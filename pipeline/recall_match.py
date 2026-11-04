"""Which complaint themes does each recall cover? Claude decides, from an embedding shortlist
-> data/clean/signals/recall_labels.parquet.

signals.py first matched recalls to themes by embedding alone: a recall's defect summary to the
RECALL_TOP_K most similar clusters, if at least RECALL_MIN_SIM. Read by hand, only about six in
ten matches between 0.55 and 0.6 were right, and every label the model learns from ("a recall
covered this pair") inherits that noise.

Now the embedding only shortlists: each recall's CANDIDATES most similar defect themes with
similarity >= CANDIDATE_MIN_SIM (wider than the old rule, so matches it missed get a chance).
Claude (the bulk labelling model, through the Batch API) reads the recall's component and defect
summary next to each candidate's name and summary and lists the themes that describe the same
defect: the same part failing the same way. Every recall with a candidate is asked, which covers
every match the embedding rule would make (its clusters are always on the shortlist), so all
recall-theme matches, including those behind the theme_recalls feature, come from one method.

A label is kept per recall for the cluster fit (themes change with a refit) and the text of its
summary (NHTSA amends summaries); anything else new is asked on the next run. signals.prepare
uses Claude's themes for every labelled recall and falls back to the embedding rule for the
rest (recalls published since the last `make match`).
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from pipeline import batch
from pipeline.config import CLEAN, EMBED_MODEL, NAMING_MODEL
from pipeline.embed import Encoder

CANDIDATES = 6
CANDIDATE_MIN_SIM = 0.40
RECALL_CHARS = 1500

SYSTEM = """You match U.S. vehicle safety recalls to themes of owner complaints filed with \
NHTSA. You get one recall (its component and defect description) and a short list of \
complaint themes, each with an id, a name and a one-sentence summary of what owners report.

List the ids of the themes that describe the same defect as the recall: the same part failing \
in the same way, so that owners who experienced the recalled defect would have filed their \
complaints in that theme. Rules:
- Same system but a different failure does not match: a recall for a leaking brake hose does \
not match a theme about brake pads wearing early.
- A theme that describes the symptom owners would notice does match: a recall for a fuel pump \
that can fail matches a theme about the engine stalling while driving.
- Themes about recall paperwork, dealers or parts availability never match.
- Return an empty list when no theme matches. Most recalls match one theme, some two, many none."""

SCHEMA = {
    "type": "object",
    "properties": {"theme_ids": {"type": "array", "items": {"type": "integer"}}},
    "required": ["theme_ids"],
    "additionalProperties": False,
}

LABELS_SCHEMA = pa.schema([
    ("fit_id", pa.string()), ("campaign_id", pa.string()), ("text_md5", pa.string()),
    ("candidates", pa.list_(pa.int32())), ("theme_ids", pa.list_(pa.int32())),
    ("model", pa.string())])


def theme_catalog(clean: Path, con: duckdb.DuckDBPyConnection) -> dict[int, dict]:
    """theme_id -> name and summary, for defect themes of the current fit (empty before
    `make name`)."""
    out = clean / "clusters"
    names, themes = out / "names.parquet", out / "themes.parquet"
    if not names.exists():
        return {}
    fit = con.execute(f"SELECT any_value(fit_id) FROM '{(out / 'clusters.parquet').as_posix()}'"
                      ).fetchone()[0]
    # Same theme ids as signals._themes: the theme a cluster was grouped into, else itself.
    grouped = (f"LEFT JOIN '{themes.as_posix()}' t ON t.fit_id = n.fit_id "
               f"AND t.cluster_id = n.cluster_id" if themes.exists() else
               "LEFT JOIN (SELECT NULL::INT AS theme_id, NULL AS theme_name) t ON FALSE")
    rows = con.execute(f"""
        SELECT coalesce(t.theme_id, n.cluster_id) AS theme_id,
               any_value(coalesce(t.theme_name, n.name)) AS name,
               string_agg(n.summary, ' ' ORDER BY n.cluster_id) AS summary
        FROM '{names.as_posix()}' n {grouped}
        WHERE n.fit_id = ? AND n.kind = 'defect'
        GROUP BY 1""", [fit]).fetchall()
    return {int(t): {"name": nm, "summary": sm} for t, nm, sm in rows}


def candidates(con: duckdb.DuckDBPyConnection, recall_vectors: np.ndarray,
               campaign_ids: list[str], centroids: np.ndarray, cluster_ids: np.ndarray,
               k: int = CANDIDATES, min_sim: float = CANDIDATE_MIN_SIM) -> pd.DataFrame:
    """Each recall's k most similar defect themes (a theme's similarity is its best cluster's)
    at or above min_sim. Needs the `theme_of` table in con."""
    if not len(campaign_ids) or not len(cluster_ids):
        return pd.DataFrame(columns=["campaign_id", "theme_id", "similarity"])
    sims = recall_vectors.astype(np.float32) @ centroids.astype(np.float32).T
    rows, cols = np.nonzero(sims >= min_sim)
    con.register("cand_raw", pa.table({
        "campaign_id": pa.array([campaign_ids[i] for i in rows], pa.string()),
        "cluster_id": pa.array(cluster_ids[cols], pa.int32()),
        "similarity": pa.array(sims[rows, cols], pa.float32())}))
    df = con.execute(f"""
        WITH t AS (SELECT campaign_id, theme_id, max(similarity) AS similarity
                   FROM cand_raw JOIN theme_of USING (cluster_id) GROUP BY ALL),
        ranked AS (SELECT *, row_number() OVER (PARTITION BY campaign_id
                                                ORDER BY similarity DESC, theme_id) AS rk
                   FROM t)
        SELECT campaign_id, theme_id, similarity
        FROM ranked WHERE rk <= {k} ORDER BY campaign_id, similarity DESC""").df()
    con.unregister("cand_raw")
    return df


def prompt(recall: dict, cands: list[int], catalog: dict[int, dict]) -> str:
    lines = "\n".join(f"{t} | {catalog[t]['name']} | {catalog[t]['summary']}" for t in cands)
    return (f"Recall {recall['campaign_id']}\n"
            f"Component: {recall['components'] or 'unknown'}\n"
            f"Defect: {recall['text'][:RECALL_CHARS]}\n\n"
            f"Complaint themes (id | name | summary):\n{lines}")


def load_labels(path: Path, fit_id: str) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=LABELS_SCHEMA.names)
    df = pq.read_table(path).to_pandas()
    return df[df["fit_id"] == fit_id]


def run(clean: Path = CLEAN, model_name: str = EMBED_MODEL, model: str = NAMING_MODEL,
        encode: Encoder | None = None, client=None, wait: bool = True,
        poll_seconds: float = batch.POLL_SECONDS) -> bool:
    """Label every relevant recall that has no label for the current fit and summary text.
    True when all are labelled."""
    from pipeline.signals import _recall_vectors, _themes, recall_text

    out = clean / "signals"
    out.mkdir(parents=True, exist_ok=True)
    path, state = out / "recall_labels.parquet", out / "recall_match_batch.json"
    con = duckdb.connect()
    catalog = theme_catalog(clean, con)
    if not catalog:
        raise SystemExit("[match] clusters are not named yet; run `make name` first")
    clusters = pq.read_table(clean / "clusters" / "clusters.parquet",
                             columns=["fit_id", "cluster_id", "raw_centroid"])
    fit_id = clusters.column("fit_id")[0].as_py()
    con.execute(f"CREATE TEMP TABLE theme_of AS {_themes(clean, con)}")

    rv = _recall_vectors(clean, model_name, encode)
    cent = clusters.column("raw_centroid").combine_chunks()
    cand = candidates(
        con, rv.column("embedding").combine_chunks().values.to_numpy(zero_copy_only=False)
        .reshape(rv.num_rows, -1), rv.column("campaign_id").to_pylist(),
        cent.values.to_numpy(zero_copy_only=False).reshape(clusters.num_rows, -1),
        clusters.column("cluster_id").to_numpy())
    shortlist = {cid: g["theme_id"].astype(int).tolist()
                 for cid, g in cand.groupby("campaign_id")}
    asked = set(shortlist)
    md5 = dict(zip(rv.column("campaign_id").to_pylist(), rv.column("text_md5").to_pylist(),
                   strict=True))
    # The same summary text the recall vectors (and their text_md5) were made from.
    info = {r["campaign_id"]: r for r in con.execute(f"""
        SELECT campaign_id, array_to_string(components, '; ') AS components,
               defect_summary AS summary
        FROM '{(clean / 'recall_timeline.parquet').as_posix()}'
        WHERE campaign_id IN (SELECT unnest(?))""", [sorted(asked)]).df().to_dict("records")}

    labels = load_labels(path, fit_id)

    def missing() -> list[str]:
        have = set(zip(labels["campaign_id"], labels["text_md5"], strict=True))
        return [c for c in sorted(asked) if (c, md5[c]) not in have]

    def build() -> list[dict]:
        return [batch.request(c, model, SYSTEM,
                              prompt({"campaign_id": c, "components": info[c]["components"],
                                      "text": recall_text(info[c]["summary"] or "")},
                                     shortlist[c], catalog), SCHEMA, max_tokens=2000)
                for c in missing()]

    def store(answers: dict[str, dict]) -> None:
        nonlocal labels
        new = [{"fit_id": fit_id, "campaign_id": c, "text_md5": md5[c],
                "candidates": shortlist[c],
                # Only ids from the shortlist count; anything else is a slip.
                "theme_ids": [t for t in dict.fromkeys(a["theme_ids"]) if t in shortlist[c]],
                "model": model}
               for c, a in answers.items() if c in shortlist]
        if not new:
            return
        keep = labels[~labels["campaign_id"].isin({r["campaign_id"] for r in new})]
        labels = pd.concat([keep, pd.DataFrame(new)], ignore_index=True)
        old = (pq.read_table(path).to_pandas() if path.exists()
               else pd.DataFrame(columns=LABELS_SCHEMA.names))
        other_fits = old[old["fit_id"] != fit_id]
        pq.write_table(pa.Table.from_pandas(pd.concat([other_fits, labels], ignore_index=True),
                                            schema=LABELS_SCHEMA, preserve_index=False), path)

    print(f"[match] {len(asked):,} recalls have a candidate theme; "
          f"{len(missing()):,} need a label")
    client = client or (batch.make_client() if missing() or state.exists() else None)
    done = True if client is None else batch.label_or_warn(
        client, state, "match", build, store, wait, poll_seconds)
    matched = labels[labels["campaign_id"].isin(asked)]
    if len(matched):
        n_themes = matched["theme_ids"].map(len)
        print(f"[match] {len(matched):,} recalls labelled: {(n_themes > 0).mean():.0%} cover at "
              f"least one theme, {n_themes.sum():,} recall-theme matches")
    return done
