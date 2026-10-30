"""Name each complaint cluster with Claude, then group duplicates into themes.

Outputs in data/clean/clusters/:
  names.parquet   one row per cluster: name, one-sentence summary, kind
  themes.parquet  one row per cluster: theme_id and theme_name

1. Naming. For every cluster of the current fit that has no name yet, Claude reads the cluster's
   distinctive terms, its top component groups and the narratives nearest its centroid. `kind`
   says what the cluster is: "defect" (one malfunction), "non_defect" (dealers, recall
   paperwork, parts availability) or "mixed" (different failures that share wording, such as
   NHTSA's template text). Only defect clusters are trend signals; the rest are kept for
   completeness. Requests go through the Message Batches API (half price; usually done within
   the hour). The batch id is saved as soon as it is created, so an interrupted run picks the
   same batch up again instead of paying twice.
2. Themes. Clustering splits some problems in two (several Takata airbag clusters: recall
   notices, parts not available, ruptures). Centroid similarity cannot tell those apart from
   neighbouring but different problems (a leaking fuel tank and a tank that will not fill score
   as alike as two Takata clusters), so one request shows Claude every name and summary and asks
   which clusters describe the same problem. A cluster left out is a theme of its own.

Names and themes are tied to the cluster fit_id: after `cluster --refit` the old ones no longer
apply and the next `name` run asks for new ones.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from pipeline.config import CLEAN, NAMING_MODEL, ROOT

EXAMPLE_CHARS = 600  # per narrative; the start says what failed, the rest is mostly story
N_EXAMPLES = 15
POLL_SECONDS = 30

SYSTEM = """You label clusters of U.S. vehicle-owner complaints filed with NHTSA. Each cluster \
was found by grouping similar complaint narratives. Given its most distinctive terms, the \
component categories its complaints were filed under and sample narratives, describe the \
theme the narratives share.

- name: 3 to 7 words, specific and plain, naming the part and what goes wrong, e.g. \
"Transmission slips or shifts harshly" or "Airbag warning light stays on". Sentence case, \
no make or model unless the whole cluster is about one, no trailing period.
- summary: one sentence a car owner would understand, saying what owners report.
- kind: "defect" if most narratives describe the same vehicle or equipment malfunction; \
"non_defect" if they are mainly about dealers, recall notices or paperwork, parts availability, \
customer service or crashes with no failure described; "mixed" if they describe different \
failures that only share wording or a vehicle.

Placeholders such as [XXX], [NAME] and [PHONE] mark removed personal details; ignore them."""

SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "summary": {"type": "string"},
        "kind": {"type": "string", "enum": ["defect", "non_defect", "mixed"]},
    },
    "required": ["name", "summary", "kind"],
    "additionalProperties": False,
}


NAMES_SCHEMA = pa.schema([("fit_id", pa.string()), ("cluster_id", pa.int32()),
                          ("name", pa.string()), ("summary", pa.string()),
                          ("kind", pa.string()), ("model", pa.string())])
THEMES_SCHEMA = pa.schema([("fit_id", pa.string()), ("cluster_id", pa.int32()),
                           ("theme_id", pa.int32()), ("theme_name", pa.string())])

THEMES_SYSTEM = """You are given named clusters of U.S. vehicle-owner complaints. Clustering \
sometimes splits one problem into several clusters. Group together clusters that describe the \
same underlying problem with the same part, for example several clusters about one airbag \
inflator recall. Do not group clusters that only share a system (brake pads wearing and brake \
lines rusting are different problems) or a vehicle. Give each group a name of 3 to 7 words in \
the same style as the cluster names. Leave clusters that have no duplicate out of every group."""

THEMES_FORMAT = {
    "type": "object",
    "properties": {"groups": {"type": "array", "items": {
        "type": "object",
        "properties": {"cluster_ids": {"type": "array", "items": {"type": "integer"}},
                       "name": {"type": "string"}},
        "required": ["cluster_ids", "name"],
        "additionalProperties": False}}},
    "required": ["groups"],
    "additionalProperties": False,
}


def prompt(terms: list[str], groups: list[str], examples: list[str]) -> str:
    shown = "\n".join(
        f"{i}. {t[:EXAMPLE_CHARS]}{'...' if len(t) > EXAMPLE_CHARS else ''}"
        for i, t in enumerate(examples, 1))
    return (f"Distinctive terms: {', '.join(terms)}\n"
            f"Component categories: {', '.join(groups) or 'unknown'}\n\n"
            f"Sample narratives:\n{shown}")


def request(custom_id: str, text: str, model: str) -> dict:
    return {
        "custom_id": custom_id,
        "params": {
            "model": model,
            "max_tokens": 4000,
            "system": SYSTEM,
            "messages": [{"role": "user", "content": text}],
            "output_config": {"effort": "low",
                              "format": {"type": "json_schema", "schema": SCHEMA}},
        },
    }


def _make_client():
    import anthropic  # pip install -e ".[llm]"
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    return anthropic.Anthropic()


def _pending(state: Path, fit_id: str) -> str | None:
    if not state.exists():
        return None
    saved = json.loads(state.read_text())
    if saved["fit_id"] != fit_id:  # the clusters were refit since; that batch is useless now
        state.unlink()
        return None
    return saved["batch_id"]


def _collect(client, batch_id: str, fit_id: str, model: str) -> tuple[list[dict], int]:
    rows, failed = [], 0
    for r in client.messages.batches.results(batch_id):
        if r.result.type != "succeeded" or r.result.message.stop_reason != "end_turn":
            failed += 1
            continue
        text = next(b.text for b in r.result.message.content if b.type == "text")
        rows.append({"fit_id": fit_id, "cluster_id": int(r.custom_id.removeprefix("cluster-")),
                     **json.loads(text), "model": model})
    return rows, failed


def _name_clusters(client, clean: Path, clusters: list[dict], names: list[dict], fit_id: str,
                   model: str, wait: bool, poll_seconds: float, state: Path) -> bool:
    """Name unnamed clusters into `names` (in place). False if a batch is still running."""
    named = {r["cluster_id"] for r in names}
    batch_id = _pending(state, fit_id)
    if batch_id is None:
        todo = [c for c in clusters if c["cluster_id"] not in named]
        if not todo:
            return True
        con = duckdb.connect()
        con.register("wanted", pa.table({"narrative_id": [i for c in todo
                                                          for i in c["examples"][:N_EXAMPLES]]}))
        texts = dict(con.execute(f"""
            SELECT narrative_id, any_value(description)
            FROM '{(clean / 'narratives.parquet').as_posix()}'
            SEMI JOIN wanted USING (narrative_id) GROUP BY narrative_id""").fetchall())
        requests = [request(f"cluster-{c['cluster_id']}",
                            prompt(c["terms"], c["component_groups"] or [],
                                   [texts[i] for i in c["examples"][:N_EXAMPLES]]), model)
                    for c in todo]
        batch_id = client.messages.batches.create(requests=requests).id
        state.write_text(json.dumps({"batch_id": batch_id, "fit_id": fit_id}))
        print(f"[name] submitted batch {batch_id}: {len(requests)} clusters to {model}")

    while True:
        batch = client.messages.batches.retrieve(batch_id)
        if batch.processing_status == "ended":
            break
        if not wait:
            print(f"[name] batch {batch_id} still {batch.processing_status}; rerun to collect")
            return False
        print(f"[name] batch {batch_id}: {batch.request_counts.processing} still processing",
              flush=True)
        time.sleep(poll_seconds)

    rows, failed = _collect(client, batch_id, fit_id, model)
    names += [r for r in rows if r["cluster_id"] not in named]
    names.sort(key=lambda r: r["cluster_id"])
    state.unlink()
    print(f"[name] {len(rows)} clusters named, {failed} failed "
          f"({len(names)}/{len(clusters)} named; rerun to retry failures)")
    return True


def group_themes(names: list[dict], groups: list[dict]) -> list[dict]:
    """Theme per cluster from Claude's groups. Ids that are unknown or already in an earlier
    group are ignored, so a bad answer can only leave clusters ungrouped."""
    by_id = {r["cluster_id"]: r for r in names}
    theme = {}
    for g in groups:
        members = [i for i in dict.fromkeys(g["cluster_ids"]) if i in by_id and i not in theme]
        if len(members) > 1:
            for i in members:
                theme[i] = (min(members), g["name"])
    return [{"fit_id": r["fit_id"], "cluster_id": r["cluster_id"],
             "theme_id": theme.get(r["cluster_id"], (r["cluster_id"],))[0],
             "theme_name": theme.get(r["cluster_id"], (None, r["name"]))[1]} for r in names]


def _themes(client, names: list[dict], model: str) -> list[dict]:
    listing = "\n".join(f"{r['cluster_id']} | {r['name']} | {r['kind']} | {r['summary']}"
                        for r in names)
    msg = client.messages.create(
        model=model, max_tokens=16000, system=THEMES_SYSTEM,
        messages=[{"role": "user", "content": f"id | name | kind | summary\n{listing}"}],
        output_config={"effort": "medium",
                       "format": {"type": "json_schema", "schema": THEMES_FORMAT}})
    if msg.stop_reason != "end_turn":
        raise SystemExit(f"[name] theme grouping stopped early ({msg.stop_reason}); rerun")
    groups = json.loads(next(b.text for b in msg.content if b.type == "text"))["groups"]
    return group_themes(names, groups)


def run(clean: Path = CLEAN, model: str = NAMING_MODEL, wait: bool = True, client=None,
        poll_seconds: float = POLL_SECONDS) -> None:
    out = clean / "clusters"
    clusters = pq.read_table(out / "clusters.parquet",
                             columns=["fit_id", "cluster_id", "terms", "component_groups",
                                      "examples"]).to_pylist()
    if not clusters:
        raise SystemExit("[name] no clusters; run `make cluster` first")
    fit_id = clusters[0]["fit_id"]
    names_path, themes_path = out / "names.parquet", out / "themes.parquet"
    names = [r for r in (pq.read_table(names_path).to_pylist() if names_path.exists() else [])
             if r["fit_id"] == fit_id]
    client = client or _make_client()

    before = len(names)
    done = _name_clusters(client, clean, clusters, names, fit_id, model, wait, poll_seconds,
                          out / "naming_batch.json")
    if len(names) != before or not names_path.exists():
        pq.write_table(pa.Table.from_pylist(names, schema=NAMES_SCHEMA), names_path)
    if not done or len(names) < len(clusters):
        return  # themes need every cluster named

    themes = pq.read_table(themes_path).to_pylist() if themes_path.exists() else []
    if themes and themes[0]["fit_id"] == fit_id and len(themes) == len(names):
        print(f"[name] all {len(clusters)} clusters named and grouped")
        return
    themes = _themes(client, names, model)
    pq.write_table(pa.Table.from_pylist(themes, schema=THEMES_SCHEMA), themes_path)
    n_themes = len({t["theme_id"] for t in themes})
    print(f"[name] {len(clusters)} clusters grouped into {n_themes} themes")
