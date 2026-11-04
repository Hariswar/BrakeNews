"""Early-warning signals: which vehicles complain about a theme far more than other vehicles do,
and would that have flagged past defects before NHTSA's recall? -> data/clean/signals/.

Unit: a vehicle (make, model, model year) and a complaint theme (a cluster, or a group of
clusters once `make name` has run; then only "defect" themes count).

Signal. Disproportionality, the method drug-safety agencies use on adverse-event reports, over
a rolling window of the last WINDOW_MONTHS months of complaints:
                      theme T   other complaints
    vehicle V            a            b
    other vehicles       c            d
  PRR = (a / (a + b)) / (c / (c + d)): how much more often V's complaints are about T than
  everyone else's are. V and T signal when a >= MIN_COMPLAINTS, PRR >= MIN_PRR and the Yates
  chi-square >= MIN_CHI2 (Evans et al., 2001). Comparing shares rather than raw counts cancels
  what inflates counts without a defect: a popular model, an older vehicle, a year with more
  complaints overall. The onset is the first month a pair signals.

Recalls. A recall covers a pair when it names the same make, model and model year and matches
the pair's theme. Which themes a recall matches comes from recall_match.py (`make match`): an
embedding shortlist that Claude narrows to the themes describing the same defect. A recall
Claude has not labelled yet (published since the last `make match`, or before clusters were
named) falls back to the embedding rule: its defect summary, embedded with the narratives'
model, matches the RECALL_TOP_K clusters whose (unadjusted) centroids it is most similar to, if
at least RECALL_MIN_SIM. Both are much tighter than matching on NHTSA's component category
("ENGINE" covers dozens of unrelated themes).

Surge. A second, independent flag from surge.py: the vehicle's own complaints have shifted
toward the theme in the last few months, beyond what its history predicts (binomial test,
false discovery rate controlled per month). The PRR compares a vehicle with other vehicles; a
surge compares it with its own past.

Backtest. For pairs a later recall covered, did the signal (or a surge) fire before the recall,
and how early? For signals old enough to have been acted on, how often did a recall follow
within FOLLOW_MONTHS, against pairs that reached the same complaint count without signalling?
The same for surges, and for signals with and without a surge. None of these numbers is a
model score: recalls are not the only outcome of a real defect, and many follow manufacturer
findings rather than complaints.

Outputs:
  theme_monthly.parquet  complaints per vehicle x theme x month (for charts)
  signals.parquet        one row per vehicle x theme: counts, onset, surge onset, current
                         window and surge, recall
  recall_themes.parquet  campaign_id -> matched theme_id, similarity (embedding rule only),
                         method ("claude" or "embedding")
  backtest.json          the numbers above
Recall-summary vectors are cached in data/clean/embeddings/<model>/recalls.parquet.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from pipeline import surge
from pipeline.config import CLEAN, EMBED_MODEL
from pipeline.embed import Encoder, load_encoder

WINDOW_MONTHS = 24
MIN_COMPLAINTS = 3
MIN_PRR = 2.0
MIN_CHI2 = 4.0
# Judged by reading samples of matches (October 2026): below 0.5 most are wrong, 0.55-0.6 about
# six in ten are right, above 0.6 eight or nine in ten. The wrong ones are mostly trailer, RV and
# bus recalls, which share few vehicles with the complaints anyway.
RECALL_TOP_K = 2
RECALL_MIN_SIM = 0.55
FOLLOW_MONTHS = 24


# "Ford Motor Company (Ford) is recalling certain 2015 Lincoln MKZ vehicles manufactured ...":
# every recall since about 2012 opens like this, and left in, the shared opening pulls summaries
# toward the complaint clusters about recall notices instead of the defect itself.
_RECALL_OPENING = re.compile(r"^.{0,400}?\bIS RECALLING\b[^.]*(?:\.(?=\s)|$)\s*|"
                             r"^VEHICLE DESCRIPTION:[^.]*\.\s*", re.IGNORECASE | re.DOTALL)


def recall_text(summary: str) -> str:
    """The part of a recall's defect summary that describes the defect."""
    rest = _RECALL_OPENING.sub("", summary.strip(), count=1)
    return rest or summary.strip()


def _recall_vectors(clean: Path, model_name: str, encode: Encoder | None) -> pa.Table:
    """campaign_id -> vector of its defect summary, encoding only new or changed summaries."""
    con = duckdb.connect()
    recalls = con.execute(f"""
        SELECT campaign_id, defect_summary FROM '{(clean / 'recall_timeline.parquet').as_posix()}'
        WHERE nullif(trim(defect_summary), '') IS NOT NULL ORDER BY campaign_id""").fetchall()
    recalls = [(cid, recall_text(t)) for cid, t in recalls]
    md5 = [hashlib.md5(t.encode()).hexdigest() for _, t in recalls]
    cache = clean / "embeddings" / model_name.replace("/", "--") / "recalls.parquet"
    cached = {}
    if cache.exists():
        old = pq.read_table(cache).to_pylist()
        cached = {(r["campaign_id"], r["text_md5"]): r["embedding"] for r in old}
    todo = [i for i, (cid, _) in enumerate(recalls) if (cid, md5[i]) not in cached]
    if todo:
        encode = encode or load_encoder(model_name)
        print(f"[signals] embedding {len(todo):,} recall summaries")
        vecs = np.asarray(encode([recalls[i][1] for i in todo]), dtype=np.float16)
        for i, v in zip(todo, vecs, strict=True):
            cached[(recalls[i][0], md5[i])] = v.tolist()
    vectors = np.array([cached[(cid, md5[i])] for i, (cid, _) in enumerate(recalls)],
                       dtype=np.float16).reshape(len(recalls), -1)
    table = pa.table({
        "campaign_id": [cid for cid, _ in recalls], "text_md5": md5,
        "embedding": pa.FixedSizeListArray.from_arrays(pa.array(vectors.reshape(-1)),
                                                       max(vectors.shape[1], 1)),
    })
    cache.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, cache, compression="zstd")
    return table


def match_recalls(recall_vectors: np.ndarray, campaign_ids: list[str], centroids: np.ndarray,
                  cluster_ids: np.ndarray, top_k: int = RECALL_TOP_K,
                  min_sim: float = RECALL_MIN_SIM) -> pa.Table:
    """Each recall's top_k most similar clusters with similarity >= min_sim."""
    if len(campaign_ids) == 0 or len(cluster_ids) == 0:
        return pa.table({"campaign_id": pa.array([], pa.string()),
                         "cluster_id": pa.array([], pa.int32()),
                         "similarity": pa.array([], pa.float32())})
    sims = recall_vectors.astype(np.float32) @ centroids.astype(np.float32).T
    k = min(top_k, len(cluster_ids))
    best = np.argsort(-sims, axis=1)[:, :k]
    rows, cols = np.nonzero(np.take_along_axis(sims, best, axis=1) >= min_sim)
    return pa.table({
        "campaign_id": pa.array([campaign_ids[i] for i in rows], pa.string()),
        "cluster_id": pa.array(cluster_ids[best[rows, cols]], pa.int32()),
        "similarity": pa.array(sims[rows, best[rows, cols]], pa.float32()),
    })


def _themes(clean: Path, con: duckdb.DuckDBPyConnection) -> str:
    """SQL for cluster_id -> theme_id over the clusters that count as defects."""
    out = clean / "clusters"
    names, themes = out / "names.parquet", out / "themes.parquet"
    fit = con.execute(f"SELECT any_value(fit_id) FROM '{(out / 'clusters.parquet').as_posix()}'"
                      ).fetchone()[0]
    named = names.exists() and con.execute(
        f"SELECT count(*) FROM '{names.as_posix()}' WHERE fit_id = ?", [fit]).fetchone()[0]
    if not named:
        print("[signals] clusters not named yet: every cluster counts as a theme "
              "(`make name` adds kinds and merges duplicates)")
        return (f"SELECT cluster_id, cluster_id AS theme_id "
                f"FROM '{(out / 'clusters.parquet').as_posix()}'")
    grouped = themes.exists() and con.execute(
        f"SELECT count(*) FROM '{themes.as_posix()}' WHERE fit_id = ?", [fit]).fetchone()[0]
    theme = (f"(SELECT theme_id FROM '{themes.as_posix()}' t WHERE t.fit_id = n.fit_id "
             f"AND t.cluster_id = n.cluster_id)" if grouped else "n.cluster_id")
    return (f"SELECT n.cluster_id, {theme} AS theme_id FROM '{names.as_posix()}' n "
            f"WHERE n.fit_id = '{fit}' AND n.kind = 'defect'")


def _prr_sql(a: str, nv: str, ct: str, n: str) -> tuple[str, str]:
    """PRR and Yates chi-square from a, the vehicle total, the theme total and the grand total.
    Half a complaint stands in for an empty `c` cell so the PRR stays finite."""
    b, c, d = f"({nv} - {a})", f"({ct} - {a})", f"({n} - {nv} - {ct} + {a})"
    prr = f"(({a}) / ({nv})) / (greatest({c}, 0.5) / greatest({c} + {d}, 1))"
    chi2 = (f"({n}) * pow(greatest(abs(({a}) * {d} - {b} * {c}) - ({n}) / 2.0, 0), 2) / "
            f"greatest(({nv}) * ({c} + {d}) * (({a}) + {c}) * ({b} + {d}), 1)")
    return prr, chi2


def _is_signal(a: str, prr: str, chi2: str) -> str:
    return f"({a} >= {MIN_COMPLAINTS} AND {prr} >= {MIN_PRR} AND {chi2} >= {MIN_CHI2})"


def prepare(clean: Path, model_name: str, encode: Encoder | None,
            out: Path) -> tuple[duckdb.DuckDBPyConnection, int, pa.Table]:
    """A connection holding the tables signals and model features share:
      theme_of  cluster_id -> theme_id, defect themes only
      base      one row per complaint x vehicle: month index mi, theme_id (NULL for none),
                crash / fire / injury / death / towed flags
      monthly   per vehicle x theme x month with a complaint: k (that month), a (window),
                cum, nv, ct, n, prr, chi2, signal
      surge     per row of monthly: k3, nv3, expected3, surge_p, surge_q, surge (surge.py)
      recall_theme  campaign_id -> theme_id, similarity, method
      covering  recalls covering a vehicle x theme: campaign_id, recall_mi, report_date
    Returns the connection, the last month index of data and the recall-to-theme matches."""
    con = duckdb.connect()
    con.execute("SET preserve_insertion_order = false")
    p = {k: (clean / f"{k}.parquet").as_posix()
         for k in ("complaints", "narratives", "recalls")}
    assignments = (clean / "clusters" / "assignments.parquet").as_posix()
    clusters = pq.read_table(clean / "clusters" / "clusters.parquet",
                             columns=["fit_id", "cluster_id", "raw_centroid"])

    # Recall -> clusters, through the defect summaries.
    rv = _recall_vectors(clean, model_name, encode)
    cent = clusters.column("raw_centroid").combine_chunks()
    matches = match_recalls(
        rv.column("embedding").combine_chunks().values.to_numpy(zero_copy_only=False)
        .reshape(rv.num_rows, -1),
        rv.column("campaign_id").to_pylist(),
        cent.values.to_numpy(zero_copy_only=False).reshape(clusters.num_rows, -1),
        clusters.column("cluster_id").to_numpy())
    con.register("recall_clusters", matches)
    con.execute(f"CREATE TEMP TABLE theme_of AS {_themes(clean, con)}")

    # Claude's labels (recall_match.py) for this fit and this summary text replace the
    # embedding rule; an empty label means "covers none of the themes".
    labels = clean / "signals" / "recall_labels.parquet"
    con.register("recall_md5", rv.select(["campaign_id", "text_md5"]))
    con.execute("CREATE TEMP TABLE labelled (campaign_id VARCHAR, theme_ids INT[])")
    if labels.exists():
        con.execute(f"""
            INSERT INTO labelled
            SELECT l.campaign_id, l.theme_ids FROM '{labels.as_posix()}' l
            JOIN recall_md5 m ON m.campaign_id = l.campaign_id AND m.text_md5 = l.text_md5
            WHERE l.fit_id = ?""", [clusters.column("fit_id")[0].as_py()])
    con.execute("""
        CREATE TEMP TABLE recall_theme AS
        SELECT rc.campaign_id, t.theme_id, max(rc.similarity) AS similarity,
               'embedding' AS method
        FROM recall_clusters rc JOIN theme_of t USING (cluster_id)
        WHERE rc.campaign_id NOT IN (SELECT campaign_id FROM labelled)
        GROUP BY ALL
        UNION ALL
        SELECT DISTINCT l.campaign_id, u.theme_id, NULL::FLOAT AS similarity,
               'claude' AS method
        FROM labelled l, unnest(l.theme_ids) AS u(theme_id)
        WHERE u.theme_id IN (SELECT theme_id FROM theme_of)
    """)
    # One row per complaint x vehicle; theme NULL when the narrative has none (or no narrative),
    # so those complaints still count in the vehicle and grand totals.
    con.execute(f"""
        CREATE TEMP TABLE base AS
        SELECT c.odi_number, c.make, c.model, c.model_year,
               year(c.received_date) * 12 + month(c.received_date) - 1 AS mi, t.theme_id,
               bool_or(c.crash) AS crash, bool_or(c.fire) AS fire,
               bool_or(c.injured > 0) AS injury, bool_or(c.deaths > 0) AS death,
               bool_or(c.towed) AS towed
        FROM '{p['complaints']}' c
        LEFT JOIN '{p['narratives']}' n USING (complaint_id)
        LEFT JOIN '{assignments}' a USING (narrative_id)
        LEFT JOIN theme_of t USING (cluster_id)
        WHERE c.make IS NOT NULL AND c.model IS NOT NULL AND c.model_year IS NOT NULL
          AND c.received_date IS NOT NULL
        GROUP BY ALL
    """)
    last_mi = con.execute("SELECT max(mi) FROM base").fetchone()[0]
    w = WINDOW_MONTHS - 1

    # Rolling-window counts at every month a vehicle x theme pair has a complaint.
    prr, chi2 = _prr_sql("a", "nv", "ct", "n")
    con.execute(f"""
        CREATE TEMP TABLE monthly AS
        WITH vt AS (SELECT make, model, model_year, theme_id, mi, count(DISTINCT odi_number) k
                    FROM base WHERE theme_id IS NOT NULL GROUP BY ALL),
        v AS (SELECT make, model, model_year, mi, count(DISTINCT odi_number) k
              FROM base GROUP BY ALL),
        t AS (SELECT theme_id, mi, count(DISTINCT odi_number) k
              FROM base WHERE theme_id IS NOT NULL GROUP BY ALL),
        g AS (SELECT mi, count(DISTINCT odi_number) k FROM base GROUP BY ALL),
        vt_r AS (SELECT *, sum(k) OVER (PARTITION BY make, model, model_year, theme_id
                                        ORDER BY mi RANGE BETWEEN {w} PRECEDING AND CURRENT ROW) a,
                           sum(k) OVER (PARTITION BY make, model, model_year, theme_id
                                        ORDER BY mi ROWS UNBOUNDED PRECEDING) cum
                 FROM vt),
        v_r AS (SELECT make, model, model_year, mi, sum(k) OVER (
                    PARTITION BY make, model, model_year ORDER BY mi
                    RANGE BETWEEN {w} PRECEDING AND CURRENT ROW) nv FROM v),
        t_r AS (SELECT theme_id, mi, sum(k) OVER (PARTITION BY theme_id ORDER BY mi
                    RANGE BETWEEN {w} PRECEDING AND CURRENT ROW) ct FROM t),
        g_r AS (SELECT mi, sum(k) OVER (ORDER BY mi RANGE BETWEEN {w} PRECEDING AND CURRENT ROW) n
                FROM g)
        SELECT vt_r.*, nv, ct, n, {prr} AS prr, {chi2} AS chi2,
               {_is_signal('a', prr, chi2)} AS signal
        FROM vt_r JOIN v_r USING (make, model, model_year, mi) JOIN t_r USING (theme_id, mi)
        JOIN g_r USING (mi)
    """)
    surge.add(con)
    # Recalls covering each pair: same vehicle, matched to one of the theme's clusters.
    con.execute(f"""
        CREATE TEMP TABLE covering AS
        SELECT DISTINCT r.make, r.model, r.model_year, t.theme_id, r.campaign_id,
               year(r.report_date) * 12 + month(r.report_date) - 1 AS recall_mi, r.report_date
        FROM '{p['recalls']}' r
        JOIN recall_theme t USING (campaign_id)
        WHERE r.report_date IS NOT NULL
    """)
    return con, last_mi, con.sql(
        "SELECT * FROM recall_theme ORDER BY campaign_id, theme_id").to_arrow_table()


def run(clean: Path = CLEAN, model_name: str = EMBED_MODEL, encode: Encoder | None = None) -> None:
    out = clean / "signals"
    out.mkdir(parents=True, exist_ok=True)
    con, last_mi, matches = prepare(clean, model_name, encode, out)
    pq.write_table(matches, out / "recall_themes.parquet")
    w = WINDOW_MONTHS - 1
    con.execute(f"""
        COPY (SELECT make, model, model_year, theme_id,
                     make_date((mi // 12)::INT, (mi % 12 + 1)::INT, 1) AS month, k AS complaints
              FROM monthly ORDER BY ALL)
        TO '{(out / 'theme_monthly.parquet').as_posix()}' (FORMAT parquet, COMPRESSION zstd)
    """)

    # The window ending at the last month of data, for pairs quiet that month too.
    prr_r, chi2_r = _prr_sql("r.a", "rv.nv", "rt.ct", "rg.n")
    lo = last_mi - w
    con.execute(f"""
        CREATE TEMP TABLE recent AS
        WITH b AS (SELECT * FROM base WHERE mi >= {lo}),
        r AS (SELECT make, model, model_year, theme_id, count(DISTINCT odi_number) a
              FROM b WHERE theme_id IS NOT NULL GROUP BY ALL),
        rv AS (SELECT make, model, model_year, count(DISTINCT odi_number) nv FROM b GROUP BY ALL),
        rt AS (SELECT theme_id, count(DISTINCT odi_number) ct
               FROM b WHERE theme_id IS NOT NULL GROUP BY ALL),
        rg AS (SELECT count(DISTINCT odi_number) n FROM b)
        SELECT r.make, r.model, r.model_year, r.theme_id, r.a AS recent_complaints,
               {prr_r} AS recent_prr, {chi2_r} AS recent_chi2,
               {_is_signal('r.a', prr_r, chi2_r)} AS active
        FROM r JOIN rv USING (make, model, model_year) JOIN rt USING (theme_id), rg
    """)

    con.execute(f"""
        COPY (
            WITH pairs AS (
                SELECT make, model, model_year, theme_id, sum(k) AS complaints,
                       min(mi) AS first_mi, max(mi) AS last_mi,
                       min(mi) FILTER (signal) AS onset_mi, max(prr) FILTER (signal) AS peak_prr,
                       min(mi) FILTER (cum >= {MIN_COMPLAINTS}) AS reached_mi,
                       min(mi) FILTER (surge) AS surge_onset_mi,
                       arg_max(surge, mi) AS last_surge, arg_max(surge_q, mi) AS last_surge_q,
                       arg_max(k3, mi) AS last_k3, arg_max(expected3, mi) AS last_expected3
                FROM monthly JOIN surge USING (make, model, model_year, theme_id, mi)
                GROUP BY ALL
            ),
            -- The first recall dated after the pair's first complaint, and whether one came
            -- earlier (then later complaints are about an already-recalled problem).
            rec AS (
                SELECT p.make, p.model, p.model_year, p.theme_id,
                       arg_min(cv.campaign_id, cv.recall_mi)
                           FILTER (cv.recall_mi > p.first_mi) AS recall_campaign,
                       min(cv.report_date) FILTER (cv.recall_mi > p.first_mi) AS recall_date,
                       min(cv.recall_mi) FILTER (cv.recall_mi > p.first_mi) AS recall_mi,
                       bool_or(cv.recall_mi <= p.first_mi) AS recalled_before_complaints
                FROM pairs p JOIN covering cv USING (make, model, model_year, theme_id)
                GROUP BY ALL
            ),
            pre AS (
                SELECT b.make, b.model, b.model_year, b.theme_id,
                       count(DISTINCT b.odi_number) AS complaints_before_recall
                FROM base b JOIN rec USING (make, model, model_year, theme_id)
                WHERE b.mi < rec.recall_mi GROUP BY ALL
            )
            SELECT p.make, p.model, p.model_year, p.theme_id, p.complaints,
                   make_date((first_mi // 12)::INT, (first_mi % 12 + 1)::INT, 1) AS first_month,
                   make_date((last_mi // 12)::INT, (last_mi % 12 + 1)::INT, 1) AS last_month,
                   CASE WHEN onset_mi IS NOT NULL THEN
                        make_date((onset_mi // 12)::INT, (onset_mi % 12 + 1)::INT, 1) END
                       AS onset_month,
                   p.peak_prr,
                   CASE WHEN surge_onset_mi IS NOT NULL THEN
                        make_date((surge_onset_mi // 12)::INT, (surge_onset_mi % 12 + 1)::INT, 1)
                   END AS surge_onset_month,
                   coalesce(r.recent_complaints, 0) AS recent_complaints, r.recent_prr,
                   r.recent_chi2, coalesce(r.active, false) AS active,
                   -- Surging now: the pair's latest complaint month is among the last
                   -- SURGE_MONTHS months of data and it surged then.
                   p.last_mi > {last_mi - surge.SURGE_MONTHS} AND p.last_surge AS surging,
                   CASE WHEN p.last_mi > {last_mi - surge.SURGE_MONTHS} THEN p.last_k3 END
                       AS recent_k3,
                   CASE WHEN p.last_mi > {last_mi - surge.SURGE_MONTHS} THEN p.last_expected3 END
                       AS recent_expected3,
                   CASE WHEN p.last_mi > {last_mi - surge.SURGE_MONTHS} THEN p.last_surge_q END
                       AS recent_surge_q,
                   rec.recall_campaign, rec.recall_date,
                   coalesce(rec.recalled_before_complaints, false) AS recalled_before_complaints,
                   coalesce(pre.complaints_before_recall, 0) AS complaints_before_recall,
                   rec.recall_mi - p.onset_mi AS lead_months,
                   p.onset_mi, p.reached_mi, p.surge_onset_mi, rec.recall_mi
            FROM pairs p
            LEFT JOIN recent r USING (make, model, model_year, theme_id)
            LEFT JOIN rec USING (make, model, model_year, theme_id)
            LEFT JOIN pre USING (make, model, model_year, theme_id)
            ORDER BY p.make, p.model, p.model_year, p.theme_id
        ) TO '{(out / 'signals.parquet').as_posix()}' (FORMAT parquet, COMPRESSION zstd)
    """)
    backtest = _backtest(con, (out / "signals.parquet").as_posix(), last_mi)
    (out / "backtest.json").write_text(json.dumps(backtest, indent=2) + "\n")
    active, surging = con.execute(
        f"SELECT count(*) FILTER (active), count(*) FILTER (surging) "
        f"FROM '{(out / 'signals.parquet').as_posix()}' "
        f"WHERE recall_campaign IS NULL AND NOT recalled_before_complaints").fetchone()
    by = dict(con.execute("SELECT method, count(*) FROM recall_theme GROUP BY 1").fetchall())
    print(f"[signals] recall-to-theme matches: {by.get('claude', 0):,} checked by Claude, "
          f"{by.get('embedding', 0):,} from the embedding rule (`make match` checks them)")
    print(f"[signals] "
          f"{active:,} active signals and {surging:,} surges with no covering recall")
    for k, v in backtest.items():
        print(f"[signals]   {k}: {v}")


def _backtest(con: duckdb.DuckDBPyConnection, signals: str, last_mi: int) -> dict:
    s = f"(SELECT * FROM '{signals}' WHERE NOT recalled_before_complaints)"
    recalled = con.execute(f"""
        SELECT count(*), count(*) FILTER (onset_mi < recall_mi),
               median(lead_months) FILTER (onset_mi < recall_mi)
        FROM {s} WHERE recall_mi IS NOT NULL AND complaints_before_recall >= {MIN_COMPLAINTS}
    """).fetchone()
    cutoff = last_mi - FOLLOW_MONTHS  # signals since then have not had time to be followed up
    flagged = con.execute(f"""
        SELECT count(*), count(*) FILTER (recall_mi > onset_mi
                                          AND recall_mi <= onset_mi + {FOLLOW_MONTHS})
        FROM {s} WHERE onset_mi <= {cutoff} AND (recall_mi IS NULL OR recall_mi > onset_mi)
    """).fetchone()
    unflagged = con.execute(f"""
        SELECT count(*), count(*) FILTER (recall_mi > reached_mi
                                          AND recall_mi <= reached_mi + {FOLLOW_MONTHS})
        FROM {s} WHERE onset_mi IS NULL AND reached_mi <= {cutoff}
          AND (recall_mi IS NULL OR recall_mi > reached_mi)
    """).fetchone()
    rate = flagged[1] / flagged[0] if flagged[0] else None
    base = unflagged[1] / unflagged[0] if unflagged[0] else None

    surged = con.execute(f"""
        SELECT count(*) FILTER (surge_onset_mi < recall_mi),
               median(recall_mi - surge_onset_mi) FILTER (surge_onset_mi < recall_mi),
               count(*) FILTER (surge_onset_mi < recall_mi OR onset_mi < recall_mi)
        FROM {s} WHERE recall_mi IS NOT NULL AND complaints_before_recall >= {MIN_COMPLAINTS}
    """).fetchone()

    def followed(onset: str, where: str = "TRUE") -> tuple[int, float | None]:
        n, hit = con.execute(f"""
            SELECT count(*), count(*) FILTER (recall_mi > {onset}
                                              AND recall_mi <= {onset} + {FOLLOW_MONTHS})
            FROM {s} WHERE {onset} <= {cutoff} AND (recall_mi IS NULL OR recall_mi > {onset})
              AND {where}
        """).fetchone()
        return n, (hit / n if n else None)

    n_surge, surge_rate = followed("surge_onset_mi")
    # Signals split by whether the pair had already surged when it signalled.
    with_surge = "coalesce(surge_onset_mi <= onset_mi, false)"
    _, sig_surge = followed("onset_mi", with_surge)
    _, sig_only = followed("onset_mi", f"NOT {with_surge}")

    def r4(x):
        return round(x, 4) if x is not None else None

    return {
        f"recalled pairs with {MIN_COMPLAINTS}+ complaints before the recall": recalled[0],
        "share flagged before the recall": round(recalled[1] / recalled[0], 3)
        if recalled[0] else None,
        "median months flagged ahead of the recall": recalled[2],
        "signals old enough to judge": flagged[0],
        f"share followed by a covering recall within {FOLLOW_MONTHS} months": round(rate, 4)
        if rate is not None else None,
        f"same share for pairs that reached {MIN_COMPLAINTS} complaints without a signal":
            round(base, 4) if base is not None else None,
        "lift": round(rate / base, 1) if rate and base else None,
        "share of recalled pairs that surged before the recall": round(surged[0] / recalled[0], 3)
        if recalled[0] else None,
        "median months surged ahead of the recall": surged[1],
        "share flagged by a signal or a surge before the recall": round(surged[2] / recalled[0], 3)
        if recalled[0] else None,
        "surges old enough to judge": n_surge,
        f"share of surges followed by a covering recall within {FOLLOW_MONTHS} months":
            r4(surge_rate),
        "surge lift over pairs without a signal": round(surge_rate / base, 1)
        if surge_rate and base else None,
        "same share for signals that had already surged": r4(sig_surge),
        "same share for signals that had not": r4(sig_only),
    }
