"""Training table for the recall-risk model -> data/clean/model/training.parquet.

One row per vehicle (make, model, model year) x complaint theme x snapshot month, for every
pair with at least one complaint in the WINDOW_MONTHS-month window ending that month.
Snapshots fall every SNAPSHOT_MONTHS months, counted back from the last month of data so the
last snapshot is "now". Every feature uses only complaints and recalls dated up to the end of
the snapshot month, so a row could have been computed on that day.

Why snapshots and not the months a pair received a complaint: the current risk list has to be
scored as of today, also for pairs that have gone quiet. A model trained only on months with a
complaint never sees a quiet pair, and scoring each pair at its last complaint month left half
of the top of the list months out of date. With snapshots, training and scoring are the same
question: as of this month, how likely is a covering recall in the next HORIZON_MONTHS?
quiet_months says how long the pair has been quiet.

Label: a recall covering the pair (same vehicle, matched to the theme; see signals.py) dated
in the HORIZON_MONTHS months after the snapshot. Snapshots in the last HORIZON_MONTHS months of
data have no label yet (NULL); the last one is what the model scores. Snapshots at or after a
covering recall are dropped: once recalled, the question is settled. recall_mi (not a feature)
is the month of the pair's first covering recall.

Features, all as of the snapshot month:
  complaints        k (that month), a (24-month window), cum (all so far), k3 and k3_prev
                    (last three months and the three before), growth = (k3 + 1) / (k3_prev + 1),
                    quiet_months (since the pair's last complaint)
  disproportion     prr, chi2, signal (the signals.py rule on this window), signal_months
                    (complaint months the rule fired so far), vehicle_total and theme_total
                    (window), share (a / vehicle_total)
  surge             (surge.py, on this snapshot) vehicle_k3 (the vehicle's complaints in the
                    last three months), expected3, surge_ratio = (k3 + 0.5) / (expected3 + 0.5),
                    surge_logp (-log10 of the binomial p-value), surge (the flag, false
                    discovery rate per snapshot among pairs with a recent complaint),
                    surge_months (complaint months surged so far)
  severity          shares of the window's complaints with a crash, fire, injury, death or tow
  age               vehicle_age (years since the model year), months_active (since the
                    pair's first complaint)
  context           sibling_signals: signalling complaint months for the same make, model and
                    theme in other model years in the window; theme_recalls: recalls matched to
                    this theme for any vehicle so far; make_recalls_5y: the make's recalls in
                    the last 5 years
  categories        make, theme_id
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa

from pipeline import surge
from pipeline.config import CLEAN, EMBED_MODEL
from pipeline.embed import Encoder
from pipeline.signals import WINDOW_MONTHS, _is_signal, _prr_sql, prepare

HORIZON_MONTHS = 24
SNAPSHOT_MONTHS = 3

FEATURES = (
    "k", "a", "cum", "k3", "k3_prev", "growth", "quiet_months", "prr", "chi2", "signal",
    "signal_months", "vehicle_k3", "expected3", "surge_ratio", "surge_logp", "surge",
    "surge_months", "vehicle_total", "theme_total", "share", "crash_share", "fire_share",
    "injury_share", "death_share", "towed_share", "vehicle_age", "months_active",
    "sibling_signals", "theme_recalls", "make_recalls_5y", "make", "theme_id",
)
CATEGORICAL = ("make", "theme_id")


def snapshot_months(first_mi: int, last_mi: int, every: int = SNAPSHOT_MONTHS) -> list[int]:
    """Month indexes from first_mi to last_mi that are a multiple of `every` before last_mi."""
    return list(range(last_mi, first_mi - 1, -every))[::-1]


def build(con, last_mi: int, recalls: str, out: Path,
          every: int = SNAPSHOT_MONTHS) -> None:
    """Write training.parquet from the tables signals.prepare left in con."""
    w = WINDOW_MONTHS - 1
    first_mi = con.execute("SELECT min(mi) FROM monthly").fetchall()[0][0]
    con.register("snap_list", pa.table({"s": pa.array(snapshot_months(first_mi, last_mi, every),
                                                      pa.int64())}))
    con.execute("CREATE OR REPLACE TEMP TABLE snaps AS SELECT s FROM snap_list")
    # Per pair and complaint month: counts, severity and running totals of the monthly flags.
    con.execute("""
        CREATE OR REPLACE TEMP TABLE pm AS
        WITH sev AS (
            SELECT make, model, model_year, theme_id, mi,
                   count(DISTINCT odi_number) FILTER (crash) AS crash,
                   count(DISTINCT odi_number) FILTER (fire) AS fire,
                   count(DISTINCT odi_number) FILTER (injury) AS injury,
                   count(DISTINCT odi_number) FILTER (death) AS death,
                   count(DISTINCT odi_number) FILTER (towed) AS towed
            FROM base WHERE theme_id IS NOT NULL GROUP BY ALL
        )
        SELECT mo.make, mo.model, mo.model_year, mo.theme_id, mo.mi, mo.k, mo.cum, mo.signal,
               s.crash, s.fire, s.injury, s.death, s.towed,
               sum(mo.signal::INT) OVER pair AS signal_cum,
               sum(sg.surge::INT) OVER pair AS surge_cum
        FROM monthly mo JOIN sev s USING (make, model, model_year, theme_id, mi)
        JOIN surge sg USING (make, model, model_year, theme_id, mi)
        WINDOW pair AS (PARTITION BY mo.make, mo.model, mo.model_year, mo.theme_id
                        ORDER BY mo.mi ROWS UNBOUNDED PRECEDING)
    """)
    prr, chi2 = _prr_sql("r.a", "v.nv", "t.ct", "g.n")
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE snap_rows AS
        WITH r AS (  -- each pair at each snapshot whose window holds one of its complaints
            SELECT pm.make, pm.model, pm.model_year, pm.theme_id, sn.s,
                   sum(pm.k) AS a,
                   coalesce(sum(pm.k) FILTER (pm.mi >= sn.s - 2), 0) AS k3,
                   coalesce(sum(pm.k) FILTER (pm.mi BETWEEN sn.s - 5 AND sn.s - 3), 0)
                       AS k3_prev,
                   coalesce(sum(pm.k) FILTER (pm.mi = sn.s), 0) AS k,
                   max(pm.mi) AS last_complaint_mi,
                   arg_max(pm.cum, pm.mi) AS cum,
                   arg_max(pm.signal_cum, pm.mi) AS signal_months,
                   arg_max(pm.surge_cum, pm.mi) AS surge_months,
                   count(*) FILTER (pm.signal) AS own_signals,
                   sum(pm.crash) AS crash_w, sum(pm.fire) AS fire_w,
                   sum(pm.injury) AS injury_w, sum(pm.death) AS death_w,
                   sum(pm.towed) AS towed_w
            FROM pm JOIN snaps sn ON sn.s BETWEEN pm.mi AND pm.mi + {w}
            GROUP BY ALL
        ),
        vm AS (SELECT make, model, model_year, mi, count(DISTINCT odi_number) AS k
               FROM base GROUP BY ALL),
        v AS (SELECT vm.make, vm.model, vm.model_year, sn.s, sum(vm.k) AS nv,
                     coalesce(sum(vm.k) FILTER (vm.mi >= sn.s - 2), 0) AS nv3
              FROM vm SEMI JOIN (SELECT DISTINCT make, model, model_year FROM pm) used
                   USING (make, model, model_year)
              JOIN snaps sn ON sn.s BETWEEN vm.mi AND vm.mi + {w}
              GROUP BY ALL),
        tm AS (SELECT theme_id, mi, count(DISTINCT odi_number) AS k
               FROM base WHERE theme_id IS NOT NULL GROUP BY ALL),
        t AS (SELECT tm.theme_id, sn.s, sum(tm.k) AS ct
              FROM tm JOIN snaps sn ON sn.s BETWEEN tm.mi AND tm.mi + {w} GROUP BY ALL),
        gm AS (SELECT mi, count(DISTINCT odi_number) AS k FROM base GROUP BY ALL),
        g AS (SELECT sn.s, sum(gm.k) AS n
              FROM gm JOIN snaps sn ON sn.s BETWEEN gm.mi AND gm.mi + {w} GROUP BY ALL),
        mm AS (  -- signalling complaint months per make, model and theme, any model year
            SELECT pm.make, pm.model, pm.theme_id, sn.s, count(*) AS signals
            FROM pm JOIN snaps sn ON sn.s BETWEEN pm.mi AND pm.mi + {w}
            WHERE pm.signal GROUP BY ALL),
        first_c AS (SELECT make, model, model_year, theme_id, min(mi) AS first_mi
                    FROM pm GROUP BY ALL),
        first_recall AS (SELECT make, model, model_year, theme_id, min(recall_mi) AS recall_mi
                         FROM covering GROUP BY ALL),
        theme_rec AS (
            SELECT ct.theme_id, sn.s, count(DISTINCT ct.campaign_id) AS n
            FROM (SELECT DISTINCT theme_id, campaign_id, recall_mi FROM covering) ct
            JOIN snaps sn ON ct.recall_mi <= sn.s GROUP BY ALL),
        make_rec AS (
            SELECT mr.make, sn.s, count(DISTINCT mr.campaign_id) AS n
            FROM (SELECT DISTINCT make, campaign_id,
                         year(report_date) * 12 + month(report_date) - 1 AS recall_mi
                  FROM '{recalls}' WHERE report_date IS NOT NULL) mr
            JOIN snaps sn ON mr.recall_mi BETWEEN sn.s - 59 AND sn.s GROUP BY ALL)
        SELECT row_number() OVER () AS rid,
               r.*, v.nv, v.nv3, t.ct, g.n, {prr} AS prr, {chi2} AS chi2,
               {_is_signal('r.a', prr, chi2)}::INT AS signal,
               coalesce(mm.signals, 0) - r.own_signals AS sibling_signals,
               f.first_mi, fr.recall_mi,
               coalesce(tr.n, 0) AS theme_recalls, coalesce(mk.n, 0) AS make_recalls_5y
        FROM r JOIN v USING (make, model, model_year, s) JOIN t USING (theme_id, s)
        JOIN g USING (s) JOIN first_c f USING (make, model, model_year, theme_id)
        LEFT JOIN mm USING (make, model, theme_id, s)
        LEFT JOIN first_recall fr USING (make, model, model_year, theme_id)
        LEFT JOIN theme_rec tr USING (theme_id, s)
        LEFT JOIN make_rec mk USING (make, s)
        WHERE fr.recall_mi IS NULL OR r.s < fr.recall_mi
    """)

    # Surge on each snapshot: the same test as surge.py, with the false discovery rate
    # controlled per snapshot among pairs with a complaint in the last three months.
    x = con.execute("SELECT rid, s, k3, nv3, a, nv, ct, n FROM snap_rows ORDER BY rid").fetchnumpy()
    expected, p = surge.surge_test(x["k3"], x["nv3"], x["a"] - x["k3"], x["nv"] - x["nv3"],
                                   x["ct"] / x["n"])
    tested = x["k3"] > 0
    q = np.ones(len(p))
    if tested.any():
        q[tested] = surge.fdr_by_month(p[tested], x["s"][tested])
    flag = tested & (x["k3"] >= surge.MIN_SURGE_COMPLAINTS) & (q <= surge.MAX_FDR)
    con.register("surge_snap", pa.table({"rid": x["rid"], "expected3": expected,
                                         "surge_p": p, "surge": flag.astype(np.int32)}))
    con.execute(f"""
        COPY (
        SELECT r.make, r.model, r.model_year, r.theme_id,
               make_date((r.s // 12)::INT, (r.s % 12 + 1)::INT, 1) AS month, r.s AS mi,
               r.k, r.a, r.cum, r.k3, r.k3_prev, (r.k3 + 1) / (r.k3_prev + 1) AS growth,
               r.s - r.last_complaint_mi AS quiet_months,
               r.prr, r.chi2, r.signal, r.signal_months,
               r.nv3 AS vehicle_k3, x.expected3,
               (r.k3 + 0.5) / (x.expected3 + 0.5) AS surge_ratio,
               -log10(greatest(x.surge_p, 1e-300)) AS surge_logp, x.surge, r.surge_months,
               r.nv AS vehicle_total, r.ct AS theme_total, r.a / r.nv AS share,
               r.crash_w / r.a AS crash_share, r.fire_w / r.a AS fire_share,
               r.injury_w / r.a AS injury_share, r.death_w / r.a AS death_share,
               r.towed_w / r.a AS towed_share,
               r.s / 12.0 - r.model_year AS vehicle_age, r.s - r.first_mi AS months_active,
               r.sibling_signals, r.theme_recalls, r.make_recalls_5y,
               CASE WHEN r.s <= {last_mi - HORIZON_MONTHS}
                    THEN (r.recall_mi IS NOT NULL
                          AND r.recall_mi <= r.s + {HORIZON_MONTHS})::INT
               END AS label, r.recall_mi
        FROM snap_rows r JOIN surge_snap x USING (rid)
        ORDER BY r.s, r.make, r.model, r.model_year, r.theme_id
        ) TO '{(out / 'training.parquet').as_posix()}' (FORMAT parquet, COMPRESSION zstd)
    """)
    con.unregister("surge_snap")
    con.execute("DROP TABLE snap_rows")


def run(clean: Path = CLEAN, model_name: str = EMBED_MODEL, encode: Encoder | None = None) -> None:
    out = clean / "model"
    out.mkdir(parents=True, exist_ok=True)
    con, last_mi, _ = prepare(clean, model_name, encode, out)
    build(con, last_mi, (clean / "recalls.parquet").as_posix(), out)
    rows, labelled, positive = con.execute(
        f"SELECT count(*), count(label), sum(label) FROM "
        f"'{(out / 'training.parquet').as_posix()}'").fetchone()
    print(f"[features] {rows:,} pair snapshots, {labelled:,} labelled, {positive:,} followed by "
          f"a covering recall within {HORIZON_MONTHS} months ({positive / max(labelled, 1):.2%})")
