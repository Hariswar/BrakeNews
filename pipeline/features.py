"""Training table for the recall-risk model -> data/clean/model/training.parquet.

One row per vehicle (make, model, model year) x complaint theme x month in which that pair
received a complaint: the moment someone watching the data would reassess it. Every feature
uses only complaints and recalls dated up to the end of that month, so a row could have been
computed on that day.

Label: a recall covering the pair (same vehicle, matched to the theme; see signals.py) dated
in the next HORIZON_MONTHS months. Rows from the last HORIZON_MONTHS months of data have no
label yet (NULL): they are what the model scores. Rows dated after a covering recall are
dropped; once recalled, the question the model answers is settled.

Features, all as of the row's month:
  complaints        k (this month), a (24-month window), cum (all so far), k3 and k3_prev
                    (last three months and the three before), growth = (k3 + 1) / (k3_prev + 1)
  disproportion     prr, chi2, signal (the signals.py rule), signal_months (months signalled so
                    far), vehicle_total and theme_total (window), share (a / vehicle_total)
  severity          shares of the window's complaints with a crash, fire, injury, death or tow
  age               vehicle_age (years since the model year), months_active (since the
                    pair's first complaint)
  context           sibling_signals: signalling rows for the same make, model and theme in other
                    model years in the window; theme_recalls: recalls matched to this theme for
                    any vehicle so far; make_recalls_5y: the make's recalls in the last 5 years
  categories        make, theme_id
"""

from __future__ import annotations

from pathlib import Path

from pipeline.config import CLEAN, EMBED_MODEL
from pipeline.embed import Encoder
from pipeline.signals import WINDOW_MONTHS, prepare

HORIZON_MONTHS = 24

FEATURES = (
    "k", "a", "cum", "k3", "k3_prev", "growth", "prr", "chi2", "signal", "signal_months",
    "vehicle_total", "theme_total", "share", "crash_share", "fire_share", "injury_share",
    "death_share", "towed_share", "vehicle_age", "months_active", "sibling_signals",
    "theme_recalls", "make_recalls_5y", "make", "theme_id",
)
CATEGORICAL = ("make", "theme_id")


def build(con, last_mi: int, recalls: str, out: Path) -> None:
    w = WINDOW_MONTHS - 1
    con.execute(f"""
        COPY (
        WITH sev AS (  -- per pair and month: complaints with each severity flag
            SELECT make, model, model_year, theme_id, mi,
                   count(DISTINCT odi_number) FILTER (crash) AS crash,
                   count(DISTINCT odi_number) FILTER (fire) AS fire,
                   count(DISTINCT odi_number) FILTER (injury) AS injury,
                   count(DISTINCT odi_number) FILTER (death) AS death,
                   count(DISTINCT odi_number) FILTER (towed) AS towed
            FROM base WHERE theme_id IS NOT NULL GROUP BY ALL
        ),
        m AS (
            SELECT mo.*, s.crash, s.fire, s.injury, s.death, s.towed,
                   min(mo.mi) OVER pair_all AS first_mi,
                   sum(mo.k) OVER (pair ORDER BY mo.mi RANGE BETWEEN 2 PRECEDING AND CURRENT ROW)
                       AS k3,
                   coalesce(sum(mo.k) OVER (pair ORDER BY mo.mi
                                            RANGE BETWEEN 5 PRECEDING AND 3 PRECEDING), 0)
                       AS k3_prev,
                   sum(mo.signal::INT) OVER (pair ORDER BY mo.mi ROWS UNBOUNDED PRECEDING)
                       AS signal_months,
                   sum(s.crash) OVER win AS crash_w, sum(s.fire) OVER win AS fire_w,
                   sum(s.injury) OVER win AS injury_w, sum(s.death) OVER win AS death_w,
                   sum(s.towed) OVER win AS towed_w,
                   sum(mo.signal::INT) OVER (PARTITION BY mo.make, mo.model, mo.theme_id
                                             ORDER BY mo.mi
                                             RANGE BETWEEN {w} PRECEDING AND CURRENT ROW)
                       - sum(mo.signal::INT) OVER win AS sibling_signals
            FROM monthly mo JOIN sev s USING (make, model, model_year, theme_id, mi)
            WINDOW pair AS (PARTITION BY mo.make, mo.model, mo.model_year, mo.theme_id),
                   pair_all AS (PARTITION BY mo.make, mo.model, mo.model_year, mo.theme_id),
                   win AS (PARTITION BY mo.make, mo.model, mo.model_year, mo.theme_id
                           ORDER BY mo.mi RANGE BETWEEN {w} PRECEDING AND CURRENT ROW)
        ),
        -- The first covering recall of each pair: rows after it are dropped, and a label is a
        -- recall date inside the horizon.
        first_recall AS (
            SELECT make, model, model_year, theme_id, min(recall_mi) AS recall_mi
            FROM covering GROUP BY ALL
        ),
        theme_rec AS (  -- recalls matched to a theme, by month, for "so far" counts
            SELECT theme_id, recall_mi, count(DISTINCT campaign_id) AS n
            FROM covering GROUP BY ALL
        ),
        make_rec AS (
            SELECT make, year(report_date) * 12 + month(report_date) - 1 AS recall_mi,
                   count(DISTINCT campaign_id) AS n
            FROM '{recalls}' WHERE report_date IS NOT NULL GROUP BY ALL
        )
        SELECT m.make, m.model, m.model_year, m.theme_id,
               make_date((m.mi // 12)::INT, (m.mi % 12 + 1)::INT, 1) AS month, m.mi,
               m.k, m.a, m.cum, m.k3, m.k3_prev, (m.k3 + 1) / (m.k3_prev + 1) AS growth,
               m.prr, m.chi2, m.signal::INT AS signal, m.signal_months,
               m.nv AS vehicle_total, m.ct AS theme_total, m.a / m.nv AS share,
               m.crash_w / m.a AS crash_share, m.fire_w / m.a AS fire_share,
               m.injury_w / m.a AS injury_share, m.death_w / m.a AS death_share,
               m.towed_w / m.a AS towed_share,
               m.mi / 12.0 - m.model_year AS vehicle_age, m.mi - m.first_mi AS months_active,
               m.sibling_signals,
               coalesce((SELECT sum(n) FROM theme_rec t
                         WHERE t.theme_id = m.theme_id AND t.recall_mi <= m.mi), 0)
                   AS theme_recalls,
               coalesce((SELECT sum(n) FROM make_rec r WHERE r.make = m.make
                         AND r.recall_mi BETWEEN m.mi - 59 AND m.mi), 0) AS make_recalls_5y,
               CASE WHEN m.mi <= {last_mi - HORIZON_MONTHS}
                    THEN (f.recall_mi IS NOT NULL AND f.recall_mi <= m.mi + {HORIZON_MONTHS})::INT
               END AS label
        FROM m LEFT JOIN first_recall f USING (make, model, model_year, theme_id)
        WHERE f.recall_mi IS NULL OR m.mi < f.recall_mi
        ORDER BY m.mi, m.make, m.model, m.model_year, m.theme_id
        ) TO '{(out / 'training.parquet').as_posix()}' (FORMAT parquet, COMPRESSION zstd)
    """)


def run(clean: Path = CLEAN, model_name: str = EMBED_MODEL, encode: Encoder | None = None) -> None:
    out = clean / "model"
    out.mkdir(parents=True, exist_ok=True)
    con, last_mi, _ = prepare(clean, model_name, encode, out)
    build(con, last_mi, (clean / "recalls.parquet").as_posix(), out)
    rows, labelled, positive = con.execute(
        f"SELECT count(*), count(label), sum(label) FROM "
        f"'{(out / 'training.parquet').as_posix()}'").fetchone()
    print(f"[features] {rows:,} rows, {labelled:,} labelled, {positive:,} followed by a covering "
          f"recall within {HORIZON_MONTHS} months ({positive / max(labelled, 1):.2%})")
