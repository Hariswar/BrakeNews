"""Surge detection: has a vehicle's complaint mix shifted toward a theme in the last few months,
beyond what its own history predicts?

The PRR in signals.py compares a vehicle with other vehicles. A surge compares a vehicle with
its own past, so it catches a problem that starts late in a model's life (a part wearing out, a
software update) even on a vehicle that has always complained about that theme a little.

At every month a vehicle x theme pair has a complaint:
  k3        the pair's complaints in the last SURGE_MONTHS months
  nv3       all the vehicle's complaints in those months
  share0    the theme's expected share of the vehicle's complaints: its share over the rest of
            the window (the WINDOW_MONTHS - SURGE_MONTHS months before), shrunk toward the
            theme's national share by PRIOR_WEIGHT complaints' worth so a vehicle with little
            history falls back to the national mix
  expected3 = share0 * nv3
  surge_p   exact binomial tail P(X >= k3), X ~ Binomial(nv3, share0)
  surge_q   Benjamini-Hochberg adjusted p-value (statsmodels) across every pair tested that
            month, since thousands of pairs are tested at once
  surge     k3 >= MIN_SURGE_COMPLAINTS and surge_q <= MAX_FDR

Using shares of the vehicle's own complaints, not counts, cancels what raises every theme at
once: a model year selling more, an ageing fleet, NHTSA's complaint volume changing.
"""

from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd

SURGE_MONTHS = 3
PRIOR_WEIGHT = 30.0
MIN_SURGE_COMPLAINTS = 3
MAX_FDR = 0.05


def surge_test(k3, nv3, hist_a, hist_n, theme_share,
               prior: float = PRIOR_WEIGHT) -> tuple[np.ndarray, np.ndarray]:
    """Expected recent complaints and the one-sided binomial p-value of seeing k3 or more."""
    from scipy.stats import binom

    k3, nv3 = np.asarray(k3, float), np.asarray(nv3, float)
    share0 = ((np.asarray(hist_a, float) + prior * np.asarray(theme_share, float))
              / (np.maximum(np.asarray(hist_n, float), 0) + prior))
    share0 = np.clip(share0, 1e-9, 1 - 1e-9)
    return share0 * nv3, binom.sf(k3 - 1, nv3, share0)


def fdr_by_month(p: np.ndarray, month: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg q-values, each month's tests a separate family."""
    from statsmodels.stats.multitest import multipletests

    q = np.empty(len(p))
    for idx in pd.Series(np.arange(len(p))).groupby(month).indices.values():
        q[idx] = multipletests(p[idx], method="fdr_bh")[1]
    return q


def add(con: duckdb.DuckDBPyConnection) -> None:
    """Create temp table `surge` (pair, mi, k3, nv3, expected3, surge_p, surge_q, surge) from
    the `base` and `monthly` tables that signals.prepare builds."""
    s = SURGE_MONTHS - 1
    df = con.execute(f"""
        WITH v AS (SELECT make, model, model_year, mi, count(DISTINCT odi_number) AS k
                   FROM base GROUP BY ALL),
        v3 AS (SELECT make, model, model_year, mi,
                      sum(k) OVER (PARTITION BY make, model, model_year ORDER BY mi
                                   RANGE BETWEEN {s} PRECEDING AND CURRENT ROW) AS nv3
               FROM v),
        m AS (SELECT *, sum(k) OVER (PARTITION BY make, model, model_year, theme_id ORDER BY mi
                                     RANGE BETWEEN {s} PRECEDING AND CURRENT ROW) AS k3
              FROM monthly)
        SELECT m.make, m.model, m.model_year, m.theme_id, m.mi, m.k3, v3.nv3,
               m.a - m.k3 AS hist_a, m.nv - v3.nv3 AS hist_n, m.ct / m.n AS theme_share
        FROM m JOIN v3 USING (make, model, model_year, mi)
    """).df()
    expected, p = surge_test(df["k3"], df["nv3"], df["hist_a"], df["hist_n"], df["theme_share"])
    q = fdr_by_month(p, df["mi"].to_numpy()) if len(df) else p
    out = df[["make", "model", "model_year", "theme_id", "mi", "k3", "nv3"]].assign(
        expected3=expected, surge_p=p, surge_q=q,
        surge=(df["k3"].to_numpy() >= MIN_SURGE_COMPLAINTS) & (q <= MAX_FDR))
    con.register("surge_df", out)
    con.execute("CREATE OR REPLACE TEMP TABLE surge AS SELECT * FROM surge_df")
    con.unregister("surge_df")
