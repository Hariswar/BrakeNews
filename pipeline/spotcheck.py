"""Phase 1 gate helper: print random clean records next to their raw source rows."""

from __future__ import annotations

import duckdb

from pipeline.config import CLEAN, RAW

COMPLAINT_FIELDS = (
    ("make", "MAKETXT"), ("model", "MODELTXT"), ("model_year", "YEARTXT"),
    ("component", "COMPDESC"), ("component_group", "COMPDESC"), ("received_date", "LDATE"),
    ("fail_date", "FAILDATE"), ("crash", "CRASH"), ("injured", "INJURED"),
    ("speed", "VEH_SPEED"), ("state", "STATE"),
)
RECALL_FIELDS = (
    ("manufacturer", "MFGNAME"), ("report_date", "RCDATE"), ("owner_notice_date", "ODATE"),
    ("potentially_affected", "POTAFF"), ("do_not_drive", "DO_NOT_DRIVE"),
    ("park_outside", "PARK_OUTSIDE"),
)


def _pairs(fields: tuple[tuple[str, str], ...]) -> str:
    return ", ".join(f"c.{a}, r.{b}" for a, b in fields)


def _show(con: duckdb.DuckDBPyConnection, label: str, sql: str,
          fields: tuple[tuple[str, str], ...]) -> None:
    for row in con.execute(sql).fetchall():
        print(f"\n{label} {row[0]}")
        for i, (a, _) in enumerate(fields):
            print(f"  {a:<20} clean={row[1 + 2 * i]!s:<40} raw={row[2 + 2 * i]!s}")


def run(n: int = 20, seed: int = 42) -> None:
    con = duckdb.connect()
    _show(con, "complaint", f"""
        SELECT c.complaint_id, {_pairs(COMPLAINT_FIELDS)}
        FROM (SELECT * FROM '{(CLEAN / "complaints.parquet").as_posix()}'
              USING SAMPLE {n} ROWS (reservoir, {seed})) c
        JOIN '{(RAW / "complaints.parquet").as_posix()}' r ON trim(r.CMPLID) = c.complaint_id
    """, COMPLAINT_FIELDS)
    # Campaign-level fields repeat on every raw row of a campaign, so any one row will do.
    raw_campaigns = ", ".join(f"max({b}) AS {b}" for _, b in RECALL_FIELDS)
    _show(con, "recall", f"""
        SELECT c.campaign_id, {_pairs(RECALL_FIELDS)}
        FROM (SELECT * FROM '{(CLEAN / "recall_timeline.parquet").as_posix()}'
              USING SAMPLE {n} ROWS (reservoir, {seed})) c
        JOIN (SELECT trim(CAMPNO) AS campaign_id, {raw_campaigns}
              FROM read_parquet('{(RAW / "recalls_*.parquet").as_posix()}') GROUP BY 1) r
          ON r.campaign_id = c.campaign_id
    """, RECALL_FIELDS)
