"""Phase 1 gate helper: print random clean complaints next to their raw records."""

from __future__ import annotations

import duckdb

from pipeline.config import CLEAN, RAW

FIELDS = (
    ("make", "MAKETXT"), ("model", "MODELTXT"), ("model_year", "YEARTXT"),
    ("component", "COMPDESC"), ("received_date", "LDATE"), ("crash", "CRASH"),
    ("injured", "INJURED"),
)


def run(n: int = 20, seed: int = 42) -> None:
    con = duckdb.connect()
    rows = con.execute(f"""
        SELECT c.complaint_id, {", ".join(f"c.{a}, r.{b}" for a, b in FIELDS)}
        FROM (SELECT * FROM '{(CLEAN / "complaints.parquet").as_posix()}'
              USING SAMPLE {n} ROWS (reservoir, {seed})) c
        JOIN '{(RAW / "complaints.parquet").as_posix()}' r
          ON trim(r.CMPLID) = c.complaint_id AND trim(upper(r.COMPDESC)) = c.component
    """).fetchall()
    for row in rows:
        print(f"\ncomplaint {row[0]}")
        for i, (a, _) in enumerate(FIELDS):
            print(f"  {a:<14} clean={row[1 + 2 * i]!s:<40} raw={row[2 + 2 * i]!s}")
