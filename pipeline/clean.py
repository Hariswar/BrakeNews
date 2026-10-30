"""Build typed, de-duplicated tables in data/clean from the raw Parquet files.

Outputs:
  complaints.parquet      one row per complaint x component, vehicles only, PII columns dropped
  recalls.parquet         one row per recall x vehicle x component
  recall_timeline.parquet one row per recall campaign with dates, vehicles and components
"""

from __future__ import annotations

from pathlib import Path

import duckdb

from pipeline.config import CLEAN, COMPLAINT_COLUMNS, COMPLAINT_PII_COLUMNS, RAW


def _date(col: str) -> str:
    return f"try_strptime(nullif(trim({col}), ''), '%Y%m%d')::DATE"


def _int(col: str) -> str:
    return f"try_cast(nullif(trim({col}), '') AS INTEGER)"


def _flag(col: str) -> str:
    return f"(upper(trim({col})) = 'Y')"


def _name(col: str) -> str:
    """Upper-case, trim and collapse internal whitespace (make/model/component names)."""
    return f"nullif(regexp_replace(upper(trim({col})), '\\s+', ' ', 'g'), '')"


def _year(col: str) -> str:
    return f"nullif({_int(col)}, 9999)"


def _component_group(col: str) -> str:
    # 'ELECTRICAL SYSTEM:12V/24V/48V BATTERY' -> 'ELECTRICAL SYSTEM'
    return f"nullif(trim(split_part({_name(col)}, ':', 1)), '')"


def clean_complaints(con: duckdb.DuckDBPyConnection, src: Path, out: Path) -> int:
    keep = [c for c in COMPLAINT_COLUMNS if c not in COMPLAINT_PII_COLUMNS]
    con.execute(f"""
        COPY (
            WITH dedup AS (
                SELECT DISTINCT {", ".join(keep)} FROM '{src.as_posix()}'
            )
            SELECT
                trim(CMPLID)                    AS complaint_id,
                trim(ODINO)                     AS odi_number,
                {_name("MFR_NAME")}             AS manufacturer,
                {_name("MAKETXT")}              AS make,
                {_name("MODELTXT")}             AS model,
                {_year("YEARTXT")}              AS model_year,
                {_name("COMPDESC")}             AS component,
                {_component_group("COMPDESC")}  AS component_group,
                {_date("LDATE")}                AS received_date,
                {_date("FAILDATE")}             AS fail_date,
                {_date("DATEA")}                AS added_date,
                {_flag("CRASH")}                AS crash,
                {_flag("FIRE")}                 AS fire,
                coalesce({_int("INJURED")}, 0)  AS injured,
                coalesce({_int("DEATHS")}, 0)   AS deaths,
                {_flag("MEDICAL_ATTN")}         AS medical_attention,
                {_flag("VEHICLES_TOWED_YN")}    AS towed,
                {_flag("POLICE_RPT_YN")}        AS police_report,
                nullif({_int("MILES")}, 0)      AS miles,
                nullif({_int("VEH_SPEED")}, 0)  AS speed,
                nullif(trim(STATE), '')         AS state,
                nullif(trim(CMPL_TYPE), '')     AS complaint_type,
                nullif(trim(FUEL_TYPE), '')     AS fuel_type,
                nullif(trim(DRIVE_TRAIN), '')   AS drive_train,
                nullif(trim(TRANS_TYPE), '')    AS transmission,
                CDESCR                          AS description_raw
            FROM dedup
            WHERE upper(trim(PROD_TYPE)) = 'V'
        ) TO '{out.as_posix()}' (FORMAT parquet, COMPRESSION zstd)
    """)
    return con.execute(f"SELECT count(*) FROM '{out.as_posix()}'").fetchone()[0]


def clean_recalls(con: duckdb.DuckDBPyConnection, srcs: list[Path], out: Path) -> int:
    union = " UNION ALL ".join(f"SELECT * FROM '{p.as_posix()}'" for p in srcs)
    con.execute(f"""
        COPY (
            WITH dedup AS (SELECT DISTINCT * EXCLUDE (RECORD_ID) FROM ({union}))
            SELECT
                trim(CAMPNO)                    AS campaign_id,
                {_name("MAKETXT")}              AS make,
                {_name("MODELTXT")}             AS model,
                {_year("YEARTXT")}              AS model_year,
                {_name("COMPNAME")}             AS component,
                {_component_group("COMPNAME")}  AS component_group,
                {_name("MFGNAME")}              AS manufacturer,
                trim(RCLTYPECD)                 AS recall_type,
                {_date("RCDATE")}               AS report_date,
                {_date("ODATE")}                AS owner_notice_date,
                {_date("BGMAN")}                AS mfg_begin,
                {_date("ENDMAN")}               AS mfg_end,
                {_int("POTAFF")}                AS potentially_affected,
                nullif(trim(INFLUENCED_BY), '') AS influenced_by,
                DESC_DEFECT                     AS defect_summary,
                CONEQUENCE_DEFECT               AS consequence_summary,
                CORRECTIVE_ACTION               AS corrective_action,
                {_flag("DO_NOT_DRIVE")}         AS do_not_drive,
                {_flag("PARK_OUTSIDE")}         AS park_outside
            FROM dedup
            WHERE upper(trim(RCLTYPECD)) = 'V'
        ) TO '{out.as_posix()}' (FORMAT parquet, COMPRESSION zstd)
    """)
    return con.execute(f"SELECT count(*) FROM '{out.as_posix()}'").fetchone()[0]


def build_recall_timeline(con: duckdb.DuckDBPyConnection, recalls: Path, out: Path) -> int:
    con.execute(f"""
        COPY (
            SELECT
                campaign_id,
                min(report_date)                        AS report_date,
                min(owner_notice_date)                  AS owner_notice_date,
                any_value(manufacturer)                 AS manufacturer,
                list_sort(list_distinct(list(make)))    AS makes,
                list_sort(list_distinct(list(model)))   AS models,
                min(model_year)                         AS first_model_year,
                max(model_year)                         AS last_model_year,
                list_sort(list_distinct(list(component_group))) AS component_groups,
                list_sort(list_distinct(list(component)))       AS components,
                max(potentially_affected)               AS potentially_affected,
                any_value(influenced_by)                AS influenced_by,
                any_value(defect_summary)               AS defect_summary,
                bool_or(do_not_drive)                   AS do_not_drive,
                bool_or(park_outside)                   AS park_outside
            FROM '{recalls.as_posix()}'
            GROUP BY campaign_id
            ORDER BY report_date
        ) TO '{out.as_posix()}' (FORMAT parquet, COMPRESSION zstd)
    """)
    return con.execute(f"SELECT count(*) FROM '{out.as_posix()}'").fetchone()[0]


def run(raw: Path = RAW, clean: Path = CLEAN) -> None:
    clean.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    n = clean_complaints(con, raw / "complaints.parquet", clean / "complaints.parquet")
    print(f"[clean] complaints: {n:,} rows")
    srcs = sorted(raw.glob("recalls_*.parquet"))
    n = clean_recalls(con, srcs, clean / "recalls.parquet")
    print(f"[clean] recalls: {n:,} rows")
    n = build_recall_timeline(con, clean / "recalls.parquet", clean / "recall_timeline.parquet")
    print(f"[clean] recall_timeline: {n:,} campaigns")
