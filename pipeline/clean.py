"""Build typed, de-duplicated tables in data/clean from the raw Parquet files.

Outputs:
  complaints.parquet      one row per complaint x vehicle x component x narrative, vehicles only,
                          PII columns dropped
  recalls.parquet         one row per recall x vehicle x component (a handful have two rows
                          because NHTSA lists two production date ranges)
  recall_timeline.parquet one row per recall campaign with dates, vehicles and components

Values that cannot be right become NULL rather than being guessed at. The rules come from
profiling the October 2026 NHTSA files:
  - dates before 1949, the earliest model year in either file (sentinels such as 19010101 and
    11111111, typos such as 00220729)
  - incident dates after the complaint was received, or before the year preceding the model year
  - manufacturing dates more than a year after the recall report (typos such as 30190430)
  - miles outside 1-999,998 and speeds outside 0-160 mph (negatives, 999, 657); speed 0 means
    the vehicle was stationary and is kept
  - 10+ deaths or 21+ injuries in one complaint: on review nearly all are typos or junk entries
  - STATE values that are not USPS codes ('00', 'UN', '??')
Complaints re-entered under a new CMPLID (same ODI number, vehicle, component and narrative;
common in 1995-2002) keep only the first-received copy.
"""

from __future__ import annotations

from pathlib import Path

import duckdb

from pipeline.config import CLEAN, COMPONENT_GROUP_ALIASES, RAW, US_STATE_CODES

# Windows-1252 punctuation (’ “ ” – •) that reached the recall file as C1 control characters
# U+0080-U+009F. Bytes 81, 8D, 8F, 90 and 9D are undefined in cp1252 and are left alone.
_CP1252 = bytes(b for b in range(0x80, 0xA0) if b not in (0x81, 0x8D, 0x8F, 0x90, 0x9D))


def _lit(s: str) -> str:
    return "'" + s.replace("'", "''") + "'"


def _date(col: str) -> str:
    d = f"try_strptime(nullif(trim({col}), ''), '%Y%m%d')::DATE"
    return f"CASE WHEN {d} >= DATE '1949-01-01' THEN {d} END"


def _int(col: str) -> str:
    return f"try_cast(nullif(trim({col}), '') AS INTEGER)"


def _int_between(col: str, lo: int, hi: int) -> str:
    return f"CASE WHEN {_int(col)} BETWEEN {lo} AND {hi} THEN {_int(col)} END"


def _flag(col: str) -> str:
    # Complaints use Y/N; the recall DO_NOT_DRIVE and PARK_OUTSIDE fields use Yes/No.
    return f"(upper(trim({col})) IN ('Y', 'YES'))"


def _name(col: str) -> str:
    """Upper-case, trim and collapse internal whitespace (make/model/component names)."""
    return f"nullif(regexp_replace(upper(trim({col})), '\\s+', ' ', 'g'), '')"


def _year(col: str) -> str:
    return f"nullif({_int(col)}, 9999)"


def _component(col: str) -> str:
    # 'LANE DEPARTURE: ASSIST ' -> 'LANE DEPARTURE:ASSIST'
    return f"nullif(rtrim(regexp_replace({_name(col)}, '\\s*:\\s*', ':', 'g'), ':'), '')"


def _component_group(col: str) -> str:
    # 'ELECTRICAL SYSTEM:12V/24V/48V BATTERY' -> 'ELECTRICAL SYSTEM', then one name per group
    group = f"split_part({_component(col)}, ':', 1)"
    aliases = " ".join(
        f"WHEN {_lit(old)} THEN {_lit(new)}" for old, new in COMPONENT_GROUP_ALIASES.items()
    )
    return f"CASE {group} {aliases} ELSE nullif({group}, '') END"


def _text(col: str) -> str:
    return f"translate({col}, {_lit(_CP1252.decode('latin-1'))}, {_lit(_CP1252.decode('cp1252'))})"


def clean_complaints(con: duckdb.DuckDBPyConnection, src: Path, out: Path) -> int:
    states = ", ".join(_lit(s) for s in US_STATE_CODES)
    con.execute(f"""
        COPY (
            WITH typed AS (
                SELECT
                    trim(CMPLID)                        AS complaint_id,
                    trim(ODINO)                         AS odi_number,
                    {_name("MFR_NAME")}                 AS manufacturer,
                    {_name("MAKETXT")}                  AS make,
                    {_name("MODELTXT")}                 AS model,
                    {_year("YEARTXT")}                  AS model_year,
                    {_component("COMPDESC")}            AS component,
                    {_component_group("COMPDESC")}      AS component_group,
                    {_date("LDATE")}                    AS received_date,
                    {_date("FAILDATE")}                 AS fail_date,
                    {_date("DATEA")}                    AS added_date,
                    {_flag("CRASH")}                    AS crash,
                    {_flag("FIRE")}                     AS fire,
                    {_int_between("INJURED", 0, 20)}    AS injured,
                    {_int_between("DEATHS", 0, 9)}      AS deaths,
                    {_flag("MEDICAL_ATTN")}             AS medical_attention,
                    {_flag("VEHICLES_TOWED_YN")}        AS towed,
                    {_flag("POLICE_RPT_YN")}            AS police_report,
                    {_int_between("MILES", 1, 999_998)} AS miles,
                    {_int_between("VEH_SPEED", 0, 160)} AS speed,
                    CASE WHEN upper(trim(STATE)) IN ({states})
                         THEN upper(trim(STATE)) END    AS state,
                    nullif(trim(CMPL_TYPE), '')         AS complaint_type,
                    nullif(trim(FUEL_TYPE), '')         AS fuel_type,
                    nullif(trim(DRIVE_TRAIN), '')       AS drive_train,
                    nullif(trim(TRANS_TYPE), '')        AS transmission,
                    CDESCR                              AS description_raw
                FROM '{src.as_posix()}'
                WHERE upper(trim(PROD_TYPE)) = 'V'
            )
            SELECT * REPLACE (
                CASE WHEN fail_date <= received_date
                      AND (model_year IS NULL OR year(fail_date) >= model_year - 1)
                     THEN fail_date END AS fail_date
            )
            FROM typed
            QUALIFY row_number() OVER (
                PARTITION BY odi_number, make, model, model_year, component, description_raw
                ORDER BY received_date, try_cast(complaint_id AS BIGINT)
            ) = 1
        ) TO '{out.as_posix()}' (FORMAT parquet, COMPRESSION zstd)
    """)
    return con.execute(f"SELECT count(*) FROM '{out.as_posix()}'").fetchone()[0]


def clean_recalls(con: duckdb.DuckDBPyConnection, srcs: list[Path], out: Path) -> int:
    union = " UNION ALL ".join(f"SELECT * FROM '{p.as_posix()}'" for p in srcs)
    con.execute(f"""
        COPY (
            WITH typed AS (
                SELECT
                    trim(CAMPNO)                    AS campaign_id,
                    {_name("MAKETXT")}              AS make,
                    {_name("MODELTXT")}             AS model,
                    {_year("YEARTXT")}              AS model_year,
                    {_component("COMPNAME")}        AS component,
                    {_component_group("COMPNAME")}  AS component_group,
                    {_name("MFGNAME")}              AS manufacturer,
                    trim(RCLTYPECD)                 AS recall_type,
                    {_date("RCDATE")}               AS report_date,
                    {_date("ODATE")}                AS owner_notice_date,
                    {_date("BGMAN")}                AS mfg_begin,
                    {_date("ENDMAN")}               AS mfg_end,
                    {_int("POTAFF")}                AS potentially_affected,
                    nullif(trim(INFLUENCED_BY), '') AS influenced_by,
                    {_text("DESC_DEFECT")}          AS defect_summary,
                    {_text("CONEQUENCE_DEFECT")}    AS consequence_summary,
                    {_text("CORRECTIVE_ACTION")}    AS corrective_action,
                    {_flag("DO_NOT_DRIVE")}         AS do_not_drive,
                    {_flag("PARK_OUTSIDE")}         AS park_outside
                FROM ({union})
                WHERE upper(trim(RCLTYPECD)) = 'V'
            )
            -- DISTINCT also folds rows that differed only in dropped columns (part numbers, notes)
            SELECT DISTINCT * REPLACE (
                CASE WHEN mfg_begin <= report_date + INTERVAL 1 YEAR THEN mfg_begin END
                    AS mfg_begin,
                CASE WHEN mfg_end <= report_date + INTERVAL 1 YEAR THEN mfg_end END
                    AS mfg_end
            )
            FROM typed
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
            ORDER BY report_date, campaign_id
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
