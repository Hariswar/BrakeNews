from datetime import date
from pathlib import Path

import duckdb
import pytest

from pipeline import clean
from pipeline.config import COMPLAINT_COLUMNS, RECALL_COLUMNS
from pipeline.ingest import tsv_to_parquet

COMPLAINT = dict(
    CMPLID="1", ODINO="11", MAKETXT=" honda ", MODELTXT="CR-V", YEARTXT="2021",
    COMPDESC="POWER TRAIN:AUTOMATIC TRANSMISSION", LDATE="20220115", CRASH="N", FIRE="N",
    INJURED="0", DEATHS="0", PROD_TYPE="V", CDESCR='Car "slips" when shifting',
    VIN="5J6RW2H5XM", VEHICLE_OPERATOR="JANE DOE", CITY="SPRINGFIELD", STATE="MO",
)
RECALL = dict(
    RECORD_ID="1", CAMPNO="22V123000", MAKETXT="HONDA", MODELTXT="CR-V", YEARTXT="2021",
    COMPNAME="POWER TRAIN:AUTOMATIC TRANSMISSION", RCLTYPECD="V", RCDATE="20220601",
    POTAFF="1000", DO_NOT_DRIVE="No", PARK_OUTSIDE="No",
)


def _row(columns, **values):
    return "\t".join(values.get(c, "") for c in columns)


def _build(tmp_path: Path, complaints: list[dict], recalls: list[dict]):
    """Write raw rows through ingest's TSV reader, run clean, return (connection, clean dir)."""
    raw, out = tmp_path / "raw", tmp_path / "clean"
    raw.mkdir()
    con = duckdb.connect()
    for name, columns, rows in (("complaints", COMPLAINT_COLUMNS, complaints),
                                ("recalls_post_2010", RECALL_COLUMNS, recalls)):
        txt = tmp_path / f"{name}.txt"
        txt.write_text("".join(_row(columns, **r) + "\n" for r in rows), encoding="utf-8")
        tsv_to_parquet(con, txt, columns, raw / f"{name}.parquet")
    clean.run(raw=raw, clean=out)
    return con, out


def test_column_counts_match_nhtsa_layouts():
    assert len(COMPLAINT_COLUMNS) == 51
    assert len(RECALL_COLUMNS) == 29


def test_ingest_and_clean_roundtrip(tmp_path: Path):
    tire = {**COMPLAINT, "CMPLID": "2", "PROD_TYPE": "T"}
    con, out = _build(tmp_path, [COMPLAINT, COMPLAINT, tire], [RECALL])

    rows = con.execute(f"SELECT * FROM '{out / 'complaints.parquet'}'").fetchdf()
    assert len(rows) == 1  # exact duplicate removed, tire complaint dropped
    r = rows.iloc[0]
    assert r.make == "HONDA" and r.model_year == 2021
    assert r.component_group == "POWER TRAIN"
    assert str(r.received_date)[:10] == "2022-01-15"
    assert r.description_raw == 'Car "slips" when shifting'
    assert "vin" not in rows.columns and "vehicle_operator" not in rows.columns
    values = {str(v) for v in rows.to_numpy().ravel()}
    assert not values & {"5J6RW2H5XM", "JANE DOE", "SPRINGFIELD"}  # no PII under any name

    tl = con.execute(f"SELECT * FROM '{out / 'recall_timeline.parquet'}'").fetchdf()
    assert list(tl.campaign_id) == ["22V123000"]
    assert list(tl.component_groups[0]) == ["POWER TRAIN"]


def test_ingest_skips_blank_lines_and_strips_crlf(tmp_path: Path):
    # The recall files use CRLF line endings and contain a few blank lines.
    txt = tmp_path / "r.txt"
    line = _row(RECALL_COLUMNS, **{**RECALL, "PARK_OUTSIDE": "Yes"})
    txt.write_text(f"{line}\r\n\r\n{line}\r\n", newline="")
    con = duckdb.connect()
    assert tsv_to_parquet(con, txt, RECALL_COLUMNS, tmp_path / "r.parquet") == 2
    assert con.execute(f"SELECT DISTINCT PARK_OUTSIDE FROM '{tmp_path / 'r.parquet'}'").fetchall() \
        == [("Yes",)]


def test_ingest_rejects_a_short_line_deep_in_the_file(tmp_path: Path):
    # Past DuckDB's sniffer sample, so this fails only if padding is off rather than being
    # silently filled with NULLs.
    good = _row(RECALL_COLUMNS, **RECALL) + "\n"
    txt = tmp_path / "r.txt"
    txt.write_text(good * 50_000 + "22V999000\tHONDA\n" + good * 10)
    with pytest.raises(duckdb.InvalidInputException):
        tsv_to_parquet(duckdb.connect(), txt, RECALL_COLUMNS, tmp_path / "r.parquet")


def test_reentered_complaints_keep_first_received_copy(tmp_path: Path):
    first = {**COMPLAINT, "CMPLID": "10", "LDATE": "20000105"}
    rereceived = {**first, "CMPLID": "9", "LDATE": "20000301"}  # lower id, received later
    other_issue = {**first, "CMPLID": "12", "CDESCR": "Engine gaskets failed."}
    other_part = {**first, "CMPLID": "13", "COMPDESC": "STEERING"}
    con, out = _build(tmp_path, [first, rereceived, other_issue, other_part], [RECALL])
    ids = con.execute(f"SELECT complaint_id FROM '{out / 'complaints.parquet'}' ORDER BY 1")
    assert [r[0] for r in ids.fetchall()] == ["10", "12", "13"]


def test_impossible_complaint_values_become_null(tmp_path: Path):
    # CMPLID -> (raw overrides, clean column, expected value). Received 2022-01-15, MY 2021.
    cases = {
        "1": ({"FAILDATE": "19010101"}, "fail_date", None),  # NHTSA's "unknown" sentinel
        "2": ({"FAILDATE": "20220120"}, "fail_date", None),  # after the complaint arrived
        "3": ({"FAILDATE": "20150101"}, "fail_date", None),  # years before the model year
        "4": ({"FAILDATE": "20200701"}, "fail_date", date(2020, 7, 1)),  # MY2021 sold in 2020
        "5": ({"VEH_SPEED": "0"}, "speed", 0),  # stationary, a real value
        "6": ({"VEH_SPEED": "999"}, "speed", None),
        "7": ({"VEH_SPEED": "-70"}, "speed", None),
        "8": ({"MILES": "-95000"}, "miles", None),
        "9": ({"MILES": "9999999"}, "miles", None),
        "10": ({"MILES": "45000"}, "miles", 45000),
        "11": ({"DEATHS": "99"}, "deaths", None),
        "12": ({"DEATHS": "2"}, "deaths", 2),
        "13": ({"INJURED": "16"}, "injured", 16),
        "14": ({"INJURED": "40"}, "injured", None),
        "15": ({"STATE": "00"}, "state", None),
        "16": ({"STATE": " ca"}, "state", "CA"),
        "17": ({"CRASH": "Y"}, "crash", True),
        "18": ({"COMPDESC": " lane departure: assist "}, "component", "LANE DEPARTURE:ASSIST"),
        "19": ({"COMPDESC": "ENGINE AND ENGINE COOLING:COOLING SYSTEM"}, "component_group",
               "ENGINE"),
        "20": ({"COMPDESC": "SERVICE BRAKES, HYDRAULIC:FOUNDATION COMPONENTS"},
               "component_group", "SERVICE BRAKES"),
    }
    complaints = [{**COMPLAINT, "CMPLID": k, "ODINO": k, **raw} for k, (raw, _, _) in cases.items()]
    con, out = _build(tmp_path, complaints, [RECALL])
    cur = con.execute(f"SELECT * FROM '{out / 'complaints.parquet'}'")
    cols = [d[0] for d in cur.description]
    got = {row[0]: dict(zip(cols, row, strict=True)) for row in cur.fetchall()}
    for cmplid, (_, col, want) in cases.items():
        assert got[cmplid][col] == want, (cmplid, col)


def test_recall_cleaning(tmp_path: Path):
    recall = {
        **RECALL, "DO_NOT_DRIVE": "Yes", "ODATE": "11111111", "BGMAN": "20210105",
        "ENDMAN": "30190430", "COMPNAME": "ENGINE AND ENGINE COOLING:ENGINE",
        "DESC_DEFECT": "THE PUMP\x92S SEAL MAY LEAK (\x93SEEP\x94).",
    }
    other_part_number = {**recall, "RECORD_ID": "2", "MFR_COMP_PTNO": "12345-AB"}
    con, out = _build(tmp_path, [COMPLAINT], [recall, other_part_number])

    rows = con.execute(f"""
        SELECT do_not_drive, owner_notice_date, mfg_begin, mfg_end, component_group,
               defect_summary
        FROM '{out / 'recalls.parquet'}'""").fetchall()
    assert rows == [(True, None, date(2021, 1, 5), None, "ENGINE",
                     "THE PUMP’S SEAL MAY LEAK (“SEEP”).")]
    tl = con.execute(f"""
        SELECT do_not_drive, park_outside, component_groups
        FROM '{out / 'recall_timeline.parquet'}'""").fetchall()
    assert tl == [(True, False, ["ENGINE"])]
