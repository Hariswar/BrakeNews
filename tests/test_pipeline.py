from datetime import date
from pathlib import Path

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from pipeline import clean, embed, scrub
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


@pytest.mark.parametrize("raw, want", [
    ("CALL ME AT (314) 555-0182 OR EMAIL JDOE77@GMAIL.COM. *AK",
     "CALL ME AT [PHONE] OR EMAIL [EMAIL]."),
    ("VIN 1HGCM82633A004352, LAST 8 OF VIN 3A004352.*DT*AK", "VIN [VIN], LAST 8 OF VIN [VIN]."),
    ("MR. AND MRS. SMITH MET MR. JOHN W. SMITH. HE SAID NO.",
     "MR. AND MRS. [NAME] MET MR. [NAME]. HE SAID NO."),
    ("MR. SMITH CONTACTED ME ABOUT IT.", "MR. [NAME] CONTACTED ME ABOUT IT."),
    ("My name is maria lopez and I live at 1234 Oak St, Springfield.",
     "My name is [NAME] and I live at [ADDRESS], Springfield."),
    ("CASE NUMBER: 1-8765432109 WAS OPENED. CRM:98765432101234. PHONE# 702/555-1234.",
     "CASE NUMBER: [ID] WAS OPENED. CRM:[NUMBER]. PHONE# [PHONE]."),
    ("PHOTOS: HTTPS://WWW.FLICKR.COM/PHOTOS/JDOE/ALBUM-1 LICENSE PLATE ABC1234, ZIP CODE 65401.",
     "PHOTOS: [URL] LICENSE PLATE [PLATE], ZIP CODE [ZIP]."),
    # Non-breaking and doubled spaces must not hide details.
    ("CALLED (800)\xa0555-0199. MY  NAME IS DUANE  SMITH AND",
     "CALLED [PHONE]. MY NAME IS [NAME] AND"),
    ("GMy name is Duane Smith and I", "GMy name is [NAME] and I"),
    # Left alone: NHTSA's own redactions, model names, vehicle facts, recall and bulletin numbers,
    # "MS"/"DR" without a period, emphasis that looks like a reviewer code.
    ("MY NAME IS [XXX]. MY 2019 TOYOTA MR2 HAS 4 WHEEL DRIVE, 45,000 MILES, 65 MPH.",
     "MY NAME IS [XXX]. MY 2019 TOYOTA MR2 HAS 4 WHEEL DRIVE, 45,000 MILES, 65 MPH."),
    ("ESTIMATED VEHICLES AFFECTED: 234567  2019 HONDA CIVIC",
     "ESTIMATED VEHICLES AFFECTED: 234567 2019 HONDA CIVIC"),
    ("RECALL 15V123000, RECALL NUMBER 1234567890, TSB #123456789. THE AIRBAG DID *NOT* DEPLOY.",
     "RECALL 15V123000, RECALL NUMBER 1234567890, TSB #123456789. THE AIRBAG DID *NOT* DEPLOY."),
    ("MS GULF COAST, MISS FIRE ON CYL 3, HIS DR. APPOINTMENTS. *NO ANSWER REQUIRED",
     "MS GULF COAST, MISS FIRE ON CYL 3, HIS DR. APPOINTMENTS. *NO ANSWER REQUIRED"),
    ("MY 2004 CHEVROLET BLAZER, 4 DR. LT HAS A FUEL GAUGE PROBLEM",
     "MY 2004 CHEVROLET BLAZER, 4 DR. LT HAS A FUEL GAUGE PROBLEM"),
])
def test_scrub_masks_personal_details_only(raw: str, want: str):
    assert scrub.scrub(raw) == want


def test_scrub_and_embed_are_incremental(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(embed, "MIN_FREE_BYTES", 0)
    con = duckdb.connect()
    calls = []

    def write_complaints(rows):
        pq.write_table(pa.table({"complaint_id": [r[0] for r in rows],
                                 "description_raw": [r[1] for r in rows]}),
                       tmp_path / "complaints.parquet")

    def fake_encode(texts):
        calls.append(len(texts))
        return np.full((len(texts), 4), 0.5, dtype=np.float32)

    def vectors():
        return con.execute(f"""SELECT narrative_id, embedding FROM
            read_parquet('{tmp_path / "embeddings/fake--model/*.parquet"}')""").fetchall()

    # One complaint listed under two components repeats its text; empty texts are skipped.
    rows = [("1", "BRAKES FAILED. CALL 314-555-0182 *AK"),
            ("2", "BRAKES FAILED. CALL 314-555-0182"), ("3", None), ("4", "*TR")]
    write_complaints(rows)
    scrub.run(clean=tmp_path)
    got = con.execute(f"SELECT complaint_id, narrative_id, description FROM "
                      f"'{tmp_path / 'narratives.parquet'}' ORDER BY 1").fetchall()
    assert [(r[0], r[2]) for r in got] == [("1", "BRAKES FAILED. CALL [PHONE]"),
                                           ("2", "BRAKES FAILED. CALL [PHONE]")]
    assert got[0][1] == got[1][1]  # same text, one narrative_id, one vector

    embed.run(clean=tmp_path, model_name="fake/model", encode=fake_encode)
    embed.run(clean=tmp_path, model_name="fake/model", encode=fake_encode)  # nothing new to do
    assert calls == [1]
    write_complaints(rows + [("5", "ENGINE STALLED ON THE HIGHWAY.")])
    scrub.run(clean=tmp_path)
    embed.run(clean=tmp_path, model_name="fake/model", encode=fake_encode)
    assert calls == [1, 1]  # only the new text was encoded
    ids = {r[0] for r in con.execute(
        f"SELECT narrative_id FROM '{tmp_path / 'narratives.parquet'}'").fetchall()}
    assert {v[0] for v in vectors()} == ids and all(v[1] == [0.5] * 4 for v in vectors())
