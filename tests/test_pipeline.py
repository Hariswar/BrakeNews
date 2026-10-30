from pathlib import Path

import duckdb

from pipeline import clean
from pipeline.config import COMPLAINT_COLUMNS, RECALL_COLUMNS
from pipeline.ingest import tsv_to_parquet


def _row(columns, **values):
    return "\t".join(values.get(c, "") for c in columns)


def test_column_counts_match_nhtsa_layouts():
    assert len(COMPLAINT_COLUMNS) == 51
    assert len(RECALL_COLUMNS) == 29


def test_ingest_and_clean_roundtrip(tmp_path: Path):
    raw, out = tmp_path / "raw", tmp_path / "clean"
    raw.mkdir()
    con = duckdb.connect()

    complaint = dict(
        CMPLID="1", ODINO="11", MAKETXT=" honda ", MODELTXT="CR-V", YEARTXT="2021",
        COMPDESC="POWER TRAIN:AUTOMATIC TRANSMISSION", LDATE="20220115", CRASH="N",
        INJURED="0", PROD_TYPE="V", CDESCR='Car "slips" when shifting', VIN="5J6RW2H5XM",
        VEHICLE_OPERATOR="JANE DOE",
    )
    tire = {**complaint, "CMPLID": "2", "PROD_TYPE": "T"}
    lines = [_row(COMPLAINT_COLUMNS, **complaint)] * 2 + [_row(COMPLAINT_COLUMNS, **tire)]
    (tmp_path / "c.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    tsv_to_parquet(con, tmp_path / "c.txt", COMPLAINT_COLUMNS, raw / "complaints.parquet")

    recall = dict(
        RECORD_ID="1", CAMPNO="22V123000", MAKETXT="HONDA", MODELTXT="CR-V", YEARTXT="2021",
        COMPNAME="POWER TRAIN:AUTOMATIC TRANSMISSION", RCLTYPECD="V", RCDATE="20220601",
        POTAFF="1000",
    )
    (tmp_path / "r.txt").write_text(_row(RECALL_COLUMNS, **recall) + "\n")
    tsv_to_parquet(con, tmp_path / "r.txt", RECALL_COLUMNS, raw / "recalls_post_2010.parquet")

    clean.run(raw=raw, clean=out)

    rows = con.execute(f"SELECT * FROM '{out / 'complaints.parquet'}'").fetchdf()
    assert len(rows) == 1  # exact duplicate removed, tire complaint dropped
    r = rows.iloc[0]
    assert r.make == "HONDA" and r.model_year == 2021
    assert r.component_group == "POWER TRAIN"
    assert str(r.received_date)[:10] == "2022-01-15"
    assert r.description_raw == 'Car "slips" when shifting'
    assert "vin" not in rows.columns and "vehicle_operator" not in rows.columns

    tl = con.execute(f"SELECT * FROM '{out / 'recall_timeline.parquet'}'").fetchdf()
    assert list(tl.campaign_id) == ["22V123000"]
    assert list(tl.component_groups[0]) == ["POWER TRAIN"]
