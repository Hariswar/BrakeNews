import json
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import duckdb
import numpy as np
import pandas as pd
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
    ("TL* THE CONTACT OWNS A 2011 HONDA PILOT.", "THE CONTACT OWNS A 2011 HONDA PILOT."),
    ("DT*: THE CONTACT STATED.", "THE CONTACT STATED."),
    ("F** THIS CAR", "F** THIS CAR"),
    # Street addresses and ZIPs anywhere; names in sign-offs at the end.
    ("TOWED TO ELM CHEVROLET (301 E CHURCH ST, ELMIRA, NY 14901) AND P.O. BOX 8234, SC 29202.",
     "TOWED TO ELM CHEVROLET ([ADDRESS], ELMIRA, NY [ZIP]) AND [ADDRESS], SC [ZIP]."),
    ("PLEASE HELP. SINCERELY, SCOTT A. HEFFNER", "PLEASE HELP. SINCERELY, [NAME]"),
    ("INTERNET SITES THANK YOU STEVE OHMAN", "INTERNET SITES THANK YOU [NAME]"),
    ("IT IS SO THIN... THANKS ! Sharon.", "IT IS SO THIN... THANKS ! [NAME]."),
    # Left alone: ordinary sentences that look like sign-offs or addresses.
    ("I SINCERELY HOPE SO. ATTENTION IS RESPECTFULLY REQUESTED. THANK YOU FOR YOUR TIME.",
     "I SINCERELY HOPE SO. ATTENTION IS RESPECTFULLY REQUESTED. THANK YOU FOR YOUR TIME."),
    ("DROVE 45 MILES DOWN THE ROAD, 4 WHEEL DRIVE, 35 MPH ON HWY 70, 2019 HONDA CIVIC LX.",
     "DROVE 45 MILES DOWN THE ROAD, 4 WHEEL DRIVE, 35 MPH ON HWY 70, 2019 HONDA CIVIC LX."),
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


def _write_vectors(clean: Path, ids: list[str], vectors: np.ndarray):
    out = clean / "embeddings" / "fake--model"
    out.mkdir(parents=True, exist_ok=True)
    v = vectors.astype(np.float16)
    pq.write_table(pa.table({
        "narrative_id": ids,
        "embedding": pa.FixedSizeListArray.from_arrays(pa.array(v.reshape(-1)), v.shape[1]),
    }), out / "part-0.parquet")


def _themed_corpus(tmp_path: Path, per_theme: int = 40):
    """Two well-separated themes plus a stray narrative pointing elsewhere."""
    rng = np.random.default_rng(1)
    themes = {"BRAKES GRIND AND PEDAL SINKS": np.eye(8)[0], "AIRBAG LIGHT STAYS ON": np.eye(8)[1]}
    ids, texts, vecs = [], [], []
    for t, (text, base) in enumerate(themes.items()):
        for i in range(per_theme):
            ids.append(f"n{t}-{i:03d}")
            texts.append(f"{text} AFTER {i} MILES")
            v = base + rng.normal(0, 0.05, 8)
            vecs.append(v / np.linalg.norm(v))
    ids.append("stray")
    texts.append("RADIO STOPPED WORKING")
    vecs.append(np.eye(8)[5])
    pq.write_table(pa.table({"complaint_id": [f"c-{i}" for i in ids], "narrative_id": ids,
                             "description": texts}), tmp_path / "narratives.parquet")
    pq.write_table(pa.table({
        "complaint_id": [f"c-{i}" for i in ids], "odi_number": [f"o-{i}" for i in ids],
        "component_group": ["SERVICE BRAKES" if i.startswith("n0") else "AIR BAGS" for i in ids],
        "make": ["HONDA"] * len(ids), "model": ["CIVIC"] * len(ids),
    }), tmp_path / "complaints.parquet")
    _write_vectors(tmp_path, ids, np.array(vecs))
    return ids, texts, np.array(vecs)


def test_vehicle_offsets_remove_what_a_vehicle_shares(tmp_path: Path):
    from pipeline import cluster

    # Theme on axis 0 or 1; every FORD F-150 narrative also leans hard along axis 2.
    rows = [("a", "FORD", "F-150", [1, 0, 3]), ("b", "FORD", "F-150", [0, 1, 3]),
            ("c", "FORD", "RANGER", [1, 0, 0]), ("d", "KIA", "RIO", [0, 1, 0]),
            ("e", None, None, [1, 0, 0])]
    ids = [r[0] for r in rows]
    pq.write_table(pa.table({"complaint_id": ids, "narrative_id": ids, "description": ids}),
                   tmp_path / "narratives.parquet")
    pq.write_table(pa.table({"complaint_id": ids, "make": [r[1] for r in rows],
                             "model": [r[2] for r in rows]}), tmp_path / "complaints.parquet")
    v = np.array([r[3] for r in rows], dtype=np.float32)
    group, means = cluster.vehicle_offsets(tmp_path, np.array(ids), v, min_group=2)
    x = cluster.adjust(v, group, means)
    assert group[0] == group[1] != group[2]  # F-150 has its own mean; RANGER falls back to FORD
    assert group[3] == group[4] == len(means) - 1  # too few KIA rows, unknown make: unchanged
    assert v[0] @ v[1] / 10 == pytest.approx(0.9) and x[0] @ x[1] < 0  # alike via the vehicle
    assert np.allclose(x[3], [0, 1, 0]) and np.allclose(x[4], [1, 0, 0])


def _fit_cluster(tmp_path):
    from pipeline import cluster

    # The identity "reducer" stands in for UMAP: the toy vectors are already low-dimensional.
    cluster.run(clean=tmp_path, model_name="fake/model", refit=True, min_cluster_size=10,
                reducer=lambda x: x)
    return cluster


def test_cluster_fits_assigns_and_keeps_fit(tmp_path: Path):
    ids, texts, vecs = _themed_corpus(tmp_path)
    cluster = _fit_cluster(tmp_path)
    out = tmp_path / "clusters"
    cl = pq.read_table(out / "clusters.parquet").to_pylist()
    a = {r["narrative_id"]: r["cluster_id"]
         for r in pq.read_table(out / "assignments.parquet").to_pylist()}
    assert len(cl) == 2
    assert a["stray"] == -1  # far from every centroid
    for c in cl:
        members = [i for i, v in a.items() if v == c["cluster_id"]]
        theme = members[0][:2]
        assert all(m.startswith(theme) for m in members) and len(members) >= 38
        assert c["narratives"] == len(members) and c["complaints"] == len(members)
        assert c["component_groups"] == ["SERVICE BRAKES" if theme == "n0" else "AIR BAGS"]
        assert set(c["examples"]) <= set(members)
        assert ("airbag" in c["terms"]) == (theme == "n1")
        assert np.argmax(c["raw_centroid"]) == int(theme[1])  # the theme's own axis

    # A new complaint arrives: a plain run assigns it to the existing fit without refitting.
    fit_id = cl[0]["fit_id"]
    narr = pq.read_table(tmp_path / "narratives.parquet").to_pylist()
    pq.write_table(pa.Table.from_pylist(narr + [{"complaint_id": "c-new", "narrative_id": "new",
                                                 "description": texts[0]}]),
                   tmp_path / "narratives.parquet")
    # "gone" is a vector whose text a scrub rule has since changed: it must be skipped.
    _write_vectors(tmp_path, ids + ["new", "gone"], np.vstack([vecs, vecs[0], vecs[1]]))
    cluster.run(clean=tmp_path, model_name="fake/model",
                reducer=lambda x: pytest.fail("refit without --refit"))
    a = {r["narrative_id"]: r["cluster_id"]
         for r in pq.read_table(out / "assignments.parquet").to_pylist()}
    assert a["new"] == a["n0-000"] and "gone" not in a
    assert pq.read_table(out / "clusters.parquet").column("fit_id")[0].as_py() == fit_id


class _FakeBatches:
    """Stands in for client.messages.batches: answers each request after one poll."""

    def __init__(self):
        self.created, self.polls = [], 0

    def create(self, requests):
        self.created.append(requests)
        return SimpleNamespace(id=f"batch-{len(self.created)}")

    def retrieve(self, batch_id):
        self.polls += 1
        status = "ended" if self.polls % 2 == 0 else "in_progress"
        return SimpleNamespace(processing_status=status,
                               request_counts=SimpleNamespace(processing=1))

    def results(self, batch_id):
        for r in self.created[int(batch_id.split("-")[1]) - 1]:
            if "AIRBAG" in r["params"]["messages"][0]["content"] and len(self.created) == 1:
                yield SimpleNamespace(custom_id=r["custom_id"],
                                      result=SimpleNamespace(type="errored"))
                continue
            body = {"name": f"Theme of {r['custom_id']}", "summary": "Owners report it.",
                    "kind": "defect"}
            msg = SimpleNamespace(stop_reason="end_turn",
                                  content=[SimpleNamespace(type="text", text=json.dumps(body))])
            yield SimpleNamespace(custom_id=r["custom_id"],
                                  result=SimpleNamespace(type="succeeded", message=msg))


class _FakeMessages:
    """Stands in for client.messages: the batches above plus one theme-grouping request."""

    def __init__(self):
        self.batches, self.theme_requests = _FakeBatches(), []

    def create(self, **params):
        self.theme_requests.append(params)
        body = {"groups": [{"cluster_ids": [1, 0], "name": "Both toy themes"}]}
        return SimpleNamespace(stop_reason="end_turn",
                               content=[SimpleNamespace(type="text", text=json.dumps(body))])


def test_names_are_resumable_and_retry_failures(tmp_path: Path):
    from pipeline import names

    _themed_corpus(tmp_path)
    _fit_cluster(tmp_path)
    messages = _FakeMessages()
    batches = messages.batches
    client = SimpleNamespace(messages=messages)

    names.run(clean=tmp_path, model="fake-model", wait=False, client=client)  # submit only
    assert (tmp_path / "clusters" / "naming_batch.json").exists()
    req = batches.created[0][0]
    assert req["params"]["model"] == "fake-model"
    assert "Sample narratives:\n1. " in req["params"]["messages"][0]["content"]
    names.run(clean=tmp_path, model="fake-model", client=client, poll_seconds=0)  # same batch
    assert len(batches.created) == 1
    got = pq.read_table(tmp_path / "clusters" / "names.parquet").to_pylist()
    assert len(got) == 1 and got[0]["kind"] == "defect"  # the airbag request failed
    assert not messages.theme_requests  # themes wait until every cluster has a name

    names.run(clean=tmp_path, model="fake-model", client=client, poll_seconds=0)
    assert len(batches.created[1]) == 1  # only the failed cluster is asked again
    got = pq.read_table(tmp_path / "clusters" / "names.parquet").to_pylist()
    assert sorted(r["cluster_id"] for r in got) == [0, 1]
    themes = pq.read_table(tmp_path / "clusters" / "themes.parquet").to_pylist()
    assert [(t["theme_id"], t["theme_name"]) for t in themes] == [(0, "Both toy themes")] * 2
    assert "Theme of cluster-1" in messages.theme_requests[0]["messages"][0]["content"]
    names.run(clean=tmp_path, model="fake-model", client=client)  # nothing left to do
    assert len(batches.created) == 2 and len(messages.theme_requests) == 1


def test_theme_groups_ignore_bad_ids_and_repeats():
    from pipeline.names import group_themes

    names = [{"fit_id": "f", "cluster_id": i, "name": f"n{i}"} for i in range(5)]
    groups = [{"cluster_ids": [3, 1, 99], "name": "A"},  # 99 does not exist
              {"cluster_ids": [1, 2], "name": "B"},  # 1 is already in A: B has one member left
              {"cluster_ids": [4, 4], "name": "C"}]  # a repeat is not a group
    got = {t["cluster_id"]: (t["theme_id"], t["theme_name"]) for t in group_themes(names, groups)}
    assert got == {0: (0, "n0"), 1: (1, "A"), 2: (2, "n2"), 3: (1, "A"), 4: (4, "n4")}


def _signal_corpus(tmp_path: Path):
    """F-150 brake complaints all 2020 (recalled 2021-06), Honda airbag complaints elsewhere.
    Returns a fake encoder for the recall summaries."""
    # Cluster 0 = brakes (axis 0), cluster 1 = airbags (axis 1).
    out = tmp_path / "clusters"
    out.mkdir()
    eye = np.eye(4, dtype=np.float32)
    pq.write_table(pa.table({
        "fit_id": ["f", "f"], "cluster_id": pa.array([0, 1], pa.int32()),
        "raw_centroid": pa.FixedSizeListArray.from_arrays(pa.array(eye[:2].reshape(-1)), 4),
    }), out / "clusters.parquet")
    complaints, narratives, assignments = [], [], []

    def complain(make, model, year, cluster, when):
        i = str(len(complaints))
        complaints.append({"complaint_id": i, "odi_number": i, "make": make, "model": model,
                           "model_year": year, "received_date": when, "fail_date": when,
                           "crash": False, "fire": False, "injured": 0, "deaths": 0,
                           "towed": False, "miles": 1000 * int(i), "speed": 40,
                           "component": "SERVICE BRAKES" if cluster == 0 else "AIR BAGS"})
        text = ("THE BRAKE PEDAL WENT TO THE FLOOR" if cluster == 0
                else "THE AIRBAG LIGHT STAYS ON")
        narratives.append({"complaint_id": i, "narrative_id": f"n{i}", "description": text})
        assignments.append({"narrative_id": f"n{i}", "cluster_id": cluster})

    for m in range(1, 11):  # the F-150 files brake complaints all year
        complain("FORD", "F-150", 2020, 0, date(2020, m, 5))
    for v in ("CIVIC", "ACCORD", "CR-V", "PILOT"):  # everyone else mostly complains of airbags
        for m in range(1, 13):
            complain("HONDA", v, 2019, 1, date(2020, m, 9))
        complain("HONDA", v, 2019, 0, date(2020, 6, 9))
    complain("HONDA", "CIVIC", 2019, 1, date(2026, 1, 1))  # makes 2026 the last month of data
    pq.write_table(pa.Table.from_pylist(complaints), tmp_path / "complaints.parquet")
    pq.write_table(pa.Table.from_pylist(narratives), tmp_path / "narratives.parquet")
    pq.write_table(pa.Table.from_pylist(assignments), out / "assignments.parquet")
    summaries = ["THE BRAKE HOSE MAY LEAK.", "THE AIRBAG INFLATOR MAY RUPTURE."]
    pq.write_table(pa.table({
        "campaign_id": ["21V001000", "19V002000"], "defect_summary": summaries,
        "report_date": [date(2021, 6, 1), date(2019, 3, 1)],
        "makes": [["FORD"], ["HONDA"]], "models": [["F-150"], ["CIVIC"]],
        "first_model_year": [2020, 2019], "last_model_year": [2020, 2019],
        "potentially_affected": [1000, 2000],
        "components": [["SERVICE BRAKES:HYDRAULIC"], ["AIR BAGS:FRONTAL"]],
    }), tmp_path / "recall_timeline.parquet")
    pq.write_table(pa.table({
        "campaign_id": ["21V001000", "19V002000"], "make": ["FORD", "HONDA"],
        "model": ["F-150", "CIVIC"], "model_year": [2020, 2019],
        "report_date": [date(2021, 6, 1), date(2019, 3, 1)],
        "component": ["SERVICE BRAKES", "AIR BAGS"], "defect_summary": summaries,
    }), tmp_path / "recalls.parquet")

    def fake_encode(texts):
        return np.array([eye[0] if "BRAKE" in t else eye[1] for t in texts])
    return fake_encode


def test_signals_flag_disproportion_before_the_recall(tmp_path: Path):
    from pipeline import signals

    fake_encode = _signal_corpus(tmp_path)
    signals.run(clean=tmp_path, model_name="fake/model", encode=fake_encode)
    sig = {(r["model"], r["theme_id"]): r for r in pq.read_table(
        tmp_path / "signals" / "signals.parquet").to_pylist()}
    f150 = sig[("F-150", 0)]
    assert f150["onset_month"] == date(2020, 3, 1)  # the third complaint, PRR far above 2
    assert f150["recall_campaign"] == "21V001000" and f150["lead_months"] == 15
    assert f150["complaints_before_recall"] == 10 and not f150["active"]  # quiet since 2020
    assert sig[("CIVIC", 1)]["onset_month"] is None  # same share as everyone else
    assert sig[("CIVIC", 1)]["recalled_before_complaints"]  # recalled in 2019, before them
    assert sig[("CIVIC", 0)]["recall_campaign"] is None  # the airbag recall is not a brake one
    matches = pq.read_table(tmp_path / "signals" / "recall_themes.parquet").to_pylist()
    assert {(m["campaign_id"], m["theme_id"], m["method"]) for m in matches} == {
        ("21V001000", 0, "embedding"), ("19V002000", 1, "embedding")}
    bt = json.loads((tmp_path / "signals" / "backtest.json").read_text())
    assert bt["share flagged before the recall"] == 1.0
    assert bt["median months flagged ahead of the recall"] == 15
    # March 2020: three brake complaints out of the F-150's three, where the national brake
    # share so far is 3 / 15 = 0.2: p = 0.2^3 = 0.008, q = 0.04 among March's five tests.
    assert f150["surge_onset_month"] == date(2020, 3, 1) and not f150["surging"]
    assert sig[("CIVIC", 1)]["surge_onset_month"] is None
    assert bt["share of recalled pairs that surged before the recall"] == 1.0
    assert bt["median months surged ahead of the recall"] == 15

    calls = []
    signals.run(clean=tmp_path, model_name="fake/model",
                encode=lambda t: calls.append(t) or fake_encode(t))
    assert calls == []  # recall vectors come from the cache


def test_claude_recall_labels_replace_the_embedding_rule(tmp_path: Path):
    from pipeline import signals
    from pipeline.recall_match import LABELS_SCHEMA

    fake_encode = _signal_corpus(tmp_path)
    signals.run(clean=tmp_path, model_name="fake/model", encode=fake_encode)
    vectors = pq.read_table(tmp_path / "embeddings" / "fake--model" / "recalls.parquet")
    md5 = dict(zip(vectors["campaign_id"].to_pylist(), vectors["text_md5"].to_pylist(),
                   strict=True))

    def label(rows):
        pq.write_table(pa.Table.from_pylist(rows, schema=LABELS_SCHEMA),
                       tmp_path / "signals" / "recall_labels.parquet")
        signals.run(clean=tmp_path, model_name="fake/model", encode=fake_encode)
        sig = {(r["model"], r["theme_id"]): r for r in pq.read_table(
            tmp_path / "signals" / "signals.parquet").to_pylist()}
        return sig, {(m["campaign_id"], m["theme_id"], m["method"]) for m in pq.read_table(
            tmp_path / "signals" / "recall_themes.parquet").to_pylist()}

    # Claude says the brake-hose recall covers no theme: the F-150 pair is no longer recalled.
    row = {"fit_id": "f", "campaign_id": "21V001000", "text_md5": md5["21V001000"],
           "candidates": [0, 1], "theme_ids": [], "model": "m"}
    sig, matches = label([row])
    assert sig[("F-150", 0)]["recall_campaign"] is None
    assert matches == {("19V002000", 1, "embedding")}
    # A label for another fit, or for an older summary text, is ignored.
    for stale in ({**row, "fit_id": "old"}, {**row, "text_md5": "0" * 32}):
        sig, matches = label([stale])
        assert sig[("F-150", 0)]["recall_campaign"] == "21V001000"
    # Theme ids that do not exist (or are not defect themes) are dropped.
    sig, matches = label([{**row, "theme_ids": [0, 77]}])
    assert matches == {("21V001000", 0, "claude"), ("19V002000", 1, "embedding")}


def test_features_are_quarterly_snapshots_of_the_past(tmp_path: Path):
    from pipeline import features

    features.run(clean=tmp_path, model_name="fake/model", encode=_signal_corpus(tmp_path))
    rows = pq.read_table(tmp_path / "model" / "training.parquet").to_pylist()
    # Data runs to 2026-01, so snapshots fall in January, April, July and October.
    assert {r["month"].month for r in rows} == {1, 4, 7, 10}
    f150 = {r["month"]: r for r in rows if r["model"] == "F-150"}
    # Brake complaints Jan-Oct 2020, recalled 2021-06: snapshots up to the one before it.
    assert sorted(f150) == [date(2020, m, 1) for m in (1, 4, 7, 10)] + [
        date(2021, 1, 1), date(2021, 4, 1)]
    apr = f150[date(2020, 4, 1)]
    assert (apr["a"], apr["k3"], apr["k3_prev"], apr["k"]) == (4, 3, 1, 1)
    assert apr["signal"] == 1 and apr["signal_months"] == 2  # fired in March and April
    quiet = f150[date(2021, 1, 1)]  # no complaint since October
    assert (quiet["k"], quiet["k3"], quiet["k3_prev"], quiet["a"]) == (0, 0, 3, 10)
    assert quiet["quiet_months"] == 3 and f150[date(2021, 4, 1)]["quiet_months"] == 6
    assert all(r["label"] == 1 for r in f150.values())  # every snapshot within 24 months
    civic = [r for r in rows if r["model"] == "CIVIC" and r["theme_id"] == 0]
    # One brake complaint, June 2020: in the window of eight snapshots, Jul 2020 to Apr 2022.
    assert [r["month"] for r in civic][0] == date(2020, 7, 1) and len(civic) == 8
    assert [r["label"] for r in civic] == [0] * 8  # no brake recall for the Civic
    assert [r["theme_recalls"] for r in civic][3:5] == [0, 1]  # the F-150's, from 2021-06
    # Rows after a covering recall are dropped: the airbag recall (2019) predates every
    # Civic airbag complaint.
    assert not [r for r in rows if r["model"] == "CIVIC" and r["theme_id"] == 1]


def test_snapshot_months_end_at_the_last_month():
    from pipeline.features import snapshot_months

    assert snapshot_months(0, 10, 3) == [1, 4, 7, 10]
    assert snapshot_months(5, 5, 3) == [5]


def test_surge_compares_a_vehicle_with_its_own_past():
    from pipeline.surge import fdr_by_month, surge_test

    # History: 10 of 100 complaints about the theme. Recent: 10 of 20, or 2 of 20.
    expected, p = surge_test(k3=[10, 2], nv3=[20, 20], hist_a=[10, 10], hist_n=[100, 100],
                             theme_share=[0.1, 0.1])
    assert np.allclose(expected, 2.0) and p[0] < 1e-4 and p[1] > 0.3
    # No history: the expected share is the national one.
    expected, _ = surge_test(k3=[1], nv3=[10], hist_a=[0], hist_n=[0], theme_share=[0.3])
    assert np.isclose(expected[0], 3.0)
    # A vehicle that always complained about the theme cannot surge on it.
    _, p = surge_test(k3=[5], nv3=[5], hist_a=[500], hist_n=[500], theme_share=[0.1])
    assert p[0] > 0.5

    p = np.array([0.001, 0.02, 0.5, 0.001])
    q = fdr_by_month(p, np.array([1, 1, 1, 2]))
    assert np.all(q >= p) and q[3] == 0.001  # alone in its month: nothing to adjust for
    assert np.isclose(q[0], 0.003) and np.isclose(q[1], 0.03)


def test_model_splits_leave_a_horizon_gap():
    from pipeline.model import selection_ends, splits

    mi = pd.Series(range(0, 200))
    s = splits(mi, last_labelled=199, test_months=48, validation_months=36, gap=24)
    train, val, test = (mi[s[k]] for k in ("train", "validation", "test"))
    assert test.min() == 152 and test.max() == 199 and len(val) == 36
    assert test.min() - val.max() > 24 and val.min() - train.max() > 24
    # A fold ending earlier has a bounded test block.
    assert mi[splits(mi, 150, 48, 36, 24)["test"]].max() == 150
    # Selection folds end where the main validation block ends, then step back.
    assert selection_ends(199, folds=3, test_months=48, gap=24) == [127, 79, 31]
    assert selection_ends(199, 1, 48, 24)[0] == val.max()


def _learnable_training(tmp_path: Path) -> None:
    """Recalls follow fires, whatever the PRR says; the rule only looks at the PRR."""
    from pipeline.features import FEATURES

    rng = np.random.default_rng(0)
    n = 30_000
    mi = rng.integers(24_000, 24_200, n)
    fire = rng.random(n)
    df = pd.DataFrame({f: rng.random(n) for f in FEATURES})
    df = df.assign(make=rng.choice(["FORD", "KIA"], n), theme_id=rng.integers(0, 5, n),
                   fire_share=fire, mi=mi, signal=(rng.random(n) < 0.3).astype(int),
                   model="X", model_year=2020,
                   month=pd.to_datetime({"year": mi // 12, "month": mi % 12 + 1, "day": 1}),
                   label=(rng.random(n) < np.where(fire > 0.9, 0.3, 0.01)).astype(float))
    df.loc[df["mi"] > 24_175, "label"] = np.nan  # the last 24 months have no label yet
    (tmp_path / "model").mkdir()
    df.to_parquet(tmp_path / "model" / "training.parquet")


def test_model_beats_the_rule_when_risk_is_learnable(tmp_path: Path):
    pytest.importorskip("lightgbm")
    pytest.importorskip("mlflow")
    from pipeline import model

    _learnable_training(tmp_path)
    m = model.run(clean=tmp_path, models_dir=tmp_path / "models", runs_dir=tmp_path / "mlruns",
                  test_months=36, validation_months=24, max_rounds=200)
    t = m["test"]
    assert t["model_precision_at_rule_volume"] > 2 * t["rule_precision"]
    assert t["average_precision"] > 2 * t["base_rate"]
    assert m["feature_importance"]["fire_share"] == max(m["feature_importance"].values())
    scores = pd.read_parquet(tmp_path / "model" / "risk_scores.parquet")
    assert len(scores) and scores["risk"].is_monotonic_decreasing
    assert (tmp_path / "models" / "recall_risk.txt").exists()
    assert (tmp_path / "mlruns" / "mlflow.db").exists()


def test_selection_prefers_the_features_that_carry_the_risk(tmp_path: Path):
    pytest.importorskip("lightgbm")
    pytest.importorskip("mlflow")
    from pipeline import model

    _learnable_training(tmp_path)
    sel = model.select(clean=tmp_path, runs_dir=tmp_path / "mlruns",
                       feature_sets={"with_fire": ("fire_share", "prr", "k3"),
                                     "without_fire": ("prr", "k3")},
                       param_sets={"fast": model.PARAM_SETS["fast"]}, folds=2,
                       test_months=24, validation_months=24, max_rounds=200)
    assert sel["winner"]["feature_set"] == "with_fire"
    # Labels run to month 24175 (2014-08); the final test block is its last 24 months, from
    # 2012-09. Each fold's test block ends a horizon before that, then 24 months earlier.
    assert [t.split(" to ")[1] for t in sel["fold_test_blocks"]] == ["2010-08", "2008-08"]
    assert json.loads((tmp_path / "model" / "selection.json").read_text())["candidates"]


class _FakeClaude:
    """Scripted stand-in for anthropic.Anthropic(): each create() returns the next response."""

    def __init__(self, script):
        self.script, self.calls = list(script), []
        self.beta = SimpleNamespace(messages=SimpleNamespace(stream=self._stream))

    def _stream(self, **kwargs):
        resp = self._create(**kwargs)

        class Stream:
            def __enter__(self):
                return SimpleNamespace(get_final_message=lambda: resp)

            def __exit__(self, *exc):
                return False
        return Stream()

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        content = self.script.pop(0)
        return SimpleNamespace(
            content=content, stop_details=None,
            stop_reason="tool_use" if any(b.type == "tool_use" for b in content) else "end_turn",
            usage=SimpleNamespace(input_tokens=1000, output_tokens=200,
                                  cache_read_input_tokens=0, cache_creation_input_tokens=0))


def _report(ids, campaigns=("21V001000",)):
    return [SimpleNamespace(type="text", text=json.dumps({
        "headline": "Brake pedal sinks to the floor", "assessment": "likely_defect",
        "confidence": "medium", "summary": "s", "failure_pattern": "f", "safety_risk": "r",
        "evidence": [{"finding": "pedal to floor", "complaint_ids": list(ids)}],
        "related_recalls": [{"campaign_id": c, "relation": "covers_this_problem", "note": "n"}
                            for c in campaigns],
        "open_questions": []}))]


def test_investigator_reads_evidence_checks_citations_and_caches(tmp_path: Path):
    pytest.importorskip("langgraph")
    from pipeline import investigate, signals

    signals.run(clean=tmp_path, model_name="fake/model", encode=_signal_corpus(tmp_path))
    pair = investigate.Pair("FORD", "F-150", 2020, 0)
    use = lambda i, name, args: SimpleNamespace(type="tool_use", id=i, name=name, input=args)  # noqa: E731
    fake = _FakeClaude([
        [use("t1", "search_complaints", {"keyword": "pedal", "severe_only": False, "limit": 3,
                                         "offset": 0}),
         use("t2", "vehicle_recalls", {}), use("t3", "drop_tables", {}),
         use("t4", "theme_recalls", {"limit": 5}), use("t5", "compare_model_years", {}),
         use("t6", "other_vehicles_with_theme", {"limit": 5})],
        # A made-up complaint and recall, and a real complaint no tool showed it (newest first:
        # the search returned 9, 8 and 7).
        _report(["9", "999", "0"], ["21V001000", "99V999000"]),
        _report(["9", "8"]),
    ])
    rec = investigate.run(clean=tmp_path, pair=pair, client=fake, review_model=None)[0]

    first = fake.calls[0]
    assert first["output_config"]["format"]["schema"] == investigate.REPORT_SCHEMA
    assert first["tool_choice"] == {"type": "auto"}
    dossier = json.loads(first["messages"][0]["content"].split("(JSON):\n", 1)[1])
    assert dossier["complaints_with_narrative"]["complaints"] == 10
    assert dossier["signal"]["onset_month"] == "2020-03-01"
    # The tool results went back in one user message; the unknown tool came back as an error.
    results = fake.calls[1]["messages"][2]["content"]
    found = json.loads(results[0]["content"])
    assert found["matching"] == 10 and found["returned"] == 3
    assert found["complaints"][0]["narrative"] == "THE BRAKE PEDAL WENT TO THE FLOOR"
    assert json.loads(results[1]["content"])["recalls"][0]["campaign_id"] == "21V001000"
    assert results[2]["is_error"] and not any(r["is_error"] for r in results[:2] + results[3:])
    assert [r["campaign_id"] for r in json.loads(results[3]["content"])["recalls"]] == [
        "21V001000"]  # matched to the brake theme; the airbag recall is not
    assert [y["model_year"] for y in json.loads(results[4]["content"])["model_years"]] == [2020]
    # Honda brake complaints exist, but from 2020: none in the last two years.
    assert json.loads(results[5]["content"])["vehicles"] == []
    # The made-up citations were sent back once, then the corrected report passed.
    revise = fake.calls[2]["messages"][-1]["content"]
    assert "999" in revise and "99V999000" in revise and "21V001000" not in revise
    assert "never returned by a tool; read them (search_complaints) before citing them, or " \
           "drop them: 0" in revise
    assert rec["check"] == {"passed": True, "problems": [], "revisions": 1}
    assert rec["report"]["evidence"][0]["complaint_ids"] == ["9", "8"]
    assert rec["status"] == "unreviewed" and rec["review"]["rounds"] == 0
    assert [c["name"] for c in rec["tool_calls"]][:3] == [
        "search_complaints", "vehicle_recalls", "drop_tables"]
    saved = tmp_path / "investigations" / "FORD_F-150_2020_t0.json"
    assert json.loads(saved.read_text())["evidence_key"] == rec["evidence_key"]

    # Same evidence: skipped without calling the model.
    again = _FakeClaude([])
    assert investigate.run(clean=tmp_path, pair=pair, client=again,
                           review_model=None)[0]["created"] == rec["created"]
    assert again.calls == []


def _use(i, name, args):
    return SimpleNamespace(type="tool_use", id=i, name=name, input=args)


def _search(i="s", limit=5):
    return _use(i, "search_complaints", {"keyword": "", "severe_only": False, "limit": limit,
                                         "offset": 0})


def _review(verdict, checks, issues=(), assessment="likely_defect"):
    return [SimpleNamespace(type="text", text=json.dumps({
        "verdict": verdict, "assessment": assessment, "confidence": "medium",
        "finding_checks": [{"finding_number": i, "support": s, "note": f"note {i}"}
                           for i, s in checks],
        "issues": [{"severity": sev, "issue": text, "fix": "fix it", "complaint_ids": ids}
                   for sev, text, ids in issues],
        "summary": "review summary"}))]


def test_reviewer_fact_checks_the_report_and_sends_issues_back(tmp_path: Path):
    pytest.importorskip("langgraph")
    from pipeline import investigate, signals

    signals.run(clean=tmp_path, model_name="fake/model", encode=_signal_corpus(tmp_path))
    pair = investigate.Pair("FORD", "F-150", 2020, 0)
    revised = _report(["9", "7"])
    payload = json.loads(revised[0].text)
    payload["assessment"] = "possible_defect"
    payload["evidence"].append({"finding": "second", "complaint_ids": ["9"]})
    revised[0].text = json.dumps(payload)
    fake = _FakeClaude([
        [_search()],                      # investigator reads 9, 8, 7, 6, 5
        _report(["9", "8"]),              # first draft passes the citation check
        [_use("r1", "vehicle_recalls", {})],
        # The reviewer checks a finding that does not exist and cites a made-up complaint...
        _review("approved", [(0, "supported"), (3, "supported")],
                [("minor", "bad cite", ["999"])]),
        # ...fixes that, and asks for a revision.
        _review("approved", [(0, "partly_supported")],
                [("major", "complaint 7 contradicts finding 0", ["7"])],
                assessment="possible_defect"),
        revised,                          # investigator revises (7 was read through a tool)
        _review("approved", [(0, "supported"), (1, "supported")],
                assessment="possible_defect"),
    ])
    rec = investigate.run(clean=tmp_path, pair=pair, client=fake)[0]
    assert fake.script == []

    # The reviewer has a conversation of its own: the dossier, the numbered report and the
    # cited complaints in full; not the investigator's messages.
    first_review = fake.calls[2]
    assert first_review["system"] == investigate.REVIEW_SYSTEM
    assert first_review["output_config"]["format"]["schema"] == investigate.REVIEW_SCHEMA
    assert len(first_review["messages"]) == 1
    brief = first_review["messages"][0]["content"]
    assert brief.startswith("Fact-check this report.")
    report_shown = json.loads(brief.split("Report (JSON):\n", 1)[1].split("\n\nCited", 1)[0])
    assert report_shown["evidence"][0]["finding_number"] == 0
    cited = json.loads(brief.split("Cited complaints (JSON, full narratives):\n", 1)[1])
    assert [c["odi_number"] for c in cited] == ["9", "8"]
    assert cited[0]["narrative"] == "THE BRAKE PEDAL WENT TO THE FLOOR"
    # Its own mistakes went back to it once.
    fix = fake.calls[4]["messages"][-1]["content"]
    assert "there is no finding 3" in fix and "999" in fix
    # The issues went to the investigator, in its own conversation.
    feedback = fake.calls[5]["messages"][-1]["content"]
    assert fake.calls[5]["system"] == investigate.SYSTEM
    assert "Finding 0 is partly supported: note 0" in feedback
    assert "[major] complaint 7 contradicts finding 0 (complaints 7)" in feedback
    assert "would assess this as possible_defect" in feedback
    # The second review sees the revised report and only the newly cited complaint.
    second = fake.calls[6]["messages"]
    assert len(second) == 7 and second[-1]["content"].startswith("The analyst revised")
    new = json.loads(second[-1]["content"].split("Newly cited complaints (JSON):\n", 1)[1])
    assert [c["odi_number"] for c in new] == ["7"]

    assert rec["status"] == "approved" and rec["report"]["assessment"] == "possible_defect"
    rv = rec["review"]
    assert rv["verdict"] == "approved" and rv["rounds"] == 2 and rv["agrees"]
    first = rv["reviews"][0]
    assert first["verdict"] == "needs_revision" and first["verdict_given"] == "approved"
    assert first["problems"] == [] and first["of_report"]["assessment"] == "likely_defect"
    assert rec["review_tool_calls"] == [{"name": "vehicle_recalls", "input": {}}]
    by_agent = rec["usage"]["by_agent"]
    assert by_agent["investigator"]["input_tokens"] == 3000
    assert by_agent["reviewer"]["input_tokens"] == 4000
    assert rec["usage"]["estimated_usd"] == pytest.approx(7 * (1000 * 4 + 200 * 20) / 1e6)
    index = pd.read_parquet(tmp_path / "investigations" / "index.parquet")
    assert index[["slug", "status", "assessment", "review_rounds"]].values.tolist() == [
        ["FORD_F-150_2020_t0", "approved", "possible_defect", 2]]


def test_reviewer_disputes_after_its_last_round(tmp_path: Path):
    pytest.importorskip("langgraph")
    from pipeline import investigate, signals

    signals.run(clean=tmp_path, model_name="fake/model", encode=_signal_corpus(tmp_path))
    unsupported = _review("approved", [(0, "not_supported")], assessment="not_a_defect")
    fake = _FakeClaude([[_search()], _report(["9"]), unsupported, _report(["9"]), unsupported])
    rec = investigate.run(clean=tmp_path, pair=investigate.Pair("FORD", "F-150", 2020, 0),
                          client=fake)[0]
    assert fake.script == [] and len(fake.calls) == 5
    assert rec["status"] == "disputed" and rec["review"]["rounds"] == 2
    assert rec["review"]["agrees"] is False and rec["review"]["assessment"] == "not_a_defect"


def test_minor_issues_of_an_approved_report_are_fixed_without_another_review(tmp_path: Path):
    pytest.importorskip("langgraph")
    from pipeline import investigate, signals

    signals.run(clean=tmp_path, model_name="fake/model", encode=_signal_corpus(tmp_path))
    fake = _FakeClaude([
        [_search()], _report(["9", "8"]),
        _review("approved", [(0, "partly_supported")], [("minor", "8 does not fit", ["8"])]),
        _report(["9", "999"]),  # the edit cites a made-up complaint: checked, sent back once
        _report(["9"]),
    ])
    rec = investigate.run(clean=tmp_path, pair=investigate.Pair("FORD", "F-150", 2020, 0),
                          client=fake)[0]
    assert fake.script == [] and len(fake.calls) == 5
    edit = fake.calls[3]["messages"][-1]["content"]
    assert edit.startswith("A second reader fact-checked your report and approved it")
    assert "[minor] 8 does not fit (complaints 8)" in edit
    assert "999" in fake.calls[4]["messages"][-1]["content"]
    assert rec["status"] == "approved" and rec["review"]["minor_fixes_applied"]
    assert rec["review"]["rounds"] == 1 and rec["report"]["evidence"][0]["complaint_ids"] == ["9"]
    assert rec["check"] == {"passed": True, "problems": [], "revisions": 1}


def test_batch_runs_pairs_within_budget_and_survives_failures(tmp_path: Path):
    pytest.importorskip("langgraph")
    import anthropic

    from pipeline import investigate, signals

    signals.run(clean=tmp_path, model_name="fake/model", encode=_signal_corpus(tmp_path))
    ford = investigate.Pair("FORD", "F-150", 2020, 0)
    empty = investigate.Pair("FORD", "F-150", 2020, 1)  # no airbag complaints on the F-150
    civic = investigate.Pair("HONDA", "CIVIC", 2019, 1)
    fake = _FakeClaude([[_search()], _report(["9"])])  # $0.016 for the F-150
    out = investigate.run(clean=tmp_path, pairs=[empty, ford, civic], client=fake,
                          review_model=None, workers=1, budget=0.01)
    assert "no complaints" in out[0]["error"] and out[1]["report"]["headline"]
    assert out[2] == {"pair": civic._asdict(), "skipped": "budget $0.01 reached"}
    index = pd.read_parquet(tmp_path / "investigations" / "index.parquet")
    assert index["slug"].tolist() == ["FORD_F-150_2020_t0"]

    # A lasting API error (no credit, bad key) stops the pairs not yet started.
    import httpx2

    def refused(msg):
        req = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
        return anthropic.BadRequestError(msg, response=httpx2.Response(400, request=req),
                                         body=None)

    class NoCredit(_FakeClaude):
        def _create(self, **kwargs):
            raise refused("credit balance is too low")

    out = investigate.run(clean=tmp_path, pairs=[civic, ford], client=NoCredit([]),
                          review_model=None, workers=1, force=True)
    assert "BadRequestError" in out[0]["error"]
    assert out[1] == {"pair": ford._asdict(), "skipped": "stopped after an API error"}


def test_batch_keeps_going_after_network_errors_and_counts_spend_of_failed_pairs(
        tmp_path: Path, monkeypatch):
    pytest.importorskip("langgraph")
    import anthropic

    from pipeline import investigate, signals

    monkeypatch.setattr(investigate, "RETRY_SECONDS", 0)
    signals.run(clean=tmp_path, model_name="fake/model", encode=_signal_corpus(tmp_path))
    ford = investigate.Pair("FORD", "F-150", 2020, 0)
    civic = investigate.Pair("HONDA", "CIVIC", 2019, 1)

    # A network drop is retried, mid-stream included; if it lasts, only that pair fails.
    class Down(_FakeClaude):
        def _create(self, **kwargs):
            self.calls.append(kwargs)
            raise anthropic.APIConnectionError(message="down", request=None)

    down = Down([])
    out = investigate.run(clean=tmp_path, pairs=[civic, ford], client=down,
                          review_model=None, workers=1)
    assert ["APIConnectionError" in r["error"] for r in out] == [True, True]
    assert len(down.calls) == 2 * investigate.ATTEMPTS

    class ReadTimeout(Exception):  # what httpx2 raises when a stream goes quiet
        pass

    class Flaky(_FakeClaude):
        def _create(self, **kwargs):
            if not self.calls:
                self.calls.append(kwargs)
                raise ReadTimeout("stream went quiet")
            return super()._create(**kwargs)

    flaky = Flaky([[_search()], _report(["9"])])
    rec = investigate.run(clean=tmp_path, pair=ford, client=flaky, review_model=None)[0]
    assert rec["check"]["passed"] and len(flaky.calls) == 3

    # Spend is counted per response, so a pair that fails half way still uses up budget.
    def declines(self, **kwargs):
        if self.calls:
            self.calls.append(kwargs)
            return SimpleNamespace(content=[], stop_reason="refusal", stop_details="cyber",
                                   usage=SimpleNamespace(input_tokens=1000, output_tokens=0,
                                                         cache_read_input_tokens=0,
                                                         cache_creation_input_tokens=0))
        return _FakeClaude._create(self, **kwargs)

    half = _FakeClaude([[_search()]])
    half._create = declines.__get__(half)
    out = investigate.run(clean=tmp_path, pairs=[civic, ford], client=half,
                          review_model=None, workers=1, budget=0.01, force=True)
    assert "declined" in out[0]["error"]  # $0.008 + $0.004 spent before it failed
    assert out[1] == {"pair": ford._asdict(), "skipped": "budget $0.01 reached"}


def test_complaints_can_be_read_in_full_and_searched_literally(tmp_path: Path):
    pytest.importorskip("langgraph")
    from pipeline import investigate, signals

    signals.run(clean=tmp_path, model_name="fake/model", encode=_signal_corpus(tmp_path))
    ev = investigate.Evidence(tmp_path, investigate.Pair("FORD", "F-150", 2020, 0))
    got = ev.run_tool("read_complaints", {"odi_numbers": ["3", "10", "3"]})
    assert [c["odi_number"] for c in got["complaints"]] == ["3"]
    assert got["not_about_this_vehicle_and_theme"] == ["10"]  # a Honda complaint
    found = ev.search_complaints(keyword="pedal")
    assert found["matching"] == 10 and not found["complaints"][0]["truncated"]
    assert ev.search_complaints(keyword="BRAKE%FLOOR")["matching"] == 0  # no wildcards
    assert ev.search_complaints(keyword="b_ake")["matching"] == 0


def test_an_edit_after_approval_that_changes_the_conclusion_is_not_approved():
    from pipeline import investigate

    review = {"verdict": "approved",
              "of_report": {"assessment": "likely_defect", "confidence": "high"}}
    report = {"assessment": "likely_defect", "confidence": "high"}
    assert investigate._status([review], report) == "approved"
    assert investigate._status([review], {**report, "confidence": "medium"}) == \
        "changed_after_review"
    assert investigate._status([{**review, "verdict": "needs_revision"}], report) == "disputed"
    assert investigate._status([], report) == "unreviewed"


class _FakeBatchAPI:
    """Stand-in for client.messages.batches: answer(params) gives each request's JSON."""

    def __init__(self, answer, fail=()):
        self.answer, self.fail, self.created, self.batches = answer, set(fail), [], {}
        self.messages = SimpleNamespace(batches=self)

    def create(self, requests):
        bid = f"b{len(self.created)}"
        self.created.append(requests)
        self.batches[bid] = requests
        return SimpleNamespace(id=bid)

    def retrieve(self, bid):
        return SimpleNamespace(processing_status="ended",
                               request_counts=SimpleNamespace(processing=0))

    def results(self, bid):
        for r in self.batches[bid]:
            if r["custom_id"] in self.fail:
                yield SimpleNamespace(custom_id=r["custom_id"],
                                      result=SimpleNamespace(type="errored"))
                continue
            text = json.dumps(self.answer(r["params"]))
            msg = SimpleNamespace(stop_reason="end_turn",
                                  content=[SimpleNamespace(type="text", text=text)])
            yield SimpleNamespace(custom_id=r["custom_id"],
                                  result=SimpleNamespace(type="succeeded", message=msg))


def _name_corpus(tmp_path: Path):
    """The signal corpus with both clusters named as defects."""
    encode = _signal_corpus(tmp_path)
    pq.write_table(pa.table({
        "fit_id": ["f", "f"], "cluster_id": pa.array([0, 1], pa.int32()),
        "name": ["Brake pedal sinks", "Airbag light on"],
        "summary": ["Owners report the brake pedal sinking.", "Owners report airbag lights."],
        "kind": ["defect", "defect"], "model": ["m", "m"],
    }), tmp_path / "clusters" / "names.parquet")
    return encode


def test_recall_match_labels_once_and_resumes(tmp_path: Path):
    from pipeline import recall_match, signals

    encode = _name_corpus(tmp_path)
    prompts = {}

    def answer(params):
        text = params["messages"][0]["content"]
        prompts[text.split("\n", 1)[0]] = text
        # The brake recall: theme 0, plus an id that was not on the shortlist.
        return {"theme_ids": [0, 5] if "BRAKE" in text else []}

    fake = _FakeBatchAPI(answer)
    assert not recall_match.run(clean=tmp_path, model_name="fake/model", encode=encode,
                                client=fake, wait=False)  # submitted, not collected
    assert (tmp_path / "signals" / "recall_match_batch.json").exists()
    assert recall_match.run(clean=tmp_path, model_name="fake/model", encode=encode, client=fake)
    assert len(fake.created) == 1 and len(fake.created[0]) == 2  # collected, not resubmitted
    # Each recall saw only its shortlist: the brake recall is nowhere near the airbag theme.
    assert "0 | Brake pedal sinks" in prompts["Recall 21V001000"]
    assert "Airbag" not in prompts["Recall 21V001000"]
    labels = {r["campaign_id"]: r["theme_ids"] for r in pq.read_table(
        tmp_path / "signals" / "recall_labels.parquet").to_pylist()}
    assert labels == {"21V001000": [0], "19V002000": []}

    # Nothing new: no request at all.
    assert recall_match.run(clean=tmp_path, model_name="fake/model", encode=encode,
                            client=_FakeBatchAPI(lambda p: pytest.fail("asked again")))
    signals.run(clean=tmp_path, model_name="fake/model", encode=encode)
    sig = {(r["model"], r["theme_id"]): r for r in pq.read_table(
        tmp_path / "signals" / "signals.parquet").to_pylist()}
    assert sig[("F-150", 0)]["recall_campaign"] == "21V001000"
    # Claude said the airbag recall covers no theme, so the Civic airbag pair is unrecalled.
    assert not sig[("CIVIC", 1)]["recalled_before_complaints"]


def test_verify_confirms_only_alerts_whose_complaints_fit_the_theme(tmp_path: Path):
    from pipeline import signals, verify

    encode = _name_corpus(tmp_path)
    # Recent complaints: a Camry with real brake failures, a Fit whose "brake" complaints are
    # mostly about other drivers' headlights, and airbag complaints that keep shares honest.
    tables = {k: pq.read_table(tmp_path / f).to_pylist() for k, f in (
        ("complaints", "complaints.parquet"), ("narratives", "narratives.parquet"),
        ("assignments", "clusters/assignments.parquet"))}

    def add(model, cluster, when, text):
        i = f"x{len(tables['complaints'])}"
        tables["complaints"].append({**tables["complaints"][0], "complaint_id": i,
                                     "odi_number": i, "make": "TOYOTA" if model == "CAMRY"
                                     else "HONDA" if model == "FIT" else "KIA",
                                     "model": model, "model_year": 2024, "received_date": when,
                                     "fail_date": when})
        tables["narratives"].append({"complaint_id": i, "narrative_id": f"n{i}",
                                     "description": text})
        tables["assignments"].append({"narrative_id": f"n{i}", "cluster_id": cluster})

    for m in range(9, 14):
        add("CAMRY", 0, date(2025 + m // 13, (m - 1) % 12 + 1, 3),
            "BRAKE PEDAL SANK" if m < 12 else "ONCOMING HEADLIGHTS BLIND ME")
    for m in range(10, 14):
        add("FIT", 0, date(2025 + m // 13, (m - 1) % 12 + 1, 4),
            "BRAKE PEDAL SANK" if m == 10 else "ONCOMING HEADLIGHTS BLIND ME")
    for d in range(1, 21):
        add("SOUL", 1, date(2025, 6, d), "AIRBAG LIGHT ON")
    for k, f in (("complaints", "complaints.parquet"), ("narratives", "narratives.parquet"),
                 ("assignments", "clusters/assignments.parquet")):
        pq.write_table(pa.Table.from_pylist(tables[k]), tmp_path / f)
    signals.run(clean=tmp_path, model_name="fake/model", encode=encode)

    def judge(params):
        return {"on_theme": "no" if "BLIND" in params["messages"][0]["content"] else "yes"}

    soul = sorted(r["narrative_id"] for r in tables["narratives"]
                  if r["description"] == "AIRBAG LIGHT ON")
    fake = _FakeBatchAPI(judge, fail={f"t1-{soul[0]}"})  # one answer goes missing
    alerts = verify.run(clean=tmp_path, client=fake)
    got = {r["model"]: r for r in alerts.to_dict("records")}
    assert set(got) == {"CAMRY", "FIT", "SOUL"}  # the active signals
    assert (got["CAMRY"]["on_theme"], got["CAMRY"]["off_theme"]) == (3, 2)
    assert got["CAMRY"]["status"] == "confirmed"
    assert (got["FIT"]["on_theme"], got["FIT"]["off_theme"]) == (1, 3)
    assert got["FIT"]["status"] == "not_confirmed"
    # 19 on theme and one unanswered: confirmed whatever the missing answer would be.
    assert (got["SOUL"]["on_theme"], got["SOUL"]["unchecked"]) == (19, 1)
    assert got["SOUL"]["status"] == "confirmed"
    # Every complaint was asked once, about its own theme, with the theme described.
    asked = [r["custom_id"] for r in fake.created[0]]
    assert len(asked) == len(set(asked)) == 29
    assert sum(a.startswith("t0-") for a in asked) == 9
    brake = next(r for r in fake.created[0] if r["custom_id"].startswith("t0-"))
    assert "Brake pedal sinks" in brake["params"]["messages"][0]["content"]
    # A rerun asks only for the missing answer.
    again = _FakeBatchAPI(judge)
    alerts = verify.run(clean=tmp_path, client=again)
    assert [r["custom_id"] for r in again.created[0]] == [f"t1-{soul[0]}"]
    assert alerts.set_index("model").loc["SOUL", "on_theme"] == 20


@pytest.mark.parametrize("on, off, unchecked, want", [
    (3, 2, 0, "confirmed"), (3, 4, 0, "mixed"), (2, 0, 0, "not_confirmed"),
    (5, 0, 2, "confirmed"),       # whatever the two missing answers say
    (2, 0, 1, "unchecked"),       # the missing answer decides
    (0, 9, 2, "not_confirmed"),   # cannot reach three on theme
])
def test_alert_status(on, off, unchecked, want):
    from pipeline.verify import status

    assert status(on, off, unchecked) == want


def test_investigator_picks_confirmed_alerts_by_risk(tmp_path: Path):
    from pipeline import investigate

    (tmp_path / "model").mkdir()
    pd.DataFrame({
        "make": ["FORD", "JEEP", "KIA", "HONDA"], "model": ["F-150", "WRANGLER", "SOUL", "FIT"],
        "model_year": [2024, 2024, 2024, 2024], "theme_id": [41, 120, 7, 3],
        "risk": [0.41, 0.27, 0.30, None],
        "status": ["not_confirmed", "confirmed", "confirmed", "confirmed"],
    }).to_parquet(tmp_path / "model" / "alerts.parquet")
    got = investigate.top_pairs(tmp_path, n=5)
    assert [p.model for p in got] == ["SOUL", "WRANGLER"]  # no glare campaign, no unscored


def test_batch_results_download_is_retried(monkeypatch):
    from pipeline import batch

    class ReadTimeout(Exception):
        pass

    calls = []

    def flaky(client, batch_id):
        calls.append(batch_id)
        if len(calls) == 1:
            raise ReadTimeout("dropped mid-stream")
        return {"a": {}}, 0

    monkeypatch.setattr(batch, "_read_results", flaky)
    monkeypatch.setattr(batch.time, "sleep", lambda s: None)
    assert batch._results(None, "b1") == ({"a": {}}, 0) and len(calls) == 2
    monkeypatch.setattr(batch, "_read_results",
                        lambda c, b: (_ for _ in ()).throw(ValueError("bad")))
    with pytest.raises(ValueError):  # not a network error: no retry
        batch._results(None, "b1")


def test_labelling_survives_an_api_refusal(tmp_path: Path):
    anthropic = pytest.importorskip("anthropic")
    from pipeline import batch

    class Broke:
        class messages:  # noqa: N801
            class batches:  # noqa: N801
                @staticmethod
                def create(requests):
                    raise anthropic.APIConnectionError(request=None)

    stored = []
    ok = batch.label_or_warn(Broke(), tmp_path / "state.json", "t",
                             lambda: [{"custom_id": "a"}], stored.append)
    assert ok is False and stored == [] and not (tmp_path / "state.json").exists()


def test_export_publishes_only_approved_reports_on_confirmed_alerts(tmp_path: Path):
    pytest.importorskip("langgraph")
    from pipeline import export, investigate, signals

    signals.run(clean=tmp_path, model_name="fake/model", encode=_signal_corpus(tmp_path))
    fake = _FakeClaude([[_search()], _report(["9", "8"]),
                        _review("approved", [(0, "supported")])])
    investigate.run(clean=tmp_path, pair=investigate.Pair("FORD", "F-150", 2020, 0), client=fake)
    reports = tmp_path / "investigations"
    rec = json.loads((reports / "FORD_F-150_2020_t0.json").read_text())
    rec["report"]["summary"] = "Owner wrote: call me at 555-123-4567 any time."
    (reports / "FORD_F-150_2020_t0.json").write_text(json.dumps(rec))
    disputed = {**rec, "pair": {**rec["pair"], "model_year": 2021}, "status": "disputed"}
    (reports / "FORD_F-150_2021_t0.json").write_text(json.dumps(disputed))
    gone = {**rec, "pair": {**rec["pair"], "model_year": 2019}}
    (reports / "FORD_F-150_2019_t0.json").write_text(json.dumps(gone))
    (tmp_path / "model").mkdir()
    pd.DataFrame({"make": ["FORD"] * 2, "model": ["F-150"] * 2, "model_year": [2020, 2021],
                  "theme_id": [0, 0], "status": ["confirmed", "confirmed"]}).to_parquet(
        tmp_path / "model" / "alerts.parquet")
    out = tmp_path / "site"
    (out / "investigations").mkdir(parents=True)
    (out / "investigations" / "old-report.json").write_text("{}")

    published = export.run(clean=tmp_path, out=out)
    # 2021 is disputed and 2019 is no longer an alert: only 2020 goes out.
    assert [p["slug"] for p in published] == ["2020-ford-f-150-theme-0"]
    files = sorted(f.name for f in (out / "investigations").glob("*.json"))
    assert files == ["2020-ford-f-150-theme-0.json"]  # the stale file was removed
    pub = json.loads((out / "investigations" / files[0]).read_text())
    assert pub["reading"] == {"key": "consistent_pattern", "confidence": "medium",
                              "label": "Consistent failure pattern in owner complaints"}
    assert "likely_defect" not in json.dumps(pub)
    assert pub["summary"] == "Owner wrote: call me at [PHONE] any time."
    assert pub["findings"] == [{"text": "pedal to floor", "complaints": ["9", "8"]}]
    assert pub["recalls"][0]["relation_label"] == "Covers this problem"
    assert pub["signal"]["complaints"] == 10 and pub["signal"]["signal_since"] == "2020-03-01"
    assert pub["provenance"]["human_reviewed"] is False and pub["provenance"]["note"]
    assert not {"tool_calls", "usage", "dossier", "review_tool_calls"} & set(pub)
    index = json.loads((out / "investigations.json").read_text())
    assert [p["slug"] for p in index["investigations"]] == ["2020-ford-f-150-theme-0"]

    # A rerun on the same reports leaves the files as they are.
    before = (out / "investigations" / files[0]).stat().st_mtime_ns
    export.run(clean=tmp_path, out=out)
    assert (out / "investigations" / files[0]).stat().st_mtime_ns == before
    method = json.loads((out / "method.json").read_text())
    assert method["data"]["complaints"] > 0 and method["backtest"]["recalled_pairs"] >= 1
    assert method["alerts"] == {"confirmed": 2}

    # Deploy check: no vehicle data yet, then ready, then a report from an older prompt.
    with pytest.raises(SystemExit):
        export.check_deploy(clean=tmp_path, out=out)
    (out / "vehicles").mkdir()
    (out / "vehicles" / "ford--f-150.json").write_text("{}")
    export.check_deploy(clean=tmp_path, out=out)
    old = json.loads((reports / "FORD_F-150_2020_t0.json").read_text())
    old["prompt_version"] = investigate.PROMPT_VERSION - 1
    (reports / "FORD_F-150_2020_t0.json").write_text(json.dumps(old))
    with pytest.raises(SystemExit):
        export.check_deploy(clean=tmp_path, out=out)


def test_vehicle_theme_flags_only_confirmed_unrecalled_signals():
    from pipeline import vehicles

    row = {"theme_id": 3, "complaints": 30, "recent_complaints": 20, "recent_prr": 8.04,
           "active": True, "surging": True, "onset_month": date(2025, 9, 1),
           "surge_onset_month": date(2025, 6, 1), "risk": 0.12345, "alert_status": "confirmed"}
    t = vehicles._theme(row, 120, [])
    assert t == {"id": 3, "complaints": 30, "share": 0.25, "recent_complaints": 20,
                 "recent_prr": 8.0, "recall_risk": 0.1235, "flagged": True, "rising": True,
                 "flagged_since": "2025-06"}
    # Never checked: counts and risk, no flag. Checked and off-theme: no risk either.
    unchecked = vehicles._theme({**row, "alert_status": None}, 120, [])
    assert "flagged" not in unchecked and unchecked["recall_risk"] == 0.1235
    off = vehicles._theme({**row, "alert_status": "not_confirmed"}, 120, [])
    assert "flagged" not in off and "recall_risk" not in off
    # A recall already covers it: listed, not flagged.
    recalled = vehicles._theme(row, 120, ["25V001000"])
    assert recalled["recalls"] == ["25V001000"]
    assert "flagged" not in recalled and "recall_risk" not in recalled


def test_vehicle_export(tmp_path: Path):
    from pipeline import signals, vehicles

    encode = _signal_corpus(tmp_path)
    # Named themes (cluster = theme) and an older spelling of the CR-V for model year 2018.
    pq.write_table(pa.table({"fit_id": ["f", "f"], "cluster_id": pa.array([0, 1], pa.int32()),
                             "name": ["Brakes fail", "Airbag light on"], "summary": ["", ""],
                             "kind": ["defect", "defect"], "model": ["m", "m"]}),
                   tmp_path / "clusters" / "names.parquet")
    pq.write_table(pa.table({"fit_id": ["f", "f"], "cluster_id": pa.array([0, 1], pa.int32()),
                             "theme_id": pa.array([0, 1], pa.int32()),
                             "theme_name": ["Brakes fail", "Airbag light on"]}),
                   tmp_path / "clusters" / "themes.parquet")
    complaints = pq.read_table(tmp_path / "complaints.parquet").to_pylist()
    for m in range(1, 13):
        i = str(1000 + m)
        complaints.append({**complaints[-1], "complaint_id": i, "odi_number": i, "model": "CR V",
                           "model_year": 2018, "received_date": date(2019, m, 2)})
    complaints.append({**complaints[-1], "complaint_id": "2000", "odi_number": "2000",
                       "make": "UNKNOWN", "model": "UNKNOWN"})
    pq.write_table(pa.Table.from_pylist(complaints), tmp_path / "complaints.parquet")
    recalls = pq.read_table(tmp_path / "recalls.parquet").to_pylist()
    for r in recalls:
        r.update(consequence_summary="CRASH RISK.", corrective_action="DEALERS WILL FIX IT.",
                 potentially_affected=10, do_not_drive=False, park_outside=r["make"] == "FORD")
    pq.write_table(pa.Table.from_pylist(recalls), tmp_path / "recalls.parquet")
    signals.run(clean=tmp_path, model_name="fake/model", encode=encode)
    out = tmp_path / "site"
    (out / "vehicles").mkdir(parents=True)
    (out / "vehicles" / "old--model.json").write_text("{}")

    index = vehicles.run(clean=tmp_path, out=out, min_complaints=10)
    files = sorted(f.name for f in (out / "vehicles").glob("*.json"))
    assert files == ["ford--f-150.json", "honda--accord.json", "honda--civic.json",
                     "honda--cr-v.json", "honda--pilot.json"]  # stale file gone, no UNKNOWN
    assert len(index) == 5
    crv = json.loads((out / "vehicles" / "honda--cr-v.json").read_text())
    # Two spellings, no shared model year: one page, named after the bigger spelling.
    assert crv["model"] == "CR-V" and [y["model_year"] for y in crv["years"]] == [2018, 2019]
    assert [y["nhtsa"] for y in crv["years"]] == [["HONDA", "CR V"], ["HONDA", "CR-V"]]
    ford = json.loads((out / "vehicles" / "ford--f-150.json").read_text())
    year = ford["years"][0]
    assert year["complaints"] == 10 and year["first_received"] == "2020-01"
    assert year["monthly"] == {"start": "2020-01", "counts": [1] * 10}
    assert year["recalls"] == ["21V001000"]
    assert year["themes"][0]["id"] == 0 and year["themes"][0]["recalls"] == ["21V001000"]
    assert "flagged" not in year["themes"][0]
    assert ford["recalls"]["21V001000"]["park_outside"] is True
    assert ford["recalls"]["21V001000"]["remedy"] == "DEALERS WILL FIX IT."
    themes = json.loads((out / "themes.json").read_text())["themes"]
    assert themes == {"0": "Brakes fail", "1": "Airbag light on"}

    before = (out / "vehicles" / "ford--f-150.json").stat().st_mtime_ns
    vehicles.run(clean=tmp_path, out=out, min_complaints=10)
    assert (out / "vehicles" / "ford--f-150.json").stat().st_mtime_ns == before
