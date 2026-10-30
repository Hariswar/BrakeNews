import json
from datetime import date
from pathlib import Path
from types import SimpleNamespace

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


def test_signals_flag_disproportion_before_the_recall(tmp_path: Path):
    from pipeline import signals

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
                           "model_year": year, "received_date": when})
        narratives.append({"complaint_id": i, "narrative_id": f"n{i}"})
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
    pq.write_table(pa.table({
        "campaign_id": ["21V001000", "19V002000"],
        "defect_summary": ["THE BRAKE HOSE MAY LEAK.", "THE AIRBAG INFLATOR MAY RUPTURE."],
    }), tmp_path / "recall_timeline.parquet")
    pq.write_table(pa.table({
        "campaign_id": ["21V001000", "19V002000"], "make": ["FORD", "HONDA"],
        "model": ["F-150", "CIVIC"], "model_year": [2020, 2019],
        "report_date": [date(2021, 6, 1), date(2019, 3, 1)],
    }), tmp_path / "recalls.parquet")

    def fake_encode(texts):
        return np.array([eye[0] if "BRAKE" in t else eye[1] for t in texts])

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
    assert {(m["campaign_id"], m["cluster_id"]) for m in matches} == {
        ("21V001000", 0), ("19V002000", 1)}
    bt = json.loads((tmp_path / "signals" / "backtest.json").read_text())
    assert bt["share flagged before the recall"] == 1.0
    assert bt["median months flagged ahead of the recall"] == 15

    calls = []
    signals.run(clean=tmp_path, model_name="fake/model",
                encode=lambda t: calls.append(t) or fake_encode(t))
    assert calls == []  # recall vectors come from the cache
