"""Phase 5, part 2: data for the site's vehicle pages (no API calls)
-> site/src/data/vehicles/<make>--<model>.json and theme names site/src/data/themes.json (with
   each theme's most common NHTSA component group, which the site maps to a category icon).

The vehicle files are about 20 MB and change with every data refresh, so they are not committed
(.gitignore): the site builds without them (no vehicle pages), and a full build runs locally
after `make export`.

A vehicle is a make, model and model year as NHTSA spells them. Only vehicles with at least
MIN_COMPLAINTS distinct complaints get a page: 7,226 of about 32,700 vehicles, holding about 93%
of all complaints (October 2026). Below that, theme shares and signals rest on a handful of reports.

One file per make and model holds all its model years, so recall texts shared across years are
written once. For each model year:
  - totals over its distinct complaints (ODI numbers): crashes, fires, people injured and killed,
    first and last received, and complaints received per month;
  - its problem themes (defect themes only, as in signals.py), with the vehicle's complaint count,
    the share of its complaints, the PRR over the last 24 months and the recalls matched to the
    theme on this vehicle;
  - its recalls, newest first.

Flags follow the same rule as published investigations (export.py): a theme is shown as flagged
only when verify.py confirmed that the complaints behind the alert are about that theme on that
vehicle. Signals that were never checked, or whose complaints turned out to be about something
else, show their counts but no flag, and the model's recall estimate is withheld for the latter.
No complaint narratives are exported.

The output directory belongs to this step: files of models that no longer qualify are removed.
Output is sorted and stable, so reruns on unchanged data give no git diff.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import duckdb

from pipeline.config import CLEAN, ROOT

SCHEMA_VERSION = 1
SITE_DATA = ROOT / "site" / "src" / "data"
MIN_COMPLAINTS = 25
MIN_THEME_COMPLAINTS = 3  # themes with fewer complaints are noise at the vehicle level
MAX_THEMES = 20  # per vehicle, plus any flagged theme beyond that
UNRELIABLE = ("not_confirmed", "mixed")  # verify.py: complaints not (all) about the theme


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "unnamed"


def _month(d) -> str | None:
    return None if d is None else d.strftime("%Y-%m")


def _num(x, digits: int | None = None):
    if x is None:
        return None
    return round(float(x), digits) if digits is not None else int(x)


def _text(s: str | None) -> str | None:
    s = re.sub(r"\s+", " ", s or "").strip()
    return s or None


def _rows(con: duckdb.DuckDBPyConnection, sql: str) -> list[dict]:
    rel = con.sql(sql)
    cols = rel.columns
    return [dict(zip(cols, r, strict=True)) for r in rel.fetchall()]


def load(clean: Path = CLEAN, min_complaints: int = MIN_COMPLAINTS) -> dict:
    """Everything the vehicle pages need, grouped by (make, model)."""
    p = {k: (clean / k).as_posix() for k in (
        "complaints.parquet", "recalls.parquet", "signals/signals.parquet",
        "signals/recall_themes.parquet", "model/risk_scores.parquet", "model/alerts.parquet",
        "clusters/themes.parquet", "clusters/names.parquet", "clusters/clusters.parquet")}
    con = duckdb.connect()
    con.execute("SET preserve_insertion_order = false")

    # One row per complaint and vehicle (a complaint can name several components).
    con.execute(f"""
        CREATE TEMP TABLE c AS
        SELECT odi_number, make, model, model_year, min(received_date) AS received,
               bool_or(crash) AS crash, bool_or(fire) AS fire,
               max(coalesce(injured, 0)) AS injured, max(coalesce(deaths, 0)) AS deaths
        FROM '{p['complaints.parquet']}'
        WHERE make IS NOT NULL AND model IS NOT NULL AND model_year IS NOT NULL
          AND received_date IS NOT NULL
        GROUP BY ALL
    """)
    # Complaints filed without a vehicle get no page.
    con.execute(f"""
        CREATE TEMP TABLE v AS
        SELECT make, model, model_year, count(*) AS complaints,
               count(*) FILTER (crash) AS crashes, count(*) FILTER (fire) AS fires,
               sum(injured) AS injured, sum(deaths) AS deaths,
               min(received) AS first_received, max(received) AS last_received
        FROM c WHERE make <> 'UNKNOWN' AND model <> 'UNKNOWN'
        GROUP BY ALL HAVING count(*) >= {int(min_complaints)}
    """)
    through = con.sql("SELECT max(received) FROM c").fetchall()[0][0]
    vehicles = _rows(con, "SELECT * FROM v ORDER BY make, model, model_year")
    monthly = _rows(con, """
        SELECT make, model, model_year, strftime(received, '%Y-%m') AS month, count(*) AS n
        FROM c JOIN v USING (make, model, model_year) GROUP BY ALL ORDER BY ALL""")

    # Theme names for the current fit, defect themes only (as signals.py counts them).
    con.execute(f"""
        CREATE TEMP TABLE theme AS
        SELECT DISTINCT t.theme_id, t.theme_name
        FROM '{p['clusters/themes.parquet']}' t
        JOIN '{p['clusters/names.parquet']}' n USING (fit_id, cluster_id)
        WHERE n.kind = 'defect'
          AND t.fit_id = (SELECT any_value(fit_id) FROM '{p['clusters/clusters.parquet']}')
    """)
    themes = dict(con.sql("SELECT theme_id, theme_name FROM theme ORDER BY 1").fetchall())
    # Each theme's most common NHTSA component group over its complaints' component rows,
    # "UNKNOWN OR OTHER" only when nothing else is named.
    groups: dict[int, str] = {}
    a = (clean / "clusters" / "assignments.parquet").as_posix()
    n = (clean / "narratives.parquet").as_posix()
    if Path(a).exists() and Path(n).exists():
        names = (clean / "clusters" / "names.parquet").as_posix()
        groups = dict(con.sql(f"""
            WITH cl AS (SELECT DISTINCT t.cluster_id, t.theme_id
                        FROM '{p['clusters/themes.parquet']}' t
                        JOIN '{names}' n USING (fit_id, cluster_id)
                        WHERE n.kind = 'defect' AND t.fit_id = (
                            SELECT any_value(fit_id) FROM '{p['clusters/clusters.parquet']}')),
            g AS (SELECT cl.theme_id, c.component_group AS grp, count(*) AS k
                  FROM '{p['complaints.parquet']}' c
                  JOIN '{n}' nr USING (complaint_id)
                  JOIN '{a}' a USING (narrative_id)
                  JOIN cl USING (cluster_id)
                  WHERE c.component_group IS NOT NULL
                  GROUP BY ALL)
            SELECT theme_id, arg_max(grp, (grp <> 'UNKNOWN OR OTHER', k)) FROM g GROUP BY 1
        """).fetchall())

    # Recalls of these vehicles, and which of the vehicle's themes each one covers.
    con.execute(f"""
        CREATE TEMP TABLE r AS
        SELECT r.make, r.model, r.model_year, r.campaign_id, r.report_date, r.component,
               r.defect_summary, r.consequence_summary, r.corrective_action,
               r.potentially_affected, r.do_not_drive, r.park_outside
        FROM '{p['recalls.parquet']}' r JOIN v USING (make, model, model_year)
    """)
    recalls = _rows(con, """
        SELECT make, model, campaign_id, min(report_date) AS report_date,
               list(DISTINCT component ORDER BY component) FILTER (component IS NOT NULL)
                   AS components,
               any_value(defect_summary) AS summary,
               any_value(consequence_summary) AS consequence,
               any_value(corrective_action) AS remedy,
               max(potentially_affected) AS affected,
               bool_or(do_not_drive) AS do_not_drive, bool_or(park_outside) AS park_outside
        FROM r GROUP BY ALL ORDER BY ALL""")
    vehicle_recalls = _rows(con, """
        SELECT make, model, model_year, campaign_id, min(report_date) AS report_date
        FROM r GROUP BY ALL ORDER BY make, model, model_year, report_date DESC, campaign_id""")
    covering = _rows(con, f"""
        SELECT DISTINCT r.make, r.model, r.model_year, t.theme_id, r.campaign_id
        FROM r JOIN '{p['signals/recall_themes.parquet']}' t USING (campaign_id)
        ORDER BY ALL""")

    alerts = clean / "model" / "alerts.parquet"
    alert_join = (f"LEFT JOIN '{alerts.as_posix()}' a USING (make, model, model_year, theme_id)"
                  if alerts.exists() else "")
    alert_status = "a.status" if alerts.exists() else "NULL"
    risk = clean / "model" / "risk_scores.parquet"
    risk_join = (f"LEFT JOIN '{risk.as_posix()}' k USING (make, model, model_year, theme_id)"
                 if risk.exists() else "")
    risk_col = "k.risk" if risk.exists() else "NULL"
    pair_rows = _rows(con, f"""
        SELECT s.make, s.model, s.model_year, s.theme_id, s.complaints, s.recent_complaints,
               s.recent_prr, s.active, s.surging, s.onset_month, s.surge_onset_month,
               {risk_col} AS risk, {alert_status} AS alert_status
        FROM '{p['signals/signals.parquet']}' s
        JOIN v USING (make, model, model_year)
        JOIN theme USING (theme_id)
        {risk_join}
        {alert_join}
        ORDER BY s.make, s.model, s.model_year, s.complaints DESC, s.theme_id
    """)
    return {"through": through, "vehicles": vehicles, "monthly": monthly, "themes": themes,
            "theme_groups": groups,
            "recalls": recalls, "vehicle_recalls": vehicle_recalls, "covering": covering,
            "pairs": pair_rows}


def _dense(counts: dict[str, int]) -> dict:
    """{"2025-07": 3, "2025-09": 1} -> {"start": "2025-07", "counts": [3, 0, 1]}"""
    def index(ym: str) -> int:
        y, m = ym.split("-")
        return int(y) * 12 + int(m) - 1
    first = min(counts, key=index)
    n = index(max(counts, key=index)) - index(first) + 1
    dense = [0] * n
    for ym, k in counts.items():
        dense[index(ym) - index(first)] = k
    return {"start": first, "counts": dense}


def _theme(row: dict, vehicle_complaints: int, recalls: list[str]) -> dict:
    """A theme on one vehicle. Fields that are false, empty or unknown are left out."""
    confirmed = row["alert_status"] == "confirmed"
    flagged = confirmed and bool(row["active"] or row["surging"]) and not recalls
    risk = None if (row["alert_status"] in UNRELIABLE or recalls) else row["risk"]
    flags = [m for m in (row["onset_month"], row["surge_onset_month"]) if m is not None]
    t = {
        "id": int(row["theme_id"]),
        "complaints": _num(row["complaints"]),
        "share": _num(row["complaints"] / vehicle_complaints, 4),
        "recent_complaints": _num(row["recent_complaints"]),
        "recent_prr": _num(row["recent_prr"], 1),
        "recall_risk": _num(risk, 4),
        "recalls": recalls,
    }
    if flagged:
        t["flagged"] = True
        t["rising"] = bool(row["surging"])
        t["flagged_since"] = _month(min(flags)) if flags else None
    # Identity checks: 0 == False, and a theme id or count of 0 must stay.
    return {k: v for k, v in t.items() if v is not None and v is not False and v != []}


def build(data: dict) -> tuple[dict[tuple[str, str], dict], list[dict]]:
    """(make slug, model slug) -> public model record, and an index of all models."""
    by_vehicle: dict[tuple, dict] = {}
    for v in data["vehicles"]:
        key = (v["make"], v["model"], v["model_year"])
        by_vehicle[key] = {
            "model_year": int(v["model_year"]),
            "complaints": int(v["complaints"]), "crashes": int(v["crashes"]),
            "fires": int(v["fires"]), "injured": int(v["injured"]), "deaths": int(v["deaths"]),
            "first_received": _month(v["first_received"]),
            "last_received": _month(v["last_received"]),
            "monthly": {}, "themes": [], "recalls": [],
        }
    sparse: dict[tuple, dict[str, int]] = {}
    for m in data["monthly"]:
        sparse.setdefault((m["make"], m["model"], m["model_year"]), {})[m["month"]] = int(m["n"])
    for key, counts in sparse.items():
        by_vehicle[key]["monthly"] = _dense(counts)
    for r in data["vehicle_recalls"]:
        by_vehicle[(r["make"], r["model"], r["model_year"])]["recalls"].append(r["campaign_id"])
    cover: dict[tuple, list[str]] = {}
    for c in data["covering"]:
        cover.setdefault((c["make"], c["model"], c["model_year"], c["theme_id"]), []).append(
            c["campaign_id"])

    themes_of: dict[tuple, list[dict]] = {}
    for row in data["pairs"]:
        key = (row["make"], row["model"], row["model_year"])
        recalls = cover.get((*key, row["theme_id"]), [])
        themes_of.setdefault(key, []).append(
            _theme(row, by_vehicle[key]["complaints"], recalls))
    for key, rows in themes_of.items():
        kept = [t for t in rows if t["complaints"] >= MIN_THEME_COMPLAINTS][:MAX_THEMES]
        kept += [t for t in rows if t.get("flagged") and t not in kept]
        by_vehicle[key]["themes"] = kept

    recalls_of: dict[tuple[str, str], dict] = {}
    for r in data["recalls"]:
        recalls_of.setdefault((r["make"], r["model"]), {})[r["campaign_id"]] = {
            "date": r["report_date"].isoformat() if r["report_date"] else None,
            "components": list(r["components"] or []),
            "summary": _text(r["summary"]),
            "consequence": _text(r["consequence"]),
            "remedy": _text(r["remedy"]),
            "affected": _num(r["affected"]),
            "do_not_drive": bool(r["do_not_drive"]),
            "park_outside": bool(r["park_outside"]),
        }

    # Each model year keeps NHTSA's spelling (`nhtsa`, for links and matching reports).
    # Spellings that give the same slug ("MERCEDES BENZ" and "MERCEDES-BENZ", "CR V" and "CR-V")
    # share a make page and, where their model years do not overlap, one model page; names shown
    # come from the spelling with the most complaints. Overlapping ones get "-2", "-3" slugs.
    years_of: dict[tuple[str, str], list[dict]] = {}
    for (make, model, _), rec in by_vehicle.items():
        rec["nhtsa"] = [make, model]
        years_of.setdefault((make, model), []).append(rec)
    total = {sp: sum(r["complaints"] for r in recs) for sp, recs in years_of.items()}
    make_total: dict[str, int] = {}
    for (make, _), n in total.items():
        make_total[make] = make_total.get(make, 0) + n
    make_label: dict[str, str] = {}
    for make in sorted(make_total, key=lambda mk: (-make_total[mk], mk)):
        make_label.setdefault(slug(make), make)

    groups: dict[tuple[str, str], list[tuple[str, str]]] = {}
    for sp in sorted(years_of):
        groups.setdefault((slug(sp[0]), slug(sp[1])), []).append(sp)
    models: dict[tuple[str, str], dict] = {}
    index = []
    for (make_slug, model_slug), members in sorted(groups.items()):
        buckets: list[dict] = []
        for sp in sorted(members, key=lambda sp: (-total[sp], sp)):
            years = {r["model_year"] for r in years_of[sp]}
            bucket = next((b for b in buckets if not years & b["years"]), None)
            if bucket is None:
                buckets.append({"members": [sp], "years": years})
            else:
                bucket["members"].append(sp)
                bucket["years"] |= years
        for i, b in enumerate(buckets):
            ms = model_slug if i == 0 else f"{model_slug}-{i + 1}"
            recs = sorted((r for sp in b["members"] for r in years_of[sp]),
                          key=lambda y: y["model_year"])
            recalls: dict[str, dict] = {}
            for sp in b["members"]:
                recalls.update(recalls_of.get(sp, {}))
            rec = {
                "schema_version": SCHEMA_VERSION, "make": make_label[make_slug],
                "model": b["members"][0][1], "make_slug": make_slug, "model_slug": ms,
                "data_through": _month(data["through"]), "recalls": recalls, "years": recs,
            }
            models[(make_slug, ms)] = rec
            index.append({
                "make": rec["make"], "model": rec["model"], "make_slug": make_slug,
                "model_slug": ms,
                "years": [[y["model_year"], y["complaints"]] for y in recs],
                "flagged": sorted({y["model_year"] for y in recs
                                   if any(t.get("flagged") for t in y["themes"])}),
            })
    return models, index


def _write(path: Path, data) -> bool:
    text = json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    if path.exists() and path.read_text() == text:
        return False
    path.write_text(text)
    return True


def run(clean: Path = CLEAN, out: Path = SITE_DATA,
        min_complaints: int = MIN_COMPLAINTS) -> list[dict]:
    data = load(clean, min_complaints)
    models, index = build(data)
    out_dir = out / "vehicles"
    out_dir.mkdir(parents=True, exist_ok=True)
    names, changed = set(), 0
    for rec in models.values():
        name = f"{rec['make_slug']}--{rec['model_slug']}"
        names.add(name)
        changed += _write(out_dir / f"{name}.json", rec)
    removed = 0
    for f in out_dir.glob("*.json"):
        if f.stem not in names:
            f.unlink()
            removed += 1
    _write(out / "themes.json", {
        "schema_version": SCHEMA_VERSION,
        "themes": {str(k): v for k, v in data["themes"].items()},
        "component_groups": {str(k): v for k, v in data.get("theme_groups", {}).items()
                             if k in data["themes"]},
    })

    years = sum(len(m["years"]) for m in index)
    flagged = sum(len(m["flagged"]) for m in index)
    size = sum(f.stat().st_size for f in out_dir.glob("*.json"))
    where = out_dir.relative_to(ROOT) if out_dir.is_relative_to(ROOT) else out_dir
    print(f"[vehicles] {years:,} vehicles (>= {min_complaints} complaints) in {len(index):,} "
          f"models -> {where} ({size / 1e6:.1f} MB)")
    print(f"[vehicles]   {changed} files written or changed, {removed} removed; "
          f"{flagged:,} vehicles with a confirmed flag")
    return index
