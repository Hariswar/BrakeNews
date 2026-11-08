"""Phase 4, part 3: publish the investigation reports for the site
-> site/src/data/investigations/<slug>.json and site/src/data/investigations.json.

Only reports that are ready for the public are exported:
  - the reviewer approved them (investigate.py status `approved`: not disputed, and not
    changed after the review), and
  - the vehicle x theme pair is still a confirmed alert (verify.py: its recent complaints are
    really about the theme). A report on a pair that has dropped out describes old data.

The site names real manufacturers' products, so the investigator's labels are not shown as
they are. Each assessment becomes a description of what the complaints show (LABELS), not a
claim that a defect exists, and every report carries a provenance note: written by an AI from
public NHTSA complaints, fact-checked by a second AI, not reviewed by a person.

A public record keeps what a reader needs: the vehicle and theme, the report's text, findings
with the ODI numbers of the complaints they rest on, related recalls, the signal's numbers
(counts, severity, PRR, surge, risk score, monthly timeline) and provenance. Internal material
stays out: tool calls, token usage, the reviewer's issue list, the full dossier. No complaint
narratives are exported. All free text goes through scrub.scrub() once more, as a last guard
against personal details a quote might carry.

The export directory belongs to this step: files of reports that are no longer published are
removed. Output is sorted and stable, so reruns on unchanged reports give no git diff.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from pipeline.config import CLEAN, ROOT
from pipeline.investigate import write_index
from pipeline.scrub import scrub

SCHEMA_VERSION = 1
SITE_DATA = ROOT / "site" / "src" / "data"

# assessment -> (public key, public wording)
LABELS = {
    "likely_defect": ("consistent_pattern", "Consistent failure pattern in owner complaints"),
    "possible_defect": ("emerging_pattern", "Emerging pattern, limited evidence"),
    "insufficient_evidence": ("unclear", "Not enough detail to judge"),
    "not_a_defect": ("not_explained_by_defect", "Signal not explained by a defect"),
}
RELATIONS = {
    "covers_this_problem": "Covers this problem",
    "similar_problem_other_vehicle": "Similar problem on another vehicle",
    "unrelated": "Looked at, unrelated",
}
NOTE = ("Written by an AI (Claude) from public NHTSA complaint and recall data, and "
        "fact-checked against the cited complaints by a second AI. Not reviewed by a person. "
        "It describes what owners report; it is not a finding that a defect exists.")


def url_slug(make: str, model: str, model_year: int, theme: str) -> str:
    raw = f"{model_year} {make} {model} {theme}".lower()
    return re.sub(r"[^a-z0-9]+", "-", raw).strip("-")


def _clean(text: str) -> str:
    return scrub(text).strip()


def _round(x, digits: int):
    return None if x is None else round(float(x), digits)


def _signal(d: dict) -> dict:
    sev = d.get("complaints_with_narrative") or {}
    sig = d.get("signal") or {}
    risk = d.get("risk") or {}
    check = d.get("complaint_check") or {}
    timeline = next((v for k, v in d.items() if k.startswith("monthly_complaints")), [])
    return {
        "complaints": sev.get("complaints"),
        "first_received": sev.get("first_received"),
        "last_received": sev.get("last_received"),
        "crashes": sev.get("crashes"), "fires": sev.get("fires"),
        "complaints_with_injuries": sev.get("with_injuries"), "injured": sev.get("injured"),
        "deaths": sev.get("deaths"), "towed": sev.get("towed"),
        "median_miles": _round(sev.get("median_miles"), 0),
        "recent_complaints": sig.get("recent_complaints"),
        "recent_prr": _round(sig.get("recent_prr"), 1),
        "peak_prr": _round(sig.get("peak_prr"), 1),
        "signal_since": sig.get("onset_month"), "signalling": sig.get("active"),
        "surging": sig.get("surging"), "surge_since": sig.get("surge_onset_month"),
        "last_3_months": sig.get("recent_k3") and int(sig["recent_k3"]),
        "last_3_months_expected": _round(sig.get("recent_expected3"), 1),
        "recall_risk": _round(risk.get("risk"), 4), "risk_rank": risk.get("rank"),
        "ranked_pairs": risk.get("of_pairs"),
        "checked_complaints": check.get("window_complaints"),
        "checked_on_theme": check.get("on_theme"),
        "monthly_complaints": timeline,
    }


def public_record(rec: dict, slug: str) -> dict:
    rep, d = rec["report"], rec["dossier"]
    key, label = LABELS[rep["assessment"]]
    return {
        "schema_version": SCHEMA_VERSION,
        "slug": slug,
        "vehicle": {k: rec["pair"][k] for k in ("make", "model", "model_year")},
        "theme": {"id": rec["pair"]["theme_id"], "name": d["theme"]["name"]},
        "data_through": d.get("data_through"),
        "reading": {"key": key, "label": label, "confidence": rep["confidence"]},
        "headline": _clean(rep["headline"]),
        "summary": _clean(rep["summary"]),
        "failure_pattern": _clean(rep["failure_pattern"]),
        "safety_risk": _clean(rep["safety_risk"]),
        "findings": [{"text": _clean(e["finding"]), "complaints": list(e["complaint_ids"])}
                     for e in rep["evidence"]],
        "recalls": [{"campaign_id": r["campaign_id"], "relation": r["relation"],
                     "relation_label": RELATIONS[r["relation"]], "note": _clean(r["note"])}
                    for r in rep["related_recalls"]],
        "open_questions": [_clean(q) for q in rep["open_questions"]],
        "signal": _signal(d),
        "provenance": {
            "written": rec["created"][:10], "investigator_model": rec["model"],
            "reviewer_model": rec.get("review_model"), "review": "approved",
            "human_reviewed": False, "note": NOTE,
        },
    }


def _write(path: Path, data) -> bool:
    text = json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    if path.exists() and path.read_text() == text:
        return False
    path.write_text(text)
    return True


def run(clean: Path = CLEAN, out: Path = SITE_DATA) -> list[dict]:
    index = write_index(clean)
    ready = index[(index["status"] == "approved") & (index["alert_status"] == "confirmed")]
    held = index.drop(ready.index)
    out_dir = out / "investigations"
    out_dir.mkdir(parents=True, exist_ok=True)

    published, slugs, changed = [], set(), 0
    for row in ready.itertuples(index=False):
        rec = json.loads((clean / "investigations" / row.file).read_text())
        slug = url_slug(row.make, row.model, row.model_year, row.theme_name or "")
        if slug in slugs:  # two themes with one name: keep the URLs apart
            slug = f"{slug}-{row.theme_id}"
        slugs.add(slug)
        pub = public_record(rec, slug)
        changed += _write(out_dir / f"{slug}.json", pub)
        published.append({
            "slug": slug, "vehicle": pub["vehicle"], "theme": pub["theme"],
            "reading": pub["reading"], "headline": pub["headline"],
            "recall_risk": pub["signal"]["recall_risk"], "risk_rank": pub["signal"]["risk_rank"],
            "surging": pub["signal"]["surging"], "written": pub["provenance"]["written"],
        })
    removed = 0
    for f in out_dir.glob("*.json"):
        if f.stem not in slugs:
            f.unlink()
            removed += 1
    published.sort(key=lambda p: (p["risk_rank"] is None, p["risk_rank"], p["slug"]))
    _write(out / "investigations.json", {"schema_version": SCHEMA_VERSION,
                                         "investigations": published})

    where = out_dir.relative_to(ROOT) if out_dir.is_relative_to(ROOT) else out_dir
    print(f"[export] {len(published)} reports published -> {where}")
    print(f"[export]   {changed} written or changed, {removed} removed")
    for row in held.itertuples(index=False):
        why = (f"review {row.status}" if row.status != "approved"
               else f"alert {row.alert_status or 'gone'}")
        print(f"[export]   held back: {row.slug} ({why})")
    return published
