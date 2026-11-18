import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { vehicleVisualization, reportVisualization } from "../src/lib/visualization/adapter.ts";
import { SYSTEMS, displayValue } from "../src/lib/visualization/mapping.ts";
import type { ModelRecord, YearRecord } from "../src/lib/vehicles";

type Report = Parameters<typeof reportVisualization>[0];
const source = JSON.parse(readFileSync(new URL("../src/data/investigations/2026-honda-pilot-steering-wheel-clunks-or-clicks-when-turning.json", import.meta.url), "utf8")) as Report;
function freeze<T>(value: T): T {
  if (value && typeof value === "object") { Object.values(value).forEach(freeze); Object.freeze(value); }
  return value;
}
function fixture() {
  const report = structuredClone(source);
  const year = { model_year: report.vehicle.model_year, nhtsa: [report.vehicle.make, report.vehicle.model], themes: [
    { id: report.theme.id, complaints: 12, share: .2, recent_complaints: 9, recall_risk: .2734, recalls: ["TEST-MATCH"], flagged: true, rising: false },
    { id: -1, complaints: 7, share: .1, recent_complaints: 3, recalls: [], flagged: false, rising: true },
  ] } as YearRecord;
  const model: ModelRecord = { schema_version: 1, make_slug: "honda", model_slug: "pilot", make: report.vehicle.make, model: report.vehicle.model, data_through: report.data_through!, years: [year], recalls: { "TEST-MATCH": { date: "2026-09-15", components: [], summary: null, consequence: null, remedy: null, affected: null, do_not_drive: false, park_outside: false } } };
  return { report, year, model };
}
const labels = { name: (id: number) => `Existing theme ${id}`, category: () => "steering" as const };

test("vehicle adapter preserves published numbers, flags, categories and source inputs", () => {
  const { report, year, model } = freeze(fixture());
  const before = JSON.stringify({ report, year, model });
  const view = vehicleVisualization(model, year, [report], labels, "2026 Honda Pilot", "https://www.nhtsa.gov/");
  const p = view.problems[0];
  assert.equal(p.complaints, year.themes[0].complaints);
  assert.equal(p.recallRisk, .2734);
  assert.equal(p.flagged, true);
  assert.equal(p.rising, false);
  assert.equal(p.category, "steering");
  assert.equal(p.evidenceHref, `#problem-${report.theme.id}`);
  assert.deepEqual(p.recallIds, ["TEST-MATCH"]);
  assert.deepEqual(p.monthly, report.signal.monthly_complaints);
  assert.notEqual(p.monthly, report.signal.monthly_complaints);
  assert.equal(JSON.stringify({ report, year, model }), before);
});

test("a different vehicle or snapshot never supplies a problem's history", () => {
  for (const change of ["vehicle", "snapshot"] as const) {
    const { report, year, model } = fixture();
    if (change === "vehicle") report.vehicle.model_year -= 1;
    else report.data_through = "2000-01-01";
    const p = vehicleVisualization(model, year, [report], labels, "Vehicle", "/source").problems[0];
    assert.deepEqual(p.monthly, []);
    assert.deepEqual(p.events, []);
    assert.equal(p.reportHref, null);
  }
});

test("missing problem history stays missing instead of borrowing aggregate vehicle counts", () => {
  const { report, year, model } = fixture();
  year.monthly = { start: "2026-01", counts: [100, 200, 300] };
  const view = vehicleVisualization(model, year, [report], labels, "Vehicle", "/source");
  assert.deepEqual(view.problems[1].monthly, []);
  assert.equal(view.problems[1].recallRisk, null);
});

test("report adapter retains unknown values and the original AI-written symptom", () => {
  const { report } = fixture();
  report.signal.complaints = null; report.signal.signalling = null; report.signal.surging = null; report.signal.recall_risk = null;
  report.findings = [];
  const view = reportVisualization(freeze(report), "steering", "Vehicle", "/source");
  assert.equal(view.symptom, report.failure_pattern);
  assert.equal(view.problems[0].complaints, null);
  assert.equal(view.problems[0].flagged, null);
  assert.equal(view.problems[0].rising, null);
  assert.equal(view.problems[0].recallRisk, null);
  assert.equal(view.problems[0].evidenceHref, "#report");
});

test("only existing same-snapshot matched recall dates enter the replay", () => {
  const { report, model } = fixture();
  report.recalls = [
    { campaign_id: "TEST-MATCH", relation: "covers_this_problem", relation_label: "Match", note: "Fixture" },
    { campaign_id: "OTHER", relation: "similar_problem_other_vehicle", relation_label: "Other", note: "Fixture" },
    { campaign_id: "UNKNOWN-DATE", relation: "covers_this_problem", relation_label: "Match", note: "Fixture" },
  ];
  const view = reportVisualization(report, "steering", "Vehicle", "/source", model);
  assert.deepEqual(view.problems[0].recallIds, ["TEST-MATCH", "UNKNOWN-DATE"]);
  assert.deepEqual(view.problems[0].events.filter((e) => e.label.startsWith("Matched")), [{ date: "2026-09-15", label: "Matched recall #TEST-MATCH", href: "https://www.nhtsa.gov/recalls?nhtsaId=TEST-MATCH" }]);
  model.data_through = "2000-01";
  assert.equal(reportVisualization(report, "steering", "Vehicle", "/source", model).problems[0].events.filter((e) => e.label.startsWith("Matched")).length, 0);
  model.data_through = report.data_through!;
  model.years = [];
  assert.equal(reportVisualization(report, "steering", "Vehicle", "/source", model).problems[0].events.filter((e) => e.label.startsWith("Matched")).length, 0);
});

test("color modes use existing counts and booleans without adding a system total or score", () => {
  const { report, year, model } = fixture();
  const { problems } = vehicleVisualization(model, year, [report], labels, "Vehicle", "/source");
  assert.equal(displayValue(problems, "steering", "volume"), 12); // not 12 + 7
  assert.equal(displayValue(problems, "steering", "flags"), 1);
  assert.equal(displayValue(problems, "steering", "rising"), 1);
  assert.equal(displayValue(problems, "steering", "recalls"), 1);
  assert.equal(displayValue(problems, "steering", "neutral"), 0);
  assert.equal(displayValue(problems, "electrical", "volume"), 0);
  assert.deepEqual(SYSTEMS.filter((s) => s.location === "unlocalized").map((s) => s.key), ["electrical", "assist", "other"]);
});
