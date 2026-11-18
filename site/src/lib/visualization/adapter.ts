// Presentation-only projections. No source writes, raw data access, network calls or scoring.
import type { ModelRecord, YearRecord } from "../vehicles";
import type { CollectionEntry } from "astro:content";
import type { CategoryKey, VisualizationData, VisualProblem, VisualEvent } from "./types";

type Report = CollectionEntry<"investigations">["data"];
interface Labels { name: (id: number) => string; category: (id: number) => CategoryKey; }
const sameMonth = (a: string | null, b: string | null) => !!a && !!b && a.slice(0, 7) === b.slice(0, 7);
const reportHref = (r: Report) => `/investigations/${r.slug}/`;

function eventsFor(report: Report, model?: ModelRecord): VisualEvent[] {
  const events: VisualEvent[] = [];
  if (report.signal.signal_since) events.push({ date: report.signal.signal_since, label: "First statistical signal (retrospective)", href: "/method/#backtest" });
  if (report.signal.surge_since) events.push({ date: report.signal.surge_since, label: "First surge signal (retrospective)", href: "/method/#flags" });
  if (model && sameMonth(model.data_through, report.data_through)) {
    for (const recall of report.recalls.filter((r) => r.relation === "covers_this_problem")) {
      const date = model.recalls[recall.campaign_id]?.date;
      if (date) events.push({ date, label: `Matched recall #${recall.campaign_id}`, href: `https://www.nhtsa.gov/recalls?nhtsaId=${encodeURIComponent(recall.campaign_id)}` });
    }
  }
  return events.sort((a, b) => a.date.localeCompare(b.date));
}

export function vehicleVisualization(model: ModelRecord, year: YearRecord, reports: readonly Report[], labels: Labels, vehicle: string, sourceHref: string): VisualizationData {
  const matching = reports.filter((r) => r.vehicle.make === year.nhtsa[0] && r.vehicle.model === year.nhtsa[1] && r.vehicle.model_year === year.model_year);
  return {
    version: 1, vehicle, snapshot: model.data_through, sourceHref,
    problems: year.themes.map((theme): VisualProblem => {
      const report = matching.find((r) => r.theme.id === theme.id && sameMonth(r.data_through, model.data_through));
      return {
        id: theme.id, name: labels.name(theme.id), category: labels.category(theme.id),
        complaints: theme.complaints, flagged: theme.flagged, rising: theme.rising,
        recallRisk: theme.recall_risk ?? null, recallIds: [...theme.recalls],
        evidenceHref: `#problem-${theme.id}`, reportHref: report ? reportHref(report) : null,
        monthly: report ? report.signal.monthly_complaints.map((m) => ({ ...m })) : [],
        events: report ? eventsFor(report, model) : [],
      };
    }),
  };
}

export function reportVisualization(report: Report, category: CategoryKey, vehicle: string, sourceHref: string, model?: ModelRecord): VisualizationData {
  // Only enrich recall dates from the same vehicle and snapshot. Unknown dates stay unknown.
  const sameVehicle = model?.years.some((y) => y.model_year === report.vehicle.model_year && y.nhtsa[0] === report.vehicle.make && y.nhtsa[1] === report.vehicle.model);
  return {
    version: 1, vehicle, snapshot: report.data_through, sourceHref, featuredProblem: report.theme.id,
    symptom: report.failure_pattern,
    problems: [{
      id: report.theme.id, name: report.theme.name, category,
      complaints: report.signal.complaints, flagged: report.signal.signalling,
      rising: report.signal.surging, recallRisk: report.signal.recall_risk,
      recallIds: report.recalls.filter((r) => r.relation === "covers_this_problem").map((r) => r.campaign_id),
      evidenceHref: report.findings.length ? "#findings" : "#report", reportHref: "#report",
      monthly: report.signal.monthly_complaints.map((m) => ({ ...m })),
      events: eventsFor(report, sameVehicle ? model : undefined),
    }],
  };
}
