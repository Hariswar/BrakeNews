// Published investigation reports (pipeline/export.py), sorted and grouped for the lists.
import { getCollection, type CollectionEntry } from "astro:content";

export type Report = CollectionEntry<"investigations">["data"];

// Highest estimated recall chance first (the export's risk rank), then by slug.
export async function allReports(): Promise<Report[]> {
  return (await getCollection("investigations"))
    .map((e) => e.data)
    .sort(
      (a, b) =>
        (a.signal.risk_rank ?? Infinity) - (b.signal.risk_rank ?? Infinity) ||
        a.slug.localeCompare(b.slug),
    );
}

export interface ProblemGroup {
  themeId: number;
  name: string;
  reports: Report[];
}

// Reports on the same problem (theme) together, in the order of each group's top report.
// Problems with one report stay apart and are returned as `singles`.
export function groupByProblem(reports: Report[]): { groups: ProblemGroup[]; singles: Report[] } {
  const byTheme = new Map<number, ProblemGroup>();
  for (const r of reports) {
    const g = byTheme.get(r.theme.id) ?? { themeId: r.theme.id, name: r.theme.name, reports: [] };
    g.reports.push(r);
    byTheme.set(r.theme.id, g);
  }
  const all = [...byTheme.values()];
  return {
    groups: all.filter((g) => g.reports.length > 1),
    singles: all.filter((g) => g.reports.length === 1).map((g) => g.reports[0]),
  };
}

export function latestDataMonth(reports: Report[]): string | null {
  return reports.map((r) => r.data_through).filter((d): d is string => !!d).sort().at(-1) ?? null;
}
