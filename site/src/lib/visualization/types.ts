import type { CategoryKey } from "../categories";

export type { CategoryKey };
export type ModelFamily = "sedan" | "suv" | "pickup";
export type DisplayMode = "neutral" | "volume" | "flags" | "rising" | "recalls";
export interface VisualMonth { readonly month: string; readonly complaints: number; }
export interface VisualEvent { readonly date: string; readonly label: string; readonly href: string | null; }
export interface VisualProblem {
  readonly id: number;
  readonly name: string;
  readonly category: CategoryKey;
  readonly complaints: number | null;
  readonly flagged: boolean | null;
  readonly rising: boolean | null;
  readonly recallRisk: number | null;
  readonly recallIds: readonly string[];
  readonly evidenceHref: string;
  readonly reportHref: string | null;
  readonly monthly: readonly VisualMonth[];
  readonly events: readonly VisualEvent[];
}
export interface VisualizationData {
  readonly version: 1;
  readonly vehicle: string;
  readonly snapshot: string | null;
  readonly sourceHref: string;
  readonly problems: readonly VisualProblem[];
  readonly featuredProblem?: number;
  readonly symptom?: string;
}
