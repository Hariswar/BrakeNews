import type { CategoryKey, DisplayMode, VisualProblem } from "./types";

// These are existing category IDs, never new complaint classifications.
export const SYSTEMS: { key: CategoryKey; label: string; location: "broad" | "unlocalized"; note: string }[] = [
  { key: "engine", label: "Engine & powertrain", location: "broad", note: "Broad powertrain region, including transmission. Layouts vary by vehicle." },
  { key: "brakes", label: "Brakes", location: "broad", note: "Wheel and braking regions; no individual failed component is identified." },
  { key: "electrical", label: "Electrical & lights", location: "unlocalized", note: "Electrical reports can involve many locations. No exact hotspot is assigned." },
  { key: "restraints", label: "Airbags, belts & seats", location: "broad", note: "General occupant area; not a verified fault location." },
  { key: "steering", label: "Steering, suspension & tires", location: "broad", note: "Broad steering and wheel region. These categories remain grouped as in the existing data." },
  { key: "body", label: "Body, doors & glass", location: "broad", note: "Body and glass surfaces; highlighting is not a diagnosis." },
  { key: "assist", label: "Driver assistance", location: "unlocalized", note: "Assistance systems span sensors and software. Explore their reports using this list." },
  { key: "other", label: "Other", location: "unlocalized", note: "These reports cannot be reliably placed on a specific physical region." },
];

// Colour scales only: counts/flags are copied from the published records, never estimated.
// Volume uses an existing individual problem count rather than inventing a system total.
export function displayValue(problems: readonly VisualProblem[], category: CategoryKey, mode: DisplayMode): number {
  const matching = problems.filter((p) => p.category === category);
  if (mode === "volume") return Math.max(0, ...matching.map((p) => p.complaints ?? 0));
  if (mode === "flags") return Number(matching.some((p) => p.flagged === true));
  if (mode === "rising") return Number(matching.some((p) => p.rising === true));
  if (mode === "recalls") return Number(matching.some((p) => p.recallIds.length > 0));
  return 0;
}
