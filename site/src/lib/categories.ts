// Problem categories for chips and icons, from each theme's most common NHTSA component group
// (themes.json, written by pipeline/vehicles.py). Presentation only: categories never change a
// number or a claim.
import themesFile from "../data/themes.json";

export type CategoryKey =
  | "engine"
  | "brakes"
  | "electrical"
  | "restraints"
  | "steering"
  | "body"
  | "assist"
  | "other";

export const CATEGORIES: Record<CategoryKey, string> = {
  engine: "Engine & powertrain",
  brakes: "Brakes",
  electrical: "Electrical & lights",
  restraints: "Airbags, belts & seats",
  steering: "Steering, suspension & tires",
  body: "Body, doors & glass",
  assist: "Driver assistance",
  other: "Other",
};

const BY_GROUP: Record<string, CategoryKey> = {
  ENGINE: "engine",
  "POWER TRAIN": "engine",
  "FUEL/PROPULSION SYSTEM": "engine",
  "VEHICLE SPEED CONTROL": "engine",
  "SERVICE BRAKES": "brakes",
  "PARKING BRAKE": "brakes",
  "ELECTRICAL SYSTEM": "electrical",
  "EXTERIOR LIGHTING": "electrical",
  "INTERIOR LIGHTING": "electrical",
  "AIR BAGS": "restraints",
  "SEAT BELTS": "restraints",
  SEATS: "restraints",
  "CHILD SEAT": "restraints",
  STEERING: "steering",
  SUSPENSION: "steering",
  WHEELS: "steering",
  TIRES: "steering",
  STRUCTURE: "body",
  VISIBILITY: "body",
  "LATCHES/LOCKS/LINKAGES": "body",
  "FORWARD COLLISION AVOIDANCE": "assist",
  "LANE DEPARTURE": "assist",
  "BACK OVER PREVENTION": "assist",
  "ELECTRONIC STABILITY CONTROL (ESC)": "assist",
  "TRACTION CONTROL SYSTEM": "assist",
};

const GROUPS: Record<string, string> = (themesFile as { component_groups?: Record<string, string> })
  .component_groups ?? {};

export function categoryOf(themeId: number): CategoryKey {
  return BY_GROUP[GROUPS[String(themeId)] ?? ""] ?? "other";
}

export function categoryLabel(key: CategoryKey): string {
  return CATEGORIES[key];
}
