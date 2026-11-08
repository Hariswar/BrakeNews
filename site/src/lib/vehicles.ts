// Shared access to the vehicle data (pipeline/vehicles.py) for the /vehicles pages.
import { getCollection, type CollectionEntry } from "astro:content";
import themesFile from "../data/themes.json";
import { makeName, modelName } from "./format";

export type ModelRecord = CollectionEntry<"vehicles">["data"];
export type YearRecord = ModelRecord["years"][number];
export type VehicleTheme = YearRecord["themes"][number];
export type Recall = ModelRecord["recalls"][string];

const THEMES: Record<string, string> = themesFile.themes;

export function themeName(id: number): string {
  return THEMES[String(id)] ?? `Problem ${id}`;
}

let cache: ModelRecord[] | undefined;

// Every model, sorted by make then model as NHTSA spells them.
export async function allModels(): Promise<ModelRecord[]> {
  cache ??= (await getCollection("vehicles"))
    .map((e) => e.data)
    .sort((a, b) => a.make.localeCompare(b.make) || a.model.localeCompare(b.model));
  return cache;
}

export interface Make {
  make: string;
  slug: string;
  label: string;
  models: ModelRecord[];
}

export async function allMakes(): Promise<Make[]> {
  const byMake = new Map<string, Make>();
  for (const m of await allModels()) {
    const entry = byMake.get(m.make_slug) ?? { make: m.make, slug: m.make_slug, label: makeName(m.make), models: [] };
    entry.models.push(m);
    byMake.set(m.make_slug, entry);
  }
  return [...byMake.values()].sort((a, b) => a.label.localeCompare(b.label));
}

export const makeUrl = (m: { make_slug: string }) => `/vehicles/${m.make_slug}/`;
export const modelUrl = (m: { make_slug: string; model_slug: string }) =>
  `/vehicles/${m.make_slug}/${m.model_slug}/`;
export const yearUrl = (m: { make_slug: string; model_slug: string }, year: number) =>
  `${modelUrl(m)}${year}/`;

export const modelLabel = (m: { make: string; model: string }) => `${makeName(m.make)} ${modelName(m.model)}`;

// The page of a make, model and year as NHTSA spells them, if one was built.
export async function findVehicleUrl(make: string, model: string, year: number): Promise<string | null> {
  for (const m of await allModels()) {
    const y = m.years.find((y) => y.model_year === year && y.nhtsa[0] === make && y.nhtsa[1] === model);
    if (y) return yearUrl(m, year);
  }
  return null;
}

export function monthlyList(monthly: YearRecord["monthly"]): { month: string; complaints: number }[] {
  const [y, mo] = monthly.start.split("-").map(Number);
  const start = y * 12 + mo - 1;
  return monthly.counts.map((complaints, i) => {
    const idx = start + i;
    return { month: `${Math.floor(idx / 12)}-${String((idx % 12) + 1).padStart(2, "0")}`, complaints };
  });
}

// Older recall texts are in capitals. Lower-case them and restore sentence starts. Kept as
// written: words with digits (model names, part numbers) and known acronyms. Names in `names`
// (upper case, e.g. the makes and this page's model) get a capital: other model names stay lower
// case, since many are ordinary words ("escape", "pilot") that recall texts also use as words.
const KEEP = new Set([
  "ABS", "ESC", "TPMS", "FMVSS", "NHTSA", "VIN", "SRS", "LED", "HID", "PCM", "ECM", "BCM", "USA",
  "U.S.", "OTA", "EV", "PHEV", "HVAC", "SUV", "RV", "LLC",
]);
export function readable(text: string | null, names: Set<string> = new Set()): string | null {
  if (!text) return text;
  const letters = text.replace(/[^A-Za-z]/g, "");
  const upper = letters.replace(/[^A-Z]/g, "").length;
  if (letters.length < 20 || upper / letters.length < 0.8) return text;
  const lowered = text.replace(/[A-Za-z0-9.'&-]+/g, (w) => {
    const bare = w.replace(/[.,]+$/, "");
    if (/\d/.test(w) || KEEP.has(w) || KEEP.has(bare)) return w;
    if (names.has(bare)) return w[0] + w.slice(1).toLowerCase();
    return w.toLowerCase();
  });
  return lowered.replace(/(^|[.!?]\s+)([a-z])/g, (_, pre, c) => pre + c.toUpperCase());
}

// Upper-case names readable() capitalizes: every make, minus words that are ordinary English
// ("CHEVROLET TRUCK", "FOREST RIVER") or names of something else.
const NOT_NAMES = new Set([
  "RAM", "MINI", "SMART", "GENESIS", "TRUCK", "FOREST", "RIVER", "GRAND", "DESIGN", "INDIAN",
  "SANTA", "VICTORY", "STALLION", "VANGUARD", "WORKHORSE", "KEYSTONE", "LAND", "EAGLE", "LUCID",
  "PUERTO", "RICO", "SATURN", "MERCURY", "SHASTA", "UNKNOWN",
]);
let makeWords: Set<string> | undefined;
export async function nameWords(...extra: string[]): Promise<Set<string>> {
  makeWords ??= new Set(
    (await allModels()).flatMap((m) => m.make.split(/[\s-]+/)).filter((w) => w.length > 2 && !NOT_NAMES.has(w)),
  );
  return new Set([...makeWords, ...extra.flatMap((e) => e.split(/[\s-]+/)).filter((w) => w.length > 2)]);
}
