// The same compact index powers free-text search and the guided dropdowns.
export type VehicleIndex = [string, string, [string, string, number[]][]][];
export interface SearchVehicle {
  label: string;
  href: string;
  terms: string;
}

export function normalizeVehicleQuery(value: string): string {
  return value.normalize("NFKD").replace(/[\u0300-\u036f]/g, "").toLowerCase().replace(/[^a-z0-9]+/g, " ").trim();
}

export function searchVehicles(vehicles: SearchVehicle[], query: string, limit = 8): SearchVehicle[] {
  const tokens = normalizeVehicleQuery(query).split(/\s+/).filter(Boolean);
  if (!tokens.length || limit <= 0) return [];
  const matches: SearchVehicle[] = [];
  for (const vehicle of vehicles) {
    // Owners often omit model punctuation: CRV, F150 and CX5 should find CR-V, F-150 and CX-5.
    const compact = vehicle.terms.replace(/ /g, "");
    if (tokens.every((token) => vehicle.terms.includes(token) || compact.includes(token))) matches.push(vehicle);
    if (matches.length >= limit) break;
  }
  return matches;
}

export function indexVehicles(index: VehicleIndex): SearchVehicle[] {
  return index.flatMap(([make, makeSlug, models]) =>
    models.flatMap(([model, modelSlug, years]) =>
      [...years].sort((a, b) => b - a).map((year) => {
        const label = `${year} ${make} ${model}`;
        return { label, href: `/vehicles/${makeSlug}/${modelSlug}/${year}/`, terms: normalizeVehicleQuery(label) };
      }),
    ),
  );
}
