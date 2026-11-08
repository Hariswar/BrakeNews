// Display helpers shared by the pages. NHTSA spells makes and models in capitals.

const MAKES: Record<string, string> = {
  KIA: "Kia",
  RAM: "Ram",
  "MERCEDES-BENZ": "Mercedes-Benz",
  "ROLLS-ROYCE": "Rolls-Royce",
  "LAND ROVER": "Land Rover",
  MINI: "MINI",
};

// Words of three letters or fewer are usually model codes (MDX, RAV4, CX-5): keep them as
// NHTSA writes them. Longer words are names (ODYSSEY -> Odyssey).
function titleWord(word: string): string {
  if (word.length <= 3 || /\d/.test(word)) return word;
  return word
    .split("-")
    .map((p) => (p.length <= 3 || /\d/.test(p) ? p : p[0] + p.slice(1).toLowerCase()))
    .join("-");
}

export function makeName(make: string): string {
  return MAKES[make] ?? make.split(" ").map(titleWord).join(" ");
}

export function modelName(model: string): string {
  return model.split(" ").map(titleWord).join(" ");
}

export interface Vehicle {
  make: string;
  model: string;
  model_year: number;
}

export function vehicleName(v: Vehicle): string {
  return `${v.model_year} ${makeName(v.make)} ${modelName(v.model)}`;
}

// NHTSA's own page for the vehicle: complaints, recalls and investigations.
export function nhtsaVehicleUrl(v: Vehicle): string {
  const part = (s: string) => encodeURIComponent(s);
  return `https://www.nhtsa.gov/vehicle/${v.model_year}/${part(v.make)}/${part(v.model)}`;
}

export function nhtsaRecallUrl(campaignId: string): string {
  return `https://www.nhtsa.gov/recalls?nhtsaId=${encodeURIComponent(campaignId)}`;
}

const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
const MONTHS_LONG = [
  "January", "February", "March", "April", "May", "June",
  "July", "August", "September", "October", "November", "December",
];

// "2025-10" or "2025-10-01" -> "Oct 2025" (long: "October 2025")
export function monthLabel(ym: string, long = false): string {
  const [y, m] = ym.split("-").map(Number);
  return `${(long ? MONTHS_LONG : MONTHS)[m - 1]} ${y}`;
}

// "2026-11-08" -> "November 8, 2026"
export function dateLabel(ymd: string): string {
  const [y, m, d] = ymd.split("-").map(Number);
  return `${MONTHS_LONG[m - 1]} ${d}, ${y}`;
}

export function num(n: number): string {
  return n.toLocaleString("en-US");
}

export function percent(p: number): string {
  const v = p * 100;
  return `${v < 10 ? v.toFixed(1) : Math.round(v)}%`;
}

export function plural(n: number, one: string, many = `${one}s`): string {
  return `${num(n)} ${n === 1 ? one : many}`;
}
