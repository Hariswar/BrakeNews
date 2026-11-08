import assert from "node:assert/strict";
import test from "node:test";
import { indexVehicles, normalizeVehicleQuery, searchVehicles, type VehicleIndex } from "../src/lib/vehicle-search.ts";

const index: VehicleIndex = [
  ["Honda", "honda", [["Pilot", "pilot", [2024, 2025]], ["CR-V", "cr-v", [2025]]]],
  ["Toyota", "toyota", [["Grand Highlander", "grand-highlander", [2025]]]],
  ["Mercedes-Benz", "mercedes-benz", [["C300", "c300", [2023]]]],
  ["Ford", "ford", [["F-150", "f-150", [2025]]]],
  ["Mazda", "mazda", [["CX-5", "cx-5", [2025]]]],
];
const vehicles = indexVehicles(index);

test("a full vehicle query opens the exact indexed route", () => {
  assert.deepEqual(searchVehicles(vehicles, "2025 Honda Pilot").map((v) => v.href), ["/vehicles/honda/pilot/2025/"]);
});
test("search ignores case, punctuation, extra whitespace and token order", () => {
  assert.equal(searchVehicles(vehicles, " PILOT   hOnDa 2025 ")[0]?.label, "2025 Honda Pilot");
  assert.equal(searchVehicles(vehicles, "honda cr-v 2025")[0]?.href, "/vehicles/honda/cr-v/2025/");
  assert.equal(searchVehicles(vehicles, "mercedes benz c300")[0]?.label, "2023 Mercedes-Benz C300");
  assert.equal(normalizeVehicleQuery("Citroën"), "citroen");
});
test("partial queries work, with newest years first within a model", () => {
  assert.deepEqual(searchVehicles(vehicles, "honda pil").map((v) => v.label), ["2025 Honda Pilot", "2024 Honda Pilot"]);
  assert.equal(searchVehicles(vehicles, "grand high")[0]?.label, "2025 Toyota Grand Highlander");
});
test("common unpunctuated model names find the canonical indexed vehicle", () => {
  assert.equal(searchVehicles(vehicles, "2025 honda crv")[0]?.href, "/vehicles/honda/cr-v/2025/");
  assert.equal(searchVehicles(vehicles, "ford f150")[0]?.href, "/vehicles/ford/f-150/2025/");
  assert.equal(searchVehicles(vehicles, "mazda cx5")[0]?.href, "/vehicles/mazda/cx-5/2025/");
});
test("empty, punctuation-only and unknown queries have no results", () => {
  for (const query of ["", "  ", "---", "Honda Highlander", "2026 Honda Pilot"]) assert.deepEqual(searchVehicles(vehicles, query), []);
});
test("results respect the limit and index conversion does not mutate source years", () => {
  assert.equal(searchVehicles(vehicles, "honda", 1).length, 1);
  assert.deepEqual(searchVehicles(vehicles, "honda", 0), []);
  assert.deepEqual(index[0][2][0][2], [2024, 2025]);
});
test("empty exported indexes stay empty", () => {
  assert.deepEqual(indexVehicles([]), []);
});
