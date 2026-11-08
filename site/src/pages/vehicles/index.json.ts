// Compact index for the vehicle picker: [make label, make slug, [[model label, model slug, [years]]]].
import { allMakes } from "../../lib/vehicles";
import { modelName } from "../../lib/format";

export async function GET() {
  const makes = (await allMakes()).map((mk) => [
    mk.label,
    mk.slug,
    [...mk.models]
      .sort((a, b) => modelName(a.model).localeCompare(modelName(b.model)))
      .map((m) => [modelName(m.model), m.model_slug, m.years.map((y) => y.model_year).sort((a, b) => b - a)]),
  ]);
  return new Response(JSON.stringify(makes), { headers: { "Content-Type": "application/json" } });
}
