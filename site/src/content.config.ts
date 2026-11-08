// Investigation reports exported by the pipeline (`make export`, pipeline/export.py).
// The schema mirrors export.public_record(); a mismatch fails the build instead of
// rendering a page with missing fields.
import { defineCollection } from "astro:content";
import { glob } from "astro/loaders";
import { z } from "astro/zod";

const month = z.string().regex(/^\d{4}-\d{2}(-\d{2})?$/);
const count = z.number().int().nonnegative().nullable();

const investigations = defineCollection({
  loader: glob({ pattern: "*.json", base: "./src/data/investigations" }),
  schema: z.object({
    schema_version: z.literal(1),
    slug: z.string(),
    vehicle: z.object({ make: z.string(), model: z.string(), model_year: z.number().int() }),
    theme: z.object({ id: z.number().int(), name: z.string() }),
    data_through: month.nullable(),
    reading: z.object({
      key: z.enum(["consistent_pattern", "emerging_pattern", "unclear", "not_explained_by_defect"]),
      label: z.string(),
      confidence: z.enum(["low", "medium", "high"]),
    }),
    headline: z.string(),
    summary: z.string(),
    failure_pattern: z.string(),
    safety_risk: z.string(),
    findings: z.array(z.object({ text: z.string(), complaints: z.array(z.string()) })),
    recalls: z.array(
      z.object({
        campaign_id: z.string(),
        relation: z.enum(["covers_this_problem", "similar_problem_other_vehicle", "unrelated"]),
        relation_label: z.string(),
        note: z.string(),
      }),
    ),
    open_questions: z.array(z.string()),
    signal: z.object({
      complaints: count,
      first_received: month.nullable(),
      last_received: month.nullable(),
      crashes: count,
      fires: count,
      complaints_with_injuries: count,
      injured: count,
      deaths: count,
      towed: count,
      median_miles: z.number().nullable(),
      recent_complaints: count,
      recent_prr: z.number().nullable(),
      peak_prr: z.number().nullable(),
      signal_since: month.nullable(),
      signalling: z.boolean().nullable(),
      surging: z.boolean().nullable(),
      surge_since: month.nullable(),
      last_3_months: count,
      last_3_months_expected: z.number().nullable(),
      recall_risk: z.number().nullable(),
      risk_rank: count,
      ranked_pairs: count,
      checked_complaints: count,
      checked_on_theme: count,
      monthly_complaints: z.array(z.object({ month, complaints: z.number().int() })),
    }),
    provenance: z.object({
      written: z.string(),
      investigator_model: z.string(),
      reviewer_model: z.string().nullable(),
      review: z.literal("approved"),
      human_reviewed: z.literal(false),
      note: z.string(),
    }),
  }),
});

// Vehicle pages: one file per make and model (pipeline/vehicles.py). The files are not
// committed; without them the collection is empty and no vehicle pages are built.
const vehicleTheme = z.object({
  id: z.number().int(),
  complaints: z.number().int(),
  share: z.number(),
  recent_complaints: z.number().int().default(0),
  recent_prr: z.number().optional(),
  recall_risk: z.number().optional(),
  recalls: z.array(z.string()).default([]),
  flagged: z.boolean().default(false),
  rising: z.boolean().default(false),
  flagged_since: month.optional(),
});

const vehicles = defineCollection({
  loader: glob({ pattern: "*.json", base: "./src/data/vehicles" }),
  schema: z.object({
    schema_version: z.literal(1),
    make: z.string(),
    model: z.string(),
    make_slug: z.string(),
    model_slug: z.string(),
    data_through: month,
    recalls: z.record(
      z.string(),
      z.object({
        date: z.string().nullable(),
        components: z.array(z.string()),
        summary: z.string().nullable(),
        consequence: z.string().nullable(),
        remedy: z.string().nullable(),
        affected: z.number().int().nullable(),
        do_not_drive: z.boolean(),
        park_outside: z.boolean(),
      }),
    ),
    years: z.array(
      z.object({
        model_year: z.number().int(),
        nhtsa: z.tuple([z.string(), z.string()]), // NHTSA's make and model spelling for this year
        complaints: z.number().int(),
        crashes: z.number().int(),
        fires: z.number().int(),
        injured: z.number().int(),
        deaths: z.number().int(),
        first_received: month,
        last_received: month,
        monthly: z.object({ start: month, counts: z.array(z.number().int()) }),
        themes: z.array(vehicleTheme),
        recalls: z.array(z.string()),
      }),
    ),
  }),
});

export const collections = { investigations, vehicles };
