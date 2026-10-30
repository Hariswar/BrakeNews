# BrakeNews

Early warning of vehicle defects from public NHTSA owner complaints: what owners report for each
car, which problems a car has far more often than other cars, and whether a recall covers them.

Everything is rebuilt from NHTSA's public flat files on a laptop (developed on an 8 GB M1).

## What it does

```
NHTSA flat files ─► ingest ─► clean ─► scrub ─► embed ─► cluster ─► name ─► signals ─► site
 2.2M complaint rows  Parquet  typed   personal   1.6M     128       Claude    vehicle x
 177k recall rows              tables  details    vectors  themes    labels    theme alerts
                                       masked
```

| Step | `make` | What it does | Time |
|---|---|---|---|
| ingest | `ingest` | Downloads the complaint and recall files and converts them to Parquet | download time |
| clean | `clean-data` | Typed, de-duplicated tables; impossible values (dates in 1901, 999 mph) become NULL; the complaint form's and the recall file's component names are mapped onto one scheme | ~10 s |
| scrub | `scrub` | Masks personal details NHTSA's redaction missed (phones, e-mails, VINs, names after titles or in sign-offs, street addresses, case numbers) | ~7 min |
| embed | `embed` | One sentence embedding per distinct narrative (all-MiniLM-L6-v2) | ~1.6 h first run, minutes after |
| cluster | `cluster` | Groups narratives into recurring failure themes | ~9 min fit, ~1 min after |
| name | `name` | Claude names each theme, marks non-defects, merges duplicates | < 1 h, < $1 |
| signals | `signals` | Flags vehicles with an unusual share of a theme, matches recalls, backtests | ~2 min |

`make build` runs ingest, clean and scrub; the later steps each have their own target.

## Results so far

Backtest over 1995 to October 2026 (`data/clean/signals/backtest.json`), before cluster naming:

- **73%** of recalls that had at least three matching complaints beforehand would have been
  flagged before NHTSA's recall, a median of **20 months** ahead (2,653 vehicle-theme pairs).
- A flagged vehicle-theme pair is **3.9x** as likely to get a covering recall within two years
  as a pair that reached the same number of complaints without a flag (2.2% vs 0.56%).

Read the second number carefully: most flags are not followed by a recall. Recalls are not the
only outcome of a real defect (extended warranties, service bulletins and investigations are
others), and many recalls follow a manufacturer's own testing rather than complaints. The
clusters are also fit on all years at once, so the backtest knows which themes exist before they
emerge; the signal itself only uses complaints received up to the month it is computed for.

## How it works

**Personal details.** NHTSA redacts names as `[XXX]` but misses a lot. The scrubber replaces
e-mails, phone numbers, VINs, licence plates, claim and case numbers, street addresses and ZIP
codes, and names after a title, "my name is" or a sign-off ("THANKS, EDUARDO") with typed
placeholders such as `[PHONE]`, and reports how many texts each rule touched. A name with no cue
in front of it cannot be found by pattern, so the site quotes narratives sparingly. Columns that
identify people (VIN, city, dealer details) are dropped in `clean`.

**Themes, not vehicles.** NHTSA staff open most narratives with "THE CONTACT OWNS A 2012 FORD
FUSION", so embeddings group by car as much as by failure: a first clustering put 55 of 143
clusters under a single make. Subtracting each make and model's mean vector before clustering
removes that shared component, leaving themes such as power-steering assist loss, Nissan CVT
failures or Takata airbag parts being unavailable. HDBSCAN cannot run on 1.6M points in 8 GB, so
UMAP and HDBSCAN find themes on a 150k sample and every narrative is assigned to its nearest
theme centroid (or to none, if it is less similar than that theme's own members usually are).

**Naming.** Claude (Haiku, through the Batch API) reads each theme's distinctive terms and the
narratives nearest its centroid, names it, and says whether it is a defect, a non-defect (recall
paperwork, dealers) or a mix of failures that only share wording. A second request groups
themes that are one problem split in two. Only defect themes become signals.

**Signals.** For each vehicle (make, model, model year) and theme, over a rolling 24-month
window: the proportional reporting ratio, i.e. how much more often this vehicle's complaints are
about this theme than everyone else's, the method drug-safety agencies use on adverse-event
reports. A pair signals at 3+ complaints, PRR >= 2 and chi-square >= 4 (Evans et al., 2001).
Comparing shares rather than counts cancels what inflates counts without a defect: popular
models, older cars, years with more complaints overall.

**Recall matching.** Each recall's defect summary (minus the "X is recalling certain..."
opening) is embedded with the same model and matched to its most similar themes, with the
similarity cutoff set by reading samples of matches. This is far tighter than NHTSA's component
category, where "ENGINE" covers dozens of unrelated problems.

**Incremental by design.** Every narrative is keyed by a hash of its scrubbed text. A monthly
NHTSA refresh embeds only new complaints, and tightening a scrub rule re-embeds only the texts
it changed (214,000 texts in 9 minutes when the reviewer-code rule was added). Clusters are
kept until `cluster --refit`, so their names stay valid across refreshes.

## Setup

Python 3.12 and Node 22.

```sh
make setup               # Python venv with all pipeline dependencies
make build               # ingest + clean + scrub
make embed cluster       # embeddings and themes
cp .env.example .env     # add ANTHROPIC_API_KEY, needed only for `make name`
make name signals
make test lint
make setup-site site-dev # the Astro site
```

Data lands in `data/raw` and `data/clean` (not committed). Every step is resumable: an
interrupted run continues where it stopped.

## Layout

```
pipeline/   one module per step; config.py holds paths, column layouts and model names
tests/      end-to-end tests on small synthetic data (no downloads, no API calls)
site/       Astro site (vehicle pages come in Phase 5)
```

## Roadmap

1. Data: ingest, clean, spot-check against the raw files. **Done.**
2. Scrub, embed, cluster, name with Claude. **Built; naming needs an API key to run.**
3. Recall-risk model. Signals and backtest **started**; next, a model that ranks signals by
   how likely a recall is, from signal strength, crashes, fires, injuries and growth.
4. LangGraph agents that investigate a signal and write it up.
5. Generated vehicle pages on the site.

## Data and disclaimer

Data: [NHTSA complaint and recall flat files](https://static.nhtsa.gov/odi/ffdd/). BrakeNews is
not affiliated with or endorsed by NHTSA and is not safety advice; check your vehicle's open
recalls at [nhtsa.gov/recalls](https://www.nhtsa.gov/recalls).

MIT licence.
