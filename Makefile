PY ?= python3.12
VENV := .venv
BIN := $(VENV)/bin

.PHONY: setup setup-site build ingest clean-data scrub embed cluster name match signals features select model risk verify investigate export spotcheck test lint site-dev site-build deploy

setup:            ## Python env with every pipeline dependency
	$(PY) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install -e ".[all]"

setup-site:       ## Node dependencies for the Astro site
	cd site && npm install

build:            ## One-command rebuild: raw NHTSA files -> clean tables + scrubbed narratives
	$(BIN)/python -m pipeline build

ingest:
	$(BIN)/python -m pipeline ingest

clean-data:
	$(BIN)/python -m pipeline clean

scrub:
	$(BIN)/python -m pipeline scrub

embed:            ## Phase 2: sentence embeddings of the scrubbed narratives (slow, resumable)
	$(BIN)/python -m pipeline embed

cluster:          ## Phase 2: assign narratives to themes; first run fits them (~15 min)
	$(BIN)/python -m pipeline cluster

name:             ## Phase 2: name unnamed clusters with Claude (needs ANTHROPIC_API_KEY)
	$(BIN)/python -m pipeline name

match:            ## Claude checks which themes each recall covers (Batch API, ~$1; resumable)
	$(BIN)/python -m pipeline match

signals:          ## Phase 3: vehicle x theme signals and surges, recall matching, backtest
	$(BIN)/python -m pipeline signals

features:         ## Phase 3: one row per vehicle x theme x month, label = recall within 24 months
	$(BIN)/python -m pipeline features

select:           ## Phase 3: compare feature/parameter sets on earlier folds (~3 min)
	$(BIN)/python -m pipeline select

model:            ## Phase 3: train and backtest the recall-risk model, score current pairs
	$(BIN)/python -m pipeline model

risk: signals features model  ## Phase 3 end to end (select is run by hand when features change)

verify:           ## Claude checks the complaints behind current alerts (Batch API, ~$2; resumable)
	$(BIN)/python -m pipeline verify

investigate:      ## Phase 4: investigate + fact-check the top confirmed alerts (TOP=n BUDGET=$)
	$(BIN)/python -m pipeline investigate --top $(or $(TOP),1) --budget $(or $(BUDGET),5)

export:           ## Site data: approved reports + vehicle pages -> site/src/data (no API calls)
	$(BIN)/python -m pipeline export

spotcheck:
	$(BIN)/python -m pipeline spotcheck -n 20

test:
	$(BIN)/pytest -q

lint:
	$(BIN)/ruff check pipeline tests

site-dev:
	cd site && npm run dev

site-build:
	cd site && npm run build

# Cloudflare Pages. Once: `cd site && npx wrangler@4 login` and
# `npx wrangler@4 pages project create brakenews --production-branch main`.
# The vehicle pages are built from local data (not in git), so deploys run from this machine.
deploy:           ## Phase 5: check, export, build and publish the site to Cloudflare Pages
	$(BIN)/python -m pipeline export
	$(BIN)/python -c "from pipeline import export; export.check_deploy()"
	cd site && npm run build
	cd site && npx --yes wrangler@4 pages deploy dist --project-name brakenews --branch main --commit-dirty=true
