PY ?= python3.12
VENV := .venv
BIN := $(VENV)/bin

.PHONY: setup setup-site build ingest clean-data scrub embed cluster name signals spotcheck test lint site-dev site-build

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

signals:          ## Phase 3 start: vehicle x theme signals, recall matching, backtest
	$(BIN)/python -m pipeline signals

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
