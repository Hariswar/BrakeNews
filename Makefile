PY ?= python3.12
VENV := .venv
BIN := $(VENV)/bin

.PHONY: setup setup-site build ingest clean-data spotcheck test lint site-dev site-build

setup:            ## Python env with every pipeline dependency
	$(PY) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install -e ".[all]"

setup-site:       ## Node dependencies for the Astro site
	cd site && npm install

build:            ## One-command rebuild: raw NHTSA files -> clean tables
	$(BIN)/python -m pipeline build

ingest:
	$(BIN)/python -m pipeline ingest

clean-data:
	$(BIN)/python -m pipeline clean

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
