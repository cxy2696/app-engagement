PYTHON ?= .venv/bin/python
PIP ?= $(PYTHON) -m pip
ARTIFACT ?= artifacts/demo

.PHONY: help install install-graph data benchmark train demo serve test lint format typecheck check verify report paper docker clean lock

help:
	@echo "install        install core + dev dependencies"
	@echo "install-graph  install torch + torch-geometric"
	@echo "data           build data/processed from data/raw"
	@echo "benchmark      run all configured models"
	@echo "train          train item-kNN only (fast demo)"
	@echo "demo           build data + train item-kNN + print serve command"
	@echo "serve          serve the latest item-kNN artefact on :8000"
	@echo "test           run pytest"
	@echo "lint           ruff check"
	@echo "format         ruff format"
	@echo "typecheck      mypy"
	@echo "check          lint + format check + typecheck + test"
	@echo "report         generate reports/ from artifacts/runs/benchmark.json"
	@echo "paper          build docs/paper/engagegraph-report.pdf"
	@echo "docker         build the container image"
	@echo "lock           regenerate requirements-lock.txt"
	@echo "clean          remove caches and generated runs (asks for confirmation)"

install:
	$(PIP) install -e ".[dev,als,plots]"

install-graph:
	$(PIP) install -e ".[graph]"

data:
	$(PYTHON) -m engagegraph data build --config configs/dataset.yaml --out data/processed

benchmark:
	$(PYTHON) -m engagegraph benchmark --config configs/benchmark.yaml

train:
	$(PYTHON) -m engagegraph train --config configs/itemknn.yaml

demo: data
	$(PYTHON) scripts/build_demo.py
	@echo "Build the dashboard with: make serve"

serve:
	$(PYTHON) -m engagegraph serve --artifact "$(ARTIFACT)" --port 8000

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check src tests scripts

format:
	$(PYTHON) -m ruff format src tests scripts
	$(PYTHON) -m ruff check --fix src tests scripts

typecheck:
	$(PYTHON) -m mypy src/engagegraph

check: lint test
	$(PYTHON) -m ruff format --check src tests scripts
	$(PYTHON) -m mypy src/engagegraph

verify:
	$(PYTHON) scripts/verify_reproducibility.py

report:
	$(PYTHON) -m engagegraph report --benchmark artifacts/runs/benchmark.json --out reports

paper:
	cd docs/paper && latexmk -pdf -interaction=nonstopmode main.tex && cp main.pdf engagegraph-report.pdf

docker:
	docker build -t engagegraph:2.3.1 .

lock:
	$(PYTHON) -m pip freeze --exclude-editable > requirements-lock.txt

clean:
	@echo "This removes caches and artifacts/runs. Press Ctrl-C to abort."
	@sleep 2
	rm -rf .pytest_cache .ruff_cache .mypy_cache artifacts/runs src/*.egg-info build dist
