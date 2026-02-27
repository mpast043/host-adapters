.PHONY: all install test test-fast lint format clean

PYTHON := python3
PIP := pip3

all: install test

install:
	$(PIP) install -r requirements.txt

# Full gate: policy engine tests + contract compliance suite (starts CGF server)
test:
	$(PYTHON) -m pytest -q tests/
	./tools/run_contract_suite.sh

# Quick iteration: policy engine tests only (no CGF server required)
test-fast:
	$(PYTHON) -m pytest -q tests/

lint:
	ruff check .
	ruff check --select I .
	python tools/schema_lint.py --dir ./

format:
	ruff check --select I --fix .
	ruff format .

format-js:
	npx prettier --write "**/*.{mjs,js,json}"

clean:
	rm -rf **/__pycache__ .pytest_cache *.log *.jsonl
