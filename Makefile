.PHONY: all install test lint format clean

PYTHON := python3
PIP := pip3

all: install test

install:
	$(PIP) install -r requirements.txt

test:
	$(PYTHON) -m pytest -v tools/contract_compliance_tests.py

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
