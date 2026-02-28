.PHONY: all install test test-fast lint format clean workflow-auto workflow-audit openclaw-opt-check

PYTHON := python3
PIP := pip3
DATA_REPO ?= /tmp/openclaws/Repos/host-adapters-experimental-data/host-adapters

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

workflow-auto:
	$(PYTHON) tools/run_workflow_auto.py --repo-root . --artifacts-root "$(DATA_REPO)"

workflow-audit:
	@RUN_DIR=$$(ls -dt "$(DATA_REPO)"/RUN_* 2>/dev/null | head -1); \
	if [ -z "$$RUN_DIR" ]; then \
		echo "No RUN_* directory found in $(DATA_REPO)"; \
		exit 1; \
	fi; \
	$(PYTHON) tools/validate_workflow_auto_run.py --run-dir "$$RUN_DIR" --output "$(DATA_REPO)/workflow_audit_latest.json" || true; \
	$(PYTHON) -c "import json; d=json.load(open('$(DATA_REPO)/workflow_audit_latest.json')); score=max(0,100-(d.get('errors',0)*20)-(d.get('warnings',0)*5)); print('RUN={run} READY={ready} SCORE={score}/100 ERRORS={errors} WARNINGS={warnings}'.format(run=d.get('run_dir'), ready=d.get('ready'), score=score, errors=d.get('errors',0), warnings=d.get('warnings',0)))"

openclaw-opt-check:
	$(PYTHON) tools/openclaw_opt_check.py --output "$(DATA_REPO)/openclaw_adapter_data/openclaw_opt_check_latest.json"
