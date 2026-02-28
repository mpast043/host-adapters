.PHONY: all install test test-fast lint format clean workflow-auto workflow-auto-once workflow-auto-supervisor workflow-audit openclaw-opt-check local-compute-mcp framework-selection-plan

PYTHON := python3
PIP := pip3
DATA_REPO ?= /tmp/openclaws/Repos/host-adapters-experimental-data/host-adapters
WORKFLOW_MAX_CYCLES ?= 6
WORKFLOW_SLEEP_SECONDS ?= 2
WORKFLOW_UNTIL_RESOLVED ?= 0
WORKFLOW_TIER_C_AFTER_CYCLE ?=
WORKFLOW_TIER_C_JUSTIFICATION ?= Escalate Tier C to resolve persistent UNDERDETERMINED selection
WORKFLOW_RESEARCH_ON_UNDERDETERMINED ?= 0
WORKFLOW_RESEARCH_AUTO_ESCALATE_TIER_C ?= 0
WORKFLOW_FOCUS_OBJECTIVE ?= A

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
	$(PYTHON) tools/workflow_auto_multi_agent.py --repo-root . --artifacts-root "$(DATA_REPO)" --max-cycles $(WORKFLOW_MAX_CYCLES) --sleep-seconds $(WORKFLOW_SLEEP_SECONDS) --focus-objective $(WORKFLOW_FOCUS_OBJECTIVE) $(if $(filter 1,$(WORKFLOW_UNTIL_RESOLVED)),--until-resolved,) $(if $(WORKFLOW_TIER_C_AFTER_CYCLE),--tier-c-after-cycle $(WORKFLOW_TIER_C_AFTER_CYCLE),) $(if $(WORKFLOW_TIER_C_JUSTIFICATION),--tier-c-justification "$(WORKFLOW_TIER_C_JUSTIFICATION)",) $(if $(filter 1,$(WORKFLOW_RESEARCH_ON_UNDERDETERMINED)),--research-on-underdetermined,) $(if $(filter 1,$(WORKFLOW_RESEARCH_AUTO_ESCALATE_TIER_C)),--research-auto-escalate-tier-c,)

workflow-auto-supervisor:
	$(PYTHON) tools/run_workflow_auto_supervisor.py --repo-root . --artifacts-root "$(DATA_REPO)" --max-cycles $(WORKFLOW_MAX_CYCLES) --sleep-seconds $(WORKFLOW_SLEEP_SECONDS) --focus-objective $(WORKFLOW_FOCUS_OBJECTIVE) $(if $(filter 1,$(WORKFLOW_UNTIL_RESOLVED)),--until-resolved,) $(if $(WORKFLOW_TIER_C_AFTER_CYCLE),--tier-c-after-cycle $(WORKFLOW_TIER_C_AFTER_CYCLE),) $(if $(WORKFLOW_TIER_C_JUSTIFICATION),--tier-c-justification "$(WORKFLOW_TIER_C_JUSTIFICATION)",) $(if $(filter 1,$(WORKFLOW_RESEARCH_ON_UNDERDETERMINED)),--research-on-underdetermined,) $(if $(filter 1,$(WORKFLOW_RESEARCH_AUTO_ESCALATE_TIER_C)),--research-auto-escalate-tier-c,)

workflow-auto-once:
	$(PYTHON) tools/run_workflow_auto.py --repo-root . --artifacts-root "$(DATA_REPO)" --resume-latest --focus-objective $(WORKFLOW_FOCUS_OBJECTIVE)

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

local-compute-mcp:
	$(PYTHON) tools/local_compute_mcp.py

framework-selection-plan:
	$(PYTHON) tools/plan_framework_selection_tests.py --repo-root . --artifacts-root "$(DATA_REPO)"
