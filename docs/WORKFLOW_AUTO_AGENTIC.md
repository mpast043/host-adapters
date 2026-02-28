# WORKFLOW_AUTO Agentic Mode

`make workflow-auto` now uses a multi-agent coordinator with three scoped roles:

- `planner`: runs `tools/plan_framework_selection_tests.py`
- `researcher`: runs `tools/research_framework_selection.py`
- `executor`: runs `tools/run_workflow_auto.py`

Each role runs in a separate subprocess via `tools/workflow_auto_agent_role.py` and is restricted to role-allowlisted commands.

## Default behavior

- Research is enabled on underdetermined cycles.
- Research may auto-escalate Tier C when evidence indicates repeated underdetermination.
- Tier C remains blocked unless escalation/override is active.

## Commands

Run agentic mode:

```bash
make workflow-auto DATA_REPO=/tmp/openclaws/Repos/host-adapters-experimental-data/host-adapters
```

Run legacy supervisor mode:

```bash
make workflow-auto-supervisor DATA_REPO=/tmp/openclaws/Repos/host-adapters-experimental-data/host-adapters
```

## Key outputs

- Agentic role outputs:
  - `RUN_*/results/agentic/planner_result_cycle_*.json`
  - `RUN_*/results/agentic/researcher_result_cycle_*.json`
  - `RUN_*/results/agentic/executor_result_cycle_*.json`
- Agentic event ledger:
  - `RUN_*/logs/agentic_events.jsonl`

