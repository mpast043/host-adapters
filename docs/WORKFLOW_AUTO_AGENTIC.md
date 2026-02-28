# WORKFLOW_AUTO Agentic Mode

`make workflow-auto` now uses a multi-agent coordinator with three scoped roles:

- `planner`: runs `tools/plan_framework_selection_tests.py`
- `researcher`: runs `tools/research_framework_selection.py`
- `executor`: runs `tools/run_workflow_auto.py`

Each role runs in a separate subprocess via `tools/workflow_auto_agent_role.py` and is restricted to role-allowlisted commands.

## Default behavior

- Focus objective is `A` (platform readiness) by default.
- Tier C auto-escalation is disabled by default.
- Tier C remains blocked unless explicitly enabled via override flags.
- Under this default, autonomy iterates on Tier A evidence and alternatives until a conclusive readiness status is reached.

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
