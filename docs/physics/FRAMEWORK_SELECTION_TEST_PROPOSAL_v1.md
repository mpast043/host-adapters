# Framework With Selection Test Proposal v1

Date: 2026-02-28  
Canonical repo: `/tmp/openclaws/Repos/host-adapters`  
Canonical PDF: `/Users/meganpastore/Clawdbot/Repos/capacity-demo/Framework with selection.pdf`

## Known Starting Point

1. Internal framework consistency is currently supported by local Tier A checks (`13/13` and `12/12` pass logs captured in external data repo artifacts).
2. Independent local Claim 3P matrix reruns (open/cyclic, Ising/Heisenberg, L=8 and L=16) are reproducible and currently trend to rejection under present falsifiers.
3. PDF scope language explicitly limits physical claims:
   - p58: no claim of new physical dynamics.
   - p59: capacity boundaries are not automatically physical phase transitions.
4. PDF stress/regression section (p60-p64) reports internal consistency passes; these are internal checks, not external physical establishment.

## Step 0: Importance Gate (must pass before any test is scheduled)

For every proposed test, answer:

1. What decision changes if this test passes vs fails?
2. Which critical path objective does it serve?
   A. Platform readiness (MCP execution, contract suite, artifact retention, reproducibility, fail-modes)  
   B. Supported claims regression (Claim 2, Claim 3)  
   C. Research branch exploration (Claim 3P, speculative variants)

Tier rules:
- Tier A: always allowed; highest priority.
- Tier B: allowed if it protects a SUPPORTED claim or prevents regression.
- Tier C: blocked by default. Requires explicit override: `RUN_OVERRIDE_TIER_C=1` and a one-line justification.

Stop condition:
If time budget remains and only Tier C work is queued, reallocate to Tier A hardening tasks.

## What Is Now Enforced in Code

Planner: `tools/plan_framework_selection_tests.py`  
Catalog: `docs/physics/framework_selection_test_catalog_v1.json`

Enforcement:
1. Step 0 fields are mandatory (`decision_if_pass`, `decision_if_fail`, `critical_path_objective`, `tier`).
2. Tier B admission is blocked unless regression-protective metadata is true.
3. Tier C is hard-blocked unless override is enabled.
4. Compute-required tests are blocked when compute target is unavailable.
5. Budget scheduler is deterministic and emits both JSON and Markdown plans.
6. Remaining-budget stop condition reallocation to Tier A hardening is explicit and recorded.

## Workflow_AUTO Integration

`tools/run_workflow_auto.py` now uses the planner in Step 4 instead of static exploration proposals.

Additional behavior added:
1. Step 0 writes `results/capabilities.json` with MCP/tool/skill availability snapshot.
2. Step 4 planning artifacts:
   - `results/science/exploration_proposals.json`
   - `results/science/exploration_proposals.md`
   - `results/science/exploration_selected.json`
3. Step 4 execution supports platform hardening tasks and science tests, logs all commands, and records skipped planned tests.

## End-to-End Autonomous Loop (Practical Operation)

1. Discover capabilities and MCP targets.
2. Run baseline contract + science suites.
3. Build gated test queue with Step 0 and tier policy.
4. Execute selected queue under budget and safety constraints.
5. Aggregate evidence and write selection ledger with witness pointers.
6. Re-plan next queue from new artifacts; repeat until:
   - completion criteria met, or
   - hard-stop failure class reached (contract/runtime/selection).

## Completion Criteria for Physics-Facing Decision

A run can report a final completion only when all are true:
1. Workflow contract artifacts pass validation.
2. Selection ledger is complete (witnesses + underdetermined next-test recommendation).
3. Physics status is explicitly layered:
   - internal consistency,
   - numerical reproducibility,
   - external physical claim strength.
4. Any remaining speculative items are labeled Tier C / underdetermined.

## Commands

Plan only:
```bash
cd /tmp/openclaws/Repos/host-adapters
make framework-selection-plan
```

Run end-to-end workflow:
```bash
cd /tmp/openclaws/Repos/host-adapters
make workflow-auto
```

Allow Tier C exploration explicitly:
```bash
cd /tmp/openclaws/Repos/host-adapters
RUN_OVERRIDE_TIER_C=1 python3 tools/run_workflow_auto.py \
  --repo-root . \
  --artifacts-root /tmp/openclaws/Repos/host-adapters-experimental-data/host-adapters \
  --tier-c-justification "Need Claim 3P branch evidence for boundary-condition sensitivity."
```
