# Framework With Selection: Start-to-Finish Checklist

Date: 2026-02-28  
Canonical repo: `/tmp/openclaws/Repos/host-adapters`  
Canonical data repo: `/tmp/openclaws/Repos/host-adapters-experimental-data/host-adapters`  
Canonical proposal PDF: `/Users/meganpastore/Clawdbot/Repos/capacity-demo/Framework with selection.pdf`

## 0. Setup and Canonical Paths

1. Use `host-adapters` as code/truth for workflow logic.
2. Write all `RUN_*` artifacts to external data repo.
3. Confirm latest run path exists under `DATA_REPO`.

## 1. Importance Gate (Before Scheduling Tests)

For each proposed test, require:
1. `decision_if_pass`
2. `decision_if_fail`
3. `critical_path_objective` (`A`, `B`, `C`)
4. `tier` (`A`, `B`, `C`)

Tier policy:
1. Tier A always allowed.
2. Tier B allowed only if it protects supported claims or prevents regression.
3. Tier C blocked unless `RUN_OVERRIDE_TIER_C=1` plus one-line justification.

Stop-condition policy:
1. If only Tier C remains and budget remains, reallocate to Tier A hardening.

## 2. Research Stage (Local + External)

1. Read local signals from latest run (`campaign_index.csv`, `workflow_auto_status.json`).
2. Query external sources (arXiv/Crossref) with objective-aware queries.
3. Produce ranked evidence and recommended execution keys.
4. Keep objective focus:
   - `A`: readiness/hardening
   - `B`: Claim 2 and Claim 3 regressions
   - `C`: Claim 3P exploration

## 3. Plan Stage (Deterministic Queue)

1. Run planner against `framework_selection_test_catalog_v1.json`.
2. Enforce Step 0 + tier policy + compute availability.
3. Generate:
   - `results/science/exploration_proposals.json`
   - `results/science/exploration_selected.json`
   - `results/science/exploration_proposals.md`

## 4. Test Stage (Agentic Execution)

1. Use multi-agent loop:
   - planner
   - researcher
   - executor
2. Execute selected tests only.
3. Write campaign evidence:
   - `results/science/campaign/campaign_index.csv`
   - `results/science/campaign/campaign_report.md`
   - per-test artifacts + verdicts

## 5. Assess Stage (Selection + Verdict)

1. Build selection outputs:
   - `results/selection/selection_manifest.json`
   - `results/selection/ledger.jsonl`
   - `results/selection/selection_report.md`
   - `results/selection/evidence_index.json`
2. Objective-focused conclusive rule:
   - If focus is `A`, `B`, or `C` and selection is `ACCEPTED` or `REJECTED`, mark run `COMPLETE`.
   - If selection is `UNDERDETERMINED`, continue looping.

## 6. Iterate Until Resolved

1. For objective `B` (math/physics supported claims), keep looping until conclusive.
2. Use external research every underdetermined cycle.
3. Keep Tier C gated unless explicitly overridden.

## 7. Retention and Audit

1. Package retained run artifact (`retained_*.tar.gz`).
2. Validate contract schema:
   - `tools/validate_workflow_auto_run.py`
3. Preserve replayability and witness traceability.

## 8. Primary Commands

Objective B autonomous loop (recommended):

```bash
cd /tmp/openclaws/Repos/host-adapters
make workflow-physics-auto DATA_REPO=/tmp/openclaws/Repos/host-adapters-experimental-data/host-adapters
```

General agentic loop with custom focus:

```bash
cd /tmp/openclaws/Repos/host-adapters
make workflow-auto DATA_REPO=/tmp/openclaws/Repos/host-adapters-experimental-data/host-adapters WORKFLOW_FOCUS_OBJECTIVE=B WORKFLOW_UNTIL_RESOLVED=1 WORKFLOW_RESEARCH_ON_UNDERDETERMINED=1
```

Explicit Tier C override (only when intentionally exploring speculative branch):

```bash
cd /tmp/openclaws/Repos/host-adapters
RUN_OVERRIDE_TIER_C=1 make workflow-auto DATA_REPO=/tmp/openclaws/Repos/host-adapters-experimental-data/host-adapters WORKFLOW_FOCUS_OBJECTIVE=C WORKFLOW_UNTIL_RESOLVED=1 WORKFLOW_RESEARCH_ON_UNDERDETERMINED=1 WORKFLOW_TIER_C_JUSTIFICATION="Explicit Tier C approval"
```

## 9. Completion Definition for Framework Evaluation

A run is considered complete for framework decisioning when:
1. Contract artifacts validate.
2. Selection ledger is complete and traceable.
3. Focused objective reaches conclusive `ACCEPTED` or `REJECTED`.
4. Claims are labeled by tier and evidence strength (internal consistency vs reproducibility vs physical interpretation).
