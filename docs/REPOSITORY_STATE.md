# Repository State Audit

**Audit Date:** 2026-03-01
**Task:** 1.1 - Audit Current State (Framework Physics Validation)
**Purpose:** Document repository state before committing physics results

---

## Summary

| Category | Count | Action |
|----------|-------|--------|
| Modified Files | 50+ | Review for commit |
| Untracked Files | 30+ | Triage needed |
| Critical Physics Results | 6 | **KEEP - commit** |
| Working Files | 10 | May archive |
| Generated Outputs | 5 | May regenerate |

---

## Modified Files (50 files)

These files have uncommitted changes in the working directory:

### Core Framework (10 files)
| File | Lines Changed | Category |
|------|---------------|----------|
| `cgf_policy/__init__.py` | 2 | Framework core |
| `cgf_policy/compiler.py` | 72 | Framework core |
| `cgf_policy/evaluator.py` | 77 | Framework core |
| `cgf_policy/fields.py` | 11 | Framework core |
| `cgf_policy/types.py` | 34 | Framework core |
| `server/cgf_schemas_v03.py` | 429 | Server schemas |
| `server/cgf_server_v03.py` | 607 | Server implementation |
| `sdk/python/cgf_sdk/adapter_base.py` | 183 | SDK |
| `sdk/python/cgf_sdk/cgf_client.py` | 140 | SDK |
| `sdk/python/cgf_sdk/errors.py` | 125 | SDK |

### Adapters (3 files)
| File | Lines Changed | Category |
|------|---------------|----------|
| `adapters/langgraph_adapter_v01.py` | 598 | Adapter implementation |
| `adapters/openclaw_adapter_v01.py` | 465 | Adapter implementation |
| `adapters/openclaw_adapter_v02.py` | 560 | Adapter implementation |

### Experiments (14 files)
| File | Lines Changed | Category |
|------|---------------|----------|
| `experiments/claim3/exp3_claim3_entanglement_max_mincut_runner.py` | 278 | Claim 3 experiment |
| `experiments/claim3/exp3_claim3_optionB_runner.py` | 254 | Claim 3 experiment |
| `experiments/claim3/exp3_claim3_physical_convergence_runner.py` | 441 | Claim 3 experiment |
| `experiments/claim3/exp3_claim3_quimb_runner.py` | 271 | Claim 3 experiment |
| `experiments/claim3/w03_controls_runner.py` | 53 | Claim 3 controls |
| `prototype/experiments/exp1_spectral_dim/exp1_spectral_dim.py` | 72 | Prototype |
| `prototype/experiments/exp2_mera_tradeoff/exp2_mera_tradeoff.py` | 192 | Prototype |
| `prototype/experiments/exp2_mera_tradeoff/exp2b_asymptotic.py` | 186 | Prototype |
| `prototype/experiments/exp3_mera_spectral/exp3_entanglement_entropy.py` | 345 | Prototype |
| `prototype/experiments/exp3_mera_spectral/exp3_mera_spectral.py` | 277 | Prototype |
| `prototype/experiments/exp3_mera_spectral/exp3_mera_spectral_v2.py` | 296 | Prototype |
| `prototype/experiments/exp3b_windowed_regime/exp3b_windowed_regime.py` | 378 | Prototype |
| `prototype/experiments/exp3b_windowed_regime/run_extended.py` | 35 | Prototype |
| `prototype/experiments/exp3b_windowed_regime/test_kcut_*.py` | 113 | Prototype tests |

### Tests (5 files)
| File | Category |
|------|----------|
| `tests/test_circuit_breaker.py` | Unit test |
| `tests/test_framework_selection_planner.py` | Unit test |
| `tests/test_outcome_reporting.py` | Unit test |
| `tests/test_policy_engine.py` | Unit test |
| `tests/test_server_auth.py` | Unit test |

### Tools (13 files)
| File | Category |
|------|----------|
| `tools/contract_compliance_tests.py` | Testing tool |
| `tools/framework_claim_checks.py` | Physics validation |
| `tools/local_compute_mcp.py` | Compute infrastructure |
| `tools/normalize_selection_ledger.py` | Data normalization |
| `tools/openclaw_opt_check.py` | Optimization check |
| `tools/plan_framework_selection_tests.py` | Test planning |
| `tools/replay_governance_timeline.py` | Replay tool |
| `tools/replay_verify.py` | Verification tool |
| `tools/research_framework_selection.py` | Research tool |
| `tools/run_contract_suite.sh` | Script |
| `tools/run_physics_audit.py` | Physics audit |
| `tools/run_workflow_auto.py` | Workflow automation |
| `tools/run_workflow_auto_supervisor.py` | Workflow supervisor |

### Documentation (1 file)
| File | Category |
|------|----------|
| `docs/physics/framework_selection_test_catalog_v1.json` | Test catalog (+102 lines) |

---

## Untracked Files (30+ files)

### Critical Physics Results (KEEP - Commit These)

| File | Reason |
|------|--------|
| `docs/physics/FRAMEWORK_SELECTION_PHYSICS_RESULTS_PAPER_v1.md` | **PRIMARY OUTPUT** - Physics results paper |
| `experiments/physics/spec_P1.md` | Experiment specification P1 |
| `experiments/physics/spec_P2.md` | Experiment specification P2 |
| `experiments/physics/spec_P3.md` | Experiment specification P3 |
| `experiments/physics/provenance_P1.json` | Experiment provenance P1 |
| `experiments/physics/provenance_P2.json` | Experiment provenance P2 |
| `experiments/physics/provenance_P3.json` | Experiment provenance P3 |

### Working Files (May Archive)

| File | Reason |
|------|--------|
| `docs/plans/2026-03-01-framework-selection-testing-plan.md` | Planning document |
| `docs/plans/2026-03-02-framework-physics-validation-plan.md` | Planning document |
| `memory/2026-02-27.md` | Working memory |
| `memory/2026-02-28.md` | Working memory |
| `WORKFLOW_AUTO.md` | Workflow documentation |
| `config/` | Configuration directory |
| `jobs/` | Jobs directory |
| `tools/batch_status.py` | Batch tool |
| `tools/batch_worker.json` | Batch configuration |
| `tools/batch_worker.py` | Batch tool |
| `tools/fetch_results.py` | Results fetcher |
| `tools/generate_framework_traceability_summary.py` | Summary generator |

### Generated Outputs (May Regenerate)

| File | Reason |
|------|--------|
| `experiments/physics/patch_P1.diff` | Generated diff |
| `experiments/physics/patch_P2.diff` | Generated diff |
| `experiments/physics/patch_P3.diff` | Generated diff |
| `experiments/physics/baseline/` | Baseline data directory |
| `experiments/physics/exp_p1_spectral_dimension_runner.py` | Generated runner |
| `experiments/physics/exp_p2_capacity_plateau_runner.py` | Generated runner |
| `experiments/physics/exp_p2_runner.py` | Generated runner |
| `experiments/physics/exp_p3_gluing_excision_stability_runner.py` | Generated runner |

### Temporary/Artifact Files (Consider Removing)

| File | Reason |
|------|--------|
| `LAST_RUN_PATH.txt` | Temporary path file |
| `PYEOF` | Likely temp file artifact |
| `sdk/python/examples/min_adapter.py` | Example (modified) |

---

## Git Status Summary

```
Last commit: 062f1e0 docs: major revision of physics validation roadmap after literature review
Branch state: 50+ modified files, 30+ untracked files
```

---

## Recommended Actions

### Immediate (Before Physics Commit)
1. Review modified files for completeness
2. Stage critical physics results for commit
3. Archive or remove temporary files

### For Commit Triage
- **Commit First:** Physics results paper, specs, provenance files
- **Commit Second:** Modified experiment runners, framework changes
- **Commit Third:** Tools and tests
- **Consider Excluding:** Temporary files (PYEOF, LAST_RUN_PATH.txt)

---

## Notes

- This audit captures state as of 2026-03-01
- No commits were made during this audit
- File categorization is based on path patterns and naming conventions
- Full content review may be needed for final categorization