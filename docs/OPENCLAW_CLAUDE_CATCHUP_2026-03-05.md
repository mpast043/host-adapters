# OpenClaw / Claude Catch-Up (2026-03-05)

This file is the canonical sync note for the XXZ framework gate state.

## What Is Authoritative Now

1. Scope gate is **observed-data ED gate** only:
   - `experiments/physics/PHYS_BORDER_XXZ_ED_runner_v1.py`
2. Canonical gate spec artifacts:
   - `docs/SCOPE_RULE.md`
   - `docs/SCOPE_RULE.json`
3. Stable regression evidence bundle:
   - `outputs/regression_stable_20260305/REGRESSION_EVIDENCE_20260305.md`
   - `outputs/regression_stable_20260305/REGRESSION_EVIDENCE_20260305.json`

## Corrections To Old Narrative

- Literature benchmark is **not** a scope gate. It is regression/interpretation only.
- Do **not** encode "gapped => c=0.5" in gate logic.
- Scope decisions are based on `delta_aicc_sat_minus_log = AICc_sat - AICc_log` from observed `S(L/2, L)`.
- Solver overlap failure blocks pooling and forces `INCONCLUSIVE`.

## Final Three Regression Checks (Stable Evidence)

A) Solver mismatch (degraded DMRG)
- Path: `outputs/regression_stable_20260305/A_solver_mismatch/run_ed_25c6c279/xxz_boundary_results.json`
- Result: `INCONCLUSIVE` with `SOLVER_MISMATCH`, pooling disabled.

B) Solver pass bulk (parity-controlled)
- Path: `outputs/regression_stable_20260305/B_solver_pass_bulk/xxz_boundary_results.json`
- Result:
  - `Δ=0.5` -> `OUT_OF_SCOPE` / `REJECT` (log preferred)
  - `Δ=2.0` -> `IN_SCOPE` / `ACCEPT` (saturation preferred)
  - overlap pass + pooled fit used.

C) Mixed-even warning/oscillation test
- Paths:
  - `outputs/regression_stable_20260305/C_mixed_even_warning/xxz_boundary_results.json`
  - `outputs/regression_stable_20260305/C_mixed_even_warning/run.log`
- Result: explicit mixed-even parity warning and oscillation-contaminated signature (`alternation_fraction=1.0`, oscillation-corrected fit selected).

## Operational Default

Use `L mod 4 = 0` grid for gate runs unless explicitly running an oscillation diagnostic.
