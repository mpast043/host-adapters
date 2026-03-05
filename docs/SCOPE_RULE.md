# Scope Rule (Canonical)

This artifact defines the authoritative framework scope gate for XXZ boundary classification.

## Authoritative Gate Definition

- Gate runner: `experiments/physics/PHYS_BORDER_XXZ_ED_runner_v1.py`
- Data source for gating: observed `S(L/2, L)` only
- Boundary condition: `OBC`
- Partition rule: `ell = L/2`
- Model comparison metric: `delta_aicc_sat_minus_log = AICc_sat - AICc_log`
- Fit families:
  - Critical candidate: `S(L) = alpha*ln(L) + beta`
  - Gapped candidate: `S(L) = S_inf - A*exp(-L/xi)`
- Optional finite-size correction (auto/force modes):
  - `+ B*(-1)^(L/2)*exp(-L/lambda)` on both candidates

## Allowed Grids

- Default grid (parity-controlled): `L ≡ 0 (mod 4)`, e.g. `8,12,16,20,24,28,32,36,40,44`
- Alternate grid: even-L only, with explicit finite-size diagnostics required
- ED policy: ED may be capped (e.g. `ed_max_L=16`) with DMRG supplying larger-L points

## Decision Bands

- `delta_aicc_sat_minus_log <= -2` -> `IN_SCOPE`
- `delta_aicc_sat_minus_log >= +2` -> `OUT_OF_SCOPE`
- `|delta_aicc_sat_minus_log| < 2` -> `INCONCLUSIVE`

## Solver Mismatch Behavior

- Overlap calibration is required before ED+DMRG pooling.
- Strict mode: all overlap points must pass tolerance.
- If overlap status is `MISMATCH`, `MISSING`, or `INPUT_ERROR`:
  - Gate verdict must be `INCONCLUSIVE`
  - Gate reason must reflect calibration failure (`SOLVER_MISMATCH`, `SOLVER_OVERLAP_MISSING`, `SOLVER_INPUT_ERROR`)
  - AICc may be computed for audit only, but is not allowed to decide scope.

## Proxy Policy

- Proxy/literature/integration runners are regression tools only.
- They are not allowed to override authoritative gate decisions.
- Canonical proxy status:
  - `experiments/physics/PHYS_BORDER_XXZ_BENCHMARK_runner_v1.py` -> theory baseline only
  - `experiments/physics/PHYS_BORDER_XXZ_runner_v1.py` -> integration regression only

