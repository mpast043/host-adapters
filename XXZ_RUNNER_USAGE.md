# XXZ Runner Usage Guide

This document defines which XXZ runner should be used for which purpose.

## Runner Roles

| Runner | Primary purpose | Produces fits.json | Suitable for framework correctness gate |
|---|---|---|---|
| `experiments/physics/PHYS_BORDER_XXZ_BENCHMARK_runner_v1.py` | Literature/theory baseline (phase expectations) | No (`result.json` only) | No |
| `experiments/physics/PHYS_BORDER_XXZ_ED_runner_v1.py` | Physics truth baseline from exact diagonalization (ED) | No (stores ED model fits in aggregate JSON) | Yes (for scope/sign logic) |
| `experiments/physics/PHYS_BORDER_XXZ_runner_v1.py` | End-to-end integration with MERA fitting | Yes (`delta_*/fits.json`) | Not as sole gate (uses Δ-phase proxy models for now) |

## Recommendation

Use a two-step policy:

1. **Correctness gate (required):** `PHYS_BORDER_XXZ_ED_runner_v1.py`
2. **Integration regression (informational):** `PHYS_BORDER_XXZ_runner_v1.py`

## Correctness Gate Spec (locked)

The ED gate is authoritative and uses observed data only:

- Data: observed `S(L/2, L)` from OBC XXZ ED
- Even `L` only (keeps `ell=L/2` integer by construction)
- No theory formula in gate decision path

Model definitions and parameter counts:

- Critical candidate: `S(L) = alpha*ln(L) + beta` (`k=2`, natural log)
- Gapped candidate: `S(L) = S_inf - A*exp(-L/xi)` (`k=3`)
- Physical domains: `S_inf >= 0`, `A >= 0`, `xi > 0`
- Oscillation-corrected candidates (used in `auto`/`force` modes when strong alternation is present):
  - `S(L) = alpha*ln(L) + beta + B*(-1)^(L/2)*exp(-L/lambda)` (`k=4`)
  - `S(L) = S_inf - A*exp(-L/xi) + B*(-1)^(L/2)*exp(-L/lambda)` (`k=5`)

Model selection and decision:

- Record both AIC and AICc with explicit `n` and `k` per model
- `delta_aicc_sat_minus_log = AICc_sat - AICc_log`
- Decision thresholds:
  - `delta <= -2` -> `IN_SCOPE` (saturating preferred)
  - `delta >= +2` -> `OUT_OF_SCOPE` (log-linear preferred)
  - `|delta| < 2` -> `INCONCLUSIVE`

Pre-registered Δ sets:

- Bulk set (expect decisive): `0.5, 0.8, 1.4, 2.0`
- Boundary set (inconclusive acceptable near transition): `0.95, 1.00, 1.05, 1.10`

Solver overlap calibration (ED + DMRG):

- Overlap points (recommended): `L = 12, 14, 16`
- Strict policy: **all overlap points must pass** before pooling
- Per overlap point diagnostics include `|S_ED-S_DMRG|`, relative error, pass/fail
- If overlap check fails or overlap data are missing: gate result is `INCONCLUSIVE`
- DMRG error handling: failed DMRG points must be encoded as `status=ERROR` with no numeric entropy (`S_dmrg` absent/null). Numeric fallback defaults are not allowed.

Pooling modes:

- `ed_only` (default): fit on ED points only
- `pool_if_overlap_pass`: pool DMRG-only points into the fit series **only** when strict overlap status is `PASS`
- `--ed-max-L` can cap ED cost (for example ED to `L<=16`) while still allowing larger-L DMRG points to be pooled after calibration pass.

Oscillation handling:

- `--oscillation-mode auto` (default): detect strong residual sign alternation and switch to oscillation-corrected fits only if AICc improves enough.
- `--oscillation-mode off`: baseline fits only.
- `--oscillation-mode force`: use oscillation-corrected fits whenever alternation is detectable.

Interpretation of `INCONCLUSIVE`:

- `AICC_BAND_INDETERMINATE` or `AICC_UNDEFINED`: insufficient evidence for model preference (often small `n` / near boundary)
- `SOLVER_MISMATCH` or `SOLVER_OVERLAP_MISSING`: solver calibration prerequisite failed
- `SOLVER_INPUT_ERROR`: DMRG overlap input reported non-OK status or missing entropy at required overlap points

Important: `INCONCLUSIVE` in the pre-registered boundary set is acceptable, expected, and not a gate failure.

## Why

- The literature benchmark encodes expected labels and is useful as a reference, but it does not compute observed framework fit outcomes.
- The ED runner computes model-comparison behavior from observed exact diagonalization outputs and is the strongest available source for correctness checks.
- The full XXZ runner currently approximates Δ regimes with proxy models (`ising_open` / `heisenberg_open`) for optimization, so it is valuable for regression but should not be the only correctness authority.

## CI Guidance

- Block merges on ED scope/sign checks passing.
- Run full XXZ integration as non-blocking until true per-Δ XXZ optimization is implemented.
