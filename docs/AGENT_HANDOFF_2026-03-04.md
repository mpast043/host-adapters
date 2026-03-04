# Agent Handoff (Claude Code / OpenClaw)

Date: 2026-03-04
Repo: `/tmp/openclaws/Repos/host-adapters`

## Purpose

This note captures the cleanup after the physics runner renaming/import regression so any agent can continue without re-debugging import failures.

## What Was Broken

After runner renames (`exp_*` to `PHYS_*`), compatibility drift introduced two issues:

1. `PHYS_CAPACITY_PLATEAU_runner_v2.py` and `PHYS_BORDER_XXZ_runner_v1.py` imported:
   - `from claim3.PHYS_PHYSICAL_CONVERGENCE_runner_v2 import optimize_mera_for_fidelity`
   - but `optimize_mera_for_fidelity` no longer existed in v2.

2. `PHYS_GLUING_STABILITY_v1_runner_v2.py` imported:
   - `claim3.exp3_claim3_physical_convergence_runner` (legacy module path no longer present).

## Fixes Applied

### 1) Restored v2 compatibility API

File:
- `experiments/physics/claim3/PHYS_PHYSICAL_CONVERGENCE_runner_v2.py`

Changes:
- Added backwards-compatible function:
  - `optimize_mera_for_fidelity(...) -> OptimizationResult`
  - wraps `optimize_mera_for_model(...)` and returns the expected object contract used by P2/P3 callers.
- Extended `OptimizationResult` with defaulted legacy fields:
  - `converged: bool = True`
  - `num_steps: Optional[int] = None`

### 2) Fixed claim3 package exports

File:
- `experiments/physics/claim3/__init__.py`

Changes:
- Exported both:
  - `optimize_mera_for_fidelity`
  - `optimize_mera_for_model`
- Kept `exact_diagonalization`, `Config`, `EDResult` exports.

### 3) Fixed stale legacy import in old P3 v2 runner

File:
- `experiments/physics/PHYS_GLUING_STABILITY_v1_runner_v2.py`

Changes:
- Repointed import from removed module path to:
  - `claim3.PHYS_PHYSICAL_CONVERGENCE_runner_v2`

## Validation Performed

All three runners now import and parse CLI args successfully:

- `python3 experiments/physics/PHYS_CAPACITY_PLATEAU_runner_v2.py --help` -> exit 0
- `python3 experiments/physics/PHYS_BORDER_XXZ_runner_v1.py --help` -> exit 0
- `python3 experiments/physics/PHYS_GLUING_STABILITY_v1_runner_v2.py --help` -> exit 0

Package API check:

- `from claim3 import exact_diagonalization, optimize_mera_for_fidelity, optimize_mera_for_model, Config, EDResult` -> OK

Runtime smoke check:

- `exact_diagonalization(L=4, ...)` + `optimize_mera_for_fidelity(...)` returns an `OptimizationResult` object with expected fields.

## Known Environment Caveats (Not Code Regressions)

1. OpenMP duplicate runtime warning may appear on this machine:
   - `OMP: Error #15 ... libomp.dylib already initialized`
   - Workaround for local smoke tests: `KMP_DUPLICATE_LIB_OK=TRUE`

2. Running TN optimization from `python - <<'PY'` can trigger multiprocessing spawn issues (`<stdin>` path):
   - Use script file execution instead of stdin for full runs.

## Current Focused Files

- `experiments/physics/claim3/PHYS_PHYSICAL_CONVERGENCE_runner_v2.py`
- `experiments/physics/claim3/__init__.py`
- `experiments/physics/PHYS_GLUING_STABILITY_v1_runner_v2.py`

## Suggested Next Step

Run one short real command from file context (not stdin) to confirm numerical pipeline stability, e.g.:

`python3 experiments/physics/PHYS_CAPACITY_PLATEAU_runner_v2.py --L 8 --A_size 4 --model ising_cyclic --chi_sweep 2,4 --fit_steps 5 --seed 42 --output /tmp/p2_smoke`

