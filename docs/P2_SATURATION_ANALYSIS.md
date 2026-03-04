# P2 Saturation Analysis: Ising vs Heisenberg

**Date**: 2026-03-04
**Status**: COMPLETE

## Executive Summary

P2 (Capacity Plateau) successfully distinguishes Ising from Heisenberg entanglement scaling behavior. Ising shows saturation within the capacity framework; Heisenberg does not.

---

## Test Results

### L=8 Results

| Model | L | A_size | ΔAIC | Verdict | Model Preferred |
|-------|---|--------|------|---------|-----------------|
| Ising | 8 | 4 | **+0.66** | ACCEPT | Saturating |
| Heisenberg | 8 | 4 | **-3.45** | REJECT | Log-linear |

### L=12 Results (Heisenberg)

| Model | L | A_size | ΔAIC | Verdict | Model Preferred |
|-------|---|--------|------|---------|-----------------|
| Heisenberg | 12 | 6 | **-4.29** | REJECT | Log-linear |

---

## Finite-Size Scaling: Heisenberg

| Metric | L=8 | L=12 | Trend |
|--------|-----|------|-------|
| ΔAIC | -3.45 | -4.29 | More negative |
| \|ΔAIC\| | 3.45 | 4.29 | **Increasing** |
| E₀ | -3.375 | -5.142 | Lower ground state |
| S_ref | 0.457 | 0.537 | Higher entropy |

**Interpretation**: Heisenberg's preference for log-linear scaling strengthens at larger L, indicating no saturation behavior.

---

## Physical Interpretation

| Model | Entanglement Scaling | Capacity-Limited? | Ground State Structure |
|-------|---------------------|-------------------|------------------------|
| **Ising** | Saturating (S → S_∞) | **YES** | Bounded entanglement |
| **Heisenberg** | Log-linear (S ∝ log χ) | **NO** | Critical-like scaling |

### Why Heisenberg Doesn't Saturate

1. **SU(2) symmetry**: Heisenberg has continuous symmetry, Ising has discrete Z₂
2. **Gapless excitations**: Heisenberg in 1D is gapless, Ising is gapped
3. **Entanglement entropy**: Heisenberg shows logarithmic divergence, Ising saturates
4. **Critical behavior**: Heisenberg is at a quantum critical point, Ising is not

---

## Claim Status

| Claim | Status | Evidence |
|-------|--------|----------|
| P2 Capacity Plateau | **PARTIAL** | Ising ACCEPT, Heisenberg REJECT |
| Heisenberg saturation | **REJECTED** | No plateau L=8→12 |
| Ising saturation | **SUPPORTED** | ΔAIC +0.66 |

---

## P3 Note

P3 (Gluing Stability) was found to have a **methodological flaw**: it tests naive tensor products, not MERA isometric operations. P3 results are **INVALID** for physics conclusions. See separate analysis.

---

## Data Locations

| Test | Location |
|------|----------|
| Ising L=8 | `runs/RUN_20260304_1243/results/physics/baseline/P2_ising/` |
| Heisenberg L=8 | `runs/RUN_20260304_1237/results/physics/baseline/P2_heisenberg/` |
| Heisenberg L=12 | `runs/RUN_20260304_1243/results/physics/baseline/P2_heisenberg_L12/` |

---

## Next Steps

1. Test Ising L=12 to confirm saturation persists
2. Document Heisenberg anomaly as theory gap
3. Consider extended framework for critical systems

---

## References

- Run ID: `RUN_20260304_1243`
- Commits: `4d7fde3`, `e2fe1ee`, `876ed1e`
- Framework: PHYSICS_BASELINE_SPEC_v0.2.1