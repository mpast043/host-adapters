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

## Literature Support

### Key References (from existing research notes)

| Paper | Key Finding | Relevance |
|-------|-------------|-----------|
| de Boer et al. (PRD 2019) | Capacity = Var(H_A), not S | Framework mapping |
| Lyu et al. (PRR 2021) | Tensor RG extracts d_s | Method for staircase |
| Wald et al. (PRR 2020) | Gap closes as π²/ln(L) at criticality | Explains ΔAIC behavior |

### Physical Explanation

**Why Heisenberg doesn't saturate:**

| Model | c | Gap | Entanglement | Capacity Bound? |
|-------|---|-----|--------------|-----------------|
| Ising | 1/2 | Gapped | Bounded | YES |
| Heisenberg | 1 | Gapless | log(L) growth | NO |

**Literature confirms:** Heisenberg is a 1D quantum critical system with logarithmically growing entanglement entropy. The capacity framework applies to gapped systems with bounded entanglement.

### Prior Experimental Evidence

From `docs/RESULTS_REFERENCE.md`:

| Prior Result | Our Confirmation |
|--------------|-------------------|
| Heisenberg fidelity ~0.68 | Confirmed different behavior |
| Cyclic ΔAIC +0.66 vs open -10.83 | Confirmed Ising cyclic: +0.66 |
| Log-linear preferred for all | Confirmed (ΔAIC negative for Heisenberg) |

---

## References

- Run ID: `RUN_20260304_1243`
- Commits: `4d7fde3`, `e2fe1ee`, `876ed1e`
- Framework: PHYSICS_BASELINE_SPEC_v0.2.1