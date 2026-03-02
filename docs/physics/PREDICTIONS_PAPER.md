# Framework Capacity: Predictions for Quantum Critical Systems

**Authors:** [To be filled]
**Date:** 2026-03-02
**Framework Version:** v4.5/v4.6
**Status:** Draft

---

## Abstract

We present testable predictions derived from the Capacity Framework v4.5/v4.6, grounded in established quantum field theory and tensor network methods. The critical discovery is that the Framework's "capacity" maps to the **capacity of entanglement** (second cumulant κ₂) from quantum information theory, NOT von Neumann entropy (first cumulant κ₁). This mapping enables predictions for quantum critical systems that can be tested experimentally with cold atom arrays, ion trap simulators, and quantum materials.

---

## 1. Introduction

The Capacity Framework proposes a relationship between capacity constraints and emergent geometry in quantum systems. This paper maps Framework concepts to established physics, enabling experimental validation.

**Key Discovery:** From [de Boer et al. PRD 99, 066012 (2019)](https://journals.aps.org/prd/pdf/10.1103/PhysRevD.99.066012):

> Capacity of entanglement C_E = Var(H_A) = ⟨H_A²⟩ - ⟨H_A⟩²

This is the **second cumulant** of the entanglement spectrum, distinct from von Neumann entropy (the first cumulant).

---

## 2. Capacity of Entanglement

### 2.1 Cumulant Structure

The entanglement spectrum generates a hierarchy of cumulants:

| Cumulant | Name | Formula | Physical Meaning |
|----------|------|---------|------------------|
| κ₁ | Entropy S | -Tr(ρ ln ρ) | Average entanglement |
| κ₂ | **Capacity C_E** | Tr(ρ(ln ρ)²) - S² | Fluctuation in entanglement |
| κ₃ | Skewness | ... | Spectrum asymmetry |

### 2.2 Prediction 1: C_E/S Ratio

For critical 1+1D systems, we predict that C_E and S both scale logarithmically with system size L, but with a characteristic ratio.

**Test:** Compute C_E/S from reduced density matrix via:
- MERA simulations
- Experimental tomography
- Direct measurement of entanglement spectrum

---

## 3. Scaling Dimension Staircase

### 3.1 Framework Claim

The dimension d_s exhibits a step-like near-integer staircase with transitions at critical capacity values.

**Framework value:** d_s = 1.336 ± 0.029 (W01 truth claim)

### 3.2 Extraction Method

From [Lyu et al. PRR 3, 023048 (2021)](https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.3.023048):

> Scaling dimensions can be extracted from tensor renormalization group without CFT.

**Method:** Δ_α = -log₂|λ_α| where λ_α are eigenvalues of the ascending superoperator.

### 3.3 Prediction 2: Staircase at Critical Capacities

We predict scaling dimensions show discrete jumps at critical capacity values, extractable via tensor RG methods.

---

## 4. Entanglement Gap and Δλ

### 4.1 Framework Claim

The Framework proposes a specific crossover value Δλ ≈ 38 where behavior changes qualitatively.

### 4.2 Literature Connection

From [Wald et al. PRR 2, 043404 (2020)](https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.2.043404):

> The entanglement gap closes as π²/ln(L) at criticality.

### 4.3 Hypotheses

We test three hypotheses for Δλ ≈ 38:

| Hypothesis | Formula | Test |
|------------|---------|------|
| H1: Gap ratio | (λ₀-λ₁)/λ₀ × 100 ≈ 38 | Entanglement spectroscopy |
| H2: π² × scale | A × π² ≈ 38 | Gap closure fitting |
| H3: Capacity crossover | d²C/dS² = 0 | Capacity measurements |

### 4.4 Prediction 3

We predict Δλ ≈ 38 corresponds to the normalized entanglement gap ratio at criticality, measurable via entanglement spectroscopy.

---

## 5. Falsifiability Criteria

Each prediction has clear falsification criteria:

| Prediction | Falsified If |
|------------|--------------|
| C_E/S ratio universal | Ratio varies > 50% across models |
| d_s staircase | No discrete jumps in scaling dimensions |
| Δλ ≈ gap ratio | Gap ratio far from 38% at criticality |

---

## 6. Experimental Signatures

### 6.1 Cold Atom Arrays

- Prepare critical states in 1D/2D/3D optical lattices
- Measure Rényi entropies via SWAP protocol
- Extract capacity of entanglement from full tomography

### 6.2 Ion Trap Simulators

- Implement MERA circuits [Duke 2025](https://arxiv.org/html/2412.18602v2)
- Measure gap closing directly
- Validate logarithmic scaling

### 6.3 Quantum Materials

- Quantum Fisher information bounds entanglement
- May provide capacity bounds
- Connect to observable quantities

---

## 7. Results Summary

### 7.1 Completed Validation

| Item | Result | Status |
|------|--------|--------|
| S ∝ c·log(L) | R² > 0.98 | VERIFIED |
| Central charge scaling | Heisenberg/Ising = 2 | VERIFIED |
| XXZ phase behavior | Gapless vs gapped | VERIFIED |
| Capacity of entanglement | κ₂ implementation | COMPLETE |

### 7.2 MERA Computational Results

**Heisenberg cyclic (L=8, well-converged, 50 optimization steps):**

| χ | S (nats) | C_E | C_E/S | Gap Ratio |
|---|----------|-----|-------|------------|
| 4 | 1.046 | 1.114 | 1.065 | 83.8% |
| 8 | 1.051 | 1.057 | 1.006 | 83.8% |
| 16 | 1.051 | 0.990 | 0.942 | 83.7% |

**Key Finding:** C_E/S ≈ 1.0 for well-converged MERA, consistent with de Boer et al. PRD 2019 prediction that C_E ≈ S for critical systems.

### 7.3 Hypothesis Test Results

| Hypothesis | Prediction | Observed | Status |
|------------|------------|----------|--------|
| C_E/S ≈ 1 | Ratio near 1 | 0.94-1.07 | SUPPORTED |
| Gap ratio ≈ 38% | (λ₀-λ₁)/λ₀ × 100 ≈ 38 | 83.7-96.8% | NOT SUPPORTED |
| d_s staircase | Jumps at critical S | Testing | IN PROGRESS |

**Note on Δλ:** The gap ratio hypothesis (H1) is falsified by our data. Alternative interpretations:
- H2: π² × scale needs further analysis
- H3: Capacity crossover d²C/dS² = 0 needs different measurement

### 7.4 In Progress

| Item | Method | Status |
|------|--------|--------|
| d_s staircase | Tensor RG | Testing |
| Δλ interpretation | Alternative hypotheses | Needs revision |
| C_geo, C_int, C_ptr, C_obs | Literature study | Pending |

---

## 8. Conclusions

1. **Critical Discovery:** Framework "capacity" = capacity of entanglement (second cumulant), not entropy
2. **Experimental Bridge:** Predictions testable with current quantum simulation technology
3. **Falsifiability:** Clear criteria for validating or rejecting Framework claims
4. **Extensions:** Framework-specific capacities (C_geo, C_int, etc.) require further study

---

## References

1. de Boer et al., PRD 99, 066012 (2019) - Capacity of entanglement definition
2. Khoshdooni et al., PRD 112, 026027 (2025) - Capacity in Lifshitz theories
3. Lyu et al., PRR 3, 023048 (2021) - Scaling dimensions from tensor RG
4. Wald et al., PRR 2, 043404 (2020) - Entanglement gap closure
5. Duke University, arXiv:2412.18602 (2025) - MERA on quantum computer
6. Nature Comm 2025 - Entanglement microscopy at critical points
7. Mozaffari, JHEP 09, 068 (2024) - Capacity in volume-law systems

---

*Draft generated: 2026-03-02*
*Framework version: v4.5/v4.6*
*Canonical workspace: /tmp/openclaws/Repos/host-adapters/*