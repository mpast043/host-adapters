# Framework v4.5/v4.6 to Physics Mapping

**Status:** Canonical Reference
**Last Updated:** 2026-03-02
**Framework Document:** Framework with selection.pdf

---

## Executive Summary

This document provides the definitive mapping between Framework concepts and established physics. The critical discovery is that **Framework "capacity" maps to capacity of entanglement (second cumulant kappa_2)**, NOT von Neumann entropy (first cumulant kappa_1).

---

## Capacity Measures

### Primary Mapping

| Framework Symbol | Physics Concept | Definition | Status | Reference |
|------------------|-----------------|------------|--------|-----------|
| C (generic) | Capacity of entanglement C_E | Var(H_A) = <H_A^2> - <H_A>^2 | **MAPPED** | [de Boer PRD 2019](https://journals.aps.org/prd/pdf/10.1103/PhysRevD.99.066012) |

### Framework-Specific Capacity Measures

| Framework Symbol | Proposed Mapping | Status | Notes |
|------------------|------------------|--------|-------|
| C_geo | Geometric capacity | NEEDS STUDY | May relate to holographic dual |
| C_int | Intrinsic capacity | NEEDS STUDY | May be different cumulant |
| C_ptr | Pointer capacity | NEEDS STUDY | May relate to measurement |
| C_obs | Observable capacity | NEEDS STUDY | May relate to accessible entanglement |

### Cumulant Structure

From statistical mechanics and QFT:

| Cumulant | Name | Formula | Framework Relevance |
|----------|------|---------|---------------------|
| kappa_1 | Entropy S | -Tr(rho ln rho) | Previously misidentified as "capacity" |
| kappa_2 | **Capacity C_E** | Tr(rho(ln rho)^2) - S^2 | **KEY MAPPING** |
| kappa_3 | Skewness | <H^3> - 3<H><H^2> + 2<H>^3 | May relate to C_int |
| kappa_4 | Kurtosis | ... | Higher-order structure |

---

## Dimension Measures

| Framework Symbol | Physics Concept | Definition | Status | Reference |
|------------------|-----------------|------------|--------|-----------|
| d_s | Spectral dimension | From return probability P(tau) ~ tau^(-d_s/2) | MAPPED | Standard QFT |
| d_s | Scaling dimension | From tensor RG eigenvalues | TESTED | [Lyu PRR 2021](https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.3.023048) |
| d_s staircase | RG flow transitions | Jumps at critical capacity | TESTING | Framework claim |

### Known CFT Scaling Dimensions

| Model | c | Primary Fields | d_s Values |
|-------|---|----------------|------------|
| Ising | 1/2 | 1, sigma, epsilon | 0, 0.125, 1.0 |
| Heisenberg | 1 | SU(2)_1 | 0, 0.5, 1.0, 1.5... |
| XXZ Delta<=1 | 1 | Gapless | n/2 for n in N |
| XXZ Delta>1 | - | Gapped | Non-universal |

---

## Critical Values

| Framework Value | Physics Interpretation | Status | Notes |
|-----------------|------------------------|--------|-------|
| Delta_lambda ~ 38 | Entanglement gap ratio? | TESTING | See DELTA_LAMBDA_ANALYSIS.md |
| S_c = log(2^n) | Critical entanglement | VERIFIED | From CFT |
| c=1, c=1/2 | Central charge | VERIFIED | Heisenberg, Ising |
| d_s = 1.336 | Framework W01 value | TESTING | May be effective dimension |

---

## Claims Status Summary

### Tier A: Algebraic (Mathematical)

| Claim | Physics Grounded? | Notes |
|-------|-------------------|-------|
| Factorisation theorem | N/A | Pure math |
| Capacity staircase structure | Partial | See d_s staircase |
| Eigenvalue bounds | N/A | Pure math |

### Tier B: Observer Structure (W02-W20)

| Claim | Physics Grounded? | Notes |
|-------|-------------------|-------|
| W01: d_s = 1.336 | Partial | Needs tensor RG validation |
| W02-W20 | Math only | Not physics claims |

### Tier C: Higher-Stakes

| Claim | Physics Grounded? | Notes |
|-------|-------------------|-------|
| C propto S | **REVISED** | C = capacity of entanglement (kappa_2) |
| d_s staircase | Testing | Tensor RG method available |
| Delta_lambda ~ 38 | Testing | Gap analysis implemented |

---

## Experimental Connections

### Recent Experimental Results

| Experiment | Finding | Framework Connection |
|------------|---------|---------------------|
| [Nature 2026](https://www.nature.com/articles/s41467-025-66775-9) | First c measurement (5% error) | Validate our c values |
| [Duke 2025](https://arxiv.org/html/2412.18602v2) | MERA on quantum computer | Gap closing observed |
| [Nature Comm 2025](https://www.nature.com/articles/s41467-024-55354-z) | Entanglement microscopy | Critical point structure |

### Testable Predictions

| Prediction | How to Test | Expected Result |
|------------|-------------|-----------------|
| C_E/S ratio ~ constant | Measure C_E and S | Model-dependent |
| Gap ratio ~ 38% | Entanglement spectroscopy | At criticality |
| d_s staircase | Scaling dimension extraction | At capacity transitions |

---

## Outstanding Questions

1. **What is C_geo, C_int, C_ptr, C_obs in physics terms?**
   - May be different cumulants of entanglement spectrum
   - May relate to geometric/observable restrictions
   - Requires further study

2. **Is d_s = 1.336 a single scaling dimension or effective average?**
   - Could be weighted average over operators
   - Could be specific to the measurement protocol
   - Tensor RG extraction will clarify

3. **What physical quantity does Delta_lambda represent?**
   - Three hypotheses under test
   - Gap ratio percentage most promising
   - Real MERA data needed

---

## Implementation Status

| Item | File | Status |
|------|------|--------|
| Capacity of entanglement | `entanglement_utils.py` | Implemented |
| Capacity runner | `entanglement_capacity_runner_real.py` | Implemented |
| Scaling dimensions | `scaling_dimensions_runner.py` | Implemented |
| Gap analysis | `entanglement_gap_analysis.py` | Implemented |
| Literature review | `CAPACITY_OF_ENTANGLEMENT_LITERATURE.md` | Complete |
| Delta_lambda analysis | `DELTA_LAMBDA_ANALYSIS.md` | Complete |

---

## Key References

1. [de Boer et al., PRD 99, 066012 (2019)](https://journals.aps.org/prd/pdf/10.1103/PhysRevD.99.066012) - Capacity of entanglement definition
2. [Khoshdooni et al., PRD 112, 026027 (2025)](https://journals.aps.org/prd/abstract/10.1103/7cg6-m7dn) - Capacity in Lifshitz theories
3. [Lyu et al., PRR 3, 023048 (2021)](https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.3.023048) - Scaling dimensions from tensor RG
4. [Wald et al., PRR 2, 043404 (2020)](https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.2.043404) - Entanglement gap closure
5. [Mozaffari, JHEP 09, 068 (2024)](https://arxiv.org/abs/2407.16028) - Capacity in volume-law systems

---

*Generated: 2026-03-02*
*Workspace: /tmp/openclaws/Repos/host-adapters/ (CANONICAL)*
*Framework: v4.5/v4.6 with selection.pdf*