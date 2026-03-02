# Physics Validation Roadmap

**Goal:** Map every Framework claim to physically testable predictions and validate against real systems.

**Status:** Planning - defines scope for future implementation

**Last Updated:** 2026-03-02 (after literature review)

---

## Critical Discovery from Literature Review

### Capacity of Entanglement = Established Physics Concept

From [de Boer, Järvelä, Keski-Vakkuri (PRD 99, 066012, 2019)](https://journals.aps.org/prd/pdf/10.1103/PhysRevD.99.066012):

> **Capacity of entanglement** is defined as the second cumulant (variance) of the entanglement spectrum/modular Hamiltonian.

This is **NOT generic entanglement entropy**. The Framework's "capacity" may map to this established concept.

**Key Properties from Literature:**
- Capacity of entanglement tracks entanglement entropy in CFTs
- It measures fluctuations in the modular Hamiltonian
- It behaves monotonically under RG flow in relativistic theories

### Other Critical References

| Paper | Key Finding | Framework Connection |
|-------|-------------|---------------------|
| [Nature 2026: Measuring central charge](https://www.nature.com/articles/s41467-025-66775-9) | First experimental c measurement (5% error) | Validate our c=1, c=1/2 values |
| [PRR 2020: Entanglement gap closure](https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.2.043404) | Gap closes as π²/ln(L) at criticality | **Key for Δλ ≈ 38** |
| [PRR 2021: Scaling dimensions from tensor RG](https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.3.023048) | Extracts d_s without CFT | **Method for d_s staircase** |
| [Quantum 2025: MERA quantum advantage](https://quantum-journal.org/papers/q-2025-02-11-1631/) | Polynomial advantage over classical | Validates MERA approach |
| [arXiv 2024: Capacity and volume law](https://arxiv.org/abs/2407.16028) | Capacity in volume-law systems | Extension beyond area-law |

---

## Revised Framework Capacity Mapping

### Previous (Incorrect) Hypothesis

| Framework | Physical | Status |
|-----------|----------|--------|
| C | Entanglement entropy S | INCOMPLETE |

### Revised Hypothesis (Based on Literature)

| Framework Symbol | Literature Concept | Definition | Status |
|------------------|---------------------|------------|--------|
| C (generic) | **Capacity of entanglement** | Var(ρ_A) = ⟨H_A²⟩ - ⟨H_A⟩² | NEW MAPPING |
| C_geo | Geometric capacity? | TBD - may be different cumulant | NEEDS STUDY |
| C_int | Intrinsic capacity? | TBD - may be different cumulant | NEEDS STUDY |
| C_ptr | Pointer capacity? | TBD - operational definition | NEEDS STUDY |
| C_obs | Observable capacity? | TBD - measurable limit | NEEDS STUDY |

**Critical insight:** The Framework's "capacity" is likely the **second cumulant of the entanglement spectrum**, NOT the first (von Neumann entropy).

### Cumulant Structure of Entanglement

From statistical mechanics, the entanglement spectrum generates cumulants:

| Cumulant | Name | Physical Meaning |
|----------|------|------------------|
| κ₁ | Entropy S = ⟨H_A⟩ | Average entanglement |
| κ₂ | **Capacity** C = Var(H_A) | Fluctuation in entanglement |
| κ₃ | Skewness | Asymmetry of spectrum |
| κ₄ | Kurtosis | Peakedness of spectrum |

**Hypothesis:** Framework capacities C_geo, C_int, etc. may be different cumulants or geometric variants.

---

## Framework Claims Requiring Physical Validation

### Tier A: Algebraic Claims (Mathematical)

These are mathematically proven and require no physical validation:
- Factorisation theorem
- Capacity staircase structure
- Eigenvalue bounds

### Tier B: Numerical Claims (Simulation-Validated)

| Claim | Description | Status | Physical Connection Needed |
|-------|-------------|--------|----------------------------|
| W02 | Poset infimum | PASS | Connect d_s to scaling dimensions |
| W03 | Memory excision | PASS | Relate to quantum operations |
| W04-W20 | Observer structure | PASS | Map to physical observables |

### Tier C: Physics Claims (Require Real System Validation)

| Claim | Description | Literature Basis | Status |
|-------|-------------|------------------|--------|
| C ∝ S | Capacity-entanglement relation | [de Boer PRD 2019](https://journals.aps.org/prd/pdf/10.1103/PhysRevD.99.066012) | REVISE to C = Var(H) |
| d_s staircase | Dimension jumps at critical values | [Lyu PRR 2021](https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.3.023048) | USE tensor RG method |
| Δλ ≈ 38 | Crossover critical value | [Wald PRR 2020](https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.2.043404) | Gap closure at criticality |

---

## d_s Staircase Testing

### Framework Claim

The dimension d_s exhibits a **step-like near-integer staircase** with transitions at critical capacity values.

### Literature Connection

From [Lyu et al. PRR 2021](https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.3.023048):

> Scaling dimensions can be extracted from **linearized tensor renormalization group transformations** without using conformal field theory.

This provides a **direct method** to obtain d_s from MERA!

### Test Methodology

```python
def test_ds_staircase():
    """
    Use tensor RG method from Lyu et al. to extract scaling dimensions.

    Expected: d_s shows staircase structure at critical entanglement values.
    """
    for model in ["heisenberg", "ising", "xxz"]:
        # 1. Run MERA to obtain tensor network
        mera = run_mera(L=16, chi=32, model=model)

        # 2. Linearize RG transformation
        T = get_transfer_matrix(mera)
        eigenvalues = np.linalg.eigvals(T)

        # 3. Extract scaling dimensions
        # From Lyu et al.: d_s ∝ -ln(λ_i / λ_0)
        d_s = extract_scaling_dimensions(eigenvalues)

        # 4. Check for staircase structure
        # Expected: d_s jumps at specific entanglement values
```

### Success Criteria

| Criterion | Prediction | Test Method |
|-----------|------------|--------------|
| d_s extracted from tensor RG | Real, positive values | [Lyu PRR 2021](https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.3.023048) |
| Staircase at critical S | Jumps at S = log(2^n) | Compare to entanglement entropy |
| Universal across models | Same d_s for same universality class | Compare Heisenberg (c=1) vs Ising (c=1/2) |

---

## Δλ ≈ 38 Crossover Testing

### Framework Claim

There is a **specific crossover value Δλ ≈ 38** where behavior changes qualitatively.

### Literature Connection

From [Wald et al. PRR 2020](https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.2.043404):

> The entanglement (Schmidt) gap **closes logarithmically** at quantum critical points: **δξ ∝ π²/ln(L)**.

**Key insight:** The gap closure behavior at criticality may relate to Δλ.

### Possible Interpretations

| Hypothesis | Formula | Literature Basis |
|------------|---------|------------------|
| Δλ = gap ratio | (λ₀ - λ₁)/λ₀ | Entanglement spectrum ratio |
| Δλ ∝ π² | Δλ ≈ π² × k | From gap closure formula |
| Δλ = crossover in capacity | d²C/dS² = 0 | Second derivative of capacity |

### Test Methodology

```python
def test_delta_lambda():
    """
    Test if Δλ ≈ 38 relates to entanglement gap closure.

    From Wald et al.: gap closes as π²/ln(L) at criticality.
    """
    for model in ["heisenberg", "ising", "xxz"]:
        for L in [2, 4, 8, 16, 32]:
            # 1. Compute entanglement spectrum
            spectrum = compute_entanglement_spectrum(L, model)

            # 2. Calculate gap and gap ratios
            gap = spectrum[0] - spectrum[1]
            gap_ratio = gap / spectrum[0]

            # 3. Check for values near 38 or 1/38
            # 4. Check if π² × scale ≈ 38
            pi_squared_approx = np.pi**2 * spectrum[0]

            # 5. Check crossover in capacity vs S
            # Capacity of entanglement = Var(H_A) = κ₂
            capacity = compute_capacity_of_entanglement(L, model)
```

### Success Criteria

| Hypothesis | Prediction | Status |
|------------|------------|--------|
| Δλ = gap ratio × 100 | gap_ratio × 100 ≈ 38 | NOT TESTED |
| Δλ = π² × scale | π² × scale ≈ 38 | NOT TESTED |
| Δλ = capacity crossover | d²C/dS² zero crossing | NOT TESTED |

---

## Capacity of Entanglement Testing

### Literature Definition

From [de Boer et al. PRD 2019](https://journals.aps.org/prd/pdf/10.1103/PhysRevD.99.066012):

```
Capacity of Entanglement: C = Var(H_A) = ⟨H_A²⟩ - ⟨H_A⟩²
                        = Σ λᵢ (ln λᵢ)² - (Σ λᵢ ln λᵢ)²
                        = d²(S)/dβ² at β=1
```

where λᵢ are the eigenvalues of the reduced density matrix.

### Relationship to Entropy

| Quantity | Formula | First Cumulant | Second Cumulant |
|----------|---------|----------------|-----------------|
| Entropy | S = -Σ λᵢ ln λᵢ | ⟨H_A⟩ | κ₁ |
| Capacity | C = Var(H_A) | ⟨H_A²⟩ - ⟨H_A⟩² | κ₂ |
| Skewness | γ | | κ₃ |

### Test Methodology

```python
def compute_capacity_of_entanglement(rho):
    """
    Compute capacity of entanglement (second cumulant of spectrum).

    C = Σ λᵢ (ln λᵢ)² - (Σ λᵢ ln λᵢ)²
    """
    eigenvalues = np.linalg.eigvalsh(rho)
    eigenvalues = eigenvalues[eigenvalues > 1e-12]

    # Entropy (first cumulant)
    S = -np.sum(eigenvalues * np.log(eigenvalues))

    # Capacity (second cumulant)
    log_lam = np.log(eigenvalues)
    C = np.sum(eigenvalues * log_lam**2) - S**2

    return S, C

def test_capacity_staircase():
    """
    Test if capacity of entanglement shows staircase structure.
    """
    for model in ["heisenberg", "ising", "xxz"]:
        for L in [2, 4, 8, 16]:
            for chi in [4, 8, 16, 32]:
                mera = run_mera(L, chi, model)
                rho = compute_reduced_density_matrix(mera)
                S, C = compute_capacity_of_entanglement(rho)

                # Check for staircase in C vs S or C vs chi
```

---

## Truth Infrastructure Claims Requiring Physics Grounding

### W01: d_s = 1.336 ± 0.029

**Current Status:** Measured from MERA, not connected to physics.

**Literature Connection:** Use [Lyu et al. PRR 2021](https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.3.023048) method to extract scaling dimensions.

**Test:**
```python
# Compare d_s from MERA to d_s from tensor RG
d_s_mera = 1.336
d_s_rg = extract_scaling_dimensions_from_tensor_RG(mera)
# Should match within uncertainty
```

### W02: Heisenberg Leverage

**Current Status:** PASS, but not physics-grounded.

**Literature Connection:** Compare to [Nature 2026 experimental c values](https://www.nature.com/articles/s41467-025-66775-9).

**Test:**
- Our Heisenberg c=1
- Nature 2026: c=1 with 5% error
- Should match within experimental uncertainty

### W03: Memory Excision Controls

**Current Status:** PASS (18/18 controls).

**Physical Interpretation Needed:**

| Control | Current | Physical Meaning |
|---------|---------|------------------|
| Positive (gluing) | S_glued ≈ S_sum | Entanglement of assembly |
| Negative | S_glued ≠ S_sum | Non-additive entanglement |

**Literature:** Need to find physical interpretation of assembly entanglement.

---

## Implementation Tasks (Revised)

### Task 1: Study Capacity of Entanglement Literature

**Goal:** Understand established physics concept

**Steps:**
1. Read [de Boer PRD 2019](https://journals.aps.org/prd/pdf/10.1103/PhysRevD.99.066012) in detail
2. Extract formulas for all cumulants
3. Identify if Framework uses κ₁ (entropy) or κ₂ (capacity)
4. Document in `docs/physics/CAPACITY_OF_ENTANGLEMENT.md`

**Output:** Understanding of what "capacity" means in physics

### Task 2: Implement Capacity of Entanglement Calculation

**Goal:** Compute C = Var(H_A) from MERA

**Steps:**
1. Add to `entanglement_utils.py`:
   ```python
   def capacity_of_entanglement(rho):
       """Second cumulant of entanglement spectrum."""
       eigenvalues = np.linalg.eigvalsh(rho)
       eigenvalues = eigenvalues[eigenvalues > 1e-12]
       S = -np.sum(eigenvalues * np.log(eigenvalues))
       log_lam = np.log(eigenvalues)
       C = np.sum(eigenvalues * log_lam**2) - S**2
       return C
   ```
2. Test on known systems (maximally mixed, Bell state)
3. Run on all models

**Output:** Capacity values for Heisenberg, Ising, XXZ

### Task 3: Extract Scaling Dimensions via Tensor RG

**Goal:** Use [Lyu et al. PRR 2021](https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.3.023048) method

**Steps:**
1. Implement linearized tensor RG
2. Extract scaling dimensions from MERA transfer matrix
3. Compare to W01 value (d_s = 1.336)
4. Check for staircase structure

**Output:** `docs/physics/SCALING_DIMENSIONS.md`

### Task 4: Test Δλ Gap Closure Hypothesis

**Goal:** Connect Δλ to entanglement gap at criticality

**Steps:**
1. Compute entanglement gap across system sizes
2. Check if gap/λ₀ ≈ 38 or π² × scale ≈ 38
3. Study gap closure behavior at criticality
4. Compare to [Wald PRR 2020](https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.2.043404)

**Output:** `outputs/delta_lambda_analysis/`

### Task 5: Compare to Experimental Data

**Goal:** Validate against [Nature 2026](https://www.nature.com/articles/s41467-025-66775-9) measurements

**Steps:**
1. Extract their c values for Ising and XXZ
2. Compare to our computed c values
3. Check if within 5% error margin
4. Document discrepancies

**Output:** Comparison table in validation summary

### Task 6: Document Framework-Physics Mapping

**Goal:** Create definitive mapping document

**Steps:**
1. Map Framework C_geo, C_int, C_ptr, C_obs to cumulants or variants
2. Map d_s to scaling dimensions
3. Map Δλ to entanglement gap ratio
4. Document any mismatches

**Output:** `docs/physics/FRAMEWORK_PHYSICS_MAPPING.md`

### Task 7: Draft Predictions Paper

**Goal:** Write paper with all connections

**Steps:**
1. Compile all computational results
2. Write derivation of capacity-entanglement connection
3. Document testable predictions with literature references
4. Include falsifiability criteria
5. Submit to appropriate journal

**Output:** `docs/physics/PREDICTIONS_PAPER.md`

---

## Literature Check Requirement

**This must be done BEFORE any implementation:**

```markdown
## Literature Check (REQUIRED)

Before implementing any physics validation:

1. **Search arXiv, PR journals** for key terms
2. **Identify established concepts** that match Framework terminology
3. **Find experimental papers** with comparable data
4. **Check for existing methods** to extract quantities
5. **Document connections** in mapping file

Current gaps addressed:
- ✅ Capacity of entanglement (de Boer PRD 2019)
- ✅ Scaling dimensions from tensor RG (Lyu PRR 2021)
- ✅ Entanglement gap closure (Wald PRR 2020)
- ✅ Experimental c measurement (Nature 2026)
- ✅ MERA quantum advantage (Quantum 2025)
```

---

## Test Matrix (Revised)

### Models to Test

| Model | c | Phase | Status |
|-------|---|-------|--------|
| Heisenberg cyclic | 1 | Gapless | S tested, C not tested |
| Ising cyclic | 1/2 | Critical | S tested, C not tested |
| XXZ Δ=0 | 1 | Gapless XY | S tested, C not tested |
| XXZ Δ=2 | - | Gapped | S tested, C not tested |

### Measures to Extract

| Measure | Formula | From | Status |
|---------|---------|------|--------|
| Entropy S | κ₁ = -Σλᵢlnλᵢ | MERA | ✅ DONE |
| **Capacity C** | κ₂ = Var(H_A) | MERA | ❌ TODO |
| Scaling dims d_s | From tensor RG | MERA | ❌ TODO |
| Entanglement gap | λ₀ - λ₁ | MERA | ✅ DONE |
| Gap ratio | (λ₀-λ₁)/λ₀ | MERA | ❌ TODO |

---

## Timeline (Revised)

| Week | Task | Deliverable |
|------|------|-------------|
| 1 | Study capacity of entanglement literature | Understanding |
| 2 | Implement C calculation | Code + results |
| 3 | Extract scaling dimensions | d_s values |
| 4 | Test Δλ hypothesis | Gap analysis |
| 5 | Compare to experimental data | Validation |
| 6 | Document mapping | Mapping file |
| 7 | Draft predictions paper | Paper draft |
| **Total** | **7 weeks** | Full physics grounding |

---

## Success Criteria Summary

| Criterion | Target | Current Status |
|-----------|--------|----------------|
| Capacity of entanglement C | Extract from all models | ❌ NOT IMPLEMENTED |
| C vs S correlation | R² > 0.95 | ❌ NOT TESTED (was testing S only) |
| d_s from tensor RG | Match W01 value | ❌ NOT TESTED |
| Δλ = gap ratio | ≈ 38 | ❌ NOT TESTED |
| Match experimental c | Within 5% | ❌ NOT COMPARED |
| Universal across models | All models agree | ✅ Gapless models agree for S |

---

## Key References

1. [de Boer et al., PRD 99, 066012 (2019)](https://journals.aps.org/prd/pdf/10.1103/PhysRevD.99.066012) - Capacity of entanglement
2. [Lyu et al., PRR 3, 023048 (2021)](https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.3.023048) - Scaling dimensions from tensor RG
3. [Wald et al., PRR 2, 043404 (2020)](https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.2.043404) - Entanglement gap closure
4. [Köylüoğlu et al., Nature 2026](https://www.nature.com/articles/s41467-025-66775-9) - Experimental central charge
5. [Quantum 2025](https://quantum-journal.org/papers/q-2025-02-11-1631/) - MERA quantum advantage

---

*Generated: 2026-03-02*
*Revised: 2026-03-02 (after literature review)*
*Status: PLANNING - Literature check complete, implementation pending*
*Workspace: /tmp/openclaws/Repos/host-adapters/ (CANONICAL)*