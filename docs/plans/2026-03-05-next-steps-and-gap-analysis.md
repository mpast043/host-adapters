# Next Steps Plan — Physics Validation Phase 2

**Date**: 2026-03-05
**Status**: PROPOSED
**Based on**: Complete review of host-adapters repository state

---

## Current State Assessment

### What's Done

| Category | Count | Status |
|----------|-------|--------|
| Foundation claims (F01-F20) | 20/20 | ALL SUPPORTED |
| Physics tests completed | 4/5 | P1 SUPPORTED, P2/P2κ₂/P4 SCOPE_CORRECT |
| Derived claims resolved | 4/5 | D1/D2 SUPPORTED, D3 deprecated, D4 SCOPE_CORRECT |
| Core infrastructure | Complete | entanglement_utils, runners, gap analysis |
| Claim taxonomy | Complete | F/P/D naming convention established |

### Key Findings So Far

1. **C ∝ S with slope 0.5** — Capacity scales exactly as half the entropy (R²=1.0) on Ising L=8, χ=2-64
2. **Framework scope is real physics** — Correctly distinguishes gapped (Ising, ΔAIC>0) from critical (Heisenberg, ΔAIC<0) systems
3. **κ₂ confirms κ₁ conclusions** — Both cumulants show bounded growth for gapped, log growth for critical
4. **Heisenberg κ₂ grows 45% faster than κ₁** — Slope 0.47 vs 0.33; enhanced capacity fluctuations at criticality
5. **P3 gluing test is invalid** — Naive tensor product destroys entanglement; needs MERA isometric approach

### What's Missing

| Gap | Priority | Effort | Impact |
|-----|----------|--------|--------|
| No real MERA C_E data for multiple models | HIGH | Medium | Validates C_E/S ratio universality |
| Scaling dimension extraction uses placeholders | HIGH | High | d_s staircase is untested |
| Gap analysis uses placeholder data | HIGH | Medium | Δλ ≈ 38 is untested |
| XXZ boundary test not run | MEDIUM | Low | Validates scope transition at Δ=1 |
| D5_WINDOWED has no evidence | MEDIUM | Medium | Windowed regime untested |
| host-adapters-experimental-data not integrated | LOW | Low | May contain additional run data |
| C_E/S ratio not tested against π²/6 prediction | HIGH | Low | Literature comparison missing |

---

## Proposed Next Steps

### Priority 1: Run Real Computations (Blocking for Paper)

#### 1A. Real MERA Capacity-Entropy Correlation (Est. 2-3 hours)

**Goal**: Get C_E/S data for Ising AND Heisenberg at multiple L and χ values.

**Current state**: Only have Ising L=8 with C/S = 0.5. Need:
- Heisenberg L=4,8,16 at χ=4,8,16,32
- Ising L=4,8,16 at χ=4,8,16,32
- XXZ Δ=0.5,1.5,2.0 at L=8, χ=16

**Key question**: Is C_E/S = 0.5 universal, or does it vary by model/universality class?

**Deliverable**: `outputs/capacity_entanglement/` with per-model JSON results

#### 1B. Real Entanglement Gap Computation (Est. 1-2 hours)

**Goal**: Replace placeholder gap data with actual computed values.

**Current state**: `entanglement_gap_analysis.py` uses `gap = 1/L^0.5` placeholder.

**Need**:
- Compute actual gaps from ground-state reduced density matrices
- Test gap_ratio × 100 vs 38 prediction
- Fit gap closure to π²/ln(L) for critical models

**Deliverable**: `outputs/gap_analysis/` with real data

#### 1C. XXZ Boundary Test Execution (Est. 1 hour)

**Goal**: Demonstrate scope boundary at Δ=1 in XXZ model.

**Current state**: Design document exists, implementation is placeholder.

**Need**:
- Run exact diagonalization for L=8,12 at Δ=0, 0.5, 1.0, 1.1, 1.5, 2.0
- Compute ΔAIC for each Δ value
- Confirm sharp transition at Δ≈1

**Deliverable**: `outputs/xxz_boundary/results.json`

### Priority 2: Complete Analysis (Required for Paper)

#### 2A. Scaling Dimension Extraction (Est. 3-4 hours)

**Goal**: Extract real scaling dimensions from MERA ascending superoperator.

**Current state**: `scaling_dimensions_runner.py` returns CFT reference values, not extracted values.

**Need**:
- Implement ascending superoperator eigenvalue extraction from MERA
- Compare extracted dimensions to known CFT values (Ising: σ=0.125, ε=1.0; Heisenberg: 0, 0.5, 1.0)
- Test staircase structure hypothesis

**Blocker**: Requires full MERA tensor network with accessible internal structure (quimb).

**Deliverable**: `outputs/scaling_dimensions/` with extracted vs known comparison

#### 2B. C_E/S Ratio Literature Comparison (Est. 30 min)

**Goal**: Compare observed C_E/S ratio to de Boer PRD 2019 predictions.

**Current finding**: C_E/S = 0.5 on Ising.
**Literature prediction**: For 1+1D CFT, capacity and entropy have related scaling but the ratio depends on Rényi index and geometry.

**Need**: Careful comparison — is 0.5 consistent with theory, or anomalous?

### Priority 3: Documentation and Paper (Can Start in Parallel)

#### 3A. FRAMEWORK_PHYSICS_MAPPING.md (Est. 1 hour)

**Goal**: Create the canonical symbol-to-physics mapping document referenced in CLAUDE.md but not yet written.

#### 3B. PREDICTIONS_PAPER.md (Est. 2-3 hours)

**Goal**: Full draft paper with:
- Abstract, Introduction, Methods, Results, Discussion, Conclusions
- All numerical results tabulated
- Falsifiability criteria
- Experimental signatures for laboratory verification

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| C_E/S = 0.5 is artifact of simple model | Medium | High | Test on multiple models and system sizes |
| Scaling dimension extraction fails with quimb | Medium | High | Fall back to known CFT comparison |
| Δλ ≈ 38 has no support | High | Medium | Document as NOT SUPPORTED; focus on what IS supported |
| P3 isometric gluing too complex | High | Low | Already low priority; scope identification (P2) suffices |

---

## Recommended Execution Order

```
Week 1: Priority 1A + 1B + 1C (run computations)
         Priority 3B (start paper draft with existing results)
Week 2: Priority 2A (scaling dimensions)
         Priority 2B (literature comparison)
         Priority 3A (mapping document)
Week 3: Priority 3B (finalize paper with all results)
         Review and iterate
```

---

## Success Criteria for Paper Readiness

| Criterion | Current | Target |
|-----------|---------|--------|
| C_E/S ratio measured on ≥3 models | 1 model | ≥3 models |
| Gap ratio tested against 38% | Placeholder only | Real data |
| Scaling dimensions extracted | Placeholder only | At least Ising + Heisenberg |
| XXZ boundary test complete | Design only | Full results |
| Framework scope validated | Partial | Complete with XXZ transition |
| All claims categorized | Done (30 claims) | Done |
| Paper draft complete | Not started | Full draft |

---

*Generated: 2026-03-05*
