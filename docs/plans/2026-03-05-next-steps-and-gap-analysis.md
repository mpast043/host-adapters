# Next Steps Plan — Physics Validation Phase 2

**Date**: 2026-03-05
**Status**: UPDATED (integrated experimental-data repo findings)
**Based on**: Complete review of host-adapters + host-adapters-experimental-data repositories

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
| XXZ boundary gate (bulk) | Complete | Δ=0.5→REJECT, Δ=2.0→ACCEPT (SCOPE_VALIDATED) |
| XXZ boundary gate (boundary) | Complete | Δ=0.95-1.1 all REJECT (SCOPE_MISMATCH at phase boundary) |
| DMRG overlap calibration | Complete | High-accuracy data for boundary + bulk deltas, L=8→44 |
| Regression test suite | Complete | A/B/C tests all pass (solver mismatch, bulk, oscillation) |

### Key Findings So Far

1. **C ∝ S with slope 0.5** — Capacity scales exactly as half the entropy (R²=1.0) on Ising L=8, χ=2-64
2. **Framework scope is real physics** — Correctly distinguishes gapped (Ising, ΔAIC>0) from critical (Heisenberg, ΔAIC<0) systems
3. **κ₂ confirms κ₁ conclusions** — Both cumulants show bounded growth for gapped, log growth for critical
4. **Heisenberg κ₂ grows 45% faster than κ₁** — Slope 0.47 vs 0.33; enhanced capacity fluctuations at criticality
5. **P3 gluing test is invalid** — Naive tensor product destroys entanglement; needs MERA isometric approach
6. **XXZ boundary gate works for bulk deltas** — Δ=0.5 (gapless) correctly REJECT, Δ=2.0 (gapped) correctly ACCEPT, with ED+DMRG pooling (n=10 points, L=8→44)
7. **Phase boundary detection is limited** — Δ=1.05 and 1.1 show log-linear preference (ΔAICc≈+22) when expected IN_SCOPE; near-critical correlation lengths exceed accessible system sizes
8. **Solver calibration is robust** — ED-DMRG overlap at L=8,12,16 matches to ~1e-9 or better across all tested deltas
9. **Parity control (L mod 4 = 0) eliminates oscillation** — Mixed-even grids show alternation_fraction=1.0; L_mod_4_eq_0 grids show 0.22
10. **Regression suite validates gate logic** — Degraded DMRG correctly triggers SOLVER_MISMATCH → INCONCLUSIVE; oscillation correction fires when appropriate

### What's Missing (Updated)

| Gap | Priority | Effort | Impact | Status |
|-----|----------|--------|--------|--------|
| No real MERA C_E data for multiple models | HIGH | Medium | Validates C_E/S ratio universality | OPEN |
| Scaling dimension extraction uses placeholders | HIGH | High | d_s staircase is untested | OPEN |
| Gap analysis uses placeholder data | HIGH | Medium | Δλ ≈ 38 is untested | OPEN |
| ~~XXZ boundary test not run~~ | ~~MEDIUM~~ | ~~Low~~ | ~~Validates scope transition at Δ=1~~ | **DONE** (experimental-data) |
| ~~host-adapters-experimental-data not integrated~~ | ~~LOW~~ | ~~Low~~ | ~~May contain additional run data~~ | **DONE** (reviewed) |
| D5_WINDOWED has no evidence | MEDIUM | Medium | Windowed regime untested | OPEN |
| C_E/S ratio not tested against π²/6 prediction | HIGH | Low | Literature comparison missing | OPEN |
| Phase boundary scope mismatch at Δ≈1 | NEW | High | Need larger L or alternative metric for near-critical gap detection | OPEN |

### Experimental-Data Repo Integration

The `host-adapters-experimental-data` repo (commit `4494c44`) contains completed test results:

**DMRG Overlap Data** (high-accuracy, bond_dim=[64,128,192,256], tol=1e-8):
- `dmrg_overlap_high_accuracy_L4k_boundary.json` — Δ={0.95,1.0,1.05,1.1}, L=8→44
- `dmrg_overlap_high_accuracy_L4k_bulk.json` — Δ={0.5,2.0}, L=8→44
- `dmrg_overlap_high_accuracy_bulk.json` — Δ={0.5,2.0}, L=8→24 (mixed-even grid)

**XXZ Boundary ED Gate v2** (11 runs + summary):
- Boundary deltas (0.95,1.0,1.05,1.1): All REJECT, ΔAICc≈+22 → log-linear strongly preferred
- Bulk regression (Δ=0.5,2.0): SCOPE_VALIDATED (correct verdicts)
- Solver mismatch test: SCOPE_INCONCLUSIVE (correct gate behavior)
- Mixed-even oscillation test: SCOPE_VALIDATED (oscillation correction fires)

**Key result**: The scope gate correctly classifies well-separated phases (Δ=0.5 gapless, Δ=2.0 gapped) but cannot resolve the phase boundary (Δ=1.05,1.1) with accessible system sizes. This is physically expected — the XXZ gap at Δ=1+ε scales as e^{-π²/√(2ε)}, giving correlation lengths ξ≈10³ for Δ=1.05.

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

#### ~~1C. XXZ Boundary Test Execution~~ — COMPLETED

**Status**: Done in `host-adapters-experimental-data` (commit `4494c44`).

**Results**:
- Bulk deltas (Δ=0.5, 2.0): **SCOPE_VALIDATED** — correct verdicts with ED+DMRG pooling
- Boundary deltas (Δ=0.95, 1.0, 1.05, 1.1): **SCOPE_MISMATCH** — all show log-linear preference
- Solver calibration: **PASS** across all deltas (ED-DMRG overlap ~1e-9)
- Regression suite: 3/3 tests pass (solver mismatch, bulk pass, mixed-even oscillation)

**Interpretation**: The sharp phase transition at Δ=1 cannot be resolved by AICc model selection at accessible system sizes (L≤44). The XXZ gap at Δ=1+ε is exponentially small: Δ_gap ~ e^{-π²/√(2ε)}, giving ξ≈10³ for Δ=1.05. Entropy growth remains logarithmic until L >> ξ. This is a physics limitation, not a framework error.

**New action item**: Consider alternative boundary detection strategies:
- Direct gap measurement from entanglement spectrum at finite L
- Curvature analysis of S(L) series (d²S/d(lnL)² sign change)
- Finite-size crossover scaling analysis

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

#### 2C. Phase Boundary Detection Alternatives (NEW — Est. 2-3 hours)

**Goal**: Find a metric that detects the XXZ Δ=1 phase transition at accessible system sizes.

**Motivation**: AICc model selection fails at Δ=1.05,1.1 because ξ >> L_max=44.

**Candidates**:
1. Entanglement spectrum gap (λ₀-λ₁) vs Δ — may show qualitative change
2. Capacity-entropy ratio κ₂/κ₁ — experimental-data shows κ₂ data at boundary deltas
3. Finite-size scaling crossover — fit S(L,Δ) jointly across Δ values
4. Second derivative d²S/d(lnL)² — sign change at phase boundary

**Data available**: ED measurements at L=4,6,8,10 with S and κ₂ for Δ=0.95,1.0,1.05,1.1 (from `xxz_boundary_ed_gate_v2`)

### Priority 3: Documentation and Paper (Can Start in Parallel)

#### 3A. FRAMEWORK_PHYSICS_MAPPING.md (Est. 1 hour)

**Goal**: Create the canonical symbol-to-physics mapping document referenced in CLAUDE.md but not yet written.

**Note**: This document already exists in `host-adapters-experimental-data` at `docs/physics/FRAMEWORK_PHYSICS_MAPPING.md` (canonical, last updated 2026-03-02). Should be copied/synced to this repo.

#### 3B. PREDICTIONS_PAPER.md (Est. 2-3 hours)

**Goal**: Full draft paper with:
- Abstract, Introduction, Methods, Results, Discussion, Conclusions
- All numerical results tabulated
- Falsifiability criteria
- Experimental signatures for laboratory verification

**Note**: Draft already exists in `host-adapters-experimental-data` at `docs/physics/PREDICTIONS_PAPER.md`. Should be used as starting point.

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| C_E/S = 0.5 is artifact of simple model | Medium | High | Test on multiple models and system sizes |
| Scaling dimension extraction fails with quimb | Medium | High | Fall back to known CFT comparison |
| Δλ ≈ 38 has no support | High | Medium | Document as NOT SUPPORTED; focus on what IS supported |
| P3 isometric gluing too complex | High | Low | Already low priority; scope identification (P2) suffices |
| Phase boundary unresolvable at accessible L | **Confirmed** | Medium | Use alternative detection metrics (spectrum gap, κ₂/κ₁); document as finite-size limitation |
| AICc gate misclassifies near-critical systems | **Confirmed** | Medium | Gate is correct for well-separated phases; document boundary limitations explicitly |

---

## Recommended Execution Order (Updated)

```
Phase 1 (immediate):
  - 1A: Real MERA C_E data for multiple models
  - 1B: Real entanglement gap computation
  - 3A: Sync FRAMEWORK_PHYSICS_MAPPING.md from experimental-data
  - 3B: Sync PREDICTIONS_PAPER.md from experimental-data

Phase 2 (next):
  - 2A: Scaling dimension extraction
  - 2B: C_E/S literature comparison
  - 2C: Phase boundary detection alternatives (using existing κ₂ data)

Phase 3 (finalize):
  - 3B: Complete paper draft with all results
  - Review and iterate
```

Note: 1C (XXZ boundary test) is COMPLETE. Results show the gate works for bulk phases and fails at the boundary — this is itself a publishable finding about finite-size limitations of AICc-based scope classification.

---

## Success Criteria for Paper Readiness

| Criterion | Current | Target | Status |
|-----------|---------|--------|--------|
| C_E/S ratio measured on ≥3 models | 1 model | ≥3 models | OPEN |
| Gap ratio tested against 38% | Placeholder only | Real data | OPEN |
| Scaling dimensions extracted | Placeholder only | At least Ising + Heisenberg | OPEN |
| XXZ boundary test complete | **Full results (bulk + boundary)** | Full results | **DONE** |
| Framework scope validated | **Bulk VALIDATED; boundary MISMATCH** | Complete with XXZ transition | **PARTIAL** |
| All claims categorized | Done (30 claims) | Done | **DONE** |
| Paper draft complete | Draft exists in experimental-data repo | Full draft | **PARTIAL** |
| DMRG solver calibration | **PASS (all deltas, L=8-16)** | Verified | **DONE** |
| Regression test suite | **3/3 pass** | All pass | **DONE** |
| Phase boundary detection | **Identified as open problem** | Strategy chosen | OPEN |

---

## Key Data Cross-References

| Data | Location |
|------|----------|
| DMRG overlap (boundary deltas) | `experimental-data/dmrg_overlap_high_accuracy_L4k_boundary.json` |
| DMRG overlap (bulk deltas) | `experimental-data/dmrg_overlap_high_accuracy_L4k_bulk.json` |
| XXZ boundary gate results | `experimental-data/xxz_boundary_ed_gate_v2/run_ed_622c8dc8/xxz_boundary_results.json` |
| Boundary summary | `experimental-data/xxz_boundary_boundaryset_L4k_summary.json` |
| Regression evidence | `experimental-data/regression_stable_20260305/REGRESSION_EVIDENCE_20260305.json` |
| Physics mapping (canonical) | `experimental-data/docs/physics/FRAMEWORK_PHYSICS_MAPPING.md` |
| Predictions paper (draft) | `experimental-data/docs/physics/PREDICTIONS_PAPER.md` |
| Capacity-of-entanglement lit review | `experimental-data/docs/physics/CAPACITY_OF_ENTANGLEMENT_LITERATURE.md` |
| Claim 1-3 verdicts | `experimental-data/experiments/physics/exp{1,2,3}_*/` |

---

*Generated: 2026-03-05, Updated: 2026-03-05 (experimental-data integration)*
