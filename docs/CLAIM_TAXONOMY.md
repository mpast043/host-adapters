# Claim Taxonomy — Unified Naming Convention

**Date**: 2026-03-04
**Purpose**: Clean, logical mapping of all claims and tests

---

## Overview

The previous naming convention (P-series, W-series, Claim numbers) was confusing and inconsistent. This document establishes a clean taxonomy.

---

## Taxonomy Structure

```
CAPACITY_FRAMEWORK/
├── FOUNDATION/          # Mathematical foundations (W-series)
│   ├── F01-F20         # Renamed from W01-W20
│
├── PHYSICS/             # Physical predictions (P-series)
│   ├── P1_SPECTRAL     # Spectral dimension
│   ├── P2_CAPACITY     # Capacity plateau
│   ├── P3_GLUING       # Gluing/excision (INVALID)
│   └── P4_MERA         # MERA convergence
│
└── DERIVED/             # Derived/secondary claims
    ├── D1_HOLOGRAPHIC  # Holographic bound (was Claim 3-v3)
    ├── D2_ALLOCATOR    # MERA allocator (was Claim 2)
    ├── D3_SPECTRAL      # Spectral bridge (was Claim 3)
    ├── D4_PHYSICAL      # Physical convergence (was Claim 3P)
    └── D5_WINDOWED      # Windowed regime (was Claim 3B)
```

---

## Clean Mapping

### Foundation Claims (Mathematical)

| Old Name | New Name | Statement | Status |
|----------|----------|-----------|--------|
| W01 | F01_ORDER | Componentwise order/filter properties hold | ✅ SUPPORTED |
| W02 | F02_INFIMUM | Componentwise infimum yields well-defined shared capacity | ✅ SUPPORTED |
| W03 | F03_EXCISION | Memory excision consistent with general excision | ✅ SUPPORTED |
| W04 | F04_SELF_REF | Framework can represent itself without contradiction | ✅ SUPPORTED |
| W05 | F05_MIXED | Mixed-regime coherence is non-contradictory | ✅ SUPPORTED |
| W06 | F06_DEPTH | Depth-vector projection D(n) is monotone | ✅ SUPPORTED |
| W07 | F07_ISOLATION | Cross-axis isolation preserves geometric gluing | ✅ SUPPORTED |
| W08 | F08_MONOTONE | N(C) is monotone non-decreasing | ✅ SUPPORTED |
| W09 | F09_DELTAT | Delta-T(C) is well-defined semiclassically | ✅ SUPPORTED |
| W10 | F10_OBSERVERS | Observers do not generate classical structure | ✅ SUPPORTED |
| W11 | F11_CPMT | CPMT operationalizes Annex T correctly | ✅ SUPPORTED |
| W12 | F12_TRIAD | Observer triad maps to substrate triad | ✅ SUPPORTED |
| W13 | F13_DECOMP | C_obs decomposition is backward compatible | ✅ SUPPORTED |
| W14 | F14_EJECTION | Observer ejection expands classical core | ✅ SUPPORTED |
| W15 | F15_ORTHOGONAL | Pointer-accuracy orthogonality consistent | ✅ SUPPORTED |
| W16 | F16_TIME | Distinguishing t_E and t_B resolves circularity | ✅ SUPPORTED |
| W17 | F17_FIXEDPT | Local fixed points compatible with global | ✅ SUPPORTED |
| W18 | F18_COMPRESS | Compression governance integrates with T.7 | ✅ SUPPORTED |
| W19 | F19_META | C_obs^meta limitation acknowledged as scope boundary | ✅ SUPPORTED |
| W20 | F20_NONNEG | Non-negotiability principle self-applied | ✅ SUPPORTED |

**Summary**: All 20 foundation claims SUPPORTED. These are mathematical properties of the capacity framework that have been verified through documentation and regression tests.

---

### Physics Tests (Predictions)

| Old Name | New Name | What It Tests | Status |
|----------|----------|---------------|--------|
| P1 | P1_SPECTRAL | Spectral dimension matches Sierpinski | ✅ SUPPORTED (d_s=1.365) |
| P2 | P2_CAPACITY | Entanglement capacity saturates for gapped systems | ✅ SCOPE_CORRECT |
| P2κ₂ | P2_CAPACITY_K2 | Capacity-of-entanglement (variance) saturates | ✅ SCOPE_CORRECT |
| P3 | P3_GLUING | Gluing/excision stability | 🔴 INVALID_METHODOLOGY |
| P4 | P4_MERA | MERA physical convergence | ✅ SCOPE_CORRECT |

**Summary**: 4/5 physics tests complete. P3 invalid due to naive tensor product flaw.

---

### Derived Claims (Secondary)

| Old Name | New Name | What It Tests | Status |
|----------|----------|---------------|--------|
| Claim 1 | (merged into P1_SPECTRAL) | Spectral dimension | ✅ SUPPORTED |
| Claim 2 | D2_ALLOCATOR | MERA as optimal capacity allocator | ✅ SUPPORTED |
| Claim 3 | D3_SPECTRAL | MERA spectral dimension bridge | ❌ NOT_SUPPORTED |
| Claim 3-v3 | D1_HOLOGRAPHIC | Entanglement entropy holographic bound | ✅ SUPPORTED |
| Claim 3P | D4_PHYSICAL | Physical Hamiltonian convergence | ❌ SCOPE_CORRECT |
| Claim 3B | D5_WINDOWED | Windowed regime transition | 🟡 NO_EVIDENCE |

**Summary**: 
- D2, D1: SUPPORTED
- D3: NOT_SUPPORTED (replaced by D1)
- D4: REJECTED (correctly identifies Heisenberg out of scope)
- D5: NO_EVIDENCE (needs investigation)

---

## Status Summary

| Category | Total | Supported | Scope-Correct | Not Supported | No Evidence | Invalid |
|----------|-------|-----------|--------------|--------------|-------------|---------|
| **Foundation (F)** | 20 | 20 | - | 0 | 0 | 0 |
| **Physics (P)** | 5 | 1 | 3 | 0 | 0 | 1 |
| **Derived (D)** | 5 | 2 | 1 | 1 | 1 | 0 |
| **TOTAL** | 30 | 23 | 4 | 1 | 1 | 1 |

---

## Reconciliation Actions

| Action | File Change | Status |
|--------|--------------|--------|
| Rename W01-W20 → F01-F20 | Update claim_map JSON | Pending |
| Merge Claim 1 → P1_SPECTRAL | Already consistent | Done |
| Rename Claim 2 → D2_ALLOCATOR | Update references | Pending |
| Rename Claim 3-v3 → D1_HOLOGRAPHIC | Update references | Pending |
| Rename Claim 3P → D4_PHYSICAL | Update references | Pending |
| Rename Claim 3B → D5_WINDOWED | Update references | Pending |
| Archive D3_SPECTRAL (NOT_SUPPORTED) | Mark deprecated | Pending |

---

## Green Light Status (Clean Names)

| Test | Status | Verdict |
|------|--------|---------|
| P1_SPECTRAL | ✅ | SUPPORTED |
| P2_CAPACITY | ✅ | SCOPE_CORRECT (Ising in, Heisenberg out) |
| P2_CAPACITY_K2 | ✅ | SCOPE_CORRECT (confirms P2) |
| P3_GLUING | 🔴 | INVALID_METHODOLOGY |
| P4_MERA | ✅ | SCOPE_CORRECT |
| D1_HOLOGRAPHIC | ✅ | SUPPORTED |
| D2_ALLOCATOR | ✅ | SUPPORTED |
| D4_PHYSICAL | ✅ | SCOPE_CORRECT |
| D5_WINDOWED | 🟡 | NO_EVIDENCE |
| F01-F20 | ✅ | ALL_SUPPORTED |

**Blockers:**
- P3_GLUING (invalid methodology)
- D5_WINDOWED (no evidence)

---

## Simple Reference

### What's Supported?
- All foundation claims (F01-F20)
- P1_SPECTRAL
- P2_CAPACITY, P2_CAPACITY_K2, P4_MERA (for gapped systems)
- D1_HOLOGRAPHIC, D2_ALLOCATOR

### What's Out of Scope?
- P2_CAPACITY, P4_MERA for critical systems (Heisenberg) — correctly identified

### What Needs Work?
- P3_GLUING — redesign with isometric gluing
- D5_WINDOWED — investigate and implement test

### What's Deprecated?
- D3_SPECTRAL — replaced by D1_HOLOGRAPHIC

---

## Implementation

To use this taxonomy:

```python
# Old (confusing)
claim_id = "W14"  # What is this?
test_id = "P2"     # vs Claim 2?
claim_id = "Claim 3P"  # vs P3? vs W03?

# New (clear)
claim_id = "F14_EJECTION"  # Foundation claim about observer ejection
test_id = "P2_CAPACITY"    # Physics test about capacity plateau
claim_id = "D4_PHYSICAL"   # Derived claim about physical convergence
```

---

## Commit Summary

| Commit | Change |
|--------|--------|
| Pending | Create CLAIM_TAXONOMY.md |
| Pending | Update claim_map JSON with new names |
| Pending | Update PHYSICS_BASELINE_STATUS_v2.json |
| Pending | Update all references in runners |