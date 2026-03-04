# Green Light Roadmap

**Date**: 2026-03-04
**Goal**: Validated, production-ready capacity framework

---

## Current Status Summary

### Traffic Light Assessment

| Component | Status | Color | Blocker? |
|-----------|--------|-------|----------|
| P1 Spectral Dimension | SUPPORTED | 🟢 | No |
| P2 Capacity Plateau | SCOPE_CORRECT | 🟢 | No |
| P2κ₂ Capacity Variance | SCOPE_CORRECT | 🟢 | No |
| P3 Gluing Stability | INVALID_METHODOLOGY | 🔴 | **Yes** |
| P4 MERA Convergence | SCOPE_CORRECT | 🟢 | No |
| W01-W20 Truth Claims | ALL_SUPPORTED | 🟢 | No |
| Claim 2 MERA Allocator | SUPPORTED | 🟢 | No |
| Claim 3-v3 Holographic | SUPPORTED | 🟢 | No |
| Claim 3B Windowed | NO_EVIDENCE | 🟡 | Maybe |
| Scope Documentation | COMPLETE | 🟢 | No |

### Blocking Issues

| Issue | Severity | Impact | Effort |
|-------|----------|--------|--------|
| P3 Invalid Methodology | 🔴 HIGH | Only red status in test suite | Medium |
| Claim 3B No Evidence | 🟡 MEDIUM | Incomplete claim validation | Low |

---

## High-Value Steps (Ranked by Impact/Effort)

### 1. XXZ Boundary Test (HIGHEST VALUE)

**Why:** Directly validates that the scope boundary is REAL and SHARP

| Metric | Value |
|--------|-------|
| Impact | HIGH - Confirms scope is physically grounded |
| Effort | LOW - Reuse existing P2 runner with XXZ model |
| Timeline | 1-2 hours |

**Design:**
```
XXZ Model: H = Σ (J/4)(X_i X_{i+1} + Y_i Y_{i+1} + Δ Z_i Z_{i+1})

Test points:
- Δ = 1.5  → Gapped (IN SCOPE)     → Expect: P2 ACCEPT
- Δ = 1.1  → Near-critical         → Expect: P2 marginal
- Δ = 1.0  → Critical (OUT SCOPE)  → Expect: P2 REJECT
- Δ = 0.5  → Gapless XY            → Expect: P2 REJECT
```

**Green Light Contribution:** If P2 transitions from ACCEPT to REJECT at Δ≈1, this confirms the framework's scope is physically meaningful.

---

### 2. P3 Redesign with Isometric Gluing (UNBLOCKS TEST SUITE)

**Why:** Only red status in test suite - must fix for complete validation

| Metric | Value |
|--------|-------|
| Impact | HIGH - Completes test suite |
| Effort | MEDIUM - Requires MERA isometric operations |
| Timeline | 4-8 hours |

**Design:**
```python
# Current (WRONG):
psi_glued = np.kron(psi_A, psi_B)  # Assumes uncorrelated

# Correct approach:
psi_glued = mera_isometric_glue(mera_A, mera_B, bond_structure)
# Uses MERA causal cone structure
# Preserves entanglement between partitions
```

**Green Light Contribution:** Removes the only 🔴 status from test suite.

---

### 3. Claim 3B Investigation (COMPLETES CLAIMS)

**Why:** Only claim with NO_EVIDENCE status

| Metric | Value |
|--------|-------|
| Impact | MEDIUM - Completes claim validation |
| Effort | LOW-MEDIUM - Need to understand what 3B tests |
| Timeline | 2-4 hours |

**Status:** Need to identify what Claim 3B actually tests (windowed regime transition)

**Green Light Contribution:** Moves all claims to SUPPORTED or NOT_SUPPORTED (no NO_EVIDENCE).

---

### 4. Critical Extension (LONG-TERM VALUE)

**Why:** Would extend framework to gapless/critical systems

| Metric | Value |
|--------|-------|
| Impact | HIGH - Extends scope |
| Effort | HIGH - Requires theoretical work |
| Timeline | Days-Weeks |

**Design:**
```
Critical Capacity Bound: C ≤ (c/6) log(L)

For Heisenberg (c=1): C ≤ 0.17 log(L)
Measured: C ≈ 0.47 log(L) (discrepancy to investigate)
```

**Green Light Contribution:** Not required for current green light, but enables future expansion.

---

## Recommended Execution Order

### Phase 1: Quick Wins (Today)

| Step | Action | Time | Outcome |
|------|--------|------|---------|
| 1.1 | XXZ boundary test | 2h | Validate scope boundary |
| 1.2 | Claim 3B investigation | 1h | Identify what needs testing |

### Phase 2: Complete Test Suite (This Week)

| Step | Action | Time | Outcome |
|------|--------|------|---------|
| 2.1 | P3 isometric gluing redesign | 4-8h | Remove 🔴 status |
| 2.2 | Claim 3B runner (if needed) | 2-4h | Complete claims |

### Phase 3: Strengthen Evidence (Optional)

| Step | Action | Time | Outcome |
|------|--------|------|---------|
| 3.1 | L=16 verification | 2h | Cross-scale validation |
| 3.2 | Additional gapped models | 4h | Generalize scope evidence |

---

## Green Light Criteria

A **green light** requires:

| Criterion | Current | Target | Gap |
|-----------|---------|--------|-----|
| All P-tests complete | 4/5 | 5/5 | P3 invalid |
| No INVALID status | 1 invalid | 0 | Fix P3 |
| All claims have evidence | 1 no-evidence | 0 | Fix 3B |
| Scope documented | ✅ | ✅ | None |
| Scope boundary validated | 🟡 | ✅ | XXZ test |

---

## Highest ROI Action

**XXZ Boundary Test**

This single test provides:
1. **Direct validation** that scope boundary is real
2. **Low effort** - reuse P2 runner with XXZ Hamiltonian
3. **High impact** - confirms framework correctly identifies gapped vs critical
4. **Publishable result** - demonstrates scope-aware framework

**Implementation:**
```python
# Add to existing P2 runner
MODELS = {
    'ising': {...},
    'heisenberg': {...},
    'xxz_delta_1.5': {'delta': 1.5, 'expected': 'GAPPED'},
    'xxz_delta_1.1': {'delta': 1.1, 'expected': 'MARGINAL'},
    'xxz_delta_1.0': {'delta': 1.0, 'expected': 'CRITICAL'},
    'xxz_delta_0.5': {'delta': 0.5, 'expected': 'GAPLESS'},
}
```

---

## Decision Matrix

| Action | Impact | Effort | ROI | Priority |
|--------|--------|--------|-----|----------|
| XXZ Boundary Test | HIGH | LOW | ⭐⭐⭐⭐⭐ | **1** |
| P3 Redesign | HIGH | MEDIUM | ⭐⭐⭐⭐ | **2** |
| Claim 3B | MEDIUM | LOW | ⭐⭐⭐ | **3** |
| Critical Extension | HIGH | HIGH | ⭐⭐ | **4** |

---

## Recommendation

**Start with XXZ Boundary Test today.**

This will:
- Confirm scope boundary is physically grounded
- Provide strongest evidence for framework validity
- Take only 1-2 hours
- Enable immediate progress toward green light

**Then fix P3 methodology** to remove the only red status.

**Result:** Complete test suite with validated scope = GREEN LIGHT.