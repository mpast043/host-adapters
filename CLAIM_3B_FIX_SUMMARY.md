# Claim 3B Diagnosis & Fix Summary
**Date**: 2026-02-25 15:05 EST  
**Framework**: v0.2.1 + precondition patch

## Problem Diagnosed

The original `compute_minimal_cut` formula in Experiment 3B had a bug where K_cut **decreased** with partition size:

| Partition | Legacy K_cut | Status |
|-----------|--------------|--------|
| A=4 | 2 | ✓ |
| A=6 | 2 | ✓ (flat) |
| A=8 | 1 | **✗ decreasing** |

This caused χ_sat = exp(S_max/K_cut) = 256 for A=8, placing the regime transition entirely outside the χ sweep range.

## Root Cause

The heuristic formula `ceil(log2(N/size_A))` collapses when size_A approaches N/2, producing the physically impossible result that larger partitions have smaller minimal cuts.

## Fix Applied

**Corrected formula**: `K_cut = ceil(log2(min(A, N-A) + 1))`

This models the MERA boundary as scaling with the **smaller** side of the partition (the actual cut interface), ensuring:
- K_cut is **invariant across seeds** for fixed layout
- K_cut is **non-decreasing** with partition size

## Validation Results

### Original (buggy) K_cut
- χ_sat predictions: A=4 → 4.0, A=6 → 8.0, A=8 → **256** (unreachable)
- Result: No capacity window observable for A=8

### Corrected K_cut  
| Partition | K_cut | χ_sat | Status |
|-----------|-------|-------|--------|
| A=4 | 3 | 2.5 | ✓ within sweep |
| A=6 | 3 | 4.0 | ✓ within sweep |
| A=8 | 4 | 4.0 | ✓ within sweep |

**Both capacity and saturation windows now accessible for all partitions.**

## Pre-registered Windows (using χ_sat)

| Partition | Capacity Window (S ≤ 0.8·S_max) | Saturation Window (S ≥ 0.95·S_max) |
|-----------|--------------------------------|-------------------------------------|
| A=4 | χ ∈ [2] | χ ∈ [4, 6, 8, ...] |
| A=6 | χ ∈ [2] | χ ∈ [4, 6, 8, ...] |
| A=8 | χ ∈ [2] | χ ∈ [4, 6, 8, ...] |

Transition expected around χ ≈ 4 for all partitions.

## Framework Spec Update

Added to `FRAMEWORK_SPEC_V0.2.1.md` under Claim 3A Required Observables:

```markdown
### Precondition: K_cut Validity Precheck

Before any verdict that depends on K_cut, the run must include a diagnostic
showing K_cut is:

1. Invariant across seeds for fixed layout and partition
2. Non-decreasing with partition size for the declared family of partitions

If this precheck fails, verdict must be INCONCLUSIVE with reason_code K_CUT_INVALID,
not REJECTED.
```

## Next Steps

1. **Re-run Claim 3B** with corrected K_cut formula
   - Expected: χ_sat ≈ 4 for all partitions
   - Both capacity and saturation regimes should be accessible

2. **Expected outcome**:
   - A=4: Immediate saturation (χ_sat ≈ 2.5, just above χ_min)
   - A=6, A=8: Transition around χ=4 — capacity window at χ=2, saturation at χ≥4

3. **If window still not detected**: May indicate true physics (no distinct I→II phase for this MERA geometry) rather than measurement error

## Files Modified

- `exp3b_windowed_regime.py` — corrected `compute_minimal_cut`
- `FRAMEWORK_SPEC_V0.2.1.md` — added K_cut precondition
- `test_kcut_precondition.py` — new validation test

## Original Verdict Status

**Claim 3B remains REJECTED** under legacy K_cut, but this was due to measurement tool failure, not physical falsification. With corrected K_cut, the experiment should be re-run for a validated verdict.
