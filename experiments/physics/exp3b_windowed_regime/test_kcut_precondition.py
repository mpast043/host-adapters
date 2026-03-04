#!/usr/bin/env python3
"""
K_cut Precondition Test for Claim 3B
Validates K_cut is invariant across seeds and non-decreasing with A.
Frame v0.2.1 compliance: Required before regime detection verdicts.
"""

import numpy as np
import sys
sys.path.insert(0, '/tmp/openclaws/Repos/host-adapters/prototype/experiments/exp3b_windowed_regime')

from exp3b_windowed_regime import compute_minimal_cut, compute_minimal_cut_legacy

def test_kcut_invariant_and_scaling(formula_name: str, kcut_fn):
    """Test K_cut invariance and scaling properties."""
    print(f"\n{'='*60}")
    print(f"Testing: {formula_name}")
    print(f"{'='*60}")
    
    N = 16
    A_values = [4, 6, 8]
    seeds = range(10)
    
    K_by_A = {}
    
    for A in A_values:
        Ks = []
        for seed in seeds:
            np.random.seed(seed)
            Ks.append(kcut_fn(A, N))
        K_by_A[A] = Ks
        
        print(f"\n  Partition A={A}:")
        print(f"    K_cut across seeds: {Ks}")
        print(f"    Unique values: {sorted(set(Ks))}")
        print(f"    Median: {np.median(Ks):.1f}")
        
        # Check invariance (deterministic formula should be invariant)
        variance = max(Ks) - min(Ks)
        if variance > 0:
            print(f"    ⚠️  WARNING: K_cut varies by {variance} across seeds")
            invariant = False
        else:
            print(f"    ✓ Invariant across seeds")
            invariant = True
    
    # Check non-decreasing with A
    medians = {A: int(np.median(Ks)) for A, Ks in K_by_A.items()}
    print(f"\n  Median K_cut by A: {medians}")
    
    non_decreasing = True
    for i in range(len(A_values) - 1):
        A_curr, A_next = A_values[i], A_values[i + 1]
        if medians[A_next] < medians[A_curr]:
            print(f"    ⚠️  VIOLATION: K_cut decreased from A={A_curr} ({medians[A_curr]}) to A={A_next} ({medians[A_next]})")
            non_decreasing = False
    
    if non_decreasing:
        print(f"    ✓ Non-decreasing with partition size")
    
    # Compute χ_sat predictions
    print(f"\n  χ_sat Predictions:")
    for A in A_values:
        K = medians[A]
        S_max = A * np.log(2)
        chi_sat = np.exp(S_max / K)
        
        if chi_sat <= 2:
            status = "immediate saturation"
        elif chi_sat >= 64:
            status = "unreachable in sweep"
        else:
            status = f"transition at χ≈{chi_sat:.0f}"
        
        print(f"    A={A}: S_max={S_max:.2f}, K={K}, χ_sat={chi_sat:.1f} → {status}")
    
    return invariant and non_decreasing

if __name__ == "__main__":
    print("="*60)
    print("K_CUT PRECONDITION TEST")
    print("Framework v0.2.1 Compliance Check")
    print("="*60)
    
    # Test legacy formula
    legacy_pass = test_kcut_invariant_and_scaling("LEGACY (buggy)", compute_minimal_cut_legacy)
    
    # Test corrected formula
    corrected_pass = test_kcut_invariant_and_scaling("CORRECTED", compute_minimal_cut)
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Legacy formula: {'PASS' if legacy_pass else 'FAIL'} (expected: FAIL)")
    print(f"Corrected formula: {'PASS' if corrected_pass else 'FAIL'} (expected: PASS)")
    
    if not legacy_pass and corrected_pass:
        print("\n✓ Fix validated: K_cut now scales correctly")
        sys.exit(0)
    else:
        print("\n⚠️  Unexpected result - needs investigation")
        sys.exit(1)
