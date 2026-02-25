#!/usr/bin/env python3
"""
Claim 3B Extended Run - L=32, A=16, finer early χ sampling
Tests window hypothesis with parameters that place χ_sat in resolvable range
"""

import sys
sys.path.insert(0, '/tmp/openclaws/Repos/host-adapters/prototype/experiments/exp3b_windowed_regime')

from pathlib import Path
from exp3b_windowed_regime import run_windowed_experiment

# User parameters: L=32, A=16, fine early χ sweep, 7 seeds
N = 32
partitions = [16]  # Single partition at half-system
chi_values = [2, 3, 4, 5, 6, 8, 10, 12, 16]  # Fine early sampling
seeds = list(range(7))  # 0,1,2,3,4,5,6

output_dir = Path("outputs/exp3b_windowed_regime/run_l32_a16")

# Calculate expected χ_sat
import numpy as np
A = 16
S_max = A * np.log(2)
K_cut = max(1, int(np.ceil(np.log2(min(A, N - A) + 1))))
chi_sat = np.exp(S_max / K_cut)

print("=" * 70)
print("CLAIM 3B EXTENDED RUN")
print("=" * 70)
print(f"System: N={N}, Partition A={A}")
print(f"S_max = {S_max:.2f}")
print(f"K_cut = {K_cut}")
print(f"Predicted χ_sat = {chi_sat:.1f}")
print(f"χ sweep: {chi_values}")
print(f"χ_sat position: between χ={int(chi_sat//1)} and χ={int(chi_sat//1)+1}")
print(f"Seeds: {seeds} ({len(seeds)} restarts)")
print("=" * 70)

# Pre-registered windows around predicted transition
# W1: very early (pure capacity regime expected)
# W2: early (approaching transition)
# W3: transition zone
windows = {
    "W1": [2, 3, 4],      # Pure capacity
    "W2": [2, 3, 4, 5, 6], # Approaching saturation
    "W3": [4, 5, 6, 8],    # Transition zone
    "W4": [8, 10, 12, 16]  # Saturation regime
}

# Patch the windows into the module
import exp3b_windowed_regime
original_windows = exp3b_windowed_regime.run_windowed_experiment.__code__.co_consts

# We need to modify the function to use custom windows
# Let's monkey-patch or just call directly with modified code
# Actually, simpler: copy the file and modify

if __name__ == "__main__":
    # Custom windows for fine-grained analysis
    custom_windows = {
        "W1": [2, 3, 4],        # Pure capacity
        "W2": [2, 3, 4, 5, 6],  # Approaching transition
        "W3": [4, 5, 6, 8],     # Transition zone
        "W4": [8, 10, 12, 16]   # Saturation
    }
    
    result = run_windowed_experiment(
        output_dir=output_dir,
        chi_values=chi_values,
        partitions=partitions,
        seeds=seeds,
        state_type="entanglement_max",
        N=N,
        windows=custom_windows
    )
    
    print(f"\n{'=' * 70}")
    print(f"FINAL VERDICT: {result['verdict']}")
    print(f"{'=' * 70}")
    
    # Post-hoc: Did we hit the predicted transition?
    print(f"\nPost-hoc analysis:")
    print(f"Predicted χ_sat = {chi_sat:.1f}")
    print(f"If transition observed around χ≈{chi_sat:.0f}, falsifiers should pass")
    print(f"If no transition or at wrong χ, hypothesis may need revision")
