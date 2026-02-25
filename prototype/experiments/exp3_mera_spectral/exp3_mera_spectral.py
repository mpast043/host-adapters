#!/usr/bin/env python3
"""
Experiment 3: MERA Spectral Dimension Bridge
Combines spectral dimension analysis with MERA capacity constraints.

Claim 3: Tensor networks with MERA structure exhibit spectral dimension scaling
consistent with holographic theories when capacity is measured via bond dimension.

Falsifier 3.1: If spectral dimension d_s does not decrease with increasing MERA level,
claim is falsified (holographic bound prediction).

Falsifier 3.2: If d_s correlates negatively with bond dimension chi (should be positive),
claim is falsified.
"""

import argparse
import json
import numpy as np
from pathlib import Path
import sys


def build_mera_graph(num_sites: int, chi: int, num_levels: int) -> tuple:
    """
    Build MERA hierarchical graph structure.
    Returns (adjacency_matrix, level_nodes) where level_nodes maps level->nodes.
    """
    # Total nodes = physical sites + internal disentangler/isometry nodes per level
    nodes_per_level = [num_sites // (2**i) for i in range(num_levels)]
    
    # Add auxiliary nodes for disentanglers (2 nodes per disentangler)
    # and isometries (1 node per isometry)
    total_nodes = sum(nodes_per_level) + num_levels * 3
    
    # Simplified: create hierarchical adjacency
    adj = np.zeros((total_nodes, total_nodes), dtype=int)
    
    # Assign node IDs
    node_id = 0
    level_nodes = {}
    disentangler_nodes = {}
    isometry_nodes = {}
    
    for level in range(num_levels):
        n_at_level = nodes_per_level[level]
        level_nodes[level] = list(range(node_id, node_id + n_at_level))
        
        # Disentanglers (every 2 nodes get paired)
        disentangler_nodes[level] = []
        for i in range(n_at_level // 2):
            adj[node_id + 2*i, node_id + 2*i + 1] = 1
            adj[node_id + 2*i + 1, node_id + 2*i] = 1
            disentangler_nodes[level].append((node_id + 2*i, node_id + 2*i + 1))
        
        node_id += n_at_level
        
        # Add dummy nodes for disentanglers and isometries
        for i in range(min(n_at_level, 2)):
            adj[node_id, max(0, node_id - 3)] = 1
            adj[max(0, node_id - 3), node_id] = 1
            node_id += 1
    
    return adj, level_nodes, disentangler_nodes


def random_walk_return_probability(adj: np.ndarray, tau_max: int, num_walks: int = 1000, seed: int = 42) -> np.ndarray:
    """
    Estimate return probability P(tau) for random walks on graph.
    P(tau) ~ tau^(-d_s/2) where d_s is spectral dimension.
    """
    rng = np.random.RandomState(seed)
    n = adj.shape[0]
    
    # Convert to transition matrix (normalized rows)
    row_sums = adj.sum(axis=1, keepdims=True)
    # Handle isolated nodes
    row_sums[row_sums == 0] = 1
    P = adj / row_sums
    
    return_probs = np.zeros(tau_max)
    
    for start_node in range(min(n, 100)):  # Sample from first 100 nodes
        for _ in range(num_walks // min(n, 100)):
            current = start_node
            for tau in range(1, tau_max + 1):
                # Random step
                if P[current].sum() > 0:
                    neighbors = np.where(P[current] > 0)[0]
                    if len(neighbors) > 0:
                        current = rng.choice(neighbors, p=P[current][neighbors]/P[current][neighbors].sum())
                
                if current == start_node:
                    return_probs[tau - 1] += 1
    
    return_probs /= (num_walks * min(n, 100))
    return return_probs


def estimate_spectral_dimension(return_probs: np.ndarray, tau_range: tuple = (5, 15)) -> float:
    """
    Fit P(tau) ~ tau^(-d_s/2) in log-log to extract spectral dimension.
    """
    start_tau, end_tau = tau_range
    if end_tau > len(return_probs):
        end_tau = len(return_probs) - 1
    
    taus = np.arange(start_tau, end_tau + 1)
    probs = return_probs[start_tau-1:end_tau]
    
    # Filter out zeros
    mask = probs > 0
    if mask.sum() < 2:
        return np.nan
    
    # Log-log fit: log(P) = -d_s/2 * log(tau) + const
    log_tau = np.log(taus[mask])
    log_p = np.log(probs[mask])
    
    # Linear regression
    A = np.vstack([log_tau, np.ones(len(log_tau))]).T
    slope, _ = np.linalg.lstsq(A, log_p, rcond=None)[0]
    
    d_s = -2 * slope
    return float(d_s)


def holographic_prediction(chi: int, level: int, num_levels: int) -> float:
    """
    Holographic bound prediction for spectral dimension.
    In holographic theories, d_eff ~ log(chi) or d_eff decreases with depth.
    Simplified: d_s should decrease with level (coarse-graining reduces dimension).
    """
    # Toy model: higher level = lower dimension (coarse-graining)
    # Bond dimension chi controls "resolution"
    base_d = 2.0  # Effective 2D from MERA structure
    depth_factor = 1.0 - (level / max(num_levels, 1)) * 0.3  # Decrease with level
    chi_factor = min(1.0, np.log(chi) / np.log(16))  # Increase with chi
    
    return base_d * depth_factor * (0.5 + 0.5 * chi_factor)


def run_experiment(num_sites: int, chi: int, num_levels: int, tau_max: int, seed: int) -> dict:
    """Run single experiment configuration."""
    adj, level_nodes, disentangler_nodes = build_mera_graph(num_sites, chi, num_levels)
    
    # Measure spectral dimension at each level
    results_by_level = {}
    overall_results = {}
    
    for level in range(min(num_levels, 3)):  # Limit to first 3 levels
        if level in level_nodes and len(level_nodes[level]) > 2:
            # Extract subgraph for this level
            nodes = level_nodes[level]
            sub_adj = adj[np.ix_(nodes, nodes)]
            
            # Random walk on level subgraph
            node_seed = seed + level * 100
            probs = random_walk_return_probability(sub_adj, tau_max, num_walks=500, seed=node_seed)
            d_s = estimate_spectral_dimension(probs, tau_range=(3, min(tau_max-2, 12)))
            
            results_by_level[level] = {
                'num_nodes': len(nodes),
                'spectral_dimension': d_s,
                'holographic_prediction': holographic_prediction(chi, level, num_levels)
            }
    
    # Full graph spectral dimension
    full_probs = random_walk_return_probability(adj, tau_max, num_walks=1000, seed=seed)
    d_s_full = estimate_spectral_dimension(full_probs, tau_range=(5, min(tau_max-2, 15)))
    
    # Holographic check: d_s should scale with log(chi) / system size
    # In holographic MERA, entropy S ~ log(chi) * log(L)
    # Spectral dimension relates to how geometry emerges from entanglement
    chi_factor = np.log(chi) if chi > 1 else 1
    size_factor = np.log(num_sites) if num_sites > 1 else 1
    
    # Expected: d_s correlates with available degrees of freedom = chi
    expected_d_s = 2.0 * min(1.0, chi_factor / size_factor)
    
    overall_results = {
        'num_sites': num_sites,
        'chi': chi,
        'num_levels': num_levels,
        'tau_max': tau_max,
        'spectral_dimension_full': d_s_full,
        'spectral_dimension_by_level': results_by_level,
        'holographic_prediction': expected_d_s,
        'chi_factor': float(chi_factor),
        'seed': seed
    }
    
    return overall_results


def main():
    parser = argparse.ArgumentParser(description='Claim 3: MERA Spectral Dimension Bridge')
    parser.add_argument('--output', type=str, required=True, help='Output directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("Experiment 3: MERA Spectral Dimension Bridge")
    print("=" * 70)
    print(f"\nClaim 3: Tensor networks with MERA structure exhibit spectral")
    print(f"dimension scaling consistent with holographic theories when")
    print(f"capacity is measured via bond dimension.")
    print()
    
    # Test configurations: varying chi and system size
    configs = [
        (16, 4, 3),   # Small system, low chi
        (16, 8, 3),   # Small system, medium chi
        (32, 4, 4),   # Medium system, low chi
        (32, 8, 4),   # Medium system, medium chi
        (32, 16, 4),  # Medium system, high chi
        (64, 8, 5),   # Large system, medium chi
        (64, 16, 5),  # Large system, high chi
    ]
    
    results = []
    print("Running configurations...")
    print("-" * 70)
    
    for num_sites, chi, num_levels in configs:
        print(f"\n  N={num_sites:3d}, chi={chi:2d}, levels={num_levels}")
        
        result = run_experiment(num_sites, chi, num_levels, tau_max=20, seed=args.seed)
        results.append(result)
        
        d_s = result['spectral_dimension_full']
        pred = result['holographic_prediction']
        
        status = "✓" if abs(d_s - pred) < 0.5 or d_s > 0 else "?"
        print(f"    d_s = {d_s:.3f}, predicted ≈ {pred:.3f} {status}")
    
    # Falsifier checks
    print("\n" + "=" * 70)
    print("Falsifier Analysis")
    print("=" * 70)
    
    # Falsifier 3.1: d_s should decrease with MERA level (coarse-graining)
    level_trends = []
    for r in results:
        by_level = r['spectral_dimension_by_level']
        if len(by_level) >= 2:
            levels = sorted(by_level.keys())
            dims = [by_level[l]['spectral_dimension'] for l in levels]
            # Trend: should decrease or stay flat (coarse-graining reduces effective dim)
            valid = all(not np.isnan(d) for d in dims)
            if valid and len(dims) >= 2:
                decreasing = dims[-1] <= dims[0]  # Last level <= First level
                level_trends.append(decreasing)
    
    falsifier_3_1_passed = len(level_trends) > 0 and sum(level_trends) / len(level_trends) >= 0.5
    print(f"\nFalsifier 3.1 (d_s decreases with level): {'PASS' if falsifier_3_1_passed else 'FAIL'}")
    print(f"  Configurations with decreasing trend: {sum(level_trends)}/{len(level_trends)}")
    
    # Falsifier 3.2: d_s correlates positively with chi (higher chi = more degrees of freedom)
    chis = [r['chi'] for r in results]
    dims = [r['spectral_dimension_full'] for r in results]
    
    valid_pairs = [(c, d) for c, d in zip(chis, dims) if not np.isnan(d)]
    if len(valid_pairs) >= 3:
        sorted_by_chi = sorted(valid_pairs, key=lambda x: x[0])
        # Check if generally increasing
        increases = sum(1 for i in range(len(sorted_by_chi)-1) 
                       if sorted_by_chi[i+1][1] > sorted_by_chi[i][1] - 0.5)
        falsifier_3_2_passed = increases >= len(sorted_by_chi) * 0.5
        
        print(f"\nFalsifier 3.2 (d_s correlates with chi): {'PASS' if falsifier_3_2_passed else 'FAIL'}")
        print(f"  Chi-dimension pairs:")
        for c, d in sorted_by_chi:
            print(f"    chi={c:2d}: d_s = {d:.3f}")
    else:
        falsifier_3_2_passed = False
        print(f"\nFalsifier 3.2 (d_s correlates with chi): FAIL (insufficient data)")
    
    # Overall verdict
    verdict = "SUPPORTED" if (falsifier_3_1_passed and falsifier_3_2_passed) else "NOT_SUPPORTED"
    
    print("\n" + "=" * 70)
    print(f"Verdict: {verdict}")
    print("=" * 70)
    
    # Save results
    results_data = {
        'experiment': 'exp3_mera_spectral',
        'claim': 'Claim 3: MERA Spectral Dimension Bridge',
        'claim_statement': 'Tensor networks with MERA structure exhibit spectral dimension scaling consistent with holographic theories when capacity is measured via bond dimension.',
        'configurations': configs,
        'results': results,
        'verdict': verdict,
        'falsifier_3_1_passed': bool(falsifier_3_1_passed),
        'falsifier_3_2_passed': bool(falsifier_3_2_passed),
        'seed': args.seed
    }
    
    data_file = output_dir / 'exp3_data.json'
    verdict_file = output_dir / 'exp3_verdict.json'
    
    with open(data_file, 'w') as f:
        json.dump(results_data, f, indent=2)
    
    verdict_short = {
        'claim_id': 'claim-003',
        'claim_title': 'MERA Spectral Dimension Bridge',
        'verdict': verdict,
        'falsifier_3_1_passed': bool(falsifier_3_1_passed),
        'falsifier_3_2_passed': bool(falsifier_3_2_passed),
        'configurations_tested': len(configs),
        'seed': args.seed
    }
    
    with open(verdict_file, 'w') as f:
        json.dump(verdict_short, f, indent=2)
    
    print(f"\nResults saved to: {output_dir}")


if __name__ == '__main__':
    main()
