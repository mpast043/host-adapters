#!/usr/bin/env python3
"""
Experiment 3 v2: MERA Spectral Dimension Bridge (Improved)

Claim 3: Tensor networks with MERA structure exhibit spectral dimension scaling
consistent with holographic theories when capacity is measured via bond dimension.

Key Insight: In MERA, bond dimension chi controls the effective dimensionality.
Holographic bound: effective dimension d_eff ~ log(chi) / log(renormalization_scale)

Falsifier 3.1: If MERA level (coarse-graining depth) does not reduce effective 
degrees of freedom (measured via return probability timescale), claim is falsified.

Falsifier 3.2: If bond dimension chi does not correlate with spectral dimension
(e.g., larger chi should give higher d_s or faster diffusion), claim is falsified.
"""

import argparse
import json
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict


def build_mera_hierarchy(num_sites: int, chi: int, num_levels: int) -> Dict:
    """
    Build a simplified MERA hierarchy as a tree-like graph.
    
    Structure:
    - Level 0: Physical sites
    - Level 1-n: Coarse-grained layers with bond dimension chi
    """
    nodes = []
    edges = []
    node_counter = 0
    level_starts = []
    
    for level in range(num_levels + 1):
        level_starts.append(node_counter)
        
        # Number of nodes at this level decreases with renormalization
        if level == 0:
            n_nodes = num_sites
        else:
            # Coarse-graining: ~ half the nodes per level
            n_nodes = max(2, num_sites // (2 ** level))
        
        level_nodes = list(range(node_counter, node_counter + n_nodes))
        nodes.extend([(n, level) for n in level_nodes])
        
        # Add edges: nodes connect to "parent" nodes in next level or within level
        if level > 0:
            # Connect to previous level (renormalization)
            prev_level_nodes = list(range(level_starts[level-1], level_starts[level]))
            for i, node in enumerate(level_nodes):
                # Each node connects to 2 nodes in previous level (disentangler + isometry)
                parent1 = prev_level_nodes[min(2*i, len(prev_level_nodes)-1)]
                parent2 = prev_level_nodes[min(2*i+1, len(prev_level_nodes)-1)]
                edges.append((node, parent1))
                edges.append((node, parent2))
        
        node_counter += n_nodes
    
    return {
        'nodes': nodes,
        'edges': edges,
        'num_total_nodes': node_counter,
        'level_starts': level_starts,
        'chi': chi
    }


def build_adjacency_matrix(graph: Dict) -> np.ndarray:
    """Convert edge list to adjacency matrix."""
    n = graph['num_total_nodes']
    adj = np.zeros((n, n), dtype=np.float64)
    
    for i, j in graph['edges']:
        adj[i, j] = 1.0
        adj[j, i] = 1.0  # Undirected
    
    return adj


def heat_kernel_trace(adj: np.ndarray, tau_max: int, dt: float = 0.1, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute heat kernel trace K(tau) = sum of return probabilities.
    K(tau) ~ tau^(-d_s/2) for spectral dimension d_s.
    """
    np.random.seed(seed)
    n = adj.shape[0]
    
    # Add self-loops for stability
    adj_diagonal = adj + np.eye(n) * np.diag(adj).sum() / n
    
    # Normalized Laplacian
    degrees = adj_diagonal.sum(axis=1)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(degrees + 1e-10))
    L = np.eye(n) - D_inv_sqrt @ adj_diagonal @ D_inv_sqrt
    
    # Compute heat kernel eigenvalues and eigenvectors
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(L)
    except np.linalg.LinAlgError:
        # Fallback: use simple random walk
        return simple_random_walk_trace(adj, tau_max, seed)
    
    # Heat kernel trace: sum over eigenvalues
    taus = np.arange(1, tau_max + 1)
    K_tau = np.zeros(tau_max)
    
    for i, tau in enumerate(taus):
        # Z(tau) = sum_i exp(-lambda_i * tau)
        K_tau[i] = np.sum(np.exp(-eigenvalues[1:] * tau * dt))  # Skip lambda=0
    
    return taus.astype(np.float64), K_tau


def simple_random_walk_trace(adj: np.ndarray, tau_max: int, seed: int, num_samples: int = 200) -> Tuple[np.ndarray, np.ndarray]:
    """
    Monte Carlo random walk return probability.
    """
    np.random.seed(seed)
    n = adj.shape[0]
    
    # Transition matrix
    degrees = adj.sum(axis=1)
    P = np.zeros_like(adj)
    for i in range(n):
        if degrees[i] > 0:
            P[i] = adj[i] / degrees[i]
    
    taus = np.arange(1, tau_max + 1)
    return_probs = np.zeros(tau_max)
    
    # Sample from nodes (not all nodes for efficiency)
    sample_nodes = np.random.choice(n, size=min(num_samples, n), replace=False)
    
    for start in sample_nodes:
        current = start
        for tau in range(tau_max):
            if P[current].sum() > 0:
                neighbors = np.where(P[current] > 0)[0]
                probs = P[current, neighbors]
                probs = probs / probs.sum()
                current = np.random.choice(neighbors, p=probs)
            
            if current == start:
                return_probs[tau] += 1.0
    
    return_probs /= len(sample_nodes)
    return taus.astype(np.float64), return_probs


def estimate_spectral_dimension_supertorso(K_tau: np.ndarray, taus: np.ndarray, fit_range: Tuple[int, int] = (3, 15)) -> float:
    """
    Estimate spectral dimension from slope of log(K(tau)) vs log(tau).
    For small tau: log(K) ~ constant - (d_s/2) * log(tau)
    """
    start, end = fit_range
    start = max(0, start)
    end = min(len(taus), end)
    
    if end - start < 3:
        return np.nan
    
    log_tau = np.log(taus[start:end])
    log_K = np.log(K_tau[start:end] + 1e-15)
    
    # Linear fit
    A = np.vstack([log_tau, np.ones(len(log_tau))]).T
    slope, intercept = np.linalg.lstsq(A, log_K, rcond=None)[0]
    
    # d_s = -2 * slope
    d_s = -2.0 * slope
    
    return max(0, d_s)  # Spectral dimension should be non-negative


def holographic_prediction(chi: int, num_sites: int) -> float:
    """
    Holographic bound prediction: 
    In AdS/CFT-inspired MERA, area law suggests d_s ~ log(chi) / log(system_size)
    
    Simplified model: d_s = c * log(chi) where c ~ 1 for reasonable chi
    """
    if chi <= 1:
        return 1.0
    
    # Effective dimension from holographic bound
    # chi represents "resolution" in bulk
    d_eff = 1.0 + np.log(chi) / np.log(2)
    
    # Normalize by system size factor
    size_factor = 1.0 + 0.1 * np.log(num_sites / 16)
    
    return d_eff / size_factor


def run_claim3_experiment(config: Dict, seed: int) -> Dict:
    """Run single Claim 3 experiment."""
    num_sites = config['num_sites']
    chi = config['chi']
    num_levels = config['num_levels']
    
    # Build graph
    graph = build_mera_hierarchy(num_sites, chi, num_levels)
    adj = build_adjacency_matrix(graph)
    
    # Compute spectral dimension via heat kernel
    taus, K_tau = heat_kernel_trace(adj, tau_max=30, dt=0.05, seed=seed)
    d_s = estimate_spectral_dimension_supertorso(K_tau, taus, fit_range=(3, 20))
    
    # Holographic prediction
    d_s_pred = holographic_prediction(chi, num_sites)
    
    # Level-by-level analysis
    level_results = {}
    for level in range(min(num_levels + 1, 4)):
        if level < len(graph['level_starts']):
            start = graph['level_starts'][level]
            end = graph['level_starts'][level + 1] if level + 1 < len(graph['level_starts']) else graph['num_total_nodes']
            n_level = end - start
            
            if n_level >= 4:  # Minimum for analysis
                # Extract subgraph and compute spectral dim
                sub_adj = adj[start:end, start:end]
                if sub_adj.sum() > 0:
                    _, K_level = heat_kernel_trace(sub_adj, tau_max=15, dt=0.05, seed=seed + level * 111)
                    d_level = estimate_spectral_dimension_supertorso(K_level, np.arange(1, 16), fit_range=(2, 10))
                    level_results[level] = {
                        'num_nodes': n_level,
                        'spectral_dimension': d_level
                    }
    
    return {
        'config': config,
        'spectral_dimension': float(d_s) if not np.isnan(d_s) else 0.0,
        'holographic_prediction': float(d_s_pred),
        'level_results': level_results,
        'heat_kernel_trace': K_tau[:10].tolist()  # First 10 for debugging
    }


def main():
    parser = argparse.ArgumentParser(description='Claim 3: MERA Spectral Dimension Bridge')
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("Experiment 3 v2: MERA Spectral Dimension Bridge")
    print("=" * 70)
    print("\nClaim 3: MERA tensor networks exhibit spectral dimension scaling")
    print("consistent with holographic theories (d_s ~ capacity ~ log(chi))")
    print()
    
    # Test configurations
    configs = [
        {'num_sites': 16, 'chi': 2, 'num_levels': 2},
        {'num_sites': 16, 'chi': 4, 'num_levels': 2},
        {'num_sites': 32, 'chi': 4, 'num_levels': 3},
        {'num_sites': 32, 'chi': 8, 'num_levels': 3},
        {'num_sites': 64, 'chi': 4, 'num_levels': 4},
        {'num_sites': 64, 'chi': 8, 'num_levels': 4},
        {'num_sites': 64, 'chi': 16, 'num_levels': 4},
    ]
    
    results = []
    print("Running configurations...")
    print("-" * 70)
    
    for i, config in enumerate(configs):
        print(f"\n  Config {i+1}: N={config['num_sites']}, chi={config['chi']}, levels={config['num_levels']}")
        
        result = run_claim3_experiment(config, seed=args.seed + i * 1000)
        results.append(result)
        
        d_s = result['spectral_dimension']
        pred = result['holographic_prediction']
        match = abs(d_s - pred) < 1.0
        
        print(f"    Spectral dim: {d_s:.3f}")
        print(f"    Predicted:    {pred:.3f}")
        print(f"    Match: {'✓' if match else '✗'}")
    
    # Falsifier Analysis
    print("\n" + "=" * 70)
    print("Falsifier Analysis")
    print("=" * 70)
    
    # Falsifier 3.1: Coarse-graining reduces effective dimension
    level_checks = []
    for r in results:
        levels = r['level_results']
        if len(levels) >= 2:
            dims = [levels[l]['spectral_dimension'] for l in sorted(levels.keys())]
            non_nan = [d for d in dims if d > 0]
            if len(non_nan) >= 2:
                # Check if dimension decreases with level (coarse-graining)
                decreasing = non_nan[-1] <= non_nan[0]
                level_checks.append(decreasing)
    
    falsifier_3_1_passed = len(level_checks) > 0 and sum(level_checks) / len(level_checks) >= 0.5
    print(f"\nFalsifier 3.1 (d_s decreases with coarse-graining level): {'PASS' if falsifier_3_1_passed else 'FAIL'}")
    print(f"  {sum(level_checks)}/{len(level_checks)} configurations show decreasing trend")
    
    # Falsifier 3.2: chi correlates with d_s
    chis = [r['config']['chi'] for r in results]
    dims = [r['spectral_dimension'] for r in results]
    
    # Compute correlation
    valid = [(c, d) for c, d in zip(chis, dims) if d > 0]
    if len(valid) >= 3:
        chi_vals = [x[0] for x in valid]
        dim_vals = [x[1] for x in valid]
        
        if len(set(chi_vals)) > 1 and len(set(dim_vals)) > 1:
            correlation = np.corrcoef(chi_vals, dim_vals)[0, 1]
            falsifier_3_2_passed = correlation > 0.0  # Positive correlation expected
        else:
            correlation = 0.0
            falsifier_3_2_passed = False
    else:
        correlation = 0.0
        falsifier_3_2_passed = False
    
    print(f"\nFalsifier 3.2 (d_s correlates positively with chi): {'PASS' if falsifier_3_2_passed else 'FAIL'}")
    print(f"  Correlation: {correlation:.3f}")
    print(f"  Chi-d_s pairs:")
    for c, d in zip(chis, dims):
        print(f"    chi={c:2d}: d_s = {d:.3f}")
    
    # Verdict
    verdict = "SUPPORTED" if (falsifier_3_1_passed and falsifier_3_2_passed) else "NOT_SUPPORTED"
    
    print("\n" + "=" * 70)
    print(f"Verdict: {verdict}")
    print("=" * 70)
    
    # Save outputs
    full_results = {
        'experiment': 'exp3_mera_spectral_v2',
        'claim': 'Claim 3: MERA Spectral Dimension Bridge',
        'claim_statement': 'Tensor networks with MERA structure exhibit spectral dimension scaling consistent with holographic theories when capacity is measured via bond dimension.',
        'results': results,
        'verdict': verdict,
        'falsifier_3_1_passed': bool(falsifier_3_1_passed),
        'falsifier_3_2_passed': bool(falsifier_3_2_passed),
        'correlation_chi_ds': float(correlation),
        'seed': args.seed
    }
    
    with open(output_dir / 'exp3_data.json', 'w') as f:
        json.dump(full_results, f, indent=2)
    
    verdict_short = {
        'claim_id': 'claim-003',
        'claim_title': 'MERA Spectral Dimension Bridge',
        'verdict': verdict,
        'falsifier_3_1_passed': bool(falsifier_3_1_passed),
        'falsifier_3_2_passed': bool(falsifier_3_2_passed),
        'correlation': float(correlation),
        'sample_count': len(configs),
        'seed': args.seed
    }
    
    with open(output_dir / 'exp3_verdict.json', 'w') as f:
        json.dump(verdict_short, f, indent=2)
    
    print(f"\nResults saved to: {output_dir}")


if __name__ == '__main__':
    main()
