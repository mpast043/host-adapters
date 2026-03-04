#!/usr/bin/env python3
"""
K_cut validation test and corrected min-cut implementation
Tests that K_cut:
1. Is invariant across seeds for fixed layout and partition
2. Is nondecreasing with partition size for contiguous intervals
"""

import numpy as np
import sys
from pathlib import Path

# Add repo to path
sys.path.insert(0, str(Path("/tmp/openclaws/Repos/host-adapters")))


def kcut_mincut_legcount(mera, site_inds, A_sites):
    """
    K_cut computed as min s-t cut on an index-expanded graph.
    
    Key property:
    - Each tensor-network index becomes its own node, so multiplicities are preserved.
    - Internal tensor-index edges have capacity 1 (each counts as one cut leg).
    - Physical legs are connected with BIG capacity to prevent cutting boundary legs.
    """
    import networkx as nx
    
    A_set = set(A_sites)
    # Map physical site index -> side
    phys_side = {ind: ("A" if i in A_set else "B") for i, ind in enumerate(site_inds)}
    
    SRC, SNK = "__SRC__", "__SNK__"
    BIG = 10**9
    
    G = nx.DiGraph()
    G.add_node(SRC)
    G.add_node(SNK)
    
    # Collect all unique tensors from ind_map (comprehensive)
    all_tensors = set()
    for ind, ts in mera.ind_map.items():
        all_tensors.update(ts)
    all_tensors = list(all_tensors)
    
    t_index = {id(t): f"T{ti}" for ti, t in enumerate(all_tensors)}
    for node in t_index.values():
        G.add_node(node)
    
    # For every index, create an explicit index node
    for ind, ts in mera.ind_map.items():
        ind_node = f"I::{ind}"
        G.add_node(ind_node)
        is_phys = ind in phys_side
        
        # Connect tensor <-> index edges
        # Internal edges count as 1 leg; physical edges are BIG (not cuttable)
        cap = BIG if is_phys else 1
        for t in ts:
            tn = t_index.get(id(t))
            if tn is None:
                continue  # Skip if tensor not in our index
            G.add_edge(tn, ind_node, capacity=cap)
            G.add_edge(ind_node, tn, capacity=cap)
        
        # Attach physical indices to SRC/SNK with BIG capacity
        if is_phys:
            if phys_side[ind] == "A":
                G.add_edge(SRC, ind_node, capacity=BIG)
                G.add_edge(ind_node, SRC, capacity=BIG)
            else:
                G.add_edge(ind_node, SNK, capacity=BIG)
                G.add_edge(SNK, ind_node, capacity=BIG)
    
    cut_val, _ = nx.minimum_cut(G, SRC, SNK, capacity="capacity")
    return int(cut_val)


def test_kcut_scaling_sanity():
    """
    Precondition test for K_cut validity.
    Validates:
    1. K_cut is invariant across seeds for fixed layout and partition
    2. K_cut is nondecreasing with partition size for contiguous intervals
    """
    import quimb.tensor as qtn
    
    L = 16
    chi = 8  # Fixed chi for layout test
    A_sizes = [4, 6, 8]
    n_seeds = 10
    
    def get_site_inds(mera, L):
        return [mera.site_ind(i) for i in range(L)]
    
    K_by_A = {}
    
    print(f"Testing K_cut scaling for L={L}, chi={chi}")
    print(f"Partition sizes: {A_sizes}")
    print(f"Seeds per partition: {n_seeds}")
    print("-" * 60)
    
    for A in A_sizes:
        Ks = []
        for seed in range(n_seeds):
            np.random.seed(seed)
            mera = qtn.MERA.rand(L, max_bond=chi, dtype="complex128")
            mera.isometrize_()
            site_inds = get_site_inds(mera, L)
            K = kcut_mincut_legcount(mera, site_inds, list(range(A)))
            Ks.append(K)
        K_by_A[A] = Ks
        
        unique_Ks = sorted(set(Ks))
        print(f"A={A}: K_cut values = {Ks}")
        print(f"       Unique values: {unique_Ks}")
        print(f"       Invariant: {'PASS' if len(unique_Ks) == 1 else 'FAIL - VARIES!'}")
        print()
    
    # Check invariance across seeds
    print("-" * 60)
    print("VALIDATION RESULTS:")
    print("-" * 60)
    
    all_invariant = True
    for A, Ks in K_by_A.items():
        unique_Ks = set(Ks)
        if len(unique_Ks) > 1:
            all_invariant = False
            print(f"❌ A={A}: K_cut varies with seed: {unique_Ks}")
        else:
            print(f"✓ A={A}: K_cut invariant across seeds (={Ks[0]})")
    
    # Check nondecreasing with A
    med = {A: int(np.median(Ks)) for A, Ks in K_by_A.items()}
    print()
    print(f"Median K_cut by A: {med}")
    
    nondecreasing = True
    if med[6] < med[4]:
        nondecreasing = False
        print(f"❌ K_cut decreased from A=4 to A=6: {med[4]} -> {med[6]}")
    else:
        print(f"✓ A=4 -> A=6: {med[4]} -> {med[6]} (nondecreasing)")
    
    if med[8] < med[6]:
        nondecreasing = False
        print(f"❌ K_cut decreased from A=6 to A=8: {med[6]} -> {med[8]}")
    else:
        print(f"✓ A=6 -> A=8: {med[6]} -> {med[8]} (nondecreasing)")
    
    print()
    print("-" * 60)
    if all_invariant and nondecreasing:
        print("✅ ALL CHECKS PASS - K_cut is valid for regime detection")
        return True, med
    else:
        print("❌ CHECKS FAILED - K_cut validity precheck failed")
        return False, med


if __name__ == "__main__":
    passed, medians = test_kcut_scaling_sanity()
    sys.exit(0 if passed else 1)
