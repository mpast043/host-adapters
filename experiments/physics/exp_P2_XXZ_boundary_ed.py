#!/usr/bin/env python3
"""
P2 XXZ Boundary Test - ED Only Version
Quick validation using exact diagonalization (no torch required)

Tests the capacity framework scope boundary using ED entanglement entropy
for various system sizes L. For critical systems, S ~ log(L). For gapped
systems, S saturates to a constant.

Usage:
    python3 exp_P2_XXZ_boundary_ed.py --deltas 0.5,1.0,1.1,1.5,2.0 --L_values 4,6,8
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np


def build_heisenberg_hamiltonian(L: int) -> np.ndarray:
    """Build Heisenberg Hamiltonian (Δ=1 case)."""
    dim = 2 ** L
    H = np.zeros((dim, dim), dtype=np.complex128)
    
    I = np.array([[1, 0], [0, 1]], dtype=np.float64)
    Sx = 0.5 * np.array([[0, 1], [1, 0]], dtype=np.float64)
    Sy = 0.5 * np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    Sz = 0.5 * np.array([[1, 0], [0, -1]], dtype=np.float64)
    
    for i in range(L - 1):
        for S1, S2 in [(Sx, Sx), (Sy, Sy), (Sz, Sz)]:
            ops = [I] * L
            ops[i] = S1
            ops[i + 1] = S2
            
            SdotS = ops[0]
            for op in ops[1:]:
                SdotS = np.kron(SdotS, op)
            
            H += SdotS
    
    return H


def build_ising_hamiltonian(L: int, j: float = 1.0, h: float = 1.0) -> np.ndarray:
    """Build transverse-field Ising Hamiltonian."""
    dim = 2 ** L
    H = np.zeros((dim, dim), dtype=np.float64)
    
    I = np.array([[1, 0], [0, 1]], dtype=np.float64)
    X = np.array([[0, 1], [1, 0]], dtype=np.float64)
    Z = np.array([[1, 0], [0, -1]], dtype=np.float64)
    
    for i in range(L - 1):
        ops = [I] * L
        ops[i] = Z
        ops[i + 1] = Z
        ZZ = ops[0]
        for op in ops[1:]:
            ZZ = np.kron(ZZ, op)
        H -= j * ZZ
    
    for i in range(L):
        ops = [I] * L
        ops[i] = X
        X_i = ops[0]
        for op in ops[1:]:
            X_i = np.kron(X_i, op)
        H -= h * X_i
    
    return H


def build_xxz_hamiltonian(L: int, delta: float, J: float = 1.0) -> np.ndarray:
    """
    Build XXZ Hamiltonian.
    H = J Σ_i [Sx_i Sx_{i+1} + Sy_i Sy_{i+1} + Δ Sz_i Sz_{i+1}]
    
    Δ > 1: Gapped (Ising-like)
    Δ = 1: Critical (Heisenberg)
    Δ < 1: Gapless (XY-like)
    """
    dim = 2 ** L
    H = np.zeros((dim, dim), dtype=np.complex128)
    
    I = np.array([[1, 0], [0, 1]], dtype=np.float64)
    Sx = 0.5 * np.array([[0, 1], [1, 0]], dtype=np.float64)
    Sy = 0.5 * np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    Sz = 0.5 * np.array([[1, 0], [0, -1]], dtype=np.float64)
    
    for i in range(L - 1):  # Open boundary
        # XX + YY term (isotropic)
        for S1, S2 in [(Sx, Sx), (Sy, Sy)]:
            ops = [I] * L
            ops[i] = S1
            ops[i + 1] = S2
            SdotS = ops[0]
            for op in ops[1:]:
                SdotS = np.kron(SdotS, op)
            H += J * SdotS
        
        # Δ·ZZ term
        ops = [I] * L
        ops[i] = Sz
        ops[i + 1] = Sz
        ZZ = ops[0]
        for op in ops[1:]:
            ZZ = np.kron(ZZ, op)
        H += delta * J * ZZ
    
    return H


def compute_entanglement_entropy(psi: np.ndarray, L: int, A_size: int) -> float:
    """Compute bipartite entanglement entropy S = -Tr(ρ_A log ρ_A)."""
    # Reshape into bipartite form
    psi = psi.reshape(2**A_size, 2**(L - A_size))
    
    # Schmidt decomposition
    u, s, vh = np.linalg.svd(psi, full_matrices=False)
    
    # Entropy from singular values
    s = s[s > 1e-12]  # Filter numerical zeros
    s = s / np.sum(s)  # Normalize
    entropy = -np.sum(s * np.log2(s + 1e-12))
    
    return float(entropy)


def compute_capacity_of_entanglement(psi: np.ndarray, L: int, A_size: int) -> float:
    """Compute capacity κ₂ = Var(ln ρ_A)."""
    psi = psi.reshape(2**A_size, 2**(L - A_size))
    u, s, vh = np.linalg.svd(psi, full_matrices=False)
    
    s = s[s > 1e-12]
    s = s / np.sum(s)
    
    # ln ρ_A eigenvalues are 2*ln(s)
    log_vals = 2 * np.log(s + 1e-12)
    mean_log = np.sum(s * log_vals)
    var_log = np.sum(s * (log_vals - mean_log)**2)
    
    return float(var_log)


def run_xxz_ed(L: int, delta: float) -> Dict:
    """Run ED for XXZ model and compute entanglement quantities."""
    H = build_xxz_hamiltonian(L, delta)
    
    # Diagonalize
    eigenvalues, eigenvectors = np.linalg.eigh(H)
    E0 = float(eigenvalues[0])
    psi0 = eigenvectors[:, 0]
    
    # Compute entanglement for half-chain
    A_size = L // 2
    S = compute_entanglement_entropy(psi0, L, A_size)
    kappa2 = compute_capacity_of_entanglement(psi0, L, A_size)
    
    return {
        "L": L,
        "delta": delta,
        "E0": E0,
        "S_entropy": S,
        "kappa2": kappa2,
        "A_size": A_size,
    }


def fit_log_scaling(L_values: List[int], S_values: List[float]) -> Dict:
    """Fit S = a*log(L) + b."""
    log_L = np.log(L_values)
    S = np.array(S_values)
    
    X = np.column_stack([log_L, np.ones_like(log_L)])
    coef, _, _, _ = np.linalg.lstsq(X, S, rcond=None)
    a, b = float(coef[0]), float(coef[1])
    
    yhat = X @ coef
    rss = float(np.sum((S - yhat)**2))
    
    n = len(L_values)
    aic = n * np.log(rss / n + 1e-12) + 2 * 2
    bic = n * np.log(rss / n + 1e-12) + 2 * np.log(n)
    
    return {"a": a, "b": b, "rss": rss, "aic": float(aic), "bic": float(bic)}


def fit_saturation(L_values: List[int], S_values: List[float]) -> Dict:
    """Fit S = S_inf - c*L^(-alpha)."""
    L = np.array(L_values, dtype=float)
    S = np.array(S_values, dtype=float)
    max_S = float(np.max(S))
    
    best = {"S_inf": float("nan"), "c": float("nan"), "alpha": float("nan"),
            "rss": float("inf"), "aic": float("inf"), "bic": float("inf")}
    
    for S_inf_mult in [1.0, 1.02, 1.05, 1.1, 1.15, 1.2, 1.3, 1.5]:
        S_inf = max_S * S_inf_mult + 0.01
        delta = S_inf - S
        valid = delta > 1e-12
        if np.sum(valid) < 3:
            continue
        
        log_d = np.log(delta[valid])
        log_L_valid = np.log(L[valid])
        
        X = np.column_stack([-log_L_valid, np.ones_like(log_L_valid)])
        coef, _, _, _ = np.linalg.lstsq(X, log_d, rcond=None)
        alpha, log_c0 = float(coef[0]), float(coef[1])
        c = np.exp(log_c0)
        
        yhat = S_inf - c * np.power(L, -alpha)
        rss = float(np.sum((S - yhat)**2))
        
        n = len(L_values)
        aic = n * np.log(rss / n + 1e-12) + 2 * 3
        bic = n * np.log(rss / n + 1e-12) + 3 * np.log(n)
        
        if rss < best["rss"]:
            best = {"S_inf": S_inf, "c": c, "alpha": alpha, 
                    "rss": rss, "aic": float(aic), "bic": float(bic)}
    
    return best


def run_xxz_boundary_test(cfg: Dict) -> Dict:
    """Run XXZ boundary test across Δ values."""
    print("=" * 60)
    print("XXZ BOUNDARY TEST (ED Only)")
    print("=" * 60)
    print(f"L values: {cfg['L_values']}")
    print(f"Δ values: {cfg['deltas']}")
    print()
    
    results = []
    
    for delta in cfg["deltas"]:
        print(f"\n[Δ = {delta}]")
        expected = "IN_SCOPE" if delta > 1 else "OUT_OF_SCOPE"
        print(f"  Expected scope: {expected} ({'gapped' if delta > 1 else 'critical'})")
        
        # Run ED for each L
        measurements = []
        for L in cfg["L_values"]:
            print(f"  L={L}...", end=" ")
            result = run_xxz_ed(L, delta)
            measurements.append(result)
            print(f"S={result['S_entropy']:.4f}, κ₂={result['kappa2']:.4f}")
        
        # Extract S values
        L_values = [m["L"] for m in measurements]
        S_values = [m["S_entropy"] for m in measurements]
        
        # Fit models
        log_model = fit_log_scaling(L_values, S_values)
        sat_model = fit_saturation(L_values, S_values)
        delta_aic = sat_model["aic"] - log_model["aic"]
        
        # Verdict
        # Δ > 1 (gapped): expect saturation (ΔAIC > 0)
        # Δ ≤ 1 (critical): expect log growth (ΔAIC < 0)
        expected_sign = "positive" if delta > 1 else "negative"
        actual_sign = "positive" if delta_aic > 0 else "negative"
        scope_correct = (expected_sign == actual_sign)
        
        if delta > 1:
            verdict = "ACCEPT" if delta_aic > 0 else "REJECT"
        else:
            verdict = "REJECT" if delta_aic < 0 else "ACCEPT"
        
        print(f"  ΔAIC = {delta_aic:.4f} ({'saturation' if delta_aic > 0 else 'log growth'})")
        print(f"  Verdict: {verdict}, Scope correct: {scope_correct}")
        
        results.append({
            "delta": delta,
            "expected_scope": expected,
            "measurements": measurements,
            "log_model": log_model,
            "sat_model": sat_model,
            "delta_aic": float(delta_aic),
            "verdict": verdict,
            "scope_correct": scope_correct,
        })
    
    # Summary
    scope_matches = sum(1 for r in results if r["scope_correct"])
    total = len(results)
    all_correct = scope_matches == total
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Δ':>6} | {'Expected':>12} | {'ΔAIC':>8} | {'Verdict':>8} | {'Scope OK':>8}")
    print("-" * 60)
    for r in results:
        print(f"{r['delta']:>6.2f} | {r['expected_scope']:>12} | {r['delta_aic']:>8.4f} | {r['verdict']:>8} | {str(r['scope_correct']):>8}")
    print("-" * 60)
    print(f"Scope matches: {scope_matches}/{total}")
    print(f"Overall: {'SCOPE_VALIDATED' if all_correct else 'SCOPE_MISMATCH'}")
    
    return {
        "metadata": {
            "test": "P2_XXZ_BOUNDARY_ED",
            "version": "1.0.0",
            "method": "exact_diagonalization",
        },
        "config": cfg,
        "results": results,
        "summary": {
            "scope_matches": scope_matches,
            "total": total,
            "all_correct": all_correct,
            "overall_verdict": "SCOPE_VALIDATED" if all_correct else "SCOPE_MISMATCH",
        }
    }


def main():
    p = argparse.ArgumentParser(description="P2 XXZ Boundary Test (ED Only)")
    p.add_argument("--deltas", default="0.5,1.0,1.1,1.5,2.0",
                   help="Comma-separated Δ values")
    p.add_argument("--L_values", default="4,6,8,10",
                   help="Comma-separated L values")
    p.add_argument("--output", default=None, help="Output directory (optional)")
    a = p.parse_args()
    
    cfg = {
        "deltas": [float(x) for x in a.deltas.split(",")],
        "L_values": [int(x) for x in a.L_values.split(",")],
    }
    
    res = run_xxz_boundary_test(cfg)
    
    if a.output:
        out_dir = Path(a.output)
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "xxz_boundary_results.json", "w") as f:
            json.dump(res, f, indent=2)
        print(f"\nResults written to {out_dir}")


if __name__ == "__main__":
    main()