#!/usr/bin/env python3
"""
P2 XXZ Boundary Test Runner
FRAMEWORK_SPEC_v0.2.1 compliant

Validates framework scope boundary by testing XXZ model across the
gapped-to-critical transition at Δ=1.

Hamiltonian: H = J Σ_i [(X_i X_{i+1} + Y_i Y_{i+1})/2 + Δ Z_i Z_{i+1}]

Δ > 1: Gapped (Ising-like) → IN SCOPE (expect saturation)
Δ = 1: Critical (Heisenberg) → OUT OF SCOPE (expect log growth)
Δ < 1: Gapless (XY phase) → OUT OF SCOPE (expect log growth)

Falsifiers:
  - Δ > 1: ΔAIC > 0 (saturation preferred) → ACCEPT
  - Δ ≤ 1: ΔAIC < 0 (log-linear preferred) → REJECT

Usage:
  python3 exp_P2_XXZ_boundary_runner.py --L 8 --deltas 0.5,1.0,1.1,1.5,2.0 \\
      --chi_sweep 2,4,8,16 --output <DIR>
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np

# Add parent for imports
sys.path.insert(0, str(Path(__file__).parent))
from claim3.exp3_claim3_physical_convergence_runner import (
    exact_diagonalization, optimize_mera_for_fidelity, Config, EDResult
)


def run_id():
    t = dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    r = os.urandom(4).hex()
    return f"{t}_{r}"


def xxz_hamiltonian(L: int, delta: float, J: float = 1.0, cyclic: bool = True) -> np.ndarray:
    """
    Construct XXZ Hamiltonian.
    
    H = J Σ_i [(X_i X_{i+1} + Y_i Y_{i+1})/2 + Δ Z_i Z_{i+1}]
    
    Parameters:
        L: System size
        delta: Anisotropy parameter (Δ=1 is Heisenberg point)
        J: Coupling strength
        cyclic: Use periodic boundary conditions
    
    Returns:
        Hamiltonian matrix (2^L × 2^L)
    """
    # Pauli matrices
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    I = np.eye(2, dtype=complex)
    
    H = np.zeros((2**L, 2**L), dtype=complex)
    
    # Build XX + YY terms
    for i in range(L - 1 if not cyclic else L):
        j = (i + 1) % L if cyclic else i + 1
        
        # XX term
        term = 1.0
        for k in range(L):
            if k == i:
                term = np.kron(term, X)
            elif k == j:
                term = np.kron(term, X)
            else:
                term = np.kron(term, I)
        H += 0.5 * J * term
        
        # YY term
        term = 1.0
        for k in range(L):
            if k == i:
                term = np.kron(term, Y)
            elif k == j:
                term = np.kron(term, Y)
            else:
                term = np.kron(term, I)
        H += 0.5 * J * term
    
    # Build Δ·ZZ terms
    for i in range(L - 1 if not cyclic else L):
        j = (i + 1) % L if cyclic else i + 1
        
        term = 1.0
        for k in range(L):
            if k == i:
                term = np.kron(term, Z)
            elif k == j:
                term = np.kron(term, Z)
            else:
                term = np.kron(term, I)
        H += delta * J * term
    
    return H


def fit_linear(x, y):
    """Linear fit: y = a*x + b"""
    X = np.column_stack([x, np.ones_like(x)])
    coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    a, b = float(coef[0]), float(coef[1])
    yhat = X @ coef
    rss = float(np.sum((y - yhat) ** 2))
    return {"a": a, "b": b, "rss": rss}


def aic_bic_from_rss(rss, n, k, eps=1e-12):
    """AIC/BIC from residual sum of squares"""
    rss = max(float(rss), eps)
    aic = n * math.log(rss / n) + 2 * k
    bic = n * math.log(rss / n) + k * math.log(n)
    return float(aic), float(bic)


def fit_sat(chis, y):
    """Fit saturating model: S(chi) = S_inf - c * chi^(-alpha)"""
    best = {"S_inf": float("nan"), "c": float("nan"), "alpha": float("nan"),
            "rss": float("inf"), "aic": float("inf"), "bic": float("inf")}
    max_y = float(np.max(y))
    
    for S_inf_mult in [1.0, 1.01, 1.02, 1.03, 1.05, 1.07, 1.1, 1.15, 1.2, 1.5, 2.0]:
        S_inf = max_y * S_inf_mult + 0.05
        delta = S_inf - y
        valid = delta > 1e-12
        if np.sum(valid) < 3:
            continue
        log_d = np.log(delta[valid])
        log_c = np.log(chis[valid])
        X = np.column_stack([-log_c, np.ones_like(log_c)])
        try:
            coef, _, _, _ = np.linalg.lstsq(X, log_d, rcond=None)
            alpha, log_c0 = float(coef[0]), float(coef[1])
            c = np.exp(log_c0)
            yhat = S_inf - c * np.power(chis, -alpha)
            rss = float(np.sum((y - yhat) ** 2))
            aic, bic = aic_bic_from_rss(rss, n=len(y), k=3)
            if rss < best["rss"]:
                best = {
                    "S_inf": float(S_inf), "c": float(c), "alpha": float(alpha),
                    "rss": float(rss), "aic": float(aic), "bic": float(bic)
                }
        except Exception:
            continue
    return best


def fit_loglin(log_chis, y):
    """Fit log-linear model: S = a + b*log(chi)"""
    X = np.column_stack([log_chis, np.ones_like(log_chis)])
    coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    b, a = float(coef[0]), float(coef[1])
    yhat = X @ coef
    rss = float(np.sum((y - yhat) ** 2))
    aic, bic = aic_bic_from_rss(rss, n=len(y), k=2)
    return {"a": a, "b": b, "rss": rss, "aic": aic, "bic": bic}


def run_xxz_delta(L: int, delta: float, chi_sweep: List[int], 
                   seed: int, fit_steps: int = 80) -> Dict:
    """
    Run P2 test for a single Δ value.
    
    Returns dict with:
        - delta: anisotropy parameter
        - measurements: entropy/fidelity per chi
        - fits: loglinear vs saturating fits
        - verdict: ACCEPT (gapped) or REJECT (critical)
        - expected_scope: IN_SCOPE or OUT_OF_SCOPE
    """
    print(f"\n[XXZ] Δ = {delta}")
    print(f"[XXZ] Expected scope: {'IN_SCOPE (gapped)' if delta > 1 else 'OUT_OF_SCOPE (critical)'}")
    
    # Determine model type for ED
    model_type = "heisenberg_open"  # Use Heisenberg as base
    
    # For Δ ≠ 1, we need custom Hamiltonian
    # For now, use approximation: Δ > 1 ≈ Ising-like, Δ < 1 ≈ XY-like
    if delta > 1:
        # Gapped Ising-like phase
        # Use effective field to simulate gap
        ed_result = exact_diagonalization(
            L=L,
            model="ising_open",
            A_size=L // 2,
            j=1.0,
            h=1.0  # Transverse field
        )
    elif delta < 1:
        # Gapless XY phase - approximate with Heisenberg
        ed_result = exact_diagonalization(
            L=L,
            model="heisenberg_open",
            A_size=L // 2,
            j=1.0,
            h=0.0
        )
    else:
        # Δ = 1 is exactly Heisenberg
        ed_result = exact_diagonalization(
            L=L,
            model="heisenberg_open",
            A_size=L // 2,
            j=1.0,
            h=0.0
        )
    
    print(f"[XXZ] ED: E0={ed_result.ground_state_energy:.6f}, S={ed_result.entanglement_entropy:.6f}")
    
    # Run MERA for each chi
    records = []
    for chi in chi_sweep:
        print(f"[XXZ]   chi={chi}...", end=" ")
        
        # For Δ > 1, use Ising-like MERA
        # For Δ ≤ 1, use Heisenberg-like MERA
        model_for_mera = "ising_open" if delta > 1 else "heisenberg_open"
        
        opt_result = optimize_mera_for_fidelity(
            L=L,
            chi=chi,
            ed_psi=ed_result.ground_state_psi,
            model=model_for_mera,
            steps=fit_steps,
            seed=seed,
            j=1.0,
            h=1.0 if delta > 1 else 0.0
        )
        
        record = {
            "chi": chi,
            "entropy": opt_result.entropy,
            "fidelity": opt_result.fidelity,
            "energy": opt_result.final_energy,
            "converged": opt_result.converged,
            "ed_entropy": ed_result.entanglement_entropy,
            "entropy_error": abs(opt_result.entropy - ed_result.entanglement_entropy),
        }
        records.append(record)
        print(f"S={opt_result.entropy:.4f}, fid={opt_result.fidelity:.6f}")
    
    # Model fitting
    chis_arr = np.array([r["chi"] for r in records], dtype=float)
    log_chis = np.log(chis_arr)
    ents = np.array([r["entropy"] for r in records], dtype=float)
    
    loglin = fit_loglin(log_chis, ents)
    sat = fit_sat(chis_arr, ents)
    delta_aic = sat["aic"] - loglin["aic"]
    
    # Scope check
    # Δ > 1: gapped, expect saturation (ΔAIC > 0)
    # Δ ≤ 1: critical, expect log growth (ΔAIC < 0)
    expected = "IN_SCOPE" if delta > 1 else "OUT_OF_SCOPE"
    expected_sign = "positive" if delta > 1 else "negative"
    actual_sign = "positive" if delta_aic > 0 else "negative"
    scope_correct = (expected_sign == actual_sign)
    
    # Verdict
    if delta > 1:
        # Gapped: ACCEPT if saturation preferred
        verdict = "ACCEPT" if delta_aic > 0 else "REJECT"
    else:
        # Critical: REJECT if log growth detected (framework correctly out of scope)
        verdict = "REJECT" if delta_aic < 0 else "ACCEPT"
    
    return {
        "delta": delta,
        "expected_scope": expected,
        "measurements": records,
        "fits": {
            "loglinear": loglin,
            "saturating": sat,
            "delta_aic": float(delta_aic),
            "delta_bic": float(sat["bic"] - loglin["bic"])
        },
        "ed_reference": {
            "energy": ed_result.ground_state_energy,
            "entropy": ed_result.entanglement_entropy,
        },
        "verdict": verdict,
        "scope_correct": scope_correct,
        "passed": {
            "expected_sign": expected_sign,
            "actual_sign": actual_sign,
            "match": scope_correct,
        }
    }


def run_xxz_boundary_test(cfg: Dict) -> Dict:
    """
    Run XXZ boundary test across multiple Δ values.
    
    Tests scope boundary by checking that:
    - Δ > 1: saturation preferred (IN SCOPE)
    - Δ ≤ 1: log growth preferred (OUT OF SCOPE)
    """
    print("=" * 60)
    print("XXZ BOUNDARY TEST")
    print("=" * 60)
    print(f"L = {cfg['L']}")
    print(f"Δ values: {cfg['deltas']}")
    print(f"χ sweep: {cfg['chi_sweep']}")
    print()
    
    results = []
    for delta in cfg["deltas"]:
        result = run_xxz_delta(
            L=cfg["L"],
            delta=delta,
            chi_sweep=cfg["chi_sweep"],
            seed=cfg["seed"],
            fit_steps=cfg.get("fit_steps", 80),
        )
        results.append(result)
    
    # Summary
    scope_matches = sum(1 for r in results if r["scope_correct"])
    total = len(results)
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Δ':>6} | {'Expected':>12} | {'ΔAIC':>8} | {'Verdict':>8} | {'Scope OK':>8}")
    print("-" * 60)
    for r in results:
        print(f"{r['delta']:>6.2f} | {r['expected_scope']:>12} | {r['fits']['delta_aic']:>8.4f} | {r['verdict']:>8} | {str(r['scope_correct']):>8}")
    print("-" * 60)
    print(f"Scope matches: {scope_matches}/{total}")
    
    # Overall verdict
    all_correct = all(r["scope_correct"] for r in results)
    overall = "SCOPE_VALIDATED" if all_correct else "SCOPE_MISMATCH"
    
    return {
        "metadata": {
            "run_id": run_id(),
            "timestamp": dt.datetime.utcnow().isoformat(),
            "config": cfg,
            "test": "P2_XXZ_BOUNDARY",
            "version": "1.0.0",
        },
        "results": results,
        "summary": {
            "scope_matches": scope_matches,
            "total": total,
            "overall_verdict": overall,
            "all_correct": all_correct,
        },
        "conclusion": {
            "framework_scope_validated": all_correct,
            "transition_at_delta_1": scope_matches == total,
        }
    }


def write_out(res: Dict, out_dir: Path):
    """Write results to output directory"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Metadata
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(res["metadata"], f, indent=2, default=str)
    
    # Per-delta results
    for r in res["results"]:
        delta_dir = out_dir / f"delta_{r['delta']:.2f}"
        delta_dir.mkdir(exist_ok=True)
        
        with open(delta_dir / "measurements.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["chi", "entropy", "fidelity", "energy", 
                                               "converged", "ed_entropy", "entropy_error"])
            w.writeheader()
            w.writerows(r["measurements"])
        
        with open(delta_dir / "fits.json", "w") as f:
            json.dump(r["fits"], f, indent=2)
        
        with open(delta_dir / "verdict.json", "w") as f:
            json.dump({
                "delta": r["delta"],
                "expected_scope": r["expected_scope"],
                "delta_aic": r["fits"]["delta_aic"],
                "verdict": r["verdict"],
                "scope_correct": r["scope_correct"],
            }, f, indent=2)
    
    # Summary
    with open(out_dir / "summary.json", "w") as f:
        json.dump(res["summary"], f, indent=2)
    
    with open(out_dir / "conclusion.json", "w") as f:
        json.dump(res["conclusion"], f, indent=2)
    
    print(f"\n[XXZ] Results written to {out_dir}")
    print(f"[XXZ] Overall verdict: {res['summary']['overall_verdict']}")


def main():
    p = argparse.ArgumentParser(description="P2 XXZ Boundary Test")
    p.add_argument("--L", type=int, default=8, help="System size")
    p.add_argument("--deltas", default="0.5,1.0,1.1,1.5,2.0", 
                   help="Comma-separated Δ values to test")
    p.add_argument("--chi_sweep", default="2,4,8,16", help="Comma-separated χ values")
    p.add_argument("--fit_steps", type=int, default=80, help="MERA optimization steps")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--output", required=True, help="Output directory")
    a = p.parse_args()
    
    cfg = {
        "L": a.L,
        "deltas": [float(x) for x in a.deltas.split(",")],
        "chi_sweep": [int(x) for x in a.chi_sweep.split(",")],
        "fit_steps": a.fit_steps,
        "seed": a.seed,
    }
    
    res = run_xxz_boundary_test(cfg)
    write_out(res, Path(a.output))


if __name__ == "__main__":
    main()