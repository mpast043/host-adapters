#!/usr/bin/env python3
"""
Claim 3P Physical Convergence Runner
FRAMEWORK_SPEC_v0.2.1 compliant

Phase 1: ED Gold Standard Test (L=8, L=12)
- Exact diagonalization for Ising and Heisenberg models
- MERA variational optimization to maximize fidelity to ED ground state
- Acceptance criteria: P3.1-P3.4

Usage:
  python3 exp3_claim3_physical_convergence_runner.py --L 12 --A_size 6 \\
    --model ising_open --j 1.0 --h 1.0 --chi_sweep 2,3,4,6,8,12,16 \\
    --restarts_per_chi 3 --fit_steps 120 --seed 42 --output <RUN_DIR>
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np


# ============================================================
# Configuration and Types
# ============================================================

@dataclass
class Config:
    L: int
    A_size: int
    model: str  # 'ising_open' or 'heisenberg_open'
    chi_sweep: List[int]
    restarts_per_chi: int
    fit_steps: int
    seed: int
    output_dir: Path
    # Ising parameters
    j: float = 1.0
    h: float = 1.0
    # Thresholds
    eps_fid: float = 1e-4
    eps_S: float = 1e-3
    

@dataclass
class EDResult:
    """Exact diagonalization reference data"""
    ground_state_energy: float
    ground_state_psi: np.ndarray
    entanglement_entropy: float
    n_sites: int
    

@dataclass  
class OptimizationResult:
    """Result from one MERA optimization run"""
    chi: int
    restart_idx: int
    fidelity: float
    entropy: float
    final_energy: float
    converged: bool
    num_steps: int
    seed: int


# ============================================================
# Utility Functions
# ============================================================

def make_run_id() -> str:
    t = dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    r = os.urandom(4).hex()
    return f"{t}_{r}"


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


def compute_entanglement_entropy(psi: np.ndarray, L: int, A_size: int) -> float:
    """
    Compute von Neumann entropy of reduced density matrix for contiguous partition.
    psi: full wavefunction (2^L,)
    A_size: number of sites in partition A (sites 0 to A_size-1)
    """
    # Reshape to (2^A, 2^(L-A))
    dim_A = 2 ** A_size
    dim_B = 2 ** (L - A_size)
    psi_matrix = psi.reshape(dim_A, dim_B)
    
    # Reduced density matrix rho_A = Tr_B(|psi><psi|)
    # SVD gives singular values which are sqrt(eigenvalues of rho_A)
    U, s, Vh = np.linalg.svd(psi_matrix, full_matrices=False)
    
    # Eigenvalues of rho_A are s^2
    eigvals = s ** 2
    eigvals = eigvals[eigvals > 1e-12]  # numerical cutoff
    
    # von Neumann entropy
    S = -np.sum(eigvals * np.log(eigvals))
    return float(S)


def fidelity(psi1: np.ndarray, psi2: np.ndarray) -> float:
    """Compute |<psi1|psi2>|^2"""
    overlap = np.vdot(psi1, psi2)
    return float(np.abs(overlap) ** 2)


# ============================================================
# Exact Diagonalization
# ============================================================

def build_ising_hamiltonian(L: int, j: float = 1.0, h: float = 1.0) -> np.ndarray:
    """
    Build Ising Hamiltonian: H = -J * sum_i Z_i Z_{i+1} - h * sum_i X_i
    Open boundary conditions.
    """
    dim = 2 ** L
    H = np.zeros((dim, dim), dtype=np.float64)
    
    # Pauli matrices
    I = np.array([[1, 0], [0, 1]], dtype=np.float64)
    X = np.array([[0, 1], [1, 0]], dtype=np.float64)
    Z = np.array([[1, 0], [0, -1]], dtype=np.float64)
    
    for i in range(L - 1):  # Z-Z interaction
        # Build operator: I^(i-1) @ Z @ Z @ I^(L-i-2)
        ops = [I] * L
        ops[i] = Z
        ops[i + 1] = Z
        
        ZZ = ops[0]
        for op in ops[1:]:
            ZZ = np.kron(ZZ, op)
        H -= j * ZZ
    
    for i in range(L):  # X field
        ops = [I] * L
        ops[i] = X
        
        X_i = ops[0]
        for op in ops[1:]:
            X_i = np.kron(X_i, op)
        H -= h * X_i
    
    return H


def build_heisenberg_hamiltonian(L: int) -> np.ndarray:
    """
    Build Heisenberg Hamiltonian: H = sum_i J * (S_i . S_{i+1})
    With J=1 and open boundary conditions.
    """
    dim = 2 ** L
    H = np.zeros((dim, dim), dtype=np.float64)
    
    # Pauli spin matrices (S = 0.5 * sigma)
    I = np.array([[1, 0], [0, 1]], dtype=np.float64)
    Sx = 0.5 * np.array([[0, 1], [1, 0]], dtype=np.float64)
    Sy = 0.5 * np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    Sz = 0.5 * np.array([[1, 0], [0, -1]], dtype=np.float64)
    
    def spin_op(ops: List[np.ndarray], site: int) -> np.ndarray:
        """Build product operator with op at site and I elsewhere"""
        full_ops = [I] * L
        full_ops[site] = ops[0]
        if len(ops) > 1:
            full_ops[site + 1] = ops[1] if site + 1 < L else I
        
        result = full_ops[0]
        for op in full_ops[1:]:
            result = np.kron(result, op)
        return result
    
    for i in range(L - 1):
        # S_i . S_{i+1} = Sx_i Sx_{i+1} + Sy_i Sy_{i+1} + Sz_i Sz_{i+1}
        # Build each term
        for S1, S2 in [(Sx, Sx), (Sy, Sy), (Sz, Sz)]:
            ops = [I] * L
            ops[i] = S1
            ops[i + 1] = S2
            
            SdotS = ops[0]
            for op in ops[2:]:
                SdotS = np.kron(SdotS, op)
            
            # For mixed real/complex, promote to complex
            if np.iscomplexobj(S1) or np.iscomplexobj(S2):
                H = H.astype(np.complex128)
                SdotS = SdotS.astype(np.complex128)
            
            H += SdotS
    
    return H


def exact_diagonalization(L: int, model: str, A_size: int, 
                          j: float = 1.0, h: float = 1.0) -> EDResult:
    """
    Perform exact diagonalization for small systems.
    Returns ground state and entanglement entropy.
    """
    print(f"  [ED] Building {model} Hamiltonian for L={L}...")
    
    if model == "ising_open":
        H = build_ising_hamiltonian(L, j, h)
    elif model == "heisenberg_open":
        H = build_heisenberg_hamiltonian(L)
    else:
        raise ValueError(f"Unknown model: {model}")
    
    print(f"  [ED] Diagonalizing {H.shape[0]}x{H.shape[0]} matrix...")
    
    # Get ground state
    if np.iscomplexobj(H):
        # Use eigh for Hermitian
        eigenvalues, eigenvectors = np.linalg.eigh(H.astype(np.complex128))
    else:
        eigenvalues, eigenvectors = np.linalg.eigh(H)
    
    idx = np.argmin(eigenvalues)
    E0 = float(eigenvalues[idx])
    psi0 = eigenvectors[:, idx]
    
    print(f"  [ED] Ground state energy: {E0:.6f}")
    
    # Compute entanglement entropy
    S = compute_entanglement_entropy(psi0, L, A_size)
    print(f"  [ED] Entanglement entropy S={S:.6f}")
    
    return EDResult(
        ground_state_energy=E0,
        ground_state_psi=psi0,
        entanglement_entropy=S,
        n_sites=L
    )


# ============================================================
# MERA Optimization (simplified)
# ============================================================

def optimize_mera_for_fidelity(L: int, chi: int, ed_psi: np.ndarray, 
                                 model: str, steps: int, seed: int,
                                 j: float = 1.0, h: float = 1.0) -> OptimizationResult:
    """
    MERA variational optimization to minimize energy and maximize fidelity to ED ground state.
    Uses quimb's TNOptimizer for proper gradient-based optimization.
    """
    import quimb.tensor as qtn
    from quimb.tensor.optimize import TNOptimizer
    import torch
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Build Hamiltonian as MPO
    if model == "ising_open":
        H_mpo = qtn.MPO_ham_ising(L, j=j, bx=h, cyclic=False)
        # Also get dense H for fidelity check
        H_dense = build_ising_hamiltonian(L, j, h)
    else:  # heisenberg
        H_mpo = qtn.MPO_ham_heis(L, cyclic=False)
        H_dense = build_heisenberg_hamiltonian(L)
        H_dense = H_dense.astype(np.float64).real
    
    # Initialize random MERA
    mera = qtn.MERA.rand(L, max_bond=chi, dtype="float64")
    mera.isometrize_()
    
    # Normalize target state
    ed_psi = ed_psi / np.linalg.norm(ed_psi)
    
    # Use negative fidelity as loss function (TNOptimizer minimizes)
    def neg_fidelity(mera_tn):
        """Negative fidelity for minimization"""
        psi_mera = mera_tn.to_dense().reshape(-1)
        # Normalize
        norm = np.linalg.norm(psi_mera)
        if norm < 1e-12:
            return 1.0  # Bad fidelity
        psi_mera = psi_mera / norm
        
        # Compute overlap (can use numpy for small systems)
        overlap = np.vdot(ed_psi, psi_mera)
        return -np.abs(overlap) ** 2  # Negative because we minimize
    
    # Optimize with TNOptimizer
    opt = TNOptimizer(
        mera,
        neg_fidelity,
        loss_target=-0.999,  # Stop when fidelity > 0.999
        optimizer="L-BFGS-B",
        autodiff_backend="torch",
        progbar=False,
    )
    
    try:
        mera_opt = opt.optimize(maxiter=steps)
        converged = True
    except Exception as e:
        # If optimization fails, use last state
        mera_opt = mera
        converged = False
    
    # Get optimized state as dense vector
    psi_mera = mera_opt.to_dense().reshape(-1)
    psi_mera = psi_mera / np.linalg.norm(psi_mera)
    
    # Compute observables
    fid = fidelity(psi_mera, ed_psi)
    entropy = compute_entanglement_entropy(psi_mera, L, L // 2)
    energy = float((psi_mera.conj() @ H_dense @ psi_mera).real)
    
    return OptimizationResult(
        chi=chi,
        restart_idx=0,
        fidelity=fid,
        entropy=entropy,
        final_energy=energy,
        converged=converged,
        num_steps=steps,
        seed=seed
    )


def run_mera_with_restarts(L: int, chi: int, ed_psi: np.ndarray, 
                           ed_result: EDResult, model: str, steps: int,
                           restarts: int, seed_base: int, j: float = 1.0, 
                           h: float = 1.0) -> List[OptimizationResult]:
    """
    Run MERA optimization with multiple restarts, return best results.
    """
    results = []
    
    for restart in range(restarts):
        seed = seed_base + restart * 1000 + chi * 10000
        print(f"    [MERA] chi={chi}, restart={restart+1}/{restarts}, seed={seed}")
        
        result = optimize_mera_for_fidelity(
            L=L, chi=chi, ed_psi=ed_psi, model=model, 
            steps=steps, seed=seed, j=j, h=h
        )
        result.restart_idx = restart
        results.append(result)
        print(f"      fidelity={result.fidelity:.6f}, S={result.entropy:.4f}")
    
    return results


# ============================================================
# Falsifiers (P3.1-P3.4)
# ============================================================

def falsifier_p31_nondecreasing_fidelity(best_fidelities: List[float], 
                                         chi_values: List[int],
                                         eps_fid: float = 1e-4) -> Dict[str, Any]:
    """
    P3.1: Best-of-restarts fidelity is nondecreasing in chi within epsilon.
    """
    violations = []
    for i in range(len(best_fidelities) - 1):
        if best_fidelities[i+1] < best_fidelities[i] - eps_fid:
            violations.append({
                "chi_i": chi_values[i],
                "chi_i+1": chi_values[i+1],
                "fid_i": best_fidelities[i],
                "fid_i+1": best_fidelities[i+1],
                "drop": best_fidelities[i] - best_fidelities[i+1]
            })
    
    return {
        "passed": len(violations) == 0,
        "violations": violations,
        "n_checks": len(best_fidelities) - 1,
        "eps_fid": eps_fid
    }


def falsifier_p32_entropy_convergence(S_best: List[float], S_ref: float,
                                       chi_values: List[int],
                                       eps_S: float = 1e-3) -> Dict[str, Any]:
    """
    P3.2: Entropy error |S_best(chi) - S_ref| is nonincreasing in chi.
    """
    errors = [abs(s - S_ref) for s in S_best]
    violations = []
    
    for i in range(len(errors) - 1):
        if errors[i+1] > errors[i] + eps_S:
            violations.append({
                "chi_i": chi_values[i],
                "chi_i+1": chi_values[i+1],
                "error_i": errors[i],
                "error_i+1": errors[i+1],
                "increase": errors[i+1] - errors[i]
            })
    
    return {
        "passed": len(violations) == 0,
        "errors": errors,
        "violations": violations,
        "n_checks": len(errors) - 1,
        "eps_S": eps_S
    }


def falsifier_p33_final_thresholds(best_fidelity: float, final_entropy_error: float,
                                   L: int, model: str) -> Dict[str, Any]:
    """
    P3.3: Final thresholds based on system size.
    """
    if L == 8:
        fid_threshold = 0.95
    elif L == 12:
        fid_threshold = 0.90
    else:
        fid_threshold = 0.85  # fallback
    
    S_threshold = 0.15  # nats
    
    fid_passed = best_fidelity >= fid_threshold
    S_passed = final_entropy_error <= S_threshold
    
    return {
        "passed": fid_passed and S_passed,
        "fid_passed": fid_passed,
        "S_passed": S_passed,
        "best_fidelity": best_fidelity,
        "final_entropy_error": final_entropy_error,
        "fid_threshold": fid_threshold,
        "S_threshold": S_threshold,
        "L": L,
        "model": model
    }


def fit_saturating_model(chi_values: List[int], S_values: List[float], 
                        S_ref: float) -> Tuple[float, float, float]:
    """
    Fit S(chi) = S_inf * chi / (chi + c) (saturating exponential-like).
    Returns (S_inf, c, rss).
    """
    chi_arr = np.array(chi_values, dtype=float)
    S_arr = np.array(S_values, dtype=float)
    
    # Initial guess
    S_inf_guess = S_ref
    c_guess = np.mean(chi_arr)
    
    best_rss = float('inf')
    best_params = (S_inf_guess, c_guess)
    
    # Grid search for robustness
    for S_inf in np.linspace(S_ref * 0.5, S_ref * 1.5, 50):
        for c in np.linspace(1, max(chi_arr) * 2, 50):
            S_pred = S_inf * chi_arr / (chi_arr + c)
            rss = np.sum((S_arr - S_pred) ** 2)
            if rss < best_rss:
                best_rss = rss
                best_params = (S_inf, c)
    
    return best_params[0], best_params[1], best_rss


def fit_log_linear_model(chi_values: List[int], S_values: List[float]) -> Tuple[float, float, float]:
    """
    Fit S(chi) = a * log(chi) + b.
    Returns (a, b, rss).
    """
    chi_arr = np.array(chi_values, dtype=float)
    S_arr = np.array(S_values, dtype=float)
    
    log_chi = np.log(chi_arr)
    
    # Linear regression
    X = np.column_stack([log_chi, np.ones(len(log_chi))])
    coef, _, _, _ = np.linalg.lstsq(X, S_arr, rcond=None)
    a, b = coef[0], coef[1]
    
    S_pred = a * log_chi + b
    rss = np.sum((S_arr - S_pred) ** 2)
    
    return a, b, rss


def aic_bic_from_rss(rss: float, n: int, k: int) -> Tuple[float, float]:
    """Compute AIC/BIC from RSS."""
    rss = max(float(rss), 1e-12)
    aic = n * math.log(rss / n) + 2 * k
    bic = n * math.log(rss / n) + k * math.log(n)
    return float(aic), float(bic)


def falsifier_p34_model_selection(chi_values: List[int], S_best: List[float],
                                   S_ref: float) -> Dict[str, Any]:
    """
    P3.4: Saturating model must win over log-linear with Delta AIC >= 10 and Delta BIC >= 10.
    """
    # Saturing model fit
    S_inf_sat, c_sat, rss_sat = fit_saturating_model(chi_values, S_best, S_ref)
    n = len(chi_values)
    aic_sat, bic_sat = aic_bic_from_rss(rss_sat, n, k=2)  # 2 params: S_inf, c
    
    # Log-linear model fit
    a_log, b_log, rss_log = fit_log_linear_model(chi_values, S_best)
    aic_log, bic_log = aic_bic_from_rss(rss_log, n, k=2)  # 2 params: a, b
    
    delta_aic = aic_log - aic_sat  # positive means saturating wins
    delta_bic = bic_log - bic_sat
    
    saturating_wins = delta_aic >= 10 and delta_bic >= 10
    
    return {
        "passed": saturating_wins,
        "saturating_wins": saturating_wins,
        "delta_aic": delta_aic,
        "delta_bic": delta_bic,
        "S_inf_sat": S_inf_sat,
        "c_sat": c_sat,
        "rss_sat": rss_sat,
        "aic_sat": aic_sat,
        "bic_sat": bic_sat,
        "a_log": a_log,
        "b_log": b_log,
        "rss_log": rss_log,
        "aic_log": aic_log,
        "bic_log": bic_log,
        "n_points": n
    }


# ============================================================
# Main Runner
# ============================================================

def main():
    ap = argparse.ArgumentParser(description="Claim 3P: Physical Hamiltonian Convergence Test")
    ap.add_argument("--L", type=int, required=True, help="System size (8 or 12)")
    ap.add_argument("--A_size", type=int, required=True, help="Partition size (L/2)")
    ap.add_argument("--model", choices=["ising_open", "heisenberg_open"], required=True)
    ap.add_argument("--j", type=float, default=1.0, help="Ising coupling (for ising)")
    ap.add_argument("--h", type=float, default=1.0, help="Ising field (for ising)")
    ap.add_argument("--chi_sweep", default="2,3,4,6,8,12,16", help="Comma-separated chi values")
    ap.add_argument("--restarts_per_chi", type=int, default=3)
    ap.add_argument("--fit_steps", type=int, default=120, help="Optimization steps")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", type=Path, required=True, help="Output directory")
    ap.add_argument("--eps_fid", type=float, default=1e-4)
    ap.add_argument("--eps_S", type=float, default=1e-3)
    
    args = ap.parse_args()
    
    # Parse chi sweep
    chi_sweep = [int(x.strip()) for x in args.chi_sweep.split(",") if x.strip()]
    
    config = Config(
        L=args.L,
        A_size=args.A_size,
        model=args.model,
        chi_sweep=chi_sweep,
        restarts_per_chi=args.restarts_per_chi,
        fit_steps=args.fit_steps,
        seed=args.seed,
        output_dir=Path(args.output),
        j=args.j,
        h=args.h,
        eps_fid=args.eps_fid,
        eps_S=args.eps_S
    )
    
    run_id = make_run_id()
    run_dir = config.output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("CLAIM 3P: Physical Hamiltonian Convergence Test")
    print("=" * 70)
    print(f"Model: {config.model}")
    print(f"System: L={config.L}, A={config.A_size}")
    print(f"Chi sweep: {config.chi_sweep}")
    print(f"Restarts: {config.restarts_per_chi}")
    print(f"Steps: {config.fit_steps}")
    print(f"Output: {run_dir}")
    print("=" * 70)
    
    # Phase 1: Exact Diagonalization
    print("\n[Phase 1] Exact Diagonalization (Gold Standard)")
    ed_result = exact_diagonalization(
        L=config.L, model=config.model, A_size=config.A_size,
        j=config.j, h=config.h
    )
    
    # Phase 2: MERA optimization across chi sweep
    print("\n[Phase 2] MERA Variational Optimization")
    
    all_results = []
    best_results_per_chi = []
    
    for chi in config.chi_sweep:
        print(f"\n  Chi = {chi}")
        results = run_mera_with_restarts(
            L=config.L, chi=chi, ed_psi=ed_result.ground_state_psi,
            ed_result=ed_result, model=config.model, steps=config.fit_steps,
            restarts=config.restarts_per_chi, seed_base=config.seed,
            j=config.j, h=config.h
        )
        
        # Pick best fidelity result
        best_result = max(results, key=lambda r: r.fidelity)
        best_results_per_chi.append(best_result)
        all_results.extend(results)
        
        print(f"    Best: fidelity={best_result.fidelity:.6f}, S={best_result.entropy:.4f}")
    
    # Extract arrays for falsifiers
    best_fidelities = [r.fidelity for r in best_results_per_chi]
    best_entropies = [r.entropy for r in best_results_per_chi]
    
    # Run falsifiers
    print("\n[Phase 3] Acceptance Criteria Evaluation")
    
    p31 = falsifier_p31_nondecreasing_fidelity(
        best_fidelities, config.chi_sweep, config.eps_fid
    )
    print(f"  P3.1 Fidelity nondecreasing: {'PASS' if p31['passed'] else 'FAIL'}")
    
    p32 = falsifier_p32_entropy_convergence(
        best_entropies, ed_result.entanglement_entropy, 
        config.chi_sweep, config.eps_S
    )
    print(f"  P3.2 Entropy convergence: {'PASS' if p32['passed'] else 'FAIL'}")
    
    final_fid = best_fidelities[-1]
    final_S_error = abs(best_entropies[-1] - ed_result.entanglement_entropy)
    p33 = falsifier_p33_final_thresholds(final_fid, final_S_error, config.L, config.model)
    print(f"  P3.3 Final thresholds: {'PASS' if p33['passed'] else 'FAIL'}")
    print(f"        Fidelity: {final_fid:.4f} (threshold: {p33['fid_threshold']})")
    print(f"        S error: {final_S_error:.4f} (threshold: {p33['S_threshold']})")
    
    p34 = falsifier_p34_model_selection(config.chi_sweep, best_entropies, 
                                         ed_result.entanglement_entropy)
    print(f"  P3.4 Model selection: {'PASS' if p34['passed'] else 'FAIL'}")
    print(f"        Delta AIC: {p34['delta_aic']:.2f}")
    print(f"        Delta BIC: {p34['delta_bic']:.2f}")
    
    # Final verdict
    all_passed = p31['passed'] and p32['passed'] and p33['passed'] and p34['passed']
    
    if all_passed:
        verdict = "SUPPORTED"
    elif not p33['passed']:
        verdict = "REJECTED"
    else:
        verdict = "INCONCLUSIVE"
    
    print(f"\n{'=' * 70}")
    print(f"FINAL VERDICT: {verdict}")
    print(f"{'=' * 70}")
    
    # Save artifacts
    print(f"\n[Phase 4] Saving artifacts to {run_dir}")
    
    # Metrics per chi
    metrics = []
    for chi, best in zip(config.chi_sweep, best_results_per_chi):
        metrics.append({
            "chi": chi,
            "fidelity_best": best.fidelity,
            "entropy_best": best.entropy,
            "energy_estimated": best.final_energy,
            "seed": best.seed,
            "restart_idx": best.restart_idx
        })
    
    write_json(run_dir / "metrics.json", {
        "chi_sweep": config.chi_sweep,
        "metrics_per_chi": metrics,
        "ed_reference": {
            "ground_state_energy": ed_result.ground_state_energy,
            "entanglement_entropy": ed_result.entanglement_entropy
        }
    })
    
    # Verdict
    verdict_data = {
        "claim": "3P",
        "model": config.model,
        "L": config.L,
        "verdict": verdict,
        "falsifiers": {
            "P3.1": p31,
            "P3.2": p32,
            "P3.3": p33,
            "P3.4": p34
        },
        "timestamp": dt.datetime.utcnow().isoformat() + "Z"
    }
    write_json(run_dir / "verdict.json", verdict_data)
    
    # Manifest
    manifest = {
        "run_id": run_id,
        "timestamp": dt.datetime.utcnow().isoformat() + "Z",
        "config": {
            "L": config.L,
            "A_size": config.A_size,
            "model": config.model,
            "chi_sweep": config.chi_sweep,
            "restarts_per_chi": config.restarts_per_chi,
            "fit_steps": config.fit_steps,
            "seed": config.seed,
            "eps_fid": config.eps_fid,
            "eps_S": config.eps_S
        },
        "ed_reference": {
            "ground_state_energy": float(ed_result.ground_state_energy),
            "entanglement_entropy": float(ed_result.entanglement_entropy),
            "n_sites": ed_result.n_sites
        },
        "verdict": verdict,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__
    }
    write_json(run_dir / "manifest.json", manifest)
    
    # Raw results CSV
    with open(run_dir / "raw_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["chi", "restart_idx", "seed", "fidelity", "entropy", "energy", "converged"])
        for r in all_results:
            writer.writerow([r.chi, r.restart_idx, r.seed, r.fidelity, r.entropy, 
                           r.final_energy, r.converged])
    
    print(f"  Saved: metrics.json, verdict.json, manifest.json, raw_results.csv")
    print(f"  Run directory: {run_dir}")
    
    return 0 if verdict == "SUPPORTED" else 1


if __name__ == "__main__":
    sys.exit(main())
