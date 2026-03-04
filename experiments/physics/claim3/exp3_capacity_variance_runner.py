#!/usr/bin/env python3
"""
Capacity-of-Entanglement (κ₂) Runner

Tests the SECOND cumulant of the entanglement spectrum (variance),
which literature identifies as the TRUE "capacity of entanglement"
(de Boer, Järvelä, Keski-Vakkuri PRD 99, 066012, 2019).

This is distinct from κ₁ (von Neumann entropy) tested in P2.

Key Insight:
- Capacity of entanglement C = Var(H_A) = ⟨H_A²⟩ - ⟨H_A⟩²
- This is the variance of the modular Hamiltonian eigenvalues
- For critical systems: C ∝ c (central charge)
- For gapped systems: C saturates

This may explain why Heisenberg "fails" P2 saturation tests:
- P2 tests κ₁ (entropy), not κ₂ (capacity)
- Heisenberg may have bounded κ₂ despite unbounded κ₁
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import json
import datetime as dt
from pathlib import Path


def build_hamiltonian(L: int, model: str, boundary: str = "cyclic") -> tuple:
    """Build Hamiltonian for 1D spin chain."""
    
    from scipy.sparse import csr_matrix
    from scipy.linalg import eigh
    
    dim = 2 ** L
    H = np.zeros((dim, dim), dtype=np.complex128)
    
    def sigma_z(i: int) -> np.ndarray:
        """Pauli Z on site i."""
        op = np.eye(1)
        for j in range(L):
            if j == i:
                op = np.kron(op, np.array([[1, 0], [0, -1]]))
            else:
                op = np.kron(op, np.eye(2))
        return op
    
    def sigma_x(i: int) -> np.ndarray:
        """Pauli X on site i."""
        op = np.eye(1)
        for j in range(L):
            if j == i:
                op = np.kron(op, np.array([[0, 1], [1, 0]]))
            else:
                op = np.kron(op, np.eye(2))
        return op
    
    def sigma_y(i: int) -> np.ndarray:
        """Pauli Y on site i."""
        op = np.eye(1)
        for j in range(L):
            if j == i:
                op = np.kron(op, np.array([[0, -1j], [1j, 0]]))
            else:
                op = np.kron(op, np.eye(2))
        return op
    
    if model == "ising":
        # Ising model: H = -J Σ σz_i σz_{i+1} - h Σ σx_i
        J, h = 1.0, 0.5  # Critical point
        for i in range(L - 1):
            H -= J * sigma_z(i) @ sigma_z(i + 1)
        for i in range(L):
            H -= h * sigma_x(i)
        if boundary == "cyclic":
            H -= J * sigma_z(L - 1) @ sigma_z(0)
    
    elif model == "heisenberg":
        # Heisenberg XXX model: H = J Σ (σx_i σx_{i+1} + σy_i σy_{i+1} + σz_i σz_{i+1})
        J = 1.0
        for i in range(L - 1):
            H += J * (
                sigma_x(i) @ sigma_x(i + 1) +
                sigma_y(i) @ sigma_y(i + 1) +
                sigma_z(i) @ sigma_z(i + 1)
            )
        if boundary == "cyclic":
            H += J * (
                sigma_x(L - 1) @ sigma_x(0) +
                sigma_y(L - 1) @ sigma_y(0) +
                sigma_z(L - 1) @ sigma_z(0)
            )
    
    else:
        raise ValueError(f"Unknown model: {model}")
    
    return H


def compute_capacity_of_entanglement(rho_A: np.ndarray) -> dict:
    """
    Compute capacity of entanglement (κ₂) and related quantities.
    
    Capacity of entanglement C = Var(H_A) = ⟨H_A²⟩ - ⟨H_A⟩²
    
    For reduced density matrix ρ_A with eigenvalues λ_i:
    - κ₁ (entropy): S = -Σ λ_i log(λ_i)
    - κ₂ (capacity): C = Σ λ_i (log λ_i)² - (Σ λ_i log λ_i)²
    
    Returns dict with all cumulants.
    """
    
    # Get eigenvalues (entanglement spectrum)
    eigvals = np.linalg.eigvalsh(rho_A)
    eigvals = eigvals[eigvals > 1e-12]  # Remove zeros
    
    # Log of eigenvalues
    log_eig = np.log(eigvals)
    
    # First cumulant: Entropy S = -⟨H_A⟩
    kappa_1 = -np.sum(eigvals * log_eig)
    
    # Second cumulant: Capacity C = ⟨H_A²⟩ - ⟨H_A⟩²
    kappa_2 = np.sum(eigvals * log_eig**2) - kappa_1**2
    
    # Third cumulant: Skewness
    kappa_3 = np.sum(eigvals * log_eig**3) - 3 * kappa_1 * kappa_2 - kappa_1**3
    
    # Effective dimension
    d_eff = np.exp(kappa_1)
    
    # Spectrum statistics
    spectrum = -log_eig  # Modular Hamiltonian eigenvalues
    mean_H = kappa_1
    var_H = kappa_2
    std_H = np.sqrt(kappa_2)
    
    return {
        "kappa_1_entropy": float(kappa_1),
        "kappa_2_capacity": float(kappa_2),
        "kappa_3_skewness": float(kappa_3),
        "d_eff": float(d_eff),
        "spectrum_mean": float(mean_H),
        "spectrum_var": float(var_H),
        "spectrum_std": float(std_H),
        "spectrum_min": float(np.min(spectrum)),
        "spectrum_max": float(np.max(spectrum)),
        "spectrum_range": float(np.max(spectrum) - np.min(spectrum)),
        "n_eigenvalues": int(len(eigvals))
    }


def reduced_density_matrix(psi: np.ndarray, A_size: int, L: int) -> np.ndarray:
    """Compute reduced density matrix for subsystem A."""
    
    B_size = L - A_size
    
    # Reshape into A and B indices
    psi = psi.reshape(2**A_size, 2**B_size)
    
    # Compute density matrix
    rho_A = psi @ psi.conj().T
    
    return rho_A


def run_capacity_test(L: int, A_size: int, model: str, boundary: str = "cyclic") -> dict:
    """Run capacity-of-entanglement test for given system."""
    
    print(f"\n{'='*60}")
    print(f"CAPACITY-OF-ENTANGLEMENT TEST (κ₂)")
    print(f"{'='*60}")
    print(f"Model: {model}, Boundary: {boundary}")
    print(f"L={L}, A_size={A_size}")
    print(f"{'='*60}\n")
    
    # Build and diagonalize Hamiltonian
    print(f"[ED] Building {model} Hamiltonian for L={L}...")
    H = build_hamiltonian(L, model, boundary)
    
    print(f"[ED] Diagonalizing {H.shape[0]}x{H.shape[1]}...")
    eigvals_full, eigvecs = np.linalg.eigh(H)
    
    # Ground state
    E0 = eigvals_full[0]
    psi_0 = eigvecs[:, 0]
    
    print(f"[ED] E0 = {E0:.6f}")
    
    # Compute reduced density matrix
    print(f"[RDM] Computing reduced density matrix (A_size={A_size})...")
    rho_A = reduced_density_matrix(psi_0, A_size, L)
    
    # Compute all cumulants
    print(f"[CUMULANTS] Computing entanglement cumulants...")
    cumulants = compute_capacity_of_entanglement(rho_A)
    
    print(f"\n[RESULTS]")
    print(f"  κ₁ (Entropy):     S = {cumulants['kappa_1_entropy']:.6f}")
    print(f"  κ₂ (Capacity):    C = {cumulants['kappa_2_capacity']:.6f}")
    print(f"  κ₃ (Skewness):    γ = {cumulants['kappa_3_skewness']:.6f}")
    print(f"  d_eff:            {cumulants['d_eff']:.4f}")
    print(f"  Spectrum range:   {cumulants['spectrum_range']:.6f}")
    
    # Compare with literature expectations
    print(f"\n[LITERATURE COMPARISON]")
    
    # Central charge expectation
    if model == "heisenberg":
        c_expected = 1.0
        model_name = "Heisenberg (c=1)"
    elif model == "ising":
        c_expected = 0.5
        model_name = "Ising (c=1/2)"
    else:
        c_expected = None
        model_name = model
    
    if c_expected is not None:
        # For critical systems: C/c should be related to log(L_A)
        L_A = A_size
        expected_C_c_ratio = (1/6) * np.log(L_A)  # Rough CFT prediction
        
        print(f"  Model: {model_name}")
        print(f"  C/c ratio: {cumulants['kappa_2_capacity']/c_expected:.4f}")
        print(f"  S/c ratio: {cumulants['kappa_1_entropy']/c_expected:.4f}")
    
    result = {
        "L": L,
        "A_size": A_size,
        "model": model,
        "boundary": boundary,
        "E0": float(E0),
        "cumulants": cumulants,
        "timestamp": dt.datetime.utcnow().isoformat(),
        "test_type": "capacity_of_entanglement_k2"
    }
    
    return result


def run_scaling_study(model: str, L_values: list, boundary: str = "cyclic") -> dict:
    """Run scaling study for capacity vs system size."""
    
    results = []
    
    print(f"\n{'='*60}")
    print(f"SCALING STUDY: {model.upper()}")
    print(f"{'='*60}\n")
    
    for L in L_values:
        A_size = L // 2  # Half-chain bipartition
        result = run_capacity_test(L, A_size, model, boundary)
        results.append(result)
    
    # Analyze scaling
    print(f"\n{'='*60}")
    print(f"SCALING ANALYSIS")
    print(f"{'='*60}\n")
    
    L_arr = np.array([r["L"] for r in results])
    S_arr = np.array([r["cumulants"]["kappa_1_entropy"] for r in results])
    C_arr = np.array([r["cumulants"]["kappa_2_capacity"] for r in results])
    
    # Fit entropy: S = a + b*log(L)
    log_L = np.log(L_arr)
    coeffs_S = np.polyfit(log_L, S_arr, 1)
    
    # Fit capacity: C = a + b*log(L) vs C = constant
    coeffs_C_log = np.polyfit(log_L, C_arr, 1)
    coeffs_C_const = np.polyfit(np.ones(len(L_arr)), C_arr, 0)
    
    # Compare models
    residuals_log = np.sum((C_arr - coeffs_C_log[0]*log_L - coeffs_C_log[1])**2)
    residuals_const = np.sum((C_arr - coeffs_C_const[0])**2)
    
    # Compute ΔAIC
    n = len(L_arr)
    k_log = 2
    k_const = 1
    AIC_log = n * np.log(residuals_log / n) + 2 * k_log
    AIC_const = n * np.log(residuals_const / n) + 2 * k_const
    delta_AIC = AIC_const - AIC_log  # Positive favors constant
    
    print(f"  Entropy scaling: S = {coeffs_S[1]:.4f} + {coeffs_S[0]:.4f} × log(L)")
    print(f"  Capacity scaling (log): C = {coeffs_C_log[1]:.4f} + {coeffs_C_log[0]:.4f} × log(L)")
    print(f"  Capacity scaling (const): C = {coeffs_C_const[0]:.4f}")
    print(f"\n  Model comparison:")
    print(f"    ΔAIC (const vs log): {delta_AIC:.4f}")
    
    if delta_AIC > 2:
        verdict = "CAPACITY_SATURATES"
        print(f"    Verdict: {verdict} (capacity bounded)")
    else:
        verdict = "CAPACITY_GROWS"
        print(f"    Verdict: {verdict} (capacity grows with L)")
    
    return {
        "model": model,
        "boundary": boundary,
        "results": results,
        "scaling": {
            "entropy_slope": float(coeffs_S[0]),
            "entropy_intercept": float(coeffs_S[1]),
            "capacity_log_slope": float(coeffs_C_log[0]),
            "capacity_const": float(coeffs_C_const[0]),
            "delta_AIC": float(delta_AIC)
        },
        "verdict": verdict
    }


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Capacity-of-Entanglement (κ₂) Test")
    parser.add_argument("--L", type=int, nargs="+", default=[8, 12, 16],
                        help="System sizes to test")
    parser.add_argument("--model", type=str, default="ising",
                        choices=["ising", "heisenberg"],
                        help="Model to test")
    parser.add_argument("--boundary", type=str, default="cyclic",
                        choices=["cyclic", "open"],
                        help="Boundary condition")
    parser.add_argument("--A_size", type=int, default=None,
                        help="Subsystem size (default: L/2)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory")
    
    args = parser.parse_args()
    
    # Run scaling study
    study_result = run_scaling_study(args.model, args.L, args.boundary)
    
    # Save results
    if args.output:
        out_path = Path(args.output)
        out_path.mkdir(parents=True, exist_ok=True)
        
        out_file = out_path / "capacity_variance_results.json"
        with open(out_file, "w") as f:
            json.dump(study_result, f, indent=2)
        
        print(f"\n[SAVED] Results written to {out_file}")
    
    # Print comparison summary
    print(f"\n{'='*60}")
    print(f"FINAL VERDICT")
    print(f"{'='*60}")
    print(f"  Model: {args.model.upper()}")
    print(f"  κ₂ (Capacity): {study_result['verdict']}")
    print(f"  ΔAIC: {study_result['scaling']['delta_AIC']:.4f}")
    
    return study_result


if __name__ == "__main__":
    main()