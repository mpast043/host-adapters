#!/usr/bin/env python3
"""
Extract scaling dimensions and entanglement quantities from quantum spin chains.

Two approaches:
  1. ED ground state -> reduced density matrix -> S, C_E, gap, spectrum scaling dims
  2. Random MERA -> ascending superoperator -> scaling dims (structural test only)

For scaling dimensions from the entanglement spectrum (Li-Haldane):
    Delta_n = -ln(lambda_n / lambda_0) / (2*pi / ln(L))

For scaling dimensions from ascending superoperator:
    Delta_alpha = -log2(|mu_alpha| / |mu_0|)

References:
    Li & Haldane, PRL 101, 010504 (2008) — entanglement spectrum
    Lyu et al. PRR 3, 023048 (2021) — scaling dims from tensor RG
    de Boer et al. PRD 99, 066012 (2019) — capacity of entanglement
"""

from __future__ import annotations

import argparse
import json
import datetime as dt
from datetime import timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy.sparse.linalg import eigsh


# ---------------------------------------------------------------------------
# Exact diagonalization
# ---------------------------------------------------------------------------

def ed_ground_state(model: str, L: int, delta: float = 1.0,
                    j: float = 1.0, h: float = 1.0) -> Tuple[float, np.ndarray]:
    """Get ground state energy and wavefunction via ED."""
    import quimb as qu

    if "heisenberg" in model:
        cyclic = "cyclic" in model
        H = qu.ham_heis(L, cyclic=cyclic, sparse=True)
    elif "ising" in model:
        cyclic = "cyclic" in model
        H = qu.ham_ising(L, jz=j, bx=h, cyclic=cyclic, sparse=True)
    elif "xxz" in model:
        cyclic = "cyclic" in model
        H = qu.ham_heis(L, j=(1.0, 1.0, delta), cyclic=cyclic, sparse=True)
    else:
        raise ValueError(f"Unknown model: {model}")

    E, psi = eigsh(H, k=1, which='SA')
    return float(E[0]), psi[:, 0]


def reduced_density_matrix(psi: np.ndarray, L: int, n_A: int) -> np.ndarray:
    """Compute reduced density matrix for the first n_A sites of L-site system."""
    d = 2  # spin-1/2
    d_A = d ** n_A
    d_B = d ** (L - n_A)
    psi_mat = psi.reshape(d_A, d_B)
    rho = psi_mat @ psi_mat.conj().T
    rho = 0.5 * (rho + rho.conj().T)
    rho /= np.trace(rho)
    return rho


# ---------------------------------------------------------------------------
# Entanglement quantities
# ---------------------------------------------------------------------------

def compute_all_entanglement(rho: np.ndarray) -> dict:
    """Compute S, C_E, gap ratio, and full spectrum from rho."""
    evals = np.sort(np.linalg.eigvalsh(rho))[::-1]
    evals_pos = evals[evals > 1e-15]

    S = float(-np.sum(evals_pos * np.log(evals_pos)))
    log_ev = np.log(evals_pos)
    CE = float(np.sum(evals_pos * log_ev**2) - S**2)

    gap = float(evals[0] - evals[1]) if len(evals) >= 2 else 0.0
    gap_ratio = float(gap / evals[0]) if evals[0] > 1e-15 else float('nan')

    return {
        "S": S,
        "CE": CE,
        "CE_over_S": CE / S if S > 1e-15 else None,
        "gap": gap,
        "gap_ratio": gap_ratio,
        "gap_ratio_pct": gap_ratio * 100,
        "spectrum": evals_pos.tolist(),
    }


# ---------------------------------------------------------------------------
# Scaling dimensions from entanglement spectrum (Li-Haldane)
# ---------------------------------------------------------------------------

def scaling_dims_from_spectrum(spectrum: List[float], num_dims: int = 10) -> List[dict]:
    """Extract scaling dimensions from entanglement spectrum.

    For critical 1+1D systems, the entanglement spectrum levels map to
    CFT scaling dimensions via:
        xi_n = -ln(lambda_n)
        Delta_n ~ (xi_n - xi_0) * normalization

    We report raw ratios; the normalization depends on geometry.
    """
    spec = np.array(spectrum)
    spec = spec[spec > 1e-15]
    if len(spec) < 2:
        return []

    xi = -np.log(spec)  # entanglement energies
    xi_0 = xi[0]  # ground state of entanglement Hamiltonian

    dims = []
    for i, x in enumerate(xi[:num_dims]):
        dims.append({
            "index": i,
            "lambda": float(spec[i]),
            "xi": float(x),
            "delta_xi": float(x - xi_0),
            "ratio_log2": float(-np.log2(spec[i] / spec[0])) if spec[0] > 0 else 0,
        })
    return dims


# ---------------------------------------------------------------------------
# Ascending superoperator from MERA (structural test)
# ---------------------------------------------------------------------------

def superoperator_from_random_mera(L: int, chi: int, seed: int = 42) -> List[dict]:
    """Build ascending superoperator from a RANDOM (unconverged) MERA.

    This tests the extraction machinery. For meaningful physics, we'd need
    a converged MERA (requires torch/jax for TNOptimizer).
    """
    import quimb.tensor as qtn

    np.random.seed(seed)
    mera = qtn.MERA.rand(L, max_bond=chi, dtype="float64")

    layer_results = []
    for layer_idx in range(10):
        layer_tag = f"_LAYER{layer_idx}"
        unis = []
        isos = []
        for t in mera:
            if layer_tag not in t.tags:
                continue
            data = np.array(t.data, dtype=np.complex128)
            if "_UNI" in t.tags:
                unis.append(data)
            elif "_ISO" in t.tags:
                isos.append(data)

        if not unis:
            break

        u = unis[0]
        if u.ndim != 4:
            break

        d_in = u.shape[0] * u.shape[1]
        d_out = u.shape[2] * u.shape[3]
        U = u.reshape(d_in, d_out)

        # Build superoperator on unitary: O -> U† O U
        # This is a square map: d_in^2 -> d_in^2 (since U is unitary, d_in == d_out)
        # For non-square U, restrict to the square part
        d_op = min(d_in, d_out)
        A_cols = []
        for i in range(d_op):
            for j_idx in range(d_op):
                O = np.zeros((d_op, d_op), dtype=np.complex128)
                O[i, j_idx] = 1.0
                # Pad if needed
                if d_op < d_in:
                    O_full = np.zeros((d_in, d_in), dtype=np.complex128)
                    O_full[:d_op, :d_op] = O
                    O_out = U.conj().T @ O_full @ U
                else:
                    O_out = U.conj().T @ O @ U
                # Project back to d_op x d_op
                O_out = O_out[:d_op, :d_op]
                A_cols.append(O_out.ravel())

        A = np.array(A_cols).T  # should be (d_op^2, d_op^2) = square
        eigenvalues = np.linalg.eigvals(A)
        mags = np.sort(np.abs(eigenvalues))[::-1]
        mu_0 = mags[0]

        dims = []
        for k, mag in enumerate(mags[:10]):
            if mag > 1e-12 and mu_0 > 1e-12:
                ratio = mag / mu_0
                if ratio > 1e-12:
                    dims.append({
                        "index": k,
                        "eigenvalue_mag": float(mag),
                        "scaling_dim": float(-np.log2(ratio)),
                    })

        layer_results.append({
            "layer": layer_idx,
            "mu_0": float(mu_0),
            "A_shape": list(A.shape),
            "scaling_dimensions": dims,
            "note": "RANDOM MERA — not physically meaningful",
        })

    return layer_results


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_ed_extraction(model: str, L: int, delta: float = 1.0) -> dict:
    """Full ED pipeline: ground state -> rho -> S, C_E, spectrum dims."""
    print(f"\n  {model} L={L}" + (f" delta={delta}" if "xxz" in model else ""))

    E0, psi = ed_ground_state(model, L, delta=delta)
    rho = reduced_density_matrix(psi, L, L // 2)
    ent = compute_all_entanglement(rho)
    spec_dims = scaling_dims_from_spectrum(ent["spectrum"])

    print(f"    E0={E0:.6f}  S={ent['S']:.6f}  C_E={ent['CE']:.6f}  "
          f"C_E/S={ent['CE_over_S']:.4f}  gap_ratio={ent['gap_ratio_pct']:.1f}%")
    if spec_dims:
        dims_str = ", ".join(f"{d['ratio_log2']:.3f}" for d in spec_dims[:6])
        print(f"    Spectrum scaling dims (log2 ratio): [{dims_str}]")

    return {
        "model": model, "L": L, "delta": delta,
        "energy": E0,
        **ent,
        "spectrum_scaling_dims": spec_dims,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Extract scaling dimensions + C_E via ED and MERA superoperator")
    parser.add_argument("--models", default="heisenberg_cyclic,ising_cyclic",
                        help="Comma-separated models")
    parser.add_argument("--L-values", default="4,6,8,10,12",
                        help="Comma-separated system sizes")
    parser.add_argument("--chi", type=int, default=8,
                        help="Bond dim for MERA structural test")
    parser.add_argument("--delta", type=float, default=1.0,
                        help="XXZ anisotropy")
    parser.add_argument("--output", default="outputs/scaling_extraction")
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",")]
    L_values = [int(x) for x in args.L_values.split(",")]

    all_results = {}

    # ---- Part 1: ED extraction across models and sizes ----
    print("=" * 60)
    print("PART 1: Exact Diagonalization")
    print("=" * 60)

    for model in models:
        print(f"\n--- {model} ---")
        model_results = []
        for L in L_values:
            try:
                r = run_ed_extraction(model, L, delta=args.delta)
                model_results.append(r)
            except Exception as e:
                print(f"    L={L} FAILED: {e}")
        all_results[model] = model_results

    # ---- Part 2: MERA superoperator structural test ----
    print(f"\n{'='*60}")
    print("PART 2: MERA Ascending Superoperator (random MERA, structural test)")
    print("=" * 60)

    mera_layers = superoperator_from_random_mera(L=8, chi=args.chi)
    for layer in mera_layers:
        dims = [d["scaling_dim"] for d in layer["scaling_dimensions"][:6]]
        print(f"  Layer {layer['layer']}: mu_0={layer['mu_0']:.4f}  "
              f"dims={[round(d, 3) for d in dims]}  [{layer['note']}]")

    # ---- Summary ----
    print(f"\n{'='*60}")
    print("SUMMARY: C_E/S across models and sizes")
    print("=" * 60)

    for model, results in all_results.items():
        print(f"\n  {model}:")
        print(f"    {'L':>4}  {'E0':>10}  {'S':>8}  {'C_E':>8}  {'C_E/S':>7}  {'gap%':>6}")
        for r in results:
            print(f"    {r['L']:>4}  {r['energy']:>10.6f}  {r['S']:>8.6f}  "
                  f"{r['CE']:>8.6f}  {r['CE_over_S']:>7.4f}  {r['gap_ratio_pct']:>6.1f}%")

    # C_E/S universality check
    print(f"\n  C_E/S ratio comparison (L=8 or largest available):")
    for model, results in all_results.items():
        # Pick L=8 if available, else largest
        target = [r for r in results if r["L"] == 8]
        if not target:
            target = [results[-1]] if results else []
        if target:
            r = target[0]
            print(f"    {model}: C_E/S = {r['CE_over_S']:.4f} (L={r['L']})")

    print(f"\n  Known CFT scaling dimensions:")
    print(f"    Ising (c=1/2):    0, 0.125, 1.0")
    print(f"    Heisenberg (c=1): 0, 0.5, 1.0, 1.5")
    print(f"    Framework W01:    d_s = 1.336 +/- 0.029")

    # ---- Save ----
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = dt.datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_file = out_dir / f"{ts}_scaling_extraction.json"

    output = {
        "generated_at_utc": ts,
        "method": "ED (exact ground state) + MERA superoperator (random, structural)",
        "ed_results": all_results,
        "mera_superoperator_structural": mera_layers,
        "known_cft": {
            "ising": [0, 0.125, 1.0],
            "heisenberg": [0, 0.5, 1.0, 1.5],
            "framework_W01": {"ds": 1.336, "error": 0.029},
        },
    }
    out_file.write_text(json.dumps(output, indent=2, default=str))
    print(f"\n  Saved: {out_file}")


if __name__ == "__main__":
    main()
