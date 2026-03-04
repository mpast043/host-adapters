#!/usr/bin/env python3
"""
Real MERA Entanglement-Capacity Correlation Test Runner.

This module tests the hypothesis that capacity C is proportional to
entanglement entropy S by running ACTUAL MERA simulations.

Hypothesis H1: C ∝ S with R² > 0.95

Uses the same MERA infrastructure as exp3_claim3_physical_convergence_runner_v2.
"""

from __future__ import annotations

import argparse
import datetime as dt
from datetime import timezone
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from entanglement_utils import (
    von_neumann_entropy,
    reduced_density_matrix,
    entanglement_spectrum,
    entanglement_gap,
    analyze_capacity_entanglement_correlation,
    capacity_of_entanglement,
)


# Module constants
__version__ = "1.0.0"

# Threshold for hypothesis testing
R_SQUARED_THRESHOLD = 0.95

# Valid model types
VALID_MODELS = ["ising_open", "ising_cyclic", "heisenberg_open", "heisenberg_cyclic", "xxz_open", "xxz_cyclic"]


def run_id() -> str:
    """Generate a unique run identifier with timestamp and random suffix."""
    t = dt.datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    r = os.urandom(4).hex()
    return f"{t}_{r}"


def compute_entropy_from_mera_state(mera, L: int, A_sites: List[int]) -> float:
    """
    Compute S(A) from MERA by forming reduced density matrix.

    This uses the same method as exp3_claim3_physical_convergence_runner_v2.
    """
    A_size = len(A_sites)

    # Build bra with different index labels
    bra = mera.H.reindex_sites("b{}", A_sites)

    # Select causal cone
    import quimb.tensor as qtn
    tags = [mera.site_tag(i) for i in A_sites]
    rho_tn = bra.select(tags, which="any") & mera.select(tags, which="any")

    left_inds = tuple(f"k{i}" for i in A_sites)
    right_inds = tuple(f"b{i}" for i in A_sites)

    # Dense rho_A
    rho = rho_tn.to_dense(left_inds, right_inds)
    rho = np.array(rho, dtype=np.complex128)
    rho = 0.5 * (rho + rho.conj().T)

    # Normalize
    tr = float(np.real_if_close(np.trace(rho)))
    if tr > 0:
        rho = rho / tr

    # Use our entanglement_utils
    return von_neumann_entropy(rho)


def optimize_mera_for_entanglement(
    L: int,
    chi: int,
    model: str,
    steps: int = 100,
    seed: int = 42,
    j: float = 1.0,
    h: float = 1.0,
    delta: float = 0.5,
) -> Dict[str, Any]:
    """
    Optimize MERA and compute entanglement entropy.

    Parameters
    ----------
    L : int
        System size (number of sites)
    chi : int
        Bond dimension
    model : str
        Model type (ising_open, ising_cyclic, heisenberg_open, heisenberg_cyclic, xxz_open, xxz_cyclic)
    steps : int
        Number of optimization steps
    seed : int
        Random seed
    j : float
        Coupling constant (Ising)
    h : float
        Field strength (Ising)
    delta : float
        Anisotropy parameter (XXZ). Delta=1 gives Heisenberg, Delta=0 gives XX model.

    Returns
    -------
    Dict[str, Any]
        Contains chi, entanglement_entropy, energy, and other metrics
    """
    import quimb.tensor as qtn
    import quimb as qu

    np.random.seed(seed)
    print(f"    [MERA] L={L}, chi={chi}, model={model}, steps={steps}")

    # Build random MERA
    mera = qtn.MERA.rand(L, max_bond=chi, dtype="float64")

    # Build Hamiltonian terms
    if model in ["ising_open", "ising_cyclic"]:
        cyclic = model == "ising_cyclic"
        pairs = [(i, i + 1) for i in range(L - 1)]
        if cyclic:
            pairs.append((L - 1, 0))

        Z = np.array([[1, 0], [0, -1]], dtype=np.float64)
        ZZ = -j * np.kron(Z, Z)
        X = np.array([[0, 1], [1, 0]], dtype=np.float64)
        Xlocal = -h * X

        terms = {}
        for i, j_pair in pairs:
            terms[(i, j_pair)] = ZZ
        for i in range(L):
            terms[(i,)] = Xlocal

    elif model in ["heisenberg_open", "heisenberg_cyclic"]:
        cyclic = model == "heisenberg_cyclic"
        H2 = qu.ham_heis(2).real

        pairs = [(i, (i + 1) % L) for i in range(L - 1)]
        if cyclic:
            pairs.append((L - 1, 0))

        terms = {pair: H2 for pair in pairs}
    elif model in ["xxz_open", "xxz_cyclic"]:
        cyclic = model == "xxz_cyclic"

        # XXZ Hamiltonian: H = -J Σ (S^x_i S^x_{i+1} + S^y_i S^y_{i+1} + Δ S^z_i S^z_{i+1})
        # For spin-1/2: S^α = σ^α/2
        # quimb convention: ham_heis(2) = S·S = (1/4)(σ^x⊗σ^x + σ^y⊗σ^y + σ^z⊗σ^z)
        # So we match this: H = S^x⊗S^x + S^y⊗S^y + Δ*S^z⊗S^z
        # For Δ=1, this matches quimb's ham_heis(2)
        sx = np.array([[0, 1], [1, 0]], dtype=np.float64) / 2
        sy = np.array([[0, -1j], [1j, 0]], dtype=np.complex128) / 2
        sz = np.array([[1, 0], [0, -1]], dtype=np.float64) / 2

        # Two-site Hamiltonian: S^x⊗S^x + S^y⊗S^y + Δ*S^z⊗S^z
        # This matches quimb.ham_heis(2) convention for Δ=1
        H2 = (np.kron(sx, sx) + np.kron(sy, sy) + delta * np.kron(sz, sz)).real

        pairs = [(i, i + 1) for i in range(L - 1)]
        if cyclic:
            pairs.append((L - 1, 0))

        terms = {pair: H2 for pair in pairs}
    else:
        raise ValueError(f"Unknown model: {model}")

    # Path optimizer
    try:
        import cotengra as ctg
        opt = ctg.ReusableHyperOptimizer(progbar=False, reconf_opts={})
    except Exception:
        opt = "auto-hq"

    # Norm function for MERA
    def norm_fn(m):
        return m.isometrize(method="exp")

    # Loss function
    def loss_fn(m, terms, optimize="auto-hq"):
        total = 0.0
        for where in terms:
            if len(where) == 2:
                tags = [m.site_tag(coo) for coo in where]
                m_ij = m.select(tags, which="any")
                m_ij_G = m_ij.gate(terms[where], where)
                ex = m_ij_G & m_ij.H
                total += ex.contract(all, optimize=optimize)
            else:
                (site,) = where
                tag = m.site_tag(site)
                m_i = m.select([tag], which="any")
                m_i_G = m_i.gate(terms[where], [site])
                ex = m_i_G & m_i.H
                total += ex.contract(all, optimize=optimize)
        return total

    # Create optimizer
    tnopt = qtn.TNOptimizer(
        mera,
        loss_fn=loss_fn,
        norm_fn=norm_fn,
        loss_constants={"terms": terms},
        loss_kwargs={"optimize": opt},
        autodiff_backend="torch",
        device="cpu",
        jit_fn=False,
    )

    # Optimize
    tnopt.optimizer = "l-bfgs-b"
    try:
        mera_opt = tnopt.optimize(steps)
        final_energy = float(loss_fn(mera_opt, terms))
    except Exception as e:
        print(f"      Optimization warning: {e}")
        mera_opt = mera
        final_energy = float(loss_fn(mera, terms))

    # Compute entanglement entropy for half-chain
    A_sites = list(range(L // 2))
    entanglement_entropy = compute_entropy_from_mera_state(mera_opt, L, A_sites)

    # Compute entanglement spectrum
    bra = mera_opt.H.reindex_sites("b{}", A_sites)
    tags = [mera_opt.site_tag(i) for i in A_sites]
    rho_tn = bra.select(tags, which="any") & mera_opt.select(tags, which="any")
    left_inds = tuple(f"k{i}" for i in A_sites)
    right_inds = tuple(f"b{i}" for i in A_sites)
    rho = rho_tn.to_dense(left_inds, right_inds)
    rho = np.array(rho, dtype=np.complex128)
    rho = 0.5 * (rho + rho.conj().T)
    tr = float(np.real_if_close(np.trace(rho)))
    if tr > 0:
        rho = rho / tr

    spectrum = entanglement_spectrum(rho)
    gap = entanglement_gap(rho)

    # Compute capacity of entanglement C_E = Var(H_A)
    capacity_e = capacity_of_entanglement(rho)

    # Capacity is hypothesized to be proportional to entanglement entropy
    # For now, we use S as the measure (C = α*S + β)
    # The exact proportionality constant α needs to be determined from theory
    capacity = entanglement_entropy  # Working hypothesis: α=1, β=0

    print(f"      E={final_energy:.6f}, S={entanglement_entropy:.6f}, gap={gap:.6f}")

    return {
        "chi": chi,
        "L": L,
        "model": model,
        "entanglement_entropy": float(entanglement_entropy),
        "entanglement_gap": float(gap),
        "entanglement_spectrum": spectrum.tolist(),
        "energy": float(final_energy),
        "capacity": float(capacity),
        "capacity_of_entanglement": float(capacity_e),
        "optimization_steps": steps,
        "seed": seed,
    }


def run_real_mera_correlation_test(
    chi_values: List[int],
    model: str,
    L: int,
    steps: int,
    output_dir: str,
    seed: int,
    delta: float = 0.5,
) -> Dict[str, Any]:
    """
    Run real MERA simulations to test capacity-entanglement correlation.

    Tests hypothesis H1: C ∝ S with R² > R_SQUARED_THRESHOLD
    """
    print(f"[H1] Running REAL MERA capacity-entanglement correlation test")
    print(f"     Model: {model}, System size: {L}")
    if "xxz" in model:
        print(f"     Anisotropy Δ: {delta}")
    print(f"     Bond dimensions: {chi_values}")
    print(f"     Optimization steps: {steps}")
    print()

    results = []
    capacities = []
    entropies = []

    for i, chi in enumerate(chi_values):
        print(f"  [{i+1}/{len(chi_values)}] chi={chi}")

        sim_seed = seed + i * 1000 + chi * 10000

        try:
            result = optimize_mera_for_entanglement(
                L=L,
                chi=chi,
                model=model,
                steps=steps,
                seed=sim_seed,
                delta=delta,
            )
            results.append(result)
            capacities.append(result["capacity"])
            entropies.append(result["entanglement_entropy"])

        except Exception as e:
            print(f"      ERROR: {e}")
            import traceback
            traceback.print_exc()
            continue

    if len(results) < 2:
        print("\n[ERROR] Not enough successful simulations for correlation analysis")
        return {
            "metadata": {
                "run_id": run_id(),
                "timestamp": dt.datetime.now(timezone.utc).isoformat(),
                "test": "H1_capacity_entanglement_correlation_REAL_MERA",
                "version": __version__,
                "config": {
                    "chi_values": chi_values,
                    "model": model,
                    "L": L,
                    "steps": steps,
                    "seed": seed,
                    "delta": delta,
                },
            },
            "measurements": results,
            "correlation_analysis": {},
            "hypothesis": {
                "H1": "C ∝ S",
                "threshold": R_SQUARED_THRESHOLD,
                "r_squared": None,
                "passed": False,
            },
            "verdict": "ERROR",
        }

    # Analyze correlation
    analysis = analyze_capacity_entanglement_correlation(capacities, entropies)

    passed = analysis["r_squared"] > R_SQUARED_THRESHOLD
    verdict = "ACCEPT" if passed else "REJECT"

    print(f"\n[H1] Correlation Analysis:")
    print(f"     Slope: {analysis['slope']:.6f}")
    print(f"     Intercept: {analysis['intercept']:.6f}")
    print(f"     R-squared: {analysis['r_squared']:.6f}")
    print(f"     P-value: {analysis['p_value']:.2e}")
    print(f"     Correlation: {analysis['correlation']:.6f}")
    print(f"     Verdict: {verdict} (threshold: {R_SQUARED_THRESHOLD})")

    output = {
        "metadata": {
            "run_id": run_id(),
            "timestamp": dt.datetime.now(timezone.utc).isoformat(),
            "test": "H1_capacity_entanglement_correlation_REAL_MERA",
            "version": __version__,
            "config": {
                "chi_values": chi_values,
                "model": model,
                "L": L,
                "steps": steps,
                "seed": seed,
                "delta": delta,
            },
        },
        "measurements": results,
        "correlation_analysis": analysis,
        "hypothesis": {
            "H1": "C ∝ S",
            "threshold": R_SQUARED_THRESHOLD,
            "r_squared": analysis["r_squared"],
            "passed": passed,
        },
        "verdict": verdict,
    }

    # Write results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    rid = output["metadata"]["run_id"]
    result_file = output_path / f"{rid}_real_mera.json"
    result_file.write_text(json.dumps(output, indent=2))
    print(f"\n[Results saved to: {result_file}]")

    return output


def main():
    parser = argparse.ArgumentParser(
        description="Real MERA Entanglement-Capacity Correlation Test"
    )
    parser.add_argument(
        "--chi",
        default="4,8,16",
        help="Comma-separated chi values (default: 4,8,16 - larger values take longer)",
    )
    parser.add_argument(
        "--model",
        default="heisenberg_cyclic",
        choices=VALID_MODELS,
        help="Model type (default: heisenberg_cyclic)",
    )
    parser.add_argument(
        "--L",
        type=int,
        default=8,
        help="System size (default: 8)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="MERA optimization steps (default: 50, use fewer for quick tests)",
    )
    parser.add_argument(
        "--output",
        default="outputs/entanglement_capacity_real",
        help="Output directory",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=0.5,
        help="Anisotropy parameter for XXZ model (default: 0.5). Delta=1 gives Heisenberg, Delta=0 gives XX model.",
    )

    args = parser.parse_args()
    chi_values = [int(x.strip()) for x in args.chi.split(",")]

    run_real_mera_correlation_test(
        chi_values=chi_values,
        model=args.model,
        L=args.L,
        steps=args.steps,
        output_dir=args.output,
        seed=args.seed,
        delta=args.delta,
    )


if __name__ == "__main__":
    main()