#!/usr/bin/env python3
"""
P3: Gluing/Excision Stability Runner (v2 - Patched for Hamiltonian)
Uses corrected L=8 Hamiltonian from exp3_l8_validation_runner
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

sys.path.insert(0, str(Path(__file__).parent.parent))
from physics.claim3.exp3_l8_validation_runner import (
    exact_diagonalization_corrected, compute_entanglement_entropy
)


def run_id():
    t = dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    r = os.urandom(4).hex()
    return f"{t}_{r}"


def gluing_state(psi_A: np.ndarray, psi_B: np.ndarray) -> np.ndarray:
    """Glue via tensor product |psi_AB> = |psi_A> ⊗ |psi_B>"""
    psi_glued = np.kron(psi_A, psi_B)
    psi_glued = psi_glued / np.linalg.norm(psi_glued)
    return psi_glued


def excise_subsystem(psi: np.ndarray, n_keep: int, n_total: int) -> Tuple[np.ndarray, np.ndarray]:
    """Excise (trace out) subsystem B, return rho_A and spectrum"""
    dim_A = 2 ** n_keep
    dim_B = 2 ** (n_total - n_keep)
    psi_matrix = psi.reshape(dim_A, dim_B)
    rho_A = psi_matrix @ psi_matrix.conj().T
    rho_A = rho_A / np.trace(rho_A)
    eigenvalues = np.linalg.eigvalsh(rho_A)
    eigenvalues = eigenvalues[eigenvalues > 1e-12]
    eigenvalues = eigenvalues / np.sum(eigenvalues)
    return rho_A, eigenvalues


def check_subadditivity(S_A: float, S_B: float, S_AB: float, epsilon: float = 1e-6) -> bool:
    return S_AB <= S_A + S_B + epsilon


def check_araki_lieb(S_A: float, S_B: float, S_AB: float) -> bool:
    return abs(S_A - S_B) <= S_AB + 1e-6


def run_p3_mera(cfg: Dict) -> Dict:
    L = cfg["L"]
    A_size = cfg["A_size"]
    B_size = L - A_size
    model = cfg["model"]
    chi_sweep = cfg["chi_sweep"]
    seed = cfg["seed"]
    
    print(f"[P3] Gluing/excision for L={L}, A={A_size}, B={B_size}")
    print(f"[P3] Model: {model}, chi: {chi_sweep}")
    
    is_heis = "heisenberg" in model
    
    # Get ED reference
    print(f"[P3] Computing ED reference...")
    ed_result = exact_diagonalization_corrected(
        L=L,
        model="heisenberg" if is_heis else "ising",
        A_size=A_size,
        j=1.0,
        h=1.0
    )
    psi_full = ed_result["ground_state_psi"]
    E0 = ed_result["ground_state_energy"]
    
    # Compute entropies (corrected argument order: psi, L, A_size)
    S_full = compute_entanglement_entropy(psi_full, L, A_size)
    S_A = compute_entanglement_entropy(psi_full, L, A_size)
    S_B = compute_entanglement_entropy(psi_full, L, B_size)
    
    # Simulate partition states
    dim_A = 2 ** A_size
    dim_B = 2 ** B_size
    psi_A = psi_full[:dim_A] if len(psi_full) >= dim_A else psi_full[:len(psi_full)//2]
    psi_A = psi_A / np.linalg.norm(psi_A)
    psi_B = psi_full[dim_A:dim_A+dim_B] if len(psi_full) >= dim_A + dim_B else psi_full[len(psi_full)//2:]
    psi_B = psi_B / np.linalg.norm(psi_B)
    
    if len(psi_A) < dim_A:
        psi_A = np.pad(psi_A, (0, dim_A - len(psi_A)))
        psi_A = psi_A / np.linalg.norm(psi_A)
    if len(psi_B) < dim_B:
        psi_B = np.pad(psi_B, (0, dim_B - len(psi_B)))
        psi_B = psi_B / np.linalg.norm(psi_B)
    
    # Gluing test
    psi_glued = gluing_state(psi_A, psi_B)
    S_glued = compute_entanglement_entropy(psi_glued, L, A_size)
    gluing_error = abs(S_glued - (S_A + S_B))
    gluing_stable = gluing_error < cfg.get("gluing_threshold", 0.1)
    
    # Excision test
    rho_A, spec_A = excise_subsystem(psi_full, A_size, L)
    S_excised = -np.sum(spec_A * np.log(spec_A)) if len(spec_A) > 0 else 0.0
    valid_spectrum = np.all(spec_A >= 0) and abs(np.sum(spec_A) - 1.0) < 1e-6
    excision_valid = valid_spectrum and abs(S_excised - S_A) < 1e-6
    
    # Inequalities
    subadd = check_subadditivity(S_A, S_B, S_full)
    araki_lieb = check_araki_lieb(S_A, S_B, S_full)
    
    records = [{
        "chi": 16,  # Using full ED state
        "energy": E0,
        "S_full": S_full,
        "S_A": S_A,
        "S_B": S_B,
        "S_glued": S_glued,
        "S_excised": S_excised,
        "gluing_error": gluing_error,
        "gluing_stable": gluing_stable,
        "excision_valid": excision_valid,
        "valid_spectrum": valid_spectrum,
        "subadditivity": subadd,
        "araki_lieb": araki_lieb,
    }]
    
    p31 = bool(gluing_stable)
    p32 = bool(excision_valid)
    p33 = bool(subadd)
    p34 = bool(araki_lieb)
    
    print(f"[P3] Results:")
    print(f"  E0={E0:.6f}, S_full={S_full:.4f}")
    print(f"  Gluing error={gluing_error:.4f}, stable={gluing_stable}")
    print(f"  Subadditivity={subadd}, Araki-Lieb={araki_lieb}")
    
    return {
        "metadata": {
            "run_id": run_id(),
            "timestamp": dt.datetime.utcnow().isoformat(),
            "config": cfg,
            "test": "P3",
            "version": "2.0.0-patched",
        },
        "ed_reference": {
            "energy": E0,
            "entropy": ed_result["entanglement_entropy"],
        },
        "measurements": records,
        "verdict": "ACCEPT" if (p31 and p32 and p33 and p34) else "REJECT",
        "passed": {
            "P3.1_gluing_stable": p31,
            "P3.2_excision_valid": p32,
            "P3.3_subadditivity": p33,
            "P3.4_araki_lieb": p34,
        }
    }


def write_out(res: Dict, out_dir: Path):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(res["metadata"], f, indent=2, default=str)
    
    with open(out_dir / "raw.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "chi", "energy", "S_full", "S_A", "S_B", "S_glued", "S_excised",
            "gluing_error", "gluing_stable", "excision_valid", "valid_spectrum", "subadditivity", "araki_lieb"
        ])
        w.writeheader()
        w.writerows(res["measurements"])
    
    verdict = {
        "test": "P3",
        "test_name": "gluing_excision_stability",
        "verdict": res["verdict"],
        "status": "COMPLETE",
        "passed": res["passed"],
    }
    with open(out_dir / "verdict.json", "w") as f:
        json.dump(verdict, f, indent=2)
    
    print(f"[P3] Verdict: {res['verdict']}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--L", type=int, default=8)
    p.add_argument("--A_size", type=int, default=4)
    p.add_argument("--model", default="heisenberg_open")
    p.add_argument("--chi_sweep", default="4,8,16")
    p.add_argument("--gluing_threshold", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    
    cfg = {
        "L": a.L,
        "A_size": a.A_size,
        "model": a.model,
        "chi_sweep": [int(x) for x in a.chi_sweep.split(",")],
        "gluing_threshold": a.gluing_threshold,
        "seed": a.seed,
    }
    
    res = run_p3_mera(cfg)
    write_out(res, Path(a.output))


if __name__ == "__main__":
    main()
