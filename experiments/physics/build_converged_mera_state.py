#!/usr/bin/env python3
"""Build a fitted MERA state and export an ascending-superoperator artifact."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import quimb as qu
import quimb.tensor as qtn
from scipy.sparse.linalg import eigsh


def _hamiltonian(model: str, L: int, boundary: str):
    cyclic = boundary == "cyclic"
    if model == "ising":
        return qu.ham_ising(L, jz=1.0, bx=1.0, cyclic=cyclic, sparse=True)
    if model == "heisenberg":
        return qu.ham_heis(L, j=1.0, cyclic=cyclic, sparse=True)
    raise ValueError("model must be one of: ising, heisenberg")


def _target_tensor_network(psi: np.ndarray, site_inds: tuple[str, ...]) -> qtn.TensorNetwork:
    t = qtn.Tensor(psi.reshape((2,) * len(site_inds)), inds=site_inds)
    return qtn.TensorNetwork([t])


def _fidelity(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na <= 1e-15 or nb <= 1e-15:
        return 0.0
    return float(abs(np.vdot(a, b)) / (na * nb))


def _choose_isometry(mera: qtn.MERA, layer: int) -> qtn.Tensor:
    layer_tag = f"_LAYER{layer}"
    candidates = [t for t in mera.tensors if "_ISO" in t.tags and layer_tag in t.tags and t.ndim == 3]
    if not candidates:
        raise ValueError(f"No rank-3 isometry found for {layer_tag}")
    return candidates[0]


def _matrix_unit_basis(dim: int) -> list[np.ndarray]:
    basis: list[np.ndarray] = []
    for i in range(dim):
        for j in range(dim):
            e = np.zeros((dim, dim), dtype=float)
            e[i, j] = 1.0
            basis.append(e)
    return basis


def _ascending_superoperator_from_isometry(iso_tensor: qtn.Tensor) -> tuple[np.ndarray, dict[str, float | int]]:
    """Build A: O -> V^T (O ⊗ I) V from a 3-index isometry tensor.

    For iso shape (d_left, d_right, chi), V has shape (d_left * d_right, chi).
    We use local operators O on d_left and spectator I on d_right.
    """
    arr = np.asarray(iso_tensor.data)
    if np.iscomplexobj(arr):
        # We keep the artifact JSON-friendly and runner-compatible.
        arr = arr.real

    d_left, d_right, chi = arr.shape
    V = arr.reshape(d_left * d_right, chi)

    basis_in = _matrix_unit_basis(d_left)
    basis_out = _matrix_unit_basis(chi)
    identity_right = np.eye(d_right, dtype=float)

    superop = np.zeros((chi * chi, d_left * d_left), dtype=float)
    for j, op_in in enumerate(basis_in):
        lifted = np.kron(op_in, identity_right)
        op_out = V.T @ lifted @ V
        for i, basis_op_out in enumerate(basis_out):
            superop[i, j] = float(np.vdot(basis_op_out, op_out))

    # Normalize by spectral radius so leading |lambda| = 1 when square.
    spectral_radius = 0.0
    if superop.shape[0] == superop.shape[1]:
        eigvals = np.linalg.eigvals(superop)
        spectral_radius = float(np.max(np.abs(eigvals))) if eigvals.size else 0.0
        if spectral_radius > 1e-15:
            superop = superop / spectral_radius

    dims = {
        "d_left": int(d_left),
        "d_right": int(d_right),
        "chi": int(chi),
        "superop_rows": int(superop.shape[0]),
        "superop_cols": int(superop.shape[1]),
        "spectral_radius_before_normalization": spectral_radius,
    }
    return superop, dims


def build_and_export(
    model: str,
    boundary: str,
    L: int,
    max_bond: int,
    fit_steps: int,
    layer: int,
    seed: int | None,
    output: Path,
) -> dict[str, Any]:
    if L <= 0 or (L & (L - 1)) != 0:
        raise ValueError("L must be a positive power of 2")

    if seed is not None:
        np.random.seed(seed)

    H = _hamiltonian(model=model, L=L, boundary=boundary)
    evals, evecs = eigsh(H, k=1, which="SA")
    e0 = float(evals[0])
    psi0 = np.asarray(evecs[:, 0], dtype=float)

    mera = qtn.MERA.rand(L, max_bond=max_bond, phys_dim=2, dtype=float)
    target = _target_tensor_network(psi0, tuple(mera.site_inds))
    mera_fit = mera.fit(
        target,
        method="als",
        steps=fit_steps,
        solver_dense="lstsq",
        inplace=False,
        progbar=False,
    )

    mera_vec = np.asarray(mera_fit.to_dense(mera_fit.site_inds), dtype=float)
    fid = _fidelity(mera_vec, psi0)
    iso = _choose_isometry(mera_fit, layer=layer)
    superop, dims = _ascending_superoperator_from_isometry(iso)

    payload: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "boundary": boundary,
        "L": L,
        "max_bond": max_bond,
        "fit_steps": fit_steps,
        "layer": layer,
        "seed": seed,
        "ground_energy": e0,
        "fit_fidelity": fid,
        "ascending_superoperator": superop.tolist(),
        "ascending_map_metadata": dims,
        "ascending_map_formula": "A(O) = V^T (O ⊗ I) V",
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fit MERA to ED ground state and export ascending superoperator JSON."
    )
    parser.add_argument("--model", choices=["ising", "heisenberg"], default="ising")
    parser.add_argument("--boundary", choices=["open", "cyclic"], default="open")
    parser.add_argument("--L", type=int, default=8, help="System size (power of 2).")
    parser.add_argument("--max-bond", type=int, default=4)
    parser.add_argument("--fit-steps", type=int, default=8)
    parser.add_argument("--layer", type=int, default=1, help="MERA layer from which to extract isometry.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/ascending_superoperator/converged_mera_state.json"),
    )
    args = parser.parse_args()

    payload = build_and_export(
        model=args.model,
        boundary=args.boundary,
        L=args.L,
        max_bond=args.max_bond,
        fit_steps=args.fit_steps,
        layer=args.layer,
        seed=args.seed,
        output=args.output,
    )
    print(json.dumps(
        {
            "output": str(args.output),
            "fit_fidelity": payload["fit_fidelity"],
            "ground_energy": payload["ground_energy"],
            "superoperator_shape": [
                len(payload["ascending_superoperator"]),
                len(payload["ascending_superoperator"][0]) if payload["ascending_superoperator"] else 0,
            ],
        },
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
