#!/usr/bin/env python3
"""
P2 XXZ Boundary Test - ED Correctness Gate (OBC, observed-data only)

Authoritative scope gate based on finite-size scaling of observed entanglement
entropy S(L/2, L) from exact diagonalization (open boundary conditions).

Gate models:
  - Critical candidate:  S(L) = alpha * ln(L) + beta      (k=2)
  - Gapped candidate:    S(L) = S_inf - A * exp(-L/xi)    (k=3)

Model selection:
  - AICc is used for both models
  - delta_aicc_sat_minus_log = AICc_sat - AICc_log
  - Decision thresholds:
      delta <= -2  -> IN_SCOPE (saturating)
      delta >= +2  -> OUT_OF_SCOPE (log-linear)
      |delta| < 2  -> INCONCLUSIVE

Pooling:
  - ed_only: fit on ED series only
  - pool_if_overlap_pass: add DMRG-only points to fit series only when
    strict overlap calibration status is PASS

Usage:
    python3 exp_P2_XXZ_boundary_ed.py --deltas 0.5,1.0,1.1,1.5,2.0 --L_values 4,6,8
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

EPS_AIC = 1e-12
DELTA_AICC_THRESHOLD = 2.0
MIN_VIABLE_L_POINTS = 8

# Entropy is computed in bits (log2). Log-model uses natural log in L by specification.
ENTROPY_UNITS = "bits"
ALPHA_TO_C_EFF_OBC_FACTOR = 6.0 * math.log(2.0)  # c_eff ≈ 6 * alpha(bits) * ln(2) for OBC
OSCILLATION_ALT_THRESHOLD = 0.75
OSCILLATION_AICC_IMPROVE_MIN = 2.0

# Pre-registered sets for robust reporting.
DELTA_SET_BULK = [0.5, 0.8, 1.4, 2.0]
DELTA_SET_BOUNDARY = [0.95, 1.0, 1.05, 1.1]


def _delta_key(delta: float) -> float:
    """Canonical key for matching delta values across sources."""
    return round(float(delta), 8)


def _load_dmrg_overlap_lookup(path: Path) -> Dict[float, Dict[int, Dict[str, Any]]]:
    """
    Load DMRG overlap data as mapping: delta -> {L -> {status, S}}.

    Supported JSON formats:
      1) {"data":[{"delta":1.0,"L":12,"S_entropy":0.7}, ...]}
      2) [{"delta":1.0,"L":12,"S_entropy":0.7}, ...]
      3) {"1.0":{"12":0.7,"14":0.8}, "1.1":{"12":0.6}}
    """
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    lookup: Dict[float, Dict[int, Dict[str, Any]]] = {}

    def put(delta_v: Any, L_v: Any, S_v: Any = None, status_v: Any = None) -> None:
        d = _delta_key(float(delta_v))
        L = int(L_v)
        status = str(status_v).upper() if status_v is not None else "OK"
        entry: Dict[str, Any] = {"status": status, "S": None}
        # Guardrail: only accept numeric S when status is OK.
        if status == "OK" and S_v is not None:
            entry["S"] = float(S_v)
        lookup.setdefault(d, {})[L] = entry

    if isinstance(raw, dict) and "data" in raw and isinstance(raw["data"], list):
        for row in raw["data"]:
            if not isinstance(row, dict):
                continue
            if "delta" in row and "L" in row:
                put(
                    row["delta"],
                    row["L"],
                    row.get("S_entropy", row.get("S")),
                    row.get("status", "OK"),
                )
        return lookup

    if isinstance(raw, list):
        for row in raw:
            if not isinstance(row, dict):
                continue
            if "delta" in row and "L" in row:
                put(
                    row["delta"],
                    row["L"],
                    row.get("S_entropy", row.get("S")),
                    row.get("status", "OK"),
                )
        return lookup

    if isinstance(raw, dict):
        for d_key, l_map in raw.items():
            if not isinstance(l_map, dict):
                continue
            for L_key, s_or_obj in l_map.items():
                if isinstance(s_or_obj, dict):
                    put(d_key, L_key, s_or_obj.get("S_entropy", s_or_obj.get("S")), s_or_obj.get("status", "OK"))
                else:
                    put(d_key, L_key, s_or_obj, "OK")
        return lookup

    raise ValueError(f"Unsupported DMRG overlap JSON format: {path}")


def _evaluate_overlap_calibration(
    delta: float,
    measurements: List[Dict[str, Any]],
    dmrg_lookup: Optional[Dict[float, Dict[int, Dict[str, Any]]]],
    overlap_L_values: List[int],
    abs_tol: float,
    rel_tol: float,
) -> Dict[str, Any]:
    """
    Strict overlap calibration (all overlap points must pass).

    Returns status in {"PASS","MISMATCH","MISSING"}.
    """
    diag = {
        "enabled": dmrg_lookup is not None,
        "delta": delta,
        "required_L_values": overlap_L_values,
        "abs_tol": abs_tol,
        "rel_tol": rel_tol,
        "strict_all_must_pass": True,
        "calibration_pass": False,
        "status": "PASS",
        "reason": "OK",
        "required_points": len(overlap_L_values),
        "checked_points": 0,
        "passed_points": 0,
        "failed_points": 0,
        "missing_points": 0,
        "error_points": 0,
        "overlap_points": [],
    }
    if dmrg_lookup is None:
        return diag

    ed_by_L = {int(m["L"]): float(m["S_entropy"]) for m in measurements}
    dmrg_by_L = dmrg_lookup.get(_delta_key(delta), {})

    for L in overlap_L_values:
        point = {"L": L}
        if L not in ed_by_L or L not in dmrg_by_L:
            diag["missing_points"] += 1
            point["status"] = "MISSING"
            point["S_ed"] = ed_by_L.get(L)
            point["S_dmrg"] = None
            point["dmrg_status"] = None
            diag["overlap_points"].append(point)
            continue

        s_ed = ed_by_L[L]
        dmrg_entry = dmrg_by_L[L]
        dmrg_status = str(dmrg_entry.get("status", "OK")).upper()
        s_dmrg = dmrg_entry.get("S", None)

        if dmrg_status != "OK" or s_dmrg is None:
            diag["error_points"] += 1
            point["status"] = "ERROR"
            point["S_ed"] = s_ed
            point["S_dmrg"] = None
            point["dmrg_status"] = dmrg_status
            point["delta_s"] = None
            point["rel_delta_s"] = None
            point["epsilon_abs"] = abs_tol
            point["epsilon_rel"] = rel_tol
            diag["overlap_points"].append(point)
            continue

        s_dmrg = float(s_dmrg)
        delta_s = abs(s_ed - s_dmrg)
        rel_s = delta_s / max(1.0, abs(s_ed))
        pass_abs = delta_s <= abs_tol
        pass_rel = rel_s <= rel_tol
        pass_point = pass_abs or pass_rel

        point.update(
            {
                "status": "PASS" if pass_point else "FAIL",
                "S_ed": s_ed,
                "S_dmrg": s_dmrg,
                "dmrg_status": dmrg_status,
                "delta_s": delta_s,
                "rel_delta_s": rel_s,
                "epsilon_abs": abs_tol,
                "epsilon_rel": rel_tol,
                "pass_abs": bool(pass_abs),
                "pass_rel": bool(pass_rel),
            }
        )
        diag["overlap_points"].append(point)
        diag["checked_points"] += 1
        if pass_point:
            diag["passed_points"] += 1
        else:
            diag["failed_points"] += 1

    if diag["error_points"] > 0:
        diag["status"] = "MISMATCH"
        diag["reason"] = "SOLVER_INPUT_ERROR"
    elif diag["missing_points"] > 0:
        diag["status"] = "MISSING"
        diag["reason"] = "SOLVER_OVERLAP_MISSING"
    elif diag["failed_points"] > 0:
        diag["status"] = "MISMATCH"
        diag["reason"] = "SOLVER_MISMATCH"
    else:
        diag["status"] = "PASS"
        diag["reason"] = "OK"

    diag["calibration_pass"] = bool(diag["status"] == "PASS")

    return diag


def resolve_output_dir(base_out: Path, run_id_value: str, mode: str = "append") -> Path:
    """Resolve output directory with append-or-overwrite semantics."""
    base_out = Path(base_out)
    if mode == "overwrite":
        base_out.mkdir(parents=True, exist_ok=True)
        return base_out

    if not base_out.exists():
        base_out.mkdir(parents=True, exist_ok=True)
        return base_out

    if not any(base_out.iterdir()):
        return base_out

    candidate = base_out / f"run_{run_id_value}"
    suffix = 1
    while candidate.exists():
        candidate = base_out / f"run_{run_id_value}_{suffix:02d}"
        suffix += 1
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


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


def _schmidt_probabilities(psi: np.ndarray, L: int, A_size: int) -> np.ndarray:
    """Return normalized Schmidt probabilities p_i = s_i^2."""
    # Reshape into bipartite form
    psi = psi.reshape(2**A_size, 2**(L - A_size))

    # Schmidt decomposition
    _, s, _ = np.linalg.svd(psi, full_matrices=False)
    p = np.square(s)
    p = p[p > 1e-14]
    p = p / np.sum(p)
    return p


def compute_entanglement_entropy(psi: np.ndarray, L: int, A_size: int) -> float:
    """Compute bipartite entanglement entropy S = -Tr(ρ_A log₂ ρ_A)."""
    p = _schmidt_probabilities(psi, L, A_size)
    entropy = -np.sum(p * np.log2(p + 1e-14))
    return float(entropy)


def compute_capacity_of_entanglement(psi: np.ndarray, L: int, A_size: int) -> float:
    """Compute capacity κ₂ = Var(ln ρ_A)."""
    p = _schmidt_probabilities(psi, L, A_size)
    log_vals = np.log(p + 1e-14)
    mean_log = np.sum(p * log_vals)
    var_log = np.sum(p * np.square(log_vals - mean_log))
    return float(var_log)


def run_xxz_ed(L: int, delta: float) -> Dict:
    """
    Run ED for XXZ model and compute entanglement quantities.

    Uses sparse Lanczos (SciPy eigsh) through quimb's sparse Heisenberg builder,
    which is practical up to the L-ranges used in this gate.
    """
    import quimb as qu
    from scipy.sparse.linalg import eigsh

    # XXZ with OBC: Jx=Jy=1, Jz=delta
    H_sparse = qu.ham_heis(L, j=(1.0, 1.0, float(delta)), cyclic=False, sparse=True, stype="csr")
    eigenvalues, eigenvectors = eigsh(H_sparse, k=1, which="SA", tol=1e-9, maxiter=200000)
    E0 = float(np.real(eigenvalues[0]))
    psi0 = np.asarray(eigenvectors[:, 0], dtype=np.complex128)
    
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
        "ed_solver": "sparse_lanczos",
    }


def _aic_bic_from_rss(rss: float, n: int, k: int) -> Dict:
    """Compute AIC/BIC with rss floor for numerical stability."""
    rss = max(float(rss), EPS_AIC)
    aic = n * math.log(rss / n) + 2 * k
    bic = n * math.log(rss / n) + k * math.log(n)
    return {"aic": float(aic), "bic": float(bic)}


def _aicc_from_aic(aic: float, n: int, k: int) -> float:
    """Small-sample corrected AIC."""
    denom = n - k - 1
    if denom <= 0:
        return float("inf")
    return float(aic + (2 * k * (k + 1)) / denom)


def _staggered_sign(L_values: List[int]) -> np.ndarray:
    """Return staggered factor (-1)^(L/2) for even-L finite-size oscillations."""
    return np.array([(-1) ** (int(L) // 2) for L in L_values], dtype=float)


def _alternation_fraction(residuals: np.ndarray) -> float:
    """Fraction of adjacent residual signs that alternate (+/-)."""
    s = np.sign(np.asarray(residuals, dtype=float))
    valid_idx = np.where(np.abs(s) > 0)[0]
    if len(valid_idx) < 2:
        return 0.0
    alt = 0
    total = 0
    prev = s[valid_idx[0]]
    for idx in valid_idx[1:]:
        cur = s[idx]
        total += 1
        if cur * prev < 0:
            alt += 1
        prev = cur
    return float(alt / total) if total > 0 else 0.0


def fit_log_scaling(L_values: List[int], S_values: List[float]) -> Dict:
    """Fit critical model S(L) = alpha*ln(L) + beta using natural log."""
    log_L = np.log(L_values)  # natural log by specification
    S = np.array(S_values)
    
    X = np.column_stack([log_L, np.ones_like(log_L)])
    coef, _, _, _ = np.linalg.lstsq(X, S, rcond=None)
    a, b = float(coef[0]), float(coef[1])
    
    yhat = X @ coef
    rss = float(np.sum((S - yhat)**2))
    
    n = len(L_values)
    k = 2
    scores = _aic_bic_from_rss(rss, n=n, k=k)
    aicc = _aicc_from_aic(scores["aic"], n=n, k=k)

    # Post-hoc interpretation only (not used for gating):
    c_eff_obc = float(ALPHA_TO_C_EFF_OBC_FACTOR * a)  # valid if critical and OBC, with S in bits

    return {
        "alpha": a,
        "beta": b,
        "rss": rss,
        "aic": scores["aic"],
        "aicc": aicc,
        "bic": scores["bic"],
        "n": n,
        "k": k,
        "log_base": "e",
        "entropy_units": ENTROPY_UNITS,
        "c_eff_obc_from_alpha_bits": c_eff_obc,
        "alpha_to_c_eff_obc_factor": float(ALPHA_TO_C_EFF_OBC_FACTOR),
        "fit_status": "OK",
        "optimizer": "linear_least_squares",
        "converged_clean": bool(np.isfinite(rss)),
    }


def fit_saturation(L_values: List[int], S_values: List[float]) -> Dict:
    """
    Fit gapped model S(L) = S_inf - A*exp(-L/xi).

    Parameter domains:
      S_inf >= 0, A >= 0, xi > 0
    """
    L = np.array(L_values, dtype=float)
    S = np.array(S_values, dtype=float)
    max_S = float(np.max(S))
    n = len(L_values)
    k = 3

    best = {
        "S_inf": float("nan"),
        "A": float("nan"),
        "xi": float("nan"),
        "rss": float("inf"),
        "aic": float("inf"),
        "aicc": float("inf"),
        "bic": float("inf"),
        "n": n,
        "k": k,
        "constraints": {"S_inf_gte_0": True, "A_gte_0": True, "xi_gt_0": True},
        "fit_status": "NO_VALID_CANDIDATE",
        "optimizer": "grid_search_closed_form_A",
        "candidates_total": 0,
        "candidates_valid": 0,
        "boundary_hits": {"S_inf": False, "A": False, "xi": False},
        "at_constraint": {"S_inf_gte_0": False, "A_gte_0": False, "xi_gt_0": False},
        "converged_clean": False,
    }

    # S_inf grid (must be >= max observed if model is monotone from below).
    s_inf_grid = np.linspace(max_S + 1e-6, max(max_S + 0.5, max_S * 1.5), 40)
    # xi grid over broad physically-reasonable range for finite chains.
    xi_grid = np.logspace(np.log10(0.5), np.log10(max(2.0 * np.max(L), 4.0)), 60)

    total_candidates = 0
    valid_candidates = 0

    for S_inf in s_inf_grid:
        if S_inf < 0:
            continue
        for xi in xi_grid:
            total_candidates += 1
            if xi <= 0:
                continue
            phi = np.exp(-L / xi)
            denom = float(np.dot(phi, phi))
            if denom <= EPS_AIC:
                continue

            # Closed-form LS estimate of A for fixed (S_inf, xi).
            A = float(np.dot(phi, (S_inf - S)) / denom)
            if A < 0:
                continue
            valid_candidates += 1

            yhat = S_inf - A * phi
            rss = float(np.sum((S - yhat) ** 2))
            scores = _aic_bic_from_rss(rss, n=n, k=k)
            aicc = _aicc_from_aic(scores["aic"], n=n, k=k)

            if rss < best["rss"]:
                best = {
                    "S_inf": float(S_inf),
                    "A": float(A),
                    "xi": float(xi),
                    "rss": rss,
                    "aic": scores["aic"],
                    "aicc": aicc,
                    "bic": scores["bic"],
                    "n": n,
                    "k": k,
                    "constraints": {"S_inf_gte_0": True, "A_gte_0": True, "xi_gt_0": True},
                    "fit_status": "OK",
                    "optimizer": "grid_search_closed_form_A",
                }

    best["candidates_total"] = total_candidates
    best["candidates_valid"] = valid_candidates
    if best["fit_status"] == "OK":
        s_inf_min = float(s_inf_grid[0])
        s_inf_max = float(s_inf_grid[-1])
        xi_min = float(xi_grid[0])
        xi_max = float(xi_grid[-1])

        hit_s_inf = abs(best["S_inf"] - s_inf_min) <= 1e-10 or abs(best["S_inf"] - s_inf_max) <= 1e-10
        hit_xi = abs(best["xi"] - xi_min) <= 1e-10 or abs(best["xi"] - xi_max) <= 1e-10
        hit_A = abs(best["A"]) <= 1e-12

        best["boundary_hits"] = {"S_inf": bool(hit_s_inf), "A": bool(hit_A), "xi": bool(hit_xi)}
        best["at_constraint"] = {
            "S_inf_gte_0": bool(abs(best["S_inf"]) <= 1e-12),
            "A_gte_0": bool(hit_A),
            "xi_gt_0": bool(abs(best["xi"] - xi_min) <= 1e-10),
        }
        best["converged_clean"] = bool(not (hit_s_inf or hit_A or hit_xi))
    else:
        best["boundary_hits"] = {"S_inf": False, "A": False, "xi": False}
        best["at_constraint"] = {"S_inf_gte_0": False, "A_gte_0": False, "xi_gt_0": False}
        best["converged_clean"] = False

    return best


def fit_log_scaling_with_oscillation(L_values: List[int], S_values: List[float]) -> Dict:
    """
    Fit critical model with staggered finite-size correction:
      S(L) = alpha*ln(L) + beta + B*(-1)^(L/2)*exp(-L/lambda)
    """
    L = np.array(L_values, dtype=float)
    S = np.array(S_values, dtype=float)
    signs = _staggered_sign(L_values)
    n = len(L_values)
    k = 4

    best = {
        "alpha": float("nan"),
        "beta": float("nan"),
        "B": float("nan"),
        "lambda": float("nan"),
        "rss": float("inf"),
        "aic": float("inf"),
        "aicc": float("inf"),
        "bic": float("inf"),
        "n": n,
        "k": k,
        "fit_status": "NO_VALID_CANDIDATE",
        "optimizer": "grid_lambda_linear_least_squares",
        "converged_clean": False,
        "boundary_hits": {"lambda": False},
    }

    lambda_grid = np.logspace(np.log10(0.5), np.log10(max(2.0 * np.max(L), 4.0)), 80)
    logL = np.log(L)
    for lam in lambda_grid:
        if lam <= 0:
            continue
        osc = signs * np.exp(-L / lam)
        X = np.column_stack([logL, np.ones_like(logL), osc])
        coef, _, _, _ = np.linalg.lstsq(X, S, rcond=None)
        alpha, beta, B = float(coef[0]), float(coef[1]), float(coef[2])
        yhat = X @ coef
        rss = float(np.sum((S - yhat) ** 2))
        scores = _aic_bic_from_rss(rss, n=n, k=k)
        aicc = _aicc_from_aic(scores["aic"], n=n, k=k)
        if rss < best["rss"]:
            best = {
                "alpha": alpha,
                "beta": beta,
                "B": B,
                "lambda": float(lam),
                "rss": rss,
                "aic": scores["aic"],
                "aicc": aicc,
                "bic": scores["bic"],
                "n": n,
                "k": k,
                "fit_status": "OK",
                "optimizer": "grid_lambda_linear_least_squares",
            }

    if best["fit_status"] == "OK":
        lam_min = float(lambda_grid[0])
        lam_max = float(lambda_grid[-1])
        hit_lam = abs(best["lambda"] - lam_min) <= 1e-10 or abs(best["lambda"] - lam_max) <= 1e-10
        best["boundary_hits"] = {"lambda": bool(hit_lam)}
        best["converged_clean"] = bool(not hit_lam)
    else:
        best["boundary_hits"] = {"lambda": False}
        best["converged_clean"] = False

    return best


def fit_saturation_with_oscillation(L_values: List[int], S_values: List[float]) -> Dict:
    """
    Fit gapped model with staggered finite-size correction:
      S(L) = S_inf - A*exp(-L/xi) + B*(-1)^(L/2)*exp(-L/lambda)
    """
    L = np.array(L_values, dtype=float)
    S = np.array(S_values, dtype=float)
    signs = _staggered_sign(L_values)
    n = len(L_values)
    k = 5
    max_S = float(np.max(S))

    best = {
        "S_inf": float("nan"),
        "A": float("nan"),
        "B": float("nan"),
        "xi": float("nan"),
        "lambda": float("nan"),
        "rss": float("inf"),
        "aic": float("inf"),
        "aicc": float("inf"),
        "bic": float("inf"),
        "n": n,
        "k": k,
        "constraints": {"S_inf_gte_0": True, "A_gte_0": True, "xi_gt_0": True, "lambda_gt_0": True},
        "fit_status": "NO_VALID_CANDIDATE",
        "optimizer": "grid_Sinf_xi_lambda_linear_least_squares",
        "converged_clean": False,
        "boundary_hits": {"S_inf": False, "A": False, "xi": False, "lambda": False},
    }

    s_inf_grid = np.linspace(max_S + 1e-6, max(max_S + 0.5, max_S * 1.5), 24)
    xi_grid = np.logspace(np.log10(0.5), np.log10(max(2.0 * np.max(L), 4.0)), 36)
    lambda_grid = np.logspace(np.log10(0.5), np.log10(max(2.0 * np.max(L), 4.0)), 24)

    for S_inf in s_inf_grid:
        if S_inf < 0:
            continue
        target = S - S_inf
        for xi in xi_grid:
            if xi <= 0:
                continue
            f1 = -np.exp(-L / xi)  # coefficient is A>=0
            for lam in lambda_grid:
                if lam <= 0:
                    continue
                f2 = signs * np.exp(-L / lam)  # coefficient is B (unconstrained)
                X = np.column_stack([f1, f2])
                coef, _, _, _ = np.linalg.lstsq(X, target, rcond=None)
                A, B = float(coef[0]), float(coef[1])
                if A < 0:
                    continue
                yhat = S_inf + X @ coef
                rss = float(np.sum((S - yhat) ** 2))
                scores = _aic_bic_from_rss(rss, n=n, k=k)
                aicc = _aicc_from_aic(scores["aic"], n=n, k=k)
                if rss < best["rss"]:
                    best = {
                        "S_inf": float(S_inf),
                        "A": float(A),
                        "B": float(B),
                        "xi": float(xi),
                        "lambda": float(lam),
                        "rss": rss,
                        "aic": scores["aic"],
                        "aicc": aicc,
                        "bic": scores["bic"],
                        "n": n,
                        "k": k,
                        "constraints": {"S_inf_gte_0": True, "A_gte_0": True, "xi_gt_0": True, "lambda_gt_0": True},
                        "fit_status": "OK",
                        "optimizer": "grid_Sinf_xi_lambda_linear_least_squares",
                    }

    if best["fit_status"] == "OK":
        s_inf_min = float(s_inf_grid[0])
        s_inf_max = float(s_inf_grid[-1])
        xi_min = float(xi_grid[0])
        xi_max = float(xi_grid[-1])
        lam_min = float(lambda_grid[0])
        lam_max = float(lambda_grid[-1])
        hit_s_inf = abs(best["S_inf"] - s_inf_min) <= 1e-10 or abs(best["S_inf"] - s_inf_max) <= 1e-10
        hit_xi = abs(best["xi"] - xi_min) <= 1e-10 or abs(best["xi"] - xi_max) <= 1e-10
        hit_lam = abs(best["lambda"] - lam_min) <= 1e-10 or abs(best["lambda"] - lam_max) <= 1e-10
        hit_A = abs(best["A"]) <= 1e-12
        best["boundary_hits"] = {"S_inf": bool(hit_s_inf), "A": bool(hit_A), "xi": bool(hit_xi), "lambda": bool(hit_lam)}
        best["converged_clean"] = bool(not (hit_s_inf or hit_A or hit_xi or hit_lam))
    else:
        best["boundary_hits"] = {"S_inf": False, "A": False, "xi": False, "lambda": False}
        best["converged_clean"] = False

    return best


def run_xxz_boundary_test(cfg: Dict) -> Dict:
    """Run XXZ boundary correctness gate across Δ values."""
    if any(L % 2 != 0 for L in cfg["L_values"]):
        raise ValueError("L_values must be even so half-chain partition (ell=L/2) stays integer.")

    # Parity control diagnostic: OBC + half-chain cut can show strong parity oscillations on mixed-even grids.
    L_req = list(cfg["L_values"])
    if all((L % 4) == 0 for L in L_req):
        parity_control = "L_mod_4_eq_0"
    elif all((L % 4) == 2 for L in L_req):
        parity_control = "L_mod_4_eq_2"
    else:
        parity_control = "mixed_even"
    cfg["parity_control"] = parity_control
    if parity_control == "mixed_even":
        print("[WARN] L grid is mixed-even. OBC half-chain entropy can exhibit strong parity oscillations.")
        print("       Recommended: use L ≡ 0 (mod 4) (or L ≡ 2 (mod 4)) OR enable oscillation_mode=auto/force.")

    ed_max_L = cfg.get("ed_max_L")
    if ed_max_L is not None:
        ed_L_schedule = [L for L in cfg["L_values"] if L <= ed_max_L]
    else:
        ed_L_schedule = list(cfg["L_values"])
    if not ed_L_schedule:
        raise ValueError("No ED L points remain after applying ed_max_L.")

    n_points = len(ed_L_schedule)
    if n_points < MIN_VIABLE_L_POINTS:
        print(
            f"[WARN] Only {n_points} L-points provided; minimum viable is {MIN_VIABLE_L_POINTS} "
            "for stable AICc discrimination."
        )

    print("=" * 60)
    print("XXZ BOUNDARY TEST (ED Correctness Gate)")
    print("=" * 60)
    print("[GATE] Uses observed S(L/2, L) data only; theory formulas are interpretation-only.")
    print(f"L values (requested): {cfg['L_values']}")
    print(f"L values (ED schedule): {ed_L_schedule}")
    print(f"Δ values: {cfg['deltas']}")
    print(f"[PARITY] parity_control={parity_control}")
    pooling_mode = cfg.get("pooling_mode", "ed_only")
    print(f"[POOLING] mode={pooling_mode}")
    dmrg_lookup = None
    if cfg.get("dmrg_overlap_json"):
        dmrg_path = Path(cfg["dmrg_overlap_json"])
        dmrg_lookup = _load_dmrg_overlap_lookup(dmrg_path)
        print(f"[CALIBRATION] DMRG overlap enabled: {dmrg_path}")
        print(
            f"[CALIBRATION] overlap L values={cfg['overlap_L_values']}, "
            f"abs_tol={cfg['overlap_abs_tol']}, rel_tol={cfg['overlap_rel_tol']}, strict=all-must-pass"
        )
    print()
    
    results = []
    
    for delta in cfg["deltas"]:
        print(f"\n[Δ = {delta}]")
        expected = "IN_SCOPE" if delta > 1 else "OUT_OF_SCOPE"
        print(f"  Expected scope: {expected} ({'gapped' if delta > 1 else 'critical'})")
        
        # Run ED for each L
        measurements = []
        for L in ed_L_schedule:
            print(f"  L={L}...", end=" ")
            result = run_xxz_ed(L, delta)
            measurements.append(result)
            print(f"S={result['S_entropy']:.4f}, κ₂={result['kappa2']:.4f}")

        overlap_diag = _evaluate_overlap_calibration(
            delta=delta,
            measurements=measurements,
            dmrg_lookup=dmrg_lookup,
            overlap_L_values=cfg["overlap_L_values"],
            abs_tol=cfg["overlap_abs_tol"],
            rel_tol=cfg["overlap_rel_tol"],
        )
        if overlap_diag["enabled"]:
            print(
                f"  [CALIBRATION] status={overlap_diag['status']} "
                f"passed={overlap_diag['passed_points']}/{overlap_diag['required_points']} "
                f"failed={overlap_diag['failed_points']} missing={overlap_diag['missing_points']} "
                f"errors={overlap_diag['error_points']}"
            )
        
        # Build fit series according to pooling mode.
        ed_L_values = [int(m["L"]) for m in measurements]
        ed_S_values = [float(m["S_entropy"]) for m in measurements]
        fit_map = {L: S for L, S in zip(ed_L_values, ed_S_values)}
        source_by_L = {L: "ED" for L in ed_L_values}
        fit_source = "ED_ONLY"
        pooling_applied = False
        pooling_reason = "POOLING_DISABLED"

        # Only allow pooling points that are in the pre-registered L grid.
        allowed_L = set(int(x) for x in cfg["L_values"])

        if pooling_mode == "pool_if_overlap_pass":
            if dmrg_lookup is None:
                pooling_reason = "NO_DMRG_INPUT"
            elif overlap_diag["status"] != "PASS":
                pooling_reason = overlap_diag["reason"]
            else:
                dmrg_by_L = dmrg_lookup.get(_delta_key(delta), {})
                added = 0
                for L_dmrg, dmrg_entry in dmrg_by_L.items():
                    L_int = int(L_dmrg)
                    if L_int not in allowed_L:
                        # Never pool unregistered L values (prevents drift from DMRG JSON contents).
                        continue
                    dmrg_status = str(dmrg_entry.get("status", "OK")).upper()
                    s_dmrg = dmrg_entry.get("S", None)
                    if L_int in fit_map:
                        continue
                    if dmrg_status != "OK" or s_dmrg is None:
                        continue
                    fit_map[L_int] = float(s_dmrg)
                    source_by_L[L_int] = "DMRG"
                    added += 1
                if added > 0:
                    fit_source = "ED_PLUS_DMRG"
                    pooling_applied = True
                    pooling_reason = "POOL_APPLIED_OVERLAP_PASS"
                else:
                    pooling_reason = "NO_NEW_DMRG_POINTS"

        fit_L_values = sorted(fit_map.keys())
        fit_S_values = [fit_map[L] for L in fit_L_values]

        # Fit baseline models on selected fit series.
        log_model_base = fit_log_scaling(fit_L_values, fit_S_values)
        sat_model_base = fit_saturation(fit_L_values, fit_S_values)

        # Detect strong staggered finite-size oscillation from baseline log residuals.
        fit_L_arr = np.array(fit_L_values, dtype=float)
        fit_S_arr = np.array(fit_S_values, dtype=float)
        yhat_log_base = log_model_base["alpha"] * np.log(fit_L_arr) + log_model_base["beta"]
        residual_alt_fraction = _alternation_fraction(fit_S_arr - yhat_log_base)
        oscillation_detected = bool(
            residual_alt_fraction >= cfg["oscillation_alt_threshold"] and len(fit_L_values) >= 6
        )

        log_model_osc = None
        sat_model_osc = None
        use_oscillation_corrected = False
        aicc_improve_log = None
        aicc_improve_sat = None

        if cfg["oscillation_mode"] != "off" and oscillation_detected:
            log_model_osc = fit_log_scaling_with_oscillation(fit_L_values, fit_S_values)
            sat_model_osc = fit_saturation_with_oscillation(fit_L_values, fit_S_values)
            if log_model_osc["fit_status"] == "OK" and sat_model_osc["fit_status"] == "OK":
                aicc_improve_log = float(log_model_base["aicc"] - log_model_osc["aicc"])
                aicc_improve_sat = float(sat_model_base["aicc"] - sat_model_osc["aicc"])
                if cfg["oscillation_mode"] == "force":
                    use_oscillation_corrected = True
                else:
                    use_oscillation_corrected = bool(
                        max(aicc_improve_log, aicc_improve_sat) >= cfg["oscillation_aicc_improve_min"]
                    )

        fit_variant = "oscillation_corrected" if use_oscillation_corrected else "base"
        log_model = log_model_osc if use_oscillation_corrected and log_model_osc is not None else log_model_base
        sat_model = sat_model_osc if use_oscillation_corrected and sat_model_osc is not None else sat_model_base

        aic_saturating = float(sat_model["aic"])
        aic_loglinear = float(log_model["aic"])
        aicc_saturating = float(sat_model["aicc"])
        aicc_loglinear = float(log_model["aicc"])
        delta_aic_sat_minus_log = aic_saturating - aic_loglinear
        delta_aic_log_minus_sat = -delta_aic_sat_minus_log
        delta_aicc_sat_minus_log = aicc_saturating - aicc_loglinear
        delta_aicc_log_minus_sat = -delta_aicc_sat_minus_log

        if aicc_saturating + EPS_AIC < aicc_loglinear:
            preferred_model = "saturating"
        elif aicc_loglinear + EPS_AIC < aicc_saturating:
            preferred_model = "log-linear"
        else:
            preferred_model = "tie"

        # Gate decision based on AICc threshold band.
        if overlap_diag["enabled"] and overlap_diag["status"] != "PASS":
            gate_decision = "INCONCLUSIVE"
            gate_reason = overlap_diag["reason"]
        elif not np.isfinite(delta_aicc_sat_minus_log):
            gate_decision = "INCONCLUSIVE"
            gate_reason = "AICC_UNDEFINED"
        elif delta_aicc_sat_minus_log <= -DELTA_AICC_THRESHOLD:
            gate_decision = "IN_SCOPE"
            gate_reason = "AICC_SATURATION_PREFERRED"
        elif delta_aicc_sat_minus_log >= DELTA_AICC_THRESHOLD:
            gate_decision = "OUT_OF_SCOPE"
            gate_reason = "AICC_LOG_PREFERRED"
        else:
            gate_decision = "INCONCLUSIVE"
            gate_reason = "AICC_BAND_INDETERMINATE"

        aicc_used_for_gating = not (overlap_diag["enabled"] and overlap_diag["status"] != "PASS")

        # Verdict semantics:
        # - IN_SCOPE => ACCEPT
        # - OUT_OF_SCOPE => REJECT
        # - INCONCLUSIVE => INCONCLUSIVE
        expected_preferred_model = "saturating" if delta > 1 else "log-linear"
        expected_sign_sat_minus_log = "negative" if expected_preferred_model == "saturating" else "positive"
        actual_sign_sat_minus_log = (
            "negative" if delta_aic_sat_minus_log < -EPS_AIC else
            "positive" if delta_aic_sat_minus_log > EPS_AIC else
            "zero"
        )
        if not np.isfinite(delta_aicc_sat_minus_log):
            actual_sign_aicc_sat_minus_log = "undefined"
        else:
            actual_sign_aicc_sat_minus_log = (
                "negative" if delta_aicc_sat_minus_log < -EPS_AIC else
                "positive" if delta_aicc_sat_minus_log > EPS_AIC else
                "zero"
            )

        if gate_decision == "INCONCLUSIVE":
            verdict = "INCONCLUSIVE"
            scope_correct = None
        else:
            verdict = "ACCEPT" if gate_decision == "IN_SCOPE" else "REJECT"
            scope_correct = (gate_decision == expected)

        print(
            f"  ΔAICc(sat-log) = {delta_aicc_sat_minus_log:.4f} "
            f"({'saturation' if preferred_model == 'saturating' else 'log growth' if preferred_model == 'log-linear' else 'tie'})"
        )
        print(
            f"  [FIT] variant={fit_variant}, oscillation_detected={oscillation_detected}, "
            f"alt_fraction={residual_alt_fraction:.3f}"
        )
        print(
            f"  [POOLING] source={fit_source}, applied={pooling_applied}, "
            f"n_fit={len(fit_L_values)}, reason={pooling_reason}"
        )
        print(
            f"  Gate decision: {gate_decision}, reason={gate_reason}, "
            f"Verdict: {verdict}, Scope correct: {scope_correct}"
        )
        if not aicc_used_for_gating:
            print("  [GATING] AICc comparison computed for audit only; not used due to solver calibration failure.")
        
        results.append({
            "delta": delta,
            "expected_scope": expected,
            "measurements": measurements,
            "log_model": log_model,
            "sat_model": sat_model,
            "log_model_base": log_model_base,
            "sat_model_base": sat_model_base,
            "log_model_osc": log_model_osc,
            "sat_model_osc": sat_model_osc,
            "fit_variant": fit_variant,
            "oscillation_diagnostics": {
                "mode": cfg["oscillation_mode"],
                "alternation_fraction": float(residual_alt_fraction),
                "alternation_threshold": float(cfg["oscillation_alt_threshold"]),
                "detected": oscillation_detected,
                "aicc_improve_log": aicc_improve_log,
                "aicc_improve_sat": aicc_improve_sat,
                "aicc_improve_threshold": float(cfg["oscillation_aicc_improve_min"]),
                "used_for_gating": use_oscillation_corrected,
            },
            # Legacy key kept for compatibility:
            # delta_aic == AIC_saturating - AIC_loglinear
            "delta_aic": float(delta_aic_sat_minus_log),
            "delta_aic_sat_minus_log": float(delta_aic_sat_minus_log),
            "delta_aic_log_minus_sat": float(delta_aic_log_minus_sat),
            "delta_aicc_sat_minus_log": float(delta_aicc_sat_minus_log),
            "delta_aicc_log_minus_sat": float(delta_aicc_log_minus_sat),
            "aic_saturating": aic_saturating,
            "aic_loglinear": aic_loglinear,
            "aicc_saturating": aicc_saturating,
            "aicc_loglinear": aicc_loglinear,
            "preferred_model": preferred_model,
            "expected_preferred_model": expected_preferred_model,
            "expected_sign_sat_minus_log": expected_sign_sat_minus_log,
            "actual_sign_sat_minus_log": actual_sign_sat_minus_log,
            "actual_sign_aicc_sat_minus_log": actual_sign_aicc_sat_minus_log,
            "gate_decision": gate_decision,
            "gate_reason": gate_reason,
            "solver_calibration": overlap_diag,
            "model_selection_audit": {
                "aicc_used_for_gating": bool(aicc_used_for_gating),
                "aicc_evaluated_for_audit": True,
                "blocked_by_solver_calibration": bool(not aicc_used_for_gating),
                "blocking_reason": overlap_diag["reason"] if not aicc_used_for_gating else None,
            },
            "fit_series": {
                "source": fit_source,
                "pooling_mode": pooling_mode,
                "pooling_applied": pooling_applied,
                "pooling_reason": pooling_reason,
                "n_fit_points": len(fit_L_values),
                "L_values": fit_L_values,
                "source_by_L": {str(L): source_by_L[L] for L in fit_L_values},
                "parity_control": parity_control,
            },
            "verdict": verdict,
            "scope_correct": scope_correct,
        })
    
    # Summary
    conclusive = [r for r in results if r["scope_correct"] is not None]
    inconclusive = [r for r in results if r["scope_correct"] is None]
    scope_matches = sum(1 for r in conclusive if r["scope_correct"])
    total = len(results)
    all_correct = (len(inconclusive) == 0 and scope_matches == total)
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Δ':>6} | {'Expected':>12} | {'ΔAICc':>10} | {'Verdict':>12} | {'Scope OK':>8}")
    print("-" * 70)
    for r in results:
        print(
            f"{r['delta']:>6.2f} | {r['expected_scope']:>12} | {r['delta_aicc_sat_minus_log']:>10.4f} | "
            f"{r['verdict']:>12} | {str(r['scope_correct']):>8}"
        )
    print("-" * 70)
    print(f"Scope matches (conclusive only): {scope_matches}/{len(conclusive)}")
    print(f"Inconclusive points: {len(inconclusive)}")
    print(f"Overall: {'SCOPE_VALIDATED' if all_correct else 'SCOPE_MISMATCH' if len(inconclusive) == 0 else 'SCOPE_INCONCLUSIVE'}")
    reason_counts: Dict[str, int] = {}
    for r in results:
        reason_counts[r["gate_reason"]] = reason_counts.get(r["gate_reason"], 0) + 1
    
    return {
        "metadata": {
            "test": "P2_XXZ_BOUNDARY_ED",
            "version": "2.0.0",
            "method": "exact_diagonalization",
            "validation_mode": "physics_truth_baseline",
            "framework_correctness_gate": True,
            "boundary_condition": "OBC",
            "partition_rule": "half_chain_ell_equals_L_over_2",
            "log_model": "S(L)=alpha*ln(L)+beta",
            "sat_model": "S(L)=S_inf-A*exp(-L/xi)",
            "entropy_units": ENTROPY_UNITS,
            "alpha_to_c_eff_obc_factor": float(ALPHA_TO_C_EFF_OBC_FACTOR),
            "parity_control": parity_control,
            "log_model_osc": "S(L)=alpha*ln(L)+beta+B*(-1)^(L/2)*exp(-L/lambda)",
            "sat_model_osc": "S(L)=S_inf-A*exp(-L/xi)+B*(-1)^(L/2)*exp(-L/lambda)",
            "decision_metric": "delta_aicc_sat_minus_log",
            "decision_threshold_abs": DELTA_AICC_THRESHOLD,
            "decision_rule": "delta<=-2:IN_SCOPE; delta>=+2:OUT_OF_SCOPE; otherwise:INCONCLUSIVE",
            "gate_data_source": "observed_S_L2_L_only",
            "even_L_only": True,
            "solver_overlap_required_for_pooling": True,
            "solver_overlap_strict_mode": "all_must_pass",
            "pooling_mode": pooling_mode,
            "oscillation_mode": cfg["oscillation_mode"],
            "oscillation_alt_threshold": cfg["oscillation_alt_threshold"],
            "oscillation_aicc_improve_min": cfg["oscillation_aicc_improve_min"],
        },
        "config": cfg,
        "results": results,
        "summary": {
            "scope_matches": scope_matches,
            "conclusive_total": len(conclusive),
            "inconclusive_total": len(inconclusive),
            "total": total,
            "all_correct": all_correct,
            "gate_reason_counts": reason_counts,
            "overall_verdict": "SCOPE_VALIDATED" if all_correct else ("SCOPE_MISMATCH" if len(inconclusive) == 0 else "SCOPE_INCONCLUSIVE"),
        }
    }


def main():
    p = argparse.ArgumentParser(description="P2 XXZ Boundary Test (ED Only)")
    p.add_argument(
        "--delta-set",
        choices=["custom", "bulk", "boundary"],
        default="custom",
        help="Use pre-registered Δ set: bulk or boundary; custom uses --deltas.",
    )
    p.add_argument("--deltas", default="0.5,1.0,1.1,1.5,2.0",
                   help="Comma-separated Δ values")
    p.add_argument("--L_values", default="4,6,8,10",
                   help="Comma-separated L values")
    p.add_argument(
        "--ed-max-L",
        type=int,
        default=None,
        help="Optional maximum L to run with ED (lets DMRG provide larger-L points for pooling).",
    )
    p.add_argument(
        "--pooling-mode",
        choices=["ed_only", "pool_if_overlap_pass"],
        default="ed_only",
        help=(
            "ed_only: fit gate models on ED points only; "
            "pool_if_overlap_pass: add DMRG-only points only if overlap calibration passes."
        ),
    )
    p.add_argument(
        "--dmrg-overlap-json",
        default=None,
        help="Optional JSON file with DMRG overlap entropy data for strict solver calibration.",
    )
    p.add_argument(
        "--overlap-L-values",
        default="12,14,16",
        help="Comma-separated overlap L values for ED/DMRG calibration.",
    )
    p.add_argument(
        "--overlap-abs-tol",
        type=float,
        default=1e-3,
        help="Absolute tolerance for |S_ED - S_DMRG| overlap checks.",
    )
    p.add_argument(
        "--overlap-rel-tol",
        type=float,
        default=1e-3,
        help="Relative tolerance for overlap checks (used as secondary pass criterion).",
    )
    p.add_argument(
        "--oscillation-mode",
        choices=["auto", "off", "force"],
        default="auto",
        help=(
            "auto: use oscillation-corrected fits when strong alternation is detected and AICc improves; "
            "off: use baseline fits only; force: use oscillation-corrected fits when detectable."
        ),
    )
    p.add_argument(
        "--oscillation-alt-threshold",
        type=float,
        default=OSCILLATION_ALT_THRESHOLD,
        help="Minimum residual sign-alternation fraction to consider staggered oscillation detected.",
    )
    p.add_argument(
        "--oscillation-aicc-improve-min",
        type=float,
        default=OSCILLATION_AICC_IMPROVE_MIN,
        help="Minimum AICc improvement required in auto mode to switch to oscillation-corrected gating.",
    )
    p.add_argument("--output", default=None, help="Output directory (base directory, optional)")
    p.add_argument(
        "--output-mode",
        choices=["append", "overwrite"],
        default="append",
        help="append: create run_<id> subdir when output exists; overwrite: write directly into output dir",
    )
    a = p.parse_args()

    if a.delta_set == "bulk":
        deltas = DELTA_SET_BULK
    elif a.delta_set == "boundary":
        deltas = DELTA_SET_BOUNDARY
    else:
        deltas = [float(x) for x in a.deltas.split(",")]

    cfg = {
        "deltas": deltas,
        "L_values": [int(x) for x in a.L_values.split(",")],
        "ed_max_L": int(a.ed_max_L) if a.ed_max_L is not None else None,
        "delta_set": a.delta_set,
        "min_viable_L_points": MIN_VIABLE_L_POINTS,
        "pooling_mode": a.pooling_mode,
        "dmrg_overlap_json": a.dmrg_overlap_json,
        "overlap_L_values": [int(x) for x in a.overlap_L_values.split(",") if x.strip()],
        "overlap_abs_tol": float(a.overlap_abs_tol),
        "overlap_rel_tol": float(a.overlap_rel_tol),
        "oscillation_mode": a.oscillation_mode,
        "oscillation_alt_threshold": float(a.oscillation_alt_threshold),
        "oscillation_aicc_improve_min": float(a.oscillation_aicc_improve_min),
    }
    
    res = run_xxz_boundary_test(cfg)
    
    if a.output:
        run_id_value = f"ed_{os.urandom(4).hex()}"
        out_dir = resolve_output_dir(Path(a.output), run_id_value, mode=a.output_mode)
        with open(out_dir / "xxz_boundary_results.json", "w") as f:
            json.dump(res, f, indent=2)
        print(f"\nResults written to {out_dir}")


if __name__ == "__main__":
    main()
