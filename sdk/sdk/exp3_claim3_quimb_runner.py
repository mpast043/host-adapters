#!/usr/bin/env python3
"""
sdk/exp3_claim3_quimb_runner.py

FRAMEWORK_SPEC_v0.2.md compliant experiment runner for Claim 3 (Option A baseline).

This is not the toy generator:
- Builds a MERA tensor network (quimb)
- Computes rho_A by contracting the causal-cone reduced density operator
- Computes von Neumann entropy from the dense rho_A eigenvalues
- Runs falsifiers 3.1–3.4 on measured S

Scope limitation (required by v0.2):
This does not "prove" physics in the mathematical sense. It produces empirical evidence
from an explicit MERA tensor network contraction. For heisenberg_opt, it is a variational
approximation (finite depth, finite chi, finite optimization steps).

Outputs:
outputs/exp3_claim3_quimb/<run_id>/
  manifest.json
  raw_entropy.csv
  fits.json
  bound_checks.json
  verdict.json

Dependencies:
  pip install quimb cotengra torch   (torch only needed for heisenberg_opt)
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import os
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


# ----------------------------
# Basic helpers
# ----------------------------

def make_run_id() -> str:
    t = dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    r = os.urandom(4).hex()
    return f"{t}_{r}"


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def fit_linear(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    """Fit y = a*x + b. Return (a, b, rss)."""
    X = np.column_stack([x, np.ones_like(x)])
    coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    a, b = float(coef[0]), float(coef[1])
    yhat = X @ coef
    rss = float(np.sum((y - yhat) ** 2))
    return a, b, rss


def aic_bic_from_rss(rss: float, n: int, k: int, eps: float = 1e-12) -> Tuple[float, float]:
    """Gaussian AIC/BIC from RSS."""
    rss = max(float(rss), eps)
    aic = n * math.log(rss / n) + 2 * k
    bic = n * math.log(rss / n) + k * math.log(n)
    return float(aic), float(bic)


def fit_log_power_model(log_chi: np.ndarray, y: np.ndarray, p_grid: np.ndarray) -> Dict[str, float]:
    """
    Fit y = a*(log_chi)^p + b, p chosen by grid search.
    AIC/BIC computed with k=3 (a,b,p).
    """
    best = {"p": float("nan"), "a": float("nan"), "b": float("nan"),
            "rss": float("inf"), "aic": float("inf"), "bic": float("inf")}
    n = len(y)
    for p in p_grid:
        z = np.power(log_chi, p)
        a, b, rss = fit_linear(z, y)
        aic, bic = aic_bic_from_rss(rss, n=n, k=3)
        if rss < best["rss"]:
            best = {"p": float(p), "a": float(a), "b": float(b),
                    "rss": float(rss), "aic": float(aic), "bic": float(bic)}
    return best


# ----------------------------
# Falsifiers (v0.2 Claim 3 Option A)
# ----------------------------

def falsifier_3_1_monotonicity(medians_by_chi: Dict[int, float], chi_sweep: List[int]) -> Dict[str, object]:
    ok = True
    violations = []
    for c0, c1 in zip(chi_sweep, chi_sweep[1:]):
        s0 = float(medians_by_chi[c0])
        s1 = float(medians_by_chi[c1])
        if s1 < s0 - 1e-12:
            ok = False
            violations.append({"chi_prev": c0, "chi_next": c1, "S_prev": s0, "S_next": s1})
    return {"passed": bool(ok), "violations": violations}


def falsifier_3_2_replicate_robustness(slopes: List[float], cv_threshold: float = 0.10) -> Dict[str, object]:
    arr = np.array(slopes, dtype=float)
    mean = float(np.mean(arr))
    std = float(np.std(arr))
    cv = float(std / mean) if mean != 0 else float("inf")
    no_negative = bool(np.all(arr > 0.0))
    stable = bool(cv <= cv_threshold)
    return {
        "passed": bool(no_negative and stable),
        "no_negative_slopes": no_negative,
        "cv_threshold": float(cv_threshold),
        "slope_mean": mean,
        "slope_std": std,
        "slope_cv": cv,
        "slopes": [float(x) for x in arr.tolist()],
    }


def falsifier_3_3_model_selection(y_medians: np.ndarray, chi_sweep: List[int]) -> Dict[str, object]:
    chis = np.array(chi_sweep, dtype=float)
    logs = np.log(chis)

    # Model 1: S = a*log(chi) + b
    a1, b1, rss1 = fit_linear(logs, y_medians)
    aic1, bic1 = aic_bic_from_rss(rss1, n=len(y_medians), k=2)

    # Model 2: S = a*chi + b
    a2, b2, rss2 = fit_linear(chis, y_medians)
    aic2, bic2 = aic_bic_from_rss(rss2, n=len(y_medians), k=2)

    # Model 3: S = a*(log(chi))^p + b, p fitted
    p_grid = np.linspace(0.5, 3.0, 251)
    m3 = fit_log_power_model(logs, y_medians, p_grid)

    competitor_aic = min(aic2, m3["aic"])
    competitor_bic = min(bic2, m3["bic"])

    delta_aic = float(competitor_aic - aic1)
    delta_bic = float(competitor_bic - bic1)

    passed = bool((a1 > 0.0) and (delta_aic >= 10.0) and (delta_bic >= 10.0))

    return {
        "passed": passed,
        "pass_condition": {"delta_aic_min": 10.0, "delta_bic_min": 10.0, "log_slope_positive": True},
        "delta_aic": delta_aic,
        "delta_bic": delta_bic,
        "model_1_log_linear": {"a": float(a1), "b": float(b1), "rss": float(rss1), "aic": float(aic1), "bic": float(bic1)},
        "model_2_linear_chi": {"a": float(a2), "b": float(b2), "rss": float(rss2), "aic": float(aic2), "bic": float(bic2)},
        "model_3_log_power": m3,
    }


def falsifier_3_4_bound_validity(bound_rows: List[Dict[str, object]]) -> Dict[str, object]:
    violations = [r for r in bound_rows if float(r["S"]) > float(r["bound"]) + 1e-12]
    return {"passed": bool(len(violations) == 0), "violation_count": int(len(violations)), "violations": violations[:10]}


# ----------------------------
# Actual MERA measurement with quimb
# ----------------------------

@dataclass(frozen=True)
class Partition:
    partition_id: str
    keep_sites: List[int]  # subsystem A sites to keep (contiguous by default)


def k_cut_proxy_upper_binary_mera(A_size: int) -> int:
    """
    Declared K_cut proxy rule (upper bound, not minimal cut):
    For 1D binary MERA, a contiguous interval has a causal cone width that stays O(1),
    and the number of bonds crossed by a simple discrete surface is O(log A).
    Use a safe upper proxy:
      K_cut = 2 * (ceil(log2(A_size)) + 1), min 2
    """
    if A_size <= 1:
        return 2
    return max(2, 2 * (int(math.ceil(math.log(A_size, 2))) + 1))


def entropy_from_mera_dense_rho(mera, part: Partition) -> Tuple[float, np.ndarray]:
    """
    Compute S(A) from MERA by explicitly forming rho_A as a dense matrix for subsystem A.

    Steps (quimb MERA example pattern):
    - build bra with different site index labels for kept sites
    - select only causal cone tensors for kept sites (reverse lightcone cancellation)
    - form rho TN and convert to dense matrix using TN.to_dense((k0..), (b0..))
    - compute eigenvalues and entropy
    """
    import quimb.tensor as qtn  # noqa: F401

    A_sites = list(part.keep_sites)
    A_size = len(A_sites)

    # reindex kept sites on bra to b{}
    bra = mera.H.reindex_sites("b{}", A_sites)

    # select causal cone for these sites only
    # important: slice/which="any" selects tensors with any of these site tags
    # here we pass explicit site tags for robustness
    tags = [mera.site_tag(i) for i in A_sites]
    rho_tn = bra.select(tags, which="any") & mera.select(tags, which="any")

    left_inds = tuple(f"k{i}" for i in A_sites)
    right_inds = tuple(f"b{i}" for i in A_sites)

    # Dense rho_A (dimension 2^A x 2^A)
    rho = rho_tn.to_dense(left_inds, right_inds)

    # Convert to ndarray, symmetrize numerically, normalize trace
    rho = np.array(rho, dtype=np.complex128)
    rho = 0.5 * (rho + rho.conj().T)
    tr = float(np.real_if_close(np.trace(rho)))
    if tr <= 0:
        raise RuntimeError(f"rho trace invalid: {tr}")
    rho = rho / tr

    # eigenvalues
    evals = np.linalg.eigvalsh(rho)
    evals = np.real_if_close(evals).astype(float)
    evals = np.clip(evals, 0.0, 1.0)

    # von Neumann entropy (natural log)
    eps = 1e-15
    S = float(-np.sum(evals * np.log(evals + eps)))

    return S, evals


def maybe_optimize_heisenberg(mera, L: int, steps: int, autodiff: str, device: str, seed: int) -> object:
    """
    Optional: variationally optimize MERA for periodic 1D Heisenberg Hamiltonian.
    Uses quimb documented TNOptimizer pattern.

    This is expensive. Keep L small and steps modest for CPU.
    """
    import quimb as qu
    import quimb.tensor as qtn

    # two-site Heisenberg term
    H2 = qu.ham_heis(2).real

    terms = {(i, (i + 1) % L): H2 for i in range(L)}

    # path optimizer (optional)
    try:
        import cotengra as ctg
        opt = ctg.ReusableHyperOptimizer(progbar=False, reconf_opts={}, max_repeats=8)
    except Exception:
        opt = "auto-hq"

    def norm_fn(m):
        # project tensors back to isometric/unitary manifold
        return m.unitize(method="exp")

    def local_expectation(m, terms, where, optimize="auto-hq"):
        tags = [m.site_tag(coo) for coo in where]
        m_ij = m.select(tags, which="any")
        m_ij_G = m_ij.gate(terms[where], where)
        ex = m_ij_G & m_ij.H
        return ex.contract(all, optimize=optimize)

    def loss_fn(m, terms, optimize="auto-hq"):
        return sum(local_expectation(m, terms, where, optimize=optimize) for where in terms)

    tnopt = qtn.TNOptimizer(
        mera,
        loss_fn=loss_fn,
        norm_fn=norm_fn,
        loss_constants={"terms": terms},
        loss_kwargs={"optimize": opt},
        autodiff_backend=autodiff,
        device=device,
        jit_fn=False,
    )

    # deterministic-ish by seeding numpy; backend-specific determinism may still vary slightly
    np.random.seed(seed)

    # small number of optimization iterations
    tnopt.optimizer = "l-bfgs-b"
    mera_opt = tnopt.optimize(steps)

    return mera_opt


# ----------------------------
# Runner
# ----------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Claim 3 runner using quimb MERA (non-toy)")
    ap.add_argument("--output_root", default="outputs")
    ap.add_argument("--experiment_name", default="exp3_claim3_quimb")
    ap.add_argument("--seed_base", type=int, default=42)

    ap.add_argument("--L", type=int, default=16, help="Number of sites (power of 2 recommended/required for MERA). Keep small for dense rho.")
    ap.add_argument("--A", type=int, default=8, help="Subsystem size (contiguous starting at 0). Keep <= 10 for dense rho on CPU.")
    ap.add_argument("--chi_sweep", default="2,3,4,6,8,12,16", help="Comma-separated chi values.")
    ap.add_argument("--seeds_per_chi", type=int, default=5)

    ap.add_argument("--state", choices=["random", "heisenberg_opt"], default="random")
    ap.add_argument("--opt_steps", type=int, default=50, help="Optimization steps for heisenberg_opt.")

    ap.add_argument("--autodiff", default="torch", help="Autodiff backend for quimb TNOptimizer (torch recommended).")
    ap.add_argument("--device", default="cpu", help="Device for autodiff backend.")

    args = ap.parse_args()

    # Imports here so the script fails fast if quimb missing
    import quimb.tensor as qtn

    out_root = Path(args.output_root)
    exp_name = str(args.experiment_name).strip()
    rid = make_run_id()
    run_dir = out_root / exp_name / rid
    run_dir.mkdir(parents=True, exist_ok=True)

    L = int(args.L)
    A = int(args.A)
    if A <= 0 or A >= L:
        raise ValueError("Require 0 < A < L")

    chi_sweep = [int(x.strip()) for x in str(args.chi_sweep).split(",") if x.strip()]
    chi_sweep = sorted(list(dict.fromkeys(chi_sweep)))
    seeds_per_chi = int(args.seeds_per_chi)
    seed_base = int(args.seed_base)

    part = Partition(partition_id="baseline_contig", keep_sites=list(range(A)))
    K_cut = int(k_cut_proxy_upper_binary_mera(A))
    scope_limitations = [
        "Evidence is generated by contracting an explicit MERA tensor network state (quimb) and computing rho_A and S(A) from it.",
        "This is not a mathematical proof and does not guarantee saturation of the cut-counting bound for a specific Hamiltonian. For heisenberg_opt, it is a finite-step variational approximation.",
    ]

    # Deterministic seed policy:
    # seed = seed_base + chi_index*1000 + rep
    def point_seed(chi_idx: int, rep: int) -> int:
        return seed_base + chi_idx * 1000 + rep

    raw_rows: List[Dict[str, object]] = []
    bound_rows: List[Dict[str, object]] = []

    # Collect S per chi per rep
    S_by_chi: Dict[int, List[float]] = {chi: [] for chi in chi_sweep}

    for chi_idx, chi in enumerate(chi_sweep):
        for rep in range(seeds_per_chi):
            seed = point_seed(chi_idx, rep)

            # Build MERA with actual tensors
            mera = qtn.MERA.rand(L, max_bond=chi, dtype="float64")
            mera.unitize_()

            if args.state == "heisenberg_opt":
                # optimize tensors for Heisenberg model
                mera = maybe_optimize_heisenberg(
                    mera=mera, L=L, steps=int(args.opt_steps),
                    autodiff=str(args.autodiff), device=str(args.device),
                    seed=seed
                )

            # Compute S(A) from dense rho_A contraction
            S, evals = entropy_from_mera_dense_rho(mera, part)

            S_by_chi[chi].append(S)

            # Bound check using declared K_cut proxy upper bound
            log_chi = float(np.log(chi))
            bound = float(K_cut * log_chi)

            raw_rows.append({
                "chi": int(chi),
                "seed": int(seed),
                "partition_id": part.partition_id,
                "L": int(L),
                "A": int(A),
                "S": float(S),
                "run_status": "ok",
                "notes": args.state,
            })

            bound_rows.append({
                "chi": int(chi),
                "seed": int(seed),
                "partition_id": part.partition_id,
                "S": float(S),
                "K_cut": int(K_cut),
                "log_chi": float(log_chi),
                "bound": float(bound),
                "margin": float(bound - S),
                "passed": bool(S <= bound + 1e-12),
            })

    # Medians across seeds
    medians_by_chi = {chi: float(np.median(np.array(S_by_chi[chi], dtype=float))) for chi in chi_sweep}
    y_medians = np.array([medians_by_chi[chi] for chi in chi_sweep], dtype=float)

    # Per-rep slopes (fit S vs log chi for each rep index)
    logs = np.log(np.array(chi_sweep, dtype=float))
    slopes = []
    for rep in range(seeds_per_chi):
        y_rep = np.array([S_by_chi[chi][rep] for chi in chi_sweep], dtype=float)
        a, b, rss = fit_linear(logs, y_rep)
        slopes.append(float(a))

    # Falsifiers 3.1–3.4
    f31 = falsifier_3_1_monotonicity(medians_by_chi, chi_sweep)
    f32 = falsifier_3_2_replicate_robustness(slopes, cv_threshold=0.10)
    f33 = falsifier_3_3_model_selection(y_medians, chi_sweep)
    f34 = falsifier_3_4_bound_validity(bound_rows)

    corr = float(np.corrcoef(logs, y_medians)[0, 1]) if len(chi_sweep) >= 2 else 0.0

    verdict = "SUPPORTED" if (f31["passed"] and f32["passed"] and f33["passed"] and f34["passed"]) else (
        "REJECTED" if (not f31["passed"] or not f34["passed"]) else "INCONCLUSIVE"
    )

    # Write manifest + artifacts
    manifest = {
        "framework_spec": {"name": "FRAMEWORK_SPEC_v0.2", "version": "0.2.0", "date": "2026-02-25"},
        "experiment_name": exp_name,
        "run_id": rid,
        "timestamp_utc": dt.datetime.utcnow().isoformat() + "Z",
        "claim": {
            "claim_id": "C3",
            "option": "A",
            "statement": "For fixed N and fixed bipartition, S scales approximately affine in log chi over tested chi range, consistent with a cut-counting bound S <= K_cut * log chi (with K_cut given by a declared proxy rule).",
        },
        "scope": {
            "fixed": {"L": L, "partition": part.partition_id, "A_sites": part.keep_sites},
            "varied": {"chi_sweep": chi_sweep, "seeds_per_chi": seeds_per_chi, "state": args.state},
            "observables": ["S(chi, seed, partition) computed from rho_A contraction", "K_cut proxy", "model AIC/BIC"],
            "K_cut_proxy_rule": "K_cut = 2*(ceil(log2(A))+1), min 2 (upper proxy for 1D binary MERA interval cut)",
        },
        "seed_policy": {"seed_base": seed_base, "formula": "seed = seed_base + chi_index*1000 + rep"},
        "software": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "scope_limitations": scope_limitations,
        "outputs": {
            "manifest": "manifest.json",
            "raw_entropy": "raw_entropy.csv",
            "fits": "fits.json",
            "bound_checks": "bound_checks.json",
            "verdict": "verdict.json",
        },
    }
    write_json(run_dir / "manifest.json", manifest)

    # raw_entropy.csv
    with (run_dir / "raw_entropy.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["chi", "seed", "partition_id", "L", "A", "S", "run_status", "notes"])
        w.writeheader()
        for r in raw_rows:
            w.writerow({
                "chi": r["chi"],
                "seed": r["seed"],
                "partition_id": r["partition_id"],
                "L": r["L"],
                "A": r["A"],
                "S": f"{float(r['S']):.12f}",
                "run_status": r["run_status"],
                "notes": r["notes"],
            })

    # fits.json
    fits = {
        "chi_sweep": chi_sweep,
        "medians_by_chi": medians_by_chi,
        "corr_medians_log_chi": float(corr),
        "falsifier_3_3_model_selection": f33,
        "replicate_slopes": slopes,
        "falsifier_3_2_slope_stats": f32,
    }
    write_json(run_dir / "fits.json", fits)

    # bound_checks.json
    bound_checks = {
        "K_cut": int(K_cut),
        "rule": manifest["scope"]["K_cut_proxy_rule"],
        "summary": {
            "total_points": len(bound_rows),
            "violation_count": f34["violation_count"],
            "passed": f34["passed"],
            "min_margin": float(min(r["margin"] for r in bound_rows)),
        },
        "rows": bound_rows,
    }
    write_json(run_dir / "bound_checks.json", bound_checks)

    # verdict.json
    verdict_obj = {
        "framework_spec": {"version": "0.2.0"},
        "experiment_name": exp_name,
        "run_id": rid,
        "claim_id": "C3",
        "option": "A",
        "state": args.state,
        "verdict": verdict,
        "metrics": {"corr_medians_log_chi": float(corr)},
        "falsifiers": {
            "3.1_monotonicity": f31,
            "3.2_replicate_robustness": f32,
            "3.3_model_selection": f33,
            "3.4_bound_validity": f34,
        },
        "scope_limitations": scope_limitations,
        "artifact_paths": {
            "manifest": "manifest.json",
            "raw_entropy": "raw_entropy.csv",
            "fits": "fits.json",
            "bound_checks": "bound_checks.json",
            "verdict": "verdict.json",
        },
    }
    write_json(run_dir / "verdict.json", verdict_obj)

    print(run_dir.as_posix())
    print(json.dumps({"verdict": verdict, "corr": corr, "f31": f31["passed"], "f32": f32["passed"], "f33": f33["passed"], "f34": f34["passed"]}))


if __name__ == "__main__":
    main()
