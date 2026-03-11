#!/usr/bin/env python3
"""Entanglement gap analysis for Δλ ≈ 38 testing.

From Wald et al. PRR 2, 043404 (2020):
The entanglement (Schmidt) gap closes as π²/ln(L) at criticality.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import quimb as qu
from scipy.optimize import curve_fit
from scipy.sparse.linalg import eigsh

try:
    from experiments.physics.entanglement_utils import reduced_density_matrix
except ModuleNotFoundError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from experiments.physics.entanglement_utils import reduced_density_matrix


def _resolve_model_alias(model: str) -> str:
    aliases = {
        "ising": "ising_cyclic",
        "tfim": "ising_cyclic",
        "xxz": "heisenberg_cyclic",
        "heisenberg": "heisenberg_cyclic",
    }
    return aliases.get(model, model)


def build_hamiltonian(model: str, L: int):
    """Build sparse 1D spin-chain Hamiltonian."""
    if L < 2:
        raise ValueError("L must be >= 2")

    model = _resolve_model_alias(model)
    if model == "ising_open":
        return qu.ham_ising(L, jz=1.0, bx=1.0, cyclic=False, sparse=True)
    if model == "ising_cyclic":
        return qu.ham_ising(L, jz=1.0, bx=1.0, cyclic=True, sparse=True)
    if model == "heisenberg_open":
        return qu.ham_heis(L, j=1.0, cyclic=False, sparse=True)
    if model == "heisenberg_cyclic":
        return qu.ham_heis(L, j=1.0, cyclic=True, sparse=True)

    raise ValueError(
        "Unsupported model. Use one of: "
        "ising_open, ising_cyclic, heisenberg_open, heisenberg_cyclic, "
        "or aliases ising/heisenberg/xxz."
    )


def ground_state_from_ed(H) -> tuple[float, np.ndarray]:
    """Compute ED ground state and energy via sparse eigensolver."""
    eigenvalues, eigenvectors = eigsh(H, k=1, which="SA")
    e0 = float(eigenvalues[0])
    psi0 = np.asarray(eigenvectors[:, 0], dtype=complex)
    return e0, psi0


def partial_trace(psi: np.ndarray, keep_sites: int, total_sites: int) -> np.ndarray:
    """Compute rho_A by tracing out sites [keep_sites, ..., total_sites-1]."""
    if keep_sites <= 0 or keep_sites >= total_sites:
        raise ValueError("keep_sites must satisfy 0 < keep_sites < total_sites")
    subsystem_a = list(range(keep_sites))
    return reduced_density_matrix(psi, subsystem_A=subsystem_a, total_sites=total_sites)


def _sorted_spectrum(rho: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    rho_h = 0.5 * (rho + rho.conj().T)
    eigenvalues = np.linalg.eigvalsh(rho_h).real
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    if eigenvalues.sum() <= eps:
        raise ValueError("rho must have positive trace")
    eigenvalues = eigenvalues / eigenvalues.sum()
    eigenvalues = np.sort(eigenvalues)[::-1]
    return eigenvalues[eigenvalues > eps]


def entanglement_gap(rho: np.ndarray, eps: float = 1e-12) -> float:
    """Compute entanglement gap λ₀ - λ₁."""
    eigenvalues = _sorted_spectrum(rho, eps=eps)
    if eigenvalues.size < 2:
        return 0.0
    return float(eigenvalues[0] - eigenvalues[1])


def gap_ratio(rho: np.ndarray, eps: float = 1e-12) -> float:
    """Compute normalized gap ratio (λ₀ - λ₁) / λ₀."""
    eigenvalues = _sorted_spectrum(rho, eps=eps)
    if eigenvalues.size < 2 or eigenvalues[0] < eps:
        return 0.0
    return float((eigenvalues[0] - eigenvalues[1]) / eigenvalues[0])


def entanglement_gap_from_rho(rho: np.ndarray, eps: float = 1e-12) -> tuple[float, float, np.ndarray]:
    """Compatibility helper: return (gap, ratio, sorted_spectrum_desc)."""
    spectrum = _sorted_spectrum(rho, eps=eps)
    if spectrum.size < 2:
        return 0.0, 0.0, spectrum
    gap = float(spectrum[0] - spectrum[1])
    ratio = float(gap / spectrum[0]) if spectrum[0] > eps else 0.0
    return gap, ratio, spectrum


def compute_ed_entanglement_gaps(
    model: str,
    L_values: Sequence[int],
    keep_sites: int | None = None,
) -> Dict[str, Any]:
    """Compute ED entanglement gaps across system sizes."""
    canonical_model = _resolve_model_alias(model)
    records: List[Dict[str, float]] = []

    for L in L_values:
        keep = keep_sites if keep_sites is not None else L // 2
        H = build_hamiltonian(canonical_model, L)
        e0, psi0 = ground_state_from_ed(H)

        rho_a = partial_trace(psi0, keep_sites=keep, total_sites=L)
        spectrum = _sorted_spectrum(rho_a)
        lam0 = float(spectrum[0]) if spectrum.size > 0 else 0.0
        lam1 = float(spectrum[1]) if spectrum.size > 1 else 0.0
        gap = float(lam0 - lam1) if spectrum.size > 1 else 0.0
        ratio = float(gap / lam0) if lam0 > 1e-12 else 0.0

        records.append(
            {
                "L": int(L),
                "ground_energy": e0,
                "gap": gap,
                "gap_ratio": ratio,
                "lambda0": lam0,
                "lambda1": lam1,
            }
        )

    return {
        "model": canonical_model,
        "records": records,
        "L_values": [r["L"] for r in records],
        "gaps": [r["gap"] for r in records],
        "gap_ratios": [r["gap_ratio"] for r in records],
    }


def test_gap_closure_critical(
    L_values: List[int],
    gaps: List[float],
) -> Dict[str, Any]:
    """Fit gap(L) = A * π² / ln(L)."""
    L_arr = np.array(L_values, dtype=float)
    gap_arr = np.array(gaps, dtype=float)

    valid = (L_arr > 2) & np.isfinite(gap_arr) & (gap_arr >= 0.0)
    L_arr = L_arr[valid]
    gap_arr = gap_arr[valid]

    if L_arr.size < 3:
        return {"error": "Insufficient data points for fitting"}

    def gap_model(L, A):
        return A * np.pi**2 / np.log(L)

    try:
        popt, pcov = curve_fit(gap_model, L_arr, gap_arr, p0=[1.0], maxfev=10000)
        A_fit = float(popt[0])
        A_err = float(np.sqrt(max(0.0, pcov[0, 0]))) if pcov.size else float("nan")

        pred = gap_model(L_arr, A_fit)
        ss_res = float(np.sum((gap_arr - pred) ** 2))
        ss_tot = float(np.sum((gap_arr - np.mean(gap_arr)) ** 2))
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 1e-15 else 1.0

        return {
            "fit_A": A_fit,
            "fit_A_error": A_err,
            "pi_squared": float(np.pi**2),
            "A_times_pi_squared": float(A_fit * np.pi**2),
            "close_to_38": bool(abs(A_fit * np.pi**2 - 38.0) < 5.0),
            "r_squared": r_squared,
            "L_values": [int(x) for x in L_arr.tolist()],
            "gap_values": [float(x) for x in gap_arr.tolist()],
        }
    except Exception as exc:
        return {"error": str(exc)}


def fit_gap_closure_log(
    L_values: Sequence[int],
    gaps: Sequence[float],
    *,
    target_delta_lambda: float = 38.0,
    tolerance: float = 5.0,
) -> Dict[str, Any]:
    """Compatibility fit helper with explicit target/tolerance knobs."""
    base = test_gap_closure_critical(list(L_values), list(gaps))
    if "error" in base:
        return base

    close = abs(base["A_times_pi_squared"] - target_delta_lambda) <= tolerance
    return {
        "fit_A": base["fit_A"],
        "fit_A_error": base["fit_A_error"],
        "A_times_pi_squared": base["A_times_pi_squared"],
        "close_to_target": bool(close),
        "target_delta_lambda": float(target_delta_lambda),
        "tolerance": float(tolerance),
        "r_squared": float(base["r_squared"]),
        "L_values": base["L_values"],
        "gap_values": base["gap_values"],
    }


def test_delta_lambda_hypothesis(gap_results: Dict[str, Any]) -> Dict[str, Any]:
    """Test simple Δλ ≈ 38 hypotheses."""
    hypotheses: Dict[str, Any] = {}

    if "gap_ratios" in gap_results:
        ratios = gap_results["gap_ratios"]
        if isinstance(ratios, dict):
            values = [float(r) for r in ratios.values() if float(r) >= 0.0]
        else:
            values = [float(r) for r in ratios if float(r) >= 0.0]
        percentages = [100.0 * r for r in values]
        if percentages:
            mean_pct = float(np.mean(percentages))
            hypotheses["H1_gap_ratio_percent"] = {
                "values": percentages,
                "mean": mean_pct,
                "mean_near_38": bool(abs(mean_pct - 38.0) < 10.0),
            }

    closure = gap_results.get("gap_closure_test", {})
    if "fit_A" in closure:
        a_pi2 = float(closure["fit_A"] * np.pi**2)
        hypotheses["H2_pi_squared_scale"] = {
            "A_times_pi_squared": a_pi2,
            "near_38": bool(abs(a_pi2 - 38.0) < 5.0),
        }

    return hypotheses


def run_gap_analysis(
    model: str,
    L_values: List[int],
    chi: int,
    output_dir: str = "outputs/gap_analysis",
) -> Dict[str, Any]:
    """Run ED-based entanglement gap analysis."""
    canonical_model = _resolve_model_alias(model)
    ed = compute_ed_entanglement_gaps(model=canonical_model, L_values=L_values)

    results: Dict[str, Any] = {
        "model": canonical_model,
        "requested_model": model,
        "L_values": [int(v) for v in L_values],
        "chi": int(chi),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "gaps": {str(r["L"]): float(r["gap"]) for r in ed["records"]},
        "gap_ratios": {str(r["L"]): float(r["gap_ratio"]) for r in ed["records"]},
        "ground_energies": {str(r["L"]): float(r["ground_energy"]) for r in ed["records"]},
        "lambda0": {str(r["L"]): float(r["lambda0"]) for r in ed["records"]},
        "lambda1": {str(r["L"]): float(r["lambda1"]) for r in ed["records"]},
        "records": ed["records"],
    }

    gaps = [results["gaps"][str(L)] for L in L_values if str(L) in results["gaps"]]
    results["gap_closure_test"] = test_gap_closure_critical(L_values, gaps)
    results["delta_lambda_hypotheses"] = test_delta_lambda_hypothesis(results)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result_file = output_path / f"{timestamp}_{canonical_model}_gap_analysis.json"
    result_file.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    results["result_file"] = str(result_file)
    return results


def _parse_L_values(value: str) -> List[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description="Entanglement gap analysis for Δλ testing")
    parser.add_argument("--model", default="heisenberg", help="Model to analyze")
    parser.add_argument("--L", default="4,8,16,32", help="Comma-separated system sizes")
    parser.add_argument("--chi", type=int, default=16, help="Bond dimension (metadata only for ED run)")
    parser.add_argument("--output", default="outputs/gap_analysis", help="Output directory")
    args = parser.parse_args()

    L_values = _parse_L_values(args.L)
    print(f"Running gap analysis for {args.model}")
    print(f"L values: {L_values}, chi: {args.chi}")

    results = run_gap_analysis(
        model=args.model,
        L_values=L_values,
        chi=args.chi,
        output_dir=args.output,
    )

    print("\nGap closure test:")
    closure = results["gap_closure_test"]
    if "error" in closure:
        print(f"  Error: {closure['error']}")
    else:
        print(f"  Fit A: {closure['fit_A']:.6f}")
        print(f"  A × pi^2: {closure['A_times_pi_squared']:.6f}")
        print(f"  Close to 38? {closure['close_to_38']}")
        print(f"  R^2: {closure['r_squared']:.6f}")

    print(f"\nResults saved to: {results['result_file']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
