#!/usr/bin/env python3
"""Entanglement gap analysis for Δλ ≈ 38 testing.

From Wald et al. PRR 2, 043404 (2020):
The entanglement (Schmidt) gap closes as π²/ln(L) at criticality.

Framework claim: Specific crossover value Δλ ≈ 38.
"""

import argparse
import json
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
from typing import Dict, List, Any
from scipy.optimize import curve_fit


def entanglement_gap(rho: np.ndarray, eps: float = 1e-12) -> float:
    """Compute entanglement gap λ₀ - λ₁.

    Args:
        rho: Reduced density matrix
        eps: Numerical cutoff

    Returns:
        Entanglement gap: λ₀ - λ₁
    """
    eigenvalues = np.linalg.eigvalsh(rho)
    eigenvalues = np.sort(eigenvalues)[::-1]  # Descending
    eigenvalues = eigenvalues[eigenvalues > eps]

    if len(eigenvalues) < 2:
        return 0.0
    return float(eigenvalues[0] - eigenvalues[1])


def gap_ratio(rho: np.ndarray, eps: float = 1e-12) -> float:
    """Compute normalized gap ratio (λ₀ - λ₁) / λ₀.

    Args:
        rho: Reduced density matrix
        eps: Numerical cutoff

    Returns:
        Gap ratio: (λ₀ - λ₁) / λ₀
    """
    eigenvalues = np.linalg.eigvalsh(rho)
    eigenvalues = np.sort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[eigenvalues > eps]

    if len(eigenvalues) < 2 or eigenvalues[0] < eps:
        return 0.0
    return float((eigenvalues[0] - eigenvalues[1]) / eigenvalues[0])


def test_gap_closure_critical(
    L_values: List[int],
    gaps: List[float]
) -> Dict[str, Any]:
    """Test if gap closes as π²/ln(L) at criticality.

    From Wald et al.: δξ ∝ π²/ln(L)

    Args:
        L_values: System sizes
        gaps: Corresponding entanglement gaps

    Returns:
        Dict with fit results and Δλ estimate
    """
    L_arr = np.array(L_values, dtype=float)
    gap_arr = np.array(gaps, dtype=float)

    # Filter valid data
    valid = (L_arr > 2) & (gap_arr > 0)
    L_arr = L_arr[valid]
    gap_arr = gap_arr[valid]

    if len(L_arr) < 3:
        return {"error": "Insufficient data points for fitting"}

    # Fit gap = A * π² / ln(L)
    def gap_model(L, A):
        return A * np.pi**2 / np.log(L)

    try:
        popt, pcov = curve_fit(gap_model, L_arr, gap_arr, p0=[1.0])
        A_fit = float(popt[0])
        A_err = float(np.sqrt(pcov[0, 0]))

        return {
            "fit_A": A_fit,
            "fit_A_error": A_err,
            "pi_squared": float(np.pi**2),
            "A_times_pi_squared": float(A_fit * np.pi**2),
            "close_to_38": bool(abs(A_fit * np.pi**2 - 38) < 5),
            "L_values": [int(x) for x in L_arr],
            "gap_values": [float(x) for x in gap_arr],
        }
    except Exception as e:
        return {"error": str(e)}


def test_delta_lambda_hypothesis(gap_results: Dict) -> Dict[str, Any]:
    """Test hypotheses for Δλ ≈ 38.

    Possible interpretations:
    1. Δλ = gap_ratio × 100 (normalized gap percentage)
    2. Δλ = π² × scale (from gap closure formula)
    3. Δλ = crossover in capacity second derivative

    Args:
        gap_results: Dict with gap analysis results

    Returns:
        Dict with hypothesis test results
    """
    hypotheses = {}

    # H1: Gap ratio percentage
    if "gap_ratios" in gap_results:
        ratios = gap_results["gap_ratios"]
        # ratios is a dict with string keys, need to iterate over values
        percentages = [r * 100 for r in ratios.values() if r > 0]
        if percentages:
            hypotheses["H1_gap_ratio_percent"] = {
                "values": percentages,
                "mean": float(np.mean(percentages)),
                "mean_near_38": bool(abs(np.mean(percentages) - 38) < 10),
            }

    # H2: π² × scale
    if "gap_closure_test" in gap_results and "fit_A" in gap_results.get("gap_closure_test", {}):
        A = gap_results["gap_closure_test"]["fit_A"]
        hypotheses["H2_pi_squared_scale"] = {
            "A_times_pi_squared": float(A * np.pi**2),
            "near_38": bool(abs(A * np.pi**2 - 38) < 5),
        }

    return hypotheses


def run_gap_analysis(
    model: str,
    L_values: List[int],
    chi: int,
    output_dir: str = "outputs/gap_analysis"
) -> Dict[str, Any]:
    """Run gap analysis for a model.

    This is a placeholder that returns expected scaling for testing.
    Real implementation would run MERA and compute gaps.

    Args:
        model: Model name
        L_values: System sizes
        chi: Bond dimension
        output_dir: Output directory

    Returns:
        Dict with analysis results
    """
    results = {
        "model": model,
        "L_values": L_values,
        "chi": chi,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "gaps": {},
        "gap_ratios": {},
    }

    # Placeholder: expected gap values for critical models
    # Real implementation would compute from MERA
    for L in L_values:
        # Approximate gap scaling at criticality
        # Gap ~ 1/L^(z) for gapless systems
        if model in ["heisenberg", "xxz"]:
            gap = 1.0 / L**0.5
        else:
            gap = 0.5 / L**0.5

        results["gaps"][str(L)] = gap
        results["gap_ratios"][str(L)] = gap / (1 - gap/2)

    # Test gap closure
    gaps = list(results["gaps"].values())
    results["gap_closure_test"] = test_gap_closure_critical(L_values, gaps)

    # Test Δλ hypotheses
    results["delta_lambda_hypotheses"] = test_delta_lambda_hypothesis(results)

    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%SZ")
    result_file = output_path / f"{timestamp}_{model}_gap_analysis.json"
    result_file.write_text(json.dumps(results, indent=2))

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Entanglement gap analysis for Δλ testing"
    )
    parser.add_argument(
        "--model",
        default="heisenberg",
        help="Model to analyze"
    )
    parser.add_argument(
        "--L",
        default="4,8,16,32",
        help="Comma-separated system sizes"
    )
    parser.add_argument(
        "--chi",
        type=int,
        default=16,
        help="Bond dimension"
    )
    parser.add_argument(
        "--output",
        default="outputs/gap_analysis",
        help="Output directory"
    )

    args = parser.parse_args()
    L_values = [int(x) for x in args.L.split(",")]

    print(f"Running gap analysis for {args.model}")
    print(f"L values: {L_values}, chi: {args.chi}")

    results = run_gap_analysis(
        model=args.model,
        L_values=L_values,
        chi=args.chi,
        output_dir=args.output,
    )

    print(f"\nGap closure test:")
    if "error" in results["gap_closure_test"]:
        print(f"  Error: {results['gap_closure_test']['error']}")
    else:
        print(f"  Fit A: {results['gap_closure_test']['fit_A']:.4f}")
        print(f"  A × π²: {results['gap_closure_test']['A_times_pi_squared']:.4f}")
        print(f"  Close to 38? {results['gap_closure_test']['close_to_38']}")

    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()