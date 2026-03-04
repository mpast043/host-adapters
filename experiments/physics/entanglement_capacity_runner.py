#!/usr/bin/env python3
"""
Entanglement-Capacity Correlation Test Runner.

This module tests the hypothesis that capacity C is proportional to
entanglement entropy S by running MERA simulations at various bond dimensions.

Hypothesis H1: C ∝ S with R² > 0.95

For now, uses placeholder data where:
- S ∝ log(chi) (entanglement entropy scales logarithmically with bond dimension)
- C = 0.5 * S (capacity is proportional to entropy)
"""

from __future__ import annotations

import argparse
import datetime as dt
from datetime import timezone
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import warnings

import numpy as np

from entanglement_utils import (
    analyze_capacity_entanglement_correlation,
)


# Module constants
__version__ = "1.0.0"

# Threshold for hypothesis testing
R_SQUARED_THRESHOLD = 0.95

# Simulation parameters
NOISE_FACTOR = 0.02
MIN_ENTROPY = 0.1
CAPACITY_FACTOR = 0.5
SPECTRUM_SIZE = 8

# Valid model types
VALID_MODELS = ["ising", "heisenberg", "xxz"]


def run_id() -> str:
    """Generate a unique run identifier with timestamp and random suffix."""
    t = dt.datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    r = os.urandom(4).hex()
    return f"{t}_{r}"


def simulate_mera(chi: int, model: str, L: int, seed: Optional[int] = None) -> Dict[str, Any]:
    """
    Simulate a MERA network at a given bond dimension.

    This is a placeholder implementation that generates physically plausible
    placeholder data for testing the capacity-entanglement correlation.

    In a real implementation, this would run an actual MERA optimization.

    Parameters
    ----------
    chi : int
        Bond dimension (maximum Schmidt rank). Must be > 0.
    model : str
        Model type (e.g., 'ising', 'heisenberg', 'xxz')
    L : int
        System size (number of sites). Must be > 0.
    seed : Optional[int]
        Random seed for reproducibility

    Returns
    -------
    Dict[str, Any]
        Dictionary containing MERA simulation results:
        - chi: Bond dimension
        - model: Model type
        - L: System size
        - entanglement_entropy: Computed entanglement entropy
        - capacity: Computed capacity
        - spectrum: Entanglement spectrum (placeholder)

    Raises
    ------
    ValueError
        If chi <= 0, L <= 0, or model is not a valid model type.
    """
    # Input validation
    if chi <= 0:
        raise ValueError(f"Bond dimension chi must be > 0, got {chi}")
    if L <= 0:
        raise ValueError(f"System size L must be > 0, got {L}")
    if model not in VALID_MODELS:
        raise ValueError(f"Invalid model '{model}'. Valid models are: {VALID_MODELS}")

    if seed is not None:
        np.random.seed(seed)

    # Placeholder: S ∝ log(chi) for critical systems
    # This is physically motivated: for a critical system, the entanglement
    # entropy of a subsystem of size L_A scales as S ≈ (c/3) * log(L_A)
    # where c is the central charge. For MERA with bond dimension chi,
    # the effective correlation length scales as chi^κ, giving
    # S ∝ log(chi) for the accessible entropy
    base_entropy = np.log(chi)

    # Add some noise to make it more realistic
    noise = NOISE_FACTOR * np.random.randn()
    entanglement_entropy = max(MIN_ENTROPY, base_entropy + noise)

    # Capacity is proportional to entropy: C = CAPACITY_FACTOR * S
    # This tests the hypothesis H1
    capacity = CAPACITY_FACTOR * entanglement_entropy

    # Generate a placeholder entanglement spectrum
    # For a maximally entangled state, spectrum would be uniform
    # For a low-entanglement state, it would be concentrated
    n_spectrum = min(chi, SPECTRUM_SIZE)  # Limit spectrum size for display
    spectrum = np.exp(-np.arange(n_spectrum) / entanglement_entropy)
    spectrum = spectrum / np.sum(spectrum)  # Normalize

    return {
        "chi": chi,
        "model": model,
        "L": L,
        "entanglement_entropy": float(entanglement_entropy),
        "capacity": float(capacity),
        "spectrum": spectrum.tolist(),
    }


def compute_capacity_from_mera(mera_result: Dict[str, Any]) -> float:
    """
    Extract capacity from a MERA simulation result.

    Parameters
    ----------
    mera_result : Dict[str, Any]
        Dictionary containing MERA simulation results

    Returns
    -------
    float
        The capacity value from the simulation
    """
    return mera_result["capacity"]


def compute_entanglement_from_mera(
    mera_result: Dict[str, Any],
    subsystem_size: Optional[int] = None
) -> float:
    """
    Compute entanglement entropy from a MERA simulation result.

    Parameters
    ----------
    mera_result : Dict[str, Any]
        Dictionary containing MERA simulation results
    subsystem_size : Optional[int]
        Size of subsystem for computing entropy (not used in placeholder)

    Returns
    -------
    float
        The entanglement entropy from the simulation
    """
    return mera_result["entanglement_entropy"]


def run_correlation_test(
    chi_values: List[int],
    model: str,
    L: int,
    output_dir: str,
    seed: Optional[int] = None
) -> Dict[str, Any]:
    """
    Run the capacity-entanglement correlation test.

    Tests hypothesis H1: C ∝ S with R² > R_SQUARED_THRESHOLD

    Parameters
    ----------
    chi_values : List[int]
        List of bond dimensions to test
    model : str
        Model type (e.g., 'ising', 'heisenberg')
    L : int
        System size
    output_dir : str
        Directory to save results
    seed : Optional[int]
        Random seed for reproducibility

    Returns
    -------
    Dict[str, Any]
        Dictionary containing test results

    Raises
    ------
    ValueError
        If chi_values is empty.
    """
    # Handle empty chi_values list
    if not chi_values:
        warnings.warn("chi_values list is empty, no simulations will be run", UserWarning)
        return {
            "metadata": {
                "run_id": run_id(),
                "timestamp": dt.datetime.now(timezone.utc).isoformat(),
                "test": "H1_capacity_entanglement_correlation",
                "version": __version__,
                "config": {
                    "chi_values": [],
                    "model": model,
                    "L": L,
                    "seed": seed,
                },
            },
            "measurements": [],
            "correlation_analysis": {},
            "hypothesis": {
                "H1": "C ∝ S",
                "threshold": R_SQUARED_THRESHOLD,
                "r_squared": None,
                "passed": False,
            },
            "verdict": "REJECT",
        }

    print(f"[H1] Running capacity-entanglement correlation test")
    print(f"     Model: {model}, System size: {L}")
    print(f"     Bond dimensions: {chi_values}")

    # Run MERA simulations for each chi value
    mera_results = []
    capacities = []
    entropies = []

    for i, chi in enumerate(chi_values):
        # Use a different seed for each simulation if seed is provided
        sim_seed = seed + i if seed is not None else None

        result = simulate_mera(chi, model, L, seed=sim_seed)
        mera_results.append(result)

        capacity = compute_capacity_from_mera(result)
        entropy = compute_entanglement_from_mera(result)

        capacities.append(capacity)
        entropies.append(entropy)

        print(f"     chi={chi}: S={entropy:.4f}, C={capacity:.4f}")

    # Analyze the correlation
    correlation_result = analyze_capacity_entanglement_correlation(
        capacities, entropies
    )

    # Test hypothesis H1: R² > R_SQUARED_THRESHOLD
    r_squared = correlation_result["r_squared"]
    passed = r_squared > R_SQUARED_THRESHOLD

    # Build the output
    output = {
        "metadata": {
            "run_id": run_id(),
            "timestamp": dt.datetime.now(timezone.utc).isoformat(),
            "test": "H1_capacity_entanglement_correlation",
            "version": __version__,
            "config": {
                "chi_values": chi_values,
                "model": model,
                "L": L,
                "seed": seed,
            },
        },
        "measurements": [
            {
                "chi": r["chi"],
                "entanglement_entropy": r["entanglement_entropy"],
                "capacity": r["capacity"],
            }
            for r in mera_results
        ],
        "correlation_analysis": correlation_result,
        "hypothesis": {
            "H1": "C ∝ S",
            "threshold": R_SQUARED_THRESHOLD,
            "r_squared": r_squared,
            "passed": passed,
        },
        "verdict": "ACCEPT" if passed else "REJECT",
    }

    # Write results to output directory
    write_results(output, output_dir)

    print(f"\n[H1] Correlation Analysis:")
    print(f"     Slope: {correlation_result['slope']:.6f}")
    print(f"     Intercept: {correlation_result['intercept']:.6f}")
    print(f"     R-squared: {r_squared:.6f}")
    print(f"     P-value: {correlation_result['p_value']:.2e}")
    print(f"     Verdict: {output['verdict']} (threshold: {R_SQUARED_THRESHOLD})")

    return output


def write_results(results: Dict[str, Any], output_dir: str) -> None:
    """
    Write results to JSON files in the output directory.

    Parameters
    ----------
    results : Dict[str, Any]
        Results dictionary to write
    output_dir : str
        Directory to write results to

    Raises
    ------
    OSError
        If there is an error creating the output directory or writing files.
    """
    output_path = Path(output_dir)
    try:
        output_path.mkdir(parents=True, exist_ok=True)

        # Write metadata
        with open(output_path / "metadata.json", "w") as f:
            json.dump(results["metadata"], f, indent=2)

        # Write raw measurements
        with open(output_path / "measurements.json", "w") as f:
            json.dump(results["measurements"], f, indent=2)

        # Write correlation analysis
        with open(output_path / "correlation.json", "w") as f:
            json.dump(results["correlation_analysis"], f, indent=2)

        # Write verdict
        verdict_data = {
            "test": "H1",
            "verdict": results["verdict"],
            "status": "COMPLETE",
            "hypothesis": results["hypothesis"],
            "passed": {"H1_correlation": results["hypothesis"]["passed"]},
        }
        with open(output_path / "verdict.json", "w") as f:
            json.dump(verdict_data, f, indent=2)

        # Write complete results
        with open(output_path / "results.json", "w") as f:
            json.dump(results, f, indent=2)
    except OSError as e:
        raise OSError(f"Failed to write results to {output_dir}: {e}") from e


def parse_chi_values(chi_str: str) -> List[int]:
    """
    Parse chi values from command line argument.

    Parameters
    ----------
    chi_str : str
        Comma-separated string of chi values (e.g., "2,4,8,16,32")

    Returns
    -------
    List[int]
        List of parsed chi values

    Raises
    ------
    ValueError
        If the input string is malformed or contains invalid values.
    """
    try:
        values = [int(x.strip()) for x in chi_str.split(",")]
    except ValueError as e:
        raise ValueError(
            f"Invalid chi values format: '{chi_str}'. "
            "Expected comma-separated integers (e.g., '2,4,8,16,32')"
        ) from e

    # Validate that all values are positive
    for v in values:
        if v <= 0:
            raise ValueError(f"Chi value must be > 0, got {v}")

    return values


def main():
    """CLI entry point for the entanglement-capacity correlation runner."""
    parser = argparse.ArgumentParser(
        description="Run capacity-entanglement correlation test (H1: C ∝ S)"
    )
    parser.add_argument(
        "--chi",
        type=str,
        default="2,4,8,16,32",
        help="Comma-separated list of bond dimensions (default: 2,4,8,16,32)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="ising",
        help="Model type (default: ising)"
    )
    parser.add_argument(
        "--L",
        type=int,
        default=64,
        help="System size (default: 64)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/entanglement_capacity",
        help="Output directory for results (default: results/entanglement_capacity)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (default: None)"
    )

    args = parser.parse_args()

    chi_values = parse_chi_values(args.chi)

    run_correlation_test(
        chi_values=chi_values,
        model=args.model,
        L=args.L,
        output_dir=args.output,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()