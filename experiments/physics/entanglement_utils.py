"""
Entanglement Utility Module for Physics Grounding Implementation.

This module provides utilities for computing entanglement-related quantities
to test the hypothesis that capacity C is proportional to entanglement entropy S.

Functions include:
- von Neumann entropy calculation
- Renyi entropy calculation
- Reduced density matrix computation
- Entanglement spectrum and gap analysis
- Capacity-entanglement correlation analysis
"""

from typing import List, Dict, Any

import numpy as np
from scipy import linalg
from scipy import stats


# Module-level constants
NUMERICAL_TOLERANCE = 1e-12
QUBIT_DIM = 2


def _validate_density_matrix(rho: np.ndarray) -> None:
    """
    Validate that an array is a valid density matrix.

    A valid density matrix must be:
    - 2-dimensional and square
    - Hermitian (rho = rho^dagger)
    - Trace equal to 1 (within numerical tolerance)
    - Positive semidefinite (all eigenvalues >= 0)

    Parameters
    ----------
    rho : np.ndarray
        Array to validate as a density matrix

    Raises
    ------
    ValueError
        If any validation check fails, with a descriptive error message
    """
    # Check that rho is a numpy array
    if not isinstance(rho, np.ndarray):
        raise ValueError(f"Density matrix must be a numpy array, got {type(rho).__name__}")

    # Check that rho is 2-dimensional
    if rho.ndim != 2:
        raise ValueError(f"Density matrix must be 2-dimensional, got {rho.ndim} dimensions")

    # Check that rho is square
    if rho.shape[0] != rho.shape[1]:
        raise ValueError(f"Density matrix must be square, got shape {rho.shape}")

    # Check Hermiticity: rho should equal its conjugate transpose
    if not np.allclose(rho, rho.conj().T, atol=NUMERICAL_TOLERANCE):
        hermitian_diff = np.max(np.abs(rho - rho.conj().T))
        raise ValueError(
            f"Density matrix must be Hermitian (equal to its conjugate transpose). "
            f"Maximum deviation from Hermiticity: {hermitian_diff:.2e}"
        )

    # Check trace equals 1
    trace = np.trace(rho)
    if not np.isclose(trace, 1.0, atol=NUMERICAL_TOLERANCE):
        raise ValueError(
            f"Density matrix must have trace equal to 1. Got trace = {trace:.6f}"
        )

    # Check positive semidefinite: all eigenvalues must be >= 0
    eigenvalues = linalg.eigvalsh(rho)
    min_eigenvalue = np.min(eigenvalues)
    if min_eigenvalue < -NUMERICAL_TOLERANCE:
        raise ValueError(
            f"Density matrix must be positive semidefinite. "
            f"Minimum eigenvalue: {min_eigenvalue:.2e}"
        )


def _get_valid_eigenvalues(rho: np.ndarray, eps: float) -> np.ndarray:
    """
    Compute eigenvalues of a density matrix and filter out negligible values.

    Parameters
    ----------
    rho : np.ndarray
        Density matrix (assumed to be validated)
    eps : float
        Threshold below which eigenvalues are considered negligible

    Returns
    -------
    np.ndarray
        Array of eigenvalues greater than eps

    Raises
    ------
    ValueError
        If all eigenvalues are filtered out (empty spectrum)
    """
    eigenvalues = linalg.eigvalsh(rho)
    valid_eigenvalues = eigenvalues[eigenvalues > eps]

    if len(valid_eigenvalues) == 0:
        raise ValueError(
            f"All eigenvalues are below threshold eps={eps:.2e}. "
            f"This may indicate a degenerate or invalid density matrix. "
            f"Eigenvalue range: [{np.min(eigenvalues):.2e}, {np.max(eigenvalues):.2e}]"
        )

    return valid_eigenvalues


def von_neumann_entropy(rho: np.ndarray, eps: float = NUMERICAL_TOLERANCE) -> float:
    """
    Compute the von Neumann entropy of a density matrix.

    S = -Tr(ρ log ρ)

    Parameters
    ----------
    rho : np.ndarray
        Density matrix (Hermitian, positive semidefinite, trace 1)
    eps : float, optional
        Small value to avoid log(0), default 1e-12

    Returns
    -------
    float
        Von Neumann entropy in nats

    Raises
    ------
    ValueError
        If rho is not a valid density matrix or all eigenvalues are filtered out
    """
    _validate_density_matrix(rho)

    # Get valid eigenvalues of the density matrix
    eigenvalues = _get_valid_eigenvalues(rho, eps)

    # Compute entropy: S = -sum(λ log λ)
    entropy = -np.sum(eigenvalues * np.log(eigenvalues))

    return float(entropy)


def renyi_entropy(rho: np.ndarray, alpha: float, eps: float = NUMERICAL_TOLERANCE) -> float:
    """
    Compute the Renyi entropy of a density matrix.

    S_α = 1/(1-α) log(Tr(ρ^α))

    For alpha == 1, this becomes the von Neumann entropy.

    Parameters
    ----------
    rho : np.ndarray
        Density matrix (Hermitian, positive semidefinite, trace 1)
    alpha : float
        Renyi entropy order (must be > 0, alpha != 1 uses the general formula)
    eps : float, optional
        Small value to avoid numerical issues, default 1e-12

    Returns
    -------
    float
        Renyi entropy in nats

    Raises
    ------
    ValueError
        If rho is not a valid density matrix, alpha <= 0, or all eigenvalues are filtered out
    """
    # Validate alpha parameter
    if alpha <= 0:
        raise ValueError(
            f"Renyi entropy order alpha must be > 0, got alpha = {alpha}"
        )

    _validate_density_matrix(rho)

    # For alpha = 1, use von Neumann entropy (the limit as alpha -> 1)
    if np.isclose(alpha, 1.0):
        return von_neumann_entropy(rho, eps)

    # Get valid eigenvalues of the density matrix
    eigenvalues = _get_valid_eigenvalues(rho, eps)

    # Compute Tr(ρ^α)
    trace_rho_alpha = np.sum(np.power(eigenvalues, alpha))

    # Compute Renyi entropy: S_α = 1/(1-α) log(Tr(ρ^α))
    entropy = (1.0 / (1.0 - alpha)) * np.log(trace_rho_alpha)

    return float(entropy)


def reduced_density_matrix(psi: np.ndarray, subsystem_A: List[int], total_sites: int) -> np.ndarray:
    """
    Compute the reduced density matrix for subsystem A.

    ρ_A = Tr_B(|ψ⟩⟨ψ|)

    Parameters
    ----------
    psi : np.ndarray
        Wavefunction as a flattened array with 2^total_sites elements
    subsystem_A : List[int]
        List of site indices belonging to subsystem A (0-indexed)
    total_sites : int
        Total number of sites in the full system

    Returns
    -------
    np.ndarray
        Reduced density matrix for subsystem A

    Raises
    ------
    ValueError
        If psi dimensions don't match total_sites or subsystem_A indices are out of range
    """
    # Validate total_sites
    if total_sites <= 0:
        raise ValueError(f"total_sites must be positive, got {total_sites}")

    # Validate psi length matches 2^total_sites
    expected_length = QUBIT_DIM ** total_sites
    if len(psi) != expected_length:
        raise ValueError(
            f"Wavefunction length must be {QUBIT_DIM}^{total_sites} = {expected_length}, "
            f"got length {len(psi)}"
        )

    # Validate subsystem_A indices
    if not subsystem_A:
        raise ValueError("subsystem_A cannot be empty")

    max_index = max(subsystem_A)
    min_index = min(subsystem_A)

    if min_index < 0:
        raise ValueError(
            f"subsystem_A indices must be non-negative, got minimum index {min_index}"
        )

    if max_index >= total_sites:
        raise ValueError(
            f"subsystem_A indices must be less than total_sites ({total_sites}), "
            f"got maximum index {max_index}"
        )

    # Check for duplicate indices
    if len(subsystem_A) != len(set(subsystem_A)):
        raise ValueError("subsystem_A contains duplicate indices")

    # Reshape wavefunction into tensor form
    # Shape: (QUBIT_DIM, QUBIT_DIM, ..., QUBIT_DIM) with total_sites dimensions
    psi_tensor = psi.reshape([QUBIT_DIM] * total_sites)

    # Determine subsystem B (complement of A)
    all_sites = set(range(total_sites))
    subsystem_B = sorted(all_sites - set(subsystem_A))

    # Number of sites in each subsystem
    n_A = len(subsystem_A)
    n_B = len(subsystem_B)

    # Dimension of each subsystem
    dim_A = QUBIT_DIM ** n_A
    dim_B = QUBIT_DIM ** n_B

    # Rearrange axes: first A indices, then B indices
    # This groups all A sites together and all B sites together
    axis_order = subsystem_A + subsystem_B
    psi_reordered = np.transpose(psi_tensor, axis_order)

    # Reshape into matrix form: (dim_A, dim_B)
    psi_matrix = psi_reordered.reshape(dim_A, dim_B)

    # Compute the density matrix |ψ⟩⟨ψ| and trace over B
    # |ψ⟩⟨ψ| is a (dim_A, dim_B) x (dim_A, dim_B) matrix
    # Tracing over B gives a (dim_A, dim_A) matrix
    # ρ_A = ψ ψ^† where we sum over the B index
    rho_A = psi_matrix @ psi_matrix.conj().T

    return rho_A


def entanglement_spectrum(rho: np.ndarray, eps: float = NUMERICAL_TOLERANCE) -> np.ndarray:
    """
    Compute the entanglement spectrum (eigenvalues of reduced density matrix).

    Parameters
    ----------
    rho : np.ndarray
        Reduced density matrix
    eps : float, optional
        Small value to filter out negligible eigenvalues, default 1e-12

    Returns
    -------
    np.ndarray
        Eigenvalues sorted in descending order

    Raises
    ------
    ValueError
        If rho is not a valid density matrix or all eigenvalues are filtered out
    """
    _validate_density_matrix(rho)

    # Get valid eigenvalues
    eigenvalues = _get_valid_eigenvalues(rho, eps)

    # Sort in descending order
    eigenvalues_sorted = np.sort(eigenvalues)[::-1]

    return eigenvalues_sorted


def entanglement_gap(rho: np.ndarray, eps: float = NUMERICAL_TOLERANCE) -> float:
    """
    Compute the entanglement gap.

    The entanglement gap is the difference between the two largest
    Schmidt values (eigenvalues of the reduced density matrix).

    Parameters
    ----------
    rho : np.ndarray
        Reduced density matrix
    eps : float, optional
        Small value for numerical stability, default 1e-12

    Returns
    -------
    float
        Entanglement gap: λ_0 - λ_1

    Raises
    ------
    ValueError
        If rho is not a valid density matrix or all eigenvalues are filtered out
    """
    spectrum = entanglement_spectrum(rho, eps)

    if len(spectrum) < 2:
        # If there's only one eigenvalue, the gap is zero
        return 0.0

    return float(spectrum[0] - spectrum[1])


def capacity_of_entanglement(rho: np.ndarray, eps: float = NUMERICAL_TOLERANCE) -> float:
    """
    Compute capacity of entanglement (second cumulant of spectrum).

    C_E = Tr(ρ(ln ρ)²) - [Tr(ρ ln ρ)]² = Var(H_A)

    From de Boer et al. PRD 99, 066012 (2019).

    Parameters
    ----------
    rho : np.ndarray
        Reduced density matrix (Hermitian, positive semidefinite, trace 1)
    eps : float, optional
        Cutoff for numerical stability, default 1e-12

    Returns
    -------
    float
        Capacity of entanglement (dimensionless)

    Raises
    ------
    ValueError
        If rho is not a valid density matrix or all eigenvalues are filtered out
    """
    _validate_density_matrix(rho)

    # Get valid eigenvalues of the density matrix
    eigenvalues = _get_valid_eigenvalues(rho, eps)

    # Compute log of eigenvalues
    log_lam = np.log(eigenvalues)

    # Entropy (first cumulant): S = -sum(λ log λ)
    S = -np.sum(eigenvalues * log_lam)

    # Capacity (second cumulant): C = sum(λ (log λ)²) - S²
    C = np.sum(eigenvalues * log_lam**2) - S**2

    return float(C)


def capacity_from_entanglement(S: float, normalization: float = 1.0) -> float:
    """
    Convert entanglement entropy to capacity.

    This implements the hypothesis that capacity C is proportional
    to entanglement entropy S.

    Parameters
    ----------
    S : float
        Entanglement entropy in nats
    normalization : float, optional
        Proportionality constant, default 1.0

    Returns
    -------
    float
        Capacity C = normalization * S
    """
    return normalization * S


def analyze_capacity_entanglement_correlation(
    capacities: List[float],
    entropies: List[float]
) -> Dict[str, Any]:
    """
    Analyze the correlation between capacity and entanglement entropy.

    Uses linear regression to test the hypothesis that capacity C
    is proportional to entanglement entropy S.

    Parameters
    ----------
    capacities : List[float]
        List of capacity values
    entropies : List[float]
        List of entanglement entropy values

    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - slope: Regression slope
        - intercept: Regression intercept
        - r_squared: Coefficient of determination
        - p_value: P-value for the slope
        - std_err: Standard error of the slope
        - n_points: Number of data points
        - correlation: Pearson correlation coefficient

    Raises
    ------
    ValueError
        If lists are empty or have different lengths
    """
    # Validate inputs
    if not capacities:
        raise ValueError("capacities list cannot be empty")

    if not entropies:
        raise ValueError("entropies list cannot be empty")

    if len(capacities) != len(entropies):
        raise ValueError(
            f"capacities and entropies must have the same length. "
            f"Got {len(capacities)} capacities and {len(entropies)} entropies"
        )

    # Convert to numpy arrays
    capacities = np.array(capacities)
    entropies = np.array(entropies)

    # Perform linear regression
    result = stats.linregress(entropies, capacities)

    # Compute r_squared
    r_squared = result.rvalue ** 2

    return {
        'slope': float(result.slope),
        'intercept': float(result.intercept),
        'r_squared': float(r_squared),
        'p_value': float(result.pvalue),
        'std_err': float(result.stderr),
        'n_points': len(capacities),
        'correlation': float(result.rvalue)
    }


def analyze_capacity_entropy_ratio(results: List[Dict]) -> Dict:
    """Analyze C_E / S ratio across models.

    For critical 1+1D systems, we expect C_E and S to both scale
    logarithmically with system size, but with different coefficients.

    Args:
        results: List of dicts with 'entropy' and 'capacity_of_entanglement' keys

    Returns:
        Dict with ratio analysis
    """
    ratios = []
    for r in results:
        S = r.get("entropy") or r.get("S")
        C_E = r.get("capacity_of_entanglement") or r.get("C_E")
        if S is not None and C_E is not None and S > 0:
            ratios.append(C_E / S)

    if not ratios:
        return {"error": "No valid data points"}

    return {
        "mean_ratio": float(np.mean(ratios)),
        "std_ratio": float(np.std(ratios)),
        "min_ratio": float(np.min(ratios)),
        "max_ratio": float(np.max(ratios)),
        "n_points": len(ratios),
        "ratios": ratios,
    }