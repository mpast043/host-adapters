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


def von_neumann_entropy(rho: np.ndarray, eps: float = 1e-12) -> float:
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
    """
    # Get eigenvalues of the density matrix
    eigenvalues = linalg.eigvalsh(rho)

    # Filter out eigenvalues below threshold to avoid log(0)
    eigenvalues = eigenvalues[eigenvalues > eps]

    # Compute entropy: S = -sum(λ log λ)
    entropy = -np.sum(eigenvalues * np.log(eigenvalues))

    return float(entropy)


def renyi_entropy(rho: np.ndarray, alpha: float, eps: float = 1e-12) -> float:
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
    """
    # For alpha = 1, use von Neumann entropy (the limit as alpha -> 1)
    if np.isclose(alpha, 1.0):
        return von_neumann_entropy(rho, eps)

    # Get eigenvalues of the density matrix
    eigenvalues = linalg.eigvalsh(rho)

    # Filter out eigenvalues below threshold
    eigenvalues = eigenvalues[eigenvalues > eps]

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
    """
    # Reshape wavefunction into tensor form
    # Shape: (2, 2, ..., 2) with total_sites dimensions
    psi_tensor = psi.reshape([2] * total_sites)

    # Determine subsystem B (complement of A)
    all_sites = set(range(total_sites))
    subsystem_B = sorted(all_sites - set(subsystem_A))

    # Number of sites in each subsystem
    n_A = len(subsystem_A)
    n_B = len(subsystem_B)

    # Dimension of each subsystem
    dim_A = 2 ** n_A
    dim_B = 2 ** n_B

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


def entanglement_spectrum(rho: np.ndarray, eps: float = 1e-12) -> np.ndarray:
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
    """
    # Get eigenvalues
    eigenvalues = linalg.eigvalsh(rho)

    # Filter out eigenvalues below threshold
    eigenvalues = eigenvalues[eigenvalues > eps]

    # Sort in descending order
    eigenvalues_sorted = np.sort(eigenvalues)[::-1]

    return eigenvalues_sorted


def entanglement_gap(rho: np.ndarray, eps: float = 1e-12) -> float:
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
    """
    spectrum = entanglement_spectrum(rho, eps)

    if len(spectrum) < 2:
        # If there's only one eigenvalue, the gap is zero
        return 0.0

    return float(spectrum[0] - spectrum[1])


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
    """
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