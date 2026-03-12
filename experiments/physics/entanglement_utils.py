"""Entanglement utilities and MERA scaling-dimension extraction helpers."""

from __future__ import annotations

import math
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

try:
    from scipy import stats as scipy_stats
except Exception:  # pragma: no cover - scipy is optional
    scipy_stats = None


def _validate_square_matrix(matrix: np.ndarray, *, name: str) -> np.ndarray:
    arr = np.asarray(matrix, dtype=complex)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"{name} must be a square 2D matrix")
    return arr


def entanglement_spectrum(rho: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Return sorted non-zero eigenvalues of a density matrix."""
    rho = _validate_square_matrix(rho, name="rho")
    rho_h = 0.5 * (rho + rho.conj().T)
    eigvals = np.linalg.eigvalsh(rho_h).real
    eigvals = np.clip(eigvals, 0.0, None)

    total = float(np.sum(eigvals))
    if total <= eps:
        raise ValueError("rho must have positive trace")
    eigvals = eigvals / total

    spec = eigvals[eigvals > eps]
    return np.sort(spec)[::-1]


def von_neumann_entropy(rho: np.ndarray, eps: float = 1e-12) -> float:
    """Compute S(ρ) = -Tr(ρ ln ρ) in natural log units."""
    spec = entanglement_spectrum(rho, eps=eps)
    return float(-np.sum(spec * np.log(spec)))


def renyi_entropy(rho: np.ndarray, alpha: float = 2.0, eps: float = 1e-12) -> float:
    """Compute Renyi entropy S_α(ρ) = ln Tr(ρ^α) / (1 - α)."""
    if alpha <= 0:
        raise ValueError("alpha must be > 0")
    if np.isclose(alpha, 1.0):
        return von_neumann_entropy(rho, eps=eps)
    spec = entanglement_spectrum(rho, eps=eps)
    return float(np.log(np.sum(spec ** alpha)) / (1.0 - alpha))


def reduced_density_matrix(psi: np.ndarray, subsystem_A: Sequence[int], total_sites: int) -> np.ndarray:
    """Compute reduced density matrix ρ_A = Tr_B |ψ><ψ| for qubit systems."""
    if total_sites <= 0:
        raise ValueError("total_sites must be positive")
    if not subsystem_A:
        raise ValueError("subsystem_A cannot be empty")
    if len(set(subsystem_A)) != len(subsystem_A):
        raise ValueError("subsystem_A contains duplicate indices")
    if any(i < 0 for i in subsystem_A):
        raise ValueError("subsystem_A indices must be non-negative")
    if any(i >= total_sites for i in subsystem_A):
        raise ValueError("subsystem_A indices must be less than total_sites")

    expected_size = 2 ** total_sites
    vec = np.asarray(psi, dtype=complex).reshape(-1)
    if vec.size != expected_size:
        raise ValueError(f"Wavefunction length must be {expected_size} for total_sites={total_sites}")

    norm = np.linalg.norm(vec)
    if norm <= 1e-15:
        raise ValueError("psi must have non-zero norm")
    vec = vec / norm

    psi_tensor = vec.reshape((2,) * total_sites)
    subsystem_B = [i for i in range(total_sites) if i not in subsystem_A]
    perm = list(subsystem_A) + subsystem_B

    psi_ab = np.transpose(psi_tensor, axes=perm).reshape(2 ** len(subsystem_A), 2 ** len(subsystem_B))
    rho_a = psi_ab @ psi_ab.conj().T
    rho_a = 0.5 * (rho_a + rho_a.conj().T)

    tr = np.trace(rho_a).real
    if tr <= 0:
        raise ValueError("computed reduced density matrix has non-positive trace")
    return rho_a / tr


def entanglement_gap(rho: np.ndarray, eps: float = 1e-12) -> float:
    """Compute Schmidt/entanglement gap λ0 - λ1."""
    spec = entanglement_spectrum(rho, eps=eps)
    if spec.size < 2:
        return 0.0
    return float(spec[0] - spec[1])


def capacity_of_entanglement(rho: np.ndarray, eps: float = 1e-12) -> float:
    """Compute C_E = Var(H_A), where H_A = -ln(ρ_A)."""
    spec = entanglement_spectrum(rho, eps=eps)
    if spec.size <= 1:
        return 0.0
    modular = -np.log(spec)
    mean_1 = float(np.sum(spec * modular))
    mean_2 = float(np.sum(spec * modular * modular))
    return mean_2 - mean_1 * mean_1


def capacity_from_entanglement(entropy: float, normalization: float = 1.0) -> float:
    """Simple linear capacity proxy C = normalization * S."""
    return float(normalization * entropy)


def _correlation_p_value(correlation: float, n_points: int) -> float:
    if n_points < 3 or not np.isfinite(correlation):
        return float("nan")
    corr = float(np.clip(correlation, -1.0, 1.0))
    if np.isclose(abs(corr), 1.0):
        return 0.0

    t_stat = abs(corr) * math.sqrt((n_points - 2) / max(1e-15, 1.0 - corr * corr))
    if scipy_stats is not None:
        return float(2.0 * scipy_stats.t.sf(t_stat, n_points - 2))

    # Normal approximation fallback when scipy is unavailable.
    return float(math.erfc(t_stat / math.sqrt(2.0)))


def analyze_capacity_entanglement_correlation(
    capacities: Sequence[float],
    entropies: Sequence[float],
) -> dict[str, float]:
    """Fit C = mS + b and report correlation diagnostics."""
    cap = np.asarray(list(capacities), dtype=float)
    ent = np.asarray(list(entropies), dtype=float)

    if cap.size == 0:
        raise ValueError("capacities list cannot be empty")
    if ent.size == 0:
        raise ValueError("entropies list cannot be empty")
    if cap.size != ent.size:
        raise ValueError("capacities and entropies must have the same length")

    if cap.size < 2:
        slope = float("nan")
        intercept = float(cap[0])
        correlation = float("nan")
    else:
        slope, intercept = np.polyfit(ent, cap, deg=1)
        correlation = float(np.corrcoef(ent, cap)[0, 1])

    r_squared = float(correlation * correlation) if np.isfinite(correlation) else float("nan")
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "correlation": float(correlation),
        "r_squared": r_squared,
        "p_value": _correlation_p_value(correlation, int(cap.size)),
        "n_points": int(cap.size),
    }


def analyze_capacity_entropy_ratio(results: Sequence[Mapping[str, float]]) -> dict[str, float | None]:
    """Summarize C_E/S ratio, expected to trend toward π²/6 for 1+1D CFTs."""
    ratios: list[float] = []
    for result in results:
        entropy = result.get("entropy")
        capacity = result.get("capacity_of_entanglement")
        if entropy is None or capacity is None:
            continue
        entropy_f = float(entropy)
        capacity_f = float(capacity)
        if entropy_f > 0:
            ratios.append(capacity_f / entropy_f)

    expected = float(np.pi**2 / 6.0)
    if not ratios:
        return {
            "mean_ratio": None,
            "std_ratio": None,
            "expected_ratio": expected,
            "deviation_percent": None,
        }

    mean = float(np.mean(ratios))
    return {
        "mean_ratio": mean,
        "std_ratio": float(np.std(ratios)),
        "expected_ratio": expected,
        "deviation_percent": float(100.0 * abs(mean - expected) / expected),
    }


def _default_operator_basis(local_dim: int) -> np.ndarray:
    """Return matrix-unit basis E_ij for d x d operators."""
    if local_dim <= 0:
        raise ValueError("local_dim must be positive")
    basis = np.zeros((local_dim * local_dim, local_dim, local_dim), dtype=complex)
    n = 0
    for i in range(local_dim):
        for j in range(local_dim):
            basis[n, i, j] = 1.0
            n += 1
    return basis


def _resolve_operator_basis(operator_basis: np.ndarray | None, local_dim: int) -> np.ndarray:
    if operator_basis is None:
        return _default_operator_basis(local_dim)
    basis = np.asarray(operator_basis, dtype=complex)
    if basis.ndim != 3 or basis.shape[1] != basis.shape[2]:
        raise ValueError("operator_basis must have shape (n, d, d)")
    return basis


def _resolve_ascending_map(mera_state: Any) -> Callable[[np.ndarray], np.ndarray]:
    keys = (
        "ascending_map",
        "ascend",
        "ascending_map_fn",
        "ascending_operator_map",
        "ascend_operator",
    )

    if isinstance(mera_state, Mapping):
        for key in keys:
            fn = mera_state.get(key)
            if callable(fn):
                return fn
    for key in keys:
        if hasattr(mera_state, key):
            fn = getattr(mera_state, key)
            if callable(fn):
                return fn

    raise ValueError(
        "Could not resolve ascending map from MERA state. Provide either "
        "'ascending_superoperator' or a callable like 'ascending_map'."
    )


def build_ascending_superoperator(
    mera_state: Any,
    *,
    local_dim: int | None = None,
    operator_basis: np.ndarray | None = None,
) -> np.ndarray:
    """Build ascending superoperator A for a converged MERA fixed point.

    Supported inputs:
    1. `mera_state["ascending_superoperator"]`: direct square matrix.
    2. `mera_state["ascending_superoperators"]`: list of square matrices to average.
    3. A callable map (e.g. `ascending_map(op)`) plus an operator basis.
    """
    if isinstance(mera_state, Mapping):
        if "ascending_superoperator" in mera_state:
            return _validate_square_matrix(
                np.asarray(mera_state["ascending_superoperator"], dtype=complex),
                name="ascending_superoperator",
            )
        if "ascending_superoperators" in mera_state:
            mats = [
                _validate_square_matrix(np.asarray(m, dtype=complex), name="ascending_superoperators entry")
                for m in mera_state["ascending_superoperators"]
            ]
            if not mats:
                raise ValueError("ascending_superoperators cannot be empty")
            shape = mats[0].shape
            if any(m.shape != shape for m in mats):
                raise ValueError("all ascending_superoperators entries must have the same shape")
            return np.mean(mats, axis=0)

    d = local_dim
    if d is None and isinstance(mera_state, Mapping):
        d = int(mera_state.get("local_dim", 2))
    if d is None:
        d = 2

    basis = _resolve_operator_basis(operator_basis, d)
    asc_map = _resolve_ascending_map(mera_state)

    n_ops, op_dim, _ = basis.shape
    superop = np.zeros((n_ops, n_ops), dtype=complex)
    for j in range(n_ops):
        asc_op = np.asarray(asc_map(basis[j]), dtype=complex)
        if asc_op.shape != (op_dim, op_dim):
            raise ValueError(
                "ascending_map returned operator with shape "
                f"{asc_op.shape}, expected {(op_dim, op_dim)}"
            )
        for i in range(n_ops):
            superop[i, j] = np.vdot(basis[i], asc_op)
    return superop


def diagonalize_ascending_superoperator(
    superoperator: np.ndarray,
    *,
    max_eigs: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Diagonalize A and sort eigenmodes by descending |λ|."""
    superoperator = _validate_square_matrix(np.asarray(superoperator, dtype=complex), name="superoperator")
    eigvals, eigvecs = np.linalg.eig(superoperator)
    order = np.argsort(-np.abs(eigvals))
    if max_eigs is not None:
        order = order[:max_eigs]
    return eigvals[order], eigvecs[:, order]


def scaling_dimensions_from_eigenvalues(eigenvalues: Iterable[complex], eps: float = 1e-14) -> np.ndarray:
    """Compute Δ_α = -log₂|λ_α| from ascending-superoperator eigenvalues."""
    lam = np.asarray(list(eigenvalues), dtype=complex)
    mags = np.abs(lam)
    dims = np.full(mags.shape, np.inf, dtype=float)
    mask = mags > eps
    dims[mask] = -np.log2(mags[mask])
    return dims


def extract_scaling_dimensions_mera(
    mera_state: Any,
    num_dims: int = 10,
    *,
    local_dim: int | None = None,
    operator_basis: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Build A, diagonalize it, and compute scaling dimensions Δ_α."""
    if num_dims <= 0:
        raise ValueError("num_dims must be positive")

    superop = build_ascending_superoperator(
        mera_state,
        local_dim=local_dim,
        operator_basis=operator_basis,
    )
    eigvals, eigvecs = diagonalize_ascending_superoperator(superop, max_eigs=num_dims)
    scaling_dims = scaling_dimensions_from_eigenvalues(eigvals)
    return {
        "ascending_superoperator": superop,
        "eigenvalues": eigvals,
        "eigenvectors": eigvecs,
        "scaling_dimensions": scaling_dims,
    }

