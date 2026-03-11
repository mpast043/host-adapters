"""Tests for MERA ascending-superoperator scaling dimension extraction."""

from __future__ import annotations

import numpy as np

from experiments.physics.entanglement_utils import (
    build_ascending_superoperator,
    extract_scaling_dimensions_mera,
    scaling_dimensions_from_eigenvalues,
)


def test_build_ascending_superoperator_passthrough_matrix():
    matrix = np.diag([1.0, 0.5, 0.25]).astype(complex)
    superop = build_ascending_superoperator({"ascending_superoperator": matrix})
    assert np.allclose(superop, matrix)


def test_build_ascending_superoperator_from_callable():
    def asc_map(operator: np.ndarray) -> np.ndarray:
        return 0.5 * operator

    superop = build_ascending_superoperator({"ascending_map": asc_map}, local_dim=2)
    assert superop.shape == (4, 4)
    assert np.allclose(superop, 0.5 * np.eye(4))


def test_scaling_dimensions_formula():
    eigenvalues = np.array([1.0, 0.5, -0.25, 0.0], dtype=complex)
    dims = scaling_dimensions_from_eigenvalues(eigenvalues)
    assert np.isclose(dims[0], 0.0)
    assert np.isclose(dims[1], 1.0)
    assert np.isclose(dims[2], 2.0)
    assert np.isinf(dims[3])


def test_extract_scaling_dimensions_mera_end_to_end():
    matrix = np.diag([1.0, 0.5, 0.25, 0.125]).astype(complex)
    out = extract_scaling_dimensions_mera({"ascending_superoperator": matrix}, num_dims=4)
    dims = out["scaling_dimensions"]
    assert np.allclose(dims, np.array([0.0, 1.0, 2.0, 3.0]))
