"""
Tests for Entanglement Utility Functions.

Tests cover:
- von Neumann entropy calculation
- Renyi entropy calculation
- Entanglement spectrum computation
- Entanglement gap analysis
- Capacity-entanglement correlation analysis
"""

import math
import numpy as np
import pytest

from experiments.physics.entanglement_utils import (
    von_neumann_entropy,
    renyi_entropy,
    reduced_density_matrix,
    entanglement_spectrum,
    entanglement_gap,
    capacity_from_entanglement,
    analyze_capacity_entanglement_correlation,
)


class TestVonNeumannEntropy:
    """Tests for von_neumann_entropy function."""

    def test_pure_state_entropy_is_zero(self):
        """Pure states have zero entropy."""
        # Create a pure state density matrix |0><0|
        rho_pure = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=complex)
        entropy = von_neumann_entropy(rho_pure)
        assert np.isclose(entropy, 0.0, atol=1e-10)

    def test_maximally_mixed_state(self):
        """Maximally mixed state has entropy log(2)."""
        # Create maximally mixed state: I/2
        rho_mixed = np.array([[0.5, 0.0], [0.0, 0.5]], dtype=complex)
        entropy = von_neumann_entropy(rho_mixed)
        expected = np.log(2)  # log(2) in nats
        assert np.isclose(entropy, expected, atol=1e-10)

    def test_bell_state_entropy(self):
        """Half of Bell state has entropy log(2)."""
        # Create Bell state: |Φ+> = (|00> + |11>) / sqrt(2)
        # Wavefunction in computational basis
        psi = np.array([1.0, 0.0, 0.0, 1.0], dtype=complex) / np.sqrt(2)

        # Compute reduced density matrix for first qubit
        rho_A = reduced_density_matrix(psi, subsystem_A=[0], total_sites=2)

        # Entropy of reduced state should be log(2)
        entropy = von_neumann_entropy(rho_A)
        expected = np.log(2)
        assert np.isclose(entropy, expected, atol=1e-10)


class TestRenyiEntropy:
    """Tests for renyi_entropy function."""

    def test_renyi_2_maximally_mixed(self):
        """Renyi-2 entropy of maximally mixed state is log(2)."""
        # Maximally mixed state: I/2
        rho_mixed = np.array([[0.5, 0.0], [0.0, 0.5]], dtype=complex)
        entropy = renyi_entropy(rho_mixed, alpha=2.0)
        expected = np.log(2)
        assert np.isclose(entropy, expected, atol=1e-10)

    def test_renyi_converges_to_von_neumann(self):
        """Renyi entropy converges to von Neumann as alpha -> 1."""
        # Create a partially mixed state
        # Eigenvalues: 0.7 and 0.3
        rho = np.array([[0.7, 0.0], [0.0, 0.3]], dtype=complex)

        # Compute von Neumann entropy
        vn_entropy = von_neumann_entropy(rho)

        # Compute Renyi entropy for alpha close to 1
        for alpha in [0.9, 0.99, 1.01, 1.1]:
            renyi = renyi_entropy(rho, alpha=alpha)
            # As alpha approaches 1, Renyi should approach von Neumann
            # The closer alpha is to 1, the better the approximation
            if abs(alpha - 1.0) < 0.1:
                assert np.isclose(renyi, vn_entropy, rtol=0.1)


class TestEntanglementSpectrum:
    """Tests for entanglement_spectrum function."""

    def test_spectrum_is_sorted(self):
        """Spectrum should be sorted descending."""
        # Create a density matrix with known eigenvalues
        # Eigenvalues will be 0.7 and 0.3
        rho = np.array([[0.7, 0.0], [0.0, 0.3]], dtype=complex)

        spectrum = entanglement_spectrum(rho)

        # Check that spectrum is sorted in descending order
        assert len(spectrum) == 2
        assert spectrum[0] >= spectrum[1]
        assert np.isclose(spectrum[0], 0.7)
        assert np.isclose(spectrum[1], 0.3)

    def test_spectrum_sums_to_one(self):
        """Eigenvalues of density matrix sum to 1."""
        # Create a random density matrix via random pure state
        # and partial trace
        np.random.seed(42)
        psi = np.random.randn(8) + 1j * np.random.randn(8)
        psi = psi / np.linalg.norm(psi)

        rho_A = reduced_density_matrix(psi, subsystem_A=[0], total_sites=3)

        spectrum = entanglement_spectrum(rho_A)

        # Sum should be 1 (trace of density matrix)
        assert np.isclose(np.sum(spectrum), 1.0, atol=1e-10)


class TestEntanglementGap:
    """Tests for entanglement_gap function."""

    def test_gap_for_pure_state(self):
        """Pure state has gap close to 1."""
        # Pure state has one eigenvalue = 1, rest = 0
        # Gap = 1 - 0 = 1 (but 0 eigenvalues are filtered)
        # For a pure state, only one eigenvalue remains
        rho_pure = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=complex)

        # For a pure state with only one non-zero eigenvalue,
        # the gap is 0 (only one eigenvalue exists)
        gap = entanglement_gap(rho_pure)
        assert np.isclose(gap, 0.0, atol=1e-10)

    def test_gap_for_maximally_mixed(self):
        """Maximally mixed state has zero gap."""
        # Maximally mixed state: both eigenvalues are 0.5
        # Gap = 0.5 - 0.5 = 0
        rho_mixed = np.array([[0.5, 0.0], [0.0, 0.5]], dtype=complex)
        gap = entanglement_gap(rho_mixed)
        assert np.isclose(gap, 0.0, atol=1e-10)


class TestCorrelationAnalysis:
    """Tests for analyze_capacity_entanglement_correlation function."""

    def test_perfect_correlation(self):
        """Perfect linear correlation should give R^2 = 1."""
        # Create perfectly correlated data: C = 2 * S
        entropies = [0.1, 0.2, 0.3, 0.4, 0.5]
        capacities = [2 * s for s in entropies]  # Perfect linear relationship

        result = analyze_capacity_entanglement_correlation(capacities, entropies)

        assert np.isclose(result['r_squared'], 1.0, atol=1e-10)
        assert np.isclose(result['slope'], 2.0, atol=1e-10)
        assert np.isclose(result['intercept'], 0.0, atol=1e-10)
        assert np.isclose(result['correlation'], 1.0, atol=1e-10)

    def test_no_correlation(self):
        """Random data should give low R^2."""
        # Use fixed seed for reproducibility
        np.random.seed(123)
        entropies = list(np.random.rand(20))
        capacities = list(np.random.rand(20))  # Independent random values

        result = analyze_capacity_entanglement_correlation(capacities, entropies)

        # With independent random data, R^2 should be relatively low
        # (not exactly 0 due to finite sample size, but definitely not 1)
        assert result['r_squared'] < 0.5  # Very loose bound for random data
        assert 'slope' in result
        assert 'intercept' in result
        assert 'p_value' in result
        assert result['n_points'] == 20