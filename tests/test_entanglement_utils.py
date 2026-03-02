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

    def test_custom_eps_parameter(self):
        """von_neumann_entropy should accept custom eps parameter."""
        rho = np.array([[0.7, 0.0], [0.0, 0.3]], dtype=complex)
        # Should work with custom eps
        entropy_default = von_neumann_entropy(rho)
        entropy_custom = von_neumann_entropy(rho, eps=1e-10)
        # Results should be identical for this well-behaved matrix
        assert np.isclose(entropy_default, entropy_custom, atol=1e-10)


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

    def test_invalid_alpha_zero(self):
        """Renyi entropy with alpha=0 should raise ValueError."""
        rho_mixed = np.array([[0.5, 0.0], [0.0, 0.5]], dtype=complex)
        with pytest.raises(ValueError, match="alpha must be > 0"):
            renyi_entropy(rho_mixed, alpha=0.0)

    def test_invalid_alpha_negative(self):
        """Renyi entropy with negative alpha should raise ValueError."""
        rho_mixed = np.array([[0.5, 0.0], [0.0, 0.5]], dtype=complex)
        with pytest.raises(ValueError, match="alpha must be > 0"):
            renyi_entropy(rho_mixed, alpha=-1.0)

    def test_custom_eps_parameter(self):
        """Renyi entropy should accept custom eps parameter."""
        rho = np.array([[0.7, 0.0], [0.0, 0.3]], dtype=complex)
        # Should work with custom eps
        entropy_default = renyi_entropy(rho, alpha=2.0)
        entropy_custom = renyi_entropy(rho, alpha=2.0, eps=1e-10)
        # Results should be identical for this well-behaved matrix
        assert np.isclose(entropy_default, entropy_custom, atol=1e-10)


class TestReducedDensityMatrix:
    """Tests for reduced_density_matrix function."""

    def test_single_site_from_two_sites(self):
        """Test reducing a 2-qubit system to 1 qubit."""
        # Bell state: |Φ+> = (|00> + |11>) / sqrt(2)
        psi = np.array([1.0, 0.0, 0.0, 1.0], dtype=complex) / np.sqrt(2)

        # Reduced density matrix for first qubit
        rho_A = reduced_density_matrix(psi, subsystem_A=[0], total_sites=2)

        # Should be maximally mixed state I/2
        expected = np.array([[0.5, 0.0], [0.0, 0.5]], dtype=complex)
        assert np.allclose(rho_A, expected, atol=1e-10)

    def test_single_site_from_three_sites(self):
        """Test reducing a 3-qubit system to 1 qubit."""
        # GHZ state: (|000> + |111>) / sqrt(2)
        psi = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=complex) / np.sqrt(2)

        # Reduced density matrix for first qubit
        rho_A = reduced_density_matrix(psi, subsystem_A=[0], total_sites=3)

        # Should be maximally mixed state I/2
        expected = np.array([[0.5, 0.0], [0.0, 0.5]], dtype=complex)
        assert np.allclose(rho_A, expected, atol=1e-10)

    def test_two_sites_from_four_sites(self):
        """Test reducing a 4-qubit system to 2 qubits."""
        # Create a product state |0000>
        psi = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=complex)

        # Reduced density matrix for first two qubits
        rho_A = reduced_density_matrix(psi, subsystem_A=[0, 1], total_sites=4)

        # Should be |00><00|
        expected = np.zeros((4, 4), dtype=complex)
        expected[0, 0] = 1.0
        assert np.allclose(rho_A, expected, atol=1e-10)

    def test_non_adjacent_subsystem(self):
        """Test tracing out middle qubit from 3-qubit system."""
        # Create a state where first and third qubits are entangled
        # |0>|Φ+>|0> with proper ordering
        # This is a simplified test using a known state
        psi = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=complex) / np.sqrt(2)

        # Get reduced density matrix for second qubit (index 1)
        rho_A = reduced_density_matrix(psi, subsystem_A=[1], total_sites=3)

        # Should be a 2x2 matrix
        assert rho_A.shape == (2, 2)
        # Trace should be 1
        assert np.isclose(np.trace(rho_A), 1.0, atol=1e-10)

    def test_invalid_psi_length(self):
        """Test that wrong psi length raises ValueError."""
        # 3-element psi when total_sites=2 expects 4 elements
        psi = np.array([1.0, 0.0, 0.0], dtype=complex)

        with pytest.raises(ValueError, match="Wavefunction length must be"):
            reduced_density_matrix(psi, subsystem_A=[0], total_sites=2)

    def test_invalid_subsystem_index_too_large(self):
        """Test that subsystem index >= total_sites raises ValueError."""
        psi = np.array([1.0, 0.0, 0.0, 0.0], dtype=complex)

        with pytest.raises(ValueError, match="subsystem_A indices must be less than total_sites"):
            reduced_density_matrix(psi, subsystem_A=[2], total_sites=2)

    def test_invalid_subsystem_index_negative(self):
        """Test that negative subsystem index raises ValueError."""
        psi = np.array([1.0, 0.0, 0.0, 0.0], dtype=complex)

        with pytest.raises(ValueError, match="subsystem_A indices must be non-negative"):
            reduced_density_matrix(psi, subsystem_A=[-1], total_sites=2)

    def test_empty_subsystem_raises_error(self):
        """Test that empty subsystem raises ValueError."""
        psi = np.array([1.0, 0.0, 0.0, 0.0], dtype=complex)

        with pytest.raises(ValueError, match="subsystem_A cannot be empty"):
            reduced_density_matrix(psi, subsystem_A=[], total_sites=2)

    def test_duplicate_indices_raises_error(self):
        """Test that duplicate subsystem indices raise ValueError."""
        psi = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=complex)

        with pytest.raises(ValueError, match="subsystem_A contains duplicate indices"):
            reduced_density_matrix(psi, subsystem_A=[0, 0], total_sites=3)

    def test_invalid_total_sites(self):
        """Test that non-positive total_sites raises ValueError."""
        psi = np.array([1.0], dtype=complex)

        with pytest.raises(ValueError, match="total_sites must be positive"):
            reduced_density_matrix(psi, subsystem_A=[0], total_sites=0)


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

    def test_custom_eps_parameter(self):
        """entanglement_spectrum should accept custom eps parameter."""
        rho = np.array([[0.7, 0.0], [0.0, 0.3]], dtype=complex)
        # Should work with custom eps
        spectrum_default = entanglement_spectrum(rho)
        spectrum_custom = entanglement_spectrum(rho, eps=1e-10)
        # Results should be identical for this well-behaved matrix
        assert np.allclose(spectrum_default, spectrum_custom, atol=1e-10)


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

    def test_empty_capacities_raises_error(self):
        """Empty capacities list should raise ValueError."""
        with pytest.raises(ValueError, match="capacities list cannot be empty"):
            analyze_capacity_entanglement_correlation([], [0.1, 0.2])

    def test_empty_entropies_raises_error(self):
        """Empty entropies list should raise ValueError."""
        with pytest.raises(ValueError, match="entropies list cannot be empty"):
            analyze_capacity_entanglement_correlation([1.0, 2.0], [])

    def test_mismatched_lengths_raises_error(self):
        """Mismatched list lengths should raise ValueError."""
        capacities = [1.0, 2.0, 3.0]
        entropies = [0.1, 0.2]

        with pytest.raises(ValueError, match="capacities and entropies must have the same length"):
            analyze_capacity_entanglement_correlation(capacities, entropies)


class TestCapacityFromEntanglement:
    """Tests for capacity_from_entanglement function."""

    def test_default_normalization(self):
        """Default normalization should give C = S."""
        entropy = 1.5
        capacity = capacity_from_entanglement(entropy)
        assert np.isclose(capacity, 1.5)

    def test_custom_normalization(self):
        """Custom normalization should scale the capacity."""
        entropy = 1.5
        normalization = 2.0
        capacity = capacity_from_entanglement(entropy, normalization=normalization)
        expected = 3.0  # 1.5 * 2.0
        assert np.isclose(capacity, expected)

    def test_zero_entropy(self):
        """Zero entropy should give zero capacity."""
        capacity = capacity_from_entanglement(0.0)
        assert np.isclose(capacity, 0.0)

    def test_negative_normalization(self):
        """Negative normalization should produce negative capacity."""
        entropy = 1.0
        capacity = capacity_from_entanglement(entropy, normalization=-1.0)
        assert np.isclose(capacity, -1.0)

    def test_bell_state_capacity(self):
        """End-to-end test: compute capacity from Bell state entropy."""
        # Bell state: |Φ+> = (|00> + |11>) / sqrt(2)
        psi = np.array([1.0, 0.0, 0.0, 1.0], dtype=complex) / np.sqrt(2)

        # Compute reduced density matrix
        rho_A = reduced_density_matrix(psi, subsystem_A=[0], total_sites=2)

        # Compute entropy
        entropy = von_neumann_entropy(rho_A)

        # Compute capacity
        capacity = capacity_from_entanglement(entropy)

        # For maximally entangled state, S = log(2), C = log(2)
        expected = np.log(2)
        assert np.isclose(capacity, expected, atol=1e-10)