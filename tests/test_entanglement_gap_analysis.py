"""Tests for ED-based entanglement gap analysis."""

from __future__ import annotations

import numpy as np

from experiments.physics.entanglement_gap_analysis import (
    compute_ed_entanglement_gaps,
    entanglement_gap_from_rho,
    fit_gap_closure_log,
)


def test_entanglement_gap_from_rho():
    rho = np.diag([0.7, 0.3]).astype(complex)
    gap, ratio, spectrum = entanglement_gap_from_rho(rho)
    assert np.isclose(gap, 0.4)
    assert np.isclose(ratio, 0.4 / 0.7)
    assert np.allclose(spectrum, np.array([0.7, 0.3]))


def test_fit_gap_closure_log_recovers_A():
    L_values = [8, 16, 32, 64]
    true_A = 0.75
    gaps = [true_A * np.pi**2 / np.log(L) for L in L_values]
    fit = fit_gap_closure_log(L_values, gaps, target_delta_lambda=38.0, tolerance=5.0)

    assert np.isclose(fit["fit_A"], true_A, atol=1e-10)
    assert np.isclose(fit["A_times_pi_squared"], true_A * np.pi**2, atol=1e-10)
    assert fit["r_squared"] > 0.999999


def test_compute_ed_entanglement_gaps_small_system():
    out = compute_ed_entanglement_gaps(model="ising_open", L_values=[4])
    assert out["model"] == "ising_open"
    assert len(out["records"]) == 1
    record = out["records"][0]
    assert record["L"] == 4
    assert record["gap"] >= 0.0
    assert 0.0 <= record["gap_ratio"] <= 1.0
