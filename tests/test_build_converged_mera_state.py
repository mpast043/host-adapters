"""Tests for ascending superoperator extraction from MERA isometries."""

from __future__ import annotations

import numpy as np
import quimb.tensor as qtn

from experiments.physics.build_converged_mera_state import _ascending_superoperator_from_isometry


def test_ascending_superoperator_from_isometry_shape_and_finiteness():
    # Identity-like 2x2 -> 4 isometry reshaped as rank-3 tensor.
    iso = np.eye(4, dtype=float).reshape(2, 2, 4)
    tensor = qtn.Tensor(iso, inds=("i0", "i1", "ic"))

    superop, dims = _ascending_superoperator_from_isometry(tensor)
    assert superop.shape == (16, 4)
    assert dims["d_left"] == 2
    assert dims["d_right"] == 2
    assert dims["chi"] == 4
    assert np.all(np.isfinite(superop))
