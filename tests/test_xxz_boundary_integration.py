"""
Integration tests for XXZ boundary benchmark.

Compares observed ΔAIC from framework model fitting against theoretical expectations:

- Critical systems (Δ ≤ 1): Theory predicts log-linear scaling (c=1), so expected ΔAIC > 0
- Gapped systems (Δ > 1): Theory predicts saturation (c=0), so expected ΔAIC < 0

Sign convention in this module:
    delta_aic = AIC_saturating - AIC_loglinear
Lower AIC is preferred.

This validates that the framework's model selection correctly identifies the physics.
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest


DELTAS = ("0.50", "1.00", "1.10", "1.50", "2.00")
EXPECTED_AIC_SIGN = {
    "0.50": "positive",  # critical -> log-linear preferred
    "1.00": "positive",  # critical -> log-linear preferred
    "1.10": "negative",  # gapped -> saturating preferred
    "1.50": "negative",  # gapped -> saturating preferred
    "2.00": "negative",  # gapped -> saturating preferred
}


def _load_metadata(run_dir: Path) -> dict:
    metadata_path = run_dir / "metadata.json"
    if not metadata_path.exists():
        return {}
    try:
        with open(metadata_path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _is_strict_correctness_gate_run(run_dir: Path) -> bool:
    """True only when artifacts are marked as physics-correctness gate outputs."""
    metadata = _load_metadata(run_dir)
    if not metadata:
        return False
    if metadata.get("framework_correctness_gate") is True:
        return True
    validation_mode = metadata.get("validation_mode")
    if validation_mode == "integration_proxy_models":
        return False
    return False


def _delta_aic_sign(delta_aic: float, tol: float = 1e-12) -> str:
    if delta_aic > tol:
        return "positive"
    if delta_aic < -tol:
        return "negative"
    return "zero"


def _winner_from_delta_aic(delta_aic: float, tol: float = 1e-12) -> str:
    # By definition here: delta_aic = AIC_saturating - AIC_loglinear
    # Lower AIC is preferred.
    if delta_aic > tol:
        return "log-linear"
    if delta_aic < -tol:
        return "saturating"
    return "tie"


@pytest.fixture(scope="module")
def framework_output_dir() -> Path:
    """Resolve framework output location for integration artifacts.

    Override with XXZ_FRAMEWORK_OUTPUT_DIR when running outside the default repo layout.
    """
    configured = os.getenv("XXZ_FRAMEWORK_OUTPUT_DIR")
    if configured:
        output_dir = Path(configured).expanduser()
    else:
        output_dir = Path(__file__).resolve().parents[1] / "outputs" / "xxz_boundary_test"

    if not output_dir.exists():
        pytest.skip(
            f"Framework output directory not found: {output_dir}. "
            "Set XXZ_FRAMEWORK_OUTPUT_DIR or run the XXZ boundary benchmark first."
        )

    def is_xxz_fit_run_dir(path: Path) -> bool:
        return all((path / f"delta_{d}" / "fits.json").exists() for d in DELTAS)

    def is_framework_xxz_run(path: Path) -> bool:
        metadata_path = path / "metadata.json"
        if not metadata_path.exists():
            return True
        try:
            with open(metadata_path, encoding="utf-8") as f:
                metadata = json.load(f)
            # Reject literature-baseline outputs for this integration test.
            return metadata.get("test") == "P2_XXZ_BOUNDARY"
        except Exception:
            return False

    candidates = []
    if is_xxz_fit_run_dir(output_dir) and is_framework_xxz_run(output_dir):
        candidates.append(output_dir)

    for child in output_dir.iterdir():
        if child.is_dir() and is_xxz_fit_run_dir(child) and is_framework_xxz_run(child):
            candidates.append(child)

    if not candidates:
        pytest.skip(
            f"No XXZ fits artifacts found under {output_dir}. "
            "Expected delta_*/fits.json in output dir or one-level run_* subdirs."
        )

    # Prefer the most recently modified candidate so append-mode runs are picked up.
    return max(candidates, key=lambda p: p.stat().st_mtime)


class TestXXZBoundaryIntegration:
    """Integration tests for XXZ boundary benchmark against framework results."""

    def test_framework_runs_exist(self, framework_output_dir):
        """Verify framework output directory and delta subdirectories exist."""
        for delta_str in DELTAS:
            delta_dir = framework_output_dir / f"delta_{delta_str}"
            assert delta_dir.exists(), f"Delta directory not found: {delta_dir}"

    def test_fits_json_structure(self, framework_output_dir):
        """Verify fits.json files have expected structure."""
        for delta_str in DELTAS:
            fits_path = framework_output_dir / f"delta_{delta_str}" / "fits.json"
            assert fits_path.exists(), f"fits.json not found for delta={delta_str}"

            with open(fits_path, encoding="utf-8") as f:
                fits = json.load(f)

            # Check required keys
            assert "loglinear" in fits, f"Missing 'loglinear' key in {fits_path}"
            assert "saturating" in fits, f"Missing 'saturating' key in {fits_path}"
            assert "delta_aic" in fits, f"Missing 'delta_aic' key in {fits_path}"
            assert "delta_bic" in fits, f"Missing 'delta_bic' key in {fits_path}"

    def test_delta_aic_signs_match_physics(self, framework_output_dir):
        """Verify ΔAIC signs match expected physics (critical vs gapped).

        Sign convention: delta_aic = AIC_sat - AIC_log
        - Critical systems (Δ ≤ 1): log-linear should win => ΔAIC > 0
        - Gapped systems (Δ > 1): saturating should win => ΔAIC < 0
        """
        if not _is_strict_correctness_gate_run(framework_output_dir):
            pytest.skip(
                "Selected artifacts are integration/proxy outputs (framework_correctness_gate=false). "
                "Run ED correctness-gate artifacts for strict physics sign checks."
            )

        mismatches = []
        for delta_str, expected_sign in EXPECTED_AIC_SIGN.items():
            fits_path = framework_output_dir / f"delta_{delta_str}" / "fits.json"

            with open(fits_path, encoding="utf-8") as f:
                fits = json.load(f)

            delta_aic = fits["delta_aic"]
            observed_aic_sign = _delta_aic_sign(delta_aic)

            matches = observed_aic_sign == expected_sign

            if not matches:
                mismatches.append({
                    "delta": float(delta_str),
                    "observed_sign": observed_aic_sign,
                    "expected_sign": expected_sign,
                    "delta_aic": delta_aic,
                })

        # Report mismatches
        if mismatches:
            print("\n=== ΔAIC Sign Mismatches ===")
            for m in mismatches:
                print(f"  Δ={m['delta']}: observed={m['observed_sign']} (ΔAIC={m['delta_aic']:.4f}), "
                      f"expected={m['expected_sign']}")

        assert len(mismatches) == 0, (
            f"Found {len(mismatches)} ΔAIC sign mismatches. "
            "Framework model selection does not match expected XXZ boundary physics."
        )

    @pytest.mark.parametrize("delta_str", DELTAS)
    def test_delta_aic_values(self, framework_output_dir, delta_str):
        """Check actual ΔAIC values from framework fits."""
        fits_path = framework_output_dir / f"delta_{delta_str}" / "fits.json"

        with open(fits_path, encoding="utf-8") as f:
            fits = json.load(f)

        delta_aic = fits["delta_aic"]
        aic_saturating = fits["saturating"]["aic"]
        aic_loglinear = fits["loglinear"]["aic"]

        # Verify ΔAIC = AIC_sat - AIC_log
        computed_delta_aic = aic_saturating - aic_loglinear
        assert np.isclose(delta_aic, computed_delta_aic, atol=1e-10), (
            f"Delta {delta_str}: delta_aic {delta_aic} != computed {computed_delta_aic}"
        )

        # Verify model preference implied by delta_aic matches raw AIC ordering
        winner = _winner_from_delta_aic(delta_aic)
        if winner == "saturating":
            assert aic_saturating < aic_loglinear
        elif winner == "log-linear":
            assert aic_loglinear < aic_saturating
        else:
            assert np.isclose(aic_saturating, aic_loglinear, atol=1e-10)

    def test_aic_model_comparison_data(self, framework_output_dir):
        """Verify model comparison data structure is complete."""
        for delta_str in DELTAS:
            fits_path = framework_output_dir / f"delta_{delta_str}" / "fits.json"

            with open(fits_path, encoding="utf-8") as f:
                fits = json.load(f)

            # Log-linear model should have: a, b, rss, aic, bic
            loglinear = fits["loglinear"]
            assert "a" in loglinear, f"Missing 'a' in loglinear for delta {delta_str}"
            assert "b" in loglinear, f"Missing 'b' in loglinear for delta {delta_str}"
            assert "rss" in loglinear, f"Missing 'rss' in loglinear for delta {delta_str}"
            assert "aic" in loglinear, f"Missing 'aic' in loglinear for delta {delta_str}"
            assert "bic" in loglinear, f"Missing 'bic' in loglinear for delta {delta_str}"

            # Saturating model should have: S_inf, c, alpha, rss, aic, bic
            saturating = fits["saturating"]
            assert "S_inf" in saturating, f"Missing 'S_inf' in saturating for delta {delta_str}"
            assert "c" in saturating, f"Missing 'c' in saturating for delta {delta_str}"
            assert "alpha" in saturating, f"Missing 'alpha' in saturating for delta {delta_str}"
            assert "rss" in saturating, f"Missing 'rss' in saturating for delta {delta_str}"
            assert "aic" in saturating, f"Missing 'aic' in saturating for delta {delta_str}"
            assert "bic" in saturating, f"Missing 'bic' in saturating for delta {delta_str}"

    def test_observed_vs_expected_delta_aic_summary(self, framework_output_dir):
        """Generate a summary comparing observed vs expected ΔAIC signs."""
        if not _is_strict_correctness_gate_run(framework_output_dir):
            pytest.skip(
                "Selected artifacts are integration/proxy outputs (framework_correctness_gate=false). "
                "Run ED correctness-gate artifacts for strict physics sign checks."
            )

        # Collect results
        results = []
        expected = EXPECTED_AIC_SIGN

        all_match = True
        for delta_str, expected_sign in expected.items():
            fits_path = framework_output_dir / f"delta_{delta_str}" / "fits.json"
            with open(fits_path, encoding="utf-8") as f:
                fits = json.load(f)

            delta_aic = fits["delta_aic"]
            observed_sign = _delta_aic_sign(delta_aic)
            matches = observed_sign == expected_sign
            all_match = all_match and matches

            results.append({
                "delta": float(delta_str),
                "observed_delta_aic": delta_aic,
                "observed_aic_sign": observed_sign,
                "expected_aic_sign": expected_sign,
                "matches": matches,
            })

        # Print summary (can be used for debugging or reporting)
        print("\n" + "=" * 70)
        print("XXZ Boundary Framework Integration Test Summary")
        print("=" * 70)
        print(f"{'Δ':>6} | {'observed ΔAIC':>14} | {'observed sign':>14} | {'expected sign':>14} | {'match':>8}")
        print("-" * 70)
        for r in results:
            print(f"{r['delta']:>6.2f} | {r['observed_delta_aic']:>14.4f} | {r['observed_aic_sign']:>14} | {r['expected_aic_sign']:>14} | {str(r['matches']):>8}")
        print("=" * 70)
        print(f"All matches: {all_match}")

        # Verify all match
        assert all_match, "Not all ΔAIC signs match expected physics"

    def test_consistency_across_runs(self, framework_output_dir):
        """Test that multiple Δ points produce consistent results within expected range."""
        for delta_str in DELTAS:
            fits_path = framework_output_dir / f"delta_{delta_str}" / "fits.json"

            with open(fits_path, encoding="utf-8") as f:
                fits = json.load(f)

            delta_aic = fits["delta_aic"]

            # Verify reasonable magnitude (not infinite or NaN)
            assert np.isfinite(delta_aic), f"Delta {delta_str}: delta_aic is not finite"
            assert abs(delta_aic) < 1000, f"Delta {delta_str}: delta_aic magnitude seems excessive"

            # Verify RSS is positive (good fit convergence)
            assert fits["loglinear"]["rss"] > 0, f"Delta {delta_str}: log-linear RSS should be positive"
            assert fits["saturating"]["rss"] > 0, f"Delta {delta_str}: saturating RSS should be positive"

    def test_framework_version_metadata(self, framework_output_dir):
        """Check if framework output includes version info."""
        metadata_path = framework_output_dir / "metadata.json"

        if metadata_path.exists():
            with open(metadata_path, encoding="utf-8") as f:
                metadata = json.load(f)

            # Optional: verify framework version if present
            assert "version" in metadata or "test" in metadata, (
                "Framework output should include version or test name in metadata"
            )

    def test_observed_delta_aic_storage_format(self, framework_output_dir):
        """Verify framework stores observed model-selection data in expected format.

        The framework's fits.json should contain:
        - observed_delta_aic: float (computed from AIC values)
        - aic_saturating: float
        - aic_loglinear: float
        - winner_model: "saturating" or "log-linear"
        """
        for delta_str in DELTAS:
            fits_path = framework_output_dir / f"delta_{delta_str}" / "fits.json"

            with open(fits_path, encoding="utf-8") as f:
                fits = json.load(f)

            # Observed values (computed by framework)
            aic_saturating = fits["saturating"]["aic"]
            aic_loglinear = fits["loglinear"]["aic"]
            observed_delta_aic = fits["delta_aic"]

            # Verify structure
            assert isinstance(observed_delta_aic, (int, float)), (
                f"Delta {delta_str}: observed_delta_aic should be numeric"
            )
            assert isinstance(aic_saturating, (int, float)), (
                f"Delta {delta_str}: aic_saturating should be numeric"
            )
            assert isinstance(aic_loglinear, (int, float)), (
                f"Delta {delta_str}: aic_loglinear should be numeric"
            )

            # Verify ΔAIC calculation
            computed_delta_aic = aic_saturating - aic_loglinear
            assert np.isclose(observed_delta_aic, computed_delta_aic, atol=1e-10), (
                f"Delta {delta_str}: delta_aic mismatch"
            )

            # Verify winner implied by delta_aic is consistent with raw AIC values
            winner_model = _winner_from_delta_aic(observed_delta_aic)
            if winner_model == "saturating":
                assert aic_saturating < aic_loglinear
            elif winner_model == "log-linear":
                assert aic_loglinear < aic_saturating
            else:
                assert np.isclose(aic_saturating, aic_loglinear, atol=1e-10)
