#!/usr/bin/env python3
"""Tests for Step 3 truth infrastructure.

Tests:
- Truth label generation and validation
- Verdict adapter (verdict.json -> SelectionResult)
- Verification engine comparison logic
- Danger gap detection
- Gate pass/fail criteria
"""

import json
import pytest
from pathlib import Path
from tempfile import TemporaryDirectory

from experiments.truth.schema import (
    TruthLabel,
    TruthArtifactRef,
    save_truth_label,
    load_truth_label,
    truth_label_digest,
)
from experiments.truth.truth_generator import TruthLabelGenerator, parse_claim_list
from experiments.selection.schema import SelectionResult, save_selection_result
from experiments.verification.verification_engine import (
    VerificationEngine,
    step3_gate_metrics,
    step3_gate_pass,
    load_truth_set,
)
from experiments.verification.ledger import VerificationLedger, LedgerEntry
from experiments.verification.run_step3 import ScienceVerdictSelectionEngine


class TestTruthLabelSchema:
    """Tests for TruthLabel schema."""

    def test_boolean_truth_label(self):
        """Test creating a boolean truth label."""
        label = TruthLabel(
            schema_version="truth.v1",
            claim_id="W02",
            truth_type="boolean",
            expected={"should_pass": True},
            tolerance={},
            source="analytical_theorem",
            confidence=1.0,
        )
        assert label.claim_id == "W02"
        assert label.truth_type == "boolean"
        assert label.expected["should_pass"] is True

    def test_scalar_truth_label(self):
        """Test creating a scalar truth label."""
        label = TruthLabel(
            schema_version="truth.v1",
            claim_id="P01",
            truth_type="scalar",
            expected={"d_s": 2.0},
            tolerance={"d_s": 0.2},
            source="analytical",
            confidence=1.0,
        )
        assert label.truth_type == "scalar"
        assert label.expected["d_s"] == 2.0

    def test_truth_label_roundtrip(self):
        """Test save and load roundtrip."""
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_label.json"
            label = TruthLabel(
                schema_version="truth.v1",
                claim_id="W03",
                truth_type="boolean",
                expected={"should_pass": True},
                tolerance={},
                source="analytical",
                confidence=1.0,
                substrate_id="test_substrate",
                seed=42,
            )
            save_truth_label(path, label)
            loaded = load_truth_label(path)
            assert loaded.claim_id == label.claim_id
            assert loaded.truth_type == label.truth_type
            assert loaded.expected == label.expected
            assert loaded.substrate_id == label.substrate_id
            assert loaded.seed == label.seed

    def test_truth_label_digest(self):
        """Test that identical labels have identical digests."""
        label1 = TruthLabel(
            schema_version="truth.v1",
            claim_id="W02",
            truth_type="boolean",
            expected={"should_pass": True},
            tolerance={},
            source="analytical",
            confidence=1.0,
        )
        label2 = TruthLabel(
            schema_version="truth.v1",
            claim_id="W02",
            truth_type="boolean",
            expected={"should_pass": True},
            tolerance={},
            source="analytical",
            confidence=1.0,
        )
        assert truth_label_digest(label1) == truth_label_digest(label2)


class TestTruthLabelGenerator:
    """Tests for TruthLabelGenerator."""

    def test_generate_observer_claim(self):
        """Test generating truth for observer claims."""
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            gen = TruthLabelGenerator(root)

            # Test W02
            label = gen.generate("W02")
            assert label.claim_id == "W02"
            assert label.truth_type == "boolean"
            assert label.expected["should_pass"] is True
            assert label.source == "analytical_theorem"

    def test_generate_regression_claim(self):
        """Test generating truth for regression claims."""
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            gen = TruthLabelGenerator(root)

            # Test claim2
            label = gen.generate("claim2_seed_perturbation")
            assert label.claim_id == "claim2_seed_perturbation"
            assert label.truth_type == "boolean"
            assert label.expected["should_pass"] is True

    def test_unknown_claim_raises(self):
        """Test that unknown claims raise KeyError."""
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            gen = TruthLabelGenerator(root)

            with pytest.raises(KeyError):
                gen.generate("UNKNOWN_CLAIM")


class TestVerificationEngine:
    """Tests for VerificationEngine."""

    def test_boolean_match(self):
        """Test boolean truth match."""
        engine = VerificationEngine()

        truth = TruthLabel(
            schema_version="truth.v1",
            claim_id="W02",
            truth_type="boolean",
            expected={"should_pass": True},
            tolerance={},
            source="analytical",
            confidence=1.0,
        )

        selection = SelectionResult(
            schema_version="selection.v1",
            claim_id="W02",
            predicted={"verdict": "PASS"},
            confidence=1.0,
        )

        ledger = engine.verify({"W02": truth}, {"W02": selection})
        assert len(ledger.entries) == 1
        assert ledger.entries[0].status == "VERIFIED"
        assert ledger.entries[0].match is True

    def test_boolean_mismatch(self):
        """Test boolean truth mismatch."""
        engine = VerificationEngine()

        truth = TruthLabel(
            schema_version="truth.v1",
            claim_id="W02",
            truth_type="boolean",
            expected={"should_pass": False},
            tolerance={},
            source="analytical",
            confidence=1.0,
        )

        selection = SelectionResult(
            schema_version="selection.v1",
            claim_id="W02",
            predicted={"verdict": "PASS"},
            confidence=1.0,
        )

        ledger = engine.verify({"W02": truth}, {"W02": selection})
        assert ledger.entries[0].status == "FALSIFIED"
        assert ledger.entries[0].match is False

    def test_danger_gap_detection(self):
        """Test danger gap detection (false positive)."""
        engine = VerificationEngine()

        # Truth says should NOT pass
        truth = TruthLabel(
            schema_version="truth.v1",
            claim_id="DANGER",
            truth_type="boolean",
            expected={"should_pass": False},
            tolerance={},
            source="analytical",
            confidence=1.0,
        )

        # Selection says PASS (false positive - DANGEROUS)
        selection = SelectionResult(
            schema_version="selection.v1",
            claim_id="DANGER",
            predicted={"verdict": "PASS"},
            confidence=1.0,
        )

        ledger = engine.verify({"DANGER": truth}, {"DANGER": selection})
        assert ledger.entries[0].danger_gap is True

    def test_missing_selection(self):
        """Test handling of missing selection."""
        engine = VerificationEngine()

        truth = TruthLabel(
            schema_version="truth.v1",
            claim_id="MISSING",
            truth_type="boolean",
            expected={"should_pass": True},
            tolerance={},
            source="analytical",
            confidence=1.0,
        )

        ledger = engine.verify({"MISSING": truth}, {})
        assert ledger.entries[0].status == "MISSING"

    def test_scalar_tolerance(self):
        """Test scalar truth with tolerance."""
        engine = VerificationEngine()

        truth = TruthLabel(
            schema_version="truth.v1",
            claim_id="P01",
            truth_type="scalar",
            expected={"d_s": 2.0},
            tolerance={"d_s": 0.2},
            source="analytical",
            confidence=1.0,
        )

        # Within tolerance
        selection = SelectionResult(
            schema_version="selection.v1",
            claim_id="P01",
            predicted={"d_s": 1.85},
            confidence=0.95,
        )

        ledger = engine.verify({"P01": truth}, {"P01": selection})
        assert ledger.entries[0].match is True

        # Outside tolerance
        selection2 = SelectionResult(
            schema_version="selection.v1",
            claim_id="P01",
            predicted={"d_s": 1.5},
            confidence=0.95,
        )

        ledger2 = engine.verify({"P01": truth}, {"P01": selection2})
        assert ledger2.entries[0].match is False


class TestGateMetrics:
    """Tests for Step 3 gate metrics."""

    def test_all_passing(self):
        """Test metrics with all claims verified."""
        ledger = VerificationLedger()
        for i in range(5):
            ledger.add(LedgerEntry(
                claim_id=f"W{i:02d}",
                status="VERIFIED",
                match=True,
                confidence=1.0,
            ))

        metrics = step3_gate_metrics(ledger, ["W00", "W01", "W02", "W03", "W04"])

        assert metrics["selection_truth"] == 1.0
        assert metrics["selection_danger_gap"] == 0
        assert metrics["selection_coverage"] == 1.0
        assert metrics["selection_confidence"] == 1.0

    def test_gate_pass_criteria(self):
        """Test gate pass/fail criteria."""
        passing_metrics = {
            "selection_truth": 0.97,
            "selection_danger_gap": 0,
            "selection_coverage": 1.0,
            "selection_confidence": 0.95,
        }

        passed, reasons = step3_gate_pass(passing_metrics)
        assert passed is True
        assert len(reasons) == 0

    def test_gate_fail_on_danger_gap(self):
        """Test gate fails on danger gaps."""
        failing_metrics = {
            "selection_truth": 0.97,
            "selection_danger_gap": 1,
            "selection_coverage": 1.0,
            "selection_confidence": 0.95,
        }

        passed, reasons = step3_gate_pass(failing_metrics)
        assert passed is False
        assert "selection_danger_gap" in reasons


class TestScienceVerdictAdapter:
    """Tests for ScienceVerdictSelectionEngine."""

    def test_load_verdict(self):
        """Test loading a verdict.json file."""
        with TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            science_dir = tmpdir / "results" / "science" / "claim_W02"
            science_dir.mkdir(parents=True)

            verdict = {
                "generated_at_utc": "2026-03-01T22:27:10.001074+00:00",
                "claim_id": "W02",
                "verdict": "PASS",
                "rationale": "Componentwise infimum was <= both vectors in all sampled cases.",
                "metrics": {"samples": 2000, "dims": 5, "failures": 0},
            }
            (science_dir / "verdict.json").write_text(json.dumps(verdict))

            engine = ScienceVerdictSelectionEngine(tmpdir)
            result = engine.run_selection("W02")

            assert result.claim_id == "W02"
            assert result.predicted["verdict"] == "PASS"
            assert result.predicted["should_pass"] is True
            assert result.confidence == 1.0  # 0 failures / 2000 samples

    def test_verdict_with_failures(self):
        """Test confidence inference with failures."""
        with TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            science_dir = tmpdir / "results" / "science" / "claim_W03"
            science_dir.mkdir(parents=True)

            verdict = {
                "claim_id": "W03",
                "verdict": "PASS",
                "metrics": {"samples": 100, "failures": 5},
            }
            (science_dir / "verdict.json").write_text(json.dumps(verdict))

            engine = ScienceVerdictSelectionEngine(tmpdir)
            result = engine.run_selection("W03")

            # Confidence should be 1 - 5/100 = 0.95
            assert result.confidence == 0.95

    def test_supported_verdict(self):
        """Test that SUPPORTED verdict is treated as pass."""
        with TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            science_dir = tmpdir / "results" / "science" / "claim2_seed321"
            science_dir.mkdir(parents=True)

            verdict = {
                "claim_id": "claim-002",
                "verdict": "SUPPORTED",
                "sample_count": 20,
            }
            (science_dir / "exp2_verdict.json").write_text(json.dumps(verdict))

            # Register the claim pattern
            from experiments.verification.run_step3 import ScienceVerdictSelectionEngine

            engine = ScienceVerdictSelectionEngine(tmpdir)
            engine.CLAIM_PATTERNS["claim2_seed_perturbation"] = ("claim2_seed321", "exp2_verdict.json")
            result = engine.run_selection("claim2_seed_perturbation")

            assert result.predicted["verdict"] == "SUPPORTED"
            assert result.predicted["should_pass"] is True


class TestParseClaimList:
    """Tests for parse_claim_list."""

    def test_parse_txt_file(self):
        """Test parsing a text file with claims."""
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "claims.txt"
            path.write_text("W02\nW03\n# comment\nW04\n")

            claims = parse_claim_list(path)
            assert claims == ["W02", "W03", "W04"]

    def test_parse_json_file(self):
        """Test parsing a JSON file with claims."""
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "claims.json"
            path.write_text('["W02", "W03", "W04"]')

            claims = parse_claim_list(path)
            assert claims == ["W02", "W03", "W04"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])