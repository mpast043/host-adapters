"""Tests for P8 Policy Engine v1.0.

Validates:
1. Deterministic rule ordering
2. Tie-breaking
3. Field validation
4. Missing field handling
5. Server integration (explanation fields)
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from cgf_policy import (
    load_policy_bundle, evaluate, compute_bundle_hash,
    Condition, Rule, PolicyBundle, DecisionResult, DecisionType,
    ALLOWED_FIELD_PATHS
)
from cgf_policy.compiler import _validate_bundle_fields
from cgf_policy.evaluator import _condition_matches


# ============== FIXTURES ==============

@pytest.fixture
def sample_bundle():
    """Create a minimal valid policy bundle."""
    return PolicyBundle(
        policy_version="1.0.0",
        bundle_hash="",
        rules=[
            Rule(
                id="high-priority",
                priority=100,
                when=[Condition(field="proposal.tool_name", op="eq", value="danger")],
                decision=DecisionType.BLOCK,
                confidence=1.0
            ),
            Rule(
                id="low-priority",
                priority=10,
                when=[Condition(field="proposal.tool_name", op="eq", value="safe")],
                decision=DecisionType.ALLOW,
                confidence=0.9
            ),
            Rule(
                id="default-allow",
                priority=0,
                when=[],
                decision=DecisionType.ALLOW,
                confidence=0.5
            )
        ]
    )


@pytest.fixture
def tie_break_bundle():
    """Bundle with same-priority rules for tie-break test."""
    return PolicyBundle(
        policy_version="1.0.0",
        bundle_hash="",
        rules=[
            Rule(
                id="aaa-rule",
                priority=50,
                when=[Condition(field="proposal.risk_tier", op="eq", value="high")],
                decision=DecisionType.BLOCK
            ),
            Rule(
                id="zzz-rule",
                priority=50,
                when=[Condition(field="proposal.risk_tier", op="eq", value="high")],
                decision=DecisionType.CONSTRAIN
            )
        ]
    )


# ============== TESTS ==============

class TestDeterministicOrdering:
    """Rule 1: Deterministic order - higher priority wins."""
    
    def test_higher_priority_wins(self, sample_bundle):
        """Higher priority rule should win over lower priority."""
        proposal = {"tool_name": "danger", "action_type": "tool_call", "risk_tier": "medium", "size_bytes": 0, "sensitivity_hint": "medium", "estimated_cost": {"tokens": 0}}
        context = {}
        signals = {}
        
        result = evaluate(proposal, context, signals, sample_bundle)
        
        assert result.decision == DecisionType.BLOCK
        assert result.matched_rule_ids == ["high-priority"]
        assert result.decision_confidence == 1.0
    
    def test_lower_priority_used_if_high_no_match(self, sample_bundle):
        """Lower priority used if high priority doesn't match."""
        proposal = {"tool_name": "safe", "action_type": "tool_call", "risk_tier": "medium", "size_bytes": 0, "sensitivity_hint": "medium", "estimated_cost": {"tokens": 0}}
        context = {}
        signals = {}
        
        result = evaluate(proposal, context, signals, sample_bundle)
        
        assert result.decision == DecisionType.ALLOW
        assert result.matched_rule_ids == ["low-priority"]


class TestTieBreaking:
    """Rule 2: Tie-break - same priority, id lexicographic wins."""
    
    def test_lexicographic_id_wins(self, tie_break_bundle):
        """With same priority, lexicographically first id should win."""
        proposal = {"tool_name": "", "action_type": "tool_call", "risk_tier": "high", "size_bytes": 0, "sensitivity_hint": "medium", "estimated_cost": {"tokens": 0}}
        context = {}
        signals = {}
        
        result = evaluate(proposal, context, signals, tie_break_bundle)
        
        # "aaa-rule" comes before "zzz-rule" lexicographically
        assert result.matched_rule_ids == ["aaa-rule"]
        assert result.decision == DecisionType.BLOCK


class TestFieldValidation:
    """Rule 3: Compiler rejects unknown fields."""
    
    def test_unknown_field_rejected(self):
        """Unknown fields in conditions should be rejected."""
        bundle_data = {
            "policy_version": "1.0.0",
            "bundle_hash": "",
            "rules": [
                {
                    "id": "bad-rule",
                    "priority": 50,
                    "when": [{"field": "unknown.field", "op": "eq", "value": "x"}],
                    "decision": "BLOCK"
                }
            ]
        }
        
        with pytest.raises(ValueError) as exc_info:
            _validate_bundle_fields(bundle_data)
        
        assert "unknown.field" in str(exc_info.value)
    
    def test_unknown_operator_rejected(self):
        """Unknown operators should be rejected."""
        bundle_data = {
            "policy_version": "1.0.0",
            "bundle_hash": "",
            "rules": [
                {
                    "id": "bad-rule",
                    "priority": 50,
                    "when": [{"field": "proposal.tool_name", "op": "magic_compare", "value": "x"}],
                    "decision": "BLOCK"
                }
            ]
        }
        
        with pytest.raises(ValueError) as exc_info:
            _validate_bundle_fields(bundle_data)
        
        assert "magic_compare" in str(exc_info.value)


class TestMissingFields:
    """Rule 4: Evaluator handles missing optional fields."""
    
    def test_missing_field_treated_as_non_match(self):
        """Missing optional fields should make condition not match (not crash)."""
        condition = Condition(
            field="proposal.tool_name",
            op="eq",
            value="danger"
        )
        
        # Payload missing proposal.tool_name
        payload = {
            "proposal": {"action_type": "tool_call"},  # no tool_name
            "context": {},
            "signals": {}
        }
        
        # Should not crash, should return False
        matches = _condition_matches(condition, payload)
        assert matches is False
    
    def test_missing_field_falls_through_to_default(self, sample_bundle):
        """Missing fields cause non-match, fall through to default rule."""
        # Proposal with no tool_name
        proposal = {"action_type": "tool_call", "risk_tier": "medium", "size_bytes": 0, "sensitivity_hint": "medium", "estimated_cost": {"tokens": 0}}
        context = {}
        signals = {}
        
        result = evaluate(proposal, context, signals, sample_bundle)
        
        # Should hit default-allow (no conditions)
        assert result.decision == DecisionType.ALLOW
        assert result.matched_rule_ids == ["default-allow"]


class TestOperators:
    """Additional operator tests."""
    
    def test_in_operator(self):
        """Test 'in' operator for list membership."""
        bundle = PolicyBundle(
            policy_version="1.0.0",
            bundle_hash="",
            rules=[
                Rule(
                    id="denylist-check",
                    priority=100,
                    when=[Condition(field="proposal.tool_name", op="in", value=["bad1", "bad2"])],
                    decision=DecisionType.BLOCK
                ),
                Rule(id="default", priority=0, when=[], decision=DecisionType.ALLOW)
            ]
        )
        
        proposal = {"tool_name": "bad1", "action_type": "tool_call", "risk_tier": "medium", "size_bytes": 0, "sensitivity_hint": "medium", "estimated_cost": {"tokens": 0}}
        result = evaluate(proposal, {}, {}, bundle)
        
        assert result.decision == DecisionType.BLOCK
        
        # Non-matching value
        proposal["tool_name"] = "good"
        result = evaluate(proposal, {}, {}, bundle)
        assert result.decision == DecisionType.ALLOW
    
    def test_gt_operator(self):
        """Test 'gt' operator for numeric comparison."""
        bundle = PolicyBundle(
            policy_version="1.0.0",
            bundle_hash="",
            rules=[
                Rule(
                    id="large-write",
                    priority=100,
                    when=[Condition(field="proposal.size_bytes", op="gt", value=1000)],
                    decision=DecisionType.CONSTRAIN
                ),
                Rule(id="default", priority=0, when=[], decision=DecisionType.ALLOW)
            ]
        )
        
        # Large size
        proposal = {"size_bytes": 5000, "action_type": "memory_write", "risk_tier": "medium", "tool_name": "", "sensitivity_hint": "medium", "estimated_cost": {"tokens": 0}}
        result = evaluate(proposal, {}, {}, bundle)
        
        assert result.decision == DecisionType.CONSTRAIN
        
        # Small size
        proposal["size_bytes"] = 500
        result = evaluate(proposal, {}, {}, bundle)
        assert result.decision == DecisionType.ALLOW


class TestExplanation:
    """Decision explanation generation."""
    
    def test_explanation_includes_rule_id(self, sample_bundle):
        """Explanation should include matched rule id."""
        proposal = {"tool_name": "danger", "action_type": "tool_call", "risk_tier": "medium", "size_bytes": 0, "sensitivity_hint": "medium", "estimated_cost": {"tokens": 0}}
        
        result = evaluate(proposal, {}, {}, sample_bundle)
        
        assert "high-priority" in result.explanation_text
        assert "BLOCK" in result.explanation_text
    
    def test_explanation_includes_conditions(self, sample_bundle):
        """Explanation should summarize matched conditions."""
        proposal = {"tool_name": "danger", "action_type": "tool_call", "risk_tier": "medium", "size_bytes": 0, "sensitivity_hint": "medium", "estimated_cost": {"tokens": 0}}
        
        result = evaluate(proposal, {}, {}, sample_bundle)
        
        assert "proposal.tool_name" in result.explanation_text
        assert "conditions:" in result.explanation_text


class TestBundleHash:
    """Bundle hash computation and verification."""
    
    def test_hash_deterministic(self, sample_bundle):
        """Same bundle should produce same hash."""
        hash1 = compute_bundle_hash(sample_bundle)
        hash2 = compute_bundle_hash(sample_bundle)
        
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA256 hex
    
    def test_hash_changes_with_content(self, sample_bundle):
        """Changing bundle content changes hash."""
        hash1 = compute_bundle_hash(sample_bundle)
        
        # Modify bundle
        sample_bundle.rules[0].priority = 99
        hash2 = compute_bundle_hash(sample_bundle)
        
        assert hash1 != hash2


class TestLoadBundle:
    """Bundle loading from file."""
    
    def test_load_v1_bundle(self, tmp_path):
        """Load actual v1 policy bundle."""
        bundle_path = Path(__file__).parent.parent / "policy" / "policy_bundle_v1.json"
        
        if bundle_path.exists():
            bundle = load_policy_bundle(bundle_path)
            
            assert bundle.policy_version == "1.0.0"
            assert bundle.bundle_hash  # Should have computed hash
            assert len(bundle.rules) > 0
            assert len(bundle.fail_modes) > 0
        else:
            pytest.skip("Policy bundle not found")
    
    def test_hash_filled_on_load(self, tmp_path):
        """Empty hash is filled on load."""
        bundle_data = {
            "policy_version": "1.0.0",
            "bundle_hash": "",  # Empty
            "rules": [
                {"id": "test", "priority": 50, "when": [], "decision": "ALLOW"}
            ]
        }
        
        path = tmp_path / "test_bundle.json"
        with open(path, "w") as f:
            json.dump(bundle_data, f)
        
        bundle = load_policy_bundle(path)
        
        assert bundle.bundle_hash  # Should be filled
        assert len(bundle.bundle_hash) == 64  # SHA256 hex


class TestSchemaIntegration:
    """Server schema integration tests."""
    
    def test_decision_result_has_policy_fields(self):
        """DecisionResult model has P8 fields."""
        result = DecisionResult(
            decision=DecisionType.ALLOW,
            decision_confidence=0.9,
            matched_rule_ids=["rule1"],
            explanation_text="Test explanation"
        )
        
        assert result.matched_rule_ids == ["rule1"]
        assert result.explanation_text == "Test explanation"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])