"""Policy evaluation engine.

Deterministic rule matching with explainable decisions.
Sorts rules by (-priority, id) and returns first match.
"""

import re
from datetime import datetime
from typing import Any, Dict, List, Tuple, Optional

from .fields import safe_get
from .types import (
    PolicyBundle, Rule, Condition, Operator, DecisionType,
    DecisionResult, ConstraintConfig
)


def evaluate(
    proposal: Dict[str, Any],
    context: Dict[str, Any],
    signals: Dict[str, Any],
    policy_bundle: PolicyBundle,
    default_confidence: float = 0.85
) -> DecisionResult:
    """Evaluate a proposal against policy rules.
    
    Deterministic matching:
    1. Sort rules by (-priority, id) descending
    2. First matching rule wins
    3. Generate human-readable explanation
    
    Args:
        proposal: Proposal fields (action_type, tool_name, etc.)
        context: Context fields (recent_errors, etc.)
        signals: Capacity signals (token_rate, etc.)
        policy_bundle: Loaded policy bundle with rules
        default_confidence: Default confidence if rule doesn't specify
        
    Returns:
        DecisionResult with decision, confidence, matched rules, explanation
    """
    # Build payload for field access
    payload = {
        "proposal": proposal,
        "context": context,
        "signals": signals
    }
    
    # Sort rules: higher priority first, then lexicographic by id for tie-breaking
    sorted_rules = sorted(
        policy_bundle.rules,
        key=lambda r: (-r.priority, r.id)
    )
    
    # Find first matching rule
    for rule in sorted_rules:
        if _rule_matches(rule, payload):
            return _make_decision_result(rule, payload, default_confidence)
    
    # No match - should never happen if default rule exists
    return DecisionResult(
        decision=DecisionType.ALLOW,
        decision_confidence=default_confidence,
        matched_rule_ids=[],
        explanation_text="No policy rules matched; defaulting to ALLOW"
    )


def _rule_matches(rule: Rule, payload: Dict[str, Any]) -> bool:
    """Check if all conditions in a rule match.
    
    Missing fields: condition evaluates to False (non-match)
    """
    for condition in rule.when:
        if not _condition_matches(condition, payload):
            return False
    return True


def _condition_matches(condition: Condition, payload: Dict[str, Any]) -> bool:
    """Evaluate a single condition against payload.
    
    Missing fields: returns False (safe default)
    """
    # Get field value (returns None if missing or not allowed)
    field_value = safe_get(payload, condition.field)
    
    if field_value is None:
        # Missing field = condition doesn't match
        return False
    
    op = condition.op
    target = condition.value
    
    try:
        if op == Operator.EQ:
            return field_value == target
        elif op == Operator.NE:
            return field_value != target
        elif op == Operator.IN:
            return field_value in target
        elif op == Operator.NOT_IN:
            return field_value not in target
        elif op == Operator.GT:
            return field_value > target
        elif op == Operator.GTE:
            return field_value >= target
        elif op == Operator.LT:
            return field_value < target
        elif op == Operator.LTE:
            return field_value <= target
        elif op == Operator.CONTAINS:
            return target in field_value
        elif op == Operator.REGEX:
            return bool(re.search(target, str(field_value)))
    except (TypeError, ValueError):
        # Type mismatch = condition doesn't match
        return False
    
    return False


def _make_decision_result(
    rule: Rule,
    payload: Dict[str, Any],
    default_confidence: float
) -> DecisionResult:
    """Build DecisionResult from matched rule."""
    
    confidence = rule.confidence if rule.confidence is not None else default_confidence
    
    # Generate explanation
    explanation = _generate_explanation(rule, payload)
    
    return DecisionResult(
        decision=rule.decision,
        decision_confidence=confidence,
        matched_rule_ids=[rule.id],
        explanation_text=explanation,
        constraint=rule.constraint,
        audited=rule.audit
    )


def _generate_explanation(rule: Rule, payload: Dict[str, Any]) -> str:
    """Generate human-readable explanation of rule match.
    
    Includes:
    - Rule id
    - Matched conditions summary
    - Decision and reasoning
    """
    parts = [f"Rule '{rule.id}' matched:"]
    
    # Summarize matched conditions
    if rule.when:
        cond_summaries = []
        for cond in rule.when:
            val = safe_get(payload, cond.field)
            cond_summaries.append(f"{cond.field} {cond.op} {cond.value} (got: {val})")
        parts.append(f"conditions: {' AND '.join(cond_summaries)}")
    else:
        parts.append("no conditions (default rule)")
    
    # Add decision
    parts.append(f"→ Decision: {rule.decision.value}")
    
    return "; ".join(parts)


def get_fail_mode(
    action_type: str,
    risk_tier: str,
    policy_bundle: PolicyBundle
) -> str:
    """Get fail mode for action_type:risk_tier combination.
    
    Returns configured fail_mode or 'fail_closed' as safe default.
    """
    for fm in policy_bundle.fail_modes:
        if fm.action_type == action_type and fm.risk_tier == risk_tier:
            return fm.fail_mode
    return "fail_closed"  # Safe default