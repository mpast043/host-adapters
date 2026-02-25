"""CGF Policy Engine v1.0 - Deterministic + Explainable.

Host-agnostic policy evaluation with deterministic rule matching
and human-readable explanations.
"""

from .fields import ALLOWED_FIELDS, safe_get, ALLOWED_FIELD_PATHS
from .types import Condition, Rule, PolicyBundle, DecisionResult, DecisionType, Operator
from .compiler import load_policy_bundle, compute_bundle_hash, validate_policy_bundle
from .evaluator import evaluate, get_fail_mode

__version__ = "1.0.0"

__all__ = [
    "ALLOWED_FIELDS",
    "ALLOWED_FIELD_PATHS",
    "safe_get",
    "Condition",
    "Rule",
    "PolicyBundle",
    "DecisionResult",
    "DecisionType",
    "Operator",
    "load_policy_bundle",
    "compute_bundle_hash",
    "validate_policy_bundle",
    "evaluate",
    "get_fail_mode",
]