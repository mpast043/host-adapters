"""Policy engine type definitions (Pydantic models).

Defines the data structures for conditions, rules, policy bundles,
and decision results. All structures are JSON-serializable for
reproducibility and audit trails.
"""

from enum import Enum
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, field_validator


class Operator(str, Enum):
    """Allowed comparison operators."""
    EQ = "eq"
    NE = "ne"
    IN = "in"
    NOT_IN = "not_in"
    GT = "gt"
    GTE = "gte"
    LT = "lt"
    LTE = "lte"
    CONTAINS = "contains"
    REGEX = "regex"


class DecisionType(str, Enum):
    """Governance decision types."""
    ALLOW = "ALLOW"
    BLOCK = "BLOCK"
    CONSTRAIN = "CONSTRAIN"
    DEFER = "DEFER"
    AUDIT = "AUDIT"


class Condition(BaseModel):
    """Single condition: field OP value."""
    field: str = Field(..., description="Dot-separated field path")
    op: Operator = Field(..., description="Comparison operator")
    value: Any = Field(..., description="Value to compare against")
    
    @field_validator("field")
    @classmethod
    def validate_field_path(cls, v: str) -> str:
        from .fields import ALLOWED_FIELD_PATHS
        if v not in ALLOWED_FIELD_PATHS:
            raise ValueError(f"Field '{v}' not in allowed fields list")
        return v


class ConstraintConfig(BaseModel):
    """Constraint configuration for CONSTRAIN decisions."""
    type: str = Field(..., description="Constraint type")
    params: Dict[str, Any] = Field(default_factory=dict)


class Rule(BaseModel):
    """Policy rule with conditions and decision."""
    id: str = Field(..., description="Unique rule identifier")
    priority: int = Field(default=0, ge=0, le=1000, description="Higher = evaluated first")
    when: List[Condition] = Field(default_factory=list, description="Conditions (all must match)")
    decision: DecisionType = Field(..., description="Decision if all conditions match")
    constraint: Optional[ConstraintConfig] = None
    audit: bool = Field(default=False, description="Force audit logging")
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Override confidence")


class FailModeConfig(BaseModel):
    """Fail mode configuration per action_type:risk_tier."""
    action_type: str
    risk_tier: str
    fail_mode: str = Field(..., pattern="^(fail_closed|fail_open|defer)$")
    timeout_ms: int = Field(default=500, gt=0)
    rationale: str = ""


class PolicyBundle(BaseModel):
    """Complete policy bundle with rules and metadata."""
    policy_version: str = Field(..., pattern=r"^\d+\.\d+\.\d+$")
    bundle_hash: str = Field(default="", description="SHA256 of canonical JSON (filled by compiler)")
    rules: List[Rule] = Field(default_factory=list)
    fail_modes: List[FailModeConfig] = Field(default_factory=list)
    
    @field_validator("rules")
    @classmethod
    def validate_unique_rule_ids(cls, v: List[Rule]) -> List[Rule]:
        ids = [r.id for r in v]
        if len(ids) != len(set(ids)):
            from collections import Counter
            dups = [item for item, count in Counter(ids).items() if count > 1]
            raise ValueError(f"Duplicate rule ids: {dups}")
        return v


class DecisionResult(BaseModel):
    """Result of policy evaluation."""
    decision: DecisionType
    decision_confidence: float = Field(ge=0.0, le=1.0)
    matched_rule_ids: List[str] = Field(default_factory=list)
    explanation_text: str = ""
    constraint: Optional[ConstraintConfig] = None
    audited: bool = False