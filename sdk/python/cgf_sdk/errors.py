"""
sdk/python/cgf_sdk/errors.py

Standardized governance errors for HostAdapter implementations.
All errors include structured data for logging and debugging.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import traceback


@dataclass
class GovernanceError(Exception):
    """Base class for governance errors."""
    message: str
    error_code: str
    proposal_id: Optional[str] = None
    decision_id: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "error_code": self.error_code,
            "proposal_id": self.proposal_id,
            "decision_id": self.decision_id,
            "context": self.context
        }
    
    def __str__(self) -> str:
        return f"[{self.error_code}] {self.message}"


@dataclass 
class ActionBlockedError(GovernanceError):
    """Action was blocked by CGF policy."""
    reason_code: str = "BLOCKED_BY_POLICY"
    
    def __init__(self, message: str, reason_code: str = "BLOCKED_BY_POLICY", **kwargs):
        super().__init__(
            message=message,
            error_code="ACTION_BLOCKED",
            **kwargs
        )
        self.reason_code = reason_code


@dataclass
class ActionConstrainedError(GovernanceError):
    """Action requires constraints that failed to apply."""
    constraint_type: Optional[str] = None
    constraint_error: Optional[str] = None
    
    def __init__(
        self,
        message: str,
        constraint_type: Optional[str] = None,
        constraint_error: Optional[str] = None,
        **kwargs
    ):
        super().__init__(
            message=message,
            error_code="CONSTRAINT_FAILED",
            **kwargs
        )
        self.constraint_type = constraint_type
        self.constraint_error = constraint_error


@dataclass
class FailModeError(GovernanceError):
    """CGF unreachable; fail mode applied."""
    fail_mode: str = "fail_closed"
    applied_decision: str = "BLOCK"
    risk_tier: str = "high"
    
    def __init__(
        self,
        message: str,
        fail_mode: str = "fail_closed",
        applied_decision: str = "BLOCK",
        risk_tier: str = "high",
        **kwargs
    ):
        super().__init__(
            message=message,
            error_code="FAIL_MODE_APPLIED",
            **kwargs
        )
        self.fail_mode = fail_mode
        self.applied_decision = applied_decision
        self.risk_tier = risk_tier
    
    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update({
            "fail_mode": self.fail_mode,
            "applied_decision": self.applied_decision,
            "risk_tier": self.risk_tier
        })
        return base


@dataclass
class CGFConnectionError(GovernanceError):
    """Failed to connect to CGF server."""
    retry_count: int = 0
    
    def __init__(self, message: str, retry_count: int = 0, **kwargs):
        super().__init__(
            message=message,
            error_code="CGF_UNREACHABLE",
            **kwargs
        )
        self.retry_count = retry_count


@dataclass
class CGFEvaluationError(GovernanceError):
    """CGF returned error during evaluation."""
    status_code: Optional[int] = None
    response_body: Optional[str] = None
    
    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        response_body: Optional[str] = None,
        **kwargs
    ):
        super().__init__(
            message=message,
            error_code="EVALUATION_ERROR",
            **kwargs
        )
        self.status_code = status_code
        self.response_body = response_body


@dataclass
class CGFRegistryError(GovernanceError):
    """CGF registration failed."""
    status_code: Optional[int] = None
    
    def __init__(self, message: str, status_code: Optional[int] = None, **kwargs):
        super().__init__(
            message=message,
            error_code="REGISTRY_ERROR",
            **kwargs
        )
        self.status_code = status_code


@dataclass
class SchemaValidationError(GovernanceError):
    """Payload failed schema validation."""
    validation_errors: List[str] = field(default_factory=list)
    
    def __init__(self, message: str, validation_errors: Optional[List[str]] = None, **kwargs):
        super().__init__(
            message=message,
            error_code="SCHEMA_VIOLATION",
            **kwargs
        )
        self.validation_errors = validation_errors or []


@dataclass
class ProposalExpiredError(GovernanceError):
    """Proposal TTL exceeded."""
    ttl_seconds: Optional[float] = None
    
    def __init__(self, message: str, ttl_seconds: Optional[float] = None, **kwargs):
        super().__init__(
            message=message,
            error_code="PROPOSAL_EXPIRED",
            **kwargs
        )
        self.ttl_seconds = ttl_seconds


@dataclass
class DecisionNotMadeError(GovernanceError):
    """Expected decision but none produced."""
    pass


class ErrorConverter:
    """Convert standard exceptions to GovernanceError."""
    
    @staticmethod
    def from_exception(
        exc: Exception,
        proposal_id: Optional[str] = None
    ) -> GovernanceError:
        """Wrap any exception as GovernanceError."""
        if isinstance(exc, GovernanceError):
            return exc
        
        return GovernanceError(
            message=str(exc),
            error_code="UNEXPECTED_ERROR",
            proposal_id=proposal_id,
            context={
                "exception_type": type(exc).__name__,
                "traceback": traceback.format_exc()
            }
        )
