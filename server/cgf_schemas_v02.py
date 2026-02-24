"""
cgf_schemas_v02.py - Backward compatibility shim for v0.2

Re-exports from cgf_schemas_v03 for v0.2 compatibility.
This allows v0.2 adapters to import "v02" schemas while using v0.3 data structures.
"""

from cgf_schemas_v03 import (
    ActionType,
    DecisionType,
    RiskTier,
    FailMode,
    HostEventType,
    HostConfig,
    HostProposal,
    CapacitySignals,
    HostEvaluationRequest,
    HostEvaluationResponse,
    CGFDecision,
    HostOutcomeReport,
    HostEvent,
    FailModeConfig,
    ReplayPack,
    CrossHostCompatibilityReport,
    validate_event_payload,
    is_compatible_version,
    COMPATIBLE_VERSIONS,
)

from cgf_schemas_v03 import (
    ToolCallParams,
    ToolCallParams as ToolParams,
    MemoryWriteParams,
    HostContext as ExecutionContext,
    HostContext,
    HostAdapterRegistration as HostAdapterRegistrationRequest,
    HostAdapterRegistrationResponse,
    get_event_required_fields,
)

# v0.2 constant
SCHEMA_VERSION = "0.2.0"

# Convenience: auto-import everything for '*' imports
__all__ = [
    "SCHEMA_VERSION",
    "ActionType",
    "DecisionType",
    "RiskTier",
    "FailMode",
    "HostEventType",
    "HostConfig",
    "ToolParams",
    "ToolCallParams",
    "MemoryWriteParams",
    "HostAdapterRegistrationRequest",
    "HostAdapterRegistrationResponse",
    "HostProposal",
    "ExecutionContext",
    "HostContext",
    "CapacitySignals",
    "HostEvaluationRequest",
    "HostEvaluationResponse",
    "CGFDecision",
    "HostOutcomeReport",
    "HostEvent",
    "FailModeConfig",
    "ReplayPack",
    "CrossHostCompatibilityReport",
    "validate_event_payload",
    "is_compatible_version",
    "COMPATIBLE_VERSIONS",
    "get_event_required_fields",
]
