"""Policy field definitions and safe accessors.

Defines the host-agnostic field namespace that policies can reference.
All field access goes through safe_get() to handle missing fields gracefully.
"""

from typing import Any, Dict, Optional

# ============== ALLOWED FIELDS ==============
# Host-agnostic field namespace - NO adapter/runtime specific fields allowed

ALLOWED_FIELDS = {
    # Proposal fields
    "proposal.action_type": "Action type (tool_call, memory_write, message_send, workflow_step)",
    "proposal.tool_name": "Tool identifier for tool_call actions",
    "proposal.size_bytes": "Size for memory operations",
    "proposal.sensitivity_hint": "Sensitivity level (low, medium, high)",
    "proposal.risk_tier": "Risk tier (low, medium, high)",
    "proposal.estimated_cost.tokens": "Estimated token cost",
    
    # Signals fields
    "signals.token_rate_60s": "Token rate over last 60 seconds",
    "signals.error_rate": "Error rate (0.0 - 1.0)",
    "signals.avg_latency_ms": "Average latency in milliseconds",
}

# Flattened field paths for quick lookup
ALLOWED_FIELD_PATHS = set(ALLOWED_FIELDS.keys())


def safe_get(payload: Dict[str, Any], field_path: str) -> Optional[Any]:
    """Safely get a field value from payload by path.
    
    Args:
        payload: Nested dict containing proposal, context, signals
        field_path: Dot-separated path like "proposal.action_type"
        
    Returns:
        Value if field exists and is in allowed list, None otherwise
    """
    # Reject unknown fields
    if field_path not in ALLOWED_FIELD_PATHS:
        return None
    
    # Navigate path
    parts = field_path.split(".")
    current = payload
    
    for part in parts:
        if not isinstance(current, dict):
            return None
        current = current.get(part)
        if current is None:
            return None
    
    return current