"""
cgf_server_v03.py - Capacity Governance Framework Server v0.3

Features:
- Data-driven policy from JSON config
- Backward compatible with v0.2.x schemas
- Support for multiple hosts (OpenClaw, LangGraph)
- Enhanced replay with cross-host comparison

Policy v0.3:
- Fail modes configured via policy_config_v03.json
- Risk tier inference from side_effects (not hardcoded)
- Host-agnostic: no branching on host_type
"""

import hmac
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel

# Import v0.3 schemas
try:
    from cgf_schemas_v03 import (
        SCHEMA_VERSION,
        MIN_COMPATIBLE_VERSION,
        is_compatible_version,
        normalize_for_processing,
        PolicyConfig,
        HostAdapterRegistration,
        HostAdapterRegistrationResponse,
        HostEvaluationRequest,
        HostEvaluationResponse,
        HostOutcomeReport,
        HostEvent,
        CGFDecision,
        ConstraintConfig,
        ReplayPack,
        HostEventType,
        ActionType,
        DecisionType,
        RiskTier
    )
    HAS_V03 = True
except ImportError:
    # Fallback to v0.2
    from cgf_schemas_v02 import (
        SCHEMA_VERSION,
        HostAdapterRegistration,
        HostAdapterRegistrationResponse,
        HostEvaluationRequest,
        HostEvaluationResponse,
        HostOutcomeReport,
        HostEvent,
        CGFDecision,
        ConstraintConfig,
        ReplayPack,
        HostEventType,
        ActionType,
        DecisionType,
        RiskTier
    )
    HAS_V03 = False
    print("Warning: Using v0.2 schemas (cgf_schemas_v03 not found)")

# ============== UTILITIES ==============

def generate_id(prefix: str = "id") -> str:
    """Generate a unique ID with prefix."""
    import uuid
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


# ============== POLICY ENGINE v1.0 ==============
# P8: Load policy bundle at startup
try:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from cgf_policy import load_policy_bundle, evaluate, PolicyBundle, get_fail_mode
    from cgf_policy.types import DecisionType as PolicyDecisionType
    HAS_POLICY_ENGINE = True
except ImportError as e:
    HAS_POLICY_ENGINE = False
    print(f"Warning: Policy engine not available: {e}")

# Load policy bundle from env or default
POLICY_BUNDLE_PATH = Path(os.environ.get("CGF_POLICY_BUNDLE_PATH", Path(__file__).parent.parent / "policy" / "policy_bundle_v1.json"))
POLICY_BUNDLE: PolicyBundle | None = None

if HAS_POLICY_ENGINE and POLICY_BUNDLE_PATH.exists():
    try:
        POLICY_BUNDLE = load_policy_bundle(POLICY_BUNDLE_PATH)
        print(f"Policy bundle loaded: {POLICY_BUNDLE.policy_version} (hash: {POLICY_BUNDLE.bundle_hash[:16]}...)")
    except Exception as e:
        print(f"Warning: Failed to load policy bundle: {e}")
else:
    print(f"Warning: Policy bundle not found at {POLICY_BUNDLE_PATH}")

# ============== CONFIGURATION ==============

# Strict mode: when CGF_STRICT=1 the catch-all default-allow rule is replaced
# with AUDIT so unknown tools are flagged rather than silently permitted.
# Dev default is off (CGF_STRICT=0) to avoid breaking existing workflows.
CGF_STRICT: bool = os.environ.get("CGF_STRICT", "0") == "1"
if CGF_STRICT:
    print("⚠ CGF_STRICT=1: default-allow rule overridden to AUDIT for unknown tools")

# Optional bearer-token auth for write endpoints.
# When CGF_AUTH_TOKEN is empty (default) auth is disabled — all requests pass.
# When set, POST /v1/register, /v1/evaluate, /v1/outcomes/report require
# "Authorization: Bearer <token>". GET /v1/health is always unprotected.
CGF_AUTH_TOKEN: str = os.environ.get("CGF_AUTH_TOKEN", "")
if CGF_AUTH_TOKEN:
    print("CGF_AUTH_TOKEN set: bearer-token auth enabled on write endpoints")


def require_auth(authorization: Optional[str] = Header(None)) -> None:
    """FastAPI dependency — enforces bearer token when CGF_AUTH_TOKEN is set."""
    if not CGF_AUTH_TOKEN:
        return
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Unauthorized")
    token = authorization[len("Bearer "):]
    if not hmac.compare_digest(token.encode(), CGF_AUTH_TOKEN.encode()):
        raise HTTPException(status_code=401, detail="Unauthorized")

# Load policy config
POLICY_CONFIG_PATH = Path(__file__).parent / "policy_config_v03.json"
if POLICY_CONFIG_PATH.exists():
    with open(POLICY_CONFIG_PATH) as f:
        POLICY_DATA = json.load(f)
else:
    POLICY_DATA = {}
    print("Warning: policy_config_v03.json not found, using defaults")

# ============== STATE ==============

adapters: Dict[str, Dict] = {}
proposals: Dict[str, Dict] = {}
decisions: Dict[str, CGFDecision] = {}
outcomes: Dict[str, HostOutcomeReport] = {}
events: List[HostEvent] = []

# ============== METRICS STATE ==============
metrics_state = {
    "decision_count": {},  # {decision: {action_type: {risk_tier: count}}}
    "decision_latency_ms": [],  # List of latency values
    "outcomes_count": {"committed": {}, "quarantined": {}},  # {action_type: count}
    "evaluate_timeouts": 0,
    "proposals_received": 0,
    "proposals_evaluated": 0,
}

LATENCY_BUCKETS = [10, 50, 100, 250, 500, 1000, 2500, 5000]

# ============== APP ==============

app = FastAPI(title="CGF Server v0.3", version="0.3.0")

# ============== STRUCTURED LOGGING ==============

def log_structured(level: str, message: str, **fields):
    """Emit structured JSON log for observability."""
    import time
    log_entry = {
        "timestamp": time.time(),
        "level": level,
        "message": message,
        "source": "cgf_server_v03",
        **fields
    }
    print(json.dumps(log_entry), flush=True)


def log_correlated(
    level: str,
    message: str,
    *,
    adapter_id: Optional[str] = None,
    proposal_id: Optional[str] = None,
    decision_id: Optional[str] = None,
    event_type: Optional[str] = None,
    **extra,
):
    """Structured log with standard governance correlation IDs.

    Always includes adapter_id, proposal_id, decision_id, and event_type so
    that log lines can be filtered by a single grep across any of these keys.
    """
    log_structured(
        level,
        message,
        **{k: v for k, v in {
            "adapter_id": adapter_id,
            "proposal_id": proposal_id,
            "decision_id": decision_id,
            "event_type": event_type,
            **extra,
        }.items() if v is not None},
    )

# ============== METRICS HELPERS ==============

def record_decision_metrics(decision: DecisionType, action_type: str, risk_tier: str, latency_ms: float):
    """Record decision metrics."""
    decision_key = decision.value if hasattr(decision, 'value') else str(decision)
    action_key = action_type.value if hasattr(action_type, 'value') else str(action_type)
    risk_key = risk_tier.value if hasattr(risk_tier, 'value') else str(risk_tier)
    
    # Count by decision/action/risk
    if decision_key not in metrics_state["decision_count"]:
        metrics_state["decision_count"][decision_key] = {}
    if action_key not in metrics_state["decision_count"][decision_key]:
        metrics_state["decision_count"][decision_key][action_key] = {}
    
    current = metrics_state["decision_count"][decision_key][action_key].get(risk_key, 0)
    metrics_state["decision_count"][decision_key][action_key][risk_key] = current + 1
    
    # Record latency
    metrics_state["decision_latency_ms"].append(latency_ms)

def record_outcome_metrics(executed: bool, action_type: str, committed: bool = True):
    """Record outcome metrics."""
    action_key = action_type.value if hasattr(action_type, 'value') else str(action_type)
    status = "committed" if committed else "quarantined"
    
    if action_key not in metrics_state["outcomes_count"][status]:
        metrics_state["outcomes_count"][status][action_key] = 0
    
    metrics_state["outcomes_count"][status][action_key] += 1

def format_prometheus_metrics() -> str:
    """Generate Prometheus text format metrics."""
    lines = [
        "# HELP cgf_decision_count Total decisions by type",
        "# TYPE cgf_decision_count counter",
    ]
    
    # decision_count{decision="ALLOW",action_type="tool_call",risk_tier="high"} 42
    for decision, actions in metrics_state["decision_count"].items():
        for action, risks in actions.items():
            for risk, count in risks.items():
                lines.append(f'cgf_decision_count{{decision="{decision}",action_type="{action}",risk_tier="{risk}"}} {count}')
    
    if not metrics_state["decision_count"]:
        lines.append('cgf_decision_count 0')
    
    lines.extend([
        "",
        "# HELP cgf_decision_latency_ms Decision latency histogram",
        "# TYPE cgf_decision_latency_ms histogram",
    ])
    
    # Histogram buckets
    for bucket in LATENCY_BUCKETS:
        count = sum(1 for l in metrics_state["decision_latency_ms"] if l <= bucket)
        lines.append(f'cgf_decision_latency_ms_bucket{{le="{bucket}"}} {count}')
    
    # +Inf bucket
    total = len(metrics_state["decision_latency_ms"])
    lines.append(f'cgf_decision_latency_ms_bucket{{le="+Inf"}} {total}')
    
    if total > 0:
        avg_latency = sum(metrics_state["decision_latency_ms"]) / total
        lines.append(f'cgf_decision_latency_ms_sum {sum(metrics_state["decision_latency_ms"])}')
    else:
        lines.append('cgf_decision_latency_ms_sum 0')
    lines.append(f'cgf_decision_latency_ms_count {total}')
    
    lines.extend([
        "",
        "# HELP cgf_outcomes_count_total Outcomes by status",
        "# TYPE cgf_outcomes_count_total counter",
    ])
    
    for status, actions in metrics_state["outcomes_count"].items():
        for action, count in actions.items():
            lines.append(f'cgf_outcomes_count_total{{success="true",action_type="{action}",committed="{status == "committed"}"}} {count}')
    
    if not (metrics_state["outcomes_count"]["committed"] or metrics_state["outcomes_count"]["quarantined"]):
        lines.append('cgf_outcomes_count_total 0')
    
    lines.extend([
        "",
        "# HELP cgf_evaluate_timeout_count Evaluation timeout count",
        "# TYPE cgf_evaluate_timeout_count counter",
        f'cgf_evaluate_timeout_count {metrics_state["evaluate_timeouts"]}',
        "",
        "# HELP cgf_proposals_received_total Total proposals received",
        "# TYPE cgf_proposals_received_total counter",
        f'cgf_proposals_received_total {metrics_state["proposals_received"]}',
        "",
        "# HELP cgf_proposals_evaluated_total Total proposals evaluated",
        "# TYPE cgf_proposals_evaluated_total counter",
        f'cgf_proposals_evaluated_total {metrics_state["proposals_evaluated"]}',
    ])
    
    return "\n".join(lines)

def get_policy_config() -> PolicyConfig:
    """Get policy config."""
    if HAS_V03 and POLICY_DATA:
        return PolicyConfig(**POLICY_DATA)
    # Default fallback
    return PolicyConfig()

def evaluate_policy(proposal: Any, context: Any, signals: Any) -> CGFDecision:
    """Host-agnostic policy evaluation using Policy Engine v1.0.
    
    Key invariant: No branching on host_type!
    Only uses: action_params, risk_tier, capacity_signals
    """
    decision_id = generate_id("dec")
    
    # P8: Use Policy Engine v1.0 if available
    if HAS_POLICY_ENGINE and POLICY_BUNDLE is not None:
        return _evaluate_with_policy_engine(proposal, context, signals, decision_id)
    
    # Fallback: use legacy policy v0.3 (for backward compatibility)
    return _evaluate_legacy(proposal, context, signals, decision_id)


def _evaluate_with_policy_engine(proposal: Any, context: Any, signals: Any, decision_id: str) -> CGFDecision:
    """Evaluate using Policy Engine v1.0."""
    # Build proposal dict
    proposal_dict = {
        "action_type": proposal.action_type.value if hasattr(proposal.action_type, 'value') else str(proposal.action_type),
        "tool_name": proposal.action_params.get("tool_name", ""),
        "size_bytes": proposal.action_params.get("size_bytes", 0),
        "sensitivity_hint": proposal.action_params.get("sensitivity_hint", "medium"),
        "risk_tier": proposal.risk_tier.value if hasattr(proposal.risk_tier, 'value') else str(proposal.risk_tier),
        "estimated_cost": {"tokens": proposal.action_params.get("estimated_tokens", 0)}
    }
    
    # Build context dict
    context_dict = {
        "recent_errors": context.recent_errors if hasattr(context, 'recent_errors') else 0,
        "adapter_id": context.adapter_id if hasattr(context, 'adapter_id') else None
    }
    
    # Build signals dict
    signals_dict = {
        "token_rate_60s": signals.token_rate if hasattr(signals, 'token_rate') else 0.0,
        "error_rate": signals.error_rate if hasattr(signals, 'error_rate') else 0.0,
        "avg_latency_ms": 0.0  # Placeholder
    }
    
    # Evaluate with policy engine
    result = evaluate(proposal_dict, context_dict, signals_dict, POLICY_BUNDLE)

    # CGF_STRICT: override the catch-all default-allow to AUDIT for unknown tools
    if CGF_STRICT and result.matched_rule_ids == ["default-allow"]:
        from cgf_policy.types import DecisionType as _PDT
        result = result.__class__(
            decision=_PDT.AUDIT,
            decision_confidence=result.decision_confidence,
            matched_rule_ids=result.matched_rule_ids,
            explanation_text=f"[STRICT MODE] {result.explanation_text}",
            constraint=result.constraint,
            audited=True,
        )

    # Convert policy DecisionType to schema DecisionType
    decision_type_map = {
        PolicyDecisionType.ALLOW: DecisionType.ALLOW,
        PolicyDecisionType.BLOCK: DecisionType.BLOCK,
        PolicyDecisionType.CONSTRAIN: DecisionType.CONSTRAIN,
        PolicyDecisionType.DEFER: DecisionType.DEFER,
        PolicyDecisionType.AUDIT: DecisionType.AUDIT,
    }
    
    # Build constraint if needed
    constraint = None
    if result.constraint:
        # Resolve template values in params
        params = result.constraint.params.copy()
        if "target_namespace" in params and params["target_namespace"] == "_quarantine_":
            params["target_namespace"] = f"_quarantine_{datetime.now().timestamp()}"
        if "source_namespace" in params and params["source_namespace"] == "default":
            params["source_namespace"] = proposal.action_params.get("namespace", "default")
        
        constraint = ConstraintConfig(
            type=result.constraint.type,
            params=params
        )
    
    # Map reason code from matched rule
    reason_code = result.matched_rule_ids[0].upper().replace("-", "_") if result.matched_rule_ids else "POLICY_MATCH"
    
    return CGFDecision(
        decision_id=decision_id,
        proposal_id=proposal.proposal_id,
        decision=decision_type_map[result.decision],
        confidence=result.decision_confidence,
        justification=result.explanation_text[:200] if len(result.explanation_text) > 200 else result.explanation_text,
        reason_code=reason_code,
        constraint=constraint,
        # P8: New fields for explainability
        policy_version=POLICY_BUNDLE.policy_version,
        matched_rule_ids=result.matched_rule_ids,
        explanation_text=result.explanation_text
    )


def _evaluate_legacy(proposal: Any, context: Any, signals: Any, decision_id: str) -> CGFDecision:
    """Legacy v0.3 policy evaluation (fallback)."""
    policy = get_policy_config()
    
    action_type = proposal.action_type
    action_params = proposal.action_params
    risk_tier = proposal.risk_tier
    
    tool_name = action_params.get("tool_name", "")
    size_bytes = action_params.get("size_bytes", 0)
    sensitivity_hint = action_params.get("sensitivity_hint", "medium")
    
    confidence_threshold = policy.confidence_thresholds.get(risk_tier, 0.6)
    default_confidence = 0.85
    
    # Rule 1: Denylist check
    if tool_name in policy.tool_denylist:
        return CGFDecision(
            decision_id=decision_id,
            proposal_id=proposal.proposal_id,
            decision=DecisionType.BLOCK,
            confidence=1.0,
            justification="Tool is in denylist",
            reason_code="DENYLISTED_TOOL"
        )
    
    # Rule 2: Large memory write threshold
    if action_type == ActionType.MEMORY_WRITE and size_bytes > policy.memory_size_threshold_bytes:
        return CGFDecision(
            decision_id=decision_id,
            proposal_id=proposal.proposal_id,
            decision=DecisionType.CONSTRAIN,
            confidence=0.9,
            justification="Large memory write requires quarantine",
            reason_code="LARGE_WRITE_THRESHOLD",
            constraint=ConstraintConfig(
                type="quarantine_namespace",
                params={
                    "target_namespace": f"_quarantine_{datetime.now().timestamp()}",
                    "source_namespace": action_params.get("namespace", "default")
                }
            )
        )
    
    # Rule 3: High sensitivity
    if sensitivity_hint == "high" and default_confidence < confidence_threshold:
        return CGFDecision(
            decision_id=decision_id,
            proposal_id=proposal.proposal_id,
            decision=DecisionType.CONSTRAIN,
            confidence=default_confidence,
            justification=f"High sensitivity with insufficient confidence (< {confidence_threshold})",
            reason_code="HIGH_SENSITIVITY_LOW_CONFIDENCE",
            constraint=ConstraintConfig(type="quarantine_namespace", params={})
        )
    
    # Rule 4: High risk with errors
    if risk_tier == RiskTier.HIGH and context.recent_errors > 3:
        return CGFDecision(
            decision_id=decision_id,
            proposal_id=proposal.proposal_id,
            decision=DecisionType.BLOCK,
            confidence=0.75,
            justification="High risk tier with recent errors",
            reason_code="HIGH_RISK_WITH_ERRORS"
        )
    
    # Default: ALLOW (or AUDIT in strict mode for unknown tools)
    if CGF_STRICT:
        return CGFDecision(
            decision_id=decision_id,
            proposal_id=proposal.proposal_id,
            decision=DecisionType.AUDIT,
            confidence=default_confidence,
            justification="[STRICT MODE] Unknown tool — audit required; set CGF_STRICT=0 to allow by default",
            reason_code="STRICT_DEFAULT_AUDIT"
        )
    return CGFDecision(
        decision_id=decision_id,
        proposal_id=proposal.proposal_id,
        decision=DecisionType.ALLOW,
        confidence=default_confidence,
        justification="Within policy thresholds",
        reason_code="DEFAULT_ALLOW"
    )

def log_event(event: HostEvent):
    """Log event to memory and optionally to file."""
    events.append(event)
    
    # Persist
    event_dir = Path("./cgf_data")
    event_dir.mkdir(exist_ok=True)
    event_file = event_dir / "events.jsonl"
    
    with open(event_file, "a") as f:
        f.write(json.dumps(event.model_dump() if hasattr(event, 'model_dump') else event.__dict__) + "\n")

def get_fail_mode_config(action_type: ActionType, risk_tier: RiskTier) -> Dict[str, Any]:
    """Get fail mode from policy config."""
    policy = get_policy_config()
    
    for fm in policy.fail_modes:
        if fm.action_type == action_type and fm.risk_tier == risk_tier:
            return {
                "fail_mode": fm.fail_mode.value,
                "timeout_ms": fm.timeout_ms,
                "rationale": fm.rationale
            }
    
    # Default
    return {
        "fail_mode": "fail_closed",
        "timeout_ms": 500,
        "rationale": "Default fail-closed for safety"
    }

# ============== ENDPOINTS ==============

@app.get("/")
def root():
    """Root endpoint."""
    return {
        "service": "CGF Server",
        "version": "0.3.0",
        "schema_version": SCHEMA_VERSION if HAS_V03 else "0.2.0",
        "policy_version": POLICY_DATA.get("policy_version", "default"),
        "endpoints": [
            "POST /v1/register",
            "POST /v1/evaluate",
            "POST /v1/outcomes/report",
            "GET /v1/proposals/{proposal_id}/replay"
        ],
        "capabilities": {
            "hosts_supported": ["openclaw", "langgraph", "custom"],
            "action_types": ["tool_call", "memory_write"],
            "event_types": 19
        }
    }

@app.post("/v1/register")
def register_adapter(reg: HostAdapterRegistration, _auth: None = Depends(require_auth)):
    """Register a host adapter."""
    adapter_id = generate_id("adp")
    
    # Validate schema version
    request_version = reg.schema_version if hasattr(reg, 'schema_version') else "0.2.0"
    if HAS_V03 and not is_compatible_version(request_version):
        raise HTTPException(
            status_code=400,
            detail=f"Incompatible schema version: {request_version}. Min: {MIN_COMPATIBLE_VERSION}"
        )
    
    adapters[adapter_id] = {
        "adapter_id": adapter_id,
        "adapter_type": reg.adapter_type,
        "host_config": reg.host_config.model_dump() if hasattr(reg.host_config, 'model_dump') else reg.host_config.__dict__,
        "registered_at": datetime.now().timestamp(),
        "schema_version": request_version
    }
    
    # Log event
    event = HostEvent(
        schema_version=SCHEMA_VERSION if HAS_V03 else "0.2.0",
        event_type=HostEventType.ADAPTER_REGISTERED,
        adapter_id=adapter_id,
        timestamp=datetime.now().timestamp(),
        payload={
            "adapter_type": reg.adapter_type,
            "host_type": reg.host_config.host_type if hasattr(reg.host_config, 'host_type') else "unknown",
            "version": reg.host_config.version if hasattr(reg.host_config, 'version') else "0.0.0"
        }
    )
    log_event(event)
    
    # P0 Fix: Include fail_mode_table from policy in registration response
    policy = get_policy_config()
    fail_mode_table = policy.fail_modes if HAS_V03 else [
        {"action_type": "tool_call", "risk_tier": "high", "fail_mode": "fail_closed", "timeout_ms": 500, "rationale": "Default: block on CGF down"},
        {"action_type": "tool_call", "risk_tier": "medium", "fail_mode": "defer", "timeout_ms": 500, "rationale": "Default: defer when CGF down"},
        {"action_type": "tool_call", "risk_tier": "low", "fail_mode": "fail_open", "timeout_ms": 500, "rationale": "Default: allow on CGF down"},
        {"action_type": "memory_write", "risk_tier": "high", "fail_mode": "fail_closed", "timeout_ms": 500, "rationale": "Default: block on CGF down"},
        {"action_type": "memory_write", "risk_tier": "medium", "fail_mode": "fail_closed", "timeout_ms": 500, "rationale": "Default: block on CGF down"},
        {"action_type": "memory_write", "risk_tier": "low", "fail_mode": "fail_open", "timeout_ms": 500, "rationale": "Default: allow on CGF down"},
    ]
    
    log_correlated(
        "info", "Adapter registered",
        adapter_id=adapter_id,
        event_type="ADAPTER_REGISTERED",
        adapter_type=reg.adapter_type,
        schema_version=request_version,
    )

    return HostAdapterRegistrationResponse(
        schema_version=SCHEMA_VERSION if HAS_V03 else "0.2.0",
        adapter_id=adapter_id,
        registered_at=datetime.now().timestamp(),
        expires_at=datetime.now().timestamp() + 3600,  # 1 hour
        fail_mode_table=fail_mode_table
    )

@app.post("/v1/evaluate")
def evaluate_proposal(req: HostEvaluationRequest, _auth: None = Depends(require_auth)):
    """Evaluate a proposal."""
    # Validate schema
    request_version = req.schema_version if hasattr(req, 'schema_version') else "0.2.0"
    if HAS_V03 and not is_compatible_version(request_version):
        raise HTTPException(
            status_code=400,
            detail=f"Incompatible schema version: {request_version}"
        )
    
    # Store proposal
    proposals[req.proposal.proposal_id] = {
        "proposal": req.proposal.model_dump() if hasattr(req.proposal, 'model_dump') else req.proposal.__dict__,
        "context": req.context.model_dump() if hasattr(req.context, 'model_dump') else req.context.__dict__,
        "signals": req.capacity_signals.model_dump() if hasattr(req.capacity_signals, 'model_dump') else req.capacity_signals.__dict__,
        "adapter_id": req.adapter_id
    }
    
    # Log proposal received
    log_event(HostEvent(
        schema_version=SCHEMA_VERSION if HAS_V03 else "0.2.0",
        event_type=HostEventType.PROPOSAL_RECEIVED,
        adapter_id=req.adapter_id or "unknown",
        timestamp=datetime.now().timestamp(),
        proposal_id=req.proposal.proposal_id,
        payload={
            "action_type": req.proposal.action_type.value if hasattr(req.proposal.action_type, 'value') else str(req.proposal.action_type),
            "action_params_hash": str(hash(str(req.proposal.action_params)))[:16],
            "risk_tier": req.proposal.risk_tier.value if hasattr(req.proposal.risk_tier, 'value') else str(req.proposal.risk_tier)
        }
    ))
    
    # Evaluate with policy
    decision = evaluate_policy(req.proposal, req.context, req.capacity_signals)
    decisions[decision.decision_id] = decision

    log_correlated(
        "info", "Decision made",
        adapter_id=req.adapter_id,
        proposal_id=req.proposal.proposal_id,
        decision_id=decision.decision_id,
        event_type="DECISION_MADE",
        decision=decision.decision.value,
        confidence=decision.confidence,
        reason_code=decision.reason_code,
    )
    
    # Log decision
    log_event(HostEvent(
        schema_version=SCHEMA_VERSION if HAS_V03 else "0.2.0",
        event_type=HostEventType.DECISION_MADE,
        adapter_id=req.adapter_id or "unknown",
        timestamp=datetime.now().timestamp(),
        proposal_id=req.proposal.proposal_id,
        decision_id=decision.decision_id,
        payload={
            "decision_type": decision.decision.value,
            "confidence": decision.confidence,
            "justification": decision.justification
        }
    ))
    
    # Log enforcement
    enforcement_events = {
        DecisionType.ALLOW: HostEventType.ACTION_ALLOWED,
        DecisionType.BLOCK: HostEventType.ACTION_BLOCKED,
        DecisionType.CONSTRAIN: HostEventType.ACTION_CONSTRAINED,
        DecisionType.DEFER: HostEventType.ACTION_DEFERRED,
        DecisionType.AUDIT: HostEventType.ACTION_AUDITED
    }
    
    if decision.decision in enforcement_events:
        payload = {
            "decision_id": decision.decision_id,
            "proposal_id": req.proposal.proposal_id
        }
        if decision.decision == DecisionType.BLOCK:
            payload["justification"] = decision.justification
            payload["reason_code"] = decision.reason_code
        
        log_event(HostEvent(
            schema_version=SCHEMA_VERSION if HAS_V03 else "0.2.0",
            event_type=enforcement_events[decision.decision],
            adapter_id=req.adapter_id or "unknown",
            timestamp=datetime.now().timestamp(),
            proposal_id=req.proposal.proposal_id,
            decision_id=decision.decision_id,
            payload=payload
        ))
    
    return HostEvaluationResponse(
        schema_version=SCHEMA_VERSION if HAS_V03 else "0.2.0",
        decision=decision
    )

@app.post("/v1/outcomes/report")
def report_outcome(outcome: HostOutcomeReport, _auth: None = Depends(require_auth)):
    """Report execution outcome."""
    outcomes[outcome.proposal_id] = outcome
    
    log_event(HostEvent(
        schema_version=SCHEMA_VERSION if HAS_V03 else "0.2.0",
        event_type=HostEventType.OUTCOME_LOGGED,
        adapter_id=outcome.adapter_id,
        timestamp=datetime.now().timestamp(),
        proposal_id=outcome.proposal_id,
        decision_id=outcome.decision_id,
        payload={
            "success": outcome.success,
            "duration_ms": outcome.duration_ms,
            "committed": outcome.committed,
            "quarantined": outcome.quarantined
        }
    ))
    
    log_correlated(
        "info", "Outcome received",
        adapter_id=outcome.adapter_id,
        proposal_id=outcome.proposal_id,
        decision_id=outcome.decision_id,
        event_type="OUTCOME_LOGGED",
        success=outcome.success,
        committed=outcome.committed,
    )

    return {"received": True}

@app.get("/v1/proposals/{proposal_id}/replay")
def get_replay(proposal_id: str):
    """Get replay pack for proposal."""
    if proposal_id not in proposals:
        raise HTTPException(status_code=404, detail="Proposal not found")
    
    proposal_data = proposals[proposal_id]
    decision = None
    outcome = None
    proposal_events = [e for e in events if e.proposal_id == proposal_id]
    
    # Find decision
    for d in decisions.values():
        if d.proposal_id == proposal_id:
            decision = d
            break
    
    # Find outcome
    if proposal_id in outcomes:
        outcome = outcomes[proposal_id]
    
    # Determine completeness
    if outcome:
        completeness = "full"
    elif decision:
        completeness = "decision-only"
    else:
        completeness = "partial"
    
    replay = ReplayPack(
        schema_version=SCHEMA_VERSION if HAS_V03 else "0.2.0",
        replay_id=generate_id("replay"),
        created_at=datetime.now().timestamp(),
        completeness=completeness,
        proposal=proposal_data["proposal"],
        decision=decision,
        outcome=outcome,
        events=proposal_events
    )
    
    return {
        "schema_version": SCHEMA_VERSION if HAS_V03 else "0.2.0",
        "replay": replay.model_dump() if hasattr(replay, 'model_dump') else replay.__dict__
    }

@app.get("/v1/health")
def health():
    """Health check."""
    return {
        "status": "healthy",
        "adapters": len(adapters),
        "proposals": len(proposals),
        "events": len(events),
        "schema_version": SCHEMA_VERSION if HAS_V03 else "0.2.0"
    }

# ============== MAIN ==============

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("CGF_PORT", "8080"))
    print("=" * 60)
    print(f"CGF Server v0.3")
    print(f"Schema: {SCHEMA_VERSION if HAS_V03 else '0.2.0'}")
    print(f"Policy: {POLICY_DATA.get('policy_version', 'default')}")
    print(f"Port: {port}")
    print("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=port)
