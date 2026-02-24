# HostAdapter v1 Specification

**Version**: 1.0.0  
**Status**: Ratified  
**Schema Compatibility**: 0.2.x, 0.3.x

---

## Purpose

The HostAdapter protocol defines the integration contract between Host Systems (OpenClaw, LangGraph, etc.) and the Capacity Governance Framework (CGF). It provides:

1. **Observation**: Extract action proposals from host execution context
2. **Governance**: Submit to CGF for evaluation
3. **Enforcement**: Apply decisions (ALLOW, CONSTRAIN, AUDIT, DEFER, BLOCK)
4. **Audit**: Emit canonical events for replay and compliance

---

## Core Concepts

### Action Types

```
tool_call      → Execute a tool/function
message_send   → Send a message to a channel
memory_write   → Write to session/memory store
workflow_step  → Advance workflow state
```

### Decision Types

| Decision | Meaning | Host Action |
|----------|---------|-------------|
| **ALLOW** | Execute as normal | Proceed with action |
| **CONSTRAIN** | Execute with constraints | Apply constraints, then proceed |
| **AUDIT** | Log and execute | Log, then proceed |
| **DEFER** | Delay until human approval | Queue, wait for signal |
| **BLOCK** | Reject | Raise governance error |

### Fail Modes

When CGF is unreachable:

| Risk Tier | Fail Mode | Rationale |
|-----------|-----------|-----------|
| high | fail_closed | Side-effect operations blocked |
| medium | defer | Uncertain operations queued |
| low | fail_open | Read-only operations allowed |

---

## Protocol Flow

```
┌─────────────┐     1. Register      ┌──────────────┐
│   Host      │ ───────────────────▶ │  CGF Server  │
│  Adapter    │                      └──────────────┘
└─────────────┘                              │
       │                                     │
       │ 2. Proposal (action + context)      │
       │ ───────────────────────────────────▶│
       │                                     │
       │ 3. Decision (ALLOW/BLOCK/etc)       │
       │ ◀───────────────────────────────────│
       │                                     │
       │ 4. Enforcement                      │
       │ (execute or block)                  │
       │                                     │
       │ 5. Outcome Report                   │
       │ ───────────────────────────────────▶│
       │                                     │
       │ 6. Event Stream (async)             │
       │ ───────────────────────────────────▶│
```

---

## Endpoints

### POST /v1/adapters/register

Register a new host adapter.

**Request**:
```json
{
  "schema_version": "0.3.0",
  "adapter_type": "openclaw",
  "capabilities": ["tool_call", "memory_write"],
  "host_config": {
    "host_type": "openclaw",
    "namespace": "default"
  },
  "supported_actions": ["tool_call", "memory_write"]
}
```

**Response**:
```json
{
  "schema_version": "0.3.0",
  "adapter_id": "adapter-abc123",
  "registered_at": 1234567890.0,
  "status": "active",
  "cgf_endpoint": "http://127.0.0.1:8080/v1"
}
```

### POST /v1/evaluate

Submit an action proposal for evaluation.

**Request**:
```json
{
  "schema_version": "0.3.0",
  "proposal": {
    "proposal_id": "prop-xyz789",
    "adapter_id": "adapter-abc123",
    "action_type": "tool_call",
    "action_params": { ... },
    "proposed_at": 1234567890.0
  },
  "context": {
    "session_id": "session-123",
    "user_id": "user-456",
    "risk_tier": "high",
    "tool_name": "file_write",
    "side_effects": ["write", "modify"],
    "tool_args_hash": "sha256:abc...",
    "recent_errors": 0
  },
  "capacity_signals": {
    "C_geo_available": 0.8,
    "C_geo_total": 1.0,
    "C_int_available": 0.9,
    "C_gauge_available": 0.85,
    "C_ptr_available": 0.75,
    "C_obs_available": 0.95,
    "gate_fit_margin": 0.2,
    "gate_gluing_margin": 0.15
  },
  "observed_at": 1234567890.0
}
```

**Response**:
```json
{
  "schema_version": "0.3.0",
  "decision": {
    "decision_id": "dec-123",
    "proposal_id": "prop-xyz789",
    "decision": "BLOCK",
    "confidence": 1.0,
    "justification": "Tool is in denylist",
    "reason_code": "DENYLISTED_TOOL",
    "constraint": null,
    "decided_at": 1234567890.0
  },
  "evaluated_at": 1234567890.0
}
```

### POST /v1/outcomes/report

Report action execution outcome.

**Request**:
```json
{
  "schema_version": "0.3.0",
  "outcome_id": "outcome-123",
  "decision_id": "dec-123",
  "proposal_id": "prop-xyz789",
  "adapter_id": "adapter-abc123",
  "action_type": "tool_call",
  "executed": false,
  "success": false,
  "error_message": "Governance blocked: DENYLISTED_TOOL",
  "side_effects": [],
  "metadata": {},
  "observed_at": 1234567890.0
}
```

**Response**: `{"status": "committed"}` or `{"status": "quarantined"}`

---

## Event Stream

All adapters MUST emit canonical events. Events are:

1. **Ordered**: Within a proposal, events follow lifecycle
2. **Idempotent**: Same event_id = same event
3. **Complete**: Full lifecycle from proposal to outcome

### Event Types (19 Canonical)

#### Adapter Lifecycle
- `adapter_registered` — Adapter registered with CGF
- `adapter_disconnected` — Adapter heartbeat timeout

#### Proposal Lifecycle
- `proposal_received` — Proposal submitted
- `proposal_enacted` — Proposal executed successfully
- `proposal_expired` — Proposal TTL exceeded
- `proposal_revoked` — Proposal cancelled by host

#### Decision Lifecycle
- `decision_made` — CGF issued decision
- `decision_rejected` — Proposal malformed/rejected

#### Enforcement
- `action_allowed` — Action executed per ALLOW
- `action_blocked` — Action blocked per BLOCK
- `action_constrained` — Action executed with constraints
- `action_deferred` — Action queued per DEFER
- `action_audited` — Action executed per AUDIT

#### Error Handling
- `errors` — Generic error
- `constraint_failed` — Constraint application failed
- `cgf_unreachable` — CGF connection failed
- `evaluate_timeout` — Evaluation timed out

#### Outcome
- `outcome_logged` — Outcome committed to CGF
- `side_effect_reported` — Side effects confirmed

---

## Replay Pack

A `ReplayPack` contains complete governance history for a proposal:

```json
{
  "schema_version": "0.3.0",
  "proposal": { ... },
  "decision": { ... },
  "events": [
    { "event_type": "proposal_received", ... },
    { "event_type": "decision_made", ... },
    { "event_type": "action_blocked", ... },
    { "event_type": "outcome_logged", ... }
  ],
  "outcome": { ... }
}
```

---

## Implementation Requirements

### Required

1. Emit all 19 event types correctly
2. Validate schema_version on all payloads
3. Implement all 5 decision types
4. Apply fail modes per risk tier when CGF unreachable
5. Provide replay pack generation

### Recommended

1. Use typed client (SDK provided)
2. Implement circuit breaker for CGF failures
3. Log structured JSON locally for debugging
4. Support config-driven policy (not hardcoded)

### Prohibited

1. OpenClaw-specific branching in policy evaluation
2. Hard-coded risk tier inference (use config)
3. Silent failures on CGF errors (must log events)

---

## Compliance

To validate adapter compliance:

```bash
python contract_compliance_tests.py
python schema_lint.py --strict outputs/<run_dir>/
```

---

## Schema Versions

| Version | Min Compatible | Features |
|---------|----------------|----------|
| 0.3.0 | 0.2.0 | Multi-host, data-driven policy |
| 0.2.0 | 0.2.0 | Schema hardening, memory_write |

---

## Changelog

- **v1.0.0** (2026-02-24): Ratified, SDK reference implementation
