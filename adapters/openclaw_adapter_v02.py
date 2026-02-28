"""
OpenClawAdapter v0.2 - HostAdapter implementation with memory_write governance.

Implements HostAdapter v1 SPEC with schema versioning and canonical EventType enum.
Changes v0.1 -> v0.2:
- Schema version 0.2.0 on all payloads
- EventType validation
- memory_write action_type support
- Session store interception
"""

import asyncio
import hashlib
import json
import os
import sys
import time
import uuid
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable

import aiohttp

# Resolve SDK path relative to this file so it works from any cwd
_SDK_PATH = Path(__file__).parent.parent / "sdk" / "python"
if str(_SDK_PATH) not in sys.path:
    sys.path.insert(0, str(_SDK_PATH))

from cgf_sdk.errors import (
    GovernanceError,
    ActionBlockedError,
    ActionConstrainedError,
    FailModeError,
    CGFConnectionError,
    CGFRegistryError,
)

from cgf_schemas_v02 import (
    SCHEMA_VERSION,
    ActionType,
    DecisionType,
    RiskTier,
    FailMode,
    HostEventType,
    HostConfig,
    HostProposal,
    HostContext,
    CapacitySignals,
    ToolCallParams,
    MemoryWriteParams,
    HostOutcomeReport,
    HostEvent,
    get_event_required_fields
)

# ============== CONFIGURATION ==============

class Config:
    CGF_ENDPOINT = os.environ.get("CGF_ENDPOINT", "http://127.0.0.1:8080")
    TIMEOUT_MS = int(os.environ.get("CGF_TIMEOUT_MS", "500"))
    DATA_DIR = Path(os.environ.get("CGF_DATA_DIR", "./openclaw_adapter_data"))
    SCHEMA_VERSION = SCHEMA_VERSION

Config.DATA_DIR.mkdir(exist_ok=True)

# ============== ADAPTER ==============

class OpenClawAdapter:
    """
    OpenClaw HostAdapter v0.2 with schema validation and memory_write governance.
    """
    
    def __init__(self, cgf_endpoint: str = Config.CGF_ENDPOINT,
                 adapter_type: str = "openclaw",
                 host_config: Optional[Dict] = None,
                 timeout_ms: int = Config.TIMEOUT_MS,
                 data_dir: Optional[Path] = None):
        self.cgf_endpoint = cgf_endpoint.rstrip('/')
        self.adapter_type = adapter_type
        self.host_config = host_config or {"host_type": "openclaw", "namespace": "default"}
        self.timeout_ms = timeout_ms
        self.data_dir = data_dir or Config.DATA_DIR
        self.data_dir.mkdir(exist_ok=True)
        
        self.adapter_id: Optional[str] = None
        self.event_count = 0
        self.proposal_count = 0
        self.session = None
        
        # P0 Fix: Cache fail_mode_table from CGF, starts empty
        self.fail_mode_table: Dict[str, FailMode] = {}
        
        # P0 Fix: Last-resort safe defaults (only used if no table cached)
        self._default_risk_tiers = {
            "high": FailMode.FAIL_CLOSED,
            "medium": FailMode.FAIL_CLOSED,  # Conservative: defer or block
            "low": FailMode.FAIL_OPEN
        }
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, *args):
        if self.session:
            await self.session.close()
    
    # ============== SCHEMA VALIDATION ==============
    
    def emit_event(self, event_type: HostEventType, payload: Dict[str, Any], 
                   proposal_id: Optional[str] = None, decision_id: Optional[str] = None) -> HostEvent:
        """
        Emit a canonical event with schema validation.
        Validates event_type against enum and required fields.
        """
        event = HostEvent(
            schema_version=SCHEMA_VERSION,
            event_type=event_type,
            adapter_id=self.adapter_id or "unregistered",
            timestamp=datetime.now().timestamp(),
            proposal_id=proposal_id,
            decision_id=decision_id,
            payload=payload
        )
        
        # Validate required fields
        required = get_event_required_fields(event_type)
        for field, field_type in required.items():
            if field not in payload:
                raise ValueError(f"Event {event_type.value} missing required field: {field}")
            value = payload[field]
            if not isinstance(value, field_type):
                raise TypeError(f"Event {event_type.value} field {field}: expected {field_type}, got {type(value)}")
        
        # Write to local JSONL
        event_path = self.data_dir / "events.jsonl"
        with open(event_path, "a") as f:
            f.write(json.dumps(event.model_dump()) + "\n")
        
        self.event_count += 1
        return event
    
    # ============== OBSERVATION ==============

    @staticmethod
    def _extract_target_path(tool_args: Dict[str, Any]) -> Optional[str]:
        """Best-effort extraction of a target path from tool arguments."""
        if not isinstance(tool_args, dict):
            return None
        for key in ("path", "file", "filepath", "filename", "target", "storePath", "output", "out_path"):
            value = tool_args.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None

    @staticmethod
    def _normalize_tool_name(tool_name: str, tool_args: Dict[str, Any]) -> str:
        """Map known autonomous workflow executions to audited aliases."""
        name = str(tool_name or "").strip()
        if name in {"exec", "shell", "bash", "python_exec", "subprocess"} and isinstance(tool_args, dict):
            command_parts: List[str] = []
            for key in ("cmd", "command", "shell_command", "script", "input"):
                value = tool_args.get(key)
                if isinstance(value, str):
                    command_parts.append(value.lower())
            command_blob = " ".join(command_parts)
            if any(
                marker in command_blob
                for marker in ("workflow-auto", "make workflow-auto", "run_workflow_auto.py", "tools/run_workflow_auto.py")
            ):
                return "workflow_auto_exec"
        return name
    
    def observe_proposal_tool(self, tool_name: str, tool_args: Dict, 
                              session_key: Optional[str], agent_id: Optional[str]) -> HostProposal:
        """Observe a tool call proposal."""
        self.proposal_count += 1
        normalized_tool_name = self._normalize_tool_name(tool_name, tool_args)
        target_path = self._extract_target_path(tool_args if isinstance(tool_args, dict) else {})
        
        # Canonicalize args for hash
        args_str = json.dumps(tool_args, sort_keys=True, separators=(',', ':'))
        args_hash = hashlib.sha256(args_str.encode()).hexdigest()[:32]
        
        # Determine risk tier
        side_effect_tools = {'file_write', 'fs_write', 'write', 'save', 'exec', 'shell', 'eval', 'workflow_auto_exec'}
        risk_tier = RiskTier.HIGH if normalized_tool_name in side_effect_tools else RiskTier.MEDIUM
        
        proposal_id = f"prop-{int(time.time() * 1000)}-{uuid.uuid4().hex[:6]}"
        
        params = ToolCallParams(
            tool_name=normalized_tool_name,
            tool_args_hash=args_hash,
            side_effects_hint=["write"] if normalized_tool_name in side_effect_tools else [],
            idempotent_hint=False,
            resource_hints=[]
        )
        action_params = params.model_dump()
        if target_path:
            action_params["target_path"] = target_path
        
        return HostProposal(
            proposal_id=proposal_id,
            timestamp=datetime.now().timestamp(),
            action_type=ActionType.TOOL_CALL,
            action_params=action_params,
            context_refs=[session_key or "unknown", agent_id or "unknown"],
            estimated_cost={"tokens": len(args_str) // 4, "latency_ms": 500},
            risk_tier=risk_tier,
            priority=0
        )
    
    def observe_memory_write(self, namespace: str, content: bytes,
                            session_key: Optional[str] = None,
                            agent_id: Optional[str] = None,
                            sensitivity_hint: str = "medium",
                            ttl: Optional[int] = None,
                            operation: str = "update") -> HostProposal:
        """
        Observe a memory write proposal (v0.2).
        
        Intercepts session store persistence at:
        pi-embedded-helpers-CMf7l1vP.js:6254 -> updateSessionStore()
        """
        self.proposal_count += 1
        
        # Hash content only - do not send full content
        content_hash = hashlib.sha256(content).hexdigest()
        size_bytes = len(content)
        
        # Sensitivity inference from content size
        if sensitivity_hint == "medium" and size_bytes > 1_000_000:  # 1MB
            sensitivity_hint = "high"
        
        proposal_id = f"prop-mem-{int(time.time() * 1000)}-{uuid.uuid4().hex[:6]}"
        
        params = MemoryWriteParams(
            namespace=namespace,
            size_bytes=size_bytes,
            ttl=ttl,
            sensitivity_hint=sensitivity_hint,
            content_hash=content_hash,
            context_refs=[session_key or "unknown", agent_id or "unknown"],
            operation=operation
        )
        
        # Risk tier based on sensitivity
        risk_tier = RiskTier.HIGH if sensitivity_hint == "high" else RiskTier.MEDIUM
        
        return HostProposal(
            proposal_id=proposal_id,
            timestamp=datetime.now().timestamp(),
            action_type=ActionType.MEMORY_WRITE,
            action_params=params.model_dump(),
            context_refs=[session_key or "unknown", agent_id or "unknown"],
            estimated_cost={"bytes": size_bytes, "latency_ms": 100},
            risk_tier=risk_tier,
            priority=0
        )
    
    def observe_context(self, session_key: Optional[str] = None,
                       agent_id: Optional[str] = None,
                       recent_errors: int = 0) -> HostContext:
        """Observe execution context."""
        return HostContext(
            agent_id=agent_id,
            session_id=session_key,
            turn_number=0,
            recent_errors=recent_errors,
            memory_growth_rate=0.0
        )
    
    def observe_signals(self) -> CapacitySignals:
        """Observe capacity signals."""
        return CapacitySignals(
            token_rate=0.0,
            tool_call_rate=0.0,
            error_rate=0.0,
            memory_growth=0.0
        )
    
    # ============== REGISTRATION ==============
    
    async def register(self) -> str:
        """Register adapter with CGF server."""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()
        
        payload = {
            "schema_version": SCHEMA_VERSION,
            "adapter_type": self.adapter_type,
            "host_config": self.host_config,
            "features": ["tool_call", "memory_write"],
            "risk_tiers": {k: v.value for k, v in self._default_risk_tiers.items()},
            "timestamp": datetime.now().timestamp()
        }
        
        try:
            async with self.session.post(
                f"{self.cgf_endpoint}/v1/register",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=5.0)
            ) as resp:
                data = await resp.json()
                if not isinstance(data, dict):
                    raise GovernanceError(
                        message=f"Invalid registration response: {type(data)}",
                        error_code="INVALID_RESPONSE"
                    )
                self.adapter_id = data.get("adapter_id") or f"local-{uuid.uuid4().hex[:12]}"
                
                # P0 Fix: Cache fail_mode_table from registration
                fmt = data.get("fail_mode_table")
                if fmt:
                    self._update_fail_mode_table(fmt)
                
                self.emit_event(
                    HostEventType.ADAPTER_REGISTERED,
                    {
                        "adapter_id": self.adapter_id,
                        "host_type": self.host_config.get("host_type", "unknown"),
                        "version": "0.2.0",
                        "fail_mode_table_cached": len(self.fail_mode_table) > 0
                    }
                )
                return self.adapter_id
        except (GovernanceError, CGFRegistryError):
            raise
        except Exception as e:
            self.adapter_id = f"local-{uuid.uuid4().hex[:12]}"
            raise CGFRegistryError(
                message=f"Registration failed: {e}",
                context={"original_error": type(e).__name__}
            )
    
    # ============== EVALUATION ==============
    
    async def evaluate(self, proposal: HostProposal, context: HostContext,
                      signals: CapacitySignals) -> Dict:
        """Send proposal to CGF for evaluation."""
        if not self.adapter_id:
            await self.register()
        
        # Ensure session is open
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()
        
        payload = {
            "schema_version": SCHEMA_VERSION,
            "adapter_id": self.adapter_id,
            "host_config": self.host_config,
            "proposal": proposal.model_dump(),
            "context": context.model_dump(),
            "capacity_signals": signals.model_dump()
        }
        
        self.emit_event(
            HostEventType.PROPOSAL_RECEIVED,
            {
                "proposal_id": proposal.proposal_id,
                "action_type": proposal.action_type.value,
                "action_params_hash": hashlib.sha256(
                    json.dumps(proposal.action_params, sort_keys=True).encode()
                ).hexdigest()[:16],
                "risk_tier": proposal.risk_tier.value
            },
            proposal_id=proposal.proposal_id
        )
        
        try:
            async with self.session.post(
                f"{self.cgf_endpoint}/v1/evaluate",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=self.timeout_ms / 1000)
            ) as resp:
                data = await resp.json()
                
                if not isinstance(data, dict):
                    raise GovernanceError(
                        message=f"Invalid response format: {type(data)}",
                        error_code="INVALID_RESPONSE",
                        proposal_id=proposal.proposal_id
                    )
                
                decision_raw = data.get("decision")
                if isinstance(decision_raw, dict):
                    decision = decision_raw
                    decision_type = DecisionType(decision.get("decision", "BLOCK"))
                elif isinstance(decision_raw, str):
                    # Server returned just the decision type
                    decision = {
                        "decision_id": "unknown",
                        "proposal_id": proposal.proposal_id,
                        "decision": decision_raw,
                        "confidence": 1.0,
                        "justification": "Received string decision"
                    }
                    decision_type = DecisionType(decision_raw)
                else:
                    decision = {}
                    decision_type = DecisionType.BLOCK
                
                self.emit_event(
                    HostEventType.DECISION_MADE,
                    {
                        "decision_id": decision.get("decision_id"),
                        "proposal_id": proposal.proposal_id,
                        "decision_type": decision_type.value,
                        "confidence": decision.get("confidence", 0.0),
                        "justification": decision.get("justification", "")
                    },
                    proposal_id=proposal.proposal_id,
                    decision_id=decision.get("decision_id")
                )
                
                return decision
        except asyncio.TimeoutError:
            raise CGFConnectionError(
                message="CGF evaluation timeout",
                proposal_id=proposal.proposal_id,
                context={"is_timeout": True}
            )
        except (GovernanceError,):
            raise
        except Exception as e:
            raise CGFConnectionError(
                message=f"CGF evaluation failed: {e}",
                proposal_id=proposal.proposal_id,
                context={"is_timeout": False, "original_error": type(e).__name__}
            )
    
    # ============== ENFORCEMENT ==============
    
    async def enforce_allow(self, proposal: HostProposal, decision_id: str) -> Dict:
        """Enforce ALLOW decision."""
        self.emit_event(
            HostEventType.ACTION_ALLOWED,
            {
                "decision_id": decision_id,
                "proposal_id": proposal.proposal_id,
                "executed_at": datetime.now().timestamp()
            },
            proposal_id=proposal.proposal_id,
            decision_id=decision_id
        )
        return {"allowed": True}
    
    async def enforce_block(self, proposal: HostProposal, decision_id: str,
                           justification: str, reason_code: Optional[str] = None):
        """Enforce BLOCK decision."""
        self.emit_event(
            HostEventType.ACTION_BLOCKED,
            {
                "decision_id": decision_id,
                "proposal_id": proposal.proposal_id,
                "justification": justification,
                "reason_code": reason_code or "BLOCKED_BY_POLICY"
            },
            proposal_id=proposal.proposal_id,
            decision_id=decision_id
        )
        raise ActionBlockedError(
            message=f"BLOCKED: {justification}",
            reason_code=reason_code or "BLOCKED_BY_POLICY",
            proposal_id=proposal.proposal_id,
            decision_id=decision_id
        )
    
    async def enforce_constrain_memory(self, proposal: HostProposal, decision_id: str,
                                       constraint: Dict) -> Dict:
        """
        Enforce CONSTRAIN decision for memory write.
        
        Returns modified namespace and constraint parameters.
        Constraint type: "quarantine_namespace" redirects write to _quarantine_ namespace.
        """
        constraint_type = constraint.get("type", "unknown")
        
        self.emit_event(
            HostEventType.ACTION_CONSTRAINED,
            {
                "decision_id": decision_id,
                "proposal_id": proposal.proposal_id,
                "constraint_type": constraint_type,
                "constraint_params": constraint.get("params", {})
            },
            proposal_id=proposal.proposal_id,
            decision_id=decision_id
        )
        
        if constraint_type == "quarantine_namespace":
            target_ns = constraint["params"].get("target_namespace", "_quarantine_")
            return {
                "allowed": True,
                "constrained": True,
                "target_namespace": target_ns,
                "original_namespace": constraint["params"].get("source_namespace"),
                "quarantined": True
            }
        
        # Unknown constraint type - fail closed
        self.emit_event(
            HostEventType.CONSTRAINT_FAILED,
            {
                "decision_id": decision_id,
                "proposal_id": proposal.proposal_id,
                "constraint_type": constraint_type,
                "error": f"Unknown constraint type: {constraint_type}",
                "fallback_decision": DecisionType.BLOCK.value
            },
            proposal_id=proposal.proposal_id,
            decision_id=decision_id
        )
        
        raise ActionConstrainedError(
            message=f"CONSTRAINT_FAILED: Unknown constraint type {constraint_type}",
            constraint_type=constraint_type,
            constraint_error="unsupported constraint type",
            proposal_id=proposal.proposal_id,
            decision_id=decision_id
        )
    
    # ============== FAIL MODE ==============
    
    def _update_fail_mode_table(self, fail_mode_table: List[Dict]) -> None:
        """Update cached fail_mode_table from CGF registration."""
        self.fail_mode_table = {}
        for entry in fail_mode_table:
            key = f"{entry['action_type']}:{entry['risk_tier']}"
            self.fail_mode_table[key] = FailMode(entry['fail_mode'])
    
    def _get_fail_mode(self, action_type: str, risk_tier: str) -> FailMode:
        """Get fail mode from cached table or safe defaults."""
        key = f"{action_type}:{risk_tier}"
        # Use cached table if available
        if key in self.fail_mode_table:
            return self.fail_mode_table[key]
        # Last-resort safe default
        return self._default_risk_tiers.get(risk_tier, FailMode.FAIL_CLOSED)
    
    def apply_fail_mode(self, proposal: HostProposal, error: Exception) -> DecisionType:
        """Apply fail mode based on risk tier and action type.
        
        Uses fail_mode_table from CGF /register response (P0 Fix).
        Falls back to safe defaults only if no table cached.
        """
        action_type = proposal.action_type.value if hasattr(proposal.action_type, 'value') else str(proposal.action_type)
        risk_tier = proposal.risk_tier.value if hasattr(proposal.risk_tier, 'value') else proposal.risk_tier
        fail_mode = self._get_fail_mode(action_type, risk_tier)
        
        error_type = "timeout" if isinstance(error, asyncio.TimeoutError) else "unreachable"
        event_type = HostEventType.EVALUATE_TIMEOUT if error_type == "timeout" else HostEventType.CGF_UNREACHABLE
        
        self.emit_event(
            event_type,
            {
                "proposal_id": proposal.proposal_id,
                **({"timeout_ms": self.timeout_ms} if error_type == "timeout" else {"endpoint": self.cgf_endpoint}),
                "error_type": error_type,
                "fail_mode_applied": fail_mode.value
            },
            proposal_id=proposal.proposal_id
        )
        
        # Map fail mode to decision
        if fail_mode == FailMode.FAIL_CLOSED:
            return DecisionType.BLOCK
        elif fail_mode == FailMode.FAIL_OPEN:
            return DecisionType.ALLOW
        else:  # DEFER
            return DecisionType.DEFER
    
    # ============== OUTCOME REPORTING ==============
    
    async def report_outcome(self, proposal: HostProposal, decision_id: str,
                            executed: bool, success: bool,
                            committed: Optional[bool] = None,
                            quarantined: Optional[bool] = None,
                            duration_ms: float = 0.0,
                            error: Optional[str] = None):
        """Report execution outcome to CGF."""
        outcome = HostOutcomeReport(
            schema_version=SCHEMA_VERSION,
            adapter_id=self.adapter_id,
            proposal_id=proposal.proposal_id,
            decision_id=decision_id,
            executed=executed,
            executed_at=datetime.now().timestamp(),
            duration_ms=duration_ms,
            success=success,
            committed=committed,
            quarantined=quarantined,
            errors=[error] if error else [],
            result_summary="Executed" if success else "Failed"
        )
        
        http_err: Optional[Exception] = None
        try:
            async with self.session.post(
                f"{self.cgf_endpoint}/v1/outcomes/report",
                json=outcome.model_dump(),
                timeout=aiohttp.ClientTimeout(total=5.0)
            ) as resp:
                await resp.json()
        except Exception as e:
            http_err = e
            # Fall through to local JSONL backup

        if http_err is not None:
            # Local JSONL fallback — safe atomic-ish write
            try:
                self.data_dir.mkdir(parents=True, exist_ok=True)
                outcome_path = self.data_dir / "outcomes_local.jsonl"
                record = json.dumps({**outcome.model_dump(), "report_error": str(http_err)}) + "\n"
                with open(outcome_path, "a") as f:
                    f.write(record)
            except Exception as fs_err:
                # Double failure: surface to stderr and raise — no silent drops
                import sys as _sys
                _sys.stderr.write(json.dumps({
                    "level": "ERROR",
                    "event": "outcome_loss",
                    "proposal_id": outcome.proposal_id,
                    "decision_id": decision_id,
                    "http_error": str(http_err),
                    "fs_error": str(fs_err),
                }) + "\n")
                _sys.stderr.flush()
                raise GovernanceError(
                    message=f"Outcome loss: HTTP report failed ({http_err}), JSONL fallback failed ({fs_err})",
                    error_code="OUTCOME_LOSS",
                    proposal_id=outcome.proposal_id,
                    decision_id=decision_id
                )
        
        self.emit_event(
            HostEventType.OUTCOME_LOGGED,
            {
                "proposal_id": proposal.proposal_id,
                "decision_id": decision_id,
                "success": success,
                "duration_ms": duration_ms
            },
            proposal_id=proposal.proposal_id,
            decision_id=decision_id
        )
    
    # ============== GOVERNANCE HOOKS ==============
    
    async def governance_hook_tool(self, tool_name: str, tool_args: Dict,
                                   session_key: Optional[str], agent_id: Optional[str]) -> Dict:
        """Full governance cycle for tool call."""
        proposal = self.observe_proposal_tool(tool_name, tool_args, session_key, agent_id)
        context = self.observe_context(session_key, agent_id)
        signals = self.observe_signals()
        
        decision_data = await self.evaluate(proposal, context, signals)
        decision = DecisionType(decision_data.get("decision", "BLOCK"))
        decision_id = decision_data.get("decision_id")
        
        if decision == DecisionType.ALLOW:
            return await self.enforce_allow(proposal, decision_id)
        elif decision == DecisionType.BLOCK:
            await self.enforce_block(
                proposal, decision_id,
                decision_data.get("justification", "Blocked by policy"),
                decision_data.get("reason_code")
            )
        elif decision == DecisionType.CONSTRAIN:
            # Tools v0.2: pass through with constraint
            return {
                "allowed": True,
                "constrained": True,
                "constraint": decision_data.get("constraint")
            }
        
        return {"allowed": True}  # Default allow for unknown
    
    async def governance_hook_memory(self, namespace: str, content: bytes,
                                     session_key: Optional[str] = None,
                                     agent_id: Optional[str] = None,
                                     sensitivity: str = "medium") -> Dict:
        """
        Full governance cycle for memory write.
        
        Returns:
            {
                "allowed": bool,
                "constrained": bool,
                "quarantined": bool,
                "target_namespace": str (quarantine path if constrained)
            }
        """
        proposal = self.observe_memory_write(
            namespace, content, session_key, agent_id, sensitivity
        )
        context = self.observe_context(session_key, agent_id)
        signals = self.observe_signals()
        
        start = time.time()
        
        try:
            decision_data = await self.evaluate(proposal, context, signals)
        except CGFConnectionError as conn_err:
            # Apply fail mode from cached table (P0 fix wired up)
            fail_decision = self.apply_fail_mode(proposal, conn_err)
            if fail_decision == DecisionType.BLOCK:
                raise ActionBlockedError(
                    message="Blocked: CGF unreachable, fail_closed applied",
                    reason_code="FAIL_MODE_BLOCKED",
                    proposal_id=proposal.proposal_id
                )
            elif fail_decision == DecisionType.DEFER:
                raise FailModeError(
                    message="CGF unreachable, action deferred",
                    fail_mode="defer",
                    proposal_id=proposal.proposal_id
                )
            # fail_open — proceed with ALLOW
            decision_data = {"decision": "ALLOW", "decision_id": "fail-mode"}
        
        decision = DecisionType(decision_data.get("decision", "BLOCK"))
        decision_id = decision_data.get("decision_id", "unknown")
        
        duration_ms = (time.time() - start) * 1000
        
        if decision == DecisionType.ALLOW:
            result = await self.enforce_allow(proposal, decision_id)
            await self.report_outcome(
                proposal, decision_id, executed=True, success=True,
                committed=True, duration_ms=duration_ms
            )
            return {**result, "constrained": False, "quarantined": False, "target_namespace": namespace}
        
        elif decision == DecisionType.CONSTRAIN:
            result = await self.enforce_constrain_memory(
                proposal, decision_id, decision_data.get("constraint", {})
            )
            await self.report_outcome(
                proposal, decision_id, executed=True, success=True,
                committed=True, quarantined=True, duration_ms=duration_ms
            )
            return result
        
        elif decision == DecisionType.BLOCK:
            await self.enforce_block(
                proposal, decision_id,
                decision_data.get("justification", "Blocked by policy"),
                decision_data.get("reason_code")
            )
        
        return {"allowed": False, "blocked": True}

    def get_recent_events(self, limit: int = 100) -> List[Dict]:
        """Retrieve recent events from local JSONL log."""
        event_path = self.data_dir / "events.jsonl"
        if not event_path.exists():
            return []
        
        events = []
        with open(event_path) as f:
            lines = f.readlines()
            for line in lines[-limit:]:
                try:
                    events.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    continue
        return events

# ============== MAIN ==============

if __name__ == "__main__":
    async def demo():
        adapter = OpenClawAdapter()
        
        print("OpenClawAdapter v0.2 Demo")
        print("-" * 40)
        
        # Demo memory write governance
        test_cases = [
            ("default", b"small data", "low"),
            ("sensitive_data", b"x" * 2_000_000, "high"),  # 2MB, should constrain
            ("large_file", b"y" * 15_000_000, "medium"),  # 15MB, should constrain
        ]
        
        for namespace, content, sensitivity in test_cases:
            print(f"\nMemory write: namespace={namespace}, size={len(content)}, sensitivity={sensitivity}")
            try:
                result = await adapter.governance_hook_memory(
                    namespace, content, sensitivity=sensitivity
                )
                print(f"  -> {result}")
            except (ActionBlockedError, GovernanceError) as e:
                print(f"  -> BLOCKED: {e}")
        
        print("\n" + "-" * 40)
        print(f"Total events: {adapter.event_count}")
    
    asyncio.run(demo())
