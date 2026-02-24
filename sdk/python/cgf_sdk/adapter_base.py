"""
sdk/python/cgf_sdk/adapter_base.py

Base HostAdapter class with event emission, replay pack building,
and fail mode application helpers.
"""

import json
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable

from .cgf_client import CGFClient, ClientConfig
from .errors import (
    GovernanceError,
    FailModeError,
    CGFConnectionError,
    ErrorConverter
)


class DecisionType(str, Enum):
    """Canonical decision types."""
    ALLOW = "ALLOW"
    CONSTRAIN = "CONSTRAIN"
    AUDIT = "AUDIT"
    DEFER = "DEFER"
    BLOCK = "BLOCK"


class RiskTier(str, Enum):
    """Canonical risk tiers."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class FailMode(str, Enum):
    """Fail mode options."""
    FAIL_CLOSED = "fail_closed"
    FAIL_OPEN = "fail_open"
    DEFER = "defer"


@dataclass
class Event:
    """Canonical event structure."""
    event_id: str
    event_type: str
    timestamp: float
    proposal_id: Optional[str] = None
    decision_id: Optional[str] = None
    data: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "timestamp": self.timestamp,
            "proposal_id": self.proposal_id,
            "decision_id": self.decision_id,
            "data": self.data
        }


@dataclass
class ReplayPack:
    """Complete governance history for a proposal."""
    schema_version: str
    proposal: Dict[str, Any]
    decision: Optional[Dict[str, Any]] = None
    events: List[Event] = field(default_factory=list)
    outcome: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "proposal": self.proposal,
            "decision": self.decision,
            "events": [e.to_dict() for e in self.events],
            "outcome": self.outcome
        }


class HostAdapter(ABC):
    """
    Base class for HostAdapter implementations.
    
    Subclasses override:
    - build_proposal(action, context)
    - apply_decision(decision) -> outcome
    - execute_action(action, decision) -> result
    """
    
    def __init__(
        self,
        adapter_type: str,
        host_config: Dict[str, Any],
        capabilities: List[str],
        config: Optional[ClientConfig] = None,
        data_dir: Optional[Path] = None
    ):
        self.adapter_type = adapter_type
        self.host_config = host_config
        self.capabilities = capabilities
        self.config = config or ClientConfig()
        self.data_dir = data_dir or Path(f"./{adapter_type}_cgf_data")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.client = CGFClient(self.config)
        self.adapter_id: Optional[str] = None
        self._events: List[Event] = []
        self._proposals: Dict[str, ReplayPack] = {}
        self._fail_mode_policy: Dict[RiskTier, FailMode] = {
            RiskTier.HIGH: FailMode.FAIL_CLOSED,
            RiskTier.MEDIUM: FailMode.DEFER,
            RiskTier.LOW: FailMode.FAIL_OPEN
        }
    
    # ============ Abstract Methods ============
    
    @abstractmethod
    def build_proposal(
        self,
        action_type: str,
        action_params: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Build a proposal from host action.
        
        Returns proposal dict with:
        - proposal_id
        - adapter_id
        - action_type
        - action_params
        - proposed_at
        """
        pass
    
    @abstractmethod
    def apply_decision(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply a CGF decision.
        
        Returns outcome with:
        - executed: bool
        - success: bool
        - error_message: Optional[str]
        - side_effects: List[str]
        """
        pass
    
    # ============ Registration ============
    
    async def register(self) -> Dict[str, Any]:
        """Register with CGF."""
        result = await self.client.register_async(
            adapter_type=self.adapter_type,
            capabilities=self.capabilities,
            host_config=self.host_config
        )
        self.adapter_id = result.get("adapter_id")
        
        self.emit_event(
            "adapter_registered",
            data={
                "adapter_id": self.adapter_id,
                "host_type": self.host_config.get("host_type"),
                "capabilities": self.capabilities
            }
        )
        
        return result
    
    # ============ Governance ============
    
    async def evaluate_and_execute(
        self,
        action_type: str,
        action_params: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Full governance lifecycle: evaluate, decide, execute, report.
        
        Returns the final outcome dict.
        """
        proposal = self.build_proposal(action_type, action_params, context)
        proposal_id = proposal["proposal_id"]
        
        # Initialize replay pack
        replay = ReplayPack(
            schema_version=self.config.schema_version,
            proposal=proposal
        )
        self._proposals[proposal_id] = replay
        
        # Emit proposal received
        self.emit_event(
            "proposal_received",
            proposal_id=proposal_id,
            data={"action_type": action_type}
        )
        
        try:
            # Get decision from CGF
            result = await self.client.evaluate_async(
                proposal=proposal,
                context=context or {},
                capacity_signals=self._get_capacity_signals()
            )
            
            decision = result.get("decision", {})
            replay.decision = decision
            decision_id = decision.get("decision_id")
            
            # Emit decision made
            self.emit_event(
                "decision_made",
                proposal_id=proposal_id,
                decision_id=decision_id,
                data={
                    "decision": decision.get("decision"),
                    "confidence": decision.get("confidence")
                }
            )
            
            # Apply decision
            outcome = self.apply_decision(decision)
            replay.outcome = outcome
            
            # Emit enforcement event
            decision_type = decision.get("decision")
            event_type = self._decision_to_event(decision_type)
            self.emit_event(
                event_type,
                proposal_id=proposal_id,
                decision_id=decision_id,
                data={"executed": outcome.get("executed")}
            )
            
            # Report outcome
            outcome["schema_version"] = self.config.schema_version
            outcome["outcome_id"] = str(uuid.uuid4())
            outcome["proposal_id"] = proposal_id
            outcome["decision_id"] = decision_id
            outcome["adapter_id"] = self.adapter_id
            outcome["observed_at"] = time.time()
            
            await self.client.report_outcome_async(outcome)
            
            self.emit_event(
                "outcome_logged",
                proposal_id=proposal_id,
                decision_id=decision_id,
                data={"outcome_id": outcome.get("outcome_id")}
            )
            
            return outcome
            
        except CGFConnectionError as e:
            # Apply fail mode
            risk_tier = RiskTier(context.get("risk_tier", "high"))
            return await self._apply_fail_mode(e, proposal_id, risk_tier)
        
    def _decision_to_event(self, decision: str) -> str:
        """Map decision to enforcement event type."""
        mapping = {
            "ALLOW": "action_allowed",
            "BLOCK": "action_blocked",
            "CONSTRAIN": "action_constrained",
            "AUDIT": "action_audited",
            "DEFER": "action_deferred"
        }
        return mapping.get(decision, "action_blocked")
    
    async def _apply_fail_mode(
        self,
        error: CGFConnectionError,
        proposal_id: str,
        risk_tier: RiskTier
    ) -> Dict[str, Any]:
        """Apply configured fail mode when CGF unreachable."""
        fail_mode = self._fail_mode_policy.get(risk_tier, FailMode.FAIL_CLOSED)
        
        outcome = {
            "executed": fail_mode == FailMode.FAIL_OPEN,
            "success": fail_mode == FailMode.FAIL_OPEN,
            "error_message": f"CGF unreachable; applied {fail_mode.value}",
            "side_effects": []
        }
        
        self.emit_event(
            "cgf_unreachable",
            proposal_id=proposal_id,
            data={
                "fail_mode": fail_mode.value,
                "risk_tier": risk_tier.value,
                "error": str(error)
            }
        )
        
        if fail_mode == FailMode.FAIL_CLOSED:
            raise FailModeError(
                message="CGF unreachable; blocked per fail mode",
                fail_mode=fail_mode.value,
                applied_decision="BLOCK",
                risk_tier=risk_tier.value,
                proposal_id=proposal_id
            )
        
        # Emit appropriate event
        if fail_mode == FailMode.FAIL_OPEN:
            self.emit_event("action_allowed", proposal_id=proposal_id, data={"via_fail_mode": True})
        elif fail_mode == FailMode.DEFER:
            self.emit_event("action_deferred", proposal_id=proposal_id, data={"via_fail_mode": True})
        
        return outcome
    
    # ============ Event Helpers ============
    
    def emit_event(
        self,
        event_type: str,
        proposal_id: Optional[str] = None,
        decision_id: Optional[str] = None,
        data: Optional[Dict[str, Any]] = None
    ) -> Event:
        """
        Emit a canonical event.
        
        Events are:
        - Stored in memory for replay packs
        - Written to JSONL for persistence
        """
        event = Event(
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            timestamp=time.time(),
            proposal_id=proposal_id,
            decision_id=decision_id,
            data=data or {}
        )
        
        self._events.append(event)
        
        # Write to file
        self._write_event(event)
        
        return event
    
    def _write_event(self, event: Event):
        """Persist event to JSONL."""
        event_file = self.data_dir / "events.jsonl"
        with open(event_file, "a") as f:
            f.write(json.dumps(event.to_dict()) + "\n")
    
    def build_replay_pack(self, proposal_id: str) -> Optional[ReplayPack]:
        """
        Build a replay pack for a proposal.
        
        Returns complete governance history or None if proposal not found.
        """
        return self._proposals.get(proposal_id)
    
    def get_events(
        self,
        proposal_id: Optional[str] = None,
        event_type: Optional[str] = None
    ) -> List[Event]:
        """Filter events by proposal and/or type."""
        events = self._events
        if proposal_id:
            events = [e for e in events if e.proposal_id == proposal_id]
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        return events
    
    # ============ Utility ============
    
    def _get_capacity_signals(self) -> Dict[str, float]:
        """Get current capacity signals (override for custom)."""
        return {
            "C_geo_available": 0.95,
            "C_int_available": 0.95,
            "C_gauge_available": 0.95,
            "C_ptr_available": 0.95,
            "C_obs_available": 0.95,
            "gate_fit_margin": 0.2,
            "gate_gluing_margin": 0.15
        }
    
    def set_fail_mode(self, risk_tier: RiskTier, fail_mode: FailMode):
        """Configure fail mode for risk tier."""
        self._fail_mode_policy[risk_tier] = fail_mode
    
    def summary(self) -> Dict[str, Any]:
        """Get adapter summary."""
        return {
            "adapter_id": self.adapter_id,
            "adapter_type": self.adapter_type,
            "proposals_count": len(self._proposals),
            "events_count": len(self._events),
            "data_dir": str(self.data_dir)
        }
