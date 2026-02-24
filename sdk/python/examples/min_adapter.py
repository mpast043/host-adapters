#!/usr/bin/env python3
"""
sdk/python/examples/min_adapter.py

Minimal HostAdapter example that runs against cgf_server_v03
without OpenClaw/LangGraph dependencies.

This demonstrates:
1. SDK-based adapter registration
2. Proposal evaluation
3. Event emission
4. Replay pack generation
5. Fail mode handling

Usage:
    # Terminal 1: Start CGF server
    python cgf_server_v03.py

    # Terminal 2: Run this example
    python sdk/python/examples/min_adapter.py
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add workspace to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

# Use SDK
from sdk.python.cgf_sdk import CGFClient, HostAdapter
from sdk.python.cgf_sdk.cgf_client import ClientConfig
from sdk.python.cgf_sdk.adapter_base import ReplayPack
from sdk.python.cgf_sdk.errors import FailModeError


class MinimalAdapter(HostAdapter):
    """
    Minimal adapter that demonstrates the HostAdapter contract.
    
    Simulates a simple workflow executor that:
    - Accepts tool_call actions
    - Allows any tool except "exec" and "bash" (denylist)
    - Emits events for replay
    """
    
    def __init__(self):
        config = ClientConfig(
            endpoint="http://127.0.0.1:8080",
            timeout_ms=500,
            schema_version="0.3.0"
        )
        
        super().__init__(
            adapter_type="minimal",
            host_config={
                "host_type": "minimal",
                "namespace": "default"
            },
            capabilities=["tool_call"],
            config=config,
            data_dir=Path("./min_adapter_data")
        )
        
        # Local denylist (mirrors CGF policy)
        self.denylist = {"exec", "bash", "rm -rf /"}
        self.execution_log: List[Dict] = []
    
    def build_proposal(
        self,
        action_type: str,
        action_params: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Build proposal from action."""
        import uuid
        import time
        
        return {
            "proposal_id": f"prop-{uuid.uuid4().hex[:8]}",
            "adapter_id": self.adapter_id,
            "action_type": action_type,
            "action_params": action_params,
            "proposed_at": time.time()
        }
    
    def apply_decision(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """Apply CGF decision locally."""
        decision_type = decision.get("decision")
        tool_name = decision.get("tool_name", "unknown")
        
        if decision_type == "ALLOW":
            # Execute the tool
            result = self._execute_tool(tool_name, decision.get("tool_args", {}))
            return {
                "executed": True,
                "success": result["success"],
                "error_message": result.get("error"),
                "side_effects": result.get("side_effects", [])
            }
        
        elif decision_type == "BLOCK":
            reason = decision.get("reason_code", "BLOCKED_BY_POLICY")
            return {
                "executed": False,
                "success": False,
                "error_message": f"Blocked: {reason}",
                "side_effects": []
            }
        
        elif decision_type == "CONSTRAIN":
            constraint = decision.get("constraint", {})
            result = self._execute_with_constraint(tool_name, constraint)
            return {
                "executed": result["executed"],
                "success": result["success"],
                "error_message": result.get("error"),
                "side_effects": result.get("side_effects", [])
            }
        
        else:
            # DEFER or unknown: defer execution
            return {
                "executed": False,
                "success": True,
                "error_message": f"Deferred: {decision_type}",
                "side_effects": []
            }
    
    def _execute_tool(self, tool_name: str, tool_args: Dict) -> Dict:
        """Simulate tool execution."""
        if tool_name in self.denylist:
            return {
                "success": False,
                "error": f"Tool {tool_name} is denylisted",
                "side_effects": []
            }
        
        self.execution_log.append({"tool": tool_name, "args": tool_args})
        return {
            "success": True,
            "result": f"Executed {tool_name} with {tool_args}",
            "side_effects": [f"{tool_name}_called"]
        }
    
    def _execute_with_constraint(self, tool_name: str, constraint: Dict) -> Dict:
        """Execute with constraints applied."""
        # Simulate constraint application
        constraint_type = constraint.get("type", "none")
        
        if constraint_type == "quarantine":
            # Would move to quarantine in real implementation
            return {
                "executed": True,
                "success": True,
                "result": f"Quarantined execution of {tool_name}",
                "side_effects": [f"{tool_name}_quarantined"]
            }
        
        return self._execute_tool(tool_name, {})


async def run_demo():
    """Run minimal adapter demo."""
    print("=" * 70)
    print("MINIMAL ADAPTER SDK DEMO v0.1")
    print("=" * 70)
    
    adapter = MinimalAdapter()
    
    # Check CGF health
    print("\n[1] Checking CGF health...")
    health = adapter.client.health()
    print(f"    Status: {health.get('status')}")
    
    if health.get('status') != 'healthy':
        print("    ⚠️  CGF not available. Continuing in fail-mode demonstration...")
    else:
        # Register
        print("\n[2] Registering adapter...")
        try:
            reg = await adapter.register()
            print(f"    Adapter ID: {reg.get('adapter_id')}")
        except Exception as e:
            print(f"    ⚠️  Registration failed: {e}")
            print("    Continuing without registration...")
    
    # Test scenarios
    scenarios = [
        ("ALLOW", "ls", {"path": "/"}, "low"),
        ("BLOCK", "file_write", {"path": "/etc/shadow"}, "high"),
        ("CONSTRAIN", "upload", {"size_bytes": 50_000_000}, "medium"),
    ]
    
    print("\n[3] Running scenarios...")
    
    for expected, tool, args, risk in scenarios:
        print(f"\n    → Scenario: {tool} (expected: {expected}, risk: {risk})")
        
        context = {
            "tool_name": tool,
            "tool_args": args,
            "risk_tier": risk,
            "side_effects": ["read"] if tool == "ls" else ["write"],
            "recent_errors": 0
        }
        
        try:
            outcome = await adapter.evaluate_and_execute(
                action_type="tool_call",
                action_params={"tool": tool, "args": args},
                context=context
            )
            
            print(f"      Executed: {outcome.get('executed')}")
            print(f"      Success: {outcome.get('success')}")
            if outcome.get('error_message'):
                print(f"      Message: {outcome.get('error_message')}")
                
        except FailModeError as e:
            print(f"      Fail mode applied: {e.fail_mode}")
            print(f"      Blocked per: {e.risk_tier}")
        except Exception as e:
            print(f"      ⚠️  Error: {type(e).__name__}: {e}")
    
    # Summary
    print("\n[4] Adapter summary...")
    summary = adapter.summary()
    print(f"    Proposals: {summary['proposals_count']}")
    print(f"    Events: {summary['events_count']}")
    print(f"    Data dir: {summary['data_dir']}")
    
    # Show replay packs
    print("\n[5] Replay packs...")
    for proposal_id, replay in adapter._proposals.items():
        print(f"    Proposal: {proposal_id}")
        if replay.decision:
            print(f"      Decision: {replay.decision.get('decision')}")
        print(f"      Events: {len(replay.events)}")
    
    # Export first replay pack
    if adapter._proposals:
        first_proposal = list(adapter._proposals.keys())[0]
        replay = adapter.build_replay_pack(first_proposal)
        print(f"\n[6] Sample replay pack (truncated):")
        replay_dict = replay.to_dict()
        print(f"    Schema: {replay_dict['schema_version']}")
        print(f"    Proposal ID: {replay_dict['proposal']['proposal_id']}")
        print(f"    Events: {[e['event_type'] for e in replay_dict['events'][:3]]}")
        
        # Save to file
        replay_file = Path("./min_adapter_data/replay_pack_sample.json")
        replay_file.parent.mkdir(parents=True, exist_ok=True)
        with open(replay_file, "w") as f:
            json.dump(replay_dict, f, indent=2)
        print(f"    Saved to: {replay_file}")
    
    # Check event log
    print("\n[7] Event counts by type...")
    event_types = {}
    for event in adapter._events:
        event_types[event.event_type] = event_types.get(event.event_type, 0) + 1
    for etype, count in sorted(event_types.items()):
        print(f"    {etype}: {count}")
    
    print("\n" + "=" * 70)
    print("Demo complete. Files written to: ./min_adapter_data/")
    print("=" * 70)
    
    return True


if __name__ == "__main__":
    result = asyncio.run(run_demo())
    sys.exit(0 if result else 1)
