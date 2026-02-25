#!/usr/bin/env python3
"""Replay verification tool for P8 Policy Engine.

Validates deterministic replay by re-evaluating a replay pack's
proposal against the current policy bundle and asserting decision matches.

Usage:
    python3 tools/replay_verify.py --replaypack path/to/replay.json --policy policy/policy_bundle_v1.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from cgf_policy import load_policy_bundle, evaluate, DecisionType


def load_replay_pack(path: Path) -> Dict[str, Any]:
    """Load replay pack from JSON."""
    with open(path) as f:
        return json.load(f)


def extract_proposal_context_signals(replay: Dict[str, Any]) -> Tuple[Dict, Dict, Dict]:
    """Extract proposal, context, and signals from replay pack."""
    # Try direct fields first (v0.3+ format)
    if "proposal" in replay:
        proposal = replay["proposal"]
        context = replay.get("context", {})
        signals = replay.get("signals", {})
        return proposal, context, signals
    
    # Fallback: build from events
    proposal_events = [e for e in replay.get("events", []) if e.get("event_type") == "PROPOSAL_RECEIVED"]
    decision_events = [e for e in replay.get("events", []) if e.get("event_type") == "DECISION_MADE"]
    
    if proposal_events and decision_events:
        proposal_event = proposal_events[0]
        decision_event = decision_events[0]
        
        # Reconstruct proposal from event payload
        proposal = {
            "action_type": proposal_event.get("payload", {}).get("action_type", "tool_call"),
            "tool_name": "",  # May not be available in legacy events
            "size_bytes": 0,
            "sensitivity_hint": "medium",
            "risk_tier": proposal_event.get("payload", {}).get("risk_tier", "medium"),
        }
        
        context = {}  # May not be available in legacy
        signals = {}  # May not be available in legacy
        
        return proposal, context, signals
    
    raise ValueError("Replay pack missing required proposal/context/signals data")


def extract_recorded_decision(replay: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Extract the recorded decision from replay pack."""
    # Direct decision field (v0.3+)
    if "decision" in replay:
        return replay["decision"]
    
    # Look in events
    for event in replay.get("events", []):
        if event.get("event_type") == "DECISION_MADE":
            payload = event.get("payload", {})
            return {
                "decision": payload.get("decision_type", "ALLOW"),
                "confidence": payload.get("confidence", 1.0),
                "justification": payload.get("justification", "")
            }
    
    return None


def format_diff(key: str, recorded: Any, recomputed: Any) -> str:
    """Format a diff line."""
    return f"  {key}:\n    recorded: {recorded}\n    recomputed: {recomputed}"


def verify_replay(replay_path: Path, policy_path: Path, verbose: bool = False) -> Tuple[bool, str]:
    """Verify replay pack against policy bundle.
    
    Returns:
        (success, message)
    """
    try:
        # Load policy bundle
        policy_bundle = load_policy_bundle(policy_path)
        
        # Load replay pack
        replay = load_replay_pack(replay_path)
        
        # Extract recorded decision
        recorded = extract_recorded_decision(replay)
        if recorded is None:
            return False, "No decision found in replay pack"
        
        # Extract proposal/context/signals
        proposal, context, signals = extract_proposal_context_signals(replay)
        
        # Re-evaluate with current policy
        result = evaluate(proposal, context, signals, policy_bundle)
        
        # Build diff
        diffs = []
        
        # Compare decision type
        recorded_decision = recorded.get("decision", "ALLOW")
        recomputed_decision = result.decision.value
        if recorded_decision != recomputed_decision:
            diffs.append(format_diff("decision", recorded_decision, recomputed_decision))
        
        # Compare confidence (allow small tolerance)
        recorded_conf = recorded.get("confidence", 1.0)
        recomputed_conf = result.decision_confidence
        if abs(recorded_conf - recomputed_conf) > 0.01:
            diffs.append(format_diff("confidence", recorded_conf, recomputed_conf))
        
        # Build output
        output_lines = []
        output_lines.append(f"Replay: {replay_path}")
        output_lines.append(f"Policy: {policy_path} (v{policy_bundle.policy_version})")
        output_lines.append(f"Bundle hash: {policy_bundle.bundle_hash[:32]}...")
        output_lines.append("")
        output_lines.append(f"Recorded decision: {recorded_decision}")
        output_lines.append(f"Recomputed decision: {recomputed_decision}")
        output_lines.append(f"Matched rules: {result.matched_rule_ids}")
        
        if verbose:
            output_lines.append("")
            output_lines.append("Explanation:")
            output_lines.append(f"  {result.explanation_text}")
        
        if diffs:
            output_lines.append("")
            output_lines.append("MISMATCHES DETECTED:")
            output_lines.extend(diffs)
            output_lines.append("")
            output_lines.append("VERIFICATION: FAILED")
            return False, "\n".join(output_lines)
        else:
            output_lines.append("")
            output_lines.append("VERIFICATION: PASSED")
            return True, "\n".join(output_lines)
            
    except FileNotFoundError as e:
        return False, f"File not found: {e}"
    except Exception as e:
        return False, f"Verification error: {e}"


def main():
    parser = argparse.ArgumentParser(
        description="Verify replay pack decisions against policy bundle"
    )
    parser.add_argument(
        "--replaypack",
        type=Path,
        required=True,
        help="Path to replay pack JSON file"
    )
    parser.add_argument(
        "--policy",
        type=Path,
        default=Path(__file__).parent.parent / "policy" / "policy_bundle_v1.json",
        help="Path to policy bundle (default: policy/policy_bundle_v1.json)"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show detailed explanation"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output JSON format"
    )
    
    args = parser.parse_args()
    
    success, message = verify_replay(args.replaypack, args.policy, args.verbose)
    
    if args.json:
        output = {
            "success": success,
            "message": message,
            "replaypack": str(args.replaypack),
            "policy": str(args.policy)
        }
        print(json.dumps(output, indent=2))
    else:
        print(message)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()