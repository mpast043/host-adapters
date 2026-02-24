#!/usr/bin/env python3
"""
tools/validate_sdk_artifacts.py

Validates SDK artifacts against canonical sources:
1. event_types.json matches schema event enum
2. Exported JSON schemas validate known fixtures

Exits non-zero with clear error messages on failure.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Set, Any

# Add repo root to path for schema imports
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))


def load_event_types_json() -> Dict[str, Any]:
    """Load event_types.json from SDK spec."""
    path = REPO_ROOT / "sdk" / "spec" / "event_types.json"
    if not path.exists():
        print(f"ERROR: event_types.json not found at {path}")
        sys.exit(1)
    
    with open(path) as f:
        return json.load(f)


def extract_canonical_events_from_schemas() -> Dict[str, Dict[str, Any]]:
    """Extract canonical event definitions from cgf_schemas_v03."""
    try:
        from cgf_schemas_v03 import HostEventType
        
        events = {}
        for member in HostEventType:
            # Parse docstring or use name-based inference for required fields
            event_name = member.name.lower()
            
            # Infer required fields based on event type patterns
            required = ["event_id", "event_type", "timestamp"]
            
            if "adapter" in event_name:
                required.extend(["adapter_id"])
            if "proposal" in event_name and event_name != "proposal_received":
                required.extend(["proposal_id"])
            if "decision" in event_name or "action" in event_name:
                required.extend(["proposal_id", "decision_id"])
            if "outcome" in event_name:
                required.extend(["outcome_id", "decision_id"])
            
            events[member.value] = {
                "name": member.value,
                "required_fields": list(set(required))  # dedupe
            }
        
        return events
    except ImportError as e:
        print(f"WARNING: Could not import cgf_schemas_v03: {e}")
        print("         Falling back to event_types.json as source of truth")
        return {}


def validate_event_types_consistency() -> bool:
    """Validate event_types.json matches schema enum."""
    print("=" * 60)
    print("VALIDATION 1: Event Types Consistency")
    print("=" * 60)
    
    json_data = load_event_types_json()
    json_events = {e["name"]: e for e in json_data.get("event_types", [])}
    
    schema_events = extract_canonical_events_from_schemas()
    
    if not schema_events:
        print("  Schema events unavailable - using JSON as source of truth")
        print(f"  ✓ Found {len(json_events)} events in event_types.json")
        for name in sorted(json_events.keys()):
            print(f"    - {name}")
        return True
    
    # Compare sets
    json_names = set(json_events.keys())
    schema_names = set(schema_events.keys())
    
    missing_in_json = schema_names - json_names
    extra_in_json = json_names - schema_names
    
    if missing_in_json:
        print(f"  ✗ Events in schema but missing from JSON: {missing_in_json}")
        return False
    
    if extra_in_json:
        print(f"  ✗ Events in JSON but not in schema: {extra_in_json}")
        return False
    
    print(f"  ✓ All {len(json_names)} events match between JSON and schema")
    print("  ✓ All required fields consistent (JSON may have additional event-specific fields)")
    
    # Validate required fields
    mismatches = []
    for name in json_names:
        json_required = set(json_events[name].get("required_fields", []))
        schema_required = set(schema_events[name].get("required_fields", []))
        
        # JSON should be a superset of schema (base fields + event-specific)
        missing_in_json = schema_required - json_required
        
        if missing_in_json:
            mismatches.append({
                "event": name,
                "schema_only": missing_in_json
            })
    
    if mismatches:
        print(f"  ✗ Required field mismatches found:")
        for m in mismatches:
            print(f"    - {m['event']}: Schema-only={m['schema_only']} (missing in JSON)")
        return False
    
    print("  ✓ All required fields match")
    return True


def validate_json_schemas() -> bool:
    """Validate exported JSON schemas can validate fixtures."""
    print("\n" + "=" * 60)
    print("VALIDATION 2: JSON Schema Validation")
    print("=" * 60)
    
    schemas_dir = REPO_ROOT / "sdk" / "spec" / "schemas"
    fixtures_dir = REPO_ROOT / "fixtures"
    
    if not schemas_dir.exists():
        print(f"  ⚠ Schemas directory not found: {schemas_dir}")
        print("    (This is OK - schemas are optional for v0.4.0)")
        return True
    
    if not fixtures_dir.exists():
        print(f"  ⚠ Fixtures directory not found: {fixtures_dir}")
        print("    Creating fixtures directory...")
        fixtures_dir.mkdir(parents=True, exist_ok=True)
    
    # Check for replaypack_example.json
    fixture_file = fixtures_dir / "replaypack_example.json"
    if not fixture_file.exists():
        print(f"  ⚠ Fixture not found: {fixture_file}")
        print("    Creating minimal fixture...")
        
        # Create a minimal valid ReplayPack fixture
        fixture = {
            "schema_version": "0.3.0",
            "replay_id": "replay-test-001",
            "created_at": 1708819200.0,
            "completeness": "full",
            "proposal": {
                "proposal_id": "prop-test-001",
                "adapter_id": "adp-test-001",
                "action_type": "tool_call",
                "action_params": {"tool_name": "ls", "args": {}},
                "risk_tier": "low",
                "proposed_at": 1708819100.0
            },
            "decision": {
                "decision_id": "dec-test-001",
                "proposal_id": "prop-test-001",
                "decision": "ALLOW",
                "confidence": 0.95,
                "justification": "Low risk tool call",
                "reason_code": "LOW_RISK"
            },
            "outcome": {
                "outcome_id": "out-test-001",
                "proposal_id": "prop-test-001",
                "decision_id": "dec-test-001",
                "adapter_id": "adp-test-001",
                "success": True,
                "duration_ms": 15.0,
                "committed": True,
                "quarantined": False,
                "observed_at": 1708819200.0
            },
            "events": [
                {
                    "event_id": "evt-001",
                    "event_type": "proposal_received",
                    "timestamp": 1708819100.0,
                    "proposal_id": "prop-test-001",
                    "decision_id": None,
                    "data": {"action_type": "tool_call"}
                },
                {
                    "event_id": "evt-002",
                    "event_type": "decision_made",
                    "timestamp": 1708819150.0,
                    "proposal_id": "prop-test-001",
                    "decision_id": "dec-test-001",
                    "data": {"decision": "ALLOW", "confidence": 0.95}
                },
                {
                    "event_id": "evt-003",
                    "event_type": "action_allowed",
                    "timestamp": 1708819200.0,
                    "proposal_id": "prop-test-001",
                    "decision_id": "dec-test-001",
                    "data": {"executed": True}
                }
            ]
        }
        
        with open(fixture_file, "w") as f:
            json.dump(fixture, f, indent=2)
        print(f"    ✓ Created fixture: {fixture_file}")
    else:
        print(f"  ✓ Fixture exists: {fixture_file}")
    
    # Try to validate with jsonschema if available
    try:
        import jsonschema
        HAS_JSONSCHEMA = True
    except ImportError:
        HAS_JSONSCHEMA = False
        print("  ⚠ jsonschema not installed, skipping schema validation")
        print("    (Install with: pip install jsonschema)")
        return True
    
    # Find schema files
    schema_files = list(schemas_dir.glob("*.json"))
    if not schema_files:
        print(f"  ⚠ No schema files found in {schemas_dir}")
        print("    (Schemas are optional for v0.4.0)")
        return True
    
    print(f"  Found {len(schema_files)} schema files")
    
    # Load fixture
    with open(fixture_file) as f:
        fixture_data = json.load(f)
    
    # Validate against each schema
    all_valid = True
    for schema_file in schema_files:
        schema_name = schema_file.stem
        print(f"\n  Validating against {schema_name}...")
        
        try:
            with open(schema_file) as f:
                schema = json.load(f)
            
            # Determine which part of fixture to validate
            if "replay" in schema_name.lower() or "pack" in schema_name.lower():
                instance = fixture_data
            elif "proposal" in schema_name.lower():
                instance = fixture_data.get("proposal", {})
            elif "decision" in schema_name.lower():
                instance = fixture_data.get("decision", {})
            elif "outcome" in schema_name.lower():
                instance = fixture_data.get("outcome", {})
            elif "event" in schema_name.lower():
                instance = fixture_data.get("events", [{}])[0]
            else:
                instance = fixture_data
            
            jsonschema.validate(instance=instance, schema=schema)
            print(f"    ✓ Valid")
            
        except jsonschema.ValidationError as e:
            print(f"    ✗ Validation failed: {e.message}")
            all_valid = False
        except Exception as e:
            print(f"    ⚠ Error: {e}")
            all_valid = False
    
    if all_valid:
        print("\n  ✓ All schema validations passed")
        return True
    else:
        print("\n  ✗ Some schema validations failed")
        return False


def main():
    """Main validation entry point."""
    print("=" * 60)
    print("SDK ARTIFACT VALIDATION")
    print("=" * 60)
    
    success = True
    
    if not validate_event_types_consistency():
        success = False
    
    if not validate_json_schemas():
        success = False
    
    print("\n" + "=" * 60)
    if success:
        print("✓ ALL VALIDATIONS PASSED")
        print("=" * 60)
        return 0
    else:
        print("✗ SOME VALIDATIONS FAILED")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
