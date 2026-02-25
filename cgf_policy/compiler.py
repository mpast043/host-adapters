"""Policy bundle compiler and validator.

Loads policy bundles from JSON, validates structure,
computes bundle hashes, and rejects invalid configurations.
"""

import hashlib
import json
from pathlib import Path
from typing import Union, Tuple

from .types import PolicyBundle, Rule, Condition, Operator, FailModeConfig
from .fields import ALLOWED_FIELD_PATHS


def load_policy_bundle(path: Union[str, Path]) -> PolicyBundle:
    """Load and validate a policy bundle from JSON file.
    
    Args:
        path: Path to policy bundle JSON file
        
    Returns:
        Validated PolicyBundle with computed hash
        
    Raises:
        FileNotFoundError: If bundle file doesn't exist
        ValueError: If bundle validation fails
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Policy bundle not found: {path}")
    
    with open(path, "r") as f:
        data = json.load(f)
    
    # Pre-validate fields before Pydantic parsing
    _validate_bundle_fields(data)
    
    # Parse with Pydantic
    bundle = PolicyBundle.model_validate(data)
    
    # Compute and fill hash if empty
    computed_hash = compute_bundle_hash(bundle)
    
    if not bundle.bundle_hash:
        # Fill empty hash
        bundle.bundle_hash = computed_hash
        # Write back with computed hash
        _write_bundle_with_hash(path, bundle)
    elif bundle.bundle_hash != computed_hash:
        # Hash mismatch - invalid bundle
        raise ValueError(
            f"Bundle hash mismatch: expected {computed_hash}, got {bundle.bundle_hash}"
        )
    
    return bundle


def compute_bundle_hash(bundle: PolicyBundle) -> str:
    """Compute SHA256 hash of canonical JSON representation.
    
    Creates deterministic hash for audit and replay verification.
    Hash excludes the bundle_hash field itself to enable computation.
    
    Args:
        bundle: Policy bundle (bundle_hash is ignored)
        
    Returns:
        Hex-encoded SHA256 hash
    """
    # Create copy without bundle_hash for hashing
    bundle_dict = bundle.model_dump(exclude={"bundle_hash"})
    
    # Canonical JSON: sorted keys, no extra whitespace
    canonical = json.dumps(bundle_dict, sort_keys=True, separators=(",", ":"))
    
    # Compute hash
    hash_bytes = hashlib.sha256(canonical.encode("utf-8")).digest()
    return hash_bytes.hex()


def _validate_bundle_fields(data: dict) -> None:
    """Pre-validate bundle fields before Pydantic parsing.
    
    Checks:
    - policy_version is present
    - All rule conditions use allowed fields and operators
    - Rule ops are valid Operator values
    """
    if "policy_version" not in data:
        raise ValueError("policy_version is required")
    
    rules = data.get("rules", [])
    for rule in rules:
        conditions = rule.get("when", [])
        for cond in conditions:
            field = cond.get("field")
            op = cond.get("op")
            
            # Validate field
            if field and field not in ALLOWED_FIELD_PATHS:
                raise ValueError(f"Rule '{rule.get('id', 'unknown')}': field '{field}' not allowed")
            
            # Validate operator
            if op and op not in [o.value for o in Operator]:
                raise ValueError(f"Rule '{rule.get('id', 'unknown')}': operator '{op}' not allowed")


def _write_bundle_with_hash(path: Path, bundle: PolicyBundle) -> None:
    """Write bundle back to file with computed hash."""
    data = bundle.model_dump()
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def validate_policy_bundle(path: Union[str, Path]) -> Tuple[bool, str]:
    """Validate a policy bundle file without loading fully.
    
    Returns:
        (is_valid, message)
    """
    try:
        path = Path(path)
        
        if not path.exists():
            return False, f"File not found: {path}"
        
        with open(path, "r") as f:
            data = json.load(f)
        
        # Check required fields
        if "policy_version" not in data:
            return False, "Missing policy_version"
        
        rules = data.get("rules", [])
        if not rules:
            return False, "No rules defined"
        
        # Check unique rule ids
        ids = [r.get("id") for r in rules]
        if len(ids) != len(set(ids)):
            return False, "Duplicate rule ids"
        
        # Validate conditions
        for rule in rules:
            conditions = rule.get("when", [])
            for cond in conditions:
                field = cond.get("field")
                op = cond.get("op")
                
                if field and field not in ALLOWED_FIELD_PATHS:
                    return False, f"Unknown field: {field}"
                
                if op and op not in [o.value for o in Operator]:
                    return False, f"Unknown operator: {op}"
        
        return True, "Valid"
        
    except json.JSONDecodeError as e:
        return False, f"Invalid JSON: {e}"
    except Exception as e:
        return False, str(e)