"""Truth plugin for observer structure claims (W02-W20).

These are theorem-level claims with boolean truths - they should always pass
if the framework is correct. The truth is analytical, derived from the
mathematical structure of the observer poset.

Claim taxonomy:
- W02: Poset infimum
- W03: Memory excision consistency
- W04: Self-reference consistency
- W06: Depth vector monotonicity
- W07: Cross-axis isolation
- W08: Class splitting monotonicity
- W09: Delta-t well defined
- W10: Observer non-influence
- W11: CPMT Annex T conformance
- W12: Observer triad mapping
- W13: Cobs decomposition compat
- W14: Ejection expands core
- W15: Pointer accuracy orthogonality
- W16: Time consistency monotone
- W17: Local-global fixed point compat
- W18: Compression governance T7
- W19: Meta limitation acknowledged
- W20: Non-negotiability self-application
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from ..schema import TruthLabel


SCHEMA_VERSION = "truth.v1"

# All observer claims should pass if framework is correct
OBSERVER_CLAIMS = {
    "W02": "Poset infimum exists and is well-defined",
    "W03": "Memory excision matched the general excision transform",
    "W04": "Self-reference is consistent under observer transformations",
    "W06": "Depth vector is monotonically increasing",
    "W07": "Cross-axis isolation is maintained",
    "W08": "Class splitting is monotonic with respect to capacity",
    "W09": "Delta-t is well-defined for all observer states",
    "W10": "Observer does not influence observed system",
    "W11": "CPMT Annex T constraints are satisfied",
    "W12": "Observer triad mapping is bijective",
    "W13": "C_obs decomposition is compatible with capacity structure",
    "W14": "Ejection expands core region",
    "W15": "Pointer accuracy is orthogonal to capacity",
    "W16": "Time consistency is monotone",
    "W17": "Local-global fixed point is compatible",
    "W18": "Compression governance T7 is satisfied",
    "W19": "Meta limitation is acknowledged",
    "W20": "Non-negotiability self-application is consistent",
}


def generate_truth(claim_id: str, root: Path, substrate_id: Optional[str] = None, seed: Optional[int] = None) -> TruthLabel:
    """Generate truth label for observer claims.

    These are analytical truths - the claim should always pass if the
    framework implementation is correct.
    """
    if claim_id not in OBSERVER_CLAIMS:
        raise KeyError(f"Unknown observer claim: {claim_id}")

    return TruthLabel(
        schema_version=SCHEMA_VERSION,
        claim_id=claim_id,
        truth_type="boolean",
        expected={"should_pass": True},
        tolerance={},
        source="analytical_theorem",
        confidence=1.0,
        substrate_id=substrate_id,
        seed=seed,
    )