"""Truth plugin for regression claims (Claim 2, Claim 3).

These are empirical claims that test specific behaviors of the framework.
The truth values are based on expected outcomes from the regression tests.

Claim taxonomy:
- claim2_seed_perturbation: Seed perturbation should produce consistent results
- claim2_baseline: Baseline for Claim 2
- claim3_optionb_regime_check: Option B regime should match expected behavior
- claim3_baseline: Baseline for Claim 3
- claim3_explore: Exploration for Claim 3
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from ..schema import TruthLabel


SCHEMA_VERSION = "truth.v1"

# Regression claims - should pass if implementation is correct
REGRESSION_CLAIMS = {
    "claim2_seed_perturbation": {
        "description": "Seed perturbation produces consistent spectral dimension",
        "expected": {"should_pass": True},
    },
    "claim2_baseline": {
        "description": "Claim 2 baseline behavior",
        "expected": {"should_pass": True},
    },
    "claim3_optionb_regime_check": {
        "description": "Option B regime check produces expected filter behavior",
        "expected": {"should_pass": True},
    },
    "claim3_baseline": {
        "description": "Claim 3 baseline behavior",
        "expected": {"should_pass": True},
    },
    "claim3_explore": {
        "description": "Claim 3 exploration",
        "expected": {"should_pass": True},
    },
}


def generate_truth(claim_id: str, root: Path, substrate_id: Optional[str] = None, seed: Optional[int] = None) -> TruthLabel:
    """Generate truth label for regression claims.

    These are empirical truths based on expected regression test outcomes.
    """
    # Handle claim IDs with or without claim_ prefix
    normalized_id = claim_id
    if claim_id.startswith("claim_"):
        normalized_id = claim_id[6:]  # Remove "claim_" prefix

    # Check for known claim patterns
    if normalized_id in REGRESSION_CLAIMS:
        claim_info = REGRESSION_CLAIMS[normalized_id]
    elif claim_id in REGRESSION_CLAIMS:
        claim_info = REGRESSION_CLAIMS[claim_id]
    else:
        raise KeyError(f"Unknown regression claim: {claim_id}")

    return TruthLabel(
        schema_version=SCHEMA_VERSION,
        claim_id=claim_id,
        truth_type="boolean",
        expected=claim_info["expected"],
        tolerance={},
        source="regression_test",
        confidence=1.0,
        substrate_id=substrate_id,
        seed=seed,
    )