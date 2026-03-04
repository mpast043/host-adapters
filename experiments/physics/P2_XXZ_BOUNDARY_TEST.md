# P2 XXZ Boundary Test — Scope Validation

**Date**: 2026-03-04
**Purpose**: Validate that framework correctly identifies gapped vs critical systems

---

## Physical Background

The XXZ model provides a clean transition from gapped to critical:

| Δ Value | Phase | Central Charge | Scope |
|---------|-------|----------------|-------|
| Δ = 0 | Free boson | c = 1 | OUT OF SCOPE (critical) |
| Δ = 0.5 | Gapless XY | c = 1 | OUT OF SCOPE (critical) |
| Δ = 1 | Heisenberg point | c = 1 | OUT OF SCOPE (critical) |
| Δ = 1.1 | Near-critical | c ≈ 1 | MARGINAL |
| Δ = 1.5 | Gapped Ising-like | c = 0 | IN SCOPE (gapped) |
| Δ = 2.0 | Gapped | c = 0 | IN SCOPE (gapped) |

**Hamiltonian:**
```
H = J Σ_i [ (X_i X_{i+1} + Y_i Y_{i+1})/2 + Δ Z_i Z_{i+1} ]
```

**Key insight:** As Δ crosses 1 from above, the system transitions from gapped to gapless.

---

## Test Design

### Predictions

If framework scope is correct:
- **Δ > 1**: P2 should ACCEPT (saturation model preferred)
- **Δ ≤ 1**: P2 should REJECT (log-linear model preferred)
- **Δ ≈ 1**: Marginal (boundary effects)

### Test Points

| Δ | Expected Behavior | P2 Verdict |
|---|-------------------|-------------|
| 0.0 | Critical (free boson) | REJECT |
| 0.5 | Critical (XY phase) | REJECT |
| 1.0 | Critical (Heisenberg) | REJECT |
| 1.1 | Near-critical | MARGINAL |
| 1.5 | Gapped | ACCEPT |
| 2.0 | Gapped | ACCEPT |

### Parameters

- L = 8, 12 (finite-size scaling)
- Boundary: cyclic
- χ = 4, 8, 16 (bond dimension sweep)

---

## Implementation

```python
"""
P2 XXZ Boundary Test — Validate scope transition at Δ=1

Tests whether the capacity framework correctly identifies:
- Gapped systems (Δ > 1) as IN SCOPE (saturation expected)
- Critical systems (Δ ≤ 1) as OUT OF SCOPE (log growth expected)
"""

import numpy as np
from pathlib import Path
import json

# Hamiltonian construction
def xxz_hamiltonian(L, delta, J=1.0):
    """
    Construct XXZ Hamiltonian.
    
    H = J Σ_i [ (X_i X_{i+1} + Y_i Y_{i+1})/2 + Δ Z_i Z_{i+1} ]
    
    Parameters:
        L: System size
        delta: Anisotropy parameter (Δ=1 is Heisenberg)
        J: Coupling strength
    """
    # Pauli matrices
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    
    H = np.zeros((2**L, 2**L), dtype=complex)
    
    for i in range(L):
        j = (i + 1) % L  # Cyclic boundary
        
        # XX + YY term
        for pauli in [X, Y]:
            term = 1
            for k in range(L):
                if k == i or k == j:
                    term = np.kron(term, pauli)
                else:
                    term = np.kron(term, np.eye(2))
            H += 0.5 * J * term
        
        # Δ·ZZ term
        term = 1
        for k in range(L):
            if k == i or k == j:
                term = np.kron(term, Z)
            else:
                term = np.kron(term, np.eye(2))
        H += delta * J * term
    
    return H

def compute_entropy_and_capacity(H, L, chi_values):
    """
    Compute entanglement entropy and capacity for various bond dimensions.
    
    Returns:
        entropy_values: S(χ) for each χ
        capacity_values: κ₂(χ) for each χ
    """
    # Diagonalize
    eigenvalues, eigenvectors = np.linalg.eigh(H)
    ground_state = eigenvectors[:, 0]
    
    # Reshape to MPS form
    psi = ground_state.reshape([2] * L)
    
    results = []
    for chi in chi_values:
        # Compute reduced density matrix for half-chain
        # (Simplified - would use proper MPS in full implementation)
        
        # Entanglement entropy S(χ)
        # Capacity κ₂(χ) = variance of entanglement spectrum
        
        # Placeholder for actual MPS computation
        # In practice, would use quimb or similar
        pass
    
    return results

def model_selection(L_values, delta_values, chi_values):
    """
    Compare saturating vs log-linear models for capacity scaling.
    
    For each Δ:
    - Fit C(L) = C_sat vs C(L) = a*log(L) + b
    - Compute ΔAIC
    - Positive ΔAIC → saturation (IN SCOPE)
    - Negative ΔAIC → log growth (OUT OF SCOPE)
    """
    results = {}
    
    for delta in delta_values:
        delta_results = {
            'delta': delta,
            'expected_scope': 'IN_SCOPE' if delta > 1 else 'OUT_OF_SCOPE',
            'measurements': []
        }
        
        for L in L_values:
            H = xxz_hamiltonian(L, delta)
            # Compute capacity for various χ
            # ... (implementation details)
            pass
        
        # Compute ΔAIC
        # Positive → saturation model preferred
        # Negative → log-linear model preferred
        
        # results[delta] = {'delta_aic': ..., 'verdict': ...}
        pass
    
    return results

def run_xxz_boundary_test():
    """
    Main test runner.
    
    Tests XXZ model at various Δ values to validate scope boundary.
    """
    test_config = {
        'L_values': [8, 12],
        'delta_values': [0.0, 0.5, 1.0, 1.1, 1.5, 2.0],
        'chi_values': [4, 8, 16],
        'boundary': 'cyclic'
    }
    
    expected_results = {
        0.0: 'REJECT',   # Critical
        0.5: 'REJECT',   # Critical
        1.0: 'REJECT',   # Critical (Heisenberg)
        1.1: 'MARGINAL', # Near boundary
        1.5: 'ACCEPT',   # Gapped
        2.0: 'ACCEPT'    # Gapped
    }
    
    print("XXZ Boundary Test")
    print("=================")
    print(f"Testing Δ values: {test_config['delta_values']}")
    print(f"L values: {test_config['L_values']}")
    print()
    
    # Run tests
    results = []
    for delta in test_config['delta_values']:
        print(f"Δ = {delta}: ", end="")
        # ... implementation
        expected = expected_results[delta]
        print(f"Expected: {expected}")
    
    return results

if __name__ == '__main__':
    run_xxz_boundary_test()
```

---

## Expected Outcomes

### If Framework is Correct

| Δ | ΔAIC Sign | Verdict | Scope |
|---|-----------|---------|-------|
| 0.0 | Negative | REJECT | OUT OF SCOPE (critical) |
| 0.5 | Negative | REJECT | OUT OF SCOPE (critical) |
| 1.0 | Negative | REJECT | OUT OF SCOPE (critical) |
| 1.1 | Near zero | MARGINAL | Boundary |
| 1.5 | Positive | ACCEPT | IN SCOPE (gapped) |
| 2.0 | Positive | ACCEPT | IN SCOPE (gapped) |

### Success Criterion

**Framework scope is validated if:**
- Δ > 1: ΔAIC > 0 (saturation preferred)
- Δ ≤ 1: ΔAIC < 0 (log-linear preferred)
- Sharp transition at Δ ≈ 1

---

## Significance

This test would:
1. **Validate scope boundary** — Confirm framework correctly identifies gapped vs critical
2. **Establish physical grounding** — Show scope is not arbitrary but reflects real physics
3. **Enable extension** — If successful, framework can be trusted for other models
4. **Provide publication-quality evidence** — Clean demonstration of scope-aware validation

---

## Files

| File | Purpose |
|------|---------|
| `exp_P2_XXZ_boundary_runner.py` | Test runner (to implement) |
| `xxz_results.json` | Results per Δ value |
| `xxz_boundary_summary.md` | Analysis and conclusions |