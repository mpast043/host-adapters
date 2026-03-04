# P3 Gluing/Excision Stability Test - Methodological Note

**Date**: 2026-03-04
**Status**: ARCHIVED - Methodological Flaw Identified

---

## Executive Summary

The P3 gluing/excision stability test contains a **fundamental methodological flaw** that invalidates its conclusions for physics claims.

---

## The Flaw

P3 tests **naive tensor product gluing**:

```python
psi_glued = np.kron(psi_A, psi_B)
```

This assumes partitions A and B are **uncorrelated**, which is **false for entangled ground states**.

### Why This is Wrong

1. **Entangled ground states have correlations** between partitions
2. **Naive tensor product destroys entanglement** structure
3. **Resulting state is not the true ground state** of the combined system

### What Should Have Been Tested

For MERA states, the correct approach is:

1. **Isometric gluing**: Use MERA isometric operations
2. **Causal cone preservation**: Maintain causal cone structure
3. **Entanglement structure**: Preserve entanglement between partitions

---

## Test Results (ARCHIVED)

| Model | L | Gluing Error | Verdict | Valid? |
|-------|---|--------------|---------|--------|
| Ising | 8 | 0.178 | REJECT | ❌ Invalid |
| Heisenberg | 8 | 0.914 | REJECT | ❌ Invalid |

Both results are **invalid for physics conclusions** due to methodological flaw.

---

## Correct Approach

### MERA Isometric Gluing

```python
def mera_glue(mera_A, mera_B, bond_structure):
    """
    Correct approach: Use MERA isometric operations.
    
    1. Identify shared bonds between A and B
    2. Apply isometries to preserve entanglement
    3. Verify causal cone structure is maintained
    """
    # Apply MERA ascending/descending superoperators
    # Preserve entanglement structure across boundary
    pass
```

This requires:
- Full MERA tensor network implementation
- Isometry constraints
- Causal cone analysis

---

## Recommendation

1. **Archive P3 results** with this note
2. **Do not use for physics claims**
3. **Implement P3 v3** with MERA isometric gluing if needed
4. **Priority: LOW** - P2 already provides scope identification

---

## Context

P3 was intended to test whether MERA states can be glued together while preserving physical properties. However, the naive tensor product approach tests the wrong concept entirely.

The correct implementation would require significant additional infrastructure (full MERA with isometries). Given that P2 already correctly identifies framework scope, P3 has lower priority.