# Capacity of Entanglement Literature Values

**Key Discovery:** Framework "capacity" maps to **capacity of entanglement** (second cumulant κ₂), NOT von Neumann entropy (first cumulant κ₁).

From [de Boer, Järvelä, Keski-Vakkuri (PRD 99, 066012, 2019)](https://journals.aps.org/prd/pdf/10.1103/PhysRevD.99.066012):

> Capacity of entanglement C_E = Var(H_A) = ⟨H_A²⟩ - ⟨H_A⟩²

## Cumulant Structure of Entanglement

| Cumulant | Name | Formula | Physical Meaning |
|----------|------|---------|------------------|
| κ₁ | Entropy S | -Tr(ρ ln ρ) | Average entanglement |
| κ₂ | **Capacity** C | Tr(ρ(ln ρ)²) - S² | Fluctuation in entanglement |
| κ₃ | Skewness | ... | Spectrum asymmetry |

## Literature Predictions

### From [Khoshdooni et al. PRD 112, 026027 (2025)](https://journals.aps.org/prd/abstract/10.1103/7cg6-m7dn)

For Lifshitz theories with dynamical exponent z:
- C_E has universal logarithmic term
- C_E shows monotonic behavior under RG for z=1 (relativistic)
- For non-relativistic (z>1), C_E is NOT monotonic

### From [de Boer et al. PRD 99, 066012 (2019)](https://journals.aps.org/prd/pdf/10.1103/PhysRevD.99.066012)

For interval of length ℓ in 1+1D CFT:
- C_E ∝ c × (π²/6) for low-lying states
- Capacity tracks entropy for small subsystems
- Related to quantum fluctuations in holographic duals

### From [Mozaffari JHEP 09 (2024) 068](https://arxiv.org/abs/2407.16028)

For volume-law systems:
- C_E behaves differently than in area-law systems
- Non-local theories show enhanced capacity

## Test Predictions

| Model | Central Charge c | Expected C_E Behavior |
|-------|------------------|------------------------|
| Heisenberg | 1 | Logarithmic growth with L |
| Ising | 1/2 | Half the Heisenberg coefficient |
| XXZ Δ≤1 | 1 | Same as Heisenberg (gapless) |
| XXZ Δ>1 | - | Different behavior (gapped) |

## Key Physical Insights

1. **C_E = 0 for pure states**: The modular Hamiltonian is constant
2. **C_E = 0 for maximally mixed states**: The modular Hamiltonian is constant
3. **C_E > 0 for partially mixed states**: Non-trivial entanglement spectrum structure
4. **C_E tracks S for critical systems**: Both grow logarithmically

## References

1. de Boer et al., PRD 99, 066012 (2019)
2. Khoshdooni et al., PRD 112, 026027 (2025)
3. Mozaffari, JHEP 09, 068 (2024)
4. Arias et al., JHEP 03, 175 (2023)