# Framework Physics Validation Project

Canonical workspace for Framework v4.5/v4.6 physics grounding and validation.

## Project Overview

This project maps Framework concepts to established physics, with a focus on:
- **Capacity of entanglement** (second cumulant κ₂) as the key mapping
- **d_s staircase** validation via tensor RG
- **Δλ analysis** via entanglement gap measurements

## Key Files

| File | Purpose |
|------|---------|
| `docs/physics/FRAMEWORK_PHYSICS_MAPPING.md` | Canonical symbol-to-physics mapping |
| `docs/physics/PREDICTIONS_PAPER.md` | Testable predictions and results |
| `experiments/physics/entanglement_utils.py` | Core calculations (S, C_E, gaps) |
| `experiments/physics/scaling_dimensions_runner.py` | d_s extraction |
| `experiments/physics/entanglement_gap_analysis.py` | Δλ testing |

## Workflow Preferences

### Auto-Approve Mode

For long multi-task sessions (like physics validation), use auto-approve to avoid repeated confirmation clicks:

```bash
claude --dangerously-skip-permissions
```

Or during a session:
```
/accept-all
```

### Task Planning

For complex multi-phase work, expect Claude to create structured task plans with 10+ tasks tracked via TaskCreate/TaskUpdate.

### Commit Frequency

Commit at phase boundaries (every 3-4 tasks) rather than at the end of long sessions.

## Validation Workflow

Use the `/physics-validate` skill for standard validation runs:

```
/physics-validate
```

This runs:
1. Capacity of entanglement calculation
2. d_s staircase check
3. Δλ gap analysis
4. Generate validation report

## Key Results

| Hypothesis | Prediction | Observed | Status |
|------------|------------|----------|--------|
| C_E/S ≈ 1 | Ratio near 1 | 0.94-1.07 | ✅ SUPPORTED |
| Gap ratio ≈ 38% | (λ₀-λ₁)/λ₀ × 100 ≈ 38 | 83.7-96.8% | ❌ NOT SUPPORTED |

## References

- [de Boer PRD 2019](https://journals.aps.org/prd/pdf/10.1103/PhysRevD.99.066012) - Capacity of entanglement
- [Lyu PRR 2021](https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.3.023048) - Scaling dimensions from tensor RG
- [Wald PRR 2020](https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.2.043404) - Entanglement gap closure