# Framework Physics Validation Results Reference

**Generated:** 2026-03-04
**Sources:**
- `host-adapters/` - Main repo
- `host-adapters-experimental-data/` - Experimental data

---

## Quick Reference

[Summary table will go here]

---

## Claims Summary

### Claim 1: Spectral Dimension as Capacity-Limited Effective Geometry

**Verdict:** ✅ SUPPORTED

**Key Metrics:**
- Falsifier 1.1: PASSED (all 7 configs within 25% of expected β=0.683)
- Falsifier 1.2: PASSED (3/7 high-cap configs show plateau-like d_eff tail)
- Sample count: 30
- Seed: 42

---

### Claim 2: MERA as Optimal Capacity Allocator

**Verdict:** ✅ SUPPORTED

**Key Metrics:**
- Falsifier 2.1: PASSED
- Falsifier 2.2: PASSED
- Slope: 0.1546
- Sample count: 20
- Seed: 42

**Additional Evidence (exp2b_asymptotic):**
- Verdict: ✅ SUPPORTED
- Savings preserved at scale: true
- Entanglement scaling preserved: true
- Savings at n=16: 9.47
- Savings at n=128: 11.74

---

### Claim 3: MERA Spectral Dimension Bridge

**Verdict:** ❌ NOT_SUPPORTED

**Key Metrics:**
- Falsifier 3.1: FAILED
- Falsifier 3.2: FAILED
- Correlation: -0.590
- Sample count: 7
- Seed: 42

**Note:** This claim was retested with refined methodology (see Claim 3-v3 below).

---

### Claim 3-v3: Entanglement Entropy Holographic Bound

**Verdict:** ✅ SUPPORTED

**Key Metrics:**
- Falsifier 3.1: PASSED
- Falsifier 3.2: PASSED
- Correlation: 0.996
- Ratio CV: 0.036
- Sample count: 11
- Seed: 42

**Note:** This is a refined formulation of Claim 3 with stronger theoretical grounding

---

### Claim 3P: Physical Hamiltonian Convergence

**Summary:** All configurations REJECTED across all model/boundary/L combinations tested.

**Results by Configuration:**

| Model | Boundary | L | Verdict | Best Fidelity | Final Entropy Error | ΔAIC | Key Failures |
|-------|----------|---|---------|---------------|---------------------|------|--------------|
| ising | open | 8 | REJECTED | 0.0036 | 1.86 | -10.83 | P3.1, P3.2, P3.3, P3.4 |
| ising | cyclic | 8 | REJECTED | 0.9997 | 0.0033 | +0.69 | P3.2, P3.4 |
| ising | cyclic | 16 | REJECTED | 0.9993 | 0.0018 | -4.42 | P3.2, P3.4 |
| heisenberg | cyclic | 8 | REJECTED | 0.68 | 0.31 | -32.66 | P3.3, P3.4 |

**Falsifier Details:**
- **P3.1 (Fidelity Monotonicity):** Passed for cyclic boundaries, failed for open boundary
- **P3.2 (Entropy Convergence):** Failed for ising_cyclic (non-monotonic entropy error), passed for heisenberg
- **P3.3 (Physical Convergence):** Strongly failed for open boundary and heisenberg (fidelity < 0.95), passed for cyclic ising
- **P3.4 (Model Selection):** Failed across all configurations - log-linear model preferred over saturating model

**Critical Discoveries:**
1. **Cyclic boundary improved ΔAIC by 60x vs open boundary** (from -10.83 to +0.69)
2. Cyclic ising achieved excellent physical convergence (fidelity ~0.9997) but failed statistical tests
3. Heisenberg model shows fundamentally different convergence behavior (fidelity only 0.68)
4. All configurations show preference for log-linear scaling over saturating behavior (P3.4)

**Data Sources:**
- `/tmp/openclaws/Repos/host-adapters-experimental-data/runs/RUN_20260227_151454/results/science/claim3p_ising/`
- `/tmp/openclaws/Repos/host-adapters-experimental-data/runs/RUN_20260227_161644/results/claim3p_cyclic/`
- `/tmp/openclaws/Repos/host-adapters-experimental-data/runs/RUN_20260228_033223/results/science/claim3p_*/`

---

## Experiment Runs

### RUN_20260227_151454 - Initial Smoke Tests

**Date:** 2026-02-27
**Mode:** LOCAL_ONLY
**Status:** PARTIAL

**Claims Tested:**
- Claim 2: MERA Optimal Allocator
- Claim 3P: Physical Convergence (Ising model)
- Claim 3: Spectral Dimension Bridge
- Claim 3v4: Partition-dependent behavior
- Claim 3B: Alternative formulation

**Verdicts:**
| Claim | Verdict | Notes |
|-------|---------|-------|
| Claim 2 | ✅ SUPPORTED | Slope: 0.1546 |
| Claim 3P | ❌ REJECTED | ΔAIC: -40.13 (open boundary) |
| Claim 3 | ❌ NOT_SUPPORTED | Correlation: -0.590 |
| Claim 3v4 | ❌ NOT_SUPPORTED | Partition-dependent confirmed |
| Claim 3B | 🔄 NO_EVIDENCE | Insufficient data |

**Key Findings:**
- CGF Layer: PASS (8/8 contracts passed)
- Selection framework validation successful in LOCAL_ONLY mode
- Time budget: 14.5/20 minutes used
- Baseline suite completed; exploration candidates generated

---

### RUN_20260227_161644 - Extended Chi Sweep

**Date:** 2026-02-27
**Continuation Of:** RUN_20260227_151454
**Status:** PARTIAL

**Claims Tested:**
- Claim 3P: Physical Convergence (cyclic boundary)
- Claim 3: Extended chi parameter sweep

**Verdicts:**
| Claim | Verdict | Notes |
|-------|---------|-------|
| Claim 2 | ✅ SUPPORTED | Confirmed from baseline |
| Claim 3P | ❌ REJECTED | ΔAIC: +0.69 (cyclic, improved from -40.13) |
| Claim 3 | 🔄 TENTATIVE_EVOLUTION | F3.2 passes with extended chi |
| Claim 3v4 | ❌ NOT_SUPPORTED | Partition-dependent confirmed |
| Claim 3B | 🔄 NO_EVIDENCE | Insufficient data |

**Key Findings:**
- **Critical Discovery:** Cyclic boundary improved ΔAIC by 60× vs open boundary
- Physical convergence excellent (fidelity ~0.9997)
- Statistical threshold refinement needed for small systems
- Runtime fix applied: Added 'ising_cyclic' to argparse choices
- Time budget: 16/20 minutes used

---

### RUN_20260228_184320 - Final Selection

**Date:** 2026-02-28
**Mode:** COMPLETE
**Status:** PARTIAL (LINT_DEBT)

**Claims Tested:**
- Baseline suite: claim2, claim3, claim3p (local deterministic)
- Exploration subset under budget constraints

**Verdicts:**
| Claim | Verdict | Notes |
|-------|---------|-------|
| Claim 2 | ✅ SUPPORTED | From baseline run |
| Claim 3P | ❌ REJECTED | From baseline run |
| Claim 3 | 🔄 UNDERDETERMINED | Selection pending |

**Key Findings:**
- All workflow steps completed: PASS
- Contract verification: PASS
- Science status: NOT_RUN (lint debt blocked)
- Selection status: UNDERDETERMINED
- Port rotation used for CGF startup: 8080, 18080, 28080, 38080
- LOCAL_ONLY downscoping auto-applied when compute MCP unavailable
- Evidence retained under results/ directory

---

## Raw Data Paths

[File paths to all verdict files]