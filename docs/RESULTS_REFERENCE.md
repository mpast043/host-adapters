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

## Experiment Runs

[All run verdicts will go here]

---

## Raw Data Paths

[File paths to all verdict files]