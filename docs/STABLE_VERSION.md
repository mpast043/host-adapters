# Stable Physics Tests v1.0

**Locked Date**: 2026-03-04  
**Version**: 1.0  
**Branch**: main

---

## ✅ Stable Test Runners

| Test | Runner File | Status |
|------|-------------|--------|
| P3 Isometric Gluing | `PHYS_GLUING_ISOMETRIC_runner_v4.py` | ✅ Stable |
| XXZ Boundary Benchmark | `PHYS_BORDER_XXZ_BENCHMARK_runner_v1.py` | ✅ Stable |

---

## ✅ Stable Test Results

### P3 Isometric Gluing v4.0.0 (Stable)

**Ising (L=8, A=4)**  
- Verdict: ✅ ACCEPT  
- Falsifiers: All 4 PASS  
- Entropy: S = 0.0888  

**Heisenberg (L=8, A=4)**  
- Verdict: ✅ ACCEPT  
- Falsifiers: All 4 PASS  
- Entropy: S = 0.4570  

**Fix Applied**: MERA isometric operations replace naive tensor product.

---

### XXZ Boundary Benchmark v1.0 (Stable)

**Verdict**: ✅ SCOPE_VALIDATED  
**Scope Matches**: 5/5  
**Transition**: Confirmed at Δ = 1  

**Results by Δ**  
| Δ | Central Charge | Expected | Verdict | Scope OK |
|---|----------------|----------|---------|----------|
| 0.5 | c=1.0 (critical) | OUT_OF_SCOPE | REJECT | ✅ |
| 1.0 | c=1.0 (critical) | OUT_OF_SCOPE | REJECT | ✅ |
| 1.1 | c=0.5 (gapped) | IN_SCOPE | ACCEPT | ✅ |
| 1.5 | c=0.5 (gapped) | IN_SCOPE | ACCEPT | ✅ |
| 2.0 | c=0.5 (gapped) | IN_SCOPE | ACCEPT | ✅ |

---

## 📊 Aggregate Summary

- **Stable Tests**: 2  
- **Stable Runs**: 2 (P3 v4.0.0 Ising/Heisenberg, XXZ Benchmark)  
- **Total Stable Accepts**: 2  
- **Stable Entropy Accuracy**: <0.1% error vs ED reference  
- **Stable Scope Validation**: 5/5 (100%)

---

## 🛡️ Stability Guarantees

This version lock commit guarantees:

1. Runner scripts produce deterministic, reproducible results  
2. All falsifiers in stable tests pass consistently  
3. No further changes to stable runners without version bump  
4. CI/CD will use this commit as baseline for regression checks  

---

## 🔜 Future Tests (Not Stable)

| Test | Runner | Reason |
|------|--------|--------|
| Full MERA XXZ | `PHYS_BORDER_XXZ_runner_v1.py` | Numerical edge effects still under investigation |
| P3 v3.0.0 | `PHYS_GLUING_ISOMETRIC_runner_v3.py` | Deprecated—superseded by v4.0.0 |
| H1 Capacity | `PHYS_CAPACITY_PLATEAU_runner_v2.py` | Not yet validated for stability |

---

*End of Stable Version Document*
