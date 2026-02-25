# Claim 3B: Windowed Regime Detection
**Framework v0.2.1 Extension | Date: 2026-02-25**

## Hypothesis
There exists an early χ window where log-linear scaling (S ~ a·log χ + b) is decisively preferred over the saturating model, before saturation effects dominate at higher χ. This validates the existence of a capacity-limited regime (Regime I) as a distinct phase preceding saturation (Regime II).

## Claim Statement
For entanglement-maximizing MERA states with fixed N and partition size A, there exists a pre-registered early χ window [χ_min, χ_early] where:
1. Log-linear model is decisively preferred (ΔAIC ≥ 10 over saturating)
2. The extended sweep [χ_min, χ_max] shows saturating model preference
3. This indicates a transition from capacity-limited to saturation physics

## Scope
| Parameter | Value |
|-----------|-------|
| Fixed | N=16 spins, entanglement-max state construction |
| Varied | A ∈ {4, 6, 8} (three partition sizes) |
| χ sweep | [2, 4, 6, 8, 12, 16, 24, 32, 48, 64] |
| Pre-registered windows | W1=[2,8], W2=[2,12], W3=[2,16] |
| Extended comparison | [2,64] full sweep |
| Seeds | 42, 7, 99 (3 per config) |

## Falsifiers

### F3B.1 Monotonicity (Inherited from Claim 3A)
Entanglement entropy S must be non-decreasing with χ for each partition.
- Pass: S(χ_{i+1}) ≥ S(χ_i) − ε for all adjacent pairs
- Fail: Any decrease → REJECTED

### F3B.2 Window Detection (Primary)
For at least one pre-registered window W ∈ {W1, W2, W3} and at least two partition sizes:
- Log-linear AIC < Saturating AIC − 10
- Slope a > 0 in log-linear fit
- **Pass**: Regime I detected in at least one window for ≥2 partitions
- **Fail**: No window shows log-linear preference → REJECT capacity window hypothesis

### F3B.3 Transition Consistency
If F3B.2 passes (Regime I detected), the extended sweep must show Regime II:
- On [2,64]: Saturating model preferred OR AIC difference < 10 (inconclusive)
- Fitted S_∞ within declared tolerance of S_max(A) = A·ln(2)
- **Pass**: Transition pattern confirmed (I → II)
- **Fail**: No transition → claim may be vacuous or incorrectly formulated

### F3B.4 Cross-Partition Robustness
Windowed Regime I detection must generalize:
- **Pass**: F3B.2 satisfied for ≥2 of 3 partition sizes
- **Fail**: Only single partition shows window → weak support, INCONCLUSIVE

## Verdict Mapping
| Verdict | Condition |
|---------|-----------|
| **SUPPORTED (capacity window)** | F3B.1–F3B.4 all pass |
| **INCONCLUSIVE** | F3B.1 pass, but F3B.2 weak or F3B.4 fail |
| **REJECTED** | F3B.1 fail, or F3B.2 fail across all windows |

## Theoretical Justification
Physical systems often show:
1. **Low χ**: Each added bond dimension provides near-independent entanglement channel → log S growth
2. **High χ**: Channels saturate, disentanglers become constrained → S approaches theoretical max
3. **Transition**: χ_crit ~ exp(S_max / K_cut) where geometric and capacity factors balance

The windowed approach tests whether this physical intuition manifests in MERA numerics.
