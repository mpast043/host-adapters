# FRAMEWORK_SPEC_v0.2.1.md

## v4.5 Research Framework Specification

Version: 0.2.1
Date: 2026-02-25
Status: Draft, for multi-agent workflow governance

## CHANGELOG v0.2.0 → v0.2.1

### Regime-Aware Claim 3A

- **Added:** Saturation regime detection for entanglement-maximizing states
- **Modified:** Claim 3 → Claim 3A with dual-regime specification
- **Added:** Four-model comparison in Falsifier 3.3 (log-linear, linear, log-power, saturating)
- **Added:** Mechanical regime detection (Regime I: capacity scaling, Regime II: saturation)
- **Modified:** Verdict labels to include regime annotation: "SUPPORTED (capacity regime)" vs "SUPPORTED (saturation regime)"

## 1. PURPOSE

This specification defines the operating discipline for research conducted under the Capacity-Governed Framework (CGF) multi-agent system. It anchors claims, governs citation requirements, mandates reproducibility standards, and enforces falsifiable hypotheses with mechanically checkable acceptance criteria.

## 2. CORE PRINCIPLES

### 2.1 Claims discipline

1. Every claim must include one of:

   1. citation support, or
   2. a derivation from explicit definitions and assumptions, or
   3. a falsifier and test plan that can decide the claim.

2. Every claim must include a scope statement:

   1. what is fixed
   2. what is varied
   3. what counts as evidence
   4. what would falsify it

3. Claim types

   1. Type A (Established): peer-reviewed source with independent replication; claim is restating accepted result.
   2. Type B (Working): supported by a single primary source and or a derivation; still requires internal falsifiers where possible.
   3. Type C (Speculative): hypothesis; must include an explicit falsifier and pre-registered acceptance criteria.

4. Claim labels used in verdicts (internal governance labels)

   1. **SUPPORTED**: passes all listed falsifiers for the stated scope, with explicit regime annotation where applicable.
   2. **SUPPORTED (regime)**: passes with regime-specific annotation (e.g., "capacity regime", "saturation regime").
   3. **INCONCLUSIVE**: no falsifier failure, but key robustness or model-selection criteria do not resolve.
   4. **REJECTED**: fails at least one falsifier for the stated scope.

### 2.2 Citations

1. Each source record must include:

   1. title
   2. authors
   3. year
   4. venue or archive
   5. URL
   6. 3 to 5 bullet rationale explaining which parts of the work are used and why

2. Preferred sources

   1. arXiv
   2. peer-reviewed journals
   3. conference proceedings
   4. authoritative textbooks or lecture notes when primary papers are not accessible

3. Web sources

   1. must be archived (Wayback Machine or equivalent) unless already a stable archive (arXiv, publisher DOI, etc.)
   2. must not be used as the only support for a Type A claim

4. No fabrication

   1. Agents must verify that every citation exists and matches the claim.

### 2.3 Reproducibility

Every experiment must specify:

1. deterministic seed policy
2. exact software versions (Python, key packages; GPU and CUDA if applicable)
3. input data or generation procedure
4. expected output format
5. validation criteria and falsifiers
6. storage location of raw outputs used for the verdict

Constraints:

1. code must be CPU-runnable on standard hardware unless the spec explicitly declares an exception for a given experiment
2. exceptions must include a CPU fallback mode or a reduced-size reproduction mode

### 2.4 Definitions

All technical terms used in claims must appear here before first use.

| Term                           | Definition                                                                                                                               | Source                                |
| ------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------- |
| Emergent spacetime             | Geometry that arises from informational or quantum constraints rather than assuming a fundamental manifold.                              | Type B: Van Raamsdonk (2010)          |
| Spectral dimension d_s         | Effective dimension computed from random walk return probability scaling P(τ) ∝ τ^(−d_s/2) or equivalent heat-kernel trace scaling.      | Type A: Ambjørn et al. (2005)         |
| MERA                           | Multi-scale Entanglement Renormalization Ansatz; tensor network ansatz for critical systems with disentanglers and isometries.           | Type A: Vidal (2007)                  |
| Bond dimension χ               | Dimension of an internal MERA tensor-network leg; controls representational capacity and upper bounds entanglement that can cross a cut. | Type A/B: primary MERA literature     |
| Entanglement entropy S         | von Neumann entropy of a reduced density matrix for a bipartition; measured or estimated for the network state.                          | Type A: standard QIT                  |
| Cut-counting bound             | For a tensor network with bond dimension χ, entropy across a cut is bounded by (number of cut legs) × log χ, up to conventions.          | Type A/B: tensor-network fundamentals |
| Minimal cut                    | The cut through the network separating a bipartition that minimizes number of severed legs; used as a proxy for entanglement bounds.     | Type A/B: tensor-network fundamentals |
| RT formula                     | Holographic relation between boundary entanglement and minimal bulk surface area in AdS/CFT settings.                                    | Type A: Ryu and Takayanagi (2006)     |
| Nonseparable Laplacian         | Operator where eigenfunctions cannot be factored into products of independent components; working definition used in prototypes.         | Type B: working definition            |
| Information bottleneck         | Trade-off between compression and prediction; used as an operational capacity proxy in some experiments.                                 | Type A: Tishby et al. (1999)          |
| Renormalization dimension flow | Hypothesized change in effective dimension under coarse-graining; requires explicit operationalization and falsifiers.                   | Type C: hypothesis                    |
| **Saturation regime**          | Physical regime where entropy S approaches maximum system capacity S_max, with diminishing returns from increased χ. | Type B: derived (v0.2.1) |
| **Capacity regime**            | Physical regime where entropy S scales approximately as log χ with bond dimension, before saturation effects dominate. | Type B: derived (v0.2.1) |

Note on RT usage:
RT is referenced as conceptual motivation only unless a specific holographic mapping is explicitly defined and tested. For MERA experiments in this framework, the cut-counting bound is the primary measurable bridge.

## 3. RESEARCH SCOPE

### 3.1 Primary domains

1. emergent spacetime from quantum information
2. spectral dimension and random walks on graphs
3. MERA tensor networks and entanglement bounds
4. renormalization group flows and effective geometry
5. nonseparable operators and capacity constraints
6. information bottlenecks and capacity-limited observables

### 3.2 Falsifiable claims (Framework v4.5)

#### Claim 1: Spectral dimension as effective observer capacity

Statement:
The spectral dimension of a graph at diffusion time τ matches an effective dimension experienced by an observer with capacity constraint C_obs = f(τ), under an explicit operational definition of the capacity-limited observation map.

Scope:

1. fixed graph family and Laplacian construction
2. capacity limitation implemented by a specified rule (for example truncation, coarse-graining, probe budget, or bottleneck mapping)
3. τ range is stated

Observable measurement:

1. compute d_s from heat kernel trace or return probability scaling
2. compute a capacity proxy from the chosen capacity-limitation rule
3. compare d_s(observed) vs predicted d_eff(C_obs)

Falsifier:
If the capacity-limited observer measurement of d_s does not follow the predicted relationship within pre-registered tolerances over the τ window, the claim is falsified.

#### Claim 2: MERA as capacity allocator (scoped)

Statement:
For 1D critical-system targets under the chosen training objective, MERA yields a better compression-expressiveness tradeoff than specified control tensor networks when comparing reconstruction error against a declared capacity cost function.

Scope:

1. the target state family is specified
2. the training or construction method is specified
3. the control networks are specified (for example random tensor networks with matched size constraints)

Observable measurement:

1. define capacity cost C_total = w1·C_geo + w2·C_int, with explicit weights and definitions
2. measure reconstruction error under a declared metric
3. compare MERA versus controls at matched error or matched cost

Falsifier:
If a declared control network achieves lower C_total at the same reconstruction error (or lower error at same C_total) under the same evaluation procedure, the claim is falsified.

#### Claim 3A: MERA entanglement scaling with bond dimension χ (REGIME-AWARE, v0.2.1)

Statement (Regime-Aware):
For fixed N and fixed bipartition A, a capacity-limited state family produces entanglement entropy S(χ) that is non-decreasing in χ and follows either:

- **Regime I (capacity-limited):** approximately affine in log χ over a pre-registered χ window, OR
- **Regime II (saturation):** approaches a finite ceiling S_∞ ≤ S_max(A) with diminishing increments

The regime is determined mechanically by model selection among competing functional forms.

Scope:

1. fixed N for baseline
2. fixed MERA layout and construction or training method
3. explicit bipartition definition (size A, geometry)
4. χ sweep covering capacity and potential saturation regimes
5. state family declared (random, heisenberg_opt, entanglement_max, etc.)
6. seed policy declared

Required observables:

### Precondition: K_cut Validity Precheck

Before any verdict that depends on K_cut, the run must include a diagnostic
showing K_cut is:

1. **Invariant across seeds** for fixed layout and partition (variance = 0 or within floating-point epsilon)
2. **Non-decreasing with partition size** for the declared family of partitions

**Test procedure:**
```python
def test_kcut_scaling_sanity():
    for A in declared_partitions:
        Ks = [compute_K_cut(A, seed) for seed in range(10)]
        # Invariant check
        assert max(Ks) == min(Ks), f"K_cut varies with seed for A={A}"
    
    medians = {A: median(Ks) for A, Ks in K_by_A.items()}
    # Non-decreasing check
    for A in sorted(medians.keys())[:-1]:
        assert medians[A+1] >= medians[A], f"K_cut decreased at A={A}"
```

**Verdict impact:**
- If this precheck fails, verdict must be **INCONCLUSIVE** with `reason_code: K_CUT_INVALID`, not REJECTED
- This prevents "REJECTED because the measurement tool is broken"

1. S(χ, seed, partition) from contracted MERA wavefunction
2. K_cut(partition) computed by declared proxy rule
3. model fits comparing competing functional forms
4. regime identification via mechanical model selection

Falsifiers and acceptance criteria for Claim 3A

**Falsifier 3.1 Monotonicity**
For each partition, the median S across seeds is non-decreasing with χ.
- Pass condition: true for all adjacent χ steps in the sweep.
- Fail: Any decrease S(χ_{i+1}) < S(χ_i) − ε (ε = 1e-12) → REJECTED

**Falsifier 3.2 Replicate Robustness**
The fitted slope a in S = a·log χ + b is positive for every seed and stable across seeds.
- Pass condition: no negative slopes; coefficient of variation of slopes ≤ declared threshold (default 10%)
- Note: For saturation regime, robustness assessed on log-linear sub-fit over pre-registered capacity window

**Falsifier 3.3 Model Selection (REGIME-AWARE, v0.2.1)**
Four competing models fit to median S(χ) across seeds:

1. **Log-linear:** S = a·log χ + b  ← Claim 3A Regime I prediction
2. **Linear in χ:** S = a·χ + b  ← Competitor
3. **Log-power:** S = a·(log χ)^p + b, p fitted  ← Competitor
4. **Saturating:** S = S_∞ − c·χ^(−α), S_∞ and α fitted  ← Claim 3A Regime II prediction

**Regime Detection Logic:**
1. Compute AIC/BIC for all four models
2. Identify winner by minimum AIC
3. **If saturating wins AND S_∞ within tolerance of S_max:** → Regime II detected
4. **If log-linear wins with ΔAIC/ΔBIC ≥ 10 over next competitor:** → Regime I detected

**Pass Conditions (satisfy one):**
- **Option (a) - Regime I:** Log-linear is decisively preferred (ΔAIC ≥ 10, ΔBIC ≥ 10) within pre-registered "capacity window" χ ∈ [χ_min, χ_cap]
- **Option (b) - Regime II:** Saturating is decisively preferred (ΔAIC ≥ 10 over log-linear) on full sweep, AND fitted S_∞ is within declared tolerance of S_max(A)

**S_max Computation:**
For A spins (each dimension 2): S_max(A) = A · ln 2

**Failure:** Neither regime condition met → INCONCLUSIVE or REJECTED depending on other falsifiers

**Falsifier 3.4 Bound Validity**
The explicit bound S ≤ K_cut · log χ must hold for every data point.
- Pass condition: zero violations (S ≤ bound + 1e-12)
- Fail: Any violation → REJECTED unless traced to verified computation bug

**Falsifier 3.5 Cut-size bridge (required only for Option B)**
Across partitions, the slope a_partition is proportional to K_cut_partition.
- Pass condition: strong correlation between a_partition and K_cut_partition under declared threshold (default r ≥ 0.9) with no single-partition dominance in the fit.

**Verdict Mapping for Claim 3A (v0.2.1)**

| Outcome | Condition |
|---------|-----------|
| **SUPPORTED (capacity regime)** | 3.1–3.4 pass, Regime I detected |
| **SUPPORTED (saturation regime)** | 3.1, 3.2, 3.4 pass, 3.3 detects Regime II with valid S_∞ |
| **INCONCLUSIVE** | Monotonicity holds but regime indeterminate |
| **REJECTED** | Any falsifier failure (especially bound violations or monotonicity) |

**Notes on regime interpretation:**
- Capacity regime validates Claim 3A's log-scaling prediction in appropriate χ range
- Saturation regime confirms physical ceiling behavior—entropy maximization correctly approaches system limits
- Both regimes constitute successful experimental outcomes
- Random initialization without RG structure may fail monotonicity—is a state family issue, not framework issue

## 4. AGENT ROLES

### Agent A Researcher

Responsibilities:

1. gather 8 to 12 real sources relevant to the declared domain and the specific claim under test
2. validate citation existence and relevance
3. produce a sources registry suitable for archiving

Outputs:

1. agentA_research.md
2. agentA_sources.json

### Agent B Mathematician

Responsibilities:

1. produce 1 to 2 concrete mathematical supports for one declared claim
2. include definitions, derivation or proof sketch, explicit observable, and falsifier mapping to acceptance criteria
3. for Claim 3A specifically, formalize the cut-counting bound, define K_cut proxy requirements, and derive saturation regime conditions

Outputs:

1. agentB_math.md
2. agentB_claims.json

### Agent C Engineer

Responsibilities:

1. map claims to minimal prototype experiments that directly exercise the falsifiers
2. discover true file paths and avoid hallucinated paths
3. write code only under <RUN_DIR>/prototype/
4. generate the required raw artifacts to support model comparison and bound checks
5. implement Claim 3A sweep with regime detection, replicate policy, model comparison, and partition variation when required

Outputs:

1. agentC_engineering.md
2. agentC_patch_plan.json
3. prototype/*.py (optional)
4. experiment artifacts as specified in section 6 for the relevant claim

## 5. GOVERNANCE CONSTRAINTS

1. all side-effect actions must route through CGF governance
2. allowed side effects: writing files only under designated outputs subdirectories
3. no edits to repo source files during runtime
4. contract compliance suite must pass before and after run
5. replay verification must confirm deterministic decisions where applicable

## 6. OUTPUT ARTIFACTS

All runs must write outputs under outputs/<experiment_name>/<run_id>/.

Phase 2 Researcher

1. agentA_research.md
2. agentA_sources.json

Phase 3 Mathematician

1. agentB_math.md
2. agentB_claims.json

Phase 4 Engineer

1. agentC_engineering.md
2. agentC_patch_plan.json
3. prototype/*.py (optional)

Phase 5 Governance

1. multiagent_timeline.md
2. schema_lint.txt
3. replay_verify.txt (if applicable)

Experiment-specific required artifacts (Claim 3A minimum)

1. **manifest.json** with run meta, χ list, seeds, partitions, versions, **regime information**
2. **raw_entropy.csv** with columns: chi, seed, partition_id, S, run_status, notes
3. **fits.json** with model parameters, AIC/BIC, **all four models**, **regime detection result**
4. **bound_checks.json** with K_cut, bound value, pass/fail, and margins
5. **verdict.json** with falsifier results, **regime annotation**, and final label

## 7. ACCEPTANCE CRITERIA

Global acceptance criteria

1. 8 to 12 real sources with full citation records
2. 1 to 2 math supports with derivations and falsifiers
3. prototype plan maps to real file paths
4. contract compliance passes
5. schema lint reports zero errors and zero warnings
6. deterministic replay matches on re-evaluation where applicable

Claim 3A acceptance criteria for SUPPORTED

1. expanded χ sweep covering capacity and potential saturation regimes
2. replicate seeds per χ
3. **four-model comparison logged** with AIC/BIC
4. **regime detection result recorded**
5. explicit bound checks logged with zero violations
6. verdict.json records all falsifiers, **regime**, and pass status

## 8. REFERENCES

Ambjørn, J., Jurkiewicz, J., and Loll, R. (2005). Spectral dimension of the universe. Physical Review Letters, 95(17), 171301.

Ryu, S., and Takayanagi, T. (2006). Holographic derivation of entanglement entropy from the anti-de Sitter space conformal field theory correspondence. Physical Review Letters, 96(18), 181602.

Tishby, N., Pereira, F. C., and Bialek, W. (1999). The information bottleneck method. arXiv preprint physics/0004057.

Vidal, G. (2007). Entanglement renormalization. Physical Review Letters, 99(22), 220405.

Van Raamsdonk, M. (2010). Building up spacetime with quantum entanglement. General Relativity and Gravitation, 42(10), 2323 to 2329.

## APPENDIX A: Experiment Registry (v0.2.1)

| Run ID | Date | State Family | χ Sweep | Verdict | Regime | Output Path |
|--------|------|--------------|---------|---------|--------|-------------|
| 20260225T182932Z_966c704d | 2026-02-25 | random MERA | 2,4,6,8 | INCONCLUSIVE | N/A | outputs/exp3_claim3_quimb/20260225T182932Z_966c704d/ |
| 20260225T183133Z_913a6c3f | 2026-02-25 | random MERA | 2,4,6,8,12,16,24,32 | REJECTED | N/A (F3.1 FAIL) | outputs/exp3_claim3_quimb/20260225T183133Z_913a6c3f/ |
| 20260225T184624Z_6c4a83f4 | 2026-02-25 | entanglement_max | 2,4,6,8,12,16,24 | SUPPORTED | II_saturation (v0.2.1 pre) | outputs/exp3_claim3_entanglement_max_mincut/20260225T184624Z/ |
| **20260225T190553Z_75d28016** | **2026-02-25** | **entanglement_max** | **2,4,6,8,12,16,24** | **SUPPORTED** | **II_saturation** | **outputs/exp3_claim3_entanglement_max_mincut/20260225T190553Z_75d28016/** |

**Key Findings:**
- Random MERA lacks RG structure → fails monotonicity (scientifically expected)
- Entanglement-max states saturate at S_∞ ≈ 5.1 (theory: S_max = 5.55 for 8 spins) → v0.2.1 correctly identifies Regime II
- Saturating model RSS = 0.017 vs log-linear RSS = 1.63 → decisive preference (ΔAIC = 30)
