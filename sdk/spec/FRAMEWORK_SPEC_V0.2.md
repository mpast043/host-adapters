# FRAMEWORK_SPEC_v0.2.md

## v4.5 Research Framework Specification

Version: 0.2.0
Date: 2026-02-25
Status: Draft, for multi-agent workflow governance

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

   1. SUPPORTED: passes all listed falsifiers for the stated scope.
   2. INCONCLUSIVE: no falsifier failure, but key robustness or model-selection criteria do not resolve.
   3. REJECTED: fails at least one falsifier for the stated scope.

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

#### Claim 3: MERA entanglement scaling with bond dimension χ (hardening target)

Statement (Option A scope, eligible for SUPPORTED now):
For fixed N and a fixed bipartition, MERA entanglement entropy S scales approximately affine in log χ over the tested χ range, consistent with the cut-counting bound S ≤ K_cut · log χ.

Statement (Option B scope, only after additional bridge tests):
Across multiple partitions, the fitted slope of S versus log χ scales with the partition-dependent cut proxy K_cut, consistent with minimal-cut intuition.

Scope:

1. fixed N for baseline
2. fixed MERA layout and construction or training method
3. explicit bipartition definition for baseline
4. χ sweep and seed policy declared

Required observables:

1. S(χ, seed, partition)
2. K_cut(partition) computed by a declared proxy rule
3. model fits comparing S versus log χ against alternatives

Falsifiers and acceptance criteria for Claim 3
These are required for the verdict label to be mechanically earned.

Falsifier 3.1 Monotonicity
For each partition, the median S across seeds is non-decreasing with χ.
Pass condition: true for all adjacent χ steps in the sweep.

Falsifier 3.2 Replicate robustness
The fitted slope a in S = a·log χ + b is positive for every seed and stable across seeds.
Pass condition: no negative slopes; coefficient of variation of slopes is at or below a declared threshold (default 10 percent) unless overridden with justification.

Falsifier 3.3 Model selection
The log model must outperform plausible alternatives, not just correlate well.
Required fitted models:

1. S = a·log χ + b
2. S = a·χ + b
3. S = a·(log χ)^p + b with p fitted
   Pass condition: ΔAIC and ΔBIC at least 10 versus the next-best non-log model, or best holdout error if holdout evaluation is used.

Falsifier 3.4 Bound validity
The explicit bound S ≤ K_cut · log χ must hold for every data point.
Pass condition: zero violations. Any violation is REJECTED unless traced to a verified computation bug and rerun.

Falsifier 3.5 Cut-size bridge (required only for Option B)
Across partitions, the slope a_partition is proportional to K_cut_partition.
Pass condition: strong correlation between a_partition and K_cut_partition under a declared threshold (default r ≥ 0.9) with no single-partition dominance in the fit.

Verdict mapping for Claim 3

1. SUPPORTED Option A: 3.1 through 3.4 pass for baseline partition, and 3.2 passes.
2. SUPPORTED Option B: 3.1 through 3.5 pass across partitions.
3. INCONCLUSIVE: monotonicity holds but model selection or robustness does not resolve.
4. REJECTED: any falsifier failure (especially bound violations).

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
3. for Claim 3 specifically, formalize the cut-counting bound and define K_cut proxy requirements

Outputs:

1. agentB_math.md
2. agentB_claims.json

### Agent C Engineer

Responsibilities:

1. map claims to minimal prototype experiments that directly exercise the falsifiers
2. discover true file paths and avoid hallucinated paths
3. write code only under <RUN_DIR>/prototype/
4. generate the required raw artifacts to support model comparison and bound checks
5. implement Claim 3 sweep, replicate policy, model comparison, and partition variation when required

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

Experiment-specific required artifacts (Claim 3 minimum)

1. manifest.json with run meta, χ list, seeds, partitions, versions
2. raw_entropy.csv with columns: chi, seed, partition_id, S, run_status, notes
3. fits.json with model parameters and AIC BIC and or holdout errors
4. bound_checks.json with K_cut, bound value, pass fail, and margins
5. verdict.json with falsifier results and final label

## 7. ACCEPTANCE CRITERIA

Global acceptance criteria

1. 8 to 12 real sources with full citation records
2. 1 to 2 math supports with derivations and falsifiers
3. prototype plan maps to real file paths
4. contract compliance passes
5. schema lint reports zero errors and zero warnings
6. deterministic replay matches on re-evaluation where applicable

Claim 3 acceptance criteria for SUPPORTED (Option A)

1. expanded χ sweep beyond four points
2. replicate seeds per χ
3. model comparison logged with AIC BIC or holdout error
4. explicit bound checks logged with zero violations
5. verdict.json records all falsifiers and their pass status

## 8. REFERENCES

Ambjørn, J., Jurkiewicz, J., and Loll, R. (2005). Spectral dimension of the universe. Physical Review Letters, 95(17), 171301.

Ryu, S., and Takayanagi, T. (2006). Holographic derivation of entanglement entropy from the anti-de Sitter space conformal field theory correspondence. Physical Review Letters, 96(18), 181602.

Tishby, N., Pereira, F. C., and Bialek, W. (1999). The information bottleneck method. arXiv preprint physics/0004057.

Vidal, G. (2007). Entanglement renormalization. Physical Review Letters, 99(22), 220405.

Van Raamsdonk, M. (2010). Building up spacetime with quantum entanglement. General Relativity and Gravitation, 42(10), 2323 to 2329.
