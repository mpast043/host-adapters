# FRAMEWORK_SPEC_v0.1.md
## v4.5 Research Framework Specification

**Version:** 0.1.0  
**Date:** 2026-02-24  
**Status:** Draft - For Multi-Agent Workflow Governance

---

## 1. PURPOSE

This specification defines the operating discipline for research conducted under the Capacity-Governed Framework (CGF) multi-agent system. It anchors all claims, governs citation requirements, mandates reproducibility standards, and establishes falsifiable hypotheses.

---

## 2. CORE PRINCIPLES

### 2.1 Claims Discipline
- Every claim must have a **citation** or **derivation**.
- Claims are categorized:
  - **Type A (Established)**: Peer-reviewed source with independent replication.
  - **Type B (Working)**: Single source or derivation from first principles.
  - **Type C (Speculative)**: Hypothesis with explicit falsifier defined.

### 2.2 Citations
- Each source requires: title, authors, year, venue, URL, and 3–5 bullet rationale.
- Preferred sources: arXiv, peer-reviewed journals, conference proceedings.
- Web sources must be archived (Wayback Machine or equivalent).

### 2.3 Reproducibility
- All experiments must specify:
  - Deterministic seed(s)
  - Exact software versions (Python, GPU, CUDA if applicable)
  - Input data or generation procedure
  - Expected output format and validation criteria
- Code must be CPU-runnable on standard hardware (no exotic dependencies).

### 2.4 Definitions
All technical terms used in claims must appear here first:

| Term | Definition | Source |
|------|------------|--------|
| **Emergent Spacetime** | Geometry that arises from informational/quantum constraints rather than fundamental manifold. | Type B: Van Raamsdonk et al. |
| **Spectral Dimension** | Effective dimension computed from random walk return probability decay on a graph or manifold. | Type A: Ambjørn et al. (2005) |
| **MERA** | Multi-scale Entanglement Renormalization Ansatz - tensor network for critical systems. | Type A: Vidal (2007) |
| **RT Formula** | Ryu-Takayanagi formula relating entanglement entropy to minimal surface area. | Type A: Ryu & Takayanagi (2006) |
| **Nonseparable Laplacian** | Operator where eigenfunctions cannot be factored into product of independent components. | Type B: Working definition |
| **Information Bottleneck** | Trade-off between compression and prediction in information theory. | Type A: Tishby et al. (1999) |
| **Renormalization Dimension Flow** | Change in effective dimension under coarse-graining/scale transformation. | Type C: Hypothesis |

---

## 3. RESEARCH SCOPE

### 3.1 Primary Domains
1. Emergent spacetime from quantum information
2. Spectral dimension and random walks on graphs
3. MERA tensor networks and holography
4. Renormalization group flows and geometric emergence
5. Nonseparable operators and capacity constraints
6. Information bottlenecks and effective geometry

### 3.2 Falsifiable Claims (Framework v4.5)

#### Claim 1: Spectral Dimension as Effective Observer Capacity
**Statement:** The spectral dimension of a graph at diffusion time τ equals the effective dimension experienced by an observer with capacity constraint C_obs = f(τ).

**Derivation Sketch:**
1. Random walk return probability P(τ) ∝ τ^(-d_s/2) defines spectral dimension d_s.
2. Observer with capacity C can resolve at most C degrees of freedom.
3. At timescale τ, available resolution constrains visible subgraph.
4. Proven for finite graphs with Laplacian L having |supp(ψ)| ≤ C.

**Observable Measurement:** 
- Compute d_s via heat kernel trace on sample graphs.
- Measure capacity via information bottleneck compression ratio.
- Compare: d_s(observed) vs. d_s(capacity-limited).

**Falsifier:** If capacity-constrained observers measure d_s that does NOT correlate with τ^(-d_eff/2) where d_eff = f(C), claim is falsified.

#### Claim 2: MERA as Optimal Capacity Allocator
**Statement:** For 1D critical systems, MERA minimizes the capacity cost function C_geo + C_int while preserving entanglement structure.

**Derivation Sketch:**
1. Each tensor in MERA consumes C_geo (physical indices) + C_int (bond dimensions).
2. Isometries reduce bond dimension while preserving reduced density matrix.
3. Disentanglers optimize for minimal mutual information across cut.
4. Result: MERA achieves optimal compression-expressiveness tradeoff.

**Observable Measurement:**
- Compare MERA bond dimensions to random tensor networks.
- Measure reconstruction error vs. capacity cost.
- Compute entanglement entropy across any cut using RT formula.

**Falsifier:** If random tensor network achieves lower (C_geo + C_int) at same reconstruction error, claim is falsified.

---

## 4. AGENT ROLES

### Agent A: Researcher
- Gathers 8–12 real sources relevant to domains in 3.1.
- Validates citations (no fabrication).
- Outputs: `agentA_research.md`, `agentA_sources.json`.

### Agent B: Mathematician
- Produces 1–2 concrete math supports for Claim 1 or Claim 2.
- Must include: definitions, derivation/proof sketch, observable, falsifier.
- Outputs: `agentB_math.md`, `agentB_claims.json`.

### Agent C: Engineer
- Maps claims to minimal prototype experiments.
- Discovers true file paths (no hallucination).
- Writes code ONLY under `<RUN_DIR>/prototype/`.
- Outputs: `agentC_engineering.md`, `agentC_patch_plan.json`, optional code.

---

## 5. GOVERNANCE CONSTRAINTS

1. All side-effect actions must route through CGF governance.
2. Allowed side effects: writing files ONLY under designated `outputs/` subdirectories.
3. No edits to repo source files during runtime.
4. Contract compliance suite must pass (8/8 tests) before and after run.
5. Replay verification must confirm deterministic decisions.

---

## 6. OUTPUT ARTIFACTS

### Phase 2 (Researcher)
- `agentA_research.md`
- `agentA_sources.json`

### Phase 3 (Mathematician)
- `agentB_math.md`
- `agentB_claims.json`

### Phase 4 (Engineer)
- `agentC_engineering.md`
- `agentC_patch_plan.json`
- `prototype/*.py` (optional)

### Phase 5 (Governance)
- `multiagent_timeline.md`
- `schema_lint.txt`
- `replay_verify.txt` (if applicable)

---

## 7. ACCEPTANCE CRITERIA

- [ ] 8–12 real sources with full citations.
- [ ] 1–2 math claims with derivations and falsifiers.
- [ ] Prototype plan maps to real file paths.
- [ ] Contract compliance: 8/8 tests pass.
- [ ] Schema lint: 0 errors, 0 warnings.
- [ ] Deterministic replay: decisions match on re-evaluation.

---

## 8. REFERENCES

Ambjørn, J., Jurkiewicz, J., & Loll, R. (2005). Spectral dimension of the universe. *Physical Review Letters*, 95(17), 171301.

Ryu, S., & Takayanagi, T. (2006). Holographic derivation of entanglement entropy from the anti-de Sitter space/conformal field theory correspondence. *Physical Review Letters*, 96(18), 181602.

Tishby, N., Pereira, F. C., & Bialek, W. (1999). The information bottleneck method. *arXiv preprint physics/0004057*.

Vidal, G. (2007). Entanglement renormalization. *Physical Review Letters*, 99(22), 220405.

Van Raamsdonk, M. (2010). Building up spacetime with quantum entanglement. *General Relativity and Gravitation*, 42(10), 2323–2329.
