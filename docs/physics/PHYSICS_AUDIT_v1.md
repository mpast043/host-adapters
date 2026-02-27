# PHYSICS_AUDIT_v1

Date: 2026-02-27
Canonical repo: `/tmp/openclaws/Repos/host-adapters`

## Scope
This audit separates three layers:
1. Internal formal consistency under stated assumptions.
2. Local numerical reproducibility.
3. External physical-claim strength.

## Tier A (Internal Math Consistency)
Status: PASS on local checks.

Evidence:
- `capacity-demo` framework tests: `13 passed` (`docs/state/physics_audit_logs/tier_a_test_framework_b.txt`)
- theorem validation tests: `12 passed` (`docs/state/physics_audit_logs/tier_a_test_theorem_validation.txt`)
- quick validation harness completed (`docs/state/physics_audit_logs/tier_a_validation_harness_quick.txt`)

Interpretation:
- The local theorem/consistency harness is healthy for the implemented assumptions and code paths.

## Tier B (Independent Numerical Replication in host-adapters)
Status: matrix executed for `ising_open`, `ising_cyclic`, `heisenberg_open`, `heisenberg_cyclic` at `L=8` and `L=16`.

Primary outputs (latest matrix set):
- `host-adapters-experimental-data/host-adapters/outputs/physics_audit/ising_open_L8/20260227T230414Z_e158654b/verdict.json`
- `host-adapters-experimental-data/host-adapters/outputs/physics_audit/ising_open_L16/20260227T230447Z_3552cc17/verdict.json`
- `host-adapters-experimental-data/host-adapters/outputs/physics_audit/ising_cyclic_L8/20260227T230424Z_5620bb56/verdict.json`
- `host-adapters-experimental-data/host-adapters/outputs/physics_audit/ising_cyclic_L16/20260227T230514Z_da778d49/verdict.json`
- `host-adapters-experimental-data/host-adapters/outputs/physics_audit/heisenberg_open_L8/20260227T230434Z_1f12583a/verdict.json`
- `host-adapters-experimental-data/host-adapters/outputs/physics_audit/heisenberg_open_L16/20260227T230541Z_479e6ba9/verdict.json`
- `host-adapters-experimental-data/host-adapters/outputs/physics_audit/heisenberg_cyclic_L8/20260227T230440Z_30ffdd56/verdict.json`
- `host-adapters-experimental-data/host-adapters/outputs/physics_audit/heisenberg_cyclic_L16/20260227T230706Z_410aac0d/verdict.json`

Observed pattern:
- All listed Tier B verdicts are `REJECTED` under current falsifier thresholds.
- Typical failure driver is `P3.4` (saturating model does not beat log-linear by required margin), with occasional `P3.2` or `P3.3` failures depending on model/L.

## Tier C (Physical-Interpretation Audit)
### Bin A: Mathematically Proved Under Assumptions
- Internal monotonicity/consistency claims in the framework stress-test section (PDF pages 60-64) are internally marked PASS.
- Local theorem/consistency tests in `capacity-demo` pass.

### Bin B: Empirically Supported in Local Experiments
- Claim 3P runner behavior is numerically reproducible across boundary conditions and model variants.
- Reproducibility currently supports a stable **rejection** outcome under the implemented P3 falsifiers, not support for the targeted convergence claim.

### Bin C: Conceptual/Speculative (Not Physically Established)
- External physical interpretation remains limited.
- The framework itself explicitly states non-claims about introducing new physical dynamics (PDF page 58) and warns that capacity boundaries are not automatically physical phase transitions (PDF page 59).

## PDF Claim Mapping (with citations)
1. Claim family: internal consistency / regression integrity
- PDF basis: pages 60-64 report stress-test claims as PASS.
- Audit class: Bin A (under framework assumptions).
- Code evidence: Tier A pass logs in `docs/state/physics_audit_logs/`.

2. Claim family: selection as separate observer component
- PDF basis: page 25 (selection as additional degree of freedom), page 26 (selection as unformalized/dark component).
- Audit class: Bin C for physical interpretation; Bin A/B only for formal/operational framing.
- Code evidence: selection contract tooling (`tools/validate_workflow_auto_run.py`, `results/selection/ledger.jsonl`) supports operational governance checks, not physical establishment.

3. Claim family: strong physical convergence/generalized physical validity
- PDF basis: non-claims and terminology limits on pages 58-59.
- Audit class: Bin B/C depending on exact claim wording.
- Code evidence: Tier B matrix verdicts in `host-adapters-experimental-data/host-adapters/outputs/physics_audit/*/verdict.json` are currently REJECTED under present falsifiers.

## Source-of-Truth Notes
- Canonical framework PDF used: `/Users/meganpastore/Clawdbot/Repos/capacity-demo/Framework with selection.pdf` (64 pages).
- Alternate file `/Users/meganpastore/Clawdbot/Repos/framework-recon-clone/Framework v4 5 CANONICAL UPDATED.pdf` is empty and excluded from evidence.

## Layered Verdict (Math-Physics Accuracy)
1. Internal formal consistency: supported for tested assumptions.
2. Local numerical reproducibility: supported, but currently reproduces rejection outcomes for Claim 3P thresholds.
3. External physical establishment: not established; must remain explicitly labeled as limited/speculative unless new external-calibration evidence is added.
