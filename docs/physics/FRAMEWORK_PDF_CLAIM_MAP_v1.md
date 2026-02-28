# Framework v4.5 PDF Claim Map (Full Explicit Claim Coverage)

Date: 2026-02-28  
Source PDF: `/Users/meganpastore/Clawdbot/Repos/capacity-demo/Framework with selection.pdf` (64 pages)  
Method: page-wise extraction with `pypdf`, then mapping to local executable evidence in `host-adapters` and `host-adapters-experimental-data`.

## Evidence Classes

1. `DOC_ONLY`: claim is asserted/verified inside the PDF itself; no independent executable confirmation yet.
2. `LOCAL_EXEC`: claim has independent executable local evidence.
3. `OPEN_OR_SPEC`: axiom/conjecture/non-claim/scope guard; not a falsifiable pass/fail claim in current harness.

## A. Explicit `Claim:` Statements in the PDF (20/20)

These are the complete explicit claim statements found in the PDF (`Annex W`, pages 59-63).

| Claim ID | PDF Page | Claim (verbatim-short) | Current Status | Evidence | Next Needed |
|---|---:|---|---|---|---|
| W01 | 59 | If `C1 <= C2`, then filter composition/order properties hold. | PARTIAL (`LOCAL_EXEC` + `DOC_ONLY`) | PDF Annex W Test 1; `/Users/meganpastore/Clawdbot/Repos/capacity-demo/tests/test_framework_b.py` (filter monotonicity), `/Users/meganpastore/Clawdbot/Repos/capacity-demo/tests/test_theorem_validation.py` (monotonicity checks), `/tmp/openclaws/Repos/host-adapters-experimental-data/host-adapters/docs/state/physics_audit_logs/tier_a_test_framework_b.txt` | Add direct assertion of full W01 identity set inside host-adapters regression suite. |
| W02 | 59 | Componentwise infimum yields well-defined shared capacity. | DOC_ONLY | PDF Annex W Test 2 | Add executable lattice/poset proofs in test harness. |
| W03 | 60 | Memory excision is consistent with general excision. | DOC_ONLY | PDF Annex W Test 3 | Add CPMT↔Annex-T consistency tests in code. |
| W04 | 60 | Framework can represent itself without contradiction. | DOC_ONLY | PDF Annex W Test 4 | Add formalization/replay test for self-reference constraints. |
| W05 | 60 | Mixed-regime capacity profiles are coherent. | PARTIAL (`LOCAL_EXEC` + `DOC_ONLY`) | PDF Annex W Test 5; Objective-B run `/tmp/openclaws/Repos/host-adapters-experimental-data/host-adapters/RUN_20260228_052622/results/science/campaign/campaign_index.csv` (`claim3_optionb_regime_check=PASS`) | Add explicit mixed-regime synthetic fixtures tied to Annex S examples. |
| W06 | 60 | Triadic depth maps to capacity vector via monotone `D(n)`. | DOC_ONLY | PDF Annex W Test 6 | Add executable monotone `D(n)` checks from prealgebra model implementation. |
| W07 | 60 | Cross-axis isolation preserves geometric gluing. | DOC_ONLY | PDF Annex W Test 7 | Add explicit cross-axis perturbation simulations. |
| W08 | 61 | `N(C)` is monotone non-decreasing in `C`. | PARTIAL (`LOCAL_EXEC` + `DOC_ONLY`) | PDF Annex W Test 8; theorem validation monotonicity checks in `/Users/meganpastore/Clawdbot/Repos/capacity-demo/tests/test_theorem_validation.py` | Add direct `N(C)` computation regression per Annex W definition. |
| W09 | 61 | `ΔT(C)` is well-defined in semiclassical regime. | DOC_ONLY | PDF Annex W Test 9 | Add executable `ΔT` well-definedness proof obligations. |
| W10 | 61 | Observers do not generate classical structure. | DOC_ONLY | PDF Annex W Test 10; Appendix L theorem statement | Add independent model checking for observer non-influence assumptions. |
| W11 | 61 | CPMT operationalizes Annex T correctly. | DOC_ONLY | PDF Annex W Test 11; Annex U/T mapping text | Add executable conformance suite for CPMT invariants. |
| W12 | 61 | Observer triad maps consistently to substrate triad. | DOC_ONLY | PDF Annex W Test 12 | Add typed mapping tests (`Access/Selection/Commitment` ↔ `M/ω/U`). |
| W13 | 62 | `C_obs` decomposition is backward compatible. | DOC_ONLY | PDF Annex W Test 13 | Add schema compatibility tests and migration fixtures. |
| W14 | 62 | Ejection theorem expands classical core. | DOC_ONLY (formal set-theory argument) | PDF Annex W Test 14; theorem statement in Section 8.7 | Add finite-ensemble constructive simulation test. |
| W15 | 62 | Pointer-accuracy orthogonality is consistent. | DOC_ONLY | PDF Annex W Test 15 | Add memory model tests for `p(m)` vs `a(m)` independence. |
| W16 | 62 | Distinguishing `t_E` and `t_B` resolves circularity. | DOC_ONLY | PDF Annex W Test 16 | Add executable time-consistency constraints in simulation. |
| W17 | 62 | Local fixed points are compatible with global axiom/conjecture structure. | DOC_ONLY | PDF Annex W Test 17; uniqueness marked conjectural | Add constructive multi-fixed-point toy model. |
| W18 | 63 | Compression governance integrates without contradiction. | DOC_ONLY | PDF Annex W Test 18 | Add governance-policy replay tests over compression events. |
| W19 | 63 | `C_obs^meta` limit is acknowledged (not solved). | DOC_ONLY | PDF Annex W Test 19 | Keep as scope-bound; no pass/fail beyond consistency acknowledgment. |
| W20 | 63 | Non-negotiability principle self-application holds. | DOC_ONLY | PDF Annex W Test 20 | Add policy-enforced self-argument blocking tests in governance runtime. |

## B. Foundational and Scope Claims Outside Annex W

| ID | PDF Page(s) | Statement | Class | Current Status | Evidence |
|---|---:|---|---|---|---|
| F01 | 2 | Axiom: Triadic irreducibility. | OPEN_OR_SPEC | ASSUMED | Section `-1.2` text in PDF. |
| F02 | 4 | Axiom: Fixed-point selection (`φ = Ψ(φ)`). | OPEN_OR_SPEC | ASSUMED | Section `-1.7` text in PDF. |
| F03 | 4 | Theorem: Uniform fixed-point property (`φ*` fixed point). | DOC_ONLY | PROOF_SKETCH_ONLY | Section `-1.8` proof sketch in PDF. |
| F04 | 5 | Conjecture: uniqueness of fixed point under regularity. | OPEN_OR_SPEC | OPEN | Section `-1.8` conjecture text. |
| F05 | 19/39/61 | Observer non-influence theorem restated. | DOC_ONLY | INTERNAL_ONLY | Section 8.7 + Appendix L + Annex W Test 10. |
| F06 | 28/31 | Non-claim: no BH information recovery mechanism; no SM group derivation/uniqueness claim. | OPEN_OR_SPEC | SCOPE_GUARD | Sections 12.4 and 16.2 text. |
| F07 | 58 | Non-claim: framework does not posit new dynamics/DoF/fundamental laws. | OPEN_OR_SPEC | SCOPE_GUARD | Annex V text. |
| F08 | 58 | Non-claim: unresolved micro distinctions do not cease to exist. | OPEN_OR_SPEC | SCOPE_GUARD | Annex V text. |
| F09 | 58-59 | Non-claim: capacity boundaries are not automatically physical phase transitions. | OPEN_OR_SPEC | SCOPE_GUARD | Annex V/U.4 terminology note. |
| F10 | 58-59 | Negative-case claim: some systems have no stable capacity boundaries. | DOC_ONLY | INTERNAL_ONLY | Annex V/U.3 text. |
| F11 | 53 | Annex T/U memory toy is algorithmic, not biological phenomenology claim. | OPEN_OR_SPEC | SCOPE_GUARD | Annex T.12/U.1 text. |

## C. Current Executable Coverage Snapshot (What Is Actually Tested)

1. Objective-B executable regressions are currently passing in a fresh autonomous run:
   - `/tmp/openclaws/Repos/host-adapters-experimental-data/host-adapters/RUN_20260228_052622/results/science/campaign/campaign_index.csv`
   - `claim2_seed_perturbation=PASS`
   - `claim3_optionb_regime_check=PASS`
2. Selection ledger for same run marks Objective-B claim set as accepted:
   - `/tmp/openclaws/Repos/host-adapters-experimental-data/host-adapters/RUN_20260228_052622/results/selection/ledger.jsonl`
3. Independent theorem/unit checks in `capacity-demo` (prior physics audit logs):
   - `/tmp/openclaws/Repos/host-adapters-experimental-data/host-adapters/docs/state/physics_audit_logs/tier_a_test_framework_b.txt` (`13 passed`)
   - `/tmp/openclaws/Repos/host-adapters-experimental-data/host-adapters/docs/state/physics_audit_logs/tier_a_test_theorem_validation.txt` (`12 passed`)

## D. Bottom Line (Accuracy/Support, Strictly)

1. Full explicit PDF claims are now mapped (`20/20`) with citations and status.
2. The PDF’s internal consistency claims are documented as `DOC_ONLY` unless independently executed.
3. A subset has independent local executable support (`LOCAL_EXEC`), mainly monotonicity/regression-oriented items and Objective-B runs.
4. This map does **not** assert that every PDF claim is physically established in external reality; it separates internal consistency from independent executable support and open/speculative scope.
