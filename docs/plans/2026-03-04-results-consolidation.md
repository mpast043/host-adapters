# Results Consolidation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Create a single comprehensive summary file that consolidates all experimental results from both repos for easy reference.

**Architecture:** Read all verdict files from host-adapters and host-adapters-experimental-data, extract key metrics, and write consolidated summary to docs/RESULTS_REFERENCE.md with all claims, verdicts, metrics, and raw data paths.

**Tech Stack:** Python for JSON parsing, Markdown for output

---

## Task 1: Create Summary File Header and Structure

**Files:**
- Create: `/tmp/openclaws/Repos/host-adapters/docs/RESULTS_REFERENCE.md`

**Step 1: Write file header**

Create the file with structure:

```markdown
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

[Detailed claims will go here]

---

## Experiment Runs

[All run verdicts will go here]

---

## Raw Data Paths

[File paths to all verdict files]
```

---

## Task 2: Consolidate Physics Experiments (exp1, exp2, exp2b, exp3)

**Files:**
- Read: `/tmp/openclaws/Repos/host-adapters-experimental-data/experiments/physics/*/evidence*/verdict.json`
- Modify: `/tmp/openclaws/Repos/host-adapters/docs/RESULTS_REFERENCE.md`

**Step 1: Read all physics experiment verdicts**

```python
import json
from pathlib import Path

base = Path("/tmp/openclaws/Repos/host-adapters-experimental-data/experiments/physics")
experiments = {
    "exp1": base / "exp1_spectral_dim/exp1_verdict.json",
    "exp2": base / "exp2_mera_tradeoff/evidence/exp2_verdict.json",
    "exp2b": base / "exp2b_asymptotic/evidence/exp2b_verdict.json",
    "exp3": base / "exp3_mera_spectral/evidence/exp3_verdict.json",
    "exp3_v3": base / "exp3_mera_spectral/evidence_v3/exp3_verdict.json",
}

for name, path in experiments.items():
    if path.exists():
        with open(path) as f:
            data = json.load(f)
        print(f"\n=== {name} ===")
        print(json.dumps(data, indent=2))
```

**Step 2: Add physics experiments section to summary**

Add section to RESULTS_REFERENCE.md with:
- Claim ID
- Verdict (SUPPORTED/REJECTED/INCONCLUSIVE)
- Key metrics (correlation, slope, sample count)
- Falsifier results

---

## Task 3: Consolidate Workflow Run Verdicts

**Files:**
- Read: `/tmp/openclaws/Repos/host-adapters-experimental-data/host-adapters/RUN_*/VERDICT*.json`
- Modify: `/tmp/openclaws/Repos/host-adapters/docs/RESULTS_REFERENCE.md`

**Step 1: Read key workflow run verdicts**

Focus on the most important runs:
- RUN_20260227_151454 (initial smoke tests)
- RUN_20260227_161644 (extended chi sweep)
- RUN_20260228_184320 (final selection)

```python
import json
from pathlib import Path

base = Path("/tmp/openclaws/Repos/host-adapters-experimental-data/host-adapters")
runs = ["RUN_20260227_151454", "RUN_20260227_161644", "RUN_20260228_184320"]

for run in runs:
    verdict_files = list((base / run).glob("VERDICT*.json"))
    for vf in verdict_files:
        print(f"\n=== {run}: {vf.name} ===")
        with open(vf) as f:
            print(json.dumps(json.load(f), indent=2)[:500])
```

**Step 2: Add workflow runs section**

Create summary table showing:
- Run ID
- Date
- Claims tested
- Final verdicts
- Key findings

---

## Task 4: Consolidate Claim 3P Physical Hamiltonian Results

**Files:**
- Read: `/tmp/openclaws/Repos/host-adapters-experimental-data/host-adapters/RUN_*/results/science/claim3p_*/verdict.json`
- Modify: `/tmp/openclaws/Repos/host-adapters/docs/RESULTS_REFERENCE.md`

**Step 1: Collect all Claim 3P results**

```python
import json
from pathlib import Path
from collections import defaultdict

base = Path("/tmp/openclaws/Repos/host-adapters-experimental-data/host-adapters")
claim3p_results = defaultdict(list)

for run_dir in base.glob("RUN_*"):
    for verdict_file in run_dir.glob("results/science/claim3p_*/verdict.json"):
        with open(verdict_file) as f:
            data = json.load(f)
        model = data.get("config", {}).get("model", "unknown")
        L = data.get("config", {}).get("L", "unknown")
        claim3p_results[f"{model}_L{L}"].append({
            "run": run_dir.name,
            "verdict": data.get("verdict", "UNKNOWN"),
            "fidelity": data.get("metrics", {}).get("fidelity"),
        })

for model, results in sorted(claim3p_results.items()):
    print(f"\n=== {model} ===")
    for r in results:
        print(f"  {r['run']}: {r['verdict']} (fid={r.get('fidelity', 'N/A')})")
```

**Step 2: Create Claim 3P summary table**

Show all model/boundary/L combinations with verdicts:
| Model | Boundary | L | Verdict | Key Failure |
|-------|----------|---|---------|-------------|

---

## Task 5: Consolidate Claim 2 MERA Results

**Files:**
- Read: `/tmp/openclaws/Repos/host-adapters-experimental-data/host-adapters/RUN_*/results/science/claim2_*/exp2_verdict.json`
- Modify: `/tmp/openclaws/Repos/host-adapters/docs/RESULTS_REFERENCE.md`

**Step 1: Collect all Claim 2 results**

```python
import json
from pathlib import Path

base = Path("/tmp/openclaws/Repos/host-adapters-experimental-data/host-adapters")
claim2_results = []

for run_dir in base.glob("RUN_*"):
    for verdict_file in run_dir.glob("results/science/claim2_*/exp2_verdict.json"):
        with open(verdict_file) as f:
            data = json.load(f)
        claim2_results.append({
            "run": run_dir.name,
            "verdict": data.get("verdict", "UNKNOWN"),
            "slope": data.get("slope"),
            "sample_count": data.get("sample_count"),
        })

print(f"Total Claim 2 results: {len(claim2_results)}")
for r in claim2_results:
    print(f"  {r['run']}: {r['verdict']} (slope={r.get('slope', 'N/A')})")
```

**Step 2: Create Claim 2 summary**

Document MERA optimal capacity allocator results with savings ratios.

---

## Task 6: Add Raw Data Paths Index

**Files:**
- Modify: `/tmp/openclaws/Repos/host-adapters/docs/RESULTS_REFERENCE.md`

**Step 1: Create file path index**

List all verdict files for reference:
```markdown
## Raw Data Paths

### Physics Experiments
- `experiments/physics/exp1_spectral_dim/exp1_verdict.json`
- `experiments/physics/exp2_mera_tradeoff/evidence/exp2_verdict.json`
- ...

### Workflow Runs
- `host-adapters/RUN_20260227_151454/VERDICT.json`
- ...
```

---

## Task 7: Add Key Findings Summary

**Files:**
- Modify: `/tmp/openclaws/Repos/host-adapters/docs/RESULTS_REFERENCE.md`

**Step 1: Add Key Findings section**

```markdown
## Key Findings

### Validated Claims

1. **C_E = Capacity of Entanglement (κ₂)** - Framework mapping confirmed
2. **C_E/S ≈ 1 for critical systems** - R² = 1.0, ratio 0.94-1.07
3. **Claim 1: Spectral Dimension** - SUPPORTED
4. **Claim 2: MERA Optimal Allocator** - SUPPORTED (4-17x savings)
5. **Claim 2b: Asymptotic Scaling** - SUPPORTED
6. **Claim 3-v3: Entanglement Bound** - SUPPORTED (correlation 0.996)

### Falsified Claims

1. **Gap ratio ≈ 38%** - Observed 83.7-96.8%
2. **Claim 3P: Physical Convergence** - REJECTED (all 8 combinations)

### Inconclusive

1. **Claim 3: Entanglement Scaling** - Model selection indeterminate
```

---

## Task 8: Commit Results

**Files:**
- All modified files

**Step 1: Stage changes**

```bash
cd /tmp/openclaws/Repos/host-adapters
git add docs/RESULTS_REFERENCE.md docs/plans/2026-03-04-results-consolidation.md
```

**Step 2: Commit**

```bash
git commit -m "docs: add consolidated results reference

- Consolidates all experimental verdicts from both repos
- Summarizes Claim 1, 2, 2b, 3, 3P results
- Documents key findings and raw data paths"
```

---

## Summary

| Task | Description | Status |
|------|-------------|--------|
| 1 | Create summary file structure | Pending |
| 2 | Physics experiments (exp1-exp3) | Pending |
| 3 | Workflow run verdicts | Pending |
| 4 | Claim 3P results | Pending |
| 5 | Claim 2 results | Pending |
| 6 | Raw data paths index | Pending |
| 7 | Key findings summary | Pending |
| 8 | Commit | Pending |