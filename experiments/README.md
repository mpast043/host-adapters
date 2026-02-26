# Experiments: MERA Capacity-Governed Systems

This directory contains experimental validation code for tensor network-based capacity allocation systems.

## Claims Overview

| Claim | Description | Status | Key Script |
|-------|-------------|--------|------------|
| 3A | Entanglement-max saturation | SUPPORTED | `exp3_claim3_entanglement_max_mincut_runner.py` |
| 3B | Windowed regime detection | **REJECTED** | `exp3_claim3_entanglement_max_mincut_runner.py` |
| 3P | Physical Hamiltonian convergence | **PARTIAL** | `exp3_claim3_physical_convergence_runner_v2.py` |

## Quick Start

### Claim 3P: Physical Hamiltonian Convergence Test

Run MERA variational optimization against Exact Diagonalization ground states:

```bash
# Activate environment
source .venv/bin/activate

# Run Ising L=8
python experiments/claim3/exp3_claim3_physical_convergence_runner_v2.py \
    --model ising_open \
    --L 8 \
    --A 4 \
    --chi 2 3 4 6 8 12 16 \
    --steps 120 \
    --restarts 3

# Run Heisenberg L=8
python experiments/claim3/exp3_claim3_physical_convergence_runner_v2.py \
    --model heisenberg_open \
    --L 8 \
    --A 4 \
    --chi 2 3 4 6 8 12 16 \
    --steps 150 \
    --restarts 3

# Run Ising L=16 (requires sparse ED)
python experiments/claim3/exp3_claim3_physical_convergence_runner_v2.py \
    --model ising_open \
    --L 16 \
    --A 8 \
    --chi 2 3 4 6 8 12 16 \
    --steps 120 \
    --restarts 3
```

### Claim 3A/3B: Entanglement-max Min-cut

```bash
python experiments/claim3/exp3_claim3_entanglement_max_mincut_runner.py \
    --L 16 --chi 2 4 6 8
```

## Important Scripts

### exp3_claim3_physical_convergence_runner_v2.py

**Purpose:** MERA optimization with energy-based local terms (not fidelity-based)

**Key Features:**
- Sparse ED for L=16 (65K×65K Hilbert space feasible)
- Energy optimization with `TNOptimizer` + L-BFGS-B
- Schmidt decomposition for entropy (never materialize full ρ)
- Multiple restarts per χ to avoid local minima

**Arguments:**
| Flag | Description | Default |
|------|-------------|---------|
| `--model` | Model type: `ising_open`, `heisenberg_open` | (required) |
| `--L` | System size (must be power of 2) | 8 |
| `--A` | Partition size | L/2 |
| `--chi` | Bond dimensions to sweep | 2 4 8 16 |
| `--steps` | Optimization steps per restart | 120 |
| `--restarts` | Number of random restarts per χ | 3 |
| `--seed` | Base random seed | 42 |
| `--output` | Output directory | `outputs/claim3P_final/` |

**Outputs:**
```
outputs/claim3P_final/{model}_L{L}/{timestamp}_{hash}/
├── manifest.json      # Run parameters and verdict
├── metrics.json       # Best results per χ
├── raw_results.csv    # All restart data
├── verdict.json       # P3.1–P3.4 criterion evaluation
└── plots/             # Convergence plots (if generated)
```

### exp3_claim3_entanglement_max_mincut_runner.py

**Purpose:** Min-cut capacity analysis for entanglement-max states

**Models:** `entanglement_max`, `area_law`, `volume_law`

## Technical Notes

### Hamiltonian Convention (CRITICAL)

⚠️ **Bug discovered 2026-02-25:** Two different ED implementations exist:

| Commit | ED Method | Ising L=8 E₀ |
|--------|-----------|--------------|
| `ac4432a` | Dense manual builder | -9.84 (WRONG) |
| `9720dfa` | Sparse quimb | -4.22 (CORRECT) |

**Always use commit `9720dfa` or later** for consistent results.

### MERA Constraints

- System size `L` must be **power of 2** (8, 16, 32, ...)
- Bond dimension `χ` controls expressivity vs computational cost
- Optimization uses `isometrize(method="exp")` norm constraint

### Sparse ED for L=16

For L=16 (65,536-dimensional Hilbert space):

```python
# Dense: ~34GB RAM, OOM killed
H_dense = np.linalg.eigh(H)  # DON'T DO THIS

# Sparse: ~500MB RAM, seconds
H_sparse = qu.ham_ising(L, jz=1.0, bx=1.0, sparse=True)
E0 = scipy.sparse.eigsh(H_sparse, k=1, which='SA')[0]
```

Schmidt decomposition for entropy:
```python
# Never materialize full ρ
psi_mat = psi0.reshape(2**A_size, 2**(L-A_size))
U, s, Vh = np.linalg.svd(psi_mat, full_matrices=False)
probs = s**2
S = -np.sum(probs * np.log(probs))
```

## Results Summary

See `CONSOLIDATED_REPORT_CLAIM3P.md` for full analysis.

**Key Findings:**
- L=8: MERA achieves perfect fidelity (1.0) but model selection favors log-linear
- L=16 Ising: MERA insufficient at χ≤16 for critical systems (fidelity only 0.072)
- Energy optimization converges even when fidelity remains poor

## Citation

If using these experiments, cite:
```
Capacity-Governed Systems Framework v0.2.1
Claim 3P: Physical Hamiltonian Convergence Test
Repository: github.com/mpast043/host-adapters
```
