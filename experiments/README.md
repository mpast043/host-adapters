# Claim 3 Experiments

FRAMEWORK_SPEC_v0.2 compliant experiment runners for Claim 3: MERA entanglement scaling with bond dimension χ.

## Runners

| File | Mode | Purpose | Dependencies |
|------|------|---------|--------------|
| `exp3_claim3_quimb_runner.py` | Option A | Full MERA tensor network via quimb | quimb, cotengra, torch |
| `exp3_claim3_optionB_runner.py` | Option A/B | Simplified entropy generator + Falsifier 3.5 | numpy only |

### Choosing a Runner

**Use `quimb_runner` when:**
- You need physically accurate MERA entanglement entropy
- You want to test actual tensor network contraction
- You have time for slower execution

**Use `optionB_runner` when:**
- You need fast iteration
- You're testing partition variation (Falsifier 3.5)
- You want deterministic synthetic data

## Usage

```bash
# Install dependencies
pip install -r requirements-exp3.txt

# Run quimb-based experiment (Option A)
python experiments/claim3/exp3_claim3_quimb_runner.py \
  --L 16 \
  --A_sites 8 \
  --chi_sweep 2,4,8,16,32 \
  --seeds_per_chi 7 \
  --state identity

# Run optionB experiment (includes partition variation)
python experiments/claim3/exp3_claim3_optionB_runner.py \
  --num_sites 64 \
  --subsystem_sizes 32,16,8,4 \
  --chi_sweep 2,4,6,8,12,16,24,32 \
  --seeds_per_chi 7
```

## Outputs

All runners write to: `outputs/<experiment_name>/<run_id>/`

| Artifact | Description |
|----------|-------------|
| `manifest.json` | Run metadata, versions, scope |
| `raw_entropy.csv` | Raw S(χ, seed, partition) data |
| `fits.json` | Model parameters, AIC/BIC |
| `bound_checks.json` | K_cut, bound, margin per point |
| `verdict.json` | FINAL: SUPPORTED / INCONCLUSIVE / REJECTED |

## Falsifier Coverage

| Falsifier | quimb_runner | optionB_runner |
|-----------|--------------|----------------|
| 3.1 Monotonicity | ✅ | ✅ |
| 3.2 Replicate robustness (CV≤10%) | ✅ | ✅ |
| 3.3 Model selection (ΔAIC/BIC≥10) | ✅ | ✅ |
| 3.4 Bound validity (S≤K·logχ) | ✅ | ✅ |
| 3.5 Cut-size bridge | ❌ (Option A) | ✅ (Option B) |

## K_cut Proxy Rule

Both runners use: `K_cut = 2 * (ceil(log2(A_size)) + 1)`, min 2

This is an upper proxy for 1D binary MERA interval cut.

## Spec Compliance

- Framework spec: v0.2
- Deterministic seed policy: `seed = seed_base + chi_index*1000 + rep`
- CPU runnable: Yes (quimb_runner may benefit from GPU for torch)
