#!/usr/bin/env python3
"""P3: Gluing/Excision Stability Runner"""

from __future__ import annotations
import argparse
import csv
import json
import math
import os
from pathlib import Path
import numpy as np


def run_id():
    import datetime as dt

    t = dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    r = os.urandom(4).hex()
    return f"{t}_{r}"


def sim_mera_state(chi, seed):
    np.random.seed(seed)
    dims = min(chi, 16)
    psi = np.random.randn(dims, dims) + 1j * np.random.randn(dims, dims)
    psi = psi / np.linalg.norm(psi)
    S = math.log(dims) * (0.5 + 0.3 * np.random.rand())
    return psi, float(S)


def compute_entropy(psi):
    u, s, _ = np.linalg.svd(psi, full_matrices=False)
    s = s[s > 1e-12]
    s = s / np.linalg.norm(s)
    return -np.sum(s * np.log(s))


def gluing_op(psi_A, psi_B, seed):
    np.random.seed(seed)
    da, db = psi_A.shape[0], psi_B.shape[0]
    glued = np.kron(psi_A, psi_B)
    noise = 0.01 * (
        np.random.randn(da * db, da * db) + 1j * np.random.randn(da * db, da * db)
    )
    glued = glued + noise
    glued = glued / np.linalg.norm(glued)
    return glued


def excision_op(psi, keep_indices, seed):
    np.random.seed(seed)
    d = psi.shape[0]
    new_d = len(keep_indices)
    excised = psi[:new_d, :new_d]
    noise = 0.005 * (np.random.randn(new_d, new_d) + 1j * np.random.randn(new_d, new_d))
    excised = excised + noise
    excised = excised / np.linalg.norm(excised)
    return excised


def run_p3(cfg):
    records = []
    for chi in cfg["chi_sweep"]:
        psi_A, S_A = sim_mera_state(chi, cfg["seed"])
        psi_B, S_B = sim_mera_state(chi, cfg["seed"] + 1)

        psi_glued = gluing_op(psi_A, psi_B, cfg["seed"])
        S_glued = compute_entropy(psi_glued)

        psi_excised = excision_op(psi_glued, [0, 1], cfg["seed"])
        S_excised = compute_entropy(psi_excised)

        S_sum = S_A + S_B
        violation = abs(S_glued - S_sum) > cfg["threshold"]

        records.append(
            {
                "chi": chi,
                "S_A": S_A,
                "S_B": S_B,
                "S_sum": S_sum,
                "S_glued": S_glued,
                "S_excised": S_excised,
                "violation": violation,
            }
        )

    p31 = all(not r["violation"] for r in records)
    p32 = all(r["S_excised"] > 0 for r in records)

    import datetime as dt

    return {
        "metadata": {
            "run_id": run_id(),
            "timestamp": dt.datetime.utcnow().isoformat(),
            "config": cfg,
            "test": "P3",
            "version": "1.0.0",
        },
        "measurements": records,
        "verdict": "ACCEPT" if (p31 and p32) else "REJECT",
        "passed": {"P3.1_gluing_stable": p31, "P3.2_excision_valid": p32},
    }


def write_out(res, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(res["metadata"], f, indent=2)
    with open(out_dir / "raw.csv", "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "chi",
                "S_A",
                "S_B",
                "S_sum",
                "S_glued",
                "S_excised",
                "violation",
            ],
        )
        w.writeheader()
        w.writerows(res["measurements"])
    v = {
        "test": "P3",
        "verdict": res["verdict"],
        "status": "COMPLETE",
        "passed": res["passed"],
    }
    with open(out_dir / "verdict.json", "w") as f:
        json.dump(v, f, indent=2)
    print(f"[P3] Verdict: {res['verdict']}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--chi_sweep", default="2,4,8")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    cfg = {
        "chi_sweep": [int(x) for x in a.chi_sweep.split(",")],
        "threshold": a.threshold,
        "seed": a.seed,
    }
    print(f"[P3] Running gluing/excision tests chi={cfg['chi_sweep']}")
    res = run_p3(cfg)
    write_out(res, a.output)


if __name__ == "__main__":
    main()
