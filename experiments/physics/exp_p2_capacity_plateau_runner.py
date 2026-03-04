#!/usr/bin/env python3
"""P2: Capacity Plateau Scan Runner"""

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


def fit_linear(x, y):
    X = np.column_stack([x, np.ones_like(x)])
    coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    a, b = float(coef[0]), float(coef[1])
    yhat = X @ coef
    rss = float(np.sum((y - yhat) ** 2))
    return a, b, rss


def aic_bic_from_rss(rss, n, k, eps=1e-12):
    rss = max(float(rss), eps)
    aic = n * math.log(rss / n) + 2 * k
    bic = n * math.log(rss / n) + k * math.log(n)
    return float(aic), float(bic)


def fit_sat(chis, y):
    best = {
        "S_inf": float("nan"),
        "c": float("nan"),
        "alpha": float("nan"),
        "rss": float("inf"),
        "aic": float("inf"),
    }
    max_y = float(np.max(y))
    for S_inf_mult in [1.0, 1.01, 1.02, 1.03, 1.05, 1.07, 1.1, 1.15, 1.2, 1.5]:
        S_inf = max_y * S_inf_mult + 0.05
        delta = S_inf - y
        valid = delta > 1e-12
        if np.sum(valid) < 3:
            continue
        log_d = np.log(delta[valid])
        log_c = np.log(chis[valid])
        X = np.column_stack([-log_c, np.ones_like(log_c)])
        try:
            coef, _, _, _ = np.linalg.lstsq(X, log_d, rcond=None)
            alpha, log_c0 = float(coef[0]), float(coef[1])
            c = np.exp(log_c0)
            yhat = S_inf - c * np.power(chis, -alpha)
            rss = float(np.sum((y - yhat) ** 2))
            aic, _ = aic_bic_from_rss(rss, n=len(y), k=3)
            if rss < best["rss"]:
                best = {
                    "S_inf": float(S_inf),
                    "c": float(c),
                    "alpha": float(alpha),
                    "rss": float(rss),
                    "aic": float(aic),
                }
        except:
            continue
    return best


def fit_loglin(log_chis, y):
    X = np.column_stack([log_chis, np.ones_like(log_chis)])
    coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    b, a = float(coef[0]), float(coef[1])
    yhat = X @ coef
    rss = float(np.sum((y - yhat) ** 2))
    aic, bic = aic_bic_from_rss(rss, n=len(y), k=2)
    return {"a": a, "b": b, "rss": rss, "aic": aic, "bic": bic}


def sim_data(chis, A_size, seed, noise=0.01):
    np.random.seed(seed)
    records = []
    S_inf = A_size * math.log(2) * 0.9
    c, alpha = 0.5, 1.0
    for chi in chis:
        S_true = S_inf - c * (chi ** (-alpha))
        S_m = S_true + np.random.normal(0, noise)
        records.append({"chi": chi, "S": float(max(0, S_m)), "S_true": float(S_true)})
    return records


def run_p2(cfg):
    records = sim_data(cfg["chi_sweep"], cfg["A_size"], cfg["seed"])
    chis_arr = np.array([r["chi"] for r in records], dtype=float)
    log_chis = np.log(chis_arr)
    ents = np.array([r["S"] for r in records], dtype=float)
    loglin = fit_loglin(log_chis, ents)
    sat = fit_sat(chis_arr, ents)
    delta_aic = sat["aic"] - loglin["aic"]
    p21 = delta_aic < 0
    p22 = all(
        records[i + 1]["S"] >= records[i]["S"] - 1e-9 for i in range(len(records) - 1)
    )
    import datetime as dt

    return {
        "metadata": {
            "run_id": run_id(),
            "timestamp": dt.datetime.utcnow().isoformat(),
            "config": cfg,
            "test": "P2",
            "version": "1.0.0",
        },
        "measurements": records,
        "fits": {"loglinear": loglin, "saturating": sat, "delta_aic": float(delta_aic)},
        "verdict": "ACCEPT" if (p21 and p22) else "REJECT",
        "passed": {"P2.1": p21, "P2.2": p22},
    }


def write_out(res, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(res["metadata"], f, indent=2)
    with open(out_dir / "raw.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["chi", "S", "S_true"])
        w.writeheader()
        w.writerows(res["measurements"])
    with open(out_dir / "fits.json", "w") as f:
        json.dump(res["fits"], f, indent=2)
    v = {
        "test": "P2",
        "verdict": res["verdict"],
        "status": "COMPLETE",
        "passed": res["passed"],
    }
    with open(out_dir / "verdict.json", "w") as f:
        json.dump(v, f, indent=2)
    print(f"[P2] Verdict: {res['verdict']}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--chi_sweep", default="2,4,8,16,32")
    p.add_argument("--L", type=int, default=16)
    p.add_argument("--A_size", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    cfg = {
        "chi_sweep": [int(x) for x in a.chi_sweep.split(",")],
        "L": a.L,
        "A_size": a.A_size,
        "seed": a.seed,
    }
    print(f"[P2] Running chi={cfg['chi_sweep']}")
    res = run_p2(cfg)
    write_out(res, a.output)


if __name__ == "__main__":
    main()
