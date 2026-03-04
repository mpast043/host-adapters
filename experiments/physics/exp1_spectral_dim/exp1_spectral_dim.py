import json
import numpy as np
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import sys

SIERPINSKI_D_S = 2 * np.log(3) / np.log(5)


def sierpinski_gasket_edges(level: int) -> Tuple[int, List[Tuple[int, int]]]:
    if level < 0:
        raise ValueError("level must be >= 0")
    if level == 0:
        return 3, [(0, 1), (1, 2), (0, 2)]

    N_prev, edges_prev = sierpinski_gasket_edges(level - 1)

    def offset_edges(edges: List[Tuple[int, int]], offset: int) -> List[Tuple[int, int]]:
        return [(u + offset, v + offset) for (u, v) in edges]

    off_T = 0
    off_L = N_prev
    off_R = 2 * N_prev

    edges = []
    edges += offset_edges(edges_prev, off_T)
    edges += offset_edges(edges_prev, off_L)
    edges += offset_edges(edges_prev, off_R)

    parent = list(range(3 * N_prev))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    A_t, B_t, C_t = 0, 1, 2
    T_B = off_T + B_t
    T_C = off_T + C_t
    L_A = off_L + A_t
    L_C = off_L + C_t
    R_A = off_R + A_t
    R_B = off_R + B_t

    union(T_B, L_A)
    union(T_C, R_A)
    union(L_C, R_B)

    rep_to_new: Dict[int, int] = {}
    new_id = 0

    def get_new(x: int) -> int:
        nonlocal new_id
        r = find(x)
        if r not in rep_to_new:
            rep_to_new[r] = new_id
            new_id += 1
        return rep_to_new[r]

    edge_set = set()
    for (u, v) in edges:
        nu, nv = get_new(u), get_new(v)
        if nu == nv:
            continue
        a, b = (nu, nv) if nu < nv else (nv, nu)
        edge_set.add((a, b))

    return new_id, sorted(edge_set)


def adjacency_from_edges(N: int, edges: List[Tuple[int, int]]) -> np.ndarray:
    A = np.zeros((N, N), dtype=int)
    for u, v in edges:
        A[u, v] = 1
        A[v, u] = 1
    return A


def compute_laplacian(adjacency: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    degrees = np.sum(adjacency, axis=1)
    L = np.diag(degrees) - adjacency
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    idx = np.argsort(eigenvalues)
    return eigenvalues[idx], eigenvectors[:, idx]


def capacity_limited_return_prob(eigenvalues: np.ndarray, tau: float, C_obs: int) -> float:
    N = len(eigenvalues)
    C = min(C_obs, N)
    return float(np.mean(np.exp(-tau * eigenvalues[:C])))


def full_return_prob(eigenvalues: np.ndarray, tau: float) -> float:
    return float(np.mean(np.exp(-tau * eigenvalues)))


def choose_tau_cut(
    taus: np.ndarray,
    P_vals: np.ndarray,
    plateau: float,
    p_cut_floor: float = 0.25,
    p_cut_cap: float = 0.95,
    plateau_mult: float = 3.0,
) -> Tuple[Optional[float], float]:
    taus = np.asarray(taus, dtype=float)
    P_vals = np.asarray(P_vals, dtype=float)
    p_cut = max(p_cut_floor, min(p_cut_cap, plateau * plateau_mult))
    idxs = np.where(P_vals < p_cut)[0]
    if len(idxs) == 0:
        return None, float(p_cut)
    return float(taus[idxs[0]]), float(p_cut)


def local_beta_scan(
    taus: np.ndarray,
    P_vals: np.ndarray,
    min_window: int = 10,
    max_window: int = 24,
    plateau: Optional[float] = None,
    plateau_margin: float = 0.08,
    p_high: float = 0.98,
    beta_min: float = 0.55,
    beta_max: float = 1.2,
    p_cut_floor: float = 0.25,
) -> Tuple[float, Dict]:
    taus = np.asarray(taus, dtype=float)
    P_vals = np.asarray(P_vals, dtype=float)

    tau_cut = None
    p_cut = None
    if plateau is not None and plateau > 0:
        tau_cut, p_cut = choose_tau_cut(taus, P_vals, plateau, p_cut_floor=p_cut_floor)

    mask = (taus > 0) & (P_vals > 0) & (P_vals < p_high)
    if plateau is not None and plateau > 0:
        mask = mask & (P_vals > plateau * (1.0 + plateau_margin))
    if tau_cut is not None:
        mask = mask & (taus <= tau_cut)

    taus = taus[mask]
    P_vals = P_vals[mask]

    if len(taus) < max(12, min_window):
        return float("nan"), {
            "reason": "insufficient_points_after_mask",
            "points": int(len(taus)),
            "tau_cut": tau_cut,
            "p_cut": p_cut,
            "plateau": float(plateau) if plateau is not None else None,
        }

    log_t = np.log(taus)
    log_p = np.log(P_vals)

    n = len(taus)
    best = None

    max_w = min(max_window, n)
    for w in range(min_window, max_w + 1):
        for start in range(0, n - w + 1):
            end = start + w
            x = log_t[start:end]
            y = log_p[start:end]
            m, a = np.polyfit(x, y, 1)
            beta = float(-m)
            if not np.isfinite(beta):
                continue
            if beta < beta_min or beta > beta_max:
                continue
            y_hat = a + m * x
            mse = float(np.mean((y - y_hat) ** 2))
            cand = (mse, -w, start, beta)
            if best is None or cand < best:
                best = cand

    if best is None:
        return float("nan"), {
            "reason": "no_fit_after_constraints",
            "points": int(n),
            "tau_cut": tau_cut,
            "p_cut": p_cut,
            "plateau": float(plateau) if plateau is not None else None,
        }

    mse, neg_w, start, beta = best
    w = -neg_w
    diag = {
        "mse": float(mse),
        "window_size": int(w),
        "window_start_idx": int(start),
        "window_end_idx": int(start + w),
        "tau_min": float(taus[start]),
        "tau_max": float(taus[start + w - 1]),
        "points_used": int(n),
        "tau_cut": tau_cut,
        "p_cut": p_cut,
        "plateau": float(plateau) if plateau is not None else None,
    }
    return float(beta), diag


def effective_dimension(taus: np.ndarray, P_vals: np.ndarray) -> np.ndarray:
    taus = np.asarray(taus, dtype=float)
    P_vals = np.asarray(P_vals, dtype=float)
    mask = (taus > 0) & (P_vals > 0)
    taus = taus[mask]
    P_vals = P_vals[mask]
    log_t = np.log(taus)
    log_p = np.log(P_vals)
    dlogp = np.gradient(log_p, log_t)
    return -2.0 * dlogp


def compute_critical_time(eigenvalues: np.ndarray, C_obs: int) -> float:
    N = len(eigenvalues)
    C = min(C_obs, N)
    idx = min(C, N - 1)
    lam = float(eigenvalues[idx])
    if lam <= 0:
        return float("inf")
    return 1.0 / lam


def default_capacities(N: int) -> List[int]:
    base = [3, 5, 8, 13, 21]
    fracs = [0.1, 0.2, 0.35, 0.5, 0.7, 0.85, 1.0]
    caps = base + [max(3, int(frac * N)) for frac in fracs]
    caps = sorted(set([c for c in caps if 2 <= c <= N]))
    return caps


def run_experiment(
    seed: int,
    levels: List[int],
    capacities: Optional[List[int]] = None,
    n_tau_points: int = 90,
) -> List[Dict]:
    np.random.seed(seed)
    results: List[Dict] = []

    for level in levels:
        N, edges = sierpinski_gasket_edges(level)
        adj = adjacency_from_edges(N, edges)
        eigenvalues, _ = compute_laplacian(adj)

        lam_max = float(eigenvalues[-1]) if eigenvalues[-1] > 0 else 1.0
        taus = np.logspace(-3, np.log10(5e3 / lam_max), n_tau_points)

        caps = capacities if capacities is not None else default_capacities(N)

        for C_obs in caps:
            if C_obs < 2 or C_obs > N:
                continue

            P_full_vals = np.array([full_return_prob(eigenvalues, t) for t in taus], dtype=float)
            P_C_vals = np.array([capacity_limited_return_prob(eigenvalues, t, C_obs) for t in taus], dtype=float)

            beta_full, beta_full_diag = local_beta_scan(taus, P_full_vals, plateau=1.0 / float(N))
            beta_C, beta_C_diag = local_beta_scan(taus, P_C_vals, plateau=1.0 / float(C_obs))

            d_eff_C = effective_dimension(taus, P_C_vals)
            k0 = int(0.75 * len(taus))
            tail = d_eff_C[k0:]
            plateau_std = float(np.std(tail))
            has_plateau = plateau_std < 0.05

            tau_crit = compute_critical_time(eigenvalues, C_obs)

            expected_beta = float(SIERPINSKI_D_S / 2.0)
            beta_diff = float(abs(beta_C - expected_beta)) if np.isfinite(beta_C) else float("inf")

            results.append(
                {
                    "level": int(level),
                    "N": int(N),
                    "C_obs": int(C_obs),
                    "C_ratio": float(C_obs / N),
                    "taus": taus.tolist(),
                    "P_full": P_full_vals.tolist(),
                    "P_C": P_C_vals.tolist(),
                    "beta_full": float(beta_full) if np.isfinite(beta_full) else None,
                    "beta_C": float(beta_C) if np.isfinite(beta_C) else None,
                    "beta_full_fit": beta_full_diag,
                    "beta_C_fit": beta_C_diag,
                    "tau_crit": float(tau_crit),
                    "has_plateau": bool(has_plateau),
                    "plateau_std": float(plateau_std),
                    "expected_d_s": float(SIERPINSKI_D_S),
                    "expected_beta": expected_beta,
                    "beta_diff_from_expected": beta_diff,
                }
            )

    return results


def test_falsifier_1(results: List[Dict], rel_tolerance: float = 0.25) -> Tuple[bool, str]:
    expected = float(SIERPINSKI_D_S / 2.0)
    checked = 0
    failures = []

    for r in results:
        if r["C_ratio"] < 0.7:
            continue
        checked += 1
        if r["beta_C"] is None:
            reason = ""
            if isinstance(r.get("beta_C_fit"), dict):
                reason = r["beta_C_fit"].get("reason", "")
            failures.append(f"L{r['level']} C={r['C_obs']}: beta_C=None {reason}")
            continue
        err = abs(r["beta_C"] - expected) / expected
        if err > rel_tolerance:
            failures.append(
                f"L{r['level']} C={r['C_obs']} (C/N={r['C_ratio']:.2f}): beta={r['beta_C']:.3f} err={err:.2f}"
            )

    if checked == 0:
        return False, "No configs met C_ratio>=0.7 to evaluate falsifier"
    if len(failures) == 0:
        return True, f"All {checked} configs within {rel_tolerance*100:.0f}% of expected β={expected:.3f}"
    return False, f"{len(failures)}/{checked} failures (sample): {failures[:3]}"


def test_falsifier_2(results: List[Dict], min_fraction_with_plateau: float = 0.2) -> Tuple[bool, str]:
    high_cap = [r for r in results if r["C_ratio"] >= 0.7]
    if not high_cap:
        return False, "No configs met C_ratio>=0.7 for plateau evaluation"

    plateau_count = sum(1 for r in high_cap if r["has_plateau"])
    total = len(high_cap)

    if plateau_count >= total * min_fraction_with_plateau:
        return True, f"{plateau_count}/{total} high-cap configs show plateau-like d_eff tail"
    return False, f"Only {plateau_count}/{total} high-cap configs show plateau-like d_eff tail"


def evaluate_claim_1(results: List[Dict], seed: int) -> Dict:
    passed_1, msg_1 = test_falsifier_1(results)
    passed_2, msg_2 = test_falsifier_2(results)

    overall = "SUPPORTED" if (passed_1 and passed_2) else "REJECTED"

    return {
        "claim_id": "claim-001",
        "claim_title": "Spectral Dimension as Capacity-Limited Effective Geometry",
        "verdict": overall,
        "falsifier_results": {
            "falsifier_1.1": {"passed": bool(passed_1), "message": msg_1},
            "falsifier_1.2": {"passed": bool(passed_2), "message": msg_2},
        },
        "sample_count": int(len(results)),
        "seed": int(seed),
    }


def save_results(results: List[Dict], verdict: Dict, output_dir: Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "exp1_data.json", "w") as f:
        json.dump(results, f, indent=2)

    with open(output_dir / "exp1_verdict.json", "w") as f:
        json.dump(verdict, f, indent=2)

    try:
        import matplotlib.pyplot as plt

        by_level: Dict[int, List[Dict]] = {}
        for r in results:
            by_level.setdefault(r["level"], []).append(r)

        reps = []
        for lvl, arr in by_level.items():
            arr_sorted = sorted(arr, key=lambda x: x["C_ratio"], reverse=True)
            reps.append(arr_sorted[0])

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        ax = axes[0]
        for r in reps:
            label = f"L{r['level']} N={r['N']} C={r['C_obs']} (C/N={r['C_ratio']:.2f})"
            ax.loglog(r["taus"], r["P_C"], label=label)
        ax.set_xlabel("τ")
        ax.set_ylabel("P_C(τ)")
        ax.set_title("Capacity-limited return probability")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        c_ratios = [r["C_ratio"] for r in results if r["beta_C"] is not None]
        betas = [r["beta_C"] for r in results if r["beta_C"] is not None]
        ax.scatter(c_ratios, betas, alpha=0.6)
        ax.axhline(y=SIERPINSKI_D_S / 2.0, linestyle="--", label=f"Expected β≈{SIERPINSKI_D_S/2:.3f}")
        ax.set_xlabel("C_obs / N")
        ax.set_ylabel("β fit")
        ax.set_title("Spectral exponent vs capacity")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / "exp1_plots.png", dpi=150, bbox_inches="tight")

    except Exception:
        return


def main() -> int:
    parser = argparse.ArgumentParser(description="Experiment 1: Spectral Dimension vs Capacity")
    parser.add_argument("--output", default="./results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--levels", type=int, nargs="*", default=[2, 3, 4])
    args = parser.parse_args()

    print("=== Experiment 1: Spectral Dimension vs Capacity ===")
    print(f"Expected d_s: {SIERPINSKI_D_S:.4f}  (expected β = {SIERPINSKI_D_S/2:.4f})")
    print(f"Levels: {args.levels}")
    print()

    results = run_experiment(seed=args.seed, levels=args.levels)
    print(f"Generated {len(results)} data points")

    verdict = evaluate_claim_1(results, seed=args.seed)
    f1 = verdict["falsifier_results"]["falsifier_1.1"]["passed"]
    f2 = verdict["falsifier_results"]["falsifier_1.2"]["passed"]

    print()
    print(f"Falsifier 1.1: {'PASS' if f1 else 'FAIL'}")
    print(verdict["falsifier_results"]["falsifier_1.1"]["message"])
    print()
    print(f"Falsifier 1.2: {'PASS' if f2 else 'FAIL'}")
    print(verdict["falsifier_results"]["falsifier_1.2"]["message"])
    print()
    print(f"Verdict: {verdict['verdict']}")

    save_results(results, verdict, Path(args.output))
    return 0 if verdict["verdict"] == "SUPPORTED" else 1


if __name__ == "__main__":
    sys.exit(main())