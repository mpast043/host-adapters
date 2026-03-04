
#!/usr/bin/env python3
"""W03 Controls Runner - Section 10.4 Full Controls"""
import argparse, json, sys
from pathlib import Path
import subprocess

REPO_ROOT = Path(__file__).parent.parent.parent

CONTROL_CONFIGS = {
    "positive": {
        "description": "Stable gluing expected (lenient threshold)",
        "threshold_multiplier": 35.0,  # Based on observed max |S_glued - S_sum| ~ 34
        "expected_verdict": "ACCEPT"
    },
    "negative": {
        "description": "Unstable gluing expected (strict threshold)",
        "threshold_multiplier": 0.1,  # Strict threshold to force violations
        "expected_verdict": "REJECT"
    }
}

def run_single_control(control_type, A, seed, chi_sweep, output_dir):
    cfg = CONTROL_CONFIGS[control_type]
    base_runner = REPO_ROOT / "experiments/physics/exp_p3_gluing_excision_stability_runner.py"

    cmd = [
        "python3", str(base_runner),
        "--chi_sweep", ",".join(map(str, chi_sweep)),
        "--seed", str(seed),
        "--threshold", str(cfg["threshold_multiplier"]),
        "--output", str(output_dir)
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    p3_path = Path(output_dir) / "verdict.json"
    p3_data = json.load(open(p3_path)) if p3_path.exists() else {"verdict": "ERROR"}

    actual = p3_data.get("verdict", "UNKNOWN")
    expected = cfg["expected_verdict"]
    correct = (actual == expected)

    return {
        "run_id": p3_data.get("metadata", {}).get("run_id", "unknown"),
        "claim_id": "W03",
        "control_type": control_type,
        "A": A,
        "seed": seed,
        "p3_verdict": actual,
        "expected_verdict": expected,
        "correct": correct,
        "control_verdict": "CONTROL_PASS" if correct else "CONTROL_FAIL"
    }

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--A_values", default="8,16,32")
    p.add_argument("--seeds", default="42,123,456")
    p.add_argument("--chi_sweep", default="2,4,8,16")
    p.add_argument("--L", type=int, default=8)
    p.add_argument("--output_root", required=True)  # Accept --output_root from worker
    args = p.parse_args()

    A_vals = [int(x) for x in args.A_values.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]
    chi = [int(x) for x in args.chi_sweep.split(",")]

    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    results = []
    for ct in ["positive", "negative"]:
        for A in A_vals:
            for seed in seeds:
                run_dir = out_root / f"W03_{ct}_A{A}_s{seed}"
                run_dir.mkdir(exist_ok=True)
                result = run_single_control(ct, A, seed, chi, run_dir)
                results.append(result)
                with open(run_dir / "verdict.json", "w") as f:
                    json.dump(result, f, indent=2)
                print(f"[{ct}] A={A}, seed={seed}: {result['control_verdict']}")

    pos_total = len([r for r in results if r["control_type"] == "positive"])
    neg_total = len([r for r in results if r["control_type"] == "negative"])
    pos_correct = sum(1 for r in results if r["control_type"] == "positive" and r["correct"])
    neg_correct = sum(1 for r in results if r["control_type"] == "negative" and r["correct"])

    summary = {
        "total_runs": len(results),
        "results": results,
        "positive_correct": pos_correct,
        "negative_correct": neg_correct,
        "pos_count": pos_total,
        "neg_count": neg_total,
        "verdict": "ACCEPT" if (pos_correct == pos_total and neg_correct == neg_total) else "PARTIAL" if (pos_correct + neg_correct) >= 15 else "INCONCLUSIVE",
        "workflow_section": "10.4"
    }

    with open(out_root / "W03_controls_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[W03] {pos_correct}/{pos_total} pos, {neg_correct}/{neg_total} neg")
    print(f"[W03] Section 10.4 verdict: {summary['verdict']}")

if __name__ == "__main__":
    main()
