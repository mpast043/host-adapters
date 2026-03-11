#!/usr/bin/env python3
"""Compute MERA scaling dimensions from an ascending superoperator payload."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from experiments.physics.entanglement_utils import extract_scaling_dimensions_mera
except ModuleNotFoundError:
    import sys

    # Allow direct invocation: python experiments/physics/scaling_dimensions_runner.py
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from experiments.physics.entanglement_utils import extract_scaling_dimensions_mera


def _complex_list(values: Any) -> list[dict[str, float]]:
    out: list[dict[str, float]] = []
    for value in values:
        z = complex(value)
        out.append({"real": float(z.real), "imag": float(z.imag)})
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build ascending superoperator, diagonalize, compute Δ = -log2(|λ|)."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="JSON file containing MERA payload (e.g. ascending_superoperator).",
    )
    parser.add_argument("--num-dims", type=int, default=10, help="Number of leading scaling dimensions to return.")
    parser.add_argument(
        "--local-dim",
        type=int,
        default=2,
        help="Local operator dimension when constructing A from an ascending map.",
    )
    parser.add_argument("--output", type=Path, default=None, help="Optional output JSON path.")
    args = parser.parse_args()

    payload = json.loads(args.input.read_text(encoding="utf-8"))
    result = extract_scaling_dimensions_mera(
        payload,
        num_dims=args.num_dims,
        local_dim=args.local_dim,
    )

    response = {
        "num_dims": int(len(result["scaling_dimensions"])),
        "eigenvalues": _complex_list(result["eigenvalues"]),
        "scaling_dimensions": [float(x) for x in result["scaling_dimensions"]],
    }

    text = json.dumps(response, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
