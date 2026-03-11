#!/usr/bin/env python3
"""Build and inspect a MERA ascending superoperator end-to-end."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Any

import numpy as np

try:
    from experiments.physics.entanglement_utils import (
        build_ascending_superoperator,
        diagonalize_ascending_superoperator,
        scaling_dimensions_from_eigenvalues,
    )
except ModuleNotFoundError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from experiments.physics.entanglement_utils import (
        build_ascending_superoperator,
        diagonalize_ascending_superoperator,
        scaling_dimensions_from_eigenvalues,
    )


def _parse_factory(spec: str) -> tuple[str, str]:
    if ":" not in spec:
        raise ValueError("Factory spec must be 'module.submodule:function_name'")
    module_name, fn_name = spec.split(":", 1)
    if not module_name or not fn_name:
        raise ValueError("Factory spec must be 'module.submodule:function_name'")
    return module_name, fn_name


def _load_state_from_factory(factory_spec: str, kwargs: dict[str, Any]) -> Any:
    module_name, fn_name = _parse_factory(factory_spec)
    module = importlib.import_module(module_name)
    fn = getattr(module, fn_name, None)
    if fn is None or not callable(fn):
        raise ValueError(f"Factory function '{fn_name}' not found in module '{module_name}'")
    return fn(**kwargs)


def _complex_pairs(values: np.ndarray) -> list[dict[str, float]]:
    out: list[dict[str, float]] = []
    for v in values:
        z = complex(v)
        out.append({"real": float(z.real), "imag": float(z.imag)})
    return out


def _matrix_to_pairs(matrix: np.ndarray) -> list[list[dict[str, float]]]:
    rows: list[list[dict[str, float]]] = []
    for row in matrix:
        rows.append(_complex_pairs(np.asarray(row)))
    return rows


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build ascending superoperator from a MERA state payload or factory, "
            "then diagonalize and compute Delta_alpha = -log2(|lambda_alpha|)."
        )
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--state-json",
        type=Path,
        help="JSON file with state payload (e.g. includes ascending_superoperator).",
    )
    source.add_argument(
        "--state-factory",
        type=str,
        help=(
            "Python callable spec 'module.submodule:function'. "
            "Function should return a converged MERA state object/dict."
        ),
    )
    parser.add_argument(
        "--factory-kwargs-json",
        type=Path,
        default=None,
        help="Optional JSON kwargs passed to --state-factory function.",
    )
    parser.add_argument(
        "--local-dim",
        type=int,
        default=None,
        help="Local operator dimension used when constructing from ascending_map.",
    )
    parser.add_argument(
        "--num-dims",
        type=int,
        default=10,
        help="Number of leading eigenmodes/scaling dimensions to report.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output JSON path.",
    )
    args = parser.parse_args()

    if args.state_json is not None:
        mera_state = _load_json(args.state_json)
    else:
        kwargs = {}
        if args.factory_kwargs_json is not None:
            kwargs = _load_json(args.factory_kwargs_json)
            if not isinstance(kwargs, dict):
                raise ValueError("--factory-kwargs-json must decode to a JSON object")
        mera_state = _load_state_from_factory(args.state_factory, kwargs)

    superop = build_ascending_superoperator(mera_state, local_dim=args.local_dim)
    eigvals, eigvecs = diagonalize_ascending_superoperator(superop, max_eigs=args.num_dims)
    deltas = scaling_dimensions_from_eigenvalues(eigvals)

    response = {
        "superoperator_shape": [int(superop.shape[0]), int(superop.shape[1])],
        "ascending_superoperator": _matrix_to_pairs(superop),
        "eigenvalues": _complex_pairs(eigvals),
        "scaling_dimensions": [float(x) for x in deltas],
        "formula": "Delta_alpha = -log2(|lambda_alpha|)",
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
