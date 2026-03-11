"""Tests for ascending_superoperator_runner CLI."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_runner_with_state_json(tmp_path: Path):
    state_path = tmp_path / "state.json"
    out_path = tmp_path / "out.json"
    state_path.write_text(
        json.dumps(
            {
                "ascending_superoperator": [
                    [1.0, 0.0, 0.0],
                    [0.0, 0.5, 0.0],
                    [0.0, 0.0, 0.25],
                ]
            }
        ),
        encoding="utf-8",
    )

    repo = Path(__file__).resolve().parents[1]
    script = repo / "experiments" / "physics" / "ascending_superoperator_runner.py"
    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--state-json",
            str(state_path),
            "--num-dims",
            "3",
            "--output",
            str(out_path),
        ],
        cwd=str(repo),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr

    data = json.loads(out_path.read_text(encoding="utf-8"))
    assert data["superoperator_shape"] == [3, 3]
    dims = data["scaling_dimensions"]
    assert len(dims) == 3
    assert abs(dims[0] - 0.0) < 1e-12
    assert abs(dims[1] - 1.0) < 1e-12
    assert abs(dims[2] - 2.0) < 1e-12
