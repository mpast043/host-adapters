from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PLANNER = REPO_ROOT / "tools/plan_framework_selection_tests.py"
CATALOG = REPO_ROOT / "docs/physics/framework_selection_test_catalog_v1.json"


def _run_planner(tmp_path: Path, *, env: dict[str, str] | None = None, extra: list[str] | None = None) -> dict:
    out_json = tmp_path / "plan.json"
    cmd = [
        sys.executable,
        str(PLANNER),
        "--repo-root",
        str(REPO_ROOT),
        "--artifacts-root",
        str(tmp_path),
        "--catalog",
        str(CATALOG),
        "--max-minutes",
        "200",
        "--max-runs",
        "50",
        "--output-json",
        str(out_json),
        "--output-md",
        str(tmp_path / "plan.md"),
    ]
    if extra:
        cmd.extend(extra)
    subprocess.run(cmd, check=True, env=env)
    return json.loads(out_json.read_text(encoding="utf-8"))


def test_tier_c_blocked_by_default(tmp_path: Path) -> None:
    plan = _run_planner(tmp_path, extra=["--local-only"])
    selected_tiers = {row["tier"] for row in plan["selected"]}
    assert "C" not in selected_tiers
    assert any("TIER_C_BLOCKED" in ";".join(row["gate"]["reasons"]) for row in plan["blocked"] if row.get("tier") == "C")


def test_tier_c_allowed_with_override(tmp_path: Path) -> None:
    env = os.environ.copy()
    env["RUN_OVERRIDE_TIER_C"] = "1"
    plan = _run_planner(
        tmp_path,
        env=env,
        extra=["--tier-c-justification", "Needed for Claim 3P exploration", "--compute-target", "local-compute"],
    )
    selected_tiers = {row["tier"] for row in plan["selected"]}
    assert "C" in selected_tiers


def test_step0_field_validation_blocks_incomplete_test(tmp_path: Path) -> None:
    bad_catalog = tmp_path / "bad_catalog.json"
    bad_catalog.write_text(
        json.dumps(
            {
                "tests": [
                    {
                        "test_id": "bad_case",
                        "tier": "B",
                        "critical_path_objective": "B",
                        "estimated_minutes": 1,
                        "protects_supported_claim": True,
                        "prevents_regression": True,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    out_json = tmp_path / "plan.json"
    cmd = [
        sys.executable,
        str(PLANNER),
        "--repo-root",
        str(REPO_ROOT),
        "--artifacts-root",
        str(tmp_path),
        "--catalog",
        str(bad_catalog),
        "--output-json",
        str(out_json),
        "--output-md",
        str(tmp_path / "plan.md"),
    ]
    subprocess.run(cmd, check=True)
    plan = json.loads(out_json.read_text(encoding="utf-8"))
    assert not plan["selected"]
    assert "STEP0_FAILURE" in ";".join(plan["blocked"][0]["gate"]["reasons"])
