from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "prepare-qwen35-aq4-phase7-fidelity.py"
OLD_SPLIT = ROOT / "benchmarks/results/2026-07-14/qwen35-9b-aq4-production-opt-v0.1/p2/fidelity-holdout-split-v0.1"


def run(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, str(TOOL), *arguments], text=True, capture_output=True, check=False)


def test_new_phase7_fixture_pool_is_formally_split_and_disjoint(tmp_path: Path) -> None:
    output = tmp_path / "phase7-preparation"
    result = run("--output", str(output), "--vocab-size", "1000")

    assert result.returncode == 0, result.stderr
    verification = run("--output", str(output), "--verify")
    assert verification.returncode == 0, verification.stderr
    report = json.loads(verification.stdout)
    assert report["status"] == "valid"
    assert report["formal_calibration"] == 24
    assert report["formal_holdout"] == 24
    assert report["execution_view_case_count"] == 24
    assert report["phase1_to_6_distinct_contexts"] == 3

    audit = json.loads((output / "selection-audit.json").read_text(encoding="utf-8"))
    assert audit["status"] == "valid_disjoint_selection"
    assert len(audit["formal_split"]["case_ids"]) == 48
    assert len(audit["formal_split"]["context_token_ids_sha256"]) == 48
    assert all(values == [] for values in audit["intersections"].values())
    assert len(audit["retired_2026_07_15_split"]["case_ids"]) == 48
    assert len(audit["retired_2026_07_15_no_go"]["case_ids"]) == 19

    formal_holdout = [json.loads(line) for line in (output / "formal-split/holdout-cases.jsonl").read_text(encoding="utf-8").splitlines()]
    execution_rows = [json.loads(line) for line in (output / "holdout-execution-view/calibration-cases.jsonl").read_text(encoding="utf-8").splitlines()]
    assert {row["case_id"] for row in formal_holdout} == {row["case_id"] for row in execution_rows}
    assert all(row["subset"] == "calibration" for row in execution_rows)
    assert (output / "formal-split/policy.json").read_bytes() == (output / "holdout-execution-view/policy.json").read_bytes()


def test_phase7_preparation_reuses_the_frozen_policy_bytes_only() -> None:
    spec = importlib.util.spec_from_file_location("phase7_preparation_policy", TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    assert module.PROTOCOL.policy() == json.loads((OLD_SPLIT / "policy.json").read_text(encoding="utf-8"))
    assert module.TOKEN_DOMAIN != b"ullm.aq4_p2_fidelity_split.v1\0"
