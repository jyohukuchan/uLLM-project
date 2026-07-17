from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "stage-aq4-phase7-fidelity-capture-binary.py"
BINARY = "ullm-aq4-fidelity-capture"
COMMIT = "d3ea48d543456a07a2796ee804671c3da513c268"


def run(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, str(TOOL), *arguments], text=True, capture_output=True, check=False)


def test_stage_is_content_copy_with_one_link_and_immutable_receipt(tmp_path: Path) -> None:
    source = tmp_path / BINARY
    source.write_bytes(b"phase7 detached fidelity capture\n" * 1024)
    source.chmod(0o700)
    os.link(source, tmp_path / "deps-peer")
    assert source.stat().st_nlink == 2
    stage = tmp_path / "stage"

    created = run("--source", str(source), "--output", str(stage), "--source-commit", COMMIT)
    assert created.returncode == 0, created.stderr
    report = json.loads(created.stdout)
    staged = stage / BINARY
    assert staged.stat().st_nlink == 1
    assert stat.S_IMODE(staged.stat().st_mode) == 0o555
    assert hashlib.sha256(staged.read_bytes()).hexdigest() == hashlib.sha256(source.read_bytes()).hexdigest()
    assert report["status"] == "valid"
    assert json.loads((stage / "staging-receipt.json").read_text(encoding="utf-8"))["source_commit"] == COMMIT
    assert stat.S_IMODE((stage / "staging-receipt.json").stat().st_mode) == 0o444

    verified = run("--verify", "--source", str(source), "--output", str(stage), "--source-commit", COMMIT)
    assert verified.returncode == 0, verified.stderr


def test_stage_refuses_reuse(tmp_path: Path) -> None:
    source = tmp_path / BINARY
    source.write_bytes(b"phase7 source\n")
    source.chmod(0o700)
    stage = tmp_path / "stage"
    assert run("--source", str(source), "--output", str(stage), "--source-commit", COMMIT).returncode == 0
    reused = run("--source", str(source), "--output", str(stage), "--source-commit", COMMIT)
    assert reused.returncode != 0
    assert "refusing to overwrite existing staging directory" in reused.stderr
