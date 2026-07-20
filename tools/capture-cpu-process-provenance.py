#!/usr/bin/env python3
"""Capture live CPU process commands and the exact Git tool blob they loaded."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any


VISIBLE_DEVICE_KEYS = (
    "CUDA_VISIBLE_DEVICES",
    "HIP_VISIBLE_DEVICES",
    "ROCR_VISIBLE_DEVICES",
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_blob(commit: str, path: Path) -> bytes:
    return subprocess.check_output(["git", "show", f"{commit}:{path.as_posix()}"])


def process_record(pid: int, expected_tool: str) -> dict[str, Any]:
    proc = Path("/proc") / str(pid)
    if not proc.is_dir():
        raise RuntimeError(f"process is not live: {pid}")
    command = [
        item.decode("utf-8", errors="surrogateescape")
        for item in (proc / "cmdline").read_bytes().split(b"\0")
        if item
    ]
    if not any(item.endswith(expected_tool) for item in command):
        raise RuntimeError(f"PID {pid} command does not contain expected tool {expected_tool}")
    environment = {}
    for item in (proc / "environ").read_bytes().split(b"\0"):
        if b"=" not in item:
            continue
        key, value = item.split(b"=", 1)
        decoded_key = key.decode("utf-8", errors="replace")
        if decoded_key in VISIBLE_DEVICE_KEYS:
            environment[decoded_key] = value.decode("utf-8", errors="replace")
    stat = (proc / "stat").read_text(encoding="utf-8").split()
    return {
        "pid": pid,
        "command": command,
        "working_directory": os.readlink(proc / "cwd"),
        "executable": os.readlink(proc / "exe"),
        "process_start_clock_ticks_since_boot": int(stat[21]),
        "visible_device_environment": environment,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid", action="append", type=int, required=True)
    parser.add_argument("--tool-path", type=Path, required=True)
    parser.add_argument("--tool-git-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--note", action="append", default=[])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite process provenance: {output}")
    tool_path = args.tool_path
    blob = git_blob(args.tool_git_commit, tool_path)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    payload: dict[str, Any] = {
        "schema_version": "cpu-process-launch-provenance-v0.1",
        "captured_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "captured while every listed process was live",
        "tool_path": tool_path.as_posix(),
        "tool_git_commit_loaded_at_launch": args.tool_git_commit,
        "tool_git_blob_sha256": sha256_bytes(blob),
        "current_worktree_tool_sha256_at_capture": sha256_file(tool_path.resolve()),
        "workspace_git_head_at_capture": head,
        "processes": [process_record(pid, tool_path.as_posix()) for pid in args.pid],
        "notes": args.note,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(output), "process_count": len(args.pid)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
