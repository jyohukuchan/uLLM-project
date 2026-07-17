#!/usr/bin/env python3
"""Compile and stage the host-only R9700 identity guard without running it.

The resulting executable is an nlink=1 copy that later service-window
drivers pass to the established read-only HIP/ASIC guard.  This helper never
executes the guard, opens HIP, queries a GPU, touches systemd, or handles the
runtime lock; invoking the compiler is its only side effect.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any


SCHEMA = "ullm.aq4_p2_production_r9700_guard_staging.v1"
NAME = "query-hip-device-identity"
MAX_RECEIPT = 512 * 1024


class GuardStageError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GuardStageError(message)


def sha(path: Path) -> str:
    info = path.lstat()
    require(stat.S_ISREG(info.st_mode) and not stat.S_ISLNK(info.st_mode), f"regular file required: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def regular(path: Path, label: str, mode: int | None = None, executable: bool = False) -> os.stat_result:
    try:
        info = path.lstat()
    except OSError as error:
        raise GuardStageError(f"{label} is unavailable: {path}: {error}") from error
    require(stat.S_ISREG(info.st_mode) and not stat.S_ISLNK(info.st_mode) and info.st_size > 0, f"{label} must be a nonempty regular non-symlink file")
    if mode is not None:
        require(stat.S_IMODE(info.st_mode) == mode, f"{label} mode differs")
    if executable:
        require(info.st_mode & 0o111, f"{label} is not executable")
    return info


def write_new(path: Path, raw: bytes, mode: int = 0o444) -> None:
    require(not os.path.lexists(path), f"refusing to overwrite: {path}")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0), mode)
    try:
        offset = 0
        while offset < len(raw):
            count = os.write(descriptor, raw[offset:])
            require(count > 0, f"short write: {path}")
            offset += count
        os.fsync(descriptor)
        os.fchmod(descriptor, mode)
    finally:
        os.close(descriptor)


def check_commit(value: str) -> None:
    require(len(value) == 40 and set(value) <= set("0123456789abcdef"), "source commit must be lowercase SHA-1")


def preparation_hash(preparation: Path) -> str:
    path = preparation / "preparation-manifest.json"
    regular(path, "preparation manifest", 0o444)
    return sha(path)


def verify(output: Path, *, preparation: Path | None = None, source_commit: str | None = None) -> dict[str, Any]:
    output = output.absolute()
    info = output.lstat()
    require(stat.S_ISDIR(info.st_mode) and not stat.S_ISLNK(info.st_mode) and stat.S_IMODE(info.st_mode) == 0o555, "guard staging directory differs")
    require(set(item.name for item in output.iterdir()) == {NAME, "guard-staging-receipt.json", "SHA256SUMS"}, "guard staging member set differs")
    binary = output / NAME
    binary_info = regular(binary, "staged guard binary", 0o555, executable=True)
    require(binary_info.st_nlink == 1, "staged guard binary must have nlink=1")
    receipt_path = output / "guard-staging-receipt.json"
    receipt_info = regular(receipt_path, "guard staging receipt", 0o444)
    require(receipt_info.st_nlink == 1 and receipt_info.st_size <= MAX_RECEIPT, "guard staging receipt differs")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    require(isinstance(receipt, dict) and receipt.get("schema_version") == SCHEMA and receipt.get("status") == "staged", "guard staging receipt schema differs")
    require(receipt.get("binary", {}).get("sha256") == sha(binary), "guard binary receipt hash differs")
    if source_commit is not None:
        check_commit(source_commit)
        require(receipt.get("source_commit") == source_commit, "guard source commit differs")
    if preparation is not None:
        require(receipt.get("preparation_manifest_sha256") == preparation_hash(preparation.absolute()), "guard preparation binding differs")
    sums = output / "SHA256SUMS"
    regular(sums, "guard SHA256SUMS", 0o444)
    expected = f"{sha(binary)}  {NAME}\n{sha(receipt_path)}  guard-staging-receipt.json\n"
    require(sums.read_text(encoding="ascii") == expected, "guard SHA256SUMS differs")
    return {"schema_version": SCHEMA, "status": "valid", "output": str(output), "binary": {"path": str(binary), "sha256": sha(binary), "nlink": binary_info.st_nlink}, "receipt_sha256": sha(receipt_path)}


def stage(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output.absolute()
    preparation = args.preparation.absolute()
    source = args.source.absolute()
    compiler = args.compiler.absolute()
    check_commit(args.source_commit)
    require(not os.path.lexists(output), f"guard staging output already exists: {output}")
    require(output.parent.is_dir() and not output.parent.is_symlink(), "guard staging parent differs")
    regular(source, "guard source")
    regular(compiler, "HIP compiler", executable=True)
    prep_sha = preparation_hash(preparation)
    output.mkdir(mode=0o700)
    temporary = output / f".{NAME}.incomplete"
    try:
        command = [str(compiler), "-std=c++20", "-O2", str(source), "-o", str(temporary)]
        result = subprocess.run(command, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        require(result.returncode == 0, f"HIP guard compilation failed with exit {result.returncode}: {result.stderr.decode('utf-8', 'replace')[-2048:]}")
        regular(temporary, "compiled guard", executable=True)
        os.chmod(temporary, 0o555)
        os.rename(temporary, output / NAME)
        binary = output / NAME
        binary_info = regular(binary, "staged guard binary", 0o555, executable=True)
        require(binary_info.st_nlink == 1, "compiled guard staging did not produce nlink=1")
        receipt = {
            "schema_version": SCHEMA,
            "status": "staged",
            "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "source_commit": args.source_commit,
            "preparation_manifest_sha256": prep_sha,
            "source": {"path": str(source), "sha256": sha(source)},
            "compiler": {"path": str(compiler), "sha256": sha(compiler), "argv": command[:-1] + ["<staged-output>"]},
            "binary": {"path": NAME, "sha256": sha(binary), "bytes": binary_info.st_size, "mode_octal": "0555", "nlink": 1},
            "compile_stdout_sha256": hashlib.sha256(result.stdout).hexdigest(),
            "compile_stderr_sha256": hashlib.sha256(result.stderr).hexdigest(),
            "guard_execution": "not_run_by_staging_tool",
        }
        write_new(output / "guard-staging-receipt.json", json.dumps(receipt, ensure_ascii=True, sort_keys=True, indent=2).encode("utf-8") + b"\n")
        write_new(output / "SHA256SUMS", f"{sha(binary)}  {NAME}\n{sha(output / 'guard-staging-receipt.json')}  guard-staging-receipt.json\n".encode("ascii"))
        os.chmod(output, 0o555)
    except Exception:
        # Keep an incomplete failed attempt for inspection and never retry into
        # the same name automatically.
        raise
    return verify(output, preparation=preparation, source_commit=args.source_commit)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--compiler", type=Path, default=Path("/opt/rocm-7.2.1/bin/hipcc"))
    parser.add_argument("--source-commit")
    parser.add_argument("--verify", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        if args.verify:
            require(args.source is None, "--source is invalid with --verify")
            result = verify(args.output, preparation=args.preparation, source_commit=args.source_commit)
        else:
            require(args.source is not None and args.source_commit is not None, "--source and --source-commit are required when staging")
            result = stage(args)
    except (GuardStageError, OSError, ValueError, subprocess.SubprocessError) as error:
        print(f"AQ4 P2 R9700 guard staging failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
