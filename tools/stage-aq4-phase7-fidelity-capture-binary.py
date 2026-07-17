#!/usr/bin/env python3
"""Create/verify a detached nlink=1 Phase 7 fidelity-capture executable.

Cargo's release executable may be hard-linked to its ``deps`` peer.  The GPU
window may execute only the content copy created here: a regular, executable,
single-link file whose bytes are bound to a clean-source commit.  Creation is
create-new only and verification never repairs an existing stage.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import stat
import sys
from pathlib import Path
from typing import Any


SCHEMA = "ullm.aq4_phase7_fidelity_capture_binary_staging.v1"
BINARY_NAME = "ullm-aq4-fidelity-capture"
DIRECTORY_MODE = 0o555
BINARY_MODE = 0o555
IMMUTABLE_MODE = 0o444
COPY_CHUNK = 1024 * 1024
MAX_RECEIPT = 128 * 1024


class StageError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise StageError(message)


def identity(info: os.stat_result) -> tuple[int, int, int, int, int, int, int]:
    return info.st_dev, info.st_ino, info.st_mode, info.st_size, info.st_mtime_ns, info.st_ctime_ns, info.st_nlink


def mode(info: os.stat_result) -> str:
    return format(stat.S_IMODE(info.st_mode), "04o")


def nofollow() -> int:
    if not hasattr(os, "O_NOFOLLOW"):
        raise StageError("O_NOFOLLOW is unavailable")
    return os.O_NOFOLLOW


def regular(path: Path, label: str, *, expected_mode: int | None = None, single_link: bool = False, executable: bool = False) -> os.stat_result:
    try:
        info = path.lstat()
    except OSError as error:
        raise StageError(f"{label} is unavailable: {path}: {error}") from error
    require(not stat.S_ISLNK(info.st_mode) and stat.S_ISREG(info.st_mode), f"{label} must be a regular non-symlink file: {path}")
    require(info.st_size > 0, f"{label} is empty: {path}")
    if expected_mode is not None:
        require(mode(info) == format(expected_mode, "04o"), f"{label} mode differs: {path}")
    if single_link:
        require(info.st_nlink == 1, f"{label} must have nlink=1: {path}")
    if executable:
        require(info.st_mode & 0o111, f"{label} is not executable: {path}")
    return info


def directory(path: Path, label: str, *, expected_mode: int | None = None) -> os.stat_result:
    try:
        info = path.lstat()
    except OSError as error:
        raise StageError(f"{label} is unavailable: {path}: {error}") from error
    require(not stat.S_ISLNK(info.st_mode) and stat.S_ISDIR(info.st_mode), f"{label} must be a real directory: {path}")
    if expected_mode is not None:
        require(mode(info) == format(expected_mode, "04o"), f"{label} mode differs: {path}")
    return info


def absent(path: Path, label: str) -> None:
    try:
        path.lstat()
    except FileNotFoundError:
        return
    except OSError as error:
        raise StageError(f"cannot inspect {label}: {error}") from error
    raise StageError(f"refusing to overwrite existing {label}: {path}")


def fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def sha_stable(path: Path, label: str, *, expected_mode: int | None = None, single_link: bool = False, max_bytes: int | None = None) -> tuple[str, os.stat_result, bytes | None]:
    before = regular(path, label, expected_mode=expected_mode, single_link=single_link)
    if max_bytes is not None:
        require(before.st_size <= max_bytes, f"{label} exceeds its bounded size")
    fd = -1
    digest = hashlib.sha256()
    saved = bytearray() if max_bytes is not None else None
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | nofollow())
        require(identity(before) == identity(os.fstat(fd)), f"{label} changed while opening")
        while True:
            chunk = os.read(fd, COPY_CHUNK)
            if not chunk:
                break
            digest.update(chunk)
            if saved is not None:
                saved.extend(chunk)
                require(len(saved) <= max_bytes, f"{label} exceeds its bounded size while reading")
        require(identity(before) == identity(os.fstat(fd)), f"{label} changed while reading")
    finally:
        if fd >= 0:
            os.close(fd)
    after = regular(path, label, expected_mode=expected_mode, single_link=single_link)
    require(identity(before) == identity(after), f"{label} changed after reading")
    return digest.hexdigest(), after, None if saved is None else bytes(saved)


def write_new(path: Path, payload: bytes, file_mode: int) -> os.stat_result:
    absent(path, "staging file")
    fd = -1
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0) | nofollow(), file_mode)
        offset = 0
        while offset < len(payload):
            count = os.write(fd, payload[offset:])
            require(count > 0, "short write creating staging evidence")
            offset += count
        os.fsync(fd)
        os.fchmod(fd, file_mode)
        opened = os.fstat(fd)
        require(stat.S_ISREG(opened.st_mode) and opened.st_nlink == 1 and mode(opened) == format(file_mode, "04o"), "staging evidence identity differs while writing")
    finally:
        if fd >= 0:
            os.close(fd)
    return regular(path, "staging evidence", expected_mode=file_mode, single_link=True)


def copy_binary(source: Path, destination: Path) -> tuple[str, os.stat_result, os.stat_result]:
    source_before = regular(source, "Cargo fidelity capture binary", executable=True)
    require(source.name == BINARY_NAME, f"unexpected Cargo binary basename: {source.name}")
    absent(destination, "staged fidelity capture binary")
    source_fd = destination_fd = -1
    digest = hashlib.sha256()
    copied = 0
    try:
        source_fd = os.open(source, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | nofollow())
        require(identity(source_before) == identity(os.fstat(source_fd)), "Cargo fidelity capture binary changed while opening")
        destination_fd = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0) | nofollow(), BINARY_MODE)
        while True:
            chunk = os.read(source_fd, COPY_CHUNK)
            if not chunk:
                break
            digest.update(chunk)
            copied += len(chunk)
            offset = 0
            while offset < len(chunk):
                count = os.write(destination_fd, chunk[offset:])
                require(count > 0, "short write while staging fidelity capture binary")
                offset += count
        require(copied == source_before.st_size and identity(source_before) == identity(os.fstat(source_fd)), "Cargo fidelity capture binary changed while copying")
        os.fsync(destination_fd)
        os.fchmod(destination_fd, BINARY_MODE)
        staged = os.fstat(destination_fd)
        require(stat.S_ISREG(staged.st_mode) and staged.st_nlink == 1 and mode(staged) == format(BINARY_MODE, "04o"), "staged fidelity capture binary identity differs")
    finally:
        if destination_fd >= 0:
            os.close(destination_fd)
        if source_fd >= 0:
            os.close(source_fd)
    source_after = regular(source, "Cargo fidelity capture binary", executable=True)
    require(identity(source_before) == identity(source_after), "Cargo fidelity capture binary changed after copying")
    staged_after = regular(destination, "staged fidelity capture binary", expected_mode=BINARY_MODE, single_link=True, executable=True)
    return digest.hexdigest(), source_after, staged_after


def record(info: os.stat_result) -> dict[str, int | str]:
    return {"device": info.st_dev, "inode": info.st_ino, "bytes": info.st_size, "mode_octal": mode(info), "nlink": info.st_nlink, "uid": info.st_uid, "gid": info.st_gid}


def check_commit(commit: str) -> None:
    require(len(commit) == 40 and all(character in "0123456789abcdef" for character in commit), "source commit must be a lowercase 40-character SHA-1")


def verify(stage: Path, source: Path | None = None, source_commit: str | None = None) -> dict[str, Any]:
    stage_info = directory(stage, "fidelity-capture staging directory", expected_mode=DIRECTORY_MODE)
    require(set(os.listdir(stage)) == {BINARY_NAME, "staging-receipt.json", "SHA256SUMS"}, "fidelity-capture staging members differ")
    binary_sha, binary_info, _ = sha_stable(stage / BINARY_NAME, "staged fidelity capture binary", expected_mode=BINARY_MODE, single_link=True)
    receipt_sha, receipt_info, receipt_raw = sha_stable(stage / "staging-receipt.json", "staging receipt", expected_mode=IMMUTABLE_MODE, single_link=True, max_bytes=MAX_RECEIPT)
    assert receipt_raw is not None
    try:
        receipt = json.loads(receipt_raw)
    except json.JSONDecodeError as error:
        raise StageError(f"staging receipt is invalid JSON: {error}") from error
    require(isinstance(receipt, dict) and receipt.get("schema_version") == SCHEMA and receipt.get("status") == "staged", "staging receipt schema/status differs")
    require(receipt.get("source_commit") and isinstance(receipt.get("source"), dict) and isinstance(receipt.get("staged_binary"), dict), "staging receipt identity is missing")
    require(receipt["staged_binary"].get("path") == BINARY_NAME and receipt["staged_binary"].get("sha256") == binary_sha and receipt["staged_binary"].get("nlink") == 1 and receipt["staged_binary"].get("mode_octal") == "0555", "staging receipt staged binary differs")
    require(receipt["source"].get("sha256") == binary_sha, "staging receipt source SHA differs")
    if source is not None:
        source_sha, _, _ = sha_stable(source, "Cargo fidelity capture binary")
        require(receipt["source"].get("path") == str(source) and source_sha == binary_sha, "staging source binding differs")
    if source_commit is not None:
        check_commit(source_commit)
        require(receipt.get("source_commit") == source_commit, "staging source commit differs")
    _, sums_info, sums_raw = sha_stable(stage / "SHA256SUMS", "staging SHA256SUMS", expected_mode=IMMUTABLE_MODE, single_link=True, max_bytes=MAX_RECEIPT)
    assert sums_raw is not None
    require(sums_raw == f"{binary_sha}  {BINARY_NAME}\n{receipt_sha}  staging-receipt.json\n".encode(), "staging SHA256SUMS differs")
    require(identity(stage_info) == identity(directory(stage, "fidelity-capture staging directory", expected_mode=DIRECTORY_MODE)), "staging directory changed while verifying")
    return {"schema_version": SCHEMA, "status": "valid", "stage_directory": str(stage), "source_commit": receipt["source_commit"], "binary": {"path": str(stage / BINARY_NAME), "sha256": binary_sha, **record(binary_info)}, "receipt": {"path": str(stage / "staging-receipt.json"), "sha256": receipt_sha, **record(receipt_info)}, "sha256sums": {"path": str(stage / "SHA256SUMS"), **record(sums_info)}}


def stage(source: Path, output: Path, source_commit: str) -> dict[str, Any]:
    check_commit(source_commit)
    directory(output.parent, "staging parent")
    absent(output, "staging directory")
    output.mkdir(mode=0o700)
    binary_sha, source_info, staged_info = copy_binary(source, output / BINARY_NAME)
    fsync_dir(output)
    receipt = {"schema_version": SCHEMA, "status": "staged", "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "source_commit": source_commit, "source": {"path": str(source), "sha256": binary_sha, **record(source_info)}, "staged_binary": {"path": BINARY_NAME, "sha256": binary_sha, **record(staged_info)}}
    write_new(output / "staging-receipt.json", (json.dumps(receipt, ensure_ascii=True, sort_keys=True, indent=2) + "\n").encode(), IMMUTABLE_MODE)
    receipt_sha, _, _ = sha_stable(output / "staging-receipt.json", "staging receipt", expected_mode=IMMUTABLE_MODE, single_link=True)
    write_new(output / "SHA256SUMS", f"{binary_sha}  {BINARY_NAME}\n{receipt_sha}  staging-receipt.json\n".encode(), IMMUTABLE_MODE)
    os.chmod(output, DIRECTORY_MODE)
    fsync_dir(output)
    fsync_dir(output.parent)
    return verify(output, source, source_commit)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--source-commit")
    parser.add_argument("--verify", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        source = args.source.absolute() if args.source is not None else None
        if args.verify:
            result = verify(args.output.absolute(), source, args.source_commit)
        else:
            require(source is not None and args.source_commit is not None, "--source and --source-commit are required when creating staging")
            result = stage(source, args.output.absolute(), args.source_commit)
    except (StageError, OSError, ValueError) as error:
        print(f"AQ4 Phase 7 fidelity-capture staging failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
