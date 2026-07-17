#!/usr/bin/env python3
"""Stage the two clean-build AQ4 P2 binaries as detached nlink=1 copies.

The P2 service window must never execute Cargo's possibly hard-linked output.
This tool creates a new, immutable staging directory containing only the
resident timing driver, full-vector calibration driver, receipt, and checksum
manifest.  It is CPU-only and does not query a GPU or a service.
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


SCHEMA = "ullm.aq4_p2_production_baseline_binary_staging.v1"
BINARIES = ("ullm-aq4-p2-resident-driver", "ullm-aq4-p2-calibration")
DIRECTORY_MODE = 0o555
BINARY_MODE = 0o555
IMMUTABLE_MODE = 0o444
COPY_CHUNK = 1024 * 1024
MAX_RECEIPT_BYTES = 256 * 1024


class StageError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise StageError(message)


def identity(info: os.stat_result) -> tuple[int, int, int, int, int, int, int]:
    return (
        info.st_dev,
        info.st_ino,
        info.st_mode,
        info.st_nlink,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def mode(info: os.stat_result) -> str:
    return format(stat.S_IMODE(info.st_mode), "04o")


def record(info: os.stat_result) -> dict[str, int | str]:
    return {
        "device": info.st_dev,
        "inode": info.st_ino,
        "bytes": info.st_size,
        "mode_octal": mode(info),
        "nlink": info.st_nlink,
        "uid": info.st_uid,
        "gid": info.st_gid,
    }


def regular(
    path: Path,
    label: str,
    *,
    expected_mode: int | None = None,
    single_link: bool = False,
    executable: bool = False,
) -> os.stat_result:
    try:
        info = path.lstat()
    except OSError as error:
        raise StageError(f"{label} is unavailable: {path}: {error}") from error
    require(stat.S_ISREG(info.st_mode) and not stat.S_ISLNK(info.st_mode), f"{label} must be a regular non-symlink file: {path}")
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
    require(stat.S_ISDIR(info.st_mode) and not stat.S_ISLNK(info.st_mode), f"{label} must be a real directory: {path}")
    if expected_mode is not None:
        require(mode(info) == format(expected_mode, "04o"), f"{label} mode differs: {path}")
    return info


def absent(path: Path, label: str) -> None:
    try:
        path.lstat()
    except FileNotFoundError:
        return
    except OSError as error:
        raise StageError(f"cannot inspect {label}: {path}: {error}") from error
    raise StageError(f"refusing to overwrite existing {label}: {path}")


def sha_stable(
    path: Path,
    label: str,
    *,
    expected_mode: int | None = None,
    single_link: bool = False,
    maximum: int | None = None,
) -> tuple[str, os.stat_result, bytes | None]:
    before = regular(path, label, expected_mode=expected_mode, single_link=single_link)
    if maximum is not None:
        require(before.st_size <= maximum, f"{label} exceeds bounded size")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0))
    digest = hashlib.sha256()
    collected = bytearray() if maximum is not None else None
    try:
        require(identity(os.fstat(descriptor)) == identity(before), f"{label} changed while opening")
        while chunk := os.read(descriptor, COPY_CHUNK):
            digest.update(chunk)
            if collected is not None:
                collected.extend(chunk)
                require(len(collected) <= maximum, f"{label} exceeds bounded size while reading")
        require(identity(os.fstat(descriptor)) == identity(before), f"{label} changed while reading")
    finally:
        os.close(descriptor)
    require(identity(regular(path, label, expected_mode=expected_mode, single_link=single_link)) == identity(before), f"{label} changed after reading")
    return digest.hexdigest(), before, None if collected is None else bytes(collected)


def write_new(path: Path, raw: bytes, file_mode: int) -> os.stat_result:
    absent(path, "staging member")
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        file_mode,
    )
    try:
        offset = 0
        while offset < len(raw):
            written = os.write(descriptor, raw[offset:])
            require(written > 0, "short write while creating staging member")
            offset += written
        os.fsync(descriptor)
        os.fchmod(descriptor, file_mode)
        info = os.fstat(descriptor)
        require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1 and mode(info) == format(file_mode, "04o"), "staging member identity differs while writing")
    finally:
        os.close(descriptor)
    return regular(path, "staging member", expected_mode=file_mode, single_link=True)


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def copy_binary(source: Path, destination: Path, expected_name: str) -> tuple[str, os.stat_result, os.stat_result]:
    before = regular(source, f"Cargo {expected_name} binary", executable=True)
    require(source.name == expected_name, f"unexpected Cargo binary name: {source.name} (expected {expected_name})")
    absent(destination, f"staged {expected_name} binary")
    source_fd = os.open(source, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0))
    destination_fd = -1
    digest = hashlib.sha256()
    copied = 0
    try:
        require(identity(os.fstat(source_fd)) == identity(before), f"Cargo {expected_name} binary changed while opening")
        destination_fd = os.open(
            destination,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
            BINARY_MODE,
        )
        while chunk := os.read(source_fd, COPY_CHUNK):
            digest.update(chunk)
            copied += len(chunk)
            offset = 0
            while offset < len(chunk):
                written = os.write(destination_fd, chunk[offset:])
                require(written > 0, f"short write staging {expected_name}")
                offset += written
        require(copied == before.st_size and identity(os.fstat(source_fd)) == identity(before), f"Cargo {expected_name} binary changed while copying")
        os.fsync(destination_fd)
        os.fchmod(destination_fd, BINARY_MODE)
        staged = os.fstat(destination_fd)
        require(stat.S_ISREG(staged.st_mode) and staged.st_nlink == 1 and mode(staged) == "0555", f"staged {expected_name} identity differs")
    finally:
        if destination_fd >= 0:
            os.close(destination_fd)
        os.close(source_fd)
    after = regular(source, f"Cargo {expected_name} binary", executable=True)
    require(identity(after) == identity(before), f"Cargo {expected_name} binary changed after copying")
    staged_after = regular(destination, f"staged {expected_name} binary", expected_mode=BINARY_MODE, single_link=True, executable=True)
    return digest.hexdigest(), after, staged_after


def check_commit(commit: str) -> None:
    require(len(commit) == 40 and set(commit) <= set("0123456789abcdef"), "source commit must be a lowercase 40-character SHA-1")


def preparation_binding(preparation: Path) -> tuple[str, str]:
    manifest = preparation / "preparation-manifest.json"
    regular(manifest, "preparation manifest", expected_mode=0o444, single_link=True)
    try:
        value = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise StageError(f"preparation manifest is invalid: {error}") from error
    require(value.get("schema_version") == "ullm.aq4_p2_production_baseline_preparation.v1" and value.get("status") == "prepared", "preparation manifest schema/status differs")
    declared = value.get("preparation_sha256")
    require(isinstance(declared, str) and len(declared) == 64 and set(declared) <= set("0123456789abcdef"), "preparation manifest self-hash is invalid")
    return sha_stable(manifest, "preparation manifest", expected_mode=0o444, single_link=True)[0], declared


def verify(
    output: Path,
    *,
    resident_source: Path | None = None,
    calibration_source: Path | None = None,
    source_commit: str | None = None,
    preparation: Path | None = None,
) -> dict[str, Any]:
    output = output.absolute()
    directory(output, "baseline binary staging directory", expected_mode=DIRECTORY_MODE)
    expected = {*BINARIES, "staging-receipt.json", "SHA256SUMS"}
    require(set(os.listdir(output)) == expected, "baseline staging member set differs")
    binaries: dict[str, dict[str, Any]] = {}
    for name in BINARIES:
        digest, info, _ = sha_stable(output / name, f"staged {name}", expected_mode=BINARY_MODE, single_link=True)
        binaries[name] = {"sha256": digest, **record(info)}
    receipt_digest, receipt_info, receipt_bytes = sha_stable(
        output / "staging-receipt.json",
        "staging receipt",
        expected_mode=IMMUTABLE_MODE,
        single_link=True,
        maximum=MAX_RECEIPT_BYTES,
    )
    assert receipt_bytes is not None
    try:
        receipt = json.loads(receipt_bytes)
    except json.JSONDecodeError as error:
        raise StageError(f"staging receipt is invalid JSON: {error}") from error
    require(isinstance(receipt, dict) and receipt.get("schema_version") == SCHEMA and receipt.get("status") == "staged", "staging receipt schema/status differs")
    if source_commit is not None:
        check_commit(source_commit)
        require(receipt.get("source_commit") == source_commit, "staging source commit differs")
    if preparation is not None:
        prep_file_hash, prep_self_hash = preparation_binding(preparation.absolute())
        require(receipt.get("preparation_manifest_sha256") == prep_file_hash, "staging preparation manifest hash differs")
        require(receipt.get("preparation_sha256") == prep_self_hash, "staging preparation self-hash differs")
    sources = receipt.get("sources")
    staged = receipt.get("staged_binaries")
    require(isinstance(sources, dict) and isinstance(staged, dict), "staging receipt members are missing")
    supplied = {
        BINARIES[0]: resident_source.absolute() if resident_source is not None else None,
        BINARIES[1]: calibration_source.absolute() if calibration_source is not None else None,
    }
    for name in BINARIES:
        source = sources.get(name)
        destination = staged.get(name)
        require(isinstance(source, dict) and isinstance(destination, dict), f"staging receipt lacks {name}")
        require(source.get("sha256") == binaries[name]["sha256"], f"staging receipt source digest differs: {name}")
        require(destination.get("sha256") == binaries[name]["sha256"] and destination.get("path") == name and destination.get("nlink") == 1 and destination.get("mode_octal") == "0555", f"staging receipt destination differs: {name}")
        if supplied[name] is not None:
            digest, _, _ = sha_stable(supplied[name], f"Cargo {name} binary")
            require(source.get("path") == str(supplied[name]) and digest == binaries[name]["sha256"], f"staging source binding differs: {name}")
    sums_digest, sums_info, sums_bytes = sha_stable(
        output / "SHA256SUMS",
        "staging SHA256SUMS",
        expected_mode=IMMUTABLE_MODE,
        single_link=True,
        maximum=MAX_RECEIPT_BYTES,
    )
    assert sums_bytes is not None
    expected_sums = "".join(
        f"{binaries[name]['sha256']}  {name}\n" for name in BINARIES
    ) + f"{receipt_digest}  staging-receipt.json\n"
    require(sums_bytes == expected_sums.encode("ascii"), "staging SHA256SUMS differs")
    return {
        "schema_version": SCHEMA,
        "status": "valid",
        "stage_directory": str(output),
        "source_commit": receipt.get("source_commit"),
        "binaries": {name: {"path": str(output / name), **facts} for name, facts in binaries.items()},
        "receipt": {"path": str(output / "staging-receipt.json"), "sha256": receipt_digest, **record(receipt_info)},
        "sha256sums": {"path": str(output / "SHA256SUMS"), "sha256": sums_digest, **record(sums_info)},
    }


def stage(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output.absolute()
    check_commit(args.source_commit)
    directory(output.parent, "baseline staging parent")
    absent(output, "baseline staging directory")
    preparation = args.preparation.absolute()
    preparation_manifest_sha, preparation_sha = preparation_binding(preparation)
    output.mkdir(mode=0o700)
    try:
        inputs = {
            BINARIES[0]: args.resident_source.absolute(),
            BINARIES[1]: args.calibration_source.absolute(),
        }
        sources: dict[str, dict[str, Any]] = {}
        staged: dict[str, dict[str, Any]] = {}
        for name, source in inputs.items():
            digest, source_info, staged_info = copy_binary(source, output / name, name)
            sources[name] = {"path": str(source), "sha256": digest, **record(source_info)}
            staged[name] = {"path": name, "sha256": digest, **record(staged_info)}
        receipt = {
            "schema_version": SCHEMA,
            "status": "staged",
            "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "source_commit": args.source_commit,
            "preparation_manifest_path": str(preparation / "preparation-manifest.json"),
            "preparation_manifest_sha256": preparation_manifest_sha,
            "preparation_sha256": preparation_sha,
            "sources": sources,
            "staged_binaries": staged,
        }
        write_new(output / "staging-receipt.json", json.dumps(receipt, ensure_ascii=True, sort_keys=True, indent=2).encode("utf-8") + b"\n", IMMUTABLE_MODE)
        receipt_sha, _, _ = sha_stable(output / "staging-receipt.json", "staging receipt", expected_mode=IMMUTABLE_MODE, single_link=True)
        sums = "".join(f"{sources[name]['sha256']}  {name}\n" for name in BINARIES) + f"{receipt_sha}  staging-receipt.json\n"
        write_new(output / "SHA256SUMS", sums.encode("ascii"), IMMUTABLE_MODE)
        os.chmod(output, DIRECTORY_MODE)
        fsync_directory(output)
        fsync_directory(output.parent)
    except Exception:
        raise
    return verify(
        output,
        resident_source=args.resident_source.absolute(),
        calibration_source=args.calibration_source.absolute(),
        source_commit=args.source_commit,
        preparation=preparation,
    )


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--resident-source", type=Path)
    parser.add_argument("--calibration-source", type=Path)
    parser.add_argument("--source-commit")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        if args.verify:
            result = verify(
                args.output,
                resident_source=args.resident_source,
                calibration_source=args.calibration_source,
                source_commit=args.source_commit,
                preparation=args.preparation,
            )
        else:
            require(args.resident_source is not None and args.calibration_source is not None and args.source_commit is not None, "--resident-source, --calibration-source, and --source-commit are required when staging")
            result = stage(args)
    except (StageError, OSError, ValueError) as error:
        print(f"AQ4 P2 production baseline binary staging failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
