#!/usr/bin/env python3
"""Parse one new-current-identity rocprof capture for a P2 detail window.

This parser is intentionally separate from the historical P2 profile tools.
It makes no identity assumption beyond the supplied new window result and its
trace binding, treats unclassified kernels as blockers by default, and does
not feed profiler timings into normal p50/p95 measurements.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import stat
import sys
from pathlib import Path
from typing import Any


SCHEMA = "ullm.aq4_p2_production_baseline_profile.v1"
MAX_ROWS = 500_000
MAX_TRACE_BYTES = 256 * 1024 * 1024
MAX_PROFILE_FILES = 128
MAX_PROFILE_TOTAL_BYTES = 512 * 1024 * 1024

FAMILIES: dict[str, tuple[str, ...]] = {
    "runtime_support": (r"__amd_rocclr_", r"memset", r"fillbuffer", r"copybuffer"),
    "embedding": (r"bf16.*row", r"embedding"),
    "aq4_projection": (r"aq4.*(?:matvec|gemm|projection|register.*bm8)", r"(?:matvec|gemm).*aq4"),
    "attention": (r"paged.*(?:attn|attention|causal.*gqa|decode)", r"qk.*norm.*rope"),
    "recurrent": (r"linear.*attn", r"gated.*delta", r"recurrent", r"qkv.*prepare"),
    "normalization": (r"rmsnorm", r"rms.*norm", r"silu.*mul", r"sigmoid.*mul", r"rope", r"(?:^|_)add(?:_|$)"),
    "head": (r"top1", r"lm.*head", r"argmax"),
}
COMPILED = {family: tuple(re.compile(pattern, re.IGNORECASE) for pattern in patterns) for family, patterns in FAMILIES.items()}


class ProfileError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ProfileError(message)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def trace_file(profile_dir: Path, suffix: str) -> Path | None:
    candidates = sorted(profile_dir.rglob(f"*_{suffix}.csv"))
    if not candidates:
        return None
    require(len(candidates) == 1, f"expected exactly one {suffix} CSV, got {len(candidates)}")
    path = candidates[0]
    require(path.is_file() and not path.is_symlink() and path.stat().st_size <= MAX_TRACE_BYTES, f"{suffix} trace differs")
    return path


def profile_inventory(profile_dir: Path) -> list[dict[str, Any]]:
    """Hash every raw profiler member that is allowed to influence parsing.

    rocprofv3 can create a shallow output directory containing metadata in
    addition to the CSV traces.  Keeping an inventory binds the parser result
    to that complete bounded raw capture rather than only to whichever CSV
    happened to be selected by a glob.
    """

    members: list[dict[str, Any]] = []
    total = 0
    for path in sorted(profile_dir.rglob("*"), key=lambda item: item.as_posix()):
        info = path.lstat()
        require(not stat.S_ISLNK(info.st_mode), f"profile member is a symlink: {path}")
        if stat.S_ISDIR(info.st_mode):
            continue
        require(stat.S_ISREG(info.st_mode), f"profile member is not a regular file: {path}")
        require(info.st_size <= MAX_TRACE_BYTES, f"profile member exceeds per-file bound: {path}")
        total += info.st_size
        require(total <= MAX_PROFILE_TOTAL_BYTES, "profile capture exceeds total byte bound")
        members.append(
            {
                "path": path.relative_to(profile_dir).as_posix(),
                "bytes": info.st_size,
                "sha256": sha(path),
            }
        )
        require(len(members) <= MAX_PROFILE_FILES, "profile capture exceeds file-count bound")
    require(members, "profile directory is empty")
    return members


def column(names: list[str], aliases: tuple[str, ...], label: str, required: bool = True) -> str | None:
    matches = [name for name in aliases if name in names]
    if required:
        require(len(matches) == 1, f"trace {label} column differs")
    else:
        require(len(matches) <= 1, f"trace {label} aliases are ambiguous")
    return matches[0] if matches else None


def classify(name: str) -> str | None:
    hits = [family for family, patterns in COMPILED.items() if any(pattern.search(name) for pattern in patterns)]
    require(len(hits) <= 1, f"kernel matches multiple P2 families: {name}: {hits}")
    return hits[0] if hits else None


def parse_kernel(path: Path) -> dict[str, Any]:
    intervals: list[tuple[int, int, str, str | None]] = []
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames is not None and len(reader.fieldnames) == len(set(reader.fieldnames)), "kernel trace header differs")
        name_col = column(reader.fieldnames, ("Kernel_Name", "KernelName", "Name", "kernel_name"), "kernel name")
        start_col = column(reader.fieldnames, ("Start_Timestamp", "BeginNs", "start_ns"), "start")
        end_col = column(reader.fieldnames, ("End_Timestamp", "EndNs", "end_ns"), "end")
        phase_col = column(reader.fieldnames, ("Phase", "phase"), "phase", required=False)
        assert name_col and start_col and end_col
        for line_number, row in enumerate(reader, 2):
            require(line_number <= MAX_ROWS + 1 and None not in row, "kernel trace row bound/shape differs")
            try:
                start = int(str(row[start_col]))
                end = int(str(row[end_col]))
            except ValueError as error:
                raise ProfileError(f"kernel timestamps differ at line {line_number}") from error
            require(0 <= start <= end, f"kernel timestamps are invalid at line {line_number}")
            name = str(row[name_col])
            require(name, f"kernel name is empty at line {line_number}")
            intervals.append((start, end, name, str(row[phase_col]) if phase_col and row.get(phase_col) else None))
    require(intervals, "kernel trace has no intervals")
    per_family: dict[str, dict[str, int]] = {family: {"kernel_count": 0, "inclusive_ns": 0} for family in FAMILIES}
    per_family["unclassified"] = {"kernel_count": 0, "inclusive_ns": 0}
    unknown: set[str] = set()
    for start, end, name, _phase in intervals:
        family = classify(name) or "unclassified"
        per_family[family]["kernel_count"] += 1
        per_family[family]["inclusive_ns"] += end - start
        if family == "unclassified":
            unknown.add(name)
    spans = sorted((start, end) for start, end, _name, _phase in intervals)
    union = 0
    current_start, current_end = spans[0]
    for start, end in spans[1:]:
        if start > current_end:
            union += current_end - current_start
            current_start, current_end = start, end
        else:
            current_end = max(current_end, end)
    union += current_end - current_start
    return {
        "path": str(path),
        "sha256": sha(path),
        "kernel_count": len(intervals),
        "gpu_union_ns": union,
        "families": per_family,
        "unknown_kernel_names": sorted(unknown),
        "unclassified_fraction_of_union": per_family["unclassified"]["inclusive_ns"] / union if union else 0.0,
    }


def parse_api(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"status": "not_captured", "launch_count": None, "sync_count": None}
    launch = sync = 0
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames is not None, "HIP API trace header differs")
        name_col = column(reader.fieldnames, ("Name", "API_Name", "ApiName", "name"), "HIP API name")
        assert name_col
        for number, row in enumerate(reader, 2):
            require(number <= MAX_ROWS + 1 and None not in row, "HIP API trace row differs")
            name = str(row[name_col]).lower()
            launch += int("launchkernel" in name)
            sync += int("synchronize" in name or "streamwait" in name)
    return {"status": "captured", "path": str(path), "sha256": sha(path), "launch_count": launch, "sync_count": sync}


def parse_copy(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"status": "not_captured", "transfer_bytes": None}
    total = 0
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames is not None, "copy trace header differs")
        size_col = column(reader.fieldnames, ("Size", "Bytes", "size", "bytes"), "copy size", required=False)
        if size_col is not None:
            for number, row in enumerate(reader, 2):
                require(number <= MAX_ROWS + 1 and None not in row, "copy trace row differs")
                try:
                    size = int(float(str(row[size_col])))
                except ValueError as error:
                    raise ProfileError(f"copy size differs at line {number}") from error
                require(size >= 0, "copy size is negative")
                total += size
    return {"status": "captured", "path": str(path), "sha256": sha(path), "transfer_bytes": total}


def parse(args: argparse.Namespace) -> dict[str, Any]:
    profile_dir = args.profile_dir.absolute()
    require(profile_dir.is_dir() and not profile_dir.is_symlink(), "profile directory must be a real directory")
    window = args.window_result.absolute()
    binding = args.trace_binding.absolute()
    require(window.is_file() and binding.is_file(), "window result/trace binding is unavailable")
    window_value = json.loads(window.read_text(encoding="utf-8"))
    binding_value = json.loads(binding.read_text(encoding="utf-8"))
    require(
        window_value.get("schema_version") == "ullm.aq4_p2_production_baseline_window_result.v1"
        and window_value.get("kind") == "detailed_profile"
        and window_value.get("status") == "partial_observability",
        "profile parser accepts successful detailed-profile windows only",
    )
    require(
        binding_value.get("schema_version") == "ullm.aq4_p2_production_baseline_window_result.v1"
        and binding_value.get("status") == "partial_observability"
        and isinstance(binding_value.get("executor_trace_sha256"), str)
        and isinstance(binding_value.get("executor_record_sidecar_sha256"), str),
        "window trace binding differs",
    )
    kernel_path = trace_file(profile_dir, "kernel_trace")
    require(kernel_path is not None, "kernel trace is not captured")
    kernel = parse_kernel(kernel_path)
    api = parse_api(trace_file(profile_dir, "hip_api_trace"))
    copy = parse_copy(trace_file(profile_dir, "memory_copy_trace"))
    inventory = profile_inventory(profile_dir)
    complete = kernel["unclassified_fraction_of_union"] <= args.maximum_unclassified_fraction
    result = {
        "schema_version": SCHEMA,
        "status": "profiled_diagnostic" if complete else "blocked_unclassified_kernel_time",
        "window": {"result_path": str(window), "result_sha256": sha(window), "trace_binding_path": str(binding), "trace_binding_sha256": sha(binding)},
        "raw_profile": {
            "root": str(profile_dir),
            "members": inventory,
            "member_count": len(inventory),
        },
        "kernel": kernel,
        "launch_sync": api,
        "transfer": copy,
        "workspace": {"status": "not_observed_by_rocprof_kernel_trace"},
        "fallback": {"status": "not_observed_by_rocprof_kernel_trace"},
        "profile_not_used_for_normal_p50_p95": True,
        "maximum_unclassified_fraction": args.maximum_unclassified_fraction,
        "profile_hash_binding": {
            "window_result_sha256": sha(window),
            "executor_trace_binding_sha256": sha(binding),
            "kernel_trace_sha256": kernel["sha256"],
            "hip_api_trace_sha256": api.get("sha256"),
            "memory_copy_trace_sha256": copy.get("sha256"),
        },
    }
    require(not os.path.lexists(args.output), f"profile output already exists: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=True, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return result


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-dir", type=Path, required=True)
    parser.add_argument("--window-result", type=Path, required=True)
    parser.add_argument("--trace-binding", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maximum-unclassified-fraction", type=float, default=0.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        require(0.0 <= args.maximum_unclassified_fraction <= 1.0, "maximum-unclassified-fraction is outside [0,1]")
        result = parse(args)
    except (ProfileError, OSError, ValueError) as error:
        print(f"AQ4 P2 production profile parse failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps({"schema_version": SCHEMA, "status": result["status"]}, ensure_ascii=True, sort_keys=True))
    return 0 if result["status"] == "profiled_diagnostic" else 2


if __name__ == "__main__":
    raise SystemExit(main())
