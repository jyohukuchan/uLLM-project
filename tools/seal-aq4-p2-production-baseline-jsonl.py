#!/usr/bin/env python3
"""Seal the completed normal P2 windows into one immutable baseline JSONL.

This is a CPU-only post-window reducer.  It accepts no partial matrix: every
normal window must have its own successful, hash-bound sidecar and raw trace.
Only measured sanitized records are copied, in frozen window/case/run order.
The source traces remain in their window directories and their hashes are
retained in the aggregate manifest; no logits, prompts, generated text, or
other unbounded payload is materialized by this tool.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterator


SCHEMA = "ullm.aq4_p2_production_baseline_jsonl.v1"
MAX_LINE_BYTES = 64 * 1024
FORBIDDEN_RECORD_FIELDS = {"prompt_token_ids", "generated_token_ids", "generated_text", "full_logits"}


class SealError(ValueError):
    """The collected baseline evidence is incomplete or does not bind."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SealError(message)


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def sha(path: Path) -> str:
    try:
        info = path.lstat()
    except OSError as error:
        raise SealError(f"file is unavailable: {path}: {error}") from error
    require(stat.S_ISREG(info.st_mode) and not stat.S_ISLNK(info.st_mode), f"regular non-symlink file required: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load(path: Path, label: str) -> Any:
    try:
        raw = path.read_bytes()
        require(len(raw) <= 16 * 1024 * 1024, f"{label} exceeds bounded JSON size")
        return json.loads(raw, object_pairs_hook=reject_duplicates, parse_constant=reject_constant)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise SealError(f"{label} is invalid: {error}") from error


def reject_duplicates(items: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, child in items:
        if key in value:
            raise SealError(f"duplicate JSON key: {key}")
        value[key] = child
    return value


def reject_constant(token: str) -> Any:
    raise SealError(f"non-finite JSON token: {token}")


def parse_sums(path: Path) -> dict[str, str]:
    text = path.read_text(encoding="ascii")
    result: dict[str, str] = {}
    for number, line in enumerate(text.splitlines(), 1):
        digest, separator, name = line.partition("  ")
        require(separator == "  " and len(digest) == 64 and set(digest) <= set("0123456789abcdef"), f"SHA256SUMS line {number} is invalid")
        require(name and "/" not in name and name not in result, f"SHA256SUMS member {number} is invalid")
        result[name] = digest
    require(result, "SHA256SUMS is empty")
    return result


def bound_member(root: Path, sums: dict[str, str], name: str) -> Path:
    require(name in sums, f"window SHA256SUMS omits {name}: {root.name}")
    path = root / name
    require(sha(path) == sums[name], f"window SHA256SUMS hash differs for {name}: {root.name}")
    return path


def records(path: Path, label: str) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            raw = line.encode("utf-8")
            require(0 < len(raw) <= MAX_LINE_BYTES and line.endswith("\n"), f"{label} line {number} is oversized or unterminated")
            try:
                value = json.loads(line, object_pairs_hook=reject_duplicates, parse_constant=reject_constant)
            except json.JSONDecodeError as error:
                raise SealError(f"{label} line {number} is invalid JSON: {error}") from error
            require(isinstance(value, dict), f"{label} line {number} is not an object")
            require(not (FORBIDDEN_RECORD_FIELDS & set(value)), f"{label} line {number} contains an unsanitized field")
            yield value


def write_new(path: Path, raw: bytes, mode: int = 0o444) -> None:
    require(not os.path.lexists(path), f"refusing to overwrite immutable output: {path}")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0), mode)
    try:
        offset = 0
        while offset < len(raw):
            written = os.write(descriptor, raw[offset:])
            require(written > 0, f"short write creating {path}")
            offset += written
        os.fsync(descriptor)
        os.fchmod(descriptor, mode)
    finally:
        os.close(descriptor)


def expected_matrix(preparation: Path) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], str]:
    plan = load(preparation / "window-plan.json", "window plan")
    cases = load(preparation / "baseline-cases.json", "baseline cases")
    require(isinstance(plan, dict) and isinstance(plan.get("windows"), list), "window plan schema differs")
    require(isinstance(cases, dict) and isinstance(cases.get("cases"), list), "baseline case schema differs")
    by_id = {str(case.get("case_id")): case for case in cases["cases"] if isinstance(case, dict) and isinstance(case.get("case_id"), str)}
    require(len(by_id) == len(cases["cases"]), "baseline case IDs differ")
    normal = [window for window in plan["windows"] if isinstance(window, dict) and window.get("kind") == "normal_measurement"]
    require(normal and plan.get("normal_window_count") == len(normal), "normal window plan differs")
    expected_ids: set[str] = set()
    for window in normal:
        require(isinstance(window.get("window_id"), str) and isinstance(window.get("case_ids"), list), "normal window entry differs")
        for case_id in window["case_ids"]:
            require(isinstance(case_id, str) and case_id in by_id and case_id not in expected_ids, "normal case plan differs")
            expected_ids.add(case_id)
            require(by_id[case_id].get("status") == "planned", f"normal case is not planned: {case_id}")
    planned = {case_id for case_id, case in by_id.items() if case.get("status") == "planned"}
    require(expected_ids == planned, "normal windows do not cover exactly the planned matrix")
    return normal, by_id, sha(preparation / "preparation-manifest.json")


def collect(preparation: Path, windows_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    normal, cases, preparation_sha = expected_matrix(preparation)
    rows: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    expected_counts: Counter[str] = Counter()
    input_windows: list[dict[str, Any]] = []
    for window in normal:
        window_id = str(window["window_id"])
        root = windows_root / window_id
        require(root.is_dir() and not root.is_symlink(), f"normal window output is unavailable: {window_id}")
        sums = parse_sums(root / "SHA256SUMS")
        result_path = bound_member(root, sums, "window-result.json")
        binding_path = bound_member(root, sums, "trace-hash-binding.json")
        sidecar_path = bound_member(root, sums, "executor-record-sidecar.jsonl")
        trace_path = bound_member(root, sums, "executor-trace.jsonl")
        sidecar_sha = sha(sidecar_path)
        trace_sha = sha(trace_path)
        result = load(result_path, f"window result {window_id}")
        binding = load(binding_path, f"trace binding {window_id}")
        require(result.get("status") == "partial_observability" and result.get("kind") == "normal_measurement" and result.get("window_id") == window_id, f"normal window result differs: {window_id}")
        require(binding.get("status") == "partial_observability", f"normal trace binding status differs: {window_id}")
        require(binding.get("preparation_manifest_sha256") == preparation_sha, f"normal preparation binding differs: {window_id}")
        require(binding.get("executor_record_sidecar_sha256") == sidecar_sha, f"normal sidecar binding differs: {window_id}")
        require(binding.get("executor_trace_sha256") == trace_sha, f"normal trace binding differs: {window_id}")
        expected_case_ids = [str(case_id) for case_id in window["case_ids"]]
        expected_case_set = set(expected_case_ids)
        local_counts: Counter[str] = Counter()
        for record in records(sidecar_path, f"sidecar {window_id}"):
            if record.get("run_kind") != "measured":
                continue
            require(record.get("status") == "ok", f"normal measured record is not ok: {window_id}")
            case_id = record.get("case_id")
            require(isinstance(case_id, str) and case_id in expected_case_set, f"normal record case differs: {window_id}")
            execution = cases[case_id].get("execution")
            require(isinstance(execution, dict), f"normal case execution differs: {case_id}")
            require(record.get("requested_m") == execution.get("requested_m") and record.get("resolved_m") == execution.get("resolved_m"), f"normal M resolution differs: {case_id}")
            require(record.get("actual_token_batch_width") == execution.get("resolved_m"), f"normal actual M differs: {case_id}")
            local_counts[case_id] += 1
            counts[case_id] += 1
            rows.append(
                {
                    "schema_version": SCHEMA,
                    "preparation_manifest_sha256": preparation_sha,
                    "window_id": window_id,
                    "sidecar_sha256": sidecar_sha,
                    "trace_sha256": trace_sha,
                    "record": record,
                }
            )
        expected_runs = int(window.get("measured_runs_per_case", 0))
        require(expected_runs > 0 and local_counts == Counter({case_id: expected_runs for case_id in expected_case_ids}), f"normal measured run count differs: {window_id}")
        expected_counts.update({case_id: expected_runs for case_id in expected_case_ids})
        input_windows.append(
            {
                "window_id": window_id,
                "window_result_sha256": sha(result_path),
                "trace_hash_binding_sha256": sha(binding_path),
                "executor_record_sidecar_sha256": sha(sidecar_path),
                "executor_trace_sha256": sha(trace_path),
                "measured_rows": sum(local_counts.values()),
            }
        )
    expected_rows = sum(int(window["measured_runs_per_case"]) * len(window["case_ids"]) for window in normal)
    require(len(rows) == expected_rows and counts == expected_counts, "aggregate measured row count differs")
    manifest = {
        "schema_version": SCHEMA,
        "status": "sealed",
        "preparation_manifest_sha256": preparation_sha,
        "normal_window_inputs": input_windows,
        "case_count": len(counts),
        "measured_row_count": len(rows),
        "record_sanitization": "sanitized executor sidecar only; raw traces remain hash-bound inputs",
        "baseline_jsonl_sha256": None,
    }
    return rows, manifest


def paths(output: Path) -> tuple[Path, Path, Path]:
    require(output.name == "baseline-measurements.jsonl", "output basename must be baseline-measurements.jsonl")
    return output, output.with_name("baseline-measurements-manifest.json"), output.with_name("baseline-measurements.SHA256SUMS")


def seal(args: argparse.Namespace) -> dict[str, Any]:
    require(args.preparation is not None and args.windows_root is not None, "--preparation and --windows-root are required when sealing")
    preparation = args.preparation.absolute()
    windows_root = args.windows_root.absolute()
    output = args.output.absolute()
    require(output.parent == windows_root, "aggregate output must be directly under preparation/windows")
    data_path, manifest_path, sums_path = paths(output)
    require(not any(os.path.lexists(path) for path in (data_path, manifest_path, sums_path)), "aggregate output path already exists")
    rows, manifest = collect(preparation, windows_root)
    raw_rows = b"".join(canonical(row) + b"\n" for row in rows)
    manifest["baseline_jsonl_sha256"] = hashlib.sha256(raw_rows).hexdigest()
    raw_manifest = json.dumps(manifest, ensure_ascii=True, sort_keys=True, indent=2, allow_nan=False).encode("utf-8") + b"\n"
    write_new(data_path, raw_rows)
    write_new(manifest_path, raw_manifest)
    sums = f"{sha(data_path)}  {data_path.name}\n{sha(manifest_path)}  {manifest_path.name}\n".encode("ascii")
    write_new(sums_path, sums)
    return verify(args)


def verify(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output.absolute()
    data_path, manifest_path, sums_path = paths(output)
    for path in (data_path, manifest_path, sums_path):
        info = path.lstat()
        require(stat.S_ISREG(info.st_mode) and not stat.S_ISLNK(info.st_mode) and stat.S_IMODE(info.st_mode) == 0o444, f"immutable aggregate member differs: {path.name}")
    manifest = load(manifest_path, "aggregate manifest")
    require(isinstance(manifest, dict) and manifest.get("schema_version") == SCHEMA and manifest.get("status") == "sealed", "aggregate manifest schema differs")
    require(manifest.get("baseline_jsonl_sha256") == sha(data_path), "aggregate JSONL hash differs")
    expected_sums = f"{sha(data_path)}  {data_path.name}\n{sha(manifest_path)}  {manifest_path.name}\n"
    require(sums_path.read_text(encoding="ascii") == expected_sums, "aggregate SHA256SUMS differs")
    return {
        "schema_version": SCHEMA,
        "status": "valid",
        "output": str(data_path),
        "baseline_jsonl_sha256": sha(data_path),
        "case_count": manifest.get("case_count"),
        "measured_row_count": manifest.get("measured_row_count"),
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preparation", type=Path)
    parser.add_argument("--windows-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        result = verify(args) if args.verify else seal(args)
    except (SealError, OSError, UnicodeError, ValueError) as error:
        print(f"AQ4 P2 baseline JSONL sealing failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
