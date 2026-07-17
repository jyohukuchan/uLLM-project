#!/usr/bin/env python3
"""Build the P2 ranked bottleneck report without inventing missing evidence.

This consumes the bounded sanitized executor sidecars and their hash bindings.
Wall-time p50/p95 and variation are calculated only from measured successful
runs.  Launch/sync, transfer, workspace, and semantic fallback remain hard
blockers until a current-identity detailed profile/trace contributes them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


SCHEMA = "ullm.aq4_p2_production_baseline_bottleneck_report.v1"


class ReportError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ReportError(message)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load(path: Path, label: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ReportError(f"{label} is invalid: {error}") from error


def percentile(values: list[float], fraction: float) -> float:
    require(values, "percentile input is empty")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def sidecar_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ReportError(f"sidecar line {line_number} is invalid: {error}") from error
            require(isinstance(record, dict), f"sidecar line {line_number} is not an object")
            records.append(record)
    return records


def profile_summary(path: Path, preparation: Path) -> dict[str, Any]:
    """Load one external detailed-profile result without reusing its timings.

    Kernel durations in rocprof can overlap across streams.  They are retained
    as explicitly *inclusive diagnostic* family values, never summed into a
    normal wall-time total.
    """

    value = load(path, f"profile {path.name}")
    require(value.get("schema_version") == "ullm.aq4_p2_production_baseline_profile.v1", f"profile schema differs: {path.name}")
    binding = value.get("profile_hash_binding")
    window = value.get("window")
    require(isinstance(binding, dict) and isinstance(window, dict), f"profile binding differs: {path.name}")
    window_path = Path(str(window.get("result_path", "")))
    trace_path = Path(str(window.get("trace_binding_path", "")))
    require(window_path.is_file() and trace_path.is_file(), f"profile window inputs are unavailable: {path.name}")
    require(binding.get("window_result_sha256") == sha(window_path), f"profile window hash differs: {path.name}")
    require(binding.get("executor_trace_binding_sha256") == sha(trace_path), f"profile trace binding hash differs: {path.name}")
    trace = load(trace_path, f"profile trace binding {path.name}")
    require(trace.get("preparation_manifest_sha256") == sha(preparation / "preparation-manifest.json"), f"profile preparation binding differs: {path.name}")
    raw = value.get("raw_profile")
    kernel = value.get("kernel")
    launch_sync = value.get("launch_sync")
    transfer = value.get("transfer")
    require(isinstance(raw, dict) and isinstance(raw.get("members"), list), f"profile raw inventory differs: {path.name}")
    require(isinstance(kernel, dict) and isinstance(kernel.get("families"), dict), f"profile kernel summary differs: {path.name}")
    require(isinstance(launch_sync, dict) and isinstance(transfer, dict), f"profile API/copy summary differs: {path.name}")
    return {
        "profile": path.name,
        "status": value.get("status"),
        "window_id": load(window_path, f"profile window result {path.name}").get("window_id"),
        "profile_sha256": sha(path),
        "raw_profile_member_count": len(raw["members"]),
        "kernel": kernel,
        "launch_sync": launch_sync,
        "transfer": transfer,
        "workspace": value.get("workspace"),
        "fallback": value.get("fallback"),
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    preparation = args.preparation.absolute()
    windows_root = args.windows_root.absolute()
    prep = load(preparation / "preparation-manifest.json", "preparation manifest")
    require(prep.get("schema_version") == "ullm.aq4_p2_production_baseline_preparation.v1" and prep.get("status") == "prepared", "preparation manifest differs")
    measured: dict[str, list[dict[str, Any]]] = defaultdict(list)
    inputs: list[dict[str, Any]] = []
    incomplete: list[str] = []
    for child in sorted(windows_root.iterdir(), key=lambda item: item.name):
        if not child.is_dir() or child.name.endswith("-profile-raw"):
            continue
        result_path = child / "window-result.json"
        binding_path = child / "trace-hash-binding.json"
        sidecar = child / "executor-record-sidecar.jsonl"
        if not result_path.exists() and not binding_path.exists() and not sidecar.exists():
            continue
        require(result_path.is_file() and binding_path.is_file() and sidecar.is_file(), f"window artifact is partial: {child}")
        result = load(result_path, f"window result {child.name}")
        binding = load(binding_path, f"trace binding {child.name}")
        require(binding.get("preparation_manifest_sha256") == sha(preparation / "preparation-manifest.json"), f"preparation hash binding differs: {child.name}")
        require(binding.get("executor_record_sidecar_sha256") == sha(sidecar), f"sidecar hash binding differs: {child.name}")
        inputs.append({"window": child.name, "result_sha256": sha(result_path), "binding_sha256": sha(binding_path), "sidecar_sha256": sha(sidecar), "status": result.get("status")})
        if result.get("status") != "partial_observability":
            incomplete.append(f"window {child.name} status is {result.get('status')!r}")
        for record in sidecar_records(sidecar):
            if record.get("status") == "ok" and record.get("run_kind") == "measured":
                measured[str(record.get("case_id"))].append(record)
    plan = load(preparation / "window-plan.json", "window plan")
    expected_normal = {
        row.get("window_id")
        for row in plan.get("windows", [])
        if isinstance(row, dict) and row.get("kind") == "normal_measurement"
    }
    completed_normal = {
        load(child / "window-result.json", f"window result {child.name}").get("window_id")
        for child in windows_root.iterdir()
        if child.is_dir() and (child / "window-result.json").is_file()
    }
    missing_normal = sorted(item for item in expected_normal if item not in completed_normal)
    incomplete.extend(f"normal window not captured: {item}" for item in missing_normal)
    expected_profiles = {
        row.get("window_id")
        for row in plan.get("windows", [])
        if isinstance(row, dict) and row.get("kind") == "detailed_profile"
    }
    profiles = [profile_summary(path, preparation) for path in sorted(windows_root.glob("*-profile.json"), key=lambda item: item.name)]
    profiled_ids = {item.get("window_id") for item in profiles if item.get("status") == "profiled_diagnostic"}
    missing_profiles = sorted(item for item in expected_profiles if item not in profiled_ids)
    incomplete.extend(f"detailed profile not successfully parsed: {item}" for item in missing_profiles)
    for profile in profiles:
        if profile["status"] != "profiled_diagnostic":
            incomplete.append(f"profile {profile['profile']} status is {profile['status']!r}")
    require(inputs, "no completed window artifact is available")
    cases: list[dict[str, Any]] = []
    for case_id, records in sorted(measured.items()):
        total = [float(record["end_to_end_ms"]) for record in records if isinstance(record.get("end_to_end_ms"), (int, float))]
        prefill = [float(record["prefill_ms"]) for record in records if isinstance(record.get("prefill_ms"), (int, float))]
        decode = [float(record["decode_ms"]) for record in records if isinstance(record.get("decode_ms"), (int, float))]
        if not total:
            incomplete.append(f"case {case_id} has no measured wall-time rows")
            continue
        cases.append(
            {
                "case_id": case_id,
                "measured_run_count": len(total),
                "wall_time_ms": {"p50": percentile(total, 0.50), "p95": percentile(total, 0.95), "mean": statistics.fmean(total), "stdev": statistics.pstdev(total) if len(total) > 1 else 0.0},
                "prefill_ms": {"p50": percentile(prefill, 0.50)} if prefill else None,
                "decode_ms": {"p50": percentile(decode, 0.50)} if decode else None,
                "m_resolution": sorted({(record.get("requested_m"), record.get("resolved_m"), record.get("actual_token_batch_width")) for record in records}),
            }
        )
    ranked_wall = sorted(cases, key=lambda item: item["wall_time_ms"]["p50"], reverse=True)
    profile_families: list[dict[str, Any]] = []
    launch_ranked: list[dict[str, Any]] = []
    transfer_ranked: list[dict[str, Any]] = []
    for profile in profiles:
        kernel = profile["kernel"]
        families = kernel.get("families", {})
        if isinstance(families, dict):
            for family, facts in families.items():
                if not isinstance(facts, dict) or family == "unclassified":
                    continue
                profile_families.append(
                    {
                        "window_id": profile["window_id"],
                        "profile": profile["profile"],
                        "family": family,
                        "inclusive_ns": facts.get("inclusive_ns"),
                        "kernel_count": facts.get("kernel_count"),
                        "diagnostic_only_not_gpu_total": True,
                    }
                )
        api = profile["launch_sync"]
        if api.get("status") == "captured":
            launch_ranked.append({"window_id": profile["window_id"], "profile": profile["profile"], "launch_count": api.get("launch_count"), "sync_count": api.get("sync_count")})
        copy = profile["transfer"]
        if copy.get("status") == "captured":
            transfer_ranked.append({"window_id": profile["window_id"], "profile": profile["profile"], "transfer_bytes": copy.get("transfer_bytes")})
    profile_families.sort(key=lambda item: (-(int(item["inclusive_ns"]) if isinstance(item["inclusive_ns"], int) else -1), str(item["window_id"]), str(item["family"])))
    launch_ranked.sort(key=lambda item: (-(int(item["launch_count"]) if isinstance(item["launch_count"], int) else -1), str(item["window_id"])))
    transfer_ranked.sort(key=lambda item: (-(int(item["transfer_bytes"]) if isinstance(item["transfer_bytes"], int) else -1), str(item["window_id"])))
    observability = {
        "wall_time": "available" if cases else "not_available",
        "launch_sync": "available_from_detailed_rocprof" if launch_ranked else "not_observed",
        "transfer": "available_from_detailed_rocprof" if transfer_ranked else "not_observed",
        "workspace": "not_observed",
        "fallback": "not_observed",
    }
    missing = [
        name
        for name, status in observability.items()
        if status not in {"available", "available_from_detailed_rocprof"}
    ]
    report = {
        "schema_version": SCHEMA,
        "status": "blocked_missing_required_observability" if missing or incomplete else "ready_for_bottleneck_selection",
        "preparation_manifest_sha256": sha(preparation / "preparation-manifest.json"),
        "window_inputs": inputs,
        "detailed_profile_inputs": profiles,
        "cases": cases,
        "ranked_bottlenecks": {
            "wall_time_p50_descending": ranked_wall,
            "kernel_family_inclusive_diagnostic": {"status": "available" if profile_families else "not_observed", "ranked": profile_families, "must_not_sum_as_gpu_total": True},
            "launch_sync": {"status": observability["launch_sync"], "ranked": launch_ranked},
            "transfer": {"status": observability["transfer"], "ranked": transfer_ranked},
            "workspace": {"status": "not_observed", "ranked": []},
            "fallback": {"status": "not_observed", "ranked": []},
        },
        "observability": observability,
        "optimizer_first_family": None,
        "blockers": [
            *incomplete,
            "current-identity detailed profile must supply launch/sync, transfer, workspace, and fallback evidence before selecting the first optimizer family",
        ],
    }
    require(not os.path.lexists(args.output), f"report output already exists: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=True, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--windows-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        result = build(args)
    except (ReportError, OSError, ValueError) as error:
        print(f"AQ4 P2 production bottleneck report failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps({"schema_version": SCHEMA, "status": result["status"], "case_count": len(result["cases"])}, ensure_ascii=True, sort_keys=True))
    return 0 if result["status"] == "ready_for_bottleneck_selection" else 2


if __name__ == "__main__":
    raise SystemExit(main())
