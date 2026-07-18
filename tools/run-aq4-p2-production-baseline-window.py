#!/usr/bin/env python3
"""Validate or execute exactly one prepared AQ4 P2 baseline window.

This program owns neither systemd nor the R9700 lock.  Its ``--execute`` path
is intentionally gated for use *inside* the root-only service-stop driver,
after the driver has performed the R9700-only guard and read-only lock probe.
``--dry-run`` is CPU-only and never starts a binary, HIP runtime, profiler, or
source model.

The resident protocol gives reliable wall-time, M resolution, reset, operation
digest, and terminal-state hash records.  It does not expose raw transfer,
workspace, or semantic-fallback counters.  Those fields are written as
explicit ``not_observed`` values and cannot be promoted into a bottleneck
report until a new detailed profile trace supplies them.
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
from typing import Any, TextIO


PROTOCOL = "ullm.aq4_p2_resident_driver.v2"
P2_IDENTITY_SCHEMA = "ullm.aq4_production_p2_identity.v2"
RUN_SCHEMA = "ullm.aq4_p2_production_baseline_window_result.v1"
MAX_JSON_LINE = 4 * 1024 * 1024
SHA256_CHUNK = 1024 * 1024


class WindowError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise WindowError(message)


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def sha_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha_file(path: Path, label: str) -> str:
    try:
        before = path.lstat()
    except OSError as error:
        raise WindowError(f"{label} is unavailable: {path}: {error}") from error
    require(stat.S_ISREG(before.st_mode) and not stat.S_ISLNK(before.st_mode), f"{label} must be a regular non-symlink file")
    digest = hashlib.sha256()
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0))
    try:
        require(_identity(os.fstat(descriptor)) == _identity(before), f"{label} changed while opening")
        while chunk := os.read(descriptor, SHA256_CHUNK):
            digest.update(chunk)
        require(_identity(os.fstat(descriptor)) == _identity(before), f"{label} changed while reading")
    finally:
        os.close(descriptor)
    require(_identity(path.lstat()) == _identity(before), f"{label} changed after reading")
    return digest.hexdigest()


def _identity(info: os.stat_result) -> tuple[int, int, int, int, int, int, int]:
    return (
        info.st_dev,
        info.st_ino,
        info.st_mode,
        info.st_nlink,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def load_json(path: Path, label: str) -> Any:
    try:
        info = path.lstat()
        require(stat.S_ISREG(info.st_mode) and not stat.S_ISLNK(info.st_mode), f"{label} must be a regular non-symlink file")
        require(info.st_size <= 16 * 1024 * 1024, f"{label} exceeds bounded JSON size")
        raw = path.read_bytes()
        return json.loads(raw, object_pairs_hook=reject_duplicate_keys, parse_constant=reject_constant)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise WindowError(f"{label} is invalid: {error}") from error


def reject_duplicate_keys(items: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, child in items:
        if key in value:
            raise WindowError(f"duplicate JSON key: {key}")
        value[key] = child
    return value


def reject_constant(token: str) -> Any:
    raise WindowError(f"non-finite JSON token: {token}")


def write_new(path: Path, raw: bytes, mode: int = 0o600) -> None:
    if os.path.lexists(path):
        raise WindowError(f"refusing to overwrite window output: {path}")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0), mode)
    try:
        offset = 0
        while offset < len(raw):
            count = os.write(descriptor, raw[offset:])
            require(count > 0, f"short write to {path}")
            offset += count
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_json_new(path: Path, value: Any, mode: int = 0o600) -> None:
    write_new(path, json.dumps(value, ensure_ascii=True, sort_keys=True, indent=2, allow_nan=False).encode("utf-8") + b"\n", mode)


def run_validation(tool: Path, arguments: list[str], label: str) -> dict[str, Any]:
    result = subprocess.run(
        [sys.executable, str(tool), *arguments],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        raise WindowError(f"{label} failed: {result.stderr.strip() or result.stdout.strip()}")
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise WindowError(f"{label} returned invalid JSON: {error}") from error
    require(payload.get("status") == "valid", f"{label} did not report valid")
    return payload


def tool_path(name: str) -> Path:
    return Path(__file__).resolve().parent / name


def load_plan(preparation: Path, staging: Path, window_id: str) -> dict[str, Any]:
    preparation = preparation.absolute()
    staging = staging.absolute()
    preflight = run_validation(
        tool_path("prepare-aq4-p2-production-baseline.py"),
        ["--output", str(preparation), "--verify"],
        "preparation verification",
    )
    stage = run_validation(
        tool_path("stage-aq4-p2-production-baseline-binaries.py"),
        ["--output", str(staging), "--preparation", str(preparation), "--verify"],
        "binary staging verification",
    )
    cases_doc = load_json(preparation / "baseline-cases.json", "baseline cases")
    windows_doc = load_json(preparation / "window-plan.json", "window plan")
    require(isinstance(cases_doc, dict) and isinstance(cases_doc.get("cases"), list), "baseline cases schema differs")
    require(isinstance(windows_doc, dict) and isinstance(windows_doc.get("windows"), list), "window plan schema differs")
    windows = [item for item in windows_doc["windows"] if isinstance(item, dict) and item.get("window_id") == window_id]
    require(len(windows) == 1, f"unknown or ambiguous window ID: {window_id}")
    by_id = {str(case.get("case_id")): case for case in cases_doc["cases"] if isinstance(case, dict) and isinstance(case.get("case_id"), str)}
    require(len(by_id) == len(cases_doc["cases"]), "baseline case IDs differ")
    window = windows[0]
    for case_id in [*window.get("case_ids", []), *window.get("unsupported_case_ids", [])]:
        require(case_id in by_id, f"window references missing case {case_id}")
    return {
        "preparation": preparation,
        "staging": staging,
        "preparation_validation": preflight,
        "staging_validation": stage,
        "cases": by_id,
        "window": window,
    }


def validate_execute_environment(preparation: Path) -> dict[str, Any]:
    # The root driver drops to the service user after it has opened and locked
    # the pre-existing lock.  This process only verifies that inherited FD; it
    # never creates, opens, or acquires /run/ullm/r9700.lock itself.
    raw_fd = os.environ.get("ULLM_P2_PREHELD_LOCK_FD")
    require(raw_fd == "9", "--execute requires the root driver's pre-held lock FD 9")
    try:
        lock_info = os.fstat(9)
    except OSError as error:
        raise WindowError(f"pre-held lock FD 9 is unavailable: {error}") from error
    require(stat.S_ISREG(lock_info.st_mode), "pre-held lock FD 9 is not a regular file")
    require(os.geteuid() != 0, "resident execution must run through the service-user boundary")
    require(os.environ.get("HIP_VISIBLE_DEVICES") == "1", "HIP_VISIBLE_DEVICES must be exactly 1")
    require(os.environ.get("ULLM_HIP_VISIBLE_DEVICES") == "1", "ULLM_HIP_VISIBLE_DEVICES must be exactly 1")
    identity = load_json(preparation / "identity.json", "preparation identity")
    guards = identity.get("deployed_active", {}).get("worker", {}).get("required_environment")
    require(isinstance(guards, list) and guards and all(isinstance(name, str) and name for name in guards), "active required HIP guard set is unavailable")
    missing = sorted(name for name in guards if os.environ.get(name) != "1")
    require(not missing, f"required AQ4 HIP guards are absent: {', '.join(missing)}")
    return {
        "required_guard_count": len(guards),
        "hip_visible_devices": "1",
        "filtered_hip_ordinal": 0,
        "lock_fd": 9,
        "lock_opened_or_acquired_by_executor": False,
    }


def build_resident_execution(case: dict[str, Any]) -> dict[str, Any]:
    execution = case["execution"]
    mode = str(execution["mode"])
    if mode not in {"all_m1", "cold_batched"}:
        mode = "all_m1" if int(execution["requested_m"]) == 1 else "cold_batched"
    return {
        "scope": "full_model",
        "phase": "cold_prefill",
        "mode": mode,
        "prompt_tokens": int(execution["prompt_tokens"]),
        "cached_prefix_tokens": 0,
        "context_tokens": int(execution["context_tokens"]),
        "generated_tokens": int(execution["generated_tokens"]),
        "request_count": 1,
        "requested_m": int(execution["requested_m"]),
        "resolved_m": int(execution["resolved_m"]),
        "sampling": {"mode": "greedy", "temperature": 0.0, "top_p": 1.0, "top_k": 1, "seed": 0},
        "control": {
            "control_id": "aq4_0_target",
            "role": "target",
            "format_id": "AQ4_0",
            "implementation_id": "qwen35_aq4_rdna4_v1",
            "promotion_eligible": True,
        },
    }


def write_runtime_identity(
    output: Path,
    ready: dict[str, Any],
    case_binding: Path,
) -> Path:
    resident = ready.get("driver_identity")
    require(isinstance(resident, dict), "resident driver ready event lacks driver identity")
    required = (
        "worker_binary_sha256",
        "package_manifest_sha256",
        "package_content_sha256",
        "served_model_manifest_sha256",
    )
    require(all(isinstance(resident.get(key), str) and len(str(resident[key])) == 64 for key in required), "resident driver identity hash set differs")
    case_sha = sha_file(case_binding, "resident case binding")
    value: dict[str, Any] = {
        "schema_version": P2_IDENTITY_SCHEMA,
        "status": "bound",
        "identity_sha256": None,
        "resident_driver_identity": resident,
        "expanded_manifest_sha256": case_sha,
        "hash_binding": {
            "bound_case_manifest_sha256": case_sha,
            "worker_binary_sha256": resident["worker_binary_sha256"],
            "package_manifest_sha256": resident["package_manifest_sha256"],
            "package_content_sha256": resident["package_content_sha256"],
            "served_model_manifest_sha256": resident["served_model_manifest_sha256"],
        },
    }
    value["identity_sha256"] = sha_bytes(canonical(value))
    path = output / "runtime-identity.json"
    write_json_new(path, value)
    return path


def _read_event(stream: TextIO, expected: str) -> dict[str, Any]:
    line = stream.readline(MAX_JSON_LINE + 1)
    require(line, f"resident driver ended before {expected}")
    require(len(line.encode("utf-8")) <= MAX_JSON_LINE and line.endswith("\n"), f"resident driver emitted oversized/unterminated {expected} event")
    try:
        value = json.loads(line, object_pairs_hook=reject_duplicate_keys, parse_constant=reject_constant)
    except json.JSONDecodeError as error:
        raise WindowError(f"resident driver emitted invalid {expected} JSON: {error}") from error
    require(isinstance(value, dict) and value.get("schema_version") == PROTOCOL and value.get("event") == expected, f"resident driver event differs: expected {expected}")
    return value


def _send(command_stream: TextIO, value: dict[str, Any]) -> None:
    raw = canonical(value).decode("utf-8") + "\n"
    command_stream.write(raw)
    command_stream.flush()


def sanitized_run(event: dict[str, Any]) -> dict[str, Any]:
    timing = event.get("timing") if isinstance(event.get("timing"), dict) else {}
    audit = event.get("audit") if isinstance(event.get("audit"), dict) else {}
    state = event.get("state") if isinstance(event.get("state"), dict) else {}
    return {
        "case_id": event.get("case_id"),
        "run_index": event.get("run_index"),
        "run_kind": event.get("run_kind"),
        "status": event.get("status"),
        "requested_m": event.get("requested_m"),
        "resolved_m": event.get("resolved_m"),
        "actual_token_batch_width": event.get("actual_token_batch_width"),
        "end_to_end_ms": timing.get("end_to_end_ms"),
        "prefill_ms": timing.get("prefill_ms"),
        "decode_ms": timing.get("decode_ms"),
        "operation_digest_sha256": audit.get("deterministic_digest_sha256"),
        "request_state_sha256": state.get("request_state_sha256"),
        "fallback_status": "not_observed_by_resident_protocol",
        "workspace_status": "not_observed_by_resident_protocol",
        "transfer_status": "not_observed_by_resident_protocol",
        "launch_sync_status": "not_observed_by_resident_protocol",
    }


def produce_sums(root: Path) -> None:
    members = sorted(
        [path for path in root.iterdir() if path.is_file() and path.name != "SHA256SUMS"],
        key=lambda path: path.name,
    )
    write_new(root / "SHA256SUMS", "".join(f"{sha_file(path, path.name)}  {path.name}\n" for path in members).encode("ascii"))


def execute(plan: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    preparation: Path = plan["preparation"]
    staging: Path = plan["staging"]
    window: dict[str, Any] = plan["window"]
    environment = validate_execute_environment(preparation)
    require(window.get("kind") in {"normal_measurement", "detailed_profile"}, "window kind cannot be executed by the resident timing executor")
    output = args.output.absolute()
    require(output.parent == (preparation / "windows").absolute(), "window output must be directly under preparation/windows")
    require(not os.path.lexists(output), f"window output already exists: {output}")
    output.mkdir(mode=0o700)
    raw_path = output / "executor-trace.jsonl"
    sidecar_path = output / "executor-record-sidecar.jsonl"
    stderr_path = output / "resident-driver.stderr"
    unsupported_path = output / "unsupported-cases.jsonl"
    preflight_template = load_json(preparation / "preflight-template.json", "preflight template")
    require(isinstance(preflight_template, dict), "preflight template differs")
    # The exact schema accepted by the resident driver has no provenance field.
    # Keep provenance separate so zero placeholders cannot masquerade as device
    # observations in a later bottleneck report.
    write_json_new(output / "preflight.json", preflight_template)
    write_json_new(
        output / "preflight-provenance.json",
        {
            "schema_version": RUN_SCHEMA,
            "status": "partial_observability",
            "weights_bytes": "package-tree byte count from prepared identity",
            "persistent_state_bytes": "not_observed",
            "kv_cache_bytes": "not_observed",
            "workspace_bytes": "not_observed",
            "temporary_bytes": "not_observed",
            "vram_headroom_bytes": "not_observed",
            "gpu_process_snapshot": "service-stop window invariant, not an AMD-SMI inventory",
        },
    )
    with unsupported_path.open("x", encoding="utf-8") as unsupported:
        for case_id in window["unsupported_case_ids"]:
            case = plan["cases"][case_id]
            record = {
                "case_id": case_id,
                "status": "unsupported",
                "feature": case.get("unsupported", {}).get("feature"),
                "reason": case.get("unsupported", {}).get("reason"),
                "must_not_be_counted_as_success": True,
            }
            unsupported.write(json.dumps(record, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n")
        unsupported.flush()
        os.fsync(unsupported.fileno())
    case_binding = preparation / "resident-case-binding.json"
    fixture = preparation / "resident-fixture.json"
    policy = preparation / "policy.json"
    manifest = args.served_manifest.absolute()
    command = [
        str(staging / "ullm-aq4-p2-resident-driver"),
        "--served-model-manifest",
        str(manifest),
        "--device-index",
        "1",
        "--build-git-commit",
        str(load_json(preparation / "identity.json", "preparation identity")["clean_baseline_source"]["git_commit"]),
    ]
    all_events: list[dict[str, Any]] = []
    sanitized: list[dict[str, Any]] = []
    failed = False
    failure: str | None = None
    with raw_path.open("x", encoding="utf-8") as raw, sidecar_path.open("x", encoding="utf-8") as sidecar, stderr_path.open("xb") as stderr:
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=stderr,
            text=True,
            bufsize=1,
            env=dict(os.environ),
        )
        try:
            require(process.stdin is not None and process.stdout is not None, "resident driver pipes are unavailable")
            ready = _read_event(process.stdout, "ready")
            all_events.append(ready)
            raw.write(json.dumps(ready, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n")
            raw.flush()
            runtime_identity = write_runtime_identity(output, ready, case_binding)
            links = {
                "case_binding": {"path": str(case_binding), "sha256": sha_file(case_binding, "resident case binding")},
                "identity": {"path": str(runtime_identity), "sha256": sha_file(runtime_identity, "runtime identity")},
                "preflight": {"path": str(output / "preflight.json"), "sha256": sha_file(output / "preflight.json", "runtime preflight")},
                "policy": {"path": str(policy), "sha256": sha_file(policy, "policy")},
                "fixture": {"path": str(fixture), "sha256": sha_file(fixture, "resident fixture")},
            }
            for case_id in window["case_ids"]:
                case = plan["cases"][case_id]
                require(case.get("status") == "planned", f"executor window includes a non-planned case: {case_id}")
                execution = build_resident_execution(case)
                _send(
                    process.stdin,
                    {
                        "command": "case_begin",
                        "schema_version": PROTOCOL,
                        "case_id": case_id,
                        "case_sha256": next(
                            item["case_sha256"]
                            for item in load_json(case_binding, "resident case binding")["cases"]
                            if item["case_id"] == case_id
                        ),
                        "case_binding": links["case_binding"],
                        "identity": links["identity"],
                        "preflight": links["preflight"],
                        "policy": links["policy"],
                        "fixture": links["fixture"],
                        "execution": execution,
                    },
                )
                event = _read_event(process.stdout, "case_ready")
                all_events.append(event)
                raw.write(json.dumps(event, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n")
                for index in range(int(window["warmup_runs_per_case"]) + int(window["measured_runs_per_case"])):
                    kind = "warmup" if index < int(window["warmup_runs_per_case"]) else "measured"
                    _send(process.stdin, {"command": "run", "schema_version": PROTOCOL, "case_id": case_id, "run_index": index, "run_kind": kind})
                    event = _read_event(process.stdout, "run_complete")
                    all_events.append(event)
                    raw.write(json.dumps(event, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n")
                    safe = sanitized_run(event)
                    sanitized.append(safe)
                    sidecar.write(json.dumps(safe, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n")
                    if event.get("status") != "ok":
                        failed = True
                        failure = f"resident run failed: {case_id} index {index} status {event.get('status')}"
                        break
                if failed:
                    _send(process.stdin, {"command": "cancel", "schema_version": PROTOCOL, "case_id": case_id, "reason": "terminal_run_failure"})
                    event = _read_event(process.stdout, "cancel_complete")
                else:
                    _send(process.stdin, {"command": "case_end", "schema_version": PROTOCOL, "case_id": case_id})
                    event = _read_event(process.stdout, "case_complete")
                all_events.append(event)
                raw.write(json.dumps(event, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n")
                raw.flush()
                sidecar.flush()
                if failed:
                    break
            _send(process.stdin, {"command": "shutdown", "schema_version": PROTOCOL})
            process.stdin.close()
            return_code = process.wait(timeout=args.timeout)
            if return_code != 0 and failure is None:
                failed = True
                failure = f"resident driver exited {return_code}"
        except Exception as error:
            failed = True
            failure = str(error)
            process.kill()
            process.wait()
        finally:
            raw.flush()
            sidecar.flush()
    binding = {
        "schema_version": RUN_SCHEMA,
        "status": "partial_observability" if not failed else "failed",
        "window_id": window["window_id"],
        "preparation_manifest_sha256": sha_file(preparation / "preparation-manifest.json", "preparation manifest"),
        "staging_receipt_sha256": sha_file(staging / "staging-receipt.json", "staging receipt"),
        "executor_trace_sha256": sha_file(raw_path, "executor trace"),
        "executor_record_sidecar_sha256": sha_file(sidecar_path, "executor sidecar"),
        "run_count": len(sanitized),
        "observability": {
            "wall_time": "available",
            "m_resolution": "available",
            "state_snapshot": "terminal generated-token sequence hash only",
            "launch_sync": "not_observed",
            "transfer": "not_observed",
            "workspace": "not_observed",
            "semantic_fallback": "not_observed",
        },
    }
    write_json_new(output / "trace-hash-binding.json", binding)
    result = {
        "schema_version": RUN_SCHEMA,
        "status": "failed" if failed else "partial_observability",
        "window_id": window["window_id"],
        "kind": window["kind"],
        "executed_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "environment": environment,
        "case_count": len(window["case_ids"]),
        "unsupported_case_ids": window["unsupported_case_ids"],
        "failure": failure,
        "trace_hash_binding": "trace-hash-binding.json",
        "detailed_profile_status": "external_rocprof_required_and_bound_by_service_driver" if window["kind"] == "detailed_profile" else "not_requested",
    }
    write_json_new(output / "window-result.json", result)
    produce_sums(output)
    return result


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--staging", type=Path, required=True)
    parser.add_argument("--window", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--served-manifest", type=Path, default=Path("/etc/ullm/served-models/active.json"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm-r9700-window", action="store_true")
    parser.add_argument("--timeout", type=float, default=14_400.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        require(args.dry_run != args.execute, "choose exactly one of --dry-run or --execute")
        require(args.timeout > 0, "timeout must be positive")
        plan = load_plan(args.preparation, args.staging, args.window)
        if args.dry_run:
            result = {
                "schema_version": RUN_SCHEMA,
                "status": "dry_run_valid",
                "window_id": plan["window"]["window_id"],
                "window_kind": plan["window"]["kind"],
                "case_count": len(plan["window"]["case_ids"]),
                "unsupported_case_count": len(plan["window"]["unsupported_case_ids"]),
                "gpu_or_service_action": "none",
                "preparation_validation": plan["preparation_validation"],
                "staging_validation": plan["staging_validation"],
            }
        else:
            require(args.confirm_r9700_window, "--execute requires --confirm-r9700-window")
            result = execute(plan, args)
        print(json.dumps(result, ensure_ascii=True, sort_keys=True))
        return 0 if result.get("status") in {"dry_run_valid", "partial_observability"} else 1
    except (WindowError, OSError, ValueError, subprocess.SubprocessError) as error:
        print(f"AQ4 P2 production baseline window failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
