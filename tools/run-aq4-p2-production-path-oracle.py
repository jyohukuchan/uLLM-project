#!/usr/bin/env python3
"""Prepare or execute one current-identity AQ4 P2 full-vector path oracle.

``--dry-run`` is deliberately CPU-only: it validates the sealed P2 envelope,
the nlink=1 calibration staging receipt, and one selected anchor without
opening a model, a HIP device, a service, or the R9700 lock.  ``--execute``
is for the service-user side of one already-guarded service-stop window.  It
only verifies inherited FD 9; lock creation/acquisition belongs exclusively
to the root service-window driver.

The staged calibration binary intentionally captures a single anchor per
process.  That keeps the source/path/state comparison bounded and makes a
failure immutable and local to one single-use window.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any


SCHEMA = "ullm.aq4_p2_production_path_oracle_window.v1"
IDENTITY_SCHEMA = "ullm.aq4_production_p2_identity.v2"
MAX_JSON_BYTES = 16 * 1024 * 1024
SHA_CHUNK = 1024 * 1024


class OracleWindowError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise OracleWindowError(message)


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def sha_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha_file(path: Path, label: str) -> str:
    try:
        before = path.lstat()
    except OSError as error:
        raise OracleWindowError(f"{label} is unavailable: {path}: {error}") from error
    require(stat.S_ISREG(before.st_mode) and not stat.S_ISLNK(before.st_mode), f"{label} must be a regular non-symlink file")
    digest = hashlib.sha256()
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0))
    try:
        while chunk := os.read(descriptor, SHA_CHUNK):
            digest.update(chunk)
    finally:
        os.close(descriptor)
    return digest.hexdigest()


def reject_duplicate(items: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in items:
        if key in result:
            raise OracleWindowError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_json(path: Path, label: str) -> Any:
    try:
        info = path.lstat()
        require(stat.S_ISREG(info.st_mode) and not stat.S_ISLNK(info.st_mode), f"{label} must be a regular non-symlink file")
        require(info.st_size <= MAX_JSON_BYTES, f"{label} exceeds bounded size")
        return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicate, parse_constant=lambda token: (_ for _ in ()).throw(OracleWindowError(f"non-finite JSON token: {token}")))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise OracleWindowError(f"{label} is invalid: {error}") from error


def write_new(path: Path, value: Any, mode: int = 0o600) -> str:
    raw = json.dumps(value, ensure_ascii=True, sort_keys=True, indent=2, allow_nan=False).encode("utf-8") + b"\n"
    require(not os.path.lexists(path), f"refusing to overwrite oracle input: {path}")
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
    return sha_bytes(raw)


def tool_path(name: str) -> Path:
    return Path(__file__).resolve().parent / name


def run_validation(program: Path, arguments: list[str], label: str) -> dict[str, Any]:
    result = subprocess.run([sys.executable, str(program), *arguments], check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise OracleWindowError(f"{label} failed: {result.stderr.strip() or result.stdout.strip()}")
    try:
        value = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise OracleWindowError(f"{label} returned invalid JSON") from error
    require(value.get("status") == "valid", f"{label} did not report valid")
    return value


def load_plan(preparation: Path, staging: Path, case_id: str, served_manifest: Path) -> dict[str, Any]:
    preparation = preparation.absolute()
    staging = staging.absolute()
    preparation_validation = run_validation(
        tool_path("prepare-aq4-p2-production-baseline.py"),
        [
            "--output",
            str(preparation),
            "--verify",
            "--active-manifest",
            str(served_manifest.absolute()),
            "--verify-live-active-identity",
        ],
        "preparation verification",
    )
    staging_validation = run_validation(
        tool_path("stage-aq4-p2-production-baseline-binaries.py"),
        ["--output", str(staging), "--preparation", str(preparation), "--verify"],
        "binary staging verification",
    )
    index = load_json(preparation / "calibration-case-index.json", "calibration case index")
    fixture = load_json(preparation / "oracle-fixture.json", "oracle fixture")
    identity = load_json(preparation / "identity.json", "preparation identity")
    preflight = load_json(preparation / "preflight-template.json", "preflight template")
    contract = load_json(preparation / "oracle-contract.json", "oracle contract")
    require(isinstance(index, dict) and isinstance(index.get("cases"), list), "calibration case index schema differs")
    require(isinstance(fixture, dict) and isinstance(fixture.get("cases"), list), "oracle fixture schema differs")
    require(isinstance(identity, dict) and isinstance(identity.get("deployed_active"), dict), "preparation identity differs")
    require(isinstance(preflight, dict), "preflight template differs")
    require(isinstance(contract, dict) and contract.get("status") == "planned", "oracle contract differs")
    selected = [item for item in index["cases"] if isinstance(item, dict) and item.get("case_id") == case_id]
    require(len(selected) == 1, f"unknown or ambiguous path-oracle anchor: {case_id}")
    selected_fixture = [item for item in fixture["cases"] if isinstance(item, dict) and item.get("case_id") == case_id]
    require(len(selected_fixture) == 1, f"path-oracle fixture differs: {case_id}")
    case = selected[0]
    require(case.get("fixture_id") == case_id and case.get("prefill_requested_m") == 1 and case.get("resolved_m") == 1 and case.get("mode") == "all_m1", "path-oracle case is not all-M=1")
    return {
        "preparation": preparation,
        "staging": staging,
        "case": case,
        "fixture": fixture,
        "identity": identity,
        "preflight": preflight,
        "contract": contract,
        "preparation_validation": preparation_validation,
        "staging_validation": staging_validation,
    }


def validate_source_root(source: Path) -> dict[str, Any]:
    source = source.absolute()
    require(source.is_dir() and not source.is_symlink(), "source oracle root must be a real directory")
    manifest = source / "manifest.json"
    sums = source / "SHA256SUMS"
    require(manifest.is_file() and not manifest.is_symlink() and sums.is_file() and not sums.is_symlink(), "source oracle manifest/SHA256SUMS is unavailable")
    value = load_json(manifest, "source oracle manifest")
    require(isinstance(value, dict) and value.get("schema_version") == "ullm.qwen35_aq4_source_calibration.v1" and value.get("oracle_kind") == "independent_source_full" and value.get("status") == "available", "source oracle is not available independent evidence")
    return {"root": str(source), "manifest_sha256": sha_file(manifest, "source oracle manifest"), "sha256sums_sha256": sha_file(sums, "source oracle SHA256SUMS")}


def validate_source_identity(plan: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    """Reject a stale source identity during the CPU-only preflight.

    The calibration binary performs the complete vector validation later.
    These model and selected-fixture checks keep a known stale source oracle
    from reaching a service-stop window in the first place.
    """
    root = Path(str(source["root"]))
    manifest = load_json(root / "manifest.json", "source oracle manifest")
    require(isinstance(manifest, dict), "source oracle manifest differs")
    identity = manifest.get("identity")
    cases_binding = manifest.get("cases")
    require(isinstance(identity, dict), "source oracle model identity is missing")
    require(isinstance(cases_binding, dict), "source oracle cases binding is missing")
    model = plan["identity"]["deployed_active"].get("model")
    require(isinstance(model, dict), "frozen active model identity differs")
    source_model_id = identity.get("model_id")
    source_model_revision = identity.get("model_revision")
    expected_model_id = model.get("upstream_id")
    expected_model_revision = model.get("revision")
    require(
        isinstance(source_model_id, str)
        and isinstance(source_model_revision, str)
        and isinstance(expected_model_id, str)
        and isinstance(expected_model_revision, str)
        and source_model_id == expected_model_id
        and source_model_revision == expected_model_revision,
        "source oracle model identity differs from frozen active model",
    )
    cases_path = Path(str(cases_binding.get("path", "")))
    require(cases_path.is_absolute(), "source oracle cases path must be absolute")
    cases_sha = sha_file(cases_path, "source oracle cases")
    require(cases_binding.get("sha256") == cases_sha, "source oracle cases hash differs")
    cases = load_json(cases_path, "source oracle cases")
    require(isinstance(cases, dict) and isinstance(cases.get("cases"), list), "source oracle cases differ")
    case_id = plan["case"].get("case_id")
    selected = [item for item in cases["cases"] if isinstance(item, dict) and item.get("case_id") == case_id]
    fixture = plan["fixture"].get("cases")
    require(isinstance(fixture, list), "path-oracle fixture differs")
    selected_fixture = [item for item in fixture if isinstance(item, dict) and item.get("case_id") == case_id]
    require(len(selected) == 1 and len(selected_fixture) == 1, "source oracle does not cover the selected path-oracle anchor")
    require(
        selected[0].get("prompt_token_ids") == selected_fixture[0].get("prompt_token_ids")
        and selected[0].get("step_count") == selected_fixture[0].get("step_count"),
        "source oracle case differs from frozen path-oracle fixture",
    )
    return {
        "model_id": source_model_id,
        "model_revision": source_model_revision,
        "cases_path": str(cases_path),
        "cases_sha256": cases_sha,
        "case_id": case_id,
    }


def validate_execute_environment(identity: dict[str, Any]) -> dict[str, Any]:
    require(os.environ.get("ULLM_P2_PREHELD_LOCK_FD") == "9", "--execute requires inherited root-driver lock FD 9")
    try:
        lock = os.fstat(9)
    except OSError as error:
        raise OracleWindowError(f"pre-held lock FD 9 is unavailable: {error}") from error
    require(stat.S_ISREG(lock.st_mode), "pre-held lock FD 9 is not a regular file")
    require(os.geteuid() != 0, "path-oracle execution must cross the service-user boundary")
    require(os.environ.get("HIP_VISIBLE_DEVICES") == "1" and os.environ.get("ULLM_HIP_VISIBLE_DEVICES") == "1", "R9700-only HIP visibility differs")
    worker = identity["deployed_active"].get("worker")
    require(isinstance(worker, dict) and isinstance(worker.get("required_environment"), list), "active AQ4 guard set differs")
    missing = [name for name in worker["required_environment"] if not isinstance(name, str) or os.environ.get(name) != "1"]
    require(not missing, f"AQ4 guard set is absent: {', '.join(str(name) for name in missing)}")
    return {"lock_fd": 9, "lock_opened_or_acquired_by_executor": False, "hip_visible_devices": "1", "physical_r9700_index": 1, "calibration_requested_device_index": 1}


def target_identity(plan: dict[str, Any], case_sha: str) -> dict[str, Any]:
    active = plan["identity"]["deployed_active"]
    model = active["model"]
    worker = active["worker"]
    package = active["package"]
    package_root = str(Path(str(package["manifest_path"])).parent)
    value: dict[str, Any] = {
        "schema_version": IDENTITY_SCHEMA,
        "status": "bound",
        "identity_sha256": None,
        "model_identity": {
            "id": model["id"],
            "revision": model["revision"],
            "format_id": model["format_id"],
            "implementation_id": model["implementation_id"],
        },
        "expanded_manifest_sha256": case_sha,
        "hash_binding": {
            "bound_case_manifest_sha256": case_sha,
            "served_model_manifest_sha256": active["manifest_sha256"],
            "worker_binary_sha256": worker["sha256"],
            "package_manifest_sha256": package["manifest_sha256"],
            "package_content_sha256": package["tree"]["sha256"],
        },
        "artifacts": {
            "served_model_manifest": active["manifest_path"],
            "worker": worker["path"],
            "package_root": package_root,
        },
        "package_file_count": package["tree"]["file_count"],
    }
    value["identity_sha256"] = sha_bytes(canonical(value))
    return value


def paths_for(plan: dict[str, Any], case_id: str) -> tuple[Path, Path, Path]:
    root = plan["preparation"] / "source-oracle"
    target_parent = root / "target"
    inputs_parent = root / "target-inputs"
    require(case_id and all(char.isalnum() or char in "._-" for char in case_id), "path-oracle case ID is unsafe")
    return target_parent, inputs_parent / case_id, target_parent / case_id


def execute(plan: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    source = validate_source_root(args.source)
    source_identity = validate_source_identity(plan, source)
    environment = validate_execute_environment(plan["identity"])
    target_parent, inputs, target_output = paths_for(plan, args.case_id)
    require(args.output.absolute() == target_output.absolute(), "target output must be preparation/source-oracle/target/CASE_ID")
    require(not os.path.lexists(inputs) and not os.path.lexists(target_output), "path-oracle case was already attempted; immutable retry is forbidden")
    target_parent.mkdir(mode=0o700, exist_ok=True)
    inputs.parent.mkdir(mode=0o700, exist_ok=True)
    require(target_parent.is_dir() and inputs.parent.is_dir() and not target_parent.is_symlink() and not inputs.parent.is_symlink(), "mutable path-oracle directories differ")
    inputs.mkdir(mode=0o700)
    try:
        case_sha = write_new(inputs / "case.json", plan["case"])
        identity = target_identity(plan, case_sha)
        identity_sha = write_new(inputs / "identity.json", identity)
        preflight_sha = write_new(inputs / "preflight.json", plan["preflight"])
        preflight_provenance_sha = write_new(
            inputs / "preflight-provenance.json",
            {
                "schema_version": SCHEMA,
                "status": "partial_observability",
                "weights_bytes": "package-tree byte count from frozen identity",
                "persistent_state_bytes": "not_observed",
                "kv_cache_bytes": "not_observed",
                "workspace_bytes": "not_observed",
                "temporary_bytes": "not_observed",
                "vram_headroom_bytes": "not_observed",
                "gpu_process_snapshot": "service-stop invariant, not a device-memory inventory",
                "zero_placeholders_are_not_measurements": True,
            },
        )
        receipt = {
            "schema_version": SCHEMA,
            "status": "inputs_prepared",
            "case_id": args.case_id,
            "preparation_manifest_sha256": sha_file(plan["preparation"] / "preparation-manifest.json", "preparation manifest"),
            "staging_receipt_sha256": sha_file(plan["staging"] / "staging-receipt.json", "staging receipt"),
            "source": {**source, "identity_validation": source_identity},
            "inputs": {"case_sha256": case_sha, "identity_sha256": identity_sha, "preflight_sha256": preflight_sha, "preflight_provenance_sha256": preflight_provenance_sha, "fixture_sha256": sha_file(plan["preparation"] / "oracle-fixture.json", "oracle fixture")},
            "environment": environment,
        }
        write_new(inputs / "input-receipt.json", receipt)
        command = [
            str(plan["staging"] / "ullm-aq4-p2-calibration"),
            "--served-model-manifest", str(args.served_manifest.absolute()),
            "--fixture", str(plan["preparation"] / "oracle-fixture.json"),
            "--case", str(inputs / "case.json"),
            "--identity", str(inputs / "identity.json"),
            "--preflight", str(inputs / "preflight.json"),
            "--source", str(args.source.absolute()),
            "--output", str(target_output),
            "--case-id", args.case_id,
            "--policy-id", "aq4_p2_production_path_oracle_all_m1",
            "--oracle-kind", "aq4_target",
            "--m", "1",
            "--device-index", "1",
            "--chunk-elements", "65536",
        ]
        result = subprocess.run(command, check=False)
        final = {**receipt, "status": "capture_available" if result.returncode == 0 else "capture_failed_or_blocked", "calibration_exit_code": result.returncode, "target_output": str(target_output), "command": command}
        write_new(inputs / "execution-receipt.json", final)
        return final
    except Exception:
        # Preserve the immutable input directory for diagnosis.  It is never
        # removed or reused automatically after an attempted anchor.
        raise


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--staging", type=Path, required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--served-manifest", type=Path, default=Path("/etc/ullm/served-models/active.json"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm-r9700-window", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        require(args.dry_run != args.execute, "choose exactly one of --dry-run or --execute")
        args.preparation = args.preparation.absolute()
        args.staging = args.staging.absolute()
        args.source = args.source.absolute()
        args.output = args.output.absolute()
        args.served_manifest = args.served_manifest.absolute()
        plan = load_plan(args.preparation, args.staging, args.case_id, args.served_manifest)
        target_parent, inputs, expected_output = paths_for(plan, args.case_id)
        require(args.output == expected_output.absolute(), "output path differs from the selected immutable anchor path")
        if args.dry_run:
            source = validate_source_root(args.source)
            source_identity = validate_source_identity(plan, source)
            result = {
                "schema_version": SCHEMA,
                "status": "dry_run_valid",
                "case_id": args.case_id,
                "source": {**source, "identity_validation": source_identity},
                "planned_input_directory": str(inputs),
                "planned_target_output": str(expected_output),
                "calibration_requested_device_index": 1,
                "gpu_or_service_action": "none",
                "preparation_validation": plan["preparation_validation"],
                "staging_validation": plan["staging_validation"],
            }
        else:
            require(args.confirm_r9700_window, "--execute requires --confirm-r9700-window")
            result = execute(plan, args)
        print(json.dumps(result, ensure_ascii=True, sort_keys=True))
        return 0 if result.get("status") in {"dry_run_valid", "capture_available"} else 1
    except (OracleWindowError, OSError, ValueError, subprocess.SubprocessError) as error:
        print(f"AQ4 P2 production path oracle failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
