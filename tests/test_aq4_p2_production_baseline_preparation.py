"""CPU-only contract tests for the new current-identity AQ4 P2 envelope."""

from __future__ import annotations

import importlib.util
import hashlib
import json
import os
import stat
import struct
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / filename)
    assert spec and spec.loader
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


PREPARE = module("aq4_p2_production_prepare", "prepare-aq4-p2-production-baseline.py")
STAGE = module("aq4_p2_production_stage", "stage-aq4-p2-production-baseline-binaries.py")
PATH_ORACLE = module("aq4_p2_production_path_oracle", "run-aq4-p2-production-path-oracle.py")
PROFILE = module("aq4_p2_production_profile", "parse-aq4-p2-production-profile.py")
GUARD_STAGE = module("aq4_p2_guard_stage", "stage-aq4-p2-r9700-guard.py")
REPORT = module("aq4_p2_bottleneck_report", "build-aq4-p2-production-bottleneck-report.py")
SEAL = module("aq4_p2_baseline_jsonl_seal", "seal-aq4-p2-production-baseline-jsonl.py")


def write(path: Path, data: bytes, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    path.chmod(mode)


def initialize_source(root: Path) -> None:
    write(root / "Cargo.lock", b"# fake lock\n")
    write(root / "crates/ullm-engine/Cargo.toml", b"[package]\nname='fixture'\n")
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "-c",
            "user.name=fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )


def active_manifest(tmp_path: Path) -> tuple[Path, Path]:
    product = tmp_path / "product"
    package = product / "package"
    worker = tmp_path / "worker"
    write(worker, b"#!/bin/sh\nexit 0\n", 0o755)
    write(package / "manifest.json", b'{"package":"fixture"}\n')
    write(package / "weights.bin", b"fixture-weight")
    manifest = {
        "format": {"format_id": "AQ4_0", "implementation_id": "qwen35_aq4_rdna4_v1"},
        "worker": {
            "binary": str(worker),
            "binary_sha256": PREPARE.sha_file(worker, "worker"),
            "identity": {"device": "gfx1201", "execution_profile": "rdna4_aq4_resident"},
            "required_environment": ["ULLM_REQUIRE_HIP_AQ4_KERNEL"],
        },
        "product": {"root": str(product), "package": {"manifest_path": "package/manifest.json", "manifest_sha256": PREPARE.sha_file(package / "manifest.json", "package")}},
        "public": {"id": "fixture-aq4", "upstream_id": "Qwen/Qwen3.5-9B", "revision": "fixture-revision"},
    }
    path = tmp_path / "active.json"
    write(path, json.dumps(manifest, sort_keys=True).encode() + b"\n")
    return path, product


def source_model(tmp_path: Path) -> Path:
    model = tmp_path / "model"
    config = {"_name_or_path": "Qwen/Qwen3.5-9B", "_commit_hash": "fixture", "torch_dtype": "bfloat16"}
    index = {"weight_map": {"weight": "model-00001-of-00001.safetensors"}}
    write(model / "config.json", json.dumps(config).encode())
    write(model / "model.safetensors.index.json", json.dumps(index).encode())
    write(model / "model-00001-of-00001.safetensors", b"fixture shard")
    for name in PREPARE.TOKENIZER_CANDIDATES:
        write(model / name, f"fixture {name}".encode())
    return model


def test_preparation_matrix_and_nlink_staging_are_cpu_only(tmp_path: Path) -> None:
    source = tmp_path / "source"
    initialize_source(source)
    manifest, _product = active_manifest(tmp_path)
    model = source_model(tmp_path)
    root = tmp_path / "result-parent"
    root.mkdir()
    output = root / "preparation"
    args = type(
        "Args",
        (),
        {
            "output": output,
            "source_worktree": source,
            "active_manifest": manifest,
            "source_model": model,
        },
    )()
    created = PREPARE.create_preparation(args)
    assert created["status"] == "valid"
    verified = PREPARE.verify_preparation(output)
    assert verified["case_count"] == 133
    assert verified["normal_window_count"] == 14
    assert verified["path_oracle_window_count"] == 8
    oracle_contract = json.loads((output / "oracle-contract.json").read_text())
    assert oracle_contract["source_oracle"]["dtype"] == "float32"
    cases = json.loads((output / "baseline-cases.json").read_text())
    assert sum(case["status"] == "unsupported" for case in cases["cases"]) == 42
    assert sum(case["status"] == "planned" for case in cases["cases"]) == 91
    assert all(case["execution"]["resolved_m"] == 1 for case in cases["cases"] if case["execution"]["mode"] == "all_m1")
    decode_cases = [case for case in cases["cases"] if case["kind"] == "decode"]
    assert len(decode_cases) == 42
    assert {
        (case["execution"]["context_tokens"], case["execution"]["requested_m"], case["execution"]["resolved_m"])
        for case in decode_cases
    } == {(context, width, width) for context in PREPARE.DECODE_CONTEXTS for width in PREPARE.M_GRID}
    assert all(case["m_grid_scope"] == "decode_context_prefill" and case["decode_iteration_token_width"] == 1 for case in decode_cases)
    windows = json.loads((output / "window-plan.json").read_text())["windows"]
    decode_windows = [window for window in windows if window["window_id"].startswith("decode-c")]
    assert [window["window_id"] for window in decode_windows] == [f"decode-c{context}" for context in PREPARE.DECODE_CONTEXTS]
    assert all(len(window["case_ids"]) == len(PREPARE.M_GRID) and not window["unsupported_case_ids"] for window in decode_windows)
    assert stat.S_IMODE((output / "identity.json").stat().st_mode) == 0o444
    assert stat.S_IMODE((output / "staging").stat().st_mode) == 0o700

    binaries = tmp_path / "binaries"
    resident = binaries / "ullm-aq4-p2-resident-driver"
    calibration = binaries / "ullm-aq4-p2-calibration"
    write(resident, b"#!/bin/sh\nexit 0\n", 0o755)
    write(calibration, b"#!/bin/sh\nexit 0\n", 0o755)
    commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    stage_args = type(
        "StageArgs",
        (),
        {
            "output": output / "staging" / "baseline-binaries",
            "preparation": output,
            "resident_source": resident,
            "calibration_source": calibration,
            "source_commit": commit,
        },
    )()
    staged = STAGE.stage(stage_args)
    assert staged["status"] == "valid"
    for name in STAGE.BINARIES:
        item = output / "staging" / "baseline-binaries" / name
        assert item.stat().st_nlink == 1
        assert stat.S_IMODE(item.stat().st_mode) == 0o555


def test_window_dry_run_never_needs_gpu_or_service(tmp_path: Path) -> None:
    source = tmp_path / "source"
    initialize_source(source)
    manifest, _product = active_manifest(tmp_path)
    model = source_model(tmp_path)
    parent = tmp_path / "parent"
    parent.mkdir()
    preparation = parent / "preparation"
    PREPARE.create_preparation(
        type(
            "Args",
            (),
            {"output": preparation, "source_worktree": source, "active_manifest": manifest, "source_model": model},
        )()
    )
    binaries = tmp_path / "binaries"
    resident = binaries / "ullm-aq4-p2-resident-driver"
    calibration = binaries / "ullm-aq4-p2-calibration"
    write(resident, b"#!/bin/sh\nexit 0\n", 0o755)
    write(calibration, b"#!/bin/sh\nexit 0\n", 0o755)
    commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    STAGE.stage(
        type(
            "StageArgs",
            (),
            {"output": preparation / "staging" / "baseline-binaries", "preparation": preparation, "resident_source": resident, "calibration_source": calibration, "source_commit": commit},
        )()
    )
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/run-aq4-p2-production-baseline-window.py"),
            "--preparation",
            str(preparation),
            "--staging",
            str(preparation / "staging/baseline-binaries"),
            "--window",
            "prefill-n128",
            "--output",
            str(preparation / "windows/prefill-n128"),
            "--dry-run",
        ],
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "HIP_VISIBLE_DEVICES": "-1", "ULLM_HIP_VISIBLE_DEVICES": "-1"},
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "dry_run_valid"
    assert payload["gpu_or_service_action"] == "none"
    assert not (preparation / "windows/prefill-n128").exists()

    decode = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/run-aq4-p2-production-baseline-window.py"),
            "--preparation",
            str(preparation),
            "--staging",
            str(preparation / "staging/baseline-binaries"),
            "--window",
            "decode-c16",
            "--output",
            str(preparation / "windows/decode-c16"),
            "--dry-run",
        ],
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "HIP_VISIBLE_DEVICES": "-1", "ULLM_HIP_VISIBLE_DEVICES": "-1"},
    )
    assert decode.returncode == 0, decode.stderr
    assert json.loads(decode.stdout)["case_count"] == len(PREPARE.M_GRID)
    assert not (preparation / "windows/decode-c16").exists()


def test_guard_staging_compiles_an_nlink_one_copy_without_running_guard(tmp_path: Path) -> None:
    source = tmp_path / "source"
    initialize_source(source)
    manifest, _product = active_manifest(tmp_path)
    model = source_model(tmp_path)
    parent = tmp_path / "parent"
    parent.mkdir()
    preparation = parent / "preparation"
    PREPARE.create_preparation(
        type("Args", (), {"output": preparation, "source_worktree": source, "active_manifest": manifest, "source_model": model})()
    )
    guard_source = tmp_path / "query-hip-device-identity.cpp"
    write(guard_source, b"int main() { return 0; }\n")
    compiler = tmp_path / "fake-hipcc.py"
    write(
        compiler,
        b"#!/usr/bin/env python3\nimport os, shutil, sys\nshutil.copyfile(sys.argv[-3], sys.argv[-1])\nos.chmod(sys.argv[-1], 0o755)\n",
        0o755,
    )
    commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    output = preparation / "guard" / "r9700-guard-staging"
    staged = GUARD_STAGE.stage(
        type(
            "GuardArgs",
            (),
            {"output": output, "preparation": preparation, "source": guard_source, "compiler": compiler, "source_commit": commit},
        )()
    )
    assert staged["status"] == "valid"
    binary = output / "query-hip-device-identity"
    assert binary.stat().st_nlink == 1
    assert stat.S_IMODE(binary.stat().st_mode) == 0o555


def test_path_oracle_dry_run_validates_only_cpu_envelope_and_source_receipt(tmp_path: Path) -> None:
    source = tmp_path / "source"
    initialize_source(source)
    manifest, _product = active_manifest(tmp_path)
    model = source_model(tmp_path)
    parent = tmp_path / "parent"
    parent.mkdir()
    preparation = parent / "preparation"
    PREPARE.create_preparation(
        type("Args", (), {"output": preparation, "source_worktree": source, "active_manifest": manifest, "source_model": model})()
    )
    binaries = tmp_path / "binaries"
    resident = binaries / "ullm-aq4-p2-resident-driver"
    calibration = binaries / "ullm-aq4-p2-calibration"
    write(resident, b"#!/bin/sh\nexit 0\n", 0o755)
    write(calibration, b"#!/bin/sh\nexit 0\n", 0o755)
    commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    STAGE.stage(
        type(
            "StageArgs",
            (),
            {"output": preparation / "staging" / "baseline-binaries", "preparation": preparation, "resident_source": resident, "calibration_source": calibration, "source_commit": commit},
        )()
    )
    source_oracle = tmp_path / "source-oracle"
    source_oracle.mkdir()
    write(
        source_oracle / "manifest.json",
        json.dumps(
            {
                "schema_version": "ullm.qwen35_aq4_source_calibration.v1",
                "oracle_kind": "independent_source_full",
                "status": "available",
            },
            sort_keys=True,
        ).encode()
        + b"\n",
    )
    write(source_oracle / "SHA256SUMS", b"fixture\n")
    case_id = json.loads((preparation / "calibration-case-index.json").read_text())["cases"][0]["case_id"]
    output = preparation / "source-oracle" / "target" / case_id
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/run-aq4-p2-production-path-oracle.py"),
            "--preparation",
            str(preparation),
            "--staging",
            str(preparation / "staging/baseline-binaries"),
            "--case-id",
            case_id,
            "--source",
            str(source_oracle),
            "--output",
            str(output),
            "--dry-run",
        ],
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "HIP_VISIBLE_DEVICES": "-1", "ULLM_HIP_VISIBLE_DEVICES": "-1"},
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "dry_run_valid"
    assert payload["gpu_or_service_action"] == "none"
    assert not output.exists()


def test_profile_parser_binds_new_raw_trace_without_running_rocprof(tmp_path: Path) -> None:
    window = tmp_path / "window-result.json"
    binding = tmp_path / "trace-hash-binding.json"
    profile = tmp_path / "profile-raw"
    profile.mkdir()
    write(
        window,
        json.dumps(
            {
                "schema_version": "ullm.aq4_p2_production_baseline_window_result.v1",
                "status": "partial_observability",
                "kind": "detailed_profile",
            }
        ).encode()
        + b"\n",
    )
    write(
        binding,
        json.dumps(
            {
                "schema_version": "ullm.aq4_p2_production_baseline_window_result.v1",
                "status": "partial_observability",
                "executor_trace_sha256": "a" * 64,
                "executor_record_sidecar_sha256": "b" * 64,
            }
        ).encode()
        + b"\n",
    )
    write(profile / "detail_kernel_trace.csv", b"Kernel_Name,Start_Timestamp,End_Timestamp\naq4_matvec_kernel,10,20\n")
    write(profile / "detail_hip_api_trace.csv", b"Name\nhipLaunchKernel\n")
    write(profile / "detail_memory_copy_trace.csv", b"Bytes\n1024\n")
    output = tmp_path / "profile.json"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/parse-aq4-p2-production-profile.py"),
            "--profile-dir",
            str(profile),
            "--window-result",
            str(window),
            "--trace-binding",
            str(binding),
            "--output",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    parsed = json.loads(output.read_text())
    assert parsed["status"] == "profiled_diagnostic"
    assert parsed["kernel"]["families"]["aq4_projection"]["kernel_count"] == 1
    assert parsed["profile_hash_binding"]["memory_copy_trace_sha256"]


def test_oracle_comparator_filters_one_anchor_and_streams_vectors(tmp_path: Path) -> None:
    case_id = "p2-oracle-anchor-prefill-all-m1-n128-m1"
    hidden = struct.pack("<4096f", *([1.0] * 4096))
    logits = struct.pack("<248320f", *([0.25] * 248320))

    def artifact(root: Path, kind: str) -> None:
        (root / "vectors").mkdir(parents=True)
        write(root / "vectors/hidden.f32le", hidden)
        write(root / "vectors/logits.f32le", logits)
        row = {
            "case_id": case_id,
            "step": 0,
            "input_token_ids_sha256": "c" * 64,
            "hidden": {"offset_bytes": 0, "bytes": len(hidden), "elements": 4096, "dtype": "f32", "endianness": "little", "sha256": hashlib.sha256(hidden).hexdigest()},
            "logits": {"offset_bytes": 0, "bytes": len(logits), "elements": 248320, "dtype": "f32", "endianness": "little", "sha256": hashlib.sha256(logits).hexdigest()},
            "greedy_token_id": 0,
            "topk": [{"token_id": item, "logit": 0.25} for item in range(10)],
        }
        write(root / "rows.jsonl", json.dumps(row, sort_keys=True, separators=(",", ":")).encode() + b"\n")
        manifest = {
            "oracle_kind": kind,
            "files": {"rows": "rows.jsonl", "hidden": "vectors/hidden.f32le", "logits": "vectors/logits.f32le"},
        }
        write(root / "manifest.json", json.dumps(manifest, sort_keys=True).encode() + b"\n")
        members = ["manifest.json", "rows.jsonl", "vectors/hidden.f32le", "vectors/logits.f32le"]
        write(
            root / "SHA256SUMS",
            "".join(f"{hashlib.sha256((root / member).read_bytes()).hexdigest()}  {member}\n" for member in members).encode(),
        )

    source = tmp_path / "source"
    target = tmp_path / "target"
    artifact(source, "independent_source_full")
    artifact(target, "aq4_target")
    output = tmp_path / "comparison.json"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/compare-aq4-p2-production-oracles.py"),
            "--source",
            str(source),
            "--target",
            str(target),
            "--case-id",
            case_id,
            "--output",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    comparison = json.loads(output.read_text())
    assert comparison["row_count"] == 1
    assert comparison["case_filter"] == case_id
    assert comparison["rows"][0]["token_agreement"]
    assert comparison["state_snapshot"]["status"] == "not_captured"


def test_source_oracle_preflight_is_cpu_visible_only(tmp_path: Path) -> None:
    source = tmp_path / "source"
    initialize_source(source)
    manifest, _product = active_manifest(tmp_path)
    model = source_model(tmp_path)
    parent = tmp_path / "parent"
    parent.mkdir()
    preparation = parent / "preparation"
    PREPARE.create_preparation(
        type("Args", (), {"output": preparation, "source_worktree": source, "active_manifest": manifest, "source_model": model})()
    )
    output = preparation / "source-oracle" / "source-full"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/capture-aq4-p2-production-source-oracle.py"),
            "--preparation",
            str(preparation),
            "--model-dir",
            str(model),
            "--output",
            str(output),
            "--preflight",
        ],
        check=False,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "CUDA_VISIBLE_DEVICES": "-1",
            "HIP_VISIBLE_DEVICES": "-1",
            "ROCR_VISIBLE_DEVICES": "-1",
            "ULLM_HIP_VISIBLE_DEVICES": "-1",
        },
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "preflight_valid"
    assert payload["gpu_or_service_action"] == "none"
    assert not output.exists()


def test_bottleneck_report_keeps_workspace_fallback_as_blockers(tmp_path: Path) -> None:
    source = tmp_path / "source"
    initialize_source(source)
    manifest, _product = active_manifest(tmp_path)
    model = source_model(tmp_path)
    parent = tmp_path / "parent"
    parent.mkdir()
    preparation = parent / "preparation"
    PREPARE.create_preparation(
        type("Args", (), {"output": preparation, "source_worktree": source, "active_manifest": manifest, "source_model": model})()
    )
    windows = json.loads((preparation / "window-plan.json").read_text())["windows"]
    case_by_id = {
        case["case_id"]: case
        for case in json.loads((preparation / "baseline-cases.json").read_text())["cases"]
    }
    prep_sha = REPORT.sha(preparation / "preparation-manifest.json")

    def window_artifact(window_id: str, case_ids: list[str], kind: str) -> tuple[Path, Path]:
        root = preparation / "windows" / window_id
        root.mkdir()
        sidecar = root / "executor-record-sidecar.jsonl"
        records = []
        for case_id in case_ids:
            execution = case_by_id[case_id]["execution"]
            for run_index in range(10):
                records.append(
                    {
                        "case_id": case_id,
                        "run_index": run_index,
                        "run_kind": "measured",
                        "status": "ok",
                        "end_to_end_ms": 10.0,
                        "prefill_ms": 8.0,
                        "decode_ms": 2.0,
                        "requested_m": execution["requested_m"],
                        "resolved_m": execution["resolved_m"],
                        "actual_token_batch_width": execution["resolved_m"],
                    }
                )
        sidecar.write_text("".join(json.dumps(record, sort_keys=True) + "\n" for record in records))
        trace = root / "executor-trace.jsonl"
        trace.write_text('{"event":"fixture"}\n')
        binding = root / "trace-hash-binding.json"
        binding.write_text(
            json.dumps(
                {
                    "schema_version": "ullm.aq4_p2_production_baseline_window_result.v1",
                    "status": "partial_observability",
                    "preparation_manifest_sha256": prep_sha,
                    "executor_record_sidecar_sha256": REPORT.sha(sidecar),
                    "executor_trace_sha256": REPORT.sha(trace),
                },
                sort_keys=True,
            )
            + "\n"
        )
        result = root / "window-result.json"
        result.write_text(
            json.dumps(
                {
                    "schema_version": "ullm.aq4_p2_production_baseline_window_result.v1",
                    "status": "partial_observability",
                    "window_id": window_id,
                    "kind": kind,
                },
                sort_keys=True,
            )
            + "\n"
        )
        members = ["executor-record-sidecar.jsonl", "executor-trace.jsonl", "trace-hash-binding.json", "window-result.json"]
        (root / "SHA256SUMS").write_text(
            "".join(f"{REPORT.sha(root / member)}  {member}\n" for member in members)
        )
        return result, binding

    for window in windows:
        if window["kind"] != "normal_measurement":
            continue
        window_artifact(window["window_id"], window["case_ids"], "normal_measurement")
    for window in windows:
        if window["kind"] != "detailed_profile":
            continue
        result, binding = window_artifact(window["window_id"], window["case_ids"], "detailed_profile")
        profile = {
            "schema_version": "ullm.aq4_p2_production_baseline_profile.v1",
            "status": "profiled_diagnostic",
            "window": {"result_path": str(result), "trace_binding_path": str(binding)},
            "profile_hash_binding": {"window_result_sha256": REPORT.sha(result), "executor_trace_binding_sha256": REPORT.sha(binding)},
            "raw_profile": {"members": []},
            "kernel": {"families": {"aq4_projection": {"inclusive_ns": 100, "kernel_count": 1}}},
            "launch_sync": {"status": "captured", "launch_count": 3, "sync_count": 1},
            "transfer": {"status": "captured", "transfer_bytes": 64},
            "workspace": {"status": "not_observed"},
            "fallback": {"status": "not_observed"},
        }
        (preparation / "windows" / f"{window['window_id']}-profile.json").write_text(json.dumps(profile, sort_keys=True) + "\n")

    aggregate = preparation / "windows" / "baseline-measurements.jsonl"
    sealed = SEAL.seal(
        type(
            "SealArgs",
            (),
            {"preparation": preparation, "windows_root": preparation / "windows", "output": aggregate},
        )()
    )
    assert sealed["status"] == "valid"
    assert sealed["case_count"] == 91
    assert sealed["measured_row_count"] == 910
    assert SEAL.verify(type("SealArgs", (), {"output": aggregate})())["baseline_jsonl_sha256"] == sealed["baseline_jsonl_sha256"]
    verify = subprocess.run(
        [sys.executable, str(ROOT / "tools/seal-aq4-p2-production-baseline-jsonl.py"), "--output", str(aggregate), "--verify"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert verify.returncode == 0, verify.stderr

    output = tmp_path / "bottleneck-report.json"
    report = REPORT.build(type("ReportArgs", (), {"preparation": preparation, "windows_root": preparation / "windows", "output": output})())
    assert report["status"] == "blocked_missing_required_observability"
    assert report["optimizer_first_family"] is None
    assert report["observability"]["launch_sync"] == "available_from_detailed_rocprof"
    assert report["ranked_bottlenecks"]["kernel_family_inclusive_diagnostic"]["ranked"]
