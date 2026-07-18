#!/usr/bin/env python3
"""Create and verify the CPU-only preparation envelope for AQ4 P2 baseline runs.

This is deliberately a *new* P2 preparation format.  It does not import an
old P2 matrix or turn any previous diagnostic result into promotion evidence.
It freezes a clean source commit, the currently deployed served-model manifest
as a separate identity, the source checkpoint, deterministic representative
fixtures, and a bounded R9700-only execution plan.

No subcommand opens HIP, queries a GPU, touches systemd, or accesses the
runtime lock.  The root-only window driver is the only consumer intended to
execute the later GPU stage.
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
from typing import Any, Iterable


SCHEMA = "ullm.aq4_p2_production_baseline_preparation.v1"
IDENTITY_SCHEMA = "ullm.aq4_p2_production_baseline_identity.v1"
CASE_SCHEMA = "ullm.aq4_p2_production_baseline_case.v1"
WINDOW_SCHEMA = "ullm.aq4_p2_production_baseline_window_plan.v1"
ORACLE_WINDOW_SCHEMA = "ullm.aq4_p2_production_path_oracle_window_plan.v1"
ORACLE_SCHEMA = "ullm.aq4_p2_production_baseline_oracle_contract.v1"
PROFILE_SCHEMA = "ullm.aq4_p2_production_baseline_profile_plan.v1"
EXECUTOR_SCHEMA = "ullm.aq4_p2_production_baseline_executor_record.v1"
SOURCE_CASES_SCHEMA = "ullm.qwen35_aq4_source_calibration_cases.v1"
RESIDENT_FIXTURE_SCHEMA = "ullm.aq4_p2_case_fixture.v1"
P2_IDENTITY_SCHEMA = "ullm.aq4_production_p2_identity.v2"

PROMPT_LENGTHS = (128, 512, 1011, 1024, 1339, 2048, 3584)
DECODE_CONTEXTS = (16, 128, 512, 1024, 1339, 2048, 3584)
M_GRID = (1, 8, 16, 32, 64, 128)
TOKEN_DOMAIN = b"ullm.aq4_p2_production_baseline_fixture.v1\0"
CHECKPOINT_INDEX = "model.safetensors.index.json"
CHECKPOINT_CONFIG = "config.json"
TOKENIZER_CANDIDATES = (
    "chat_template.jinja",
    "merges.txt",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
)
MAX_JSON_BYTES = 16 * 1024 * 1024
MAX_CASES = 256
MAX_TOKEN_COUNT = 4096
SHA256_RE = set("0123456789abcdef")


class PreparationError(ValueError):
    """A preparation input is unsuitable for an immutable P2 baseline."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PreparationError(message)


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def canonical_line(value: Any) -> bytes:
    return canonical(value) + b"\n"


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha_file(path: Path, label: str) -> str:
    info = regular(path, label)
    before = file_identity(info)
    digest = hashlib.sha256()
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        require(file_identity(os.fstat(descriptor)) == before, f"{label} changed while opening")
        while chunk := os.read(descriptor, 1024 * 1024):
            digest.update(chunk)
        require(file_identity(os.fstat(descriptor)) == before, f"{label} changed while reading")
    finally:
        os.close(descriptor)
    require(file_identity(regular(path, label)) == before, f"{label} changed after reading")
    return digest.hexdigest()


def file_identity(info: os.stat_result) -> tuple[int, int, int, int, int, int, int]:
    return (
        info.st_dev,
        info.st_ino,
        info.st_mode,
        info.st_nlink,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def regular(path: Path, label: str, *, maximum: int | None = None) -> os.stat_result:
    try:
        info = path.lstat()
    except OSError as error:
        raise PreparationError(f"{label} is unavailable: {path}: {error}") from error
    require(stat.S_ISREG(info.st_mode) and not stat.S_ISLNK(info.st_mode), f"{label} must be a regular non-symlink file: {path}")
    require(info.st_size > 0, f"{label} is empty: {path}")
    if maximum is not None:
        require(info.st_size <= maximum, f"{label} exceeds {maximum} bytes: {path}")
    return info


def directory(path: Path, label: str) -> os.stat_result:
    try:
        info = path.lstat()
    except OSError as error:
        raise PreparationError(f"{label} is unavailable: {path}: {error}") from error
    require(stat.S_ISDIR(info.st_mode) and not stat.S_ISLNK(info.st_mode), f"{label} must be a real directory: {path}")
    return info


def load_json(path: Path, label: str, *, maximum: int = MAX_JSON_BYTES) -> Any:
    regular(path, label, maximum=maximum)
    try:
        raw = path.read_bytes()
        return json.loads(
            raw,
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=lambda token: (_ for _ in ()).throw(
                PreparationError(f"{label} contains non-finite JSON token {token!r}")
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise PreparationError(f"{label} is not valid JSON: {error}") from error


def reject_duplicate_keys(items: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, child in items:
        if key in value:
            raise PreparationError(f"duplicate JSON key: {key}")
        value[key] = child
    return value


def write_new(path: Path, raw: bytes, mode: int = 0o444) -> None:
    if os.path.lexists(path):
        raise PreparationError(f"refusing to overwrite preparation member: {path}")
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
        mode,
    )
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


def fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def run_git(repo: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *arguments],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        raise PreparationError(
            f"git {' '.join(arguments)} failed: {result.stderr.strip() or result.stdout.strip()}"
        )
    return result.stdout.strip()


def git_identity(repo: Path) -> dict[str, Any]:
    directory(repo, "source worktree")
    commit = run_git(repo, "rev-parse", "HEAD")
    tree = run_git(repo, "rev-parse", "HEAD^{tree}")
    require(len(commit) == 40 and set(commit) <= SHA256_RE, "source HEAD is not a lowercase SHA-1")
    require(len(tree) == 40 and set(tree) <= SHA256_RE, "source tree is not a lowercase SHA-1")
    dirty = run_git(repo, "status", "--porcelain", "--untracked-files=no")
    require(not dirty, "source worktree has tracked modifications; use a detached clean worktree")
    cargo_lock = repo / "Cargo.lock"
    return {
        "git_commit": commit,
        "git_tree": tree,
        "tracked_worktree_clean": True,
        "worktree": str(repo.resolve()),
        "cargo_lock_sha256": sha_file(cargo_lock, "Cargo.lock"),
        "engine_cargo_toml_sha256": sha_file(repo / "crates/ullm-engine/Cargo.toml", "engine Cargo.toml"),
    }


def safe_relative(root: Path, raw: str, label: str) -> Path:
    candidate = Path(raw)
    require(bool(raw) and not candidate.is_absolute(), f"{label} must be a non-empty relative path")
    require(all(part not in {"", ".", ".."} for part in candidate.parts), f"{label} is not a safe relative path")
    path = root / candidate
    regular(path, label)
    return path


def package_tree_identity(root: Path) -> dict[str, Any]:
    directory(root, "AQ4 package root")
    entries: list[dict[str, Any]] = []
    for current, directories, files in os.walk(root, followlinks=False):
        directories.sort()
        files.sort()
        current_path = Path(current)
        for name in directories:
            info = (current_path / name).lstat()
            require(not stat.S_ISLNK(info.st_mode), f"AQ4 package contains a symlink directory: {current_path / name}")
        for name in files:
            path = current_path / name
            info = regular(path, "AQ4 package file")
            relative = path.relative_to(root).as_posix()
            entries.append({"file": relative, "bytes": info.st_size, "sha256": sha_file(path, "AQ4 package file")})
    require(entries, "AQ4 package root is empty")
    entries.sort(key=lambda item: item["file"])
    aggregate = hashlib.sha256()
    total = 0
    for entry in entries:
        relative = entry["file"].encode("utf-8")
        digest = bytes.fromhex(entry["sha256"])
        aggregate.update(len(relative).to_bytes(8, "little"))
        aggregate.update(relative)
        aggregate.update(int(entry["bytes"]).to_bytes(8, "little"))
        aggregate.update(digest)
        total += int(entry["bytes"])
    return {
        "root": str(root.resolve()),
        "sha256": aggregate.hexdigest(),
        "file_count": len(entries),
        "bytes": total,
        "manifest_entries": entries,
    }


def inspect_active_manifest(path: Path) -> dict[str, Any]:
    payload = load_json(path, "active served-model manifest")
    require(isinstance(payload, dict), "active served-model manifest root must be an object")
    fmt = payload.get("format")
    worker = payload.get("worker")
    product = payload.get("product")
    public = payload.get("public")
    require(isinstance(fmt, dict) and fmt.get("format_id") == "AQ4_0", "active manifest format must be AQ4_0")
    require(isinstance(fmt, dict) and fmt.get("implementation_id") == "qwen35_aq4_rdna4_v1", "active manifest implementation differs")
    require(isinstance(worker, dict) and isinstance(worker.get("identity"), dict), "active manifest worker identity is missing")
    require(worker["identity"].get("device") == "gfx1201", "active manifest worker is not gfx1201")
    require(worker["identity"].get("execution_profile") == "rdna4_aq4_resident", "active manifest execution profile differs")
    require(isinstance(product, dict) and isinstance(product.get("root"), str), "active manifest product root is missing")
    require(isinstance(product.get("package"), dict), "active manifest package binding is missing")
    require(isinstance(public, dict), "active manifest public identity is missing")
    package_path = Path(product["root"]) / str(product["package"].get("manifest_path", ""))
    worker_path = Path(str(worker.get("binary", "")))
    regular(package_path, "active package manifest")
    regular(worker_path, "active worker binary")
    package = package_tree_identity(package_path.parent)
    promotion = payload.get("promotion", product.get("promotion"))
    return {
        "manifest_path": str(path.resolve()),
        "manifest_sha256": sha_file(path, "active served-model manifest"),
        "worker": {
            "path": str(worker_path.resolve()),
            "sha256": sha_file(worker_path, "active worker binary"),
            "manifest_sha256": worker.get("binary_sha256"),
            "identity": worker["identity"],
            "required_environment": worker.get("required_environment"),
        },
        "package": {
            "manifest_path": str(package_path.resolve()),
            "manifest_sha256": sha_file(package_path, "active package manifest"),
            "manifest_declared_sha256": product["package"].get("manifest_sha256"),
            "tree": package,
        },
        "model": {
            "id": public.get("id"),
            "upstream_id": public.get("upstream_id"),
            "revision": public.get("revision"),
            "format_id": fmt.get("format_id"),
            "implementation_id": fmt.get("implementation_id"),
        },
        "promotion": promotion,
    }


def source_checkpoint_identity(model_dir: Path) -> dict[str, Any]:
    directory(model_dir, "source checkpoint root")
    config = load_json(model_dir / CHECKPOINT_CONFIG, "source checkpoint config")
    index = load_json(model_dir / CHECKPOINT_INDEX, "source checkpoint index")
    require(isinstance(config, dict), "source checkpoint config root differs")
    require(isinstance(index, dict) and isinstance(index.get("weight_map"), dict), "source checkpoint weight index differs")
    shards = sorted(set(index["weight_map"].values()))
    require(shards and all(isinstance(name, str) for name in shards), "source checkpoint shard list differs")
    checkpoint_files: list[dict[str, Any]] = []
    for name in [CHECKPOINT_CONFIG, CHECKPOINT_INDEX, *shards]:
        path = safe_relative(model_dir, name, "source checkpoint member")
        info = regular(path, "source checkpoint member")
        checkpoint_files.append({"file": name, "bytes": info.st_size, "sha256": sha_file(path, "source checkpoint member")})
    checkpoint_files.sort(key=lambda item: item["file"])
    tokenizer_files: list[dict[str, Any]] = []
    for name in TOKENIZER_CANDIDATES:
        path = model_dir / name
        if path.exists():
            info = regular(path, "source tokenizer member")
            tokenizer_files.append({"file": name, "bytes": info.st_size, "sha256": sha_file(path, "source tokenizer member")})
    require(tokenizer_files, "source tokenizer identity has no supported files")
    tokenizer_files.sort(key=lambda item: item["file"])
    return {
        "root": str(model_dir.resolve()),
        "model_id": config.get("_name_or_path"),
        "model_revision": config.get("_commit_hash"),
        "dtype": str(config.get("torch_dtype", "bfloat16")),
        "source_checkpoint": {
            "root": str(model_dir.resolve()),
            "files": checkpoint_files,
            "aggregate_sha256": sha_bytes(canonical(checkpoint_files) + b"\n"),
        },
        "tokenizer": {
            "root": str(model_dir.resolve()),
            "files": tokenizer_files,
            "aggregate_sha256": sha_bytes(canonical(tokenizer_files) + b"\n"),
        },
    }


def token_ids(namespace: str, length: int) -> list[int]:
    require(0 < length <= MAX_TOKEN_COUNT, "fixture token length is outside the bounded range")
    result: list[int] = []
    block = 0
    while len(result) < length:
        raw = hashlib.sha256(TOKEN_DOMAIN + namespace.encode("ascii") + b"\0" + block.to_bytes(8, "big")).digest()
        for offset in range(0, len(raw), 4):
            if len(result) == length:
                break
            # Keep below Qwen3.5's special-token area and avoid token 0.
            result.append(1 + int.from_bytes(raw[offset : offset + 4], "big") % 247_999)
        block += 1
    return result


def case_digest(case: dict[str, Any]) -> str:
    clone = json.loads(json.dumps(case))
    clone["case_sha256"] = None
    return sha_bytes(canonical(clone))


def baseline_case(
    case_id: str,
    *,
    kind: str,
    prompt_tokens: int,
    requested_m: int,
    resolved_m: int | None,
    generated_tokens: int,
    mode: str,
    cached_prefix_tokens: int = 0,
    status: str = "planned",
) -> dict[str, Any]:
    require(requested_m in M_GRID, "case M is outside the fixed grid")
    case: dict[str, Any] = {
        "schema_version": CASE_SCHEMA,
        "case_id": case_id,
        "case_sha256": None,
        "kind": kind,
        "status": status,
        "device": {
            "physical_target": "r9700-rdna4",
            "hip_visibility": "HIP_VISIBLE_DEVICES=1,ULLM_HIP_VISIBLE_DEVICES=1",
            "filtered_hip_ordinal": 0,
            "architecture": "gfx1201",
            "pci_bdf": "0000:47:00.0",
            "pci_device_id": "0x7551",
        },
        "execution": {
            "scope": "direct_resident_runtime",
            "phase": "cold_prefill" if kind != "decode" else "decode_after_context_prefill",
            "mode": mode,
            "prompt_tokens": prompt_tokens,
            "context_tokens": prompt_tokens,
            "cached_prefix_tokens": cached_prefix_tokens,
            "generated_tokens": generated_tokens,
            "request_count": 1,
            "requested_m": requested_m,
            "resolved_m": resolved_m,
            "sampling": {"mode": "greedy", "temperature": 0.0, "top_p": 1.0, "top_k": 1, "seed": 0},
        },
        "observability": {
            "wall_time": "required",
            "launch_sync": "required_from_executor_or_profiler",
            "transfer": "required_or_explicitly_not_observed",
            "workspace": "required_from_trace_or_explicitly_not_observed",
            "fallback": "required",
            "state_snapshot": "streaming_anchor_or_explicitly_not_captured",
        },
    }
    if status == "unsupported":
        case["unsupported"] = {
            "feature": "cached_prefix_chunked",
            "reason": "active AQ4 production path does not advertise a cached-prefix chunked executor",
            "must_not_be_counted_as_success": True,
        }
    case["case_sha256"] = case_digest(case)
    return case


def make_cases() -> tuple[list[dict[str, Any]], dict[str, list[int]]]:
    cases: list[dict[str, Any]] = []
    fixtures: dict[str, list[int]] = {}
    for length in PROMPT_LENGTHS:
        fixture_id = f"prefill-n{length}"
        fixtures[fixture_id] = token_ids(fixture_id, length)
        cases.append(
            baseline_case(
                f"p2-baseline-prefill-all-m1-n{length}-m1",
                kind="prefill",
                prompt_tokens=length,
                requested_m=1,
                resolved_m=1,
                generated_tokens=0,
                mode="all_m1",
            )
        )
        for width in M_GRID:
            production = baseline_case(
                f"p2-baseline-prefill-production-n{length}-m{width}",
                kind="prefill",
                prompt_tokens=length,
                requested_m=width,
                resolved_m=width,
                generated_tokens=0,
                # The resident runtime has one physical width-one path.  Keep
                # M=1 explicitly in the production grid, but label its actual
                # execution mode as all_m1 rather than inventing a distinct
                # cold-batched implementation that does not exist.
                mode="all_m1" if width == 1 else "cold_batched",
            )
            # One actual all-M=1 run is the shared same-input reference for
            # every production M at this prompt length.  Duplicating it six
            # times would add false variance without adding a distinct path.
            production["all_m1_reference_case_id"] = f"p2-baseline-prefill-all-m1-n{length}-m1"
            production["case_sha256"] = case_digest(production)
            cases.append(production)
            cases.append(
                baseline_case(
                    f"p2-baseline-prefill-cached-prefix-n{length}-m{width}",
                    kind="cached_prefix_chunked",
                    prompt_tokens=length,
                    requested_m=width,
                    resolved_m=None,
                    generated_tokens=0,
                    mode="cached_prefix_chunked",
                    cached_prefix_tokens=min(128, length),
                    status="unsupported",
                )
            )
    for length in DECODE_CONTEXTS:
        fixture_id = f"decode-c{length}"
        fixtures[fixture_id] = token_ids(fixture_id, length)
        for width in M_GRID:
            decode = baseline_case(
                f"p2-baseline-decode-c{length}-m{width}-g64",
                kind="decode",
                prompt_tokens=length,
                requested_m=width,
                resolved_m=width,
                generated_tokens=64,
                mode="decode_single_token",
            )
            # M applies to the cold context-prefill chunks that establish the
            # requested decode state.  Each of the 64 subsequent decode
            # iterations remains physically width one.  The resident trace
            # records the requested/resolved prefill width and the executor
            # records fallback explicitly; neither is inferred from M=1.
            decode["m_grid_scope"] = "decode_context_prefill"
            decode["decode_iteration_token_width"] = 1
            decode["case_sha256"] = case_digest(decode)
            cases.append(decode)
    # 7 all-M=1 prefill references + 42 prefill production M-grid cases +
    # 42 cached-prefix unsupported cases + 42 decode-context M-grid cases.
    require(len(cases) == 133, "internal case matrix count differs")
    return cases, fixtures


def planned_case_ids(cases: Iterable[dict[str, Any]], predicate: Any) -> list[str]:
    return [str(case["case_id"]) for case in cases if predicate(case)]


def make_windows(cases: list[dict[str, Any]]) -> dict[str, Any]:
    windows: list[dict[str, Any]] = []
    order = 1
    # One prompt length per normal prefill window keeps a failed long context
    # from consuming a run that also contains unrelated representative points.
    for length in PROMPT_LENGTHS:
        ids = planned_case_ids(
            cases,
            lambda case, length=length: case["kind"] == "prefill"
            and case["execution"]["prompt_tokens"] == length,
        )
        unsupported = planned_case_ids(
            cases,
            lambda case, length=length: case["kind"] == "cached_prefix_chunked"
            and case["execution"]["prompt_tokens"] == length,
        )
        windows.append(
            {
                "window_id": f"prefill-n{length}",
                "order": order,
                "kind": "normal_measurement",
                "case_ids": ids,
                "unsupported_case_ids": unsupported,
                "model_loads": 1,
                "warmup_runs_per_case": 2,
                "measured_runs_per_case": 10,
                "stop_after_first_terminal_failure": True,
                "reason": "one representative prefill length per single-use service-stop window",
            }
        )
        order += 1
    # A full decode M-grid has a 64-token serial decode segment for every
    # width.  Keep one start-context per outage so a long context or an M
    # fallback cannot turn an unrelated representative point into the same
    # service-stop failure domain.
    for length in DECODE_CONTEXTS:
        ids = planned_case_ids(
            cases,
            lambda case, length=length: case["kind"] == "decode"
            and case["execution"]["context_tokens"] == length,
        )
        windows.append(
            {
                "window_id": f"decode-c{length}",
                "order": order,
                "kind": "normal_measurement",
                "case_ids": ids,
                "unsupported_case_ids": [],
                "model_loads": 1,
                "warmup_runs_per_case": 2,
                "measured_runs_per_case": 10,
                "stop_after_first_terminal_failure": True,
                "reason": "one representative decode context and its complete M grid per single-use service-stop window",
            }
        )
        order += 1
    # Detailed profiler runs intentionally do not reuse normal-window timing.
    for profile_id, case_id in (
        ("profile-prefill-n128-m1", "p2-baseline-prefill-production-n128-m1"),
        ("profile-prefill-n1024-m128", "p2-baseline-prefill-production-n1024-m128"),
        ("profile-prefill-n2048-m64", "p2-baseline-prefill-production-n2048-m64"),
        ("profile-prefill-n3584-m128", "p2-baseline-prefill-production-n3584-m128"),
        ("profile-decode-c16", "p2-baseline-decode-c16-m1-g64"),
        ("profile-decode-c3584", "p2-baseline-decode-c3584-m1-g64"),
    ):
        windows.append(
            {
                "window_id": profile_id,
                "order": order,
                "kind": "detailed_profile",
                "case_ids": [case_id],
                "unsupported_case_ids": [],
                "model_loads": 1,
                # The resident protocol has a fixed 2 + 10 lifecycle.  The
                # profile collector is external and marks its trace separately;
                # its samples are never mixed into normal timing statistics.
                "warmup_runs_per_case": 2,
                "measured_runs_per_case": 10,
                "stop_after_first_terminal_failure": True,
                "reason": "rocprof/detail trace is isolated from normal baseline timing",
            }
        )
        order += 1
    return {
        "schema_version": WINDOW_SCHEMA,
        "status": "planned",
        "gpu_policy": {
            "target": "r9700-rdna4 only",
            "execution_serialization": "exactly one root-owned service-stop window at a time",
            "service_operation": "one stop and one restore per invoked window",
            "lock_policy": "pre-existing lock read-only probe only; no creation by this plan",
        },
        "windows": windows,
        "normal_window_count": 14,
        "detailed_profile_window_count": 6,
        "proposed_total_single_use_windows": len(windows),
    }


def legacy_case(case: dict[str, Any], tokens: list[int], *, oracle: bool = False) -> dict[str, Any]:
    execution = case["execution"]
    requested = int(execution["requested_m"])
    resolved = int(execution["resolved_m"] if execution["resolved_m"] is not None else requested)
    mode = "all_m1" if requested == 1 and oracle else str(execution["mode"])
    if mode not in {"all_m1", "cold_batched"}:
        mode = "all_m1" if requested == 1 else "cold_batched"
    generated = 1 if oracle else int(execution["generated_tokens"])
    item: dict[str, Any] = {
        "case_id": case["case_id"],
        "fixture_id": case["case_id"],
        "case_sha256": None,
        "stage_id": "p2_production_baseline",
        "stage_order": 1,
        "scope": "full_model",
        "phase": "cold_prefill",
        "mode": mode,
        "baseline_mode": mode,
        "prompt_tokens": len(tokens),
        "cached_prefix_tokens": 0,
        "context_tokens": len(tokens),
        "decode_start_tokens": len(tokens) if generated else 0,
        "prefill_requested_m": requested,
        "resolved_m": 1 if mode == "all_m1" else resolved,
        "request_count": 1,
        "decode_request_count": 0,
        "generated_tokens": generated,
        "device": {
            "device_id": "r9700-rdna4",
            "backend": "hip",
            "name": "AMD Radeon Graphics",
            "architecture": "gfx1201",
            "runtime_device_index": 1,
        },
        "control_id": "aq4_0_target",
        "control": {
            "control_id": "aq4_0_target",
            "role": "target",
            "format_id": "AQ4_0",
            "implementation_id": "qwen35_aq4_rdna4_v1",
            "promotion_eligible": True,
        },
        "sampling": {"mode": "greedy", "temperature": 0.0, "top_p": 1.0, "top_k": 1, "seed": 0},
        "format_id": "AQ4_0",
        "implementation_id": "qwen35_aq4_rdna4_v1",
        "path_oracle_case_id": None,
        "path_oracle_result_sha256": None,
    }
    if mode != "all_m1":
        item["path_oracle_case_id"] = f"p2-path-anchor-{case['case_id']}"
    item["case_sha256"] = case_digest(item)
    return item


def calibration_case(case: dict[str, Any], tokens: list[int]) -> dict[str, Any]:
    # The existing full-vector calibration binary intentionally uses the
    # physical HIP device-id vocabulary (1) and RDNA4 family vocabulary.  The
    # outer R9700 guard still makes filtered HIP ordinal 0 the only visible GPU.
    item = legacy_case(case, tokens, oracle=True)
    item["device"] = {
        "device_id": "r9700-rdna4",
        "backend": "hip",
        "name": "AMD Radeon Graphics",
        "architecture": "RDNA4",
        "runtime_device_index": 1,
    }
    item["case_sha256"] = case_digest(item)
    return item


def make_oracle_cases(cases: list[dict[str, Any]], fixtures: dict[str, list[int]]) -> tuple[list[dict[str, Any]], dict[str, list[int]]]:
    selected = (
        "p2-baseline-prefill-all-m1-n128-m1",
        "p2-baseline-prefill-all-m1-n1024-m1",
        "p2-baseline-prefill-all-m1-n2048-m1",
        "p2-baseline-prefill-all-m1-n3584-m1",
        "p2-baseline-decode-c16-m1-g64",
        "p2-baseline-decode-c1024-m1-g64",
        "p2-baseline-decode-c2048-m1-g64",
        "p2-baseline-decode-c3584-m1-g64",
    )
    by_id = {str(case["case_id"]): case for case in cases}
    oracle_cases: list[dict[str, Any]] = []
    oracle_fixtures: dict[str, list[int]] = {}
    for case_id in selected:
        original = by_id[case_id]
        oracle_id = f"p2-oracle-anchor-{case_id.removeprefix('p2-baseline-')}"
        clone = json.loads(json.dumps(original))
        clone["case_id"] = oracle_id
        clone["kind"] = "oracle_anchor"
        clone["status"] = "planned"
        clone["execution"]["generated_tokens"] = 1
        clone["execution"]["requested_m"] = 1
        clone["execution"]["resolved_m"] = 1
        clone["execution"]["mode"] = "all_m1"
        clone["case_sha256"] = case_digest(clone)
        key = f"prefill-n{original['execution']['prompt_tokens']}" if original["kind"] == "prefill" else f"decode-c{original['execution']['context_tokens']}"
        oracle_cases.append(clone)
        oracle_fixtures[oracle_id] = fixtures[key]
    return oracle_cases, oracle_fixtures


def sealed_files(root: Path) -> list[Path]:
    return sorted(
        [path for path in root.iterdir() if path.is_file() and path.name != "SHA256SUMS"],
        key=lambda path: path.name,
    )


def create_preparation(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output.absolute()
    require(not os.path.lexists(output), f"preparation output already exists: {output}")
    directory(output.parent, "preparation output parent")
    source = git_identity(args.source_worktree.absolute())
    active = inspect_active_manifest(args.active_manifest.absolute())
    checkpoint = source_checkpoint_identity(args.source_model.absolute())
    cases, fixtures = make_cases()
    oracle_cases, oracle_fixtures = make_oracle_cases(cases, fixtures)
    windows = make_windows(cases)

    deployed_commit = None
    promotion = active.get("promotion")
    if isinstance(promotion, dict):
        value = promotion.get("source_commit")
        if isinstance(value, str):
            deployed_commit = value
        else:
            source_info = promotion.get("source")
            if isinstance(source_info, dict):
                value = source_info.get("git_commit")
                deployed_commit = value if isinstance(value, str) else None
    comparability = {
        "status": "comparable" if deployed_commit == source["git_commit"] else "separated_not_comparable",
        "reason": (
            "deployed active manifest does not expose a product promotion source commit"
            if deployed_commit is None
            else "deployed product promotion source commit differs from clean baseline source HEAD"
            if deployed_commit != source["git_commit"]
            else "deployed product promotion source commit matches clean baseline source HEAD"
        ),
        "clean_baseline_git_commit": source["git_commit"],
        "deployed_product_source_commit": deployed_commit,
    }
    identity: dict[str, Any] = {
        "schema_version": IDENTITY_SCHEMA,
        "status": "frozen",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "clean_baseline_source": source,
        "deployed_active": active,
        "source_checkpoint": checkpoint,
        "comparability": comparability,
        "target": {
            "device_id": "r9700-rdna4",
            "architecture": "gfx1201",
            "pci_bdf": "0000:47:00.0",
            "pci_device_id": "0x7551",
            "physical_hip_index": 1,
            "filtered_hip_ordinal": 0,
        },
        "identity_sha256": None,
    }
    identity["identity_sha256"] = sha_bytes(canonical({**identity, "identity_sha256": None}))

    resident_cases = [case for case in cases if case["status"] == "planned"]
    resident_case_bindings: list[dict[str, Any]] = []
    resident_fixture_cases: list[dict[str, Any]] = []
    for case in resident_cases:
        key = f"prefill-n{case['execution']['prompt_tokens']}" if case["kind"] == "prefill" else f"decode-c{case['execution']['context_tokens']}"
        tokens = fixtures[key]
        resident_cases_binding = legacy_case(case, tokens)
        resident_case_bindings.append(resident_cases_binding)
        resident_fixture_cases.append(
            {"case_id": case["case_id"], "prompt_token_ids": tokens, "step_count": case["execution"]["generated_tokens"]}
        )

    oracle_calibration_cases: list[dict[str, Any]] = []
    oracle_source_cases: list[dict[str, Any]] = []
    oracle_fixture_cases: list[dict[str, Any]] = []
    for case in oracle_cases:
        tokens = oracle_fixtures[case["case_id"]]
        oracle_calibration_cases.append(calibration_case(case, tokens))
        oracle_source_cases.append(
            {
                "case_id": case["case_id"],
                "prompt_token_ids": tokens,
                "step_count": 1,
                "semantic_input_id": case["case_id"],
                "observation": "streaming_anchor_first_generated_token",
            }
        )
        oracle_fixture_cases.append({"case_id": case["case_id"], "prompt_token_ids": tokens, "step_count": 1})

    oracle_contract = {
        "schema_version": ORACLE_SCHEMA,
        "status": "planned",
        "source_oracle": {
            "kind": "independent_source_full",
            "device": "cpu",
            "dtype": "float32",
            "streaming_chunk_elements": 65536,
            "max_resident_logit_rows": 1,
            "all_logit_matrix_in_memory": False,
        },
        "path_oracle": {
            "kind": "same_artifact_all_m1_target_full_vector",
            "requested_m": 1,
            "resolved_m": 1,
            "streaming_chunk_elements": 65536,
            "all_logit_matrix_in_memory": False,
        },
        "state_snapshot": {
            "kind": "streaming_anchor_execution_rows",
            "coverage": "first generated token at each anchor; no claim of a full KV byte dump",
            "hash_binding": "source row SHA-256 + source replay sequence SHA-256 + generation epoch",
        },
        "anchor_case_ids": [case["case_id"] for case in oracle_cases],
        "comparison": {
            "mode": "rowwise streaming",
            "requires_matching_source_and_target_row_hashes": True,
            "requires_sidecar_sha256s": True,
            "must_not_retain_full_logit_matrix": True,
        },
    }
    # Full-vector target capture is intentionally one anchor per service-stop
    # window.  Each capture loads a clean model process and transfers a
    # diagnostic vector; grouping anchors would hide the failure boundary and
    # keep the service unavailable for an unnecessarily long interval.
    oracle_windows = {
        "schema_version": ORACLE_WINDOW_SCHEMA,
        "status": "planned",
        "execution_serialization": "one R9700-only single-use service-stop window per anchor",
        "windows": [
            {
                "window_id": f"path-oracle-{case['case_id'].removeprefix('p2-oracle-anchor-')}",
                "order": index,
                "oracle_case_id": case["case_id"],
                "model_loads": 1,
                "requested_m": 1,
                "resolved_m": 1,
                "reason": "single full-vector all-M=1 path/state anchor",
            }
            for index, case in enumerate(oracle_cases, 1)
        ],
        "proposed_single_use_window_count": len(oracle_cases),
    }
    profile_plan = {
        "schema_version": PROFILE_SCHEMA,
        "status": "planned",
        "normal_measurement": "all planned non-unsupported matrix cases",
        "detailed_profile_case_ids": [
            entry["case_ids"][0]
            for entry in windows["windows"]
            if entry["kind"] == "detailed_profile"
        ],
        "profile_requirements": {
            "capture_only_in_detailed_profile_window": True,
            "kernel_family_mapping": "tools/profile-aq4-p2-family-exclusive.py mapping may be reused only as parsing design; raw trace must be new-current-identity evidence",
            "unknown_kernel_policy": "fail closed/unclassified, never silently attributed",
        },
    }
    executor_contract = {
        "schema_version": EXECUTOR_SCHEMA,
        "status": "planned",
        "raw_trace": {
            "format": "JSONL",
            "bounded_record_bytes": 4 * 1024 * 1024,
            "contains": ["ready", "case_ready", "run_complete", "case_complete"],
        },
        "sanitized_sidecar": {
            "format": "JSONL",
            "allowed_fields": [
                "case_id",
                "run_index",
                "run_kind",
                "status",
                "requested_m",
                "resolved_m",
                "actual_token_batch_width",
                "end_to_end_ms",
                "prefill_ms",
                "decode_ms",
                "operation_digest_sha256",
                "request_state_sha256",
                "fallback_status",
                "workspace_status",
                "transfer_status",
            ],
            "must_not_include": ["prompt_token_ids", "generated_token_ids", "generated_text", "full_logits"],
        },
        "hash_binding": {
            "requires": [
                "preparation SHA256SUMS",
                "staging receipt SHA-256",
                "raw trace SHA-256",
                "sanitized sidecar SHA-256",
            ],
            "binding_file": "trace-hash-binding.json",
        },
        "unobserved_metrics_policy": "write explicit not_observed/not_captured status; never substitute zero or success",
    }
    policy = {
        "schema_version": "ullm.aq4_production_p2_threshold_policy.v1",
        "status": "bound",
        "policy_id": "aq4-p2-production-baseline-observability-v0.1",
        "scope": "baseline collection only; no performance promotion decision",
    }
    preflight_template = {
        "weights_bytes": int(active["package"]["tree"]["bytes"]),
        "persistent_state_bytes": 0,
        "kv_cache_bytes": 0,
        "workspace_bytes": 0,
        "temporary_bytes": 0,
        "vram_headroom_bytes": 0,
        "gpu_process_snapshot": [],
    }

    output.mkdir(mode=0o700)
    try:
        # These four directories are deliberately pre-created before the
        # immutable envelope is sealed.  The preparation files themselves are
        # immutable; later CPU staging, CPU source capture, host-only guard
        # compilation, and root-only window output never need to reopen that
        # sealed member set for writing.
        for name in ("staging", "source-oracle", "guard", "windows"):
            (output / name).mkdir(mode=0o700)
        documents: dict[str, Any] = {
            "identity.json": identity,
            "baseline-cases.json": {"schema_version": CASE_SCHEMA, "cases": cases},
            "window-plan.json": windows,
            "oracle-contract.json": oracle_contract,
            "oracle-window-plan.json": oracle_windows,
            "profile-plan.json": profile_plan,
            "executor-record-contract.json": executor_contract,
            "resident-fixture.json": {"schema_version": RESIDENT_FIXTURE_SCHEMA, "cases": resident_fixture_cases},
            "resident-case-binding.json": {"cases": resident_case_bindings},
            "oracle-fixture.json": {"cases": oracle_fixture_cases},
            "source-oracle-cases.json": {"schema_version": SOURCE_CASES_SCHEMA, "cases": oracle_source_cases},
            "calibration-case-index.json": {"cases": oracle_calibration_cases},
            "policy.json": policy,
            "preflight-template.json": preflight_template,
        }
        for name, document in documents.items():
            write_new(output / name, json.dumps(document, ensure_ascii=True, sort_keys=True, indent=2, allow_nan=False).encode("utf-8") + b"\n")
        manifest: dict[str, Any] = {
            "schema_version": SCHEMA,
            "status": "prepared",
            "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "identity": {"path": "identity.json", "sha256": sha_file(output / "identity.json", "identity")},
            "matrix": {
                "prefill_prompt_lengths": list(PROMPT_LENGTHS),
                "decode_start_contexts": list(DECODE_CONTEXTS),
                "m_grid": list(M_GRID),
                "planned_case_count": len(cases),
                "executable_case_count": len(resident_cases),
                "unsupported_cached_prefix_case_count": len([case for case in cases if case["status"] == "unsupported"]),
            },
            "source_identity_comparability": comparability,
            "expected_staged_binaries": ["ullm-aq4-p2-resident-driver", "ullm-aq4-p2-calibration"],
            "window_plan": {"path": "window-plan.json", "sha256": sha_file(output / "window-plan.json", "window plan")},
            "oracle_contract": {"path": "oracle-contract.json", "sha256": sha_file(output / "oracle-contract.json", "oracle contract")},
            "oracle_window_plan": {"path": "oracle-window-plan.json", "sha256": sha_file(output / "oracle-window-plan.json", "oracle window plan")},
            "preparation_sha256": None,
        }
        manifest["preparation_sha256"] = sha_bytes(canonical({**manifest, "preparation_sha256": None}))
        write_new(output / "preparation-manifest.json", json.dumps(manifest, ensure_ascii=True, sort_keys=True, indent=2).encode("utf-8") + b"\n")
        sums = "".join(
            f"{sha_file(path, f'preparation member {path.name}')}  {path.name}\n"
            for path in sealed_files(output)
        )
        write_new(output / "SHA256SUMS", sums.encode("ascii"))
        os.chmod(output, 0o555)
        fsync_directory(output)
        fsync_directory(output.parent)
    except Exception:
        # Do not delete a partially prepared artifact automatically: retaining
        # it makes the failed attempt inspectable and prevents silent reuse.
        raise
    return verify_preparation(output)


def parse_sums(path: Path) -> dict[str, str]:
    raw = path.read_text(encoding="ascii")
    result: dict[str, str] = {}
    for number, line in enumerate(raw.splitlines(), 1):
        digest, separator, name = line.partition("  ")
        require(separator == "  " and len(digest) == 64 and set(digest) <= SHA256_RE and name, f"SHA256SUMS line {number} is invalid")
        require("/" not in name and name not in result, f"SHA256SUMS line {number} member is invalid")
        result[name] = digest
    require(result, "SHA256SUMS is empty")
    return result


def verify_preparation(output: Path) -> dict[str, Any]:
    output = output.absolute()
    info = directory(output, "preparation output")
    require(stat.S_IMODE(info.st_mode) == 0o555, "preparation directory mode must be 0555")
    expected_files = {
        "identity.json",
        "baseline-cases.json",
        "window-plan.json",
        "oracle-contract.json",
        "oracle-window-plan.json",
        "profile-plan.json",
        "executor-record-contract.json",
        "resident-fixture.json",
        "resident-case-binding.json",
        "oracle-fixture.json",
        "source-oracle-cases.json",
        "calibration-case-index.json",
        "policy.json",
        "preflight-template.json",
        "preparation-manifest.json",
        "SHA256SUMS",
    }
    expected_directories = {"staging", "source-oracle", "guard", "windows"}
    observed = {path.name for path in output.iterdir()}
    require(
        observed == expected_files | expected_directories,
        f"preparation member set differs: expected {sorted(expected_files | expected_directories)}, got {sorted(observed)}",
    )
    for name in expected_directories:
        member = directory(output / name, f"mutable preparation directory {name}")
        require(stat.S_IMODE(member.st_mode) == 0o700, f"mutable preparation directory mode differs: {name}")
    sums = parse_sums(output / "SHA256SUMS")
    require(set(sums) == expected_files - {"SHA256SUMS"}, "SHA256SUMS members differ")
    for name, digest in sums.items():
        path = output / name
        item = regular(path, f"preparation member {name}")
        require(stat.S_IMODE(item.st_mode) == 0o444, f"preparation member mode differs: {name}")
        require(sha_file(path, f"preparation member {name}") == digest, f"preparation member hash differs: {name}")
    manifest = load_json(output / "preparation-manifest.json", "preparation manifest")
    require(isinstance(manifest, dict) and manifest.get("schema_version") == SCHEMA and manifest.get("status") == "prepared", "preparation manifest schema/status differs")
    declared = manifest.get("preparation_sha256")
    require(isinstance(declared, str) and declared == sha_bytes(canonical({**manifest, "preparation_sha256": None})), "preparation manifest self-hash differs")
    identity = load_json(output / "identity.json", "identity")
    require(isinstance(identity, dict) and identity.get("schema_version") == IDENTITY_SCHEMA and identity.get("status") == "frozen", "identity schema/status differs")
    identity_hash = identity.get("identity_sha256")
    require(isinstance(identity_hash, str) and identity_hash == sha_bytes(canonical({**identity, "identity_sha256": None})), "identity self-hash differs")
    cases = load_json(output / "baseline-cases.json", "baseline cases")
    require(isinstance(cases, dict) and cases.get("schema_version") == CASE_SCHEMA and isinstance(cases.get("cases"), list), "baseline cases schema differs")
    require(len(cases["cases"]) == 133, "baseline matrix count differs")
    ids: set[str] = set()
    for case in cases["cases"]:
        require(isinstance(case, dict) and isinstance(case.get("case_id"), str) and case["case_id"] not in ids, "baseline case IDs differ")
        ids.add(case["case_id"])
        require(case.get("case_sha256") == case_digest(case), f"baseline case hash differs: {case['case_id']}")
    decode_grid = {
        (
            case.get("execution", {}).get("context_tokens"),
            case.get("execution", {}).get("requested_m"),
            case.get("execution", {}).get("resolved_m"),
        )
        for case in cases["cases"]
        if case.get("kind") == "decode"
    }
    expected_decode_grid = {(context, width, width) for context in DECODE_CONTEXTS for width in M_GRID}
    require(decode_grid == expected_decode_grid, "decode M grid/resolution differs")
    for case in cases["cases"]:
        if case.get("kind") != "decode":
            continue
        require(
            case.get("m_grid_scope") == "decode_context_prefill"
            and case.get("decode_iteration_token_width") == 1,
            f"decode M scope differs: {case.get('case_id')}",
        )
    windows = load_json(output / "window-plan.json", "window plan")
    require(isinstance(windows, dict) and windows.get("schema_version") == WINDOW_SCHEMA, "window plan schema differs")
    normal_planned: set[str] = set()
    for window in windows.get("windows", []):
        require(isinstance(window, dict) and isinstance(window.get("window_id"), str), "window entry differs")
        for case_id in window.get("case_ids", []):
            require(case_id in ids, f"window case is missing: {case_id}")
            if window.get("kind") == "normal_measurement":
                require(case_id not in normal_planned, f"normal window case is repeated: {case_id}")
                normal_planned.add(case_id)
    normal = [case["case_id"] for case in cases["cases"] if case["status"] == "planned"]
    normal_window_ids = {
        case_id
        for window in windows["windows"]
        if window.get("kind") == "normal_measurement"
        for case_id in window.get("case_ids", [])
    }
    require(set(normal) == normal_window_ids, "normal measurement windows do not cover exactly the executable matrix")
    require(windows.get("normal_window_count") == 14, "normal window count differs")
    require(windows.get("detailed_profile_window_count") == 6, "detailed profile window count differs")
    oracle_windows = load_json(output / "oracle-window-plan.json", "oracle window plan")
    require(isinstance(oracle_windows, dict) and oracle_windows.get("schema_version") == ORACLE_WINDOW_SCHEMA and isinstance(oracle_windows.get("windows"), list), "oracle window plan schema differs")
    anchors = load_json(output / "calibration-case-index.json", "calibration case index")
    require(isinstance(anchors, dict) and isinstance(anchors.get("cases"), list), "calibration case index differs")
    anchor_ids = {item.get("case_id") for item in anchors["cases"] if isinstance(item, dict)}
    planned_anchor_ids = {item.get("oracle_case_id") for item in oracle_windows["windows"] if isinstance(item, dict)}
    require(anchor_ids and anchor_ids == planned_anchor_ids and len(planned_anchor_ids) == 8, "oracle windows do not cover exactly the eight anchors")
    return {
        "schema_version": SCHEMA,
        "status": "valid",
        "output": str(output),
        "preparation_sha256": declared,
        "identity_sha256": identity_hash,
        "case_count": len(cases["cases"]),
        "normal_window_count": windows.get("normal_window_count"),
        "detailed_profile_window_count": windows.get("detailed_profile_window_count"),
        "path_oracle_window_count": oracle_windows.get("proposed_single_use_window_count"),
        "comparability": identity.get("comparability"),
    }


def verify_live_active_identity(output: Path, active_manifest: Path) -> dict[str, Any]:
    """Verify that a sealed preparation still names the live served identity.

    A preparation is deliberately immutable. This check never refreshes it in
    place: a changed active manifest requires a new preparation output.
    """
    result = verify_preparation(output)
    identity = load_json(output.absolute() / "identity.json", "identity")
    frozen = identity.get("deployed_active") if isinstance(identity, dict) else None
    require(isinstance(frozen, dict), "preparation identity lacks frozen deployed active identity")
    live = inspect_active_manifest(active_manifest.absolute())
    require(
        frozen == live,
        "frozen deployed active identity differs from live active manifest; create a new preparation output",
    )
    return {
        **result,
        "live_active_identity": {
            "manifest_path": live["manifest_path"],
            "manifest_sha256": live["manifest_sha256"],
            "model": live["model"],
            "worker_binary_sha256": live["worker"]["sha256"],
            "package_manifest_sha256": live["package"]["manifest_sha256"],
            "package_content_sha256": live["package"]["tree"]["sha256"],
        },
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="new immutable preparation directory")
    parser.add_argument("--verify", action="store_true", help="verify an existing preparation directory without modifying it")
    parser.add_argument(
        "--verify-live-active-identity",
        action="store_true",
        help="with --verify, require the frozen deployed identity to equal --active-manifest",
    )
    parser.add_argument("--source-worktree", type=Path, help="clean detached source worktree at the frozen HEAD")
    parser.add_argument("--active-manifest", type=Path, default=Path("/etc/ullm/served-models/active.json"))
    parser.add_argument("--source-model", type=Path, default=Path("/home/homelab1/datapool/ai_models/safetensors/Qwen/Qwen3.5-9B"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        if args.verify:
            result = (
                verify_live_active_identity(args.output, args.active_manifest)
                if args.verify_live_active_identity
                else verify_preparation(args.output)
            )
        else:
            require(args.source_worktree is not None, "--source-worktree is required when creating preparation")
            require(not args.verify_live_active_identity, "--verify-live-active-identity requires --verify")
            result = create_preparation(args)
    except (PreparationError, OSError, ValueError) as error:
        print(f"AQ4 P2 production baseline preparation failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
