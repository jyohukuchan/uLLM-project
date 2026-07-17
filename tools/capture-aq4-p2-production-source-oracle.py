#!/usr/bin/env python3
"""Capture the new P2 source oracle on CPU with streamed vector sidecars.

The input is the current-identity preparation envelope, not an older P2
split.  Each observation writes one final hidden row and one full pre-softmax
logit row in chunks; it never retains a sequence-by-vocabulary matrix.  The
tool refuses to run when a GPU is visible and never reads a service or lock.
"""

from __future__ import annotations

import argparse
import datetime as dt
import gc
import hashlib
import json
import os
import shutil
import stat
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, BinaryIO


SCHEMA = "ullm.qwen35_aq4_source_calibration.v1"
CASES_SCHEMA = "ullm.qwen35_aq4_source_calibration_cases.v1"
HIDDEN_SIZE = 4096
VOCAB_SIZE = 248320
TOP_K = 10
F32_BYTES = 4
RUNTIME_DTYPE = "float32"
ROW_BYTES = (HIDDEN_SIZE + VOCAB_SIZE) * F32_BYTES
DEFAULT_CHUNK_ELEMENTS = 65536
MAX_CASES = 8192
MAX_STEPS = 128
MAX_ROWS = 16384
MAX_CASE_BYTES = 4 * 1024 * 1024


class SourceError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SourceError(message)


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
        raise SourceError(f"{label} is unavailable: {path}: {error}") from error
    require(stat.S_ISREG(before.st_mode) and not stat.S_ISLNK(before.st_mode), f"{label} must be a regular non-symlink file")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0))
    digest = hashlib.sha256()
    try:
        while chunk := os.read(descriptor, 1024 * 1024):
            digest.update(chunk)
    finally:
        os.close(descriptor)
    return digest.hexdigest()


def reject_duplicate_keys(items: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, child in items:
        if key in value:
            raise SourceError(f"duplicate JSON key: {key}")
        value[key] = child
    return value


def load_json(path: Path, label: str, maximum: int = MAX_CASE_BYTES) -> Any:
    try:
        info = path.lstat()
    except OSError as error:
        raise SourceError(f"{label} is unavailable: {error}") from error
    require(stat.S_ISREG(info.st_mode) and not stat.S_ISLNK(info.st_mode), f"{label} must be a regular non-symlink file")
    require(info.st_size <= maximum, f"{label} exceeds bounded size")
    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=lambda token: (_ for _ in ()).throw(SourceError(f"non-finite JSON token {token}")),
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise SourceError(f"{label} is invalid JSON: {error}") from error


def source_cases(path: Path) -> list[dict[str, Any]]:
    value = load_json(path, "source oracle cases")
    require(isinstance(value, dict) and set(value) == {"schema_version", "cases"} and value.get("schema_version") == CASES_SCHEMA, "source oracle cases schema differs")
    cases = value.get("cases")
    require(isinstance(cases, list) and 0 < len(cases) <= MAX_CASES, "source oracle cases count differs")
    rows = 0
    seen: set[str] = set()
    result: list[dict[str, Any]] = []
    for index, case in enumerate(cases):
        require(isinstance(case, dict) and set(case) == {"case_id", "prompt_token_ids", "step_count", "semantic_input_id", "observation"}, f"source case {index} fields differ")
        case_id = case["case_id"]
        tokens = case["prompt_token_ids"]
        steps = case["step_count"]
        require(isinstance(case_id, str) and case_id and len(case_id) <= 128 and case_id not in seen, f"source case {index} ID differs")
        require(isinstance(tokens, list) and 0 < len(tokens) <= 4096 and all(type(token) is int and 0 <= token < VOCAB_SIZE for token in tokens), f"source case {index} token list differs")
        require(type(steps) is int and 0 < steps <= MAX_STEPS, f"source case {index} step count differs")
        require(isinstance(case["semantic_input_id"], str) and case["semantic_input_id"], f"source case {index} semantic input differs")
        require(isinstance(case["observation"], str) and case["observation"], f"source case {index} observation differs")
        seen.add(case_id)
        rows += steps
        require(rows <= MAX_ROWS, "source oracle row count exceeds bound")
        result.append(case)
    return result


def ensure_cpu_visibility() -> None:
    expected = {
        "CUDA_VISIBLE_DEVICES": "-1",
        "HIP_VISIBLE_DEVICES": "-1",
        "ROCR_VISIBLE_DEVICES": "-1",
        "ULLM_HIP_VISIBLE_DEVICES": "-1",
    }
    bad = {name: os.environ.get(name) for name, value in expected.items() if os.environ.get(name) != value}
    require(not bad, f"CPU source oracle visibility contract differs: {bad}")


def validate_preparation(preparation: Path) -> dict[str, Any]:
    tool = Path(__file__).resolve().parent / "prepare-aq4-p2-production-baseline.py"
    result = __import__("subprocess").run(
        [sys.executable, str(tool), "--output", str(preparation), "--verify"],
        check=False,
        stdout=__import__("subprocess").PIPE,
        stderr=__import__("subprocess").PIPE,
        text=True,
    )
    if result.returncode != 0:
        raise SourceError(f"preparation verification failed: {result.stderr.strip() or result.stdout.strip()}")
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise SourceError(f"preparation verification returned invalid JSON: {error}") from error
    require(payload.get("status") == "valid", "preparation verification did not report valid")
    return payload


def model_identity(preparation: Path, model_dir: Path) -> dict[str, Any]:
    identity = load_json(preparation / "identity.json", "preparation identity", 16 * 1024 * 1024)
    active = identity.get("deployed_active")
    checkpoint = identity.get("source_checkpoint")
    require(isinstance(active, dict) and isinstance(checkpoint, dict), "preparation identity lacks active/checkpoint binding")
    source = checkpoint.get("source_checkpoint")
    tokenizer = checkpoint.get("tokenizer")
    model = active.get("model")
    require(isinstance(source, dict) and isinstance(tokenizer, dict) and isinstance(model, dict), "preparation identity source/model fields differ")
    require(Path(str(source.get("root", ""))).resolve() == model_dir.resolve(), "requested source model differs from frozen checkpoint root")
    for value, label in ((source, "source checkpoint"), (tokenizer, "tokenizer")):
        files = value.get("files")
        require(isinstance(files, list) and files, f"{label} file identity differs")
        actual = sha_bytes(canonical(files) + b"\n")
        require(value.get("aggregate_sha256") == actual, f"{label} aggregate hash differs")
        for item in files:
            require(isinstance(item, dict) and isinstance(item.get("file"), str), f"{label} file entry differs")
            path = model_dir / item["file"]
            require(path.is_file() and not path.is_symlink(), f"{label} member is unavailable: {path}")
            require(sha_file(path, f"{label} member") == item.get("sha256"), f"{label} member hash differs: {path.name}")
    return {
        "artifact": {"package_manifest_sha256": None, "artifact_manifest_sha256": None},
        "model_id": model.get("upstream_id"),
        "model_revision": model.get("revision"),
        "source_checkpoint": {
            "aggregate_sha256": source["aggregate_sha256"],
            "dtype": checkpoint.get("dtype", "bfloat16"),
            "files": source["files"],
            "root": str(model_dir.resolve()),
        },
        "tokenizer": {
            "aggregate_sha256": tokenizer["aggregate_sha256"],
            "files": tokenizer["files"],
            "root": str(model_dir.resolve()),
        },
        "hidden_size": HIDDEN_SIZE,
        "vocab_size": VOCAB_SIZE,
    }


def token_hash(tokens: list[int]) -> str:
    return sha_bytes(canonical(tokens) + b"\n")


def f32_bytes(tensor: Any) -> bytes:
    try:
        import numpy as np
    except ImportError as error:
        raise SourceError(f"numpy is required for f32le output: {error}") from error
    values = tensor.detach().to(device="cpu", dtype=__import__("torch").float32).contiguous().flatten().numpy()
    return values.astype("<f4", copy=False).tobytes(order="C")


def write_vector(handle: BinaryIO, tensor: Any, chunk_elements: int) -> tuple[int, str, int]:
    torch = __import__("torch")
    flat = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous().flatten()
    digest = hashlib.sha256()
    nonfinite = int((~torch.isfinite(flat)).sum().item())
    offset = handle.tell()
    for chunk in flat.split(chunk_elements):
        raw = f32_bytes(chunk)
        handle.write(raw)
        digest.update(raw)
    return offset, digest.hexdigest(), nonfinite


def topk(logits: Any) -> list[dict[str, Any]]:
    torch = __import__("torch")
    flat = logits.detach().to(device="cpu", dtype=torch.float32).flatten()
    indices = torch.argsort(flat, descending=True, stable=True)[:TOP_K]
    return [{"token_id": int(token), "logit": float(flat[token])} for token in indices.tolist()]


def write_sums(root: Path) -> None:
    members = [root / "manifest.json", root / "rows.jsonl", root / "vectors/hidden.f32le", root / "vectors/logits.f32le"]
    raw = "".join(f"{sha_file(path, path.name)}  {path.relative_to(root).as_posix()}\n" for path in members)
    (root / "SHA256SUMS").write_text(raw, encoding="ascii")


def capture(args: argparse.Namespace, cases: list[dict[str, Any]], identity: dict[str, Any]) -> dict[str, Any]:
    ensure_cpu_visibility()
    try:
        import torch
        from transformers import AutoModelForCausalLM
    except ImportError as error:
        raise SourceError(f"torch and transformers are required for source capture: {error}") from error
    require(not torch.cuda.is_available(), "GPU is visible to the CPU source oracle")
    output = args.output.absolute()
    require(not os.path.lexists(output), f"source oracle output already exists: {output}")
    require(output.parent.is_dir() and not output.parent.is_symlink(), "source oracle output parent must be a real directory")
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.incomplete-", dir=output.parent))
    try:
        vectors = temporary / "vectors"
        vectors.mkdir(mode=0o700)
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(args.threads)
        started = time.monotonic()
        model = AutoModelForCausalLM.from_pretrained(
            args.model_dir,
            local_files_only=True,
            dtype=torch.float32,
            low_cpu_mem_usage=False,
            device_map=None,
        )
        model.eval()
        rows = 0
        nonfinite_rows = 0
        with (vectors / "hidden.f32le").open("xb") as hidden, (vectors / "logits.f32le").open("xb") as logits, (temporary / "rows.jsonl").open("x", encoding="utf-8") as rows_file:
            for case in cases:
                past = None
                input_tokens = list(case["prompt_token_ids"])
                try:
                    for step in range(case["step_count"]):
                        tensor = torch.tensor([input_tokens], dtype=torch.long, device="cpu")
                        with torch.inference_mode():
                            base = model.model(input_ids=tensor, past_key_values=past, use_cache=True, return_dict=True)
                            hidden_tensor = base.last_hidden_state[:, -1, :]
                            logits_tensor = model.lm_head(hidden_tensor.unsqueeze(1))[:, -1, :]
                        hidden_offset, hidden_sha, hidden_nonfinite = write_vector(hidden, hidden_tensor, args.chunk_elements)
                        logits_offset, logits_sha, logits_nonfinite = write_vector(logits, logits_tensor, args.chunk_elements)
                        finite = hidden_nonfinite == 0 and logits_nonfinite == 0
                        ranking = topk(logits_tensor) if finite else []
                        row = {
                            "case_id": case["case_id"],
                            "step": step,
                            "semantic_input_id": case["semantic_input_id"],
                            "observation": case["observation"],
                            "input_token_ids_sha256": token_hash(input_tokens),
                            "hidden": {"offset_bytes": hidden_offset, "bytes": HIDDEN_SIZE * F32_BYTES, "elements": HIDDEN_SIZE, "dtype": "f32", "endianness": "little", "sha256": hidden_sha, "nonfinite_count": hidden_nonfinite},
                            "logits": {"offset_bytes": logits_offset, "bytes": VOCAB_SIZE * F32_BYTES, "elements": VOCAB_SIZE, "dtype": "f32", "endianness": "little", "sha256": logits_sha, "nonfinite_count": logits_nonfinite},
                            "greedy_token_id": ranking[0]["token_id"] if ranking else None,
                            "topk": ranking,
                            "finite": finite,
                        }
                        rows_file.write(json.dumps(row, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
                        rows += 1
                        nonfinite_rows += int(not finite)
                        if not finite:
                            break
                        input_tokens = [ranking[0]["token_id"]]
                        past = getattr(base, "past_key_values", None)
                        del tensor, base, hidden_tensor, logits_tensor
                    if nonfinite_rows:
                        break
                finally:
                    del past
                    gc.collect()
        del model
        gc.collect()
        manifest = {
            "schema_version": SCHEMA,
            "oracle_kind": "independent_source_full",
            "status": "available" if nonfinite_rows == 0 else "blocked",
            "evidence_class": "production" if nonfinite_rows == 0 else "blocked",
            "usable_as_source_evidence": nonfinite_rows == 0,
            "promotion_eligible": False,
            "created_utc": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
            "identity": identity,
            "parent_sampled_oracle": {"status": "not_used", "reason": "new current-identity P2 source capture"},
            "vector_contract": {"hidden_shape": [HIDDEN_SIZE], "logits_shape": [VOCAB_SIZE], "dtype": "f32", "endianness": "little", "layout": "flat", "chunk_elements": args.chunk_elements, "row_bytes": ROW_BYTES, "semantic_hidden": "final_rmsnorm_hidden_used_by_lm_head", "semantic_logits": "raw_pre_softmax_lm_head_logits"},
            "limits": {"max_case_file_bytes": MAX_CASE_BYTES, "max_cases": MAX_CASES, "max_rows": MAX_ROWS, "max_steps": MAX_STEPS},
            "cases": {"path": str((args.preparation / "source-oracle-cases.json").resolve()), "sha256": sha_file(args.preparation / "source-oracle-cases.json", "source oracle cases"), "case_count": len(cases), "row_count": rows},
            "files": {"rows": "rows.jsonl", "hidden": "vectors/hidden.f32le", "logits": "vectors/logits.f32le"},
            "runtime": {"runtime": "transformers.AutoModelForCausalLM", "device": "cpu", "dtype": RUNTIME_DTYPE, "torch_num_threads": args.threads, "model_loads": 1, "inference_mode": True, "full_vocab_ranking": nonfinite_rows == 0, "max_resident_logit_rows": 1, "elapsed_seconds": time.monotonic() - started},
            "legacy_cross_check": {"status": "not_used", "reason": "old P2 identity is not reused as current evidence"},
        }
        (temporary / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=True, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        write_sums(temporary)
        for path in (temporary / "manifest.json", temporary / "rows.jsonl", vectors / "hidden.f32le", vectors / "logits.f32le", temporary / "SHA256SUMS"):
            descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        os.rename(temporary, output)
        temporary = None  # type: ignore[assignment]
        return manifest
    finally:
        if temporary is not None and temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--confirm-cpu-source-capture", action="store_true")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--chunk-elements", type=int, default=DEFAULT_CHUNK_ELEMENTS)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        require(args.preflight != args.confirm_cpu_source_capture, "choose exactly one of --preflight or --confirm-cpu-source-capture")
        require(0 < args.threads <= 32 and 0 < args.chunk_elements <= 1_048_576, "threads/chunk-elements are outside bounded limits")
        args.preparation = args.preparation.absolute()
        args.model_dir = args.model_dir.absolute()
        args.output = args.output.absolute()
        ensure_cpu_visibility()
        preparation = validate_preparation(args.preparation)
        cases = source_cases(args.preparation / "source-oracle-cases.json")
        identity = model_identity(args.preparation, args.model_dir)
        if args.preflight:
            result = {"schema_version": SCHEMA, "status": "preflight_valid", "preparation_sha256": preparation["preparation_sha256"], "case_count": len(cases), "row_count": sum(case["step_count"] for case in cases), "gpu_or_service_action": "none"}
        else:
            result = capture(args, cases, identity)
        print(json.dumps(result, ensure_ascii=True, sort_keys=True))
        return 0 if result.get("status") in {"preflight_valid", "available"} else 1
    except (SourceError, OSError, ValueError, RuntimeError) as error:
        print(f"AQ4 P2 production source oracle failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
