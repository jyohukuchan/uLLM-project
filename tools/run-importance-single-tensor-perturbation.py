#!/usr/bin/env python3
"""Run bounded C4 block perturbation or C6 full-vocabulary KL on an explicit device."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import io
import json
import math
import os
import re
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch


def load_tool(filename: str, module_name: str):
    path = Path(__file__).resolve().parent / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


SAMPLER = load_tool("run-aq-tensor-sample.py", "importance_perturb_sampler")
COLLECTOR = load_tool("collect-activation-stats.py", "importance_perturb_collector")


FORMAL_C4_SHARD_COUNT = 4
FORMAL_C4_TEMP_CACHE_LIMIT_BYTES = 8 * 1024**3


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def synchronize_device(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def elapsed_since(device: torch.device, started_at: float) -> float:
    synchronize_device(device)
    return max(time.perf_counter() - started_at, 0.0)


def emit_progress(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True), file=sys.stderr, flush=True)


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def load_codebooks(path: Path) -> tuple[dict[tuple[str, str], torch.Tensor], str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    result = {}
    for row in payload["codebooks"]:
        result[(str(row["family"]), str(row["candidate_id"]))] = torch.tensor(
            row["values_f32"], dtype=torch.float32
        )
    return result, sha256_file(path)


def load_selection(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        payload = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(payload, list)
        or not payload
        or any(not isinstance(row, dict) for row in payload)
    ):
        raise SystemExit("tensor selection must be a nonempty JSON list")
    forbidden = {
        "qtype_ud",
        "qtype_static",
        "ordinal_ud",
        "ordinal_static",
        "promotion_delta_ordinal",
        "promotion_delta_bpp",
        "promoted",
        "candidate_score",
    }
    leaked = sorted({key for row in payload for key in forbidden if key in row})
    if leaked:
        raise SystemExit(f"tensor selection contains forbidden label/score keys: {leaked}")
    return payload


def load_examples(path: Path) -> list[dict[str, Any]]:
    return list(COLLECTOR.iter_examples(path))


def make_batches(tokenizer, examples: list[dict[str, Any]], batch_size: int, sequence_length: int):
    batches = []
    for start in range(0, len(examples), batch_size):
        members = examples[start : start + batch_size]
        tensors, _ = COLLECTOR.encode_examples(tokenizer, members, sequence_length, False)
        batches.append(
            {
                "examples": members,
                "tensors": tensors,
                "record_ids": [str(item["record_id"]) for item in members],
                "domains": [str(item.get("domain", "unknown")) for item in members],
            }
        )
    return batches


def prepare_formal_c4_shards(
    tokenizer,
    prompt_paths: list[Path],
    batch_size: int,
    sequence_length: int,
) -> list[dict[str, Any]]:
    """Load four disjoint prompt shards in their explicit command-line order."""

    if len(prompt_paths) != FORMAL_C4_SHARD_COUNT:
        raise SystemExit(
            f"formal C4 requires exactly {FORMAL_C4_SHARD_COUNT} --prompt-shard arguments"
        )
    if len(set(prompt_paths)) != len(prompt_paths):
        raise SystemExit("formal C4 prompt shard paths must be distinct")
    seen_record_ids: set[str] = set()
    shards = []
    for shard_index, path in enumerate(prompt_paths):
        examples = load_examples(path)
        if not examples:
            raise SystemExit(f"formal C4 prompt shard is empty: {path}")
        record_ids = [str(item["record_id"]) for item in examples]
        duplicates = seen_record_ids.intersection(record_ids)
        if duplicates:
            raise SystemExit(
                "formal C4 prompt shards overlap in record IDs: "
                + ", ".join(sorted(duplicates))
            )
        if len(set(record_ids)) != len(record_ids):
            raise SystemExit(f"formal C4 prompt shard contains duplicate record IDs: {path}")
        seen_record_ids.update(record_ids)
        shards.append(
            {
                "shard_index": shard_index,
                "path": path,
                "sha256": sha256_file(path),
                "examples": examples,
                "batches": make_batches(tokenizer, examples, batch_size, sequence_length),
                "record_ids": record_ids,
                "domains": [str(item.get("domain", "unknown")) for item in examples],
            }
        )
    return shards


def model_module(model: torch.nn.Module, hf_stem: str) -> tuple[str, torch.nn.Module]:
    modules = dict(model.named_modules())
    without_model = hf_stem.removeprefix("model.")
    without_language_model = without_model.removeprefix("language_model.")
    candidates = list(
        dict.fromkeys(
            [
                hf_stem,
                without_model,
                f"model.{without_model}",
                without_language_model,
                f"model.{without_language_model}",
                f"language_model.{without_language_model}",
            ]
        )
    )
    for name in candidates:
        module = modules.get(name)
        if module is not None:
            return name, module
    # Multimodal source checkpoints expose text weights below
    # ``model.language_model.layers``. AutoModelForCausalLM deliberately loads
    # the text-only wrapper, where the same modules live below ``model.layers``.
    # Resolve that wrapper difference by a unique decoder-layer suffix, and
    # fail closed if an architecture exposes more than one matching tower.
    layer_marker = "layers."
    marker_index = hf_stem.find(layer_marker)
    if marker_index >= 0:
        layer_suffix = hf_stem[marker_index:]
        matches = [
            name
            for name in modules
            if name == layer_suffix or name.endswith(f".{layer_suffix}")
        ]
        if len(matches) == 1:
            return matches[0], modules[matches[0]]
        if len(matches) > 1:
            raise KeyError(
                f"ambiguous model module for {hf_stem}; decoder suffix "
                f"{layer_suffix!r} matched {matches}"
            )
    raise KeyError(f"model module not found for {hf_stem}; tried {candidates}")


def tensor_linear_module(model: torch.nn.Module, tensor_name: str) -> tuple[str, torch.nn.Linear]:
    name, module = model_module(model, tensor_name.removesuffix(".weight"))
    if not isinstance(module, torch.nn.Linear):
        raise TypeError(f"target is not Linear: {name} -> {type(module).__name__}")
    return name, module


def enclosing_layer_module(model: torch.nn.Module, linear_name: str) -> tuple[str, torch.nn.Module]:
    match = re.search(r"^(.*\.layers\.\d+)(?:\.|$)", linear_name)
    if match is None:
        raise KeyError(f"cannot derive layer module from {linear_name}")
    return model_module(model, match.group(1))


def hidden_from_output(output: Any) -> torch.Tensor:
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)) and output and torch.is_tensor(output[0]):
        return output[0]
    if hasattr(output, "last_hidden_state") and torch.is_tensor(output.last_hidden_state):
        return output.last_hidden_state
    raise TypeError(f"cannot extract block hidden state from {type(output).__name__}")


def nearest_sorted_codebook(values: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
    if not bool((codebook[1:] >= codebook[:-1]).all()):
        raise ValueError("family codebook must be sorted")
    index = torch.searchsorted(codebook, values).clamp(0, codebook.numel() - 1)
    previous = (index - 1).clamp(0, codebook.numel() - 1)
    # Existing argmin picks the lower codebook index on exact distance ties.
    choose_previous = (values - codebook[previous]).abs() <= (values - codebook[index]).abs()
    return torch.where(choose_previous, previous, index)


def quantize_weight_exact_contract(
    weight: torch.Tensor,
    tensor_name: str,
    activation_second_moment: torch.Tensor,
    candidate,
    codebook: torch.Tensor,
    max_fit_elements: int,
    scale_window: int,
    group_chunk: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    source = weight.detach().to("cpu", dtype=torch.float32)
    if source.ndim != 2 or source.shape[1] < 1:
        raise ValueError("formal perturbation quantizer requires a rank-2 weight")
    fit_groups, _ = SAMPLER.deterministic_group_partition_with_columns(
        source,
        candidate.group_size,
        max_fit_elements,
        seed=0,
        tensor_name=tensor_name,
        partition="fit",
    )
    scales = SAMPLER.scale_values(candidate.scale_format)
    tensor_scale = SAMPLER.choose_tensor_scale(fit_groups, candidate, scales, codebook)
    flat = source.flatten()
    usable = (flat.numel() // candidate.group_size) * candidate.group_size
    groups = flat[:usable].view(-1, candidate.group_size)
    output = torch.empty(flat.shape, dtype=weight.dtype, device="cpu")
    columns_per_row = int(source.shape[1])
    element_offsets = torch.arange(candidate.group_size, dtype=torch.long)
    max_code = codebook.abs().max().clamp_min(1e-12)
    index_counts = torch.zeros(candidate.codebook_entries, dtype=torch.long)
    scale_min = scales.numel()
    scale_max = 0
    improved = 0
    sse = 0.0
    ref_sse = 0.0

    with torch.inference_mode():
        for start in range(0, groups.shape[0], group_chunk):
            end = min(start + group_chunk, groups.shape[0])
            chunk = groups[start:end]
            group_ids = torch.arange(start, end, dtype=torch.long)
            columns = (group_ids[:, None] * candidate.group_size + element_offsets[None, :]) % columns_per_row
            weights = activation_second_moment.index_select(0, columns.flatten()).view_as(chunk).to(torch.float32)
            scaled = chunk / tensor_scale
            target = scaled.abs().amax(dim=1) / max_code
            center = SAMPLER.nearest_scale_indices(target, scales)
            best_error = torch.full((chunk.shape[0],), torch.inf, dtype=torch.float32)
            best_recon = torch.zeros_like(chunk)
            best_indices = torch.zeros_like(chunk, dtype=torch.long)
            best_scale_indices = torch.zeros(chunk.shape[0], dtype=torch.long)
            for offset in range(-scale_window, scale_window + 1):
                scale_indices = (center + offset).clamp(0, scales.numel() - 1)
                group_scales = scales.index_select(0, scale_indices)
                normalized = scaled / group_scales[:, None]
                indices = nearest_sorted_codebook(normalized, codebook)
                quantized = codebook.index_select(0, indices.flatten()).view_as(chunk)
                reconstruction = quantized * group_scales[:, None] * tensor_scale
                error = ((chunk - reconstruction).square() * weights).sum(dim=1)
                use = error < best_error
                best_error = torch.where(use, error, best_error)
                best_recon = torch.where(use[:, None], reconstruction, best_recon)
                best_indices = torch.where(use[:, None], indices, best_indices)
                best_scale_indices = torch.where(use, scale_indices, best_scale_indices)
            output[:usable].view(-1, candidate.group_size)[start:end] = best_recon.to(weight.dtype)
            index_counts += torch.bincount(best_indices.flatten(), minlength=candidate.codebook_entries)
            scale_min = min(scale_min, int(best_scale_indices.min()))
            scale_max = max(scale_max, int(best_scale_indices.max()))
            improved += int((best_scale_indices != center).sum())
            sse += float((chunk - best_recon).to(torch.float64).square().sum())
            ref_sse += float(chunk.to(torch.float64).square().sum())
    if usable < flat.numel():
        output[usable:] = flat[usable:].to(weight.dtype)
    return output.view_as(weight), {
        "tensor_scale": tensor_scale,
        "usable_elements": usable,
        "groups": int(groups.shape[0]),
        "group_chunk": group_chunk,
        "scale_index_min": scale_min,
        "scale_index_max": scale_max,
        "scale_window_improved_groups": improved,
        "unweighted_relative_mse_diagnostic": sse / max(ref_sse, 1e-30),
        "index_counts": [int(value) for value in index_counts.tolist()],
    }


def cache_metadata(
    args: argparse.Namespace,
    tensor_name: str,
    family: str,
    candidate_id: str,
    weight: torch.Tensor,
    codebook: torch.Tensor,
    codebook_file_sha: str,
    stats_sha: str,
) -> dict[str, Any]:
    return {
        "schema_version": "importance-quantized-weight-cache-v0.1",
        "tensor_name": tensor_name,
        "family": family,
        "candidate_id": candidate_id,
        "shape": list(weight.shape),
        "dtype": str(weight.dtype),
        "codebook_values": [float(value) for value in codebook.tolist()],
        "codebook_file_sha256": codebook_file_sha,
        "activation_stats_sha256": stats_sha,
        "max_fit_elements": args.max_fit_elements,
        "scale_window": args.scale_window,
        "group_chunk": args.group_chunk,
        "sampler_sha256": sha256_file(Path(__file__).resolve().parent / "run-aq-tensor-sample.py"),
    }


def load_or_quantize(
    args: argparse.Namespace,
    tensor_name: str,
    family: str,
    weight: torch.Tensor,
    activation: torch.Tensor,
    candidate,
    codebook: torch.Tensor,
    codebook_file_sha: str,
    stats_sha: str,
) -> tuple[torch.Tensor, dict[str, Any], Path]:
    metadata = cache_metadata(
        args, tensor_name, family, candidate.candidate_id, weight, codebook, codebook_file_sha, stats_sha
    )
    cache_path = args.quantized_cache_dir / f"{canonical_sha(metadata)}.pt"
    if cache_path.is_file():
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        if payload.get("metadata") != metadata:
            raise ValueError(f"quantized cache metadata mismatch: {cache_path}")
        return payload["weight"], payload["quantization"], cache_path
    quantized, quantization = quantize_weight_exact_contract(
        weight,
        tensor_name,
        activation,
        candidate,
        codebook,
        args.max_fit_elements,
        args.scale_window,
        args.group_chunk,
    )
    tmp = cache_path.with_name(f"{cache_path.name}.{os.getpid()}.tmp")
    torch.save(
        {"metadata": metadata, "quantization": quantization, "weight": quantized},
        tmp,
    )
    tmp.replace(cache_path)
    return quantized, quantization, cache_path


class FormalC4EphemeralCache:
    """Run-owned, layer-scoped cache with a strict active-byte ceiling."""

    def __init__(
        self,
        base_dir: Path,
        run_id: str,
        output_path: Path,
        max_active_bytes: int,
    ) -> None:
        if max_active_bytes < 1 or max_active_bytes > FORMAL_C4_TEMP_CACHE_LIMIT_BYTES:
            raise ValueError(
                "formal C4 cache limit must be in [1, 8 GiB]"
            )
        base_dir.mkdir(parents=True, exist_ok=True)
        owner = canonical_sha(
            {
                "run_id": run_id,
                "output": str(output_path),
                "pid": os.getpid(),
                "time_ns": time.time_ns(),
            }
        )[:16]
        self.root = base_dir / f"formal-c4-{owner}"
        self.root.mkdir(parents=False, exist_ok=False)
        self.max_active_bytes = int(max_active_bytes)
        self.active_bytes = 0
        self.peak_bytes = 0
        self.layer_peak_bytes = 0
        self.files_created = 0
        self._active_paths: list[Path] = []
        self._created_directories: list[Path] = []
        self._layer_dir: Path | None = None

    def begin_layer(self, layer_name: str) -> Path:
        if self._active_paths or self._layer_dir is not None or self.active_bytes:
            raise RuntimeError("formal C4 cache layer changed before prior cleanup")
        layer_key = canonical_sha({"layer": layer_name})[:16]
        layer_dir = self.root / f"layer-{layer_key}"
        layer_dir.mkdir(parents=False, exist_ok=False)
        self._created_directories.append(layer_dir)
        self._layer_dir = layer_dir
        self.layer_peak_bytes = 0
        return layer_dir

    def store(
        self,
        metadata: dict[str, Any],
        quantization: dict[str, Any],
        quantized: torch.Tensor,
    ) -> Path:
        if self._layer_dir is None:
            raise RuntimeError("formal C4 cache has no active layer")
        payload = {
            "metadata": metadata,
            "quantization": quantization,
            "weight": quantized,
        }
        buffer = io.BytesIO()
        torch.save(payload, buffer)
        serialized_bytes = buffer.getbuffer().nbytes
        if self.active_bytes + serialized_bytes > self.max_active_bytes:
            raise RuntimeError(
                "formal C4 active temporary cache would exceed 8 GiB hard cap: "
                f"active={self.active_bytes}, next={serialized_bytes}, "
                f"configured_limit={self.max_active_bytes}"
            )
        path = self._layer_dir / f"{canonical_sha(metadata)}.pt"
        if path.exists():
            raise RuntimeError(f"formal C4 cache path collision: {path}")
        try:
            with path.open("xb") as handle:
                handle.write(buffer.getbuffer())
        except BaseException:
            # The path is run-owned but is not tracked until its write completes.
            # Remove a partial file so cleanup_layer() can always remove the layer.
            if path.exists():
                path.unlink()
            raise
        actual_bytes = path.stat().st_size
        if actual_bytes != serialized_bytes:
            path.unlink()
            raise RuntimeError("formal C4 cache serialized byte count changed while writing")
        self._active_paths.append(path)
        self.active_bytes += actual_bytes
        self.peak_bytes = max(self.peak_bytes, self.active_bytes)
        self.layer_peak_bytes = max(self.layer_peak_bytes, self.active_bytes)
        self.files_created += 1
        return path

    @staticmethod
    def load(path: Path, expected_metadata: dict[str, Any]) -> dict[str, Any]:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("metadata") != expected_metadata:
            raise ValueError(f"formal C4 cache metadata mismatch: {path}")
        return payload

    def cleanup_layer(self) -> None:
        for path in reversed(self._active_paths):
            if path.exists():
                size = path.stat().st_size
                path.unlink()
                self.active_bytes -= size
        self._active_paths.clear()
        if self.active_bytes != 0:
            raise RuntimeError("formal C4 cache active-byte accounting did not return to zero")
        if self._layer_dir is not None:
            if self._layer_dir.exists():
                self._layer_dir.rmdir()
            self._layer_dir = None

    def cleanup_run(self) -> None:
        self.cleanup_layer()
        for directory in reversed(self._created_directories):
            if directory.exists():
                directory.rmdir()
        if self.root.exists():
            self.root.rmdir()


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch["tensors"].items()}


def run_model(model, tensors: dict[str, torch.Tensor]):
    try:
        return model(**tensors, use_cache=False)
    except TypeError as exc:
        if "use_cache" not in str(exc):
            raise
        return model(**tensors)


def clone_call_value(value: Any) -> Any:
    """Detach one captured block-call value from the full-model forward."""

    if torch.is_tensor(value):
        return value.detach().to("cpu").clone()
    if isinstance(value, tuple):
        return tuple(clone_call_value(item) for item in value)
    if isinstance(value, list):
        return [clone_call_value(item) for item in value]
    if isinstance(value, Mapping):
        # Gemma 4 passes shared KV state in collections.UserDict.  A plain
        # detached dict preserves the layer-facing mapping semantics while
        # ensuring every isolated replay receives an independent mutable
        # container.
        return {key: clone_call_value(item) for key, item in value.items()}
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f"unsupported captured block-call value: {type(value).__name__}")


def move_call_value(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, tuple):
        return tuple(move_call_value(item, device) for item in value)
    if isinstance(value, list):
        return [move_call_value(item, device) for item in value]
    if isinstance(value, Mapping):
        return {key: move_call_value(item, device) for key, item in value.items()}
    return value


def reference_c4(model, batches, layer_modules: dict[str, torch.nn.Module], device: torch.device):
    captured_outputs: dict[str, torch.Tensor] = {}
    captured_calls: dict[str, tuple[tuple[Any, ...], dict[str, Any]]] = {}
    references: dict[str, list[torch.Tensor]] = {name: [] for name in layer_modules}
    calls: dict[str, list[tuple[tuple[Any, ...], dict[str, Any]]]] = {
        name: [] for name in layer_modules
    }
    handles = []
    for name, module in layer_modules.items():
        def pre_hook(_module, args, kwargs, *, _name=name):
            captured_calls[_name] = (
                clone_call_value(args),
                clone_call_value(kwargs),
            )

        def hook(_module, _inputs, output, *, _name=name):
            captured_outputs[_name] = (
                hidden_from_output(output).detach().to("cpu", dtype=torch.bfloat16).clone()
            )

        handles.append(module.register_forward_pre_hook(pre_hook, with_kwargs=True))
        handles.append(module.register_forward_hook(hook))
    try:
        with torch.inference_mode():
            for batch in batches:
                captured_outputs.clear()
                captured_calls.clear()
                run_model(model, move_batch(batch, device))
                if set(captured_outputs) != set(layer_modules):
                    raise RuntimeError("reference block output hook coverage is incomplete")
                if set(captured_calls) != set(layer_modules):
                    raise RuntimeError("reference block input hook coverage is incomplete")
                for name in layer_modules:
                    references[name].append(captured_outputs[name])
                    calls[name].append(captured_calls[name])
    finally:
        for handle in handles:
            handle.remove()
    return {"outputs": references, "calls": calls}


def validate_c4_call_cache(layer_modules, reference, device: torch.device) -> dict[str, Any]:
    """Prove that isolated blocks reproduce their captured reference outputs."""

    result = {}
    with torch.inference_mode():
        for layer_name, layer_module in layer_modules.items():
            numerator = 0.0
            denominator = 0.0
            max_abs = 0.0
            calls = reference["calls"][layer_name]
            outputs = reference["outputs"][layer_name]
            for (args, kwargs), expected in zip(calls, outputs, strict=True):
                actual = hidden_from_output(
                    layer_module(
                        *move_call_value(args, device),
                        **move_call_value(kwargs, device),
                    )
                ).detach().to("cpu", dtype=torch.bfloat16)
                diff = actual.to(torch.float64) - expected.to(torch.float64)
                numerator += float(diff.square().sum())
                denominator += float(expected.to(torch.float64).square().sum())
                max_abs = max(max_abs, float(diff.abs().max()))
            relative_l2 = math.sqrt(numerator / max(denominator, 1e-30))
            if relative_l2 > 1e-7 or max_abs > 1e-6:
                raise RuntimeError(
                    f"cached C4 block call does not reproduce {layer_name}: "
                    f"relative_l2={relative_l2}, max_abs={max_abs}"
                )
            result[layer_name] = {
                "batch_count": len(calls),
                "relative_l2": relative_l2,
                "max_abs_error": max_abs,
            }
    return result


def candidate_c4_sums(
    batches,
    layer_name: str,
    layer_module,
    reference,
    device: torch.device,
) -> dict[str, float | int]:
    numerator = 0.0
    denominator = 0.0
    tokens = 0
    calls = reference["calls"][layer_name]
    outputs = reference["outputs"][layer_name]
    with torch.inference_mode():
        for batch, (args, kwargs), expected in zip(batches, calls, outputs, strict=True):
            candidate = hidden_from_output(
                layer_module(
                    *move_call_value(args, device),
                    **move_call_value(kwargs, device),
                )
            ).detach().to("cpu", dtype=torch.bfloat16)
            mask = batch["tensors"].get("attention_mask")
            if mask is None:
                valid = torch.ones(expected.shape[:-1], dtype=torch.bool)
            else:
                valid = mask.to("cpu", dtype=torch.bool)
            diff = candidate.to(torch.float64) - expected.to(torch.float64)
            ref64 = expected.to(torch.float64)
            numerator += float(diff.square().sum(dim=-1)[valid].sum())
            denominator += float(ref64.square().sum(dim=-1)[valid].sum())
            tokens += int(valid.sum())
    return {
        "numerator": numerator,
        "reference_denominator": denominator,
        "valid_tokens": tokens,
    }


def merge_c4_sums(parts: list[dict[str, float | int]]) -> dict[str, float | int]:
    return {
        "numerator": math.fsum(float(part["numerator"]) for part in parts),
        "reference_denominator": math.fsum(
            float(part["reference_denominator"]) for part in parts
        ),
        "valid_tokens": sum(int(part["valid_tokens"]) for part in parts),
    }


def c4_metrics_from_sums(
    sums: dict[str, float | int],
    *,
    execution: str = "isolated target block on cached BF16 reference block inputs",
) -> dict[str, float | int | str]:
    numerator = float(sums["numerator"])
    denominator = float(sums["reference_denominator"])
    tokens = int(sums["valid_tokens"])
    return {
        "C4_A": numerator / max(tokens, 1),
        "C4_reference_energy": denominator / max(tokens, 1),
        "C4_L": numerator / max(denominator, 1e-30),
        "valid_tokens": tokens,
        "execution": execution,
    }


def candidate_c4(batches, layer_name: str, layer_module, reference, device: torch.device):
    return c4_metrics_from_sums(
        candidate_c4_sums(batches, layer_name, layer_module, reference, device)
    )


def reference_c4_one_layer(
    model,
    batches,
    layer_name: str,
    layer_module: torch.nn.Module,
    device: torch.device,
):
    """Capture exactly one model/layer/shard reference working set."""

    return reference_c4(model, batches, {layer_name: layer_module}, device)


def formal_c4_input_signature(
    args: argparse.Namespace,
    selection: list[dict[str, Any]],
    candidates: list[Any],
    shards: list[dict[str, Any]],
    codebook_file_sha: str,
    stats_sha: str,
) -> str:
    return canonical_sha(
        {
            "schema_version": "importance-score-formal-c4-input-v0.1",
            "run_id": args.run_id,
            "model_dir": str(args.model_dir),
            "tensor_selection_sha256": sha256_file(args.tensor_selection),
            "selected_tensor_names": [str(row["hf_name"]) for row in selection],
            "candidate_ids": [candidate.candidate_id for candidate in candidates],
            "prompt_shards": [
                {
                    "index": int(shard["shard_index"]),
                    "sha256": str(shard["sha256"]),
                    "record_ids": list(shard["record_ids"]),
                }
                for shard in shards
            ],
            "activation_stats_sha256": stats_sha,
            "family_codebooks_sha256": codebook_file_sha,
            "batch_size": args.batch_size,
            "sequence_length": args.sequence_length,
            "max_fit_elements": args.max_fit_elements,
            "scale_window": args.scale_window,
            "group_chunk": args.group_chunk,
            "dtype": args.dtype,
            "seed": args.seed,
            "sampler_sha256": sha256_file(
                Path(__file__).resolve().parent / "run-aq-tensor-sample.py"
            ),
        }
    )


def completed_work_keys(
    output: Path,
    mode: str,
    *,
    formal_c4_signature: str | None = None,
) -> set[tuple[str, str, str]]:
    completed: set[tuple[str, str, str]] = set()
    if not output.is_file():
        return completed
    for line_number, line in enumerate(output.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("status") != "ok" or row.get("mode") != mode:
            continue
        if formal_c4_signature is not None:
            observed = row.get("formal_c4_input_signature_sha256")
            if observed != formal_c4_signature:
                raise SystemExit(
                    "formal C4 resume row has different frozen inputs at "
                    f"{output}:{line_number}"
                )
        completed.add((str(row["tensor_name"]), str(row["candidate_id"]), mode))
    return completed


def _layer_sort_key(layer_name: str) -> tuple[int, str]:
    match = re.search(r"\.layers\.(\d+)(?:\.|$)", layer_name)
    return (int(match.group(1)) if match else 2**31 - 1, layer_name)


def run_formal_c4_streaming(
    args: argparse.Namespace,
    model,
    selection: list[dict[str, Any]],
    candidates: list[Any],
    codebooks: dict[tuple[str, str], torch.Tensor],
    codebook_file_sha: str,
    activation_stats: dict[str, torch.Tensor],
    stats_sha: str,
    target_modules: dict[str, tuple[str, torch.nn.Linear]],
    layer_modules: dict[str, torch.nn.Module],
    target_layers: dict[str, str],
    shards: list[dict[str, Any]],
    device: torch.device,
    completed: set[tuple[str, str, str]],
    input_signature: str,
    overall_started_at: float,
) -> dict[str, Any]:
    work_keys = {
        (str(selected["hf_name"]), candidate.candidate_id, "c4")
        for selected in selection
        for candidate in candidates
    }
    completed_before_run = len(work_keys & completed)
    remaining_before_run = len(work_keys) - completed_before_run
    completed_this_run = 0
    candidate_stage_started_at = time.perf_counter()
    cache = FormalC4EphemeralCache(
        args.quantized_cache_dir,
        args.run_id,
        args.output,
        args.formal_c4_max_cache_bytes,
    )
    selected_by_layer: dict[str, list[dict[str, Any]]] = {}
    for selected in selection:
        layer_name = target_layers[str(selected["hf_name"])]
        selected_by_layer.setdefault(layer_name, []).append(selected)
    prompt_shard_metadata = [
        {
            "shard_index": int(shard["shard_index"]),
            "path": str(shard["path"]),
            "sha256": str(shard["sha256"]),
            "record_count": len(shard["examples"]),
            "record_ids": list(shard["record_ids"]),
            "domains": list(shard["domains"]),
        }
        for shard in shards
    ]
    all_record_ids = [record_id for shard in shards for record_id in shard["record_ids"]]

    emit_progress(
        {
            "event": "stage_start",
            "stage": "formal_c4_layer_shard_streaming",
            "run_id": args.run_id,
            "mode": "c4",
            "layer_count": len(selected_by_layer),
            "shard_count": len(shards),
            "tensor_candidates_total": len(work_keys),
            "tensor_candidates_already_completed": completed_before_run,
            "tensor_candidates_remaining": remaining_before_run,
            "elapsed_seconds": time.perf_counter() - overall_started_at,
        }
    )

    try:
        with args.output.open("a", encoding="utf-8", newline="\n") as handle:
            for layer_name in sorted(selected_by_layer, key=_layer_sort_key):
                layer_rows = selected_by_layer[layer_name]
                pending: list[tuple[dict[str, Any], Any]] = []
                for selected in layer_rows:
                    tensor_name = str(selected["hf_name"])
                    for candidate in candidates:
                        key = (tensor_name, candidate.candidate_id, "c4")
                        if key not in completed:
                            pending.append((selected, candidate))
                if not pending:
                    continue

                layer_started_at = time.perf_counter()
                emit_progress(
                    {
                        "event": "stage_start",
                        "stage": "formal_c4_layer",
                        "run_id": args.run_id,
                        "mode": "c4",
                        "layer_name": layer_name,
                        "pending_tensor_candidates": len(pending),
                    }
                )
                cache.begin_layer(layer_name)
                prepared: dict[tuple[str, str, str], dict[str, Any]] = {}
                shard_sums: dict[tuple[str, str, str], list[dict[str, float | int]]] = {
                    (str(selected["hf_name"]), candidate.candidate_id, "c4"): []
                    for selected, candidate in pending
                }
                elapsed_by_key = {key: 0.0 for key in shard_sums}
                shard_audits: list[dict[str, Any]] = []
                layer_peak_temp_bytes = 0
                try:
                    for selected, candidate in pending:
                        tensor_name = str(selected["hf_name"])
                        family = str(selected["canonical_family"])
                        key = (tensor_name, candidate.candidate_id, "c4")
                        _linear_name, linear = target_modules[tensor_name]
                        original_parameter = linear._parameters["weight"]
                        if original_parameter is None:
                            raise RuntimeError(f"target weight parameter missing: {tensor_name}")
                        activation = SAMPLER.activation_stats_for_tensor(
                            tensor_name,
                            tuple(int(value) for value in original_parameter.shape),
                            activation_stats,
                        )
                        codebook = codebooks.get((family, candidate.candidate_id))
                        if codebook is None:
                            raise SystemExit(
                                f"codebook missing: {family}/{candidate.candidate_id}"
                            )
                        synchronize_device(device)
                        quantize_started_at = time.perf_counter()
                        quantized, quantization = quantize_weight_exact_contract(
                            original_parameter,
                            tensor_name,
                            activation,
                            candidate,
                            codebook,
                            args.max_fit_elements,
                            args.scale_window,
                            args.group_chunk,
                        )
                        metadata = cache_metadata(
                            args,
                            tensor_name,
                            family,
                            candidate.candidate_id,
                            original_parameter,
                            codebook,
                            codebook_file_sha,
                            stats_sha,
                        )
                        cache_path = cache.store(metadata, quantization, quantized)
                        cache_elapsed_seconds = elapsed_since(device, quantize_started_at)
                        elapsed_by_key[key] += cache_elapsed_seconds
                        prepared[key] = {
                            "selected": selected,
                            "candidate": candidate,
                            "cache_path": cache_path,
                            "cache_metadata": metadata,
                            "quantization": quantization,
                        }
                        del quantized
                        emit_progress(
                            {
                                "event": "progress",
                                "stage": "formal_c4_layer_cache",
                                "run_id": args.run_id,
                                "mode": "c4",
                                "layer_name": layer_name,
                                "tensor_name": tensor_name,
                                "candidate_id": candidate.candidate_id,
                                "active_temp_bytes": cache.active_bytes,
                                "max_active_temp_bytes": cache.max_active_bytes,
                                "tensor_candidate_cache_elapsed_seconds": cache_elapsed_seconds,
                            }
                        )
                    layer_peak_temp_bytes = cache.layer_peak_bytes

                    for shard in shards:
                        shard_index = int(shard["shard_index"])
                        shard_started_at = time.perf_counter()
                        emit_progress(
                            {
                                "event": "stage_start",
                                "stage": "formal_c4_reference_shard",
                                "run_id": args.run_id,
                                "mode": "c4",
                                "layer_name": layer_name,
                                "shard_index": shard_index,
                                "batch_count": len(shard["batches"]),
                                "record_count": len(shard["examples"]),
                            }
                        )
                        reference = reference_c4_one_layer(
                            model,
                            shard["batches"],
                            layer_name,
                            layer_modules[layer_name],
                            device,
                        )
                        cache_audit = validate_c4_call_cache(
                            {layer_name: layer_modules[layer_name]}, reference, device
                        )[layer_name]
                        for selected, candidate in pending:
                            tensor_name = str(selected["hf_name"])
                            key = (tensor_name, candidate.candidate_id, "c4")
                            _linear_name, linear = target_modules[tensor_name]
                            original_parameter = linear._parameters["weight"]
                            if original_parameter is None:
                                raise RuntimeError(
                                    f"target weight parameter missing: {tensor_name}"
                                )
                            synchronize_device(device)
                            evaluation_started_at = time.perf_counter()
                            payload = cache.load(
                                prepared[key]["cache_path"],
                                prepared[key]["cache_metadata"],
                            )
                            linear._parameters["weight"] = torch.nn.Parameter(
                                payload["weight"].to(
                                    device=original_parameter.device,
                                    dtype=original_parameter.dtype,
                                ),
                                requires_grad=False,
                            )
                            try:
                                sums = candidate_c4_sums(
                                    shard["batches"],
                                    layer_name,
                                    layer_modules[layer_name],
                                    reference,
                                    device,
                                )
                            finally:
                                linear._parameters["weight"] = original_parameter
                            shard_tensor_elapsed_seconds = elapsed_since(
                                device, evaluation_started_at
                            )
                            elapsed_by_key[key] += shard_tensor_elapsed_seconds
                            shard_sums[key].append(sums)
                            del payload
                            emit_progress(
                                {
                                    "event": "progress",
                                    "stage": "formal_c4_shard_tensor_candidate",
                                    "run_id": args.run_id,
                                    "mode": "c4",
                                    "layer_name": layer_name,
                                    "shard_index": shard_index,
                                    "tensor_name": tensor_name,
                                    "candidate_id": candidate.candidate_id,
                                    "valid_tokens": int(sums["valid_tokens"]),
                                    "shard_tensor_candidate_elapsed_seconds": (
                                        shard_tensor_elapsed_seconds
                                    ),
                                }
                            )
                        shard_audits.append(
                            {
                                "shard_index": shard_index,
                                "path": str(shard["path"]),
                                "sha256": str(shard["sha256"]),
                                "record_count": len(shard["examples"]),
                                "record_ids": list(shard["record_ids"]),
                                "batch_count": len(shard["batches"]),
                                "cache_audit": cache_audit,
                                "elapsed_seconds": elapsed_since(device, shard_started_at),
                            }
                        )
                        emit_progress(
                            {
                                "event": "stage_complete",
                                "stage": "formal_c4_reference_shard",
                                "run_id": args.run_id,
                                "mode": "c4",
                                "layer_name": layer_name,
                                "shard_index": shard_index,
                                "tensor_candidates_completed": len(pending),
                                "shard_elapsed_seconds": shard_audits[-1][
                                    "elapsed_seconds"
                                ],
                            }
                        )
                        del reference
                finally:
                    cache.cleanup_layer()

                for selected, candidate in pending:
                    tensor_name = str(selected["hf_name"])
                    family = str(selected["canonical_family"])
                    key = (tensor_name, candidate.candidate_id, "c4")
                    merged_sums = merge_c4_sums(shard_sums[key])
                    metrics = c4_metrics_from_sums(
                        merged_sums,
                        execution=(
                            "four fixed shards; one model x one layer x one shard BF16 "
                            "reference cache; FP64 scalar sum aggregation"
                        ),
                    )
                    row = {
                        "schema_version": "importance-score-single-tensor-perturbation-v0.1",
                        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                        "status": "ok",
                        "run_id": args.run_id,
                        "mode": "c4",
                        "model_dir": str(args.model_dir),
                        "tensor_name": tensor_name,
                        "canonical_family": family,
                        "layer_id": int(selected["layer_id"]),
                        "layer_module": layer_name,
                        "candidate_id": candidate.candidate_id,
                        "elapsed_seconds": elapsed_by_key[key],
                        "metrics": metrics,
                        "quantization": prepared[key]["quantization"],
                        "quantized_cache": str(prepared[key]["cache_path"]),
                        "quantized_cache_retained": False,
                        "prompt_file": None,
                        "prompt_file_sha256": None,
                        "prompt_shards": prompt_shard_metadata,
                        "tensor_selection": str(args.tensor_selection),
                        "tensor_selection_sha256": sha256_file(args.tensor_selection),
                        "record_count": len(all_record_ids),
                        "record_ids": all_record_ids,
                        "sequence_length": args.sequence_length,
                        "batch_size": args.batch_size,
                        "device": str(device),
                        "require_cpu": args.require_cpu,
                        "reference_dtype": args.dtype,
                        "codebook_file_sha256": codebook_file_sha,
                        "activation_stats_sha256": stats_sha,
                        "c4_reference_cache_audit": shard_audits,
                        "formal_c4_input_signature_sha256": input_signature,
                        "c4_streaming_contract": {
                            "formal": True,
                            "shard_count": FORMAL_C4_SHARD_COUNT,
                            "reference_cache_scope": "one model x one layer x one shard",
                            "reference_cache_dtype": "bfloat16",
                            "candidate_output_storage": "none; FP64-equivalent Python scalar sums only",
                            "aggregation": "sum numerator/reference denominator/token count across fixed shards before normalization",
                            "record_overlap_count": 0,
                        },
                        "c4_cache_policy": {
                            "mode": "run-owned layer-scoped ephemeral disk cache",
                            "max_active_temp_bytes": cache.max_active_bytes,
                            "layer_peak_temp_bytes": layer_peak_temp_bytes,
                            "run_peak_temp_bytes_at_layer": cache.peak_bytes,
                            "cache_files_created_for_layer": len(pending),
                            "cleanup_completed_before_row_write": True,
                            "run_owned_root": str(cache.root),
                        },
                        "notes": args.note,
                    }
                    handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                    handle.flush()
                    completed_this_run += 1
                    if args.progress_every_tensors and (
                        completed_this_run % args.progress_every_tensors == 0
                        or completed_this_run == remaining_before_run
                    ):
                        emit_progress(
                            {
                                "event": "progress",
                                "stage": "tensor_candidates",
                                "run_id": args.run_id,
                                "mode": "c4",
                                "layer_name": layer_name,
                                "tensor_name": tensor_name,
                                "candidate_id": candidate.candidate_id,
                                "tensor_candidate_elapsed_seconds": elapsed_by_key[key],
                                "stage_elapsed_seconds": elapsed_since(
                                    device, candidate_stage_started_at
                                ),
                                "tensor_candidates_completed_this_run": completed_this_run,
                                "tensor_candidates_completed_total": (
                                    completed_before_run + completed_this_run
                                ),
                                "tensor_candidates_total": len(work_keys),
                            }
                        )
                emit_progress(
                    {
                        "event": "stage_complete",
                        "stage": "formal_c4_layer",
                        "run_id": args.run_id,
                        "mode": "c4",
                        "layer_name": layer_name,
                        "layer_elapsed_seconds": elapsed_since(device, layer_started_at),
                        "layer_peak_temp_bytes": layer_peak_temp_bytes,
                    }
                )
    finally:
        cache.cleanup_run()

    return {
        "mode": "c4",
        "tensors": len(selection),
        "candidates": len(candidates),
        "tensor_candidates_total": len(work_keys),
        "tensor_candidates_already_completed": completed_before_run,
        "tensor_candidates_completed_this_run": completed_this_run,
        "formal_c4_streaming": True,
        "formal_c4_shards": len(shards),
        "peak_temp_cache_bytes": cache.peak_bytes,
        "cache_cleanup_completed": not cache.root.exists(),
        "elapsed_seconds": elapsed_since(device, overall_started_at),
    }


def reference_c6(model, batches, device: torch.device):
    result = []
    with torch.inference_mode():
        for batch in batches:
            tensors = move_batch(batch, device)
            logits = run_model(model, tensors).logits.detach().to("cpu", dtype=torch.bfloat16)
            mask = tensors.get("attention_mask")
            result.append(
                {
                    "logits": logits,
                    "mask": (
                        mask.detach().to("cpu", dtype=torch.bool)
                        if mask is not None
                        else torch.ones(logits.shape[:-1], dtype=torch.bool)
                    ),
                }
            )
    return result


def streaming_token_kl(reference: torch.Tensor, candidate: torch.Tensor, vocab_chunk: int) -> torch.Tensor:
    reference = reference.to(torch.float32)
    candidate = candidate.to(torch.float32)
    ref_lse = torch.logsumexp(reference, dim=-1)
    cand_lse = torch.logsumexp(candidate, dim=-1)
    kl = torch.zeros(reference.shape[:-1], dtype=torch.float64)
    for start in range(0, reference.shape[-1], vocab_chunk):
        end = min(start + vocab_chunk, reference.shape[-1])
        logp = reference[..., start:end] - ref_lse[..., None]
        logq = candidate[..., start:end] - cand_lse[..., None]
        p = logp.exp()
        kl += (p * (logp - logq)).to(torch.float64).sum(dim=-1)
    return kl


def candidate_c6(model, batches, references, device: torch.device, vocab_chunk: int):
    kl_values = []
    flips = 0
    probability_drops = []
    tokens = 0
    with torch.inference_mode():
        for batch, reference in zip(batches, references, strict=True):
            tensors = move_batch(batch, device)
            candidate_logits = run_model(model, tensors).logits.detach().to("cpu", dtype=torch.bfloat16)
            reference_logits = reference["logits"]
            mask = reference["mask"]
            token_kl = streaming_token_kl(reference_logits, candidate_logits, vocab_chunk)
            kl_values.extend(float(value) for value in token_kl[mask].tolist())
            ref32 = reference_logits.to(torch.float32)
            cand32 = candidate_logits.to(torch.float32)
            ref_top = ref32.argmax(dim=-1)
            cand_top = cand32.argmax(dim=-1)
            flips += int(((ref_top != cand_top) & mask).sum())
            ref_prob = torch.softmax(ref32, dim=-1).gather(-1, ref_top[..., None]).squeeze(-1)
            cand_prob = torch.softmax(cand32, dim=-1).gather(-1, ref_top[..., None]).squeeze(-1)
            probability_drops.extend(float(value) for value in (ref_prob - cand_prob)[mask].tolist())
            tokens += int(mask.sum())
    values = np.asarray(kl_values, dtype=np.float64)
    drops = np.asarray(probability_drops, dtype=np.float64)
    return {
        "C6_A": float(values.mean()),
        "C6_L": float(values.mean()),
        "p99_token_KL": float(np.quantile(values, 0.99)),
        "top1_flip_rate": flips / max(tokens, 1),
        "reference_top1_probability_drop_mean": float(drops.mean()),
        "reference_top1_probability_drop_p99": float(np.quantile(drops, 0.99)),
        "valid_tokens": tokens,
        "temperature": 1.0,
        "vocabulary_accumulation": f"exact full vocabulary in chunks of {vocab_chunk}",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("c4", "c6"), required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--activation-stats", type=Path, required=True)
    parser.add_argument("--family-codebooks", type=Path, required=True)
    parser.add_argument("--tensor-selection", type=Path, required=True)
    parser.add_argument("--prompt-file", type=Path)
    parser.add_argument(
        "--prompt-shard",
        type=Path,
        action="append",
        default=[],
        help="Formal C4 only: repeat exactly four times in frozen shard order.",
    )
    parser.add_argument("--candidate", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quantized-cache-dir", type=Path, required=True)
    parser.add_argument("--max-tensors", type=int)
    parser.add_argument(
        "--progress-every-tensors",
        type=int,
        default=0,
        help="Emit JSON progress every N completed tensor-candidates; 0 disables it.",
    )
    parser.add_argument(
        "--formal-c4-streaming",
        action="store_true",
        help="Enforce four-shard one-layer reference streaming and ephemeral cache limits.",
    )
    parser.add_argument(
        "--formal-c4-max-cache-bytes",
        type=int,
        default=FORMAL_C4_TEMP_CACHE_LIMIT_BYTES,
        help="Active formal C4 cache cap; may be lowered but never exceed 8 GiB.",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--max-fit-elements", type=int, default=65536)
    parser.add_argument("--scale-window", type=int, default=4)
    parser.add_argument("--group-chunk", type=int, default=4096)
    parser.add_argument("--vocab-chunk", type=int, default=16384)
    parser.add_argument("--dtype", choices=("auto", "float32", "bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--require-cpu", action="store_true")
    parser.add_argument("--torch-threads", type=int, default=32)
    parser.add_argument("--torch-interop-threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--note", action="append", default=[])
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.require_cpu and args.device != "cpu":
        raise SystemExit("--require-cpu requires --device cpu")
    if (
        min(args.batch_size, args.sequence_length, args.group_chunk, args.vocab_chunk) < 1
        or args.progress_every_tensors < 0
    ):
        raise SystemExit("batch/sequence/chunk arguments must be positive; progress must be >= 0")
    if args.max_tensors is not None and args.max_tensors < 1:
        raise SystemExit("--max-tensors must be positive")
    if args.formal_c4_streaming:
        if args.mode != "c4":
            raise SystemExit("--formal-c4-streaming is valid only with --mode c4")
        if args.prompt_file is not None:
            raise SystemExit("formal C4 uses --prompt-shard, not --prompt-file")
        if len(args.prompt_shard) != FORMAL_C4_SHARD_COUNT:
            raise SystemExit(
                f"formal C4 requires exactly {FORMAL_C4_SHARD_COUNT} --prompt-shard arguments"
            )
        if not 1 <= args.formal_c4_max_cache_bytes <= FORMAL_C4_TEMP_CACHE_LIMIT_BYTES:
            raise SystemExit("formal C4 active cache cap must be in [1, 8 GiB]")
    else:
        if args.prompt_file is None:
            raise SystemExit("non-formal C4/C6 requires --prompt-file")
        if args.prompt_shard:
            raise SystemExit("--prompt-shard requires --formal-c4-streaming")
    torch.set_num_threads(args.torch_threads)
    torch.set_num_interop_threads(args.torch_interop_threads)
    torch.manual_seed(args.seed)
    args.model_dir = args.model_dir.expanduser().resolve()
    args.activation_stats = args.activation_stats.expanduser().resolve()
    args.family_codebooks = args.family_codebooks.expanduser().resolve()
    args.tensor_selection = args.tensor_selection.expanduser().resolve()
    args.prompt_file = args.prompt_file.expanduser().resolve() if args.prompt_file else None
    args.prompt_shard = [path.expanduser().resolve() for path in args.prompt_shard]
    args.output = args.output.expanduser().resolve()
    args.quantized_cache_dir = args.quantized_cache_dir.expanduser().resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.quantized_cache_dir.mkdir(parents=True, exist_ok=True)

    overall_started_at = time.perf_counter()
    emit_progress(
        {
            "event": "stage_start",
            "stage": "prepare_inputs",
            "run_id": args.run_id,
            "mode": args.mode,
        }
    )
    selection = load_selection(args.tensor_selection)
    if args.max_tensors is not None:
        selection = selection[: args.max_tensors]
    candidates = []
    for candidate_id in args.candidate:
        candidate = SAMPLER.candidate_from_id(candidate_id)
        if candidate is None:
            raise SystemExit(f"unknown candidate: {candidate_id}")
        candidates.append(candidate)
    codebooks, codebook_file_sha = load_codebooks(args.family_codebooks)
    activation_stats = SAMPLER.load_activation_stats(args.activation_stats)
    stats_path = args.activation_stats
    if stats_path.is_dir():
        stats_path = stats_path / "activation_second_moments.safetensors"
    stats_sha = sha256_file(stats_path)
    args.model_class = "auto_model" if args.mode == "c4" else "causal_lm"
    emit_progress(
        {
            "event": "stage_start",
            "stage": "load_model",
            "run_id": args.run_id,
            "mode": args.mode,
            "elapsed_seconds": time.perf_counter() - overall_started_at,
        }
    )
    tokenizer, model = COLLECTOR.load_transformers_model(args)
    device = next(model.parameters()).device
    formal_shards = (
        prepare_formal_c4_shards(
            tokenizer,
            args.prompt_shard,
            args.batch_size,
            args.sequence_length,
        )
        if args.formal_c4_streaming
        else []
    )
    if args.formal_c4_streaming:
        examples = []
        batches = []
    else:
        assert args.prompt_file is not None
        examples = load_examples(args.prompt_file)
        batches = make_batches(tokenizer, examples, args.batch_size, args.sequence_length)

    target_modules = {}
    layer_modules = {}
    target_layers = {}
    for row in selection:
        tensor_name = str(row["hf_name"])
        linear_name, linear = tensor_linear_module(model, tensor_name)
        layer_name, layer = enclosing_layer_module(model, linear_name)
        target_modules[tensor_name] = (linear_name, linear)
        layer_modules[layer_name] = layer
        target_layers[tensor_name] = layer_name

    if args.formal_c4_streaming:
        input_signature = formal_c4_input_signature(
            args,
            selection,
            candidates,
            formal_shards,
            codebook_file_sha,
            stats_sha,
        )
        completed = completed_work_keys(
            args.output,
            "c4",
            formal_c4_signature=input_signature,
        )
        summary = run_formal_c4_streaming(
            args,
            model,
            selection,
            candidates,
            codebooks,
            codebook_file_sha,
            activation_stats,
            stats_sha,
            target_modules,
            layer_modules,
            target_layers,
            formal_shards,
            device,
            completed,
            input_signature,
            overall_started_at,
        )
        print(json.dumps(summary, sort_keys=True), flush=True)
        return 0

    synchronize_device(device)
    reference_started_at = time.perf_counter()
    emit_progress(
        {
            "event": "stage_start",
            "stage": f"{args.mode}_reference",
            "run_id": args.run_id,
            "mode": args.mode,
            "batch_count": len(batches),
            "elapsed_seconds": time.perf_counter() - overall_started_at,
        }
    )
    if args.mode == "c4":
        reference = reference_c4(model, batches, layer_modules, device)
        c4_cache_audit = validate_c4_call_cache(layer_modules, reference, device)
    else:
        reference = reference_c6(model, batches, device)
        c4_cache_audit = {}

    completed = completed_work_keys(args.output, args.mode)

    work_keys = {
        (str(selected["hf_name"]), candidate.candidate_id, args.mode)
        for selected in selection
        for candidate in candidates
    }
    completed_before_run = len(work_keys & completed)
    remaining_before_run = len(work_keys) - completed_before_run
    completed_this_run = 0
    synchronize_device(device)
    candidate_stage_started_at = time.perf_counter()
    emit_progress(
        {
            "event": "stage_start",
            "stage": "tensor_candidates",
            "run_id": args.run_id,
            "mode": args.mode,
            "tensor_candidates_total": len(work_keys),
            "tensor_candidates_already_completed": completed_before_run,
            "tensor_candidates_remaining": remaining_before_run,
            "reference_elapsed_seconds": elapsed_since(device, reference_started_at),
            "elapsed_seconds": time.perf_counter() - overall_started_at,
        }
    )
    with args.output.open("a", encoding="utf-8", newline="\n") as handle:
        for selected in selection:
            tensor_name = str(selected["hf_name"])
            family = str(selected["canonical_family"])
            linear_name, linear = target_modules[tensor_name]
            layer_name = target_layers[tensor_name]
            original_parameter = linear._parameters["weight"]
            if original_parameter is None:
                raise RuntimeError(f"target weight parameter missing: {linear_name}")
            activation = SAMPLER.activation_stats_for_tensor(
                tensor_name,
                tuple(int(value) for value in original_parameter.shape),
                activation_stats,
            )
            for candidate in candidates:
                key = (tensor_name, candidate.candidate_id, args.mode)
                if key in completed:
                    continue
                synchronize_device(device)
                tensor_candidate_started_at = time.perf_counter()
                codebook = codebooks.get((family, candidate.candidate_id))
                if codebook is None:
                    raise SystemExit(f"codebook missing: {family}/{candidate.candidate_id}")
                quantized, quantization, cache_path = load_or_quantize(
                    args,
                    tensor_name,
                    family,
                    original_parameter,
                    activation,
                    candidate,
                    codebook,
                    codebook_file_sha,
                    stats_sha,
                )
                linear._parameters["weight"] = torch.nn.Parameter(
                    quantized.to(device=original_parameter.device, dtype=original_parameter.dtype),
                    requires_grad=False,
                )
                try:
                    metrics = (
                        candidate_c4(
                            batches,
                            layer_name,
                            layer_modules[layer_name],
                            reference,
                            device,
                        )
                        if args.mode == "c4"
                        else candidate_c6(model, batches, reference, device, args.vocab_chunk)
                    )
                finally:
                    linear._parameters["weight"] = original_parameter
                tensor_candidate_elapsed_seconds = elapsed_since(
                    device, tensor_candidate_started_at
                )
                row = {
                    "schema_version": "importance-score-single-tensor-perturbation-v0.1",
                    "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                    "status": "ok",
                    "run_id": args.run_id,
                    "mode": args.mode,
                    "model_dir": str(args.model_dir),
                    "tensor_name": tensor_name,
                    "canonical_family": family,
                    "layer_id": int(selected["layer_id"]),
                    "layer_module": layer_name,
                    "candidate_id": candidate.candidate_id,
                    "elapsed_seconds": tensor_candidate_elapsed_seconds,
                    "metrics": metrics,
                    "quantization": quantization,
                    "quantized_cache": str(cache_path),
                    "prompt_file": str(args.prompt_file),
                    "prompt_file_sha256": sha256_file(args.prompt_file),
                    "tensor_selection": str(args.tensor_selection),
                    "tensor_selection_sha256": sha256_file(args.tensor_selection),
                    "record_count": len(examples),
                    "record_ids": [str(item["record_id"]) for item in examples],
                    "sequence_length": args.sequence_length,
                    "batch_size": args.batch_size,
                    "device": str(device),
                    "require_cpu": args.require_cpu,
                    "reference_dtype": args.dtype,
                    "codebook_file_sha256": codebook_file_sha,
                    "activation_stats_sha256": stats_sha,
                    "c4_reference_cache_audit": c4_cache_audit.get(layer_name),
                    "notes": args.note,
                }
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                handle.flush()
                completed_this_run += 1
                if args.progress_every_tensors and (
                    completed_this_run % args.progress_every_tensors == 0
                    or completed_this_run == remaining_before_run
                ):
                    emit_progress(
                        {
                            "event": "progress",
                            "stage": "tensor_candidates",
                            "run_id": args.run_id,
                            "mode": args.mode,
                            "tensor_name": tensor_name,
                            "candidate_id": candidate.candidate_id,
                            "tensor_candidate_elapsed_seconds": tensor_candidate_elapsed_seconds,
                            "stage_elapsed_seconds": elapsed_since(
                                device, candidate_stage_started_at
                            ),
                            "tensor_candidates_completed_this_run": completed_this_run,
                            "tensor_candidates_completed_total": (
                                completed_before_run + completed_this_run
                            ),
                            "tensor_candidates_total": len(work_keys),
                        }
                    )
                del quantized
    print(
        json.dumps(
            {
                "mode": args.mode,
                "tensors": len(selection),
                "candidates": len(candidates),
                "tensor_candidates_total": len(work_keys),
                "tensor_candidates_already_completed": completed_before_run,
                "tensor_candidates_completed_this_run": completed_this_run,
                "elapsed_seconds": elapsed_since(device, overall_started_at),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
