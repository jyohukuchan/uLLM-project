#!/usr/bin/env python3
"""Run bounded CPU-only C4 block perturbation or C6 full-vocabulary KL."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import math
import os
import re
import sys
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def model_module(model: torch.nn.Module, hf_stem: str) -> tuple[str, torch.nn.Module]:
    modules = dict(model.named_modules())
    candidates = [hf_stem, hf_stem.removeprefix("model."), f"model.{hf_stem.removeprefix('model.')}"]
    for name in candidates:
        module = modules.get(name)
        if module is not None:
            return name, module
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
    if isinstance(value, dict):
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
    if isinstance(value, dict):
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


def candidate_c4(batches, layer_name: str, layer_module, reference, device: torch.device):
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
        "C4_A": numerator / max(tokens, 1),
        "C4_reference_energy": denominator / max(tokens, 1),
        "C4_L": numerator / max(denominator, 1e-30),
        "valid_tokens": tokens,
        "execution": "isolated target block on cached BF16 reference block inputs",
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
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--candidate", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quantized-cache-dir", type=Path, required=True)
    parser.add_argument("--max-tensors", type=int)
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
    if min(args.batch_size, args.sequence_length, args.group_chunk, args.vocab_chunk) < 1:
        raise SystemExit("batch/sequence/chunk arguments must be positive")
    torch.set_num_threads(args.torch_threads)
    torch.set_num_interop_threads(args.torch_interop_threads)
    torch.manual_seed(args.seed)
    args.model_dir = args.model_dir.expanduser().resolve()
    args.activation_stats = args.activation_stats.expanduser().resolve()
    args.family_codebooks = args.family_codebooks.expanduser().resolve()
    args.tensor_selection = args.tensor_selection.expanduser().resolve()
    args.prompt_file = args.prompt_file.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    args.quantized_cache_dir = args.quantized_cache_dir.expanduser().resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.quantized_cache_dir.mkdir(parents=True, exist_ok=True)

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
    tokenizer, model = COLLECTOR.load_transformers_model(args)
    device = next(model.parameters()).device
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

    if args.mode == "c4":
        reference = reference_c4(model, batches, layer_modules, device)
        c4_cache_audit = validate_c4_call_cache(layer_modules, reference, device)
    else:
        reference = reference_c6(model, batches, device)
        c4_cache_audit = {}

    completed = set()
    if args.output.is_file():
        for line in args.output.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("status") == "ok":
                completed.add((row["tensor_name"], row["candidate_id"], row["mode"]))

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
                del quantized
    print(json.dumps({"mode": args.mode, "tensors": len(selection), "candidates": len(candidates)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
