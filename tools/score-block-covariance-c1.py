#!/usr/bin/env python3
"""Score AQ4/AQ5 tensor perturbations with C1 block covariance on CPU."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import importlib.util
import json
import math
import sys
import time
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open


def load_sampler():
    path = Path(__file__).resolve().parent / "run-aq-tensor-sample.py"
    spec = importlib.util.spec_from_file_location("aq_c1_sampler", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


SAMPLER = load_sampler()


def read_labels(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return [row for row in csv.DictReader(handle, delimiter="\t") if row["eligible"] == "true"]


def read_source_roster(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    forbidden = {
        "gguf_name",
        "qtype_ud",
        "qtype_static",
        "ordinal_ud",
        "promotion_delta_ordinal",
        "promotion_delta_bpp",
        "promoted",
    }
    leaked = sorted({key for row in rows for key in forbidden if key in row})
    if leaked:
        raise ValueError(f"source roster contains forbidden label keys: {leaked}")
    return rows


def load_codebooks(path: Path) -> dict[tuple[str, str], torch.Tensor]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    result = {}
    for row in payload["codebooks"]:
        key = (str(row["family"]), str(row["candidate_id"]))
        result[key] = torch.tensor(row["values_f32"], dtype=torch.float32)
    return result


def tensor_file_map(model_dir: Path) -> dict[str, Path]:
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        return {name: model_dir / filename for name, filename in index["weight_map"].items()}
    result = {}
    for path in sorted(model_dir.glob("*.safetensors")):
        with safe_open(path, framework="pt", device="cpu") as handle:
            for name in handle.keys():
                if name in result:
                    raise ValueError(f"duplicate safetensor key: {name}")
                result[name] = path
    if not result:
        raise FileNotFoundError(f"no safetensors weights under {model_dir}")
    return result


def load_merged_covariance(
    covariance_dirs: list[Path],
) -> tuple[dict[str, torch.Tensor], dict[str, str], dict[str, Any]]:
    if len(covariance_dirs) != 4:
        raise SystemExit("formal C1 requires exactly four covariance shards")
    metadata = [json.loads((path / "metadata.json").read_text(encoding="utf-8")) for path in covariance_dirs]
    aliases = metadata[0]["module_aliases"]
    if any(item["module_aliases"] != aliases for item in metadata[1:]):
        raise SystemExit("C1 module alias coverage differs across shards")
    keys = sorted(metadata[0]["modules"])
    if any(sorted(item["modules"]) != keys for item in metadata[1:]):
        raise SystemExit("C1 covariance module coverage differs across shards")
    merged = {}
    with ExitStack() as stack:
        handles = [
            stack.enter_context(
                safe_open(path / "block_covariance_128.safetensors", framework="pt", device="cpu")
            )
            for path in covariance_dirs
        ]
        for key in keys:
            counts = [int(item["modules"][key]["activation_count"]) for item in metadata]
            total = sum(counts)
            if total <= 0:
                raise SystemExit(f"zero covariance count: {key}")
            value = sum(
                (handles[i].get_tensor(key).to(torch.float64) * counts[i] for i in range(4)),
                torch.zeros_like(handles[0].get_tensor(key), dtype=torch.float64),
            ) / float(total)
            if not bool(torch.isfinite(value).all()):
                raise SystemExit(f"non-finite covariance: {key}")
            merged[key] = value.contiguous()
    summary = {
        "shards": [
            {
                "path": str(path),
                "samples": int(item["samples_seen"]),
                "tokens": int(item["tokens_seen"]),
            }
            for path, item in zip(covariance_dirs, metadata, strict=True)
        ],
        "samples": sum(int(item["samples_seen"]) for item in metadata),
        "tokens": sum(int(item["tokens_seen"]) for item in metadata),
        "block_size": int(metadata[0]["block_size"]),
        "product_dtype": metadata[0]["product_dtype"],
        "accumulation_and_storage_dtype": metadata[0]["accumulation_and_storage_dtype"],
    }
    return merged, aliases, summary


def module_name_for_hf_weight(hf_name: str) -> str:
    return hf_name.removeprefix("model.").removesuffix(".weight")


def fit_group_ids(tensor_name: str, group_count: int, max_elements: int, group_size: int) -> set[int]:
    max_groups = max(1, max_elements // group_size)
    fit_count = 1 if group_count == 1 else min(max_groups, group_count // 2)
    key = f"aq-fit-eval-v1\0{0}\0{tensor_name}\0{group_size}".encode("utf-8")
    digest = hashlib.sha256(key).digest()
    offset = int.from_bytes(digest[:8], "little") % group_count
    step = SAMPLER.coprime_step(group_count, int.from_bytes(digest[8:16], "little"))
    return set(
        int(value)
        for value in SAMPLER.affine_group_ids(group_count, 0, fit_count, offset, step).tolist()
    )


def select_eval_block_pairs(
    tensor_name: str,
    out_features: int,
    in_features: int,
    block_size: int,
    group_size: int,
    max_pairs: int,
    max_fit_elements: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if in_features % block_size or block_size % group_size:
        raise ValueError("C1 primary requires complete 128-channel blocks and 16-weight groups")
    in_blocks = in_features // block_size
    pair_count = out_features * in_blocks
    group_count = out_features * (in_features // group_size)
    excluded = fit_group_ids(tensor_name, group_count, max_fit_elements, group_size)
    key = hashlib.sha256(f"importance-c1-block-eval-v1\0{tensor_name}".encode()).digest()
    offset = int.from_bytes(key[:8], "little") % pair_count
    step = SAMPLER.coprime_step(pair_count, int.from_bytes(key[8:16], "little"))
    rows: list[int] = []
    blocks: list[int] = []
    groups_per_block = block_size // group_size
    groups_per_row = in_features // group_size
    for position in range(pair_count):
        pair = (offset + position * step) % pair_count
        row = pair // in_blocks
        block = pair % in_blocks
        first_group = row * groups_per_row + block * groups_per_block
        if any(first_group + index in excluded for index in range(groups_per_block)):
            continue
        rows.append(row)
        blocks.append(block)
        if len(rows) >= min(max_pairs, pair_count):
            break
    if not rows:
        raise ValueError(f"no disjoint C1 evaluation blocks for {tensor_name}")
    return torch.tensor(rows, dtype=torch.long), torch.tensor(blocks, dtype=torch.long)


def reconstruct_from_state(groups: torch.Tensor, state) -> torch.Tensor:
    group_scale = state.scales.index_select(0, state.local_indices)
    quantized = state.codebook.index_select(0, state.code_indices.flatten()).view_as(groups)
    return quantized * group_scale[:, None] * float(state.tensor_scale)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--labels", type=Path)
    selection.add_argument("--source-roster", type=Path)
    parser.add_argument("--activation-stats", type=Path, required=True)
    parser.add_argument("--family-codebooks", type=Path, required=True)
    parser.add_argument("--covariance-dir", action="append", type=Path, required=True)
    parser.add_argument("--candidate", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-fit-elements", type=int, default=65536)
    parser.add_argument("--max-block-pairs", type=int, default=512)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--scale-window", type=int, default=4)
    parser.add_argument("--torch-threads", type=int, default=32)
    parser.add_argument("--torch-interop-threads", type=int, default=1)
    parser.add_argument(
        "--progress-every-tensors",
        type=int,
        default=0,
        help="Emit a compact JSON progress row to stderr every N completed tensors; 0 disables it.",
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--note", action="append", default=[])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.block_size != 128:
        raise SystemExit("the formal C1 primary uses block size 128")
    if args.progress_every_tensors < 0:
        raise SystemExit("--progress-every-tensors must be >= 0")
    torch.set_num_threads(args.torch_threads)
    torch.set_num_interop_threads(args.torch_interop_threads)
    model_dir = args.model_dir.expanduser().resolve()
    labels = (
        read_labels(args.labels.expanduser().resolve())
        if args.labels
        else read_source_roster(args.source_roster.expanduser().resolve())
    )
    activation_stats = SAMPLER.load_activation_stats(args.activation_stats.expanduser().resolve())
    codebooks = load_codebooks(args.family_codebooks.expanduser().resolve())
    covariance_dirs = [path.expanduser().resolve() for path in args.covariance_dir]
    covariances, aliases, covariance_summary = load_merged_covariance(covariance_dirs)
    file_map = tensor_file_map(model_dir)
    candidates = []
    for candidate_id in args.candidate:
        candidate = SAMPLER.candidate_from_id(candidate_id)
        if candidate is None:
            raise SystemExit(f"unknown candidate: {candidate_id}")
        candidates.append(candidate)
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    started_at = time.perf_counter()
    completed_tensors = 0
    completed_rows = 0
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for label in labels:
            tensor_started_at = time.perf_counter()
            tensor_name = label["hf_name"]
            family = label["canonical_family"]
            with safe_open(file_map[tensor_name], framework="pt", device="cpu") as source:
                tensor = source.get_tensor(tensor_name)
            if tensor.ndim != 2:
                raise SystemExit(f"eligible C1 tensor is not rank 2: {tensor_name}")
            out_features, in_features = map(int, tensor.shape)
            module = module_name_for_hf_weight(tensor_name)
            covariance_source = aliases.get(module)
            if covariance_source is None or covariance_source not in covariances:
                raise SystemExit(f"C1 covariance alias missing for {tensor_name}: {module}")
            covariance = covariances[covariance_source]
            rows, blocks = select_eval_block_pairs(
                tensor_name,
                out_features,
                in_features,
                args.block_size,
                16,
                args.max_block_pairs,
                args.max_fit_elements,
            )
            offsets = torch.arange(args.block_size, dtype=torch.long)
            columns = blocks[:, None] * args.block_size + offsets[None, :]
            values = tensor[rows[:, None], columns].to(torch.float32).contiguous()
            groups = values.view(-1, 16)
            group_columns = columns.view(-1, 16)
            moments = SAMPLER.activation_stats_for_tensor(
                tensor_name, (out_features, in_features), activation_stats
            )
            eval_weights = moments.index_select(0, group_columns.flatten()).view_as(groups)
            fit_groups, _ = SAMPLER.deterministic_group_partition_with_columns(
                tensor,
                16,
                args.max_fit_elements,
                seed=0,
                tensor_name=tensor_name,
                partition="fit",
            )
            cov_selected = covariance.index_select(0, blocks)

            for candidate in candidates:
                candidate_started_at = time.perf_counter()
                codebook = codebooks.get((family, candidate.candidate_id))
                if codebook is None:
                    raise SystemExit(f"family codebook missing: {family}/{candidate.candidate_id}")
                tensor_scale = SAMPLER.choose_tensor_scale(
                    fit_groups,
                    candidate,
                    SAMPLER.scale_values(candidate.scale_format),
                    codebook,
                )
                metrics = SAMPLER.evaluate_candidate(
                    groups,
                    candidate,
                    args.scale_window,
                    codebook,
                    tensor_scale,
                    group_weights=eval_weights,
                    weighted_scale_search=True,
                )
                state = metrics.pop("_evaluation_state")
                reconstruction = reconstruct_from_state(groups, state).view_as(values).to(torch.float64)
                values64 = values.to(torch.float64)
                error = values64 - reconstruction
                error_q = torch.einsum("pi,pij,pj->p", error, cov_selected, error)
                ref_q = torch.einsum("pi,pij,pj->p", values64, cov_selected, values64)
                sampled_a = float(error_q.sum())
                sampled_ref = float(ref_q.sum())
                population_pairs = out_features * (in_features // args.block_size)
                expansion = float(population_pairs) / float(rows.numel())
                a_full = sampled_a * expansion
                ref_full = sampled_ref * expansion
                row = {
                    "schema_version": "importance-score-c1-result-v0.1",
                    "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                    "run_id": args.run_id,
                    "model_id": label["model_id"],
                    "hf_name": tensor_name,
                    "gguf_name": label.get("gguf_name"),
                    "layer_id": int(label["layer_id"]),
                    "canonical_family": family,
                    "candidate_id": candidate.candidate_id,
                    "C1_A_estimated_full_tensor": a_full,
                    "C1_reference_energy_estimated_full_tensor": ref_full,
                    "C1_L": a_full / max(ref_full, 1e-30),
                    "sampled_block_pairs": int(rows.numel()),
                    "population_block_pairs": population_pairs,
                    "sample_expansion_factor": expansion,
                    "covariance_source_module": covariance_source,
                    "covariance_block_size": args.block_size,
                    "covariance_summary": covariance_summary,
                    "quantizer": {
                        "family_codebook": True,
                        "weighted_scale_search": True,
                        "fit_eval_overlap": 0,
                        "tensor_scale": tensor_scale,
                    },
                    "notes": args.note,
                }
                row["measurement_elapsed_seconds"] = max(
                    time.perf_counter() - candidate_started_at, 0.0
                )
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                handle.flush()
                completed_rows += 1
            completed_tensors += 1
            if args.progress_every_tensors and (
                completed_tensors % args.progress_every_tensors == 0
                or completed_tensors == len(labels)
            ):
                print(
                    json.dumps(
                        {
                            "run_id": args.run_id,
                            "stage": "c1_tensor_scoring",
                            "completed_tensors": completed_tensors,
                            "total_tensors": len(labels),
                            "completed_rows": completed_rows,
                            "tensor_name": tensor_name,
                            "tensor_elapsed_seconds": max(
                                time.perf_counter() - tensor_started_at, 0.0
                            ),
                            "elapsed_seconds": max(time.perf_counter() - started_at, 0.0),
                        },
                        sort_keys=True,
                    ),
                    file=sys.stderr,
                    flush=True,
                )
            del tensor, values, groups, fit_groups
    print(
        json.dumps(
            {
                "rows": completed_rows,
                "tensors": completed_tensors,
                "elapsed_seconds": max(time.perf_counter() - started_at, 0.0),
                "output": str(output),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
