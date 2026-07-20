#!/usr/bin/env python3
"""Collect CPU-only block-diagonal input covariance for C1.

The primary block width is fixed at 128 channels by the frozen plan.  Products
are evaluated in FP32 and accumulated/stored in FP64.  Modules that provably
share the same block input (Q/K/V, gate/up, and Qwen linear-attention input
projections) share one accumulator; the complete alias map is written to
metadata rather than being inferred downstream.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors.torch import save_file


def load_activation_collector():
    path = Path(__file__).resolve().parent / "collect-activation-stats.py"
    spec = importlib.util.spec_from_file_location("aq_activation_collector", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


COLLECTOR = load_activation_collector()


@dataclass
class BlockCovarianceAccumulator:
    sum_xx: torch.Tensor
    features: int
    block_size: int
    count: int = 0

    @classmethod
    def create(cls, features: int, block_size: int) -> "BlockCovarianceAccumulator":
        blocks = (features + block_size - 1) // block_size
        return cls(
            sum_xx=torch.zeros((blocks, block_size, block_size), dtype=torch.float64),
            features=features,
            block_size=block_size,
        )

    def add(self, values: torch.Tensor, valid_token_mask: torch.Tensor | None) -> None:
        flat = values.detach().reshape(-1, values.shape[-1])
        if valid_token_mask is not None and values.ndim >= 3:
            if tuple(valid_token_mask.shape) != tuple(values.shape[:-1]):
                raise ValueError("attention mask does not match activation prefix")
            flat = flat[valid_token_mask.detach().reshape(-1).to(dtype=torch.bool, device=flat.device)]
        if flat.numel() == 0:
            return
        if int(flat.shape[1]) != self.features or not bool(torch.isfinite(flat).all()):
            raise ValueError("invalid activation encountered during block covariance collection")
        flat32 = flat.to(torch.float32)
        if self.features % self.block_size == 0:
            blocked = flat32.reshape(flat32.shape[0], -1, self.block_size)
            gram = torch.einsum("nbi,nbj->bij", blocked, blocked)
            self.sum_xx += gram.to(torch.float64).cpu()
        else:
            for block_index, start in enumerate(range(0, self.features, self.block_size)):
                width = min(self.block_size, self.features - start)
                block = flat32[:, start : start + width]
                self.sum_xx[block_index, :width, :width] += (block.T @ block).to(torch.float64).cpu()
        self.count += int(flat.shape[0])

    def covariance(self) -> torch.Tensor:
        if self.count <= 0:
            raise ValueError("zero-count covariance accumulator")
        return (self.sum_xx / float(self.count)).contiguous()


def covariance_source_name(name: str) -> str:
    name = re.sub(r"\.self_attn\.(?:k_proj|v_proj)$", ".self_attn.q_proj", name)
    name = re.sub(r"\.mlp\.up_proj$", ".mlp.gate_proj", name)
    name = re.sub(
        r"\.linear_attn\.in_proj_(?:a|b|z)$",
        ".linear_attn.in_proj_qkv",
        name,
    )
    return name


def sha256_file(path: Path | None) -> str | None:
    if path is None or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--corpus-manifest", type=Path, required=True)
    parser.add_argument("--model-class", choices=("auto_model", "causal_lm"), default="auto_model")
    parser.add_argument("--dtype", choices=("auto", "float32", "bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--require-cpu", action="store_true")
    parser.add_argument("--max-samples", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--torch-threads", type=int, default=16)
    parser.add_argument("--torch-interop-threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shard-id", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--module-pattern",
        default=(
            r"(self_attn|linear_attn|mlp).*"
            r"(q_proj|k_proj|v_proj|o_proj|in_proj_(qkv|a|b|z)|out_proj|gate_proj|up_proj|down_proj)$"
        ),
    )
    parser.add_argument("--note", action="append", default=[])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.require_cpu and args.device != "cpu":
        raise SystemExit("--require-cpu requires --device cpu")
    if min(args.max_samples, args.batch_size, args.sequence_length, args.block_size) < 1:
        raise SystemExit("sample, batch, sequence, and block sizes must be positive")
    if args.block_size != 128:
        raise SystemExit("the formal C1 primary requires --block-size 128")
    args.model_dir = args.model_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.prompt_file = args.prompt_file.expanduser().resolve()
    args.corpus_manifest = args.corpus_manifest.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    torch.set_num_threads(args.torch_threads)
    torch.set_num_interop_threads(args.torch_interop_threads)
    torch.manual_seed(args.seed)
    tokenizer, model = COLLECTOR.load_transformers_model(args)
    pattern = re.compile(args.module_pattern)
    active_mask: dict[str, torch.Tensor | None] = {"value": None}
    aliases: dict[str, str] = {}
    source_modules: dict[str, torch.nn.Linear] = {}
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear) or not pattern.search(name):
            continue
        source = covariance_source_name(name)
        aliases[name] = source
        if name == source:
            source_modules[source] = module
    missing_sources = sorted(set(aliases.values()) - set(source_modules))
    if missing_sources:
        raise SystemExit(f"covariance source modules are missing: {missing_sources}")

    accumulators = {
        name: BlockCovarianceAccumulator.create(int(module.in_features), args.block_size)
        for name, module in source_modules.items()
    }
    handles = []
    for name, module in source_modules.items():
        accumulator = accumulators[name]

        def hook(_module, inputs, *, _accumulator=accumulator):
            if inputs and torch.is_tensor(inputs[0]) and inputs[0].is_floating_point():
                _accumulator.add(inputs[0], active_mask["value"])

        handles.append(module.register_forward_pre_hook(hook))

    device = next(model.parameters()).device
    samples_seen = 0
    tokens_seen = 0
    domains: Counter[str] = Counter()
    record_digest = hashlib.sha256()
    examples_iter = iter(COLLECTOR.iter_examples(args.prompt_file))
    try:
        with torch.inference_mode():
            while samples_seen < args.max_samples:
                examples = []
                for _ in range(min(args.batch_size, args.max_samples - samples_seen)):
                    try:
                        examples.append(next(examples_iter))
                    except StopIteration:
                        break
                if not examples:
                    break
                batch, _ = COLLECTOR.encode_examples(
                    tokenizer, examples, args.sequence_length, False
                )
                batch = {key: value.to(device) for key, value in batch.items()}
                attention_mask = batch.get("attention_mask")
                tokens_seen += int(
                    attention_mask.sum().item()
                    if attention_mask is not None
                    else batch["input_ids"].numel()
                )
                active_mask["value"] = attention_mask
                try:
                    model(**batch, use_cache=False)
                except TypeError as exc:
                    if "use_cache" not in str(exc):
                        raise
                    model(**batch)
                active_mask["value"] = None
                samples_seen += len(examples)
                for example in examples:
                    domains[str(example.get("domain", "unknown"))] += 1
                    record_digest.update(str(example["record_id"]).encode("utf-8") + b"\n")
    finally:
        for handle in handles:
            handle.remove()

    output_path = args.output_dir / "block_covariance_128.safetensors"
    save_file(
        {name: accumulator.covariance() for name, accumulator in sorted(accumulators.items())},
        str(output_path),
    )
    metadata = {
        "schema_version": "importance-score-c1-block-covariance-v0.1",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "run_id": args.run_id,
        "model_dir": str(args.model_dir),
        "prompt_file": str(args.prompt_file),
        "prompt_file_sha256": sha256_file(args.prompt_file),
        "corpus_manifest": str(args.corpus_manifest),
        "corpus_manifest_sha256": sha256_file(args.corpus_manifest),
        "shard_id": args.shard_id,
        "samples_seen": samples_seen,
        "tokens_seen": tokens_seen,
        "domain_counts": dict(sorted(domains.items())),
        "processed_record_ids_sha256": record_digest.hexdigest(),
        "sequence_length": args.sequence_length,
        "batch_size": args.batch_size,
        "block_size": args.block_size,
        "device": str(device),
        "require_cpu": args.require_cpu,
        "reference_dtype": args.dtype,
        "product_dtype": "float32",
        "accumulation_and_storage_dtype": "float64",
        "source_module_count": len(accumulators),
        "covered_linear_module_count": len(aliases),
        "module_aliases": dict(sorted(aliases.items())),
        "modules": {
            name: {
                "features": accumulator.features,
                "block_count": int(accumulator.sum_xx.shape[0]),
                "valid_width_last_block": (
                    accumulator.features - (accumulator.sum_xx.shape[0] - 1) * accumulator.block_size
                ),
                "activation_count": accumulator.count,
            }
            for name, accumulator in sorted(accumulators.items())
        },
        "output_sha256": sha256_file(output_path),
        "notes": args.note,
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"samples_seen": samples_seen, "tokens_seen": tokens_seen, "modules": len(accumulators)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
