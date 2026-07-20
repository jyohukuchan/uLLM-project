#!/usr/bin/env python3
"""Collect compact activation statistics for aq weighted-error evaluation.

The tool records per-module input second moments for selected Linear modules.
It stores reductions only; raw activations are never retained.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from safetensors.torch import save_file


DEFAULT_PROMPTS = [
    "Explain the difference between prefill and decode in large language model inference.",
    "Write a short technical note about quantizing transformer weight matrices.",
    "Summarize why activation outliers matter for post-training quantization.",
    "List practical constraints for running a language model on a consumer GPU.",
]


@dataclass
class ActivationAccumulator:
    sum_sq: torch.Tensor
    sum_abs: torch.Tensor
    max_abs: torch.Tensor
    count: int = 0

    @classmethod
    def create(cls, features: int) -> "ActivationAccumulator":
        return cls(
            sum_sq=torch.zeros(features, dtype=torch.float64),
            sum_abs=torch.zeros(features, dtype=torch.float64),
            max_abs=torch.zeros(features, dtype=torch.float32),
        )

    def add(self, values: torch.Tensor, valid_token_mask: torch.Tensor | None = None) -> None:
        if values.ndim < 2:
            raise ValueError(f"activation must have at least 2 dimensions, got {tuple(values.shape)}")
        flat = values.detach().reshape(-1, values.shape[-1])
        if valid_token_mask is not None and values.ndim >= 3:
            expected = values.shape[:-1]
            if tuple(valid_token_mask.shape) != tuple(expected):
                raise ValueError(
                    f"attention mask shape {tuple(valid_token_mask.shape)} does not match activation prefix {tuple(expected)}"
                )
            flat = flat[valid_token_mask.detach().reshape(-1).to(dtype=torch.bool, device=flat.device)]
        if flat.numel() == 0:
            return
        if not bool(torch.isfinite(flat).all()):
            raise ValueError("non-finite activation encountered")
        # Keep the reductions in FP64.  The stored moments retain that dtype so
        # downstream quadratic forms need not start from a rounded accumulator.
        flat64 = flat.to(torch.float64)
        self.sum_sq += flat64.square().sum(dim=0).cpu()
        self.sum_abs += flat64.abs().sum(dim=0).cpu()
        self.max_abs = torch.maximum(self.max_abs, flat.abs().amax(dim=0).to(torch.float32).cpu())
        self.count += int(flat.shape[0])

    def second_moment(self) -> torch.Tensor:
        if self.count == 0:
            return torch.zeros_like(self.max_abs)
        return self.sum_sq / float(self.count)

    def mean_abs(self) -> torch.Tensor:
        if self.count == 0:
            return torch.zeros_like(self.max_abs)
        return self.sum_abs / float(self.count)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256_file(path: Path | None) -> str | None:
    if path is None or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def iter_examples(path: Path | None) -> Iterable[dict]:
    if path is None:
        for index, prompt in enumerate(DEFAULT_PROMPTS):
            yield {"record_id": f"default-{index:04d}", "domain": "default", "text": prompt}
        return
    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                example = json.loads(line)
                if not isinstance(example, dict):
                    raise ValueError(f"{path}:{line_number}: JSONL record must be an object")
                if ("text" in example) == ("messages" in example):
                    raise ValueError(
                        f"{path}:{line_number}: record must contain exactly one of text/messages"
                    )
                example.setdefault("record_id", f"line-{line_number:08d}")
                example.setdefault("domain", "unknown")
                yield example
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        prompt = line.strip()
        if prompt:
            yield {"record_id": f"line-{len(prompt)}-{hashlib.sha256(prompt.encode()).hexdigest()[:16]}", "domain": "text", "text": prompt}


def dtype_from_arg(name: str) -> torch.dtype | str:
    if name == "auto":
        return "auto"
    if name == "float32":
        return torch.float32
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    raise ValueError(f"unknown dtype: {name}")


def load_transformers_model(args: argparse.Namespace):
    from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

    model_cls = AutoModel if args.model_class == "auto_model" else AutoModelForCausalLM
    dtype = dtype_from_arg(args.dtype)
    kwargs = {"trust_remote_code": args.trust_remote_code}
    kwargs["dtype"] = dtype

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir,
        trust_remote_code=args.trust_remote_code,
        local_files_only=True,
    )
    try:
        model = model_cls.from_pretrained(
            args.model_dir,
            local_files_only=True,
            **kwargs,
        )
    except TypeError as exc:
        if "dtype" not in str(exc):
            raise
        kwargs["torch_dtype"] = kwargs.pop("dtype")
        model = model_cls.from_pretrained(
            args.model_dir,
            local_files_only=True,
            **kwargs,
        )
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    if args.device != "auto":
        model.to(torch.device(args.device))
    model.eval()
    return tokenizer, model


def register_hooks(
    model: torch.nn.Module,
    module_pattern: re.Pattern[str],
    max_modules: int | None,
    active_attention_mask: dict[str, torch.Tensor | None],
) -> tuple[dict[str, ActivationAccumulator], list[torch.utils.hooks.RemovableHandle]]:
    accumulators: dict[str, ActivationAccumulator] = {}
    handles: list[torch.utils.hooks.RemovableHandle] = []

    def make_hook(name: str, in_features: int):
        def hook(_module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
            if not inputs:
                return
            values = inputs[0]
            if not torch.is_tensor(values) or not values.is_floating_point():
                return
            if int(values.shape[-1]) != in_features:
                return
            accumulators[name].add(values, active_attention_mask["value"])

        return hook

    for name, module in model.named_modules():
        if max_modules is not None and len(accumulators) >= max_modules:
            break
        if not isinstance(module, torch.nn.Linear):
            continue
        if not module_pattern.search(name):
            continue
        accumulators[name] = ActivationAccumulator.create(int(module.in_features))
        handles.append(module.register_forward_pre_hook(make_hook(name, int(module.in_features))))

    if not accumulators:
        raise SystemExit("no modules matched --module-pattern")
    return accumulators, handles


def render_example(
    tokenizer,
    example: dict,
    repeat_to_length: bool,
    sequence_length: int,
) -> tuple[str, str]:
    if "messages" in example:
        messages = example["messages"]
        if not isinstance(messages, list) or not messages:
            raise ValueError(f"record {example['record_id']} has invalid messages")
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        render_kind = "official_chat_template"
    else:
        prompt = str(example["text"])
        render_kind = "plain_text"
    if repeat_to_length and len(prompt) > 0:
        prompt = (prompt + "\n") * max(1, sequence_length // max(1, len(prompt.split())) // 2)
    return prompt, render_kind


def encode_examples(
    tokenizer,
    examples: list[dict],
    sequence_length: int,
    repeat_to_length: bool,
) -> tuple[dict[str, torch.Tensor], list[str]]:
    rendered = [
        render_example(tokenizer, example, repeat_to_length, sequence_length)
        for example in examples
    ]
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("batch padding requires a pad or EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    encoded = tokenizer(
        [item[0] for item in rendered],
        return_tensors="pt",
        truncation=True,
        max_length=sequence_length,
        padding=True,
    )
    return (
        {key: value for key, value in encoded.items() if torch.is_tensor(value)},
        [item[1] for item in rendered],
    )


def tensor_dict_for_output(accumulators: dict[str, ActivationAccumulator]) -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {}
    for name, accumulator in sorted(accumulators.items()):
        tensors[name] = accumulator.second_moment().contiguous()
        tensors[f"{name}.mean_abs"] = accumulator.mean_abs().contiguous()
        tensors[f"{name}.max_abs"] = accumulator.max_abs.contiguous()
    return tensors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prompt-file", type=Path, default=None)
    parser.add_argument("--corpus-manifest", type=Path, default=None)
    parser.add_argument("--corpus-id", default=None)
    parser.add_argument("--shard-id", default=None)
    parser.add_argument("--max-samples", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--progress-every-batches",
        type=int,
        default=0,
        help="Emit a compact JSON progress row to stderr every N batches; 0 disables it.",
    )
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--repeat-to-length", action="store_true")
    parser.add_argument(
        "--module-pattern",
        default=(
            r"(self_attn|linear_attn|mlp).*"
            r"(q_proj|k_proj|v_proj|o_proj|in_proj(_qkv|_qkvz|_ba|_[abz])?|"
            r"out_proj|gate_proj|up_proj|down_proj)$"
        ),
    )
    parser.add_argument("--max-modules", type=int, default=None)
    parser.add_argument("--model-class", choices=("auto_model", "causal_lm"), default="auto_model")
    parser.add_argument("--dtype", choices=("auto", "float32", "bfloat16", "float16"), default="auto")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--require-cpu",
        action="store_true",
        help="Fail unless --device is exactly cpu; use for offline CPU-only measurement.",
    )
    parser.add_argument("--torch-threads", type=int, default=16)
    parser.add_argument("--torch-interop-threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--run-id", default="activation-stats")
    parser.add_argument("--note", action="append", default=[])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_samples < 1 or args.batch_size < 1 or args.progress_every_batches < 0:
        raise SystemExit("--max-samples/--batch-size must be >= 1 and progress interval >= 0")
    if args.sequence_length < 1:
        raise SystemExit("--sequence-length must be >= 1")
    if args.max_modules is not None and args.max_modules < 1:
        raise SystemExit("--max-modules must be >= 1")
    if args.torch_threads < 1 or args.torch_interop_threads < 1:
        raise SystemExit("--torch-threads and --torch-interop-threads must be >= 1")
    if args.require_cpu and args.device != "cpu":
        raise SystemExit("--require-cpu requires --device cpu")

    args.model_dir = args.model_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.prompt_file = args.prompt_file.expanduser().resolve() if args.prompt_file else None
    args.corpus_manifest = args.corpus_manifest.expanduser().resolve() if args.corpus_manifest else None
    args.output_dir.mkdir(parents=True, exist_ok=True)

    torch.set_num_threads(args.torch_threads)
    torch.set_num_interop_threads(args.torch_interop_threads)
    torch.manual_seed(args.seed)

    tokenizer, model = load_transformers_model(args)
    active_attention_mask: dict[str, torch.Tensor | None] = {"value": None}
    accumulators, handles = register_hooks(
        model, re.compile(args.module_pattern), args.max_modules, active_attention_mask
    )
    device = next(model.parameters()).device
    samples_seen = 0
    tokens_seen = 0
    domain_counts: Counter[str] = Counter()
    render_kind_counts: Counter[str] = Counter()
    record_id_digest = hashlib.sha256()
    batches_seen = 0

    try:
        with torch.inference_mode():
            examples_iter = iter(iter_examples(args.prompt_file))
            while samples_seen < args.max_samples:
                examples = []
                for _ in range(min(args.batch_size, args.max_samples - samples_seen)):
                    try:
                        examples.append(next(examples_iter))
                    except StopIteration:
                        break
                if not examples:
                    break
                batch, render_kinds = encode_examples(
                    tokenizer, examples, args.sequence_length, args.repeat_to_length
                )
                batch = {key: value.to(device) for key, value in batch.items()}
                attention_mask = batch.get("attention_mask")
                if attention_mask is not None:
                    tokens_seen += int(attention_mask.sum().item())
                elif "input_ids" in batch:
                    tokens_seen += int(batch["input_ids"].numel())
                active_attention_mask["value"] = attention_mask
                try:
                    model(**batch, use_cache=False)
                except TypeError as exc:
                    if "use_cache" not in str(exc):
                        raise
                    model(**batch)
                samples_seen += len(examples)
                batches_seen += 1
                for example, render_kind in zip(examples, render_kinds, strict=True):
                    domain_counts[str(example.get("domain", "unknown"))] += 1
                    render_kind_counts[render_kind] += 1
                    record_id_digest.update(str(example["record_id"]).encode("utf-8"))
                    record_id_digest.update(b"\n")
                if args.progress_every_batches and batches_seen % args.progress_every_batches == 0:
                    print(
                        json.dumps(
                            {
                                "run_id": args.run_id,
                                "batches_seen": batches_seen,
                                "samples_seen": samples_seen,
                                "tokens_seen": tokens_seen,
                            },
                            sort_keys=True,
                        ),
                        file=sys.stderr,
                        flush=True,
                    )
                active_attention_mask["value"] = None
    finally:
        for handle in handles:
            handle.remove()

    save_file(tensor_dict_for_output(accumulators), str(args.output_dir / "activation_second_moments.safetensors"))
    metadata = {
        "schema_version": "aq-activation-stats-v0.2",
        "run_id": args.run_id,
        "timestamp_utc": utc_now(),
        "model_dir": str(args.model_dir),
        "prompt_file": str(args.prompt_file) if args.prompt_file else None,
        "prompt_file_sha256": sha256_file(args.prompt_file),
        "corpus_manifest": str(args.corpus_manifest) if args.corpus_manifest else None,
        "corpus_manifest_sha256": sha256_file(args.corpus_manifest),
        "corpus_id": args.corpus_id,
        "shard_id": args.shard_id,
        "max_samples": args.max_samples,
        "batch_size": args.batch_size,
        "batches_seen": batches_seen,
        "progress_every_batches": args.progress_every_batches,
        "samples_seen": samples_seen,
        "tokens_seen": tokens_seen,
        "domain_counts": dict(sorted(domain_counts.items())),
        "render_kind_counts": dict(sorted(render_kind_counts.items())),
        "processed_record_ids_sha256": record_id_digest.hexdigest(),
        "sequence_length": args.sequence_length,
        "repeat_to_length": args.repeat_to_length,
        "module_pattern": args.module_pattern,
        "module_count": len(accumulators),
        "model_class": args.model_class,
        "dtype": args.dtype,
        "device": str(device),
        "require_cpu": args.require_cpu,
        "torch_threads": args.torch_threads,
        "torch_interop_threads": args.torch_interop_threads,
        "seed": args.seed,
        "git_revision": git_revision(),
        "model_config_sha256": sha256_file(args.model_dir / "config.json"),
        "model_weight_index_sha256": sha256_file(args.model_dir / "model.safetensors.index.json"),
        "tokenizer_config_sha256": sha256_file(args.model_dir / "tokenizer_config.json"),
        "chat_template_sha256": sha256_file(args.model_dir / "chat_template.jinja"),
        "padding_mask_policy": "attention_mask filters [batch, sequence, hidden] pre-hook activations; this run uses one unpadded example per forward.",
        "chat_template_policy": "records with messages use tokenizer.apply_chat_template(add_generation_prompt=False); text records are tokenized verbatim",
        "reduction_dtype": "float64",
        "stored_second_moment_dtype": "float64",
        "notes": args.note,
        "modules": {
            name: {
                "input_features": int(accumulator.max_abs.numel()),
                "activation_count": accumulator.count,
            }
            for name, accumulator in sorted(accumulators.items())
        },
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
