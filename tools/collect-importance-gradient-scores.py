#!/usr/bin/env python3
"""Collect the frozen C5 Taylor and Fisher importance-score candidates.

This runner is intentionally narrow.  It evaluates exactly the frozen AQ4
baseline and AQ5 high candidate, exactly four D_fisher shards, and only the
label-blind active source roster.  Gradient tensors never survive a parameter's
post-accumulate hook: the hook performs chunked scalar reductions and clears
``parameter.grad`` before returning.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import logging
import math
import os
import statistics
import subprocess
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import torch
import torch.nn.functional as F


LOW_CANDIDATE_ID = "aq4_e4m3_g16_ts_flloyd16"
HIGH_CANDIDATE_ID = "aq5_e4m3_g16_ts_flloyd32"
LOW = LOW_CANDIDATE_ID
HIGH = HIGH_CANDIDATE_ID
CANDIDATE_IDS = (LOW_CANDIDATE_ID, HIGH_CANDIDATE_ID)
SHARD_COUNT = 4
SCHEMA_VERSION = "importance-score-gradient-c5-v0.1"
RECEIPT_SCHEMA_VERSION = "importance-score-gradient-c5-receipt-v0.1"
CHECKPOINT_SCHEMA_VERSION = "importance-score-gradient-c5-checkpoint-v0.1"
DEFAULT_HOST_CACHE_LIMIT_BYTES = 64 * 1024**3
DEFAULT_DISK_CACHE_LIMIT_BYTES = 64 * 1024**3

EMPIRICAL_METHOD_IDS = (
    "C5a_Taylor_deletion_L1",
    "C5a_Taylor_deletion_squared",
    f"C5a_Taylor_quant_{LOW_CANDIDATE_ID}",
    f"C5a_Taylor_quant_{HIGH_CANDIDATE_ID}",
    f"C5b_Empirical_Fisher_{LOW_CANDIDATE_ID}",
    f"C5b_Empirical_Fisher_{HIGH_CANDIDATE_ID}",
)
SELF_FISHER_METHOD_IDS = (
    f"C5b_Self_Fisher_{LOW_CANDIDATE_ID}",
    f"C5b_Self_Fisher_{HIGH_CANDIDATE_ID}",
)

EMPIRICAL_SCORE_COLUMNS = (
    "C5a_Taylor_quant_I",
    "C5a_Taylor_quant_A_low",
    "C5a_Taylor_quant_A_high",
    "C5a_Taylor_quant_raw_gain",
    "C5a_Taylor_quant_G",
    "C5a_Taylor_L1_S",
    "C5a_Taylor_squared_S",
    "C5b_Empirical_Fisher_I",
    "C5b_Empirical_Fisher_A_low",
    "C5b_Empirical_Fisher_A_high",
    "C5b_Empirical_Fisher_raw_gain",
    "C5b_Empirical_Fisher_G",
)
SELF_FISHER_SCORE_COLUMNS = (
    "C5b_Self_Fisher_I",
    "C5b_Self_Fisher_A_low",
    "C5b_Self_Fisher_A_high",
    "C5b_Self_Fisher_raw_gain",
    "C5b_Self_Fisher_G",
)

FORMULAS = {
    "C5a_Taylor_deletion_L1": (
        "A(t)=mean_s sum_i |g[s,i]*w[i]|; L(t)=A(t)/n_params"
    ),
    "C5a_Taylor_deletion_squared": (
        "A(t)=mean_s sum_i (g[s,i]*w[i])^2; L(t)=A(t)/n_params"
    ),
    "C5a_Taylor_quant": (
        "A(t,b)=mean_s |sum_i g[s,i]*(Q_b(w)[i]-w[i])|; "
        "L(t,b)=A(t,b)/n_params"
    ),
    "C5b_Fisher": (
        "A(t,b)=0.5*sum_i mean_observation(g[observation,i]^2)*"
        "(Q_b(w)[i]-w[i])^2; L(t,b)=A(t,b)/n_params"
    ),
    "empirical_gradient": (
        "g_s=grad of the within-record valid-next-token mean causal-LM NLL"
    ),
    "self_Fisher_gradient": (
        "g_(s,m)=grad of the within-record mean log p_theta(y|x), with every "
        "valid next token independently sampled from the full vocabulary"
    ),
}


def load_tool(filename: str, module_name: str):
    path = Path(__file__).resolve().parent / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# Reuse, rather than duplicate, the already measured name resolver and exact
# fake-quantization contract.
PERTURB = load_tool(
    "run-importance-single-tensor-perturbation.py",
    "importance_gradient_perturbation_contract",
)
COLLECTOR = PERTURB.COLLECTOR
SAMPLER = PERTURB.SAMPLER


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tensor(tensor: torch.Tensor, chunk_elements: int = 1 << 20) -> str:
    """Hash raw contiguous tensor bytes without materializing a whole byte copy."""

    if tensor.device.type != "cpu":
        raise ValueError("sha256_tensor requires a CPU tensor")
    flat = tensor.detach().contiguous().view(-1)
    digest = hashlib.sha256()
    for start in range(0, flat.numel(), chunk_elements):
        raw = flat[start : start + chunk_elements].contiguous().view(torch.uint8)
        digest.update(raw.numpy().tobytes())
    return digest.hexdigest()


def git_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def emit_progress(payload: Mapping[str, Any]) -> None:
    print(
        json.dumps(dict(payload), ensure_ascii=False, sort_keys=True),
        file=sys.stderr,
        flush=True,
    )


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def elapsed(device: torch.device, started_at: float) -> float:
    synchronize(device)
    return max(0.0, time.perf_counter() - started_at)


def write_new_atomic(path: Path, data: bytes) -> None:
    """Publish bytes atomically while refusing to replace an existing artifact."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def write_replace_atomic(path: Path, value: Mapping[str, Any]) -> None:
    """Atomically replace a resumable checkpoint (never a final result)."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    payload = (json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def receipt_path_for(output: Path) -> Path:
    if output.suffix == ".jsonl":
        return output.with_suffix(".receipt.json")
    return output.with_name(f"{output.name}.receipt.json")


def checkpoint_path_for(output: Path) -> Path:
    if output.suffix == ".jsonl":
        return output.with_suffix(".checkpoint.json")
    return output.with_name(f"{output.name}.checkpoint.json")


def require_fresh_final_outputs(output: Path, receipt: Path) -> None:
    existing = [str(path) for path in (output, receipt) if path.exists()]
    if existing:
        raise SystemExit(f"refusing to overwrite sealed C5 outputs: {existing}")


def enforce_r9700_visibility() -> dict[str, Any]:
    """Fail closed unless only physical HIP device 1 is visible as cuda:0."""

    if os.environ.get("HIP_VISIBLE_DEVICES") != "1":
        raise SystemExit("C5 requires the exact environment HIP_VISIBLE_DEVICES=1")
    for name in ("ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
        value = os.environ.get(name)
        if value not in (None, "", "1"):
            raise SystemExit(f"{name} conflicts with the required physical device 1: {value!r}")
    if torch.version.hip is None:
        raise SystemExit("C5 requires a ROCm PyTorch build")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise SystemExit("C5 requires exactly one visible ROCm device")
    properties = torch.cuda.get_device_properties(0)
    architecture = str(getattr(properties, "gcnArchName", "")).split(":", 1)[0]
    if architecture != "gfx1201":
        raise SystemExit(
            f"C5 requires logical cuda:0 to be the R9700/gfx1201, observed {architecture!r}"
        )
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    return {
        "logical_device": "cuda:0",
        "physical_visibility": "HIP_VISIBLE_DEVICES=1",
        "device_count": torch.cuda.device_count(),
        "name": properties.name,
        "architecture": architecture,
        "total_memory_bytes": int(properties.total_memory),
        "torch_version": torch.__version__,
        "rocm_version": torch.version.hip,
    }


class _FallbackLogHandler(logging.Handler):
    def __init__(self, monitor: "FallbackMonitor") -> None:
        super().__init__()
        self.monitor = monitor

    def emit(self, record: logging.LogRecord) -> None:
        self.monitor.inspect(record.getMessage(), source=f"logging:{record.name}")


class FallbackMonitor:
    """Allow only the already-audited Qwen linear-attention fallback warning."""

    ALLOWED_FRAGMENTS = (
        "fast path is not available",
        "falling back to torch implementation",
    )

    def __init__(self) -> None:
        self.allowed_messages: list[dict[str, str]] = []
        self._handler = _FallbackLogHandler(self)
        self._old_showwarning: Callable[..., Any] | None = None

    def inspect(self, message: str, *, source: str) -> None:
        lowered = message.lower()
        if "fallback" not in lowered and "falling back" not in lowered:
            return
        if any(fragment in lowered for fragment in self.ALLOWED_FRAGMENTS):
            self.allowed_messages.append({"source": source, "message": message})
            emit_progress(
                {
                    "event": "allowed_known_fallback",
                    "source": source,
                    "message": message,
                }
            )
            return
        raise RuntimeError(f"unknown fallback detected from {source}: {message}")

    def __enter__(self) -> "FallbackMonitor":
        logging.getLogger().addHandler(self._handler)
        self._old_showwarning = warnings.showwarning

        def guarded_showwarning(
            message: Warning | str,
            category: type[Warning],
            filename: str,
            lineno: int,
            file=None,
            line: str | None = None,
        ) -> None:
            self.inspect(str(message), source=f"warning:{category.__name__}")
            assert self._old_showwarning is not None
            self._old_showwarning(message, category, filename, lineno, file=file, line=line)

        warnings.showwarning = guarded_showwarning
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        logging.getLogger().removeHandler(self._handler)
        if self._old_showwarning is not None:
            warnings.showwarning = self._old_showwarning


@dataclass
class ScalarMoments:
    """Ordered, deterministic online scalar moments suitable for checkpoints."""

    count: int = 0
    mean: float = 0.0
    m2: float = 0.0
    minimum: float = math.inf
    maximum: float = -math.inf

    def add(self, raw_value: float) -> None:
        value = float(raw_value)
        if not math.isfinite(value):
            raise FloatingPointError(f"non-finite score observation: {value}")
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (value - self.mean)
        self.minimum = min(self.minimum, value)
        self.maximum = max(self.maximum, value)

    @property
    def sample_variance(self) -> float | None:
        if self.count < 2:
            return None
        return max(self.m2 / (self.count - 1), 0.0)

    def describe(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "mean": self.mean if self.count else None,
            "sample_variance": self.sample_variance,
            "minimum": self.minimum if self.count else None,
            "maximum": self.maximum if self.count else None,
        }

    def to_json(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "mean": self.mean,
            "m2": self.m2,
            "minimum": self.minimum if self.count else None,
            "maximum": self.maximum if self.count else None,
        }

    @classmethod
    def from_json(cls, value: Mapping[str, Any]) -> "ScalarMoments":
        count = int(value.get("count", 0))
        return cls(
            count=count,
            mean=float(value.get("mean", 0.0)),
            m2=float(value.get("m2", 0.0)),
            minimum=float(value["minimum"]) if count else math.inf,
            maximum=float(value["maximum"]) if count else -math.inf,
        )


@dataclass
class TensorAggregate:
    overall: dict[str, ScalarMoments] = field(default_factory=dict)
    shards: list[dict[str, ScalarMoments]] = field(
        default_factory=lambda: [{} for _ in range(SHARD_COUNT)]
    )
    mc_records: dict[str, ScalarMoments] = field(default_factory=dict)
    hook_calls: int = 0
    hook_elapsed_seconds: float = 0.0
    hook_max_seconds: float = 0.0

    def add(self, shard_index: int, values: Mapping[str, float], hook_seconds: float) -> None:
        if not 0 <= shard_index < SHARD_COUNT:
            raise ValueError(f"invalid shard index: {shard_index}")
        for key in sorted(values):
            self.overall.setdefault(key, ScalarMoments()).add(float(values[key]))
            self.shards[shard_index].setdefault(key, ScalarMoments()).add(float(values[key]))
        self.hook_calls += 1
        self.hook_elapsed_seconds += float(hook_seconds)
        self.hook_max_seconds = max(self.hook_max_seconds, float(hook_seconds))

    def add_mc_record(self, key: str, value: float) -> None:
        self.mc_records.setdefault(key, ScalarMoments()).add(value)

    def to_json(self) -> dict[str, Any]:
        return {
            "overall": {key: value.to_json() for key, value in sorted(self.overall.items())},
            "shards": [
                {key: value.to_json() for key, value in sorted(shard.items())}
                for shard in self.shards
            ],
            "mc_records": {
                key: value.to_json() for key, value in sorted(self.mc_records.items())
            },
            "hook_calls": self.hook_calls,
            "hook_elapsed_seconds": self.hook_elapsed_seconds,
            "hook_max_seconds": self.hook_max_seconds,
        }

    @classmethod
    def from_json(cls, value: Mapping[str, Any]) -> "TensorAggregate":
        shards = [
            {key: ScalarMoments.from_json(moment) for key, moment in shard.items()}
            for shard in value.get("shards", [])
        ]
        if len(shards) != SHARD_COUNT:
            raise ValueError("checkpoint aggregate does not contain exactly four shards")
        return cls(
            overall={
                key: ScalarMoments.from_json(moment)
                for key, moment in value.get("overall", {}).items()
            },
            shards=shards,
            mc_records={
                key: ScalarMoments.from_json(moment)
                for key, moment in value.get("mc_records", {}).items()
            },
            hook_calls=int(value.get("hook_calls", 0)),
            hook_elapsed_seconds=float(value.get("hook_elapsed_seconds", 0.0)),
            hook_max_seconds=float(value.get("hook_max_seconds", 0.0)),
        )


def new_aggregates(tensor_names: Iterable[str]) -> dict[str, TensorAggregate]:
    return {name: TensorAggregate() for name in tensor_names}


def aggregates_to_json(aggregates: Mapping[str, TensorAggregate]) -> dict[str, Any]:
    return {name: aggregates[name].to_json() for name in sorted(aggregates)}


def aggregates_from_json(value: Mapping[str, Any]) -> dict[str, TensorAggregate]:
    return {name: TensorAggregate.from_json(item) for name, item in sorted(value.items())}


@dataclass
class TensorQuantCache:
    tensor_name: str
    source_bf16_cpu: torch.Tensor
    quantized_bf16_cpu: dict[str, torch.Tensor]
    quantization: dict[str, dict[str, Any]] = field(default_factory=dict)
    cache_paths: dict[str, str] = field(default_factory=dict)
    content_sha256: dict[str, str] = field(default_factory=dict)
    preparation_timing: dict[str, float] = field(default_factory=dict)

    @property
    def n_params(self) -> int:
        return self.source_bf16_cpu.numel()

    @property
    def host_bytes(self) -> int:
        tensors = [self.source_bf16_cpu, *self.quantized_bf16_cpu.values()]
        return sum(tensor.numel() * tensor.element_size() for tensor in tensors)


def _scalar_stack_to_floats(names: list[str], values: list[torch.Tensor]) -> dict[str, float]:
    if not values:
        return {}
    packed = torch.stack(values).detach().to("cpu", dtype=torch.float64)
    return {name: float(value) for name, value in zip(names, packed.tolist(), strict=True)}


def reduce_gradient_scores(
    gradient: torch.Tensor,
    source_cpu: torch.Tensor,
    quantized_cpu: Mapping[str, torch.Tensor],
    *,
    mode: str,
    chunk_elements: int,
) -> dict[str, float]:
    """Reduce one full parameter gradient without retaining an elementwise score.

    Source and quantized weights are transferred from their BF16 CPU caches one
    chunk at a time.  Each delta is formed *after* both operands are converted
    to FP32.  Taylor-quant takes ``abs`` only after every signed chunk dot has
    been combined into the full-tensor dot.
    """

    if mode not in ("empirical", "self_fisher"):
        raise ValueError(f"unknown gradient mode: {mode}")
    if chunk_elements < 1:
        raise ValueError("chunk_elements must be positive")
    if source_cpu.device.type != "cpu" or any(
        tensor.device.type != "cpu" for tensor in quantized_cpu.values()
    ):
        raise ValueError("source and quantized caches must be CPU tensors")
    if tuple(gradient.shape) != tuple(source_cpu.shape):
        raise ValueError("gradient/source shape mismatch")
    if set(quantized_cpu) != set(CANDIDATE_IDS):
        raise ValueError(f"quantized cache must contain exactly {CANDIDATE_IDS}")
    if any(tuple(tensor.shape) != tuple(source_cpu.shape) for tensor in quantized_cpu.values()):
        raise ValueError("quantized/source shape mismatch")

    flat_gradient = gradient.detach().reshape(-1)
    flat_source = source_cpu.reshape(-1)
    flat_quantized = {key: value.reshape(-1) for key, value in quantized_cpu.items()}
    device = gradient.device
    accumulator_names: list[str] = ["gradient_trace"]
    if mode == "empirical":
        accumulator_names.extend(("taylor_delete_l1", "taylor_delete_squared"))
    for candidate_id in CANDIDATE_IDS:
        if mode == "empirical":
            accumulator_names.append(f"taylor_quant_signed_dot::{candidate_id}")
        accumulator_names.append(f"fisher_unhalved::{candidate_id}")
    accumulators = {
        name: torch.zeros((), device=device, dtype=torch.float64)
        for name in accumulator_names
    }

    for start in range(0, flat_gradient.numel(), chunk_elements):
        end = min(start + chunk_elements, flat_gradient.numel())
        gradient32 = flat_gradient[start:end].to(dtype=torch.float32)
        source32 = flat_source[start:end].to(device=device, dtype=torch.float32)
        gradient_squared = gradient32.square()
        accumulators["gradient_trace"] += gradient_squared.sum(dtype=torch.float64)
        if mode == "empirical":
            deletion_product = gradient32 * source32
            accumulators["taylor_delete_l1"] += deletion_product.abs().sum(
                dtype=torch.float64
            )
            accumulators["taylor_delete_squared"] += deletion_product.square().sum(
                dtype=torch.float64
            )
        for candidate_id in CANDIDATE_IDS:
            quantized32 = flat_quantized[candidate_id][start:end].to(
                device=device, dtype=torch.float32
            )
            # This subtraction is deliberately FP32.  Never cache or cast delta
            # to BF16: doing so would change both Taylor and Fisher scores.
            delta32 = quantized32 - source32
            if mode == "empirical":
                accumulators[f"taylor_quant_signed_dot::{candidate_id}"] += (
                    gradient32 * delta32
                ).sum(dtype=torch.float64)
            accumulators[f"fisher_unhalved::{candidate_id}"] += (
                gradient_squared * delta32.square()
            ).sum(dtype=torch.float64)

    output_names: list[str] = ["gradient_trace"]
    output_values: list[torch.Tensor] = [accumulators["gradient_trace"]]
    if mode == "empirical":
        output_names.extend(("taylor_delete_l1", "taylor_delete_squared"))
        output_values.extend(
            (accumulators["taylor_delete_l1"], accumulators["taylor_delete_squared"])
        )
    for candidate_id in CANDIDATE_IDS:
        if mode == "empirical":
            output_names.append(f"taylor_quant_abs_dot::{candidate_id}")
            output_values.append(
                accumulators[f"taylor_quant_signed_dot::{candidate_id}"].abs()
            )
        output_names.append(f"fisher_half::{candidate_id}")
        output_values.append(0.5 * accumulators[f"fisher_unhalved::{candidate_id}"])
    return _scalar_stack_to_floats(output_names, output_values)


class PostAccumulateGradientSession:
    """Own post-accumulate hooks for one gradient observation at a time."""

    def __init__(
        self,
        parameters: Mapping[str, torch.nn.Parameter],
        caches: Mapping[str, TensorQuantCache],
        *,
        mode: str,
        chunk_elements: int,
        progress: Callable[[Mapping[str, Any]], None] | None = None,
    ) -> None:
        if set(parameters) != set(caches):
            raise ValueError("parameter/cache roster mismatch")
        if not hasattr(torch.nn.Parameter(torch.ones(1)), "register_post_accumulate_grad_hook"):
            raise RuntimeError("PyTorch post-accumulate parameter hooks are required")
        self.parameters = dict(parameters)
        self.caches = dict(caches)
        self.mode = mode
        self.chunk_elements = chunk_elements
        self.progress = progress
        self.context: dict[str, Any] | None = None
        self.results: dict[str, dict[str, float]] = {}
        self.hook_seconds: dict[str, float] = {}
        self.handles = [
            parameter.register_post_accumulate_grad_hook(self._make_hook(name))
            for name, parameter in self.parameters.items()
        ]

    def _make_hook(self, tensor_name: str):
        def hook(parameter: torch.nn.Parameter) -> None:
            started_at = time.perf_counter()
            if self.context is None:
                parameter.grad = None
                raise RuntimeError("gradient hook fired without an active observation")
            if tensor_name in self.results:
                parameter.grad = None
                raise RuntimeError(f"gradient hook fired twice for {tensor_name}")
            gradient = parameter.grad
            if gradient is None:
                raise RuntimeError(f"post-accumulate hook has no gradient for {tensor_name}")
            try:
                values = reduce_gradient_scores(
                    gradient,
                    self.caches[tensor_name].source_bf16_cpu,
                    self.caches[tensor_name].quantized_bf16_cpu,
                    mode=self.mode,
                    chunk_elements=self.chunk_elements,
                )
            finally:
                # This is the principal memory invariant of the runner.
                parameter.grad = None
            hook_elapsed = time.perf_counter() - started_at
            self.results[tensor_name] = values
            self.hook_seconds[tensor_name] = hook_elapsed
            if self.progress is not None:
                self.progress(
                    {
                        "event": "tensor_gradient_reduced",
                        "mode": self.mode,
                        "tensor_name": tensor_name,
                        "hook_elapsed_seconds": hook_elapsed,
                        **self.context,
                    }
                )

        return hook

    def begin(self, context: Mapping[str, Any]) -> None:
        if self.context is not None:
            raise RuntimeError("a gradient observation is already active")
        stale = [name for name, parameter in self.parameters.items() if parameter.grad is not None]
        if stale:
            for parameter in self.parameters.values():
                parameter.grad = None
            raise RuntimeError(f"stale gradients before observation: {stale[:5]}")
        self.context = dict(context)
        self.results = {}
        self.hook_seconds = {}

    def finish(self) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
        if self.context is None:
            raise RuntimeError("no active gradient observation")
        missing = sorted(set(self.parameters) - set(self.results))
        extra = sorted(set(self.results) - set(self.parameters))
        uncleared = [name for name, parameter in self.parameters.items() if parameter.grad is not None]
        self.context = None
        if missing or extra or uncleared:
            for parameter in self.parameters.values():
                parameter.grad = None
            raise RuntimeError(
                "gradient hook coverage failure: "
                f"missing={missing[:8]}, extra={extra[:8]}, uncleared={uncleared[:8]}"
            )
        return dict(self.results), dict(self.hook_seconds)

    def abort(self) -> None:
        for parameter in self.parameters.values():
            parameter.grad = None
        self.context = None
        self.results = {}
        self.hook_seconds = {}

    def close(self) -> None:
        self.abort()
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def __enter__(self) -> "PostAccumulateGradientSession":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


def stable_self_fisher_seed(base_seed: int, record_id: str, draw_index: int) -> int:
    material = (
        f"importance-score-C5-self-Fisher-v0.1\0{base_seed}\0{record_id}\0{draw_index}"
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") & ((1 << 63) - 1)


def valid_next_token_view(
    logits: torch.Tensor, tensors: Mapping[str, torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Return logits/real targets at positions with a valid source and next token."""

    input_ids = tensors.get("input_ids")
    if input_ids is None:
        raise ValueError("causal-LM batch does not contain input_ids")
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("C5 requires batch=1 input IDs")
    if logits.ndim != 3 or logits.shape[:2] != input_ids.shape:
        raise ValueError("causal-LM logits do not align with input IDs")
    attention_mask = tensors.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    valid = attention_mask[:, :-1].bool() & attention_mask[:, 1:].bool()
    valid_count = int(valid.sum().item())
    if valid_count < 1:
        raise ValueError("record has no valid next-token prediction")
    return logits[:, :-1, :][valid], input_ids[:, 1:][valid], valid_count


def empirical_causal_lm_loss(
    logits: torch.Tensor, tensors: Mapping[str, torch.Tensor]
) -> tuple[torch.Tensor, int]:
    valid_logits, targets, valid_count = valid_next_token_view(logits, tensors)
    loss = F.cross_entropy(valid_logits.float(), targets, reduction="mean")
    return loss, valid_count


def sample_self_fisher_targets(
    valid_logits: torch.Tensor,
    *,
    seed: int,
) -> torch.Tensor:
    generator = torch.Generator(device=valid_logits.device)
    generator.manual_seed(seed)
    # Sampling is detached, but it is exact categorical sampling over every
    # vocabulary entry; no top-k/top-p approximation is permitted.
    probabilities = torch.softmax(valid_logits.detach().float(), dim=-1)
    targets = torch.multinomial(probabilities, 1, generator=generator).squeeze(1)
    del probabilities
    return targets


def self_fisher_log_probability(
    valid_logits: torch.Tensor,
    sampled_targets: torch.Tensor,
) -> torch.Tensor:
    if sampled_targets.shape != valid_logits.shape[:-1]:
        raise ValueError("sampled targets do not align with valid logits")
    log_probabilities = torch.log_softmax(valid_logits.float(), dim=-1)
    return log_probabilities.gather(1, sampled_targets[:, None]).squeeze(1).mean()


def _mean(moment_map: Mapping[str, ScalarMoments], key: str) -> float:
    moment = moment_map.get(key)
    if moment is None or moment.count < 1:
        raise ValueError(f"missing score observations for {key}")
    return float(moment.mean)


def flat_scores_from_moments(
    mode: str,
    moment_map: Mapping[str, ScalarMoments],
    n_params: int,
) -> dict[str, float]:
    if n_params < 1:
        raise ValueError("n_params must be positive")
    if mode == "empirical":
        taylor_low = _mean(moment_map, f"taylor_quant_abs_dot::{LOW_CANDIDATE_ID}")
        taylor_high = _mean(moment_map, f"taylor_quant_abs_dot::{HIGH_CANDIDATE_ID}")
        taylor_gain = taylor_low - taylor_high
        fisher_low = _mean(moment_map, f"fisher_half::{LOW_CANDIDATE_ID}")
        fisher_high = _mean(moment_map, f"fisher_half::{HIGH_CANDIDATE_ID}")
        fisher_gain = fisher_low - fisher_high
        return {
            "C5a_Taylor_quant_I": taylor_low / n_params,
            "C5a_Taylor_quant_A_low": taylor_low,
            "C5a_Taylor_quant_A_high": taylor_high,
            "C5a_Taylor_quant_raw_gain": taylor_gain,
            "C5a_Taylor_quant_G": max(0.0, taylor_gain),
            "C5a_Taylor_L1_S": _mean(moment_map, "taylor_delete_l1") / n_params,
            "C5a_Taylor_squared_S": _mean(moment_map, "taylor_delete_squared")
            / n_params,
            "C5b_Empirical_Fisher_I": fisher_low / n_params,
            "C5b_Empirical_Fisher_A_low": fisher_low,
            "C5b_Empirical_Fisher_A_high": fisher_high,
            "C5b_Empirical_Fisher_raw_gain": fisher_gain,
            "C5b_Empirical_Fisher_G": max(0.0, fisher_gain),
        }
    if mode == "self_fisher":
        fisher_low = _mean(moment_map, f"fisher_half::{LOW_CANDIDATE_ID}")
        fisher_high = _mean(moment_map, f"fisher_half::{HIGH_CANDIDATE_ID}")
        fisher_gain = fisher_low - fisher_high
        return {
            "C5b_Self_Fisher_I": fisher_low / n_params,
            "C5b_Self_Fisher_A_low": fisher_low,
            "C5b_Self_Fisher_A_high": fisher_high,
            "C5b_Self_Fisher_raw_gain": fisher_gain,
            "C5b_Self_Fisher_G": max(0.0, fisher_gain),
        }
    raise ValueError(f"unknown mode: {mode}")


def detailed_scores_from_moments(
    mode: str,
    moment_map: Mapping[str, ScalarMoments],
    n_params: int,
) -> dict[str, Any]:
    if mode == "empirical":
        result: dict[str, Any] = {
            "C5a_Taylor_deletion_L1": {
                "method_id": EMPIRICAL_METHOD_IDS[0],
                "A": _mean(moment_map, "taylor_delete_l1"),
                "L": _mean(moment_map, "taylor_delete_l1") / n_params,
            },
            "C5a_Taylor_deletion_squared": {
                "method_id": EMPIRICAL_METHOD_IDS[1],
                "A": _mean(moment_map, "taylor_delete_squared"),
                "L": _mean(moment_map, "taylor_delete_squared") / n_params,
            },
            "C5a_Taylor_quant": {},
            "C5b_Empirical_Fisher": {},
        }
        for candidate_id in CANDIDATE_IDS:
            taylor = _mean(moment_map, f"taylor_quant_abs_dot::{candidate_id}")
            fisher = _mean(moment_map, f"fisher_half::{candidate_id}")
            result["C5a_Taylor_quant"][candidate_id] = {
                "method_id": f"C5a_Taylor_quant_{candidate_id}",
                "A": taylor,
                "L": taylor / n_params,
            }
            result["C5b_Empirical_Fisher"][candidate_id] = {
                "method_id": f"C5b_Empirical_Fisher_{candidate_id}",
                "A": fisher,
                "L": fisher / n_params,
                "estimator": "empirical Fisher; corpus real-next-token NLL gradient",
                "small_perturbation_KL_theoretical_connection": False,
            }
        return result
    if mode == "self_fisher":
        result = {"C5b_Self_Fisher": {}}
        for candidate_id in CANDIDATE_IDS:
            fisher = _mean(moment_map, f"fisher_half::{candidate_id}")
            result["C5b_Self_Fisher"][candidate_id] = {
                "method_id": f"C5b_Self_Fisher_{candidate_id}",
                "A": fisher,
                "L": fisher / n_params,
                "estimator": "self-Fisher; full-vocabulary categorical teacher samples",
                "small_perturbation_KL_theoretical_connection": True,
            }
        return result
    raise ValueError(f"unknown mode: {mode}")


def score_columns_for_mode(mode: str) -> tuple[str, ...]:
    if mode == "empirical":
        return EMPIRICAL_SCORE_COLUMNS
    if mode == "self_fisher":
        return SELF_FISHER_SCORE_COLUMNS
    raise ValueError(f"unknown mode: {mode}")


def load_active_roster(
    roster_path: Path,
    manifest_path: Path,
    *,
    smoke: bool,
    max_tensors: int | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selection = PERTURB.load_selection(roster_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "importance-score-source-roster-v0.1":
        raise SystemExit("active roster manifest has an unexpected schema")
    if manifest.get("roster_sha256") != sha256_file(roster_path):
        raise SystemExit("active roster bytes differ from their frozen manifest")
    if int(manifest.get("roster_tensor_count", -1)) != len(selection):
        raise SystemExit("active roster count differs from its frozen manifest")
    if not str(manifest.get("status", "")).startswith("frozen"):
        raise SystemExit("tensor roster is not frozen")
    model_ids = {str(row.get("model_id")) for row in selection}
    if model_ids != {str(manifest.get("model_id"))}:
        raise SystemExit("active roster model ID differs from its frozen manifest")
    if str(manifest.get("architecture", "")).startswith("gemma") and not manifest.get(
        "source_activity_filter"
    ):
        raise SystemExit("Gemma C5 requires the activity-filtered source roster")
    names: set[str] = set()
    for row in selection:
        required = {
            "hf_name",
            "shape",
            "n_params",
            "layer_id",
            "canonical_family",
            "model_id",
            "architecture",
        }
        missing = sorted(required - set(row))
        if missing:
            raise SystemExit(f"active roster row is missing fields: {missing}")
        name = str(row["hf_name"])
        if name in names:
            raise SystemExit(f"active roster contains duplicate tensor: {name}")
        names.add(name)
        shape = tuple(int(value) for value in row["shape"])
        if len(shape) != 2 or math.prod(shape) != int(row["n_params"]):
            raise SystemExit(f"invalid active roster shape/count: {name}")
    if max_tensors is not None:
        if not smoke:
            raise SystemExit("--max-tensors is smoke-only; formal C5 must use the full roster")
        if not 1 <= max_tensors <= 2:
            raise SystemExit("C5 smoke requires one or two tensors")
        selection = selection[:max_tensors]
    return selection, manifest


def load_shards(
    paths: list[Path],
    *,
    smoke: bool,
    max_records_per_shard: int | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if len(paths) != SHARD_COUNT or len(set(paths)) != SHARD_COUNT:
        raise SystemExit("C5 requires exactly four distinct --corpus-shard paths")
    if max_records_per_shard is not None and not smoke:
        raise SystemExit("--max-records-per-shard is smoke-only")
    if smoke and max_records_per_shard != 1:
        raise SystemExit("C5 smoke must use exactly one record from each of four shards")
    seen: set[str] = set()
    metadata: list[dict[str, Any]] = []
    flattened: list[dict[str, Any]] = []
    for shard_index, path in enumerate(paths):
        examples = list(COLLECTOR.iter_examples(path))
        if max_records_per_shard is not None:
            examples = examples[:max_records_per_shard]
        if not examples:
            raise SystemExit(f"empty C5 corpus shard: {path}")
        record_ids = [str(example["record_id"]) for example in examples]
        if len(set(record_ids)) != len(record_ids):
            raise SystemExit(f"duplicate record IDs within C5 shard: {path}")
        overlap = seen.intersection(record_ids)
        if overlap:
            raise SystemExit(f"C5 shards overlap in record IDs: {sorted(overlap)[:4]}")
        seen.update(record_ids)
        metadata.append(
            {
                "shard_index": shard_index,
                "path": str(path),
                "sha256": sha256_file(path),
                "record_count": len(examples),
                "record_ids_sha256": canonical_sha(record_ids),
            }
        )
        for record_index, example in enumerate(examples):
            flattened.append(
                {
                    "shard_index": shard_index,
                    "record_index": record_index,
                    "example": example,
                }
            )
    if smoke and len(flattened) != 4:
        raise SystemExit("C5 smoke must contain exactly four records total")
    return flattened, metadata


def validate_candidate_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    candidates = {str(row.get("candidate_id")): row for row in manifest.get("candidates", [])}
    if set(candidates) != set(CANDIDATE_IDS):
        raise SystemExit(f"C5 candidate manifest must contain exactly {CANDIDATE_IDS}")
    if candidates[LOW_CANDIDATE_ID].get("role") != "b_0":
        raise SystemExit("frozen AQ4 candidate is not marked b_0")
    if candidates[HIGH_CANDIDATE_ID].get("role") != "B_high":
        raise SystemExit("frozen AQ5 candidate is not marked B_high")
    return manifest


def validate_fisher_corpus_manifest(
    path: Path, shard_paths: list[Path]
) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "importance-score-fisher-corpus-freeze-v0.1":
        raise SystemExit("D_fisher manifest has an unexpected schema")
    if not str(manifest.get("status", "")).startswith("sealed"):
        raise SystemExit("D_fisher manifest is not sealed")
    frozen_shards = manifest.get("D_fisher", {}).get("shards", [])
    if len(frozen_shards) != SHARD_COUNT:
        raise SystemExit("D_fisher manifest does not bind exactly four shards")
    for shard_index, (frozen, observed_path) in enumerate(
        zip(frozen_shards, shard_paths, strict=True)
    ):
        expected_sha = frozen.get("sha256")
        actual_sha = sha256_file(observed_path)
        if expected_sha != actual_sha:
            raise SystemExit(
                f"D_fisher shard {shard_index} differs from its sealed manifest"
            )
        frozen_path = Path(str(frozen.get("path", ""))).expanduser()
        if frozen_path.name != observed_path.name:
            raise SystemExit(
                f"D_fisher shard {shard_index} path/order differs from its manifest"
            )
    return manifest


def resolve_stats_path(path: Path) -> Path:
    return path / "activation_second_moments.safetensors" if path.is_dir() else path


def prepare_tensor_caches(
    args: argparse.Namespace,
    model: torch.nn.Module,
    selection: list[dict[str, Any]],
    codebooks: Mapping[tuple[str, str], torch.Tensor],
    codebook_file_sha: str,
    activation_stats: Mapping[str, torch.Tensor],
    stats_sha: str,
    device: torch.device,
) -> tuple[dict[str, torch.nn.Parameter], dict[str, TensorQuantCache], dict[str, Any]]:
    estimated_bytes = sum(int(row["n_params"]) * 2 * 3 for row in selection)
    if estimated_bytes > args.max_host_cache_bytes:
        raise SystemExit(
            "source+AQ4+AQ5 BF16 host cache would exceed the configured cap: "
            f"estimated={estimated_bytes}, cap={args.max_host_cache_bytes}"
        )
    candidates = {
        candidate_id: SAMPLER.candidate_from_id(candidate_id)
        for candidate_id in CANDIDATE_IDS
    }
    if any(candidate is None for candidate in candidates.values()):
        raise RuntimeError("existing sampler does not define the frozen C5 candidates")
    parameters: dict[str, torch.nn.Parameter] = {}
    caches: dict[str, TensorQuantCache] = {}
    referenced_disk_bytes = 0
    overall_started_at = time.perf_counter()
    for tensor_index, row in enumerate(selection):
        tensor_started_at = time.perf_counter()
        tensor_name = str(row["hf_name"])
        family = str(row["canonical_family"])
        linear_name, linear = PERTURB.tensor_linear_module(model, tensor_name)
        parameter = linear._parameters.get("weight")
        if parameter is None:
            raise RuntimeError(f"resolved active tensor has no parameter: {linear_name}")
        if parameter.device != device:
            raise RuntimeError(f"active parameter is not on {device}: {tensor_name}")
        if parameter.dtype != torch.bfloat16:
            raise RuntimeError(
                f"C5 source model must be BF16, got {parameter.dtype} for {tensor_name}"
            )
        expected_shape = tuple(int(value) for value in row["shape"])
        if tuple(parameter.shape) != expected_shape or parameter.numel() != int(row["n_params"]):
            raise RuntimeError(f"active roster/model shape mismatch: {tensor_name}")
        if id(parameter) in {id(value) for value in parameters.values()}:
            raise RuntimeError(f"active roster resolves two names to one parameter: {tensor_name}")
        source_hash_started_at = time.perf_counter()
        source = parameter.detach().to("cpu", dtype=torch.bfloat16).contiguous()
        activation = SAMPLER.activation_stats_for_tensor(
            tensor_name,
            expected_shape,
            activation_stats,
        )
        quantized: dict[str, torch.Tensor] = {}
        quantization: dict[str, dict[str, Any]] = {}
        cache_paths: dict[str, str] = {}
        hashes = {"source_bf16": sha256_tensor(source)}
        preparation_timing = {
            "source_copy_and_hash_seconds": time.perf_counter() - source_hash_started_at
        }
        for candidate_id in CANDIDATE_IDS:
            candidate_started_at = time.perf_counter()
            codebook = codebooks.get((family, candidate_id))
            if codebook is None:
                raise RuntimeError(f"codebook missing: {family}/{candidate_id}")
            candidate_weight, candidate_meta, cache_path = PERTURB.load_or_quantize(
                args,
                tensor_name,
                family,
                parameter,
                activation,
                candidates[candidate_id],
                codebook,
                codebook_file_sha,
                stats_sha,
            )
            candidate_weight = candidate_weight.detach().contiguous()
            if candidate_weight.device.type != "cpu" or candidate_weight.dtype != torch.bfloat16:
                raise RuntimeError(
                    f"quantized cache is not BF16 CPU for {tensor_name}/{candidate_id}"
                )
            if tuple(candidate_weight.shape) != expected_shape:
                raise RuntimeError(f"quantized cache shape mismatch: {tensor_name}/{candidate_id}")
            quantized[candidate_id] = candidate_weight
            quantization[candidate_id] = candidate_meta
            cache_paths[candidate_id] = str(cache_path)
            hashes[f"quantized_bf16::{candidate_id}"] = sha256_tensor(candidate_weight)
            hashes[f"cache_file::{candidate_id}"] = sha256_file(cache_path)
            preparation_timing[f"quantize_load_and_hash::{candidate_id}"] = (
                time.perf_counter() - candidate_started_at
            )
            referenced_disk_bytes += cache_path.stat().st_size
            if referenced_disk_bytes > args.max_disk_cache_bytes:
                raise SystemExit(
                    "referenced exact-quantized disk cache exceeds configured cap: "
                    f"bytes={referenced_disk_bytes}, cap={args.max_disk_cache_bytes}"
                )
        cache = TensorQuantCache(
            tensor_name=tensor_name,
            source_bf16_cpu=source,
            quantized_bf16_cpu=quantized,
            quantization=quantization,
            cache_paths=cache_paths,
            content_sha256=hashes,
            preparation_timing=preparation_timing,
        )
        parameters[tensor_name] = parameter
        caches[tensor_name] = cache
        emit_progress(
            {
                "event": "tensor_cache_ready",
                "tensor_index": tensor_index,
                "tensor_count": len(selection),
                "tensor_name": tensor_name,
                "n_params": cache.n_params,
                "host_cache_bytes": cache.host_bytes,
                "tensor_elapsed_seconds": elapsed(device, tensor_started_at),
                "stage_elapsed_seconds": elapsed(device, overall_started_at),
            }
        )
    actual_bytes = sum(cache.host_bytes for cache in caches.values())
    if actual_bytes != estimated_bytes:
        raise RuntimeError(
            f"BF16 host cache accounting mismatch: estimated={estimated_bytes}, actual={actual_bytes}"
        )
    return parameters, caches, {
        "policy": "source, AQ4, and AQ5 are retained as BF16 CPU tensors only",
        "delta_policy": (
            "each quantized-minus-source difference is constructed transiently in FP32; "
            "no BF16-rounded delta is stored"
        ),
        "estimated_bytes": estimated_bytes,
        "actual_bytes": actual_bytes,
        "configured_limit_bytes": args.max_host_cache_bytes,
        "referenced_disk_cache_bytes": referenced_disk_bytes,
        "configured_disk_limit_bytes": args.max_disk_cache_bytes,
        "elapsed_seconds": elapsed(device, overall_started_at),
    }


def input_signature(
    args: argparse.Namespace,
    selection: list[dict[str, Any]],
    shard_metadata: list[dict[str, Any]],
) -> str:
    return canonical_sha(
        {
            "schema_version": SCHEMA_VERSION,
            "run_id": args.run_id,
            "mode": args.mode,
            "model_dir": str(args.model_dir),
            "tensor_selection_sha256": sha256_file(args.tensor_selection),
            "tensor_selection_manifest_sha256": sha256_file(args.tensor_selection_manifest),
            "selected_tensor_names": [str(row["hf_name"]) for row in selection],
            "corpus_shards": shard_metadata,
            "activation_stats_sha256": sha256_file(resolve_stats_path(args.activation_stats)),
            "family_codebooks_sha256": sha256_file(args.family_codebooks),
            "candidate_manifest_sha256": sha256_file(args.candidate_manifest),
            "fisher_corpus_manifest_sha256": sha256_file(args.fisher_corpus_manifest),
            "candidate_ids": CANDIDATE_IDS,
            "sequence_length": args.sequence_length,
            "mc_samples": args.mc_samples if args.mode == "self_fisher" else 0,
            "mc_selection_note": args.mc_selection_note,
            "seed": args.seed,
            "gradient_chunk_elements": args.gradient_chunk_elements,
            "max_fit_elements": args.max_fit_elements,
            "scale_window": args.scale_window,
            "group_chunk": args.group_chunk,
            "dtype": args.dtype,
            "torch_threads": args.torch_threads,
            "torch_interop_threads": args.torch_interop_threads,
            "max_host_cache_bytes": args.max_host_cache_bytes,
            "max_disk_cache_bytes": args.max_disk_cache_bytes,
            "trust_remote_code": args.trust_remote_code,
            "device_contract": "HIP_VISIBLE_DEVICES=1; one gfx1201 at logical cuda:0",
            "smoke": args.smoke,
            "implementation_sha256": sha256_file(Path(__file__).resolve()),
            "perturbation_runner_sha256": sha256_file(
                Path(__file__).resolve().parent
                / "run-importance-single-tensor-perturbation.py"
            ),
        }
    )


def checkpoint_payload(
    *,
    signature: str,
    mode: str,
    next_sample_index: int,
    next_draw_index: int,
    aggregates: Mapping[str, TensorAggregate],
    inflight: Mapping[str, Any],
    sample_timings: list[dict[str, Any]],
    valid_next_tokens: Mapping[str, int],
) -> dict[str, Any]:
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "status": "resumable",
        "updated_at_utc": utc_now(),
        "input_signature_sha256": signature,
        "mode": mode,
        "cursor": {
            "next_sample_index": next_sample_index,
            "next_draw_index": next_draw_index,
        },
        "aggregates": aggregates_to_json(aggregates),
        "inflight": dict(inflight),
        "sample_timings": sample_timings,
        "valid_next_tokens": dict(valid_next_tokens),
    }


def load_checkpoint(
    path: Path,
    *,
    signature: str,
    mode: str,
) -> tuple[
    int,
    int,
    dict[str, TensorAggregate],
    dict[str, Any],
    list[dict[str, Any]],
    dict[str, int],
]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise SystemExit("C5 checkpoint has an unexpected schema")
    if payload.get("input_signature_sha256") != signature or payload.get("mode") != mode:
        raise SystemExit("C5 checkpoint is bound to different frozen inputs")
    cursor = payload.get("cursor", {})
    return (
        int(cursor.get("next_sample_index", 0)),
        int(cursor.get("next_draw_index", 0)),
        aggregates_from_json(payload.get("aggregates", {})),
        dict(payload.get("inflight", {})),
        list(payload.get("sample_timings", [])),
        {key: int(value) for key, value in payload.get("valid_next_tokens", {}).items()},
    )


def add_observation_results(
    aggregates: Mapping[str, TensorAggregate],
    results: Mapping[str, Mapping[str, float]],
    hook_seconds: Mapping[str, float],
    shard_index: int,
) -> None:
    if set(aggregates) != set(results) or set(results) != set(hook_seconds):
        raise RuntimeError("gradient result roster is incomplete")
    for tensor_name in sorted(results):
        aggregates[tensor_name].add(shard_index, results[tensor_name], hook_seconds[tensor_name])


def append_inflight(
    inflight: dict[str, Any],
    results: Mapping[str, Mapping[str, float]],
    *,
    record_id: str,
) -> None:
    if not inflight:
        inflight.update({"record_id": record_id, "values": {}})
    if inflight.get("record_id") != record_id:
        raise RuntimeError("self-Fisher inflight record mismatch")
    values = inflight.setdefault("values", {})
    for tensor_name in sorted(results):
        tensor_values = values.setdefault(tensor_name, {})
        for metric in (
            "gradient_trace",
            f"fisher_half::{LOW_CANDIDATE_ID}",
            f"fisher_half::{HIGH_CANDIDATE_ID}",
        ):
            tensor_values.setdefault(metric, []).append(float(results[tensor_name][metric]))


def finalize_inflight_mc(
    inflight: dict[str, Any],
    aggregates: Mapping[str, TensorAggregate],
    expected_draws: int,
) -> None:
    values = inflight.get("values", {})
    if set(values) != set(aggregates):
        raise RuntimeError("self-Fisher inflight tensor roster is incomplete")
    for tensor_name in sorted(values):
        for metric, observations in sorted(values[tensor_name].items()):
            if len(observations) != expected_draws:
                raise RuntimeError(
                    f"self-Fisher record has {len(observations)} draws, expected {expected_draws}"
                )
            mean = math.fsum(float(value) for value in observations) / expected_draws
            variance = statistics.variance(observations) if expected_draws > 1 else 0.0
            aggregates[tensor_name].add_mc_record(f"draw_mean::{metric}", mean)
            aggregates[tensor_name].add_mc_record(
                f"within_record_draw_sample_variance::{metric}", variance
            )
    inflight.clear()


def move_encoded_to_device(
    tokenizer,
    example: Mapping[str, Any],
    sequence_length: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    tensors, _ = COLLECTOR.encode_examples(tokenizer, [dict(example)], sequence_length, False)
    if any(int(value.shape[0]) != 1 for value in tensors.values() if value.ndim > 0):
        raise RuntimeError("C5 encoder violated the batch=1 contract")
    return {key: value.to(device) for key, value in tensors.items()}


def run_gradient_passes(
    args: argparse.Namespace,
    tokenizer,
    model: torch.nn.Module,
    samples: list[dict[str, Any]],
    parameters: Mapping[str, torch.nn.Parameter],
    caches: Mapping[str, TensorQuantCache],
    aggregates: dict[str, TensorAggregate],
    *,
    signature: str,
    checkpoint_path: Path,
    start_sample_index: int,
    start_draw_index: int,
    inflight: dict[str, Any],
    sample_timings: list[dict[str, Any]],
    valid_next_tokens: dict[str, int],
    device: torch.device,
) -> None:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    for parameter in parameters.values():
        parameter.requires_grad_(True)

    with PostAccumulateGradientSession(
        parameters,
        caches,
        mode=args.mode,
        chunk_elements=args.gradient_chunk_elements,
        progress=emit_progress,
    ) as session:
        for sample_index in range(start_sample_index, len(samples)):
            sample = samples[sample_index]
            shard_index = int(sample["shard_index"])
            record_index = int(sample["record_index"])
            example = sample["example"]
            record_id = str(example["record_id"])
            sample_started_at = time.perf_counter()
            emit_progress(
                {
                    "event": "sample_start",
                    "mode": args.mode,
                    "sample_index": sample_index,
                    "sample_count": len(samples),
                    "shard_index": shard_index,
                    "record_index": record_index,
                    "record_id": record_id,
                }
            )
            tensors = move_encoded_to_device(
                tokenizer, example, args.sequence_length, device
            )
            forward_started_at = time.perf_counter()
            output = PERTURB.run_model(model, tensors)
            logits = output.logits
            forward_seconds = elapsed(device, forward_started_at)
            valid_logits, real_targets, valid_count = valid_next_token_view(logits, tensors)
            valid_next_tokens[record_id] = valid_count

            if args.mode == "empirical":
                if start_draw_index and sample_index == start_sample_index:
                    raise RuntimeError("empirical checkpoint cannot have a nonzero draw cursor")
                backward_started_at = time.perf_counter()
                loss = F.cross_entropy(valid_logits.float(), real_targets, reduction="mean")
                session.begin(
                    {
                        "sample_index": sample_index,
                        "shard_index": shard_index,
                        "record_index": record_index,
                        "record_id": record_id,
                        "draw_index": None,
                    }
                )
                try:
                    loss.backward()
                    results, hook_seconds = session.finish()
                except BaseException:
                    session.abort()
                    raise
                backward_seconds = elapsed(device, backward_started_at)
                add_observation_results(
                    aggregates, results, hook_seconds, shard_index
                )
                sample_timings.append(
                    {
                        "sample_index": sample_index,
                        "shard_index": shard_index,
                        "record_index": record_index,
                        "record_id": record_id,
                        "draw_index": None,
                        "valid_next_tokens": valid_count,
                        "forward_seconds": forward_seconds,
                        "backward_and_reduction_seconds": backward_seconds,
                        "loss": float(loss.detach()),
                    }
                )
                next_sample, next_draw = sample_index + 1, 0
                write_replace_atomic(
                    checkpoint_path,
                    checkpoint_payload(
                        signature=signature,
                        mode=args.mode,
                        next_sample_index=next_sample,
                        next_draw_index=next_draw,
                        aggregates=aggregates,
                        inflight=inflight,
                        sample_timings=sample_timings,
                        valid_next_tokens=valid_next_tokens,
                    ),
                )
                emit_progress(
                    {
                        "event": "sample_complete",
                        "mode": args.mode,
                        "sample_index": sample_index,
                        "record_id": record_id,
                        "valid_next_tokens": valid_count,
                        "sample_elapsed_seconds": elapsed(device, sample_started_at),
                    }
                )
            else:
                draw_start = start_draw_index if sample_index == start_sample_index else 0
                if draw_start == 0 and inflight:
                    raise RuntimeError("unexpected self-Fisher inflight data at record boundary")
                if draw_start > 0 and inflight.get("record_id") != record_id:
                    raise RuntimeError("resumed self-Fisher inflight record differs from cursor")
                for draw_index in range(draw_start, args.mc_samples):
                    seed = stable_self_fisher_seed(args.seed, record_id, draw_index)
                    emit_progress(
                        {
                            "event": "mc_draw_start",
                            "mode": args.mode,
                            "sample_index": sample_index,
                            "shard_index": shard_index,
                            "record_id": record_id,
                            "draw_index": draw_index,
                            "mc_samples": args.mc_samples,
                            "seed": seed,
                        }
                    )
                    sampled_targets = sample_self_fisher_targets(valid_logits, seed=seed)
                    log_probability = self_fisher_log_probability(
                        valid_logits, sampled_targets
                    )
                    backward_started_at = time.perf_counter()
                    session.begin(
                        {
                            "sample_index": sample_index,
                            "shard_index": shard_index,
                            "record_index": record_index,
                            "record_id": record_id,
                            "draw_index": draw_index,
                            "mc_samples": args.mc_samples,
                            "seed": seed,
                        }
                    )
                    try:
                        log_probability.backward(
                            retain_graph=(draw_index + 1 < args.mc_samples)
                        )
                        results, hook_seconds = session.finish()
                    except BaseException:
                        session.abort()
                        raise
                    backward_seconds = elapsed(device, backward_started_at)
                    add_observation_results(
                        aggregates, results, hook_seconds, shard_index
                    )
                    append_inflight(inflight, results, record_id=record_id)
                    last_draw = draw_index + 1 == args.mc_samples
                    if last_draw:
                        finalize_inflight_mc(inflight, aggregates, args.mc_samples)
                    sample_timings.append(
                        {
                            "sample_index": sample_index,
                            "shard_index": shard_index,
                            "record_index": record_index,
                            "record_id": record_id,
                            "draw_index": draw_index,
                            "seed": seed,
                            "valid_next_tokens": valid_count,
                            "forward_seconds": forward_seconds if draw_index == draw_start else 0.0,
                            "forward_shared_across_record_draws": True,
                            "backward_and_reduction_seconds": backward_seconds,
                            "mean_log_probability": float(log_probability.detach()),
                        }
                    )
                    next_sample = sample_index + 1 if last_draw else sample_index
                    next_draw = 0 if last_draw else draw_index + 1
                    write_replace_atomic(
                        checkpoint_path,
                        checkpoint_payload(
                            signature=signature,
                            mode=args.mode,
                            next_sample_index=next_sample,
                            next_draw_index=next_draw,
                            aggregates=aggregates,
                            inflight=inflight,
                            sample_timings=sample_timings,
                            valid_next_tokens=valid_next_tokens,
                        ),
                    )
                    emit_progress(
                        {
                            "event": "mc_draw_complete",
                            "mode": args.mode,
                            "sample_index": sample_index,
                            "record_id": record_id,
                            "draw_index": draw_index,
                            "mc_samples": args.mc_samples,
                            "backward_and_reduction_seconds": backward_seconds,
                        }
                    )
                    del sampled_targets, log_probability, results, hook_seconds
                emit_progress(
                    {
                        "event": "sample_complete",
                        "mode": args.mode,
                        "sample_index": sample_index,
                        "record_id": record_id,
                        "valid_next_tokens": valid_count,
                        "sample_elapsed_seconds": elapsed(device, sample_started_at),
                    }
                )
            start_draw_index = 0
            del valid_logits, real_targets, logits, output, tensors

    for parameter in parameters.values():
        parameter.requires_grad_(False)
        if parameter.grad is not None:
            raise RuntimeError("target gradient survived completed hook session")


def _mc_diagnostics(aggregate: TensorAggregate) -> dict[str, Any]:
    result: dict[str, Any] = {}
    metrics = (
        "gradient_trace",
        f"fisher_half::{LOW_CANDIDATE_ID}",
        f"fisher_half::{HIGH_CANDIDATE_ID}",
    )
    for metric in metrics:
        observed = aggregate.overall.get(metric)
        draw_means = aggregate.mc_records.get(f"draw_mean::{metric}")
        within = aggregate.mc_records.get(
            f"within_record_draw_sample_variance::{metric}"
        )
        result[metric] = {
            "all_record_draw_observations": observed.describe() if observed else None,
            "per_record_draw_mean": draw_means.describe() if draw_means else None,
            "within_record_draw_sample_variance": within.describe() if within else None,
        }
    return result


def build_rows(
    args: argparse.Namespace,
    selection: list[dict[str, Any]],
    aggregates: Mapping[str, TensorAggregate],
    caches: Mapping[str, TensorQuantCache],
    *,
    signature: str,
    shard_metadata: list[dict[str, Any]],
    valid_next_tokens: Mapping[str, int],
    common_metadata: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    for row in selection:
        tensor_name = str(row["hf_name"])
        aggregate = aggregates[tensor_name]
        n_params = int(row["n_params"])
        scores = flat_scores_from_moments(args.mode, aggregate.overall, n_params)
        if tuple(scores) != score_columns_for_mode(args.mode):
            raise RuntimeError("flat score columns are not in their frozen order")
        shard_scores = []
        for shard_index, shard in enumerate(aggregate.shards):
            flat = flat_scores_from_moments(args.mode, shard, n_params)
            shard_scores.append({"shard_index": shard_index, **flat})
        cache = caches[tensor_name]
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "created_at_utc": utc_now(),
                "status": "ok",
                "run_id": args.run_id,
                "mode": args.mode,
                "model_id": str(row["model_id"]),
                "model_dir": str(args.model_dir),
                "architecture": str(row["architecture"]),
                "tensor_name": tensor_name,
                "hf_name": tensor_name,
                "layer_id": int(row["layer_id"]),
                "canonical_family": str(row["canonical_family"]),
                "shape": [int(value) for value in row["shape"]],
                "n_params": n_params,
                "candidate_ids": {
                    "low": LOW_CANDIDATE_ID,
                    "high": HIGH_CANDIDATE_ID,
                },
                "method_ids": (
                    list(EMPIRICAL_METHOD_IDS)
                    if args.mode == "empirical"
                    else list(SELF_FISHER_METHOD_IDS)
                ),
                "scores": scores,
                "shard_scores": shard_scores,
                "detailed_scores": detailed_scores_from_moments(
                    args.mode, aggregate.overall, n_params
                ),
                "observation_moments": {
                    key: moment.describe()
                    for key, moment in sorted(aggregate.overall.items())
                },
                "mc_diagnostics": (
                    _mc_diagnostics(aggregate) if args.mode == "self_fisher" else None
                ),
                "observation_count": next(iter(aggregate.overall.values())).count,
                "record_count": len(valid_next_tokens),
                "mc_samples_per_record": args.mc_samples if args.mode == "self_fisher" else None,
                "valid_next_tokens": sum(valid_next_tokens.values()),
                "hook_timing": {
                    "calls": aggregate.hook_calls,
                    "total_seconds": aggregate.hook_elapsed_seconds,
                    "maximum_seconds": aggregate.hook_max_seconds,
                },
                "quantization": cache.quantization,
                "quantized_cache_paths": cache.cache_paths,
                "weight_cache_content_sha256": cache.content_sha256,
                "weight_cache_preparation_timing": cache.preparation_timing,
                "input_signature_sha256": signature,
                "corpus_shards": shard_metadata,
                "contract": dict(common_metadata),
                "notes": args.note,
            }
        )
    return rows


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("empirical", "self_fisher"), required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--activation-stats", type=Path, required=True)
    parser.add_argument("--family-codebooks", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--fisher-corpus-manifest", type=Path, required=True)
    parser.add_argument("--tensor-selection", type=Path, required=True)
    parser.add_argument("--tensor-selection-manifest", type=Path, required=True)
    parser.add_argument("--corpus-shard", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--quantized-cache-dir", type=Path, required=True)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--mc-samples", type=int, default=4)
    parser.add_argument(
        "--mc-selection-note",
        help="Formal self-Fisher only: frozen reason for the selected MC draw count.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gradient-chunk-elements", type=int, default=1 << 20)
    parser.add_argument("--max-fit-elements", type=int, default=65536)
    parser.add_argument("--scale-window", type=int, default=4)
    parser.add_argument("--group-chunk", type=int, default=4096)
    parser.add_argument("--max-host-cache-bytes", type=int, default=DEFAULT_HOST_CACHE_LIMIT_BYTES)
    parser.add_argument("--max-disk-cache-bytes", type=int, default=DEFAULT_DISK_CACHE_LIMIT_BYTES)
    parser.add_argument("--torch-threads", type=int, default=32)
    parser.add_argument("--torch-interop-threads", type=int, default=1)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--note", action="append", default=[])
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-tensors", type=int)
    parser.add_argument("--max-records-per-shard", type=int)
    args = parser.parse_args(argv)
    args.dtype = "bfloat16"
    args.device = "cuda:0"
    args.model_class = "causal_lm"
    return args


def normalize_paths(args: argparse.Namespace) -> None:
    path_fields = (
        "model_dir",
        "activation_stats",
        "family_codebooks",
        "candidate_manifest",
        "fisher_corpus_manifest",
        "tensor_selection",
        "tensor_selection_manifest",
        "output",
        "quantized_cache_dir",
    )
    for field_name in path_fields:
        setattr(args, field_name, getattr(args, field_name).expanduser().resolve())
    args.corpus_shard = [path.expanduser().resolve() for path in args.corpus_shard]
    args.receipt = (
        args.receipt.expanduser().resolve()
        if args.receipt is not None
        else receipt_path_for(args.output)
    )
    args.checkpoint = (
        args.checkpoint.expanduser().resolve()
        if args.checkpoint is not None
        else checkpoint_path_for(args.output)
    )


def validate_args(args: argparse.Namespace) -> None:
    if len({args.output, args.receipt, args.checkpoint}) != 3:
        raise SystemExit("output, receipt, and checkpoint paths must be distinct")
    positive = (
        args.sequence_length,
        args.mc_samples,
        args.gradient_chunk_elements,
        args.max_fit_elements,
        args.group_chunk,
        args.max_host_cache_bytes,
        args.max_disk_cache_bytes,
        args.torch_threads,
        args.torch_interop_threads,
    )
    if min(positive) < 1:
        raise SystemExit("all size/count/thread arguments must be positive")
    if args.mode == "empirical" and args.mc_samples != 4:
        # MC samples do not affect empirical scores; rejecting non-default values
        # catches copy/paste mistakes while keeping one frozen CLI signature.
        raise SystemExit("--mc-samples is only configurable in self_fisher mode")
    if args.mode == "self_fisher" and not 1 <= args.mc_samples <= 64:
        raise SystemExit("self-Fisher MC samples must be in [1, 64]")
    if args.mode == "self_fisher" and not args.smoke and not args.mc_selection_note:
        raise SystemExit("formal self-Fisher requires --mc-selection-note from the smoke variance audit")
    if args.mode == "empirical" and args.mc_selection_note is not None:
        raise SystemExit("--mc-selection-note is valid only in self_fisher mode")
    if args.max_tensors is not None and args.max_tensors < 1:
        raise SystemExit("--max-tensors must be positive")
    if args.max_records_per_shard is not None and args.max_records_per_shard < 1:
        raise SystemExit("--max-records-per-shard must be positive")
    if args.smoke and (args.max_tensors is None or args.max_records_per_shard is None):
        raise SystemExit("smoke mode requires bounded tensors and records")
    if not args.smoke and (args.max_tensors is not None or args.max_records_per_shard is not None):
        raise SystemExit("formal C5 cannot silently truncate tensors or records")


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True, write_through=True)
        sys.stderr.reconfigure(line_buffering=True, write_through=True)
    args = parse_args(argv)
    normalize_paths(args)
    validate_args(args)
    require_fresh_final_outputs(args.output, args.receipt)
    if args.checkpoint.exists() and not args.resume:
        raise SystemExit(
            f"checkpoint exists; pass --resume or choose fresh paths: {args.checkpoint}"
        )
    if args.resume and not args.checkpoint.exists():
        emit_progress({"event": "resume_requested_without_checkpoint", "starting_fresh": True})
    for path in (
        args.model_dir,
        args.activation_stats,
        args.family_codebooks,
        args.candidate_manifest,
        args.fisher_corpus_manifest,
        args.tensor_selection,
        args.tensor_selection_manifest,
        *args.corpus_shard,
    ):
        if not path.exists():
            raise SystemExit(f"required C5 input is missing: {path}")
    args.quantized_cache_dir.mkdir(parents=True, exist_ok=True)

    torch.set_num_threads(args.torch_threads)
    torch.set_num_interop_threads(args.torch_interop_threads)
    torch.manual_seed(args.seed)
    device_metadata = enforce_r9700_visibility()
    device = torch.device("cuda:0")
    overall_started_at = time.perf_counter()
    emit_progress(
        {
            "event": "stage_start",
            "stage": "validate_frozen_inputs",
            "run_id": args.run_id,
            "mode": args.mode,
        }
    )
    selection, roster_manifest = load_active_roster(
        args.tensor_selection,
        args.tensor_selection_manifest,
        smoke=args.smoke,
        max_tensors=args.max_tensors,
    )
    frozen_model_dir = Path(str(roster_manifest.get("model_dir", ""))).expanduser().resolve()
    if frozen_model_dir != args.model_dir:
        raise SystemExit(
            f"model directory differs from the frozen active roster: {args.model_dir} != "
            f"{frozen_model_dir}"
        )
    samples, shard_metadata = load_shards(
        args.corpus_shard,
        smoke=args.smoke,
        max_records_per_shard=args.max_records_per_shard,
    )
    validate_candidate_manifest(args.candidate_manifest)
    validate_fisher_corpus_manifest(args.fisher_corpus_manifest, args.corpus_shard)
    signature = input_signature(args, selection, shard_metadata)

    start_sample_index = 0
    start_draw_index = 0
    aggregates = new_aggregates(str(row["hf_name"]) for row in selection)
    inflight: dict[str, Any] = {}
    sample_timings: list[dict[str, Any]] = []
    valid_next_tokens: dict[str, int] = {}
    if args.resume and args.checkpoint.exists():
        (
            start_sample_index,
            start_draw_index,
            aggregates,
            inflight,
            sample_timings,
            valid_next_tokens,
        ) = load_checkpoint(args.checkpoint, signature=signature, mode=args.mode)
        if set(aggregates) != {str(row["hf_name"]) for row in selection}:
            raise SystemExit("checkpoint tensor roster differs from frozen active roster")
        emit_progress(
            {
                "event": "checkpoint_resumed",
                "next_sample_index": start_sample_index,
                "next_draw_index": start_draw_index,
                "checkpoint": str(args.checkpoint),
            }
        )

    emit_progress(
        {
            "event": "stage_start",
            "stage": "load_model",
            "elapsed_seconds": elapsed(device, overall_started_at),
        }
    )
    with FallbackMonitor() as fallback_monitor:
        model_load_started_at = time.perf_counter()
        tokenizer, model = COLLECTOR.load_transformers_model(args)
        model_load_seconds = elapsed(device, model_load_started_at)
        if next(model.parameters()).device != device:
            raise RuntimeError("loaded model is not on logical cuda:0")
        codebooks, codebook_file_sha = PERTURB.load_codebooks(args.family_codebooks)
        activation_stats = SAMPLER.load_activation_stats(args.activation_stats)
        stats_sha = sha256_file(resolve_stats_path(args.activation_stats))
        emit_progress(
            {
                "event": "stage_start",
                "stage": "prepare_bf16_weight_caches",
                "elapsed_seconds": elapsed(device, overall_started_at),
            }
        )
        parameters, caches, host_cache_metadata = prepare_tensor_caches(
            args,
            model,
            selection,
            codebooks,
            codebook_file_sha,
            activation_stats,
            stats_sha,
            device,
        )
        del activation_stats, codebooks
        emit_progress(
            {
                "event": "stage_start",
                "stage": "gradient_passes",
                "mode": args.mode,
                "sample_count": len(samples),
                "tensor_count": len(selection),
                "mc_samples": args.mc_samples if args.mode == "self_fisher" else 0,
                "elapsed_seconds": elapsed(device, overall_started_at),
            }
        )
        gradient_passes_started_at = time.perf_counter()
        run_gradient_passes(
            args,
            tokenizer,
            model,
            samples,
            parameters,
            caches,
            aggregates,
            signature=signature,
            checkpoint_path=args.checkpoint,
            start_sample_index=start_sample_index,
            start_draw_index=start_draw_index,
            inflight=inflight,
            sample_timings=sample_timings,
            valid_next_tokens=valid_next_tokens,
            device=device,
        )
        gradient_passes_seconds = elapsed(device, gradient_passes_started_at)
        if inflight:
            raise RuntimeError("self-Fisher completed with unfinished MC observations")
        common_metadata = {
            "formulas": FORMULAS,
            "gradient_estimator": (
                "empirical Fisher and Taylor use one shared real-next-token gradient per record"
                if args.mode == "empirical"
                else "self-Fisher uses independently sampled full-vocabulary teacher targets"
            ),
            "Fisher_estimator_separation": (
                "only self_fisher is identified with the theoretical small-perturbation KL "
                "connection; empirical Fisher is a secondary corpus/task-loss sensitivity"
            ),
            "record_weighting": (
                "equal weight per record after within-record valid-next-token mean"
            ),
            "mc_weighting": (
                "equal weight per record x draw; fixed draw count makes this the mean of "
                "per-record MC means"
                if args.mode == "self_fisher"
                else None
            ),
            "mc_selection_note": args.mc_selection_note,
            "valid_next_token_mask": (
                "attention_mask[:, :-1] AND attention_mask[:, 1:]; batch size exactly one"
            ),
            "reduction": (
                "element products and quantized-minus-source differences in FP32; each chunk "
                "reduced to FP64 scalar; Taylor-quant signed chunk dots summed before one "
                "full-tensor abs; observation means in deterministic shard/record/draw order"
            ),
            "gradient_storage": (
                "post-accumulate parameter hook only; p.grad=None immediately after scalar "
                "reduction; no per-parameter gradient/Fisher tensor retained"
            ),
            "weight_cache": host_cache_metadata,
            "shard_aggregation": "four fixed shard means plus the all-record observation mean",
            "self_fisher_seed": (
                "first 63 bits of SHA256('importance-score-C5-self-Fisher-v0.1' NUL "
                "base_seed NUL record_id NUL draw_index)"
                if args.mode == "self_fisher"
                else None
            ),
        }
        rows = build_rows(
            args,
            selection,
            aggregates,
            caches,
            signature=signature,
            shard_metadata=shard_metadata,
            valid_next_tokens=valid_next_tokens,
            common_metadata=common_metadata,
        )
        output_bytes = b"".join(
            (
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            ).encode("utf-8")
            for row in rows
        )
        write_new_atomic(args.output, output_bytes)
        output_sha = sha256_file(args.output)
        seed_rows = [
            {
                "record_id": str(sample["example"]["record_id"]),
                "draw_index": draw_index,
                "seed": stable_self_fisher_seed(
                    args.seed, str(sample["example"]["record_id"]), draw_index
                ),
            }
            for sample in samples
            for draw_index in (
                range(args.mc_samples) if args.mode == "self_fisher" else []
            )
        ]
        implementation_paths = {
            "gradient_runner": Path(__file__).resolve(),
            "perturbation_runner": Path(__file__).resolve().parent
            / "run-importance-single-tensor-perturbation.py",
            "sampler": Path(__file__).resolve().parent / "run-aq-tensor-sample.py",
            "activation_collector": Path(__file__).resolve().parent
            / "collect-activation-stats.py",
            "selection_plan": Path(__file__).resolve().parents[1]
            / "docs/plans/importance-score-algorithm-selection-plan-v0.1.md",
        }
        receipt = {
            "schema_version": RECEIPT_SCHEMA_VERSION,
            "created_at_utc": utc_now(),
            "status": "sealed C5 gradient score output",
            "run_id": args.run_id,
            "mode": args.mode,
            "smoke": args.smoke,
            "model_id": str(roster_manifest["model_id"]),
            "architecture": str(roster_manifest["architecture"]),
            "result_path": str(args.output),
            "result_sha256": output_sha,
            "tensor_count": len(rows),
            "name_set_sha256": canonical_sha(
                sorted(str(row["tensor_name"]) for row in rows)
            ),
            "output": {
                "path": str(args.output),
                "sha256": output_sha,
                "bytes": len(output_bytes),
                "rows": len(rows),
                "score_columns": list(score_columns_for_mode(args.mode)),
            },
            "input_signature_sha256": signature,
            "inputs": {
                "model_dir": str(args.model_dir),
                "model_config_sha256": (
                    sha256_file(args.model_dir / "config.json")
                    if (args.model_dir / "config.json").is_file()
                    else None
                ),
                "tensor_selection": str(args.tensor_selection),
                "tensor_selection_sha256": sha256_file(args.tensor_selection),
                "tensor_selection_manifest": str(args.tensor_selection_manifest),
                "tensor_selection_manifest_sha256": sha256_file(
                    args.tensor_selection_manifest
                ),
                "roster_status": roster_manifest.get("status"),
                "activation_stats": str(resolve_stats_path(args.activation_stats)),
                "activation_stats_sha256": stats_sha,
                "family_codebooks": str(args.family_codebooks),
                "family_codebooks_sha256": codebook_file_sha,
                "candidate_manifest": str(args.candidate_manifest),
                "candidate_manifest_sha256": sha256_file(args.candidate_manifest),
                "fisher_corpus_manifest": str(args.fisher_corpus_manifest),
                "fisher_corpus_manifest_sha256": sha256_file(
                    args.fisher_corpus_manifest
                ),
                "corpus_shards": shard_metadata,
            },
            "input_hashes": {
                "source_roster": sha256_file(args.tensor_selection),
                "source_roster_manifest": sha256_file(args.tensor_selection_manifest),
                "fisher_corpus_manifest": sha256_file(args.fisher_corpus_manifest),
                "candidate_manifest": sha256_file(args.candidate_manifest),
                "activation_stats": stats_sha,
                "family_codebooks": codebook_file_sha,
                "corpus_shards": [item["sha256"] for item in shard_metadata],
            },
            "candidate_ids": {
                "low": LOW_CANDIDATE_ID,
                "high": HIGH_CANDIDATE_ID,
            },
            "method_ids": (
                list(EMPIRICAL_METHOD_IDS)
                if args.mode == "empirical"
                else list(SELF_FISHER_METHOD_IDS)
            ),
            "execution": {
                "device": device_metadata,
                "sequence_length": args.sequence_length,
                "batch_size": 1,
                "record_count": len(samples),
                "valid_next_tokens": sum(valid_next_tokens.values()),
                "mc_samples": args.mc_samples if args.mode == "self_fisher" else None,
                "mc_selection_note": args.mc_selection_note,
                "seed": args.seed,
                "seed_schedule_sha256": canonical_sha(seed_rows) if seed_rows else None,
                "gradient_chunk_elements": args.gradient_chunk_elements,
                "sample_timings": sample_timings,
                "total_elapsed_seconds": elapsed(device, overall_started_at),
                "fallback_monitor": {
                    "known_allowed_count": len(fallback_monitor.allowed_messages),
                    "known_allowed_messages": fallback_monitor.allowed_messages,
                    "unknown_count": 0,
                },
                "checkpoint_policy": (
                    "atomic checkpoint after every completed record/draw; checkpoint removed "
                    "only after final JSONL and receipt publication"
                ),
                "stdout_stderr": "write-through/line-buffered; every sample/tensor/MC event flushed",
            },
            "execution_settings": {
                "mode": args.mode,
                "candidate_ids": {
                    "low": LOW_CANDIDATE_ID,
                    "high": HIGH_CANDIDATE_ID,
                },
                "sequence_length": args.sequence_length,
                "batch_size": 1,
                "shard_count": SHARD_COUNT,
                "record_count": len(samples),
                "mc_samples": args.mc_samples if args.mode == "self_fisher" else None,
                "mc_selection_note": args.mc_selection_note,
                "seed": args.seed,
                "gradient_chunk_elements": args.gradient_chunk_elements,
                "max_fit_elements": args.max_fit_elements,
                "scale_window": args.scale_window,
                "group_chunk": args.group_chunk,
                "source_and_quantized_cache_dtype": "bfloat16",
                "dtype": "bfloat16",
                "delta_dtype": "float32",
                "scalar_reduction_dtype": "float64",
                "device_contract": {
                    "HIP_VISIBLE_DEVICES": "1",
                    "visible_device_count": 1,
                    "logical_device": "cuda:0",
                    "architecture": "gfx1201",
                },
                "torch_threads": args.torch_threads,
                "torch_interop_threads": args.torch_interop_threads,
                "max_host_cache_bytes": args.max_host_cache_bytes,
                "max_disk_cache_bytes": args.max_disk_cache_bytes,
                "trust_remote_code": args.trust_remote_code,
                "smoke": args.smoke,
            },
            "timing": {
                "total_elapsed_seconds": elapsed(device, overall_started_at),
                "model_load_seconds": model_load_seconds,
                "weight_cache_elapsed_seconds": host_cache_metadata["elapsed_seconds"],
                "gradient_passes_seconds": gradient_passes_seconds,
                "sample_timings": sample_timings,
            },
            "contract": common_metadata,
            "implementation_hashes": {
                name: {"path": str(path), "sha256": sha256_file(path)}
                for name, path in implementation_paths.items()
            },
            "workspace_git_head": git_revision(),
            "notes": args.note,
        }
        write_new_atomic(
            args.receipt,
            (json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
                "utf-8"
            ),
        )
        if args.checkpoint.exists():
            args.checkpoint.unlink()
        summary = {
            "status": "complete",
            "mode": args.mode,
            "output": str(args.output),
            "output_sha256": output_sha,
            "receipt": str(args.receipt),
            "receipt_sha256": sha256_file(args.receipt),
            "tensor_count": len(rows),
            "record_count": len(samples),
            "mc_samples": args.mc_samples if args.mode == "self_fisher" else None,
            "elapsed_seconds": elapsed(device, overall_started_at),
        }
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
