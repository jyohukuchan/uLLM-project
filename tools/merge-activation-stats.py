#!/usr/bin/env python3
"""Merge fixed activation-stat shards with FP64 weighted reductions."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_stats(path: Path) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    with safe_open(path, framework="pt", device="cpu") as handle:
        for key in handle.keys():
            result[key] = handle.get_tensor(key)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", action="append", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_dirs = [path.expanduser().resolve() for path in args.input_dir]
    output_dir = args.output_dir.expanduser().resolve()
    if len(input_dirs) != 4:
        raise SystemExit("v0.1 D_stats merge requires exactly four fixed shards")
    metadata = [json.loads((path / "metadata.json").read_text(encoding="utf-8")) for path in input_dirs]
    modules = [set(item["modules"]) for item in metadata]
    if any(module_set != modules[0] for module_set in modules[1:]):
        raise SystemExit("module coverage differs across activation-stat shards")
    if any(item.get("device") != "cpu" or not item.get("require_cpu") for item in metadata):
        raise SystemExit("all activation-stat shards must be explicitly CPU-only")
    stats = [load_stats(path / "activation_second_moments.safetensors") for path in input_dirs]
    if any(set(item) != set(stats[0]) for item in stats[1:]):
        raise SystemExit("stat key coverage differs across shards")

    output: dict[str, torch.Tensor] = {}
    for module_name in sorted(modules[0]):
        counts = [int(item["modules"][module_name]["activation_count"]) for item in metadata]
        total = sum(counts)
        if total <= 0:
            raise SystemExit(f"zero activation count for {module_name}")
        second = sum(
            (stats[index][module_name].to(torch.float64) * counts[index] for index in range(len(stats))),
            torch.zeros_like(stats[0][module_name], dtype=torch.float64),
        ) / total
        mean_abs_key = f"{module_name}.mean_abs"
        mean_abs = sum(
            (stats[index][mean_abs_key].to(torch.float64) * counts[index] for index in range(len(stats))),
            torch.zeros_like(stats[0][mean_abs_key], dtype=torch.float64),
        ) / total
        max_key = f"{module_name}.max_abs"
        max_abs = torch.stack([item[max_key].to(torch.float32) for item in stats], dim=0).amax(dim=0)
        if not bool(torch.isfinite(second).all() and torch.isfinite(mean_abs).all() and torch.isfinite(max_abs).all()):
            raise SystemExit(f"non-finite merged statistic for {module_name}")
        output[module_name] = second.contiguous()
        output[mean_abs_key] = mean_abs.contiguous()
        output[max_key] = max_abs.contiguous()

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "activation_second_moments.safetensors"
    save_file(output, str(output_path))
    summary = {
        "schema_version": "aq-activation-stats-merge-v0.1",
        "run_id": args.run_id,
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "shard_count": 4,
        "shards": [
            {
                "directory": str(path),
                "metadata_sha256": sha256_file(path / "metadata.json"),
                "stats_sha256": sha256_file(path / "activation_second_moments.safetensors"),
                "tokens_seen": int(item["tokens_seen"]),
                "samples_seen": int(item["samples_seen"]),
            }
            for path, item in zip(input_dirs, metadata, strict=True)
        ],
        "tokens_seen": sum(int(item["tokens_seen"]) for item in metadata),
        "samples_seen": sum(int(item["samples_seen"]) for item in metadata),
        "module_count": len(modules[0]),
        "stat_dtype": "float64 second moment and mean_abs; float32 max_abs",
        "padding_mask_policy": metadata[0].get("padding_mask_policy"),
        "output_sha256": sha256_file(output_path),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
