#!/usr/bin/env python3
"""Freeze a GGUF-label-blind source tensor roster for score generation."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from safetensors import safe_open


def load_sampler():
    path = Path(__file__).resolve().parent / "run-aq-tensor-sample.py"
    spec = importlib.util.spec_from_file_location("importance_source_roster_sampler", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


SAMPLER = load_sampler()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--architecture", required=True)
    parser.add_argument("--scope-prefix", required=True)
    parser.add_argument("--family", action="append", required=True)
    parser.add_argument("--group-size", type=int, default=16)
    parser.add_argument("--model-provenance-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def inactive_source_tensor_reason(
    model_config: dict[str, Any], name: str, layer_id: int
) -> str | None:
    """Return a source-only exclusion reason for weights absent from inference.

    Gemma 4 checkpoints retain per-layer K/V tensors for the trailing KV-sharing
    layers, while the architecture reuses K/V states produced by the last
    non-sharing layers.  Transformers therefore does not instantiate or load
    those trailing projection modules.  Keeping the redundant checkpoint
    tensors in the score roster would fabricate activation statistics for
    parameters that cannot affect a forward pass.
    """

    text_config = model_config.get("text_config", model_config)
    if text_config.get("model_type") != "gemma4_text":
        return None
    layer_count = int(text_config.get("num_hidden_layers", 0))
    shared_layer_count = int(text_config.get("num_kv_shared_layers", 0))
    if layer_count < 1 or shared_layer_count < 1:
        return None
    first_shared_layer = layer_count - shared_layer_count
    if first_shared_layer < 0:
        raise ValueError("num_kv_shared_layers exceeds num_hidden_layers")
    if layer_id < first_shared_layer:
        return None
    if re.search(r"\.self_attn\.(?:k_proj|v_proj)\.weight$", name):
        return "inactive_gemma4_kv_shared_projection"
    return None


def main() -> int:
    args = parse_args()
    if args.group_size < 1:
        raise SystemExit("--group-size must be positive")
    model_dir = args.model_dir.expanduser().resolve()
    provenance = args.model_provenance_manifest.expanduser().resolve()
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    allowed = set(args.family)
    model_config_path = model_dir / "config.json"
    model_config = json.loads(model_config_path.read_text(encoding="utf-8"))
    rows = []
    header_digest = hashlib.sha256()
    all_header_count = 0
    exclusion_counts: Counter[str] = Counter()
    for path in SAMPLER.iter_safetensor_files(model_dir):
        with safe_open(path, framework="pt", device="cpu") as handle:
            for name in sorted(handle.keys()):
                shape = [int(value) for value in handle.get_slice(name).get_shape()]
                all_header_count += 1
                header_digest.update(
                    canonical_json({"name": name, "shape": shape}).encode("utf-8") + b"\n"
                )
                if not name.startswith(args.scope_prefix):
                    exclusion_counts["outside_scope_prefix"] += 1
                    continue
                if not name.endswith(".weight"):
                    exclusion_counts["not_weight"] += 1
                    continue
                family = SAMPLER.family_for_tensor(name)
                if family not in allowed:
                    exclusion_counts["family_not_selected"] += 1
                    continue
                if len(shape) != 2:
                    exclusion_counts["not_2d"] += 1
                    continue
                n_params = shape[0] * shape[1]
                if n_params % args.group_size:
                    exclusion_counts["group_misaligned"] += 1
                    continue
                layer_match = re.search(r"\.layers\.(\d+)\.", name)
                if layer_match is None:
                    exclusion_counts["layer_id_unavailable"] += 1
                    continue
                layer_id = int(layer_match.group(1))
                inactive_reason = inactive_source_tensor_reason(model_config, name, layer_id)
                if inactive_reason is not None:
                    exclusion_counts[inactive_reason] += 1
                    continue
                rows.append(
                    {
                        "model_id": args.model_id,
                        "architecture": args.architecture,
                        "hf_name": name,
                        "canonical_family": family,
                        "layer_id": layer_id,
                        "shape": shape,
                        "n_params": n_params,
                        "selection_inputs": [
                            "BF16 source tensor name",
                            "BF16 source tensor shape",
                            "scope prefix",
                            "family taxonomy",
                            "group alignment",
                            "source model config active-module semantics",
                        ],
                    }
                )
    rows.sort(key=lambda row: (row["canonical_family"], row["layer_id"], row["hf_name"]))
    if not rows:
        raise SystemExit("source roster is empty")
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")
    family_counts = Counter(str(row["canonical_family"]) for row in rows)
    layer_counts = Counter(int(row["layer_id"]) for row in rows)
    manifest = {
        "schema_version": "importance-score-source-roster-v0.1",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "frozen without reading GGUF tensor names, types, ordinals, or promotion labels",
        "model_id": args.model_id,
        "architecture": args.architecture,
        "model_dir": str(model_dir),
        "model_config_path": str(model_config_path),
        "model_config_sha256": sha256_file(model_config_path),
        "source_activity_filter": {
            "inputs": ["BF16 source tensor name", "BF16 source config.json"],
            "gemma4_rule": (
                "exclude k_proj/v_proj checkpoint tensors at layer >= "
                "num_hidden_layers - num_kv_shared_layers because those decoder layers reuse "
                "previously computed K/V states and instantiate no projection module"
            ),
            "label_blind": True,
        },
        "scope_prefix": args.scope_prefix,
        "allowed_families": sorted(allowed),
        "group_size": args.group_size,
        "source_header_count": all_header_count,
        "source_name_shape_header_sha256": header_digest.hexdigest(),
        "roster_path": str(output),
        "roster_sha256": sha256_file(output),
        "roster_tensor_count": len(rows),
        "family_counts": dict(sorted(family_counts.items())),
        "layer_count": len(layer_counts),
        "layer_tensor_counts": {str(key): value for key, value in sorted(layer_counts.items())},
        "exclusion_counts": dict(sorted(exclusion_counts.items())),
        "model_provenance_manifest": str(provenance),
        "model_provenance_manifest_sha256": sha256_file(provenance),
        "family_taxonomy_implementation": {
            "path": str(Path(SAMPLER.__file__).resolve()),
            "sha256": sha256_file(Path(SAMPLER.__file__).resolve()),
        },
        "forbidden_inputs": [
            "GGUF tensor name",
            "GGUF tensor type",
            "ordinal_ud",
            "ordinal_static",
            "promotion_delta_ordinal",
            "promotion_delta_bpp",
            "promoted",
            "candidate score values",
        ],
    }
    manifest_path = output.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
