#!/usr/bin/env python3
"""Build and hash a source-only score table before opening lockbox labels."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch


FORBIDDEN_ROSTER_KEYS = {
    "gguf_name",
    "qtype_ud",
    "qtype_static",
    "ordinal_ud",
    "ordinal_static",
    "packed_bpp_ud",
    "packed_bpp_static",
    "promotion_delta_ordinal",
    "promotion_delta_bpp",
    "promoted",
}

PRIMARY_SCORE_ORDER = (
    "C0_I",
    "C1_I",
    "C4_I",
    "S_AWQ_level",
    "S_AWQ_tail",
    "S_range",
)
FROZEN_EXECUTION_SETTINGS = {
    "weight_sample_size": 65536,
    "seed": 0,
    "torch_threads": 16,
    "torch_interop_threads": 1,
    "activation_stat_shard_count": 4,
}


def load_report_module():
    path = Path(__file__).resolve().parent / "report-importance-score-formal.py"
    spec = importlib.util.spec_from_file_location("importance_prejoin_report_helpers", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


REPORT = load_report_module()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
    ).strip()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def resolve_stats_file(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path.is_dir():
        path = path / "activation_second_moments.safetensors"
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--source-roster", type=Path, required=True)
    parser.add_argument("--source-roster-manifest", type=Path, required=True)
    parser.add_argument("--combined-stats", type=Path, required=True)
    parser.add_argument("--shard-stats", type=Path, action="append", required=True)
    parser.add_argument("--c0-jsonl", type=Path, required=True)
    parser.add_argument("--c1-jsonl", type=Path)
    parser.add_argument("--c4-jsonl", type=Path)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--score-registry", type=Path, required=True)
    parser.add_argument("--corpus-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--weight-sample-size", type=int, default=65536)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--torch-threads", type=int, default=16)
    parser.add_argument("--torch-interop-threads", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if len(args.shard_stats) != 4:
        raise SystemExit("prejoin score build requires exactly four activation-stat shards")
    if min(args.weight_sample_size, args.torch_threads, args.torch_interop_threads) < 1:
        raise SystemExit("sample and thread counts must be positive")
    execution_settings = {
        "weight_sample_size": args.weight_sample_size,
        "seed": args.seed,
        "torch_threads": args.torch_threads,
        "torch_interop_threads": args.torch_interop_threads,
        "activation_stat_shard_count": len(args.shard_stats),
    }
    if execution_settings != FROZEN_EXECUTION_SETTINGS:
        raise SystemExit(
            "lockbox prejoin execution settings differ from the frozen v0.1 contract: "
            f"{execution_settings} != {FROZEN_EXECUTION_SETTINGS}"
        )
    torch.set_num_threads(args.torch_threads)
    torch.set_num_interop_threads(args.torch_interop_threads)
    output_dir = args.output_dir.expanduser().resolve()
    scores_path = output_dir / "scores-prejoin.jsonl"
    shard_path = output_dir / "shard-scores-prejoin.json"
    receipt_path = output_dir / "scores-prejoin.receipt.json"
    existing_outputs = [path for path in (scores_path, shard_path, receipt_path) if path.exists()]
    if existing_outputs:
        raise SystemExit(f"refusing to overwrite sealed prejoin outputs: {existing_outputs}")
    paths = {
        "source_roster": args.source_roster.expanduser().resolve(),
        "source_roster_manifest": args.source_roster_manifest.expanduser().resolve(),
        "combined_stats": resolve_stats_file(args.combined_stats),
        "c0_jsonl": args.c0_jsonl.expanduser().resolve(),
        "candidate_manifest": args.candidate_manifest.expanduser().resolve(),
        "score_registry": args.score_registry.expanduser().resolve(),
        "corpus_manifest": args.corpus_manifest.expanduser().resolve(),
    }
    if args.c1_jsonl:
        paths["c1_jsonl"] = args.c1_jsonl.expanduser().resolve()
    if args.c4_jsonl:
        paths["c4_jsonl"] = args.c4_jsonl.expanduser().resolve()
    shard_paths = [resolve_stats_file(path) for path in args.shard_stats]
    for name, path in [*paths.items(), *[(f"shard_stats_{i}", p) for i, p in enumerate(shard_paths)]]:
        if not path.is_file():
            raise SystemExit(f"missing prejoin input {name}: {path}")

    roster = read_jsonl(paths["source_roster"])
    forbidden_present = sorted(
        {key for row in roster for key in FORBIDDEN_ROSTER_KEYS if key in row}
    )
    if forbidden_present:
        raise SystemExit(f"source roster contains forbidden lockbox label keys: {forbidden_present}")
    roster_manifest = json.loads(paths["source_roster_manifest"].read_text(encoding="utf-8"))
    if roster_manifest.get("status") != (
        "frozen without reading GGUF tensor names, types, ordinals, or promotion labels"
    ):
        raise SystemExit("source roster manifest does not assert label-blind generation")
    if roster_manifest.get("roster_sha256") != sha256_file(paths["source_roster"]):
        raise SystemExit("source roster hash does not match its manifest")
    candidate_manifest = json.loads(paths["candidate_manifest"].read_text(encoding="utf-8"))
    candidate_ids = {item["candidate_id"] for item in candidate_manifest["candidates"]}
    if candidate_ids != {REPORT.LOW, REPORT.HIGH}:
        raise SystemExit(f"unexpected frozen candidate set: {sorted(candidate_ids)}")

    model_dir = args.model_dir.expanduser().resolve()
    combined = REPORT.SCREEN.load_stats(paths["combined_stats"])
    shards = [REPORT.SCREEN.load_stats(path) for path in shard_paths]
    c0 = REPORT.parse_quantizer_rows(paths["c0_jsonl"])
    c1 = REPORT.parse_c1(paths.get("c1_jsonl"))
    c4 = REPORT.parse_perturbation(paths.get("c4_jsonl"), "c4")
    rows, per_shard = REPORT.score_features(
        roster,
        model_dir,
        combined,
        shards,
        c0,
        c1,
        c4,
        args.weight_sample_size,
        args.seed,
    )
    forbidden_output = sorted(
        {key for row in rows for key in FORBIDDEN_ROSTER_KEYS if key in row}
    )
    if forbidden_output:
        raise RuntimeError(f"prejoin score table leaked label fields: {forbidden_output}")
    if {row["hf_name"] for row in rows} != {row["hf_name"] for row in roster}:
        raise RuntimeError("prejoin score table tensor set differs from source roster")
    candidate_score_columns = [
        score for score in PRIMARY_SCORE_ORDER if all(score in row for row in rows)
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    with scores_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")
    write_json(shard_path, per_shard)
    tensor_names = sorted(row["hf_name"] for row in rows)
    tensor_set_sha = hashlib.sha256(
        ("\n".join(tensor_names) + "\n").encode("utf-8")
    ).hexdigest()
    tool_dir = Path(__file__).resolve().parent
    input_hashes = {name: sha256_file(path) for name, path in paths.items()}
    input_hashes.update(
        {
            f"shard_stats_{index}": sha256_file(path)
            for index, path in enumerate(shard_paths)
        }
    )
    receipt = {
        "schema_version": "importance-score-prejoin-receipt-v0.1",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "sealed score table generated without accepting or opening a GGUF label manifest",
        "model_id": rows[0]["model_id"],
        "architecture": rows[0]["architecture"],
        "score_table_path": str(scores_path),
        "score_table_sha256": sha256_file(scores_path),
        "shard_scores_path": str(shard_path),
        "shard_scores_sha256": sha256_file(shard_path),
        "tensor_count": len(rows),
        "tensor_name_set_sha256": tensor_set_sha,
        "candidate_score_columns": candidate_score_columns,
        "score_columns": sorted({
            key
            for row in rows
            for key in row
            if key.startswith("C") or key.startswith("S_")
        }),
        "forbidden_label_keys_verified_absent": sorted(FORBIDDEN_ROSTER_KEYS),
        "execution_settings": execution_settings,
        "input_hashes": input_hashes,
        "implementation_hashes": {
            name: sha256_file(tool_dir / name)
            for name in (
                "build-importance-score-prejoin.py",
                "report-importance-score-formal.py",
                "summarize-importance-score-screen.py",
                "run-aq-tensor-sample.py",
                "score-block-covariance-c1.py",
                "run-importance-single-tensor-perturbation.py",
            )
        },
        "workspace_git_head": git_revision(),
    }
    write_json(receipt_path, receipt)
    print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
