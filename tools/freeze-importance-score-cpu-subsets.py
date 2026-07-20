#!/usr/bin/env python3
"""Freeze label/score-independent CPU subsets for C4 and C6."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "records": len(rows),
        "domains": dict(sorted(Counter(str(row["domain"]) for row in rows).items())),
        "source_shards": dict(sorted(Counter(str(row["shard"]) for row in rows).items())),
    }


def write_shard_jsonl_files(
    output_dir: Path, split: str, rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Write exact, non-overlapping views of an already-selected CPU subset."""
    result = []
    for shard in range(4):
        shard_rows = [row for row in rows if int(row["shard"]) == shard]
        if not shard_rows:
            raise RuntimeError(f"{split}: selected CPU subset has no rows for shard {shard}")
        result.append(
            {
                "shard": shard,
                **write_jsonl(
                    output_dir / f"{split}-cpu-subset-shard-{shard:02d}.jsonl",
                    shard_rows,
                ),
            }
        )
    if sum(int(item["records"]) for item in result) != len(rows):
        raise RuntimeError(f"{split}: shard views do not partition the selected CPU subset")
    return result


def select_records(corpus_root: Path, split: str, per_shard: int) -> list[dict[str, Any]]:
    selected = []
    for shard in range(4):
        path = corpus_root / f"{split}-shard-{shard:02d}.jsonl"
        rows = read_jsonl(path)
        rows.sort(
            key=lambda row: hashlib.sha256(
                f"importance-score-cpu-subset-v1\0{split}\0{row['domain']}\0{row['record_id']}".encode()
            ).digest()
        )
        # Round-robin domains after deterministic within-domain ordering keeps
        # the tiny CPU subset as broad as its per-shard count permits.
        by_domain: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_domain[str(row["domain"])].append(row)
        domain_order = sorted(by_domain)
        rotation = (shard * per_shard) % len(domain_order)
        domain_order = domain_order[rotation:] + domain_order[:rotation]
        shard_rows = []
        depth = 0
        while len(shard_rows) < per_shard:
            progressed = False
            for domain in domain_order:
                if depth < len(by_domain[domain]):
                    shard_rows.append(by_domain[domain][depth])
                    progressed = True
                    if len(shard_rows) >= per_shard:
                        break
            if not progressed:
                break
            depth += 1
        if len(shard_rows) != per_shard:
            raise RuntimeError(f"{split} shard {shard}: insufficient records")
        selected.extend(shard_rows)
    selected.sort(key=lambda row: (int(row["shard"]), str(row["domain"]), str(row["record_id"])))
    return selected


def select_kl_core(labels_path: Path, rate: float) -> list[dict[str, Any]]:
    with labels_path.open(encoding="utf-8", newline="") as handle:
        eligible = [row for row in csv.DictReader(handle, delimiter="\t") if row["eligible"] == "true"]
    by_family: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in eligible:
        by_family[row["canonical_family"]].append(row)
    selected = []
    for family, rows in sorted(by_family.items()):
        count = max(1, math.ceil(rate * len(rows)))
        rows.sort(
            key=lambda row: hashlib.sha256(
                (
                    "importance-score-kl-core-v1\0"
                    + family
                    + "\0"
                    + row["hf_name"]
                    + "\0"
                    + row["layer_id"]
                    + "\0"
                    + row["shape"]
                ).encode()
            ).digest()
        )
        for row in rows[:count]:
            selected.append(
                {
                    "model_id": row["model_id"],
                    "hf_name": row["hf_name"],
                    "gguf_name": row["gguf_name"],
                    "canonical_family": family,
                    "layer_id": int(row["layer_id"]),
                    "shape": json.loads(row["shape"]),
                    "selection_inputs": [
                        "model_id",
                        "hf_name",
                        "canonical_family",
                        "layer_id",
                        "shape",
                    ],
                }
            )
    selected.sort(key=lambda row: (row["canonical_family"], row["layer_id"], row["hf_name"]))
    return selected


def select_kl_core_from_source_roster(roster_path: Path, rate: float) -> list[dict[str, Any]]:
    rows = read_jsonl(roster_path)
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_family[str(row["canonical_family"])].append(row)
    selected = []
    for family, members in sorted(by_family.items()):
        count = max(1, math.ceil(rate * len(members)))
        members.sort(
            key=lambda row: hashlib.sha256(
                (
                    "importance-score-kl-core-v1\0"
                    + family
                    + "\0"
                    + str(row["hf_name"])
                    + "\0"
                    + str(row["layer_id"])
                    + "\0"
                    + canonical_json(row["shape"])
                ).encode()
            ).digest()
        )
        for row in members[:count]:
            selected.append(
                {
                    "model_id": row["model_id"],
                    "hf_name": row["hf_name"],
                    "canonical_family": family,
                    "layer_id": int(row["layer_id"]),
                    "shape": row["shape"],
                    "selection_inputs": [
                        "model_id",
                        "hf_name",
                        "canonical_family",
                        "layer_id",
                        "shape",
                    ],
                }
            )
    selected.sort(key=lambda row: (row["canonical_family"], row["layer_id"], row["hf_name"]))
    return selected


def tokenizer_counts(specs: list[str], splits: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    if not specs:
        return {}
    from transformers import AutoTokenizer

    result = {}
    for spec in specs:
        model_id, raw_path = spec.split("=", 1)
        tokenizer = AutoTokenizer.from_pretrained(
            Path(raw_path).expanduser().resolve(), local_files_only=True, trust_remote_code=True
        )
        model_counts = {}
        for split, rows in splits.items():
            tokens = 0
            domains: Counter[str] = Counter()
            for row in rows:
                text = (
                    tokenizer.apply_chat_template(
                        row["messages"], tokenize=False, add_generation_prompt=False
                    )
                    if "messages" in row
                    else row["text"]
                )
                count = min(128, len(tokenizer.encode(text, add_special_tokens=True)))
                tokens += count
                domains[str(row["domain"])] += count
            model_counts[split] = {
                "sequence_length": 128,
                "valid_tokens": tokens,
                "domain_valid_tokens": dict(sorted(domains.items())),
            }
        result[model_id] = model_counts
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-root", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--labels", type=Path)
    source.add_argument("--source-roster", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--c4-per-shard", type=int, default=4)
    parser.add_argument("--c6-per-shard", type=int, default=2)
    parser.add_argument("--kl-core-rate", type=float, default=0.10)
    parser.add_argument("--tokenizer", action="append", default=[], metavar="MODEL_ID=PATH")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.c4_per_shard < 1 or args.c6_per_shard < 1:
        raise SystemExit("per-shard counts must be positive")
    if not 0.10 <= args.kl_core_rate <= 0.15:
        raise SystemExit("KL-core requested rate must remain within the frozen 10-15% interval")
    corpus_root = args.corpus_root.expanduser().resolve()
    labels_path = args.labels.expanduser().resolve() if args.labels else None
    source_roster = args.source_roster.expanduser().resolve() if args.source_roster else None
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    c4_rows = select_records(corpus_root, "D_block", args.c4_per_shard)
    c6_rows = select_records(corpus_root, "D_KL", args.c6_per_shard)
    kl_core = (
        select_kl_core(labels_path, args.kl_core_rate)
        if labels_path is not None
        else select_kl_core_from_source_roster(source_roster, args.kl_core_rate)
    )
    c4 = write_jsonl(output_dir / "D_block-cpu-subset.jsonl", c4_rows)
    c6 = write_jsonl(output_dir / "D_KL-cpu-subset.jsonl", c6_rows)
    c4_shards = write_shard_jsonl_files(output_dir, "D_block", c4_rows)
    c6_shards = write_shard_jsonl_files(output_dir, "D_KL", c6_rows)
    kl_path = output_dir / "KL-core.json"
    kl_path.write_text(canonical_json(kl_core) + "\n", encoding="utf-8")
    family_counts = Counter(row["canonical_family"] for row in kl_core)
    if labels_path is not None:
        with labels_path.open(encoding="utf-8", newline="") as handle:
            eligible_count = sum(
                row["eligible"] == "true" for row in csv.DictReader(handle, delimiter="\t")
            )
        kl_selection_source = {
            "kind": "paired-label manifest with qtype/promotion fields excluded from selection",
            "path": str(labels_path),
            "sha256": sha256_file(labels_path),
        }
    else:
        roster_rows = read_jsonl(source_roster)
        eligible_count = len(roster_rows)
        kl_selection_source = {
            "kind": "BF16 source-only roster; no GGUF tensor label was opened",
            "path": str(source_roster),
            "sha256": sha256_file(source_roster),
        }
    manifest = {
        "schema_version": "importance-score-cpu-subsets-v0.1",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "frozen before C4/C6 execution; score and UD type values were not selection inputs",
        "selection_version": "importance-score-cpu-subset-v1 / importance-score-kl-core-v1",
        "C4": {
            **c4,
            "shard_files": c4_shards,
            "full_split_records": 128,
            "record_fraction": len(c4_rows) / 128,
            "reason": "CPU-only block perturbation feasibility cap; four records per fixed shard",
        },
        "C6": {
            **c6,
            "shard_files": c6_shards,
            "full_split_records": 64,
            "record_fraction": len(c6_rows) / 64,
            "reason": "CPU-only full-vocabulary single-tensor KL feasibility cap; two records per fixed shard",
        },
        "KL_core": {
            "path": str(kl_path),
            "sha256": sha256_file(kl_path),
            "requested_rate_per_family": args.kl_core_rate,
            "selected_tensor_count": len(kl_core),
            "eligible_tensor_count": eligible_count,
            "actual_fraction": len(kl_core) / max(1, eligible_count),
            "family_counts": dict(sorted(family_counts.items())),
            "selection_source": kl_selection_source,
            "forbidden_selection_inputs": [
                "qtype_ud",
                "ordinal_ud",
                "qtype_static",
                "promotion_delta_ordinal",
                "promoted",
                "all score values",
            ],
        },
        "model_token_counts": tokenizer_counts(
            args.tokenizer, {"D_block_cpu": c4_rows, "D_KL_cpu": c6_rows}
        ),
    }
    (output_dir / "subset-manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
