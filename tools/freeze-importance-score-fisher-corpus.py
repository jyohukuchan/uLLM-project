#!/usr/bin/env python3
"""Materialize the conditional D_fisher split without changing frozen splits."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import heapq
import importlib.util
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "importance-score-fisher-corpus-freeze-v0.1"
SELECTION_VERSION = "sha256-lowest-unused-record-v1"
SHARD_COUNT = 4
SEQUENCE_LENGTH = 128
DOMAIN_COUNTS = {
    "chat": 26,
    "code": 26,
    "multilingual_ja": 26,
    "reasoning_math": 25,
    "general": 25,
}


def load_base_tool():
    path = Path(__file__).resolve().parent / "freeze-importance-score-corpus.py"
    spec = importlib.util.spec_from_file_location("importance_score_base_corpus", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BASE = load_base_tool()


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
    ).strip()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def base_split_records(
    manifest: dict[str, Any], manifest_path: Path
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for split_name, split in sorted(manifest.get("splits", {}).items()):
        raw_path = Path(str(split.get("combined", {}).get("path", ""))).expanduser()
        path = raw_path.resolve() if raw_path.is_absolute() else (manifest_path.parent / raw_path).resolve()
        if not path.is_file():
            raise ValueError(f"base split is missing: {split_name}: {path}")
        expected = split.get("combined", {}).get("sha256")
        if expected != sha256_file(path):
            raise ValueError(f"base split hash mismatch: {split_name}: {path}")
        records.extend(read_jsonl(path))
    if not records:
        raise ValueError("base corpus manifest has no materialized split records")
    return records


def select_unused_domain_records(
    sources: list[dict[str, Any]],
    domain: str,
    count: int,
    excluded_record_ids: set[str],
    excluded_content_hashes: set[str],
) -> list[dict[str, Any]]:
    """Select the lowest-ranked normalized records after excluding frozen content."""

    heap: list[tuple[int, str, dict[str, Any]]] = []
    content_seen = set(excluded_content_hashes)
    eligible = 0
    for source in sources:
        if source["domain"] != domain:
            continue
        for record in BASE.iter_source_records(source):
            record_id = str(record["record_id"])
            content_sha = str(record["normalized_record_sha256"])
            if record_id in excluded_record_ids or content_sha in content_seen:
                continue
            content_seen.add(content_sha)
            eligible += 1
            rank = int.from_bytes(
                hashlib.sha256(
                    f"{SELECTION_VERSION}\0{record_id}\0{content_sha}".encode("utf-8")
                ).digest(),
                "big",
            )
            item = (-rank, record_id, record)
            if len(heap) < count:
                heapq.heappush(heap, item)
            elif rank < -heap[0][0]:
                heapq.heapreplace(heap, item)
    if len(heap) != count:
        raise RuntimeError(
            f"domain {domain}: requested {count} unused unique records, found {eligible}"
        )
    selected = [dict(item[2], split="D_fisher") for item in heap]
    selected.sort(key=lambda record: str(record["record_id"]))
    return selected


def tokenizer_counts(
    tokenizer_specs: list[str], records: list[dict[str, Any]]
) -> dict[str, Any]:
    if not tokenizer_specs:
        return {}
    from transformers import AutoTokenizer

    result: dict[str, Any] = {}
    for spec in tokenizer_specs:
        model_id, raw_path = spec.split("=", 1)
        path = Path(raw_path).expanduser().resolve()
        tokenizer = AutoTokenizer.from_pretrained(
            path, local_files_only=True, trust_remote_code=True
        )
        domain_tokens: Counter[str] = Counter()
        valid_tokens = 0
        for record in records:
            text = BASE.render_for_tokenizer(tokenizer, record)
            count = min(
                SEQUENCE_LENGTH,
                len(tokenizer.encode(text, add_special_tokens=True)),
            )
            valid_tokens += count
            domain_tokens[str(record["domain"])] += count
        result[model_id] = {
            "tokenizer_path": str(path),
            "tokenizer_config_sha256": sha256_file(path / "tokenizer_config.json"),
            "tokenizer_json_sha256": sha256_file(path / "tokenizer.json"),
            "chat_template_sha256": (
                sha256_file(path / "chat_template.jinja")
                if (path / "chat_template.jinja").is_file()
                else None
            ),
            "D_fisher": {
                "sequence_length": SEQUENCE_LENGTH,
                "valid_tokens": valid_tokens,
                "domain_valid_tokens": dict(sorted(domain_tokens.items())),
            },
        }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-corpus-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--tokenizer",
        action="append",
        default=[],
        metavar="MODEL_ID=PATH",
        help="Measure model-specific valid-token counts; may be repeated.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base_manifest_path = args.base_corpus_manifest.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    outputs = [
        output_dir / "D_fisher.jsonl",
        *(output_dir / f"D_fisher-shard-{index:02d}.jsonl" for index in range(SHARD_COUNT)),
        output_dir / "D_fisher-manifest.json",
    ]
    existing = [path for path in outputs if path.exists()]
    if existing:
        raise SystemExit(f"refusing to overwrite frozen D_fisher outputs: {existing}")
    if not base_manifest_path.is_file():
        raise SystemExit(f"base corpus manifest is missing: {base_manifest_path}")
    base_manifest = json.loads(base_manifest_path.read_text(encoding="utf-8"))
    if base_manifest.get("schema_version") != "importance-score-corpus-freeze-v0.1":
        raise SystemExit("base corpus manifest has an unexpected schema")
    base_records = base_split_records(base_manifest, base_manifest_path)
    excluded_record_ids = {str(record["record_id"]) for record in base_records}
    excluded_content_hashes = {
        str(record["normalized_record_sha256"]) for record in base_records
    }
    if len(excluded_record_ids) != len(base_records):
        raise SystemExit("base corpus contains duplicate record IDs")

    sources = [dict(source) for source in base_manifest.get("sources", [])]
    if not sources:
        raise SystemExit("base corpus manifest contains no sources")
    for source in sources:
        path = Path(str(source["path"])).expanduser().resolve()
        if not path.is_file():
            raise SystemExit(f"missing frozen raw source: {path}")
        actual = sha256_file(path)
        expected = source.get("actual_file_sha256") or source.get("lfs_sha256")
        if expected != actual:
            raise SystemExit(
                f"frozen raw source hash mismatch for {path}: expected {expected}, got {actual}"
            )
        source["path"] = str(path)
        source["actual_file_sha256"] = actual

    selected: list[dict[str, Any]] = []
    for domain, count in DOMAIN_COUNTS.items():
        selected.extend(
            select_unused_domain_records(
                sources,
                domain,
                count,
                excluded_record_ids,
                excluded_content_hashes,
            )
        )
    if len({str(record["record_id"]) for record in selected}) != len(selected):
        raise RuntimeError("D_fisher selection contains duplicate record IDs")
    if excluded_record_ids.intersection(str(record["record_id"]) for record in selected):
        raise RuntimeError("D_fisher record leakage into a frozen base split")
    if excluded_content_hashes.intersection(
        str(record["normalized_record_sha256"]) for record in selected
    ):
        raise RuntimeError("D_fisher normalized-content leakage into a frozen base split")

    shards = BASE.assign_shards(selected)
    combined = [record for shard in shards for record in shard]
    combined.sort(key=lambda record: (record["shard"], record["domain"], record["record_id"]))
    output_dir.mkdir(parents=True, exist_ok=False)
    combined_artifact = BASE.write_jsonl(output_dir / "D_fisher.jsonl", combined)
    shard_artifacts = [
        BASE.write_jsonl(output_dir / f"D_fisher-shard-{index:02d}.jsonl", shard)
        for index, shard in enumerate(shards)
    ]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": utc_now(),
        "status": "sealed additive C5 corpus; existing frozen splits remain byte-for-byte unchanged",
        "workspace_git_head": git_revision(),
        "implementation_hashes": {
            "freeze-importance-score-fisher-corpus.py": sha256_file(
                Path(__file__).resolve()
            ),
            "freeze-importance-score-corpus.py": sha256_file(
                Path(__file__).resolve().parent / "freeze-importance-score-corpus.py"
            ),
        },
        "source_plan": "docs/plans/importance-score-algorithm-selection-plan-v0.1.md",
        "base_corpus_manifest": {
            "path": str(base_manifest_path),
            "sha256": sha256_file(base_manifest_path),
            "schema_version": base_manifest["schema_version"],
        },
        "selection": {
            "version": SELECTION_VERSION,
            "rule": (
                "select lowest SHA-256 ranks per domain after excluding every frozen base-split "
                "record ID and normalized-content hash; score and UD labels are never inputs"
            ),
            "domain_counts": DOMAIN_COUNTS,
            "shard_count": SHARD_COUNT,
            "base_record_overlap": 0,
            "base_normalized_content_overlap": 0,
            "base_record_count_excluded": len(base_records),
        },
        "D_fisher": {
            "purpose": "C5 Taylor and Fisher gradients",
            "combined": combined_artifact,
            "shards": shard_artifacts,
        },
        "tokenization_contract": {
            "shared_raw_examples": True,
            "chat": "model tokenizer official apply_chat_template(add_generation_prompt=False)",
            "non_chat": "verbatim normalized text with model tokenizer special tokens",
            "truncation": "right truncation",
            "padding": "one unpadded record per gradient sample; attention_mask defines valid next-token positions",
            "sequence_length": SEQUENCE_LENGTH,
            "sample_weighting": "equal weight per raw record after within-record valid-next-token mean loss",
        },
        "model_token_counts": tokenizer_counts(args.tokenizer, combined),
        "sources": sources,
    }
    manifest_path = output_dir / "D_fisher-manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"manifest": str(manifest_path), "sha256": sha256_file(manifest_path)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
