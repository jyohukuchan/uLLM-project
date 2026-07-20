#!/usr/bin/env python3
"""Freeze a deterministic mixed-domain raw corpus for importance-score runs."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import heapq
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import pyarrow.parquet as pq


SCHEMA_VERSION = "importance-score-corpus-freeze-v0.1"
SELECTION_VERSION = "sha256-lowest-record-v1"
SHARD_COUNT = 4
SPLIT_DOMAIN_COUNTS = {
    "D_stats": {
        "chat": 480,
        "code": 480,
        "multilingual_ja": 480,
        "reasoning_math": 480,
        "general": 480,
    },
    "D_block": {
        "chat": 26,
        "code": 26,
        "multilingual_ja": 26,
        "reasoning_math": 25,
        "general": 25,
    },
    "D_KL": {
        "chat": 13,
        "code": 13,
        "multilingual_ja": 13,
        "reasoning_math": 13,
        "general": 12,
    },
}
SEQUENCE_LENGTHS = {"D_stats": 128, "D_block": 128, "D_KL": 128}


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clean_text(value: Any, *, limit: int = 8192) -> str:
    text = str(value or "").replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    return text.strip()[:limit]


def adapter_columns(adapter: str) -> list[str]:
    return {
        "ultrachat_messages": ["prompt_id", "messages"],
        "mbpp_code": ["task_id", "text", "code", "test_list", "test_setup_code"],
        "gsm8k_math": ["question", "answer"],
        "jparacrawl_en_ja": ["translation"],
        "fineweb_general": ["id", "text", "url", "language"],
    }[adapter]


def normalize_row(source: dict[str, Any], row_index: int, row: dict[str, Any]) -> dict[str, Any] | None:
    adapter = source["adapter"]
    source_id = source["source_id"]
    source_key = str(row.get("prompt_id") or row.get("task_id") or row.get("id") or row_index)
    base: dict[str, Any] = {
        "record_id": f"{source_id}:{source_key}",
        "domain": source["domain"],
        "source": {
            "source_id": source_id,
            "filename": source["filename"],
            "row_index": row_index,
            "source_key": source_key,
        },
    }

    if adapter == "ultrachat_messages":
        raw_messages = row.get("messages") or []
        messages = []
        for message in raw_messages[-2:]:
            role = str(message.get("role", ""))
            content = clean_text(message.get("content"), limit=4096)
            if role in {"user", "assistant"} and content:
                messages.append({"role": role, "content": content})
        if len(messages) != 2 or messages[0]["role"] != "user" or messages[1]["role"] != "assistant":
            return None
        base["messages"] = messages
    elif adapter == "mbpp_code":
        problem = clean_text(row.get("text"), limit=2048)
        code = clean_text(row.get("code"), limit=4096)
        tests = "\n".join(clean_text(item, limit=1024) for item in (row.get("test_list") or []))
        setup = clean_text(row.get("test_setup_code"), limit=1024)
        base["text"] = (
            f"Python programming problem:\n{problem}\n\nSolution:\n```python\n{setup}\n{code}\n```\n\nTests:\n{tests}"
        ).strip()
    elif adapter == "gsm8k_math":
        question = clean_text(row.get("question"), limit=4096)
        answer = clean_text(row.get("answer"), limit=4096)
        base["text"] = f"Math problem:\n{question}\n\nReasoning and answer:\n{answer}"
    elif adapter == "jparacrawl_en_ja":
        translation = row.get("translation") or {}
        english = clean_text(translation.get("en"), limit=4096)
        japanese = clean_text(translation.get("ja"), limit=4096)
        base["text"] = f"English:\n{english}\n\n日本語:\n{japanese}"
    elif adapter == "fineweb_general":
        text = clean_text(row.get("text"), limit=8192)
        if len(text) < 512:
            return None
        base["text"] = text
        base["source"]["url"] = clean_text(row.get("url"), limit=2048)
        base["source"]["language"] = str(row.get("language") or "")
    else:
        raise ValueError(f"unknown adapter: {adapter}")

    content_value = base.get("messages", base.get("text"))
    if len(canonical_json(content_value)) < 64:
        return None
    source_record_sha = sha256_bytes(canonical_json(row).encode("utf-8"))
    base["source_record_sha256"] = source_record_sha
    base["normalized_record_sha256"] = sha256_bytes(canonical_json(base).encode("utf-8"))
    return base


def iter_source_records(source: dict[str, Any]) -> Iterable[dict[str, Any]]:
    path = Path(source["path"])
    parquet = pq.ParquetFile(path)
    columns = adapter_columns(source["adapter"])
    row_index = 0
    for row_group in range(parquet.num_row_groups):
        table = parquet.read_row_group(row_group, columns=columns)
        for row in table.to_pylist():
            normalized = normalize_row(source, row_index, row)
            row_index += 1
            if normalized is not None:
                yield normalized


def select_domain_records(
    sources: list[dict[str, Any]],
    domain: str,
    count: int,
) -> list[dict[str, Any]]:
    heap: list[tuple[int, str, dict[str, Any]]] = []
    content_seen: set[str] = set()
    eligible = 0
    for source in sources:
        if source["domain"] != domain:
            continue
        for record in iter_source_records(source):
            content_sha = record["normalized_record_sha256"]
            if content_sha in content_seen:
                continue
            content_seen.add(content_sha)
            eligible += 1
            rank = int.from_bytes(
                hashlib.sha256(
                    f"{SELECTION_VERSION}\0{record['record_id']}\0{content_sha}".encode("utf-8")
                ).digest(),
                "big",
            )
            item = (-rank, record["record_id"], record)
            if len(heap) < count:
                heapq.heappush(heap, item)
            elif rank < -heap[0][0]:
                heapq.heapreplace(heap, item)
    if len(heap) != count:
        raise RuntimeError(f"domain {domain}: requested {count} unique records, found {eligible}")
    selected = [item[2] for item in heap]
    selected.sort(key=lambda record: record["record_id"])
    return selected


def assign_splits(records_by_domain: dict[str, list[dict[str, Any]]]) -> dict[str, list[dict[str, Any]]]:
    splits: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for domain, records in records_by_domain.items():
        records = sorted(
            records,
            key=lambda record: hashlib.sha256(
                f"importance-score-split-v1\0{record['record_id']}".encode("utf-8")
            ).digest(),
        )
        cursor = 0
        for split_name in ("D_stats", "D_block", "D_KL"):
            count = SPLIT_DOMAIN_COUNTS[split_name][domain]
            chunk = records[cursor : cursor + count]
            cursor += count
            for record in chunk:
                record = dict(record)
                record["split"] = split_name
                splits[split_name].append(record)
        if cursor != len(records):
            raise AssertionError(f"domain {domain}: split assignment left records unassigned")
    return splits


def assign_shards(records: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    shards: list[list[dict[str, Any]]] = [[] for _ in range(SHARD_COUNT)]
    by_domain: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_domain[record["domain"]].append(record)
    for domain_records in by_domain.values():
        domain_records.sort(
            key=lambda record: hashlib.sha256(
                f"importance-score-shard-v1\0{record['record_id']}".encode("utf-8")
            ).digest()
        )
        for index, record in enumerate(domain_records):
            record = dict(record)
            record["shard"] = index % SHARD_COUNT
            shards[index % SHARD_COUNT].append(record)
    for shard in shards:
        shard.sort(key=lambda record: (record["domain"], record["record_id"]))
    return shards


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> dict[str, Any]:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(canonical_json(record))
            handle.write("\n")
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "records": len(records),
        "domains": dict(sorted(Counter(record["domain"] for record in records).items())),
        "record_ids_sha256": sha256_bytes(
            "".join(f"{record['record_id']}\n" for record in records).encode("utf-8")
        ),
    }


def render_for_tokenizer(tokenizer, record: dict[str, Any]) -> str:
    if "messages" in record:
        return tokenizer.apply_chat_template(
            record["messages"], tokenize=False, add_generation_prompt=False
        )
    return record["text"]


def tokenizer_counts(
    tokenizer_specs: list[str],
    splits: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    if not tokenizer_specs:
        return {}
    from transformers import AutoTokenizer

    result = {}
    for spec in tokenizer_specs:
        model_id, raw_path = spec.split("=", 1)
        path = Path(raw_path).expanduser().resolve()
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=True)
        per_split = {}
        for split_name, records in splits.items():
            sequence_length = SEQUENCE_LENGTHS[split_name]
            domain_tokens: Counter[str] = Counter()
            token_count = 0
            for record in records:
                text = render_for_tokenizer(tokenizer, record)
                count = min(sequence_length, len(tokenizer.encode(text, add_special_tokens=True)))
                token_count += count
                domain_tokens[record["domain"]] += count
            per_split[split_name] = {
                "sequence_length": sequence_length,
                "valid_tokens": token_count,
                "domain_valid_tokens": dict(sorted(domain_tokens.items())),
            }
        result[model_id] = {
            "tokenizer_path": str(path),
            "tokenizer_config_sha256": sha256_file(path / "tokenizer_config.json"),
            "tokenizer_json_sha256": sha256_file(path / "tokenizer.json"),
            "chat_template_sha256": (
                sha256_file(path / "chat_template.jinja")
                if (path / "chat_template.jinja").is_file()
                else None
            ),
            "splits": per_split,
        }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--tokenizer",
        action="append",
        default=[],
        metavar="MODEL_ID=PATH",
        help="Measure model-specific post-template token counts; may be repeated.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_manifest_path = args.source_manifest.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    sources = source_manifest["sources"]

    for source in sources:
        path = Path(source["path"])
        if not path.is_file():
            raise SystemExit(f"missing source file: {path}")
        actual_sha = sha256_file(path)
        source["actual_file_sha256"] = actual_sha
        expected_sha = source.get("lfs_sha256")
        if expected_sha and actual_sha != expected_sha:
            raise SystemExit(
                f"source SHA mismatch for {path}: expected {expected_sha}, got {actual_sha}"
            )
        source["size_bytes"] = path.stat().st_size

    domains = list(SPLIT_DOMAIN_COUNTS["D_stats"])
    records_by_domain = {}
    for domain in domains:
        needed = sum(SPLIT_DOMAIN_COUNTS[split][domain] for split in SPLIT_DOMAIN_COUNTS)
        records_by_domain[domain] = select_domain_records(sources, domain, needed)
    splits = assign_splits(records_by_domain)

    split_artifacts = {}
    materialized_splits: dict[str, list[dict[str, Any]]] = {}
    for split_name, records in splits.items():
        shards = assign_shards(records)
        combined = [record for shard in shards for record in shard]
        combined.sort(key=lambda record: (record["shard"], record["domain"], record["record_id"]))
        materialized_splits[split_name] = combined
        split_artifacts[split_name] = {
            "combined": write_jsonl(output_dir / f"{split_name}.jsonl", combined),
            "shards": [
                write_jsonl(output_dir / f"{split_name}-shard-{index:02d}.jsonl", shard)
                for index, shard in enumerate(shards)
            ],
            "selection_counts": SPLIT_DOMAIN_COUNTS[split_name],
            "purpose": {
                "D_stats": "activation, covariance, and range statistics",
                "D_block": "block-output perturbation; preselected from the same raw-source pool without score/label input",
                "D_KL": "unseen single-tensor KL holdout; selected without score/UD input",
            }[split_name],
        }

    all_ids = [record["record_id"] for records in materialized_splits.values() for record in records]
    if len(all_ids) != len(set(all_ids)):
        raise RuntimeError("record leakage detected across splits")

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": utc_now(),
        "status": "formal raw corpus frozen; replaces but does not rewrite the 32-prompt CPU pilot",
        "source_plan": "docs/plans/importance-score-algorithm-selection-plan-v0.1.md",
        "source_manifest": {
            "path": str(source_manifest_path),
            "sha256_before_runtime_file_audit": sha256_file(source_manifest_path),
        },
        "selection": {
            "version": SELECTION_VERSION,
            "rule": "keep the lowest SHA-256 ranks per domain, then independent SHA-256 split and stratified four-shard assignments; score and UD labels are never inputs",
            "split_domain_counts": SPLIT_DOMAIN_COUNTS,
            "shard_count": SHARD_COUNT,
            "raw_record_normalization": "UTF-8/LF/NUL removal; bounded adapter-specific text fields; exact normalized records are stored in JSONL",
            "cross_split_record_overlap": 0,
        },
        "sources": sources,
        "splits": split_artifacts,
        "tokenization_contract": {
            "shared_raw_examples": True,
            "chat": "model tokenizer official apply_chat_template(add_generation_prompt=False)",
            "non_chat": "verbatim normalized text with model tokenizer special tokens",
            "truncation": "right truncation to the split sequence length",
            "padding": "one unpadded record per forward; attention_mask defines valid tokens",
            "sequence_lengths": SEQUENCE_LENGTHS,
        },
        "model_token_counts": tokenizer_counts(args.tokenizer, materialized_splits),
        "pilot_replacement": {
            "replaced_artifact": "32 prompt / 3,416-token provisional CPU pilot",
            "rule": "pilot results remain historical provisional evidence and are not pooled with this formal corpus",
        },
        "known_adjustments": {
            "D_stats": "2,400 balanced raw records target approximately 256k valid tokens/model after 128-token truncation; exact model counts are recorded",
            "D_block": "128 records cap at 16,384 tokens/model, matching the planned 16k scale",
            "D_KL": "64 records cap at 8,192 tokens/model, matching the planned 8k scale; CPU execution may use a smaller preregistered subset with the reduction reported",
            "D_fisher": "not materialized because C5 is conditional on the frozen Fisher escalation rule",
            "D_final": "not materialized until a candidate passes Phase 4/5 admission; algorithm selection cannot access it",
        },
    }
    manifest_path = output_dir / "corpus-manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
