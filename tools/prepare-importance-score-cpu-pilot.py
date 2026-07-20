#!/usr/bin/env python3
"""Freeze four deterministic CPU-pilot D_stats shards from a local prompt file."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shards", type=int, default=4)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.shards != 4:
        raise SystemExit("the frozen v0.1 CPU pilot uses exactly four shards")
    source = args.source.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    raw = source.read_bytes()
    prompts = [line.strip() for line in raw.decode("utf-8").splitlines() if line.strip()]
    if len(prompts) < args.shards:
        raise SystemExit("source has fewer prompts than shards")
    output_dir.mkdir(parents=True, exist_ok=True)
    assignments: list[list[tuple[int, str]]] = [[] for _ in range(args.shards)]
    for source_line, prompt in enumerate(prompts):
        # The source has English followed by Japanese prompts.  Round-robin
        # assignment gives each shard the same 4/4 language split without
        # changing or sampling raw examples.
        assignments[source_line % args.shards].append((source_line, prompt))
    records = []
    for shard_id, shard in enumerate(assignments):
        text = "\n".join(prompt for _, prompt in shard) + "\n"
        path = output_dir / f"D_stats-cpu-pilot-shard-{shard_id:02d}.txt"
        write_text(path, text)
        records.append(
            {
                "shard_id": shard_id,
                "path": str(path),
                "sha256": sha256_bytes(text.encode("utf-8")),
                "raw_source_line_indices": [line for line, _ in shard],
                "example_count": len(shard),
                "stratification": "round_robin source order; four English and four Japanese source prompts per shard"
            }
        )
    selection = {
        "schema_version": "importance-score-cpu-pilot-selection-v0.1",
        "status": "provisional CPU screen; not the formal 256k-token mixed-domain D_stats",
        "source": {
            "path": str(source),
            "sha256": sha256_bytes(raw),
            "nonempty_line_count": len(prompts)
        },
        "selection_rule": "No replacement sampling. Assign nonempty source lines by source_line_index modulo 4.",
        "shards": records,
        "planned_sequence_length": 128,
        "repeat_to_length": True,
        "token_count": "pending tokenizer measurement from collector metadata"
    }
    selection_path = output_dir / "cpu-pilot-corpus-selection.json"
    selection_path.write_text(json.dumps(selection, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(selection_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
