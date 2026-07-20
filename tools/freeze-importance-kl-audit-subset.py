#!/usr/bin/env python3
"""Freeze a bounded descriptive KL-audit union after score/label reporting."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


DEFAULT_SCORES = (
    "C0_I",
    "C1_I",
    "C4_I",
    "S_AWQ_level",
    "S_AWQ_tail",
    "S_range",
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def select_audit_rows(
    scores: list[dict[str, Any]],
    disagreements: list[dict[str, str]],
    score_columns: list[str],
    per_score_extreme: int,
    max_tensors: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_name = {str(row["hf_name"]): row for row in scores}
    if len(by_name) != len(scores):
        raise ValueError("duplicate tensor in prejoin scores")
    reasons: dict[str, set[str]] = defaultdict(set)
    mandatory: set[str] = set()
    for score in score_columns:
        if not all(score in row for row in scores):
            raise ValueError(f"score column is not complete: {score}")
        ordered = sorted(
            scores,
            key=lambda row: (
                float(row[score]),
                hashlib.sha256(str(row["hf_name"]).encode()).digest(),
            ),
        )
        for rank, row in enumerate(ordered[:per_score_extreme], start=1):
            reasons[str(row["hf_name"])].add(f"{score}:bottom:{rank}")
        for rank, row in enumerate(reversed(ordered[-per_score_extreme:]), start=1):
            reasons[str(row["hf_name"])].add(f"{score}:top:{rank}")
        mandatory.add(str(ordered[0]["hf_name"]))
        mandatory.add(str(ordered[-1]["hf_name"]))
    disagreement_missing = []
    for row in disagreements:
        name = str(row.get("hf_name", ""))
        if name not in by_name:
            disagreement_missing.append(name)
            continue
        reasons[name].add(f"disagreement:{row.get('score_id')}:{row.get('notes')}")
    if disagreement_missing:
        raise ValueError(f"disagreement tensors absent from prejoin scores: {disagreement_missing}")
    if len(mandatory) > max_tensors:
        raise ValueError(
            f"max_tensors={max_tensors} cannot preserve one top and bottom for every score "
            f"({len(mandatory)} unique mandatory tensors)"
        )
    selected_names = sorted(mandatory)
    remaining = sorted(
        set(reasons) - mandatory,
        key=lambda name: hashlib.sha256(
            (
                "importance-score-kl-audit-cap-v1\0"
                + str(by_name[name]["canonical_family"])
                + "\0"
                + str(by_name[name]["layer_id"])
                + "\0"
                + name
                + "\0"
                + "|".join(sorted(reasons[name]))
            ).encode()
        ).digest(),
    )
    selected_names.extend(remaining[: max_tensors - len(selected_names)])
    selected = []
    for name in selected_names:
        row = by_name[name]
        selected.append(
            {
                "model_id": row["model_id"],
                "hf_name": name,
                "canonical_family": row["canonical_family"],
                "layer_id": int(row["layer_id"]),
                "shape": row["shape"],
                "audit_reasons": sorted(reasons[name]),
                "selection_inputs": [
                    "frozen score ranks",
                    "post-join disagreement membership",
                    "hf_name",
                    "canonical_family",
                    "layer_id",
                    "shape",
                ],
            }
        )
    selected.sort(key=lambda row: (row["canonical_family"], row["layer_id"], row["hf_name"]))
    audit = {
        "union_tensor_count_before_cap": len(reasons),
        "mandatory_unique_top1_bottom1_count": len(mandatory),
        "selected_tensor_count": len(selected),
        "omitted_by_cpu_cap_count": len(reasons) - len(selected),
    }
    return selected, audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores-prejoin", type=Path, required=True)
    parser.add_argument("--scores-prejoin-receipt", type=Path, required=True)
    parser.add_argument("--disagreements", type=Path, required=True)
    parser.add_argument("--score", action="append", default=[])
    parser.add_argument("--per-score-extreme", type=int, default=2)
    parser.add_argument("--max-tensors", type=int, default=12)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.per_score_extreme < 1 or args.max_tensors < 1:
        raise SystemExit("extreme and tensor counts must be positive")
    scores_path = args.scores_prejoin.expanduser().resolve()
    receipt_path = args.scores_prejoin_receipt.expanduser().resolve()
    disagreements_path = args.disagreements.expanduser().resolve()
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("score_table_sha256") != sha256_file(scores_path):
        raise SystemExit("prejoin score hash differs from its sealed receipt")
    score_columns = args.score or list(DEFAULT_SCORES)
    selected, selection_audit = select_audit_rows(
        read_jsonl(scores_path),
        read_tsv(disagreements_path),
        score_columns,
        args.per_score_extreme,
        args.max_tensors,
    )
    output = args.output.expanduser().resolve()
    if output.exists() or output.with_suffix(".manifest.json").exists():
        raise SystemExit(f"refusing to overwrite frozen KL-audit selection: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(canonical_json(selected) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": "importance-score-kl-audit-subset-v0.1",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "descriptive KL-audit only; forbidden from KL-core inferential correlation",
        "selection_version": "importance-score-kl-audit-cap-v1",
        "score_columns": score_columns,
        "per_score_extreme": args.per_score_extreme,
        "max_tensors_cpu_cap": args.max_tensors,
        "cpu_cap_reason": (
            "CPU-only full-model full-vocabulary perturbation cost; preserve one top and bottom per "
            "score, then fill the remaining union by a deterministic structural hash"
        ),
        "label_dependence": (
            "Disagreement membership uses the already-joined paired teacher label exactly as specified "
            "for KL-audit; these rows are never pooled into KL-core or an admission statistic."
        ),
        "selection_audit": selection_audit,
        "selection_path": str(output),
        "selection_sha256": sha256_file(output),
        "input_hashes": {
            "scores_prejoin": sha256_file(scores_path),
            "scores_prejoin_receipt": sha256_file(receipt_path),
            "disagreements": sha256_file(disagreements_path),
        },
    }
    output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
