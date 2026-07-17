#!/usr/bin/env python3
"""Compare P2 source/path oracle sidecars row-by-row without a logit matrix.

Both vector stores are read in fixed-size F32 chunks.  The comparator emits
per-row hidden/logit metrics, top-k and greedy agreement, and (when available)
the target's streaming execution-row state binding.  Missing state evidence is
reported as ``not_captured`` rather than treated as agreement.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import stat
import sys
from pathlib import Path
from typing import Any, BinaryIO


SCHEMA = "ullm.aq4_p2_production_baseline_oracle_comparison.v1"
HIDDEN = 4096
LOGITS = 248320
F32_BYTES = 4
MAX_ROW_LINE = 64 * 1024


class CompareError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CompareError(message)


def sha_file(path: Path) -> str:
    info = path.lstat()
    require(stat.S_ISREG(info.st_mode) and not stat.S_ISLNK(info.st_mode), f"artifact member must be regular: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path, label: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicate)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CompareError(f"{label} is invalid: {error}") from error


def reject_duplicate(items: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in items:
        if key in result:
            raise CompareError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def parse_sums(root: Path) -> dict[str, str]:
    path = root / "SHA256SUMS"
    text = path.read_text(encoding="ascii")
    result: dict[str, str] = {}
    for number, line in enumerate(text.splitlines(), 1):
        digest, marker, name = line.partition("  ")
        require(marker == "  " and len(digest) == 64 and all(char in "0123456789abcdef" for char in digest) and name and "/../" not in name and not name.startswith("/"), f"SHA256SUMS line {number} differs")
        require(name not in result, f"SHA256SUMS duplicate member: {name}")
        result[name] = digest
    return result


def artifact(root: Path, label: str) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, str]]:
    root = root.absolute()
    info = root.lstat()
    require(stat.S_ISDIR(info.st_mode) and not stat.S_ISLNK(info.st_mode), f"{label} root must be a real directory")
    sums = parse_sums(root)
    required = {"manifest.json", "rows.jsonl", "vectors/hidden.f32le", "vectors/logits.f32le"}
    require(required <= set(sums), f"{label} SHA256SUMS lacks vector members")
    for relative, digest in sums.items():
        path = root / relative
        require(path.is_file() and not path.is_symlink() and sha_file(path) == digest, f"{label} SHA256SUMS mismatch: {relative}")
    manifest = load_json(root / "manifest.json", f"{label} manifest")
    require(isinstance(manifest, dict), f"{label} manifest root differs")
    files = manifest.get("files")
    require(isinstance(files, dict) and files.get("rows") == "rows.jsonl" and files.get("hidden") == "vectors/hidden.f32le" and files.get("logits") == "vectors/logits.f32le", f"{label} vector file map differs")
    rows: list[dict[str, Any]] = []
    with (root / "rows.jsonl").open("rb") as stream:
        for number, raw in enumerate(stream, 1):
            require(len(raw) <= MAX_ROW_LINE and raw.endswith(b"\n"), f"{label} row {number} is oversized/unterminated")
            try:
                row = json.loads(raw, object_pairs_hook=reject_duplicate)
            except json.JSONDecodeError as error:
                raise CompareError(f"{label} row {number} is invalid: {error}") from error
            require(isinstance(row, dict), f"{label} row {number} is not an object")
            rows.append(row)
    require(rows, f"{label} has no rows")
    return manifest, rows, sums


def vector_metrics(left: BinaryIO, right: BinaryIO, left_ref: dict[str, Any], right_ref: dict[str, Any], elements: int, chunk_elements: int) -> dict[str, float | int]:
    try:
        import numpy as np
    except ImportError as error:
        raise CompareError(f"numpy is required for streamed vector comparison: {error}") from error
    for reference, label in ((left_ref, "source"), (right_ref, "target")):
        require(reference.get("elements") == elements and reference.get("bytes") == elements * F32_BYTES and reference.get("dtype") == "f32" and reference.get("endianness") == "little", f"{label} vector reference differs")
    left.seek(int(left_ref["offset_bytes"]))
    right.seek(int(right_ref["offset_bytes"]))
    sq_error = 0.0
    sq_left = 0.0
    sq_right = 0.0
    dot = 0.0
    max_abs = 0.0
    nonfinite = 0
    seen = 0
    while seen < elements:
        count = min(chunk_elements, elements - seen)
        bytes_count = count * F32_BYTES
        lhs = left.read(bytes_count)
        rhs = right.read(bytes_count)
        require(len(lhs) == bytes_count and len(rhs) == bytes_count, "vector sidecar is short")
        a = np.frombuffer(lhs, dtype="<f4")
        b = np.frombuffer(rhs, dtype="<f4")
        finite = np.isfinite(a) & np.isfinite(b)
        nonfinite += int(count - int(finite.sum()))
        if finite.any():
            af = a[finite].astype("float64", copy=False)
            bf = b[finite].astype("float64", copy=False)
            delta = af - bf
            sq_error += float(np.dot(delta, delta))
            sq_left += float(np.dot(af, af))
            sq_right += float(np.dot(bf, bf))
            max_abs = max(max_abs, float(np.max(np.abs(delta))))
            dot += float(np.dot(af, bf))
        seen += count
    denominator = math.sqrt(sq_left) if sq_left else 0.0
    cosine_denominator = math.sqrt(sq_left * sq_right)
    return {
        "relative_l2": math.sqrt(sq_error) / denominator if denominator else math.inf,
        "cosine": dot / cosine_denominator if cosine_denominator else 0.0,
        "max_abs": max_abs,
        "nonfinite_elements": nonfinite,
    }


def top_ids(row: dict[str, Any]) -> list[int]:
    values = row.get("topk")
    require(isinstance(values, list) and values, "row topk differs")
    ids: list[int] = []
    for value in values:
        require(isinstance(value, dict) and type(value.get("token_id")) is int, "row topk entry differs")
        ids.append(value["token_id"])
    return ids


def execution_state(
    target: Path,
    target_manifest: dict[str, Any],
    target_sums: dict[str, str],
    source_row_sha256: dict[tuple[str, int], str],
) -> dict[str, Any]:
    path = target / "execution-rows.jsonl"
    if not path.is_file() or path.is_symlink():
        return {"status": "not_captured", "reason": "target artifact does not contain execution-rows.jsonl"}
    declared = target_manifest.get("execution_rows")
    require(isinstance(declared, dict) and declared.get("file") == "execution-rows.jsonl", "target execution-state manifest differs")
    require(declared.get("sha256") == target_sums.get("execution-rows.jsonl") == sha_file(path), "target execution-state hash binding differs")
    records: list[dict[str, Any]] = []
    with path.open("rb") as stream:
        for raw in stream:
            require(len(raw) <= MAX_ROW_LINE and raw.endswith(b"\n"), "execution state row differs")
            value = json.loads(raw, object_pairs_hook=reject_duplicate)
            require(isinstance(value, dict), "execution state record differs")
            key = (value.get("case_id"), value.get("step"))
            require(
                key in source_row_sha256
                and value.get("source_row_sha256") == source_row_sha256[key]
                and isinstance(value.get("source_sequence_sha256"), str)
                and isinstance(value.get("generation_epoch"), int)
                and value.get("observation_complete") is True
                and value.get("publication_committed") is True,
                "execution state source-row binding differs",
            )
            records.append(value)
    require(declared.get("record_count") == len(records) and declared.get("lockstep_with_vector_rows") is True, "execution-state record count differs")
    return {"status": "captured", "record_count": len(records), "records": records}


def compare(args: argparse.Namespace) -> dict[str, Any]:
    source_manifest, source_rows, source_sums = artifact(args.source, "source")
    target_manifest, target_rows, target_sums = artifact(args.target, "target")
    require(source_manifest.get("oracle_kind") == "independent_source_full", "source oracle kind differs")
    require(target_manifest.get("oracle_kind") in {"aq4_target", "same_artifact_all_m1"}, "target path oracle kind differs")
    if args.case_id is not None:
        source_rows = [row for row in source_rows if row.get("case_id") == args.case_id]
        target_rows = [row for row in target_rows if row.get("case_id") == args.case_id]
        require(source_rows and target_rows, f"requested oracle case is absent: {args.case_id}")
    source_index = {(row.get("case_id"), row.get("step")): row for row in source_rows}
    target_index = {(row.get("case_id"), row.get("step")): row for row in target_rows}
    require(len(source_index) == len(source_rows) and len(target_index) == len(target_rows), "oracle rows contain duplicate keys")
    require(set(source_index) == set(target_index), "source/target row keys differ")
    source_row_sha = {
        (row.get("case_id"), row.get("step")): hashlib.sha256(
            json.dumps(row, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8") + b"\n"
        ).hexdigest()
        for row in source_rows
    }
    rows: list[dict[str, Any]] = []
    with (args.source / "vectors/hidden.f32le").open("rb") as source_hidden, (args.source / "vectors/logits.f32le").open("rb") as source_logits, (args.target / "vectors/hidden.f32le").open("rb") as target_hidden, (args.target / "vectors/logits.f32le").open("rb") as target_logits:
        for key in sorted(source_index):
            source_row = source_index[key]
            target_row = target_index[key]
            require(source_row.get("input_token_ids_sha256") == target_row.get("input_token_ids_sha256"), f"input token hash differs for {key}")
            hidden = vector_metrics(source_hidden, target_hidden, source_row["hidden"], target_row["hidden"], HIDDEN, args.chunk_elements)
            logits = vector_metrics(source_logits, target_logits, source_row["logits"], target_row["logits"], LOGITS, args.chunk_elements)
            source_top = top_ids(source_row)
            target_top = top_ids(target_row)
            rows.append(
                {
                    "case_id": key[0],
                    "step": key[1],
                    "source_row_sha256": source_row_sha[key],
                    "source_greedy_token_id": source_row.get("greedy_token_id"),
                    "target_greedy_token_id": target_row.get("greedy_token_id"),
                    "token_agreement": source_row.get("greedy_token_id") == target_row.get("greedy_token_id"),
                    "top10_overlap": len(set(source_top[:10]) & set(target_top[:10])) / 10.0,
                    "hidden": hidden,
                    "logits": logits,
                }
            )
    status = "available" if all(item["hidden"]["nonfinite_elements"] == 0 and item["logits"]["nonfinite_elements"] == 0 for item in rows) else "blocked_nonfinite"
    result = {
        "schema_version": SCHEMA,
        "status": status,
        "source": {"root": str(args.source.absolute()), "manifest_sha256": source_sums["manifest.json"]},
        "target": {"root": str(args.target.absolute()), "manifest_sha256": target_sums["manifest.json"]},
        "streaming": {"chunk_elements": args.chunk_elements, "full_logit_matrix_retained": False},
        "row_count": len(rows),
        "case_filter": args.case_id,
        "rows": rows,
        "state_snapshot": execution_state(args.target, target_manifest, target_sums, source_row_sha),
    }
    require(not os.path.lexists(args.output), f"comparison output already exists: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=True, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return result


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case-id", help="compare one anchor from a multi-anchor source artifact")
    parser.add_argument("--chunk-elements", type=int, default=65536)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        require(0 < args.chunk_elements <= 1_048_576, "chunk-elements is outside bounded range")
        args.source = args.source.absolute()
        args.target = args.target.absolute()
        args.output = args.output.absolute()
        result = compare(args)
    except (CompareError, OSError, ValueError) as error:
        print(f"AQ4 P2 production oracle comparison failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps({"schema_version": SCHEMA, "status": result["status"], "row_count": result["row_count"]}, ensure_ascii=True, sort_keys=True))
    return 0 if result["status"] == "available" else 1


if __name__ == "__main__":
    raise SystemExit(main())
