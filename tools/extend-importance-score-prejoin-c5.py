#!/usr/bin/env python3
"""Append sealed C5 gradient scores to an existing label-free prejoin table."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any


BASE_SCORE_COLUMNS = (
    "C0_I",
    "C1_I",
    "C4_I",
    "S_AWQ_level",
    "S_AWQ_tail",
    "S_range",
)
C5A_SCORE_COLUMNS = (
    "C5a_Taylor_quant_I",
    "C5a_Taylor_L1_S",
    "C5a_Taylor_squared_S",
)
C5B_SCORE_COLUMNS = (
    "C5b_Self_Fisher_I",
    "C5b_Empirical_Fisher_I",
)
C5A_ALL_KEYS = (
    "C5a_Taylor_quant_I",
    "C5a_Taylor_quant_A_low",
    "C5a_Taylor_quant_A_high",
    "C5a_Taylor_quant_raw_gain",
    "C5a_Taylor_quant_G",
    "C5a_Taylor_L1_S",
    "C5a_Taylor_squared_S",
)
EMPIRICAL_FISHER_ALL_KEYS = (
    "C5b_Empirical_Fisher_I",
    "C5b_Empirical_Fisher_A_low",
    "C5b_Empirical_Fisher_A_high",
    "C5b_Empirical_Fisher_raw_gain",
    "C5b_Empirical_Fisher_G",
)
SELF_FISHER_ALL_KEYS = (
    "C5b_Self_Fisher_I",
    "C5b_Self_Fisher_A_low",
    "C5b_Self_Fisher_A_high",
    "C5b_Self_Fisher_raw_gain",
    "C5b_Self_Fisher_G",
)
WINNER_ELIGIBLE = (
    *BASE_SCORE_COLUMNS,
    "C5a_Taylor_quant_I",
    "C5b_Self_Fisher_I",
)
SECONDARY = (
    "C5a_Taylor_L1_S",
    "C5a_Taylor_squared_S",
    "C5b_Empirical_Fisher_I",
)
FORBIDDEN_LABEL_KEYS = {
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
STRUCTURAL_KEYS = (
    "model_id",
    "architecture",
    "layer_id",
    "canonical_family",
    "hf_name",
    "shape",
    "n_params",
)
CURRENT_IMPLEMENTATION_FILES = (
    "build-importance-score-prejoin.py",
    "extend-importance-score-prejoin-c5.py",
    "build-ud-tensor-labels.py",
    "build-importance-active-label-view.py",
    "collect-importance-gradient-scores.py",
    "freeze-importance-score-fisher-corpus.py",
    "freeze-importance-kl-audit-subset.py",
    "report-importance-score-formal.py",
    "report-importance-score-two-model.py",
    "run-importance-single-tensor-perturbation.py",
    "run-aq-tensor-sample.py",
    "score-block-covariance-c1.py",
    "summarize-importance-score-screen.py",
)
SEALED_STATUS = "sealed score table generated without accepting or opening a GGUF label manifest"


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


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def tensor_name_set_sha256(names: set[str]) -> str:
    return hashlib.sha256(("\n".join(sorted(names)) + "\n").encode("utf-8")).hexdigest()


def require_finite_map(values: dict[str, Any], context: str) -> None:
    for key, value in values.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{context}.{key} is not a numeric score")
        if not math.isfinite(float(value)):
            raise ValueError(f"{context}.{key} is not finite")


def verify_base(
    scores_path: Path, receipt_path: Path, shard_path: Path
) -> tuple[list[dict[str, Any]], list[dict[str, dict[str, float]]], dict[str, Any]]:
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("schema_version") != "importance-score-prejoin-receipt-v0.1":
        raise ValueError("base prejoin receipt has an unexpected schema")
    if receipt.get("status") != SEALED_STATUS:
        raise ValueError("base prejoin receipt is not label-free and sealed")
    if receipt.get("score_table_sha256") != sha256_file(scores_path):
        raise ValueError("base prejoin score table hash mismatch")
    if receipt.get("shard_scores_sha256") != sha256_file(shard_path):
        raise ValueError("base prejoin shard score hash mismatch")
    rows = read_jsonl(scores_path)
    shards = json.loads(shard_path.read_text(encoding="utf-8"))
    if not isinstance(shards, list) or len(shards) != 4:
        raise ValueError("base prejoin must contain exactly four shard score maps")
    names = {str(row.get("hf_name")) for row in rows}
    if len(names) != len(rows):
        raise ValueError("base prejoin has duplicate tensor names")
    if int(receipt.get("tensor_count", -1)) != len(rows):
        raise ValueError("base prejoin tensor count mismatch")
    if receipt.get("tensor_name_set_sha256") != tensor_name_set_sha256(names):
        raise ValueError("base prejoin tensor-name set hash mismatch")
    if set(receipt.get("candidate_score_columns", [])) != set(BASE_SCORE_COLUMNS):
        raise ValueError("base prejoin does not contain exactly the six frozen v0.1 scores")
    leaked = sorted({key for row in rows for key in FORBIDDEN_LABEL_KEYS if key in row})
    if leaked:
        raise ValueError(f"base prejoin contains teacher-label fields: {leaked}")
    if any(set(shard) != names for shard in shards):
        raise ValueError("base prejoin shard tensor coverage is not exact")
    return rows, shards, receipt


def gradient_output_path(receipt: dict[str, Any], receipt_path: Path) -> Path:
    raw = receipt.get("score_table_path") or receipt.get("result_path")
    if not raw:
        raise ValueError(f"gradient receipt has no result path: {receipt_path}")
    path = Path(str(raw)).expanduser()
    if not path.is_absolute():
        path = receipt_path.parent / path
    return path.resolve()


def verify_gradient_result(
    receipt_path: Path,
    expected_mode: str,
    expected_names: set[str],
    expected_model_id: str,
    expected_inputs: dict[str, str],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any], Path]:
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("mode") != expected_mode:
        raise ValueError(
            f"gradient receipt mode mismatch: {receipt.get('mode')} != {expected_mode}"
        )
    if receipt.get("status") != "sealed C5 gradient score output":
        raise ValueError("gradient receipt is not a sealed C5 score result")
    if receipt.get("model_id") != expected_model_id:
        raise ValueError("gradient result model differs from the base prejoin")
    if receipt.get("workspace_git_head") != git_revision():
        raise ValueError("gradient result was generated from a different git HEAD")
    result_path = gradient_output_path(receipt, receipt_path)
    if not result_path.is_file():
        raise ValueError(f"gradient result is missing: {result_path}")
    expected_hash = receipt.get("score_table_sha256") or receipt.get("result_sha256")
    if expected_hash != sha256_file(result_path):
        raise ValueError("gradient result hash differs from its receipt")
    rows = read_jsonl(result_path)
    by_name = {str(row.get("hf_name")): row for row in rows}
    if len(by_name) != len(rows) or set(by_name) != expected_names:
        raise ValueError("gradient result tensor coverage differs from the base prejoin")
    if int(receipt.get("tensor_count", -1)) != len(rows):
        raise ValueError("gradient receipt tensor count mismatch")
    name_set_hash = receipt.get("tensor_name_set_sha256")
    runner_name_set_hash = hashlib.sha256(
        canonical_json(sorted(by_name)).encode("utf-8")
    ).hexdigest()
    if name_set_hash is not None:
        expected_name_set_hash = tensor_name_set_sha256(set(by_name))
    else:
        name_set_hash = receipt.get("name_set_sha256")
        expected_name_set_hash = runner_name_set_hash
    if name_set_hash != expected_name_set_hash:
        raise ValueError("gradient receipt tensor-name set hash mismatch")
    input_hashes = receipt.get("input_hashes", {})
    for key, expected in expected_inputs.items():
        if input_hashes.get(key) != expected:
            raise ValueError(f"gradient receipt input hash mismatch: {key}")
    runner_hash_entry = receipt.get("implementation_hashes", {}).get(
        "collect-importance-gradient-scores.py"
    ) or receipt.get("implementation_hashes", {}).get("gradient_runner")
    runner_hash = (
        runner_hash_entry.get("sha256")
        if isinstance(runner_hash_entry, dict)
        else runner_hash_entry
    )
    current_runner_hash = sha256_file(
        Path(__file__).resolve().parent / "collect-importance-gradient-scores.py"
    )
    if runner_hash != current_runner_hash:
        raise ValueError("gradient collector implementation hash mismatch")
    for name, row in by_name.items():
        leaked = sorted(FORBIDDEN_LABEL_KEYS.intersection(row))
        if leaked:
            raise ValueError(f"gradient result contains teacher fields for {name}: {leaked}")
        scores = row.get("scores")
        shards = row.get("shard_scores")
        if not isinstance(scores, dict):
            raise ValueError(f"gradient row has no score map: {name}")
        if not isinstance(shards, list) or len(shards) != 4:
            raise ValueError(f"gradient row must have four shard score maps: {name}")
        require_finite_map(scores, f"{expected_mode}:{name}:scores")
        for index, shard_scores in enumerate(shards):
            if not isinstance(shard_scores, dict):
                raise ValueError(f"gradient shard score map is invalid: {name}:{index}")
            require_finite_map(shard_scores, f"{expected_mode}:{name}:shard-{index}")
    return by_name, receipt, result_path


def structural_match(base: dict[str, Any], gradient: dict[str, Any], context: str) -> None:
    for key in STRUCTURAL_KEYS:
        if key not in gradient:
            raise ValueError(f"{context} gradient row is missing structural key {key}")
        if gradient[key] != base[key]:
            raise ValueError(f"{context} structural mismatch for {key}")


def selected_score_map(
    row: dict[str, Any], keys: tuple[str, ...], context: str
) -> dict[str, float]:
    scores = row["scores"]
    missing = [key for key in keys if key not in scores]
    if missing:
        raise ValueError(f"{context} is missing required C5 scores: {missing}")
    return {key: float(scores[key]) for key in keys}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-prejoin-scores", type=Path, required=True)
    parser.add_argument("--base-prejoin-receipt", type=Path, required=True)
    parser.add_argument("--base-prejoin-shard-scores", type=Path, required=True)
    parser.add_argument("--empirical-receipt", type=Path, required=True)
    parser.add_argument("--self-fisher-receipt", type=Path)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--score-registry", type=Path, required=True)
    parser.add_argument("--corpus-manifest", type=Path, required=True)
    parser.add_argument("--fisher-corpus-manifest", type=Path, required=True)
    parser.add_argument("--stage", choices=("c5a", "full"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.stage == "full" and args.self_fisher_receipt is None:
        raise SystemExit("full C5 extension requires --self-fisher-receipt")
    if args.stage == "c5a" and args.self_fisher_receipt is not None:
        raise SystemExit("C5a-only extension must not accept a self-Fisher result")
    output_dir = args.output_dir.expanduser().resolve()
    scores_path = output_dir / "scores-prejoin.jsonl"
    shard_path = output_dir / "shard-scores-prejoin.json"
    receipt_path = output_dir / "scores-prejoin.receipt.json"
    existing = [path for path in (scores_path, shard_path, receipt_path) if path.exists()]
    if existing:
        raise SystemExit(f"refusing to overwrite sealed C5 prejoin outputs: {existing}")

    base_paths = {
        "base_prejoin_scores": args.base_prejoin_scores.expanduser().resolve(),
        "base_prejoin_receipt": args.base_prejoin_receipt.expanduser().resolve(),
        "base_prejoin_shard_scores": args.base_prejoin_shard_scores.expanduser().resolve(),
    }
    common_paths = {
        "candidate_manifest": args.candidate_manifest.expanduser().resolve(),
        "score_registry": args.score_registry.expanduser().resolve(),
        "corpus_manifest": args.corpus_manifest.expanduser().resolve(),
        "fisher_corpus_manifest": args.fisher_corpus_manifest.expanduser().resolve(),
    }
    gradient_receipts = {
        "empirical_receipt": args.empirical_receipt.expanduser().resolve(),
    }
    if args.self_fisher_receipt is not None:
        gradient_receipts["self_fisher_receipt"] = (
            args.self_fisher_receipt.expanduser().resolve()
        )
    for name, path in {**base_paths, **common_paths, **gradient_receipts}.items():
        if not path.is_file():
            raise SystemExit(f"missing C5 prejoin input {name}: {path}")

    base_rows, base_shards, base_receipt = verify_base(
        base_paths["base_prejoin_scores"],
        base_paths["base_prejoin_receipt"],
        base_paths["base_prejoin_shard_scores"],
    )
    names = {str(row["hf_name"]) for row in base_rows}
    model_id = str(base_rows[0]["model_id"])
    if any(str(row["model_id"]) != model_id for row in base_rows):
        raise ValueError("base prejoin mixes model IDs")
    common_hashes = {name: sha256_file(path) for name, path in common_paths.items()}
    for key in ("candidate_manifest", "score_registry", "corpus_manifest"):
        if base_receipt.get("input_hashes", {}).get(key) != common_hashes[key]:
            raise ValueError(f"base prejoin and current input differ: {key}")
    expected_gradient_inputs = {
        "candidate_manifest": common_hashes["candidate_manifest"],
        "fisher_corpus_manifest": common_hashes["fisher_corpus_manifest"],
        "source_roster": base_receipt.get("input_hashes", {}).get("source_roster"),
        "source_roster_manifest": base_receipt.get("input_hashes", {}).get(
            "source_roster_manifest"
        ),
    }
    if any(value is None for value in expected_gradient_inputs.values()):
        raise ValueError("base prejoin does not bind the source roster inputs")
    empirical, empirical_receipt, empirical_result_path = verify_gradient_result(
        gradient_receipts["empirical_receipt"],
        "empirical",
        names,
        model_id,
        expected_gradient_inputs,
    )
    self_fisher: dict[str, dict[str, Any]] | None = None
    self_receipt: dict[str, Any] | None = None
    self_result_path: Path | None = None
    if args.stage == "full":
        self_fisher, self_receipt, self_result_path = verify_gradient_result(
            gradient_receipts["self_fisher_receipt"],
            "self_fisher",
            names,
            model_id,
            expected_gradient_inputs,
        )

    merged_rows: list[dict[str, Any]] = []
    merged_shards = [dict(shard) for shard in base_shards]
    for base in base_rows:
        name = str(base["hf_name"])
        empirical_row = empirical[name]
        structural_match(base, empirical_row, f"empirical:{name}")
        additions = selected_score_map(empirical_row, C5A_ALL_KEYS, f"empirical:{name}")
        if args.stage == "full":
            additions.update(
                selected_score_map(
                    empirical_row, EMPIRICAL_FISHER_ALL_KEYS, f"empirical:{name}"
                )
            )
            assert self_fisher is not None
            structural_match(base, self_fisher[name], f"self_fisher:{name}")
            additions.update(
                selected_score_map(
                    self_fisher[name], SELF_FISHER_ALL_KEYS, f"self_fisher:{name}"
                )
            )
        collisions = sorted(set(base).intersection(additions))
        if collisions:
            raise ValueError(f"C5 score collides with a base column for {name}: {collisions}")
        merged = dict(base)
        merged.update(additions)
        for key, value in base.items():
            if merged[key] != value:
                raise RuntimeError(f"legacy prejoin value changed for {name}: {key}")
        merged_rows.append(merged)

        for shard_index in range(4):
            shard_additions = selected_score_map(
                {"scores": empirical_row["shard_scores"][shard_index]},
                C5A_ALL_KEYS,
                f"empirical:{name}:shard-{shard_index}",
            )
            if args.stage == "full":
                shard_additions.update(
                    selected_score_map(
                        {"scores": empirical_row["shard_scores"][shard_index]},
                        EMPIRICAL_FISHER_ALL_KEYS,
                        f"empirical:{name}:shard-{shard_index}",
                    )
                )
                assert self_fisher is not None
                shard_additions.update(
                    selected_score_map(
                        {"scores": self_fisher[name]["shard_scores"][shard_index]},
                        SELF_FISHER_ALL_KEYS,
                        f"self_fisher:{name}:shard-{shard_index}",
                    )
                )
            base_shard_scores = merged_shards[shard_index][name]
            shard_collisions = sorted(set(base_shard_scores).intersection(shard_additions))
            if shard_collisions:
                raise ValueError(
                    f"C5 shard score collides with a base column for {name}: {shard_collisions}"
                )
            merged_shards[shard_index][name] = {
                **base_shard_scores,
                **shard_additions,
            }

    output_dir.mkdir(parents=True, exist_ok=True)
    with scores_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in merged_rows:
            handle.write(canonical_json(row) + "\n")
    write_json(shard_path, merged_shards)
    c5_columns = list(C5A_SCORE_COLUMNS)
    c5_all_keys = list(C5A_ALL_KEYS)
    if args.stage == "full":
        c5_columns.extend(C5B_SCORE_COLUMNS)
        c5_all_keys.extend(SELF_FISHER_ALL_KEYS)
        c5_all_keys.extend(EMPIRICAL_FISHER_ALL_KEYS)
    candidate_score_columns = [*BASE_SCORE_COLUMNS, *c5_columns]
    winner_eligible = [
        score for score in candidate_score_columns if score in WINNER_ELIGIBLE
    ]
    secondary = [score for score in candidate_score_columns if score in SECONDARY]
    tool_dir = Path(__file__).resolve().parent
    implementation_hashes = {
        name: sha256_file(tool_dir / name) for name in CURRENT_IMPLEMENTATION_FILES
    }
    input_paths = {**base_paths, **common_paths, **gradient_receipts}
    input_paths["empirical_result"] = empirical_result_path
    if self_result_path is not None:
        input_paths["self_fisher_result"] = self_result_path
    sealed_input_hashes = {name: sha256_file(path) for name, path in input_paths.items()}
    sealed_input_hashes["source_roster"] = expected_gradient_inputs["source_roster"]
    sealed_input_hashes["source_roster_manifest"] = expected_gradient_inputs[
        "source_roster_manifest"
    ]
    c5_execution_settings = {"empirical": empirical_receipt["execution_settings"]}
    if self_receipt is not None:
        c5_execution_settings["self_fisher"] = self_receipt["execution_settings"]
    receipt = {
        "schema_version": "importance-score-prejoin-receipt-v0.1",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": SEALED_STATUS,
        "model_id": model_id,
        "architecture": merged_rows[0]["architecture"],
        "workspace_git_head": git_revision(),
        "score_table_path": str(scores_path),
        "score_table_sha256": sha256_file(scores_path),
        "shard_scores_path": str(shard_path),
        "shard_scores_sha256": sha256_file(shard_path),
        "tensor_count": len(merged_rows),
        "tensor_name_set_sha256": tensor_name_set_sha256(names),
        "candidate_score_columns": candidate_score_columns,
        "reported_score_columns": candidate_score_columns,
        "winner_eligible_score_columns": winner_eligible,
        "secondary_score_columns": secondary,
        "score_columns": sorted(
            {key for row in merged_rows for key in row if key.startswith("C") or key.startswith("S_")}
        ),
        "forbidden_label_keys_verified_absent": sorted(FORBIDDEN_LABEL_KEYS),
        "execution_settings": base_receipt["execution_settings"],
        "c5_execution_settings": c5_execution_settings,
        "input_hashes": sealed_input_hashes,
        "implementation_hashes": implementation_hashes,
        "c5_extension": {
            "stage": args.stage,
            "legacy_base_preserved_exactly": True,
            "base_score_table_sha256": sha256_file(base_paths["base_prejoin_scores"]),
            "base_receipt_sha256": sha256_file(base_paths["base_prejoin_receipt"]),
            "base_shard_scores_sha256": sha256_file(
                base_paths["base_prejoin_shard_scores"]
            ),
            "added_candidate_score_columns": c5_columns,
            "added_score_columns": sorted(set(c5_all_keys)),
            "fisher_corpus_manifest_sha256": common_hashes[
                "fisher_corpus_manifest"
            ],
            "label_or_GGUF_inputs_accepted": False,
        },
    }
    write_json(receipt_path, receipt)
    print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
