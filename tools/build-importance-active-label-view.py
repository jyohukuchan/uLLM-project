#!/usr/bin/env python3
"""Build the one-shot Gemma C5 active-label view without reopening GGUF files.

The C5 Gemma score table is a lockbox input.  This tool authenticates the
Qwen freeze, the later label-blind Gemma prejoin receipt, both sealed score
files, and the label-blind source roster before it opens the already sealed
GGUF-derived label TSV.  It then selects labels solely by the exact source and
prejoin ``hf_name`` cohort.  No GGUF command or model code is invoked.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import re
import subprocess
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any


MODEL_ID = "gemma-4-E4B-it"
ACTIVE_TENSOR_COUNT = 258
ORIGINAL_PHYSICAL_ELIGIBLE_COUNT = 294
FREEZE_STATUS = "sealed before any Gemma tensor-level score/label join"
PREJOIN_STATUS = (
    "sealed score table generated without accepting or opening a GGUF label manifest"
)
LOCKBOX_STATUS = (
    "C5 score-label join order verified against previously sealed GGUF labels"
)
SOURCE_ROSTER_STATUS = (
    "frozen without reading GGUF tensor names, types, ordinals, or promotion labels"
)

REQUIRED_REPORTED_SCORES = {
    "C0_I",
    "C1_I",
    "C4_I",
    "S_AWQ_level",
    "S_AWQ_tail",
    "S_range",
    "C5a_Taylor_quant_I",
    "C5a_Taylor_L1_S",
    "C5a_Taylor_squared_S",
    "C5b_Self_Fisher_I",
    "C5b_Empirical_Fisher_I",
}
REQUIRED_C5_WINNER_SCORES = {
    "C5a_Taylor_quant_I",
    "C5b_Self_Fisher_I",
}
REQUIRED_C5_SECONDARY_SCORES = {
    "C5a_Taylor_L1_S",
    "C5a_Taylor_squared_S",
    "C5b_Empirical_Fisher_I",
}
LOCKBOX_INPUT_HASHES = (
    "candidate_manifest",
    "score_registry",
    "corpus_manifest",
    "fisher_corpus_manifest",
)
LOCKBOX_SHARED_IMPLEMENTATIONS = (
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
    "promoted_vs_4bit_floor",
    "eligible",
    "exclusion_reason",
    "label_mode",
}
LABEL_REQUIRED_FIELDS = (
    "model_id",
    "architecture",
    "layer_id",
    "canonical_family",
    "gguf_name",
    "hf_name",
    "shape",
    "n_params",
    "qtype_ud",
    "qtype_static",
    "ordinal_ud",
    "ordinal_static",
    "packed_bpp_ud",
    "packed_bpp_static",
    "promotion_delta_ordinal",
    "promotion_delta_bpp",
    "promoted",
    "eligible",
    "exclusion_reason",
    "promoted_vs_4bit_floor",
    "label_mode",
    "gguf_shape_ne",
    "hf_shape",
    "shape_status",
    "semantic_transform_note",
)
ORDINAL = {"IQ4_XS": 0, "Q4_K": 0, "Q5_K": 1, "Q6_K": 2, "Q8_0": 3}
PACKED_BPP = {"IQ4_XS": 4.25, "Q4_K": 4.5, "Q5_K": 5.5, "Q6_K": 6.5625, "Q8_0": 8.5}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def tensor_name_set_sha256(names: set[str]) -> str:
    return hashlib.sha256(("\n".join(sorted(names)) + "\n").encode("utf-8")).hexdigest()


def git_revision() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
    ).strip()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path} line {line_number} is not a JSON object")
        rows.append(value)
    return rows


def read_label_tsv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    return fields, rows


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_tsv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields, delimiter="\t", extrasaction="raise"
        )
        writer.writeheader()
        writer.writerows(rows)


def parse_timestamp(value: Any, label: str) -> dt.datetime:
    try:
        parsed = dt.datetime.fromisoformat(str(value))
    except ValueError as error:
        raise ValueError(f"{label} is not a valid ISO-8601 timestamp") from error
    if parsed.utcoffset() is None:
        raise ValueError(f"{label} must include a timezone")
    return parsed


def require_sha256(value: Any, label: str) -> str:
    result = str(value or "")
    if re.fullmatch(r"[0-9a-f]{64}", result) is None:
        raise ValueError(f"{label} is not a lowercase SHA-256 digest")
    return result


def require_unique_names(rows: list[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    by_name: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        name = str(row.get("hf_name", ""))
        if not name:
            raise ValueError(f"{label} row {index} has no hf_name")
        if name in by_name:
            raise ValueError(f"duplicate hf_name in {label}: {name}")
        by_name[name] = row
    return by_name


def normalized_shape(value: Any, label: str) -> list[int]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as error:
            raise ValueError(f"{label} is not a JSON shape") from error
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} is not a non-empty shape")
    try:
        shape = [int(item) for item in value]
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} contains a non-integer dimension") from error
    if any(item <= 0 for item in shape):
        raise ValueError(f"{label} contains a non-positive dimension")
    return shape


def verify_lockbox_authorization(
    freeze_path: Path,
    receipt_path: Path,
    scores_path: Path,
    shard_scores_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Authenticate all score-side inputs before any old label artifact is read."""
    freeze = read_json(freeze_path)
    receipt = read_json(receipt_path)
    if not isinstance(freeze, dict) or not isinstance(receipt, dict):
        raise ValueError("candidate freeze and prejoin receipt must be JSON objects")
    if freeze.get("status") != FREEZE_STATUS:
        raise ValueError("Qwen C5 candidate freeze has an unexpected status")
    if freeze.get("development_model") != "qwen3.5-9b":
        raise ValueError("candidate freeze does not name qwen3.5-9b as development model")
    if freeze.get("lockbox_model") != MODEL_ID:
        raise ValueError(f"candidate freeze does not authorize {MODEL_ID}")
    if receipt.get("status") != PREJOIN_STATUS:
        raise ValueError("Gemma C5 prejoin receipt is not label-blind and sealed")
    if receipt.get("model_id") != MODEL_ID:
        raise ValueError(f"prejoin receipt does not name {MODEL_ID}")

    freeze_time = parse_timestamp(freeze.get("created_at_utc"), "candidate freeze timestamp")
    receipt_time = parse_timestamp(receipt.get("created_at_utc"), "prejoin receipt timestamp")
    if freeze_time >= receipt_time:
        raise ValueError("Gemma C5 prejoin receipt is not later than the Qwen freeze")

    frozen_git = str(freeze.get("workspace_git_head", ""))
    receipt_git = str(receipt.get("workspace_git_head", ""))
    if not frozen_git or frozen_git != receipt_git:
        raise ValueError("candidate freeze and prejoin receipt use different git revisions")
    if git_revision() != frozen_git:
        raise ValueError("current workspace revision differs from the frozen lockbox revision")

    embedded_score_path = Path(str(receipt.get("score_table_path", ""))).expanduser().resolve()
    embedded_shard_path = Path(str(receipt.get("shard_scores_path", ""))).expanduser().resolve()
    if embedded_score_path != scores_path:
        raise ValueError("CLI score table path differs from the sealed prejoin receipt")
    if embedded_shard_path != shard_scores_path:
        raise ValueError("CLI shard-score path differs from the sealed prejoin receipt")
    actual_score_hash = sha256_file(scores_path)
    actual_shard_hash = sha256_file(shard_scores_path)
    if require_sha256(receipt.get("score_table_sha256"), "score table receipt hash") != actual_score_hash:
        raise ValueError("sealed Gemma C5 score table differs from its receipt")
    if require_sha256(receipt.get("shard_scores_sha256"), "shard table receipt hash") != actual_shard_hash:
        raise ValueError("sealed Gemma C5 shard-score table differs from its receipt")

    freeze_inputs = freeze.get("input_hashes")
    receipt_inputs = receipt.get("input_hashes")
    if not isinstance(freeze_inputs, dict) or not isinstance(receipt_inputs, dict):
        raise ValueError("lockbox input hash maps are missing")
    input_comparison: dict[str, dict[str, str]] = {}
    for name in LOCKBOX_INPUT_HASHES:
        frozen = require_sha256(freeze_inputs.get(name), f"freeze input hash {name}")
        sealed = require_sha256(receipt_inputs.get(name), f"prejoin input hash {name}")
        if frozen != sealed:
            raise ValueError(f"lockbox input hash mismatch for {name}")
        input_comparison[name] = {"candidate_freeze": frozen, "prejoin_receipt": sealed}

    freeze_impl = freeze.get("implementation_hashes")
    receipt_impl = receipt.get("implementation_hashes")
    if not isinstance(freeze_impl, dict) or not isinstance(receipt_impl, dict):
        raise ValueError("lockbox implementation hash maps are missing")
    expected_implementations = set(LOCKBOX_SHARED_IMPLEMENTATIONS)
    if set(freeze_impl) != expected_implementations or set(receipt_impl) != expected_implementations:
        raise ValueError("freeze/prejoin implementation hash key sets differ from the C5 contract")
    implementation_comparison: dict[str, dict[str, str]] = {}
    for name in LOCKBOX_SHARED_IMPLEMENTATIONS:
        frozen = require_sha256(freeze_impl.get(name), f"freeze implementation hash {name}")
        sealed = require_sha256(receipt_impl.get(name), f"prejoin implementation hash {name}")
        if frozen != sealed:
            raise ValueError(f"lockbox implementation hash mismatch for {name}")
        implementation_comparison[name] = {
            "candidate_freeze": frozen,
            "prejoin_receipt": sealed,
        }
    current_builder_hash = sha256_file(Path(__file__).resolve())
    if require_sha256(
        freeze_impl.get("build-importance-active-label-view.py"),
        "frozen active-label-view implementation hash",
    ) != current_builder_hash:
        raise ValueError("current active-label-view builder differs from the Qwen freeze")

    reported = freeze.get("reported_score_columns")
    transferred = freeze.get("candidate_scores_transferred_unchanged")
    receipt_reported = receipt.get("reported_score_columns")
    receipt_candidates = receipt.get("candidate_score_columns")
    winner = freeze.get("winner_eligible_score_columns")
    receipt_winner = receipt.get("winner_eligible_score_columns")
    secondary = freeze.get("secondary_score_columns")
    receipt_secondary = receipt.get("secondary_score_columns")
    for value, label in (
        (reported, "freeze reported scores"),
        (transferred, "freeze transferred scores"),
        (receipt_reported, "prejoin reported scores"),
        (receipt_candidates, "prejoin candidate scores"),
        (winner, "freeze winner-eligible scores"),
        (receipt_winner, "prejoin winner-eligible scores"),
        (secondary, "freeze secondary scores"),
        (receipt_secondary, "prejoin secondary scores"),
    ):
        if not isinstance(value, list) or len(set(value)) != len(value):
            raise ValueError(f"{label} must be an explicit duplicate-free list")
    if not (reported == transferred == receipt_reported == receipt_candidates):
        raise ValueError("reported candidate score order differs across freeze and prejoin")
    if set(reported) != REQUIRED_REPORTED_SCORES:
        raise ValueError("frozen score set is not exactly legacy-v0.1 plus the five C5 scores")
    if winner != receipt_winner or secondary != receipt_secondary:
        raise ValueError("candidate score roles differ across freeze and prejoin")
    if set(winner) & set(secondary):
        raise ValueError("winner-eligible and secondary score roles overlap")
    if not REQUIRED_C5_WINNER_SCORES.issubset(winner):
        raise ValueError("Taylor-quant and self-Fisher are not both winner eligible")
    if not REQUIRED_C5_SECONDARY_SCORES.issubset(secondary):
        raise ValueError("C5 secondary score roles are incomplete")
    if not set(winner).issubset(reported) or not set(secondary).issubset(reported):
        raise ValueError("candidate score roles contain an unreported score")

    formulas = freeze.get("formulas")
    thresholds = freeze.get("thresholds")
    if not isinstance(formulas, dict) or set(formulas) != set(reported):
        raise ValueError("candidate freeze formulas do not cover the reported score set exactly")
    if not isinstance(thresholds, dict) or not thresholds:
        raise ValueError("candidate freeze thresholds are missing")
    formula_winner = [
        name for name in reported if bool(formulas[name].get("winner_eligible", True))
    ]
    formula_secondary = [
        name for name in reported if bool(formulas[name].get("secondary", False))
    ]
    if formula_winner != winner or formula_secondary != secondary:
        raise ValueError("formula role metadata differs from the frozen score role lists")

    freeze_settings = freeze.get("execution_settings")
    if not isinstance(freeze_settings, dict):
        raise ValueError("candidate freeze execution settings are missing")
    if receipt.get("execution_settings") != freeze_settings.get("prejoin_score_generation"):
        raise ValueError("base prejoin execution settings differ from the Qwen freeze")
    if receipt.get("c5_execution_settings") != freeze_settings.get("c5_gradient"):
        raise ValueError("C5 execution settings differ from the Qwen freeze")

    receipt_score_columns = receipt.get("score_columns")
    if not isinstance(receipt_score_columns, list) or not set(reported).issubset(
        receipt_score_columns
    ):
        raise ValueError("prejoin receipt score columns omit a frozen candidate")
    return freeze, receipt, {
        "freeze_timestamp": freeze_time.isoformat(),
        "prejoin_timestamp": receipt_time.isoformat(),
        "workspace_git_head": frozen_git,
        "score_table_sha256": actual_score_hash,
        "shard_scores_sha256": actual_shard_hash,
        "input_hash_comparison": input_comparison,
        "implementation_hash_comparison": implementation_comparison,
        "reported_score_columns": list(reported),
        "winner_eligible_score_columns": list(winner),
        "secondary_score_columns": list(secondary),
        "formula_sha256": canonical_json_sha256(formulas),
        "threshold_sha256": canonical_json_sha256(thresholds),
        "prejoin_execution_settings": receipt["execution_settings"],
        "c5_execution_settings": receipt["c5_execution_settings"],
    }


def verify_source_and_scores(
    source_roster_path: Path,
    source_manifest_path: Path,
    scores_path: Path,
    shard_scores_path: Path,
    receipt: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[Any]]:
    manifest = read_json(source_manifest_path)
    roster = read_jsonl(source_roster_path)
    scores = read_jsonl(scores_path)
    shards = read_json(shard_scores_path)
    if not isinstance(manifest, dict) or manifest.get("schema_version") != "importance-score-source-roster-v0.1":
        raise ValueError("active source roster manifest has an unexpected schema")
    if manifest.get("status") != SOURCE_ROSTER_STATUS:
        raise ValueError("active source roster is not label-blind and frozen")
    if manifest.get("model_id") != MODEL_ID:
        raise ValueError(f"source roster manifest does not name {MODEL_ID}")
    if int(manifest.get("roster_tensor_count", -1)) != ACTIVE_TENSOR_COUNT:
        raise ValueError(f"active source roster must contain exactly {ACTIVE_TENSOR_COUNT} tensors")
    if Path(str(manifest.get("roster_path", ""))).expanduser().resolve() != source_roster_path:
        raise ValueError("source roster path differs from its manifest")
    roster_hash = sha256_file(source_roster_path)
    manifest_hash = sha256_file(source_manifest_path)
    if require_sha256(manifest.get("roster_sha256"), "source roster manifest hash") != roster_hash:
        raise ValueError("source roster differs from its manifest")
    receipt_inputs = receipt.get("input_hashes", {})
    if require_sha256(receipt_inputs.get("source_roster"), "prejoin source roster hash") != roster_hash:
        raise ValueError("sealed prejoin receipt binds a different source roster")
    if require_sha256(
        receipt_inputs.get("source_roster_manifest"), "prejoin source roster manifest hash"
    ) != manifest_hash:
        raise ValueError("sealed prejoin receipt binds a different source roster manifest")

    roster_by_name = require_unique_names(roster, "active source roster")
    score_by_name = require_unique_names(scores, "sealed prejoin score table")
    if len(roster_by_name) != ACTIVE_TENSOR_COUNT:
        raise ValueError(f"active source roster must contain exactly {ACTIVE_TENSOR_COUNT} unique names")
    if set(score_by_name) != set(roster_by_name):
        missing = sorted(set(roster_by_name) - set(score_by_name))
        extra = sorted(set(score_by_name) - set(roster_by_name))
        raise ValueError(
            f"prejoin/source active tensor set mismatch: missing={missing}, extra={extra}"
        )
    if int(receipt.get("tensor_count", -1)) != ACTIVE_TENSOR_COUNT:
        raise ValueError("prejoin receipt tensor count is not the exact active cohort size")
    digest = tensor_name_set_sha256(set(roster_by_name))
    if require_sha256(receipt.get("tensor_name_set_sha256"), "prejoin tensor-name digest") != digest:
        raise ValueError("prejoin receipt tensor-name set digest is incorrect")

    allowed_families = set(manifest.get("allowed_families", []))
    manifest_family_counts = manifest.get("family_counts")
    if not allowed_families or not isinstance(manifest_family_counts, dict):
        raise ValueError("source roster family contract is incomplete")
    derived_family_counts: Counter[str] = Counter()
    reported_scores = list(receipt["reported_score_columns"])
    score_columns = list(receipt["score_columns"])
    for name, roster_row in roster_by_name.items():
        if roster_row.get("model_id") != MODEL_ID:
            raise ValueError(f"source roster model mismatch for {name}")
        family = str(roster_row.get("canonical_family", ""))
        if family not in allowed_families:
            raise ValueError(f"source roster family is not allowed for {name}: {family}")
        derived_family_counts[family] += 1
        shape = normalized_shape(roster_row.get("shape"), f"source roster shape for {name}")
        if len(shape) != 2 or int(roster_row.get("n_params", -1)) != math.prod(shape):
            raise ValueError(f"source roster shape/parameter count mismatch for {name}")
        score_row = score_by_name[name]
        leaked = sorted(FORBIDDEN_LABEL_KEYS.intersection(score_row))
        if leaked:
            raise ValueError(f"sealed prejoin score row leaks label fields for {name}: {leaked}")
        structural = {
            "model_id": MODEL_ID,
            "architecture": roster_row.get("architecture"),
            "canonical_family": family,
            "layer_id": int(roster_row.get("layer_id", -1)),
            "n_params": int(roster_row["n_params"]),
        }
        for key, expected in structural.items():
            if score_row.get(key) != expected:
                raise ValueError(f"prejoin/source structural mismatch for {name}: {key}")
        if normalized_shape(score_row.get("shape"), f"prejoin shape for {name}") != shape:
            raise ValueError(f"prejoin/source shape mismatch for {name}")
        for column in score_columns:
            if column not in score_row:
                raise ValueError(f"prejoin score column {column} is missing for {name}")
            try:
                value = float(score_row[column])
            except (TypeError, ValueError) as error:
                raise ValueError(f"prejoin score {column} is not numeric for {name}") from error
            if not math.isfinite(value):
                raise ValueError(f"prejoin score {column} is not finite for {name}")
        if not set(reported_scores).issubset(score_row):
            raise ValueError(f"prejoin row omits a frozen candidate for {name}")
    if dict(sorted(derived_family_counts.items())) != {
        str(key): int(value) for key, value in sorted(manifest_family_counts.items())
    }:
        raise ValueError("source roster family counts differ from its manifest")

    if not isinstance(shards, list) or len(shards) != 4:
        raise ValueError("sealed prejoin shard scores must contain exactly four shards")
    for shard_index, shard in enumerate(shards):
        if not isinstance(shard, dict) or set(shard) != set(roster_by_name):
            raise ValueError(f"shard {shard_index} does not cover the exact active tensor set")
        for name, values in shard.items():
            if not isinstance(values, dict) or not values:
                raise ValueError(f"shard {shard_index} score map is empty for {name}")
            for key, raw in values.items():
                if key in FORBIDDEN_LABEL_KEYS:
                    raise ValueError(f"shard {shard_index} leaks label field {key} for {name}")
                try:
                    value = float(raw)
                except (TypeError, ValueError) as error:
                    raise ValueError(f"shard {shard_index} score {key} is not numeric for {name}") from error
                if not math.isfinite(value):
                    raise ValueError(f"shard {shard_index} score {key} is not finite for {name}")
    return roster, scores, shards


def validate_label_value_consistency(row: dict[str, str], name: str) -> None:
    qtype_ud = row["qtype_ud"]
    qtype_static = row["qtype_static"]
    if qtype_ud not in ORDINAL or qtype_static not in ORDINAL:
        raise ValueError(f"unsupported paired qtype for active label {name}")
    expected_ud_ordinal = ORDINAL[qtype_ud]
    expected_static_ordinal = ORDINAL[qtype_static]
    if int(row["ordinal_ud"]) != expected_ud_ordinal or int(row["ordinal_static"]) != expected_static_ordinal:
        raise ValueError(f"qtype/ordinal inconsistency for active label {name}")
    if not math.isclose(float(row["packed_bpp_ud"]), PACKED_BPP[qtype_ud], abs_tol=1e-12):
        raise ValueError(f"UD qtype/bpp inconsistency for active label {name}")
    if not math.isclose(float(row["packed_bpp_static"]), PACKED_BPP[qtype_static], abs_tol=1e-12):
        raise ValueError(f"static qtype/bpp inconsistency for active label {name}")
    delta_ordinal = expected_ud_ordinal - expected_static_ordinal
    delta_bpp = PACKED_BPP[qtype_ud] - PACKED_BPP[qtype_static]
    if int(row["promotion_delta_ordinal"]) != delta_ordinal:
        raise ValueError(f"promotion ordinal inconsistency for active label {name}")
    if not math.isclose(float(row["promotion_delta_bpp"]), delta_bpp, abs_tol=1e-12):
        raise ValueError(f"promotion bpp inconsistency for active label {name}")
    expected_promoted = "true" if delta_ordinal > 0 else "false"
    if row["promoted"] != expected_promoted:
        raise ValueError(f"promotion boolean inconsistency for active label {name}")
    expected_floor_label = "true" if qtype_ud in {"Q5_K", "Q6_K", "Q8_0"} else "false"
    if row["promoted_vs_4bit_floor"] != expected_floor_label:
        raise ValueError(f"4-bit-floor label inconsistency for active label {name}")


def filter_previously_sealed_labels(
    original_labels_path: Path,
    original_audit_path: Path,
    roster: list[dict[str, Any]],
) -> tuple[list[str], list[dict[str, str]], dict[str, Any], dict[str, Any]]:
    """Open label-side artifacts only after score-side authorization has passed."""
    original_audit = read_json(original_audit_path)
    fields, rows = read_label_tsv(original_labels_path)
    if not isinstance(original_audit, dict):
        raise ValueError("original label audit must be a JSON object")
    missing_fields = sorted(set(LABEL_REQUIRED_FIELDS) - set(fields))
    if missing_fields:
        raise ValueError(f"original label TSV is missing fields: {missing_fields}")
    if original_audit.get("model_id") != MODEL_ID:
        raise ValueError(f"original label audit does not name {MODEL_ID}")

    eligible_rows = [row for row in rows if row.get("eligible") == "true"]
    if len(eligible_rows) != ORIGINAL_PHYSICAL_ELIGIBLE_COUNT:
        raise ValueError(
            "original physical label cohort must contain exactly "
            f"{ORIGINAL_PHYSICAL_ELIGIBLE_COUNT} eligible tensors"
        )
    if int(original_audit.get("eligible_core_count", -1)) != len(eligible_rows):
        raise ValueError("original label audit eligible count differs from the label TSV")
    eligible_by_name = require_unique_names(eligible_rows, "original eligible label TSV")
    roster_by_name = require_unique_names(roster, "active source roster")
    if not set(roster_by_name).issubset(eligible_by_name):
        missing = sorted(set(roster_by_name) - set(eligible_by_name))
        raise ValueError(f"active source tensors are missing from the original labels: {missing}")

    audited_family_counts_raw = original_audit.get("eligible_core_family_counts")
    if not isinstance(audited_family_counts_raw, dict):
        raise ValueError("original label audit family counts are missing")
    original_family_counts = dict(
        sorted(Counter(row["canonical_family"] for row in eligible_rows).items())
    )
    audited_family_counts = {
        str(key): int(value)
        for key, value in sorted(audited_family_counts_raw.items())
    }
    if original_family_counts != audited_family_counts:
        raise ValueError("original label family counts differ from their audit")

    pair = original_audit.get("paired_static_q4_k_m")
    if not isinstance(pair, dict):
        raise ValueError("original label audit lacks paired Q4_K_M evidence")
    if (
        pair.get("status") != "paired_exact_tensor_name_and_shape"
        or pair.get("admission_use") != "eligible"
        or float(pair.get("eligible_coverage", 0.0)) != 1.0
        or int(pair.get("eligible_paired_count", -1)) != len(eligible_rows)
        or not bool(pair.get("cohort_metadata_exact_match", False))
        or list(pair.get("pairing_errors", []))
    ):
        raise ValueError("original paired Q4_K_M label cohort is not exact and admission eligible")
    prior_order = original_audit.get("lockbox_order_audit")
    if not isinstance(prior_order, dict) or prior_order.get("status") != "order verified before invoking gguf-dump":
        raise ValueError("original Gemma labels lack their one-shot GGUF lockbox order evidence")

    selected: list[dict[str, str]] = []
    active_names = set(roster_by_name)
    for row in rows:
        name = row.get("hf_name", "")
        if name not in active_names:
            continue
        if row.get("eligible") != "true" or row.get("label_mode") != "paired_same_cohort_q4_k_m":
            raise ValueError(f"active label is not eligible and exactly paired: {name}")
        roster_row = roster_by_name[name]
        checks = {
            "model_id": MODEL_ID,
            "architecture": str(roster_row["architecture"]),
            "canonical_family": str(roster_row["canonical_family"]),
            "layer_id": str(int(roster_row["layer_id"])),
            "n_params": str(int(roster_row["n_params"])),
        }
        for key, expected in checks.items():
            if row.get(key) != expected:
                raise ValueError(f"active source/label structural mismatch for {name}: {key}")
        roster_shape = normalized_shape(roster_row["shape"], f"source shape for {name}")
        if normalized_shape(row.get("shape"), f"label shape for {name}") != roster_shape:
            raise ValueError(f"active source/label shape mismatch for {name}")
        if normalized_shape(row.get("hf_shape"), f"label HF shape for {name}") != roster_shape:
            raise ValueError(f"active label HF shape mismatch for {name}")
        if row.get("exclusion_reason") or str(row.get("shape_status", "")).startswith("fatal"):
            raise ValueError(f"active label is structurally excluded or fatal: {name}")
        validate_label_value_consistency(row, name)
        selected.append(row)
    if len(selected) != ACTIVE_TENSOR_COUNT or {row["hf_name"] for row in selected} != active_names:
        raise ValueError("filtered labels do not equal the exact 258-tensor active cohort")
    return fields, selected, original_audit, {
        "original_eligible_count": len(eligible_rows),
        "original_family_counts": original_family_counts,
        "excluded_names": sorted(set(eligible_by_name) - active_names),
        "active_family_counts": dict(
            sorted(Counter(row["canonical_family"] for row in selected).items())
        ),
    }


def build_active_label_view(
    *,
    original_labels_path: Path,
    original_audit_path: Path,
    source_roster_path: Path,
    source_manifest_path: Path,
    scores_path: Path,
    shard_scores_path: Path,
    prejoin_receipt_path: Path,
    candidate_freeze_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    paths = {
        "original_labels": original_labels_path.expanduser().resolve(),
        "original_label_audit_summary": original_audit_path.expanduser().resolve(),
        "source_roster": source_roster_path.expanduser().resolve(),
        "source_roster_manifest": source_manifest_path.expanduser().resolve(),
        "prejoin_scores": scores_path.expanduser().resolve(),
        "prejoin_shard_scores": shard_scores_path.expanduser().resolve(),
        "prejoin_receipt": prejoin_receipt_path.expanduser().resolve(),
        "qwen_candidate_freeze": candidate_freeze_path.expanduser().resolve(),
    }
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists():
        raise ValueError(f"refusing to reuse active-label-view output directory: {output_dir}")
    for label, path in paths.items():
        if not path.is_file():
            raise ValueError(f"missing {label}: {path}")

    freeze, receipt, authorization = verify_lockbox_authorization(
        paths["qwen_candidate_freeze"],
        paths["prejoin_receipt"],
        paths["prejoin_scores"],
        paths["prejoin_shard_scores"],
    )
    roster, _, _ = verify_source_and_scores(
        paths["source_roster"],
        paths["source_roster_manifest"],
        paths["prejoin_scores"],
        paths["prejoin_shard_scores"],
        receipt,
    )

    # This is deliberately the first label-side read in the workflow.
    fields, selected, original_audit, label_audit = filter_previously_sealed_labels(
        paths["original_labels"], paths["original_label_audit_summary"], roster
    )
    original_labels_hash = sha256_file(paths["original_labels"])
    original_audit_hash = sha256_file(paths["original_label_audit_summary"])
    roster_hash = sha256_file(paths["source_roster"])
    roster_manifest_hash = sha256_file(paths["source_roster_manifest"])
    prejoin_receipt_hash = sha256_file(paths["prejoin_receipt"])
    freeze_hash = sha256_file(paths["qwen_candidate_freeze"])
    created_at = dt.datetime.now(dt.timezone.utc).isoformat()

    active_audit = deepcopy(original_audit)
    active_audit["eligible_core_count"] = ACTIVE_TENSOR_COUNT
    active_audit["eligible_core_family_counts"] = label_audit["active_family_counts"]
    paired = deepcopy(original_audit["paired_static_q4_k_m"])
    paired["eligible_paired_count"] = ACTIVE_TENSOR_COUNT
    paired["eligible_coverage"] = 1.0
    paired["admission_use"] = "eligible"
    active_audit["paired_static_q4_k_m"] = paired
    active_audit["lockbox_order_audit"] = {
        "status": LOCKBOX_STATUS,
        "verified_at_utc": created_at,
        "workspace_git_head": authorization["workspace_git_head"],
        "qwen_candidate_freeze_path": str(paths["qwen_candidate_freeze"]),
        "qwen_candidate_freeze_sha256": freeze_hash,
        "qwen_candidate_freeze_created_at_utc": authorization["freeze_timestamp"],
        "prejoin_score_receipt_path": str(paths["prejoin_receipt"]),
        "prejoin_score_receipt_sha256": prejoin_receipt_hash,
        "prejoin_score_receipt_created_at_utc": authorization["prejoin_timestamp"],
        "sealed_score_table_path": str(paths["prejoin_scores"]),
        "sealed_score_table_sha256": authorization["score_table_sha256"],
        "sealed_shard_scores_path": str(paths["prejoin_shard_scores"]),
        "sealed_shard_scores_sha256": authorization["shard_scores_sha256"],
        "source_roster_path": str(paths["source_roster"]),
        "source_roster_sha256": roster_hash,
        "source_roster_manifest_path": str(paths["source_roster_manifest"]),
        "source_roster_manifest_sha256": roster_manifest_hash,
        "reported_score_columns": authorization["reported_score_columns"],
        "candidate_score_columns": authorization["reported_score_columns"],
        "winner_eligible_score_columns": authorization["winner_eligible_score_columns"],
        "secondary_score_columns": authorization["secondary_score_columns"],
        "formula_sha256": authorization["formula_sha256"],
        "threshold_sha256": authorization["threshold_sha256"],
        "execution_settings": authorization["prejoin_execution_settings"],
        "c5_execution_settings": authorization["c5_execution_settings"],
        "input_hash_comparison": authorization["input_hash_comparison"],
        "implementation_hash_comparison": authorization["implementation_hash_comparison"],
        "previously_sealed_label_source": {
            "original_labels_path": str(paths["original_labels"]),
            "original_labels_sha256": original_labels_hash,
            "original_label_audit_summary_path": str(paths["original_label_audit_summary"]),
            "original_label_audit_summary_sha256": original_audit_hash,
            "prior_lockbox_order_audit": deepcopy(original_audit["lockbox_order_audit"]),
        },
        "existing_labels_were_previously_sealed_and_opened": True,
        "gguf_reopened": False,
        "score_formulas_or_thresholds_changed": False,
    }
    active_audit["active_source_scope_derivation"] = {
        "schema_version": "importance-score-active-label-view-v0.2",
        "created_at_utc": created_at,
        "status": (
            "derived solely from exact label-blind source-roster and sealed prejoin hf_name "
            "membership; no label-value selection"
        ),
        "selection_basis": "exact common hf_name set of the active source roster and sealed C5 prejoin",
        "active_source_tensor_count": ACTIVE_TENSOR_COUNT,
        "original_physical_eligible_core_count": label_audit["original_eligible_count"],
        "original_physical_eligible_core_family_counts": label_audit["original_family_counts"],
        "excluded_inactive_shared_kv_count": len(label_audit["excluded_names"]),
        "excluded_inactive_shared_kv_names": label_audit["excluded_names"],
        "qwen_candidate_freeze_sha256": freeze_hash,
        "prejoin_receipt_sha256": prejoin_receipt_hash,
        "prejoin_score_table_sha256": authorization["score_table_sha256"],
        "prejoin_shard_scores_sha256": authorization["shard_scores_sha256"],
        "source_roster_sha256": roster_hash,
        "source_roster_manifest_sha256": roster_manifest_hash,
        "original_labels_sha256": original_labels_hash,
        "original_label_audit_summary_sha256": original_audit_hash,
        "existing_labels_were_previously_sealed_and_opened": True,
        "gguf_reopened": False,
        "score_formulas_or_thresholds_changed": False,
    }

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(exist_ok=False)
    labels_output = output_dir / "ud-tensor-labels-active.tsv"
    audit_output = output_dir / "ud-label-audit-summary-active.json"
    receipt_output = output_dir / "active-label-view.receipt.json"
    write_tsv(labels_output, fields, selected)
    write_json(audit_output, active_audit)
    output_hashes = {
        "active_labels": sha256_file(labels_output),
        "active_label_audit_summary": sha256_file(audit_output),
    }
    output_receipt = {
        "schema_version": "importance-score-active-label-view-receipt-v0.2",
        "created_at_utc": created_at,
        "status": "sealed one-shot C5 score-label join against previously sealed GGUF labels",
        "model_id": MODEL_ID,
        "workspace_git_head": authorization["workspace_git_head"],
        "selection_basis": "exact source-roster and sealed prejoin hf_name set only",
        "one_shot_join": True,
        "active_eligible_count": ACTIVE_TENSOR_COUNT,
        "active_family_counts": label_audit["active_family_counts"],
        "original_eligible_count": label_audit["original_eligible_count"],
        "excluded_count": len(label_audit["excluded_names"]),
        "excluded_names": label_audit["excluded_names"],
        "reported_score_columns": authorization["reported_score_columns"],
        "winner_eligible_score_columns": authorization["winner_eligible_score_columns"],
        "secondary_score_columns": authorization["secondary_score_columns"],
        "formula_sha256": authorization["formula_sha256"],
        "threshold_sha256": authorization["threshold_sha256"],
        "existing_labels_were_previously_sealed_and_opened": True,
        "gguf_reopened": False,
        "score_formulas_or_thresholds_changed": False,
        "input_paths": {name: str(path) for name, path in paths.items()},
        "input_hashes": {
            "original_labels": original_labels_hash,
            "original_label_audit_summary": original_audit_hash,
            "source_roster": roster_hash,
            "source_roster_manifest": roster_manifest_hash,
            "prejoin_scores": authorization["score_table_sha256"],
            "prejoin_shard_scores": authorization["shard_scores_sha256"],
            "prejoin_receipt": prejoin_receipt_hash,
            "qwen_candidate_freeze": freeze_hash,
        },
        "output_paths": {
            "active_labels": str(labels_output),
            "active_label_audit_summary": str(audit_output),
        },
        "output_hashes": output_hashes,
        "hash_chain": {
            "qwen_candidate_freeze_sha256": freeze_hash,
            "prejoin_receipt_sha256": prejoin_receipt_hash,
            "sealed_score_table_sha256": authorization["score_table_sha256"],
            "sealed_shard_scores_sha256": authorization["shard_scores_sha256"],
            "source_roster_sha256": roster_hash,
            "original_labels_sha256": original_labels_hash,
            **output_hashes,
        },
    }
    write_json(receipt_output, output_receipt)
    return output_receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-labels", type=Path, required=True)
    parser.add_argument("--original-label-audit-summary", type=Path, required=True)
    parser.add_argument("--source-roster", type=Path, required=True)
    parser.add_argument("--source-roster-manifest", type=Path, required=True)
    parser.add_argument("--prejoin-scores", type=Path, required=True)
    parser.add_argument("--prejoin-shard-scores", type=Path, required=True)
    parser.add_argument("--prejoin-receipt", type=Path, required=True)
    parser.add_argument("--qwen-candidate-freeze", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        receipt = build_active_label_view(
            original_labels_path=args.original_labels,
            original_audit_path=args.original_label_audit_summary,
            source_roster_path=args.source_roster,
            source_manifest_path=args.source_roster_manifest,
            scores_path=args.prejoin_scores,
            shard_scores_path=args.prejoin_shard_scores,
            prejoin_receipt_path=args.prejoin_receipt,
            candidate_freeze_path=args.qwen_candidate_freeze,
            output_dir=args.output_dir,
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error
    print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
