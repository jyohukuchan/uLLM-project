#!/usr/bin/env python3
"""Build and audit Qwen/Gemma Unsloth Dynamic tensor labels read-only.

The script consumes ``gguf-dump --json`` and safetensors headers only.  It
never dequantizes GGUF payloads or modifies model files.  When an explicit
same-cohort Q4_K_M is supplied, tensor names and shapes are paired exactly and
the ordinal/bpp promotion deltas are emitted.  A missing or mismatched baseline
is recorded as an explicit pairing failure rather than silently replaced by a
fallback label.

For a Gemma lockbox model, the tool refuses to invoke ``gguf-dump`` until it
has verified both the sealed Qwen candidate freeze and the source-only score
receipt, including their common implementation hashes and execution order.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

from safetensors import safe_open


UNKNOWN = "unknown"
ORDINAL = {"IQ4_XS": 0, "Q4_K": 0, "Q5_K": 1, "Q6_K": 2, "Q8_0": 3}
ESCAPE_ORDINAL = {**ORDINAL, "F16": 4, "F32": 5}
PACKED_BPP = {
    "IQ4_XS": 4.25,
    "Q4_K": 4.5,
    "Q5_K": 5.5,
    "Q6_K": 6.5625,
    "Q8_0": 8.5,
    "F16": 16.0,
    "F32": 32.0,
}
COHORT_METADATA_KEYS = (
    "general.architecture",
    "general.base_model.0.name",
    "general.base_model.0.organization",
    "general.base_model.0.repo_url",
    "general.basename",
    "general.name",
    "general.quantization_version",
    "general.quantized_by",
    "quantize.imatrix.file",
    "quantize.imatrix.dataset",
    "quantize.imatrix.entries_count",
    "quantize.imatrix.chunks_count",
)
CORE_FAMILIES = {
    "attn_q",
    "attn_k",
    "attn_v",
    "attn_o",
    "linear_attn_qkv",
    "linear_attn_a",
    "linear_attn_b",
    "linear_attn_z",
    "linear_attn_out",
    "mlp_gate",
    "mlp_up",
    "mlp_down",
}


def shape_json(shape: list[int] | tuple[int, ...] | None) -> str:
    return json.dumps(list(shape) if shape is not None else None, separators=(",", ":"))


def stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return str(value)


def write_tsv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: stringify(row.get(field)) for field in fields})


def load_safetensor_headers(model_dir: Path) -> dict[str, dict[str, Any]]:
    index_path = model_dir / "model.safetensors.index.json"
    by_file: dict[str, list[str]] = {}
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        for name, filename in index["weight_map"].items():
            by_file.setdefault(filename, []).append(name)
    else:
        for path in sorted(model_dir.glob("*.safetensors")):
            with safe_open(path, framework="pt", device="cpu") as handle:
                by_file[path.name] = list(handle.keys())
        if not by_file:
            raise FileNotFoundError(f"no safetensors weights under {model_dir}")
    headers: dict[str, dict[str, Any]] = {}
    for filename, names in sorted(by_file.items()):
        with safe_open(model_dir / filename, framework="pt", device="cpu") as handle:
            for name in names:
                view = handle.get_slice(name)
                headers[name] = {
                    "shape": [int(value) for value in view.get_shape()],
                    "dtype": str(view.get_dtype()),
                    "file": filename,
                }
    return headers


def gguf_to_hf(gguf_name: str) -> tuple[str | None, str, int | None, str]:
    """Return HF name, canonical family, layer ID, and semantic transform note."""
    globals_map = {
        "token_embd.weight": ("model.language_model.embed_tokens.weight", "token_embd", None, "identity"),
        "output_norm.weight": ("model.language_model.norm.weight", "output_norm", None, "HF_plus_1_for_additive_RMSNorm"),
        "output.weight": ("lm_head.weight", "output", None, "identity"),
    }
    if gguf_name in globals_map:
        return globals_map[gguf_name]
    match = re.fullmatch(r"blk\.(\d+)\.(.+)", gguf_name)
    if match is None:
        return None, "unknown", None, "unknown"
    layer_id = int(match.group(1))
    suffix = match.group(2)
    prefix = f"model.language_model.layers.{layer_id}."
    mapping = {
        "attn_norm.weight": ("input_layernorm.weight", "attn_norm", "HF_plus_1_for_additive_RMSNorm"),
        "post_attention_norm.weight": ("post_attention_layernorm.weight", "post_attention_norm", "HF_plus_1_for_additive_RMSNorm"),
        "ffn_down.weight": ("mlp.down_proj.weight", "mlp_down", "identity"),
        "ffn_gate.weight": ("mlp.gate_proj.weight", "mlp_gate", "identity"),
        "ffn_up.weight": ("mlp.up_proj.weight", "mlp_up", "identity"),
        "attn_q.weight": ("self_attn.q_proj.weight", "attn_q", "identity"),
        "attn_k.weight": ("self_attn.k_proj.weight", "attn_k", "identity"),
        "attn_v.weight": ("self_attn.v_proj.weight", "attn_v", "identity"),
        "attn_output.weight": ("self_attn.o_proj.weight", "attn_o", "identity"),
        "attn_q_norm.weight": ("self_attn.q_norm.weight", "attn_q_norm", "HF_plus_1_for_additive_RMSNorm"),
        "attn_k_norm.weight": ("self_attn.k_norm.weight", "attn_k_norm", "HF_plus_1_for_additive_RMSNorm"),
        "attn_gate.weight": ("linear_attn.in_proj_z.weight", "linear_attn_z", "V_head_reorder"),
        "attn_qkv.weight": ("linear_attn.in_proj_qkv.weight", "linear_attn_qkv", "V_head_reorder"),
        "ssm_a": ("linear_attn.A_log", "linear_attn_A_log", "GGUF_equals_negative_exp_HF"),
        "ssm_alpha.weight": ("linear_attn.in_proj_a.weight", "linear_attn_a", "V_head_reorder"),
        "ssm_beta.weight": ("linear_attn.in_proj_b.weight", "linear_attn_b", "V_head_reorder"),
        "ssm_conv1d.weight": ("linear_attn.conv1d.weight", "linear_attn_conv", "HF_squeeze_dim1_then_transpose_and_V_head_reorder"),
        "ssm_dt.bias": ("linear_attn.dt_bias", "linear_attn_dt_bias", "V_head_reorder"),
        "ssm_norm.weight": ("linear_attn.norm.weight", "linear_attn_norm", "identity"),
        "ssm_out.weight": ("linear_attn.out_proj.weight", "linear_attn_out", "V_head_reorder"),
    }
    item = mapping.get(suffix)
    if item is None:
        return None, "unknown", layer_id, "unknown"
    suffix_hf, family, transform = item
    return prefix + suffix_hf, family, layer_id, transform


def shape_status(gguf_shape: list[int], hf_shape: list[int], gguf_name: str) -> str:
    if gguf_name.endswith(".ssm_conv1d.weight"):
        return "fatal_rank3_singleton_axis_shape_mismatch"
    if len(gguf_shape) == 2 and list(reversed(gguf_shape)) == hf_shape:
        return "logical_match_ggml_ne_reversed"
    if gguf_shape == hf_shape:
        return "logical_match"
    return "fatal_shape_mismatch"


def fallback_label(qtype: str) -> bool:
    return qtype in {"Q5_K", "Q6_K", "Q8_0"}


def metadata_values(gguf: dict[str, Any]) -> dict[str, Any]:
    metadata = gguf.get("metadata", {})
    return {
        key: metadata[key].get("value")
        for key in COHORT_METADATA_KEYS
        if key in metadata and isinstance(metadata[key], dict)
    }


def static_candidates(root: Path) -> list[str]:
    """Read-only, deliberately strict local static-baseline search."""
    candidates: list[str] = []
    for path in root.rglob("*.gguf"):
        name = path.name.lower()
        if "qwen3.5-9b" in name and "q4_k_m" in name:
            candidates.append(str(path))
    return sorted(candidates)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_lockbox_order(
    model_id: str, candidate_freeze_path: Path, score_receipt_path: Path
) -> dict[str, Any]:
    freeze = json.loads(candidate_freeze_path.read_text(encoding="utf-8"))
    receipt = json.loads(score_receipt_path.read_text(encoding="utf-8"))
    if freeze.get("status") != "sealed before any Gemma tensor-level score/label join":
        raise ValueError("Qwen candidate freeze has an unexpected status")
    if freeze.get("lockbox_model") != model_id:
        raise ValueError(
            f"Qwen candidate freeze names {freeze.get('lockbox_model')}, not {model_id}"
        )
    if receipt.get("status") != (
        "sealed score table generated without accepting or opening a GGUF label manifest"
    ):
        raise ValueError("prejoin score receipt has an unexpected status")
    if receipt.get("model_id") != model_id:
        raise ValueError(f"prejoin score receipt names {receipt.get('model_id')}, not {model_id}")
    score_path = Path(str(receipt.get("score_table_path", ""))).expanduser().resolve()
    if not score_path.is_file() or sha256_file(score_path) != receipt.get("score_table_sha256"):
        raise ValueError("sealed prejoin score table is missing or differs from its receipt")
    freeze_hashes = freeze.get("implementation_hashes", {})
    receipt_hashes = receipt.get("implementation_hashes", {})
    compared = {}
    for name in (
        "build-importance-score-prejoin.py",
        "report-importance-score-formal.py",
        "run-aq-tensor-sample.py",
        "score-block-covariance-c1.py",
        "run-importance-single-tensor-perturbation.py",
    ):
        compared[name] = {
            "candidate_freeze": freeze_hashes.get(name),
            "prejoin_receipt": receipt_hashes.get(name),
        }
        if not compared[name]["candidate_freeze"] or (
            compared[name]["candidate_freeze"] != compared[name]["prejoin_receipt"]
        ):
            raise ValueError(f"lockbox implementation hash mismatch for {name}")
    if str(freeze.get("created_at_utc", "")) >= str(receipt.get("created_at_utc", "")):
        raise ValueError("prejoin score receipt is not later than the Qwen candidate freeze")
    return {
        "status": "order verified before invoking gguf-dump",
        "qwen_candidate_freeze_path": str(candidate_freeze_path),
        "qwen_candidate_freeze_sha256": sha256_file(candidate_freeze_path),
        "prejoin_score_receipt_path": str(score_receipt_path),
        "prejoin_score_receipt_sha256": sha256_file(score_receipt_path),
        "sealed_score_table_path": str(score_path),
        "sealed_score_table_sha256": receipt["score_table_sha256"],
        "implementation_hash_comparison": compared,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gguf-path", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-id", default="qwen3.5-9b")
    parser.add_argument("--gguf-dump", type=Path, default=Path("/home/homelab1/hf_venv/bin/gguf-dump"))
    parser.add_argument(
        "--static-gguf-path",
        type=Path,
        help="Explicit same-repository/revision Q4_K_M baseline; no heuristic candidate is auto-selected.",
    )
    parser.add_argument("--static-search-root", type=Path, default=Path("/home/homelab1/datapool/ai_models"))
    parser.add_argument("--qwen-candidate-freeze", type=Path)
    parser.add_argument("--prejoin-score-receipt", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    gguf_path = args.gguf_path.expanduser().resolve()
    model_dir = args.model_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    gguf_dump = args.gguf_dump.expanduser().resolve()
    static_root = args.static_search_root.expanduser().resolve()
    is_gemma_lockbox = args.model_id.lower().startswith("gemma")
    if is_gemma_lockbox and (
        args.qwen_candidate_freeze is None or args.prejoin_score_receipt is None
    ):
        raise SystemExit(
            "Gemma label extraction requires both --qwen-candidate-freeze and "
            "--prejoin-score-receipt"
        )
    lockbox_order_audit = None
    if is_gemma_lockbox:
        lockbox_order_audit = verify_lockbox_order(
            args.model_id,
            args.qwen_candidate_freeze.expanduser().resolve(),
            args.prejoin_score_receipt.expanduser().resolve(),
        )
    for path, label in ((gguf_path, "GGUF"), (model_dir, "model directory"), (gguf_dump, "gguf-dump")):
        if not path.exists():
            raise SystemExit(f"missing {label}: {path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    if is_gemma_lockbox:
        existing = [
            output_dir / name
            for name in (
                "ud-tensor-labels.tsv",
                "tensor-name-map.tsv",
                "eligibility-audit.tsv",
                "ud-label-audit-summary.json",
            )
            if (output_dir / name).exists()
        ]
        if existing:
            raise SystemExit(f"refusing to overwrite Gemma lockbox label artifacts: {existing}")

    dumped = subprocess.run(
        [str(gguf_dump), "--json", str(gguf_path)], check=True, text=True, capture_output=True
    )
    gguf = json.loads(dumped.stdout)
    tensors: dict[str, dict[str, Any]] = gguf["tensors"]
    static_path = args.static_gguf_path.expanduser().resolve() if args.static_gguf_path else None
    static_tensors: dict[str, dict[str, Any]] | None = None
    static_gguf: dict[str, Any] | None = None
    static_pair_errors: list[str] = []
    if static_path is not None:
        if not static_path.is_file():
            raise SystemExit(f"missing explicit static GGUF: {static_path}")
        static_dumped = subprocess.run(
            [str(gguf_dump), "--json", str(static_path)], check=True, text=True, capture_output=True
        )
        static_gguf = json.loads(static_dumped.stdout)
        static_tensors = static_gguf["tensors"]
        missing_in_static = sorted(set(tensors) - set(static_tensors))
        extra_in_static = sorted(set(static_tensors) - set(tensors))
        if missing_in_static:
            static_pair_errors.append(f"missing_tensor_names:{len(missing_in_static)}")
        if extra_in_static:
            static_pair_errors.append(f"extra_tensor_names:{len(extra_in_static)}")
        for name in sorted(set(tensors) & set(static_tensors)):
            if [int(value) for value in tensors[name]["shape"]] != [
                int(value) for value in static_tensors[name]["shape"]
            ]:
                static_pair_errors.append(f"shape_mismatch:{name}")
    hf_headers = load_safetensor_headers(model_dir)
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    text_config = config.get("text_config", {})
    architecture = str(text_config.get("model_type") or config.get("architectures", [UNKNOWN])[0])
    static = static_candidates(static_root)

    labels: list[dict[str, Any]] = []
    name_map: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    mapped_hf: set[str] = set()
    duplicate_hf: set[str] = set()

    for gguf_name in sorted(tensors):
        info = tensors[gguf_name]
        gguf_shape = [int(value) for value in info["shape"]]
        qtype = str(info["type"])
        hf_name, family, layer_id, transform = gguf_to_hf(gguf_name)
        hf_info = hf_headers.get(hf_name) if hf_name else None
        if hf_name is None:
            mapping_status = "fatal_unmatched_gguf"
            shape_state = "fatal_unmatched_gguf"
            hf_shape: list[int] | None = None
            exclusion = "unmatched_gguf_name"
        elif hf_info is None:
            mapping_status = "fatal_unmatched_hf"
            shape_state = "fatal_unmatched_hf"
            hf_shape = None
            exclusion = "mapped_hf_name_not_present"
        else:
            hf_shape = list(hf_info["shape"])
            if hf_name in mapped_hf:
                duplicate_hf.add(hf_name)
                mapping_status = "fatal_duplicate_hf_target"
                shape_state = "fatal_duplicate_hf_target"
                exclusion = "duplicate_hf_target"
            else:
                mapped_hf.add(hf_name)
                mapping_status = "mapped"
                shape_state = shape_status(gguf_shape, hf_shape, gguf_name)
                exclusion = ""

        is_quantized_core_type = qtype in ORDINAL
        matrix_hf = hf_shape is not None and len(hf_shape) == 2
        is_global = family in {"token_embd", "output"}
        if mapping_status != "mapped":
            eligible = False
        elif shape_state.startswith("fatal"):
            eligible = False
            exclusion = exclusion or "source_rank_3_serialized_squeezed_axis"
        elif not matrix_hf:
            eligible = False
            exclusion = "not_2d_weight_matrix"
        elif not is_quantized_core_type:
            eligible = False
            exclusion = "non_core_quantized_type_secondary_or_unsupported"
        elif is_global:
            eligible = False
            exclusion = "global_only_no_repeated_layer_ranking"
        elif family not in CORE_FAMILIES:
            eligible = False
            exclusion = "not_a_supported_AQ_text_linear_family"
        else:
            eligible = True
            exclusion = ""

        static_info = static_tensors.get(gguf_name) if static_tensors is not None else None
        static_qtype = str(static_info["type"]) if static_info is not None else UNKNOWN
        static_ordinal = ORDINAL.get(static_qtype, UNKNOWN)
        static_bpp = PACKED_BPP.get(static_qtype, UNKNOWN)
        pairable = (
            static_tensors is not None
            and not static_pair_errors
            and static_info is not None
            and qtype in ORDINAL
            and static_qtype in ORDINAL
        )
        delta_ordinal = ORDINAL[qtype] - int(static_ordinal) if pairable else UNKNOWN
        delta_bpp = PACKED_BPP[qtype] - float(static_bpp) if pairable else UNKNOWN
        label = {
            "model_id": args.model_id,
            "architecture": architecture,
            "layer_id": layer_id if layer_id is not None else "global",
            "canonical_family": family,
            "gguf_name": gguf_name,
            "hf_name": hf_name or UNKNOWN,
            "shape": shape_json(hf_shape if hf_shape is not None else gguf_shape),
            "n_params": int(__import__("math").prod(hf_shape if hf_shape is not None else gguf_shape)),
            "qtype_ud": qtype,
            "qtype_static": static_qtype,
            "ordinal_ud": ORDINAL.get(qtype, UNKNOWN),
            "ordinal_static": static_ordinal,
            "packed_bpp_ud": PACKED_BPP.get(qtype, UNKNOWN),
            "packed_bpp_static": static_bpp,
            "promotion_delta_ordinal": delta_ordinal,
            "promotion_delta_bpp": delta_bpp,
            "promoted": (delta_ordinal > 0) if pairable else UNKNOWN,
            "eligible": eligible,
            "exclusion_reason": exclusion,
            "promoted_vs_4bit_floor": fallback_label(qtype),
            "label_mode": "paired_same_cohort_q4_k_m" if pairable else "unpaired_fallback_exploratory",
            "gguf_shape_ne": shape_json(gguf_shape),
            "hf_shape": shape_json(hf_shape),
            "shape_status": shape_state,
            "semantic_transform_note": transform,
        }
        labels.append(label)
        name_map.append(
            {
                "record_type": "gguf_to_hf",
                "gguf_name": gguf_name,
                "hf_name": hf_name or UNKNOWN,
                "gguf_shape_ne": shape_json(gguf_shape),
                "hf_shape": shape_json(hf_shape),
                "mapping_status": mapping_status,
                "shape_status": shape_state,
                "semantic_transform_note": transform,
                "fatal": mapping_status.startswith("fatal") or shape_state.startswith("fatal"),
            }
        )
        audit.append(
            {
                "record_type": "gguf_tensor",
                "name": gguf_name,
                "counterpart": hf_name or UNKNOWN,
                "status": "eligible_core" if eligible else ("fatal" if "fatal" in (mapping_status + shape_state) else "excluded"),
                "reason": exclusion,
                "qtype_ud": qtype,
                "shape": shape_json(hf_shape if hf_shape is not None else gguf_shape),
                "canonical_family": family,
                "layer_id": layer_id if layer_id is not None else "global",
            }
        )

    for hf_name in sorted(hf_headers):
        if hf_name in mapped_hf:
            continue
        reason = "vision_audio_projector_out_of_text_cohort" if hf_name.startswith("model.visual.") else "mtp_not_serialized_by_ud_gguf"
        name_map.append(
            {
                "record_type": "hf_only_scope_excluded",
                "gguf_name": "",
                "hf_name": hf_name,
                "gguf_shape_ne": "",
                "hf_shape": shape_json(hf_headers[hf_name]["shape"]),
                "mapping_status": "scope_excluded",
                "shape_status": "not_applicable",
                "semantic_transform_note": "not_serialized_by_text_only_UD_GGUF",
                "fatal": True,
            }
        )
        audit.append(
            {
                "record_type": "hf_only_scope_excluded",
                "name": hf_name,
                "counterpart": "",
                "status": "fatal",
                "reason": reason,
                "qtype_ud": "",
                "shape": shape_json(hf_headers[hf_name]["shape"]),
                "canonical_family": "scope_excluded",
                "layer_id": "",
            }
        )

    label_fields = [
        "model_id", "architecture", "layer_id", "canonical_family", "gguf_name", "hf_name", "shape", "n_params",
        "qtype_ud", "qtype_static", "ordinal_ud", "ordinal_static", "packed_bpp_ud", "packed_bpp_static",
        "promotion_delta_ordinal", "promotion_delta_bpp", "promoted", "eligible", "exclusion_reason",
        "promoted_vs_4bit_floor", "label_mode", "gguf_shape_ne", "hf_shape", "shape_status", "semantic_transform_note",
    ]
    map_fields = [
        "record_type", "gguf_name", "hf_name", "gguf_shape_ne", "hf_shape", "mapping_status", "shape_status",
        "semantic_transform_note", "fatal",
    ]
    audit_fields = ["record_type", "name", "counterpart", "status", "reason", "qtype_ud", "shape", "canonical_family", "layer_id"]
    write_tsv(output_dir / "ud-tensor-labels.tsv", labels, label_fields)
    write_tsv(output_dir / "tensor-name-map.tsv", name_map, map_fields)
    write_tsv(output_dir / "eligibility-audit.tsv", audit, audit_fields)

    eligible = [row for row in labels if row["eligible"]]
    report = {
        "schema_version": "ud-label-audit-summary-v0.1",
        "model_id": args.model_id,
        "architecture": architecture,
        "lockbox_order_audit": lockbox_order_audit,
        "gguf_tensor_count": len(labels),
        "hf_tensor_count": len(hf_headers),
        "text_only_cohort_mapped": len(mapped_hf),
        "unmatched_gguf": sum(1 for row in labels if row["hf_name"] == UNKNOWN),
        "duplicate_hf_targets": sorted(duplicate_hf),
        "fatal_shape_mismatches": [row["gguf_name"] for row in labels if str(row["shape_status"]).startswith("fatal")],
        "scope_excluded_hf_count": len(hf_headers) - len(mapped_hf),
        "ud_type_counts": dict(sorted(Counter(row["qtype_ud"] for row in labels).items())),
        "eligible_core_count": len(eligible),
        "eligible_core_family_counts": dict(sorted(Counter(str(row["canonical_family"]) for row in eligible).items())),
        "paired_static_q4_k_m": {
            "status": (
                "paired_exact_tensor_name_and_shape"
                if static_tensors is not None and not static_pair_errors
                else ("pairing_failed" if static_tensors is not None else "not_supplied")
            ),
            "explicit_path": str(static_path) if static_path is not None else None,
            "search_root": str(static_root),
            "candidates": static,
            "pairing_errors": static_pair_errors,
            "eligible_paired_count": sum(
                1 for row in eligible if row["label_mode"] == "paired_same_cohort_q4_k_m"
            ),
            "eligible_coverage": (
                sum(1 for row in eligible if row["label_mode"] == "paired_same_cohort_q4_k_m") / len(eligible)
                if eligible else 0.0
            ),
            "admission_use": (
                "eligible" if eligible and all(row["label_mode"] == "paired_same_cohort_q4_k_m" for row in eligible)
                else "HOLD: paired binary teacher incomplete"
            ),
            "teacher_cohort_metadata": metadata_values(gguf),
            "static_cohort_metadata": metadata_values(static_gguf) if static_gguf is not None else None,
            "cohort_metadata_exact_match": (
                metadata_values(gguf) == metadata_values(static_gguf)
                if static_gguf is not None
                else False
            ),
        },
        "fallback": {
            "name": "promoted_vs_4bit_floor",
            "definition": "qtype_ud in {Q5_K, Q6_K, Q8_0}",
            "allowed_use": "exploratory ordinal report only; never AUC/Precision@K admission replacement"
        },
        "fatal_audit_rule": "Every unmatched, duplicate, and shape mismatch is listed in eligibility-audit.tsv; no row is silently skipped.",
        "text_only_scope_rule": "model.language_model.* plus lm_head are the GGUF cohort. model.visual.* and mtp.* remain listed as fatal scope exclusions rather than joined to text-backbone analysis."
    }
    (output_dir / "ud-label-audit-summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
