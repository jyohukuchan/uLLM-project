from __future__ import annotations

import csv
import importlib.util
import json
from collections import Counter
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = ROOT / "tools" / "build-importance-active-label-view.py"


def load_tool():
    spec = importlib.util.spec_from_file_location("test_importance_active_label_view_tool", TOOL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def write_tsv(path: Path, fields: tuple[str, ...], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def build_fixture(tmp_path: Path, tool) -> dict[str, Path]:
    model_id = tool.MODEL_ID
    architecture = "gemma4_text"
    families = ["attn_k", "attn_o", "attn_q", "attn_v", "mlp_down", "mlp_gate", "mlp_up"]
    active_names = [f"model.language_model.layers.{index}.self_attn.q_proj.weight" for index in range(258)]
    all_names = active_names + [
        f"model.language_model.layers.{index}.self_attn.k_proj.inactive.weight"
        for index in range(258, 294)
    ]

    roster_rows = []
    label_rows = []
    for index, name in enumerate(all_names):
        family = families[index % len(families)]
        structural = {
            "model_id": model_id,
            "architecture": architecture,
            "layer_id": index,
            "canonical_family": family,
            "hf_name": name,
            "shape": [2, 4],
            "n_params": 8,
        }
        if index < 258:
            roster_rows.append(structural)
        promoted = index % 2 == 0
        qtype_ud = "Q5_K" if promoted else "Q4_K"
        label_rows.append(
            {
                "model_id": model_id,
                "architecture": architecture,
                "layer_id": str(index),
                "canonical_family": family,
                "gguf_name": f"blk.{index}.attn_q.weight",
                "hf_name": name,
                "shape": "[2,4]",
                "n_params": "8",
                "qtype_ud": qtype_ud,
                "qtype_static": "Q4_K",
                "ordinal_ud": "1" if promoted else "0",
                "ordinal_static": "0",
                "packed_bpp_ud": "5.5" if promoted else "4.5",
                "packed_bpp_static": "4.5",
                "promotion_delta_ordinal": "1" if promoted else "0",
                "promotion_delta_bpp": "1.0" if promoted else "0.0",
                "promoted": "true" if promoted else "false",
                "eligible": "true",
                "exclusion_reason": "",
                "promoted_vs_4bit_floor": "true" if promoted else "false",
                "label_mode": "paired_same_cohort_q4_k_m",
                "gguf_shape_ne": "[4,2]",
                "hf_shape": "[2,4]",
                "shape_status": "logical_match_ggml_ne_reversed",
                "semantic_transform_note": "identity",
            }
        )

    source_roster = tmp_path / "gemma-active.jsonl"
    source_manifest = tmp_path / "gemma-active.manifest.json"
    write_jsonl(source_roster, roster_rows)
    active_family_counts = dict(sorted(Counter(row["canonical_family"] for row in roster_rows).items()))
    write_json(
        source_manifest,
        {
            "schema_version": "importance-score-source-roster-v0.1",
            "status": tool.SOURCE_ROSTER_STATUS,
            "model_id": model_id,
            "roster_path": str(source_roster.resolve()),
            "roster_sha256": tool.sha256_file(source_roster),
            "roster_tensor_count": 258,
            "allowed_families": families,
            "family_counts": active_family_counts,
        },
    )

    reported = [
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
    ]
    secondary = [
        "C5a_Taylor_L1_S",
        "C5a_Taylor_squared_S",
        "C5b_Empirical_Fisher_I",
    ]
    winner = [name for name in reported if name not in secondary]
    score_rows = []
    for index, row in enumerate(roster_rows):
        score_rows.append({**row, **{score: float(index + 1) for score in reported}})
    scores = tmp_path / "scores-prejoin.jsonl"
    shards = tmp_path / "shard-scores-prejoin.json"
    write_jsonl(scores, score_rows)
    write_json(
        shards,
        [
            {name: {"C5a_Taylor_quant_I": float(index + 1)} for index, name in enumerate(active_names)}
            for _ in range(4)
        ],
    )

    common_hashes = {name: "1" * 64 for name in tool.LOCKBOX_SHARED_IMPLEMENTATIONS}
    common_hashes["build-importance-active-label-view.py"] = tool.sha256_file(TOOL_PATH)
    current_git = "a" * 40
    common_inputs = {name: "2" * 64 for name in tool.LOCKBOX_INPUT_HASHES}
    base_settings = {
        "weight_sample_size": 65536,
        "seed": 0,
        "torch_threads": 16,
        "torch_interop_threads": 1,
        "activation_stat_shard_count": 4,
    }
    c5_settings = {
        "empirical": {"sample_count": 128},
        "self_fisher": {"sample_count": 128, "mc_samples": 4},
    }
    formulas = {
        name: {
            "definition": name,
            "winner_eligible": name in winner,
            "secondary": name in secondary,
        }
        for name in reported
    }
    freeze = tmp_path / "qwen-freeze.json"
    write_json(
        freeze,
        {
            "status": tool.FREEZE_STATUS,
            "created_at_utc": "2026-07-21T00:00:00+00:00",
            "development_model": "qwen3.5-9b",
            "lockbox_model": model_id,
            "workspace_git_head": current_git,
            "candidate_scores_transferred_unchanged": reported,
            "reported_score_columns": reported,
            "winner_eligible_score_columns": winner,
            "secondary_score_columns": secondary,
            "formulas": formulas,
            "thresholds": {"within_family_macro_rho": 0.30},
            "execution_settings": {
                "prejoin_score_generation": base_settings,
                "c5_gradient": c5_settings,
            },
            "input_hashes": common_inputs,
            "implementation_hashes": common_hashes,
        },
    )
    prejoin_receipt = tmp_path / "scores-prejoin.receipt.json"
    write_json(
        prejoin_receipt,
        {
            "status": tool.PREJOIN_STATUS,
            "created_at_utc": "2026-07-21T00:00:01+00:00",
            "model_id": model_id,
            "workspace_git_head": current_git,
            "score_table_path": str(scores.resolve()),
            "score_table_sha256": tool.sha256_file(scores),
            "shard_scores_path": str(shards.resolve()),
            "shard_scores_sha256": tool.sha256_file(shards),
            "tensor_count": 258,
            "tensor_name_set_sha256": tool.tensor_name_set_sha256(set(active_names)),
            "candidate_score_columns": reported,
            "reported_score_columns": reported,
            "winner_eligible_score_columns": winner,
            "secondary_score_columns": secondary,
            "score_columns": reported,
            "execution_settings": base_settings,
            "c5_execution_settings": c5_settings,
            "input_hashes": {
                **common_inputs,
                "source_roster": tool.sha256_file(source_roster),
                "source_roster_manifest": tool.sha256_file(source_manifest),
            },
            "implementation_hashes": common_hashes,
        },
    )

    labels = tmp_path / "ud-tensor-labels.tsv"
    write_tsv(labels, tool.LABEL_REQUIRED_FIELDS, label_rows)
    original_family_counts = dict(sorted(Counter(row["canonical_family"] for row in label_rows).items()))
    original_audit = tmp_path / "ud-label-audit-summary.json"
    write_json(
        original_audit,
        {
            "schema_version": "ud-label-audit-summary-v0.1",
            "model_id": model_id,
            "architecture": architecture,
            "eligible_core_count": 294,
            "eligible_core_family_counts": original_family_counts,
            "lockbox_order_audit": {
                "status": "order verified before invoking gguf-dump",
                "sealed_score_table_sha256": "3" * 64,
            },
            "paired_static_q4_k_m": {
                "status": "paired_exact_tensor_name_and_shape",
                "admission_use": "eligible",
                "eligible_coverage": 1.0,
                "eligible_paired_count": 294,
                "cohort_metadata_exact_match": True,
                "pairing_errors": [],
                "teacher_cohort_metadata": {"general.name": "gemma"},
                "static_cohort_metadata": {"general.name": "gemma"},
            },
        },
    )
    return {
        "original_labels_path": labels,
        "original_audit_path": original_audit,
        "source_roster_path": source_roster,
        "source_manifest_path": source_manifest,
        "scores_path": scores,
        "shard_scores_path": shards,
        "prejoin_receipt_path": prejoin_receipt,
        "candidate_freeze_path": freeze,
        "output_dir": tmp_path / "active-view",
        "current_git": current_git,
    }


def test_builds_exact_one_shot_active_label_view(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tool = load_tool()
    fixture = build_fixture(tmp_path, tool)
    current_git = fixture.pop("current_git")
    monkeypatch.setattr(tool, "git_revision", lambda: current_git)
    receipt = tool.build_active_label_view(**fixture)

    output_dir = fixture["output_dir"]
    with (output_dir / "ud-tensor-labels-active.tsv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        active_rows = list(csv.DictReader(handle, delimiter="\t"))
    audit = json.loads((output_dir / "ud-label-audit-summary-active.json").read_text())
    sealed = json.loads((output_dir / "active-label-view.receipt.json").read_text())

    assert len(active_rows) == 258
    assert {row["hf_name"] for row in active_rows} == {
        f"model.language_model.layers.{index}.self_attn.q_proj.weight"
        for index in range(258)
    }
    assert audit["eligible_core_count"] == 258
    assert audit["paired_static_q4_k_m"]["eligible_paired_count"] == 258
    assert audit["paired_static_q4_k_m"]["eligible_coverage"] == 1.0
    assert audit["paired_static_q4_k_m"]["cohort_metadata_exact_match"] is True
    assert audit["paired_static_q4_k_m"]["admission_use"] == "eligible"
    order = audit["lockbox_order_audit"]
    assert order["status"] == tool.LOCKBOX_STATUS
    assert order["existing_labels_were_previously_sealed_and_opened"] is True
    assert order["gguf_reopened"] is False
    assert order["score_formulas_or_thresholds_changed"] is False
    assert sealed == receipt
    assert sealed["one_shot_join"] is True
    assert sealed["output_hashes"]["active_labels"] == tool.sha256_file(
        output_dir / "ud-tensor-labels-active.tsv"
    )

    with pytest.raises(ValueError, match="refusing to reuse"):
        tool.build_active_label_view(**fixture)


def test_rejects_bad_score_hash_before_opening_old_label_tsv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tool = load_tool()
    fixture = build_fixture(tmp_path, tool)
    current_git = fixture.pop("current_git")
    monkeypatch.setattr(tool, "git_revision", lambda: current_git)
    receipt_path = fixture["prejoin_receipt_path"]
    receipt = json.loads(receipt_path.read_text())
    receipt["score_table_sha256"] = "f" * 64
    write_json(receipt_path, receipt)

    def forbidden_label_read(_path: Path):
        raise AssertionError("old label TSV was opened before score-side authorization")

    monkeypatch.setattr(tool, "read_label_tsv", forbidden_label_read)
    with pytest.raises(ValueError, match="score table differs"):
        tool.build_active_label_view(**fixture)


def test_rejects_score_role_drift_before_label_join(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tool = load_tool()
    fixture = build_fixture(tmp_path, tool)
    current_git = fixture.pop("current_git")
    monkeypatch.setattr(tool, "git_revision", lambda: current_git)
    receipt_path = fixture["prejoin_receipt_path"]
    receipt = json.loads(receipt_path.read_text())
    receipt["secondary_score_columns"].remove("C5b_Empirical_Fisher_I")
    write_json(receipt_path, receipt)

    with pytest.raises(ValueError, match="score roles differ"):
        tool.build_active_label_view(**fixture)
