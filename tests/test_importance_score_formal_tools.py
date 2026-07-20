from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file


def load_tool(filename: str, name: str):
    path = Path(__file__).resolve().parents[1] / "tools" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_block_covariance_accumulator_preserves_off_diagonal_terms() -> None:
    tool = load_tool("collect-block-covariance-stats.py", "test_c1_collector")
    accumulator = tool.BlockCovarianceAccumulator.create(4, 2)
    values = torch.tensor([[[1.0, 2.0, 3.0, 4.0], [2.0, 1.0, 0.0, 1.0]]])
    accumulator.add(values, torch.tensor([[1, 1]]))
    covariance = accumulator.covariance()

    assert accumulator.count == 2
    assert covariance.shape == (2, 2, 2)
    assert torch.allclose(
        covariance[0], torch.tensor([[2.5, 2.0], [2.0, 2.5]], dtype=torch.float64)
    )


def test_optimized_codebook_lookup_matches_lowest_index_argmin() -> None:
    tool = load_tool(
        "run-importance-single-tensor-perturbation.py", "test_single_tensor_perturbation"
    )
    codebook = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0])
    values = torch.tensor([-2.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.75, 2.0])
    expected = (values[:, None] - codebook[None, :]).abs().argmin(dim=1)

    assert torch.equal(tool.nearest_sorted_codebook(values, codebook), expected)


def test_perturbation_selection_rejects_label_or_score_columns(tmp_path: Path) -> None:
    tool = load_tool(
        "run-importance-single-tensor-perturbation.py", "test_perturbation_selection"
    )
    path = tmp_path / "selection.json"
    source_only = [
        {
            "model_id": "m",
            "hf_name": "model.layers.0.mlp.up_proj.weight",
            "canonical_family": "mlp_up",
            "layer_id": 0,
            "shape": [8, 8],
        }
    ]
    path.write_text(json.dumps(source_only), encoding="utf-8")
    assert tool.load_selection(path) == source_only

    source_only[0]["promotion_delta_ordinal"] = 1
    path.write_text(json.dumps(source_only), encoding="utf-8")
    with pytest.raises(SystemExit, match="forbidden label/score keys"):
        tool.load_selection(path)


def test_kl_core_selection_does_not_depend_on_type_or_promotion_columns(tmp_path: Path) -> None:
    tool = load_tool("freeze-importance-score-cpu-subsets.py", "test_cpu_subset_freezer")
    fields = [
        "model_id",
        "hf_name",
        "gguf_name",
        "canonical_family",
        "layer_id",
        "shape",
        "eligible",
        "qtype_ud",
        "promoted",
    ]
    rows = [
        {
            "model_id": "m",
            "hf_name": f"model.layers.{index}.mlp.up_proj.weight",
            "gguf_name": f"blk.{index}.ffn_up.weight",
            "canonical_family": "mlp_up",
            "layer_id": str(index),
            "shape": "[8,8]",
            "eligible": "true",
            "qtype_ud": "Q4_K" if index % 2 else "Q6_K",
            "promoted": "true" if index % 2 else "false",
        }
        for index in range(20)
    ]
    left = tmp_path / "left.tsv"
    right = tmp_path / "right.tsv"
    for path, reverse_labels in ((left, False), (right, True)):
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
            writer.writeheader()
            for row in rows:
                item = dict(row)
                if reverse_labels:
                    item["qtype_ud"] = "Q8_0" if item["qtype_ud"] == "Q4_K" else "Q4_K"
                    item["promoted"] = "false" if item["promoted"] == "true" else "true"
                writer.writerow(item)

    assert tool.select_kl_core(left, 0.10) == tool.select_kl_core(right, 0.10)


def test_cpu_subset_shard_views_are_an_exact_partition(tmp_path: Path) -> None:
    tool = load_tool("freeze-importance-score-cpu-subsets.py", "test_cpu_subset_shards")
    rows = [
        {"record_id": f"r-{shard}-{index}", "domain": "general", "shard": shard}
        for shard in range(4)
        for index in range(2)
    ]

    metadata = tool.write_shard_jsonl_files(tmp_path, "D_block", rows)

    assert [item["records"] for item in metadata] == [2, 2, 2, 2]
    restored = []
    for item in metadata:
        restored.extend(tool.read_jsonl(Path(item["path"])))
    assert restored == rows


def test_source_roster_kl_selection_is_score_blind(tmp_path: Path) -> None:
    tool = load_tool("freeze-importance-score-cpu-subsets.py", "test_roster_kl_core")
    paths = [tmp_path / "left.jsonl", tmp_path / "right.jsonl"]
    rows = [
        {
            "model_id": "m",
            "hf_name": f"model.language_model.layers.{index}.mlp.up_proj.weight",
            "canonical_family": "mlp_up",
            "layer_id": index,
            "shape": [8, 8],
        }
        for index in range(20)
    ]
    for path, reverse_scores in zip(paths, (False, True), strict=True):
        with path.open("w", encoding="utf-8") as handle:
            for index, row in enumerate(rows):
                item = dict(row)
                item["candidate_score"] = -index if reverse_scores else index
                handle.write(tool.canonical_json(item) + "\n")

    assert tool.select_kl_core_from_source_roster(
        paths[0], 0.10
    ) == tool.select_kl_core_from_source_roster(paths[1], 0.10)


def test_single_safetensors_header_audit_without_index(tmp_path: Path) -> None:
    tool = load_tool("build-ud-tensor-labels.py", "test_ud_label_builder")
    save_file(
        {"model.layers.0.mlp.up_proj.weight": torch.zeros((3, 4), dtype=torch.bfloat16)},
        str(tmp_path / "model.safetensors"),
    )
    headers = tool.load_safetensor_headers(tmp_path)

    assert headers["model.layers.0.mlp.up_proj.weight"]["shape"] == [3, 4]
    assert headers["model.layers.0.mlp.up_proj.weight"]["dtype"] == "BF16"


def test_direction_gate_counts_constant_score_as_not_positive() -> None:
    tool = load_tool("report-importance-score-formal.py", "test_formal_direction_gate")
    families = [
        {
            "family": f"f{index}",
            "n": 8,
            "label_nonconstant": True,
            "defined": index < 3,
            "tau_b": 0.5 if index < 3 else None,
        }
        for index in range(5)
    ]

    result = tool.direction_gate(families)

    assert result["defined_family_count"] == 3
    assert result["positive_tau_fraction"] == 0.6
    assert result["pass"] is False


def test_paired_cohort_gate_requires_exact_full_pairing(tmp_path: Path) -> None:
    tool = load_tool("report-importance-score-formal.py", "test_formal_pair_audit")
    path = tmp_path / "audit.json"
    path.write_text(
        json.dumps(
            {
                "paired_static_q4_k_m": {
                    "status": "paired_exact_tensor_name_and_shape",
                    "admission_use": "eligible",
                    "eligible_coverage": 1.0,
                    "eligible_paired_count": 8,
                    "cohort_metadata_exact_match": True,
                    "pairing_errors": [],
                }
            }
        ),
        encoding="utf-8",
    )

    assert tool.paired_cohort_audit(path, 8)["pass"] is True
    assert tool.paired_cohort_audit(path, 9)["pass"] is False


def test_prejoin_score_receipt_is_verified_before_label_join(tmp_path: Path) -> None:
    tool = load_tool("report-importance-score-formal.py", "test_formal_prejoin")
    score_path = tmp_path / "scores-prejoin.jsonl"
    shard_path = tmp_path / "shard-scores-prejoin.json"
    receipt_path = tmp_path / "scores-prejoin.receipt.json"
    score_row = {
        "model_id": "m",
        "architecture": "a",
        "canonical_family": "mlp_up",
        "layer_id": 0,
        "hf_name": "model.layers.0.mlp.up_proj.weight",
        "shape": [2, 2],
        "n_params": 4,
        "C0_I": 0.1,
    }
    extra_score_row = {
        **score_row,
        "hf_name": "model.layers.1.mlp.up_proj.weight",
        "layer_id": 1,
    }
    score_path.write_text(
        json.dumps(score_row, sort_keys=True)
        + "\n"
        + json.dumps(extra_score_row, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    shard_path.write_text(json.dumps([{}, {}, {}, {}]), encoding="utf-8")
    name_digest = hashlib.sha256(
        ("\n".join(sorted((score_row["hf_name"], extra_score_row["hf_name"]))) + "\n").encode()
    ).hexdigest()
    receipt_path.write_text(
        json.dumps(
            {
                "status": (
                    "sealed score table generated without accepting or opening a GGUF label manifest"
                ),
                "score_table_sha256": tool.sha256_file(score_path),
                "shard_scores_sha256": tool.sha256_file(shard_path),
                "tensor_name_set_sha256": name_digest,
                "tensor_count": 2,
                "workspace_git_head": "test",
                "implementation_hashes": {},
            }
        ),
        encoding="utf-8",
    )
    labels = [
        {
            "model_id": "m",
            "architecture": "a",
            "canonical_family": "mlp_up",
            "layer_id": "0",
            "hf_name": score_row["hf_name"],
            "gguf_name": "blk.0.ffn_up.weight",
            "shape": "[2,2]",
            "n_params": "4",
            "qtype_ud": "Q5_K",
            "qtype_static": "Q4_K",
            "ordinal_ud": "1",
            "ordinal_static": "0",
            "packed_bpp_ud": "5.5",
            "packed_bpp_static": "4.5",
            "promotion_delta_ordinal": "1",
            "promotion_delta_bpp": "1.0",
            "promoted": "true",
        }
    ]

    rows, shards, audit = tool.load_and_join_prejoin_scores(
        score_path, receipt_path, shard_path, labels
    )

    assert rows[0]["qtype_ud"] == "Q5_K"
    assert rows[0]["promoted"] is True
    assert len(shards) == 4
    assert audit["label_join_performed_after_receipt"] is True
    assert audit["unjoined_source_score_tensor_count"] == 1


def test_score_features_accepts_source_only_roster(tmp_path: Path) -> None:
    tool = load_tool("report-importance-score-formal.py", "test_formal_source_score")
    name = "model.layers.0.mlp.up_proj.weight"
    module = name.removesuffix(".weight")
    save_file({name: torch.tensor([[1.0, -1.0], [0.5, -0.5]])}, str(tmp_path / "model.safetensors"))
    stats = {
        module: torch.tensor([1.0, 4.0]),
        f"{module}.mean_abs": torch.tensor([0.5, 1.0]),
        f"{module}.max_abs": torch.tensor([1.0, 2.0]),
    }
    candidates = {
        tool.LOW: {
            "metrics": {
                "weighted_relative_mse": 0.1,
                "weighted_sse_estimated_full_tensor": 2.0,
            }
        },
        tool.HIGH: {
            "metrics": {
                "weighted_relative_mse": 0.05,
                "weighted_sse_estimated_full_tensor": 1.0,
            }
        },
    }
    roster = [
        {
            "model_id": "m",
            "architecture": "a",
            "layer_id": 0,
            "canonical_family": "mlp_up",
            "hf_name": name,
            "shape": [2, 2],
            "n_params": 4,
        }
    ]

    rows, per_shard = tool.score_features(
        roster, tmp_path, stats, [stats] * 4, {name: candidates}, {}, {}, 4, 0
    )

    assert rows[0]["C0_I"] == 0.1
    assert rows[0]["C0_G"] == 1.0
    assert "qtype_ud" not in rows[0]
    assert all(name in shard for shard in per_shard)


def test_gemma_label_builder_verifies_freeze_then_prejoin_order(tmp_path: Path) -> None:
    tool = load_tool("build-ud-tensor-labels.py", "test_lockbox_label_order")
    score_path = tmp_path / "scores-prejoin.jsonl"
    score_path.write_text("{}\n", encoding="utf-8")
    hashes = {
        "build-importance-score-prejoin.py": "a",
        "report-importance-score-formal.py": "b",
        "run-aq-tensor-sample.py": "c",
    }
    freeze_path = tmp_path / "qwen-candidate-freeze.json"
    freeze_path.write_text(
        json.dumps(
            {
                "status": "sealed before any Gemma tensor-level score/label join",
                "lockbox_model": "gemma-4-E4B-it",
                "created_at_utc": "2026-07-21T00:00:00+00:00",
                "implementation_hashes": hashes,
            }
        ),
        encoding="utf-8",
    )
    receipt_path = tmp_path / "scores-prejoin.receipt.json"
    receipt_path.write_text(
        json.dumps(
            {
                "status": (
                    "sealed score table generated without accepting or opening a GGUF label manifest"
                ),
                "model_id": "gemma-4-E4B-it",
                "created_at_utc": "2026-07-21T01:00:00+00:00",
                "score_table_path": str(score_path),
                "score_table_sha256": tool.sha256_file(score_path),
                "implementation_hashes": hashes,
            }
        ),
        encoding="utf-8",
    )

    audit = tool.verify_lockbox_order(
        "gemma-4-E4B-it", freeze_path, receipt_path
    )

    assert audit["status"] == "order verified before invoking gguf-dump"
    assert audit["sealed_score_table_sha256"] == tool.sha256_file(score_path)


def test_c1_accepts_single_file_source_roster_without_labels(tmp_path: Path) -> None:
    tool = load_tool("score-block-covariance-c1.py", "test_c1_source_roster")
    name = "model.language_model.layers.0.mlp.up_proj.weight"
    save_file({name: torch.zeros((2, 2))}, str(tmp_path / "model.safetensors"))
    roster_path = tmp_path / "roster.jsonl"
    row = {
        "model_id": "m",
        "architecture": "a",
        "layer_id": 0,
        "canonical_family": "mlp_up",
        "hf_name": name,
        "shape": [2, 2],
        "n_params": 4,
    }
    roster_path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    assert tool.read_source_roster(roster_path) == [row]
    assert tool.tensor_file_map(tmp_path)[name] == tmp_path / "model.safetensors"

    row["qtype_ud"] = "Q5_K"
    roster_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="forbidden label keys"):
        tool.read_source_roster(roster_path)
