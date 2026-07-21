from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
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


def test_activation_stats_merge_accepts_only_uniform_explicit_execution_provenance() -> None:
    tool = load_tool("merge-activation-stats.py", "test_activation_merge_provenance")

    assert tool.merged_execution_provenance(
        [{"require_cpu": True, "device": "cpu"} for _ in range(4)]
    ) == ("cpu", "cpu")
    assert tool.merged_execution_provenance(
        [{"require_cpu": False, "device": "cuda:0"} for _ in range(4)]
    ) == ("cuda:0", "gpu")


def test_activation_stats_merge_writes_gpu_execution_provenance(
    tmp_path: Path, monkeypatch
) -> None:
    tool = load_tool("merge-activation-stats.py", "test_activation_merge_metadata")
    module_name = "model.layers.0.mlp.up_proj"
    input_dirs = []
    for index in range(4):
        input_dir = tmp_path / f"shard-{index}"
        input_dir.mkdir()
        (input_dir / "metadata.json").write_text(
            json.dumps(
                {
                    "device": "cuda:0",
                    "require_cpu": False,
                    "tokens_seen": 2,
                    "samples_seen": 1,
                    "padding_mask_policy": "test",
                    "modules": {module_name: {"activation_count": 2}},
                }
            ),
            encoding="utf-8",
        )
        save_file(
            {
                module_name: torch.tensor([1.0, 2.0], dtype=torch.float64),
                f"{module_name}.mean_abs": torch.tensor([0.5, 1.0], dtype=torch.float64),
                f"{module_name}.max_abs": torch.tensor([1.0, 2.0], dtype=torch.float32),
            },
            str(input_dir / "activation_second_moments.safetensors"),
        )
        input_dirs.append(input_dir)
    output_dir = tmp_path / "merged"
    argv = ["merge-activation-stats.py"]
    for input_dir in input_dirs:
        argv.extend(("--input-dir", str(input_dir)))
    argv.extend(("--output-dir", str(output_dir), "--run-id", "gpu-merge-test"))
    monkeypatch.setattr(sys, "argv", argv)

    assert tool.main() == 0
    metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["execution_device"] == "cuda:0"
    assert metadata["device_class"] == "gpu"


@pytest.mark.parametrize(
    "metadata",
    [
        [
            {"require_cpu": True, "device": "cpu"},
            {"require_cpu": False, "device": "cuda:0"},
            {"require_cpu": True, "device": "cpu"},
            {"require_cpu": False, "device": "cuda:0"},
        ],
        [{"require_cpu": False, "device": "auto"} for _ in range(4)],
        [
            {"require_cpu": False, "device": device}
            for device in ("cuda:0", "cuda:0", "cuda:1", "cuda:0")
        ],
        [{"require_cpu": False, "device": "cpu"} for _ in range(4)],
        [{"device": "cuda:0"} for _ in range(4)],
    ],
)
def test_activation_stats_merge_rejects_mixed_or_implicit_execution_provenance(
    metadata: list[dict],
) -> None:
    tool = load_tool("merge-activation-stats.py", f"test_bad_merge_provenance_{id(metadata)}")

    with pytest.raises(SystemExit):
        tool.merged_execution_provenance(metadata)


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


def test_block_covariance_timing_snapshot_reports_throughput(monkeypatch) -> None:
    tool = load_tool("collect-block-covariance-stats.py", "test_c1_timing")
    monkeypatch.setattr(tool.time, "perf_counter", lambda: 12.0)

    assert tool.timing_snapshot(torch.device("cpu"), 10.0, 5) == (2.0, 2.5)


def test_optimized_codebook_lookup_matches_lowest_index_argmin() -> None:
    tool = load_tool(
        "run-importance-single-tensor-perturbation.py", "test_single_tensor_perturbation"
    )
    codebook = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0])
    values = torch.tensor([-2.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.75, 2.0])
    expected = (values[:, None] - codebook[None, :]).abs().argmin(dim=1)

    assert torch.equal(tool.nearest_sorted_codebook(values, codebook), expected)


def test_perturbation_progress_is_flushed_as_stderr_json(capsys) -> None:
    tool = load_tool(
        "run-importance-single-tensor-perturbation.py", "test_perturbation_progress"
    )

    tool.emit_progress({"event": "progress", "tensor_candidates_completed_this_run": 1})

    captured = capsys.readouterr()
    assert captured.out == ""
    assert json.loads(captured.err) == {
        "event": "progress",
        "tensor_candidates_completed_this_run": 1,
    }


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


def test_perturbation_selection_accepts_source_roster_jsonl(tmp_path: Path) -> None:
    tool = load_tool(
        "run-importance-single-tensor-perturbation.py", "test_perturbation_roster_selection"
    )
    path = tmp_path / "source-roster.jsonl"
    rows = [
        {
            "model_id": "m",
            "hf_name": f"model.layers.{index}.mlp.up_proj.weight",
            "canonical_family": "mlp_up",
            "layer_id": index,
            "shape": [8, 8],
        }
        for index in range(2)
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    assert tool.load_selection(path) == rows


def test_cached_c4_block_call_matches_reference_and_detects_perturbation() -> None:
    tool = load_tool(
        "run-importance-single-tensor-perturbation.py", "test_cached_c4_block_call"
    )
    layer = torch.nn.Linear(2, 2, bias=False).eval()
    with torch.no_grad():
        layer.weight.copy_(torch.eye(2))
    hidden = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]], dtype=torch.float32)
    expected = layer(hidden).to(torch.bfloat16)
    reference = {
        "calls": {"layer": [((hidden.clone(),), {})]},
        "outputs": {"layer": [expected.clone()]},
    }
    batches = [{"tensors": {"attention_mask": torch.tensor([[1, 1]])}}]

    audit = tool.validate_c4_call_cache({"layer": layer}, reference, torch.device("cpu"))
    assert audit["layer"]["relative_l2"] == 0.0

    with torch.no_grad():
        layer.weight[0, 0] += 1.0
    metrics = tool.candidate_c4(
        batches, "layer", layer, reference, torch.device("cpu")
    )
    assert metrics["C4_A"] > 0
    assert metrics["valid_tokens"] == 2


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
    c4 = {
        name: {
            tool.LOW: {"metrics": {"C4_L": 0.2, "C4_A": 4.0}},
            tool.HIGH: {"metrics": {"C4_L": 0.1, "C4_A": 2.0}},
        }
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
        roster, tmp_path, stats, [stats] * 4, {name: candidates}, {}, c4, 4, 0
    )

    assert rows[0]["C0_I"] == 0.1
    assert rows[0]["C0_G"] == 1.0
    assert rows[0]["C4_I"] == 0.2
    assert rows[0]["C4_G"] == 2.0
    assert "C4_I_subset" not in rows[0]
    assert "qtype_ud" not in rows[0]
    assert all(name in shard for shard in per_shard)


def write_lockbox_order_fixture(tool, tmp_path: Path) -> tuple[Path, Path, Path]:
    score_path = tmp_path / "scores-prejoin.jsonl"
    score_path.write_text("{}\n", encoding="utf-8")
    shared_hashes = {
        "build-importance-score-prejoin.py": "a",
        "report-importance-score-formal.py": "b",
        "summarize-importance-score-screen.py": "c",
        "run-aq-tensor-sample.py": "d",
        "score-block-covariance-c1.py": "e",
        "run-importance-single-tensor-perturbation.py": "f",
    }
    input_hashes = {
        "candidate_manifest": "candidate",
        "score_registry": "registry",
        "corpus_manifest": "corpus",
    }
    execution_settings = {
        "weight_sample_size": 65536,
        "seed": 0,
        "torch_threads": 16,
        "torch_interop_threads": 1,
        "activation_stat_shard_count": 4,
    }
    scores = ["C0_I", "S_AWQ_level"]
    freeze_path = tmp_path / "qwen-candidate-freeze.json"
    freeze_path.write_text(
        json.dumps(
            {
                "status": "sealed before any Gemma tensor-level score/label join",
                "lockbox_model": "gemma-4-E4B-it",
                "created_at_utc": "2026-07-21T00:00:00+00:00",
                "workspace_git_head": "head",
                "candidate_scores_transferred_unchanged": scores,
                "execution_settings": {"prejoin_score_generation": execution_settings},
                "input_hashes": input_hashes,
                "implementation_hashes": {
                    **shared_hashes,
                    "build-ud-tensor-labels.py": tool.sha256_file(
                        Path(tool.__file__).resolve()
                    ),
                },
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
                "workspace_git_head": "head",
                "score_table_path": str(score_path),
                "score_table_sha256": tool.sha256_file(score_path),
                "candidate_score_columns": scores,
                "score_columns": [*scores, "C0_G"],
                "execution_settings": execution_settings,
                "input_hashes": input_hashes,
                "implementation_hashes": shared_hashes,
            }
        ),
        encoding="utf-8",
    )
    return freeze_path, receipt_path, score_path


def test_gemma_label_builder_verifies_freeze_then_prejoin_order(tmp_path: Path) -> None:
    tool = load_tool("build-ud-tensor-labels.py", "test_lockbox_label_order")
    freeze_path, receipt_path, score_path = write_lockbox_order_fixture(tool, tmp_path)

    audit = tool.verify_lockbox_order(
        "gemma-4-E4B-it", freeze_path, receipt_path
    )

    assert audit["status"] == "order verified before invoking gguf-dump"
    assert audit["sealed_score_table_sha256"] == tool.sha256_file(score_path)
    assert audit["input_hash_comparison"]["score_registry"]["candidate_freeze"] == "registry"
    assert "summarize-importance-score-screen.py" in audit["implementation_hash_comparison"]


@pytest.mark.parametrize(
    ("section", "key", "value", "message"),
    [
        ("input_hashes", "score_registry", "changed", "input hash mismatch"),
        (
            "implementation_hashes",
            "summarize-importance-score-screen.py",
            "changed",
            "implementation hash mismatch",
        ),
        ("execution_settings", "seed", 1, "execution settings differ"),
    ],
)
def test_gemma_label_builder_rejects_lockbox_chain_mismatch(
    tmp_path: Path, section: str, key: str, value, message: str
) -> None:
    tool = load_tool(
        "build-ud-tensor-labels.py", f"test_lockbox_mismatch_{section}_{key}"
    )
    freeze_path, receipt_path, _score_path = write_lockbox_order_fixture(tool, tmp_path)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt[section][key] = value
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        tool.verify_lockbox_order("gemma-4-E4B-it", freeze_path, receipt_path)


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


def test_two_model_comparison_pairs_worst_model_bootstrap_draws() -> None:
    tool = load_tool("report-importance-score-two-model.py", "test_two_model_report")
    qwen = {
        "left": {"rho": np.array([0.5, 0.6, 0.7])},
        "right": {"rho": np.array([0.1, 0.2, 0.3])},
    }
    gemma = {
        "left": {"rho": np.array([0.4, 0.5, 0.6])},
        "right": {"rho": np.array([0.0, 0.1, 0.2])},
    }

    result = tool.paired_worst_model_difference(
        qwen, gemma, "left", "right", "rho"
    )

    assert result["replicates"] == 3
    assert result["left_strictly_better"] is True


def test_two_model_bootstrap_loader_requires_every_frozen_replicate(
    tmp_path: Path,
) -> None:
    tool = load_tool("report-importance-score-two-model.py", "test_two_model_bootstrap")
    scores = ["C0_I", "C1_I"]
    rows = [
        {
            "score_id": score,
            "replicate": replicate,
            "primary_rho": float(replicate),
            "primary_tau_b": float(replicate) / 2,
        }
        for score in scores
        for replicate in range(3)
    ]
    path = tmp_path / "bootstrap.parquet"
    tool.pq.write_table(pa.Table.from_pylist(rows), path)

    arrays = tool.bootstrap_arrays(path, scores, expected_replicates=3)

    assert arrays["C0_I"]["rho"].tolist() == [0.0, 1.0, 2.0]
    tool.pq.write_table(pa.Table.from_pylist(rows[:-1]), path)
    with pytest.raises(ValueError, match="exactly 3 rows per score"):
        tool.bootstrap_arrays(path, scores, expected_replicates=3)


def write_formal_report_receipt_fixture(
    tool, tmp_path: Path
) -> tuple[Path, Path, Path, dict, dict, list[str]]:
    scores = ["C0_I", "C1_I"]
    implementations = {
        "report-importance-score-formal.py": "formal-hash",
        "summarize-importance-score-screen.py": "summarizer-hash",
    }
    metrics = {
        "model_id": "m",
        "execution_settings": {"formal_report": dict(tool.EXPECTED_FORMAL_SETTINGS)},
        "implementation_hashes": implementations,
    }
    freeze = {"implementation_hashes": implementations}
    metrics_path = tmp_path / "metrics-by-model.json"
    bootstrap_path = tmp_path / "bootstrap-samples.parquet"
    receipt_path = tmp_path / "formal-report.receipt.json"
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
    bootstrap_path.write_bytes(b"sealed bootstrap")
    receipt = {
        "schema_version": "importance-score-formal-report-receipt-v0.1",
        "status": "sealed formal report outputs",
        "model_id": "m",
        "candidate_score_columns": scores,
        "execution_settings": {"formal_report": dict(tool.EXPECTED_FORMAL_SETTINGS)},
        "implementation_hashes": implementations,
        "bootstrap_contract": {
            "replicates_per_score": tool.EXPECTED_BOOTSTRAP_REPLICATES,
            "score_count": len(scores),
            "row_count": tool.EXPECTED_BOOTSTRAP_REPLICATES * len(scores),
            "expected_row_count": tool.EXPECTED_BOOTSTRAP_REPLICATES * len(scores),
        },
        "output_hashes": {
            "metrics-by-model.json": tool.sha256_file(metrics_path),
            "bootstrap-samples.parquet": tool.sha256_file(bootstrap_path),
        },
    }
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    return metrics_path, bootstrap_path, receipt_path, metrics, freeze, scores


def test_two_model_report_receipt_binds_metrics_bootstrap_settings_and_code(
    tmp_path: Path,
) -> None:
    tool = load_tool("report-importance-score-two-model.py", "test_report_receipt")
    fixture = write_formal_report_receipt_fixture(tool, tmp_path)

    receipt = tool.validate_report_receipt("model", *fixture)

    assert receipt["status"] == "sealed formal report outputs"


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("bootstrap_hash", "bootstrap samples differ"),
        ("settings", "settings differ"),
        ("implementation", "differs from the candidate freeze"),
        ("row_count", "bootstrap receipt differs"),
    ],
)
def test_two_model_report_receipt_rejects_chain_mismatch(
    tmp_path: Path, mutation: str, message: str
) -> None:
    tool = load_tool(
        "report-importance-score-two-model.py", f"test_bad_report_receipt_{mutation}"
    )
    fixture = write_formal_report_receipt_fixture(tool, tmp_path)
    metrics_path, bootstrap_path, receipt_path, metrics, freeze, scores = fixture
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if mutation == "bootstrap_hash":
        receipt["output_hashes"]["bootstrap-samples.parquet"] = "changed"
    elif mutation == "settings":
        receipt["execution_settings"]["formal_report"]["seed"] = 1
    elif mutation == "implementation":
        receipt["implementation_hashes"]["report-importance-score-formal.py"] = "changed"
    else:
        receipt["bootstrap_contract"]["row_count"] -= 1
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    with pytest.raises(SystemExit, match=message):
        tool.validate_report_receipt(
            "model",
            metrics_path,
            bootstrap_path,
            receipt_path,
            metrics,
            freeze,
            scores,
        )


def write_kl_core_fixture(tool, tmp_path: Path) -> tuple[Path, Path, list[dict]]:
    selection = [
        {
            "model_id": "m",
            "hf_name": f"model.layers.{index}.mlp.up_proj.weight",
            "canonical_family": "mlp_up",
            "layer_id": index,
            "shape": [8, 8],
        }
        for index in range(2)
    ]
    selection_path = tmp_path / "KL-core.json"
    selection_path.write_text(json.dumps(selection), encoding="utf-8")
    selection_sha = tool.sha256_file(selection_path)
    prompt_rows = [{"record_id": "r0"}, {"record_id": "r1"}]
    prompt_path = tmp_path / "D_KL-cpu-subset.jsonl"
    prompt_path.write_text(
        "".join(json.dumps(row) + "\n" for row in prompt_rows), encoding="utf-8"
    )
    prompt_sha = tool.sha256_file(prompt_path)
    manifest_path = tmp_path / "subset-manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "importance-score-cpu-subsets-v0.1",
                "C6": {
                    "path": str(prompt_path),
                    "sha256": prompt_sha,
                    "records": len(prompt_rows),
                },
                "KL_core": {
                    "path": str(selection_path),
                    "sha256": selection_sha,
                    "selected_tensor_count": len(selection),
                },
                "model_token_counts": {
                    "m": {
                        "D_KL_cpu": {
                            "sequence_length": 128,
                            "valid_tokens": 2,
                        }
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    rows = [
        {
            "status": "ok",
            "mode": "c6",
            "tensor_name": selected["hf_name"],
            "candidate_id": candidate,
            "tensor_selection_sha256": selection_sha,
            "prompt_file_sha256": prompt_sha,
            "record_count": len(prompt_rows),
            "record_ids": [row["record_id"] for row in prompt_rows],
            "sequence_length": 128,
            "metrics": {"C6_L": 0.1, "valid_tokens": 2},
        }
        for selected in selection
        for candidate in (tool.LOW, tool.HIGH)
    ]
    c6_path = tmp_path / "c6.jsonl"
    c6_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return c6_path, manifest_path, rows


def test_formal_report_accepts_exact_manifest_bound_kl_core(tmp_path: Path) -> None:
    tool = load_tool("report-importance-score-formal.py", "test_exact_kl_core")
    c6_path, manifest_path, _rows = write_kl_core_fixture(tool, tmp_path)

    parsed, audit = tool.validate_kl_core_inputs(c6_path, manifest_path)

    assert len(parsed) == 2
    assert all(set(candidates) == {tool.LOW, tool.HIGH} for candidates in parsed.values())
    assert audit["tensor_set_and_candidate_coverage_exact"] is True
    assert audit["kl_audit_rows_rejected"] is True


@pytest.mark.parametrize(
    "mutation", ("selection_hash", "prompt_hash", "extra_tensor", "wrong_mode")
)
def test_formal_report_rejects_kl_audit_or_c6_contract_mismatch(
    tmp_path: Path, mutation: str
) -> None:
    tool = load_tool(
        "report-importance-score-formal.py", f"test_bad_kl_core_{mutation}"
    )
    c6_path, manifest_path, rows = write_kl_core_fixture(tool, tmp_path)
    if mutation == "selection_hash":
        rows[0]["tensor_selection_sha256"] = "audit-selection"
    elif mutation == "prompt_hash":
        rows[0]["prompt_file_sha256"] = "wrong-corpus"
    elif mutation == "extra_tensor":
        rows[0]["tensor_name"] = "model.layers.99.mlp.up_proj.weight"
    else:
        rows[0]["mode"] = "c4"
    c6_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    with pytest.raises(ValueError):
        tool.validate_kl_core_inputs(c6_path, manifest_path)


def test_kl_audit_cap_preserves_score_extremes_without_label_fields() -> None:
    tool = load_tool("freeze-importance-kl-audit-subset.py", "test_kl_audit_subset")
    scores = [
        {
            "model_id": "m",
            "hf_name": f"model.layers.{index}.mlp.up_proj.weight",
            "canonical_family": "mlp_up",
            "layer_id": index,
            "shape": [8, 8],
            **{score: float(index) for score in tool.DEFAULT_SCORES},
        }
        for index in range(6)
    ]
    disagreements = [
        {
            "hf_name": "model.layers.3.mlp.up_proj.weight",
            "score_id": "C0_I",
            "notes": "score_high_not_promoted",
            "qtype_ud": "Q4_K",
        }
    ]

    selected, audit = tool.select_audit_rows(
        scores, disagreements, list(tool.DEFAULT_SCORES), 2, 3
    )

    names = {row["hf_name"] for row in selected}
    assert "model.layers.0.mlp.up_proj.weight" in names
    assert "model.layers.5.mlp.up_proj.weight" in names
    assert audit["selected_tensor_count"] == 3
    assert all("qtype_ud" not in row for row in selected)
