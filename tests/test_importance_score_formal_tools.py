from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

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


def test_single_safetensors_header_audit_without_index(tmp_path: Path) -> None:
    tool = load_tool("build-ud-tensor-labels.py", "test_ud_label_builder")
    save_file(
        {"model.layers.0.mlp.up_proj.weight": torch.zeros((3, 4), dtype=torch.bfloat16)},
        str(tmp_path / "model.safetensors"),
    )
    headers = tool.load_safetensor_headers(tmp_path)

    assert headers["model.layers.0.mlp.up_proj.weight"]["shape"] == [3, 4]
    assert headers["model.layers.0.mlp.up_proj.weight"]["dtype"] == "BF16"
