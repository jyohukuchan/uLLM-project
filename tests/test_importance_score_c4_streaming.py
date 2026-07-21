from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch


def load_tool(name: str):
    path = (
        Path(__file__).resolve().parents[1]
        / "tools"
        / "run-importance-single-tensor-perturbation.py"
    )
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def c4_reference(hidden: torch.Tensor, expected: torch.Tensor):
    return {
        "calls": {"layer": [((hidden.clone(),), {})]},
        "outputs": {"layer": [expected.to(torch.bfloat16).clone()]},
    }


def test_four_shard_scalar_aggregation_matches_combined_evaluation() -> None:
    tool = load_tool("test_c4_streaming_aggregation")
    layer = torch.nn.Linear(2, 2, bias=False).eval()
    with torch.no_grad():
        layer.weight.copy_(torch.eye(2))
    hidden_shards = [
        torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]),
        torch.tensor([[[2.0, -1.0]]]),
        torch.tensor([[[0.5, 1.5], [2.5, 3.5], [4.5, 5.5]]]),
        torch.tensor([[[-2.0, 1.0], [1.0, -3.0]]]),
    ]
    references = [c4_reference(hidden, layer(hidden)) for hidden in hidden_shards]
    batches = [
        [{"tensors": {"attention_mask": torch.ones(hidden.shape[:-1], dtype=torch.long)}}]
        for hidden in hidden_shards
    ]
    with torch.no_grad():
        layer.weight[0, 0] += 0.25

    parts = [
        tool.candidate_c4_sums(batch, "layer", layer, reference, torch.device("cpu"))
        for batch, reference in zip(batches, references, strict=True)
    ]
    streamed = tool.c4_metrics_from_sums(tool.merge_c4_sums(parts))

    combined_batches = [item for shard in batches for item in shard]
    combined_reference = {
        "calls": {"layer": [item for ref in references for item in ref["calls"]["layer"]]},
        "outputs": {
            "layer": [item for ref in references for item in ref["outputs"]["layer"]]
        },
    }
    combined = tool.candidate_c4(
        combined_batches,
        "layer",
        layer,
        combined_reference,
        torch.device("cpu"),
    )

    assert streamed["valid_tokens"] == combined["valid_tokens"] == 8
    assert streamed["C4_A"] == pytest.approx(combined["C4_A"], rel=0, abs=1e-15)
    assert streamed["C4_reference_energy"] == pytest.approx(
        combined["C4_reference_energy"], rel=0, abs=1e-15
    )
    assert streamed["C4_L"] == pytest.approx(combined["C4_L"], rel=0, abs=1e-15)


def test_formal_reference_helper_selects_exactly_one_layer(monkeypatch) -> None:
    tool = load_tool("test_c4_streaming_one_layer")
    observed = {}

    def fake_reference(model, batches, layer_modules, device):
        observed["model"] = model
        observed["batches"] = batches
        observed["layer_modules"] = layer_modules
        observed["device"] = device
        return {"calls": {}, "outputs": {}}

    monkeypatch.setattr(tool, "reference_c4", fake_reference)
    layer = torch.nn.Linear(2, 2)
    result = tool.reference_c4_one_layer(
        "model", ["batch"], "model.layers.7", layer, torch.device("cpu")
    )

    assert result == {"calls": {}, "outputs": {}}
    assert observed["layer_modules"] == {"model.layers.7": layer}


def test_formal_cache_hard_cap_cleanup_and_run_ownership(tmp_path: Path) -> None:
    tool = load_tool("test_c4_streaming_cache")
    base = tmp_path / "cache"
    base.mkdir()
    sentinel = base / "preexisting.keep"
    sentinel.write_text("do not delete", encoding="utf-8")
    output = tmp_path / "result.jsonl"
    metadata = {"tensor_name": "x", "candidate_id": "aq4"}
    quantization = {"groups": 1}
    weight = torch.arange(16, dtype=torch.bfloat16).view(4, 4)

    cache = tool.FormalC4EphemeralCache(base, "run", output, 1024 * 1024)
    cache.begin_layer("model.layers.0")
    path = cache.store(metadata, quantization, weight)
    assert path.is_file()
    assert cache.active_bytes == path.stat().st_size
    assert cache.layer_peak_bytes == cache.active_bytes
    assert cache.load(path, metadata)["quantization"] == quantization
    owned_root = cache.root
    cache.cleanup_layer()
    assert cache.active_bytes == 0
    cache.cleanup_run()
    assert not owned_root.exists()
    assert sentinel.read_text(encoding="utf-8") == "do not delete"

    capped = tool.FormalC4EphemeralCache(base, "run-cap", output, 1)
    capped.begin_layer("model.layers.1")
    with pytest.raises(RuntimeError, match="hard cap"):
        capped.store(metadata, quantization, weight)
    assert capped.active_bytes == 0
    capped_root = capped.root
    capped.cleanup_run()
    assert not capped_root.exists()
    assert sentinel.is_file()


def test_formal_resume_requires_identical_frozen_input_signature(tmp_path: Path) -> None:
    tool = load_tool("test_c4_streaming_resume")
    output = tmp_path / "c4.jsonl"
    signature = "a" * 64
    rows = [
        {
            "status": "ok",
            "mode": "c4",
            "tensor_name": "model.layers.0.mlp.up_proj.weight",
            "candidate_id": "aq4",
            "formal_c4_input_signature_sha256": signature,
        },
        {
            "status": "failed",
            "mode": "c4",
            "tensor_name": "ignored",
            "candidate_id": "aq4",
            "formal_c4_input_signature_sha256": signature,
        },
    ]
    output.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    assert tool.completed_work_keys(
        output, "c4", formal_c4_signature=signature
    ) == {("model.layers.0.mlp.up_proj.weight", "aq4", "c4")}
    with pytest.raises(SystemExit, match="different frozen inputs"):
        tool.completed_work_keys(output, "c4", formal_c4_signature="b" * 64)
