from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch


def load_sampler():
    path = Path(__file__).resolve().parents[1] / "tools" / "run-aq-tensor-sample.py"
    spec = importlib.util.spec_from_file_location("aq_offline_candidate_sampler", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_aq5_is_only_the_32_entry_extension_of_aq4() -> None:
    sampler = load_sampler()
    aq4 = sampler.candidate_from_id("aq4_e4m3_g16_ts_flloyd16")
    aq5 = sampler.candidate_from_id("aq5_e4m3_g16_ts_flloyd32")

    assert aq4 is not None and aq5 is not None
    assert (aq4.index_bits, aq4.codebook_entries) == (4, 16)
    assert (aq5.index_bits, aq5.codebook_entries) == (5, 32)
    assert aq5.group_size == aq4.group_size == 16
    assert aq5.scale_format == aq4.scale_format == "e4m3"
    assert aq5.tensor_scale == aq4.tensor_scale == "bf16"
    assert aq5.codebook_storage_dtype == aq4.codebook_storage_dtype == "bf16"
    assert sampler.candidate_from_id("aq5_e4m3_g16_ts_flloyd16") is None


def test_fit_eval_partition_is_disjoint_and_shared_by_candidates() -> None:
    sampler = load_sampler()
    tensor = torch.arange(8192, dtype=torch.float32).reshape(128, 64)
    kwargs = {
        "group_size": 16,
        "max_elements": 2048,
        "seed": 0,
        "tensor_name": "model.layers.0.mlp.up_proj.weight",
    }
    fit, _ = sampler.deterministic_group_partition_with_columns(
        tensor, partition="fit", **kwargs
    )
    evaluate, _ = sampler.deterministic_group_partition_with_columns(
        tensor, partition="eval", **kwargs
    )

    fit_starts = set(int(value) for value in fit[:, 0])
    eval_starts = set(int(value) for value in evaluate[:, 0])
    assert fit_starts.isdisjoint(eval_starts)
    assert fit.shape == evaluate.shape == (128, 16)


def test_aq4_and_aq5_emit_storage_rounded_codebooks_and_expected_bpp() -> None:
    sampler = load_sampler()
    groups = torch.linspace(-1.0, 1.0, 4096).reshape(-1, 16)

    for candidate_id, expected_entries, expected_bpp in (
        ("aq4_e4m3_g16_ts_flloyd16", 16, 4.5),
        ("aq5_e4m3_g16_ts_flloyd32", 32, 5.5),
    ):
        candidate = sampler.candidate_from_id(candidate_id)
        assert candidate is not None
        codebook = sampler.codebook_from_groups(
            groups,
            candidate.codebook_mode,
            codebook_entries=candidate.codebook_entries,
            iterations=candidate.lloyd_iterations,
            storage_dtype=candidate.codebook_storage_dtype,
        )
        assert codebook.numel() == expected_entries
        assert torch.equal(codebook, codebook.to(torch.bfloat16).to(torch.float32))

        tensor_scale = sampler.choose_tensor_scale(
            groups,
            candidate,
            sampler.scale_values(candidate.scale_format),
            codebook,
        )
        metrics = sampler.evaluate_candidate(
            groups,
            candidate,
            4,
            codebook,
            tensor_scale,
        )
        assert metrics["effective_bpp"] == expected_bpp
        assert torch.isfinite(torch.tensor(metrics["relative_mse"]))


def test_c0_raw_moment_scale_and_full_tensor_expansion_are_preserved() -> None:
    sampler = load_sampler()
    candidate = sampler.Candidate(
        "test", "e4m3", 2, "none", "none", "free16"
    )
    groups = torch.tensor([[1.0, 2.0]])
    recon = torch.zeros_like(groups)
    metrics = sampler.metrics_from_recon(
        groups,
        recon,
        torch.tensor([18.0]),
        candidate,
        torch.tensor([[2.0, 4.0]]),
        torch.tensor([1.0]),
        torch.tensor([1.0]),
        population_elements=4,
    )

    assert metrics["weighted_sse"] == 18.0
    assert metrics["weighted_reference_sse"] == 18.0
    assert metrics["weighted_weight_sum"] == 6.0
    assert metrics["sample_expansion_factor"] == 2.0
    assert metrics["weighted_sse_estimated_full_tensor"] == 36.0
