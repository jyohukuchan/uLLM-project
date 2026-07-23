from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = ROOT / "tools/collect-importance-gradient-scores.py"


def load_tool(module_name: str = "test_importance_gradient_tool"):
    spec = importlib.util.spec_from_file_location(module_name, TOOL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


TOOL = load_tool()


def quantized(low: torch.Tensor, high: torch.Tensor) -> dict[str, torch.Tensor]:
    return {
        TOOL.LOW_CANDIDATE_ID: low,
        TOOL.HIGH_CANDIDATE_ID: high,
    }


def moments(**values: float):
    result = {}
    for key, value in values.items():
        moment = TOOL.ScalarMoments()
        moment.add(value)
        result[key] = moment
    return result


def test_taylor_quant_abs_is_after_the_full_signed_tensor_dot() -> None:
    gradient = torch.tensor([1.0, 1.0])
    source = torch.tensor([0.0, 0.0])
    # With one element per chunk, abs-per-chunk would incorrectly produce 2.
    low = torch.tensor([1.0, -1.0])
    high = torch.tensor([2.0, -2.0])

    scores = TOOL.reduce_gradient_scores(
        gradient,
        source,
        quantized(low, high),
        mode="empirical",
        chunk_elements=1,
    )

    assert scores[f"taylor_quant_abs_dot::{TOOL.LOW_CANDIDATE_ID}"] == 0.0
    assert scores[f"taylor_quant_abs_dot::{TOOL.HIGH_CANDIDATE_ID}"] == 0.0


def test_taylor_deletion_and_fisher_formulas_include_exact_half() -> None:
    gradient = torch.tensor([2.0, 4.0])
    source = torch.tensor([3.0, 5.0])
    low = torch.tensor([4.0, 7.0])  # delta = [1, 2]
    high = source.clone()  # delta = zero

    scores = TOOL.reduce_gradient_scores(
        gradient,
        source,
        quantized(low, high),
        mode="empirical",
        chunk_elements=1,
    )

    assert scores["taylor_delete_l1"] == 26.0
    assert scores["taylor_delete_squared"] == 436.0
    assert scores[f"taylor_quant_abs_dot::{TOOL.LOW_CANDIDATE_ID}"] == 10.0
    # 0.5 * (2^2 * 1^2 + 4^2 * 2^2) = 34.
    assert scores[f"fisher_half::{TOOL.LOW_CANDIDATE_ID}"] == 34.0
    assert scores[f"fisher_half::{TOOL.HIGH_CANDIDATE_ID}"] == 0.0


def test_sample_means_then_n_param_normalization_and_flat_columns() -> None:
    aggregate = TOOL.TensorAggregate()
    first = {
        "gradient_trace": 1.0,
        "taylor_delete_l1": 8.0,
        "taylor_delete_squared": 16.0,
        f"taylor_quant_abs_dot::{TOOL.LOW_CANDIDATE_ID}": 12.0,
        f"taylor_quant_abs_dot::{TOOL.HIGH_CANDIDATE_ID}": 4.0,
        f"fisher_half::{TOOL.LOW_CANDIDATE_ID}": 10.0,
        f"fisher_half::{TOOL.HIGH_CANDIDATE_ID}": 6.0,
    }
    second = {
        "gradient_trace": 3.0,
        "taylor_delete_l1": 16.0,
        "taylor_delete_squared": 32.0,
        f"taylor_quant_abs_dot::{TOOL.LOW_CANDIDATE_ID}": 20.0,
        f"taylor_quant_abs_dot::{TOOL.HIGH_CANDIDATE_ID}": 8.0,
        f"fisher_half::{TOOL.LOW_CANDIDATE_ID}": 14.0,
        f"fisher_half::{TOOL.HIGH_CANDIDATE_ID}": 8.0,
    }
    aggregate.add(0, first, 0.1)
    aggregate.add(0, second, 0.2)

    scores = TOOL.flat_scores_from_moments("empirical", aggregate.overall, n_params=4)

    assert tuple(scores) == TOOL.EMPIRICAL_SCORE_COLUMNS
    assert scores["C5a_Taylor_quant_A_low"] == 16.0
    assert scores["C5a_Taylor_quant_A_high"] == 6.0
    assert scores["C5a_Taylor_quant_I"] == 4.0
    assert scores["C5a_Taylor_quant_raw_gain"] == 10.0
    assert scores["C5a_Taylor_quant_G"] == 10.0
    assert scores["C5a_Taylor_L1_S"] == 3.0
    assert scores["C5a_Taylor_squared_S"] == 6.0
    assert scores["C5b_Empirical_Fisher_A_low"] == 12.0
    assert scores["C5b_Empirical_Fisher_A_high"] == 7.0
    assert scores["C5b_Empirical_Fisher_I"] == 3.0
    assert scores["C5b_Empirical_Fisher_G"] == 5.0


def test_self_and_empirical_fisher_ids_and_flat_columns_are_never_conflated() -> None:
    assert set(TOOL.EMPIRICAL_METHOD_IDS).isdisjoint(TOOL.SELF_FISHER_METHOD_IDS)
    assert all("Empirical_Fisher" in item for item in TOOL.EMPIRICAL_METHOD_IDS[-2:])
    assert all("Self_Fisher" in item for item in TOOL.SELF_FISHER_METHOD_IDS)

    self_moments = moments(
        **{
            f"fisher_half::{TOOL.LOW_CANDIDATE_ID}": 6.0,
            f"fisher_half::{TOOL.HIGH_CANDIDATE_ID}": 8.0,
        }
    )
    scores = TOOL.flat_scores_from_moments("self_fisher", self_moments, n_params=2)

    assert tuple(scores) == TOOL.SELF_FISHER_SCORE_COLUMNS
    assert scores == {
        "C5b_Self_Fisher_I": 3.0,
        "C5b_Self_Fisher_A_low": 6.0,
        "C5b_Self_Fisher_A_high": 8.0,
        "C5b_Self_Fisher_raw_gain": -2.0,
        "C5b_Self_Fisher_G": 0.0,
    }
    assert not any("Empirical" in key for key in scores)


def test_ordered_aggregation_and_checkpoint_round_trip_are_deterministic() -> None:
    observations = [
        (0, {"gradient_trace": 1.0, f"fisher_half::{TOOL.LOW_CANDIDATE_ID}": 4.0}),
        (1, {"gradient_trace": 3.0, f"fisher_half::{TOOL.LOW_CANDIDATE_ID}": 2.0}),
        (2, {"gradient_trace": 2.0, f"fisher_half::{TOOL.LOW_CANDIDATE_ID}": 5.0}),
        (3, {"gradient_trace": 7.0, f"fisher_half::{TOOL.LOW_CANDIDATE_ID}": 1.0}),
    ]

    def build():
        aggregate = TOOL.TensorAggregate()
        for shard, values in observations:
            aggregate.add(shard, values, 0.0)
        return aggregate

    first = build()
    second = build()
    assert TOOL.canonical_json(first.to_json()) == TOOL.canonical_json(second.to_json())

    restored = TOOL.TensorAggregate.from_json(first.to_json())
    assert TOOL.canonical_json(restored.to_json()) == TOOL.canonical_json(first.to_json())
    assert restored.overall["gradient_trace"].mean == 3.25
    assert [shard["gradient_trace"].count for shard in restored.shards] == [1, 1, 1, 1]


def test_self_fisher_seed_and_full_vocabulary_sampling_are_deterministic() -> None:
    seed = TOOL.stable_self_fisher_seed(17, "record-a", 3)
    assert seed == TOOL.stable_self_fisher_seed(17, "record-a", 3)
    assert seed != TOOL.stable_self_fisher_seed(17, "record-a", 4)
    logits = torch.tensor([[0.0, 1.0, 2.0], [2.0, 1.0, 0.0]])
    first = TOOL.sample_self_fisher_targets(logits, seed=seed)
    second = TOOL.sample_self_fisher_targets(logits, seed=seed)
    assert torch.equal(first, second)
    assert first.shape == (2,)
    assert bool(((0 <= first) & (first < logits.shape[-1])).all())


def test_causal_lm_loss_is_batch_one_valid_next_token_masked_mean() -> None:
    logits = torch.tensor(
        [
            [
                [4.0, 0.0, 0.0],
                [0.0, 4.0, 0.0],
                [0.0, 0.0, 4.0],
                [1.0, 1.0, 1.0],
            ]
        ],
        requires_grad=True,
    )
    tensors = {
        "input_ids": torch.tensor([[0, 0, 1, 2]]),
        "attention_mask": torch.tensor([[1, 1, 1, 0]]),
    }

    loss, count = TOOL.empirical_causal_lm_loss(logits, tensors)
    expected = F.cross_entropy(
        torch.stack((logits[0, 0], logits[0, 1])),
        torch.tensor([0, 1]),
        reduction="mean",
    )

    assert count == 2
    assert torch.equal(loss, expected)


def test_post_accumulate_hook_reduces_then_immediately_clears_gradient() -> None:
    linear = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        linear.weight.copy_(torch.tensor([[1.0, 2.0]]))
    source = linear.weight.detach().cpu().clone()
    cache = TOOL.TensorQuantCache(
        tensor_name="tiny.weight",
        source_bf16_cpu=source,
        quantized_bf16_cpu=quantized(
            torch.tensor([[2.0, 1.0]]),
            torch.tensor([[1.0, 2.0]]),
        ),
    )

    with TOOL.PostAccumulateGradientSession(
        {"tiny.weight": linear.weight},
        {"tiny.weight": cache},
        mode="empirical",
        chunk_elements=1,
    ) as session:
        session.begin({"record_id": "tiny"})
        linear(torch.tensor([[3.0, 4.0]])).sum().backward()
        results, timings = session.finish()

        assert linear.weight.grad is None
        assert results["tiny.weight"]["taylor_delete_l1"] == 11.0
        assert results["tiny.weight"][
            f"taylor_quant_abs_dot::{TOOL.LOW_CANDIDATE_ID}"
        ] == 1.0
        assert timings["tiny.weight"] >= 0.0


def test_deferred_post_accumulate_hook_activation_preserves_scores() -> None:
    linear = torch.nn.Linear(2, 1, bias=False)
    source = linear.weight.detach().cpu().clone()
    cache = TOOL.TensorQuantCache(
        tensor_name="tiny.weight",
        source_bf16_cpu=source,
        quantized_bf16_cpu=quantized(source.clone(), source.clone()),
    )
    output = linear(torch.tensor([[3.0, 4.0]]))

    with TOOL.PostAccumulateGradientSession(
        {"tiny.weight": linear.weight},
        {"tiny.weight": cache},
        mode="self_fisher",
        chunk_elements=2,
        auto_activate=False,
    ) as session:
        session.activate()
        session.begin({"record_id": "tiny"})
        output.sum().backward()
        results, _ = session.finish()
        session.deactivate()

    assert results["tiny.weight"]["gradient_trace"] == 25.0
    assert linear.weight.grad is None


def test_only_gemma_self_fisher_requires_pinned_transfer_staging() -> None:
    assert TOOL.requires_pinned_transfer_staging("gemma4_text", "self_fisher")
    assert not TOOL.requires_pinned_transfer_staging("gemma4_text", "empirical")
    assert not TOOL.requires_pinned_transfer_staging("qwen3_5_text", "self_fisher")


def test_pinned_transfer_staging_path_preserves_exact_reduction_values() -> None:
    class CpuStager:
        def __init__(self) -> None:
            self.keys: list[str] = []

        def to_device(
            self, key: str, source_cpu: torch.Tensor, *, device: torch.device
        ) -> torch.Tensor:
            self.keys.append(key)
            return source_cpu.to(device=device, dtype=torch.float32)

    gradient = torch.tensor([1.5, -2.0, 0.25])
    source = torch.tensor([0.25, -0.5, 1.0], dtype=torch.bfloat16)
    candidates = quantized(
        torch.tensor([0.5, -0.25, 0.5], dtype=torch.bfloat16),
        torch.tensor([0.25, -0.5, 1.0], dtype=torch.bfloat16),
    )
    direct = TOOL.reduce_gradient_scores(
        gradient,
        source,
        candidates,
        mode="self_fisher",
        chunk_elements=2,
    )
    stager = CpuStager()
    staged = TOOL.reduce_gradient_scores(
        gradient,
        source,
        candidates,
        mode="self_fisher",
        chunk_elements=2,
        transfer_stager=stager,
    )

    assert staged == direct
    assert stager.keys == [
        "source",
        TOOL.LOW_CANDIDATE_ID,
        TOOL.HIGH_CANDIDATE_ID,
    ] * 2


def test_pinned_stager_separates_bf16_h2d_from_device_fp32_cast(
    monkeypatch,
) -> None:
    calls: list[tuple[str, dict[str, object]]] = []
    result = object()

    class FakeDeviceTensor:
        def to(self, **kwargs):
            calls.append(("device_cast", kwargs))
            return result

    class FakePinnedTensor:
        def __getitem__(self, _index):
            return self

        def copy_(self, source):
            calls.append(("host_copy", {"dtype": source.dtype}))

        def numel(self):
            return 4

        def element_size(self):
            return 2

        def to(self, **kwargs):
            calls.append(("h2d", kwargs))
            return FakeDeviceTensor()

    monkeypatch.setattr(TOOL.torch, "empty", lambda *args, **kwargs: FakePinnedTensor())
    stager = TOOL.PinnedGradientScoreStager(4)
    device = torch.device("cuda:0")

    observed = stager.to_device(
        TOOL.PinnedGradientScoreStager.SOURCE_KEY,
        torch.ones(4, dtype=torch.bfloat16),
        device=device,
    )

    assert observed is result
    assert calls == [
        ("host_copy", {"dtype": torch.bfloat16}),
        ("h2d", {"device": device, "non_blocking": False}),
        ("device_cast", {"dtype": torch.float32}),
    ]


def test_candidate_ids_are_the_frozen_low_and_only_high_pair() -> None:
    assert TOOL.LOW_CANDIDATE_ID == "aq4_e4m3_g16_ts_flloyd16"
    assert TOOL.HIGH_CANDIDATE_ID == "aq5_e4m3_g16_ts_flloyd32"
    assert TOOL.LOW == TOOL.LOW_CANDIDATE_ID
    assert TOOL.HIGH == TOOL.HIGH_CANDIDATE_ID
    assert TOOL.CANDIDATE_IDS == (
        "aq4_e4m3_g16_ts_flloyd16",
        "aq5_e4m3_g16_ts_flloyd32",
    )


@pytest.mark.parametrize("mode", ["empirical", "self_fisher"])
def test_every_score_is_finite(mode: str) -> None:
    gradient = torch.tensor([1.5, -2.0])
    source = torch.tensor([0.25, -0.5])
    values = TOOL.reduce_gradient_scores(
        gradient,
        source,
        quantized(torch.tensor([0.5, -0.25]), torch.tensor([0.25, -0.5])),
        mode=mode,
        chunk_elements=2,
    )
    assert values
    assert all(math.isfinite(value) for value in values.values())
