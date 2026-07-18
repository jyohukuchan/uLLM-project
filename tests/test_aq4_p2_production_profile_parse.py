from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "aq4_p2_production_profile_parse",
    ROOT / "tools/parse-aq4-p2-production-profile.py",
)
assert SPEC and SPEC.loader
PROFILE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PROFILE
try:
    SPEC.loader.exec_module(PROFILE)
finally:
    sys.modules.pop(SPEC.name, None)


REAL_KERNEL_FAMILIES: tuple[tuple[str, str], ...] = (
    ("__amd_rocclr_copyBuffer", "runtime_support"),
    ("__amd_rocclr_fillBufferAligned", "runtime_support"),
    ("ullm_add_f32_kernel", "normalization"),
    ("ullm_aq4_gemm_register_bm8_f32_kernel", "aq4_projection"),
    ("ullm_aq4_matvec_add_f32_kernel", "aq4_projection"),
    ("ullm_aq4_matvec_batch_f32_kernel", "aq4_projection"),
    ("ullm_aq4_matvec_f32_kernel", "aq4_projection"),
    ("ullm_aq4_matvec_gate_beta_f32_kernel", "aq4_projection"),
    ("ullm_aq4_matvec_pair_f32_kernel", "aq4_projection"),
    ("ullm_aq4_matvec_qkv_z_gate_beta_f32_kernel", "aq4_projection"),
    ("ullm_aq4_matvec_silu_mul_f32_kernel", "aq4_projection"),
    ("ullm_aq4_matvec_triple_f32_kernel", "aq4_projection"),
    ("ullm_bf16_row_f32_kernel", "embedding"),
    ("ullm_linear_attn_gate_beta_f32_kernel", "recurrent"),
    ("ullm_linear_attn_qkv_prepare_batch_f32_kernel", "recurrent"),
    ("ullm_linear_attn_qkv_prepare_batch_update_history_f32_kernel", "recurrent"),
    ("ullm_linear_attn_qkv_prepare_f32_kernel", "recurrent"),
    ("ullm_linear_attn_recurrent_f32_kernel", "recurrent"),
    ("ullm_paged_causal_gqa_chunk_f32_kernel", "attention"),
    ("ullm_paged_decode_attn_f32_kernel", "attention"),
    ("ullm_paged_decode_attn_split_merge_f32_kernel", "attention"),
    ("ullm_paged_decode_attn_split_partial_f32_kernel", "attention"),
    ("ullm_paged_kv_write_chunk_f32_kernel", "paged_validation"),
    ("ullm_paged_kv_write_f32_kernel", "paged_validation"),
    ("ullm_qwen35_qk_norm_rope_batch_f32_kernel", "paged_validation"),
    ("ullm_qwen35_qk_norm_rope_paged_kv_write_f32_kernel", "paged_validation"),
    ("ullm_rmsnorm_f32_kernel", "normalization"),
    ("ullm_segmented_rmsnorm_f32_kernel", "normalization"),
    ("ullm_segmented_rmsnorm_silu_mul_f32_kernel", "normalization"),
    ("ullm_silu_mul_f32_kernel", "normalization"),
    ("ullm_top1_f32_kernel", "head"),
)


def matching_families(name: str) -> list[str]:
    return [
        family
        for family, patterns in PROFILE.COMPILED.items()
        if any(pattern.search(name) for pattern in patterns)
    ]


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=True, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def write_kernel_trace(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["Kernel_Name", "Start_Timestamp", "End_Timestamp", "Phase"],
        )
        writer.writeheader()
        writer.writerows(rows)


def bound_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    profile = tmp_path / "profile"
    profile.mkdir()
    window = tmp_path / "window-result.json"
    binding = tmp_path / "trace-binding.json"
    write_json(
        window,
        {
            "schema_version": "ullm.aq4_p2_production_baseline_window_result.v1",
            "status": "partial_observability",
            "kind": "detailed_profile",
        },
    )
    write_json(
        binding,
        {
            "schema_version": "ullm.aq4_p2_production_baseline_window_result.v1",
            "status": "partial_observability",
            "executor_trace_sha256": "a" * 64,
            "executor_record_sidecar_sha256": "b" * 64,
        },
    )
    return profile, window, binding


def parse_profile(
    profile: Path,
    window: Path,
    binding: Path,
    output: Path,
    maximum_unclassified_fraction: float = 0.0,
) -> dict[str, object]:
    args = PROFILE.parse_args(
        [
            "--profile-dir",
            str(profile),
            "--window-result",
            str(window),
            "--trace-binding",
            str(binding),
            "--output",
            str(output),
            "--maximum-unclassified-fraction",
            str(maximum_unclassified_fraction),
        ]
    )
    return PROFILE.parse(args)


@pytest.mark.parametrize(("name", "family"), REAL_KERNEL_FAMILIES)
def test_all_real_p2_kernel_names_classify_to_one_expected_family(
    name: str, family: str
) -> None:
    assert matching_families(name) == [family]
    assert PROFILE.classify(name) == family


@pytest.mark.parametrize(
    ("name", "family"),
    [
        ("ullm_aq4_matvec_add_f32_kernel", "aq4_projection"),
        ("ullm_aq4_matvec_silu_mul_f32_kernel", "aq4_projection"),
        ("ullm_qwen35_qk_norm_rope_batch_f32_kernel", "paged_validation"),
        (
            "ullm_qwen35_qk_norm_rope_paged_kv_write_f32_kernel",
            "paged_validation",
        ),
    ],
)
def test_previously_ambiguous_kernel_names_now_classify(name: str, family: str) -> None:
    assert PROFILE.classify(name) == family


@pytest.mark.parametrize(
    ("name", "family"),
    [
        ("ullm_rope_f32_kernel", "normalization"),
        ("ullm_sigmoid_mul_f32_kernel", "normalization"),
    ],
)
def test_existing_normalization_patterns_remain_available(name: str, family: str) -> None:
    assert PROFILE.classify(name) == family


def test_unknown_and_genuinely_ambiguous_kernel_names_remain_fail_closed() -> None:
    assert PROFILE.classify("vendor_new_unreviewed_kernel") is None
    with pytest.raises(PROFILE.ProfileError, match="matches multiple P2 families"):
        PROFILE.classify("ullm_aq4_matvec_rmsnorm_f32_kernel")


def test_parse_kernel_accounts_for_every_real_name_in_a_synthetic_trace(tmp_path: Path) -> None:
    trace = tmp_path / "detail_kernel_trace.csv"
    rows = [
        {
            "Kernel_Name": name,
            "Start_Timestamp": index * 10,
            "End_Timestamp": (index + 1) * 10,
            "Phase": "prefill" if index < 20 else "decode",
        }
        for index, (name, _family) in enumerate(REAL_KERNEL_FAMILIES)
    ]
    write_kernel_trace(trace, rows)

    parsed = PROFILE.parse_kernel(trace)

    expected_counts = {
        family: sum(1 for _name, expected in REAL_KERNEL_FAMILIES if expected == family)
        for family in PROFILE.FAMILIES
    }
    assert parsed["kernel_count"] == len(REAL_KERNEL_FAMILIES)
    assert parsed["gpu_union_ns"] == len(REAL_KERNEL_FAMILIES) * 10
    assert parsed["unknown_kernel_names"] == []
    assert parsed["unclassified_fraction_of_union"] == 0.0
    assert {
        family: values["kernel_count"]
        for family, values in parsed["families"].items()
        if family != "unclassified"
    } == expected_counts
    assert parsed["families"]["unclassified"] == {
        "kernel_count": 0,
        "inclusive_ns": 0,
    }


def test_parse_binds_a_synthetic_profile_with_the_new_family(tmp_path: Path) -> None:
    profile, window, binding = bound_inputs(tmp_path)
    write_kernel_trace(
        profile / "detail_kernel_trace.csv",
        [
            {
                "Kernel_Name": "ullm_qwen35_qk_norm_rope_paged_kv_write_f32_kernel",
                "Start_Timestamp": 0,
                "End_Timestamp": 10,
                "Phase": "prefill",
            },
            {
                "Kernel_Name": "ullm_paged_kv_write_f32_kernel",
                "Start_Timestamp": 10,
                "End_Timestamp": 20,
                "Phase": "decode",
            },
        ],
    )
    (profile / "detail_hip_api_trace.csv").write_text(
        "Name\nhipLaunchKernel\nhipStreamSynchronize\n",
        encoding="utf-8",
    )
    (profile / "detail_memory_copy_trace.csv").write_text(
        "Bytes\n1024\n",
        encoding="utf-8",
    )
    output = tmp_path / "profile.json"

    result = parse_profile(profile, window, binding, output)

    assert result["status"] == "profiled_diagnostic"
    assert result["kernel"]["families"]["paged_validation"] == {
        "kernel_count": 2,
        "inclusive_ns": 20,
    }
    assert result["kernel"]["unknown_kernel_names"] == []
    assert result["launch_sync"] == {
        "status": "captured",
        "path": str(profile / "detail_hip_api_trace.csv"),
        "sha256": PROFILE.sha(profile / "detail_hip_api_trace.csv"),
        "launch_count": 1,
        "sync_count": 1,
    }
    assert result["transfer"]["transfer_bytes"] == 1024
    assert result["raw_profile"]["member_count"] == 3
    assert result["profile_hash_binding"]["kernel_trace_sha256"] == PROFILE.sha(
        profile / "detail_kernel_trace.csv"
    )
    assert json.loads(output.read_text(encoding="utf-8")) == result


def test_parse_api_accepts_the_rocprofv3_function_column_variant(tmp_path: Path) -> None:
    # The rocprofv3 build actually installed for P2 capture names the HIP API
    # trace's name column "Function" rather than "Name"/"API_Name"/"ApiName".
    profile, window, binding = bound_inputs(tmp_path)
    write_kernel_trace(
        profile / "detail_kernel_trace.csv",
        [
            {
                "Kernel_Name": "ullm_top1_f32_kernel",
                "Start_Timestamp": 0,
                "End_Timestamp": 10,
                "Phase": "decode",
            }
        ],
    )
    (profile / "detail_hip_api_trace.csv").write_text(
        "Domain,Function,Process_Id,Thread_Id,Correlation_Id,Start_Timestamp,End_Timestamp\n"
        "HIP,hipModuleLaunchKernel,1,1,1,0,10\n"
        "HIP,hipStreamSynchronize,1,1,2,10,20\n",
        encoding="utf-8",
    )
    output = tmp_path / "profile.json"

    result = parse_profile(profile, window, binding, output)

    assert result["status"] == "profiled_diagnostic"
    assert result["launch_sync"]["status"] == "captured"
    assert result["launch_sync"]["launch_count"] == 1
    assert result["launch_sync"]["sync_count"] == 1


def test_unknown_kernel_time_is_blocked_by_default_and_can_be_explicitly_allowed(
    tmp_path: Path,
) -> None:
    profile, window, binding = bound_inputs(tmp_path)
    write_kernel_trace(
        profile / "detail_kernel_trace.csv",
        [
            {
                "Kernel_Name": "ullm_top1_f32_kernel",
                "Start_Timestamp": 0,
                "End_Timestamp": 10,
                "Phase": "decode",
            },
            {
                "Kernel_Name": "vendor_new_unreviewed_kernel",
                "Start_Timestamp": 10,
                "End_Timestamp": 20,
                "Phase": "decode",
            },
        ],
    )

    blocked = parse_profile(profile, window, binding, tmp_path / "blocked.json")
    allowed = parse_profile(
        profile,
        window,
        binding,
        tmp_path / "allowed.json",
        maximum_unclassified_fraction=0.5,
    )

    assert blocked["status"] == "blocked_unclassified_kernel_time"
    assert blocked["kernel"]["unknown_kernel_names"] == ["vendor_new_unreviewed_kernel"]
    assert blocked["kernel"]["unclassified_fraction_of_union"] == 0.5
    assert allowed["status"] == "profiled_diagnostic"


def test_profile_hash_binding_changes_when_the_raw_trace_is_replaced(tmp_path: Path) -> None:
    profile, window, binding = bound_inputs(tmp_path)
    trace = profile / "detail_kernel_trace.csv"
    write_kernel_trace(
        trace,
        [
            {
                "Kernel_Name": "ullm_top1_f32_kernel",
                "Start_Timestamp": 0,
                "End_Timestamp": 10,
                "Phase": "decode",
            }
        ],
    )
    first = parse_profile(profile, window, binding, tmp_path / "first.json")

    write_kernel_trace(
        trace,
        [
            {
                "Kernel_Name": "ullm_rmsnorm_f32_kernel",
                "Start_Timestamp": 0,
                "End_Timestamp": 10,
                "Phase": "decode",
            }
        ],
    )
    second = parse_profile(profile, window, binding, tmp_path / "second.json")

    assert first["profile_hash_binding"]["kernel_trace_sha256"] != second[
        "profile_hash_binding"
    ]["kernel_trace_sha256"]
    assert first["kernel"]["families"]["head"]["kernel_count"] == 1
    assert second["kernel"]["families"]["normalization"]["kernel_count"] == 1
