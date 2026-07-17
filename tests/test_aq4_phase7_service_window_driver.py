from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_phase7_window_is_single_stop_and_freezes_before_holdout_capture() -> None:
    source = (ROOT / "tools/run-aq4-phase7-service-window.sh").read_text(encoding="utf-8")

    assert "--confirm-single-window" in source
    assert "systemctl restart" not in source
    assert source.count('systemctl stop "$SERVICE"') == 1
    assert source.count('systemctl start "$SERVICE"') == 1
    assert "RuntimeDirectoryPreserve" in source
    assert "exec 9< \"$lock\"" in source
    assert "flock -n 9" in source
    assert "run-aq4-phase3c-r9700-guard.py" in source
    assert "/opt/rocm/bin/amd-smi" in source
    assert "stage-aq4-phase7-fidelity-capture-binary.py" in source
    assert "fidelity-capture-binary-staging" in source
    assert "nlink=1" in source
    assert "path-oracle-export" not in source
    assert "source-oracles/calibration" in source
    assert "source-oracles/holdout" in source
    assert "holdout-execution-view" in source
    assert source.index("calibration_metrics_and_freeze_invoked") < source.index("holdout_target_capture_invoked")
    assert source.index("freeze-receipt.json") < source.index("holdout_target_capture_invoked")

    phase3c_guards = (
        "ULLM_REQUIRE_HIP_AQ4_MATVEC_KERNEL",
        "ULLM_REQUIRE_HIP_AQ4_MATVEC_BATCH_KERNEL",
        "ULLM_REQUIRE_HIP_AQ4_MATVEC_ADD_KERNEL",
        "ULLM_REQUIRE_HIP_AQ4_MATVEC_QKV_Z_GATE_BETA_KERNEL",
        "ULLM_REQUIRE_HIP_LINEAR_ATTN_KERNEL",
        "ULLM_REQUIRE_HIP_LINEAR_ATTN_QKV_PREPARE_BATCH_KERNEL",
        "ULLM_REQUIRE_HIP_LINEAR_ATTN_RECURRENT_KERNEL",
        "ULLM_REQUIRE_HIP_RMSNORM_KERNEL",
        "ULLM_REQUIRE_HIP_SEGMENTED_RMSNORM_SILU_MUL_KERNEL",
        "ULLM_REQUIRE_HIP_AQ4_REGISTER_BM8_KERNEL",
        "ULLM_REQUIRE_HIP_PAGED_DECODE_ATTN_KERNEL",
        "ULLM_REQUIRE_HIP_QWEN35_QK_NORM_ROPE_PAGED_KV_WRITE_KERNEL",
        "ULLM_REQUIRE_HIP_PAGED_KV_WRITE_CHUNK_KERNEL",
        "ULLM_REQUIRE_HIP_PAGED_CAUSAL_GQA_CHUNK_KERNEL",
        "ULLM_REQUIRE_HIP_QWEN35_QK_NORM_ROPE_BATCH_KERNEL",
        "ULLM_REQUIRE_HIP_BF16_ROW_KERNEL",
        "ULLM_REQUIRE_HIP_TOP1_KERNEL",
    )
    for guard in phase3c_guards:
        assert guard in source


def test_phase7_source_oracle_driver_masks_gpu_visibility_and_never_operates_service() -> None:
    source = (ROOT / "tools/run-aq4-phase7-source-oracles.sh").read_text(encoding="utf-8")

    assert "--confirm-cpu-source-capture" in source
    assert "CUDA_VISIBLE_DEVICES=-1" in source
    assert "HIP_VISIBLE_DEVICES=-1" in source
    assert "ROCR_VISIBLE_DEVICES=-1" in source
    assert "systemctl" not in source
    assert "/etc/ullm/served-models/active.json" not in source
    assert "/run/ullm/r9700.lock" not in source
