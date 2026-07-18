// Copyright 2026 uLLM contributors
// SPDX-License-Identifier: Apache-2.0

//! One-shot end-to-end Qwen3.5 AQ4 cold-prefill timing on the isolated R9700.
//!
//! The caller owns GPU isolation and must set `HIP_VISIBLE_DEVICES=1` plus the production AQ4
//! kernel guards before starting this process. This binary never modifies the environment.

use serde_json::json;
use std::env;
use std::process::ExitCode;
use std::time::{Duration, Instant};

use ullm_engine::aq4_worker_backend::QWEN35_AQ4_REQUIRED_HIP_KERNEL_ENV;
use ullm_engine::backend_operation_registry::OperationExecutionStatus;
use ullm_engine::execution_batch::ExecutionPhase;
use ullm_engine::qwen35_aq4_head_runtime::PackageLmHeadMode;
use ullm_engine::qwen35_aq4_model_runtime::{
    QWEN35_AQ4_CONTEXT_LENGTH, QWEN35_AQ4_KV_BLOCK_SIZE, Qwen35Aq4ModelLoadConfig,
    Qwen35Aq4ModelRuntime,
};
use ullm_engine::qwen35_aq4_session::{QWEN35_AQ4_ROPE_BASE, QWEN35_AQ4_ROTARY_DIM};

const PACKAGE_DIR: &str = "/home/homelab1/datapool/ullm/product/qwen35-9b-aq4-cli-v0.1/package";
const RUNTIME_HIP_DEVICE_INDEX: u32 = 1;
const REQUIRED_ARCHITECTURE: &str = "gfx1201";
const LOAD_CHUNK_BYTES: usize = 1024 * 1024;
const LM_HEAD_CHUNK_ROWS: usize = 8192;
const TOKEN_COUNT: usize = 2048;
const PREFILL_CHUNK_WIDTH: usize = 128;

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("ullm-aq4-e2e-prefill-timing: {error}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), String> {
    require_caller_environment()?;
    require_isolated_r9700("before model load")?;

    let mut model = Qwen35Aq4ModelRuntime::load(Qwen35Aq4ModelLoadConfig {
        package_dir: PACKAGE_DIR.into(),
        device_index: RUNTIME_HIP_DEVICE_INDEX,
        expected_architecture: Some(REQUIRED_ARCHITECTURE.to_string()),
        chunk_bytes: LOAD_CHUNK_BYTES,
        context_length: QWEN35_AQ4_CONTEXT_LENGTH,
        kv_block_size: QWEN35_AQ4_KV_BLOCK_SIZE,
        layer_indices: None,
        lm_head_mode: PackageLmHeadMode::GpuResidentF32,
        lm_head_chunk_rows: LM_HEAD_CHUNK_ROWS,
    })?;

    require_isolated_r9700("after model load")?;
    if model.backend() != "hip" {
        return Err(format!(
            "loaded model backend must be hip, got {}",
            model.backend()
        ));
    }
    if TOKEN_COUNT > model.geometry().context_length {
        return Err(format!(
            "timing token count {TOKEN_COUNT} exceeds loaded context length {}",
            model.geometry().context_length
        ));
    }

    let token_ids = deterministic_token_ids(model.geometry().vocab)?;

    // The reset preserves all resident weights and in-process HIPRTC/module caches while returning
    // KV and recurrent request state to the same clean baseline used by the resident driver.
    let _ = run_cold_prefill(&mut model, &token_ids)?;
    model.reset_all_request_state_synchronized()?;

    let elapsed = run_cold_prefill(&mut model, &token_ids)?;
    if elapsed.is_zero() {
        return Err("timed cold prefill completed with zero elapsed duration".to_string());
    }
    let elapsed_seconds = elapsed.as_secs_f64();
    let tokens_per_second = TOKEN_COUNT as f64 / elapsed_seconds;
    println!(
        "{}",
        json!({
            "tokens": TOKEN_COUNT,
            "chunk_width": PREFILL_CHUNK_WIDTH,
            "elapsed_seconds": elapsed_seconds,
            "tokens_per_second": tokens_per_second,
        })
    );
    Ok(())
}

fn require_caller_environment() -> Result<(), String> {
    require_environment_value("HIP_VISIBLE_DEVICES")?;
    for name in QWEN35_AQ4_REQUIRED_HIP_KERNEL_ENV {
        require_environment_value(name)?;
    }
    Ok(())
}

fn require_environment_value(name: &str) -> Result<(), String> {
    if env::var(name).ok().as_deref() != Some("1") {
        return Err(format!(
            "{name} must be set to exactly 1 by the caller; this binary does not set GPU or kernel environment variables"
        ));
    }
    Ok(())
}

fn require_isolated_r9700(stage: &str) -> Result<(), String> {
    let count = ullm_runtime_sys::device_count()
        .map_err(|error| format!("failed to query runtime device count {stage}: {error}"))?;
    if count != RUNTIME_HIP_DEVICE_INDEX + 1 {
        return Err(format!(
            "{stage}: expected exactly CPU device 0 and isolated HIP device 1, found {count} runtime devices"
        ));
    }
    let info = ullm_runtime_sys::device_info(RUNTIME_HIP_DEVICE_INDEX)
        .map_err(|error| format!("failed to query runtime device 1 {stage}: {error}"))?;
    if info.backend != "hip" || info.gcn_arch_name != REQUIRED_ARCHITECTURE {
        return Err(format!(
            "{stage}: runtime device 1 must be HIP {REQUIRED_ARCHITECTURE}, got backend={} architecture={}",
            info.backend, info.gcn_arch_name
        ));
    }
    Ok(())
}

fn deterministic_token_ids(vocab: usize) -> Result<Vec<usize>, String> {
    if vocab == 0 {
        return Err("loaded model has an empty vocabulary".to_string());
    }
    Ok((0..TOKEN_COUNT)
        .map(|index| (17 + index * 7919) % vocab)
        .collect())
}

fn run_cold_prefill(
    model: &mut Qwen35Aq4ModelRuntime,
    token_ids: &[usize],
) -> Result<Duration, String> {
    if token_ids.len() != TOKEN_COUNT || TOKEN_COUNT % PREFILL_CHUNK_WIDTH != 0 {
        return Err("fixed prefill timing sequence does not divide into M=128 chunks".to_string());
    }

    let mut offset = 0usize;
    let started = Instant::now();
    let mut elapsed = None;
    let mut actual_width = 0usize;
    while offset < token_ids.len() {
        let width = PREFILL_CHUNK_WIDTH.min(token_ids.len() - offset);
        let final_chunk = offset + width == token_ids.len();
        let label = format!("aq4-e2e-prefill-timing-prefill-{offset}");
        let step = model.dispatch_prefill_chunk_for_phase(
            &token_ids[offset..offset + width],
            QWEN35_AQ4_ROTARY_DIM,
            QWEN35_AQ4_ROPE_BASE,
            offset,
            ExecutionPhase::ColdPrefill,
            false,
            &label,
        )?;
        if step.execution_width != width {
            return Err(format!(
                "{label}: native prefill execution width {} differs from requested {width}",
                step.execution_width
            ));
        }
        if step.invocations.is_empty() {
            return Err(format!(
                "{label}: native prefill completed no layer invocations"
            ));
        }
        for invocation in step.invocations {
            if invocation.execution_width != width
                || invocation.phase != ExecutionPhase::ColdPrefill
            {
                return Err(format!(
                    "{label}: prefill invocation did not preserve cold M={width} execution"
                ));
            }
            if invocation
                .records
                .iter()
                .any(|record| record.status != OperationExecutionStatus::Succeeded)
            {
                return Err(format!("{label}: a prefill operation did not succeed"));
            }
        }
        model.synchronize()?;
        if final_chunk {
            elapsed = Some(started.elapsed());
        }
        actual_width = actual_width.max(width);
        offset += width;
    }
    if actual_width != PREFILL_CHUNK_WIDTH {
        return Err(format!(
            "actual prefill width {actual_width} differs from required M={PREFILL_CHUNK_WIDTH}"
        ));
    }
    elapsed.ok_or_else(|| "cold prefill did not reach a final synchronized chunk".to_string())
}
