// Copyright 2026 uLLM contributors
// SPDX-License-Identifier: Apache-2.0

//! Clean, per-decode-step Qwen3.5 AQ4 profiler driver for the isolated R9700.
//!
//! The warmup request is completely finished and reset before a fresh request is prefetched to
//! the requested context.  Consequently the first marked decode starts with exactly the caller's
//! context length while still benefiting from the warmup's in-process module/cache state.
//!
//! Usage: `ullm-aq4-decode-step-profile [TARGET_CONTEXT_LENGTH] [--warmup STEPS] [--measured STEPS]`

use serde_json::json;
use std::env;
use std::io::{self, Write};
use std::process::ExitCode;
use std::time::Instant;

use ullm_engine::aq4_worker_backend::QWEN35_AQ4_REQUIRED_HIP_KERNEL_ENV;
use ullm_engine::inference_api::{CancellationToken, InferenceRequest, SamplingParams};
use ullm_engine::qwen35_aq4_head_runtime::PackageLmHeadMode;
use ullm_engine::qwen35_aq4_model_runtime::{
    QWEN35_AQ4_CONTEXT_LENGTH, QWEN35_AQ4_KV_BLOCK_SIZE, Qwen35Aq4ModelLoadConfig,
};
use ullm_engine::qwen35_aq4_session::{Qwen35Aq4InferenceSession, Qwen35Aq4SessionConfig};
use ullm_engine::worker_driver::{InferenceSession, PublishedAdvance, SessionAdvance};

const PACKAGE_DIR: &str = "/home/homelab1/datapool/ullm/product/qwen35-9b-aq4-cli-v0.1/package";
const RUNTIME_HIP_DEVICE_INDEX: u32 = 1;
const REQUIRED_ARCHITECTURE: &str = "gfx1201";
const LOAD_CHUNK_BYTES: usize = 1024 * 1024;
const LM_HEAD_CHUNK_ROWS: usize = 8192;
const PREFILL_CHUNK_WIDTH: usize = 128;
const EOS_TOKEN_IDS: [usize; 2] = [248_044, 248_046];
const DEFAULT_TARGET_CONTEXT_LENGTH: usize = 1339;
const DEFAULT_WARMUP_STEPS: usize = 6;
const DEFAULT_MEASURED_STEPS: usize = 32;

#[derive(Debug, Clone, Copy)]
struct Args {
    target_context_length: usize,
    warmup_steps: usize,
    measured_steps: usize,
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("ullm-aq4-decode-step-profile: {error}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), String> {
    let args = args_from_caller()?;
    require_caller_environment()?;
    require_isolated_r9700("before model load")?;

    let largest_request_tokens = args
        .warmup_steps
        .max(args.measured_steps)
        .checked_add(1)
        .ok_or_else(|| "decode step count overflows request token count".to_string())?;
    let session_config =
        Qwen35Aq4SessionConfig::greedy(largest_request_tokens, EOS_TOKEN_IDS.to_vec())
            .with_prefill_chunk_tokens(PREFILL_CHUNK_WIDTH)?;
    let mut session = Qwen35Aq4InferenceSession::load(
        Qwen35Aq4ModelLoadConfig {
            package_dir: PACKAGE_DIR.into(),
            device_index: RUNTIME_HIP_DEVICE_INDEX,
            expected_architecture: Some(REQUIRED_ARCHITECTURE.to_string()),
            chunk_bytes: LOAD_CHUNK_BYTES,
            context_length: QWEN35_AQ4_CONTEXT_LENGTH,
            kv_block_size: QWEN35_AQ4_KV_BLOCK_SIZE,
            layer_indices: None,
            lm_head_mode: PackageLmHeadMode::GpuResidentF32,
            lm_head_chunk_rows: LM_HEAD_CHUNK_ROWS,
        },
        session_config,
    )?;

    require_isolated_r9700("after model load")?;
    if session.model().backend() != "hip" {
        return Err(format!(
            "loaded model backend must be hip, got {}",
            session.model().backend()
        ));
    }
    let context_limit = session.model().geometry().context_length;
    if args.target_context_length < 2 {
        return Err(
            "target context length must be at least 2 for native AQ4 session prefill".into(),
        );
    }
    let required_context = args
        .target_context_length
        .checked_add(largest_request_tokens)
        .ok_or_else(|| "target context plus decode steps overflows".to_string())?;
    if required_context > context_limit {
        return Err(format!(
            "target context plus the seed and longest requested decode run is {required_context}, exceeding loaded context length {context_limit}"
        ));
    }
    let prompt =
        deterministic_token_ids(args.target_context_length, session.model().geometry().vocab)?;

    // Warming a complete request, then resetting it, keeps the measured request at the exact
    // requested cache length rather than shifting every marked sample by the warmup count.
    if args.warmup_steps != 0 {
        run_unmarked_request(&mut session, &prompt, args.warmup_steps, "warmup")?;
    }
    seed_request_for_decode(&mut session, &prompt, args.measured_steps, "measured")?;

    // The shared decoder-stack and LM-head ranges are inert until this explicit opt-in.  Enabling
    // only after the fresh measured request has been seeded prevents prefill and seed top-1 from
    // appearing as measurement ranges.
    ullm_engine::roctx::enable()?;
    write_json_line(json!({
        "event": "configuration",
        "target_context_length": args.target_context_length,
        "warmup_steps": args.warmup_steps,
        "measured_steps": args.measured_steps,
        "prefill_chunk_width": PREFILL_CHUNK_WIDTH,
        "marker_ranges": [
            "ullm.aq4.decode.step.v1/...",
            "ullm.aq4.decode.decoder_stack.v1",
            "ullm.aq4.decode.lm_head_top1.v1"
        ],
        "synchronization": {
            "prefill": "session synchronizes after every prefill chunk",
            "decode_each_layer": false,
            "decode": "no explicit post-decoder synchronization; the production LM-head top-1 device-to-host readback synchronizes the same stream before prepare_advance returns",
            "publish": "CPU-only commit; no GPU synchronization",
            "shutdown": "session shutdown synchronizes after the final reset"
        }
    }))?;

    let mut samples = Vec::with_capacity(args.measured_steps);
    for step_index in 0..args.measured_steps {
        let cache_start = args
            .target_context_length
            .checked_add(step_index)
            .ok_or_else(|| "measured cache length overflows".to_string())?;
        let label =
            format!("ullm.aq4.decode.step.v1/step_index={step_index}/cache_start={cache_start}");
        let (token_id, cache_end, elapsed_seconds) = {
            let started = Instant::now();
            let _step_range = ullm_engine::roctx::range(&label);
            let (token_id, cache_end, _terminal) = prepare_and_publish_decode_step(
                &mut session,
                cache_start,
                step_index,
                args.measured_steps,
                "measured",
            )?;
            (token_id, cache_end, started.elapsed().as_secs_f64())
        };
        write_json_line(json!({
            "event": "measured_decode_step",
            "step_index": step_index,
            "token_id": token_id,
            "cache_len_start": cache_start,
            "cache_len_end": cache_end,
            "elapsed_seconds": elapsed_seconds,
        }))?;
        samples.push(elapsed_seconds);
    }

    let total_seconds = samples.iter().sum::<f64>();
    let mean_seconds = total_seconds / samples.len() as f64;
    let min_seconds = samples.iter().copied().fold(f64::INFINITY, f64::min);
    let max_seconds = samples.iter().copied().fold(0.0_f64, f64::max);
    write_json_line(json!({
        "event": "summary",
        "warmup_count": args.warmup_steps,
        "measured_count": args.measured_steps,
        "mean_step_seconds": mean_seconds,
        "min_step_seconds": min_seconds,
        "max_step_seconds": max_seconds,
        "mean_tokens_per_second": args.measured_steps as f64 / total_seconds,
    }))?;

    // The last measured publish reaches the request length limit.  Finish uses the ordinary
    // synchronized request-state reset; it intentionally happens after the outer step range.
    session.finish_and_reset()?;
    session.shutdown()
}

fn args_from_caller() -> Result<Args, String> {
    let mut target_context_length = DEFAULT_TARGET_CONTEXT_LENGTH;
    let mut warmup_steps = DEFAULT_WARMUP_STEPS;
    let mut measured_steps = DEFAULT_MEASURED_STEPS;
    let mut target_seen = false;
    let mut arguments = env::args().skip(1);
    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "--help" | "-h" => return Err(usage().to_string()),
            "--warmup" => {
                warmup_steps = parse_positive_or_zero("--warmup", arguments.next())?;
            }
            "--measured" => {
                measured_steps = parse_positive_or_zero("--measured", arguments.next())?;
            }
            value if value.starts_with('-') => {
                return Err(format!("unknown argument {value:?}; {}", usage()));
            }
            value if !target_seen => {
                target_context_length = parse_positive("TARGET_CONTEXT_LENGTH", value)?;
                target_seen = true;
            }
            value => {
                return Err(format!(
                    "unexpected positional argument {value:?}; {}",
                    usage()
                ));
            }
        }
    }
    if measured_steps == 0 {
        return Err("--measured must be positive".to_string());
    }
    Ok(Args {
        target_context_length,
        warmup_steps,
        measured_steps,
    })
}

fn usage() -> &'static str {
    "usage: ullm-aq4-decode-step-profile [TARGET_CONTEXT_LENGTH] [--warmup STEPS] [--measured STEPS]"
}

fn parse_positive(label: &str, value: impl AsRef<str>) -> Result<usize, String> {
    let value = value.as_ref();
    let parsed = value
        .parse::<usize>()
        .map_err(|error| format!("invalid {label} {value:?}: {error}"))?;
    if parsed == 0 {
        return Err(format!("{label} must be positive"));
    }
    Ok(parsed)
}

fn parse_positive_or_zero(label: &str, value: Option<String>) -> Result<usize, String> {
    let value = value.ok_or_else(|| format!("{label} requires a value"))?;
    value
        .parse::<usize>()
        .map_err(|error| format!("invalid {label} {value:?}: {error}"))
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

fn deterministic_token_ids(token_count: usize, vocab: usize) -> Result<Vec<usize>, String> {
    if vocab == 0 {
        return Err("loaded model has an empty vocabulary".to_string());
    }
    Ok((0..token_count)
        .map(|index| (17 + index * 7919) % vocab)
        .collect())
}

fn run_unmarked_request(
    session: &mut Qwen35Aq4InferenceSession,
    prompt: &[usize],
    decode_steps: usize,
    request_kind: &str,
) -> Result<(), String> {
    seed_request_for_decode(session, prompt, decode_steps, request_kind)?;
    for step_index in 0..decode_steps {
        let cache_start = prompt
            .len()
            .checked_add(step_index)
            .ok_or_else(|| "warmup cache length overflows".to_string())?;
        let _ = prepare_and_publish_decode_step(
            session,
            cache_start,
            step_index,
            decode_steps,
            request_kind,
        )?;
    }
    session.finish_and_reset()?;
    Ok(())
}

fn seed_request_for_decode(
    session: &mut Qwen35Aq4InferenceSession,
    prompt: &[usize],
    decode_steps: usize,
    request_kind: &str,
) -> Result<(), String> {
    let request = InferenceRequest::new_with_eos(
        format!("aq4-decode-step-profile-{request_kind}"),
        prompt.to_vec(),
        decode_steps
            .checked_add(1)
            .ok_or_else(|| "request seed plus decode step count overflows".to_string())?,
        EOS_TOKEN_IDS.to_vec(),
        SamplingParams::greedy_with_top_k(0, 1),
    );
    session.start_request(request, CancellationToken::new())?;
    loop {
        match session.prepare_advance()? {
            SessionAdvance::PromptProgress {
                prompt_tokens_processed,
                cache_len,
                execution_width,
            } => {
                if prompt_tokens_processed > prompt.len()
                    || cache_len != prompt_tokens_processed
                    || execution_width == 0
                {
                    return Err(format!(
                        "{request_kind} prefill violated session progress contract: processed={prompt_tokens_processed}, cache_len={cache_len}, execution_width={execution_width}"
                    ));
                }
                if prompt_tokens_processed == prompt.len() {
                    continue;
                }
            }
            SessionAdvance::Token {
                prepared,
                cache_len,
                terminal_reason,
                ..
            } => {
                if cache_len != prompt.len() {
                    return Err(format!(
                        "{request_kind} seed cache length {cache_len} differs from target {}",
                        prompt.len()
                    ));
                }
                if terminal_reason.is_some() {
                    return Err(format!(
                        "{request_kind} seed token unexpectedly terminated the request"
                    ));
                }
                match session.publish_prepared(prepared, |_| Ok(()))? {
                    PublishedAdvance::Token {
                        cache_len: published_cache_len,
                        terminal_reason: None,
                        ..
                    } if published_cache_len == prompt.len() => return Ok(()),
                    PublishedAdvance::Token {
                        cache_len: published_cache_len,
                        terminal_reason,
                        ..
                    } => {
                        return Err(format!(
                            "{request_kind} seed publication did not leave a decodable request: cache_len={published_cache_len}, terminal_reason={terminal_reason:?}"
                        ));
                    }
                    PublishedAdvance::CancellationObserved => {
                        return Err(format!(
                            "{request_kind} seed unexpectedly observed cancellation"
                        ));
                    }
                }
            }
            SessionAdvance::CancellationObserved => {
                return Err(format!(
                    "{request_kind} prefill unexpectedly observed cancellation"
                ));
            }
        }
    }
}

fn prepare_and_publish_decode_step(
    session: &mut Qwen35Aq4InferenceSession,
    cache_start: usize,
    step_index: usize,
    total_steps: usize,
    request_kind: &str,
) -> Result<
    (
        usize,
        usize,
        Option<ullm_engine::inference_api::FinishReason>,
    ),
    String,
> {
    let (prepared, token_id, cache_end, terminal_reason) = match session.prepare_advance()? {
        SessionAdvance::Token {
            prepared,
            token_id,
            cache_len,
            terminal_reason,
            ..
        } => (prepared, token_id, cache_len, terminal_reason),
        SessionAdvance::PromptProgress { .. } => {
            return Err(format!(
                "{request_kind} decode step unexpectedly returned prompt progress"
            ));
        }
        SessionAdvance::CancellationObserved => {
            return Err(format!(
                "{request_kind} decode step unexpectedly observed cancellation"
            ));
        }
    };
    let expected_cache_end = cache_start
        .checked_add(1)
        .ok_or_else(|| "decode cache length overflows".to_string())?;
    if cache_end != expected_cache_end {
        return Err(format!(
            "{request_kind} decode step {step_index} cache length mismatch: start={cache_start}, end={cache_end}, expected_end={expected_cache_end}"
        ));
    }
    let should_terminate = step_index + 1 == total_steps;
    if terminal_reason.is_some() != should_terminate {
        return Err(format!(
            "{request_kind} decode step {step_index} terminal state differs from requested fixed-length run: terminal_reason={terminal_reason:?}, total_steps={total_steps}"
        ));
    }
    match session.publish_prepared(prepared, |_| Ok(()))? {
        PublishedAdvance::Token {
            token_id: published_token_id,
            cache_len: published_cache_len,
            terminal_reason: published_terminal_reason,
            ..
        } if published_token_id == token_id
            && published_cache_len == cache_end
            && published_terminal_reason == terminal_reason => {}
        PublishedAdvance::Token {
            token_id: published_token_id,
            cache_len: published_cache_len,
            terminal_reason: published_terminal_reason,
            ..
        } => {
            return Err(format!(
                "{request_kind} decode step {step_index} publication changed token/state: token={published_token_id}/{token_id}, cache_len={published_cache_len}/{cache_end}, terminal={published_terminal_reason:?}/{terminal_reason:?}"
            ));
        }
        PublishedAdvance::CancellationObserved => {
            return Err(format!(
                "{request_kind} decode step unexpectedly observed cancellation"
            ));
        }
    }
    Ok((token_id, cache_end, terminal_reason))
}

fn write_json_line(value: serde_json::Value) -> Result<(), String> {
    let stdout = io::stdout();
    let mut stdout = stdout.lock();
    serde_json::to_writer(&mut stdout, &value)
        .map_err(|error| format!("failed to serialize diagnostic JSON line: {error}"))?;
    stdout
        .write_all(b"\n")
        .and_then(|_| stdout.flush())
        .map_err(|error| format!("failed to write diagnostic JSON line: {error}"))
}
