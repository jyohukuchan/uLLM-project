const LINEAR_ATTN_QKV_PREPARE_SHUFFLE_KEY_HEADS: usize = 16;
const LINEAR_ATTN_QKV_PREPARE_SHUFFLE_VALUE_HEADS: usize = 32;
const LINEAR_ATTN_QKV_PREPARE_SHUFFLE_KEY_DIM: usize = 128;
const LINEAR_ATTN_QKV_PREPARE_SHUFFLE_VALUE_DIM: usize = 128;
const LINEAR_ATTN_QKV_PREPARE_SHUFFLE_KERNEL_SIZE: usize = 4;

struct LinearAttnQkvPrepareShuffleFixture {
    qkv: Vec<f32>,
    conv_weight: Vec<f32>,
    conv_history: Vec<f32>,
}

struct LinearAttnQkvPrepareShuffleOutput {
    conv: Vec<f32>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    history: Vec<f32>,
}

fn linear_attn_qkv_prepare_shuffle_prototype_gpu_device() -> u32 {
    (1..device_count().unwrap())
        .find(|&candidate| {
            device_info(candidate)
                .map(|info| info.gcn_arch_name == "gfx1201")
                .unwrap_or(false)
        })
        .expect("isolated gfx1201 HIP device")
}

fn linear_attn_qkv_prepare_shuffle_prototype_channels() -> usize {
    LINEAR_ATTN_QKV_PREPARE_SHUFFLE_KEY_HEADS * LINEAR_ATTN_QKV_PREPARE_SHUFFLE_KEY_DIM * 2
        + LINEAR_ATTN_QKV_PREPARE_SHUFFLE_VALUE_HEADS * LINEAR_ATTN_QKV_PREPARE_SHUFFLE_VALUE_DIM
}

fn linear_attn_qkv_prepare_shuffle_prototype_q_scale() -> f32 {
    1.0_f32 / (LINEAR_ATTN_QKV_PREPARE_SHUFFLE_KEY_DIM as f32).sqrt()
}

fn linear_attn_qkv_prepare_shuffle_prototype_fixture() -> LinearAttnQkvPrepareShuffleFixture {
    let channels = linear_attn_qkv_prepare_shuffle_prototype_channels();
    let mut state = 0x6a09_e667_u32;
    let mut next = || {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        state as f32 / u32::MAX as f32
    };
    LinearAttnQkvPrepareShuffleFixture {
        qkv: (0..channels).map(|_| next() - 0.5).collect(),
        conv_weight: (0..channels * LINEAR_ATTN_QKV_PREPARE_SHUFFLE_KERNEL_SIZE)
            .map(|_| next() - 0.5)
            .collect(),
        conv_history: (0..channels * LINEAR_ATTN_QKV_PREPARE_SHUFFLE_KERNEL_SIZE)
            .map(|_| next() - 0.5)
            .collect(),
    }
}

fn linear_attn_qkv_prepare_shuffle_prototype_cpu_reference(
    fixture: &LinearAttnQkvPrepareShuffleFixture,
) -> LinearAttnQkvPrepareShuffleOutput {
    let mut history = fixture.conv_history.clone();
    let (conv, q, k, v) = expected_linear_attn_qkv_prepare(
        &fixture.qkv,
        &fixture.conv_weight,
        &mut history,
        LINEAR_ATTN_QKV_PREPARE_SHUFFLE_KEY_HEADS,
        LINEAR_ATTN_QKV_PREPARE_SHUFFLE_VALUE_HEADS,
        LINEAR_ATTN_QKV_PREPARE_SHUFFLE_KEY_DIM,
        LINEAR_ATTN_QKV_PREPARE_SHUFFLE_VALUE_DIM,
        LINEAR_ATTN_QKV_PREPARE_SHUFFLE_KERNEL_SIZE,
        linear_attn_qkv_prepare_shuffle_prototype_q_scale(),
        true,
    );
    LinearAttnQkvPrepareShuffleOutput {
        conv,
        q,
        k,
        v,
        history,
    }
}

#[allow(clippy::too_many_arguments)]
fn linear_attn_qkv_prepare_shuffle_prototype_launch(
    prototype: bool,
    qkv: &RuntimeBuffer,
    conv_weight: &RuntimeBuffer,
    conv_history: &mut RuntimeBuffer,
    conv_output: &mut RuntimeBuffer,
    q_output: &mut RuntimeBuffer,
    k_output: &mut RuntimeBuffer,
    v_output: &mut RuntimeBuffer,
    stream: &mut RuntimeStream,
) {
    if prototype {
        linear_attn_qkv_prepare_shuffle_prototype_f32(
            qkv,
            conv_weight,
            conv_history,
            LINEAR_ATTN_QKV_PREPARE_SHUFFLE_KEY_HEADS,
            LINEAR_ATTN_QKV_PREPARE_SHUFFLE_VALUE_HEADS,
            LINEAR_ATTN_QKV_PREPARE_SHUFFLE_KEY_DIM,
            LINEAR_ATTN_QKV_PREPARE_SHUFFLE_VALUE_DIM,
            LINEAR_ATTN_QKV_PREPARE_SHUFFLE_KERNEL_SIZE,
            linear_attn_qkv_prepare_shuffle_prototype_q_scale(),
            true,
            conv_output,
            q_output,
            k_output,
            v_output,
            Some(&mut *stream),
        )
        .unwrap();
    } else {
        linear_attn_qkv_prepare_f32(
            qkv,
            conv_weight,
            conv_history,
            LINEAR_ATTN_QKV_PREPARE_SHUFFLE_KEY_HEADS,
            LINEAR_ATTN_QKV_PREPARE_SHUFFLE_VALUE_HEADS,
            LINEAR_ATTN_QKV_PREPARE_SHUFFLE_KEY_DIM,
            LINEAR_ATTN_QKV_PREPARE_SHUFFLE_VALUE_DIM,
            LINEAR_ATTN_QKV_PREPARE_SHUFFLE_KERNEL_SIZE,
            linear_attn_qkv_prepare_shuffle_prototype_q_scale(),
            true,
            conv_output,
            q_output,
            k_output,
            v_output,
            Some(&mut *stream),
        )
        .unwrap();
    }
}

fn linear_attn_qkv_prepare_shuffle_prototype_read(
    buffer: &RuntimeBuffer,
    elements: usize,
    stream: &mut RuntimeStream,
) -> Vec<f32> {
    let mut bytes = vec![0_u8; elements * std::mem::size_of::<f32>()];
    buffer
        .copy_to_host(0, &mut bytes, Some(&mut *stream))
        .unwrap();
    stream.synchronize().unwrap();
    le_bytes_to_f32s(&bytes)
}

fn linear_attn_qkv_prepare_shuffle_prototype_run(
    device: u32,
    prototype: bool,
    fixture: &LinearAttnQkvPrepareShuffleFixture,
) -> LinearAttnQkvPrepareShuffleOutput {
    let channels = linear_attn_qkv_prepare_shuffle_prototype_channels();
    let q_elements =
        LINEAR_ATTN_QKV_PREPARE_SHUFFLE_KEY_HEADS * LINEAR_ATTN_QKV_PREPARE_SHUFFLE_KEY_DIM;
    let v_elements =
        LINEAR_ATTN_QKV_PREPARE_SHUFFLE_VALUE_HEADS * LINEAR_ATTN_QKV_PREPARE_SHUFFLE_VALUE_DIM;
    let f32_bytes = std::mem::size_of::<f32>();
    let mut context = RuntimeContext::create(device).unwrap();
    let mut stream = context.create_stream().unwrap();
    let mut qkv = context.alloc_buffer(channels * f32_bytes).unwrap();
    let mut conv_weight = context
        .alloc_buffer(fixture.conv_weight.len() * f32_bytes)
        .unwrap();
    let mut conv_history = context
        .alloc_buffer(fixture.conv_history.len() * f32_bytes)
        .unwrap();
    let mut conv_output = context.alloc_buffer(channels * f32_bytes).unwrap();
    let mut q_output = context.alloc_buffer(q_elements * f32_bytes).unwrap();
    let mut k_output = context.alloc_buffer(q_elements * f32_bytes).unwrap();
    let mut v_output = context.alloc_buffer(v_elements * f32_bytes).unwrap();
    qkv.copy_from_host(0, &f32s_to_le_bytes(&fixture.qkv), Some(&mut stream))
        .unwrap();
    conv_weight
        .copy_from_host(
            0,
            &f32s_to_le_bytes(&fixture.conv_weight),
            Some(&mut stream),
        )
        .unwrap();
    conv_history
        .copy_from_host(
            0,
            &f32s_to_le_bytes(&fixture.conv_history),
            Some(&mut stream),
        )
        .unwrap();
    stream.synchronize().unwrap();

    linear_attn_qkv_prepare_shuffle_prototype_launch(
        prototype,
        &qkv,
        &conv_weight,
        &mut conv_history,
        &mut conv_output,
        &mut q_output,
        &mut k_output,
        &mut v_output,
        &mut stream,
    );
    stream.synchronize().unwrap();
    LinearAttnQkvPrepareShuffleOutput {
        conv: linear_attn_qkv_prepare_shuffle_prototype_read(&conv_output, channels, &mut stream),
        q: linear_attn_qkv_prepare_shuffle_prototype_read(&q_output, q_elements, &mut stream),
        k: linear_attn_qkv_prepare_shuffle_prototype_read(&k_output, q_elements, &mut stream),
        v: linear_attn_qkv_prepare_shuffle_prototype_read(&v_output, v_elements, &mut stream),
        history: linear_attn_qkv_prepare_shuffle_prototype_read(
            &conv_history,
            fixture.conv_history.len(),
            &mut stream,
        ),
    }
}

fn linear_attn_qkv_prepare_shuffle_prototype_time(
    device: u32,
    prototype: bool,
    fixture: &LinearAttnQkvPrepareShuffleFixture,
    rounds: usize,
) -> f64 {
    assert!(rounds > 0);
    let channels = linear_attn_qkv_prepare_shuffle_prototype_channels();
    let q_elements =
        LINEAR_ATTN_QKV_PREPARE_SHUFFLE_KEY_HEADS * LINEAR_ATTN_QKV_PREPARE_SHUFFLE_KEY_DIM;
    let v_elements =
        LINEAR_ATTN_QKV_PREPARE_SHUFFLE_VALUE_HEADS * LINEAR_ATTN_QKV_PREPARE_SHUFFLE_VALUE_DIM;
    let f32_bytes = std::mem::size_of::<f32>();
    let mut context = RuntimeContext::create(device).unwrap();
    let mut stream = context.create_stream().unwrap();
    let mut qkv = context.alloc_buffer(channels * f32_bytes).unwrap();
    let mut conv_weight = context
        .alloc_buffer(fixture.conv_weight.len() * f32_bytes)
        .unwrap();
    let mut conv_history = context
        .alloc_buffer(fixture.conv_history.len() * f32_bytes)
        .unwrap();
    let mut conv_output = context.alloc_buffer(channels * f32_bytes).unwrap();
    let mut q_output = context.alloc_buffer(q_elements * f32_bytes).unwrap();
    let mut k_output = context.alloc_buffer(q_elements * f32_bytes).unwrap();
    let mut v_output = context.alloc_buffer(v_elements * f32_bytes).unwrap();
    qkv.copy_from_host(0, &f32s_to_le_bytes(&fixture.qkv), Some(&mut stream))
        .unwrap();
    conv_weight
        .copy_from_host(
            0,
            &f32s_to_le_bytes(&fixture.conv_weight),
            Some(&mut stream),
        )
        .unwrap();
    conv_history
        .copy_from_host(
            0,
            &f32s_to_le_bytes(&fixture.conv_history),
            Some(&mut stream),
        )
        .unwrap();
    stream.synchronize().unwrap();

    for _ in 0..3 {
        linear_attn_qkv_prepare_shuffle_prototype_launch(
            prototype,
            &qkv,
            &conv_weight,
            &mut conv_history,
            &mut conv_output,
            &mut q_output,
            &mut k_output,
            &mut v_output,
            &mut stream,
        );
    }
    stream.synchronize().unwrap();
    let started = std::time::Instant::now();
    for _ in 0..rounds {
        linear_attn_qkv_prepare_shuffle_prototype_launch(
            prototype,
            &qkv,
            &conv_weight,
            &mut conv_history,
            &mut conv_output,
            &mut q_output,
            &mut k_output,
            &mut v_output,
            &mut stream,
        );
    }
    stream.synchronize().unwrap();
    started.elapsed().as_secs_f64() * 1_000.0 / rounds as f64
}

fn assert_linear_attn_qkv_prepare_shuffle_prototype_matches_cpu(
    actual: &[f32],
    expected: &[f32],
    label: &str,
) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{label}: output lengths differ"
    );
    // The wave reduction changes FP32 association order. Keep the established AQ4 differential
    // convention: 0.05 absolute plus 1% of the CPU-reference magnitude.
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        let tolerance = 5e-2_f32 + 1e-2_f32 * expected.abs();
        assert!(
            (actual - expected).abs() <= tolerance,
            "{label} index {index}: actual={actual} expected={expected} tolerance={tolerance}"
        );
    }
}

#[test]
fn cpu_linear_attn_qkv_prepare_shuffle_prototype_rejects_cpu_backend() {
    let channels = linear_attn_qkv_prepare_shuffle_prototype_channels();
    let q_elements =
        LINEAR_ATTN_QKV_PREPARE_SHUFFLE_KEY_HEADS * LINEAR_ATTN_QKV_PREPARE_SHUFFLE_KEY_DIM;
    let v_elements =
        LINEAR_ATTN_QKV_PREPARE_SHUFFLE_VALUE_HEADS * LINEAR_ATTN_QKV_PREPARE_SHUFFLE_VALUE_DIM;
    let f32_bytes = std::mem::size_of::<f32>();
    let mut context = RuntimeContext::create(0).unwrap();
    let qkv = context.alloc_buffer(channels * f32_bytes).unwrap();
    let conv_weight = context
        .alloc_buffer(channels * LINEAR_ATTN_QKV_PREPARE_SHUFFLE_KERNEL_SIZE * f32_bytes)
        .unwrap();
    let mut conv_history = context
        .alloc_buffer(channels * LINEAR_ATTN_QKV_PREPARE_SHUFFLE_KERNEL_SIZE * f32_bytes)
        .unwrap();
    let mut conv_output = context.alloc_buffer(channels * f32_bytes).unwrap();
    let mut q_output = context.alloc_buffer(q_elements * f32_bytes).unwrap();
    let mut k_output = context.alloc_buffer(q_elements * f32_bytes).unwrap();
    let mut v_output = context.alloc_buffer(v_elements * f32_bytes).unwrap();

    let error = linear_attn_qkv_prepare_shuffle_prototype_f32(
        &qkv,
        &conv_weight,
        &mut conv_history,
        LINEAR_ATTN_QKV_PREPARE_SHUFFLE_KEY_HEADS,
        LINEAR_ATTN_QKV_PREPARE_SHUFFLE_VALUE_HEADS,
        LINEAR_ATTN_QKV_PREPARE_SHUFFLE_KEY_DIM,
        LINEAR_ATTN_QKV_PREPARE_SHUFFLE_VALUE_DIM,
        LINEAR_ATTN_QKV_PREPARE_SHUFFLE_KERNEL_SIZE,
        linear_attn_qkv_prepare_shuffle_prototype_q_scale(),
        true,
        &mut conv_output,
        &mut q_output,
        &mut k_output,
        &mut v_output,
        None,
    )
    .expect_err("the direct prototype must not stage or fall back on CPU");
    assert!(error.contains("requires a HIP gfx1201 backend"));
}

#[test]
#[ignore = "requires an isolated gfx1201 HIP device and ULLM_RUN_LINEAR_ATTN_QKV_PREPARE_SHUFFLE_PROTOTYPE_DIFFERENTIAL=1"]
fn hip_linear_attn_qkv_prepare_shuffle_prototype_qwen35_m1_matches_cpu_when_enabled() {
    assert_eq!(
        std::env::var("ULLM_RUN_LINEAR_ATTN_QKV_PREPARE_SHUFFLE_PROTOTYPE_DIFFERENTIAL").as_deref(),
        Ok("1"),
        "set ULLM_RUN_LINEAR_ATTN_QKV_PREPARE_SHUFFLE_PROTOTYPE_DIFFERENTIAL=1 before running this GPU differential test"
    );
    let _lock = AQ4_EXPERIMENTAL_ENV_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let fixture = linear_attn_qkv_prepare_shuffle_prototype_fixture();
    let expected = linear_attn_qkv_prepare_shuffle_prototype_cpu_reference(&fixture);
    let actual = linear_attn_qkv_prepare_shuffle_prototype_run(
        linear_attn_qkv_prepare_shuffle_prototype_gpu_device(),
        true,
        &fixture,
    );
    assert_linear_attn_qkv_prepare_shuffle_prototype_matches_cpu(
        &actual.conv,
        &expected.conv,
        "conv output",
    );
    assert_linear_attn_qkv_prepare_shuffle_prototype_matches_cpu(
        &actual.q,
        &expected.q,
        "q output",
    );
    assert_linear_attn_qkv_prepare_shuffle_prototype_matches_cpu(
        &actual.k,
        &expected.k,
        "k output",
    );
    assert_linear_attn_qkv_prepare_shuffle_prototype_matches_cpu(
        &actual.v,
        &expected.v,
        "v output",
    );
    assert_linear_attn_qkv_prepare_shuffle_prototype_matches_cpu(
        &actual.history,
        &expected.history,
        "conv history",
    );
    let max_q_abs_diff = actual
        .q
        .iter()
        .zip(&expected.q)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0_f32, f32::max);
    let max_k_abs_diff = actual
        .k
        .iter()
        .zip(&expected.k)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0_f32, f32::max);
    eprintln!(
        "linear-attn qkv-prepare shuffle differential passed M=1 heads=16/32 dims=128/128 kernel=4 max_q_abs_diff={max_q_abs_diff:.9} max_k_abs_diff={max_k_abs_diff:.9}"
    );
}

#[test]
#[ignore = "requires an isolated gfx1201 HIP device and ULLM_RUN_LINEAR_ATTN_QKV_PREPARE_SHUFFLE_PROTOTYPE_TIMING=1"]
fn hip_linear_attn_qkv_prepare_shuffle_prototype_qwen35_m1_timing_vs_production_when_enabled() {
    assert_eq!(
        std::env::var("ULLM_RUN_LINEAR_ATTN_QKV_PREPARE_SHUFFLE_PROTOTYPE_TIMING").as_deref(),
        Ok("1"),
        "set ULLM_RUN_LINEAR_ATTN_QKV_PREPARE_SHUFFLE_PROTOTYPE_TIMING=1 before running this GPU timing test"
    );
    let _lock = AQ4_EXPERIMENTAL_ENV_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let _require_hip = ExperimentalEnvGuard::new("ULLM_REQUIRE_HIP_LINEAR_ATTN_KERNEL", Some("1"));
    let fixture = linear_attn_qkv_prepare_shuffle_prototype_fixture();
    let gpu = linear_attn_qkv_prepare_shuffle_prototype_gpu_device();
    let production = linear_attn_qkv_prepare_shuffle_prototype_time(gpu, false, &fixture, 100);
    let shuffle = linear_attn_qkv_prepare_shuffle_prototype_time(gpu, true, &fixture, 100);
    assert!(production > 0.0 && shuffle > 0.0);
    eprintln!(
        "linear-attn qkv-prepare shuffle M=1 heads=16/32 dims=128/128 kernel=4: production={production:.6} ms, shuffle={shuffle:.6} ms, speedup={:.3}x",
        production / shuffle
    );
}
