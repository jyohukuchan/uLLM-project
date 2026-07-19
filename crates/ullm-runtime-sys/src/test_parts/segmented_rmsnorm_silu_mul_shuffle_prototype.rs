const SEGMENTED_RMSNORM_SILU_MUL_SHUFFLE_PROTOTYPE_SEGMENTS: usize = 32;
const SEGMENTED_RMSNORM_SILU_MUL_SHUFFLE_PROTOTYPE_SEGMENT_SIZE: usize = 128;
const SEGMENTED_RMSNORM_SILU_MUL_SHUFFLE_PROTOTYPE_ELEMENTS: usize =
    SEGMENTED_RMSNORM_SILU_MUL_SHUFFLE_PROTOTYPE_SEGMENTS
        * SEGMENTED_RMSNORM_SILU_MUL_SHUFFLE_PROTOTYPE_SEGMENT_SIZE;

fn segmented_rmsnorm_silu_mul_shuffle_prototype_gpu_device() -> u32 {
    (1..device_count().unwrap())
        .find(|&candidate| {
            device_info(candidate)
                .map(|info| info.gcn_arch_name == "gfx1201")
                .unwrap_or(false)
        })
        .expect("isolated gfx1201 HIP device")
}

fn segmented_rmsnorm_silu_mul_shuffle_prototype_fixture() -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut state = 0x6a09_e667_u32;
    let mut next = || {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        state as f32 / u32::MAX as f32
    };
    let input = (0..SEGMENTED_RMSNORM_SILU_MUL_SHUFFLE_PROTOTYPE_ELEMENTS)
        .map(|_| next() * 2.0 - 1.0)
        .collect();
    let weight = (0..SEGMENTED_RMSNORM_SILU_MUL_SHUFFLE_PROTOTYPE_SEGMENT_SIZE)
        .map(|_| 0.5 + next())
        .collect();
    let gate = (0..SEGMENTED_RMSNORM_SILU_MUL_SHUFFLE_PROTOTYPE_ELEMENTS)
        .map(|_| next() * 2.0 - 1.0)
        .collect();
    (input, weight, gate)
}

fn segmented_rmsnorm_silu_mul_shuffle_prototype_launch(
    prototype: bool,
    input: &RuntimeBuffer,
    weight: &RuntimeBuffer,
    gate: &RuntimeBuffer,
    epsilon: f32,
    output: &mut RuntimeBuffer,
    stream: &mut RuntimeStream,
) {
    if prototype {
        segmented_rmsnorm_silu_mul_shuffle_prototype_f32(
            input,
            weight,
            gate,
            SEGMENTED_RMSNORM_SILU_MUL_SHUFFLE_PROTOTYPE_SEGMENTS,
            SEGMENTED_RMSNORM_SILU_MUL_SHUFFLE_PROTOTYPE_SEGMENT_SIZE,
            epsilon,
            output,
            Some(&mut *stream),
        )
        .unwrap();
    } else {
        segmented_rmsnorm_silu_mul_f32(
            input,
            weight,
            gate,
            SEGMENTED_RMSNORM_SILU_MUL_SHUFFLE_PROTOTYPE_SEGMENTS,
            SEGMENTED_RMSNORM_SILU_MUL_SHUFFLE_PROTOTYPE_SEGMENT_SIZE,
            epsilon,
            output,
            Some(&mut *stream),
        )
        .unwrap();
    }
}

fn segmented_rmsnorm_silu_mul_shuffle_prototype_run(
    device: u32,
    prototype: bool,
    input_values: &[f32],
    weight_values: &[f32],
    gate_values: &[f32],
    epsilon: f32,
) -> Vec<f32> {
    assert_eq!(
        input_values.len(),
        SEGMENTED_RMSNORM_SILU_MUL_SHUFFLE_PROTOTYPE_ELEMENTS
    );
    assert_eq!(
        weight_values.len(),
        SEGMENTED_RMSNORM_SILU_MUL_SHUFFLE_PROTOTYPE_SEGMENT_SIZE
    );
    assert_eq!(
        gate_values.len(),
        SEGMENTED_RMSNORM_SILU_MUL_SHUFFLE_PROTOTYPE_ELEMENTS
    );
    let f32_bytes = std::mem::size_of::<f32>();
    let mut context = RuntimeContext::create(device).unwrap();
    let mut stream = context.create_stream().unwrap();
    let mut input = context
        .alloc_buffer(input_values.len() * f32_bytes)
        .unwrap();
    let mut weight = context
        .alloc_buffer(weight_values.len() * f32_bytes)
        .unwrap();
    let mut gate = context.alloc_buffer(gate_values.len() * f32_bytes).unwrap();
    let mut output = context
        .alloc_buffer(input_values.len() * f32_bytes)
        .unwrap();
    input
        .copy_from_host(0, &f32s_to_le_bytes(input_values), Some(&mut stream))
        .unwrap();
    weight
        .copy_from_host(0, &f32s_to_le_bytes(weight_values), Some(&mut stream))
        .unwrap();
    gate.copy_from_host(0, &f32s_to_le_bytes(gate_values), Some(&mut stream))
        .unwrap();
    stream.synchronize().unwrap();

    segmented_rmsnorm_silu_mul_shuffle_prototype_launch(
        prototype,
        &input,
        &weight,
        &gate,
        epsilon,
        &mut output,
        &mut stream,
    );
    stream.synchronize().unwrap();
    let mut output_bytes = vec![0_u8; input_values.len() * f32_bytes];
    output
        .copy_to_host(0, &mut output_bytes, Some(&mut stream))
        .unwrap();
    stream.synchronize().unwrap();
    le_bytes_to_f32s(&output_bytes)
}

fn segmented_rmsnorm_silu_mul_shuffle_prototype_time(
    device: u32,
    prototype: bool,
    input_values: &[f32],
    weight_values: &[f32],
    gate_values: &[f32],
    epsilon: f32,
    rounds: usize,
) -> f64 {
    assert!(rounds > 0);
    let f32_bytes = std::mem::size_of::<f32>();
    let mut context = RuntimeContext::create(device).unwrap();
    let mut stream = context.create_stream().unwrap();
    let mut input = context
        .alloc_buffer(input_values.len() * f32_bytes)
        .unwrap();
    let mut weight = context
        .alloc_buffer(weight_values.len() * f32_bytes)
        .unwrap();
    let mut gate = context.alloc_buffer(gate_values.len() * f32_bytes).unwrap();
    let mut output = context
        .alloc_buffer(input_values.len() * f32_bytes)
        .unwrap();
    input
        .copy_from_host(0, &f32s_to_le_bytes(input_values), Some(&mut stream))
        .unwrap();
    weight
        .copy_from_host(0, &f32s_to_le_bytes(weight_values), Some(&mut stream))
        .unwrap();
    gate.copy_from_host(0, &f32s_to_le_bytes(gate_values), Some(&mut stream))
        .unwrap();
    stream.synchronize().unwrap();

    for _ in 0..3 {
        segmented_rmsnorm_silu_mul_shuffle_prototype_launch(
            prototype,
            &input,
            &weight,
            &gate,
            epsilon,
            &mut output,
            &mut stream,
        );
    }
    stream.synchronize().unwrap();
    let started = std::time::Instant::now();
    for _ in 0..rounds {
        segmented_rmsnorm_silu_mul_shuffle_prototype_launch(
            prototype,
            &input,
            &weight,
            &gate,
            epsilon,
            &mut output,
            &mut stream,
        );
    }
    stream.synchronize().unwrap();
    started.elapsed().as_secs_f64() * 1_000.0 / rounds as f64
}

fn assert_segmented_rmsnorm_silu_mul_shuffle_prototype_matches_cpu(
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
fn cpu_segmented_rmsnorm_silu_mul_shuffle_prototype_rejects_cpu_backend() {
    let f32_bytes = std::mem::size_of::<f32>();
    let mut context = RuntimeContext::create(0).unwrap();
    let input = context
        .alloc_buffer(SEGMENTED_RMSNORM_SILU_MUL_SHUFFLE_PROTOTYPE_ELEMENTS * f32_bytes)
        .unwrap();
    let weight = context
        .alloc_buffer(SEGMENTED_RMSNORM_SILU_MUL_SHUFFLE_PROTOTYPE_SEGMENT_SIZE * f32_bytes)
        .unwrap();
    let gate = context
        .alloc_buffer(SEGMENTED_RMSNORM_SILU_MUL_SHUFFLE_PROTOTYPE_ELEMENTS * f32_bytes)
        .unwrap();
    let mut output = context
        .alloc_buffer(SEGMENTED_RMSNORM_SILU_MUL_SHUFFLE_PROTOTYPE_ELEMENTS * f32_bytes)
        .unwrap();

    let error = segmented_rmsnorm_silu_mul_shuffle_prototype_f32(
        &input,
        &weight,
        &gate,
        SEGMENTED_RMSNORM_SILU_MUL_SHUFFLE_PROTOTYPE_SEGMENTS,
        SEGMENTED_RMSNORM_SILU_MUL_SHUFFLE_PROTOTYPE_SEGMENT_SIZE,
        1e-6_f32,
        &mut output,
        None,
    )
    .expect_err("the direct prototype must not stage or fall back on CPU");
    assert!(error.contains("requires a HIP gfx1201 backend"));
}

#[test]
#[ignore = "requires an isolated gfx1201 HIP device and ULLM_RUN_SEGMENTED_RMSNORM_SILU_MUL_SHUFFLE_PROTOTYPE_DIFFERENTIAL=1"]
fn hip_segmented_rmsnorm_silu_mul_shuffle_prototype_qwen35_m1_matches_cpu_when_enabled() {
    assert_eq!(
        std::env::var("ULLM_RUN_SEGMENTED_RMSNORM_SILU_MUL_SHUFFLE_PROTOTYPE_DIFFERENTIAL")
            .as_deref(),
        Ok("1"),
        "set ULLM_RUN_SEGMENTED_RMSNORM_SILU_MUL_SHUFFLE_PROTOTYPE_DIFFERENTIAL=1 before running this GPU differential test"
    );
    let _lock = AQ4_EXPERIMENTAL_ENV_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let (input, weight, gate) = segmented_rmsnorm_silu_mul_shuffle_prototype_fixture();
    let expected = segmented_rmsnorm_silu_mul_shuffle_prototype_run(
        0, false, &input, &weight, &gate, 1e-6_f32,
    );
    let actual = segmented_rmsnorm_silu_mul_shuffle_prototype_run(
        segmented_rmsnorm_silu_mul_shuffle_prototype_gpu_device(),
        true,
        &input,
        &weight,
        &gate,
        1e-6_f32,
    );
    assert_segmented_rmsnorm_silu_mul_shuffle_prototype_matches_cpu(
        &actual,
        &expected,
        "Qwen3.5 M=1 linear-attention post output",
    );
    let max_abs_diff = actual
        .iter()
        .zip(&expected)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0_f32, f32::max);
    eprintln!(
        "segmented RMSNorm SiLU-mul shuffle prototype differential passed M=1 segments=32 segment_size=128 epsilon=0.000001000 max_abs_diff={max_abs_diff:.9}"
    );
}

#[test]
#[ignore = "requires an isolated gfx1201 HIP device and ULLM_RUN_SEGMENTED_RMSNORM_SILU_MUL_SHUFFLE_PROTOTYPE_TIMING=1"]
fn hip_segmented_rmsnorm_silu_mul_shuffle_prototype_qwen35_m1_timing_vs_production_when_enabled() {
    assert_eq!(
        std::env::var("ULLM_RUN_SEGMENTED_RMSNORM_SILU_MUL_SHUFFLE_PROTOTYPE_TIMING").as_deref(),
        Ok("1"),
        "set ULLM_RUN_SEGMENTED_RMSNORM_SILU_MUL_SHUFFLE_PROTOTYPE_TIMING=1 before running this GPU timing test"
    );
    let _lock = AQ4_EXPERIMENTAL_ENV_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let _require_hip = ExperimentalEnvGuard::new(
        "ULLM_REQUIRE_HIP_SEGMENTED_RMSNORM_SILU_MUL_KERNEL",
        Some("1"),
    );
    let gpu = segmented_rmsnorm_silu_mul_shuffle_prototype_gpu_device();
    let (input, weight, gate) = segmented_rmsnorm_silu_mul_shuffle_prototype_fixture();
    let production = segmented_rmsnorm_silu_mul_shuffle_prototype_time(
        gpu, false, &input, &weight, &gate, 1e-6_f32, 100,
    );
    let shuffle = segmented_rmsnorm_silu_mul_shuffle_prototype_time(
        gpu, true, &input, &weight, &gate, 1e-6_f32, 100,
    );
    assert!(production > 0.0 && shuffle > 0.0);
    eprintln!(
        "segmented RMSNorm SiLU-mul shuffle prototype M=1 segments=32 segment_size=128: production={production:.6} ms, shuffle={shuffle:.6} ms, speedup={:.3}x",
        production / shuffle
    );
}
