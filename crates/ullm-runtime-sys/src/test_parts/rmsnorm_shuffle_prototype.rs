const RMSNORM_SHUFFLE_PROTOTYPE_ELEMENTS: usize = 4_096;

fn rmsnorm_shuffle_prototype_gpu_device() -> u32 {
    (1..device_count().unwrap())
        .find(|&candidate| {
            device_info(candidate)
                .map(|info| info.gcn_arch_name == "gfx1201")
                .unwrap_or(false)
        })
        .expect("isolated gfx1201 HIP device")
}

fn rmsnorm_shuffle_prototype_fixture() -> (Vec<f32>, Vec<f32>) {
    let mut state = 0x6a09_e667_u32;
    let mut next = || {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        state as f32 / u32::MAX as f32
    };
    let input = (0..RMSNORM_SHUFFLE_PROTOTYPE_ELEMENTS)
        .map(|_| next() * 2.0 - 1.0)
        .collect();
    let weight = (0..RMSNORM_SHUFFLE_PROTOTYPE_ELEMENTS)
        .map(|_| 0.5 + next())
        .collect();
    (input, weight)
}

fn rmsnorm_shuffle_prototype_launch(
    prototype: bool,
    input: &RuntimeBuffer,
    weight: &RuntimeBuffer,
    epsilon: f32,
    output: &mut RuntimeBuffer,
    stream: &mut RuntimeStream,
) {
    if prototype {
        rmsnorm_shuffle_prototype_f32(
            input,
            weight,
            RMSNORM_SHUFFLE_PROTOTYPE_ELEMENTS,
            epsilon,
            output,
            Some(&mut *stream),
        )
        .unwrap();
    } else {
        rmsnorm_f32(
            input,
            weight,
            RMSNORM_SHUFFLE_PROTOTYPE_ELEMENTS,
            epsilon,
            output,
            Some(&mut *stream),
        )
        .unwrap();
    }
}

fn rmsnorm_shuffle_prototype_run(
    device: u32,
    prototype: bool,
    input_values: &[f32],
    weight_values: &[f32],
    epsilon: f32,
) -> Vec<f32> {
    assert_eq!(input_values.len(), RMSNORM_SHUFFLE_PROTOTYPE_ELEMENTS);
    assert_eq!(weight_values.len(), RMSNORM_SHUFFLE_PROTOTYPE_ELEMENTS);
    let f32_bytes = std::mem::size_of::<f32>();
    let mut context = RuntimeContext::create(device).unwrap();
    let mut stream = context.create_stream().unwrap();
    let mut input = context
        .alloc_buffer(input_values.len() * f32_bytes)
        .unwrap();
    let mut weight = context
        .alloc_buffer(weight_values.len() * f32_bytes)
        .unwrap();
    let mut output = context
        .alloc_buffer(input_values.len() * f32_bytes)
        .unwrap();
    input
        .copy_from_host(0, &f32s_to_le_bytes(input_values), Some(&mut stream))
        .unwrap();
    weight
        .copy_from_host(0, &f32s_to_le_bytes(weight_values), Some(&mut stream))
        .unwrap();
    stream.synchronize().unwrap();

    rmsnorm_shuffle_prototype_launch(
        prototype,
        &input,
        &weight,
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

fn rmsnorm_shuffle_prototype_time(
    device: u32,
    prototype: bool,
    input_values: &[f32],
    weight_values: &[f32],
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
    let mut output = context
        .alloc_buffer(input_values.len() * f32_bytes)
        .unwrap();
    input
        .copy_from_host(0, &f32s_to_le_bytes(input_values), Some(&mut stream))
        .unwrap();
    weight
        .copy_from_host(0, &f32s_to_le_bytes(weight_values), Some(&mut stream))
        .unwrap();
    stream.synchronize().unwrap();

    for _ in 0..3 {
        rmsnorm_shuffle_prototype_launch(
            prototype,
            &input,
            &weight,
            epsilon,
            &mut output,
            &mut stream,
        );
    }
    stream.synchronize().unwrap();
    let started = std::time::Instant::now();
    for _ in 0..rounds {
        rmsnorm_shuffle_prototype_launch(
            prototype,
            &input,
            &weight,
            epsilon,
            &mut output,
            &mut stream,
        );
    }
    stream.synchronize().unwrap();
    started.elapsed().as_secs_f64() * 1_000.0 / rounds as f64
}

fn assert_rmsnorm_shuffle_prototype_matches_cpu(actual: &[f32], expected: &[f32], label: &str) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{label}: output lengths differ"
    );
    // A shuffle changes the floating-point reduction order, so use the established AQ4
    // differential convention: 0.05 absolute plus 1% of the CPU reference magnitude.
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        let tolerance = 5e-2_f32 + 1e-2_f32 * expected.abs();
        assert!(
            (actual - expected).abs() <= tolerance,
            "{label} index {index}: actual={actual} expected={expected} tolerance={tolerance}"
        );
    }
}

#[test]
fn cpu_rmsnorm_shuffle_prototype_rejects_cpu_backend() {
    let f32_bytes = std::mem::size_of::<f32>();
    let mut context = RuntimeContext::create(0).unwrap();
    let input = context
        .alloc_buffer(RMSNORM_SHUFFLE_PROTOTYPE_ELEMENTS * f32_bytes)
        .unwrap();
    let weight = context
        .alloc_buffer(RMSNORM_SHUFFLE_PROTOTYPE_ELEMENTS * f32_bytes)
        .unwrap();
    let mut output = context
        .alloc_buffer(RMSNORM_SHUFFLE_PROTOTYPE_ELEMENTS * f32_bytes)
        .unwrap();

    let error = rmsnorm_shuffle_prototype_f32(
        &input,
        &weight,
        RMSNORM_SHUFFLE_PROTOTYPE_ELEMENTS,
        1e-6_f32,
        &mut output,
        None,
    )
    .expect_err("the direct prototype must not stage or fall back on CPU");
    assert!(error.contains("requires a HIP gfx1201 backend"));
}

#[test]
#[ignore = "requires an isolated gfx1201 HIP device and ULLM_RUN_RMSNORM_SHUFFLE_PROTOTYPE_DIFFERENTIAL=1"]
fn hip_rmsnorm_shuffle_prototype_m1_qwen35_shape_matches_cpu_when_enabled() {
    assert_eq!(
        std::env::var("ULLM_RUN_RMSNORM_SHUFFLE_PROTOTYPE_DIFFERENTIAL").as_deref(),
        Ok("1"),
        "set ULLM_RUN_RMSNORM_SHUFFLE_PROTOTYPE_DIFFERENTIAL=1 before running this GPU differential test"
    );
    let _lock = AQ4_EXPERIMENTAL_ENV_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let gpu = rmsnorm_shuffle_prototype_gpu_device();
    let (input, weight) = rmsnorm_shuffle_prototype_fixture();

    // Both Qwen3.5 self-attention and linear-attention decode paths call this M=1 shape;
    // input and post norms differ only in epsilon.
    for &(label, epsilon) in &[("input norm", 1e-6_f32), ("post norm", 1e-5_f32)] {
        let expected = rmsnorm_shuffle_prototype_run(0, false, &input, &weight, epsilon);
        let actual = rmsnorm_shuffle_prototype_run(gpu, true, &input, &weight, epsilon);
        assert_rmsnorm_shuffle_prototype_matches_cpu(&actual, &expected, label);
        let max_abs_diff = actual
            .iter()
            .zip(&expected)
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0_f32, f32::max);
        eprintln!(
            "RMSNorm shuffle prototype differential passed case={label} M=1 hidden=4096 epsilon={epsilon:.9} max_abs_diff={max_abs_diff:.9}"
        );
    }
}

#[test]
#[ignore = "requires an isolated gfx1201 HIP device and ULLM_RUN_RMSNORM_PRODUCTION_DIFFERENTIAL=1"]
fn hip_rmsnorm_production_m1_qwen35_shape_matches_cpu_when_enabled() {
    assert_eq!(
        std::env::var("ULLM_RUN_RMSNORM_PRODUCTION_DIFFERENTIAL").as_deref(),
        Ok("1"),
        "set ULLM_RUN_RMSNORM_PRODUCTION_DIFFERENTIAL=1 before running this GPU differential test"
    );
    let _lock = AQ4_EXPERIMENTAL_ENV_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let _require_production_kernel =
        ExperimentalEnvGuard::new("ULLM_REQUIRE_HIP_RMSNORM_KERNEL", Some("1"));
    let gpu = rmsnorm_shuffle_prototype_gpu_device();
    let (input, weight) = rmsnorm_shuffle_prototype_fixture();

    // Both Qwen3.5 self-attention and linear-attention decode paths call this M=1 shape;
    // input and post norms differ only in epsilon.
    for &(label, epsilon) in &[("input norm", 1e-6_f32), ("post norm", 1e-5_f32)] {
        let expected = rmsnorm_shuffle_prototype_run(0, false, &input, &weight, epsilon);
        let actual = rmsnorm_shuffle_prototype_run(gpu, false, &input, &weight, epsilon);
        assert_rmsnorm_shuffle_prototype_matches_cpu(&actual, &expected, label);
        let max_abs_diff = actual
            .iter()
            .zip(&expected)
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0_f32, f32::max);
        eprintln!(
            "RMSNorm production differential passed case={label} M=1 hidden=4096 epsilon={epsilon:.9} max_abs_diff={max_abs_diff:.9}"
        );
    }
}

#[test]
#[ignore = "requires an isolated gfx1201 HIP device and ULLM_RUN_RMSNORM_SHUFFLE_PROTOTYPE_TIMING=1"]
fn hip_rmsnorm_shuffle_prototype_m1_qwen35_shape_timing_vs_production_when_enabled() {
    assert_eq!(
        std::env::var("ULLM_RUN_RMSNORM_SHUFFLE_PROTOTYPE_TIMING").as_deref(),
        Ok("1"),
        "set ULLM_RUN_RMSNORM_SHUFFLE_PROTOTYPE_TIMING=1 before running this GPU timing test"
    );
    let _lock = AQ4_EXPERIMENTAL_ENV_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let _require_hip = ExperimentalEnvGuard::new("ULLM_REQUIRE_HIP_RMSNORM_KERNEL", Some("1"));
    let gpu = rmsnorm_shuffle_prototype_gpu_device();
    let (input, weight) = rmsnorm_shuffle_prototype_fixture();
    let production = rmsnorm_shuffle_prototype_time(gpu, false, &input, &weight, 1e-6_f32, 100);
    let shuffle = rmsnorm_shuffle_prototype_time(gpu, true, &input, &weight, 1e-6_f32, 100);
    assert!(production > 0.0 && shuffle > 0.0);
    eprintln!(
        "RMSNorm shuffle prototype M=1 hidden=4096: production={production:.6} ms, shuffle={shuffle:.6} ms, speedup={:.3}x",
        production / shuffle
    );
}
