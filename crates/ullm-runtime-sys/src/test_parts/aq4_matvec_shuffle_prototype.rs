#[allow(clippy::too_many_arguments)]
fn aq4_matvec_shuffle_prototype_run(
    device: u32,
    shuffle: bool,
    matrix: &Aq4WideLoadMatrixHost,
    input: &[f32],
) -> Vec<f32> {
    let mut context = RuntimeContext::create(device).unwrap();
    let mut stream = context.create_stream().unwrap();
    let matrix_device = aq4_fused_wide_load_upload(&mut context, &mut stream, matrix);
    let mut input_device = context
        .alloc_buffer(input.len() * std::mem::size_of::<f32>())
        .unwrap();
    input_device
        .copy_from_host(0, &f32s_to_le_bytes(input), Some(&mut stream))
        .unwrap();
    let mut output = context
        .alloc_buffer(matrix.rows * std::mem::size_of::<f32>())
        .unwrap();
    let row_scale_count = matrix.row_scales.as_ref().map_or(0, Vec::len);
    if shuffle {
        aq4_matvec_shuffle_prototype_f32(
            &matrix_device.index,
            &matrix_device.scale,
            &matrix_device.codebook,
            &matrix_device.scale_values,
            &input_device,
            matrix_device.row_scale.as_ref(),
            matrix.scale_count,
            matrix.group_size,
            matrix.tensor_scale,
            row_scale_count,
            matrix.rows,
            matrix.cols,
            &mut output,
            Some(&mut stream),
        )
        .unwrap();
    } else {
        aq4_matvec_f32(
            &matrix_device.index,
            &matrix_device.scale,
            &matrix_device.codebook,
            &matrix_device.scale_values,
            &input_device,
            matrix_device.row_scale.as_ref(),
            matrix.scale_count,
            matrix.group_size,
            matrix.tensor_scale,
            row_scale_count,
            matrix.rows,
            matrix.cols,
            &mut output,
            Some(&mut stream),
        )
        .unwrap();
    }
    stream.synchronize().unwrap();
    aq4_fused_wide_load_read(&output, matrix.rows, &mut stream)
}

fn aq4_matvec_shuffle_prototype_time(
    device: u32,
    shuffle: bool,
    matrix: &Aq4WideLoadMatrixHost,
    input: &[f32],
    rounds: usize,
) -> f64 {
    let mut context = RuntimeContext::create(device).unwrap();
    let mut stream = context.create_stream().unwrap();
    let matrix_device = aq4_fused_wide_load_upload(&mut context, &mut stream, matrix);
    let mut input_device = context
        .alloc_buffer(input.len() * std::mem::size_of::<f32>())
        .unwrap();
    input_device
        .copy_from_host(0, &f32s_to_le_bytes(input), Some(&mut stream))
        .unwrap();
    let mut output = context
        .alloc_buffer(matrix.rows * std::mem::size_of::<f32>())
        .unwrap();
    let row_scale_count = matrix.row_scales.as_ref().map_or(0, Vec::len);
    let mut launch = |stream: &mut RuntimeStream| {
        if shuffle {
            aq4_matvec_shuffle_prototype_f32(
                &matrix_device.index,
                &matrix_device.scale,
                &matrix_device.codebook,
                &matrix_device.scale_values,
                &input_device,
                matrix_device.row_scale.as_ref(),
                matrix.scale_count,
                matrix.group_size,
                matrix.tensor_scale,
                row_scale_count,
                matrix.rows,
                matrix.cols,
                &mut output,
                Some(stream),
            )
            .unwrap();
        } else {
            aq4_matvec_f32(
                &matrix_device.index,
                &matrix_device.scale,
                &matrix_device.codebook,
                &matrix_device.scale_values,
                &input_device,
                matrix_device.row_scale.as_ref(),
                matrix.scale_count,
                matrix.group_size,
                matrix.tensor_scale,
                row_scale_count,
                matrix.rows,
                matrix.cols,
                &mut output,
                Some(stream),
            )
            .unwrap();
        }
    };
    for _ in 0..3 {
        launch(&mut stream);
    }
    stream.synchronize().unwrap();
    let started = std::time::Instant::now();
    for _ in 0..rounds {
        launch(&mut stream);
    }
    stream.synchronize().unwrap();
    started.elapsed().as_secs_f64() * 1_000.0 / rounds as f64
}

#[test]
#[ignore = "requires an isolated gfx1201 HIP device and ULLM_RUN_AQ4_MATVEC_PRODUCTION_DIFFERENTIAL=1"]
fn hip_aq4_matvec_production_matches_cpu_for_qwen35_lm_head_when_enabled() {
    assert_eq!(
        std::env::var("ULLM_RUN_AQ4_MATVEC_PRODUCTION_DIFFERENTIAL").as_deref(),
        Ok("1"),
        "set ULLM_RUN_AQ4_MATVEC_PRODUCTION_DIFFERENTIAL=1 before running this GPU differential test"
    );
    let _lock = AQ4_EXPERIMENTAL_ENV_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    // `aq4_matvec_f32` below is the public Rust wrapper over the production C ABI. Requiring
    // the existing HIP kernel prevents a staging fallback from making this test pass.
    let _require_production_kernel =
        ExperimentalEnvGuard::new("ULLM_REQUIRE_HIP_AQ4_MATVEC_KERNEL", Some("1"));
    let _rpb = ExperimentalEnvGuard::new("ULLM_AQ4_MATVEC_RPB", Some("32"));
    let gpu = aq4_fused_wide_load_gpu_device();
    // The resident Qwen3.5 AQ4 lm_head validates matrix.rows == vocab and matrix.cols == hidden;
    // the product profile fixes those dimensions to vocab=248320 and hidden=4096.
    let mut state = 0x243f_6a88;
    let lm_head = aq4_fused_wide_load_host_matrix(248_320, 4_096, 16, true, &mut state);
    let input: Vec<f32> = (0..lm_head.cols)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            state as f32 / u32::MAX as f32 * 2.0 - 1.0
        })
        .collect();
    let expected = aq4_matvec_shuffle_prototype_run(0, false, &lm_head, &input);
    let actual = aq4_matvec_shuffle_prototype_run(gpu, false, &lm_head, &input);
    // Keep the established AQ4 differential allowance: 0.05 + 0.01 * |expected|.
    aq4_fused_wide_load_assert_matches_cpu(&actual, &expected, "Qwen3.5 AQ4 lm_head production");
}

#[test]
#[ignore = "requires an isolated gfx1201 HIP device and ULLM_RUN_AQ4_MATVEC_SHUFFLE_DIFFERENTIAL=1"]
fn hip_aq4_matvec_shuffle_prototype_matches_cpu_for_qwen35_lm_head_when_enabled() {
    assert_eq!(
        std::env::var("ULLM_RUN_AQ4_MATVEC_SHUFFLE_DIFFERENTIAL").as_deref(),
        Ok("1"),
        "set ULLM_RUN_AQ4_MATVEC_SHUFFLE_DIFFERENTIAL=1 before running this GPU differential test"
    );
    let _lock = AQ4_EXPERIMENTAL_ENV_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let _rpb = ExperimentalEnvGuard::new("ULLM_AQ4_MATVEC_RPB", Some("32"));
    let gpu = aq4_fused_wide_load_gpu_device();
    // The resident Qwen3.5 AQ4 lm_head validates matrix.rows == vocab and matrix.cols == hidden;
    // the product profile fixes those dimensions to vocab=248320 and hidden=4096.
    let mut state = 0x243f_6a88;
    let lm_head = aq4_fused_wide_load_host_matrix(248_320, 4_096, 16, true, &mut state);
    let input: Vec<f32> = (0..lm_head.cols)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            state as f32 / u32::MAX as f32 * 2.0 - 1.0
        })
        .collect();
    let expected = aq4_matvec_shuffle_prototype_run(0, false, &lm_head, &input);
    let actual = aq4_matvec_shuffle_prototype_run(gpu, true, &lm_head, &input);
    // Keep the established AQ4 differential allowance: 0.05 + 0.01 * |expected|.
    aq4_fused_wide_load_assert_matches_cpu(&actual, &expected, "Qwen3.5 AQ4 lm_head shuffle");
}

#[test]
#[ignore = "requires an isolated gfx1201 HIP device and ULLM_RUN_AQ4_MATVEC_SHUFFLE_TIMING=1"]
fn hip_aq4_matvec_shuffle_prototype_timing_vs_production_for_qwen35_lm_head_when_enabled() {
    assert_eq!(
        std::env::var("ULLM_RUN_AQ4_MATVEC_SHUFFLE_TIMING").as_deref(),
        Ok("1")
    );
    let _lock = AQ4_EXPERIMENTAL_ENV_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let _rpb = ExperimentalEnvGuard::new("ULLM_AQ4_MATVEC_RPB", Some("32"));
    let gpu = aq4_fused_wide_load_gpu_device();
    let mut state = 0x1319_8a2e;
    let lm_head = aq4_fused_wide_load_host_matrix(248_320, 4_096, 16, true, &mut state);
    let input = vec![0.125; lm_head.cols];
    let production = aq4_matvec_shuffle_prototype_time(gpu, false, &lm_head, &input, 20);
    let shuffle = aq4_matvec_shuffle_prototype_time(gpu, true, &lm_head, &input, 20);
    assert!(production.is_finite() && production > 0.0);
    assert!(shuffle.is_finite() && shuffle > 0.0);
    eprintln!(
        "AQ4 matvec shuffle lm_head [248320,4096] group16: production={production:.3} ms, shuffle={shuffle:.3} ms, speedup={:.3}x",
        production / shuffle
    );
}
