#[allow(clippy::too_many_arguments)]
fn aq4_matvec_triple_shuffle_run(
    device: u32,
    shuffle: bool,
    first: &Aq4WideLoadMatrixHost,
    second: &Aq4WideLoadMatrixHost,
    third: &Aq4WideLoadMatrixHost,
    input: &[f32],
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut context = RuntimeContext::create(device).unwrap();
    let mut stream = context.create_stream().unwrap();
    let first_device = aq4_fused_wide_load_upload(&mut context, &mut stream, first);
    let second_device = aq4_fused_wide_load_upload(&mut context, &mut stream, second);
    let third_device = aq4_fused_wide_load_upload(&mut context, &mut stream, third);
    let mut input_device = context
        .alloc_buffer(input.len() * std::mem::size_of::<f32>())
        .unwrap();
    input_device
        .copy_from_host(0, &f32s_to_le_bytes(input), Some(&mut stream))
        .unwrap();
    let mut first_output = context
        .alloc_buffer(first.rows * std::mem::size_of::<f32>())
        .unwrap();
    let mut second_output = context
        .alloc_buffer(second.rows * std::mem::size_of::<f32>())
        .unwrap();
    let mut third_output = context
        .alloc_buffer(third.rows * std::mem::size_of::<f32>())
        .unwrap();
    let first_row_scale_count = first.row_scales.as_ref().map_or(0, Vec::len);
    let second_row_scale_count = second.row_scales.as_ref().map_or(0, Vec::len);
    let third_row_scale_count = third.row_scales.as_ref().map_or(0, Vec::len);
    if shuffle {
        aq4_matvec_triple_shuffle_prototype_f32(
            &first_device.index,
            &first_device.scale,
            &first_device.codebook,
            &first_device.scale_values,
            first_device.row_scale.as_ref(),
            first.scale_count,
            first.group_size,
            first.tensor_scale,
            first_row_scale_count,
            &second_device.index,
            &second_device.scale,
            &second_device.codebook,
            &second_device.scale_values,
            second_device.row_scale.as_ref(),
            second.scale_count,
            second.group_size,
            second.tensor_scale,
            second_row_scale_count,
            &third_device.index,
            &third_device.scale,
            &third_device.codebook,
            &third_device.scale_values,
            third_device.row_scale.as_ref(),
            third.scale_count,
            third.group_size,
            third.tensor_scale,
            third_row_scale_count,
            &input_device,
            first.rows,
            second.rows,
            third.rows,
            first.cols,
            &mut first_output,
            &mut second_output,
            &mut third_output,
            Some(&mut stream),
        )
        .unwrap();
    } else {
        aq4_matvec_triple_f32(
            &first_device.index,
            &first_device.scale,
            &first_device.codebook,
            &first_device.scale_values,
            first_device.row_scale.as_ref(),
            first.scale_count,
            first.group_size,
            first.tensor_scale,
            first_row_scale_count,
            &second_device.index,
            &second_device.scale,
            &second_device.codebook,
            &second_device.scale_values,
            second_device.row_scale.as_ref(),
            second.scale_count,
            second.group_size,
            second.tensor_scale,
            second_row_scale_count,
            &third_device.index,
            &third_device.scale,
            &third_device.codebook,
            &third_device.scale_values,
            third_device.row_scale.as_ref(),
            third.scale_count,
            third.group_size,
            third.tensor_scale,
            third_row_scale_count,
            &input_device,
            first.rows,
            second.rows,
            third.rows,
            first.cols,
            &mut first_output,
            &mut second_output,
            &mut third_output,
            Some(&mut stream),
        )
        .unwrap();
    }
    stream.synchronize().unwrap();
    (
        aq4_fused_wide_load_read(&first_output, first.rows, &mut stream),
        aq4_fused_wide_load_read(&second_output, second.rows, &mut stream),
        aq4_fused_wide_load_read(&third_output, third.rows, &mut stream),
    )
}

#[allow(clippy::too_many_arguments)]
fn aq4_matvec_triple_shuffle_time(
    device: u32,
    shuffle: bool,
    first: &Aq4WideLoadMatrixHost,
    second: &Aq4WideLoadMatrixHost,
    third: &Aq4WideLoadMatrixHost,
    input: &[f32],
    rounds: usize,
) -> f64 {
    let mut context = RuntimeContext::create(device).unwrap();
    let mut stream = context.create_stream().unwrap();
    let first_device = aq4_fused_wide_load_upload(&mut context, &mut stream, first);
    let second_device = aq4_fused_wide_load_upload(&mut context, &mut stream, second);
    let third_device = aq4_fused_wide_load_upload(&mut context, &mut stream, third);
    let mut input_device = context
        .alloc_buffer(input.len() * std::mem::size_of::<f32>())
        .unwrap();
    input_device
        .copy_from_host(0, &f32s_to_le_bytes(input), Some(&mut stream))
        .unwrap();
    let mut first_output = context
        .alloc_buffer(first.rows * std::mem::size_of::<f32>())
        .unwrap();
    let mut second_output = context
        .alloc_buffer(second.rows * std::mem::size_of::<f32>())
        .unwrap();
    let mut third_output = context
        .alloc_buffer(third.rows * std::mem::size_of::<f32>())
        .unwrap();
    let first_row_scale_count = first.row_scales.as_ref().map_or(0, Vec::len);
    let second_row_scale_count = second.row_scales.as_ref().map_or(0, Vec::len);
    let third_row_scale_count = third.row_scales.as_ref().map_or(0, Vec::len);
    let mut launch = |mut stream: &mut RuntimeStream| {
        if shuffle {
            aq4_matvec_triple_shuffle_prototype_f32(
                &first_device.index,
                &first_device.scale,
                &first_device.codebook,
                &first_device.scale_values,
                first_device.row_scale.as_ref(),
                first.scale_count,
                first.group_size,
                first.tensor_scale,
                first_row_scale_count,
                &second_device.index,
                &second_device.scale,
                &second_device.codebook,
                &second_device.scale_values,
                second_device.row_scale.as_ref(),
                second.scale_count,
                second.group_size,
                second.tensor_scale,
                second_row_scale_count,
                &third_device.index,
                &third_device.scale,
                &third_device.codebook,
                &third_device.scale_values,
                third_device.row_scale.as_ref(),
                third.scale_count,
                third.group_size,
                third.tensor_scale,
                third_row_scale_count,
                &input_device,
                first.rows,
                second.rows,
                third.rows,
                first.cols,
                &mut first_output,
                &mut second_output,
                &mut third_output,
                Some(&mut stream),
            )
            .unwrap();
        } else {
            aq4_matvec_triple_f32(
                &first_device.index,
                &first_device.scale,
                &first_device.codebook,
                &first_device.scale_values,
                first_device.row_scale.as_ref(),
                first.scale_count,
                first.group_size,
                first.tensor_scale,
                first_row_scale_count,
                &second_device.index,
                &second_device.scale,
                &second_device.codebook,
                &second_device.scale_values,
                second_device.row_scale.as_ref(),
                second.scale_count,
                second.group_size,
                second.tensor_scale,
                second_row_scale_count,
                &third_device.index,
                &third_device.scale,
                &third_device.codebook,
                &third_device.scale_values,
                third_device.row_scale.as_ref(),
                third.scale_count,
                third.group_size,
                third.tensor_scale,
                third_row_scale_count,
                &input_device,
                first.rows,
                second.rows,
                third.rows,
                first.cols,
                &mut first_output,
                &mut second_output,
                &mut third_output,
                Some(&mut stream),
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
#[ignore = "requires an isolated gfx1201 HIP device and ULLM_RUN_AQ4_MATVEC_TRIPLE_SHUFFLE_DIFFERENTIAL=1"]
fn hip_aq4_matvec_triple_shuffle_prototype_matches_cpu_for_production_and_single_stream_shapes() {
    assert_eq!(
        std::env::var("ULLM_RUN_AQ4_MATVEC_TRIPLE_SHUFFLE_DIFFERENTIAL").as_deref(),
        Ok("1")
    );
    let _lock = AQ4_EXPERIMENTAL_ENV_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let _rpb = ExperimentalEnvGuard::new("ULLM_AQ4_MATVEC_TRIPLE_RPB", Some("8"));
    let gpu = aq4_fused_wide_load_gpu_device();
    let mut state = 0x9e37_79b9;
    // Qwen3.5-9B self-attention Q/K/V: Q=[8192,4096], K/V=[1024,4096]. The
    // first 1024 rows take the three-stream helper; Q's tail takes the single-stream path.
    let first = aq4_fused_wide_load_host_matrix(8_192, 4_096, 16, true, &mut state);
    let second = aq4_fused_wide_load_host_matrix(1_024, 4_096, 16, false, &mut state);
    let third = aq4_fused_wide_load_host_matrix(1_024, 4_096, 16, true, &mut state);
    let input: Vec<f32> = (0..4_096)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            state as f32 / u32::MAX as f32 * 2.0 - 1.0
        })
        .collect();
    let baseline_expected =
        aq4_matvec_triple_shuffle_run(0, false, &first, &second, &third, &input);
    let baseline_actual = aq4_matvec_triple_shuffle_run(gpu, true, &first, &second, &third, &input);
    for (label, actual, expected) in [
        ("shuffle Q", &baseline_actual.0, &baseline_expected.0),
        ("shuffle K", &baseline_actual.1, &baseline_expected.1),
        ("shuffle V", &baseline_actual.2, &baseline_expected.2),
    ] {
        aq4_fused_wide_load_assert_matches_cpu(actual, expected, label);
    }
    for stream_index in 0..3 {
        let (mut modified_first, mut modified_second, mut modified_third) =
            (first.clone(), second.clone(), third.clone());
        match stream_index {
            0 => modified_first.indices[0] ^= 0x11,
            1 => modified_second.indices[0] ^= 0x11,
            _ => modified_third.indices[0] ^= 0x11,
        }
        let expected = aq4_matvec_triple_shuffle_run(
            0,
            false,
            &modified_first,
            &modified_second,
            &modified_third,
            &input,
        );
        let actual = aq4_matvec_triple_shuffle_run(
            gpu,
            true,
            &modified_first,
            &modified_second,
            &modified_third,
            &input,
        );
        let expected_streams = [&expected.0, &expected.1, &expected.2];
        let actual_streams = [&actual.0, &actual.1, &actual.2];
        let baseline_expected_streams = [
            &baseline_expected.0,
            &baseline_expected.1,
            &baseline_expected.2,
        ];
        let baseline_actual_streams = [&baseline_actual.0, &baseline_actual.1, &baseline_actual.2];
        for (label, actual, expected) in [
            ("mutated shuffle Q", actual_streams[0], expected_streams[0]),
            ("mutated shuffle K", actual_streams[1], expected_streams[1]),
            ("mutated shuffle V", actual_streams[2], expected_streams[2]),
        ] {
            aq4_fused_wide_load_assert_matches_cpu(actual, expected, label);
        }
        aq4_fused_wide_load_assert_changed(
            expected_streams[stream_index],
            baseline_expected_streams[stream_index],
            "triple CPU stream",
        );
        aq4_fused_wide_load_assert_changed(
            actual_streams[stream_index],
            baseline_actual_streams[stream_index],
            "triple GPU stream",
        );
        for other in 0..3 {
            if other != stream_index {
                assert_f32s_close(
                    expected_streams[other],
                    baseline_expected_streams[other],
                    0.0,
                );
                assert_f32s_close(actual_streams[other], baseline_actual_streams[other], 0.0);
            }
        }
    }

    // Different group sizes force all three rows through the individual thread_sum helper.
    let single_first = aq4_fused_wide_load_host_matrix(128, 4_096, 16, true, &mut state);
    let single_second = aq4_fused_wide_load_host_matrix(128, 4_096, 8, false, &mut state);
    let single_third = aq4_fused_wide_load_host_matrix(128, 4_096, 16, true, &mut state);
    let expected = aq4_matvec_triple_shuffle_run(
        0,
        false,
        &single_first,
        &single_second,
        &single_third,
        &input,
    );
    let actual = aq4_matvec_triple_shuffle_run(
        gpu,
        true,
        &single_first,
        &single_second,
        &single_third,
        &input,
    );
    for (label, actual, expected) in [
        ("single-stream shuffle first", &actual.0, &expected.0),
        ("single-stream shuffle second", &actual.1, &expected.1),
        ("single-stream shuffle third", &actual.2, &expected.2),
    ] {
        aq4_fused_wide_load_assert_matches_cpu(actual, expected, label);
    }
}

#[test]
#[ignore = "requires an isolated gfx1201 HIP device and ULLM_RUN_AQ4_MATVEC_TRIPLE_PRODUCTION_DIFFERENTIAL=1"]
fn hip_aq4_matvec_triple_production_matches_cpu_for_production_shape() {
    assert_eq!(
        std::env::var("ULLM_RUN_AQ4_MATVEC_TRIPLE_PRODUCTION_DIFFERENTIAL").as_deref(),
        Ok("1")
    );
    let _lock = AQ4_EXPERIMENTAL_ENV_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let _rpb = ExperimentalEnvGuard::new("ULLM_AQ4_MATVEC_TRIPLE_RPB", Some("8"));
    let _require_production_kernel =
        ExperimentalEnvGuard::new("ULLM_REQUIRE_HIP_AQ4_MATVEC_TRIPLE_KERNEL", Some("1"));
    let gpu = aq4_fused_wide_load_gpu_device();
    let mut state = 0x510e_527f;
    // Qwen3.5-9B self-attention Q/K/V: Q=[8192,4096], K/V=[1024,4096].
    let first = aq4_fused_wide_load_host_matrix(8_192, 4_096, 16, true, &mut state);
    let second = aq4_fused_wide_load_host_matrix(1_024, 4_096, 16, false, &mut state);
    let third = aq4_fused_wide_load_host_matrix(1_024, 4_096, 16, true, &mut state);
    let input: Vec<f32> = (0..4_096)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            state as f32 / u32::MAX as f32 * 2.0 - 1.0
        })
        .collect();
    let expected = aq4_matvec_triple_shuffle_run(0, false, &first, &second, &third, &input);
    let actual = aq4_matvec_triple_shuffle_run(gpu, false, &first, &second, &third, &input);
    for (label, actual, expected) in [
        ("production Q", &actual.0, &expected.0),
        ("production K", &actual.1, &expected.1),
        ("production V", &actual.2, &expected.2),
    ] {
        aq4_fused_wide_load_assert_matches_cpu(actual, expected, label);
    }
}

#[test]
#[ignore = "requires an isolated gfx1201 HIP device and ULLM_RUN_AQ4_MATVEC_TRIPLE_SHUFFLE_TIMING=1"]
fn hip_aq4_matvec_triple_shuffle_prototype_timing_vs_production() {
    assert_eq!(
        std::env::var("ULLM_RUN_AQ4_MATVEC_TRIPLE_SHUFFLE_TIMING").as_deref(),
        Ok("1")
    );
    let _lock = AQ4_EXPERIMENTAL_ENV_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let _rpb = ExperimentalEnvGuard::new("ULLM_AQ4_MATVEC_TRIPLE_RPB", Some("8"));
    let gpu = aq4_fused_wide_load_gpu_device();
    let mut state = 0x243f_6a88;
    let first = aq4_fused_wide_load_host_matrix(8_192, 4_096, 16, true, &mut state);
    let second = aq4_fused_wide_load_host_matrix(1_024, 4_096, 16, false, &mut state);
    let third = aq4_fused_wide_load_host_matrix(1_024, 4_096, 16, true, &mut state);
    let input = vec![0.125; 4_096];
    let production =
        aq4_matvec_triple_shuffle_time(gpu, false, &first, &second, &third, &input, 20);
    let shuffle = aq4_matvec_triple_shuffle_time(gpu, true, &first, &second, &third, &input, 20);
    assert!(production > 0.0 && shuffle > 0.0);
    eprintln!(
        "AQ4 triple shuffle [8192/1024/1024,4096]: production={production:.3} ms, shuffle={shuffle:.3} ms, speedup={:.3}x",
        production / shuffle
    );
}
