#[test]
#[ignore = "requires an isolated gfx1201 HIP device and ULLM_RUN_AQ4_WIDE_LOAD_DIFFERENTIAL=1"]
fn hip_aq4_wide_load_prototype_m1_model_shapes_match_cpu_when_enabled() {
    assert_eq!(
        std::env::var("ULLM_RUN_AQ4_WIDE_LOAD_DIFFERENTIAL").as_deref(),
        Ok("1"),
        "set ULLM_RUN_AQ4_WIDE_LOAD_DIFFERENTIAL=1 before running this GPU differential test"
    );
    let hip_device = (1..device_count().unwrap())
        .find(|&device| {
            device_info(device)
                .map(|info| info.gcn_arch_name == "gfx1201")
                .unwrap_or(false)
        })
        .expect("isolated gfx1201 HIP device");
    let _lock = AQ4_EXPERIMENTAL_ENV_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let mut seed = 0x6a09_e667_u32;
    let mut next = || {
        seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        seed as f32 / u32::MAX as f32
    };
    for &(family, rows, cols, group_size, use_row_scale) in &[
        ("attn_q + linear_attn_qkv", 8_192_usize, 4_096, 16, true),
        ("linear_attn_z", 4_096, 4_096, 16, false),
        ("linear_attn_a + linear_attn_b", 32, 4_096, 16, true),
        ("mlp_gate + mlp_up", 12_288, 4_096, 16, true),
        ("mlp_down", 4_096, 12_288, 16, false),
        ("attn_o + linear_attn_out", 4_096, 4_096, 8, true),
        ("attn_k + attn_v", 1_024, 4_096, 8, false),
    ] {
        let index_bytes = rows * cols / 2;
        let groups = rows * cols / group_size;
        let scale_count = 7_usize;
        let indices: Vec<u8> = (0..index_bytes)
            .map(|_| ((next() * 16.0) as u8 & 15) | (((next() * 16.0) as u8 & 15) << 4))
            .collect();
        let scale_indices: Vec<u8> = (0..groups)
            .map(|_| ((next() * scale_count as f32) as u8).min(scale_count as u8 - 1))
            .collect();
        let codebook: Vec<f32> = (0..16).map(|_| next() - 0.5).collect();
        let scale_values: Vec<f32> = (0..scale_count).map(|_| 0.5 + next() * 0.5).collect();
        let input: Vec<f32> = (0..cols).map(|_| next() * 2.0 - 1.0).collect();
        let row_scales: Vec<f32> = (0..rows).map(|_| 0.75 + next() * 0.5).collect();
        let row_scale_count = if use_row_scale { rows } else { 0 };
        let run = |device: u32, wide: bool| {
            let mut context = RuntimeContext::create(device).unwrap();
            let mut stream = context.create_stream().unwrap();
            let mut index = context.alloc_buffer(index_bytes).unwrap();
            let mut scale = context.alloc_buffer(groups).unwrap();
            let mut codebook_buffer = context.alloc_buffer(64).unwrap();
            let mut scale_values_buffer = context.alloc_buffer(scale_count * 4).unwrap();
            let mut input_buffer = context.alloc_buffer(cols * 4).unwrap();
            let mut row_scale_buffer =
                use_row_scale.then(|| context.alloc_buffer(rows * 4).unwrap());
            let mut output = context.alloc_buffer(rows * 4).unwrap();
            index
                .copy_from_host(0, &indices, Some(&mut stream))
                .unwrap();
            scale
                .copy_from_host(0, &scale_indices, Some(&mut stream))
                .unwrap();
            codebook_buffer
                .copy_from_host(0, &f32s_to_le_bytes(&codebook), Some(&mut stream))
                .unwrap();
            scale_values_buffer
                .copy_from_host(0, &f32s_to_le_bytes(&scale_values), Some(&mut stream))
                .unwrap();
            input_buffer
                .copy_from_host(0, &f32s_to_le_bytes(&input), Some(&mut stream))
                .unwrap();
            if let Some(buffer) = row_scale_buffer.as_mut() {
                buffer
                    .copy_from_host(0, &f32s_to_le_bytes(&row_scales), Some(&mut stream))
                    .unwrap();
            }
            if wide {
                aq4_matvec_wide_load_prototype_f32(
                    &index,
                    &scale,
                    &codebook_buffer,
                    &scale_values_buffer,
                    &input_buffer,
                    row_scale_buffer.as_ref(),
                    scale_count,
                    group_size,
                    0.75,
                    row_scale_count,
                    rows,
                    cols,
                    &mut output,
                    Some(&mut stream),
                )
                .unwrap();
            } else {
                aq4_matvec_f32(
                    &index,
                    &scale,
                    &codebook_buffer,
                    &scale_values_buffer,
                    &input_buffer,
                    row_scale_buffer.as_ref(),
                    scale_count,
                    group_size,
                    0.75,
                    row_scale_count,
                    rows,
                    cols,
                    &mut output,
                    Some(&mut stream),
                )
                .unwrap();
            }
            stream.synchronize().unwrap();
            let mut bytes = vec![0; rows * 4];
            output
                .copy_to_host(0, &mut bytes, Some(&mut stream))
                .unwrap();
            stream.synchronize().unwrap();
            le_bytes_to_f32s(&bytes)
        };
        let expected = run(0, false);
        let actual = run(hip_device, true);
        assert_f32s_close(&actual, &expected, 1e-3);
        eprintln!("AQ4 M=1 wide-load differential family={family} rows={rows} cols={cols} group{group_size}: ok");
    }
}

#[test]
#[ignore = "requires an isolated gfx1201 HIP device and ULLM_RUN_AQ4_WIDE_LOAD_TIMING=1"]
fn hip_aq4_wide_load_prototype_m1_model_shapes_timing_vs_production_when_enabled() {
    assert_eq!(
        std::env::var("ULLM_RUN_AQ4_WIDE_LOAD_TIMING").as_deref(),
        Ok("1")
    );
    let hip_device = (1..device_count().unwrap())
        .find(|&device| {
            device_info(device)
                .map(|info| info.gcn_arch_name == "gfx1201")
                .unwrap_or(false)
        })
        .expect("isolated gfx1201 HIP device");
    let _lock = AQ4_EXPERIMENTAL_ENV_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    for &(family, rows, cols, group_size) in &[
        ("attn_q + linear_attn_qkv", 8_192_usize, 4_096, 16),
        ("linear_attn_z", 4_096, 4_096, 16),
        ("linear_attn_a + linear_attn_b", 32, 4_096, 16),
        ("mlp_gate + mlp_up", 12_288, 4_096, 16),
        ("mlp_down", 4_096, 12_288, 16),
        ("attn_o + linear_attn_out", 4_096, 4_096, 8),
        ("attn_k + attn_v", 1_024, 4_096, 8),
    ] {
        let mut context = RuntimeContext::create(hip_device).unwrap();
        let mut stream = context.create_stream().unwrap();
        let mut index = context.alloc_buffer(rows * cols / 2).unwrap();
        let mut scale = context.alloc_buffer(rows * cols / group_size).unwrap();
        let mut codebook = context.alloc_buffer(64).unwrap();
        let mut scale_values = context.alloc_buffer(4).unwrap();
        let mut input = context.alloc_buffer(cols * 4).unwrap();
        let mut output = context.alloc_buffer(rows * 4).unwrap();
        index
            .copy_from_host(0, &vec![0x87; rows * cols / 2], Some(&mut stream))
            .unwrap();
        scale
            .copy_from_host(0, &vec![0; rows * cols / group_size], Some(&mut stream))
            .unwrap();
        codebook
            .copy_from_host(
                0,
                &f32s_to_le_bytes(&(0..16).map(|v| v as f32 / 16.0 - 0.5).collect::<Vec<_>>()),
                Some(&mut stream),
            )
            .unwrap();
        scale_values
            .copy_from_host(0, &f32s_to_le_bytes(&[0.75]), Some(&mut stream))
            .unwrap();
        input
            .copy_from_host(0, &f32s_to_le_bytes(&vec![0.125; cols]), Some(&mut stream))
            .unwrap();
        for _ in 0..3 {
            aq4_matvec_f32(
                &index,
                &scale,
                &codebook,
                &scale_values,
                &input,
                None,
                1,
                group_size,
                0.75,
                0,
                rows,
                cols,
                &mut output,
                Some(&mut stream),
            )
            .unwrap();
        }
        stream.synchronize().unwrap();
        for _ in 0..3 {
            aq4_matvec_wide_load_prototype_f32(
                &index,
                &scale,
                &codebook,
                &scale_values,
                &input,
                None,
                1,
                group_size,
                0.75,
                0,
                rows,
                cols,
                &mut output,
                Some(&mut stream),
            )
            .unwrap();
        }
        stream.synchronize().unwrap();
        let started = std::time::Instant::now();
        for _ in 0..20 {
            aq4_matvec_f32(
                &index,
                &scale,
                &codebook,
                &scale_values,
                &input,
                None,
                1,
                group_size,
                0.75,
                0,
                rows,
                cols,
                &mut output,
                Some(&mut stream),
            )
            .unwrap();
        }
        stream.synchronize().unwrap();
        let production_ms = started.elapsed().as_secs_f64() * 50.0;
        let started = std::time::Instant::now();
        for _ in 0..20 {
            aq4_matvec_wide_load_prototype_f32(
                &index,
                &scale,
                &codebook,
                &scale_values,
                &input,
                None,
                1,
                group_size,
                0.75,
                0,
                rows,
                cols,
                &mut output,
                Some(&mut stream),
            )
            .unwrap();
        }
        stream.synchronize().unwrap();
        let wide_ms = started.elapsed().as_secs_f64() * 50.0;
        assert!(production_ms > 0.0 && wide_ms > 0.0);
        eprintln!("AQ4 M=1 wide-load prototype timing family={family} rows={rows} cols={cols} group{group_size}: production={production_ms:.3} ms, wide={wide_ms:.3} ms, speedup={:.3}x", production_ms / wide_ms);
    }
}
