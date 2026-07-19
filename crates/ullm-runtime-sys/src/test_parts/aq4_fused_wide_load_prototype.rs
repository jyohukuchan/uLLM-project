#[derive(Clone)]
struct Aq4WideLoadMatrixHost {
    rows: usize,
    cols: usize,
    group_size: usize,
    scale_count: usize,
    tensor_scale: f32,
    indices: Vec<u8>,
    scales: Vec<u8>,
    codebook: Vec<f32>,
    scale_values: Vec<f32>,
    row_scales: Option<Vec<f32>>,
}

struct Aq4WideLoadMatrixDevice {
    index: RuntimeBuffer,
    scale: RuntimeBuffer,
    codebook: RuntimeBuffer,
    scale_values: RuntimeBuffer,
    row_scale: Option<RuntimeBuffer>,
}

fn aq4_fused_wide_load_gpu_device() -> u32 {
    (1..device_count().unwrap())
        .find(|&device| {
            device_info(device)
                .map(|info| info.gcn_arch_name == "gfx1201")
                .unwrap_or(false)
        })
        .expect("isolated gfx1201 HIP device")
}

fn aq4_fused_wide_load_host_matrix(
    rows: usize,
    cols: usize,
    group_size: usize,
    with_row_scale: bool,
    state: &mut u32,
) -> Aq4WideLoadMatrixHost {
    let mut next = || {
        *state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        *state as f32 / u32::MAX as f32
    };
    let scale_count = 7;
    Aq4WideLoadMatrixHost {
        rows,
        cols,
        group_size,
        scale_count,
        tensor_scale: 0.5 + next(),
        indices: (0..rows * cols / 2)
            .map(|_| ((next() * 16.0) as u8 & 15) | (((next() * 16.0) as u8 & 15) << 4))
            .collect(),
        scales: (0..rows * cols / group_size)
            .map(|_| ((next() * scale_count as f32) as u8).min(scale_count as u8 - 1))
            .collect(),
        // Separate randomized codebooks and scale tables make stream aliasing visible.
        codebook: (0..16).map(|_| next() * 1.5 - 0.75).collect(),
        scale_values: (0..scale_count).map(|_| 0.25 + next()).collect(),
        row_scales: with_row_scale.then(|| (0..rows).map(|_| 0.75 + next() * 0.5).collect()),
    }
}

fn aq4_fused_wide_load_upload(
    context: &mut RuntimeContext,
    stream: &mut RuntimeStream,
    host: &Aq4WideLoadMatrixHost,
) -> Aq4WideLoadMatrixDevice {
    let mut index = context.alloc_buffer(host.indices.len()).unwrap();
    let mut scale = context.alloc_buffer(host.scales.len()).unwrap();
    let mut codebook = context.alloc_buffer(16 * 4).unwrap();
    let mut scale_values = context.alloc_buffer(host.scale_values.len() * 4).unwrap();
    index.copy_from_host(0, &host.indices, Some(stream)).unwrap();
    scale.copy_from_host(0, &host.scales, Some(stream)).unwrap();
    codebook.copy_from_host(0, &f32s_to_le_bytes(&host.codebook), Some(stream)).unwrap();
    scale_values.copy_from_host(0, &f32s_to_le_bytes(&host.scale_values), Some(stream)).unwrap();
    let row_scale = host.row_scales.as_ref().map(|values| {
        let mut buffer = context.alloc_buffer(values.len() * 4).unwrap();
        buffer.copy_from_host(0, &f32s_to_le_bytes(values), Some(stream)).unwrap();
        buffer
    });
    Aq4WideLoadMatrixDevice { index, scale, codebook, scale_values, row_scale }
}

fn aq4_fused_wide_load_read(
    output: &RuntimeBuffer,
    elements: usize,
    stream: &mut RuntimeStream,
) -> Vec<f32> {
    let mut bytes = vec![0; elements * 4];
    output.copy_to_host(0, &mut bytes, Some(stream)).unwrap();
    stream.synchronize().unwrap();
    le_bytes_to_f32s(&bytes)
}

fn aq4_fused_wide_load_assert_changed(actual: &[f32], baseline: &[f32], label: &str) {
    assert!(
        actual.iter().zip(baseline).any(|(left, right)| left.to_bits() != right.to_bits()),
        "{label} mutation did not affect its expected output stream"
    );
}

fn aq4_fused_wide_load_assert_matches_cpu(actual: &[f32], expected: &[f32], label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}: output lengths differ");
    // Match the established AQ4 differential convention: a small absolute floor plus 1%
    // relative allowance.  The wide-load kernels keep FP32 arithmetic, but repartition each
    // row's reduction across lanes, so their reduction tree is intentionally not the CPU's
    // sequential accumulation order.  Exact GPU-to-GPU checks below still guard that a mutation
    // of one packed stream cannot affect another stream.
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        let tolerance = 5e-2_f32 + 1e-2_f32 * expected.abs();
        assert!(
            (actual - expected).abs() <= tolerance,
            "{label} index {index}: actual={actual} expected={expected} tolerance={tolerance}"
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn aq4_fused_wide_load_run_qkv(
    device: u32,
    wide: bool,
    qkv: &Aq4WideLoadMatrixHost,
    z: &Aq4WideLoadMatrixHost,
    a: &Aq4WideLoadMatrixHost,
    b: &Aq4WideLoadMatrixHost,
    input: &[f32],
    a_log: &[f32],
    dt_bias: &[f32],
) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut context = RuntimeContext::create(device).unwrap();
    let mut stream = context.create_stream().unwrap();
    let qkv_device = aq4_fused_wide_load_upload(&mut context, &mut stream, qkv);
    let z_device = aq4_fused_wide_load_upload(&mut context, &mut stream, z);
    let a_device = aq4_fused_wide_load_upload(&mut context, &mut stream, a);
    let b_device = aq4_fused_wide_load_upload(&mut context, &mut stream, b);
    let mut input_device = context.alloc_buffer(input.len() * 4).unwrap();
    let mut a_log_device = context.alloc_buffer(a_log.len() * 4).unwrap();
    let mut dt_bias_device = context.alloc_buffer(dt_bias.len() * 4).unwrap();
    input_device.copy_from_host(0, &f32s_to_le_bytes(input), Some(&mut stream)).unwrap();
    a_log_device.copy_from_host(0, &f32s_to_le_bytes(a_log), Some(&mut stream)).unwrap();
    dt_bias_device.copy_from_host(0, &f32s_to_le_bytes(dt_bias), Some(&mut stream)).unwrap();
    let mut qkv_output = context.alloc_buffer(qkv.rows * 4).unwrap();
    let mut z_output = context.alloc_buffer(z.rows * 4).unwrap();
    let mut gate_output = context.alloc_buffer(a.rows * 4).unwrap();
    let mut beta_output = context.alloc_buffer(b.rows * 4).unwrap();
    let qkv_count = qkv.row_scales.as_ref().map_or(0, Vec::len);
    let z_count = z.row_scales.as_ref().map_or(0, Vec::len);
    let a_count = a.row_scales.as_ref().map_or(0, Vec::len);
    let b_count = b.row_scales.as_ref().map_or(0, Vec::len);
    if wide {
        aq4_matvec_qkv_z_gate_beta_wide_load_prototype_f32(
            &qkv_device.index, &qkv_device.scale, &qkv_device.codebook, &qkv_device.scale_values, qkv_device.row_scale.as_ref(), qkv.scale_count, qkv.group_size, qkv.tensor_scale, qkv_count,
            &z_device.index, &z_device.scale, &z_device.codebook, &z_device.scale_values, z_device.row_scale.as_ref(), z.scale_count, z.group_size, z.tensor_scale, z_count,
            &a_device.index, &a_device.scale, &a_device.codebook, &a_device.scale_values, a_device.row_scale.as_ref(), a.scale_count, a.group_size, a.tensor_scale, a_count,
            &b_device.index, &b_device.scale, &b_device.codebook, &b_device.scale_values, b_device.row_scale.as_ref(), b.scale_count, b.group_size, b.tensor_scale, b_count,
            &input_device, &a_log_device, &dt_bias_device, qkv.rows, z.rows, a.rows, qkv.cols,
            &mut qkv_output, &mut z_output, &mut gate_output, &mut beta_output, Some(&mut stream),
        ).unwrap();
    } else {
        aq4_matvec_qkv_z_gate_beta_f32(
            &qkv_device.index, &qkv_device.scale, &qkv_device.codebook, &qkv_device.scale_values, qkv_device.row_scale.as_ref(), qkv.scale_count, qkv.group_size, qkv.tensor_scale, qkv_count,
            &z_device.index, &z_device.scale, &z_device.codebook, &z_device.scale_values, z_device.row_scale.as_ref(), z.scale_count, z.group_size, z.tensor_scale, z_count,
            &a_device.index, &a_device.scale, &a_device.codebook, &a_device.scale_values, a_device.row_scale.as_ref(), a.scale_count, a.group_size, a.tensor_scale, a_count,
            &b_device.index, &b_device.scale, &b_device.codebook, &b_device.scale_values, b_device.row_scale.as_ref(), b.scale_count, b.group_size, b.tensor_scale, b_count,
            &input_device, &a_log_device, &dt_bias_device, qkv.rows, z.rows, a.rows, qkv.cols,
            &mut qkv_output, &mut z_output, &mut gate_output, &mut beta_output, Some(&mut stream),
        ).unwrap();
    }
    stream.synchronize().unwrap();
    (
        aq4_fused_wide_load_read(&qkv_output, qkv.rows, &mut stream),
        aq4_fused_wide_load_read(&z_output, z.rows, &mut stream),
        aq4_fused_wide_load_read(&gate_output, a.rows, &mut stream),
        aq4_fused_wide_load_read(&beta_output, b.rows, &mut stream),
    )
}

#[allow(clippy::too_many_arguments)]
fn aq4_fused_wide_load_run_silu(
    device: u32,
    wide: bool,
    gate: &Aq4WideLoadMatrixHost,
    up: &Aq4WideLoadMatrixHost,
    input: &[f32],
) -> Vec<f32> {
    let mut context = RuntimeContext::create(device).unwrap();
    let mut stream = context.create_stream().unwrap();
    let gate_device = aq4_fused_wide_load_upload(&mut context, &mut stream, gate);
    let up_device = aq4_fused_wide_load_upload(&mut context, &mut stream, up);
    let mut input_device = context.alloc_buffer(input.len() * 4).unwrap();
    input_device.copy_from_host(0, &f32s_to_le_bytes(input), Some(&mut stream)).unwrap();
    let mut output = context.alloc_buffer(gate.rows * 4).unwrap();
    let gate_count = gate.row_scales.as_ref().map_or(0, Vec::len);
    let up_count = up.row_scales.as_ref().map_or(0, Vec::len);
    if wide {
        aq4_matvec_silu_mul_wide_load_prototype_f32(
            &gate_device.index, &gate_device.scale, &gate_device.codebook, &gate_device.scale_values, gate_device.row_scale.as_ref(), gate.scale_count, gate.group_size, gate.tensor_scale, gate_count,
            &up_device.index, &up_device.scale, &up_device.codebook, &up_device.scale_values, up_device.row_scale.as_ref(), up.scale_count, up.group_size, up.tensor_scale, up_count,
            &input_device, gate.rows, gate.cols, &mut output, Some(&mut stream),
        ).unwrap();
    } else {
        aq4_matvec_silu_mul_f32(
            &gate_device.index, &gate_device.scale, &gate_device.codebook, &gate_device.scale_values, gate_device.row_scale.as_ref(), gate.scale_count, gate.group_size, gate.tensor_scale, gate_count,
            &up_device.index, &up_device.scale, &up_device.codebook, &up_device.scale_values, up_device.row_scale.as_ref(), up.scale_count, up.group_size, up.tensor_scale, up_count,
            &input_device, gate.rows, gate.cols, &mut output, Some(&mut stream),
        ).unwrap();
    }
    stream.synchronize().unwrap();
    aq4_fused_wide_load_read(&output, gate.rows, &mut stream)
}

#[allow(clippy::too_many_arguments)]
fn aq4_fused_wide_load_time_qkv(
    device: u32, wide: bool, qkv: &Aq4WideLoadMatrixHost, z: &Aq4WideLoadMatrixHost,
    a: &Aq4WideLoadMatrixHost, b: &Aq4WideLoadMatrixHost, input: &[f32], a_log: &[f32],
    dt_bias: &[f32], rounds: usize,
) -> f64 {
    let mut context = RuntimeContext::create(device).unwrap(); let mut stream = context.create_stream().unwrap();
    let q = aq4_fused_wide_load_upload(&mut context, &mut stream, qkv); let zq = aq4_fused_wide_load_upload(&mut context, &mut stream, z);
    let aq = aq4_fused_wide_load_upload(&mut context, &mut stream, a); let bq = aq4_fused_wide_load_upload(&mut context, &mut stream, b);
    let mut input_q = context.alloc_buffer(input.len() * 4).unwrap(); let mut a_log_q = context.alloc_buffer(a_log.len() * 4).unwrap(); let mut dt_bias_q = context.alloc_buffer(dt_bias.len() * 4).unwrap();
    input_q.copy_from_host(0, &f32s_to_le_bytes(input), Some(&mut stream)).unwrap(); a_log_q.copy_from_host(0, &f32s_to_le_bytes(a_log), Some(&mut stream)).unwrap(); dt_bias_q.copy_from_host(0, &f32s_to_le_bytes(dt_bias), Some(&mut stream)).unwrap();
    let mut q_out = context.alloc_buffer(qkv.rows * 4).unwrap(); let mut z_out = context.alloc_buffer(z.rows * 4).unwrap(); let mut gate_out = context.alloc_buffer(a.rows * 4).unwrap(); let mut beta_out = context.alloc_buffer(b.rows * 4).unwrap();
    let q_count = qkv.row_scales.as_ref().map_or(0, Vec::len); let z_count = z.row_scales.as_ref().map_or(0, Vec::len); let a_count = a.row_scales.as_ref().map_or(0, Vec::len); let b_count = b.row_scales.as_ref().map_or(0, Vec::len);
    let mut launch = |mut stream: &mut RuntimeStream| if wide {
        aq4_matvec_qkv_z_gate_beta_wide_load_prototype_f32(&q.index, &q.scale, &q.codebook, &q.scale_values, q.row_scale.as_ref(), qkv.scale_count, qkv.group_size, qkv.tensor_scale, q_count, &zq.index, &zq.scale, &zq.codebook, &zq.scale_values, zq.row_scale.as_ref(), z.scale_count, z.group_size, z.tensor_scale, z_count, &aq.index, &aq.scale, &aq.codebook, &aq.scale_values, aq.row_scale.as_ref(), a.scale_count, a.group_size, a.tensor_scale, a_count, &bq.index, &bq.scale, &bq.codebook, &bq.scale_values, bq.row_scale.as_ref(), b.scale_count, b.group_size, b.tensor_scale, b_count, &input_q, &a_log_q, &dt_bias_q, qkv.rows, z.rows, a.rows, qkv.cols, &mut q_out, &mut z_out, &mut gate_out, &mut beta_out, Some(&mut stream)).unwrap();
    } else {
        aq4_matvec_qkv_z_gate_beta_f32(&q.index, &q.scale, &q.codebook, &q.scale_values, q.row_scale.as_ref(), qkv.scale_count, qkv.group_size, qkv.tensor_scale, q_count, &zq.index, &zq.scale, &zq.codebook, &zq.scale_values, zq.row_scale.as_ref(), z.scale_count, z.group_size, z.tensor_scale, z_count, &aq.index, &aq.scale, &aq.codebook, &aq.scale_values, aq.row_scale.as_ref(), a.scale_count, a.group_size, a.tensor_scale, a_count, &bq.index, &bq.scale, &bq.codebook, &bq.scale_values, bq.row_scale.as_ref(), b.scale_count, b.group_size, b.tensor_scale, b_count, &input_q, &a_log_q, &dt_bias_q, qkv.rows, z.rows, a.rows, qkv.cols, &mut q_out, &mut z_out, &mut gate_out, &mut beta_out, Some(&mut stream)).unwrap();
    };
    for _ in 0..3 { launch(&mut stream); } stream.synchronize().unwrap();
    let started = std::time::Instant::now(); for _ in 0..rounds { launch(&mut stream); } stream.synchronize().unwrap();
    started.elapsed().as_secs_f64() * 1_000.0 / rounds as f64
}

fn aq4_fused_wide_load_time_silu(
    device: u32, wide: bool, gate: &Aq4WideLoadMatrixHost, up: &Aq4WideLoadMatrixHost,
    input: &[f32], rounds: usize,
) -> f64 {
    let mut context = RuntimeContext::create(device).unwrap(); let mut stream = context.create_stream().unwrap();
    let gate_q = aq4_fused_wide_load_upload(&mut context, &mut stream, gate); let up_q = aq4_fused_wide_load_upload(&mut context, &mut stream, up); let mut input_q = context.alloc_buffer(input.len() * 4).unwrap(); input_q.copy_from_host(0, &f32s_to_le_bytes(input), Some(&mut stream)).unwrap(); let mut output = context.alloc_buffer(gate.rows * 4).unwrap();
    let gate_count = gate.row_scales.as_ref().map_or(0, Vec::len); let up_count = up.row_scales.as_ref().map_or(0, Vec::len);
    let mut launch = |mut stream: &mut RuntimeStream| if wide {
        aq4_matvec_silu_mul_wide_load_prototype_f32(&gate_q.index, &gate_q.scale, &gate_q.codebook, &gate_q.scale_values, gate_q.row_scale.as_ref(), gate.scale_count, gate.group_size, gate.tensor_scale, gate_count, &up_q.index, &up_q.scale, &up_q.codebook, &up_q.scale_values, up_q.row_scale.as_ref(), up.scale_count, up.group_size, up.tensor_scale, up_count, &input_q, gate.rows, gate.cols, &mut output, Some(&mut stream)).unwrap();
    } else {
        aq4_matvec_silu_mul_f32(&gate_q.index, &gate_q.scale, &gate_q.codebook, &gate_q.scale_values, gate_q.row_scale.as_ref(), gate.scale_count, gate.group_size, gate.tensor_scale, gate_count, &up_q.index, &up_q.scale, &up_q.codebook, &up_q.scale_values, up_q.row_scale.as_ref(), up.scale_count, up.group_size, up.tensor_scale, up_count, &input_q, gate.rows, gate.cols, &mut output, Some(&mut stream)).unwrap();
    };
    for _ in 0..3 { launch(&mut stream); } stream.synchronize().unwrap();
    let started = std::time::Instant::now(); for _ in 0..rounds { launch(&mut stream); } stream.synchronize().unwrap();
    started.elapsed().as_secs_f64() * 1_000.0 / rounds as f64
}

#[test]
#[ignore = "requires an isolated gfx1201 HIP device and ULLM_RUN_AQ4_FUSED_WIDE_LOAD_DIFFERENTIAL=1"]
fn hip_aq4_fused_wide_load_prototypes_match_cpu_and_keep_packed_streams_independent() {
    assert_eq!(std::env::var("ULLM_RUN_AQ4_FUSED_WIDE_LOAD_DIFFERENTIAL").as_deref(), Ok("1"));
    let _lock = AQ4_EXPERIMENTAL_ENV_LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    let gpu = aq4_fused_wide_load_gpu_device();
    let mut state = 0x510e_527f;
    // Qwen3.5-9B linear-attention decode: qkv=[8192,4096], z=[4096,4096], a/b=[32,4096].
    let qkv = aq4_fused_wide_load_host_matrix(8_192, 4_096, 16, true, &mut state);
    let z = aq4_fused_wide_load_host_matrix(4_096, 4_096, 16, false, &mut state);
    let a = aq4_fused_wide_load_host_matrix(32, 4_096, 16, true, &mut state);
    let b = aq4_fused_wide_load_host_matrix(32, 4_096, 16, false, &mut state);
    let input: Vec<f32> = (0..4_096).map(|_| { state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223); state as f32 / u32::MAX as f32 * 2.0 - 1.0 }).collect();
    let a_log = vec![0.25; 32];
    let dt_bias = vec![-0.125; 32];
    let baseline_cpu = aq4_fused_wide_load_run_qkv(0, false, &qkv, &z, &a, &b, &input, &a_log, &dt_bias);
    let baseline_gpu = aq4_fused_wide_load_run_qkv(gpu, true, &qkv, &z, &a, &b, &input, &a_log, &dt_bias);
    for (label, actual, expected) in [
        ("baseline qkv", &baseline_gpu.0, &baseline_cpu.0),
        ("baseline z", &baseline_gpu.1, &baseline_cpu.1),
        ("baseline gate", &baseline_gpu.2, &baseline_cpu.2),
        ("baseline beta", &baseline_gpu.3, &baseline_cpu.3),
    ] {
        aq4_fused_wide_load_assert_matches_cpu(actual, expected, label);
    }
    for stream_index in 0..4 {
        let (mut mqkv, mut mz, mut ma, mut mb) = (qkv.clone(), z.clone(), a.clone(), b.clone());
        match stream_index { 0 => mqkv.indices[0] ^= 0x11, 1 => mz.indices[0] ^= 0x11, 2 => ma.indices[0] ^= 0x11, _ => mb.indices[0] ^= 0x11 }
        let expected = aq4_fused_wide_load_run_qkv(0, false, &mqkv, &mz, &ma, &mb, &input, &a_log, &dt_bias);
        let actual = aq4_fused_wide_load_run_qkv(gpu, true, &mqkv, &mz, &ma, &mb, &input, &a_log, &dt_bias);
        let actual_streams = [&actual.0, &actual.1, &actual.2, &actual.3];
        let expected_streams = [&expected.0, &expected.1, &expected.2, &expected.3];
        let baseline_gpu_streams = [&baseline_gpu.0, &baseline_gpu.1, &baseline_gpu.2, &baseline_gpu.3];
        for (label, actual, expected) in [
            ("mutated qkv", actual_streams[0], expected_streams[0]),
            ("mutated z", actual_streams[1], expected_streams[1]),
            ("mutated gate", actual_streams[2], expected_streams[2]),
            ("mutated beta", actual_streams[3], expected_streams[3]),
        ] {
            aq4_fused_wide_load_assert_matches_cpu(actual, expected, label);
        }
        let changed = [&expected.0, &expected.1, &expected.2, &expected.3];
        let baseline = [&baseline_cpu.0, &baseline_cpu.1, &baseline_cpu.2, &baseline_cpu.3];
        aq4_fused_wide_load_assert_changed(changed[stream_index], baseline[stream_index], "qkv/z/gate/beta");
        aq4_fused_wide_load_assert_changed(actual_streams[stream_index], baseline_gpu_streams[stream_index], "GPU qkv/z/gate/beta");
        for other in 0..4 {
            if other != stream_index {
                assert_f32s_close(changed[other], baseline[other], 0.0);
                assert_f32s_close(actual_streams[other], baseline_gpu_streams[other], 0.0);
            }
        }
    }
    // The same production M=1 fusion serves both decode paths: gate/up=[12288,4096].
    let gate = aq4_fused_wide_load_host_matrix(12_288, 4_096, 16, true, &mut state);
    let up = aq4_fused_wide_load_host_matrix(12_288, 4_096, 16, false, &mut state);
    let baseline_cpu = aq4_fused_wide_load_run_silu(0, false, &gate, &up, &input);
    let baseline_gpu = aq4_fused_wide_load_run_silu(gpu, true, &gate, &up, &input);
    aq4_fused_wide_load_assert_matches_cpu(&baseline_gpu, &baseline_cpu, "baseline SiLU-mul");
    for gate_stream in [true, false] {
        let (mut modified_gate, mut modified_up) = (gate.clone(), up.clone());
        if gate_stream { modified_gate.indices[0] ^= 0x11; } else { modified_up.indices[0] ^= 0x11; }
        let expected = aq4_fused_wide_load_run_silu(0, false, &modified_gate, &modified_up, &input);
        let actual = aq4_fused_wide_load_run_silu(gpu, true, &modified_gate, &modified_up, &input);
        aq4_fused_wide_load_assert_matches_cpu(&actual, &expected, "mutated SiLU-mul");
        aq4_fused_wide_load_assert_changed(&expected, &baseline_cpu, if gate_stream { "SiLU gate" } else { "SiLU up" });
        aq4_fused_wide_load_assert_changed(&actual, &baseline_gpu, if gate_stream { "GPU SiLU gate" } else { "GPU SiLU up" });
    }
}

#[test]
#[ignore = "requires an isolated gfx1201 HIP device and ULLM_RUN_AQ4_FUSED_WIDE_LOAD_TIMING=1"]
fn hip_aq4_fused_wide_load_prototypes_timing_vs_production() {
    assert_eq!(std::env::var("ULLM_RUN_AQ4_FUSED_WIDE_LOAD_TIMING").as_deref(), Ok("1"));
    let _lock = AQ4_EXPERIMENTAL_ENV_LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    let gpu = aq4_fused_wide_load_gpu_device();
    let mut state = 0x1f83_d9ab;
    let qkv = aq4_fused_wide_load_host_matrix(8_192, 4_096, 16, true, &mut state);
    let z = aq4_fused_wide_load_host_matrix(4_096, 4_096, 16, false, &mut state);
    let a = aq4_fused_wide_load_host_matrix(32, 4_096, 16, true, &mut state);
    let b = aq4_fused_wide_load_host_matrix(32, 4_096, 16, false, &mut state);
    let input = vec![0.125; 4_096]; let a_log = vec![0.25; 32]; let dt_bias = vec![-0.125; 32];
    let gate = aq4_fused_wide_load_host_matrix(12_288, 4_096, 16, true, &mut state);
    let up = aq4_fused_wide_load_host_matrix(12_288, 4_096, 16, false, &mut state);
    let qkv_production = aq4_fused_wide_load_time_qkv(gpu, false, &qkv, &z, &a, &b, &input, &a_log, &dt_bias, 20);
    let qkv_wide = aq4_fused_wide_load_time_qkv(gpu, true, &qkv, &z, &a, &b, &input, &a_log, &dt_bias, 20);
    let silu_production = aq4_fused_wide_load_time_silu(gpu, false, &gate, &up, &input, 20);
    let silu_wide = aq4_fused_wide_load_time_silu(gpu, true, &gate, &up, &input, 20);
    assert!(qkv_production > 0.0 && qkv_wide > 0.0 && silu_production > 0.0 && silu_wide > 0.0);
    eprintln!("AQ4 fused qkv/z gate/beta [8192/4096/32,4096]: production={qkv_production:.3} ms, wide={qkv_wide:.3} ms, speedup={:.3}x", qkv_production / qkv_wide);
    eprintln!("AQ4 fused SiLU-mul [12288,4096]: production={silu_production:.3} ms, wide={silu_wide:.3} ms, speedup={:.3}x", silu_production / silu_wide);
}
