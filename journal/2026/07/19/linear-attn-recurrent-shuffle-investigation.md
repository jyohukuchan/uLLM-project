# Linear-attention recurrent shuffle investigation

## Scope

This pass keeps the existing token-serial Gated DeltaNet recurrence. It does not attempt the
chunked/WY-transform formulation and does not change the production recurrent kernel or its
dispatch.

## Current production geometry and synchronization

Qwen3.5 uses `key_heads=16`, `value_heads=32`, `key_dim=128`, and `value_dim=128`. The HIP
launch therefore maps `grid.x = value_heads * value_dim = 4096` blocks, with one block per
`(value_head, value)` state column. The default block width is 128, so the R9700 target executes
four wave32s per block. Each lane owns one key entry and, after commit `39386e63`, keeps its
state scalar in a register throughout the sequence loop.

The resident path has two 128-way LDS reductions per timestep:

1. `sum_k((decay * state[k]) * key[k])`, used by lane 0 to form `v_prime`.
2. `sum_k(updated_state[k] * query[k])`, written as the output element.

Each reduction has one barrier after the `partial[tid]` write and seven tree-stage barriers for
offsets `64, 32, 16, 8, 4, 2, 1`, so it contributes eight barriers. The `v_prime_shared` publish
adds one more barrier and the final output barrier adds one more, for `8 + 1 + 8 + 1 = 18` per
timestep.

The `partial`-write barriers and all tree barriers before the final offset are load-bearing for
the existing LDS tree. The `v_prime_shared` barrier is load-bearing because every lane uses it in
the state update. There are two mechanically redundant *terminal tree* barriers: after the
offset-1 update in each reduction. In the first reduction, the immediately following
`v_prime_shared` barrier prevents lane 1 from reusing `partial[1]` before lane 0 has consumed it.
In the second reduction, the final output/reuse barrier provides that same protection. Thus the
safe straightforward edit is 18 to 16 barriers by retaining the final output/reuse barrier.

The final output/reuse barrier and the second terminal-tree barrier are an alternative protective
pair, not two barriers that can both be deleted in the one-array layout: without either one, a
nonzero lane could enter the next timestep and overwrite `partial[1]` before lane 0 executes its
final `partial[0] += partial[1]`. These redundancies existed before the register-residency change;
that change removed global state traffic but did not leave behind a shared-state synchronization
barrier.

## Added experimental variant

`ullm_linear_attn_recurrent_shuffle_prototype_f32_kernel` is a direct-only, exact-gfx1201 Qwen
prototype. It launches its own HIPRTC module and does not modify production dispatch. It performs
each 128-way sum as:

1. register `__shfl_down` reduction within each wave;
2. one LDS write by each wave leader and one CTA barrier;
3. a second register shuffle in wave 0.

The first reduction is followed by the existing `v_prime_shared` CTA handoff. The prototype uses
separate `current_wave_partials` and `output_wave_partials` scratch arrays. That separation makes
the first reduction of timestep `t+1` safe while wave 0 completes the output reduction of `t`;
the next `v_prime` handoff is the fence before `output_wave_partials` is reused. Its dynamic count
is therefore three CTA barriers per timestep, not 18.

The tree/reduction association order changes, so exact FP32 bit identity is not expected. The
GPU differential test uses a tight `1e-4` absolute threshold against the production kernel across
M=1, 7, 128, and 2048 and logs output/state maximum differences.

## Prefetch and input-traffic assessment

`gate` and `beta` vary every timestep and cannot be hoisted across it. They are nevertheless
block-uniform: `beta` is consumed only by lane 0, and `decay=expf(gate)` is identical in all 128
lanes. A future micro-variant could calculate the next timestep's decay in lane 0 and publish it
through the already-required `v_prime_shared` barrier, avoiding a new barrier after a one-time
prologue. That is only worthwhile if HIPRTC has *not* already scalarized the uniform load/exp,
which cannot be established without compiling/profiling the generated code; it was deferred.

`q` and `k` are already contiguous across the 128 lanes. A cooperative LDS prefetch of future
inputs would require an additional publish barrier, while a one-step per-lane register prefetch of
future `q/k` adds VGPR pressure and has no demonstrated latency benefit without GPU profiling. The
same future `q/k` stream is also duplicated across independent value columns, which cannot be
shared between CTAs in this kernel. No prefetch variant was added in this pass.

## Validation performed without GPU execution

- `cargo check -p ullm-runtime-sys --tests` passed.
- `cargo test -p ullm-runtime-sys --lib tests::cpu_linear_attn_recurrent_shuffle_prototype_rejects_cpu_backend -- --exact` passed.
- `cargo test -p ullm-runtime-sys --lib tests::cpu_linear_attn_recurrent_f32_computes_expected_values -- --exact` passed.
- No HIPRTC compilation or GPU kernel execution was performed.

## GPU validation commands

Use the isolated-device/service-stop procedure, then run:

```bash
ULLM_RUN_LINEAR_ATTN_RECURRENT_SHUFFLE_PROTOTYPE_DIFFERENTIAL=1 \
  cargo test -p ullm-runtime-sys --lib \
  tests::hip_linear_attn_recurrent_shuffle_prototype_matches_production_when_enabled \
  -- --ignored --exact --nocapture --test-threads=1

ULLM_RUN_LINEAR_ATTN_RECURRENT_SHUFFLE_PROTOTYPE_TIMING=1 \
  cargo test -p ullm-runtime-sys --lib \
  tests::hip_linear_attn_recurrent_shuffle_prototype_m2048_timing_vs_production_when_enabled \
  -- --ignored --exact --nocapture --test-threads=1
```

Only consider promotion after the M=2048 differential passes, the direct timing is repeatably
faster, and an end-to-end 2048-token profile confirms a meaningful reduction in recurrent kernel
time with no fidelity regression.
