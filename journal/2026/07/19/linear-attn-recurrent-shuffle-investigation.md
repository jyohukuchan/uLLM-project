# Linear-attention recurrent shuffle investigation

## Scope

This work keeps the existing token-serial Gated DeltaNet recurrence. It does not attempt the
chunked/WY-transform formulation and changes only the two per-timestep reductions inside the
existing production recurrent kernel.

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

## Historical prototype

The initial `ullm_linear_attn_recurrent_shuffle_prototype_f32_kernel` was a direct-only,
exact-gfx1201 Qwen prototype. It launched its own HIPRTC module and did not modify production
dispatch. It performed
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
original GPU differential used a tight `1e-4` absolute threshold against the old production
kernel across M=1, 7, 128, and 2048 and logged output/state maximum differences.

## Promotion decision

`PackageLinearAttnResidentStep::load_with_registry` fixes every current Qwen3.5 linear-attention
layer to `key_heads=16`, `value_heads=32`, `key_dim=128`, and `value_dim=128`. The production
registry independently encodes that same exact geometry for both the M=1 recurrent scan and the
M=2..128 recurrent sequence operation, and rejects alternate geometry requests. Therefore all
24 production linear-attention layers use the validated shuffle geometry; no new registry
descriptor, public ABI, or model-level dispatch path is required.

The stable runtime ABI still accepts geometrically general recurrent calls, including existing
small-shape HIP coverage. The stable launcher therefore chooses between two internal HIPRTC code
objects that both export `ullm_linear_attn_recurrent_f32_kernel`: the exact
`gfx1201 + 16/32/128/128 + 128-thread` launch loads the pure shuffle body, while every other
valid shape or explicit block-size override loads the preserved generic body. If the exact
code object cannot compile or load on a HIP installation, the launcher also safely falls back to
the generic body. This is an internal code-object choice, not a new public API, registry
descriptor, or model-level dispatch path, and it keeps the old implementation as a live generic
fallback rather than an unused source-only reference. Keeping the two bodies in separate modules
also avoids inflating the measured shuffle kernel's register/LDS allocation with generic fallback
code.

The direct-only prototype API and timing A/B harness were removed. Its isolated HIPRTC cache was
repurposed as the internal production code-object cache and now resolves the stable kernel entry
point; the prior production source remains the live generic fallback. Production-named
differential coverage now compares the exact promoted launch to the generic fallback at M=2048,
and compares production M=1, 7, and 128 results with the CPU oracle.

The promotion follows user-provided R9700 validation from the prototype revision: all four
M=1/7/128/2048 comparisons passed with output/state maximum absolute differences near `2e-9`
against the prior production body (tolerance `1e-4`), and M=2048 measured `31.476 ms` for the old
body versus `18.371 ms` for the shuffle body (`1.713x`). Those GPU results were supplied before
this CPU-only promotion pass; no HIPRTC compilation or GPU execution occurs here.

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

The initial prototype pass completed its targeted Rust and CPU checks. The promotion pass then
completed the requested CPU-only validation:

- `cargo test -p ullm-runtime-sys -- --test-threads=1`: 161 passed, 10 ignored.
- `cargo test -p ullm-engine --lib`: 743 passed, 4 ignored.
- `cargo fmt -p ullm-runtime-sys --check`: passed for the touched Rust package.
- `git diff --check`: passed.
- `cargo fmt --all --check`: was run but remains blocked by unrelated, pre-existing formatting
  drift in untouched engine files; this pass did not reformat or include those files.

No HIPRTC compilation or GPU kernel execution occurs in this pass.

## GPU validation commands

Use the isolated-device/service-stop procedure, then run:

```bash
ULLM_RUN_LINEAR_ATTN_RECURRENT_PRODUCTION_DIFFERENTIAL=1 \
  cargo test -p ullm-runtime-sys --lib \
  tests::hip_linear_attn_recurrent_production_model_shapes_match_cpu_when_enabled \
  -- --ignored --exact --nocapture --test-threads=1

ULLM_RUN_LINEAR_ATTN_RECURRENT_PRODUCTION_DIFFERENTIAL=1 \
  cargo test -p ullm-runtime-sys --lib \
  tests::hip_linear_attn_recurrent_production_m2048_matches_generic_fallback_when_enabled \
  -- --ignored --exact --nocapture --test-threads=1
```

After those production-name differentials, run the established service-stop-window end-to-end
M=2048 prefill measurement with `ullm-aq4-e2e-prefill-timing`. Confirm that fidelity remains
within the existing production guard and that the end-to-end throughput improvement survives the
full 24-layer workload.
