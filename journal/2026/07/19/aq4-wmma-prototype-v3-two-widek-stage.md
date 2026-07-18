# AQ4 WMMA prototype v3: 2 Wide-K microtiles per LDS hand-off

## Scope

This is an additive, direct-only gfx1201 AQ4 group16 M=128 experiment. It adds the v3 HIPRTC
source, a separate module cache, a C ABI, a Rust wrapper, and ignored GPU tests. It does not
change the promoted double-buffered v2 kernel, its production cache, registry, dispatch,
group8 path, attention, recurrent kernels, or any serving chunk width.

V3 accepts rows divisible by 16 and cols divisible by 64. Every current group16 production
projection is in that set: 8192x4096, 4096x4096, 32x4096, 12288x4096, and 4096x12288.

## Candidate analysis

V2 uses a 32-K Wide-K tile. One CTA covers 16 output rows by 128 batch columns. Each Wide-K
step has eight waves and two 16x16x16 FP16 WMMA instructions per wave:

    2 * 16 * 32 * 128 = 131,072 FLOP per CTA Wide-K step
    131,072 / 1,024 FP16 WMMA FLOP per CU-cycle = 128 CU-cycles at peak

V3 groups two such 32-K slabs into one pipeline stage, so a hand-off protects four WMMA
instructions per wave:

    2 * 131,072 = 262,144 FLOP per CTA stage
    262,144 / 1,024 = 256 CU-cycles at peak

The important point is not greater arithmetic intensity: it is twice as much existing compute
and LDS consumption between CTA barriers. Both slabs retain the v2 physical LDS leading
dimension of 32 half values. They are not represented as a 64-K matrix, so rocWMMA fragment
loads use the already validated ldm=32 layout rather than introducing a possible new LDS bank
pattern.

### Rejected: three 32-K pipeline stages

The v2 static LDS footprint is 26,624 B:

    2 * (16 * 32 * 2 + 128 * 32 * 2) + 8 * 16 * 16 * 4 = 26,624 B

Three 32-K stages would be 35,840 B. Against the documented 64 KiB/CU LDS budget, that caps
residency at one 8-wave CTA, whereas v2 is LDS-capped at two CTAs / 16 waves. More importantly,
simultaneously holding K+1 and K+2 input cohorts would add a second four-float4 cohort per
thread: another 16 FP32 values, in addition to the fragments and existing address state. It
would not reduce v2's barrier count. There is no profiling evidence that two current WMMA
instructions fail to hide the v2 prefetch latency, so the occupancy and VGPR risk is not
justified.

### Rejected: M=256 batch tile

A direct M=256 adaptation needs 51,200 B:

    2 * 16 * 32 * 2 + 2 * 256 * 32 * 2 + 16 * 16 * 16 * 4 = 51,200 B

It fits under 64 KiB but has one-CTA LDS residency. Its extra reuse is only of the small
weight slab; the much larger activation slab doubles with M, so its arithmetic-intensity gain
is minor. Production prefill uses M=128 chunks. Invoking M=256 at M=128 would leave half the
tile idle, and changing chunking is out of scope.

### Selected: two Wide-K microtiles per stage

V3 keeps two ping-pong stages but doubles the payload of each stage:

    2 * 2 * (16 * 32 * 2 + 128 * 32 * 2) + 8 * 16 * 16 * 4 = 45,056 B

This also limits LDS residency to one 8-wave CTA, versus v2's two 8-wave CTAs. That is the
main downside and makes v3 a prototype, not a promotion candidate by construction.

Unlike the three-stage alternative, v3 has only one v2-sized prefetch cohort live at a time:
four float4 input values per thread, plus one uint4 packed-weight value and two FP32 group
scales for the 16 loader threads. It prefetches, computes, and stores the first next-stage
32-K slab, then repeats the same sequence for the second slab before one hand-off barrier.
The counted inner loop is explicitly kept non-unrolled so GPU validation can check that the
compiler did not retain two cohorts in VGPRs.

For cols=4096, v2 has 129 CTA barriers including priming and output; v3 has 65. For
cols=12288, v2 has 385 and v3 has 193.

The prior isolated v2 timing run gives a reason to test this trade:

| shape | v1 | v2 | v1-to-v2 delta |
| --- | ---: | ---: | ---: |
| 12288x4096 | 0.896 ms | 0.734 ms | 0.162 ms |
| 4096x12288 | 1.142 ms | 0.698 ms | 0.444 ms |

V2 removed one barrier per 32-K step and also overlapped prefetch, so dividing these deltas by
128 or 384 steps is only an aggregate upper-bound signal, not a barrier-cycle measurement.
It is approximately 1.27 and 1.16 microseconds per launch-wide-K step. If all of that delta
were barrier cost and one-CTA occupancy were free, halving it again would imply 0.653 ms and
0.476 ms. Those are deliberately not forecasts: they ignore the v2 prefetch contribution and
the v3 occupancy loss. They establish that a measured A/B experiment is justified.

### Not pursued without evidence: LDS banking, fragment addresses, or spills

No GPU ISA, HIPRTC compilation, or profiler metric collection was performed in this CPU-only
change. The source alone does not establish a bank conflict or spill. V3 therefore preserves
the v2 32-K physical LDS microtile layout instead of making a speculative ldm=64 change. GPU
validation must collect LDS_Block_Size, VGPR_Count, Scratch_Size, and occupancy alongside
timing before any promotion decision.

## Implementation and validation

The unique HIPRTC entry is ullm_aq4_gemm_wmma_prototype_v3_f32_kernel. Its module cache is
separate from production, so the A/B test loads both code objects in one process without
changing the served v2 function.

The ignored differential test compares v3 against the CPU AQ4 reference on all five production
group16 shapes and exercises null and non-null row scales. The tolerance remains 0.05 absolute
plus 1 percent relative because both WMMA paths stage operands as FP16 and accumulate as FP32.

The ignored timing test warms both code objects three times, then times 20 synchronized launches
of production v2 and v3 for all five shapes. It reports ms, nominal TFLOPS, and v2/v3 speedup.

GPU commands:

    ULLM_RUN_AQ4_WMMA_PROTOTYPE_V3_DIFFERENTIAL=1 cargo test -p ullm-runtime-sys hip_aq4_wmma_prototype_v3_m128_group16_model_shapes_match_cpu_when_enabled -- --ignored --nocapture --test-threads=1

    ULLM_RUN_AQ4_WMMA_PROTOTYPE_V3_TIMING=1 cargo test -p ullm-runtime-sys hip_aq4_wmma_prototype_v3_m128_group16_model_shapes_timing_vs_wmma_prototype_when_enabled -- --ignored --nocapture --test-threads=1

Run them only in the established isolated production-service-stop window. A promotion requires
all differential cases to pass, no scratch spill, and repeatable material wins on both MLP
shapes after checking the profiler resource counters. A small win that coincides with a severe
VGPR or occupancy regression should not be promoted.

## CPU-only verification

- cargo test -p ullm-runtime-sys -- --test-threads=1: 161 passed, 0 failed, 12 ignored.
- cargo test -p ullm-engine --lib: 743 passed, 0 failed, 4 ignored.
- HIPRTC compilation and GPU execution were intentionally excluded from this change.
