# AQ4 WMMA prototype v2: 二重バッファのソフトウェア・パイプライン

## 対象範囲

`ullm_aq4_gemm_wmma_prototype_v2_f32_kernel` は、gfx1201 専用の追加実験であり、direct API
からだけ呼び出せる。`ullm_aq4_gemm_wmma_prototype_f32_kernel`、production WMMA module/cache、
AQ4 registry、dispatch 判定は変更しない。新しい C ABI と Rust wrapper には意図的に
`*_wmma_prototype_v2_*` という名前を付け、同じプロセスで v1 と v2 を比較できるようにした。

v2 の kernel guard、host ABI validation、launch grid はすでに v1 と同じ一般化済み contract を
使う。すなわち `gfx1201`、group16、M=128、`rows % 16 == 0`、`cols % 32 == 0` であり、grid は
`(rows / 16, 128 / 128)` である。このため対象には attn_q/linear_attn_qkv (`8192x4096`)、
linear_attn_z (`4096x4096`)、linear_attn_a/linear_attn_b (`32x4096`)、mlp_gate/mlp_up
(`12288x4096`)、mlp_down (`4096x12288`) の全てが含まれる。今回追加したのは、この全 shape set を
v2 で実機検証する GPU-gated test coverage である。実機 differential と timing の結果が揃うまでは
production dispatch を v1 のまま維持する。

## roofline の前提

project の architecture reference は FP16 WMMA を 1024 FLOP/clock/CU と記載している。
`rocminfo` が報告する 64 CU、2350 MHz を用いると、計算値は次のとおりである。

```text
1024 FLOP / CU-cycle * 64 CU * 2.350 GHz = 154,009.6 GFLOPS = 154.01 TFLOPS
```

AMD の R9700 product page は boost clock 2920 MHz における FP16 matrix peak を 191 TFLOPS
と記載する。`1024 * 64 * 2.920 = 191.365 TFLOPS` であり、reference table の単位が
FLOP/CU-cycle であることを独立に裏付ける。既存の `mlp_gate/mlp_up` 15.56 TFLOPS は、
2350 MHz の dense-FP16 WMMA roofline の 10.1%（boost clock の marketing peak の 8.13%）である。

## v2 の仮説と resource のトレードオフ

v1 は K=32 Wide-K tile ごとに、load publish barrier と WMMA 後の LDS 再利用 barrier を一つずつ
実行する。最終 output tile barrier を含めると、`cols=4096` / `12288` の CTA 当たりの barrier 数は
257 / 769 である。v1 の CTA 当たり LDS 使用量は 17,408 byte である。

- weight: `16 * 32 * 2 = 1,024 B`
- input: `128 * 32 * 2 = 8,192 B`
- FP32 output transpose tile: `8 * 16 * 16 * 4 = 8,192 B`

v2 は K 依存の weight/input LDS tile だけを二重化し、合計 26,624 B とする。stage 0 を最初に
準備し、current tile の二つの WMMA を発行する前に next K=32 tile を register に load する。その後、
もう一方の LDS stage に store して、一つの uniform producer/consumer barrier で公開する。最終 output
barrier を含めて 129 / 385 回となり、v1 の K-loop barrier 数をほぼ半分にする。

この 26,624 B は CTA ごとの固定値であり、`32x4096` の場合も変わらない。small-row shape は
`grid_x = 32 / 16 = 2` で CTA が二つになるだけである。gfx1201 の 64 KiB/CU LDS budget を下回ることを
kernel source の compile-time assertion でも固定した。これは resident CTA 数や VGPR occupancy を保証する
ものではないため、実機 validation では compile/launch 成功、differential、timing を必ず確認する。

LDS のみで見る resident 上限は、3 CTA / 24 wave32（32 wavefront/CU の 75%）から、2 CTA / 16 wave32
（50%）になる。FP32 input prefetch の register 使用量により、実際の occupancy はさらに下がり得る。
このため v2 は production への変更ではなく、明確な A/B 実験である。timing を解釈する前に GPU profiler
の `LDS_Block_Size`、`VGPR_Count`、`Scratch_Size`、occupancy counter を確認する。

## 隔離した service window で実行する GPU validation

```bash
ULLM_RUN_AQ4_WMMA_PROTOTYPE_V2_DIFFERENTIAL=1 \
  cargo test -p ullm-runtime-sys \
  hip_aq4_wmma_prototype_v2_m128_group16_model_shapes_match_cpu_when_enabled \
  -- --ignored --nocapture --test-threads=1
```

この test は全 5 group16 production geometry、M=128 を CPU AQ4 reference と比較する。128 回と
384 回の Wide-K iteration、optional row-scale ABI の null/non-null、small `32x4096` の 2-row-tile
launch を通す。staging tolerance は `0.05 + 0.01 * abs(expected)` である。

```bash
ULLM_RUN_AQ4_WMMA_PROTOTYPE_V2_TIMING=1 \
  cargo test -p ullm-runtime-sys \
  hip_aq4_wmma_prototype_v2_m128_group16_model_shapes_timing_vs_wmma_prototype_when_enabled \
  -- --ignored --nocapture --test-threads=1
```

timing test は各 isolated module を 3 回 warm-up し、全 5 production geometry で v1 を 20 launch、v2 を
20 launchする。ms/launch、nominal GEMM TFLOPS、v1/v2 speedup を出力する。

## CPU-only verification

```text
cargo test -p ullm-runtime-sys -- --test-threads=1
160 passed, 0 failed, 7 ignored

cargo test -p ullm-engine --lib
737 passed, 0 failed, 1 ignored

cargo test -p ullm-engine --lib backend_operation_registry::tests:: -- --test-threads=1
50 passed, 0 failed, 1 ignored

git diff --check
passed
```

`cargo fmt --all --check` は今回の変更とは無関係な既存 source files の formatting diff を報告するため
失敗する。整形対象を勝手に広げず、その baseline failure はこの変更に含めない。GPU-gated ignored test は
有効化しておらず、この作業では HIPRTC と GPU kernel を実行していない。
