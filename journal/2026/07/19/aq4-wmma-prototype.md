# AQ4 WMMA register-batch GEMM prototype

## Scope

`gfx1201` の AQ4_0 group16 に限定した、M=128 の直接起動用 rocWMMA プロトタイプを追加した。対象は Qwen3.5-9B の MLP 投影だけで、`12288x4096`（gate/up）と `4096x12288`（down）を受け付ける。通常の dispatch、SQ8/FP8、既存 AQ4 register/LDS kernel は変更していない。

## Tile と復号設計

1 CTA は出力の `[16 rows, 128 batches]` を担当する。8 wave32 の各 wave は 16 batch 列を担当し、`A=[16, 32]` と `B=[32, 16]` に対して K=16 の rocWMMA を 2 回発行する。

- AQ4 index は row-major の packed nibble であるため、連続する二つの group16 は 32 要素、16 bytes になる。各 output row の loader が 16-byte vector load を一度行い、二つの group scale を用いて `codebook[nibble] * scale_values[scale_index]` を FP16 LDS tile に復号する。
- input は batch-major FP32 から、rocWMMA の B fragment が要求する `[K, batch]` column-major FP16 LDS tile に変換する。各 wave は隣接する 16 batch 列を読む。
- WMMA accumulator は FP32 である。`tensor_scale` と optional `row_scale[row]` は K に依存しないので、FP32 accumulator を row-major LDS に store して ABI の `[batch][row]` 出力へ並べ替える際に一度だけ乗算する。CPU AQ4 式と代数的に同じであり、FP16 staging による丸めだけが差分になる。
- これは AQ4 の LUT 復号を native INT4 WMMA に変えるものではない。ただし packed layout と group16 が正確に 16-byte / 2xK16 に対応するため、Wide-K は自然に適用できる。

rocWMMA fragment API と row-major `store_matrix_sync` を使い、gfx12 raw WMMA accumulator の lane mapping を直接扱わない。HIPRTC module/cache も既存 register module と分離している。

## GPU で次に実行するもの

サービス停止ウィンドウ内で、まず differential を実行する。

```bash
ULLM_RUN_AQ4_WMMA_PROTOTYPE_DIFFERENTIAL=1 \
  cargo test -p ullm-runtime-sys \
  hip_aq4_wmma_prototype_m128_mlp_shapes_match_cpu_when_enabled \
  -- --ignored --nocapture --test-threads=1
```

これは各 target shape の random AQ4 data を CPU `aq4_matvec_batch_f32` と比較する。許容値は `0.05 + 0.01 * abs(expected)` である。両入力を FP16 に丸める一方で accumulator は FP32 なので、これは bounded random input に対する staging 誤差の許容であり、通常の FP32 AQ4 kernel の `1e-3` 互換性を主張する値ではない。layout、transpose、scale-factor の誤りはこの範囲を大きく超える。

成功後に同じウィンドウで timing を実行する。

```bash
ULLM_RUN_AQ4_WMMA_PROTOTYPE_TIMING=1 \
  cargo test -p ullm-runtime-sys \
  hip_aq4_wmma_prototype_m128_timing_vs_register_bm8_when_enabled \
  -- --ignored --nocapture --test-threads=1
```

各 shape で 3 warm-up と 20 timed launch を行い、既存 forced register-BM8 と WMMA の ms、TFLOPS、speedup を表示する。HIPRTC compile/load は warm-up 前に完了する。まず differential 合格が必須で、両 shape で register BM8 の少なくとも 2x（およそ 4 TFLOPS 以上が初期の有望ライン）なら、他の M 幅と AQ4 geometry への一般化を検討する価値がある。1.25x 未満、または片方の MLP shape が退行するなら、現 tile の一般化ではなく LDS/staging と tile shape の再設計を優先する。

## CPU-only verification

- `cargo test -p ullm-runtime-sys -- --test-threads=1`: 159 passed, 3 ignored.
- `cargo test -p ullm-engine --lib`: 733 passed, 1 ignored.
- `cargo fmt --all --check` は既存の `ullm-engine` formatting drift により失敗した。今回変更した runtime-sys Rust files はその output に含まれない。
- `git diff --check` は clean。

HIPRTC compile と GPU execution はこの作業では意図的に実行していない。特に、rocWMMA FP16 fragment の HIPRTC 許容性と実行時の fragment layout は上記 differential で初めて実機確認される。
