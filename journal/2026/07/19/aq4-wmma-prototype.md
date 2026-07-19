# AQ4 WMMA group16 M=128 production promotion

## Scope

gfx1201 専用の AQ4_0 group16 rocWMMA kernel を、当初の MLP 二形状専用プロトタイプから、M=128 かつ `rows % 16 == 0`・`cols % 32 == 0` の非ゼロ shape に一般化した。direct ABI の既存名 `ullm_runtime_aq4_matvec_batch_wmma_prototype_f32` は ABI 互換性のため維持するが、production registry では `HipAq4GemmWmmaF32` として選択する。

通常の AQ4 generic experiment classifier は変更しない。production path は新しい direct WMMA ABI を registry から明示的に呼び出すため、既存 register-BM8 の classifier・numerics・M=8/64 等の挙動を変えない。

## 対象 geometry と launch

manifest を実際に確認した group16 family は以下である。重複 family をまとめても、WMMA が扱う dimension は五つある。

| family | rows x cols | grid.x x grid.y (M=128) |
| --- | --- | --- |
| `attn_q`, `linear_attn_qkv` | `8192 x 4096` | `512 x 1` |
| `linear_attn_z` | `4096 x 4096` | `256 x 1` |
| `linear_attn_a`, `linear_attn_b` | `32 x 4096` | `2 x 1` |
| `mlp_gate`, `mlp_up` | `12288 x 4096` | `768 x 1` |
| `mlp_down` | `4096 x 12288` | `256 x 1` |

host launch は `grid_x = rows / 16`、`grid_y = batch_count / 128`、256 threads/block である。shape guard により端数 tile は存在しない。特に `linear_attn_a/b` の `rows=32` は `grid=(2,1)` となり、二つの完全な 16-row CTA を起動する。

Wide-K は二つの group16、すなわち 32 AQ4 element / 16 packed bytes を一度に読む。`cols % 32 == 0` は packed weight row stride と FP32 activation row stride を 16-byte aligned にし、各 K increment の vector load alignment も満たす。

## Dispatch boundary

production WMMA eligibility `W` は次である。

```text
backend == HIP && architecture == gfx1201 && group_size == 16 &&
rows != 0 && cols != 0 && rows % 16 == 0 && cols % 32 == 0
```

既存 group16 register-BM8 eligibility `R` は `rows % 32 == 0 && cols % 128 == 0`（同じ HIP/gfx1201/group16 前提）である。非ゼロ geometry では `R` は `W` の真部分集合である。従って「WMMA を満たさず register を満たす」`R \ W` shape は存在しない。逆の `W \ R`（たとえば `16x32`）だけが存在する。

- `R` の実モデル shape: M=2--7 は legacy、M=8--127 は register-BM8、M=128 は WMMA。
- `W \ R`: M=2--127 は legacy、M=128 のみ WMMA。
- group8 は既存の `HipAq4GemmRegisterBm8Group8F32` の M=8--128 のままである。

M=128 WMMA の feature guard は `ULLM_REQUIRE_HIP_AQ4_WMMA_GEMM_KERNEL=1`。capability probe は実際の `linear_attn_a/b` geometry (`32x4096`, M=128) を direct ABI で起動するため、小さい row-grid も HIPRTC/probe 時点で検証される。別個に registry へ昇格した ragged M=65--127 WMMA path は group16 の `ULLM_REQUIRE_HIP_AQ4_WMMA_GEMM_RAGGED_M_KERNEL=1` と group8 の `ULLM_REQUIRE_HIP_AQ4_WMMA_GEMM_GROUP8_RAGGED_M_KERNEL=1` を必要とする。

## Tile と復号設計

1 CTA は出力の `[16 rows, 128 batches]` を担当する。8 wave32 の各 wave は 16 batch 列を担当し、`A=[16, 32]` と `B=[32, 16]` に対して K=16 の rocWMMA を 2 回発行する。

- AQ4 index は row-major packed nibble であるため、連続する二つの group16 は 32 element、16 bytes になる。各 output row の loader が 16-byte vector load を一度行い、二つの group scale を用いて `codebook[nibble] * scale_values[scale_index]` を FP16 LDS tile に復号する。
- input は batch-major FP32 から、rocWMMA の B fragment が要求する `[K, batch]` column-major FP16 LDS tile に変換する。各 wave は隣接する 16 batch 列を読む。
- WMMA accumulator は FP32 である。`tensor_scale` と optional `row_scale[row]` は K に依存しないので、FP32 accumulator を row-major LDS に store して ABI の `[batch][row]` 出力へ並べ替える際に一度だけ乗算する。CPU AQ4 式と代数的に同じであり、FP16 staging による丸めだけが差分になる。
- rocWMMA fragment API と row-major `store_matrix_sync` を使い、gfx12 raw WMMA accumulator の lane mapping を直接扱わない。

## GPU で次に実行するもの

サービス停止ウィンドウ内で、まず五つの group16 dimension を CPU reference と比較する。

```bash
ULLM_RUN_AQ4_WMMA_PROTOTYPE_DIFFERENTIAL=1 \
  cargo test -p ullm-runtime-sys \
  hip_aq4_wmma_prototype_m128_group16_model_shapes_match_cpu_when_enabled \
  -- --ignored --nocapture --test-threads=1
```

これは `attn_q/linear_attn_qkv`、`linear_attn_z`、`linear_attn_a/b`、`mlp_gate/up`、`mlp_down` を実行する。許容値は `0.05 + 0.01 * abs(expected)` である。両入力を FP16 に丸める一方で accumulator は FP32 なので、これは bounded random input に対する staging 誤差の許容であり、通常の FP32 AQ4 kernel の `1e-3` 互換性を主張する値ではない。

成功後に timing を実行する。

```bash
ULLM_RUN_AQ4_WMMA_PROTOTYPE_TIMING=1 \
  cargo test -p ullm-runtime-sys \
  hip_aq4_wmma_prototype_m128_group16_model_shapes_timing_vs_register_bm8_when_enabled \
  -- --ignored --nocapture --test-threads=1
```

各 shape で 3 warm-up と 20 timed launch を行い、既存 forced register-BM8 と WMMA の ms、TFLOPS、speedup を表示する。さらに production guard を有効化した registry/probe で、M=128 は `hip.aq4-gemm-wmma-f32.gfx1201.group16.m128`、同じ geometry の M=8/64/127 は `hip.aq4-gemm-register-bm8-f32.gfx1201.group16.m8-m127` になることを確認する。

## CPU-only verification

- `cargo test -p ullm-runtime-sys -- --test-threads=1`: 160 passed, 3 ignored.
- `cargo test -p ullm-engine --lib`: 737 passed, 1 ignored.
- `cargo test -p ullm-engine --bin ullm-aq4-differential-trace`: 14 passed.
- `pytest -q tests/test_generate_served_model.py`: 26 passed.
- `cargo fmt --all --check`: 既存の無関係な formatting drift（`ullm-aq4-fidelity-capture`、複数の AQ4 diagnostic binary、`loader.rs`、`qwen35_aq4_layer_runtime.rs`）により exit 1。今回の touched Rust files は formatter output に含まれない。
- `git diff --check`: clean。

HIPRTC compile と GPU execution はこの作業では実行していない。実機での既存 MLP differential/timing 成果を production promotion の根拠にしつつ、新しい三 dimension と production registry/probe は上記の実機手順で確認する。
