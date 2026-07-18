# Paged causal GQA rocWMMA prototype

## Scope

Qwen3.5-9B の cold-prefill self-attention 用に、production scalar reader とは別 HIPRTC
module / direct ABI の validate-first prototype を追加した。既存
`ullm_paged_causal_gqa_chunk_f32_kernel`、production operation registry、AQ4、SQ8/FP8 は変更しない。

Qwen runtime を確認した結果、この reader は self-attention の全経路で
`execute_paged_causal_gqa_chunk_sigmoid_gate_f32` を呼ぶ。geometry は Q=16、KV=4、GQA=4、
head/value dim=256、page=256 であり、softmax scale は `1/sqrt(256)=0.0625` である。

## Tile design

- grid は `[ceil(M/16), 16]`、CTA は 256 thread (8 wave32)。一つの CTA が一つの Q head と
  16 query row を処理し、`kv_head = q_head / 4` を使う。
- Q tile は FP16 LDS `[16,256]`、K tile は FP16 LDS `[16,256]`、V tile は F32 LDS
  `[16,256]`。LDS は score `[16,16]` と row state を含めて 33,984 bytes で、gfx1201 の
  64 KiB budget 内である。
- wave 0 は rocWMMA `16x16x16` FP16 fragments、F32 accumulator により、16 回の K tile
  MMA で `Q[16,256] * K[16,256]^T` を計算する。K は source-major に置くことで
  rocWMMA の column-major B fragment `[K,N]` とそのまま一致する。
- 各 16-token source tile は page=256 を跨がない。source tile の先頭から
  `logical_block = source / 256` と block table を一度だけ読み、physical K/V を LDS に
  stage する。
- 各 query row は F32 running max / sum / output accumulator を保持する。causal limit
  `cached_prefix_len + row + 1` 以外を `-inf` / weight 0 とし、tile ごとに FlashAttention
  online-softmax recurrence を適用する。V accumulator と normalization、sigmoid gate は F32。
- AV は v1 では scalar F32 のままである。thread mapping は一 wave が全 16 query row の
  二つの 16-value group を持つため、V LDS read は row 間 broadcast になる。

FP16 へ丸めるのは Q/K staging のみである。AV WMMA には softmax weights の FP16 変換と V の
`[K,N]` layout/transposition が必要なので、precision と lane-layout risk を検証前に増やさない。
これは validation 後の follow-up とする。Q/K/V double buffering も同じく次段階である。

## GPU validation handoff

まず scalar GPU reader との差分を確認する。fixture は L=2048 の最終 M=128 chunk、非 identity
page table、Q=16/KV=4/dim=256、sigmoid gate を含む。許容値は
`abs_error <= 0.005 + 0.01 * abs(scalar)` である。これは Q/K FP16 staging と tile-level
online-softmax grouping の誤差用であり、pure-F32 path の tolerance ではない。両 output の
finite 性も確認する。

```bash
ULLM_RUN_PAGED_CAUSAL_GQA_WMMA_PROTOTYPE_DIFFERENTIAL=1 \
  cargo test -p ullm-runtime-sys \
  hip_paged_causal_gqa_wmma_prototype_qwen35_l2048_matches_scalar_when_enabled \
  -- --ignored --nocapture --test-threads=1
```

成功後、同じ direct API で 16 x M=128 chunk の L=2048 prefill を 3 warm-up、20 timed
iterations で scalar と比較する。表示する useful work は one self-attention layer の
`2 * Q * sum_{t=1..2048}(t) * (head_dim + value_dim)`、34.3765 GFLOP/prefill である
（8 self-attention layers なら 275.012 GFLOP）。

```bash
ULLM_RUN_PAGED_CAUSAL_GQA_WMMA_PROTOTYPE_TIMING=1 \
  cargo test -p ullm-runtime-sys \
  hip_paged_causal_gqa_wmma_prototype_qwen35_l2048_timing_vs_scalar_when_enabled \
  -- --ignored --nocapture --test-threads=1
```

GPU/HIPRTC execution はこの CPU-only 作業では実行しない。promotion 候補にするには differential
が通り、finite output を満たし、この direct timing で scalar より少なくとも 3x 速いことを
最低条件とする。10x は stretch target であり、production registry への wiring 前には e2e
prefill に regression がないことも確認する。

## CPU-only verification

- `cargo test -p ullm-runtime-sys -- --test-threads=1`: 159 passed, 7 ignored。
- `cargo test -p ullm-engine --lib`: 737 passed, 2 ignored。
- `git diff --check`: clean。
- `cargo fmt --all --check`: exit 1。`ullm-aq4-fidelity-capture.rs`、既存 AQ4 diagnostic
  binaries、`loader.rs`、`qwen35_aq4_layer_runtime.rs` など、今回と無関係な既存 formatting
  drift が原因である。これら user worktree files は変更しない。
