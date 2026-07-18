# AQ4 end-to-end prefill timing binary

`ullm-aq4-e2e-prefill-timing` を追加した。実パッケージを直接ロードし、`M=128` の
16 チャンクで 2048 token の `ColdPrefill` を実行する。最初の一回は warmup として捨て、
`reset_all_request_state_synchronized()` で request state だけを初期化してから二回目を計測する。

実行前に caller が `HIP_VISIBLE_DEVICES=1` と production AQ4 guard set の全変数を `1` に
設定していることを検証する。runtime index 1 が唯一の HIP device で、architecture が厳密に
`gfx1201` であることを load 前後で確認する。token はロード済み vocab size で剰余を取る決定的な
2048 token 列であり、すべて有効範囲に収まる。

CPU-only verification:

- `cargo build --bin ullm-aq4-e2e-prefill-timing -p ullm-engine`: passed.
- `HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1 cargo test -p ullm-engine --lib`: 737 passed, 1 ignored.
- `rustfmt --edition 2024 --check crates/ullm-engine/src/bin/ullm-aq4-e2e-prefill-timing.rs`: passed.
- `cargo fmt --all --check`: existing unrelated formatting drift prevents a repository-wide pass; the new binary is clean.

GPU execution was intentionally not run in this task.
