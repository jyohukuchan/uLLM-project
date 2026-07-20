# AQ4 matvec shuffle production promotion

## 前回の要点

Qwen3.5 lm_head（248320x4096）向けのwidth=8 sub-wave shuffle prototypeは、gfx1201実機でCPU差分と2回のタイミング計測を通過していた。

## 今回の変更点

- `ullm_aq4_matvec_f32_kernel`のHIPRTC本番sourceを、gfx1201かつRPB=32でのみwide-load重み読込みを維持したwidth=8 shuffle reductionへ差し替えた。
- 従来のwide-load/LDS reductionは`[[maybe_unused]]` rollback bodyとして保持し、他architectureまたはRPBではそのbodyを選択する。
- 本番Rust/C ABI経路を`ULLM_REQUIRE_HIP_AQ4_MATVEC_KERNEL=1`でfail-closedにし、lm_head実shapeをCPU参照と比較するignored GPU差分テストを追加した。
- 新規runtime guardは追加していない。既存の`ULLM_REQUIRE_HIP_AQ4_MATVEC_KERNEL`を再利用する。

## 検証

- `cargo test -p ullm-runtime-sys`（164 passed, 40 ignored）
- `cargo test -p ullm-engine --lib`（746 passed, 5 ignored）
- `cargo test -p ullm-engine --bin ullm-aq4-differential-trace`（14 passed）
- `python3 -m pytest -q tests/test_generate_served_model.py`（26 passed）
- `rustfmt --check crates/ullm-runtime-sys/src/test_parts/aq4_matvec_shuffle_prototype.rs`
- `git diff --check`（対象ファイル）
- `cargo fmt --all --check`は対象外ファイルのフォーマット差分のため失敗。全体formatterは実行していない。

## 次の行動

gfx1201実機でproduction-entry差分テストと実運用decodeタイミングを最終確認する。
