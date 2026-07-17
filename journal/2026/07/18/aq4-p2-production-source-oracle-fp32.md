# AQ4 P2 production source oracle FP32 switch

## 前回の要点

- `capture-aq4-p2-production-source-oracle.py` はQwen3.5-9BのCPU-only source oracleをBF16 runtimeで実行する契約だった。
- AVX2のみのThreadripper PRO 3995WXではCPU BF16行列積が極端に遅く、P2の8 anchor captureを実用時間で完了できなかった。

## 今回の変更点

- source modelの`AutoModelForCausalLM.from_pretrained(..., dtype=...)`を`torch.float32`へ変更した。capture manifestの`runtime.dtype`とpreparationの`oracle-contract.json`も`float32`へ更新した。
- `identity.source_checkpoint.dtype`はcheckpointの保存形式を示すため、実際のBF16 safetensorsに合わせて保持した。vector sidecarは従来どおり`f32` little-endianであり、runtime dtypeとは別契約である。
- P2 oracle/source calibration仕様を更新し、現行production captureのFP32 runtimeと、既存BF16 exporter・履歴artifactを分離して記述した。comparatorにはruntime BF16固定チェックはなく、F32 sidecar契約だけを検証するため変更不要だった。

## CPU-only 検証

- GPU可視化を4変数すべて`-1`にして、n128 anchor 1件のみの一時preparation fixtureを作成し、`--confirm-cpu-source-capture --threads 1`を実行した。
  - capture manifest `runtime.elapsed_seconds`: 38.94秒
  - コマンド全体（preparation/model identityのSHA-256再検証を含む）: 58.29秒
  - `runtime.dtype=float32`、`vector_contract.dtype=f32`、checkpoint dtype=`bfloat16`、SHA256SUMS検証成功。
- `pytest -q tests/test_aq4_p2_production_baseline_preparation.py tests/test_qwen35_aq4_p2_oracle.py tests/test_aq4_production_p2_evidence.py` — `36 passed, 25 subtests passed`。
- `python3 -m py_compile`（capture、prepare、compare）— 成功。
- 検証用の一時preparation/outputは実行後にゴミ箱へ移動した。本番の8ケースfixtureは変更していない。

## 実行していないこと

- GPU/HIP inference、service操作、sudo、SQ8関連の変更は一切行っていない。

## 次の行動

1. P2 source oracle本番8ケースをFP32 runtimeでcaptureする場合は、長さ加重の概算を目安にCPU-onlyで実行する。
2. source sidecarが揃った後にのみ、既存のpath oracleとstreaming comparatorによるP2比較へ進む。
