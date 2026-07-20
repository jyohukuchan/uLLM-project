# Phase 7 CPU-only preparation progress

- 新規48ケースを生成し、旧07/15 split全48件・No-Go 19件・Phase 1〜6の3 contextとのhash非重複を検証した。
- formal split（calibration 24 / holdout 24）、holdout execution view、比較器、single-window driver、runbookを追加した。
- GPU可視性を無効化したCPU-only BF16 source oracleはcalibration/holdout各24件で完走し、双方ともvalidator `valid`、nonfinite 0、CPU/BF16、model load 1回、全checksum一致を確認した。
- `prepare --verify`、staging `--verify`、shell/Python構文検査、Phase 7対象pytest 8件も成功した。GPU、service、systemd、active manifest、lockには触れていない。
- 次は親エージェントがjournal/runbook記載のroot-only rehearsal 3回を成功させた後、single service-stop windowを一回だけ実行する。

## Importance-score selection progress

- Phase 0--2のQwen CPU-only provisional screenを完了した。registry commitは`49fceeeb`、UD label auditは427 GGUF tensor/200 eligible core tensor、C0/C2/C3は200/200 coverageである。
- 同一cohort Q4_K_Mはlocalに無く、pairing・AUC/Precision@KはHOLD。AQ4/AQ5 storage contractも未完了のためgain/allocationは開始していない。
- D_statsは32 prompt/3,416 valid tokenのpilotであり、formal 256k mixed-domain corpusではない。Gemma、GPU、service/systemd、新規downloadは未実行である。
- 詳細は`benchmarks/results/2026-07-21/aq/importance-score/`のrun artifactと`journal/2026/07/21/importance-score-qwen-phase0-2-cpu-pilot.md`に記録した。
