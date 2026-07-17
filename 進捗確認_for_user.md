# Phase 7 CPU-only preparation progress

- 新規48ケースを生成し、旧07/15 split全48件・No-Go 19件・Phase 1〜6の3 contextとのhash非重複を検証した。
- formal split（calibration 24 / holdout 24）、holdout execution view、比較器、single-window driver、runbookを追加した。
- BF16 source oracleをGPU可視性を無効化したCPU-only runnerで生成中である。GPU、service、systemd、active manifest、lockには触れていない。
