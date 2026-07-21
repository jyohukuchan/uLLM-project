# Phase 7 CPU-only preparation progress

- 新規48ケースを生成し、旧07/15 split全48件・No-Go 19件・Phase 1〜6の3 contextとのhash非重複を検証した。
- formal split（calibration 24 / holdout 24）、holdout execution view、比較器、single-window driver、runbookを追加した。
- GPU可視性を無効化したCPU-only BF16 source oracleはcalibration/holdout各24件で完走し、双方ともvalidator `valid`、nonfinite 0、CPU/BF16、model load 1回、全checksum一致を確認した。
- `prepare --verify`、staging `--verify`、shell/Python構文検査、Phase 7対象pytest 8件も成功した。GPU、service、systemd、active manifest、lockには触れていない。
- 次は親エージェントがjournal/runbook記載のroot-only rehearsal 3回を成功させた後、single service-stop windowを一回だけ実行する。

## Importance-score selection progress

- GPU resumeは安全停止した。`HIP_VISIBLE_DEVICES=1`で可視GPU 1台、`gfx1201`、31.859 GiBを確認したが、Qwen D_stats 2-sample smokeの初期化直後にlinear-attention fast-path不在とPyTorch fallbackの警告を検出したため、sample forward前にSIGINTで停止した（wall 20.56秒、完走sample 0）。
- 正式Qwen run、D_stats/D_block/D_KL、Qwen/Gemma cpu-subsets、Gemma BF16/UD-Q4_K_XL/Q4_K_Mの実在を確認済み。追加downloadは0 byteで、Gemma lockboxは未開封のままである。
- 関連GPUジョブは残っておらず、停止後のR9700空きは31.791 GiB。`ullm-openai.service`には触れておらず、V620を可視化するコマンドも実行していない。次はfast-path依存関係の扱いについて人間判断が必要である。
- AQ5をAQ4と同じBF16 family-codebook/E4M3-like group-scale/tensor-scale構造の5-bit/32-entry候補としてCPU sampler/exporterへ実装した。AQ4/AQ5はdisjoint deterministic fit/evalとなり、synthetic testと実Qwen tensor smokeが成功した。
- Qwen same-revision Q4_K_M（5.68 GB）とGemma E4B BF16（15.99 GB）、同revision UD/static GGUF（計10.10 GB）を取得し、全取得ファイルのSHA-256がHugging Face LFS SHAと一致した。Gemma UDはQ4_K/Q5_K/Q6_K混在で、E4B lockboxのまま進められる。
- UltraChat、MBPP、JParaCrawl、GSM8K、FineWebを混ぜた正式raw corpusをhash選択でfreezeした。D_statsは2,400 recordsでQwen 267,068 / Gemma 267,794 valid tokens、D_blockはQwen 14,040 / Gemma 14,093、D_KLはQwen 7,032 / Gemma 7,033 tokensである。32-prompt pilotとは混ぜない。
- Qwen UD/staticは427 tensorで名前・shapeが完全一致し、eligible core 200/200のpaired coverageを確認した。Qwenの正式score計測は次で、Gemmaのtensor labelとのjoinはQwen candidate freeze後まで行わない。
