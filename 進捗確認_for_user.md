# Phase 7 CPU-only preparation progress

- 新規48ケースを生成し、旧07/15 split全48件・No-Go 19件・Phase 1〜6の3 contextとのhash非重複を検証した。
- formal split（calibration 24 / holdout 24）、holdout execution view、比較器、single-window driver、runbookを追加した。
- GPU可視性を無効化したCPU-only BF16 source oracleはcalibration/holdout各24件で完走し、双方ともvalidator `valid`、nonfinite 0、CPU/BF16、model load 1回、全checksum一致を確認した。
- `prepare --verify`、staging `--verify`、shell/Python構文検査、Phase 7対象pytest 8件も成功した。GPU、service、systemd、active manifest、lockには触れていない。
- 次は親エージェントがjournal/runbook記載のroot-only rehearsal 3回を成功させた後、single service-stop windowを一回だけ実行する。

## Importance-score selection progress

- 2026-07-21の直接許可に基づき、全GPU実行を`HIP_VISIBLE_DEVICES=1`の論理`cuda:0` (`gfx1201`) だけに限定した。Qwenの既知linear-attention torch fallback警告だけを許容し、その他の未知warning/fallbackは0件だった。V620と`ullm-openai.service`には触れていない。
- QwenはD_stats 4 shard（2,400 samples / 267,068 tokens、0.1283秒/sample）、C0/C2/C3 200 tensors、C1（16 samples、0.634秒/sample；score 0.0146秒/tensor-candidate）、C4 400 rows（平均5.943秒/tensor-candidate）、KL-core 50 rows（平均9.468秒/tensor-candidate）、descriptive KL-audit 24 rows（平均8.212秒/tensor-candidate）を完走した。
- Qwen formal reportは200 eligible tensors、10,000 bootstrap / 10,000 permutationで封印した。`C0/C1/C4/AWQ-level/AWQ-tail/range`の順でrhoは`0.0022/-0.0824/0.1988/0.3717/-0.0801/0.3285`、tau-bは`0.0114/-0.0656/0.1716/0.3137/-0.0770/0.2747`で、admission合格は0/6。最終Qwen freezeはHEAD `54910a89`、SHA-256 `0d7a552f...`、Qwen finalistsは空である。
- Gemmaは開封前にshared-KVの非実行layers 24–41 K/V 36 tensorsをconfig-onlyで除外し、258-tensor rosterを封印した。D_stats 4 shard（2,400 samples / 267,794 tokens、0.0914秒/sample）、C0/C2/C3 258 tensors、C1（16 samples、0.6955秒/sample；score 0.0231秒/tensor-candidate）、C4 516 rows（平均2.631秒/tensor-candidate）、KL-core 62 rows（平均6.226秒/tensor-candidate）、descriptive KL-audit 24 rows（平均6.885秒/tensor-candidate）を完走した。
- Gemma lockboxはQwen freeze→source-only prejoin（SHA-256 `f57b965d...`）→GGUF label-openの順で一度だけ開封した。GGUF physical core 294とsealed active roster 258の差は事前除外済みshared-KV 36個のみで、label値に依存しないactive-label viewで258/258 join、same-cohort coverage 1.0を得た。Qwenのscore式・閾値・実装hashは開封後に変更していない。
- Gemma formal reportは258 eligible tensors、10,000 bootstrap / 10,000 permutationで完走した。同じ順でrhoは`0.0937/0.0271/0.2151/0.1053/-0.0432/0.1220`、tau-bは`0.0748/0.0231/0.1732/0.0782/-0.0331/0.1011`、admission合格は0/6。teacher coverageはpaired 1.0、nonconstant 4 families、mixed 5 familiesで合格した。
- frozen worst-model ruleの最終結果は`NO-GO`、two-model finalists 0、winnerなし、phase 6非承認。lockbox receiptは`valid one-shot Gemma lockbox`、事後の式/閾値変更false、第三モデル不要と判定した。追加downloadは0 byte、GPU実行はすべて終了している。

## C5 gradient extension progress

- HEAD `532d488b`から、正式仕様どおりC5a Taylor deletion/Taylor-quantとC5b self-Fisher/empirical Fisherを追加する作業を開始した。
- 既存C0/C1/C4/C6の成果物とmanifestは不変とし、未materializeの`D_fisher`は固定済みraw sourceから既存split非重複の追加manifestとして凍結する方針である。
- 現在は既存quantizer・prejoin・formal report・Qwen freeze・Gemma one-shot lockboxの追加実装境界を監査中。GPU/serviceにはまだ触れていない。
- C5 runner、追加`D_fisher` freezer、既存prejoinのbyte-preserving extension、旧6候補の数値不変検証付きformal report、Qwen freeze、Gemma active-label one-shot joinを実装した。Taylor-quant/self-Fisherだけをwinner eligibleとし、削除Taylor 2種とempirical Fisherはsecondaryに固定した。
- C5は旧6候補と別のBH補正族にし、既存adjusted pを変更しない。CPU-onlyの関連testは現時点で全件成功しており、次は実装commit後にQwenの4 sample・2 tensor smokeから開始する。GPU/serviceには引き続き触れていない。
