# Qwen importance score Phase 0--2 CPU pilot

## 前回の要点

- `importance-score-algorithm-selection-plan-v0.1.md`は、UD GGUFのtensor別typeをexternal teacher labelにしてC0--C7を比較する実験契約をfreezeした。
- primaryは`ordinal_ud`へのmodel x canonical_family内layer方向 Spearman/Kendallであり、same-cohort `Q4_K_M`が無い場合の`promoted_vs_4bit_floor`はordinal探索専用で、AUC/Precision@K admissionには使わない。
- Qwenはdevelopment model、Gemmaは未開封lockboxであり、今回Gemmaを実行しない制約だった。

## 今回の変更点

- score-method registryとadmission gateを`49fceeeb`でcommitした。AQ6はregistryへ追加していない。
- run rootに`quantization-candidate-manifest.json`と`corpus-manifest.json`を作成した。AQ4は現行FP32 sampled evaluatorの情報だけを記録し、storage rounding、serialized byte formula、fit/eval splitなど未確定項目を`unknown`にした。AQ5は現行実装で再現不能のため全contractを`unknown`にし、両candidateをgain/allocationから除外した。
- immutable candidate manifestはrun-idへhash済みのpre-screen contractである。実行したcollector/sampler/summarizer/mergerのsource SHA-256は`experiment-manifest.json`へ別記録し、future bootstrapはsource hashを動的に採るよう修正した。これはAQ4をfinal-storage candidateへ昇格させるものではない。
- `gguf-dump --json`でQwen3.5-9B UD-Q4_K_XLの427 GGUF tensorを取得し、BF16との対応を監査した。text cohortは427/427対応、core eligibleは200。unmatched/duplicateは0、`ssm_conv1d` 24件のrank-3 shape mismatchとvision/MTP 348件のscope exclusionを`eligibility-audit.tsv`にfatalとして全件残した。
- local `/home/homelab1/datapool/ai_models`にはsame-cohort Qwen3.5-9B `Q4_K_M`が無かった。ダウンロードはせず、paired analysisとAUC/Precision@KはHOLDにした。
- GPUなし・service/systemd非接触で、既存32 promptを4 shardへ固定してCPUでD_stats pilotを実行した。32 sample、3,416 valid token、248 module、FP64 second moment/mean_absを得た。これはformal 256k mixed-domain corpusではない。
- C0は各eligible tensorで65,536要素sample、C2はcollected activation moments、C3は同数のdeterministic no-replacement weight sampleで算出した。C0 200/200行はfiniteだった。
- `ordinal_ud`へのwithin-family macro（defined family 5）の結果は次の通り。すべてQwen-only provisional値である。

| score | rho | tau-b | layer-cluster bootstrap rho 95% CI |
|---|---:|---:|---:|
| `C0_L` | 0.0825 | 0.0812 | [-0.2657, 0.2735] |
| `S_AWQ_level` | 0.3807 | 0.3209 | [0.0026, 0.6355] |
| `S_AWQ_tail` | -0.1287 | -0.1172 | [-0.3832, 0.2742] |
| `S_range` | 0.1844 | 0.1501 | [-0.1426, 0.5028] |

- fixed common-layer 10,000 replicate bootstrapを実行した。`S_AWQ_level`のrho下限は正だがtau-b下限は-0.0025であり、Gemma/paired baseline/formal corpusも無いのでadmission判定は行わない。
- C1/C4/C5/C6は未実行。C1用D_block covariance、C4のblock perturbation、C5 backward、C6 KL-core/KL-auditは今回のCPU-only pilotの範囲外とした。

## 次の行動

1. human decision: Qwen同revision・same conversion cohortのstandard `Q4_K_M`を取得してよいか判断する。local UD fileは5.6 GiBなので、static artifactもおおむね同程度の数GiB規模と見込むが、取得元revisionと実サイズをpinしてからにする。
2. human decision: formal `D_stats/D_block/D_fisher/D_KL/D_final`のraw example mix（chat/code/Japanese/multilingual/reasoning/math/general）とrecord-level hash/splitをfreezeする。現pilotはその代替ではない。
3. AQ4のfinal storage rounding/bytes/fit-eval splitと、AQ5の現行reproducerを確定する。確定前は`I_t`のみの予備screenに留め、`G_t`、`U_t`、allocationを開始しない。
4. paired staticとformal corpusが揃った後にC1/C4、fixed KL-core/KL-audit、必要時C5を実行する。Gemma lockboxはQwen側の式・設定を変更せず、別の承認turnでのみ開く。
