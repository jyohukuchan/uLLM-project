# AQ mixed-precision importance score選定計画

## 前回の要点

- AQ4_0 mixed precisionは、SQ8 overlayよりcodebook indexを5/6 bitへ増やす方向を第一候補とした。
- 2026-07-02の`aq5_e4m3_g16_ts_flloyd32`はsampled weight MSEでUD Q5_Kを僅かに上回ったが、activation/model-level validationは未完了だった。
- 現行のactivation-second-moment weighted relative MSEでは、AQ all-g8が`0.002582475 @ 5.000000 bpp`、UD Q4_K_XL実物が`0.002364278 @ 5.206019 bpp`だった。
- UD GGUFのtensor別typeを外部teacher labelにし、Qwenだけでなく別architectureでもscoreを検証する方針になった。

## 今回の変更点

- `docs/plans/importance-score-algorithm-selection-plan-v0.1.md`を新規作成した。
- 現行second-moment法が、GPTQ型局所出力誤差の対角covariance近似と同じであることを整理した。独立した「対角GPTQ」候補として重複計上しない。
- 候補を次の役割へ分けた。
  - mandatory baseline: activation-second-moment weighted relative MSE
  - primary challengers: block-covariance/Hessian reconstruction、block-output perturbation
  - cheap diagnostics: AWQ salient activation、SmoothQuant型outlier/dynamic range
  - conditional candidate: quantization-aware Fisher/Taylor
  - causal oracle: single-tensor direct KL
  - deferred escalation: full GPTQ/OmniQuant、GAMMA/CoopQ型global interaction
- tensor順位用のrelative sensitivityと、allocation用の非正規化loss gain/追加byte utilityを分離した。非加法なrelative MSEをそのままknapsackへ入れない。
- v0.1がfreezeする対象をscore-method registryに限定した。low formatは`aq4_e4m3_g16_ts_flloyd16`、既存5-bit diagnosticは`aq5_e4m3_g16_ts_flloyd32`とし、storage semanticsを含む別manifestができるまでgain測定を開始しない。AQ6は未定義のままregistryへ入れない。
- UD typeを`{IQ4_XS,Q4_K} < Q5_K < Q6_K < Q8_0`の序数と、4.25/4.50/5.50/6.5625/8.50 packed bppの連続値へ分ける設計にした。
- standard Q4_K_M自体のfamily recipeを交絡させないため、same-revision static GGUFとの差分をprimary promotion labelとした。
- paired static baselineが無いfallbackはordinal探索だけに限定し、AUC/Precision admissionはHOLDとする契約にした。
- family内Spearman/Kendall、family-level、whole-model、ROC-AUC、PR-AUC、Precision@K、byte-matched ranking、layer-cluster bootstrap、不一致一覧の出力契約を決めた。
- QwenとGemmaの両方で相関gateを通ること、worst-model metricでwinnerを選ぶこと、Gemma結果を見て変更したら第三modelを要求することを明記した。
- Qwen UD typeは今回既に閲覧済みなのでdevelopment modelとし、未開封confirmatory claimはGemma lockbox以降に限定した。
- local確認ではQwen3.5-9B UD-Q4_K_XLに十分なtype variationがあった。一方、Gemma 4 31B UD-Q8_K_XLは量子化matrixがQ8_0へ揃っておりteacherに不適だった。
- Gemma lockboxにはBF16 source `unsloth/gemma-4-E4B-it`と、同一GGUF repo/revisionの`UD-Q4_K_XL`/`Q4_K_M`を推奨し、retune時の第三modelにはGemma 4 12B-it UD-Q4_K_XLを推奨した。
- 最終mixed artifactにはall-AQ4、byte-matched random、family-only controlを置き、relative-L2全行`<=1.0`、KLのpaired改善、relative NLL非劣性を満たすGO/HOLD/NO-GO gateを定義した。task評価は別の凍結manifestができるまでv0.1 gateから除外し、SQ8のruntime authorizationとfidelity NO-GOは別判定として扱う。
- quantizer、C++/HIP、GPU、systemd、本番設定、model downloadには触れていない。

## 次の行動

1. plan v0.1のscore式、metric gate、corpus splitをfreezeする。
2. Qwenのsame-revision Q4_K_M baselineを確保できるか確認し、paired UD label manifestを作る。
3. 別途承認された実験turnでGemma 4 E4B-it BF16 sourceとGGUF repoの対応revisionをpinし、UD-Q4_K_XL/Q4_K_Mを同じGGUF revisionから用意する。
4. Qwenでlow-cost score、block-covariance/block-output、KL subsetの順に測る。
5. scoreを変更せずGemma lockboxへ適用し、両modelのworst-case metricで本命を決める。
