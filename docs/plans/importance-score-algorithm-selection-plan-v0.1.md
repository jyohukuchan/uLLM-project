# AQ Mixed-Precision Importance-Score Algorithm Selection Plan v0.1

- Status: design frozen; AQ5 offline format amendment approved and formal CPU measurement in progress
- Date: 2026-07-21
- Scope: AQ4_0から5/6 bit codebook indexへ昇格させるtensorを選ぶ重要度スコアと、その選定実験
- Primary external teacher: Unsloth Dynamic (UD) GGUFのtensor別量子化type
- Causal reference: 単一tensor摂動によるforward KL divergence

## Goal

Qwen3.5-9B向けAQ4_0 mixed-precision化について、tensorごとに「AQ4のまま残すか、5/6 bitへ昇格させるか」を判断できる重要度スコアを選定する。

選定は次の二つを分離して行う。

1. `sensitivity`: 共通の低bit候補で量子化したとき、そのtensorがどれだけ壊れやすいか。
2. `allocation utility`: 高bit化で回復する誤差を、追加storage byteで割った値。

Unsloth UDの割当ては非公開アルゴリズムの「正解」ではなく、異なるcalibration、quantizer、runtime制約、容量予算を含んだ**外部teacher label**として扱う。UDとの一致だけで最終採用せず、単一tensor KLと最終mixed modelのfidelityでも因果的に確認する。

本計画の推奨構成は以下である。

- 現行のactivation-second-moment重み付きrelative MSEを必須baselineとする。
- 本命challengerは、現行法をchannel相関へ拡張するblock-covariance scoreと、非線形・残差経路まで含めるblock-output perturbation scoreとする。
- AWQ salienceとSmoothQuant型dynamic-range統計は、安価な独立diagnosticとして残す。
- Fisher/Taylor系は高コストな二次候補、単一tensor KLは最終oracleとして限定利用する。
- raw `|gradient * weight|`、full GPTQ/OBS、OmniQuant学習、Shapley/global allocationはv0.1の本命から外し、必要条件が成立した場合だけ昇格する。

## Success Criteria

### Reproducibility

- QwenとGemmaについて、GGUFのrepo、revision、filename、file SHA-256/LFS SHA、`gguf-dump` version、GGUF quantization version、取得できる場合はllama.cpp converter/imatrix revision、全tensorのname/shape/typeを固定したlabel manifestを残せる。
- score corpus、KL holdout、tokenizer/chat template、model revision、seed、sequence length、padding mask、uLLM git commitをmanifestへ記録できる。
- 各候補は、未開封Gemma UD labelとscoreをjoinする前に数式、正規化、集約、hyperparameterを凍結する。Qwen labelは既知なのでこの意味のholdoutには数えない。
- 量子化typeの序数、packed bpp、paired static baselineとの差分を別columnで保持し、相互に置き換えない。

### Statistical admission gate

候補を「モデル間で一貫して高い相関を示す本命」と呼ぶ最低条件を、Gemma lockboxを開く前に次のように固定する。

| Metric | Admission gate |
|---|---:|
| Paired static baseline | Qwen、Gemma両方でsame-cohort Q4_K_Mを取得し、eligible tensorのpaired label coverage 100% |
| UD teacher coverage | 各modelで`n >= 4`かつlabel非定数のrepeated familyが4以上、positive/negative混在familyが3以上 |
| `ordinal_ud`に対するfamily内layer方向のmacro Spearman `rho` | Qwen、Gemmaの両方で `>= 0.30` |
| `ordinal_ud`に対するfamily内layer方向のmacro Kendall `tau-b` | 両方で `>= 0.20` |
| cluster bootstrap | 両modelでprimary `rho` の95% CI下限が `> 0` |
| family方向の一貫性 | labelが非定数のfamilyの70%以上で相関符号が正。`n >= 16`のmajor familyに `tau-b < -0.20` がない |
| promoted/not-promoted `AUC_within` | 両方で `>= 0.65` |
| global Precision@K (`K = positive count`) | prevalenceを`p`として両方で`Precision@K >= p + 0.25*(1-p)` |
| KLとの順位相関 | stratified subset上で両方 `rho >= 0.30`、かつ現行baselineより `0.05`超悪化しない |

この値は自然定数ではなくv0.1の事前判定基準である。結果を見た後に変更する場合は、同じ実験の再解釈をせずplan versionを上げ、別の未開封modelを追加する。

KL gateのprimaryは、candidateのsensitivity `I_t`またはformat非依存`S_t`と、同じ`b_0`摂動の`L_C6(t,b_0)`との相関である。quantization-aware候補では、`G_t^*`とKL側の`b_0`からmanifest登録済みhigh formatへの最大回復量との相関もsecondaryで報告する。

### Winner rule

1. admission gateを全て満たす候補だけをfinalistとする。
2. finalist間では、まず `min(Qwen rho, Gemma rho)` が最大の候補を選ぶ。
3. 候補間の差は、同じlayer-cluster bootstrap replicate上でmetric差を取るpaired CIで判定する。個別CIの重なりを有意差判定に使わない。
4. worst-model `rho`差のpaired 95% CIが0を含む候補は統計的tieとし、worst-modelのKendall、ROC-AUC、Precision@K、KL相関の順で同じpaired比較を行う。
5. それでも同等なら、backward不要、保存統計量が小さい、測定時間が短い候補を選ぶ。
6. 単一scoreが勝つまでは、UD labelへfitした線形合成や重み探索を本命にしない。
7. どの候補もgateを満たさなければ明示的に`NO-GO`とし、現行baselineを暫定controlとして維持する。

production winnerの対象はC0-C5である。C6 direct KLはoracleなので、全tensor常用winnerではなく性能上限と限定rerankerとして扱う。C6だけがgateを通る場合は「cheap score選定成功」とせず、KL対象を固定budgetで絞れる二段構成のfeasibilityを別判定する。C7はPhase 6のescalationであり、このwinner比較へ入れない。

### Final policy gate

重要度スコアの選定後も、それだけでmixed-precisionを採用しない。untouched `D_final`上で、少なくとも次を同じstorage rounding、token mask、prompt順で比較する。

- `control-aq4`: 全eligible tensorをformat manifestのexact `b_0`にした基準。
- `control-random`: 同じ追加byte budgetを使うfamily-stratified random allocation、固定5 seeds。
- `control-family`: layer scoreを使わずfamily medianだけで割り当てるpolicy。
- `candidate-mixed`: winner scoreとformat manifest登録済みhigh candidateによるmixed allocation。

GO条件を次で固定する。

| Gate | GO condition |
|---|---|
| Artifact integrity | missing/non-finite row 0、対象tensor coverage 100%、実効bppがtarget以下かつ予定値との差 `<= 0.01 bpp` |
| Logits hard ceiling | 全fidelity rowでBF16に対する`relative-L2 <= 1.0` |
| KL improvement | `candidate-mixed - control-aq4`のtoken mean KL paired差について、prompt/conversation cluster bootstrap 95% CI上限 `< 0` |
| Tail/flip non-regression | p99 KL、mean/p99 relative-L2、top-1 flip rateが`control-aq4`以下 |
| Allocation value | mean KLが5 random controlsの全てと`control-family`より低い |
| PPL non-inferiority | `Delta_NLL_rel=(NLL_candidate-NLL_control)/NLL_control`のprompt-cluster paired bootstrap 95% CI上限が`<= 0.001` |

- hard ceiling、integrity、budgetのいずれかに失敗すれば`NO-GO`。
- hard gateは通るがKL improvementまたはallocation valueを証明できなければ`HOLD/INCONCLUSIVE`であり、GOへ読み替えない。
- 全条件を通った場合だけ次のproduction planでGO候補にする。

代表taskは現repoにtask名・dataset revision・shot/template・seed・metric aggregationまで凍結した共通harnessがないため、v0.1のGO gateへ曖昧な`0.5 point`条件を入れない。production promotionを主張する前に、別versionの`task-eval-manifest.json`でこれらとtask別non-inferiority margin、CI規則を固定してから評価する。task結果を見てmanifestを選ばない。

2026-07-16のSQ8 overlayについては、runtime materializationのauthorization状態とは別に、ユーザーが示したfidelity calibrationがrelative-L2 ceilingでNO-GOだった。このfidelity結果をruntime GOで上書きせず、上記gateを新規artifactで通過するまでは比較anchor以上に扱わない。

## Non-Goals

- AQ quantizer本体、C++/HIP kernel、GGUF writer、runtime dispatchをこの作業で実装しない。
- Unslothの非公開アルゴリズム、閾値、calibration corpusを推測して再現しない。
- UD割当てを唯一のground truthとみなさない。
- AQ6の最終format、group size、codebook、global bpp budgetをこのplanだけで決めない。AQ5は2026-07-21のユーザー直接承認により、下記format contractへ限定して確定した。
- GPU benchmark、`ullm-openai.service`、本番設定、model downloadをこの作業で実行しない。
- vision/audio tensorとtext backbone tensorを、対応関係を定義せず同じ母集団へ混ぜない。

## Observed Starting Point

### Existing AQ evidence

`docs/plans/aq-activation-aware-validation-v0.1.md`にある現行baselineは、linear入力channelのsecond momentを用いる。

\[
h_j = \mathbb{E}[x_j^2]
\]

\[
L_{\mathrm{diag}}(t,b)=
\frac{\sum_{i,j} h_j\left(W_{ij}-Q_b(W)_{ij}\right)^2}
     {\sum_{i,j} h_j W_{ij}^2+\epsilon}
\]

既存のQwen3.5-9B sampled resultは以下である。

| Artifact | weighted relative MSE | effective bpp |
|---|---:|---:|
| AQ g16 weighted scale+codebook | 0.004038034 | 4.500000 |
| AQ g8 weighted scale+codebook | 0.002821072 | 5.000000 |
| AQ all-g8 combined | 0.002582475 | 5.000000 |
| Unsloth Dynamic Q4_K_XL実物 | 0.002364278 | 5.206019 |

一方、8-promptの`linear_attn.out_proj` logits smokeではg16 weightedがg8 weightedより僅かに良く、local tensor proxyの改善がend-to-end logitsの改善を保証しないことも既に観測されている。このためUD labelとの一致とKL確認の両方が必要である。

更新済みin-proj collectorでは248 modules / 744 stat keys、24/24 weighted codebooks、fallback 0まで確認済みである。さらにfull-248 project-text lossのpolicy順位はbest-to-worstで`all-g16 > all-g8 > p4p6 > p4p46 > p4p65`となり、狭いtensor MSEが好むmixed policyよりall-g16が保守的に良かった。これは「高精度tensorを多く選べば必ずmodel lossが良くなる」という仮説への既存の負の証拠であり、C6/final artifact gateを省略しない理由になる。

2026-07-02の`aq5_e4m3_g16_ts_flloyd32`は、sampled weight MSEでUD Q5_Kを僅かに上回った。したがってmixed precisionの第一方向はSQ8 overlayではなく、16-entryから32/64-entryへcodebook index幅を増やす方向とする。ただし、この結果は重要度スコアの妥当性を証明しない。

### Local UD inventory checked on 2026-07-21

| Model artifact | Dump summary | Suitability |
|---|---|---|
| `/home/homelab1/datapool/ai_models/gguf/unsloth/Qwen3.5-9B-GGUF/Qwen3.5-9B-UD-Q4_K_XL.gguf` | 427 tensors: F16 48, F32 177, IQ4_XS 10, Q4_K 77, Q5_K 67, Q6_K 24, Q8_0 24 | primary UD teacherとして利用可能 |
| `/home/homelab1/datapool/ai_models/gguf/unsloth/gemma-4-31B-it-UD-Q8_K_XL.gguf` | 833 tensors: F16 30, F32 422, Q8_0 381 | quantized matrix内のprecision variationがなく、重要度teacherには不適 |
| `/home/homelab1/datapool/ai_models/gguf/unsloth/gemma-4-31B-it-GGUF/gemma-4-31B-it-Q4_K_M.gguf` | standard Q4_K_M | paired static reference候補だが、対応するUD-Q4がlocalにない |

他のlocal候補も確認した。`safetensors/gemma-4-E2B`はBF16のbase checkpointで、対応するlocal E2B-it UD-Q4がない。`gemma-4-31B-it-FP8-Dynamic`と`gemma-4-31B-it-MXFP4`は既に量子化済みでBF16 truthとして使えない。`translategemma-12b-it`にはBF16 checkpointとstatic Q4_K_Mがあるが、同checkpointのlocal UD GGUFがない。26B-A4B uncensored artifactもstatic Q4_K_Mだけである。従って、現状localだけで「BF16 source + mixed UD-Q4 teacher + paired static Q4」を満たすGemma setはない。

Qwenでは同一family内でもlayer別typeが変わり、`blk.0/1/2.ffn_down.weight`がQ6_K/Q5_K/Q6_Kとなる。`ssm_out.weight`は全24層でQ8_0であり、これはlayer内順位相関には寄与しない一方、family全体の昇格を検出できるかという別の強いtest caseになる。

### Existing tool reuse and gaps

| Tool | Reusable part | Before measurement, close this gap |
|---|---|---|
| `tools/collect-activation-stats.py` | Linear pre-hook、second moment、mean-abs、max-absのstreaming収集 | corpus content hash、model/revision、git commit、seed、padding mask、token countをmanifest化。covariance/quantileが必要な候補は別統計として明示 |
| `tools/run-aq-weighted-sample.py` | 現行weighted samplerのentry point | wrapper自身ではなく下位samplerのprovenanceとsplitを固定 |
| `tools/run-aq-tensor-sample.py` | candidate quantization、weighted metric、scale search、weighted Lloydの再利用 | full tensorをFP32 flattenしてからsampleするためsample capはmemory capではない。replacement sampling、discovery-order、fit/eval非分離を修正対象として記録 |
| `tools/export-aq-family-codebooks.py` | plan/family/tensor filterとfamily codebook export | family全tensorの`torch.cat` memory、旧12-family artifactとのcandidate filter不整合、fallback provenanceを検証 |

codebook/tensor scaleをFP32で評価した値と、最終保存dtypeへ丸めた値を混ぜない。重要度scoreは最終storage semanticsで再測定する。

## Working Hypotheses

1. **H1: 現行baselineは強いcontrolである。** `diag(E[xx^T])`で重み付けした誤差は、input channel間のcross termを無視したlinear layer局所出力MSEの対角近似であり、安価なscoreとして理論的に明確である。full局所出力MSEそのものではない。
2. **H2: 「対角GPTQ」は独立候補ではない。** GPTQの局所目的を対角化し、誤差補償を除くと現行second-moment法へ戻る。新情報はoff-diagonal covarianceまたはinverse-Hessian compensationからのみ得られる。
3. **H3: block covarianceは最有力challengerである。** channel相関を含めれば、現行統計を自然に拡張しつつfull end-to-end perturbationより大幅に安い。
4. **H4: block-output scoreは局所MSEとKLの橋渡しになる。** attention/MLP/SSM blockの非線形、gate、residualまで含めることで、tensor単体のlinear出力誤差が見落とす伝播を捉えられる。
5. **H5: AWQ salienceとdynamic rangeは単独winnerよりdiagnosticとして有用である。** candidate quantization誤差を見ないため、同じ入力を共有するQ/K/V等を十分区別できない可能性が高い。
6. **H6: FisherはKLの安価な二次近似になり得る。** ただしgradient corpus依存と計算量が大きく、forward-only候補が失敗した場合に価値が高い。
7. **H7: direct KLは最も目的に近いが、production scoreには高価すぎる。** stratified subsetのoracleまたは上位候補のrerankerとして使う。
8. **H8: Qwenだけでの改善は不十分である。** Qwenで調整したscoreがGemmaで符号反転・大幅低下するなら、architecture固有の局所最適と判定する。
9. **H9: UD labelはbudget-coupledである。** 各tensorを独立分類するだけでなく、同じ追加byte budgetでのrankingを評価しなければならない。

## Common Score Definitions

tensor `t`のweightを`W_t`、候補format `b`による実際の量子化後weightを`Q_b(W_t)`、摂動を次で定義する。

\[
\Delta W_t(b)=Q_b(W_t)-W_t
\]

linear入力のuncentered second momentを次で定義する。

\[
C_t=\mathbb{E}_{x\sim D_{\mathrm{stats}}}[xx^T]
\]

全てのquantization-aware候補は、同じ`Q_b(W_t)`、scale/codebook fitting split、storage roundingを使う。score algorithmごとに異なる量子化結果を作ると、重要度とquantizer品質を分離できないためである。

### Format contract status

このv0.1はC0-C7の**score-method registry**に加え、2026-07-21の承認済みamendmentとしてAQ4/AQ5のoffline measurement formatをfreezeする。AQ6は引き続き未確定であり、quantization-format registryへ入れない。測定開始前に別の`quantization-candidate-manifest.json`を作り、次を全て固定することをhard prerequisiteとする。

```text
candidate_id, index_bits, codebook_entries, group_size,
group_scale_encoding, tensor_scale_encoding, codebook_storage_dtype,
scale/codebook objective, family taxonomy, fit/eval split,
seed, iterations, rounding mode, serialized byte formula, implementation revision
```

確定候補は以下である。

| Field | `aq4_e4m3_g16_ts_flloyd16` (`b_0`) | `aq5_e4m3_g16_ts_flloyd32` (`B_high`) |
|---|---|---|
| index / codebook | 4 bit / 16 entry | 5 bit / 32 entry |
| group | contiguous 16 weights | AQ4と同一 |
| group scale | unsigned-positive finite E4M3-like tableへのuint8 index、1 byte/group | AQ4と同一 |
| tensor scale | little-endian BF16、1 scalar/tensor | AQ4と同一 |
| codebook storage/scope | little-endian BF16、`model x canonical_family`ごとに1個 | entry数以外AQ4と同一 |
| objective | `D_stats` activation-second-moment weighted MSE。quantile初期値から8回Lloyd、group scaleはnearest table index `+/-4`探索 | AQ4と同一 |
| fit/eval | seed 0、tensor名とgroup sizeをhashしたaffine group permutation。各tensorの先頭最大4,096 groupをfit、次の最大4,096 groupをdisjoint eval。replacementなし、両候補でgroup index共通 | AQ4と同一 |
| rounding | BF16 castはround-to-nearest-ties-to-even。codebook距離tieは低index、scale midpoint tieは高index | AQ4と同一 |
| serialized payload bytes | tensorごとに`ceil(4n/8)+ceil(n/16)+2`、familyごとに`16*2`。container metadata/alignmentを除く | tensorごとに`ceil(5n/8)+ceil(n/16)+2`、familyごとに`32*2`。container metadata/alignmentを除く |

AQ5はAQ4のcodebook indexを4 bitから5 bitへ拡張したものだけであり、group size、scale encoding、family taxonomy、fit/eval split、objectiveを変えない。5-bit indexはcontiguous LSB-first bitstreamとし、group境界paddingを入れない。2026-07-02のFP16-rounded sampled resultはseed evidenceに限り、今回freezeしたBF16 storage semanticsで再測定する。

具体的なmachine-readable contractは`quantization-candidate-manifest-v0.2`として各formal runの`quantization-candidate-manifest.json`へ保存し、`tools/run-aq-tensor-sample.py`、`tools/export-aq-family-codebooks.py`、`tools/aq_scale_formats.py`のSHA-256とworkspace commitを`implementation revision`へ記録する。本番runtimeがAQ5をload/executeできるという意味ではなく、この調査のCPU fake-quantization contractである。

- 6-bit candidate: 未定義。exact ID、64-entry fitting、storage semantics、実装revisionが揃うまで`B_high`へ入れない。
- `SQ8_0`: high-quality diagnostic anchorであり、codebook-index mixed-precision candidateではない。

これによりAQ4/AQ5のsensitivity、gain、byte utility測定を開始できる。Gemma lockbox後にformatを追加・変更した場合はscore formulaを変えていなくても新しいexperiment versionとして扱う。

全二次形式はFP64でaccumulateする。式中の数値guardは`epsilon = epsilon_energy = 1e-30`に固定し、各正規化denominatorはguard加算前にも`> 1e-30`をeligibility条件とする。追加byteは正の整数でなければcandidate pairをrejectし、byte denominatorへepsilonを足さない。

### Sensitivity, gain, and byte utility

共通のlow formatを`b_0 = aq4_e4m3_g16_ts_flloyd16`とする。`B_high`は上記manifestにある再現可能な5/6-bit candidateだけから作り、未定義AQ6やSQ8_0を含めない。

quantization-aware候補は二つの値を分けて返す。

- `A_t(b)`: normalization前のcandidate-specific loss surrogate。C0/C1ならtoken当たりの局所出力sum-squared error、C4ならblock出力sum-squared error、C5なら二次loss近似、C6ならtoken mean KLである。
- `L_t(b)`: tensor間のsensitivity順位用に正規化したscore。現行C0ではreference output energyで割ったrelative MSEである。

相対値`L_t`はtensorごとの分母が異なり加法的でないため、追加byte utilityやknapsackへ直接入れない。

\[
I_t=L_t(b_0)
\]

\[
g_t(b)=A_t(b_0)-A_t(b)
\]

\[
G_t(b)=\max(0,g_t(b))
\]

\[
U_t(b)=
\frac{G_t(b)}{\mathrm{bytes}_t(b)-\mathrm{bytes}_t(b_0)}
\]

promotion rankingの表示用heuristicとして次を定義する。

\[
G_t^*=\max_{b\in B_{high}}G_t(b),\qquad
U_t^*=\max_{b\in B_{high}}U_t(b)
\]

- UD ordinalとのprimary correlationには`I_t`を使う。
- 「baselineより昇格したtensor」のPrecision@Kにはquantization-aware候補では`G_t^*`を使う。
- `U_t^*`はtop-K表示用heuristicに限り、最大値を与えたformatも保存する。最終割当ては全`(tensor, format)` Pareto点を保持し、「一tensorにつき一format」のmultiple-choice knapsack/integer programで解く。
- C2/C3とC5aのformat非依存deletion proxyはcandidate formatを評価しないため、固有のscalar `S_t`をsensitivity/Precision@Kに使い、gain/byte utilityは`not applicable`とする。根拠なく`S_t`を追加byteで割らない。C5のうちquantization-Taylor/Fisherだけがcandidate-specific gainを持てる。
- `g_t < 0`はclip前のraw値を保存し、candidate formatのfit failureとして一覧化する。定義上`G_t`は負にならない。
- tensor sizeが異なるため、tensor-count Kとparameter/byte-budget Kを両方報告する。
- `A_t`も単一tensorの局所摂動を足し合わせる近似であり、block間interactionを保証しない。allocationはcandidateごとに別々に作り、異なる候補の`A_t`を混ぜず、最終mixed artifact gateで必ず検証する。

## Candidate Algorithms

### Comparison table

| ID | Candidate | Required data | Relative cost / GPU | Theory | uLLM difficulty | v0.1 role |
|---|---|---|---|---|---|---|
| C0 | activation-second-moment weighted relative MSE | forward activation `E[x_j^2]`、candidate `Delta W` | 1 calibration forward + offline O(P); GPU推奨だがCPU可 | diagonal `C_t`による局所linear出力誤差 | Low; 実装済み部分を再利用 | mandatory baseline |
| C1 | GPTQ型block-covariance / Hessian reconstruction | blockwise `E[xx^T]`、candidate `Delta W` | O(sum d block_size)保存とblock matmul。full inverseはO(d^2) memory/O(d^3) factorization; GPU推奨 | `Tr(Delta W C Delta W^T)`。GPTQ/OBQはfull inverseと誤差補償を使う | Medium for block covariance; High for compensated GPTQ | primary challenger |
| C2 | AWQ型activation-only salient channel | `E|x_j|`またはRMSのみ | 1 forward、O(d)統計; GPU不要だがmodel forwardはGPUが現実的 | 大きいactivation channelに結合したweight誤差が出力へ強く効く | Low | cheap independent diagnostic |
| C3 | SmoothQuant/OmniQuant型outlier・dynamic range | activation/weight channel max、RMS、quantile、clipping residual | pure statsはLow。learned clipping/transformは反復forward/backwardでMedium-High | activation/weight間のoutlier移動、clippingによるrange最適化 | Low for stats; High for learned OmniQuant | diagnostic; learned variant deferred |
| C4 | single-tensor block-output perturbation | block input cache、reference/candidate block output | tensor x formatごとのblock forward; GPU推奨 | nonlinear/gate/residualを含むlocal reconstruction | Medium | primary challenger / KL bridge |
| C5a | Taylor `|gradient * weight|` / OBD comparison | labeled or self-supervised loss、gradient/Hessian | backward必須、GPU推奨、gradient storage大 | parameter removalの一次/二次Taylor近似 | Medium-High | documented comparison; conditional ablation |
| C5b | quantization-aware Fisher/HAWQ/SqueezeLLM型 | `E[g_i^2]`またはHVP、candidate `Delta W` | backward/HVP、GPU、複数sample。full Fisher不要でもHigh | small perturbation時のKL二次近似 | High | conditional finalist |
| C6 | single-tensor direct KL | unseen prompts、BF16/candidate logits | O(num_tensor x num_format x full-model forward); GPU必須に近い | end-to-end output distributionを直接比較 | High | oracle / top-set reranker |
| C7 | GAMMA型learned hidden-state preference、CoopQ型interaction/Shapley | block states、反復最適化または組合せ摂動 | Very High | global budgetとtensor/layer interactionを直接扱う | Very High | v0.1 deferred; failure escalation |

### C0: activation-second-moment weighted relative MSE

\[
A_{C0}(t,b)=
\mathrm{Tr}(\Delta W_t C_{t,\mathrm{diag}}\Delta W_t^T)
\]

\[
L_{C0}(t,b)=
\frac{A_{C0}(t,b)}
     {\mathrm{Tr}(W_t C_{t,\mathrm{diag}}W_t^T)+\epsilon_{energy}}
\]

利点は、labelやbackwardが不要で、実際のgroup size、scale、codebook roundingによる`Delta W`を直接評価できることである。欠点はchannel相関、後段Jacobian、residual interactionを見ないことである。

これはAWQのactivation-awareな思想と整合するが、数式としてはlinear layer output reconstructionの対角二次形式である。GPTQのfull-Hessian/inverse-Hessian algorithmそのものではない。

### C1: GPTQ-like block-covariance score

full covarianceなら局所linear出力誤差は次である。

\[
A_{C1}(t,b)=
\mathrm{Tr}(\Delta W_t C_t\Delta W_t^T)
\]

\[
L_{C1}(t,b)=
\frac{A_{C1}(t,b)}
     {\mathrm{Tr}(W_t C_tW_t^T)+\epsilon_{energy}}
\]

v0.1ではinput channelを連続128 channel blockへ分割した`blockdiag(C_t)`をpre-registerする。64/256 blockはlabelを見て選ばず、score stabilityのsensitivity analysisとしてのみ使う。128でmemory上限を超えるtensorはstreaming block accumulateに落とし、対角へsilent fallbackしない。

GPTQ本体は`H=2XX^T`のinverse/Choleskyと逐次誤差補償を用いる。従って以下を区別する。

- `C1-block`: 既に得たcandidate `Delta W`をblock covarianceで採点する。本計画の本命。
- `C1-GPTQ`: inverse Hessianを用いてcandidate自体を再構成・補償し、その最終residualを採点する。quantizer変更を伴うためv0.1ではdeferred。
- `C1-diag`: C0と同一情報なので独立candidate数へ数えない。

### C2: AWQ-like activation-only salience

channel salienceを次で定義する。

\[
a_j=\mathbb{E}[|x_j|]
\]

量子化を一切介さないpre-registered scalarを二つだけ比較する。

\[
S_{\mathrm{AWQ-level}}(t)=\log(\mathrm{mean}_j(a_j)+\epsilon)
\]

\[
S_{\mathrm{AWQ-tail}}(t)=
\frac{\sum_{j\in\mathrm{top\ 1\%}}a_j}
     {\sum_j a_j+\epsilon}
\]

top 1%のchannel数は`k=max(1, ceil(0.01*d_in))`で固定する。`level`はactivation全体の大きさ、`tail`はsalient channelへの集中を表す。AWQの本来の手法はchannel scaleを探索してweightを保護するもので、上記tensor集約はuLLM用のproxyである。Q/K/Vのように同じ入力を共有するtensorは同じ値になるため、単独winnerにしにくい。この限界そのものを不一致分析で確認する。

### C3: outlier and dynamic-range score

SmoothQuantはactivationとweightのchannel maxを等価変換で再配分し、OmniQuantはblock reconstructionを用いてclippingと変換を学習する。重要度判定ではquantizerを学習せず、まずdimensionlessなrange severityを固定する。

\[
r_x(t)=
\frac{Q_{0.99,j}(\max_{token}|x_j|)}
     {Q_{0.50,j}(\sqrt{\mathbb{E}[x_j^2]})+\epsilon}
\]

\[
r_w(t)=
\frac{Q_{0.999}(|W_t|)}
     {\mathrm{RMS}(W_t)+\epsilon}
\]

\[
S_{\mathrm{range}}(t)=\frac{1}{2}\left(\log r_x(t)+\log r_w(t)\right)
\]

maxだけではrare sampleに不安定なため、quantile版をprimaryにし、true max版はdiagnosticとする。candidate-specific clipping lossも保存できるが、それはC0/C1のquantization-aware scoreへ分類する。

`S_range`はSmoothQuant/OmniQuantが公開したtensor重要度式ではなく、両手法のoutlier/dynamic-range仮説から本計画用に導いたpre-registered proxyである。

OmniQuant型learned clippingは重要度scoreだけを得るには高価で、algorithmとquantizerを同時に変えてしまうため初期比較から外す。

### C4: block-output perturbation

tensor `t`を含むTransformer/SSM blockを`F_l`、そのblockへのreference inputを`h_l`とする。

sequence `s`のvalid-token位置を`V_s`、総valid token数を`N_V=sum_s |V_s|`とする。attention文脈を保ったsequence単位でblockを実行し、paddingを除き、hidden dimensionはsum、token方向はmeanで固定する。

\[
A_{C4}(t,b)=
\frac{1}{N_V}\sum_s\sum_{v\in V_s}
\left\|\left[F_l(H_{l,s};W)-F_l(H_{l,s};W_{t\leftarrow Q_b})\right]_v\right\|_2^2
\]

\[
L_{C4}(t,b)=
\frac{A_{C4}(t,b)}
     {\frac{1}{N_V}\sum_s\sum_{v\in V_s}\left\|\left[F_l(H_{l,s};W)\right]_v\right\|_2^2+\epsilon_{energy}}
\]

attention、gate、normalization、residualまで通したblock出力を比較するが、後段全modelは実行しない。入力cacheはBF16 reference modelから一度だけ作り、candidate modelの途中状態を次blockへ連鎖させない。これによりtensor単独効果を保つ。

C4は256k-token全量をcacheせず、`D_stats`からname hashで事前抽出した16k-tokenの`D_block`を使う。cache/execution contractは次で固定する。

- 一度に保持するのは`one model x one layer x one shard`のBF16 block inputだけで、全layer cacheを作らない。
- 4 shardをstreamし、candidate outputは保存せずFP64のnumerator/denominator/countだけをaccumulateする。
- active host cache上限4 GiB、temporary disk上限8 GiBとする。超過時はmicrobatchを縮小し、token sampleやprecisionをsilentに変更しない。
- attention mask、position information、valid-token maskをreference forwardと同じものに固定する。

GAMMAのteacher-forced hidden-state reconstructionは、この方向をglobal budget optimizationへ拡張した近年の例である。v0.1ではscoreの独立性と実装量を優先し、learned preferenceやinteger programは導入しない。

### C5: gradient, Taylor, OBD, Fisher, and HAWQ

Taylor pruning文献の代表的なparameter scoreには`(g_i w_i)^2`がある。本計画ではユーザー指定の`|gradient * weight|`も独立したL1 proxyとして、sampleごとの符号を潰してから集約する。

\[
A_{|gw|}(t)=
\mathbb{E}_{s}\left[\sum_{i\in t}|g_{s,i} w_i|\right],\qquad
L_{|gw|}(t)=\frac{A_{|gw|}(t)}{n_t}
\]

これは本計画の`L1 Taylor deletion proxy`であり、文献のsquared scoreそのものではない。weightを0にする重要度で、rounding perturbationを直接測らないため、本命ではなくC5へescalateしたときのablationとする。squared variant`E_s[sum_i(g_{s,i}w_i)^2]/n_t`も同じgradientからsecondaryで出す。

量子化に合わせた一次scoreは、gradientをsample平均してから絶対値を取らず、sampleごとの内積絶対値を平均する。

\[
A_{\mathrm{Taylor-quant}}(t,b)=
\mathbb{E}_s\left[\left|\langle g_{s,t},\Delta W_t(b)\rangle\right|\right],\qquad
L_{\mathrm{Taylor-quant}}(t,b)=\frac{A_{\mathrm{Taylor-quant}}(t,b)}{n_t}
\]

OBD/Fisher型の対角二次scoreは次である。

\[
A_{\mathrm{Fisher}}(t,b)=
\frac{1}{2}\sum_{i\in t}\mathbb{E}[g_i^2]\Delta w_i(b)^2,\qquad
L_{\mathrm{Fisher}}(t,b)=\frac{A_{\mathrm{Fisher}}(t,b)}{n_t}
\]

OBDはこの対角二次形で`Delta w=-w`と置いた削除salience`A_OBD(t)=0.5*sum_i H_ii*w_i^2`に対応する。Fisher estimatorは混同せず次を分ける。

- `self-Fisher` primary: 各`x`で`y ~ p_theta(.|x)`をsampleし、`grad log p_theta(y|x)`の二乗を平均する。small-perturbation KLとの理論接続はこちらに限る。
- `empirical Fisher` secondary: corpusの実next-tokenに対するcausal-LM gradient二乗を平均する。domain/task lossへの感度であり、self-Fisherと同一視しない。

HAWQ-V2のHessian traceはlayer平均曲率を測るが、tensor内のどこに量子化誤差が生じるかを捨てるため、`trace(H)/n * ||Delta W||^2`はablationに留める。

FisherはKLの二次近似として有望であり、Optimal Formatsはこの関係からtensor別bit allocationを導出している。ただしbackwardとper-parameter accumulatorの負担が大きい。C0/C1/C4がadmission gateを満たさない、またはKLとの不一致が系統的な場合だけfull evaluationへ進める。

### C6: single-tensor direct KL

\[
A_{C6}(t,b)=L_{C6}(t,b)=
\mathbb{E}_{x\sim D_{\mathrm{KL}}}
D_{\mathrm{KL}}\left(
p_{\mathrm{BF16}}(\cdot|x)\;\|\;
p_{t\leftarrow Q_b}(\cdot|x)
\right)
\]

C6は全tensorに共通のmodel-output KLをvalid token当たりで平均した値で、tensor固有のenergy denominatorを持たない。このためC6に限り、global loss unitの`A_C6`をそのまま順位score`L_C6`として使う。

- temperatureはprimaryで`T=1`に固定する。
- prompt部分のpaddingと、必要ならprompt tokenをmaskし、評価対象token数を明記する。
- KLはfull vocabularyをstreaming accumulateし、top-k近似をprimary値に使わない。
- token meanに加え、p99 KL、top-1 flip rate、reference top-1 probability dropを保存する。
- 同時に一つのtensorだけをfake-quantizeし、残りはBF16 referenceとする。
- 全tensorには高価なため、二つの集合を分ける。
  - `KL-core`: score値とUD type値を入力にせず、tensor name hashとfamily/layer/shapeだけで層化して固定した10-15%の確率sample。KL順位相関gateはこの集合だけで計算する。
  - `KL-audit`: 各scoreのtop/bottomと全不一致tensorのunion。原因分析には使うが、inferential correlationへ混ぜない。
- `KL-core`で層ごとの抽出率が異なる場合、model全体値はinverse-probability weightingを併記する。

これはUDが掲げるflip/KL目的に最も近い。一方、単一tensor効果は複数tensorを同時量子化したinteractionを含まない。最後にfull mixed artifactを必ず測る。

### C7: interaction-aware methods

CoopQはlayerをcooperative gameとしてShapley近似を取り、pair interactionを含むbudget allocationを解く。GAMMAはteacher-forced hidden-state reconstructionからprecision preferenceを学習し、整数計画で容量を満たす。どちらも「独立tensor scoreだけではinteractionを落とす」というriskへの有力な回答である。

ただしv0.1の目的は、まず再利用可能なtensor scoreを二つのarchitectureで選ぶことである。C7は次の場合だけPhase 6へ昇格する。

- C6単一tensor KLはUDと合うが、複数tensor mixed artifactだけが大きく崩れる。
- 不一致が同一block内のQ/K/V、gate/up/down、SSM in/outの組として集中する。
- 追加容量を揃えてもmultiple-choice allocationが一貫してfinal-model optimumを外し、pair interactionが疑われる。

## Calibration and Measurement Protocol

### Dataset split

Wikipedia単独は使わない。raw exampleを固定したうえで、chat、code、日本語を含むmultilingual、reasoning/math、一般文を混ぜる。

| Split | Suggested size | Purpose | Leakage rule |
|---|---:|---|---|
| `D_stats` | 256k tokens/model | activation、covariance、range score | score定義のfitには使えるがUD labelでsampleを選ばない |
| `D_block` | 16k tokens/model | C4 block-output perturbation | `D_stats`からname hashで事前抽出、4固定shard |
| `D_fisher` | 16k tokens/model | gradient/Fisher候補 | `D_stats`から独立、4 shardで安定性確認 |
| `D_KL` | 8k tokens/model | single-tensor exact KL | score式と対象tensor stratificationを凍結後に開く |
| `D_final` | 32k tokens/model | final mixed artifact KL/PPL | algorithm選定には使用しない |

同じraw example集合をmodel間で共有し、各model固有tokenizerと公式chat templateでtokenizeする。raw example hash、model別token count、truncation、sequence length別内訳を保存する。QwenとGemmaでtoken数が違うため、単純なtoken index対応を仮定しない。

各splitを4個のstratified shardへ分け、score rankのseed/shard stabilityを報告する。1.5M token超のUnsloth corpusを再現できないことは明記し、UD一致の上限要因として扱う。

### Eligibility and family taxonomy

core analysisへ入れる条件は以下である。

- 2-D weight matrixで、format manifestのlow/high candidateとしてstorage可能である。
- GGUF nameとBF16 tensor nameを一意に対応できる。
- candidate quantizerが同じstorage semanticsで評価できる。

以下を別stratumへ分ける。

- `token_embd.weight`、`output.weight`: layer反復がないためglobal-only。
- norm、bias、state、position/rope、non-matrix F16/F32: coreから除外。
- vision/audio/projector: text backboneと分離。
- Qwen固有SSM/linear-attention family、Gemma固有family: architecture-specific secondary analysis。

canonical recordは少なくとも次を持つ。

```text
model_id, architecture, layer_id, canonical_family, gguf_name, hf_name,
shape, n_params, qtype_ud, qtype_static, ordinal_ud, ordinal_static,
packed_bpp_ud, packed_bpp_static, promotion_delta_ordinal,
promotion_delta_bpp, promoted, eligible, exclusion_reason
```

name mappingはregexだけに任せず、unmatched、duplicate、shape mismatchをfatalにする。common-family cross-model analysisとarchitecture-specific analysisを別tableで出す。

## Extracting UD Tensor Labels

### Read-only extraction

local commandのJSON schemaでは`.tensors`がnameからmetadataへのobjectである。抽出例は次である。

```bash
/home/homelab1/hf_venv/bin/gguf-dump --json MODEL.gguf \
  | jq -r '.tensors | to_entries[] | [.key, .value.type, (.value.shape | @json)] | @tsv'
```

type count確認例:

```bash
/home/homelab1/hf_venv/bin/gguf-dump --json MODEL.gguf \
  | jq '.tensors | to_entries | group_by(.value.type) |
        map({type: .[0].value.type, count: length})'
```

`general.file_type`はmixed GGUF全体を一つの代表typeで表すだけで、tensor labelとして使わない。`general.quantized_by=Unsloth`やimatrix metadataもprovenance補助であり、UD variantを一意に証明しない。filename、配布repo、pinned revision、file hashをprimary provenanceにする。

### Ordinal and continuous targets

llama.cpp block layoutから得るnominal packed bppは次である。alignment、GGUF metadata、tokenizerは含めない。

| GGUF type | Block payload | packed bpp | Primary ordinal |
|---|---:|---:|---:|
| IQ4_XS | 136 bytes / 256 weights | 4.2500 | 0 |
| Q4_K | 144 / 256 | 4.5000 | 0 |
| Q5_K | 176 / 256 | 5.5000 | 1 |
| Q6_K | 210 / 256 | 6.5625 | 2 |
| Q8_0 | 34 / 32 | 8.5000 | 3 |
| F16 | 2 / 1 | 16.0000 | core quantized ordinalから原則除外; eligible escape analysisでは4 |
| F32 | 4 / 1 | 32.0000 | core quantized ordinalから原則除外; eligible escape analysisでは5 |

実験manifestではこのtableを参照したllama.cpp source commitを固定し、floating `master`をprovenanceにしない。

IQ4_XSとQ4_Kは圧縮率もquantization familyも異なり、qualityの全順序を保証できない。primary ordinalではtieにし、packed bpp連続値では差を残す。F16/F32はnorm/state/unsupported shape等のconverter制約で残る場合が多いためblanketに「最重要」としない。BF16 name mapping、2-D shape、block alignment、AQ対応を全て満たすものだけを`eligible_high_precision_escape`としてsecondary分析する。次の三つを並行して報告する。

1. `ordinal_ud`: coreは`{IQ4_XS,Q4_K}=0 < Q5_K < Q6_K < Q8_0`。eligible escapeだけF16/F32を4/5として別reportにする。
2. `packed_bpp_ud`: 上表の連続値。
3. `empirical_format_loss`: 同じBF16 tensorを各GGUF typeで再量子化できる場合のC0/KL実測値。これはlabelではなくmapping sensitivity analysisである。

shapeがblock sizeで割り切れない場合はnominal bppをblindly使わず、actual encoded payloadまたはconverter fallbackを記録する。

### Defining a positive promotion label

標準`Q4_K_M`自体も、familyやlayerによってQ5_K/Q6_Kを使うrecipeを持つ。従ってprimary positiveは単純な`UD type > Q4`ではなく、同一model・同一revisionのstatic baselineとのpaired ordinal差分で定義する。

\[
\Delta \mathrm{ordinal}^{UD}_t =
\mathrm{ordinal}(q^{UD}_t)-\mathrm{ordinal}(q^{static}_t)
\]

\[
\Delta \mathrm{bpp}^{UD}_t =
\mathrm{bpp}(q^{UD}_t)-\mathrm{bpp}(q^{static}_t)
\]

```text
promoted = promotion_delta_ordinal > 0
unchanged_tier = promotion_delta_ordinal == 0
demoted  = promotion_delta_ordinal < 0
lateral_format_change = promotion_delta_ordinal == 0 and promotion_delta_bpp != 0
```

`promotion_delta_bpp`は追加容量とbyte-matched evaluationに使う。例えばIQ4_XSからQ4_Kへの変更はpacked bppが増えてもprimary ordinalでは同tierなので、precision promotion positiveには数えない。

paired artifactは、同じbase-weight revisionと同じ変換cohortで作られたことをmetadata、repo history、imatrix hashで確認する。単に似たfilenameまたは同じarchitectureであるだけではpairにしない。同revision/cohortのstatic GGUFが得られない場合だけ、fallback labelを用いる。

```text
promoted_vs_4bit_floor = qtype_ud in {Q5_K, Q6_K, Q8_0}
```

fallback結果をpaired結果と同じtableへ混ぜない。fallbackは`ordinal_ud`相関のexploratory reportだけを許し、AUC/Precision@K admission gateを満たす代替にはしない。現在local Qwen directoryにはUD-Q4_K_XLしかないため、最終admission判定には将来同repo/revisionのstandard Q4_K_Mを追加するか、対応するllama.cpp converter recipe/versionを固定して再構成する必要がある。paired labelがQwenまたはGemmaの一方でも欠ける場合、判定は`HOLD: binary teacher incomplete`である。

## Statistical Validation Design

### 1. Within-family layer ranking: primary

各`model x canonical_family`について、layer方向にcandidate sensitivity `I_t`または`S_t`と`ordinal_ud`のSpearman `rho`、Kendall `tau-b`を計算する。これを唯一のprimary rank targetとする。`promotion_delta_ordinal`と`promotion_delta_bpp`に対する相関はpaired artifactがある場合のsecondary analysisであり、gateの`rho/tau-b`には使わない。

- `n < 4`またはUD labelが全てtieのfamilyはcorrelation undefinedとしてcoverageへ出し、0で埋めない。
- familyを等重みした算術macro meanをprimaryとする。小さいfamilyでは`rho=+/-1`が出やすいため、Fisher z平均は`rho`を`[-0.999,0.999]`へclipしたsensitivity analysisに留める。
- tensor数、parameter数で重み付けしたmicro値はsecondaryとする。
- `ssm_out`のような全層Q8 familyはfamily-level testへ回す。
- tieが多いのでKendallは`tau-b`を使う。

これにより、`ffn_down`というfamily自体が高精度になりやすい効果を除き、「同じfamilyのどのlayerを上げるか」を検証できる。

### 2. Family-level and whole-model ranking

family効果を完全に消すと、Qwen `ssm_out`全層Q8のような重要signalも消える。従って三つを分ける。

1. **Within-family macro primary:** 前節のfamily別`ordinal_ud`相関を等重み平均した値。Success Criteriaとwinner ruleの`rho/tau-b`は常にこれを指す。
2. **Family-level secondary:** familyごとのmedian scoreとmedian ordinal/promotion rateの順位相関。
3. **Residualized whole-model secondary:** model/family内midrank percentileを連結したSpearman/Kendall。
4. **Raw whole-model descriptive:** 全eligible tensorのSpearman/Kendall。shape、family、type recipeに強く交絡するため採否のprimaryにはしない。

residualized whole-model値は、scoreとlabelをそれぞれfamily内で`(midrank - 0.5) / n_family`へ変換してから連結して計算する。別のresidual定義へ結果後に切り替えない。modelをpoolする場合はこの変換を`model x family`内で行う。

### 3. Binary retrieval metrics

paired baselineの`promoted`をpositiveとして以下を報告する。

binary ranking scoreはquantization-aware候補では`G_t^*`、format非依存候補では`S_t`に固定する。`I_t`によるbinary値はsecondaryであり、admission AUC/Precision@Kへ都合よく切り替えない。

- positive/negativeの両方を持つfamilyごとにROC-AUCを計算し、等重みmacro平均した`AUC_within`をprimaryとする。pooled版を出す場合もmodel/family内score rankを用い、family識別だけで高くならないようにする。
- raw scoreの`AUC_global`をsecondaryとし、`ssm_out`のようなfamily-wide promotionを含む全体選択能力を測る。
- class imbalanceを可視化するPR-AUC。
- `K = actual promoted tensor count`でのPrecision@K、Recall@K、NDCG@K。
- actual UD追加byteと同じbudgetになるまで`U_t^*`順に選ぶbyte-matched Precision/Recall。C2/C3はformat-specific gainがないためこのmetricを欠測扱いにする。
- parameter-weighted版とtensor-count版。
- familyごとのPrecision@Kとmacro平均。

Precision@K gateはrandom prevalenceから到達可能な最大1.0までのgapを25%以上閉じる条件`P@K >= p + 0.25*(1-p)`とする。単純な`1.5x lift`は`p > 2/3`で理論上達成不能になるため使わない。raw liftもdescriptiveには残す。

admission gateのROC-AUCはmacro `AUC_within`を指す。positive/negativeが混在するfamilyが3未満で`AUC_within`のcoverageを満たさないmodelはscore failureではなくteacher variant不成立とし、別UD-Q4 variantへ切り替える。`AUC_global`で代用しない。

ROC-AUCだけでは「限られた5/6 bit枠の上位を当てる」能力を測れないため、Precision@Kを必須にする。逆にPrecision@KだけではKの選び方に依存するためAUCと併記する。

### 4. Uncertainty and multiplicity

- Transformer/SSM layerをclusterとして10,000回bootstrapし、95% percentile CIを出す。同一layerのQ/K/V等を独立sampleとみなさない。
- nullはmodelごとにlayer IDの共通permutationを10,000回生成し、全repeated familyのlabelへ同じlayer permutationを適用する。これによりfamily marginalと同一layer内Q/K/V等のcross-family共変動を保つ。familyごとの独立permutationは禁止し、欠損layerを持つfamilyは共通layer intersectionで計算する。
- candidateが複数あるため、探索的p-valueにはBenjamini-Hochberg correctionをかける。
- effect sizeとCIをprimaryにし、p-valueだけで選ばない。
- 4 corpus shard、sequence length、seed間のrank stabilityをSpearmanとtop-K Jaccardで報告する。

### 5. Disagreement analysis

次を必ず`disagreements.tsv`へ出す。

- score上位だがUDで昇格していないtensor。
- score下位だがUDで昇格したtensor。
- scoreとUDのfamily内rank residualが上位/下位5%のtensor。
- C0とC1/C4/C5/C6で順位が大きく割れたtensor。
- all-layer同一UD typeだがscore分散が大きいfamily。

最低限のcolumnは次とする。

```text
model, layer, family, gguf_name, shape, n_params,
ud_type, static_type, promotion_delta_ordinal, promotion_delta_bpp,
score_raw, score_family_rank, ud_family_rank, rank_residual,
activation_rms, activation_tail, range_score,
diag_mse, block_cov_mse, block_output_mse, fisher, kl, flip_rate,
qualitative_class, notes
```

定性classは、(a) name/type mapping、(b) static recipe、(c) family-wide rule、(d) calibration domain、(e) shared-input ambiguity、(f) residual/nonlinear propagation、(g) paired-tensor interaction、(h) runtime/kernel constraint、(i) quantizer fit failure、(j) unknown、を使う。

## Multi-Model Validation Design

### Qwen discovery model

Qwen3.5-9Bはlocal UD-Q4_K_XLがあり、同family内layer variationも十分あるためdiscovery modelとする。ここで以下だけを許す。

このplan作成時点でQwenのtype countと複数の具体例は既に閲覧済みである。従ってQwenを未開封holdoutまたはconfirmatory evidenceとは呼ばず、taxonomy、実装、候補screen用のexploratory/development modelとしてのみ扱う。v0.1の事前登録性はGemma lockbox以降に対して成立する。

- tensor name canonicalizationの修正。
- score implementation bugの修正。
- 事前登録した候補の単独比較。
- C1 block size等のrobustness確認。ただし最良値をGemmaへ持ち込まずprimary=128を維持する。

Qwen labelにfitしたfeature weight、family-specific threshold、manual exception listはGemmaへ持ち込まない。

### Gemma lockbox recommendation

手元のGemma 4 31B UD-Q8_K_XLは量子化matrixがQ8_0へ揃っており、layerwise teacher labelにならない。新規に用意する第一候補は次である。

- BF16 source: [`unsloth/gemma-4-E4B-it`](https://huggingface.co/unsloth/gemma-4-E4B-it)
- UD teacher: [`unsloth/gemma-4-E4B-it-GGUF`](https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF/tree/main) の `gemma-4-E4B-it-UD-Q4_K_XL.gguf`
- Paired baseline: UD teacherと同じGGUF repo/revisionの`gemma-4-E4B-it-Q4_K_M.gguf`
- 2026-07-21時点のUD-Q4 file表示サイズ: 約5.13 GB。model card上8Bで、local Qwen UDの約5.97 GBと実験規模が近い。

取得時は`main`を使い続けず、BF16 sourceとGGUF repoのcommit revision、LFS SHA、config/tokenizer revisionをmanifestへ固定する。GGUFがどのBF16 revisionから変換されたか一致を確認できなければ、weight-level score比較は実行しない。

E4Bを推す理由は次である。

- Qwenとは異なるGemma architectureで、1-model過学習を検出できる。
- 31Bよりdownload/storage/calibration負担が小さく、Qwen9Bと同程度の検証規模になる。
- UD-Q4_K_XLが公開され、Q4/Q5/Q6/Q8のlayerwise variationを期待できる。実際のtype countは取得後にdumpし、variationがなければlockbox不成立とする。

Gemmaはmultimodal familyを含むため、primary transferはtext backboneのcommon familyと、各architecture内のrepeated text familyで行う。vision/audio/projectorはsecondary reportへ分ける。

### Third-model confirmation

Gemma結果を見てscore式、threshold、feature weightを一度でも変更した場合、Gemmaはlockboxではなくなる。その場合は以下を未開封confirmatory modelにする。

1. 推奨: [`unsloth/gemma-4-12b-it-GGUF`](https://huggingface.co/unsloth/gemma-4-12b-it-GGUF) の`UD-Q4_K_XL`とpaired `Q4_K_M`。UD fileは2026-07-21時点で約7.37 GB。
2. Scale stress: [`unsloth/gemma-4-31B-it-GGUF`](https://huggingface.co/unsloth/gemma-4-31B-it-GGUF) の`UD-Q4_K_XL`、約18.8 GB。local Q4_K_Mと同一revisionでpairにできる場合のみ優先度が上がる。
3. Smoke用途のみ: Gemma 4 E2B-it UD-Q4_K_XL、約3.18 GB。安いがQwen9Bとのscale差が大きく、唯一のconfirmatory modelにはしない。

### Anti-overfitting protocol

1. score-method registry、quantization-format manifest、数式、hyperparameter、metric gate、corpus splitのhashをQwen score測定前にcommitする。
2. Qwenでdebugと探索を行う。
3. Gemma scoreを先に生成してhashし、その後UD labelとjoinする。labelを見ながらformulaを直さない。
4. Qwenで選んだcandidateをGemmaで一回評価し、worst-model ruleで判定する。
5. reverse transferはdescriptiveとして、Gemmaで得た順位をQwenへ適用する。ただしGemmaを用いたretuning後の値をholdout performanceとは呼ばない。
6. tensorをrandom splitしない。同じmodelのlayer/tensorは強く依存するため、split単位はmodel全体、追加時はarchitecture family全体とする。
7. learned ensembleは少なくとも3 modelが揃い、leave-one-model-out evaluationが可能になるまで禁止する。
8. common-familyだけの結果と、architecture-specific familyを含む結果を両方出し、片方だけ都合よく選ばない。

## Phase Breakdown

### Phase 0: Freeze the experiment contract

- このplanのscore-method registryとadmission gateをcommitする。
- `D_stats/D_block/D_fisher/D_KL/D_final`のraw corpus manifestを固定する。
- `quantization-candidate-manifest.json`でexact low/high format、storage dtype/rounding、fit/eval split、implementation revisionを固定する。未定義AQ6を名前だけで登録しない。
- Qwen/Gemma BF16とGGUF provenance schemaを固定する。

Exit criteria:

- Qwenが既知のdevelopment labelであることをmanifestへ明記し、Gemma label-score join前にscore数式とprimary metricを再現できる。
- Qwen/Gemmaで同じraw corpus、別tokenizerを使うルールが固定されている。
- low formatと少なくとも一つのhigh formatについて、同じ`Q_b(W)`を再現できるmanifest hashとstorage byte式がある。なければsensitivity screenだけに限定し、gain/allocation phaseへ進まない。

### Phase 1: Build and audit UD labels

- Qwen UD dumpからtensor manifestを生成する。
- 同revision static Q4_K_Mとのpaired labelを作る。得られなければfallbackを明示する。
- Gemma E4B-it UD-Q4_K_XLを取得後、type variationをread-only dumpで確認する。
- BF16/GGUF name mapping、shape、eligibilityを100% auditする。

Exit criteria:

- eligible tensorにunmatched/duplicateがない。
- type count、packed bpp、promotion count、extra-byte budgetが再計算可能である。
- QwenとGemmaの両方でsame-cohort paired static baselineが得られる。得られなければordinal探索は続行できるがadmission判定はHOLDとする。
- Gemmaに少なくとも複数のquantized typeと、label非定数familyがある。なければ12Bまたは31B UD-Q4へ切り替える。

### Phase 2: Low-cost score screen

- C0、C2、C3を同じ`D_stats`から計算する。
- 4 shardのrank stabilityを測る。
- Qwen UDと比較し、実装bugとtaxonomyだけを修正する。
- score式の追加・feature weight fitは行わない。

Exit criteria:

- 各scoreが全eligible tensorをcoverageする。
- baseline数値が既存artifactから合理的範囲で再現する。
- instabilityの高い統計量を特定する。

### Phase 3: Correlation-aware and causal screen

- C1 block covarianceを測る。
- C4 block-output perturbationを測る。
- score/UD type値に依存しないname-hash samplingで固定した`KL-core`と、top/bottom/disagreement用`KL-audit`を分けてC6 direct KLを測る。
- C0/C1/C4とKLの不一致を分析する。
- 必要条件を満たす場合だけC5b self-Fisherを追加し、同じgradient passからC5a L1/squared Taylor、quantization-Taylor、OBD/HAWQ ablationも出す。C5へescalateしない場合、C5aは文献比較のみで未測定と明記する。

Fisher escalation condition:

- C0/C1/C4のいずれもKL `rho >= 0.30`を満たさない、または
- C1とC4の順位が系統的に割れ、後段曲率が原因と疑われる、または
- UDとは合うがdirect KLと反対方向になる。

### Phase 4: Qwen statistical report

- within-family、family-level、whole-modelの三層で相関を出す。
- ROC-AUC、PR-AUC、Precision@K、byte-matched rankingを出す。
- cluster bootstrap、permutation、multiplicity correctionを行う。
- disagreement listとqualitative classを確定する。
- candidate式をfreezeし、Gemma lockbox manifestへ署名/hashを残す。

### Phase 5: Gemma lockbox transfer

- Qwenでfreezeしたcandidateを変更せずGemma E4Bへ適用する。
- common familyとGemma-specific familyを分離して同じmetricを出す。
- worst-model ruleでfinalistを決める。
- 結果を見て変更した場合はversionを上げ、Gemma 12Bを第三lockboxにする。

### Phase 6: Policy-level validation

- finalist scoreからformat manifestに登録済みのlow/high candidateだけでgain curveを作る。
- UDと同じ追加byte budget、およびuLLMのtarget bpp budgetでallocationを作る。
- 全`(tensor, format)` Pareto点から、一tensor一format制約付きmultiple-choice knapsack/integer allocationをprimary policyとして解く。`U_t(b)` greedyは速度用diagnosticに留める。
- final mixed modelの`D_final` KL/logits relative-L2/PPLを測る。taskは別途manifestがfreeze済みの場合だけsecondaryで測る。
- interaction failureが残る場合のみC7へ進む。

## Decision Tree

```text
exact b0と少なくとも一つのhigh-format manifestが再現可能か?
├─ No  -> ordinal sensitivity探索だけ可。gain/allocationはHOLD
└─ Yes
   |
   +-- UD-Q4 GGUFにtensor type variationがあるか?
   |   ├─ No  -> teacherとして不採用。別size/variantへ切替
   |   └─ Yes -> label auditへ
   |
   +-- 同cohort static Q4_K_Mとpairにできるか?
   |   ├─ Yes -> promotion_delta_ordinalをprimary binary label、delta_bppをbudgetに使う
   |   └─ No  -> ordinal探索だけ実行可。admissionはHOLDし、paired artifactを用意
   |
   +-- C0/C2/C3はQwenで安定するか?
   |   ├─ No  -> corpus/provenance/masking/quantizer fitを修正
   |   └─ Yes -> C1/C4とKL subsetへ進む
   |
   +-- C0/C1/C4のいずれかがKL gateを通るか?
   |   ├─ No  -> C5 Fisherを追加
   |   └─ Yes -> forward-only finalistをfreeze
   |
   +-- freezeした候補がQwenとGemmaの両方でadmission gateを通るか?
   |   ├─ No  -> NO-GO。単一model最適化を採用しない
   |   └─ Yes -> worst-model scoreでwinner選定
   |
   +-- final mixed artifactもfidelity gateを通るか?
       ├─ Yes -> mixed-precision policy候補として次planへ
       └─ No
          ├─ interactionがblock内に集中 -> C7/paired perturbationへ
          └─ 広域に分散 -> score/corpus仮説を棄却しplan versionを上げる
```

## Risks

| Risk | Consequence | Mitigation |
|---|---|---|
| `AQ4/AQ5/AQ6` aliasだけでformatを指定 | `Delta W`とbyte costを再現できない | exact candidate manifestをPhase 0 hard gateにし、未定義AQ6を除外 |
| UDをground truthと誤認 | 非公開corpusやruntime都合を学習する | teacher labelと呼び、KL/final artifactを独立gateにする |
| GGUF typeのqualityをbpp順と誤認 | IQ4_XS/Q4_K等を誤序数化 | primary ordinalでtie、packed bppとempirical lossを別分析 |
| standard Q4_K_M recipeの既存promotion | UD固有promotionを過大評価 | same-base/same-conversion-cohort paired static GGUFをprimary baselineにする |
| family/shape confounding | whole-model相関が見かけ上高くなる | within-family primary、family-level、raw globalを分離 |
| tensorを独立標本とみなす | CIが過度に狭くなる | layer-cluster bootstrap、全family共通layer permutation |
| Qwen labelへのmanual overfit | Gemmaで崩壊 | formula freeze、Gemma lockbox、retune時は第三model |
| calibration domain/length shift | score順位が不安定 | multi-domain corpus、4 shard、length別rank stability |
| Gemma multimodal tensorの混入 | architecture差でなくmodality差を測る | text/common family primary、projector別stratum |
| local Q8 Gemmaをteacherに使う | labelが定数で相関不能 | E4B/12B/31BのUD-Q4を別途用意 |
| single-tensor scoreがinteractionを無視 | final mixed modelだけ悪化 | block-output、paired disagreement test、final artifact gate、必要時C7 |
| quantizer fitとscoreを同時変更 | algorithm効果を識別不能 | 全scoreで同一`Delta W`を共有 |
| FP32評価と保存dtypeの差 | offline scoreがruntime artifactを表さない | final storage rounding後に再測定 |
| existing samplerのfull-tensor materialization | 大modelでmemory事故 | 実装時はstreaming/shape audit。silent samplingをmemory guaranteeとしない |
| direct KL/Fisherの計算量 | 全tensor評価が現実的でない | stratified subset、top/disagreement優先、段階的escalation |
| 最新研究の未再現 | paper上の利得を過信 | GAMMA/CoopQ/Optimal Formatsは一次資料の仮説として扱い、対象modelで再検証 |

## Expected Artifacts for the Future Measurement Run

この設計作業では作成しないが、実験時の出力契約を次とする。

```text
benchmarks/results/<YYYY-MM-DD>/aq/importance-score/<run-id>/
  experiment-manifest.json
  quantization-candidate-manifest.json
  corpus-manifest.json
  ud-tensor-labels.tsv
  tensor-name-map.tsv
  eligibility-audit.tsv
  scores.parquet
  metrics-by-family.tsv
  metrics-by-model.json
  bootstrap-samples.parquet
  disagreements.tsv
  kl-subset.tsv
  final-report.md
```

`run-id`にはmodel、source revision、corpus hash、score-method registry version、quantization-candidate manifest hashを含める。raw model pathだけをprovenanceにしない。

## Primary References

- Existing uLLM plans: `docs/plans/aq-activation-aware-validation-v0.1.md`, `docs/plans/aq-full-quantizer-design-v0.1.md`, `docs/research/quantization-method-survey-2026-07-01.md`
- [GPTQ](https://arxiv.org/abs/2210.17323): layer reconstruction、full Hessian inverse、error compensation
- [AWQ](https://arxiv.org/abs/2306.00978): activationによるsalient weight channel保護
- [SmoothQuant](https://arxiv.org/abs/2211.10438): activation/weight間のoutlier migration
- [OmniQuant](https://arxiv.org/abs/2308.13137): blockwise learned clipping/equivalent transformation
- [Taylor importance](https://arxiv.org/abs/1906.10771), [Optimal Brain Damage](https://proceedings.neurips.cc/paper_files/paper/1989/file/6c9882bbac1c7093bd25041881277658-Paper.pdf), [Optimal Brain Surgeon](https://proceedings.neurips.cc/paper/1992/hash/303ed4c69846ab36c2904d3ba8573050-Abstract.html), [SparseGPT](https://proceedings.mlr.press/v202/frantar23a/frantar23a.pdf): first/second-order salienceとlarge-model reconstruction
- [HAWQ-V2](https://arxiv.org/abs/1911.03852), [SqueezeLLM](https://proceedings.mlr.press/v235/kim24f.html), [Optimal Formats](https://arxiv.org/abs/2505.12988): Hessian/Fisherとmixed precision
- [GAMMA](https://arxiv.org/abs/2605.18475), [CoopQ](https://arxiv.org/abs/2509.15455): budget-aware/global interaction-aware mixed precision
- [Unsloth Dynamic 2.0 documentation](https://unsloth.ai/docs/basics/unsloth-dynamic-2.0-ggufs), [original Dynamic 4-bit analysis](https://unsloth.ai/blog/dynamic-4bit): KL/flip目標、large mixed-domain calibration、model/layer-specific allocation
- [llama.cpp tensor encoding schemes](https://github.com/ggml-org/llama.cpp/wiki/Tensor-Encoding-Schemes), [current block structures](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-common.h): GGUF type payload/bpp

## Next Actions

1. このv0.1をscore-method registryと判定基準としてfreezeし、別途exact quantization candidate manifestを作る。
2. Qwen3.5-9Bについてsame-revision Q4_K_M baselineの入手可否を確認し、UD label manifestを作る。
3. Gemma 4 E4B-itのBF16 sourceとGGUF repoを、それぞれ対応関係を確認したrevisionでpinする。UD-Q4_K_XLとQ4_K_Mは同じGGUF repo revisionに揃える。downloadは別途承認された実験turnで行う。
4. corpus splitとprovenance schemaを先に確定し、既存activation collectorの不足項目を実装planへ分離する。
5. QwenでC0/C2/C3、次にC1/C4、最後にKL subsetを測る。
6. scoreをfreezeしてGemma lockboxを一回だけ評価する。
7. admission gateを通過した場合だけ、format manifestへ登録済みの5/6-bit候補を使うallocationとfinal mixed artifact検証の次planを作る。
