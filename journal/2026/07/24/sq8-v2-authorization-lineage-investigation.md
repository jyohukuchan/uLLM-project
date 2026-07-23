# SQ8_0 v2 authorization/lineage investigation

## 前回の要点

独立フォーマット`SQ8_0`の本番昇格では、2026-07-12のcomplete campaignが
履歴worker `145a5351...b950`とlegacy起動方式に固定され、現行candidate
workerとmanifest起動方式の証跡にならないことが判明していた。現行profile
は`ullm.worker.v1`かつreasoningなしで、生成結果も
`ullm.served_model.v1`だった。ユーザーはv1例外ではなくv2で進めると決定
した。

## 今回の変更点

- `/etc/ullm/served-models/active.json`、AQ4 promotion runbook、profile、
  generator、Python/Rust loader、activation/bundle toolingを読み、
  AQ4_0本番の実際のv2経路を特定した。
- served-model v2の共通部分は、厳密な`ullm.served_model.v2` manifest、
  `ullm.worker.v2`、reasoning contract、promotion receipt hash、
  generic-reasoning release bundle、atomic activation/rollbackである。
  profile wrapper自体はAQ4 v2でも`ullm.served_model.profile.v1`のままで
  あり、SQ8向け`profile.v2`を新設する必要はない。
- 現在のAQ4 bytesは正式なcomplete bundle経路ではなく、同一model ID
  限定のdiffering-worker v2 bootstrapで稼働していることを確認した。
  そのsidecarは監査用でruntimeには読まれず、任意の新規pathを使えば再実行
  できるためscheme-wideなone-shot認可でもない。現行実装はmodel ID差を
  先に拒否するので、AQ4から独立SQ8へのcandidate-active切替にはそのまま
  使えない。
- `0cd6b9a0`から`6ad51ac5`までの「SQ8 authorization lineage v2」は、
  current mainと非連結のside historyにあるQwen3.5 AQ4_0の48個QKV/Z
  tensor向けSQ8 overlay専用実装だと確定した。AQ4 worker、overlay
  promotion/audit schema、固定履歴、request ID、48-tensor topologyを
  hard-codeしており、独立SQ8_0へ再利用してはならない。
- 現行generic bundle v1のidentity/rollback/独立validator設計とatomic
  activationは再利用できるが、exact 6-slot envelopeにはSQ8 full campaign
  の格納先がない。pre-receipt promotion evidenceへ事後campaignを結び付ける
  こともできないため、SQ8 campaign 3 referenceを追加したbundle v2と
  activation側の明示schema dispatchが必要である。現行bundle内promotion
  validatorはAQ4 schema専用で、candidate receiptとの直接cross-checkと
  browser evidence内のmanifest/worker identityも欠けている。
- SQ8 worker protocol parserはv2 reasoningを受理できるが、現行SQ8 serving
  runtimeは`reasoning_usage: None`を返すため、reasoning requestのrelease
  accountingを通せない。worker再ビルドとは別にreasoning state/accounting
  実装とCPU testが必要である。またv2 profileでもdecoderがv1 commandを
  明示compatibility modeなしに受理できるため、loaded schemaとの一致を
  強制する必要がある。
- Qwen3-14B-FP8 tokenizerのthinking tokenをread-only確認し、
  `<think>`=`151667`、`</think>`=`151668`を得た。これを用いた
  `qwen3-thinking-v1` contract案は、budget/forced-close/history/answer
  reservationを含むruntime test後に確定する。
- runbook
  `docs/plans/sq8-recovery-plan-v0.2-promotion-runbook-v0.1.md`へ、v1/v2
  比較、旧overlay lineageの切り分け、SQ8 serving promotion
  evidence/receipt構造、対応コード箇所、他served-modelとの互換条件、
  bundle v2、事前発行authorizationのatomic claim、campaign全体を包む
  cross-model temporary window、locked rollback、実装順、admission
  checklistを追記した。
- 実装コード、service、GPU、systemd、V620、active manifest、artifact、
  candidate、worker binaryには変更を加えていない。

## 次の行動

1. 別作業のworker再構築結果をreproducibility baselineとして受け取る。
   reasoning/auth変更後、同一final release commitからworker/build receiptを
   改めて作り、これだけを昇格identityにする。
2. Qwen3 reasoning contractとexact worker-schema enforcementをCPU上で
   実装・検証し、人間がdialect/budget semanticsとversioned v2 specsを
   承認する。
3. SQ8 serving promotion evidence/receipt tooling、no-clobber publication、
   generatorの厳密なcurrent-main AQ4/SQ8 dispatchを実装する。
4. generic reasoning/browser gateのprocess/identityをvalidated manifest
   由来にし、各stageで実`active.json` bytesをcandidateと比較する。
   SQ8 full campaign v2はcandidate copyを含めてend-to-endで束縛する。
5. generic bundle v1/AQ4を不変で保持し、SQ8 full campaign用3 slot、
   candidate receipt/browser identity cross-checkを持つbundle v2を追加する。
   独立再計算、AQ4回帰、mixed-schema拒否を確認する。
6. 人間がexact identity/run/output/expiry/max_attempts=1を持つauthorization
   を事前発行し、固定registryでatomic claimする仕組みを作る。campaign全体
   をactivation lockと`finally` restoreで包み、AQ4 reverse reconciliation
   とimmutable outcomeを私有copy上の全failure境界で検証する。
7. 認証修正を含む単一clean commitからfinal worker/candidateを凍結し、
   parent-onlyでfresh SQ8 full campaignとreasoning/browser campaignを
   実行する。AQ4 exact bytesへの復旧成功後にcomplete bundle v2を組み、
   最終昇格だけを`--release-bundle`で行う。
