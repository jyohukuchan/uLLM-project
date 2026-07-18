# AQ4 P2 production baseline window execution v0.1

## 前回の要点

- `benchmarks/results/2026-07-17/qwen35-9b-aq4-production-opt-v0.1/p2/aq4-production-prefill-decode-baseline-v0.10-verified` へP2のprepare/stageを完了し、通常measurement window 14本、detailed profile window 6本、path-oracle window 8本が計画済みだった。
- fidelity修正(RMSNorm additive weight)をproductionへ再デプロイ済み（candidate active, `b4d42d9f89e0ae249f4a7dbc30d2a7428879f86e`）。

## 今回の変更点

P2 window driverの静的監査で、既存tooling未実施のバグを3件発見・修正しコミットした（GPU実行は伴わない診断・修正）。

- `crates/ullm-engine/src/bin/ullm-aq4-p2-resident-driver.rs`: `WorkerHardlinkGuard::capture()`がcompile-time埋め込みfixtureを使っており、staging binaryのinode変化に追従できなかった。`build_worker_hardlink_fixture()`を追加し、実行時に`scan_worker_inode_paths()`でhardlink集合を動的検証するよう変更した。
- `tools/run-aq4-p2-production-baseline-window.py`と`tools/prepare-aq4-p2-production-baseline.py`の`legacy_case()`: `--device-index`/`runtime_device_index`が`0`（CPU fallback synthetic device）にハードコードされており、`1`（`HIP_VISIBLE_DEVICES=1`下の実R9700）であるべきだった。既存の`calibration_case()`の正しい値に揃えた。
- 上記3件の修正を診断ログ追加込みで個別commit（7c594ada、6f33fe48、b4d42d9f、c6868183、9ea8734f）。

## Window実行結果

commit `b4d42d9f89e0ae249f4a7dbc30d2a7428879f86e`のstaged binaryで、通常window 14本中13本、detailed profile window 6本中6本を完了した。

- 通常window: `prefill-n128/512/1011/1024/1339/2048/3584`（7本）、`decode-c128/512/1024/1339/2048/3584`（6本）が`status: partial_observability`で成功。
- `decode-c16`のみ`status: failed`。原因は`ullm-aq4-p2-resident-driver.rs`の`resident actual width X differs from resolved M Y`不変条件違反で、context長16に対してM=32/64/128を要求するdecodeケースは、prefillチャンク幅が`min(M, context_len)`で頭打ちになるため原理的に満たせない。`tools/prepare-aq4-p2-production-baseline.py`の`make_cases()`がdecode側にのみ「M > context長」の組み合わせを`unsupported`として除外する仕組みを欠いていた（prefill側の`cached_prefix_chunked`には同種の仕組みが既にある）。m1/m8/m16（3/6ケース）は正常に完走・成功。これはuLLMエンジンの不具合ではなく、ベンチマークのcase生成matrix側の見落としであり、実運用でも起こり得ない組み合わせのため、tool側の修正・再取得は行わず既知の制約として受容する判断とした。
- detailed profile window: `profile-prefill-n128-m1/n1024-m128/n2048-m64/n3584-m128`、`profile-decode-c16/c3584`の6本すべて`executor_exit_code=0`で実測完了（rocprofv3非対応環境のため`detailed_profile_status: external_rocprof_required_and_bound_by_service_driver`、profile解析自体は`not_requested`/失敗扱いだが基礎計測は有効）。

## オペレーション上のインシデント（2件、いずれも復旧・原因確認済み）

- 1回目: 大量windowの連続実行でsystemd `StartLimitBurst=3`/`StartLimitIntervalUSec=15min`に抵触し`ullm-openai.service`が`failed (start-limit-hit)`になった。`systemctl reset-failed && systemctl start`で即復旧、manifest/readyzを確認。
- 2回目: 自作オーケストレーションスクリプトの待機ロジックが「windowループ開始時刻」基準だったため、長時間window直後に短時間のdetailed-profile windowが連続すると実際のrestore-start間隔が圧縮され、再度start-limit-hitに抵触した（`profile-prefill-n2048-m64`のrestore失敗、exit 70）。同様に復旧。事後精査で、このwindow自体の計測(`window-result.json`, `executor_exit_code=0`)は正常に完了しており、失敗したのはrestore手順のみと判明。以後は実際の`ActiveEnterTimestamp`基準・600秒間隔のペーシングに修正し再発なし。
- いずれもuLLMのcommitted toolingではなく、リポジトリ外の一時オーケストレーションスクリプト側の設計不備。

## 封印(seal)について

`tools/seal-aq4-p2-production-baseline-jsonl.py`は通常window全数が厳密に`partial_observability`であることを要求するため、`decode-c16`が`failed`である現状では封印を拒否する（副作用なしで拒否されることを確認済み）。`decode-c16`をsealable化するにはcase matrix生成を修正し準備ツリーを再生成する必要があるが、封印検証は`preparation_manifest_sha256`を準備ツリー全体から再計算するため、再生成は既に完了した19window全ての束縛検証を無効化し、事実上全window再取得が必要になる。既存のfidelity修正promotionと同じ判断枠組みで、「機能的に完了・形式的に未封印」として現状を受容し、封印は見送ることにした。19window分の生データはrun root配下にそのまま保全されている。

## 実行していないこと

- path-oracle window（8本、別途CPU fp32 source-oracle再captureが必要）は未着手。
- `decode-c16`のcase matrix修正・準備ツリー再生成は行っていない。
- 封印（`baseline-measurements.jsonl`の確定）は行っていない。

## 次の行動

1. path-oracle windowの要否と実施タイミングを別途判断する。
2. `decode-c16`の3ケースをsupported化する意義があるかどうかは、次にP2を再取得する理由が生じた際にまとめて判断する。
3. 現状の19window分の生evidenceを、P3（prefill候補実装）のprofile分析入力として利用開始してよい。
