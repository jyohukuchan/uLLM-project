# AQ4 P2 production profile parser fixes v0.1

## 前回の要点

- P2 detailed profile window 6本（`profile-prefill-n128-m1`、`profile-prefill-n1024-m128`、`profile-prefill-n2048-m64`、`profile-prefill-n3584-m128`、`profile-decode-c16`、`profile-decode-c3584`）はrocprofv3自体は正常に実行・記録が完了していたが、後段の`tools/parse-aq4-p2-production-profile.py`がすべて`exit 73`（profile parse failure）で終わっており、P3の候補優先順位付けに使えるカーネル単位の内訳データが未取得だった。

## 今回の変更点

CPU-only・GPU再取得なしで2件のバグを修正し、既に取得済みの生rocprofトレースを再解析した。

1. **カーネルfamily分類の曖昧一致バグ**（codex terra-maxへ委任、独立検証済み、commit `4c49d13b`）: `FAMILIES`辞書の`normalization`の素の`rope`/`add`/`silu.*mul`パターンが、`attention`/`aq4_projection`の融合カーネル名（`qwen35_qk_norm_rope_paged_kv_write`、`aq4_matvec_add`等）と二重一致していた。専用`paged_validation` familyの追加とnegative lookaheadで解消。既存の6window全実カーネル名31種で分類先が一意になることを独立に再検証した。
2. **HIP APIトレースの列名不一致**: 実機のrocprofv3が出力する列名が`Function`で、`parse_api()`が受理する既存候補（`Name`/`API_Name`/`ApiName`/`name`）に含まれていなかった。`Function`を追加。値がHIP API名（`hipModuleLaunchKernel`等）であることを確認済み。
3. **行数上限(`MAX_ROWS`)の較正**: 旧値500,000は実測に対して小さすぎた。6window中5windowの実測ピーク（`prefill-n128-m1`のHIP APIトレース1,316,514行）に余裕を持たせて2,000,000へ引き上げ。`MAX_TRACE_BYTES`(256MB)/`MAX_PROFILE_TOTAL_BYTES`(512MB)は5windowの実測（最大204MB）に対し既に十分だったため変更していない。

## decode-c3584の詳細profileは意図的に断念

`decode-c3584`（M=1で3584トークンを1トークンずつprefillし、KV状態を確立してからdecode 64トークン）は他5windowの30〜150倍の規模（kernel_trace 12,629,052行/2.08GB、hip_api_trace 36,604,722行/5.85GB、合計5.85GB）。この規模に合わせて上限を引き上げると、他5windowに対する安全境界としての意味がほぼ失われる。

検討した代替案:
- 記録の先頭だけ取得: 逆効果。1回の実行は「prefill（巨大、繰り返し）→decode（本来知りたい部分）」の順であり、先頭切り詰めは欲しいdecode部分を丸ごと落とす。
- 間引き取得: 均等間引きでは依然としてprefill由来行が支配的で意味が薄い。既に完全な生トレースがディスク上にあるため、`executor-trace.jsonl`の実測タイムスタンプと突き合わせてdecode区間だけを抽出する後処理は技術的に筋が良いが、実装コストが相応にかかる。
- 断念: 今回はこれを採用。理由はP3（prefill候補選定）に`decode-c3584`のデータは不要で、`decode-c16`が同じM=1設計でdecode側のfamily構成を既に示しているため。decodeの最適化（P5）に実際に着手する際、必要なら上記の区間抽出方式を検討する。

ユーザー判断: 「一旦cでいいから、それで進めて」（c = 断念して進める）。

## Profile結果（P3候補優先順位付けの根拠）

4つのprefill profile windowすべてで一貫した傾向:

| window | aq4_projection | attention | recurrent | 残り(norm/embed/head/paged_validation/runtime) |
|---|---|---|---|---|
| prefill-n128-m1 | 81.4% | 3.6% | 8.7% | 6.3% |
| prefill-n1024-m128 | 90.3% | 3.5% | 5.8% | 0.4% |
| prefill-n2048-m64 | 86.7% | 7.0% | 5.7% | 0.6% |
| prefill-n3584-m128 | 83.0% | 11.3% | 5.4% | 0.3% |

さらに、各windowの測定10回分の`end_to_end_ms`合計とrocprofの`gpu_union_ns`（12回分）を比較すると、GPU busy時間が壁時計時間の90〜99%を占め、D2H転送やstream同期待ちによるアイドル時間はごく小さい（候補1「D2H/同期削減」の余地は薄いことを示唆）。

結論: **P3候補は#2（AQ4 BM8/registerカーネルのshape coverage・tail・scale metadata residency改善）を最優先とする**。理由は(a) 全prompt長で一貫してGPU busy時間の81〜90%を占める最大の単一要因であること、(b) ワークロードが実際にcompute-boundであり（idle/sync待ちがほぼ無い）、カーネル実装そのものの改善が直接効くこと。`attention`の比率がprompt長とともに増加する（3.6%→11.3%）点は長文脈でのcandidate #4（recurrent/self-attention chunk execution）が次点候補になりうることを示すが、現時点ではcandidate #2が明確に最優先。

## 実行していないこと

- decode-c3584の詳細profile取得。GPU/systemd/sudoを要する操作は一切行っていない（既存の生トレースへのオフライン再解析のみ）。

## 次の行動

1. 現在のAQ4 BM8/registerカーネル実装（shape coverage、tail処理、scale metadata residency）を精査し、P3-Aレーンとして具体的な改善余地を特定する。
2. P3の選抜Gate（shape/dtype/finite/greedy token一致、p50 5%超・p95 10%超回帰の禁止等）に従い、候補実装・CPU oracle比較・独立validationの順で進める。
