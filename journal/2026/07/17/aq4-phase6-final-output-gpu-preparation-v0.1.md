# AQ4 Phase 6 final-output GPU preparation v0.1

## 前回の要点

- Phase 5（`e992b3ea`,`1ed64022`）で、Qwen3.5 AQ4最終RMSNormの`1 + weight`適用漏れを修正した。CPU-only chainではfinal RMSNorm相対L2が`0.5010330688`から`0.1688691127`へ、LM-head 34-row sampleが`0.5860500940`から`0.0582126936`へ改善し、layer 0とlayer 0--11に回帰はなかった。
- 07/14の`0.6151289249`は通常Gateway requestではなく、service停止済みR9700で実行した3 row M=1/cold診断の最終logit比較値である。Phase 6では同一fixture・同一座標・同一比較式で、この修正後の値を一回だけ取り直す必要がある。

## 今回の変更点

### `0.6151289249`を出した経路の確定

- `journal/2026/07/14/qwen35-aq4-p2-path-oracle.md`、`path-oracle-v2`、`source-oracle-v2`、`qwen35_aq4_p2_oracle.compare_payloads`を照合した。
- 比較対象は**`ullm-aq4-p2-path-oracle`**である。`source-oracle-v2`と`path-oracle-v2`の3 recordを現在の比較器で再計算し、`logit_sample_bounded_relative_l2_max=0.6151289249025698`を再現した。
- 定義は、各rowで保存されたsource/AQ4 logit sampleの共通座標についてrelative L2を計算し、3 rowの最大を取るものだ。full-vocabulary指標ではない。hiddenの同じbounded metricは`0.5452883336042509`である。
- `ullm-aq4-differential-trace`はlayer別差分を取得する同一M=1診断の別toolである。07/14 attempt3の先頭32 logit sampleでは`0.6212985932808415`となり、`0.6151289249`を直接比較する経路ではない。そのためPhase 6の唯一のGPU model runには採用しなかった。

### Phase 6専用tooling

- commit `d3ea48d543456a07a2796ee804671c3da513c268`で、nlink=1 content-copy staging、bounded final-output比較器、single-use service window driver、static testsを追加した。
- `tools/stage-aq4-phase6-path-oracle-binary.py`はCargo outputのhardlink状態を許容しつつ、新規staging copyをmode `0555` / `nlink=1`で固定する。receiptと`SHA256SUMS`はmode `0444` / `nlink=1`である。
- `tools/compare-aq4-phase6-final-output.py`は07/14 baselineを`0.6151289249025698`へexact再計算してから、修正後path oracleとside-by-side比較する。`after_fix`のmetric、absolute差、改善率をcreate-new artifactへ保存する。これは正式P2 gateではないことも明記する。
- `tools/run-aq4-phase6-service-window.sh`は、Phase 3c v0.7の17 guard、既存lockのnon-creating probe、R9700-only HIP+ASIC guard、`RuntimeDirectoryPreserve=yes`、nlink=1 staging、single stop/startを再利用する。path-oracleの元のactive production contractに合わせ、17 guardが30 active production guardのsubsetであることをassertし、実行childには完全な30 guardを`env -i`で与える。
- window driverのsource binary contractは、clean source worktree外の固定`uLLM-phase6-build-target/release/ullm-aq4-p2-path-oracle`を明示的に受理する形へ整合させた。これにより、runbookの正確なhandoff commandと「targetをworktree外に置いてclean contractを維持する」設計が一致する。他のbinary pathはstop前に拒否する。
- さらにdriverが生成するwindow artifactのいずれかが既に存在する場合は、stop前に再実行を拒否する。これによりpre-stop failureを含め、同一`OUT`でのevidence上書きや無承認のsecond windowを防ぐ。rehearsal、baseline、stagingの既存evidenceだけはこの判定から除外する。
- runbookを`docs/plans/aq4-phase6-gpu-window-runbook-v0.1.md`として追加し、旧Phase 3c v0.7 evidence/runbookを上書きしなかった。

### CPU-only準備と検証

- `d3ea48d5`のclean detached worktreeを`/home/homelab1/coding-local/ultimateLLM/uLLM-phase6-clean-source`に作成した。targetをworktree外の`/home/homelab1/coding-local/ultimateLLM/uLLM-phase6-build-target`に置き、worktreeがcleanであることを維持した。
- `CARGO_BUILD_JOBS=1`で`ullm-aq4-p2-path-oracle`のrelease buildに成功した。既知のC++ `subobject-linkage` warning以外のfailureはなかった。
- host-only HIP guardをbuildしただけで、実行していない。staged path-oracleのSHA-256は`774964446f3fbfe10323242e67f3aeb95f8f34d42e84db2e46763ff782e452de`、mode `0555`、nlink `1`である。guard binaryのSHA-256は`e85043b1bc1812a1b0ebcba31fcfa0bff5402be348d713a37f44643d9885175d`、mode `0755`である。
- staging `SHA256SUMS`はstaging directory内で`sha256sum -c SHA256SUMS`を実行して全件`OK`を確認した。最初に親directoryから相対名のchecksumを呼んだためread-only verifyが失敗したが、stagingの内容は変更せず、正しいdirectoryで再検証して成功した。今後はrunbookの`cd .../path-oracle-binary-staging`形式を使う。
- `baseline-metric-reproduction/comparison.json`は3 row / 同一cases SHAでbaseline `0.6151289249025698`をexactに再現した。
- `pytest -q tests/test_stage_aq4_phase6_path_oracle_binary.py tests/test_aq4_phase6_service_window_driver.py tests/test_qwen35_aq4_path_oracle.py`は`14 passed, 15 subtests passed`。`bash -n`、`py_compile`、`git diff --check`も成功した。
- GPU、service、systemd、`/run/ullm/r9700.lock`、V620、07/16 P3 harnessのroot/artifact/environment/`rocprof`には触れていない。root-only guardも未実行である。

## 親エージェントへのhandoff

### root-only guard rehearsal

service稼働中に下の3 commandを順番に一回ずつ実行する。各`attempt-*`が`r9700-guard-rehearsal-summary.json`で`status=valid`でなければ、final windowへ進まない。

```bash
sudo /usr/bin/python3 /home/homelab1/coding-local/ultimateLLM/uLLM-project/tools/run-aq4-phase3c-r9700-guard.py --output /home/homelab1/coding-local/ultimateLLM/uLLM-project/benchmarks/results/2026-07-17/qwen35-9b-aq4-production-opt-v0.1/p2/aq4-phase6-gpu-final-output-v0.1/guard-chain-rehearsal-v0.1/attempt-1 --guard-bin /home/homelab1/coding-local/ultimateLLM/uLLM-project/benchmarks/results/2026-07-17/qwen35-9b-aq4-production-opt-v0.1/p2/aq4-phase6-gpu-final-output-v0.1/query-hip-device-identity --health-phase rehearsal-1
sudo /usr/bin/python3 /home/homelab1/coding-local/ultimateLLM/uLLM-project/tools/run-aq4-phase3c-r9700-guard.py --output /home/homelab1/coding-local/ultimateLLM/uLLM-project/benchmarks/results/2026-07-17/qwen35-9b-aq4-production-opt-v0.1/p2/aq4-phase6-gpu-final-output-v0.1/guard-chain-rehearsal-v0.1/attempt-2 --guard-bin /home/homelab1/coding-local/ultimateLLM/uLLM-project/benchmarks/results/2026-07-17/qwen35-9b-aq4-production-opt-v0.1/p2/aq4-phase6-gpu-final-output-v0.1/query-hip-device-identity --health-phase rehearsal-2
sudo /usr/bin/python3 /home/homelab1/coding-local/ultimateLLM/uLLM-project/tools/run-aq4-phase3c-r9700-guard.py --output /home/homelab1/coding-local/ultimateLLM/uLLM-project/benchmarks/results/2026-07-17/qwen35-9b-aq4-production-opt-v0.1/p2/aq4-phase6-gpu-final-output-v0.1/guard-chain-rehearsal-v0.1/attempt-3 --guard-bin /home/homelab1/coding-local/ultimateLLM/uLLM-project/benchmarks/results/2026-07-17/qwen35-9b-aq4-production-opt-v0.1/p2/aq4-phase6-gpu-final-output-v0.1/query-hip-device-identity --health-phase rehearsal-3
```

### final single window

```bash
sudo /home/homelab1/coding-local/ultimateLLM/uLLM-project/tools/run-aq4-phase6-service-window.sh /home/homelab1/coding-local/ultimateLLM/uLLM-project/benchmarks/results/2026-07-17/qwen35-9b-aq4-production-opt-v0.1/p2/aq4-phase6-gpu-final-output-v0.1 /home/homelab1/coding-local/ultimateLLM/uLLM-project/benchmarks/results/2026-07-17/qwen35-9b-aq4-production-opt-v0.1/p2/aq4-phase6-gpu-final-output-v0.1/query-hip-device-identity /home/homelab1/coding-local/ultimateLLM/uLLM-phase6-clean-source /home/homelab1/coding-local/ultimateLLM/uLLM-phase6-build-target/release/ullm-aq4-p2-path-oracle d3ea48d543456a07a2796ee804671c3da513c268 --confirm-single-window
```

new scriptsのexecution bitは確認済みである。`tools/run-aq4-phase6-service-window.sh`、`tools/stage-aq4-phase6-path-oracle-binary.py`、`tools/compare-aq4-phase6-final-output.py`はすべてmode `0755`である。guard scriptは`/usr/bin/python3`で起動するためexecution bitを必要としない。

### 期待成果物と比較

final commandは次のnew evidenceだけを書き込む。

```text
/home/homelab1/coding-local/ultimateLLM/uLLM-project/benchmarks/results/2026-07-17/qwen35-9b-aq4-production-opt-v0.1/p2/aq4-phase6-gpu-final-output-v0.1/path-oracle/
/home/homelab1/coding-local/ultimateLLM/uLLM-project/benchmarks/results/2026-07-17/qwen35-9b-aq4-production-opt-v0.1/p2/aq4-phase6-gpu-final-output-v0.1/oracle-link/
/home/homelab1/coding-local/ultimateLLM/uLLM-project/benchmarks/results/2026-07-17/qwen35-9b-aq4-production-opt-v0.1/p2/aq4-phase6-gpu-final-output-v0.1/final-output-comparison/comparison.json
```

`comparison.json`の`before_fix.agreement.logit_sample_bounded_relative_l2_max`は`0.6151289249025698`、`after_fix.agreement.logit_sample_bounded_relative_l2_max`が修正後の同一metricである。`delta.logit_sample_bounded_relative_l2_percent_reduction`と`strictly_improved`で直接比較する。driverはR9700 guard前後、lock probe、service stop/start、post-restore active/running・NRestarts不変・manifest hash不変も同じ`OUT`へ保存する。

## 次の行動

1. 親エージェントがservice稼働中のR9700-only guard rehearsalを3回実行し、各evidenceを確認する。
2. 3回すべてvalidの場合だけ、final service-stop commandを一回実行する。失敗した場合、retryや追加service操作はせずevidenceを報告する。
3. 成功した場合、`final-output-comparison/comparison.json`で`0.6151289249025698`との差を判断し、その後の正式P2 fidelity gateはPhase 7として独立holdoutで扱う。
