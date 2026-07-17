# AQ4 Phase 6 GPU final-output window runbook v0.1

## 目的と測定経路

このrunbookは、final RMSNorm additive weight修正（`e992b3ea`）を、承認済みの**一回だけ**のR9700 service-stop windowで再確認するためのものだ。GPU実行・`sudo`・service操作は親エージェントだけが行う。ここに記載するCPU-only準備は完了済みであり、このrunbook作成者はGPU、service、systemd、lockを操作していない。

07/14の比較対象は`ullm-aq4-p2-path-oracle`の3 row M=1/cold診断である。`source-oracle-v2`と`path-oracle-v2`を現行`qwen35_aq4_p2_oracle.compare_payloads`で再計算した値は次のとおりである。

| metric | 再計算値 |
| --- | ---: |
| final hidden bounded relative L2 max | `0.5452883336042509` |
| final logit bounded relative L2 max | **`0.6151289249025698`** |
| logit bounded cosine min | `0.9446401707134972` |
| logit max abs | `8.347781658172607` |

従ってPhase 6の唯一のGPU model runは`ullm-aq4-p2-path-oracle`を使う。`ullm-aq4-differential-trace`の07/14 attempt3はlayer別診断として有用だが、保存済み先頭32 logit座標だけでは`0.6212985932808415`であり、今回比較すべき`0.6151289249`そのものを出した経路ではない。

このmetricは、3 rowごとに保存されたsource/AQ4 logit座標の**共通部分**について、`sqrt(sum((aq4-source)^2)) / max(sqrt(sum(source^2)), 1e-12)`を計算し、その最大値を取るものだ。full-vocabularyのP2 fidelity gateではない。Phase 7の独立holdoutを代替しない。

## 固定identityとfixture

```text
REPO=/home/homelab1/coding-local/ultimateLLM/uLLM-project
SOURCE_COMMIT=d3ea48d543456a07a2796ee804671c3da513c268
SOURCE_TREE=/home/homelab1/coding-local/ultimateLLM/uLLM-phase6-clean-source
SOURCE_BIN=/home/homelab1/coding-local/ultimateLLM/uLLM-phase6-build-target/release/ullm-aq4-p2-path-oracle
OUT=/home/homelab1/coding-local/ultimateLLM/uLLM-project/benchmarks/results/2026-07-17/qwen35-9b-aq4-production-opt-v0.1/p2/aq4-phase6-gpu-final-output-v0.1
PACKAGE=/home/homelab1/datapool/ullm/product/qwen35-9b-aq4-cli-v0.1/package
CASES=/home/homelab1/coding-local/ultimateLLM/uLLM-project/tests/fixtures/qwen35-aq4-p2-oracle/cases.json
SOURCE_ORACLE=/home/homelab1/coding-local/ultimateLLM/uLLM-project/benchmarks/results/2026-07-14/qwen35-9b-aq4-production-opt-v0.1/p2/source-oracle-v2
BASELINE_PATH_ORACLE=/home/homelab1/coding-local/ultimateLLM/uLLM-project/benchmarks/results/2026-07-14/qwen35-9b-aq4-production-opt-v0.1/p2/path-oracle-v2
TOKENIZER_ROOT=/home/homelab1/datapool/ai_models/safetensors/Qwen/Qwen3.5-9B
LOCK=/run/ullm/r9700.lock
```

`CASES`のSHA-256は`15fed90dd2e16a5b68d4498c8632257d80ac94c56ed614696b0884c65f4836f2`である。path-oracle exporterは`SOURCE_ORACLE`から同じsource replayを再構成し、runtime sidecarに各context hashを束縛する。

| case / step | context | hash |
| --- | --- | --- |
| `fixture-prompt-0` / 0 | `[11,12,13]` | `42ea52c728680a54afafd1c1e1e45f13300c3ceb962f320f3900196a0c46215c` |
| `fixture-prompt-0` / 1 | `[11,12,13,220]` | `6af1601b9bf35d095b24c5bac3a95a01bf77d047b576441d0a5f9510eec66249` |
| `fixture-prompt-1` / 0 | `[21,22]` | `3bca9e21e3b6f741ed412f91d7696146c254ff68bd9be9ca41b1d172eb3549e6` |

## Phase 3c v0.7基盤の再利用

- `RuntimeDirectoryPreserve=yes`をstop直前に必須assertする。`/run/ullm/r9700.lock`は既存regular fileを`O_CREAT`なしでopenし、`LOCK_EX|LOCK_NB`を一回だけ試す。作成、修復、待機、unlinkはしない。
- `HIP_VISIBLE_DEVICES=1`と`ULLM_HIP_VISIBLE_DEVICES=1`を固定する。HIP guardはfiltered ordinal 0だけを調べ、同じBDFだけを`/opt/rocm/bin/amd-smi --gpu`へ渡し、`gfx1201`・PCI device ID `0x7551`・BDF一致をassertする。`amd-smi list`、対象なしquery、V620 queryは使わない。
- Phase 3c v0.7の17 required guardを、Phase 6 active production guardの必須subsetとしてassertする。path-oracleは07/14と同じactive manifest経路を再現するため、実行childにはactive manifestの完全な30 guardを`=1`で明示し、`env -i`で余分なworker環境を継承しない。17 guardは削らない。
- RPBは`QKV/Z/gate/beta=4`、`SiLU/mul=8`、`add=8`に固定する。Cargo hardlinkを直接実行せず、content copy済みのnlink=1 staging executableだけを実行する。
- 07/16に停止したP3 harnessのroot、artifact、environment、`rocprof`、recovery操作は参照・変更しない。

## CPU-only準備済みの状態

clean worktreeは`SOURCE_COMMIT`にdetachし、worktree本体の`git status --short`は空である。targetはworktree外に置き、clean contractを汚さない。

```bash
CARGO_TARGET_DIR=/home/homelab1/coding-local/ultimateLLM/uLLM-phase6-build-target CARGO_BUILD_JOBS=1 ULLM_BUILD_GIT_COMMIT=d3ea48d543456a07a2796ee804671c3da513c268 cargo build --release -p ullm-engine --bin ullm-aq4-p2-path-oracle
```

buildは成功した（既知のC++ `subobject-linkage` warningのみ）。CPU-only staging evidenceはすでに作成済みである。

- staged binary: `OUT/path-oracle-binary-staging/ullm-aq4-p2-path-oracle`
- mode: `0555`、nlink: `1`
- SHA-256: `774964446f3fbfe10323242e67f3aeb95f8f34d42e84db2e46763ff782e452de`
- staging receipt / checksum: mode `0444`、nlink `1`、`sha256sum -c`済み
- host-only HIP guard: `OUT/query-hip-device-identity`、mode `0755`、SHA-256 `e85043b1bc1812a1b0ebcba31fcfa0bff5402be348d713a37f44643d9885175d`
- baseline reproduction: `OUT/baseline-metric-reproduction/comparison.json`が`0.6151289249025698`をexactに再現済み

次の確認はGPUを使わない。

```bash
cd /home/homelab1/coding-local/ultimateLLM/uLLM-project/benchmarks/results/2026-07-17/qwen35-9b-aq4-production-opt-v0.1/p2/aq4-phase6-gpu-final-output-v0.1/path-oracle-binary-staging && sha256sum -c SHA256SUMS
test -x /home/homelab1/coding-local/ultimateLLM/uLLM-project/tools/run-aq4-phase6-service-window.sh
test -x /home/homelab1/coding-local/ultimateLLM/uLLM-project/tools/stage-aq4-phase6-path-oracle-binary.py
test -x /home/homelab1/coding-local/ultimateLLM/uLLM-project/tools/compare-aq4-phase6-final-output.py
test -x /home/homelab1/coding-local/ultimateLLM/uLLM-project/benchmarks/results/2026-07-17/qwen35-9b-aq4-production-opt-v0.1/p2/aq4-phase6-gpu-final-output-v0.1/query-hip-device-identity
test -x /home/homelab1/coding-local/ultimateLLM/uLLM-project/benchmarks/results/2026-07-17/qwen35-9b-aq4-production-opt-v0.1/p2/aq4-phase6-gpu-final-output-v0.1/path-oracle-binary-staging/ullm-aq4-p2-path-oracle
```

## 親エージェントが実行するroot-only rehearsal

serviceが稼働したまま、次の3 commandを順番に一回ずつ実行する。これはHIP identityとR9700 BDF指定済みAMD-SMI telemetryだけであり、device memory、stream、kernel、lock、service、manifestは操作しない。各attemptが`r9700-guard-rehearsal-summary.json`の`status=valid`であることを確認してから次へ進む。

```bash
sudo /usr/bin/python3 /home/homelab1/coding-local/ultimateLLM/uLLM-project/tools/run-aq4-phase3c-r9700-guard.py --output /home/homelab1/coding-local/ultimateLLM/uLLM-project/benchmarks/results/2026-07-17/qwen35-9b-aq4-production-opt-v0.1/p2/aq4-phase6-gpu-final-output-v0.1/guard-chain-rehearsal-v0.1/attempt-1 --guard-bin /home/homelab1/coding-local/ultimateLLM/uLLM-project/benchmarks/results/2026-07-17/qwen35-9b-aq4-production-opt-v0.1/p2/aq4-phase6-gpu-final-output-v0.1/query-hip-device-identity --health-phase rehearsal-1
sudo /usr/bin/python3 /home/homelab1/coding-local/ultimateLLM/uLLM-project/tools/run-aq4-phase3c-r9700-guard.py --output /home/homelab1/coding-local/ultimateLLM/uLLM-project/benchmarks/results/2026-07-17/qwen35-9b-aq4-production-opt-v0.1/p2/aq4-phase6-gpu-final-output-v0.1/guard-chain-rehearsal-v0.1/attempt-2 --guard-bin /home/homelab1/coding-local/ultimateLLM/uLLM-project/benchmarks/results/2026-07-17/qwen35-9b-aq4-production-opt-v0.1/p2/aq4-phase6-gpu-final-output-v0.1/query-hip-device-identity --health-phase rehearsal-2
sudo /usr/bin/python3 /home/homelab1/coding-local/ultimateLLM/uLLM-project/tools/run-aq4-phase3c-r9700-guard.py --output /home/homelab1/coding-local/ultimateLLM/uLLM-project/benchmarks/results/2026-07-17/qwen35-9b-aq4-production-opt-v0.1/p2/aq4-phase6-gpu-final-output-v0.1/guard-chain-rehearsal-v0.1/attempt-3 --guard-bin /home/homelab1/coding-local/ultimateLLM/uLLM-project/benchmarks/results/2026-07-17/qwen35-9b-aq4-production-opt-v0.1/p2/aq4-phase6-gpu-final-output-v0.1/query-hip-device-identity --health-phase rehearsal-3
```

rehearsalが1つでも失敗したら、service-stop commandは実行しない。原因をevidenceから報告し、windowの再試行は承認なしに行わない。

## 親エージェントが一回だけ実行するservice-stop window

3 rehearsalがすべてvalidであることを確認してからのみ、次のcommandを**一回だけ**実行する。

```bash
sudo /home/homelab1/coding-local/ultimateLLM/uLLM-project/tools/run-aq4-phase6-service-window.sh /home/homelab1/coding-local/ultimateLLM/uLLM-project/benchmarks/results/2026-07-17/qwen35-9b-aq4-production-opt-v0.1/p2/aq4-phase6-gpu-final-output-v0.1 /home/homelab1/coding-local/ultimateLLM/uLLM-project/benchmarks/results/2026-07-17/qwen35-9b-aq4-production-opt-v0.1/p2/aq4-phase6-gpu-final-output-v0.1/query-hip-device-identity /home/homelab1/coding-local/ultimateLLM/uLLM-phase6-clean-source /home/homelab1/coding-local/ultimateLLM/uLLM-phase6-build-target/release/ullm-aq4-p2-path-oracle d3ea48d543456a07a2796ee804671c3da513c268 --confirm-single-window
```

driverはstop前にstaging、clean source worktree、RMSNorm fix ancestor、active manifest SHA、`RuntimeDirectoryPreserve=yes`、17/30 guard contractを再検証する。どれかが失敗した場合はserviceを停止しない。stop後はlock probe、R9700 guard、path-oracle一回、post guard、service start一回の順に実行する。失敗時もretry、restart、lock作成/修復、V620 queryは行わない。

## 期待する成果物と比較方法

成功時には以下がcreate-newで保存される。

```text
OUT/guard-before/
OUT/guard-after/
OUT/path-oracle/{manifest.json,payload.jsonl,runtime.json,SHA256SUMS}
OUT/oracle-link/manifest.json
OUT/final-output-comparison/{comparison.json,SHA256SUMS}
OUT/{service-window-pre-stop.txt,service-window-lock-after-stop.json,service-window-post-restore.txt,service-window-result.txt}
```

`OUT/final-output-comparison/comparison.json`の次を比較する。

```text
before_fix.agreement.logit_sample_bounded_relative_l2_max
after_fix.agreement.logit_sample_bounded_relative_l2_max
delta.logit_sample_bounded_relative_l2_absolute
delta.logit_sample_bounded_relative_l2_percent_reduction
```

`before_fix`は`0.6151289249025698`に固定され、`after_fix`は同一source oracle、同一3 context、同一座標、同一bounded式で再計算される。改善の有無は`after_fix < before_fix`で機械的に示すが、量子化誤差として十分かどうかと正式P2 Go判定は別途Phase 7で判断する。
