# AQ4 P2 production baseline CPU preparation v0.1

## 前回の要点

- Qwen3.5 AQ4 fidelity の根本原因だった final RMSNorm additive weight の適用漏れは、commit `e992b3ea` 系で修正され、独立holdout 48件の正式P2 fidelity gateでは8指標中7指標が合格した。残る token agreement はmargin相関により量子化ノイズとして許容され、`851c9c9d` / `f1a3cf4c` でクローズされた。
- その問題は production prefill/decode optimization plan のP2 baseline/profile中に発見されたため、旧P2診断結果は探索資料に留め、現active identity向けのbaselineを新規取得する必要がある。

## 今回の変更点

### 新規 current-identity P2 envelope

- `tools/prepare-aq4-p2-production-baseline.py` を追加し、clean detached source、active AQ4 manifest/worker/package、source checkpoint/tokenizer、deterministic fixture、P2 matrixをimmutable envelopeに固定した。
- clean source commit は `f1a3cf4c86978b3b8900396a0b6a8caff90b97f1`、final preparation root は `benchmarks/results/2026-07-17/qwen35-9b-aq4-production-opt-v0.1/p2/aq4-production-prefill-decode-baseline-v0.4/` である。v0.1〜v0.3はCPU preparation receiptのみで、実行には使わない。
  - preparation SHA-256: `1289b145c65340f7a790113f7bbc7db60135c2870aa12a0618a9ec6739fcef49`
  - identity SHA-256: `e682b50a7c34edd288d759cac146c07679132bbfc40948efb90d42f405f038a2`
- active manifestはproduct promotion source commitを公開していないため、比較可能と偽装せず `separated_not_comparable` と記録した。以降のwindowはactive manifest/package/worker hashを再確認し、差分があれば比較不能として止まる。
- prefill代表7点とdecode context代表7点の双方にM grid `1/8/16/32/64/128`を固定した。decodeではMがcontext-prefill chunk幅を表し、64 generated tokenの各decode iterationは物理的に幅1のままなので、requested/resolved Mとfallbackを別々に記録する。planned 91件、cached-prefix chunkedの42組合せはactive AQ4 pathが未広告のため明示的に `unsupported` とし、成功件数へ含めない。
- 通常計測14、詳細profile 6、path oracle 8の、合計28本のsingle-use R9700 window planを作成した。一つの長時間service stopへ全matrixを詰め込まず、decodeはstart contextごとにM grid全体を1 windowへ隔離する。

### staged binaries、guard、single-window executors

- clean detached sourceから `ullm-aq4-p2-resident-driver` と `ullm-aq4-p2-calibration` をrelease buildし、`tools/stage-aq4-p2-production-baseline-binaries.py`でnlink=1のimmutable copiesへstageした。
  - resident SHA-256: `daf6f12a4d4aaad11b0ef5ffe717372d47b7174271671ae1b9e2f4daf1288753`
  - calibration SHA-256: `b4bbdd6f57169326f269bcccc069538a378b308d61a247de2d77766bb539d641`
- `tools/stage-aq4-p2-r9700-guard.py`を追加した。host-only HIP identity guardをclean sourceからcompile/stageするだけで、guard executableやHIP queryは実行しない。v0.4 staged guardはnlink=1、SHA-256 `0964f145bc2a931a4270d89715e2b86c1d8043d088630da52d74d05f1f40aa1f`。
- `tools/run-aq4-p2-production-baseline-window.py`とroot-only `tools/run-aq4-p2-production-baseline-service-window.sh`を追加した。後者は既存Phase 3c/6/7の契約を踏襲し、単発stop/restore、`RuntimeDirectoryPreserve=yes`、R9700-only HIP/ASIC guard、stop後の既存lock read-only probe、service-user boundary、inherit FD 9を使う。executor自身はlockをopen/create/acquireしない。
- 詳細profileだけは同driverでrocprofv3 kernel/HIP runtime/memory-copy raw CSVを捕捉する。`tools/parse-aq4-p2-production-profile.py`はraw member inventory、trace hash binding、unknown kernel fail-closed分類を作る。profile timingはnormal p50/p95へ混ぜない。

### streaming oracle/report path

- `tools/capture-aq4-p2-production-source-oracle.py`はCPU BF16 sourceのhidden/logit rowを65536 element chunksでsidecarへ出す。GPU visibilityが少しでも残れば拒否し、full logit matrixを保持しない。
- `tools/run-aq4-p2-production-path-oracle.py`と対応するroot-only service-window driverを追加した。全8 anchorは一つずつclean calibration process/windowで捕捉し、source row SHA、replay sequence SHA、generation epochをstate snapshotとして結ぶ。
- `tools/compare-aq4-p2-production-oracles.py`はanchor filterを備え、source/path vectorsをrow-wise streaming比較する。target execution rowsがあればSHA256SUMSとmanifestまで検証する。
- `tools/build-aq4-p2-production-bottleneck-report.py`はnormal p50/p95、詳細profile kernel family、launch/sync、transferをrankする。ただしworkspace/fallbackが新current-identity traceで未観測なら、zeroへ置換せずreportをblockedのままにする。
- `tools/seal-aq4-p2-production-baseline-jsonl.py`は全14 normal windowのhash-bound sanitized sidecarをCPU-onlyで再検証し、planned 91 case × 10 measured runを`baseline-measurements.jsonl`へimmutableに封印する。partial matrix、raw trace/sidecar hash不一致、M resolution不一致、unsanitized fieldは拒否する。

### Runbook

- `docs/plans/aq4-production-prefill-decode-optimization-plan-v0.1-p2-runbook-v0.1.md`を追加した。parent operator用のCPU preflight、guard rehearsal、最初のfinal single-window、source/path oracle、post-window compare/reportのcommandを明記した。

## CPU-only 検証

- `cargo build --release -p ullm-engine --bin ullm-aq4-p2-resident-driver --bin ullm-aq4-p2-calibration --target-dir /home/homelab1/coding-local/ultimateLLM/uLLM-p2-baseline-build-f1a3cf4c` — 成功。既存C++ `subobject-linkage` warningのみ。
- `pytest -q tests/test_aq4_p2_production_baseline_preparation.py` — `8 passed`。matrix/immutable preparation、nlink=1 binary stage、window dry-run、fake compiler guard stage、path-oracle dry-run、profile parser、streaming comparator、CPU source preflight、immutable baseline JSONL sealing、workspace/fallback未観測時のreport blockerを含む。
- `python3 -m py_compile`（新規P2 Python tools一式）— 成功。
- `bash -n tools/run-aq4-p2-production-baseline-service-window.sh tools/run-aq4-p2-production-path-oracle-service-window.sh` — 成功。
- real v0.4 preparation/stage/guard `--verify` — 成功。normal `prefill-n128` executor `--dry-run` — `dry_run_valid`、7 planned cases + 6 unsupported cached-prefix cases、GPU/service action `none`。decode M gridの`decode-c16`も`--dry-run`成功（6 planned cases、GPU/service action `none`）。
- actual source checkpointに対する `capture-aq4-p2-production-source-oracle.py --preflight` を、`CUDA_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1 ULLM_HIP_VISIBLE_DEVICES=-1` で実行 — `preflight_valid`、8 cases/8 rows、GPU/service action `none`。
- `git diff --check` — 成功。

## 実行していないこと

- GPU inference/HIP runtime実行、rocprof capture、AMD-SMI/HIP guard rehearsal、systemd service stop/start、`/run/ullm/r9700.lock`の取得・作成、sudoは一切実行していない。
- independent source oracleの実captureはCPU-onlyだが、9B source modelの長文脈replayを伴うためpreflightで止めた。runbookの明示confirmation commandをparent operatorが必要時に実行する。
- target path oracle、通常14 window、詳細profile 6 window、bottleneck family選択はいずれも未実施である。
- SQ8 formatのcode/tooling、V620、active manifest/package/serviceには触れていない。

## 次の行動

1. execution worktreeのtooling tracked/clean gateを満たしたうえで、parent operatorがrunbookのR9700 guard rehearsalを実行する。
2. guardがvalidなら、まず`prefill-n128`だけをsingle-use service windowとして実行・確認する。以降もwindow IDごとに直列で進める。
3. CPU source sidecar capture後、8 anchorを一つずつpath oracle windowで捕捉し、streaming comparatorを実行する。
4. workspace/fallbackを実観測するcurrent-identity traceが揃わない限り、ranked reportからoptimizer familyを選定しない。
