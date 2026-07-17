# AQ4 Phase 7 P2 fidelity single-window runbook v0.1

## Scope

This route uses the new independent 48-case fixture pool at
`benchmarks/results/2026-07-17/qwen35-9b-aq4-production-opt-v0.1/p2/aq4-phase7-p2-fidelity-preparation-v0.1`.
It excludes every member of the retired 2026-07-15 48-case split (including
the 19 No-Go rows) and all three distinct Phase 1--6 diagnostic contexts. The
formal split has 24 calibration rows and 24 holdout rows. The holdout is
captured exactly once after calibration bounds are frozen.

Do not edit or invoke the paused P3 harness, any existing Phase 3c/Phase 6
window evidence, or `path-oracle-export`. The separate Phase 6 symlink-guard
issue is neither used nor fixed by this route.

## Fixed inputs

| Input | Value |
|---|---|
| preparation root | `/home/homelab1/coding-local/ultimateLLM/uLLM-project/benchmarks/results/2026-07-17/qwen35-9b-aq4-production-opt-v0.1/p2/aq4-phase7-p2-fidelity-preparation-v0.1` |
| formal split SHA-256 | `ebd759851c2f2c1a9b27b1f529954fa0ef180c0eae1acb4a4426359006dbc43a` |
| formal policy SHA-256 | `302c3219af286a970ddf39ed090021ef102b51b2d188c0ff337f6b9dd04d1a03` |
| clean source worktree | `/home/homelab1/coding-local/ultimateLLM/uLLM-phase6-clean-source` |
| source commit | `d3ea48d543456a07a2796ee804671c3da513c268` |
| required RMSNorm ancestor | `e992b3ea1d0427744dfd83abdc98283a74c1e3b4` |
| Cargo target binary | `/home/homelab1/coding-local/ultimateLLM/uLLM-phase7-build-target/release/ullm-aq4-fidelity-capture` |

`selection-audit.json` and `preparation-manifest.json` are authoritative for
the complete case list, row hashes, old-split exclusion, and non-overlap.
Re-run the preparation verifier before measuring; it fails closed on a
changed file, symlink, or overlap.

## CPU-only preparation

These commands do not query a GPU, service, systemd unit, lock, or active
manifest. The source runner masks CUDA/HIP/ROCR visibility and performs two
CPU BF16 model loads: formal calibration first, then formal-holdout input.
Once `source-oracles/` is valid, do not run its create-new command again.

```bash
/usr/bin/python3 /home/homelab1/coding-local/ultimateLLM/uLLM-project/tools/prepare-qwen35-aq4-phase7-fidelity.py --verify --output /home/homelab1/coding-local/ultimateLLM/uLLM-project/benchmarks/results/2026-07-17/qwen35-9b-aq4-production-opt-v0.1/p2/aq4-phase7-p2-fidelity-preparation-v0.1
/home/homelab1/coding-local/ultimateLLM/uLLM-project/tools/run-aq4-phase7-source-oracles.sh /home/homelab1/coding-local/ultimateLLM/uLLM-project/benchmarks/results/2026-07-17/qwen35-9b-aq4-production-opt-v0.1/p2/aq4-phase7-p2-fidelity-preparation-v0.1 --confirm-cpu-source-capture
cd /home/homelab1/coding-local/ultimateLLM/uLLM-phase6-clean-source
CARGO_TARGET_DIR=/home/homelab1/coding-local/ultimateLLM/uLLM-phase7-build-target CARGO_BUILD_JOBS=1 ULLM_BUILD_GIT_COMMIT=d3ea48d543456a07a2796ee804671c3da513c268 cargo build --release -p ullm-engine --bin ullm-aq4-fidelity-capture
/usr/bin/python3 /home/homelab1/coding-local/ultimateLLM/uLLM-project/tools/stage-aq4-phase7-fidelity-capture-binary.py --source /home/homelab1/coding-local/ultimateLLM/uLLM-phase7-build-target/release/ullm-aq4-fidelity-capture --output /home/homelab1/coding-local/ultimateLLM/uLLM-project/benchmarks/results/2026-07-17/qwen35-9b-aq4-production-opt-v0.1/p2/aq4-phase7-p2-fidelity-preparation-v0.1/fidelity-capture-binary-staging --source-commit d3ea48d543456a07a2796ee804671c3da513c268
g++ -std=c++20 -Wall -Wextra -Werror -D__HIP_PLATFORM_AMD__ -I/opt/rocm/include /home/homelab1/coding-local/ultimateLLM/uLLM-project/tools/query-hip-device-identity.cpp -L/opt/rocm/lib -lamdhip64 -Wl,-rpath,/opt/rocm/lib -o /home/homelab1/coding-local/ultimateLLM/uLLM-project/benchmarks/results/2026-07-17/qwen35-9b-aq4-production-opt-v0.1/p2/aq4-phase7-p2-fidelity-preparation-v0.1/query-hip-device-identity
test -x /home/homelab1/coding-local/ultimateLLM/uLLM-project/benchmarks/results/2026-07-17/qwen35-9b-aq4-production-opt-v0.1/p2/aq4-phase7-p2-fidelity-preparation-v0.1/query-hip-device-identity
sha256sum /home/homelab1/coding-local/ultimateLLM/uLLM-project/benchmarks/results/2026-07-17/qwen35-9b-aq4-production-opt-v0.1/p2/aq4-phase7-p2-fidelity-preparation-v0.1/query-hip-device-identity > /home/homelab1/coding-local/ultimateLLM/uLLM-project/benchmarks/results/2026-07-17/qwen35-9b-aq4-production-opt-v0.1/p2/aq4-phase7-p2-fidelity-preparation-v0.1/query-hip-device-identity.sha256
```

The Cargo target is outside the clean source worktree. The staging tool
publishes a content copy only: executable mode `0555`, `nlink=1`, with receipt
and sums mode `0444`, `nlink=1`.

## Root-only R9700 rehearsal

After every CPU input is valid, with the production service still running,
run each command once. Every `r9700-guard-rehearsal-summary.json` must have
`status=valid`; otherwise do not consume the service-stop window.

```bash
sudo /usr/bin/python3 /home/homelab1/coding-local/ultimateLLM/uLLM-project/tools/run-aq4-phase3c-r9700-guard.py --output /home/homelab1/coding-local/ultimateLLM/uLLM-project/benchmarks/results/2026-07-17/qwen35-9b-aq4-production-opt-v0.1/p2/aq4-phase7-p2-fidelity-preparation-v0.1/guard-chain-rehearsal-v0.1/attempt-1 --guard-bin /home/homelab1/coding-local/ultimateLLM/uLLM-project/benchmarks/results/2026-07-17/qwen35-9b-aq4-production-opt-v0.1/p2/aq4-phase7-p2-fidelity-preparation-v0.1/query-hip-device-identity --health-phase rehearsal-1
sudo /usr/bin/python3 /home/homelab1/coding-local/ultimateLLM/uLLM-project/tools/run-aq4-phase3c-r9700-guard.py --output /home/homelab1/coding-local/ultimateLLM/uLLM-project/benchmarks/results/2026-07-17/qwen35-9b-aq4-production-opt-v0.1/p2/aq4-phase7-p2-fidelity-preparation-v0.1/guard-chain-rehearsal-v0.1/attempt-2 --guard-bin /home/homelab1/coding-local/ultimateLLM/uLLM-project/benchmarks/results/2026-07-17/qwen35-9b-aq4-production-opt-v0.1/p2/aq4-phase7-p2-fidelity-preparation-v0.1/query-hip-device-identity --health-phase rehearsal-2
sudo /usr/bin/python3 /home/homelab1/coding-local/ultimateLLM/uLLM-project/tools/run-aq4-phase3c-r9700-guard.py --output /home/homelab1/coding-local/ultimateLLM/uLLM-project/benchmarks/results/2026-07-17/qwen35-9b-aq4-production-opt-v0.1/p2/aq4-phase7-p2-fidelity-preparation-v0.1/guard-chain-rehearsal-v0.1/attempt-3 --guard-bin /home/homelab1/coding-local/ultimateLLM/uLLM-project/benchmarks/results/2026-07-17/qwen35-9b-aq4-production-opt-v0.1/p2/aq4-phase7-p2-fidelity-preparation-v0.1/query-hip-device-identity --health-phase rehearsal-3
```

## One and only service-stop command

The parent agent alone may run this exact command once, after three valid
rehearsals. It makes one stop/start pair and rejects any reused window output.
Before stopping, it validates preparation/source artifacts, nlink staging,
the clean fixed source, the RMSNorm-fix ancestry, active 30 guards (including
the 17 required Phase 3c subset), `RuntimeDirectoryPreserve=yes`, lock, and
the R9700-only contract.

```bash
sudo /home/homelab1/coding-local/ultimateLLM/uLLM-project/tools/run-aq4-phase7-service-window.sh /home/homelab1/coding-local/ultimateLLM/uLLM-project/benchmarks/results/2026-07-17/qwen35-9b-aq4-production-opt-v0.1/p2/aq4-phase7-p2-fidelity-preparation-v0.1 /home/homelab1/coding-local/ultimateLLM/uLLM-project/benchmarks/results/2026-07-17/qwen35-9b-aq4-production-opt-v0.1/p2/aq4-phase7-p2-fidelity-preparation-v0.1/query-hip-device-identity /home/homelab1/coding-local/ultimateLLM/uLLM-phase6-clean-source d3ea48d543456a07a2796ee804671c3da513c268 --confirm-single-window
```

The stopped interval is fixed to target calibration (one GPU model load),
metric validation and freeze, target formal holdout (one GPU model load),
post-guard, and one service start. The holdout execution view is hash-mapped
to formal holdout and cannot derive calibration bounds.

## Expected artifacts and decision

The create-new output root contains `source-oracles/{calibration,holdout}/`,
`target-calibration/`, `calibration-metrics.json`,
`calibration-metrics-validation.json`, `freeze-receipt.json`,
`target-holdout/`, `holdout-evaluation.json`, `active-pins.json`,
`phase7-guard-contract.json`, `guard-before/`, `guard-after/`, and
`service-window-*` evidence.

`holdout-evaluation.json.status=go` means the frozen policy passed: Wilson
token agreement and quality-retention rates, top-10 overlap, logits/hidden
cosine and relative-L2, and each relative-L2 row at most 1.0. It also records
hidden/logits max-abs drift per row.

The historical binding spec additionally requires zero source greedy mismatch
rows and names a logits max-abs bound, whereas the frozen policy uses a Wilson
token-agreement rate, no logits max-abs bound, and diagnostic-only hidden
max-abs. The evaluator therefore records
`formal_p2_status=blocked_contract_resolution` even if the frozen numerical
policy passes. Do not claim full P2 promotion until that document-level
incompatibility is resolved without modifying observed-policy thresholds.
