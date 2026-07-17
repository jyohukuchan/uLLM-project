# AQ4 Phase 7 P2 fidelity gate CPU-only preparation v0.1

## 前回の要点

- `e992b3ea`のQwen3.5 final RMSNorm additive weight修正により、CPU-onlyではfinal RMSNorm relative-L2が`0.5010330688`から`0.1688691127`へ、LM head sampleが`0.5860500940`から`0.0582126936`へ改善した。
- Phase 6のR9700実測では、07/14のbounded logit relative-L2最大`0.6151289249`が修正後`0.0635407831`へ改善した。ただし3 row中2 rowではgreedy top-1が一致しなかった。
- 07/15の旧splitは24 calibration / 24 holdoutの48件である。`calibration-no-go.json`には19件のNo-Go rowがあり、旧splitを同じcaseで再測定する根拠は文書から得られなかった。安全側として旧48件すべてを除外する。

## 今回の変更点

### 凍結policyと契約の監査

- `docs/proposals/aq4-p2-fidelity-holdout-protocol-v0.1.md`の既存formula bytesを変更せず再利用した。policy SHA-256は`302c3219af286a970ddf39ed090021ef102b51b2d188c0ff337f6b9dd04d1a03`である。
- 凍結policyはtoken agreement率、top-10 overlap、logits cosine / relative-L2、hidden cosine / relative-L2、hidden max-abs（診断専用）、BF16 top-1がAQ4 top-10に残る固定品質率を扱う。二項率は`n=24`の95%片側Wilson下限、連続値はcalibration平均からの事前formula、logits/hidden relative-L2の各row `> 1.0`は集計前No-Goである。
- `docs/specs/aq4-p2-calibration-evidence-binding-v0.1.md`はsource comparisonのgreedy mismatch rowを0とし、logits max-abs boundを含む5数値を明記する。一方、凍結policyはWilson token-agreement、logits max-abs boundなし、hidden max-abs diagnostic-onlyである。この文書上の不一致はpolicyを変更せず比較器で明示し、`formal_p2_status=blocked_contract_resolution`としてfail-closedにする。凍結数値policyの結果は別に`go`/`no-go`で記録する。

### 新しい独立splitとhash非重複

- 新規fixture domain `ullm.aq4_phase7_independent_fixture.v1\\0`から48 caseを生成し、各stratum（prompt長4種 × mode 2種）を既存のper-stratum SHA-256 ruleでcalibration 24 / formal holdout 24へ分割した。観測済みfidelity値は選定に使っていない。
- formal split manifest SHA-256は`ebd759851c2f2c1a9b27b1f529954fa0ef180c0eae1acb4a4426359006dbc43a`、formal calibration cases SHA-256は`a068e1e4f0adbeef5d1f50b8b85e5a12ec6538704dc4a5a1c6ae2dd23226a63d`、formal holdout cases SHA-256は`44db10127761ddc507e7e7485a587bcffbf86179d03be73ab49ad64c80cf9437`である。
- retired 07/15 split manifest `966878f3d9eb13f5b485825208f8072521724f308f5ee3d8a003b0b051198887`の全48 case/hash/context、19件No-Go list（file SHA-256 `e5f0ac76a53dff92a88a0971d575730e21f9e833b8564d21c780a5539abec143`）、Phase 1〜6の3 distinct context hashとのintersectionはすべて空だった。Phase 1〜6 context hashは`42ea52c728680a54afafd1c1e1e45f13300c3ceb962f320f3900196a0c46215c`、`6af1601b9bf35d095b24c5bac3a95a01bf77d047b576441d0a5f9510eec66249`、`3bca9e21e3b6f741ed412f91d7696146c254ff68bd9be9ca41b1d172eb3549e6`である。
- 機械検証の詳細は`benchmarks/results/2026-07-17/qwen35-9b-aq4-production-opt-v0.1/p2/aq4-phase7-p2-fidelity-preparation-v0.1/selection-audit.json`（SHA-256 `ca15c4a27143536c2fbef448cc75678a36b38f636e23e8842aa95915a77a0feb`）に固定した。

formal holdout 24件は次のとおりである（左からcase ID、case SHA-256、context token SHA-256）。

| case_id | case_sha256 | context_token_ids_sha256 |
|---|---|---|
| `p2-phase7-independent-production_server-cold_prefill-all_m1-n1011-m16-r9700-rdna4-aq4_0_target` | `1cd8636a4c60cd08b4569a76e6c8bc263d584bc2b7f89e85034cc0cbd3fb4f91` | `fbb8364ac619a9bb503f9def90b717c1dc55e0f7a05297a3be354222a6caa026` |
| `p2-phase7-independent-production_server-cold_prefill-all_m1-n1011-m64-r9700-rdna4-aq4_0_target` | `7b1d5e4a9763bdf65e7fc936203dfba1538af88f46f93bc81715df6132580936` | `145f7da75f24547f1701697c2e6222a7a3325c86fc6effe151a695f17259d354` |
| `p2-phase7-independent-production_server-cold_prefill-all_m1-n1011-m8-r9700-rdna4-aq4_0_target` | `0846bc15ed75bda45d2d84231bbc854a0535d10eb99138d3a4bfa14a6e15278e` | `174c89a48bd8ba74adc184d36d66224149923f9fb7fc7d760d89d00f509993e7` |
| `p2-phase7-independent-production_server-cold_prefill-all_m1-n1024-m1-r9700-rdna4-aq4_0_target` | `832d7657f9e478c7a172baf4962ce0814800adc672d779c309a03495a455d4d0` | `c693c2401eb0f33319cb89fe6a3cf3fd6e29307a4f80a9287f498217a807dbd3` |
| `p2-phase7-independent-production_server-cold_prefill-all_m1-n1024-m16-r9700-rdna4-aq4_0_target` | `8a8430d1c256b9c625a305926c5903da49fae6ce239e90f9984a79b00b0c6104` | `91a5ae520b8fb2f053af8c7f73bc2fd4b44799cf44e38bebdd0f7d3716d20ade` |
| `p2-phase7-independent-production_server-cold_prefill-all_m1-n1024-m32-r9700-rdna4-aq4_0_target` | `2e32423555f41cf5035ededd5f8d49497c7e33896baca144db238896047a2d4b` | `471c4be22f6d347c787f259f4ea0651141ad7d5a59ba5b5494475822aa8c1861` |
| `p2-phase7-independent-production_server-cold_prefill-all_m1-n1339-m128-r9700-rdna4-aq4_0_target` | `88329155e15bacd47aa47cae8dc7d26e729e445fb6685abb9a20f329844458ad` | `35e397a0f8038facc3422355c3e69abc4f6605971ffc6363d407c5bd061efd6f` |
| `p2-phase7-independent-production_server-cold_prefill-all_m1-n1339-m32-r9700-rdna4-aq4_0_target` | `3d71e43ac45e757241150c3967797f6ec769e990164bbac1b94a36ca528e6f93` | `4a142092bee0947cdf6e319ec820ba394ab5d7277b997a2889bc181b601ac2de` |
| `p2-phase7-independent-production_server-cold_prefill-all_m1-n1339-m64-r9700-rdna4-aq4_0_target` | `caa1647041caf49d743118ae60fb72c4d4eafaab0542e16167b1dfeaa3362fe4` | `a2bcd8cff582def52a2047b36ab64304f61fc4c6f2601247d367d32d0dc6fa6a` |
| `p2-phase7-independent-production_server-cold_prefill-all_m1-n2048-m1-r9700-rdna4-aq4_0_target` | `754fc44b8013f8575d3ee4ac0cde0b1e4a04fcc25efaf6b738005d7328c586e1` | `d239353845000272ecd1beb7cd7fcf458eb0a1ef525e5d299eb7665c2993daaa` |
| `p2-phase7-independent-production_server-cold_prefill-all_m1-n2048-m16-r9700-rdna4-aq4_0_target` | `cc4aee591a0ffe99d82d410435386ba432f7193b53dd9673cd573ed223ee9019` | `6b160d17590b08addfc43609c70c489391f4cc18dd27100edf149402efae874b` |
| `p2-phase7-independent-production_server-cold_prefill-all_m1-n2048-m8-r9700-rdna4-aq4_0_target` | `c6dbf37a5c9c4137eb9e038b0c25cf8d6785a74a48967f95ab88ea8d6831dbc5` | `ccc5731b78f7b38473f36c45cd2fe4319da69d0c69937240c30a66c35e981979` |
| `p2-phase7-independent-production_server-cold_prefill-cold_batched-n1011-m16-r9700-rdna4-aq4_0_target` | `7a55704b3050a6146c0396434e980baf78fb716c591e77c44d584b7288780970` | `561cf853732b5b42750a919161fe5e9ed20b2689e150d4af83bb80fdcbd85255` |
| `p2-phase7-independent-production_server-cold_prefill-cold_batched-n1011-m32-r9700-rdna4-aq4_0_target` | `92dd77eac12e04e467ed1cfa6200513f78d4a19af3e1a8517e4c7d0927333e30` | `76d188da383f47a29f770681a6d72e1dc1f3be574ea3d04fbcee1eb8d6e7ea70` |
| `p2-phase7-independent-production_server-cold_prefill-cold_batched-n1011-m64-r9700-rdna4-aq4_0_target` | `5a49c2e2d2f323bd478aa2ce7b8d86691f8be81c19cfb68d5080cfa0509b7040` | `d464ab390f185a01a2a336fdbe77e3b2d67f568cb727f6533f7258f1b23bb07d` |
| `p2-phase7-independent-production_server-cold_prefill-cold_batched-n1024-m128-r9700-rdna4-aq4_0_target` | `78b172aab23c41e3d282e5881d5c4baa479ddf66ce1e9885a259901dac39e54c` | `2ce0f2634fbd74f9111d20012feb9a5cfd9bcfd9488d6e9b16f8b0cf59d247d0` |
| `p2-phase7-independent-production_server-cold_prefill-cold_batched-n1024-m32-r9700-rdna4-aq4_0_target` | `04cd7889cc55bde49618f4f06dd2e84d9dd671ae291ec197657306764755f63b` | `e8e7bf9f7036eddb755b1540bf49918bf1b6733070c3a3c546a007d60bb993f3` |
| `p2-phase7-independent-production_server-cold_prefill-cold_batched-n1024-m8-r9700-rdna4-aq4_0_target` | `699649ca5c619c6223d8029fb235917097b1a8d383dd77a8f66afa66e5312539` | `8f870ca17ffba42bbd9f77bedd6cda16a72b1dc9beea7c77fe1e851538633a30` |
| `p2-phase7-independent-production_server-cold_prefill-cold_batched-n1339-m128-r9700-rdna4-aq4_0_target` | `2373a661bd6f109cafbb44a228ff94445761922c5d72d0bffa58730860f694b1` | `35f8ba932bcfa962d2aa7cd99f5ae9f0ccc48936ce74e91830c21e6ff2ac2d3f` |
| `p2-phase7-independent-production_server-cold_prefill-cold_batched-n1339-m16-r9700-rdna4-aq4_0_target` | `3d88f33a94c5a016334bd2d5905466ef58f737730f363055b261a55eb7a974d6` | `7dcfe5022e0bb8ad9feb0917a31e92f5c9f774d293f0f59090f014ce5a800ad0` |
| `p2-phase7-independent-production_server-cold_prefill-cold_batched-n1339-m8-r9700-rdna4-aq4_0_target` | `9b7f582d5fc4e334d3d62ff2d967f3432c43fabb6d8f455b8bf6e54c36e04d98` | `ba24fdee819348b717f0256eae8f369d1a8dbacea9660721d02b7aeebb408a88` |
| `p2-phase7-independent-production_server-cold_prefill-cold_batched-n2048-m1-r9700-rdna4-aq4_0_target` | `da6e77c07023018d6e517cba16acb637d918f0be0ba5d7da4432fe853cce6879` | `4dd10f2d02a07442e09aabd9bdcb2b61908a9bae6a7b8efd84e7f3b0072835d3` |
| `p2-phase7-independent-production_server-cold_prefill-cold_batched-n2048-m128-r9700-rdna4-aq4_0_target` | `6ef8dde275b0adc719e89a310b5d649e73da588ba073c7e73d800e75d0d880eb` | `86052ced7b52431428a8c482f7e5e8dca068d7a0810800195cc67b28171551ef` |
| `p2-phase7-independent-production_server-cold_prefill-cold_batched-n2048-m64-r9700-rdna4-aq4_0_target` | `a3f000ee1f3b66cc98f16fda3be2c436e366a3096afdc7fd58293d760bdb1182` | `62fad1e1a490a4c5bee00f36c6be4d0fddd7e8206702658307b72c43befa0b3a` |

### CPU-only実行準備

- `tools/run-aq4-phase7-source-oracles.sh`は`CUDA_VISIBLE_DEVICES=-1`、`HIP_VISIBLE_DEVICES=-1`、`ROCR_VISIBLE_DEVICES=-1`を空環境へ明示し、source modelとCPU BF16だけを使う。service、systemd、lock、active manifestは参照しない。
- source oracleはformal calibration 24件とformal holdout execution input 24件のfull hidden/logit vectorsをcreate-newで出力する。holdout source vectorはfreeze前に密封する入力であり、target比較・threshold導出は行わない。
- clean source worktree `/home/homelab1/coding-local/ultimateLLM/uLLM-phase6-clean-source`がcommit `d3ea48d543456a07a2796ee804671c3da513c268`でclean、かつ`e992b3ea1d0427744dfd83abdc98283a74c1e3b4`をancestorとして含むことを確認した。`CARGO_TARGET_DIR=/home/homelab1/coding-local/ultimateLLM/uLLM-phase7-build-target`、`CARGO_BUILD_JOBS=1`で`ullm-aq4-fidelity-capture`をbuildした。Cargo output SHA-256は`1600f3eae8d129683e2c111fa7f16810e5f7665aa09b83098d45e1a9b18362a0`である。
- `fidelity-capture-binary-staging/ullm-aq4-fidelity-capture`はcontent copyで、SHA-256一致、mode `0555`、`nlink=1`である。receiptと`SHA256SUMS`はmode `0444`、`nlink=1`で検証済みである。
- host-only R9700 guard binaryは`query-hip-device-identity`へcompileした。実行はしておらず、mode `0775`、`nlink=1`、SHA-256 `e85043b1bc1812a1b0ebcba31fcfa0bff5402be348d713a37f44643d9885175d`である。
- 新規比較器はtoken agreement、top-k overlap、logits cosine / relative-L2 / max-abs、hidden cosine / relative-L2 / max-abs、固定品質率を全24 holdout rowから計算し、freeze receiptとsource/target/split/identityのhash chainを再検証する。

### root-only guard rehearsal（親エージェント用）

source-oraclesがvalidになった後、serviceが稼働中のまま、下の3 commandを一回ずつ実行する。各`r9700-guard-rehearsal-summary.json`が`status=valid`でなければfinal windowへ進まない。

```bash
sudo /usr/bin/python3 /home/homelab1/coding-local/ultimateLLM/uLLM-project/tools/run-aq4-phase3c-r9700-guard.py --output /home/homelab1/coding-local/ultimateLLM/uLLM-project/benchmarks/results/2026-07-17/qwen35-9b-aq4-production-opt-v0.1/p2/aq4-phase7-p2-fidelity-preparation-v0.1/guard-chain-rehearsal-v0.1/attempt-1 --guard-bin /home/homelab1/coding-local/ultimateLLM/uLLM-project/benchmarks/results/2026-07-17/qwen35-9b-aq4-production-opt-v0.1/p2/aq4-phase7-p2-fidelity-preparation-v0.1/query-hip-device-identity --health-phase rehearsal-1
sudo /usr/bin/python3 /home/homelab1/coding-local/ultimateLLM/uLLM-project/tools/run-aq4-phase3c-r9700-guard.py --output /home/homelab1/coding-local/ultimateLLM/uLLM-project/benchmarks/results/2026-07-17/qwen35-9b-aq4-production-opt-v0.1/p2/aq4-phase7-p2-fidelity-preparation-v0.1/guard-chain-rehearsal-v0.1/attempt-2 --guard-bin /home/homelab1/coding-local/ultimateLLM/uLLM-project/benchmarks/results/2026-07-17/qwen35-9b-aq4-production-opt-v0.1/p2/aq4-phase7-p2-fidelity-preparation-v0.1/query-hip-device-identity --health-phase rehearsal-2
sudo /usr/bin/python3 /home/homelab1/coding-local/ultimateLLM/uLLM-project/tools/run-aq4-phase3c-r9700-guard.py --output /home/homelab1/coding-local/ultimateLLM/uLLM-project/benchmarks/results/2026-07-17/qwen35-9b-aq4-production-opt-v0.1/p2/aq4-phase7-p2-fidelity-preparation-v0.1/guard-chain-rehearsal-v0.1/attempt-3 --guard-bin /home/homelab1/coding-local/ultimateLLM/uLLM-project/benchmarks/results/2026-07-17/qwen35-9b-aq4-production-opt-v0.1/p2/aq4-phase7-p2-fidelity-preparation-v0.1/query-hip-device-identity --health-phase rehearsal-3
```

### final single service-stop window（親エージェント用）

3 rehearsal成功後にだけ、次を一回だけ実行する。driverは1 stop / 1 start、R9700 architecture + ASIC guard、既存lockのread-only probe、`RuntimeDirectoryPreserve=yes`、17 required guardが30 active guardのsubsetであること、nlink=1 stagingを検証する。calibration target 24件を一回loadしてfreezeし、同じ停止window内でholdout target 24件を一回loadする。`path-oracle-export`は実行しない。

```bash
sudo /home/homelab1/coding-local/ultimateLLM/uLLM-project/tools/run-aq4-phase7-service-window.sh /home/homelab1/coding-local/ultimateLLM/uLLM-project/benchmarks/results/2026-07-17/qwen35-9b-aq4-production-opt-v0.1/p2/aq4-phase7-p2-fidelity-preparation-v0.1 /home/homelab1/coding-local/ultimateLLM/uLLM-project/benchmarks/results/2026-07-17/qwen35-9b-aq4-production-opt-v0.1/p2/aq4-phase7-p2-fidelity-preparation-v0.1/query-hip-device-identity /home/homelab1/coding-local/ultimateLLM/uLLM-phase6-clean-source d3ea48d543456a07a2796ee804671c3da513c268 --confirm-single-window
```

期待成果物は`source-oracles/{calibration,holdout}/`、`target-calibration/`、`calibration-metrics.json`、`calibration-metrics-validation.json`、`freeze-receipt.json`、`target-holdout/`、`holdout-evaluation.json`、guard/lock/service restore evidenceである。判定は`holdout-evaluation.json.status`の凍結policy Go/No-Goと、`formal_p2_status`のcontract解決状態を分けて読む。

## 次の行動

1. 実行中のCPU BF16 source oracleを完走・validatorで確認し、source artifactのmanifest SHA-256、row数、nonfinite数、runtimeがCPU/BF16であることをjournalへ追記する。
2. source/oracleとbinary stagingのCPU-only最終検証、script executable bit、test suiteを再確認する。GPU guard rehearsalとservice-stop windowはこのタスクでは実行しない。
3. 親エージェントは上の3 rehearsalが全てvalidであることを確認してから、single-window commandを一回だけ実行する。旧07/15 splitやP3/Phase 3c/Phase 6の既存window evidenceを再利用・再実行しない。
