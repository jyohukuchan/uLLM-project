# SQ8_0 Qwen3-14B-FP8 production promotion runbook v0.1

Status (2026-07-24): **cutover is not authorized yet.** The immutable SQ8_0
product artifact and the complete 2026-07-12 historical campaign have both
been revalidated, but that campaign is bound to an unavailable historical
worker binary and a legacy worker invocation. It is not evidence for the
current manifest-capable SQ8 worker. This runbook therefore stops before an
`active.json` change until a new candidate identity and candidate-active
campaign are explicitly authorized.

This concerns the independent `SQ8_0` FP8 E4M3 format for
`Qwen/Qwen3-14B-FP8`. It is not AQ4_0 with FP8 applied to selected tensors.
Keep this promotion distinct from AQ4_0におけるFP8の部分適用.

Every command that starts, stops, restarts, or observes the production
service, R9700, or OpenWebUI is marked **parent-only**. It was deliberately
not run while preparing this handoff. Nothing in this document targets V620.

## 1. Scope and fixed identities

| Item | Fixed value or current observation |
|---|---|
| Target public ID | `ullm-qwen3-14b-sq8` |
| Target format | `SQ8_0`, implementation `qwen3_sq8_rdna4_v1` |
| Upstream model | `Qwen/Qwen3-14B-FP8` at `9a283b4a5efbc09ce247e0ae5b02b744739e525a` |
| Product root | `/home/homelab1/datapool/ullm/product/qwen3-14b-fp8-sq8-v0.1` |
| Product receipt SHA-256 | `f2d7b25d40adb29dae225626a87129dc5495c1251ee45563bcf4e6660cca8016` |
| Artifact manifest SHA-256 | `23977f4e9bed4bac4cc64c177c35d7f83355861426bf32027a69cf7a241552e2` |
| Artifact content SHA-256 | `2243acf1df627ff6ec13840c8ffcf35c77e89205eb36cef7561b85c9c98b9147` |
| Artifact selection | 280 tensor pairs; 13,213,670,400 payload bytes; 561 files |
| Package manifest SHA-256 | `c2133dfe392f3d5608bde17ed764ae8347c3096c500a58aa235adbeb63d1a0eb` |
| Package payload | 163 files; 3,112,499,200 bytes |
| Historical full-campaign source commit | `f647a8aa8a78ddf0846b92fa700b8fa5a0995887` |
| Historical worker SHA-256 | `145a5351db3957130200276314853e394d0fd206a69e2eab260c01141411b950` |
| Current SQ8 worker SHA-256 | `3ecc7915265e574e146f8c3d8d5de405459bdcb12c035f50e56e948da37d0454` |
| Current SQ8 worker size | 7,707,400 bytes |
| Current active manifest | `/etc/ullm/served-models/active.json` |
| Current active manifest SHA-256 | `5d015a013dcf70cea13dd9ed569d89ed2a025a17e14a6192ca18ee4cdadd1c8a` |
| Current active format / protocol | `AQ4_0` / `ullm.worker.v2` |
| Current active promotion source | `0cd760568e197e1adb4c4df3d6149591a912f709` |

The exact current active bytes are the rollback target. Do not use a stale
named candidate as a substitute for those bytes.

## 2. Immutable artifact and 2026-07-12 evidence

The product artifact itself is complete and read-only. Re-run the full hash
check before any new candidate is made:

```bash
set -euo pipefail

REPO=/home/homelab1/coding-local/ultimateLLM/uLLM-project
PRODUCT=/home/homelab1/datapool/ullm/product/qwen3-14b-fp8-sq8-v0.1

cd "$REPO"
python3 tools/validate-sq8-product-promotion.py "$PRODUCT"
```

It must report `verified: true`, `full_payloads: true`, and the hashes in
section 1. A metadata-only pass is not a substitute for this full payload
hashing pass.

The complete historical campaign is retained here:

```text
/home/homelab1/datapool/ullm/sq8-openwebui-product-20260712-v0.1
```

Its `SHA256SUMS` has SHA-256
`7ae6a710fb3b83dcfe1c7a71f1699a1f01927b330dd4b7d01e9ea7409be60a45`.
The repository mirror at
`benchmarks/results/2026-07-12/sq8-openwebui-product-20260712-v0.1` is a
thin mirror: it omits `raw-session-results.jsonl` and
`service-journal.raw.jsonl`. It must not be used as a standalone release
input, even though its remaining files match.

Validate the complete archive and reproduce the independent decision in an
isolated copy. `release-validation.json` is a validator output rather than an
input, so retain the old one outside the copied bundle before recomputing it.

```bash
set -euo pipefail

REPO=/home/homelab1/coding-local/ultimateLLM/uLLM-project
HISTORICAL_COMMIT=f647a8aa8a78ddf0846b92fa700b8fa5a0995887
HISTORICAL_WORKER_SHA=145a5351db3957130200276314853e394d0fd206a69e2eab260c01141411b950
HISTORICAL_BUNDLE=/home/homelab1/datapool/ullm/sq8-openwebui-product-20260712-v0.1

( cd "$HISTORICAL_BUNDLE" && sha256sum -c SHA256SUMS )

SOURCE_TREE=$(mktemp -d /tmp/ullm-sq8-historical-source-XXXXXX)
rmdir "$SOURCE_TREE"
git -C "$REPO" worktree add --detach "$SOURCE_TREE" "$HISTORICAL_COMMIT"

VALIDATION_ROOT=$(mktemp -d /tmp/ullm-sq8-historical-validation-XXXXXX)
cp -a --reflink=auto "$HISTORICAL_BUNDLE"/. "$VALIDATION_ROOT/bundle"
mv "$VALIDATION_ROOT/bundle/release-validation.json" \
  "$VALIDATION_ROOT/release-validation.preexisting.json"

python3 "$REPO/tools/validate-sq8-openwebui-release.py" \
  --expected-commit "$HISTORICAL_COMMIT" \
  --expected-worker-binary-sha256 "$HISTORICAL_WORKER_SHA" \
  --repo-root "$SOURCE_TREE" \
  "$VALIDATION_ROOT/bundle"

test "$(sha256sum "$VALIDATION_ROOT/bundle/release-validation.json" | awk '{print $1}')" = \
  4261908bb53881a7737f54251235284d7c4069659d65851a0c023447d0cf8e08
```

The expected independent result is `release_status: complete`, including 21
successful browser requests, five cancellation phases, 100 normal plus 20
restart resource requests, 72 latency requests, 610 resource samples, and
the planned service restart count transition 7 to 8.

### 2.1 Why the historical complete result cannot activate today

The 2026-07-12 evidence is trustworthy for its own immutable identities. It
does not bind the current candidate because all of the following differ:

- The recorded worker is `145a...b950`; that binary is no longer available in
  the staged-worker locations. The current worker is `3ecc...d045`.
- At `f647a8...`, `ullm-sq8-worker` accepted only legacy
  `--artifact PATH --package PATH` startup. The current worker additionally
  implements `--served-model-manifest PATH`, which the current service uses.
- The historical environment records the legacy worker invocation, while the
  current candidate profile uses the served-model-manifest invocation.

Rebuilding or locating `145a...b950` would preserve archival evidence only;
it would not make that old worker compatible with the present manifest-mode
service. Do not overwrite the current worker with it.

## 3. Current candidate dry-run and the decision gate

The existing profile is:

```text
deploy/served-models/qwen3-14b-sq8.profile.json
```

It currently emits `ullm.served_model.v1`, `ullm.worker.v1`, and records the
short product-plan commit `dfc63de` (full commit
`dfc63de724f945517d398791f23c273fdb1c484a`). A dry-run generated manifest
was valid but had the following distinct identity:

```text
manifest SHA-256: 0b5f7fe4fa201c799d2834ba7aee8efa5b76cc8d26e7e8b221e1dd72ed0a97b3
worker SHA-256:   3ecc7915265e574e146f8c3d8d5de405459bdcb12c035f50e56e948da37d0454
```

Reproduce that non-installing dry-run as follows. Its output must remain
outside `/etc/ullm/served-models`.

```bash
set -euo pipefail

REPO=/home/homelab1/coding-local/ultimateLLM/uLLM-project
DRY_RUN_ROOT=$(mktemp -d /tmp/ullm-sq8-manifest-dryrun-XXXXXX)
CANDIDATE="$DRY_RUN_ROOT/qwen3-14b-sq8.json"

cd "$REPO"
python3 tools/generate-served-model.py \
  --profile deploy/served-models/qwen3-14b-sq8.profile.json \
  --output "$CANDIDATE"
python3 tools/validate-served-model.py --manifest "$CANDIDATE"
sha256sum "$CANDIDATE"
```

Do not install that current v1 candidate or treat the historical campaign as
its release bundle. Before a candidate-active service window, the parent must
choose and record one of these paths:

1. **v1 SQ8 route:** explicitly approve a temporary/final v1 activation
   policy, create a clean source/worker release identity, and collect a fresh
   complete SQ8 campaign bound to that exact worker and candidate manifest.
   This is an exception to the v2 bundle-gated AQ4 route and must be reviewed
   as such.
2. **v2 release route (preferred for parity with current AQ4):** implement and
   review a SQ8 v2 manifest/receipt/release-bundle path, including a full
   40-character promotion source commit and an identity-matching worker. Only
   then use normal `--release-bundle` activation.

No command in this runbook infers that approval. Until one route is selected,
the correct result is no `active.json` change.

## 4. Browser-gate session authentication

There are two different credentials, and they must remain separate:

| Purpose | File contract | Permitted use |
|---|---|---|
| Gateway OpenAI API authentication | `/etc/ullm/openai-api-key`, root:`homelab1`, mode `0640` | API contract, direct-cancel, latency, and `/v1` checks |
| OpenWebUI frontend login | parent-provisioned private file, execution user, mode `0600` | Browser soak, Stop, failure, and browser smoke only |

The second file must contain a real OpenWebUI frontend session JWT. It is not
the OpenWebUI signing key and it is never `/etc/ullm/openai-api-key`. The
updated runners reject a non-JWT value, require an `alg` header and integer
`exp`, and reject a token too close to expiry for each gate. They do not print
the token. OpenWebUI itself remains the authority that validates its signature
and session claims.

The exposed interfaces are now:

```text
tools/run-openwebui-soak-gate.py --openwebui-session-token-file PATH
tools/run-openwebui-stop-gate.py --openwebui-session-token-file PATH
tools/run-openwebui-failure-gate.py --openwebui-session-token-file PATH
tools/run-openwebui-reasoning-browser-smoke.py --openwebui-session-token-file PATH
tools/run-sq8-full-openwebui-campaign.py --openwebui-session-token-file PATH
```

The browser containers receive that file only as
`/run/secrets/openwebui-session-token` through
`OPENWEBUI_SESSION_TOKEN_FILE`. The gateway API key is not mounted there.

Before a real campaign, the parent may check only metadata, without printing
the JWT:

```bash
test "$OPENWEBUI_SESSION_TOKEN_FILE" != /etc/ullm/openai-api-key
test -f "$OPENWEBUI_SESSION_TOKEN_FILE"
test "$(stat -c '%a' "$OPENWEBUI_SESSION_TOKEN_FILE")" = 600
test "$(stat -c '%U' "$OPENWEBUI_SESSION_TOKEN_FILE")" = "$(id -un)"
```

## 5. Required SQ8 campaign gates

Run the full orchestrator rather than composing a partial set of browser
gates. Its fixed order binds the same candidate identity through all of these
components:

| Stage | Gate and independent evidence |
|---|---|
| API contract | `run-sq8-api-contract-gate.py` and its ingestor |
| Browser smoke and soak | `run-openwebui-soak-gate.py --include-smoke` and `sq8_openwebui_gate_ingest.py` |
| Direct cancellation | `run-sq8-direct-cancel-gate.py` and its ingestor |
| Browser Stop/recovery | `run-openwebui-stop-gate.py` and `sq8_openwebui_stop_gate_ingest.py` |
| Post-header failure/recovery | `run-openwebui-failure-gate.py` and its ingestor |
| Latency | `run-sq8-http-latency-gate.py` and its ingestor |
| Resource stability | normal and restart collectors, then independent campaign validation |

`sq8_openwebui_stop_gate_ingest.py` does not authenticate a browser. It
independently reconstructs the Stop evidence bundle after the gate has run;
therefore it cannot compensate for a wrong session token.

After the parent has selected a route, prepared a clean detached source tree,
and installed the exact candidate for a temporary candidate-active campaign,
the parent-only command is:

```bash
set -euo pipefail

SOURCE_TREE=/absolute/clean/sq8-release-source
SOURCE_COMMIT=<approved-40-character-commit-including-the-session-auth-fix>
WORKER=/absolute/staged/ullm-sq8-worker
WORKER_SHA=<SHA-256-of-WORKER>
RUN_ID=sq8-openwebui-production-<date>-v0.1
OUT=/absolute/new-empty-release-directory
GATEWAY_API_KEY_FILE=/etc/ullm/openai-api-key
OPENWEBUI_SESSION_TOKEN_FILE=/absolute/private/openwebui-session.jwt

test "$(git -C "$SOURCE_TREE" rev-parse HEAD)" = "$SOURCE_COMMIT"
test -z "$(git -C "$SOURCE_TREE" status --porcelain --untracked-files=all)"
test "$(sha256sum "$WORKER" | awk '{print $1}')" = "$WORKER_SHA"
test ! -e "$OUT"

cd "$SOURCE_TREE"
python3 tools/run-sq8-full-openwebui-campaign.py \
  --execute \
  --expected-commit "$SOURCE_COMMIT" \
  --expected-worker-binary-sha256 "$WORKER_SHA" \
  --run-id "$RUN_ID" \
  --final-path "$OUT" \
  --api-key-file "$GATEWAY_API_KEY_FILE" \
  --openwebui-session-token-file "$OPENWEBUI_SESSION_TOKEN_FILE"

python3 tools/validate-sq8-openwebui-release.py \
  --expected-commit "$SOURCE_COMMIT" \
  --expected-worker-binary-sha256 "$WORKER_SHA" \
  --repo-root "$SOURCE_TREE" \
  "$OUT"
```

This is a real service/GPU/OpenWebUI operation: it intentionally exercises
Stop and post-header failure/recovery. It is parent-only and was not run for
this handoff. It must finish with `release_status: complete`; any incomplete,
identity-mismatched, or expired-session result invalidates the candidate.

## 6. Parent-only `active.json` activation and rollback

This section is executable only after section 3's policy choice and a fresh,
identity-matching complete campaign. It documents the actual switch mechanics
so that the operator does not hand-edit `active.json`.

First freeze the current rollback bytes and verify the candidate outside the
active path:

```bash
set -euo pipefail

ACTIVE=/etc/ullm/served-models/active.json
ACTIVE_SHA256=5d015a013dcf70cea13dd9ed569d89ed2a025a17e14a6192ca18ee4cdadd1c8a
CANDIDATE_ETC=/etc/ullm/served-models/candidates/qwen3-14b-sq8-<release>.json
BACKUP=/etc/ullm/served-models/active.json.before-sq8-<release>

test "$(sha256sum "$ACTIVE" | awk '{print $1}')" = "$ACTIVE_SHA256"
sudo /usr/bin/python3 "$SOURCE_TREE/tools/validate-served-model.py" \
  --manifest "$CANDIDATE_ETC"
sudo install -m 0644 -o root -g root "$ACTIVE" "$BACKUP"
test "$(sha256sum "$BACKUP" | awk '{print $1}')" = "$ACTIVE_SHA256"
```

For an explicitly approved v1 SQ8 route, use the activation tool rather than
`cp` or an editor. The service restart and OpenWebUI reconciliation below are
intentionally parent-only. Replace image identities only after a separate
identity check.

```bash
OPENWEBUI_IMAGE=ullm/open-webui:0.9.4-ullm.1
OPENWEBUI_BASE_IMAGE=ghcr.io/open-webui/open-webui@sha256:a6da0c292081d810a396ce786a10536d0b1b9ba2925dcca20ebb03f9fa90dbff

sudo /usr/bin/python3 "$SOURCE_TREE/tools/activate-served-model.py" \
  --candidate "$CANDIDATE_ETC" \
  --active-manifest "$ACTIVE" \
  --command-timeout-seconds 650 \
  --check-command-json '["/usr/bin/systemctl","restart","ullm-openai.service"]' \
  --check-command-json "[\"/usr/bin/docker\",\"run\",\"--rm\",\"--network\",\"open-webui-network\",\"--entrypoint\",\"curl\",\"$OPENWEBUI_BASE_IMAGE\",\"--fail\",\"--silent\",\"--show-error\",\"--retry\",\"120\",\"--retry-delay\",\"2\",\"--retry-max-time\",\"600\",\"--retry-all-errors\",\"http://172.20.0.1:8000/readyz\"]" \
  --reconcile-command-json '["/usr/bin/docker","stop","open-webui"]' \
  --reconcile-command-json "[\"/usr/bin/docker\",\"run\",\"--rm\",\"-v\",\"open-webui:/data\",\"-v\",\"/etc/ullm/openai-api-key:/run/secrets/ullm-api-key:ro\",\"-v\",\"/etc/ullm/served-models:/etc/ullm/served-models:ro\",\"-v\",\"$SOURCE_TREE/deploy/openwebui/configure.py:/configure.py:ro\",\"--entrypoint\",\"python\",\"$OPENWEBUI_IMAGE\",\"/configure.py\",\"--served-model-manifest\",\"/etc/ullm/served-models/active.json\",\"--base-url\",\"http://172.20.0.1:8000/v1\"]" \
  --reconcile-command-json "[\"/usr/bin/docker\",\"compose\",\"-f\",\"$SOURCE_TREE/deploy/openwebui/compose.yaml\",\"up\",\"-d\",\"--no-build\"]" \
  --final-check-command-json '["/usr/bin/curl","--fail","--silent","--show-error","--retry","30","--retry-delay","2","--retry-connrefused","http://127.0.0.1:3000/health"]' \
  --rollback-command-json '["/usr/bin/systemctl","restart","ullm-openai.service"]'
```

For a future v2 SQ8 route, add the independently validated
`--release-bundle "$BUNDLE"`, `--systemd-unit`, and `--environment-file`
arguments required by `activate-served-model.py`; do not use the v1 command
as a way to bypass that v2 release-bundle contract.

If any activation check or reconciliation command fails, the activation tool
restores the exact previous active bytes before running rollback hooks. If a
later human decision requires manual rollback after a completed activation,
the parent must first compare the backup hash, then restore it and reconcile:

```bash
test "$(sha256sum "$BACKUP" | awk '{print $1}')" = "$ACTIVE_SHA256"
sudo install -m 0644 -o root -g root "$BACKUP" "$ACTIVE"
sudo systemctl restart ullm-openai.service
# Re-run the same configure.py and docker compose reconciliation commands above.
```

## 7. Pre- and post-activation health checks

Before the switch, record these without modifying the active manifest:

```bash
sudo /usr/bin/python3 "$SOURCE_TREE/tools/validate-served-model.py" --manifest "$ACTIVE"
sha256sum "$ACTIVE"
docker run --rm --network open-webui-network --entrypoint curl \
  "$OPENWEBUI_BASE_IMAGE" --fail --silent --show-error \
  http://172.20.0.1:8000/readyz
curl --fail --silent --show-error http://127.0.0.1:3000/health
```

After a successful activation, all of the following must hold before the
window is closed:

1. `validate-served-model.py --manifest "$ACTIVE"` reports `format_id: SQ8_0`,
   public ID `ullm-qwen3-14b-sq8`, and the exact fresh candidate worker
   SHA-256.
2. Gateway `/readyz` and OpenWebUI `/health` both return success after the
   reconciliation commands complete.
3. An authenticated `/v1/models` check made with the gateway API credential
   lists `ullm-qwen3-14b-sq8`; use an approved secret-safe client, not an
   inline shell substitution that exposes the credential.
4. A real OpenWebUI browser check selects the SQ8_0 model and produces the
   expected model identity. It must use the separate session JWT, not the
   gateway API key.
5. The full campaign output and independent validator still bind the active
   manifest, worker, product artifact, source commit, and OpenWebUI image.

Any failure in these checks requires the rollback procedure above. Do not
leave a mixed active-manifest/OpenWebUI configuration state.

## 8. Dry-run record for this handoff

Completed without changing `/etc/ullm/served-models/active.json`, restarting
`ullm-openai.service`, invoking systemd, or using a GPU:

- The active AQ4_0 v2 manifest was read and validated; its hash is recorded
  in section 1.
- `validate-sq8-product-promotion.py` completed full artifact and package
  payload hashing successfully.
- The datapool full 2026-07-12 archive passed all 19 `SHA256SUMS` entries.
  A clean detached `f647a8...` source checkout reproduced the independent
  `complete` decision and the exact validator SHA-256
  `4261908bb53881a7737f54251235284d7c4069659d65851a0c023447d0cf8e08`.
- The historical repository mirror was correctly rejected as incomplete due
  to its two omitted raw files.
- A temporary current v1 candidate manifest was generated and validated; it
  proves current product/profile mechanics but also proves the worker and
  manifest identity mismatch described in section 2.1.
- On a private temporary copy only, `activate-served-model.py` successfully
  performed the v2-to-v1 SQ8_0 manifest replacement and validated the result.
  A separate forced failing check (`/usr/bin/false`) returned nonzero and
  restored the copied AQ4 active bytes exactly. In both cases the real active
  manifest remained SHA-256
  `5d015a013dcf70cea13dd9ed569d89ed2a025a17e14a6192ca18ee4cdadd1c8a`.
- Browser-gate unit tests cover rejection of a gateway API-key-shaped value
  and acceptance of a structurally valid, unexpired session JWT. No real
  session credential was read or created.
- The historical formal Stop/failure pilots remain pinned to their old gate
  and browser source SHA-256 values. The source-binding tests now explicitly
  require a fresh pilot instead of treating those immutable old pilots as
  validation of this new gate source.

The only remaining actions are policy/identity work and parent-controlled
candidate-active service execution. The safe dry-run result is therefore
**not ready for production cutover**.
