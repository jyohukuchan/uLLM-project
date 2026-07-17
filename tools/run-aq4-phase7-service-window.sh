#!/usr/bin/env bash
# Consume exactly one approved AQ4 Phase 7 full-fidelity service-stop window.
#
# CPU source vectors, the clean-source capture binary, its nlink=1 stage, and
# three successful R9700 read-only guard rehearsals must already exist.  This
# driver performs one service stop only: target calibration -> freeze -> target
# formal holdout all occur in that same stopped interval.  It never invokes the
# Phase 6 path-oracle exporter, so it does not alter or work around that
# tool's separate symlink-guard issue.

set -u -o pipefail
umask 077

if [ "$#" -ne 5 ] || [ "$5" != "--confirm-single-window" ]; then
  echo "usage: $0 PREPARATION_ROOT HIP_GUARD_BINARY SOURCE_WORKTREE SOURCE_COMMIT --confirm-single-window" >&2
  exit 64
fi
if [ "$(id -u)" -ne 0 ]; then
  echo "this driver must run as root" >&2
  exit 65
fi

OUT=$1
GUARD_BIN=$2
SOURCE_TREE=$3
SOURCE_COMMIT=$4
REPO=/home/homelab1/coding-local/ultimateLLM/uLLM-project
BUILD_TARGET=/home/homelab1/coding-local/ultimateLLM/uLLM-phase7-build-target
SOURCE_BIN="$BUILD_TARGET/release/ullm-aq4-fidelity-capture"
STAGE_DIR="$OUT/fidelity-capture-binary-staging"
CAPTURE_BIN="$STAGE_DIR/ullm-aq4-fidelity-capture"
FORMAL_SPLIT="$OUT/formal-split"
EXECUTION_VIEW="$OUT/holdout-execution-view"
SOURCE_CALIBRATION="$OUT/source-oracles/calibration"
SOURCE_HOLDOUT="$OUT/source-oracles/holdout"
CALIBRATION_CASES="$OUT/source-cases/calibration-cases.json"
HOLDOUT_CASES="$OUT/source-cases/holdout-execution-cases.json"
MANIFEST=/etc/ullm/served-models/active.json
LOCK=/run/ullm/r9700.lock
SERVICE=ullm-openai.service
REQUIRED_FIX_COMMIT=e992b3ea1d0427744dfd83abdc98283a74c1e3b4
EXPECTED_SOURCE_COMMIT=d3ea48d543456a07a2796ee804671c3da513c268
STAGING_TOOL="$REPO/tools/stage-aq4-phase7-fidelity-capture-binary.py"
PREPARATION_TOOL="$REPO/tools/prepare-qwen35-aq4-phase7-fidelity.py"
SOURCE_VALIDATE_TOOL="$REPO/tools/validate-qwen35-aq4-p2-full-calibration.py"
CAPTURE_METRICS_TOOL="$REPO/tools/capture-qwen35-aq4-fidelity.py"
CAPTURE_METRICS_VALIDATE_TOOL="$REPO/tools/validate-qwen35-aq4-fidelity-capture.py"
FREEZE_TOOL="$REPO/tools/generate-aq4-p2-fidelity-holdout.py"
HOLDOUT_EVALUATOR="$REPO/tools/evaluate-qwen35-aq4-phase7-holdout.py"
GUARD_TOOL="$REPO/tools/run-aq4-phase3c-r9700-guard.py"
LOCK_PROBE="$REPO/tools/probe-aq4-phase3c-existing-lock.py"

# The Phase 3c v0.7 set remains a mandatory subset of the active production
# guard set.  The capture child receives all 30 active guards under env -i.
PHASE3C_REQUIRED_GUARDS=(
  ULLM_REQUIRE_HIP_AQ4_MATVEC_KERNEL
  ULLM_REQUIRE_HIP_AQ4_MATVEC_BATCH_KERNEL
  ULLM_REQUIRE_HIP_AQ4_MATVEC_ADD_KERNEL
  ULLM_REQUIRE_HIP_AQ4_MATVEC_QKV_Z_GATE_BETA_KERNEL
  ULLM_REQUIRE_HIP_LINEAR_ATTN_KERNEL
  ULLM_REQUIRE_HIP_LINEAR_ATTN_QKV_PREPARE_BATCH_KERNEL
  ULLM_REQUIRE_HIP_LINEAR_ATTN_RECURRENT_KERNEL
  ULLM_REQUIRE_HIP_RMSNORM_KERNEL
  ULLM_REQUIRE_HIP_SEGMENTED_RMSNORM_SILU_MUL_KERNEL
  ULLM_REQUIRE_HIP_AQ4_REGISTER_BM8_KERNEL
  ULLM_REQUIRE_HIP_PAGED_DECODE_ATTN_KERNEL
  ULLM_REQUIRE_HIP_QWEN35_QK_NORM_ROPE_PAGED_KV_WRITE_KERNEL
  ULLM_REQUIRE_HIP_PAGED_KV_WRITE_CHUNK_KERNEL
  ULLM_REQUIRE_HIP_PAGED_CAUSAL_GQA_CHUNK_KERNEL
  ULLM_REQUIRE_HIP_QWEN35_QK_NORM_ROPE_BATCH_KERNEL
  ULLM_REQUIRE_HIP_BF16_ROW_KERNEL
  ULLM_REQUIRE_HIP_TOP1_KERNEL
)

PHASE7_REQUIRED_GUARDS=(
  ULLM_REQUIRE_HIP_AQ4_KERNEL
  ULLM_REQUIRE_HIP_AQ4_MATVEC_KERNEL
  ULLM_REQUIRE_HIP_AQ4_MATVEC_BATCH_KERNEL
  ULLM_REQUIRE_HIP_AQ4_REGISTER_BM8_KERNEL
  ULLM_REQUIRE_HIP_AQ4_MATVEC_ADD_KERNEL
  ULLM_REQUIRE_HIP_AQ4_MATVEC_PAIR_KERNEL
  ULLM_REQUIRE_HIP_AQ4_MATVEC_TRIPLE_KERNEL
  ULLM_REQUIRE_HIP_AQ4_MATVEC_QKV_Z_GATE_BETA_KERNEL
  ULLM_REQUIRE_HIP_ADD_KERNEL
  ULLM_REQUIRE_HIP_BF16_MATVEC_KERNEL
  ULLM_REQUIRE_HIP_BF16_ROW_KERNEL
  ULLM_REQUIRE_HIP_LINEAR_ATTN_GATE_BETA_KERNEL
  ULLM_REQUIRE_HIP_LINEAR_ATTN_KERNEL
  ULLM_REQUIRE_HIP_LINEAR_ATTN_QKV_PREPARE_BATCH_KERNEL
  ULLM_REQUIRE_HIP_LINEAR_ATTN_RECURRENT_KERNEL
  ULLM_REQUIRE_HIP_LINEAR_ATTN_RECURRENT_SEQUENCE_KERNEL
  ULLM_REQUIRE_HIP_PAGED_KV_WRITE_CHUNK_KERNEL
  ULLM_REQUIRE_HIP_PAGED_CAUSAL_GQA_CHUNK_KERNEL
  ULLM_REQUIRE_HIP_PAGED_DECODE_ATTN_KERNEL
  ULLM_REQUIRE_HIP_PAGED_DECODE_SPLIT_KERNEL
  ULLM_REQUIRE_HIP_PAGED_KV_WRITE_KERNEL
  ULLM_REQUIRE_HIP_QWEN35_Q_SPLIT_KERNEL
  ULLM_REQUIRE_HIP_QWEN35_QK_NORM_ROPE_BATCH_KERNEL
  ULLM_REQUIRE_HIP_QWEN35_QK_NORM_ROPE_PAGED_KV_WRITE_KERNEL
  ULLM_REQUIRE_HIP_RMSNORM_KERNEL
  ULLM_REQUIRE_HIP_ROPE_KERNEL
  ULLM_REQUIRE_HIP_SEGMENTED_RMSNORM_SILU_MUL_KERNEL
  ULLM_REQUIRE_HIP_SIGMOID_MUL_KERNEL
  ULLM_REQUIRE_HIP_SILU_MUL_KERNEL
  ULLM_REQUIRE_HIP_TOP1_KERNEL
)

PHASE7_ENV=(
  HOME=/home/homelab1
  PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
  HIP_VISIBLE_DEVICES=1
  ULLM_HIP_VISIBLE_DEVICES=1
  ULLM_BUILD_GIT_COMMIT="$SOURCE_COMMIT"
  ULLM_AQ4_MATVEC_QKV_Z_GATE_BETA_RPB=4
  ULLM_AQ4_MATVEC_SILU_MUL_RPB=8
  ULLM_AQ4_MATVEC_ADD_RPB=8
)
for guard in "${PHASE7_REQUIRED_GUARDS[@]}"; do
  PHASE7_ENV+=("$guard=1")
done

WINDOW_ARTIFACTS=(
  "$OUT/preparation-verify-window.json"
  "$OUT/source-calibration-verify-window.json"
  "$OUT/source-holdout-verify-window.json"
  "$OUT/active-pins.json"
  "$OUT/fidelity-capture-binary-staging-verify-window.json"
  "$OUT/phase7-guard-contract.json"
  "$OUT/service-window-events.tsv"
  "$OUT/service-window-result.txt"
  "$OUT/service-window-pre-stop.txt"
  "$OUT/service-window-stop.stdout"
  "$OUT/service-window-stop.stderr"
  "$OUT/service-window-stop-systemctl-show.txt"
  "$OUT/service-window-stop-systemctl-show.stderr"
  "$OUT/service-window-stop.txt"
  "$OUT/service-window-lock-after-stop.json"
  "$OUT/guard-before"
  "$OUT/target-calibration"
  "$OUT/calibration-metrics.json"
  "$OUT/calibration-metrics-validation.json"
  "$OUT/freeze-receipt.json"
  "$OUT/target-holdout"
  "$OUT/guard-after"
  "$OUT/service-window-start.stdout"
  "$OUT/service-window-start.stderr"
  "$OUT/service-window-start.txt"
  "$OUT/service-window-post-restore.txt"
  "$OUT/holdout-evaluation.json"
)

STOP_ATTEMPTED=0
RESTORE_DONE=0
RESTORE_RC=99
CALIBRATION_CAPTURE_RC=not_started
FREEZE_RC=not_started
HOLDOUT_CAPTURE_RC=not_started
AFTER_GUARD_RC=not_started
OUTCOME=not_started

event() {
  printf '%s\t%s\t%s\n' "$(date --iso-8601=seconds --utc)" "$1" "$2" >> "$OUT/service-window-events.tsv"
}

write_result() {
  printf '%s\n' \
    "recorded_at_utc=$(date --iso-8601=seconds --utc)" \
    "outcome=$OUTCOME" \
    "calibration_capture_exit_code=$CALIBRATION_CAPTURE_RC" \
    "freeze_exit_code=$FREEZE_RC" \
    "holdout_capture_exit_code=$HOLDOUT_CAPTURE_RC" \
    "post_trace_guard_exit_code=$AFTER_GUARD_RC" \
    "restore_start_exit_code=$RESTORE_RC" \
    > "$OUT/service-window-result.txt"
}

restore() {
  if [ "$STOP_ATTEMPTED" -ne 1 ] || [ "$RESTORE_DONE" -ne 0 ]; then
    return 0
  fi
  RESTORE_DONE=1
  local started_at finished_at
  started_at="$(date --iso-8601=seconds --utc)"
  event restore_start_invoked ""
  systemctl start "$SERVICE" > "$OUT/service-window-start.stdout" 2> "$OUT/service-window-start.stderr"
  RESTORE_RC=$?
  finished_at="$(date --iso-8601=seconds --utc)"
  printf '%s\n' "started_at_utc=$started_at" "finished_at_utc=$finished_at" "exit_code=$RESTORE_RC" > "$OUT/service-window-start.txt"
  event restore_start_returned "exit_code=$RESTORE_RC"
  return "$RESTORE_RC"
}

on_exit() {
  local code=$?
  trap - EXIT
  if [ "$STOP_ATTEMPTED" -eq 1 ] && [ "$RESTORE_DONE" -eq 0 ]; then
    restore
  fi
  exit "$code"
}
trap on_exit EXIT

if ! [[ "$SOURCE_COMMIT" =~ ^[0-9a-f]{40}$ ]] || [ "$SOURCE_COMMIT" != "$EXPECTED_SOURCE_COMMIT" ]; then
  echo "source commit must be the fixed clean Phase 6 commit $EXPECTED_SOURCE_COMMIT" >&2
  exit 66
fi
case "$OUT" in /*) ;; *) echo "preparation root must be an absolute path" >&2; exit 66 ;; esac
case "$SOURCE_TREE" in /*) ;; *) echo "source worktree must be an absolute path" >&2; exit 66 ;; esac
if [ ! -d "$OUT" ] || [ -L "$OUT" ]; then
  echo "preparation root contract failed" >&2
  exit 30
fi
for artifact in "${WINDOW_ARTIFACTS[@]}"; do
  if [ -e "$artifact" ] || [ -L "$artifact" ]; then
    echo "refusing to reuse a Phase 7 service-window output: $artifact" >&2
    exit 30
  fi
done
if [ ! -f "$LOCK" ] || [ -L "$LOCK" ] || [ ! -x "$CAPTURE_BIN" ] || [ ! -x "$GUARD_BIN" ] || [ ! -x /opt/rocm/bin/amd-smi ]; then
  echo "pre-stop fixed file contract failed" >&2
  exit 31
fi
for path in "$STAGING_TOOL" "$PREPARATION_TOOL" "$SOURCE_VALIDATE_TOOL" "$CAPTURE_METRICS_TOOL" "$CAPTURE_METRICS_VALIDATE_TOOL" "$FREEZE_TOOL" "$HOLDOUT_EVALUATOR" "$GUARD_TOOL" "$LOCK_PROBE" "$FORMAL_SPLIT/split-manifest.json" "$FORMAL_SPLIT/policy.json" "$FORMAL_SPLIT/calibration-cases.jsonl" "$EXECUTION_VIEW/split-manifest.json" "$EXECUTION_VIEW/policy.json" "$EXECUTION_VIEW/calibration-cases.jsonl" "$SOURCE_CALIBRATION/manifest.json" "$SOURCE_HOLDOUT/manifest.json" "$CALIBRATION_CASES" "$HOLDOUT_CASES"; do
  if [ ! -f "$path" ] || [ -L "$path" ]; then
    echo "pre-stop input contract failed: $path" >&2
    exit 32
  fi
done
if [ ! -d "$SOURCE_TREE" ] || [ -L "$SOURCE_TREE" ] || [ ! -d "$BUILD_TARGET" ] || [ -L "$BUILD_TARGET" ] || [ ! -x "$SOURCE_BIN" ] || [ ! -r "$MANIFEST" ]; then
  echo "source/build/manifest contract failed" >&2
  exit 32
fi
if [ "$(git -C "$SOURCE_TREE" rev-parse HEAD)" != "$SOURCE_COMMIT" ] || ! git -C "$SOURCE_TREE" diff --quiet || ! git -C "$SOURCE_TREE" diff --cached --quiet || [ -n "$(git -C "$SOURCE_TREE" status --porcelain --untracked-files=all)" ]; then
  echo "clean source worktree contract failed" >&2
  exit 34
fi
if ! git -C "$SOURCE_TREE" merge-base --is-ancestor "$REQUIRED_FIX_COMMIT" "$SOURCE_COMMIT"; then
  echo "source worktree does not include the final RMSNorm fix" >&2
  exit 34
fi
for tool in \
  tools/prepare-qwen35-aq4-phase7-fidelity.py \
  tools/stage-aq4-phase7-fidelity-capture-binary.py \
  tools/run-aq4-phase7-source-oracles.sh \
  tools/run-aq4-phase7-service-window.sh \
  tools/evaluate-qwen35-aq4-phase7-holdout.py \
  tools/capture-qwen35-aq4-fidelity.py \
  tools/validate-qwen35-aq4-fidelity-capture.py \
  tools/generate-aq4-p2-fidelity-holdout.py \
  tools/run-aq4-phase3c-r9700-guard.py \
  tools/probe-aq4-phase3c-existing-lock.py; do
  if ! git -C "$REPO" ls-files --error-unmatch "$tool" >/dev/null || ! git -C "$REPO" diff --quiet HEAD -- "$tool" || ! git -C "$REPO" diff --cached --quiet HEAD -- "$tool"; then
    echo "window tooling is not tracked and clean: $tool" >&2
    exit 35
  fi
done

if ! /usr/bin/python3 "$PREPARATION_TOOL" --output "$OUT" --verify > "$OUT/preparation-verify-window.json"; then
  echo "Phase 7 preparation verification failed" >&2
  exit 36
fi
if ! /usr/bin/python3 "$SOURCE_VALIDATE_TOOL" --artifact "$SOURCE_CALIBRATION" > "$OUT/source-calibration-verify-window.json" || ! /usr/bin/python3 "$SOURCE_VALIDATE_TOOL" --artifact "$SOURCE_HOLDOUT" > "$OUT/source-holdout-verify-window.json"; then
  echo "Phase 7 source-oracle verification failed" >&2
  exit 36
fi
if ! /usr/bin/python3 "$STAGING_TOOL" --verify --source "$SOURCE_BIN" --output "$STAGE_DIR" --source-commit "$SOURCE_COMMIT" > "$OUT/fidelity-capture-binary-staging-verify-window.json"; then
  echo "staged fidelity-capture binary identity contract failed" >&2
  exit 37
fi

if ! /usr/bin/python3 - "$MANIFEST" "$OUT/active-pins.json" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
output_path = Path(sys.argv[2])
phase3c = {
    "ULLM_REQUIRE_HIP_AQ4_MATVEC_KERNEL", "ULLM_REQUIRE_HIP_AQ4_MATVEC_BATCH_KERNEL", "ULLM_REQUIRE_HIP_AQ4_MATVEC_ADD_KERNEL", "ULLM_REQUIRE_HIP_AQ4_MATVEC_QKV_Z_GATE_BETA_KERNEL", "ULLM_REQUIRE_HIP_LINEAR_ATTN_KERNEL", "ULLM_REQUIRE_HIP_LINEAR_ATTN_QKV_PREPARE_BATCH_KERNEL", "ULLM_REQUIRE_HIP_LINEAR_ATTN_RECURRENT_KERNEL", "ULLM_REQUIRE_HIP_RMSNORM_KERNEL", "ULLM_REQUIRE_HIP_SEGMENTED_RMSNORM_SILU_MUL_KERNEL", "ULLM_REQUIRE_HIP_AQ4_REGISTER_BM8_KERNEL", "ULLM_REQUIRE_HIP_PAGED_DECODE_ATTN_KERNEL", "ULLM_REQUIRE_HIP_QWEN35_QK_NORM_ROPE_PAGED_KV_WRITE_KERNEL", "ULLM_REQUIRE_HIP_PAGED_KV_WRITE_CHUNK_KERNEL", "ULLM_REQUIRE_HIP_PAGED_CAUSAL_GQA_CHUNK_KERNEL", "ULLM_REQUIRE_HIP_QWEN35_QK_NORM_ROPE_BATCH_KERNEL", "ULLM_REQUIRE_HIP_BF16_ROW_KERNEL", "ULLM_REQUIRE_HIP_TOP1_KERNEL",
}
phase7 = {
    "ULLM_REQUIRE_HIP_AQ4_KERNEL", "ULLM_REQUIRE_HIP_AQ4_MATVEC_KERNEL", "ULLM_REQUIRE_HIP_AQ4_MATVEC_BATCH_KERNEL", "ULLM_REQUIRE_HIP_AQ4_REGISTER_BM8_KERNEL", "ULLM_REQUIRE_HIP_AQ4_MATVEC_ADD_KERNEL", "ULLM_REQUIRE_HIP_AQ4_MATVEC_PAIR_KERNEL", "ULLM_REQUIRE_HIP_AQ4_MATVEC_TRIPLE_KERNEL", "ULLM_REQUIRE_HIP_AQ4_MATVEC_QKV_Z_GATE_BETA_KERNEL", "ULLM_REQUIRE_HIP_ADD_KERNEL", "ULLM_REQUIRE_HIP_BF16_MATVEC_KERNEL", "ULLM_REQUIRE_HIP_BF16_ROW_KERNEL", "ULLM_REQUIRE_HIP_LINEAR_ATTN_GATE_BETA_KERNEL", "ULLM_REQUIRE_HIP_LINEAR_ATTN_KERNEL", "ULLM_REQUIRE_HIP_LINEAR_ATTN_QKV_PREPARE_BATCH_KERNEL", "ULLM_REQUIRE_HIP_LINEAR_ATTN_RECURRENT_KERNEL", "ULLM_REQUIRE_HIP_LINEAR_ATTN_RECURRENT_SEQUENCE_KERNEL", "ULLM_REQUIRE_HIP_PAGED_KV_WRITE_CHUNK_KERNEL", "ULLM_REQUIRE_HIP_PAGED_CAUSAL_GQA_CHUNK_KERNEL", "ULLM_REQUIRE_HIP_PAGED_DECODE_ATTN_KERNEL", "ULLM_REQUIRE_HIP_PAGED_DECODE_SPLIT_KERNEL", "ULLM_REQUIRE_HIP_PAGED_KV_WRITE_KERNEL", "ULLM_REQUIRE_HIP_QWEN35_Q_SPLIT_KERNEL", "ULLM_REQUIRE_HIP_QWEN35_QK_NORM_ROPE_BATCH_KERNEL", "ULLM_REQUIRE_HIP_QWEN35_QK_NORM_ROPE_PAGED_KV_WRITE_KERNEL", "ULLM_REQUIRE_HIP_RMSNORM_KERNEL", "ULLM_REQUIRE_HIP_ROPE_KERNEL", "ULLM_REQUIRE_HIP_SEGMENTED_RMSNORM_SILU_MUL_KERNEL", "ULLM_REQUIRE_HIP_SIGMOID_MUL_KERNEL", "ULLM_REQUIRE_HIP_SILU_MUL_KERNEL", "ULLM_REQUIRE_HIP_TOP1_KERNEL",
}
raw = manifest_path.read_bytes()
payload = json.loads(raw)
assert payload["format"]["format_id"] == "AQ4_0"
assert payload["format"]["implementation_id"] == "qwen35_aq4_rdna4_v1"
assert payload["worker"]["identity"]["device"] == "gfx1201"
assert payload["worker"]["identity"]["execution_profile"] == "rdna4_aq4_resident"
guards = payload["worker"]["required_environment"]
assert isinstance(guards, list) and len(guards) == len(set(guards))
assert set(guards) == phase7 and phase3c <= phase7
package = Path(payload["product"]["root"]) / payload["product"]["package"]["manifest_path"]
worker = Path(payload["worker"]["binary"])
assert package.is_file() and not package.is_symlink() and worker.is_file() and not worker.is_symlink()
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
manifest_sha = hashlib.sha256(raw).hexdigest()
package_sha = sha(package)
worker_sha = sha(worker)
assert package_sha == payload["product"]["package"]["manifest_sha256"]
assert worker_sha == payload["worker"]["binary_sha256"]
guard_digest = hashlib.sha256()
guard_digest.update(b"ullm-aq4-p2-resident-guards-v1\0")
for name in sorted(phase7):
    guard_digest.update(f"{name}=1\n".encode())
result = {
    "schema_version": "ullm.aq4_phase7_active_guard_contract.v1",
    "status": "valid",
    "served_model_manifest_sha256": manifest_sha,
    "package_manifest_sha256": package_sha,
    "worker_binary_sha256": worker_sha,
    "guard_sha256": guard_digest.hexdigest(),
    "quantized_artifact_revision": payload["public"]["revision"],
    "device_architecture": "gfx1201",
    "phase3c_required_guards": sorted(phase3c),
    "active_required_guards": sorted(guards),
    "phase3c_subset_of_active": True,
}
output_path.write_text(json.dumps(result, ensure_ascii=True, sort_keys=True, indent=2) + "\n", encoding="utf-8")
PY
then
  echo "active manifest/guard contract failed before service stop" >&2
  exit 39
fi

read -r MANIFEST_SHA PACKAGE_SHA WORKER_SHA GUARD_SHA QUANTIZED_REVISION < <(/usr/bin/python3 - "$OUT/active-pins.json" <<'PY'
import json
import sys
p=json.load(open(sys.argv[1], encoding="utf-8"))
assert p["status"] == "valid"
print(p["served_model_manifest_sha256"], p["package_manifest_sha256"], p["worker_binary_sha256"], p["guard_sha256"], p["quantized_artifact_revision"])
PY
)

active_state="$(systemctl show "$SERVICE" -p ActiveState --value)"
sub_state="$(systemctl show "$SERVICE" -p SubState --value)"
main_pid="$(systemctl show "$SERVICE" -p MainPID --value)"
restarts="$(systemctl show "$SERVICE" -p NRestarts --value)"
preserve="$(systemctl show "$SERVICE" -p RuntimeDirectoryPreserve --value)"
actual_manifest_sha="$(sha256sum "$MANIFEST" | awk '{print $1}')"
printf '%s\n' \
  "captured_at_utc=$(date --iso-8601=seconds --utc)" \
  "active_state=$active_state" \
  "sub_state=$sub_state" \
  "main_pid=$main_pid" \
  "nrestarts=$restarts" \
  "runtime_directory_preserve=$preserve" \
  "manifest_sha256=$actual_manifest_sha" \
  "source_commit=$SOURCE_COMMIT" \
  > "$OUT/service-window-pre-stop.txt"
if [ "$active_state" != active ] || [ "$sub_state" != running ] || [ "$main_pid" -le 0 ] || [ "$preserve" != yes ] || [ "$actual_manifest_sha" != "$MANIFEST_SHA" ]; then
  OUTCOME=pre_stop_baseline_failure
  write_result
  exit 33
fi

STOP_ATTEMPTED=1
stop_started_at="$(date --iso-8601=seconds --utc)"
event stop_invoked ""
systemctl stop "$SERVICE" > "$OUT/service-window-stop.stdout" 2> "$OUT/service-window-stop.stderr"
stop_rc=$?
systemctl show "$SERVICE" -p ActiveState -p SubState -p MainPID > "$OUT/service-window-stop-systemctl-show.txt" 2> "$OUT/service-window-stop-systemctl-show.stderr"
show_rc=$?
stop_finished_at="$(date --iso-8601=seconds --utc)"
printf '%s\n' "started_at_utc=$stop_started_at" "finished_at_utc=$stop_finished_at" "exit_code=$stop_rc" "post_stop_show_exit_code=$show_rc" > "$OUT/service-window-stop.txt"
event stop_returned "exit_code=$stop_rc"
if [ "$stop_rc" -ne 0 ]; then
  OUTCOME=stop_failure
  restore
  write_result
  exit 50
fi

runuser -u homelab1 -- /usr/bin/python3 "$LOCK_PROBE" "$OUT/service-window-lock-after-stop.json" "$LOCK"
probe_rc=$?
event post_stop_lock_probe "exit_code=$probe_rc"
if [ "$probe_rc" -ne 0 ]; then
  OUTCOME=lock_acquisition_failure_no_target_capture
  restore
  write_result
  exit 40
fi

/usr/bin/python3 "$GUARD_TOOL" --output "$OUT/guard-before" --guard-bin "$GUARD_BIN" --health-phase before
guard_rc=$?
event r9700_architecture_and_health_guard_before "exit_code=$guard_rc"
if [ "$guard_rc" -ne 0 ]; then
  OUTCOME=architecture_guard_failure_no_target_capture
  restore
  write_result
  exit 41
fi

run_target_capture() {
  local split_root=$1
  local source_root=$2
  local cases=$3
  local target_output=$4
  runuser -u homelab1 -- /usr/bin/env -i "${PHASE7_ENV[@]}" \
    /bin/bash -ceu '
      lock=$1
      binary=$2
      manifest=$3
      split_root=$4
      source_root=$5
      cases=$6
      output=$7
      split_sha=$8
      policy_sha=$9
      cases_sha=${10}
      manifest_sha=${11}
      package_sha=${12}
      worker_sha=${13}
      guard_sha=${14}
      revision=${15}
      exec 9< "$lock"
      flock -n 9
      exec "$binary" \
        --served-model-manifest "$manifest" \
        --split-root "$split_root" \
        --source "$source_root" \
        --cases "$cases" \
        --output "$output" \
        --device-index 1 \
        --chunk-elements 65536 \
        --expected-split-manifest-sha256 "$split_sha" \
        --expected-policy-sha256 "$policy_sha" \
        --expected-calibration-cases-sha256 "$cases_sha" \
        --expected-served-model-manifest-sha256 "$manifest_sha" \
        --expected-package-manifest-sha256 "$package_sha" \
        --expected-worker-binary-sha256 "$worker_sha" \
        --expected-guard-sha256 "$guard_sha" \
        --expected-device-architecture gfx1201 \
        --expected-quantized-artifact-revision "$revision"
    ' bash "$LOCK" "$CAPTURE_BIN" "$MANIFEST" "$split_root" "$source_root" "$cases" "$target_output" \
      "$(sha256sum "$split_root/split-manifest.json" | awk '{print $1}')" \
      "$(sha256sum "$split_root/policy.json" | awk '{print $1}')" \
      "$(sha256sum "$split_root/calibration-cases.jsonl" | awk '{print $1}')" \
      "$MANIFEST_SHA" "$PACKAGE_SHA" "$WORKER_SHA" "$GUARD_SHA" "$QUANTIZED_REVISION"
}

event calibration_target_capture_invoked "one_model_load"
run_target_capture "$FORMAL_SPLIT" "$SOURCE_CALIBRATION" "$CALIBRATION_CASES" "$OUT/target-calibration"
CALIBRATION_CAPTURE_RC=$?
event calibration_target_capture_returned "exit_code=$CALIBRATION_CAPTURE_RC"

if [ "$CALIBRATION_CAPTURE_RC" -eq 0 ]; then
  event calibration_metrics_and_freeze_invoked "cpu_only_while_service_stopped"
  runuser -u homelab1 -- /usr/bin/env -i HOME=/home/homelab1 PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    /usr/bin/python3 "$CAPTURE_METRICS_TOOL" \
      --split-root "$FORMAL_SPLIT" \
      --source "$SOURCE_CALIBRATION" \
      --active "$OUT/target-calibration" \
      --output "$OUT/calibration-metrics.json" \
      --expected-split-manifest-sha256 "$(sha256sum "$FORMAL_SPLIT/split-manifest.json" | awk '{print $1}')" \
      --expected-policy-sha256 "$(sha256sum "$FORMAL_SPLIT/policy.json" | awk '{print $1}')" \
      --expected-calibration-cases-sha256 "$(sha256sum "$FORMAL_SPLIT/calibration-cases.jsonl" | awk '{print $1}')" \
      --expected-served-model-manifest-sha256 "$MANIFEST_SHA" \
      --expected-package-manifest-sha256 "$PACKAGE_SHA" \
      --expected-worker-binary-sha256 "$WORKER_SHA" \
      --expected-guard-sha256 "$GUARD_SHA" \
      --expected-device-architecture gfx1201 \
      --expected-quantized-artifact-revision "$QUANTIZED_REVISION"
  metrics_rc=$?
  if [ "$metrics_rc" -eq 0 ]; then
    runuser -u homelab1 -- /usr/bin/env -i HOME=/home/homelab1 PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
      /usr/bin/python3 "$CAPTURE_METRICS_VALIDATE_TOOL" --metrics "$OUT/calibration-metrics.json" --split-root "$FORMAL_SPLIT" > "$OUT/calibration-metrics-validation.json"
    metrics_validate_rc=$?
  else
    metrics_validate_rc=1
  fi
  if [ "$metrics_rc" -eq 0 ] && [ "$metrics_validate_rc" -eq 0 ]; then
    runuser -u homelab1 -- /usr/bin/env -i HOME=/home/homelab1 PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
      /usr/bin/python3 "$FREEZE_TOOL" freeze --split-root "$FORMAL_SPLIT" --metrics "$OUT/calibration-metrics.json" --output "$OUT/freeze-receipt.json"
    FREEZE_RC=$?
  else
    FREEZE_RC=1
  fi
  event calibration_metrics_and_freeze_returned "metrics_exit_code=$metrics_rc metrics_validation_exit_code=$metrics_validate_rc freeze_exit_code=$FREEZE_RC"
fi

if [ "$CALIBRATION_CAPTURE_RC" -eq 0 ] && [ "$FREEZE_RC" -eq 0 ]; then
  event holdout_target_capture_invoked "one_model_load_after_frozen_receipt"
  run_target_capture "$EXECUTION_VIEW" "$SOURCE_HOLDOUT" "$HOLDOUT_CASES" "$OUT/target-holdout"
  HOLDOUT_CAPTURE_RC=$?
  event holdout_target_capture_returned "exit_code=$HOLDOUT_CAPTURE_RC"
fi

/usr/bin/python3 "$GUARD_TOOL" --output "$OUT/guard-after" --guard-bin "$GUARD_BIN" --health-phase after
AFTER_GUARD_RC=$?
event r9700_architecture_and_health_guard_after "exit_code=$AFTER_GUARD_RC"
restore
if [ "$RESTORE_RC" -ne 0 ]; then
  OUTCOME=restore_start_failure
  write_result
  exit 70
fi

post_active_state="$(systemctl show "$SERVICE" -p ActiveState --value)"
post_sub_state="$(systemctl show "$SERVICE" -p SubState --value)"
post_main_pid="$(systemctl show "$SERVICE" -p MainPID --value)"
post_restarts="$(systemctl show "$SERVICE" -p NRestarts --value)"
post_manifest_sha="$(sha256sum "$MANIFEST" | awk '{print $1}')"
printf '%s\n' \
  "captured_at_utc=$(date --iso-8601=seconds --utc)" \
  "active_state=$post_active_state" \
  "sub_state=$post_sub_state" \
  "main_pid=$post_main_pid" \
  "nrestarts=$post_restarts" \
  "manifest_sha256=$post_manifest_sha" \
  "pre_stop_nrestarts=$restarts" \
  > "$OUT/service-window-post-restore.txt"
if [ "$post_active_state" != active ] || [ "$post_sub_state" != running ] || [ "$post_main_pid" -le 0 ] || [ "$post_restarts" != "$restarts" ] || [ "$post_manifest_sha" != "$actual_manifest_sha" ]; then
  OUTCOME=post_restore_service_contract_failure
  write_result
  exit 72
fi
if [ "$CALIBRATION_CAPTURE_RC" -ne 0 ]; then
  OUTCOME=calibration_target_capture_failure_no_retry
  write_result
  exit "$CALIBRATION_CAPTURE_RC"
fi
if [ "$FREEZE_RC" -ne 0 ]; then
  OUTCOME=calibration_freeze_failure_holdout_not_captured
  write_result
  exit 73
fi
if [ "$HOLDOUT_CAPTURE_RC" -ne 0 ]; then
  OUTCOME=holdout_target_capture_failure_no_retry
  write_result
  exit "$HOLDOUT_CAPTURE_RC"
fi
if [ "$AFTER_GUARD_RC" -ne 0 ]; then
  OUTCOME=post_trace_health_guard_failure
  write_result
  exit 71
fi

event holdout_evaluation_invoked "cpu_only_after_service_restore"
if ! runuser -u homelab1 -- /usr/bin/env -i HOME=/home/homelab1 PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
  /usr/bin/python3 "$HOLDOUT_EVALUATOR" \
    --preparation-root "$OUT" \
    --freeze-receipt "$OUT/freeze-receipt.json" \
    --calibration-metrics "$OUT/calibration-metrics.json" \
    --source "$SOURCE_HOLDOUT" \
    --target "$OUT/target-holdout" \
    --output "$OUT/holdout-evaluation.json" \
    --confirm-holdout-once; then
  OUTCOME=post_restore_holdout_evaluation_failure
  write_result
  exit 74
fi
event holdout_evaluation_returned "exit_code=0"
OUTCOME=calibration_frozen_and_single_holdout_evaluated
write_result
exit 0
