#!/usr/bin/env bash
# Consume exactly one R9700-only AQ4 P2 full-vector path-oracle window.
#
# The CPU source sidecar must already exist.  This driver performs one stop,
# one read-only existing-lock probe, one before/after R9700 guard pair, and one
# service-user calibration process for exactly one anchor.  It has no V620 or
# SQ8 branch and never creates /run/ullm/r9700.lock.

set -u -o pipefail
umask 077

if [ "$#" -ne 7 ] || [ "$7" != "--confirm-single-window" ]; then
  echo "usage: $0 PREPARATION_ROOT HIP_GUARD_BINARY CLEAN_SOURCE_WORKTREE SOURCE_COMMIT SOURCE_ORACLE_ROOT CASE_ID --confirm-single-window" >&2
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
SOURCE_ORACLE=$5
CASE_ID=$6
REPO=/home/homelab1/coding-local/ultimateLLM/uLLM-project
SERVICE=ullm-openai.service
LOCK=/run/ullm/r9700.lock
MANIFEST=/etc/ullm/served-models/active.json
STAGING="$OUT/staging/baseline-binaries"
EXECUTOR="$REPO/tools/run-aq4-p2-production-path-oracle.py"
PREPARE_TOOL="$REPO/tools/prepare-aq4-p2-production-baseline.py"
STAGE_TOOL="$REPO/tools/stage-aq4-p2-production-baseline-binaries.py"
GUARD_STAGE_TOOL="$REPO/tools/stage-aq4-p2-r9700-guard.py"
GUARD_TOOL="$REPO/tools/run-aq4-phase3c-r9700-guard.py"
LOCK_PROBE="$REPO/tools/probe-aq4-phase3c-existing-lock.py"
TARGET_OUTPUT="$OUT/source-oracle/target/$CASE_ID"
INPUTS="$OUT/source-oracle/target-inputs/$CASE_ID"
EVENTS="$OUT/source-oracle/$CASE_ID-service-events.tsv"
RESULT="$OUT/source-oracle/$CASE_ID-service-result.txt"
PRE_STOP="$OUT/source-oracle/$CASE_ID-service-pre-stop.txt"
POST_RESTORE="$OUT/source-oracle/$CASE_ID-service-post-restore.txt"
LOCK_PROBE_OUTPUT="$OUT/source-oracle/$CASE_ID-service-lock-after-stop.json"
GUARD_BEFORE="$OUT/guard/$CASE_ID-oracle-before"
GUARD_AFTER="$OUT/guard/$CASE_ID-oracle-after"
GUARD_STAGE_DIR=$(dirname "$GUARD_BIN")

AQ4_GUARDS=(
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
P2_ENV=(
  HOME=/home/homelab1
  PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
  HIP_VISIBLE_DEVICES=1
  ULLM_HIP_VISIBLE_DEVICES=1
  ULLM_P2_PREHELD_LOCK_FD=9
  ULLM_BUILD_GIT_COMMIT="$SOURCE_COMMIT"
  ULLM_AQ4_MATVEC_QKV_Z_GATE_BETA_RPB=4
  ULLM_AQ4_MATVEC_SILU_MUL_RPB=8
  ULLM_AQ4_MATVEC_ADD_RPB=8
)
for name in "${AQ4_GUARDS[@]}"; do
  P2_ENV+=("$name=1")
done

STOP_ATTEMPTED=0
RESTORE_DONE=0
RESTORE_RC=99
EXECUTOR_RC=not_started
BEFORE_GUARD_RC=not_started
AFTER_GUARD_RC=not_started
OUTCOME=not_started

event() { printf '%s\t%s\t%s\n' "$(date --iso-8601=seconds --utc)" "$1" "$2" >> "$EVENTS"; }
write_result() {
  printf '%s\n' \
    "recorded_at_utc=$(date --iso-8601=seconds --utc)" \
    "case_id=$CASE_ID" \
    "outcome=$OUTCOME" \
    "executor_exit_code=$EXECUTOR_RC" \
    "guard_before_exit_code=$BEFORE_GUARD_RC" \
    "guard_after_exit_code=$AFTER_GUARD_RC" \
    "restore_start_exit_code=$RESTORE_RC" > "$RESULT"
}
restore() {
  if [ "$STOP_ATTEMPTED" -ne 1 ] || [ "$RESTORE_DONE" -ne 0 ]; then return 0; fi
  RESTORE_DONE=1
  event restore_start_invoked ""
  systemctl start "$SERVICE" > "$OUT/source-oracle/$CASE_ID-service-start.stdout" 2> "$OUT/source-oracle/$CASE_ID-service-start.stderr"
  RESTORE_RC=$?
  event restore_start_returned "exit_code=$RESTORE_RC"
  return "$RESTORE_RC"
}
on_exit() {
  local code=$?
  trap - EXIT
  if [ "$STOP_ATTEMPTED" -eq 1 ] && [ "$RESTORE_DONE" -eq 0 ]; then restore; fi
  exit "$code"
}
trap on_exit EXIT
require_regular() { if [ ! -f "$1" ] || [ -L "$1" ]; then echo "required regular file is unavailable: $1" >&2; exit 30; fi; }
require_directory() { if [ ! -d "$1" ] || [ -L "$1" ]; then echo "required real directory is unavailable: $1" >&2; exit 31; fi; }

case "$OUT" in /*) ;; *) echo "preparation root must be absolute" >&2; exit 32;; esac
case "$SOURCE_TREE" in /*) ;; *) echo "source worktree must be absolute" >&2; exit 32;; esac
case "$SOURCE_ORACLE" in /*) ;; *) echo "source oracle root must be absolute" >&2; exit 32;; esac
if ! [[ "$SOURCE_COMMIT" =~ ^[0-9a-f]{40}$ ]] || ! [[ "$CASE_ID" =~ ^[A-Za-z0-9._-]+$ ]]; then echo "source commit or case ID is unsafe" >&2; exit 32; fi
require_directory "$OUT"
require_directory "$OUT/staging"
require_directory "$OUT/source-oracle"
require_directory "$OUT/guard"
require_directory "$SOURCE_TREE"
require_directory "$SOURCE_ORACLE"
require_regular "$GUARD_BIN"
if [ "$GUARD_BIN" != "$GUARD_STAGE_DIR/query-hip-device-identity" ]; then echo "guard binary must be the staged query-hip-device-identity member" >&2; exit 34; fi
require_regular "$MANIFEST"
require_regular "$LOCK"
if [ -e "$TARGET_OUTPUT" ] || [ -L "$TARGET_OUTPUT" ] || [ -e "$INPUTS" ] || [ -L "$INPUTS" ] || [ -e "$EVENTS" ] || [ -e "$RESULT" ] || [ -e "$PRE_STOP" ] || [ -e "$POST_RESTORE" ] || [ -e "$LOCK_PROBE_OUTPUT" ] || [ -e "$GUARD_BEFORE" ] || [ -e "$GUARD_AFTER" ] || [ -e "$OUT/source-oracle/$CASE_ID-preparation-verify.json" ] || [ -e "$OUT/source-oracle/$CASE_ID-staging-verify.json" ] || [ -e "$OUT/source-oracle/$CASE_ID-guard-staging-verify.json" ] || [ -e "$OUT/source-oracle/$CASE_ID-executor-dry-run.json" ]; then
  echo "refusing to reuse an AQ4 P2 path-oracle output for $CASE_ID" >&2
  exit 33
fi
for path in "$PREPARE_TOOL" "$STAGE_TOOL" "$GUARD_STAGE_TOOL" "$EXECUTOR" "$GUARD_TOOL" "$LOCK_PROBE"; do require_regular "$path"; done
require_regular "$STAGING/ullm-aq4-p2-calibration"
if [ "$(git -C "$SOURCE_TREE" rev-parse HEAD)" != "$SOURCE_COMMIT" ] || ! git -C "$SOURCE_TREE" diff --quiet || ! git -C "$SOURCE_TREE" diff --cached --quiet || [ -n "$(git -C "$SOURCE_TREE" status --porcelain --untracked-files=all)" ]; then
  echo "clean source worktree contract failed" >&2
  exit 35
fi
for path in \
  tools/prepare-aq4-p2-production-baseline.py \
  tools/stage-aq4-p2-production-baseline-binaries.py \
  tools/stage-aq4-p2-r9700-guard.py \
  tools/run-aq4-p2-production-path-oracle.py \
  tools/run-aq4-p2-production-path-oracle-service-window.sh \
  tools/run-aq4-phase3c-r9700-guard.py \
  tools/probe-aq4-phase3c-existing-lock.py \
  tools/query-hip-device-identity.cpp; do
  if ! git -C "$REPO" ls-files --error-unmatch "$path" >/dev/null || ! git -C "$REPO" diff --quiet HEAD -- "$path" || ! git -C "$REPO" diff --cached --quiet HEAD -- "$path"; then
    echo "window tooling is not tracked and clean: $path" >&2
    exit 36
  fi
done
if ! /usr/bin/python3 "$PREPARE_TOOL" --output "$OUT" --verify > "$OUT/source-oracle/$CASE_ID-preparation-verify.json" || ! /usr/bin/python3 "$STAGE_TOOL" --output "$STAGING" --preparation "$OUT" --source-commit "$SOURCE_COMMIT" --verify > "$OUT/source-oracle/$CASE_ID-staging-verify.json"; then
  echo "P2 preparation/staging verification failed" >&2
  exit 37
fi
if ! /usr/bin/python3 "$GUARD_STAGE_TOOL" --output "$GUARD_STAGE_DIR" --preparation "$OUT" --source-commit "$SOURCE_COMMIT" --verify > "$OUT/source-oracle/$CASE_ID-guard-staging-verify.json"; then
  echo "P2 R9700 guard staging identity verification failed" >&2
  exit 37
fi
if ! /usr/bin/python3 "$EXECUTOR" --preparation "$OUT" --staging "$STAGING" --case-id "$CASE_ID" --source "$SOURCE_ORACLE" --output "$TARGET_OUTPUT" --dry-run > "$OUT/source-oracle/$CASE_ID-executor-dry-run.json"; then
  echo "P2 path-oracle executor dry-run failed" >&2
  exit 37
fi
PREPARED_COMMIT=$(/usr/bin/python3 - "$OUT/identity.json" <<'PY'
import json
import sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["clean_baseline_source"]["git_commit"])
PY
)
if [ "$PREPARED_COMMIT" != "$SOURCE_COMMIT" ] || [ "$(systemctl show "$SERVICE" -p RuntimeDirectoryPreserve --value)" != yes ]; then
  echo "prepared commit or RuntimeDirectoryPreserve=yes contract failed" >&2
  exit 38
fi

pre_state=$(systemctl show "$SERVICE" -p ActiveState --value)
pre_sub=$(systemctl show "$SERVICE" -p SubState --value)
pre_pid=$(systemctl show "$SERVICE" -p MainPID --value)
pre_restarts=$(systemctl show "$SERVICE" -p NRestarts --value)
manifest_sha=$(sha256sum "$MANIFEST" | awk '{print $1}')
printf '%s\n' "captured_at_utc=$(date --iso-8601=seconds --utc)" "active_state=$pre_state" "sub_state=$pre_sub" "main_pid=$pre_pid" "nrestarts=$pre_restarts" "manifest_sha256=$manifest_sha" "runtime_directory_preserve=yes" > "$PRE_STOP"
if [ "$pre_state" != active ] || [ "$pre_sub" != running ] || [ "$pre_pid" -le 0 ]; then echo "service pre-stop state is not active/running" >&2; exit 40; fi

event stop_invoked "single_stop"
STOP_ATTEMPTED=1
systemctl stop "$SERVICE" > "$OUT/source-oracle/$CASE_ID-service-stop.stdout" 2> "$OUT/source-oracle/$CASE_ID-service-stop.stderr"
stop_rc=$?
event stop_returned "exit_code=$stop_rc"
if [ "$stop_rc" -ne 0 ]; then OUTCOME=service_stop_failure; restore; write_result; exit "$stop_rc"; fi
if ! runuser -u homelab1 -- /usr/bin/python3 "$LOCK_PROBE" "$LOCK_PROBE_OUTPUT" "$LOCK"; then OUTCOME=lock_probe_failure_no_target_capture; restore; write_result; exit 41; fi
/usr/bin/python3 "$GUARD_TOOL" --output "$GUARD_BEFORE" --guard-bin "$GUARD_BIN" --health-phase "p2-oracle-$CASE_ID-before"
BEFORE_GUARD_RC=$?
if [ "$BEFORE_GUARD_RC" -ne 0 ]; then OUTCOME=r9700_guard_before_failure_no_target_capture; restore; write_result; exit 42; fi
event r9700_guard_before_valid ""

event executor_invoked "case_id=$CASE_ID"
runuser -u homelab1 -- /usr/bin/env -i "${P2_ENV[@]}" /bin/bash -ceu '
  lock=$1; executor=$2; preparation=$3; staging=$4; source=$5; case_id=$6; output=$7; manifest=$8
  exec 9< "$lock"
  flock -n 9
  exec /usr/bin/python3 "$executor" --preparation "$preparation" --staging "$staging" --case-id "$case_id" --source "$source" --output "$output" --served-manifest "$manifest" --execute --confirm-r9700-window
' bash "$LOCK" "$EXECUTOR" "$OUT" "$STAGING" "$SOURCE_ORACLE" "$CASE_ID" "$TARGET_OUTPUT" "$MANIFEST" > "$OUT/source-oracle/$CASE_ID-executor.stdout" 2> "$OUT/source-oracle/$CASE_ID-executor.stderr"
EXECUTOR_RC=$?
event executor_returned "exit_code=$EXECUTOR_RC"

if /usr/bin/python3 "$GUARD_TOOL" --output "$GUARD_AFTER" --guard-bin "$GUARD_BIN" --health-phase "p2-oracle-$CASE_ID-after"; then AFTER_GUARD_RC=0; else AFTER_GUARD_RC=$?; fi
event r9700_guard_after_returned "exit_code=$AFTER_GUARD_RC"
restore
if [ "$RESTORE_RC" -ne 0 ]; then OUTCOME=restore_start_failure; write_result; exit 70; fi
post_state=$(systemctl show "$SERVICE" -p ActiveState --value)
post_sub=$(systemctl show "$SERVICE" -p SubState --value)
post_pid=$(systemctl show "$SERVICE" -p MainPID --value)
post_restarts=$(systemctl show "$SERVICE" -p NRestarts --value)
post_manifest_sha=$(sha256sum "$MANIFEST" | awk '{print $1}')
printf '%s\n' "captured_at_utc=$(date --iso-8601=seconds --utc)" "active_state=$post_state" "sub_state=$post_sub" "main_pid=$post_pid" "nrestarts=$post_restarts" "manifest_sha256=$post_manifest_sha" "pre_stop_nrestarts=$pre_restarts" > "$POST_RESTORE"
if [ "$post_state" != active ] || [ "$post_sub" != running ] || [ "$post_pid" -le 0 ] || [ "$post_restarts" != "$pre_restarts" ] || [ "$post_manifest_sha" != "$manifest_sha" ]; then OUTCOME=post_restore_service_contract_failure; write_result; exit 71; fi
if [ "$EXECUTOR_RC" -ne 0 ]; then OUTCOME=executor_failure_no_retry; write_result; exit "$EXECUTOR_RC"; fi
if [ "$AFTER_GUARD_RC" -ne 0 ]; then OUTCOME=r9700_guard_after_failure; write_result; exit 72; fi
OUTCOME=single_path_oracle_window_completed
write_result
