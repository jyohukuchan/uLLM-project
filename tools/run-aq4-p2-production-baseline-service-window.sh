#!/usr/bin/env bash
# Consume exactly one AQ4 P2 production-baseline service-stop window.
#
# This is intentionally a root-only consumer.  It performs one stop and one
# restore at most, verifies the pre-existing R9700 lock through the established
# read-only probe, runs the R9700-only guard before and after capture, and
# drops the model executor to the service user with an inherited locked FD.
# It has no V620 path and never calls an unqualified AMD-SMI command.

set -u -o pipefail
umask 077

if [ "$#" -ne 6 ] || [ "$6" != "--confirm-single-window" ]; then
  echo "usage: $0 PREPARATION_ROOT HIP_GUARD_BINARY CLEAN_SOURCE_WORKTREE SOURCE_COMMIT WINDOW_ID --confirm-single-window" >&2
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
WINDOW_ID=$5
REPO=/home/homelab1/coding-local/ultimateLLM/uLLM-project
SERVICE=ullm-openai.service
LOCK=/run/ullm/r9700.lock
MANIFEST=/etc/ullm/served-models/active.json
STAGING="$OUT/staging/baseline-binaries"
WINDOW_OUTPUT="$OUT/windows/$WINDOW_ID"
EVENTS="$OUT/windows/$WINDOW_ID-service-events.tsv"
RESULT="$OUT/windows/$WINDOW_ID-service-result.txt"
PRE_STOP="$OUT/windows/$WINDOW_ID-service-pre-stop.txt"
POST_RESTORE="$OUT/windows/$WINDOW_ID-service-post-restore.txt"
LOCK_PROBE_OUTPUT="$OUT/windows/$WINDOW_ID-service-lock-after-stop.json"
GUARD_BEFORE="$OUT/guard/$WINDOW_ID-before"
GUARD_AFTER="$OUT/guard/$WINDOW_ID-after"
PREPARE_TOOL="$REPO/tools/prepare-aq4-p2-production-baseline.py"
STAGE_TOOL="$REPO/tools/stage-aq4-p2-production-baseline-binaries.py"
GUARD_STAGE_TOOL="$REPO/tools/stage-aq4-p2-r9700-guard.py"
EXECUTOR="$REPO/tools/run-aq4-p2-production-baseline-window.py"
PROFILE_PARSER="$REPO/tools/parse-aq4-p2-production-profile.py"
GUARD_TOOL="$REPO/tools/run-aq4-phase3c-r9700-guard.py"
LOCK_PROBE="$REPO/tools/probe-aq4-phase3c-existing-lock.py"
ROCPROF=/opt/rocm-7.2.1/bin/rocprofv3
GUARD_STAGE_DIR=$(dirname "$GUARD_BIN")

# This is the active AQ4-only guard set.  It is deliberately explicit rather
# than inheriting a broad environment from root or the stopped service.
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
PROFILE_RC=not_requested
BEFORE_GUARD_RC=not_started
AFTER_GUARD_RC=not_started
OUTCOME=not_started
WINDOW_KIND=
PROFILE_ENABLED=0
PROFILE_RAW="$OUT/windows/$WINDOW_ID-profile-raw"
PROFILE_REPORT="$OUT/windows/$WINDOW_ID-profile.json"

event() {
  printf '%s\t%s\t%s\n' "$(date --iso-8601=seconds --utc)" "$1" "$2" >> "$EVENTS"
}

write_result() {
  printf '%s\n' \
    "recorded_at_utc=$(date --iso-8601=seconds --utc)" \
    "window_id=$WINDOW_ID" \
    "outcome=$OUTCOME" \
    "executor_exit_code=$EXECUTOR_RC" \
    "profile_parse_exit_code=$PROFILE_RC" \
    "guard_before_exit_code=$BEFORE_GUARD_RC" \
    "guard_after_exit_code=$AFTER_GUARD_RC" \
    "restore_start_exit_code=$RESTORE_RC" \
    > "$RESULT"
}

restore() {
  if [ "$STOP_ATTEMPTED" -ne 1 ] || [ "$RESTORE_DONE" -ne 0 ]; then
    return 0
  fi
  RESTORE_DONE=1
  event restore_start_invoked ""
  systemctl start "$SERVICE" > "$OUT/windows/$WINDOW_ID-service-start.stdout" 2> "$OUT/windows/$WINDOW_ID-service-start.stderr"
  RESTORE_RC=$?
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

require_regular() {
  if [ ! -f "$1" ] || [ -L "$1" ]; then
    echo "required regular file is unavailable: $1" >&2
    exit 30
  fi
}

require_directory() {
  if [ ! -d "$1" ] || [ -L "$1" ]; then
    echo "required real directory is unavailable: $1" >&2
    exit 31
  fi
}

case "$OUT" in /*) ;; *) echo "preparation root must be absolute" >&2; exit 32 ;; esac
case "$SOURCE_TREE" in /*) ;; *) echo "source worktree must be absolute" >&2; exit 32 ;; esac
if ! [[ "$SOURCE_COMMIT" =~ ^[0-9a-f]{40}$ ]]; then
  echo "source commit must be a lowercase 40-character SHA-1" >&2
  exit 32
fi
if ! [[ "$WINDOW_ID" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "window ID is unsafe" >&2
  exit 32
fi
require_directory "$OUT"
require_directory "$OUT/staging"
require_directory "$OUT/guard"
require_directory "$OUT/windows"
require_directory "$SOURCE_TREE"
require_regular "$GUARD_BIN"
if [ "$GUARD_BIN" != "$GUARD_STAGE_DIR/query-hip-device-identity" ]; then
  echo "guard binary must be the staged query-hip-device-identity member" >&2
  exit 34
fi
require_regular "$MANIFEST"
require_regular "$LOCK"
if [ -e "$WINDOW_OUTPUT" ] || [ -L "$WINDOW_OUTPUT" ] || [ -e "$PROFILE_RAW" ] || [ -L "$PROFILE_RAW" ] || [ -e "$PROFILE_REPORT" ] || [ -L "$PROFILE_REPORT" ] || [ -e "$EVENTS" ] || [ -e "$RESULT" ] || [ -e "$PRE_STOP" ] || [ -e "$POST_RESTORE" ] || [ -e "$LOCK_PROBE_OUTPUT" ] || [ -e "$GUARD_BEFORE" ] || [ -e "$GUARD_AFTER" ] || [ -e "$OUT/windows/$WINDOW_ID-preparation-verify.json" ] || [ -e "$OUT/windows/$WINDOW_ID-staging-verify.json" ] || [ -e "$OUT/windows/$WINDOW_ID-guard-staging-verify.json" ] || [ -e "$OUT/windows/$WINDOW_ID-executor-dry-run.json" ]; then
  echo "refusing to reuse an AQ4 P2 service-window output for $WINDOW_ID" >&2
  exit 33
fi
for path in "$PREPARE_TOOL" "$STAGE_TOOL" "$GUARD_STAGE_TOOL" "$EXECUTOR" "$PROFILE_PARSER" "$GUARD_TOOL" "$LOCK_PROBE"; do
  require_regular "$path"
done
for binary in "$STAGING/ullm-aq4-p2-resident-driver" "$STAGING/ullm-aq4-p2-calibration"; do
  if [ ! -x "$binary" ] || [ -L "$binary" ]; then
    echo "staged AQ4 P2 binary is unavailable: $binary" >&2
    exit 34
  fi
done
if [ "$(git -C "$SOURCE_TREE" rev-parse HEAD)" != "$SOURCE_COMMIT" ] || ! git -C "$SOURCE_TREE" diff --quiet || ! git -C "$SOURCE_TREE" diff --cached --quiet || [ -n "$(git -C "$SOURCE_TREE" status --porcelain --untracked-files=all)" ]; then
  echo "clean source worktree contract failed" >&2
  exit 35
fi
for path in \
  tools/prepare-aq4-p2-production-baseline.py \
  tools/stage-aq4-p2-production-baseline-binaries.py \
  tools/stage-aq4-p2-r9700-guard.py \
  tools/run-aq4-p2-production-baseline-window.py \
  tools/run-aq4-p2-production-baseline-service-window.sh \
  tools/parse-aq4-p2-production-profile.py \
  tools/run-aq4-phase3c-r9700-guard.py \
  tools/probe-aq4-phase3c-existing-lock.py \
  tools/query-hip-device-identity.cpp; do
  if ! git -C "$REPO" ls-files --error-unmatch "$path" >/dev/null || ! git -C "$REPO" diff --quiet HEAD -- "$path" || ! git -C "$REPO" diff --cached --quiet HEAD -- "$path"; then
    echo "window tooling is not tracked and clean: $path" >&2
    exit 36
  fi
done

if ! /usr/bin/python3 "$PREPARE_TOOL" --output "$OUT" --verify > "$OUT/windows/$WINDOW_ID-preparation-verify.json"; then
  echo "P2 preparation verification failed" >&2
  exit 37
fi
if ! /usr/bin/python3 "$STAGE_TOOL" --output "$STAGING" --preparation "$OUT" --source-commit "$SOURCE_COMMIT" --verify > "$OUT/windows/$WINDOW_ID-staging-verify.json"; then
  echo "P2 staged binary identity verification failed" >&2
  exit 37
fi
if ! /usr/bin/python3 "$GUARD_STAGE_TOOL" --output "$GUARD_STAGE_DIR" --preparation "$OUT" --source-commit "$SOURCE_COMMIT" --verify > "$OUT/windows/$WINDOW_ID-guard-staging-verify.json"; then
  echo "P2 R9700 guard staging identity verification failed" >&2
  exit 37
fi
if ! /usr/bin/python3 "$EXECUTOR" --preparation "$OUT" --staging "$STAGING" --window "$WINDOW_ID" --output "$WINDOW_OUTPUT" --dry-run > "$OUT/windows/$WINDOW_ID-executor-dry-run.json"; then
  echo "P2 executor dry-run failed" >&2
  exit 37
fi
WINDOW_KIND=$(/usr/bin/python3 - "$OUT/window-plan.json" "$WINDOW_ID" <<'PY'
import json
import sys

plan = json.load(open(sys.argv[1], encoding="utf-8"))
rows = [row for row in plan["windows"] if row.get("window_id") == sys.argv[2]]
if len(rows) != 1 or rows[0].get("kind") not in {"normal_measurement", "detailed_profile"}:
    raise SystemExit(1)
print(rows[0]["kind"])
PY
)
if [ -z "$WINDOW_KIND" ]; then
  echo "P2 window kind is unavailable" >&2
  exit 37
fi
if [ "$WINDOW_KIND" = detailed_profile ]; then
  PROFILE_ENABLED=1
  require_regular "$ROCPROF"
  if [ ! -x "$ROCPROF" ]; then
    echo "rocprofv3 is not executable" >&2
    exit 37
  fi
fi
PREPARED_COMMIT=$(/usr/bin/python3 - "$OUT/identity.json" <<'PY'
import json
import sys
with open(sys.argv[1], encoding="utf-8") as source:
    value = json.load(source)
print(value["clean_baseline_source"]["git_commit"])
PY
)
if [ "$PREPARED_COMMIT" != "$SOURCE_COMMIT" ]; then
  echo "prepared source commit differs from supplied clean worktree" >&2
  exit 38
fi
if [ "$(systemctl show "$SERVICE" -p RuntimeDirectoryPreserve --value)" != yes ]; then
  echo "RuntimeDirectoryPreserve=yes is required before a P2 window" >&2
  exit 39
fi

pre_state=$(systemctl show "$SERVICE" -p ActiveState --value)
pre_sub=$(systemctl show "$SERVICE" -p SubState --value)
pre_pid=$(systemctl show "$SERVICE" -p MainPID --value)
pre_restarts=$(systemctl show "$SERVICE" -p NRestarts --value)
manifest_sha=$(sha256sum "$MANIFEST" | awk '{print $1}')
printf '%s\n' \
  "captured_at_utc=$(date --iso-8601=seconds --utc)" \
  "active_state=$pre_state" \
  "sub_state=$pre_sub" \
  "main_pid=$pre_pid" \
  "nrestarts=$pre_restarts" \
  "manifest_sha256=$manifest_sha" \
  "runtime_directory_preserve=yes" \
  > "$PRE_STOP"
if [ "$pre_state" != active ] || [ "$pre_sub" != running ] || [ "$pre_pid" -le 0 ]; then
  echo "service pre-stop state is not active/running" >&2
  exit 40
fi

event stop_invoked "single_stop"
STOP_ATTEMPTED=1
systemctl stop "$SERVICE" > "$OUT/windows/$WINDOW_ID-service-stop.stdout" 2> "$OUT/windows/$WINDOW_ID-service-stop.stderr"
stop_rc=$?
event stop_returned "exit_code=$stop_rc"
if [ "$stop_rc" -ne 0 ]; then
  OUTCOME=service_stop_failure
  restore
  write_result
  exit "$stop_rc"
fi
if ! runuser -u homelab1 -- /usr/bin/python3 "$LOCK_PROBE" "$LOCK_PROBE_OUTPUT" "$LOCK"; then
  OUTCOME=lock_probe_failure_no_target_capture
  restore
  write_result
  exit 41
fi
/usr/bin/python3 "$GUARD_TOOL" --output "$GUARD_BEFORE" --guard-bin "$GUARD_BIN" --health-phase "p2-$WINDOW_ID-before"
BEFORE_GUARD_RC=$?
if [ "$BEFORE_GUARD_RC" -ne 0 ]; then
  OUTCOME=r9700_guard_before_failure_no_target_capture
  restore
  write_result
  exit 42
fi
event r9700_guard_before_valid ""

event executor_invoked "window=$WINDOW_ID"
runuser -u homelab1 -- /usr/bin/env -i "${P2_ENV[@]}" \
  /bin/bash -ceu '
    lock=$1
    executor=$2
    preparation=$3
    staging=$4
    window=$5
    output=$6
    manifest=$7
    profile_enabled=$8
    profile_raw=$9
    rocprof=${10}
    exec 9< "$lock"
    flock -n 9
    runner=(/usr/bin/python3 "$executor" \
      --preparation "$preparation" \
      --staging "$staging" \
      --window "$window" \
      --output "$output" \
      --served-manifest "$manifest" \
      --execute \
      --confirm-r9700-window)
    if [ "$profile_enabled" = 1 ]; then
      mkdir -m 700 "$profile_raw"
      /usr/bin/python3 - "$profile_raw/rocprof-provenance.json" "$rocprof" "$window" "${runner[@]}" <<'"'"'PY'"'"'
import hashlib
import json
import os
import sys
from pathlib import Path

output = Path(sys.argv[1])
profiler = Path(sys.argv[2])
raw = profiler.read_bytes()
payload = {
    "schema_version": "ullm.aq4_p2_production_rocprof_provenance.v1",
    "tool": "rocprofv3",
    "profiler_path": str(profiler),
    "profiler_sha256": hashlib.sha256(raw).hexdigest(),
    "window_id": sys.argv[3],
    "runner_argv": sys.argv[4:],
    "profile_timing_used_for_normal_p50_p95": False,
}
fd = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
with os.fdopen(fd, "w", encoding="utf-8") as stream:
    json.dump(payload, stream, ensure_ascii=True, sort_keys=True, indent=2)
    stream.write("\\n")
    stream.flush()
    os.fsync(stream.fileno())
PY
      exec "$rocprof" --log-level error --kernel-trace --hip-runtime-trace \
        --memory-copy-trace --marker-trace --output-format csv \
        --output-directory "$profile_raw" --output-file "$window" -- "${runner[@]}"
    fi
    exec "${runner[@]}"
  ' bash "$LOCK" "$EXECUTOR" "$OUT" "$STAGING" "$WINDOW_ID" "$WINDOW_OUTPUT" "$MANIFEST" "$PROFILE_ENABLED" "$PROFILE_RAW" "$ROCPROF" \
  > "$OUT/windows/$WINDOW_ID-executor.stdout" 2> "$OUT/windows/$WINDOW_ID-executor.stderr"
EXECUTOR_RC=$?
event executor_returned "exit_code=$EXECUTOR_RC"

if [ "$PROFILE_ENABLED" -eq 1 ] && [ "$EXECUTOR_RC" -eq 0 ]; then
  event profile_parse_invoked "window=$WINDOW_ID"
  runuser -u homelab1 -- /usr/bin/env -i HOME=/home/homelab1 PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    /usr/bin/python3 "$PROFILE_PARSER" \
      --profile-dir "$PROFILE_RAW" \
      --window-result "$WINDOW_OUTPUT/window-result.json" \
      --trace-binding "$WINDOW_OUTPUT/trace-hash-binding.json" \
      --output "$PROFILE_REPORT" \
    > "$OUT/windows/$WINDOW_ID-profile.stdout" 2> "$OUT/windows/$WINDOW_ID-profile.stderr"
  PROFILE_RC=$?
  event profile_parse_returned "exit_code=$PROFILE_RC"
fi

if /usr/bin/python3 "$GUARD_TOOL" --output "$GUARD_AFTER" --guard-bin "$GUARD_BIN" --health-phase "p2-$WINDOW_ID-after"; then
  AFTER_GUARD_RC=0
else
  AFTER_GUARD_RC=$?
fi
event r9700_guard_after_returned "exit_code=$AFTER_GUARD_RC"
restore
if [ "$RESTORE_RC" -ne 0 ]; then
  OUTCOME=restore_start_failure
  write_result
  exit 70
fi

post_state=$(systemctl show "$SERVICE" -p ActiveState --value)
post_sub=$(systemctl show "$SERVICE" -p SubState --value)
post_pid=$(systemctl show "$SERVICE" -p MainPID --value)
post_restarts=$(systemctl show "$SERVICE" -p NRestarts --value)
post_manifest_sha=$(sha256sum "$MANIFEST" | awk '{print $1}')
printf '%s\n' \
  "captured_at_utc=$(date --iso-8601=seconds --utc)" \
  "active_state=$post_state" \
  "sub_state=$post_sub" \
  "main_pid=$post_pid" \
  "nrestarts=$post_restarts" \
  "manifest_sha256=$post_manifest_sha" \
  "pre_stop_nrestarts=$pre_restarts" \
  > "$POST_RESTORE"
if [ "$post_state" != active ] || [ "$post_sub" != running ] || [ "$post_pid" -le 0 ] || [ "$post_restarts" != "$pre_restarts" ] || [ "$post_manifest_sha" != "$manifest_sha" ]; then
  OUTCOME=post_restore_service_contract_failure
  write_result
  exit 71
fi
if [ "$EXECUTOR_RC" -ne 0 ]; then
  OUTCOME=executor_failure_no_retry
  write_result
  exit "$EXECUTOR_RC"
fi
if [ "$PROFILE_ENABLED" -eq 1 ] && [ "$PROFILE_RC" -ne 0 ]; then
  OUTCOME=profile_parse_failure_or_unclassified_kernel_time
  write_result
  exit 73
fi
if [ "$AFTER_GUARD_RC" -ne 0 ]; then
  OUTCOME=r9700_guard_after_failure
  write_result
  exit 72
fi
OUTCOME=single_window_completed_partial_observability
write_result
exit 0
