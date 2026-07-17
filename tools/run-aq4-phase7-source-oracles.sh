#!/usr/bin/env bash
# Prepare the two BF16 full-vector inputs for the Phase 7 formal gate.
#
# This script is deliberately CPU-only.  It masks CUDA/HIP visibility before
# each exporter process and never reads an active served-model manifest,
# touches systemd, or opens the R9700 lock.  The holdout source vectors are
# sealed inputs only; no target comparison or holdout aggregate is calculated
# here, so the frozen envelope is still derived before the target holdout is
# observed in the one GPU service-stop window.

set -euo pipefail
umask 077

if [ "$#" -ne 2 ] || { [ "$2" != "--preflight" ] && [ "$2" != "--confirm-cpu-source-capture" ]; }; then
  echo "usage: $0 PREPARATION_ROOT --preflight|--confirm-cpu-source-capture" >&2
  exit 64
fi

PREPARATION_ROOT=$1
MODE=$2
REPO=/home/homelab1/coding-local/ultimateLLM/uLLM-project
SOURCE_MODEL=/home/homelab1/datapool/ai_models/safetensors/Qwen/Qwen3.5-9B
LEGACY_SOURCE=/home/homelab1/coding-local/ultimateLLM/uLLM-project/benchmarks/results/2026-07-14/qwen35-9b-aq4-production-opt-v0.1/p2/source-oracle-v2
PREP_TOOL="$REPO/tools/prepare-qwen35-aq4-phase7-fidelity.py"
EXPORT_TOOL="$REPO/tools/export-qwen35-aq4-source-calibration.py"
VALIDATE_TOOL="$REPO/tools/validate-qwen35-aq4-p2-full-calibration.py"
FORMAL_SPLIT="$PREPARATION_ROOT/formal-split"
EXECUTION_VIEW="$PREPARATION_ROOT/holdout-execution-view"
CALIBRATION_CASES="$PREPARATION_ROOT/source-cases/calibration-cases.json"
HOLDOUT_CASES="$PREPARATION_ROOT/source-cases/holdout-execution-cases.json"
OUTPUT_ROOT="$PREPARATION_ROOT/source-oracles"
CALIBRATION_OUTPUT="$OUTPUT_ROOT/calibration"
HOLDOUT_OUTPUT="$OUTPUT_ROOT/holdout"

sha() {
  sha256sum "$1" | awk '{print $1}'
}

require_file() {
  if [ ! -f "$1" ] || [ -L "$1" ]; then
    echo "required regular file is unavailable: $1" >&2
    exit 65
  fi
}

require_dir() {
  if [ ! -d "$1" ] || [ -L "$1" ]; then
    echo "required real directory is unavailable: $1" >&2
    exit 66
  fi
}

require_dir "$PREPARATION_ROOT"
require_dir "$SOURCE_MODEL"
require_dir "$LEGACY_SOURCE"
require_file "$PREP_TOOL"
require_file "$EXPORT_TOOL"
require_file "$VALIDATE_TOOL"
require_file "$CALIBRATION_CASES"
require_file "$HOLDOUT_CASES"
require_file "$FORMAL_SPLIT/split-manifest.json"
require_file "$FORMAL_SPLIT/policy.json"
require_file "$FORMAL_SPLIT/calibration-cases.jsonl"
require_file "$EXECUTION_VIEW/split-manifest.json"
require_file "$EXECUTION_VIEW/policy.json"
require_file "$EXECUTION_VIEW/calibration-cases.jsonl"

/usr/bin/python3 "$PREP_TOOL" --output "$PREPARATION_ROOT" --verify

if [ -e "$OUTPUT_ROOT" ] || [ -L "$OUTPUT_ROOT" ]; then
  echo "refusing to reuse Phase 7 source-oracle output root: $OUTPUT_ROOT" >&2
  exit 67
fi

FORMAL_SPLIT_SHA=$(sha "$FORMAL_SPLIT/split-manifest.json")
FORMAL_POLICY_SHA=$(sha "$FORMAL_SPLIT/policy.json")
FORMAL_CALIBRATION_SHA=$(sha "$FORMAL_SPLIT/calibration-cases.jsonl")
CALIBRATION_CASES_SHA=$(sha "$CALIBRATION_CASES")
VIEW_SPLIT_SHA=$(sha "$EXECUTION_VIEW/split-manifest.json")
VIEW_POLICY_SHA=$(sha "$EXECUTION_VIEW/policy.json")
VIEW_CALIBRATION_SHA=$(sha "$EXECUTION_VIEW/calibration-cases.jsonl")
HOLDOUT_CASES_SHA=$(sha "$HOLDOUT_CASES")

printf '%s\n' \
  "preparation_root=$PREPARATION_ROOT" \
  "formal_split_manifest_sha256=$FORMAL_SPLIT_SHA" \
  "formal_policy_sha256=$FORMAL_POLICY_SHA" \
  "formal_calibration_cases_sha256=$FORMAL_CALIBRATION_SHA" \
  "calibration_source_cases_sha256=$CALIBRATION_CASES_SHA" \
  "execution_view_split_manifest_sha256=$VIEW_SPLIT_SHA" \
  "execution_view_policy_sha256=$VIEW_POLICY_SHA" \
  "execution_view_calibration_cases_sha256=$VIEW_CALIBRATION_SHA" \
  "holdout_execution_source_cases_sha256=$HOLDOUT_CASES_SHA" \
  "gpu_visibility_contract=CUDA_VISIBLE_DEVICES=-1,HIP_VISIBLE_DEVICES=-1,ROCR_VISIBLE_DEVICES=-1" \
  "source_model_loads=2" \
  "target_model_loads=0"

if [ "$MODE" = "--preflight" ]; then
  exit 0
fi

mkdir -m 700 "$OUTPUT_ROOT"

run_source_export() {
  local split_root=$1
  local expected_split=$2
  local expected_policy=$3
  local expected_calibration=$4
  local cases=$5
  local expected_cases=$6
  local output=$7
  env -i \
    HOME=/home/homelab1 \
    PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    TRANSFORMERS_OFFLINE=1 \
    CUDA_VISIBLE_DEVICES=-1 \
    HIP_VISIBLE_DEVICES=-1 \
    ROCR_VISIBLE_DEVICES=-1 \
    ULLM_HIP_VISIBLE_DEVICES=-1 \
    /usr/bin/python3 "$EXPORT_TOOL" \
      --model-dir "$SOURCE_MODEL" \
      --cases "$cases" \
      --output "$output" \
      --legacy-oracle "$LEGACY_SOURCE" \
      --split-root "$split_root" \
      --expected-split-manifest-sha256 "$expected_split" \
      --expected-policy-sha256 "$expected_policy" \
      --expected-calibration-cases-sha256 "$expected_calibration" \
      --expected-cases-sha256 "$expected_cases" \
      --threads 32
}

run_source_export "$FORMAL_SPLIT" "$FORMAL_SPLIT_SHA" "$FORMAL_POLICY_SHA" "$FORMAL_CALIBRATION_SHA" "$CALIBRATION_CASES" "$CALIBRATION_CASES_SHA" "$CALIBRATION_OUTPUT"
/usr/bin/python3 "$VALIDATE_TOOL" --artifact "$CALIBRATION_OUTPUT"

run_source_export "$EXECUTION_VIEW" "$VIEW_SPLIT_SHA" "$VIEW_POLICY_SHA" "$VIEW_CALIBRATION_SHA" "$HOLDOUT_CASES" "$HOLDOUT_CASES_SHA" "$HOLDOUT_OUTPUT"
/usr/bin/python3 "$VALIDATE_TOOL" --artifact "$HOLDOUT_OUTPUT"
