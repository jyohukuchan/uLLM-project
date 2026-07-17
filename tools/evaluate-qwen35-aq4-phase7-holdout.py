#!/usr/bin/env python3
"""Evaluate the one permitted AQ4 Phase 7 formal-holdout observation.

This comparator consumes only pre-existing full-vector source/AQ4 artifacts.
It never starts a model, accesses a GPU, or operates a service.  The formal
split remains the only source of holdout membership; the separate execution
view is accepted solely after its mapping to that formal holdout is verified.

The report deliberately distinguishes the frozen fidelity-policy result from
the older calibration-evidence binding specification.  The latter still says
that a source comparison has zero greedy mismatches and has hash-bound limits
for hidden/logit max-abs, while the frozen fidelity policy uses an agreement
rate and records hidden max-abs diagnostically (and has no logit-max-abs
bound).  This tool does not silently choose between those incompatible rules.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import struct
import sys
from pathlib import Path
from typing import Any, Iterator


REPO = Path(__file__).resolve().parents[1]
SCHEMA = "ullm.aq4_phase7_fidelity_holdout_evaluation.v1"
PREPARATION_SCHEMA = "ullm.aq4_phase7_fidelity_preparation.v1"
EXECUTION_VIEW_SCHEMA = "ullm.aq4_phase7_holdout_execution_view.v1"
SOURCE_CASES_SCHEMA = "ullm.qwen35_aq4_source_calibration_cases.v1"
TOP_K = 10
HIDDEN_SIZE = 4096
VOCAB_SIZE = 248320
F32_BYTES = 4
EPSILON = 1e-12


class EvaluationError(ValueError):
    """A required input is malformed, unbound, or changed while read."""


def _load_module(name: str, filename: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, REPO / "tools" / filename)
    if spec is None or spec.loader is None:  # pragma: no cover - repository corruption
        raise RuntimeError(f"cannot load {filename}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


PROTOCOL = _load_module("aq4_phase7_eval_protocol", "generate-aq4-p2-fidelity-holdout.py")
SPLIT_VALIDATOR = _load_module("aq4_phase7_eval_split_validator", "validate-aq4-p2-fidelity-holdout.py")
FULL_COMPARE = _load_module("aq4_phase7_eval_full_compare", "compare-qwen35-aq4-p2-calibration.py")
FULL_VALIDATE = FULL_COMPARE._VALIDATOR


def require(condition: bool, message: str) -> None:
    if not condition:
        raise EvaluationError(message)


def is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def sha_file(path: Path, label: str) -> str:
    try:
        return FULL_COMPARE.sha256_file(path, label)
    except Exception as error:
        raise EvaluationError(f"{label} SHA-256 failed: {error}") from error


def load_json(path: Path, label: str) -> Any:
    try:
        return FULL_COMPARE.read_json(path, label)
    except Exception as error:
        raise EvaluationError(f"{label} is invalid: {error}") from error


def load_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    try:
        return PROTOCOL.read_jsonl(path, label)
    except Exception as error:
        raise EvaluationError(f"{label} is invalid: {error}") from error


def atomic_json(path: Path, value: Any) -> None:
    if os.path.lexists(path):
        raise EvaluationError(f"refusing to overwrite holdout evaluation: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(value, ensure_ascii=True, sort_keys=True, indent=2, allow_nan=False).encode("utf-8") + b"\n"
    temporary = path.with_name(f".{path.name}.{os.getpid()}.incomplete")
    if os.path.lexists(temporary):
        raise EvaluationError(f"incomplete holdout evaluation already exists: {temporary}")
    with temporary.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    # A prechecked create-new destination and a same-directory replace are
    # sufficient here because the service driver rejects every output marker
    # before it consumes the one permitted holdout window.
    if os.path.lexists(path):
        raise EvaluationError(f"holdout evaluation appeared while writing: {path}")
    temporary.rename(path)
    directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0))
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def chunks(fd: int, offset: int, elements: int, chunk_elements: int) -> Iterator[list[float]]:
    remaining = elements
    cursor = offset
    while remaining:
        count = min(remaining, chunk_elements)
        raw = os.pread(fd, count * F32_BYTES, cursor)
        if len(raw) != count * F32_BYTES:
            raise EvaluationError("full-vector sidecar ended before row boundary")
        yield list(struct.unpack(f"<{count}f", raw))
        cursor += len(raw)
        remaining -= count


def vector_stats(reference: Iterator[list[float]], candidate: Iterator[list[float]], elements: int) -> dict[str, float | int]:
    reference_sq = 0.0
    candidate_sq = 0.0
    dot = 0.0
    delta_sq = 0.0
    max_abs = 0.0
    seen = 0
    for left, right in zip(reference, candidate):
        require(len(left) == len(right), "full-vector chunks differ in length")
        for source, active in zip(left, right):
            require(math.isfinite(source) and math.isfinite(active), "full-vector comparison contains a non-finite value")
            source = float(source)
            active = float(active)
            delta = active - source
            reference_sq += source * source
            candidate_sq += active * active
            dot += source * active
            delta_sq += delta * delta
            max_abs = max(max_abs, abs(delta))
            seen += 1
    require(seen == elements, "full-vector row element count differs")
    try:
        next(reference)
        raise EvaluationError("reference vector has surplus chunks")
    except StopIteration:
        pass
    try:
        next(candidate)
        raise EvaluationError("candidate vector has surplus chunks")
    except StopIteration:
        pass
    reference_norm = math.sqrt(reference_sq)
    candidate_norm = math.sqrt(candidate_sq)
    result = {
        "reference_norm_sq": reference_sq,
        "candidate_norm_sq": candidate_sq,
        "dot": dot,
        "delta_norm_sq": delta_sq,
        "relative_l2": math.sqrt(delta_sq) / max(reference_norm, 1e-30),
        "cosine": dot / max(reference_norm * candidate_norm, 1e-30),
        "max_abs": max_abs,
        "elements": elements,
    }
    require(all(math.isfinite(float(value)) for key, value in result.items() if key != "elements"), "derived vector metric is non-finite")
    return result


def read_source_cases(path: Path, label: str) -> tuple[list[dict[str, Any]], str]:
    try:
        cases, digest = FULL_VALIDATE.load_cases(path)
    except Exception as error:
        raise EvaluationError(f"{label} source cases are invalid: {error}") from error
    require(len(cases) == 24 and all(case["step_count"] == 1 for case in cases), f"{label} source cases must be 24 one-step rows")
    return cases, digest


def validate_preparation(preparation_root: Path) -> dict[str, Any]:
    manifest = load_json(preparation_root / "preparation-manifest.json", "Phase 7 preparation manifest")
    require(isinstance(manifest, dict) and manifest.get("schema_version") == PREPARATION_SCHEMA and manifest.get("status") == "ready_for_cpu_source_and_single_gpu_window", "Phase 7 preparation manifest schema/status differs")
    formal = preparation_root / "formal-split"
    try:
        split = SPLIT_VALIDATOR.validate(formal)
    except Exception as error:
        raise EvaluationError(f"formal split validation failed: {error}") from error
    for name, path in (
        ("formal_split_manifest_sha256", formal / "split-manifest.json"),
        ("formal_policy_sha256", formal / "policy.json"),
        ("formal_calibration_cases_sha256", formal / "calibration-cases.jsonl"),
        ("formal_holdout_cases_sha256", formal / "holdout-cases.jsonl"),
    ):
        require(manifest.get(name) == sha_file(path, name), f"preparation manifest binding differs: {name}")
    view = preparation_root / "holdout-execution-view"
    view_manifest = load_json(view / "split-manifest.json", "holdout execution-view manifest")
    view_binding = load_json(preparation_root / "view-binding.json", "holdout execution-view binding")
    require(isinstance(view_manifest, dict) and view_manifest.get("execution_view_only") is True and view_manifest.get("calibration_bounds_derivation_forbidden") is True, "execution view safety fields differ")
    require(isinstance(view_binding, dict) and view_binding.get("schema_version") == EXECUTION_VIEW_SCHEMA and view_binding.get("status") == "ready_for_one_holdout_capture", "execution-view binding schema/status differs")
    formal_holdout = load_jsonl(formal / "holdout-cases.jsonl", "formal holdout")
    execution_rows = load_jsonl(view / "calibration-cases.jsonl", "execution-view calibration")
    require(len(formal_holdout) == 24 and len(execution_rows) == 24, "formal holdout/execution-view row count differs")
    by_formal = {row.get("case_id"): row for row in formal_holdout}
    by_execution = {row.get("case_id"): row for row in execution_rows}
    require(None not in by_formal and None not in by_execution and set(by_formal) == set(by_execution) and len(by_formal) == 24, "execution-view case set differs from formal holdout")
    identity_fields = ("case_sha256", "fixture_sha256", "prompt_token_ids_sha256", "context_token_ids_sha256", "prompt_tokens", "context_tokens", "baseline_mode", "prefill_requested_m", "resolved_m", "step", "row_count")
    for case_id, formal_row in by_formal.items():
        execution_row = by_execution[case_id]
        require(execution_row.get("subset") == "calibration", f"execution-view subset differs: {case_id}")
        require(all(formal_row.get(field) == execution_row.get(field) for field in identity_fields), f"execution-view identity differs: {case_id}")
    mapping = view_binding.get("mapping")
    require(isinstance(mapping, list) and len(mapping) == 24 and {item.get("case_id") for item in mapping if isinstance(item, dict)} == set(by_formal), "execution-view mapping differs")
    require(view_binding.get("formal_split_manifest_sha256") == sha_file(formal / "split-manifest.json", "formal split manifest"), "execution-view binding formal split differs")
    require(view_binding.get("formal_holdout_sha256") == sha_file(formal / "holdout-cases.jsonl", "formal holdout"), "execution-view binding formal holdout differs")
    require(view_binding.get("execution_split_manifest_sha256") == sha_file(view / "split-manifest.json", "execution-view manifest"), "execution-view binding execution split differs")
    require(view_binding.get("execution_calibration_sha256") == sha_file(view / "calibration-cases.jsonl", "execution-view calibration"), "execution-view binding execution rows differs")
    source_cases = manifest.get("source_cases")
    require(isinstance(source_cases, dict) and isinstance(source_cases.get("execution_holdout"), dict), "preparation has no execution holdout source cases")
    execution_source = Path(source_cases["execution_holdout"].get("path", ""))
    require(execution_source.is_absolute(), "execution holdout source cases path is not absolute")
    _, execution_source_sha = read_source_cases(execution_source, "execution holdout")
    require(execution_source_sha == source_cases["execution_holdout"].get("sha256"), "execution holdout source cases SHA differs")
    return {
        "manifest": manifest,
        "formal_root": formal,
        "view_root": view,
        "formal_holdout": by_formal,
        "execution_rows": by_execution,
        "execution_source_cases": execution_source,
        "execution_source_cases_sha256": execution_source_sha,
        "split_result": split,
    }


def validate_freeze(receipt_path: Path, metrics_path: Path, preparation: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    receipt = load_json(receipt_path, "frozen calibration receipt")
    metrics = load_json(metrics_path, "calibration metrics")
    require(isinstance(receipt, dict) and receipt.get("schema_version") == PROTOCOL.RECEIPT_SCHEMA and receipt.get("status") == "frozen_calibration_envelope", "freeze receipt schema/status differs")
    require(isinstance(metrics, dict) and metrics.get("schema_version") == PROTOCOL.METRICS_SCHEMA and metrics.get("subset") == "calibration" and metrics.get("row_count") == 24, "calibration metrics schema/subset differs")
    formal = preparation["formal_root"]
    split_sha = sha_file(formal / "split-manifest.json", "formal split manifest")
    policy_sha = sha_file(formal / "policy.json", "formal policy")
    metrics_sha = sha_file(metrics_path, "calibration metrics")
    require(receipt.get("split_manifest_sha256") == split_sha and receipt.get("policy_sha256") == policy_sha and receipt.get("metrics_sha256") == metrics_sha, "freeze receipt hash binding differs")
    require(metrics.get("split_manifest_sha256") == split_sha and metrics.get("policy_sha256") == policy_sha, "calibration metrics split/policy binding differs")
    require(receipt.get("holdout_status") == "not_started" and receipt.get("holdout_evaluations_remaining") == 1, "freeze receipt no longer permits exactly one holdout evaluation")
    bounds = receipt.get("derived_bounds")
    require(isinstance(bounds, dict) and set(bounds) == set(PROTOCOL.METRICS), "freeze receipt metric bounds differ")
    return receipt, metrics


def nested_runtime(artifact: dict[str, Any], label: str) -> dict[str, Any]:
    runtime = artifact["manifest"].get("runtime")
    require(isinstance(runtime, dict) and isinstance(runtime.get("runtime"), dict), f"{label} nested runtime is missing")
    return runtime["runtime"]


def assert_source_target_binding(source: dict[str, Any], target: dict[str, Any], preparation: dict[str, Any], calibration_metrics: dict[str, Any]) -> None:
    require(source["manifest"].get("oracle_kind") == "independent_source_full", "holdout source artifact kind differs")
    require(target["manifest"].get("oracle_kind") == "aq4_target", "holdout target artifact kind differs")
    require(source["nonfinite_rows"] == 0 and target["nonfinite_rows"] == 0, "holdout artifact contains non-finite rows")
    source_cases = source["manifest"].get("cases", {})
    target_cases = target["manifest"].get("cases", {})
    expected_cases_path = preparation["execution_source_cases"].resolve()
    expected_cases_sha = preparation["execution_source_cases_sha256"]
    for label, cases in (("source", source_cases), ("target", target_cases)):
        require(Path(cases.get("path", "")).resolve() == expected_cases_path and cases.get("sha256") == expected_cases_sha and cases.get("case_count") == 24 and cases.get("row_count") == 24, f"holdout {label} cases binding differs")
    parent = target["manifest"].get("parent_sampled_oracle")
    require(isinstance(parent, dict) and Path(parent.get("path", "")).resolve() == (source["root"] / "manifest.json").resolve() and parent.get("manifest_sha256") == source["manifest_sha256"], "holdout target direct source parent binding differs")
    source_identity = source["manifest"].get("identity")
    target_identity = target["manifest"].get("identity")
    require(isinstance(source_identity, dict) and isinstance(target_identity, dict), "holdout identities are missing")
    for key in ("model_id", "model_revision"):
        require(source_identity.get(key) == target_identity.get(key), f"holdout source/target identity differs: {key}")
    require(source_identity.get("tokenizer", {}).get("aggregate_sha256") == target_identity.get("tokenizer", {}).get("aggregate_sha256"), "holdout source/target tokenizer differs")
    calibration_identity = calibration_metrics.get("identity")
    require(isinstance(calibration_identity, dict), "calibration metrics identity is missing")
    calibration_source_identity = calibration_identity.get("source_identity")
    calibration_active_identity = calibration_identity.get("active_identity")
    require(isinstance(calibration_source_identity, dict) and isinstance(calibration_active_identity, dict), "calibration source/active identity is missing")
    for key in ("model_id", "model_revision"):
        require(source_identity.get(key) == calibration_source_identity.get(key), f"holdout/calibration source identity differs: {key}")
    require(source_identity.get("source_checkpoint", {}).get("aggregate_sha256") == calibration_source_identity.get("source_checkpoint", {}).get("aggregate_sha256"), "holdout/calibration source checkpoint differs")
    require(source_identity.get("tokenizer", {}).get("aggregate_sha256") == calibration_source_identity.get("tokenizer", {}).get("aggregate_sha256"), "holdout/calibration tokenizer differs")
    for key in ("package_content_sha256", "package_manifest_sha256", "worker_binary_sha256"):
        require(target_identity.get(key) == calibration_active_identity.get(key), f"holdout/calibration active identity differs: {key}")
    target_runtime = nested_runtime(target, "holdout target")
    calibration_active_sha = calibration_metrics.get("active_manifest_sha256")
    require(isinstance(calibration_active_sha, str), "calibration active manifest SHA is missing")
    # The target calibration artifacts are per-case-set, so their artifact
    # manifest SHA differs.  These production identity pins must remain the
    # same across calibration and holdout.
    require(is_sha256(target_runtime.get("served_model_manifest_sha256")), "holdout served manifest runtime field is invalid")
    require(is_sha256(target_runtime.get("package_manifest_sha256")) and is_sha256(target_runtime.get("worker_binary_sha256")) and is_sha256(target_runtime.get("guard_sha256")), "holdout target runtime hash field is invalid")
    expected_runtime = calibration_metrics.get("identity", {}).get("active_identity", {})
    require(target_identity.get("package_manifest_sha256") == expected_runtime.get("package_manifest_sha256") and target_identity.get("worker_binary_sha256") == expected_runtime.get("worker_binary_sha256"), "holdout target package/worker differs from calibration")
    view = preparation["view_root"]
    require(target_runtime.get("split_manifest_sha256") == sha_file(view / "split-manifest.json", "execution-view manifest"), "holdout target split binding differs")
    require(target_runtime.get("policy_sha256") == sha_file(view / "policy.json", "execution-view policy"), "holdout target policy binding differs")
    require(target_runtime.get("calibration_cases_sha256") == sha_file(view / "calibration-cases.jsonl", "execution-view calibration"), "holdout target cases binding differs")
    require(target_runtime.get("device", {}).get("architecture") == "gfx1201", "holdout target device architecture differs")


def check_row_identity(formal_row: dict[str, Any], source_row: dict[str, Any], target_row: dict[str, Any]) -> None:
    require(source_row["input_token_ids_sha256"] == target_row["input_token_ids_sha256"] == formal_row["context_token_ids_sha256"], f"holdout input token identity differs: {formal_row['case_id']}")
    require(source_row["greedy_token_id"] is not None and target_row["greedy_token_id"] is not None, f"holdout greedy token is unavailable: {formal_row['case_id']}")
    require(isinstance(source_row["topk"], list) and isinstance(target_row["topk"], list) and len(source_row["topk"]) == TOP_K and len(target_row["topk"]) == TOP_K, f"holdout top-k is unavailable: {formal_row['case_id']}")


def assess_policy(policy: dict[str, Any], receipt: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = policy.get("metrics")
    bounds = receipt.get("derived_bounds")
    require(isinstance(metrics, dict) and isinstance(bounds, dict) and set(metrics) == set(PROTOCOL.METRICS) and set(bounds) == set(PROTOCOL.METRICS), "frozen policy/receipt metric set differs")
    result: dict[str, Any] = {}
    pathological: dict[str, list[str]] = {"logits_relative_l2": [], "hidden_relative_l2": []}
    for name, expected in PROTOCOL.METRICS.items():
        spec = metrics.get(name)
        receipt_bound = bounds.get(name)
        require(isinstance(spec, dict) and isinstance(receipt_bound, dict), f"frozen metric binding differs: {name}")
        values = [row["metrics"][name] for row in rows]
        require(len(values) == 24 and all(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)) for value in values), f"holdout metric values differ: {name}")
        if name in pathological:
            ceiling = spec.get("pathological_rejection_ceiling")
            require(ceiling == 1.0, f"frozen pathological ceiling differs: {name}")
            pathological[name] = [row["case_id"] for row in rows if float(row["metrics"][name]) > float(ceiling)]
        if spec.get("role") == "diagnostic_only":
            result[name] = {
                "role": "diagnostic_only",
                "observed_max": max(float(value) for value in values),
                "frozen_bound": None,
                "passed": True,
            }
            continue
        direction = spec.get("direction")
        if spec.get("aggregation") == "wilson_lower_one_sided":
            require(all(float(value) in (0.0, 1.0) for value in values), f"binary holdout metric differs: {name}")
            successes = sum(float(value) == 1.0 for value in values)
            observed = PROTOCOL.wilson_lower_one_sided(successes, len(values))
            aggregate = "wilson_lower_one_sided"
            detail = {"successes": successes, "sample_count": len(values), "observed_mean": sum(float(value) for value in values) / len(values)}
        else:
            observed = sum(float(value) for value in values) / len(values)
            aggregate = "mean"
            detail = {"sample_count": len(values)}
        bound = receipt_bound.get("bound")
        require(isinstance(bound, (int, float)) and math.isfinite(float(bound)), f"frozen metric bound differs: {name}")
        if direction == "higher":
            passed = observed + EPSILON >= float(bound)
        elif direction == "lower":
            passed = observed <= float(bound) + EPSILON
        else:
            raise EvaluationError(f"frozen metric direction differs: {name}")
        result[name] = {
            "role": "promotion",
            "aggregation": aggregate,
            "direction": direction,
            "observed": observed,
            "frozen_bound": float(bound),
            "passed": passed,
            **detail,
        }
    pathological_ok = all(not cases for cases in pathological.values())
    promotion_ok = all(item["passed"] for item in result.values() if item["role"] == "promotion")
    return {
        "metrics": result,
        "pathological_relative_l2_rejections": pathological,
        "pathological_relative_l2_passed": pathological_ok,
        "frozen_policy_passed": promotion_ok and pathological_ok,
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    preparation = validate_preparation(args.preparation_root.absolute())
    receipt, calibration_metrics = validate_freeze(args.freeze_receipt.absolute(), args.calibration_metrics.absolute(), preparation)
    try:
        source = FULL_COMPARE.load_artifact(args.source.absolute())
        target = FULL_COMPARE.load_artifact(args.target.absolute())
    except Exception as error:
        raise EvaluationError(f"holdout full-vector artifact validation failed: {error}") from error
    assert_source_target_binding(source, target, preparation, calibration_metrics)
    expected = preparation["formal_holdout"]
    require(set(source["rows"]) == {(case_id, 0) for case_id in expected} and set(target["rows"]) == {(case_id, 0) for case_id in expected}, "holdout artifact row coverage differs from formal holdout")
    rows: list[dict[str, Any]] = []
    with FULL_VALIDATE.stable_fd(source["hidden"], "holdout source hidden") as (source_hidden, _), FULL_VALIDATE.stable_fd(source["logits"], "holdout source logits") as (source_logits, _), FULL_VALIDATE.stable_fd(target["hidden"], "holdout target hidden") as (target_hidden, _), FULL_VALIDATE.stable_fd(target["logits"], "holdout target logits") as (target_logits, _):
        for case_id in sorted(expected):
            formal_row = expected[case_id]
            source_row = source["rows"][(case_id, 0)]
            target_row = target["rows"][(case_id, 0)]
            check_row_identity(formal_row, source_row, target_row)
            hidden = vector_stats(
                chunks(source_hidden, source_row["hidden"]["offset_bytes"], HIDDEN_SIZE, source["chunk_elements"]),
                chunks(target_hidden, target_row["hidden"]["offset_bytes"], HIDDEN_SIZE, target["chunk_elements"]),
                HIDDEN_SIZE,
            )
            logits = vector_stats(
                chunks(source_logits, source_row["logits"]["offset_bytes"], VOCAB_SIZE, source["chunk_elements"]),
                chunks(target_logits, target_row["logits"]["offset_bytes"], VOCAB_SIZE, target["chunk_elements"]),
                VOCAB_SIZE,
            )
            source_top = [item["token_id"] for item in source_row["topk"]]
            target_top = [item["token_id"] for item in target_row["topk"]]
            source_greedy = source_row["greedy_token_id"]
            target_greedy = target_row["greedy_token_id"]
            rows.append(
                {
                    "case_id": case_id,
                    "case_sha256": formal_row["case_sha256"],
                    "fixture_sha256": formal_row["fixture_sha256"],
                    "prompt_token_ids_sha256": formal_row["prompt_token_ids_sha256"],
                    "context_token_ids_sha256": formal_row["context_token_ids_sha256"],
                    "prompt_tokens": formal_row["prompt_tokens"],
                    "context_tokens": formal_row["context_tokens"],
                    "baseline_mode": formal_row["baseline_mode"],
                    "prefill_requested_m": formal_row["prefill_requested_m"],
                    "resolved_m": formal_row["resolved_m"],
                    "step": 0,
                    "row_count": 1,
                    "greedy": {"source": source_greedy, "target": target_greedy, "exact": source_greedy == target_greedy},
                    "ordered_top10": {"source": source_top, "target": target_top, "exact": source_top == target_top, "overlap": len(set(source_top) & set(target_top)) / TOP_K},
                    "metrics": {
                        "token_agreement_rate": float(source_greedy == target_greedy),
                        "topk_overlap_rate_k10": len(set(source_top) & set(target_top)) / TOP_K,
                        "logits_cosine": logits["cosine"],
                        "logits_relative_l2": logits["relative_l2"],
                        "hidden_cosine": hidden["cosine"],
                        "hidden_relative_l2": hidden["relative_l2"],
                        "hidden_max_abs": hidden["max_abs"],
                        "bf16_top1_retained_in_aq4_top10_rate": float(source_greedy in target_top),
                    },
                    "raw": {"hidden": hidden, "logits": logits, "source_top1_retained_in_target_top10": source_greedy in target_top},
                }
            )
    policy = load_json(preparation["formal_root"] / "policy.json", "formal frozen policy")
    policy_assessment = assess_policy(policy, receipt, rows)
    greedy_mismatches = sum(not row["greedy"]["exact"] for row in rows)
    ordered_top10_mismatches = sum(not row["ordered_top10"]["exact"] for row in rows)
    # The frozen policy is fully evaluated above.  The separate binding spec is
    # reported literally, without inventing the missing max-abs limits.
    binding_spec = {
        "document": "docs/specs/aq4-p2-calibration-evidence-binding-v0.1.md",
        "literal_source_comparison_greedy_mismatch_rows_must_equal": 0,
        "observed_greedy_mismatch_rows": greedy_mismatches,
        "literal_greedy_requirement_passed": greedy_mismatches == 0,
        "ordered_top10_mismatch_rows": ordered_top10_mismatches,
        "required_numeric_fields_named_by_spec": ["hidden_relative_l2", "hidden_max_abs", "logits_relative_l2", "logits_max_abs", "top10_overlap"],
        "frozen_policy_fields": sorted(policy["metrics"]),
        "unresolved_contract_difference": "frozen policy has no logits_max_abs bound and marks hidden_max_abs diagnostic_only; its token agreement is Wilson-rate, not zero greedy mismatch",
        "status": "requires_parent_policy_resolution_before_claiming_full_P2_go",
    }
    frozen_status = "go" if policy_assessment["frozen_policy_passed"] else "no-go"
    formal_status = "blocked_contract_resolution"
    result = {
        "schema_version": SCHEMA,
        "status": frozen_status,
        "formal_p2_status": formal_status,
        "holdout_evaluation_count": 1,
        "holdout_evaluation_contract": "one create-new report after a receipt with holdout_evaluations_remaining=1",
        "formal_split": {
            "split_manifest_sha256": sha_file(preparation["formal_root"] / "split-manifest.json", "formal split manifest"),
            "policy_sha256": sha_file(preparation["formal_root"] / "policy.json", "formal policy"),
            "holdout_cases_sha256": sha_file(preparation["formal_root"] / "holdout-cases.jsonl", "formal holdout"),
            "case_count": len(rows),
        },
        "execution_view": {
            "split_manifest_sha256": sha_file(preparation["view_root"] / "split-manifest.json", "execution-view manifest"),
            "calibration_cases_sha256": sha_file(preparation["view_root"] / "calibration-cases.jsonl", "execution-view calibration"),
            "source_cases_sha256": preparation["execution_source_cases_sha256"],
        },
        "calibration_freeze": {
            "receipt_path": str(args.freeze_receipt.absolute()),
            "receipt_sha256": sha_file(args.freeze_receipt.absolute(), "freeze receipt"),
            "calibration_metrics_path": str(args.calibration_metrics.absolute()),
            "calibration_metrics_sha256": sha_file(args.calibration_metrics.absolute(), "calibration metrics"),
        },
        "artifacts": {
            "source_manifest_sha256": source["manifest_sha256"],
            "target_manifest_sha256": target["manifest_sha256"],
            "source_path": str(source["root"].resolve()),
            "target_path": str(target["root"].resolve()),
        },
        "rows": rows,
        "frozen_policy_assessment": policy_assessment,
        "quality_task": {
            "kind": policy["quality_task"]["kind"],
            "score": policy["quality_task"]["score"],
            "observed_mean": sum(row["metrics"]["bf16_top1_retained_in_aq4_top10_rate"] for row in rows) / len(rows),
            "frozen_policy_result": policy_assessment["metrics"]["bf16_top1_retained_in_aq4_top10_rate"],
        },
        "binding_spec_observation": binding_spec,
    }
    atomic_json(args.output.absolute(), result)
    return result


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preparation-root", type=Path, required=True)
    parser.add_argument("--freeze-receipt", type=Path, required=True)
    parser.add_argument("--calibration-metrics", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--confirm-holdout-once", action="store_true")
    args = parser.parse_args(argv)
    if not args.confirm_holdout_once:
        parser.error("--confirm-holdout-once is required")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        result = evaluate(args)
    except (EvaluationError, OSError, ValueError, RuntimeError) as error:
        print(f"AQ4 Phase 7 holdout evaluation failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps({"status": result["status"], "formal_p2_status": result["formal_p2_status"], "output": str(args.output.absolute())}, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
