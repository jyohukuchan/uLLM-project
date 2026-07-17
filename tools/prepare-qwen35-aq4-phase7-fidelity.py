#!/usr/bin/env python3
"""Prepare a new, hash-bound AQ4 Phase 7 P2 fidelity split.

The 2026-07-15 fidelity split is evidence-only and must not be measured
again.  This tool creates a different 48-case production-profile fixture
pool, checks it against that retired split and the Phase 1--6 diagnostic
contexts, then delegates the official 24/24 split and frozen policy bytes to
``generate-aq4-p2-fidelity-holdout.py``.

The existing GPU capture binary accepts only a ``calibration-cases.jsonl``
view.  ``holdout-execution-view`` is therefore a deliberately separate,
hash-bound execution adapter containing the official holdout rows as its
capture input.  It is *not* an alternate formal split and cannot be used to
derive calibration bounds.  The Phase 7 evaluator verifies the mapping back
to ``formal-split/holdout-cases.jsonl`` before accepting any result.

This is fixture/preparation-only code: it does not load a model, query a GPU,
read an active served-model manifest, or operate a service.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Iterable


REPO = Path(__file__).resolve().parents[1]
PREPARATION_SCHEMA = "ullm.aq4_phase7_fidelity_preparation.v1"
AUDIT_SCHEMA = "ullm.aq4_phase7_fidelity_selection_audit.v1"
EXECUTION_VIEW_SCHEMA = "ullm.aq4_phase7_holdout_execution_view.v1"
SOURCE_CASES_SCHEMA = "ullm.qwen35_aq4_source_calibration_cases.v1"
FIXTURE_SCHEMA = "ullm.aq4_p2_case_fixture.v1"
INDEX_SCHEMA = "ullm.aq4_p2_fixture_index.v1"
EXPANDED_SCHEMA = "ullm.aq4_production_p2_expanded.v2"

PROMPT_LENGTHS = (1011, 1024, 1339, 2048)
BASELINE_MODES = ("all_m1", "cold_batched")
PREFILL_WIDTHS = (1, 8, 16, 32, 64, 128)
TOKEN_DOMAIN = b"ullm.aq4_phase7_independent_fixture.v1\0"
CASE_PREFIX = "p2-phase7-independent-production_server-cold_prefill"
DEFAULT_OLD_SPLIT = REPO / "benchmarks/results/2026-07-14/qwen35-9b-aq4-production-opt-v0.1/p2/fidelity-holdout-split-v0.1"
DEFAULT_OLD_NO_GO = REPO / "benchmarks/results/2026-07-15/qwen35-9b-aq4-production-opt-v0.1/p2/fidelity-calibration-active-v0.1/recovery/active-attempt7-schema-recovery-v0.1/calibration-no-go.json"
DEFAULT_PHASE_CASES = REPO / "tests/fixtures/qwen35-aq4-p2-oracle/cases.json"
DEFAULT_PHASE_CONTEXTS = REPO / "tests/fixtures/qwen35-aq4-layer0-hybrid-contexts-v0.1.json"
DEFAULT_INPUT_CONTROLS = (
    REPO / "tests/fixtures/aq4-p2-input-controls/gateway-request.json",
    REPO / "tests/fixtures/aq4-p2-input-controls/pure-prefill.json",
)


class PreparationError(ValueError):
    """The independent selection or its verification is invalid."""


def _load_module(name: str, filename: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, REPO / "tools" / filename)
    if spec is None or spec.loader is None:  # pragma: no cover - repository corruption
        raise RuntimeError(f"cannot load {filename}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


PROTOCOL = _load_module("aq4_phase7_fidelity_protocol", "generate-aq4-p2-fidelity-holdout.py")
SPLIT_VALIDATOR = _load_module("aq4_phase7_split_validator", "validate-aq4-p2-fidelity-holdout.py")


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha_file(path: Path, label: str) -> str:
    if path.is_symlink() or not path.is_file():
        raise PreparationError(f"{label} must be a regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path, label: str) -> Any:
    if path.is_symlink() or not path.is_file():
        raise PreparationError(f"{label} must be a regular file: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=PROTOCOL.pairs, parse_constant=PROTOCOL.no_constants)
    except (OSError, UnicodeError, json.JSONDecodeError, PROTOCOL.ProtocolError) as error:
        raise PreparationError(f"invalid {label}: {error}") from error


def write_json_new(path: Path, value: Any) -> None:
    if os.path.lexists(path):
        raise PreparationError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(value, ensure_ascii=True, sort_keys=True, indent=2, allow_nan=False).encode("utf-8") + b"\n"
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def write_text_new(path: Path, text: str) -> None:
    if os.path.lexists(path):
        raise PreparationError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())


def read_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        raise PreparationError(f"{label} must be a regular file: {path}")
    rows: list[dict[str, Any]] = []
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            value = json.loads(raw, object_pairs_hook=PROTOCOL.pairs, parse_constant=PROTOCOL.no_constants)
        except (UnicodeError, json.JSONDecodeError, PROTOCOL.ProtocolError) as error:
            raise PreparationError(f"invalid {label} line {number}: {error}") from error
        if not isinstance(value, dict):
            raise PreparationError(f"{label} line {number} is not an object")
        rows.append(value)
    return rows


def write_jsonl_new(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    encoded = "".join(json.dumps(row, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n" for row in rows)
    write_text_new(path, encoded)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PreparationError(message)


def token_id_stream(case_id: str, count: int, vocab_size: int) -> list[int]:
    """Derive non-special, deterministic fixture IDs from a new domain.

    Qwen3.5's added/special tokens begin above the normal base vocabulary
    range.  The conservative 248000 ceiling also keeps this fixture generator
    usable against a small test vocabulary.
    """

    require(2 <= vocab_size <= 248320, "vocab size is outside Qwen3.5 bounds")
    safe_limit = min(vocab_size, 248000)
    require(safe_limit >= 2, "vocab has no safe ordinary token range")
    result: list[int] = []
    block_index = 0
    while len(result) < count:
        seed = hashlib.sha256(TOKEN_DOMAIN + case_id.encode("ascii") + block_index.to_bytes(8, "big")).digest()
        for offset in range(0, len(seed), 4):
            if len(result) == count:
                break
            value = int.from_bytes(seed[offset : offset + 4], "big")
            result.append(1 + value % (safe_limit - 1))
        block_index += 1
    return result


def fixture_case_id(prompt_tokens: int, baseline_mode: str, requested_m: int) -> str:
    return f"{CASE_PREFIX}-{baseline_mode}-n{prompt_tokens}-m{requested_m}-r9700-rdna4-aq4_0_target"


def make_expanded_case(case_id: str, prompt_tokens: int, baseline_mode: str, requested_m: int) -> dict[str, Any]:
    resolved_m = 1 if baseline_mode == "all_m1" else requested_m
    item: dict[str, Any] = {
        "baseline_mode": baseline_mode,
        "cached_prefix_tokens": 0,
        "case_id": case_id,
        "case_sha256": None,
        "context_tokens": prompt_tokens,
        "control": {
            "control_id": "aq4_0_target",
            "format_id": "AQ4_0",
            "implementation_id": "qwen35_aq4_rdna4_v1",
            "promotion_eligible": True,
            "role": "target",
        },
        "control_id": "aq4_0_target",
        "decode_request_count": 0,
        "decode_start_tokens": 0,
        "device": {
            "architecture": "gfx1201",
            "backend": "hip",
            "device_id": "r9700-rdna4",
            "name": "AMD Radeon AI PRO R9700",
            "runtime_device_index": 1,
        },
        "fixture_id": case_id,
        "format_id": "AQ4_0",
        "generated_tokens": 0,
        "implementation_id": "qwen35_aq4_rdna4_v1",
        "mode": baseline_mode,
        "path_oracle_case_id": None,
        "path_oracle_result_sha256": None,
        "phase": "cold_prefill",
        "prefill_requested_m": requested_m,
        "prompt_tokens": prompt_tokens,
        "request_count": 1,
        "resolved_m": resolved_m,
        "sampling": {"mode": "greedy", "seed": 0, "temperature": 0.0, "top_k": 1, "top_p": 1.0},
        "scope": "production_server",
        "stage_id": "representative",
        "stage_order": 2,
    }
    item["case_sha256"] = PROTOCOL.case_hash(item)
    return item


def normalise_tokens(value: Any, label: str) -> list[int]:
    if not isinstance(value, list) or not value or any(type(token) is not int or token < 0 or token >= 248320 for token in value):
        raise PreparationError(f"{label} must be a non-empty Qwen3.5 token list")
    return list(value)


def diagnostic_contexts(phase_cases: Path, phase_contexts: Path, extra_inputs: Iterable[Path]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return every inspectable Phase 1--6 context and its provenance.

    The hybrid fixture is authoritative for the three distinct contexts used
    throughout Phase 1--6; input controls additionally demonstrate that the
    short prompt itself was excluded even where no generated continuation was
    recorded in a source-case file.
    """

    contexts: list[dict[str, Any]] = []
    source_files: list[dict[str, Any]] = []

    phase_case_payload = load_json(phase_cases, "Phase 1--6 source cases")
    source_files.append({"path": str(phase_cases.resolve()), "sha256": sha_file(phase_cases, "Phase 1--6 source cases")})
    if not isinstance(phase_case_payload, dict) or not isinstance(phase_case_payload.get("cases"), list):
        raise PreparationError("Phase 1--6 source cases schema differs")
    for item in phase_case_payload["cases"]:
        if not isinstance(item, dict):
            raise PreparationError("Phase 1--6 source case differs")
        tokens = normalise_tokens(item.get("prompt_token_ids"), "Phase 1--6 prompt tokens")
        contexts.append({"source": "qwen35-aq4-p2-oracle/cases.json", "case_id": item.get("case_id"), "step": 0, "token_ids": tokens, "context_token_ids_sha256": PROTOCOL.context_hash(tokens)})

    hybrid_payload = load_json(phase_contexts, "Phase 1--6 hybrid contexts")
    source_files.append({"path": str(phase_contexts.resolve()), "sha256": sha_file(phase_contexts, "Phase 1--6 hybrid contexts")})
    if not isinstance(hybrid_payload, dict) or not isinstance(hybrid_payload.get("cases"), list):
        raise PreparationError("Phase 1--6 hybrid contexts schema differs")
    for item in hybrid_payload["cases"]:
        if not isinstance(item, dict):
            raise PreparationError("Phase 1--6 hybrid context differs")
        tokens = normalise_tokens(item.get("context_token_ids"), "Phase 1--6 hybrid context tokens")
        contexts.append({"source": "qwen35-aq4-layer0-hybrid-contexts-v0.1.json", "case_id": item.get("case_id"), "step": item.get("step"), "token_ids": tokens, "context_token_ids_sha256": PROTOCOL.context_hash(tokens)})

    for control_path in extra_inputs:
        if not control_path.exists():
            continue
        payload = load_json(control_path, "Phase 1--6 input control")
        source_files.append({"path": str(control_path.resolve()), "sha256": sha_file(control_path, "Phase 1--6 input control")})
        if not isinstance(payload, dict) or "prompt_token_ids" not in payload:
            raise PreparationError(f"Phase 1--6 input control schema differs: {control_path}")
        tokens = normalise_tokens(payload["prompt_token_ids"], "Phase 1--6 input control tokens")
        contexts.append({"source": str(control_path.relative_to(REPO)), "case_id": payload.get("case_id", control_path.stem), "step": 0, "token_ids": tokens, "context_token_ids_sha256": PROTOCOL.context_hash(tokens)})

    distinct = {(item["case_id"], item["step"], item["context_token_ids_sha256"]): item for item in contexts}
    return [distinct[key] for key in sorted(distinct, key=lambda key: (str(key[0]), str(key[1]), key[2]))], source_files


def old_split_rows(root: Path) -> tuple[list[dict[str, Any]], dict[str, str]]:
    try:
        validation = SPLIT_VALIDATOR.validate(root)
    except Exception as error:
        raise PreparationError(f"retired 2026-07-15 split validation failed: {error}") from error
    calibration = read_jsonl(root / "calibration-cases.jsonl", "retired calibration cases")
    holdout = read_jsonl(root / "holdout-cases.jsonl", "retired holdout cases")
    require(len(calibration) == 24 and len(holdout) == 24, "retired split count differs")
    hashes = {
        "split_manifest_sha256": str(validation["split_manifest_sha256"]),
        "policy_sha256": str(validation["policy_sha256"]),
        "calibration_cases_sha256": sha_file(root / "calibration-cases.jsonl", "retired calibration cases"),
        "holdout_cases_sha256": sha_file(root / "holdout-cases.jsonl", "retired holdout cases"),
    }
    return calibration + holdout, hashes


def old_no_go_cases(path: Path) -> dict[str, Any]:
    payload = load_json(path, "2026-07-15 no-go receipt")
    ids: list[str] = []
    if isinstance(payload, dict):
        candidates = payload.get("failed_case_ids")
        if isinstance(candidates, list):
            ids = [value for value in candidates if isinstance(value, str)]
        elif isinstance(payload.get("failures"), list):
            ids = [item.get("case_id") for item in payload["failures"] if isinstance(item, dict) and isinstance(item.get("case_id"), str)]
        elif isinstance(payload.get("relative_l2_rejections"), dict) and isinstance(payload["relative_l2_rejections"].get("cases"), list):
            ids = [item.get("case_id") for item in payload["relative_l2_rejections"]["cases"] if isinstance(item, dict) and isinstance(item.get("case_id"), str)]
    require(ids, "2026-07-15 no-go receipt has no failed case IDs")
    return {"path": str(path.resolve()), "sha256": sha_file(path, "2026-07-15 no-go receipt"), "case_ids": sorted(set(ids))}


def build_fixture_binding(root: Path, vocab_size: int) -> tuple[Path, Path, list[dict[str, Any]]]:
    binding = root / "fixture-binding"
    fixtures = binding / "fixtures"
    cases: list[dict[str, Any]] = []
    index_rows: list[dict[str, Any]] = []
    for prompt_tokens in PROMPT_LENGTHS:
        for mode in BASELINE_MODES:
            for requested_m in PREFILL_WIDTHS:
                case_id = fixture_case_id(prompt_tokens, mode, requested_m)
                case = make_expanded_case(case_id, prompt_tokens, mode, requested_m)
                tokens = token_id_stream(case_id, prompt_tokens, vocab_size)
                fixture_path = fixtures / f"{case_id}.json"
                fixture = {"schema_version": FIXTURE_SCHEMA, "cases": [{"case_id": case_id, "prompt_token_ids": tokens}]}
                write_json_new(fixture_path, fixture)
                fixture_sha = sha_file(fixture_path, f"fixture {case_id}")
                index_rows.append(
                    {
                        "case_id": case_id,
                        "case_sha256": case["case_sha256"],
                        "context_tokens": prompt_tokens,
                        "fixture_path": str(fixture_path.relative_to(binding)),
                        "fixture_sha256": fixture_sha,
                        "generated_tokens": 0,
                        "prompt_token_ids_sha256": sha_bytes(canonical(tokens)),
                        "prompt_tokens": prompt_tokens,
                        "runnable_reason": "Phase 7 independent full-model cold-prefill fixture",
                    }
                )
                cases.append(case)
    require(len(cases) == 48, "independent profile does not contain 48 cases")
    expanded = {
        "schema_version": EXPANDED_SCHEMA,
        "status": "phase7_independent_fixture_pool",
        "fixture_domain": TOKEN_DOMAIN.decode("ascii").replace("\x00", "\\0"),
        "cases": sorted(cases, key=lambda item: item["case_id"]),
    }
    expanded_path = binding / "expanded.json"
    write_json_new(expanded_path, expanded)
    index_path = binding / "fixture-index.json"
    write_json_new(
        index_path,
        {
            "schema_version": INDEX_SCHEMA,
            "case_count": len(index_rows),
            "subset": "phase7_independent_full_context_step_zero",
            "contract": "new fixture pool; retired 2026-07-15 split and Phase 1--6 contexts are excluded by selection-audit.json",
            "expanded_manifest_sha256": sha_file(expanded_path, "expanded independent manifest"),
            "cases": sorted(index_rows, key=lambda item: item["case_id"]),
        },
    )
    return expanded_path, index_path, cases


def make_source_cases(rows: list[dict[str, Any]], destination: Path) -> None:
    cases: list[dict[str, Any]] = []
    for row in sorted(rows, key=lambda item: item["case_id"]):
        fixture_path = Path(row["fixture_path"])
        fixture = load_json(fixture_path, f"fixture {row['case_id']}")
        fixture_cases = fixture.get("cases") if isinstance(fixture, dict) else None
        require(isinstance(fixture_cases, list) and len(fixture_cases) == 1, f"fixture schema differs: {row['case_id']}")
        tokens = normalise_tokens(fixture_cases[0].get("prompt_token_ids"), f"fixture tokens {row['case_id']}")
        require(fixture_cases[0].get("case_id") == row["case_id"], f"fixture case ID differs: {row['case_id']}")
        require(len(tokens) == row["prompt_tokens"], f"fixture length differs: {row['case_id']}")
        require(PROTOCOL.context_hash(tokens) == row["context_token_ids_sha256"], f"fixture context hash differs: {row['case_id']}")
        cases.append(
            {
                "case_id": row["case_id"],
                "prompt_token_ids": tokens,
                "step_count": 1,
                "semantic_input_id": row["case_id"],
                "observation": "fidelity_full_context_step0",
            }
        )
    require(len(cases) == 24, "source case contract must contain exactly 24 rows")
    write_json_new(destination, {"schema_version": SOURCE_CASES_SCHEMA, "cases": cases})


def clone_with_subset(rows: list[dict[str, Any]], subset: str) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in rows:
        cloned = json.loads(json.dumps(row))
        cloned["subset"] = subset
        result.append(cloned)
    return sorted(result, key=lambda item: item["case_id"])


def write_standard_split_sums(root: Path) -> None:
    names = ("calibration-cases.jsonl", "holdout-cases.jsonl", "policy.json", "split-manifest.json")
    write_text_new(root / "SHA256SUMS", "".join(f"{sha_file(root / name, name)}  {name}\n" for name in names))


def build_execution_view(root: Path, formal_split: Path) -> tuple[Path, Path]:
    """Build a capture-only mirror of the formal holdout.

    The split's original calibration rows are retained as a non-executed
    companion set solely so the Rust capture binary's fixed 24/24 file
    contract remains satisfied.  The manifest calls this out explicitly;
    ordinary split validation is intentionally *not* run on this adapter.
    """

    formal_manifest = load_json(formal_split / "split-manifest.json", "formal split manifest")
    formal_calibration = read_jsonl(formal_split / "calibration-cases.jsonl", "formal calibration rows")
    formal_holdout = read_jsonl(formal_split / "holdout-cases.jsonl", "formal holdout rows")
    view = root / "holdout-execution-view"
    view.mkdir(mode=0o700)
    view_calibration = clone_with_subset(formal_holdout, "calibration")
    view_holdout = clone_with_subset(formal_calibration, "holdout")
    write_jsonl_new(view / "calibration-cases.jsonl", view_calibration)
    write_jsonl_new(view / "holdout-cases.jsonl", view_holdout)
    shutil.copyfile(formal_split / "policy.json", view / "policy.json")
    view_manifest = {
        "schema_version": PROTOCOL.SPLIT_SCHEMA,
        "status": "ready_for_calibration",
        "execution_view_schema_version": EXECUTION_VIEW_SCHEMA,
        "execution_view_only": True,
        "calibration_bounds_derivation_forbidden": True,
        "formal_split_root": str(formal_split.resolve()),
        "formal_split_manifest_sha256": sha_file(formal_split / "split-manifest.json", "formal split manifest"),
        "formal_policy_sha256": sha_file(formal_split / "policy.json", "formal policy"),
        "formal_holdout_sha256": sha_file(formal_split / "holdout-cases.jsonl", "formal holdout rows"),
        "formal_calibration_sha256": sha_file(formal_split / "calibration-cases.jsonl", "formal calibration rows"),
        "calibration_sha256": sha_file(view / "calibration-cases.jsonl", "execution-view calibration rows"),
        "holdout_sha256": sha_file(view / "holdout-cases.jsonl", "execution-view companion rows"),
        "policy_sha256": sha_file(view / "policy.json", "execution-view policy"),
        "attempt2_exclusions": formal_manifest["attempt2_exclusions"],
        "row_mapping": "execution calibration rows are byte-identical formal holdout rows except subset=calibration; see view-binding.json",
    }
    write_json_new(view / "split-manifest.json", view_manifest)
    write_standard_split_sums(view)
    binding_rows = []
    formal_by_id = {row["case_id"]: row for row in formal_holdout}
    execution_by_id = {row["case_id"]: row for row in view_calibration}
    require(set(formal_by_id) == set(execution_by_id) and len(formal_by_id) == 24, "execution view does not contain exactly the formal holdout")
    identity_fields = ("case_sha256", "fixture_sha256", "prompt_token_ids_sha256", "context_token_ids_sha256", "prompt_tokens", "context_tokens", "baseline_mode", "prefill_requested_m", "resolved_m", "step", "row_count")
    for case_id in sorted(formal_by_id):
        formal_row = formal_by_id[case_id]
        execution_row = execution_by_id[case_id]
        require(all(formal_row[field] == execution_row[field] for field in identity_fields), f"execution view identity differs: {case_id}")
        binding_rows.append({"case_id": case_id, **{field: formal_row[field] for field in identity_fields}})
    binding_path = root / "view-binding.json"
    write_json_new(
        binding_path,
        {
            "schema_version": EXECUTION_VIEW_SCHEMA,
            "status": "ready_for_one_holdout_capture",
            "formal_split_manifest_sha256": sha_file(formal_split / "split-manifest.json", "formal split manifest"),
            "formal_policy_sha256": sha_file(formal_split / "policy.json", "formal policy"),
            "formal_holdout_sha256": sha_file(formal_split / "holdout-cases.jsonl", "formal holdout rows"),
            "execution_split_manifest_sha256": sha_file(view / "split-manifest.json", "execution-view manifest"),
            "execution_policy_sha256": sha_file(view / "policy.json", "execution-view policy"),
            "execution_calibration_sha256": sha_file(view / "calibration-cases.jsonl", "execution-view calibration rows"),
            "formal_holdout_case_count": len(binding_rows),
            "mapping": binding_rows,
        },
    )
    return view, binding_path


def selection_audit(
    root: Path,
    new_cases: list[dict[str, Any]],
    formal_split: Path,
    old_root: Path,
    old_no_go: Path,
    phase_cases: Path,
    phase_contexts: Path,
    extra_inputs: Iterable[Path],
) -> Path:
    old_rows, old_hashes = old_split_rows(old_root)
    diagnostics, diagnostic_sources = diagnostic_contexts(phase_cases, phase_contexts, extra_inputs)
    retired_no_go = old_no_go_cases(old_no_go)
    formal_rows = read_jsonl(formal_split / "calibration-cases.jsonl", "formal calibration rows") + read_jsonl(formal_split / "holdout-cases.jsonl", "formal holdout rows")
    require(len(formal_rows) == 48, "formal split does not contain 48 rows")
    new_case_ids = {row["case_id"] for row in formal_rows}
    new_case_hashes = {row["case_sha256"] for row in formal_rows}
    new_context_hashes = {row["context_token_ids_sha256"] for row in formal_rows}
    old_case_ids = {row["case_id"] for row in old_rows}
    old_case_hashes = {row["case_sha256"] for row in old_rows}
    old_context_hashes = {row["context_token_ids_sha256"] for row in old_rows}
    diagnostic_case_ids = {str(row["case_id"]) for row in diagnostics if row.get("case_id") is not None}
    diagnostic_hashes = {row["context_token_ids_sha256"] for row in diagnostics}
    policy_hashes = set(PROTOCOL.ATTEMPT2_CONTEXT_HASHES)
    intersections = {
        "retired_split_case_ids": sorted(new_case_ids & old_case_ids),
        "retired_split_case_sha256": sorted(new_case_hashes & old_case_hashes),
        "retired_split_context_sha256": sorted(new_context_hashes & old_context_hashes),
        "phase1_to_6_diagnostic_case_ids": sorted(new_case_ids & diagnostic_case_ids),
        "phase1_to_6_diagnostic_context_sha256": sorted(new_context_hashes & diagnostic_hashes),
        "protocol_attempt2_context_sha256": sorted(new_context_hashes & policy_hashes),
        "retired_2026_07_15_no_go_case_ids": sorted(new_case_ids & set(retired_no_go["case_ids"])),
    }
    require(all(not values for values in intersections.values()), f"new fixture selection overlaps prohibited input: {intersections}")
    actual_phase_hashes = sorted({row["context_token_ids_sha256"] for row in diagnostics})
    require(len(actual_phase_hashes) == 3, "Phase 1--6 diagnostic context audit no longer has exactly three contexts")
    source_case_hashes = {case["case_sha256"] for case in new_cases}
    require(source_case_hashes == new_case_hashes, "formal split case hashes differ from generated independent pool")
    audit_path = root / "selection-audit.json"
    write_json_new(
        audit_path,
        {
            "schema_version": AUDIT_SCHEMA,
            "status": "valid_disjoint_selection",
            "selection_method": "new deterministic TOKEN_DOMAIN fixture pool; official per-stratum SHA-256 split; no observed fidelity result was used",
            "new_fixture_domain": TOKEN_DOMAIN.decode("ascii").replace("\x00", "\\0"),
            "formal_split": {
                "root": str(formal_split.resolve()),
                "split_manifest_sha256": sha_file(formal_split / "split-manifest.json", "formal split manifest"),
                "policy_sha256": sha_file(formal_split / "policy.json", "formal policy"),
                "calibration_cases_sha256": sha_file(formal_split / "calibration-cases.jsonl", "formal calibration rows"),
                "holdout_cases_sha256": sha_file(formal_split / "holdout-cases.jsonl", "formal holdout rows"),
                "case_count": 48,
                "case_ids": sorted(new_case_ids),
                "case_sha256": sorted(new_case_hashes),
                "context_token_ids_sha256": sorted(new_context_hashes),
            },
            "retired_2026_07_15_split": {"root": str(old_root.resolve()), "case_count": len(old_rows), **old_hashes, "case_ids": sorted(old_case_ids), "case_sha256": sorted(old_case_hashes), "context_token_ids_sha256": sorted(old_context_hashes)},
            "retired_2026_07_15_no_go": retired_no_go,
            "phase1_to_6_diagnostic": {
                "source_files": diagnostic_sources,
                "contexts": diagnostics,
                "distinct_context_token_ids_sha256": actual_phase_hashes,
                "protocol_attempt2_context_hashes": sorted(policy_hashes),
            },
            "intersections": intersections,
        },
    )
    return audit_path


def top_level_sums(root: Path) -> None:
    paths = sorted(path for path in root.rglob("*") if path.is_file() and path.name != "SHA256SUMS")
    write_text_new(root / "SHA256SUMS", "".join(f"{sha_file(path, str(path.relative_to(root)))}  {path.relative_to(root).as_posix()}\n" for path in paths))


def parse_vocab_size(model_dir: Path | None, explicit: int | None) -> int:
    if explicit is not None:
        require(2 <= explicit <= 248320, "--vocab-size is outside Qwen3.5 bounds")
        return explicit
    if model_dir is None:
        raise PreparationError("--source-model-dir or --vocab-size is required")
    config = load_json(model_dir / "config.json", "source model config")
    candidates = [config.get("vocab_size")] if isinstance(config, dict) else []
    if isinstance(config, dict) and isinstance(config.get("text_config"), dict):
        candidates.append(config["text_config"].get("vocab_size"))
    for candidate in candidates:
        if type(candidate) is int and 2 <= candidate <= 248320:
            return candidate
    raise PreparationError("Qwen3.5 source model config has no valid vocab_size")


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output.absolute()
    require(not os.path.lexists(output), f"refusing to overwrite preparation root: {output}")
    require(output.parent.is_dir() and not output.parent.is_symlink(), f"preparation parent is invalid: {output.parent}")
    old_root = args.old_split_root.absolute()
    old_no_go = args.old_no_go.absolute()
    phase_cases = args.phase_cases.absolute()
    phase_contexts = args.phase_contexts.absolute()
    extra_inputs = tuple(path.absolute() for path in args.input_control)
    vocab_size = parse_vocab_size(args.source_model_dir.absolute() if args.source_model_dir else None, args.vocab_size)
    output.mkdir(mode=0o700)
    try:
        expanded, index, cases = build_fixture_binding(output, vocab_size)
        formal_split = output / "formal-split"
        PROTOCOL.split(argparse.Namespace(expanded=expanded, fixture_index=index, output=formal_split))
        try:
            split_validation = SPLIT_VALIDATOR.validate(formal_split)
        except Exception as error:
            raise PreparationError(f"new formal split validation failed: {error}") from error
        audit_path = selection_audit(output, cases, formal_split, old_root, old_no_go, phase_cases, phase_contexts, extra_inputs)
        calibration_rows = read_jsonl(formal_split / "calibration-cases.jsonl", "formal calibration rows")
        holdout_rows = read_jsonl(formal_split / "holdout-cases.jsonl", "formal holdout rows")
        source_cases_root = output / "source-cases"
        make_source_cases(calibration_rows, source_cases_root / "calibration-cases.json")
        make_source_cases(holdout_rows, source_cases_root / "holdout-cases.json")
        execution_view, view_binding = build_execution_view(output, formal_split)
        execution_rows = read_jsonl(execution_view / "calibration-cases.jsonl", "execution-view holdout rows")
        make_source_cases(execution_rows, source_cases_root / "holdout-execution-cases.json")
        manifest_path = output / "preparation-manifest.json"
        write_json_new(
            manifest_path,
            {
                "schema_version": PREPARATION_SCHEMA,
                "status": "ready_for_cpu_source_and_single_gpu_window",
                "formal_split_root": str(formal_split.resolve()),
                "formal_split_manifest_sha256": sha_file(formal_split / "split-manifest.json", "formal split manifest"),
                "formal_policy_sha256": sha_file(formal_split / "policy.json", "formal policy"),
                "formal_calibration_cases_sha256": sha_file(formal_split / "calibration-cases.jsonl", "formal calibration rows"),
                "formal_holdout_cases_sha256": sha_file(formal_split / "holdout-cases.jsonl", "formal holdout rows"),
                "execution_view_root": str(execution_view.resolve()),
                "execution_view_split_manifest_sha256": sha_file(execution_view / "split-manifest.json", "execution-view split manifest"),
                "execution_view_calibration_cases_sha256": sha_file(execution_view / "calibration-cases.jsonl", "execution-view calibration rows"),
                "view_binding_sha256": sha_file(view_binding, "execution-view binding"),
                "selection_audit_sha256": sha_file(audit_path, "selection audit"),
                "source_cases": {
                    "calibration": {"path": str((source_cases_root / "calibration-cases.json").resolve()), "sha256": sha_file(source_cases_root / "calibration-cases.json", "calibration source cases")},
                    "formal_holdout": {"path": str((source_cases_root / "holdout-cases.json").resolve()), "sha256": sha_file(source_cases_root / "holdout-cases.json", "formal holdout source cases")},
                    "execution_holdout": {"path": str((source_cases_root / "holdout-execution-cases.json").resolve()), "sha256": sha_file(source_cases_root / "holdout-execution-cases.json", "execution holdout source cases")},
                },
                "source_model_vocab_size": vocab_size,
                "holdout_execution_contract": "formal holdout is eligible for exactly one target capture after the calibration freeze receipt; execution view cannot derive bounds",
            },
        )
        top_level_sums(output)
        return verify(output)
    except Exception:
        # Preserve an incomplete directory for forensic inspection rather than
        # silently deleting a new fixture selection.  A new output root must be
        # chosen for a subsequent attempt.
        raise


def verify_standard_sums(root: Path) -> None:
    expected = {"calibration-cases.jsonl", "holdout-cases.jsonl", "policy.json", "split-manifest.json"}
    sums = (root / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    actual: dict[str, str] = {}
    for line in sums:
        digest, sep, name = line.partition("  ")
        require(sep == "  " and name not in actual and len(digest) == 64, "execution-view SHA256SUMS differs")
        actual[name] = digest
    require(set(actual) == expected, "execution-view SHA256SUMS member set differs")
    for name, digest in actual.items():
        require(sha_file(root / name, f"execution-view {name}") == digest, f"execution-view checksum differs: {name}")


def verify(output: Path) -> dict[str, Any]:
    output = output.absolute()
    require(output.is_dir() and not output.is_symlink(), f"preparation root is invalid: {output}")
    manifest = load_json(output / "preparation-manifest.json", "preparation manifest")
    require(isinstance(manifest, dict) and manifest.get("schema_version") == PREPARATION_SCHEMA, "preparation manifest schema differs")
    formal_split = output / "formal-split"
    try:
        split_result = SPLIT_VALIDATOR.validate(formal_split)
    except Exception as error:
        raise PreparationError(f"formal split verification failed: {error}") from error
    audit = load_json(output / "selection-audit.json", "selection audit")
    require(isinstance(audit, dict) and audit.get("schema_version") == AUDIT_SCHEMA and audit.get("status") == "valid_disjoint_selection", "selection audit schema/status differs")
    intersections = audit.get("intersections")
    require(isinstance(intersections, dict) and all(value == [] for value in intersections.values()), "selection audit records a prohibited overlap")
    phase = audit.get("phase1_to_6_diagnostic")
    require(isinstance(phase, dict) and len(phase.get("distinct_context_token_ids_sha256", [])) == 3, "three Phase 1--6 contexts are not bound")
    formal_holdout = read_jsonl(formal_split / "holdout-cases.jsonl", "formal holdout rows")
    view = output / "holdout-execution-view"
    verify_standard_sums(view)
    view_manifest = load_json(view / "split-manifest.json", "execution-view manifest")
    binding = load_json(output / "view-binding.json", "execution-view binding")
    require(isinstance(binding, dict) and binding.get("schema_version") == EXECUTION_VIEW_SCHEMA and binding.get("status") == "ready_for_one_holdout_capture", "execution-view binding schema/status differs")
    require(view_manifest.get("execution_view_only") is True and view_manifest.get("calibration_bounds_derivation_forbidden") is True, "execution-view safety flags differ")
    view_calibration = read_jsonl(view / "calibration-cases.jsonl", "execution-view calibration rows")
    expected = {row["case_id"]: row for row in formal_holdout}
    actual = {row["case_id"]: row for row in view_calibration}
    require(set(expected) == set(actual) and len(expected) == 24, "execution view is not the formal holdout")
    fields = ("case_sha256", "fixture_sha256", "prompt_token_ids_sha256", "context_token_ids_sha256", "prompt_tokens", "context_tokens", "baseline_mode", "prefill_requested_m", "resolved_m", "step", "row_count")
    for case_id in expected:
        require(all(expected[case_id][field] == actual[case_id][field] for field in fields), f"execution-view mapping differs: {case_id}")
    for key, source_case in manifest.get("source_cases", {}).items():
        require(isinstance(source_case, dict), f"source case binding differs: {key}")
        path = Path(source_case.get("path", ""))
        require(path.is_file() and sha_file(path, f"source cases {key}") == source_case.get("sha256"), f"source case SHA differs: {key}")
        payload = load_json(path, f"source cases {key}")
        require(isinstance(payload, dict) and payload.get("schema_version") == SOURCE_CASES_SCHEMA and len(payload.get("cases", [])) == 24, f"source case schema/count differs: {key}")
    top_sums = output / "SHA256SUMS"
    require(top_sums.is_file() and not top_sums.is_symlink(), "top-level SHA256SUMS is unavailable")
    for line in top_sums.read_text(encoding="utf-8").splitlines():
        digest, sep, name = line.partition("  ")
        require(sep == "  " and len(digest) == 64 and name, "top-level SHA256SUMS line differs")
        require(sha_file(output / name, f"top-level {name}") == digest, f"top-level checksum differs: {name}")
    return {
        "schema_version": PREPARATION_SCHEMA,
        "status": "valid",
        "output": str(output),
        "formal_calibration": split_result["calibration"],
        "formal_holdout": split_result["holdout"],
        "formal_split_manifest_sha256": split_result["split_manifest_sha256"],
        "policy_sha256": split_result["policy_sha256"],
        "execution_view_case_count": len(view_calibration),
        "phase1_to_6_distinct_contexts": len(phase["distinct_context_token_ids_sha256"]),
        "overlap_check": "passed",
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify", action="store_true", help="verify an existing preparation root without writing")
    parser.add_argument("--source-model-dir", type=Path, help="BF16 Qwen3.5 source root; only config.json is read")
    parser.add_argument("--vocab-size", type=int, help="test-only alternative to --source-model-dir")
    parser.add_argument("--old-split-root", type=Path, default=DEFAULT_OLD_SPLIT)
    parser.add_argument("--old-no-go", type=Path, default=DEFAULT_OLD_NO_GO)
    parser.add_argument("--phase-cases", type=Path, default=DEFAULT_PHASE_CASES)
    parser.add_argument("--phase-contexts", type=Path, default=DEFAULT_PHASE_CONTEXTS)
    parser.add_argument("--input-control", type=Path, action="append", default=list(DEFAULT_INPUT_CONTROLS))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        result = verify(args.output) if args.verify else prepare(args)
    except (PreparationError, OSError, ValueError, RuntimeError) as error:
        print(f"AQ4 Phase 7 fidelity preparation failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
