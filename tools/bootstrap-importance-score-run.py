#!/usr/bin/env python3
"""Create the frozen Phase-0 artifact skeleton for one importance-score run.

This tool is intentionally metadata-only.  It does not load a model, start a
service, access a GPU, or download anything.  The AQ4/AQ5 offline format
contract is concrete; AQ6 remains deliberately unregistered.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any


UNKNOWN = "unknown"


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return UNKNOWN


def file_record(path: Path, *, required: bool = True) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.is_file(),
        "sha256": sha256_file(path) if path.is_file() else UNKNOWN,
        "required": required,
    }


def candidate_manifest(revision: str) -> dict[str, Any]:
    tools_dir = Path(__file__).resolve().parent
    implementation = {
        "workspace_git_head": revision,
        "sampler_sha256": sha256_file(tools_dir / "run-aq-tensor-sample.py"),
        "family_exporter_sha256": sha256_file(tools_dir / "export-aq-family-codebooks.py"),
        "scale_helper_sha256": sha256_file(tools_dir / "aq_scale_formats.py"),
        "scope": "offline CPU fake-quantization and codebook export only; no runtime format support implied",
    }
    common = {
        "group_size": 16,
        "group_scale_encoding": (
            "one uint8 index per group into the unsigned-positive finite E4M3-like table returned by "
            "tools/aq_scale_formats.py::scale_values('e4m3')"
        ),
        "tensor_scale_encoding": "one little-endian IEEE-754 bfloat16 scalar per tensor",
        "codebook_storage_dtype": (
            "little-endian IEEE-754 bfloat16; one shared codebook per model x canonical_family"
        ),
        "scale/codebook objective": (
            "D_stats activation-second-moment-weighted squared reconstruction error; family codebook uses "
            "8 Lloyd iterations after fixed quantile initialization; tensor scale uses the fit-group median; "
            "each group scale searches nearest table index +/-4 and minimizes the same weighted objective"
        ),
        "family taxonomy": (
            "tools/run-aq-tensor-sample.py::family_for_tensor canonical text-matrix taxonomy v1; "
            "codebook scope is exactly model_id x canonical_family; no cross-model sharing"
        ),
        "fit/eval split": (
            "per_family_sample; seed=0; for each tensor/group_size, SHA-256-keyed affine permutation of "
            "contiguous groups; first min(4096,floor(N_groups/2)) groups fit codebook/tensor scale and next "
            "min(4096,N_groups-fit_count) disjoint groups evaluate; no replacement; AQ4/AQ5 share indices"
        ),
        "C0 loss estimator": (
            "evaluation uses the raw (not tensor-normalized) D_stats E[x_j^2] moments; sampled numerator "
            "and reference energy are expanded by usable_tensor_elements/evaluation_sample_elements for "
            "A_t and gain, while their ratio is L_t; the common deterministic sample is used for AQ4/AQ5"
        ),
        "seed": 0,
        "iterations": 8,
        "rounding mode": (
            "IEEE roundTiesToEven on bfloat16 storage casts; codebook nearest ties choose the lowest index; "
            "scale-table midpoint ties choose the higher index; scale-search objective ties retain the "
            "first (lowest-offset) candidate"
        ),
        "implementation revision": implementation,
        "contract_status": "complete for offline CPU measurement",
        "eligible_for_gain_allocation": True,
        "provisional_sensitivity_screen_only": False,
    }
    aq4 = {
        **common,
        "candidate_id": "aq4_e4m3_g16_ts_flloyd16",
        "role": "b_0",
        "index_bits": 4,
        "codebook_entries": 16,
        "index_packing": "two indices per byte, earlier element in low nibble",
        "serialized byte formula": (
            "tensor payload(n)=ceil(4*n/8)+ceil(n/16)+2 bytes; family overhead=16*2=32 bytes; "
            "model payload=sum_t tensor_payload(n_t)+32*num_families; excludes container metadata/alignment"
        ),
    }
    aq5 = {
        **common,
        "candidate_id": "aq5_e4m3_g16_ts_flloyd32",
        "role": "B_high",
        "index_bits": 5,
        "codebook_entries": 32,
        "index_packing": "contiguous LSB-first 5-bit bitstream; no per-group padding",
        "serialized byte formula": (
            "tensor payload(n)=ceil(5*n/8)+ceil(n/16)+2 bytes; family overhead=32*2=64 bytes; "
            "model payload=sum_t tensor_payload(n_t)+64*num_families; excludes container metadata/alignment"
        ),
        "definition_delta_from_aq4": (
            "only codebook index width (4->5 bits), entry count (16->32), and resulting packing/byte count"
        ),
        "historical_seed_evidence": {
            "source": "journal/2026/07/02/aq5-ud-q5-replacement.md",
            "note": "The historical FP16-rounded result is seed evidence only; this frozen contract uses the same BF16 storage semantics as b_0 and must be remeasured.",
        },
    }
    return {
        "schema_version": "quantization-candidate-manifest-v0.2",
        "status": "complete offline CPU contract; AQ4 is b_0 and AQ5 is the sole B_high candidate",
        "source_plan": "docs/plans/importance-score-algorithm-selection-plan-v0.1.md",
        "generated_at_utc": utc_now(),
        "candidates": [aq4, aq5],
        "not_registered": [
            {
                "candidate_id": "AQ6",
                "reason": "Undefined: exact ID, 64-entry fitting, storage semantics, and implementation revision are absent."
            },
            {
                "candidate_id": "SQ8_0",
                "reason": "High-quality diagnostic anchor, not a codebook-index mixed-precision candidate."
            }
        ]
    }


def corpus_manifest(corpus_file: Path) -> dict[str, Any]:
    source = file_record(corpus_file)
    raw_source = {
        "source_id": "qwen35-aq-smoke-prompts-v0.1",
        "kind": "repository_local_text_lines",
        **source,
        "selection": "All nonempty lines, deterministic source order, then four contiguous shards for the CPU pilot.",
        "coverage": "technical/general, code prompts, Japanese; not a formal mixed-domain corpus with sufficient chat/reasoning/math coverage."
    }
    splits = {
        "D_stats": {
            "purpose": "activation, covariance, and range statistics",
            "suggested_token_count_per_model": 256000,
            "raw_example_sources": [raw_source],
            "hash_method": "SHA-256 of canonical UTF-8 LF source bytes and SHA-256 of the explicit selection manifest",
            "token_count_status": "pending model-tokenizer measurement",
            "status": "provisional CPU pilot only; does not satisfy the frozen mixed-domain 256k-token target"
        },
        "D_block": {
            "purpose": "C4 block-output perturbation",
            "suggested_token_count_per_model": 16000,
            "raw_example_sources": "D_stats examples preselected by name hash into four fixed shards",
            "hash_method": "SHA-256 of selected raw-example identifiers and canonical source bytes",
            "token_count_status": "not materialized",
            "status": "pending"
        },
        "D_fisher": {
            "purpose": "C5 gradient/Fisher candidates",
            "suggested_token_count_per_model": 16000,
            "raw_example_sources": "independent raw-example selection from the frozen corpus",
            "hash_method": "SHA-256 of selected raw-example identifiers and canonical source bytes",
            "token_count_status": "not materialized",
            "status": "pending"
        },
        "D_KL": {
            "purpose": "C6 exact single-tensor KL",
            "suggested_token_count_per_model": 8000,
            "raw_example_sources": "unopened holdout selection after score formula and KL-core stratification are frozen",
            "hash_method": "SHA-256 of selected raw-example identifiers and canonical source bytes",
            "token_count_status": "not materialized",
            "status": "pending"
        },
        "D_final": {
            "purpose": "final mixed-artifact KL/PPL/fidelity only",
            "suggested_token_count_per_model": 32000,
            "raw_example_sources": "unopened final holdout selection",
            "hash_method": "SHA-256 of selected raw-example identifiers and canonical source bytes",
            "token_count_status": "not materialized",
            "status": "pending"
        }
    }
    return {
        "schema_version": "importance-score-corpus-manifest-v0.1",
        "status": "schema frozen; only a small local CPU pilot source is selected",
        "source_plan": "docs/plans/importance-score-algorithm-selection-plan-v0.1.md",
        "generated_at_utc": utc_now(),
        "shared_raw_example_rule": "Use the same raw example set across models, then apply each model's tokenizer and official chat template. Do not assume identical token indices across tokenizers.",
        "tokenization_contract": {
            "tokenizer_revision": UNKNOWN,
            "chat_template": "official local chat_template.jinja when a chat source is selected",
            "truncation": "record sequence length and per-length token counts",
            "padding": "record valid-token mask and exclude padding from all reductions"
        },
        "splits": splits,
        "known_local_future_sources_not_selected": [
            "/home/homelab1/datapool/dataset/fineweb/data/ (general English web; 2.9 TB)",
            "/home/homelab1/datapool/dataset/JParaCrawl/data/ (Japanese/English parallel corpus; 581 MB)",
            "/home/homelab1/datapool/ai_models/safetensors/Hy-MT2-30B-A3B/train/data/example_data.jsonl (100-message Chinese/English chat fixture)"
        ],
        "human_decision_needed": "Freeze a sufficiently sized chat/code/Japanese/multilingual/reasoning/math/general mix and its record-level split policy before treating any score run as formal D_stats."
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--gguf-path", type=Path, required=True)
    corpus_group = parser.add_mutually_exclusive_group(required=True)
    corpus_group.add_argument("--corpus-file", type=Path)
    corpus_group.add_argument("--corpus-manifest", type=Path)
    parser.add_argument("--static-gguf-path", type=Path)
    parser.add_argument("--source-repo", default="unknown")
    parser.add_argument("--source-revision", default="unknown")
    parser.add_argument("--ud-repo", default="unknown")
    parser.add_argument("--ud-revision", default="unknown")
    parser.add_argument(
        "--registry-path",
        type=Path,
        default=Path("docs/registries/importance-score-method-registry-v0.1.json"),
    )
    parser.add_argument("--model-id", default="qwen3.5-9b")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.results_root.expanduser().resolve()
    model_dir = args.model_dir.expanduser().resolve()
    gguf_path = args.gguf_path.expanduser().resolve()
    corpus_file = args.corpus_file.expanduser().resolve() if args.corpus_file else None
    formal_corpus_manifest = (
        args.corpus_manifest.expanduser().resolve() if args.corpus_manifest else None
    )
    static_gguf_path = args.static_gguf_path.expanduser().resolve() if args.static_gguf_path else None
    registry_path = args.registry_path.expanduser().resolve()
    required_paths = [(model_dir, "model directory"), (gguf_path, "GGUF"), (registry_path, "registry")]
    required_paths.append(
        (formal_corpus_manifest, "corpus manifest")
        if formal_corpus_manifest is not None
        else (corpus_file, "corpus file")
    )
    if static_gguf_path is not None:
        required_paths.append((static_gguf_path, "static GGUF"))
    for path, label in required_paths:
        assert path is not None
        if not path.exists():
            raise SystemExit(f"missing {label}: {path}")

    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    revision = git_revision()
    candidates = candidate_manifest(revision)
    corpus = (
        json.loads(formal_corpus_manifest.read_text(encoding="utf-8"))
        if formal_corpus_manifest is not None
        else corpus_manifest(corpus_file)
    )

    # Hash canonical manifests before assigning the run-id.  The date is the
    # result partition; content hashes make runs distinguishable within it.
    root.mkdir(parents=True, exist_ok=True)
    staging = root / ".importance-score-contract-staging"
    if staging.exists():
        raise SystemExit(f"staging directory already exists: {staging}")
    staging.mkdir()
    write_json(staging / "score-method-registry.json", registry)
    write_json(staging / "quantization-candidate-manifest.json", candidates)
    if formal_corpus_manifest is not None:
        shutil.copyfile(formal_corpus_manifest, staging / "corpus-manifest.json")
    else:
        write_json(staging / "corpus-manifest.json", corpus)
    registry_hash = sha256_file(staging / "score-method-registry.json")
    candidate_hash = sha256_file(staging / "quantization-candidate-manifest.json")
    corpus_hash = sha256_file(staging / "corpus-manifest.json")
    run_id = (
        f"{args.model_id}-source-rev-{args.source_revision[:12]}-corpus-{corpus_hash[:12]}-"
        f"registry-v0.1-{registry_hash[:12]}-qcm-{candidate_hash[:12]}"
    )
    run_dir = root / run_id
    if run_dir.exists():
        raise SystemExit(f"refusing to overwrite existing run directory: {run_dir}")

    experiment = {
        "schema_version": "importance-score-experiment-manifest-v0.1",
        "run_id": run_id,
        "created_at_utc": utc_now(),
        "phase_status": {
            "phase_0": "formal score registry, AQ4/AQ5 candidate contract, corpus, and provenance coordinates frozen",
            "phase_1": "pending UD extraction/audit",
            "phase_2_to_5": "pending CPU-only formal measurement; Gemma remains unopened until Qwen candidate freeze",
            "phase_3_plus": "not started"
        },
        "execution_constraints": {
            "gpu": "forbidden for this run",
            "service_or_systemd": "forbidden for this run",
            "downloads": "already acquired artifacts are read-only inputs; acquisition provenance is recorded separately",
            "quantizer_runtime_writer_kernel_changes": "out of scope"
        },
        "source_plan": "docs/plans/importance-score-algorithm-selection-plan-v0.1.md",
        "source_plan_sha256": sha256_file(Path("docs/plans/importance-score-algorithm-selection-plan-v0.1.md")),
        "git_revision_at_bootstrap": revision,
        "model": {
            "model_id": args.model_id,
            "architecture": "unknown until config audit",
            "bf16": {
                "directory": str(model_dir),
                "config": file_record(model_dir / "config.json"),
                "weight_index": file_record(model_dir / "model.safetensors.index.json"),
                "source_repo": args.source_repo,
                "source_revision": args.source_revision
            },
            "ud_gguf": {
                "path": str(gguf_path),
                "sha256": sha256_file(gguf_path),
                "source_repo": args.ud_repo,
                "source_revision": args.ud_revision,
                "gguf_dump_version": UNKNOWN
            },
            "paired_static_q4_k_m": {
                "status": "supplied_pending_exact_name_shape_audit" if static_gguf_path else "not supplied",
                "path": str(static_gguf_path) if static_gguf_path else None,
                "sha256": sha256_file(static_gguf_path) if static_gguf_path else None
            }
        },
        "contract_hashes": {
            "score_method_registry_sha256": registry_hash,
            "quantization_candidate_manifest_sha256": candidate_hash,
            "corpus_manifest_sha256": corpus_hash
        },
        "admission_status": "HOLD pending formal Qwen measurement, Qwen candidate freeze, and one-shot Gemma lockbox evaluation.",
        "format_status": "AQ4 b_0 and AQ5 B_high are complete offline CPU contracts; AQ6 is not registered."
    }
    write_json(staging / "experiment-manifest.json", experiment)
    staging.rename(run_dir)
    print(run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
