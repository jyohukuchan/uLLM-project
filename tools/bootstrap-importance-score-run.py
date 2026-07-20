#!/usr/bin/env python3
"""Create the frozen Phase-0 artifact skeleton for one importance-score run.

This tool is intentionally metadata-only.  It does not load a model, start a
service, access a GPU, or download anything.  Unknown format-contract fields
remain the literal string ``unknown`` so a later gain/allocation stage cannot
mistake a historical sampler result for a serializable candidate.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
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
    aq4 = {
        "candidate_id": "aq4_e4m3_g16_ts_flloyd16",
        "index_bits": 4,
        "codebook_entries": 16,
        "group_size": 16,
        "group_scale_encoding": "e4m3_unsigned_positive_table; serialized_mapping=unknown",
        "tensor_scale_encoding": UNKNOWN,
        "codebook_storage_dtype": UNKNOWN,
        "scale/codebook objective": UNKNOWN,
        "family taxonomy": UNKNOWN,
        "fit/eval split": UNKNOWN,
        "seed": UNKNOWN,
        "iterations": 8,
        "rounding mode": UNKNOWN,
        "serialized byte formula": UNKNOWN,
        "implementation revision": {
            "workspace_git_head": revision,
            "sampler_sha256": "8c5d395d853cecd7f9362ee27305c040f1ac9fc35aa87ba57be96f5b649f5b2c",
            "family_exporter_sha256": "2a71abdc459c309dcda82f1f4ad0ee3ec993d3d4fc60d84459980c23be11f91b",
            "scale_helper_sha256": "8a2bbe7084afbd3fdcb43d9682b1eeac41b0168c084bf5ca40937fe093a7541a",
            "meaning": "FP32 sampled evaluator revision, not a final-storage Q_b revision"
        },
        "contract_status": "incomplete",
        "eligible_for_gain_allocation": False,
        "provisional_sensitivity_screen_only": True,
        "exclusion_reason": "Final BF16 codebook/tensor-scale rounding, serialized byte formula, family/codebook scope, objective, and fit/eval split are not reproducibly frozen.",
        "implementation_evidence": {
            "declared_sampler_fields": {
                "scale_format": "e4m3",
                "tensor_scale": "bf16",
                "codebook_storage_dtype": "bf16",
                "nominal_effective_bpp": 4.5,
                "tool_default_seed": 0,
                "lloyd_iterations": 8,
                "default_scale_window": 4
            },
            "non_contractual_behavior": [
                "The evaluator keeps codebook and tensor scale as FP32 values.",
                "Its sample is replacement sampling and uses the same samples for fitting and evaluation.",
                "The nominal 4 + 8/group_size bpp omits serialized codebook, tensor-scale, tail padding, and alignment bytes."
            ]
        }
    }
    aq5 = {
        "candidate_id": "aq5_e4m3_g16_ts_flloyd32",
        "index_bits": UNKNOWN,
        "codebook_entries": UNKNOWN,
        "group_size": UNKNOWN,
        "group_scale_encoding": UNKNOWN,
        "tensor_scale_encoding": UNKNOWN,
        "codebook_storage_dtype": UNKNOWN,
        "scale/codebook objective": UNKNOWN,
        "family taxonomy": UNKNOWN,
        "fit/eval split": UNKNOWN,
        "seed": UNKNOWN,
        "iterations": UNKNOWN,
        "rounding mode": UNKNOWN,
        "serialized byte formula": UNKNOWN,
        "implementation revision": UNKNOWN,
        "contract_status": "incomplete",
        "eligible_for_gain_allocation": False,
        "provisional_sensitivity_screen_only": False,
        "exclusion_reason": "The checked-out sampler and family exporter cannot resolve aq5_e4m3_g16_ts_flloyd32; no current implementation revision recreates Q_b.",
        "historical_evidence_not_promoted_to_contract": {
            "source": "journal/2026/07/02/aq5-ud-q5-replacement.md",
            "claims": "5-bit, 32-entry, g16, E4M3, FP16-rounded codebook/global scale, 8 iterations",
            "reason_not_frozen": "Historical report lacks an available reproducer, fit/eval split, seed, serialized byte formula, and implementation revision."
        }
    }
    return {
        "schema_version": "quantization-candidate-manifest-v0.1",
        "status": "incomplete; sensitivity-only screen permitted for the AQ4 sampled evaluator, gain/allocation prohibited",
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
    parser.add_argument("--corpus-file", type=Path, required=True)
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
    corpus_file = args.corpus_file.expanduser().resolve()
    registry_path = args.registry_path.expanduser().resolve()
    for path, label in ((model_dir, "model directory"), (gguf_path, "GGUF"), (corpus_file, "corpus file"), (registry_path, "registry")):
        if not path.exists():
            raise SystemExit(f"missing {label}: {path}")

    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    revision = git_revision()
    candidates = candidate_manifest(revision)
    corpus = corpus_manifest(corpus_file)

    # Hash canonical manifests before assigning the run-id.  The date is the
    # result partition; content hashes make runs distinguishable within it.
    root.mkdir(parents=True, exist_ok=True)
    staging = root / ".importance-score-contract-staging"
    if staging.exists():
        raise SystemExit(f"staging directory already exists: {staging}")
    staging.mkdir()
    write_json(staging / "score-method-registry.json", registry)
    write_json(staging / "quantization-candidate-manifest.json", candidates)
    write_json(staging / "corpus-manifest.json", corpus)
    registry_hash = sha256_file(staging / "score-method-registry.json")
    candidate_hash = sha256_file(staging / "quantization-candidate-manifest.json")
    corpus_hash = sha256_file(staging / "corpus-manifest.json")
    run_id = (
        f"{args.model_id}-source-rev-unknown-corpus-{corpus_hash[:12]}-"
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
            "phase_0": "contract frozen with incomplete quantization candidates",
            "phase_1": "pending UD extraction/audit",
            "phase_2": "pending CPU-only provisional screen",
            "phase_3_plus": "not started"
        },
        "execution_constraints": {
            "gpu": "forbidden for this run",
            "service_or_systemd": "forbidden for this run",
            "new_model_or_file_download": "forbidden for this run",
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
                "source_revision": UNKNOWN
            },
            "ud_gguf": {
                "path": str(gguf_path),
                "sha256": sha256_file(gguf_path),
                "source_repo": "unknown from local artifact alone",
                "source_revision": UNKNOWN,
                "gguf_dump_version": UNKNOWN
            },
            "paired_static_q4_k_m": {
                "status": "not yet audited",
                "download_permitted": False
            }
        },
        "contract_hashes": {
            "score_method_registry_sha256": registry_hash,
            "quantization_candidate_manifest_sha256": candidate_hash,
            "corpus_manifest_sha256": corpus_hash
        },
        "admission_status": "HOLD pending paired Q4_K_M on Qwen and Gemma; Gemma is explicitly out of scope for this run.",
        "format_status": "No candidate is eligible for gain/allocation; AQ4 sampled sensitivity output is provisional only."
    }
    write_json(staging / "experiment-manifest.json", experiment)
    staging.rename(run_dir)
    print(run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
