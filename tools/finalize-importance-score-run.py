#!/usr/bin/env python3
"""Record completed Qwen Phase-0--2 artifacts without changing their contract files.

The immutable registry, quantization-candidate manifest, and corpus manifest
remain byte-for-byte as hashed into the run ID.  This tool only updates the
mutable experiment manifest with observed CPU-pilot coverage and explicit HOLD
reasons for phases that were not authorized or not reproducible.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--registry-commit", default="49fceeeb")
    parser.add_argument("--gguf-dump", type=Path, default=Path("/home/homelab1/hf_venv/bin/gguf-dump"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.run_root.expanduser().resolve()
    gguf_dump = args.gguf_dump.expanduser().resolve()
    tools_dir = Path(__file__).resolve().parent
    paths = {
        "experiment": root / "experiment-manifest.json",
        "registry": root / "score-method-registry.json",
        "candidate_manifest": root / "quantization-candidate-manifest.json",
        "corpus_manifest": root / "corpus-manifest.json",
        "label_audit": root / "ud-label-audit-summary.json",
        "combined_stats": root / "activation-stats" / "combined" / "metadata.json",
        "c0": root / "c0-sampler.jsonl",
        "metrics": root / "metrics-by-model.json",
        "bootstrap": root / "bootstrap-samples.parquet",
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise SystemExit(f"cannot finalize; missing artifacts: {', '.join(missing)}")

    manifest = read_json(paths["experiment"])
    label_audit = read_json(paths["label_audit"])
    combined_stats = read_json(paths["combined_stats"])
    metrics = read_json(paths["metrics"])
    c0_rows = [json.loads(line) for line in paths["c0"].read_text(encoding="utf-8").splitlines() if line.strip()]
    c0_ok = sum(row.get("status") == "ok" for row in c0_rows)
    if c0_ok != int(label_audit["eligible_core_count"]):
        raise SystemExit(
            "C0 coverage does not match eligible labels: "
            f"{c0_ok} != {label_audit['eligible_core_count']}"
        )

    manifest["phase_status"] = {
        "phase_0": (
            f"completed: C0--C7/admission registry committed as {args.registry_commit}; "
            "AQ4/AQ5 candidate manifest remains explicitly incomplete and excludes gain/allocation."
        ),
        "phase_1": (
            "completed for Qwen only: all UD tensors were audited; same-cohort static Q4_K_M was "
            "not found locally, no download was attempted, and paired analysis is HOLD."
        ),
        "phase_2": (
            "completed as an explicitly provisional CPU-only pilot: C0 sampled evaluator plus C2/C3 "
            "cover every eligible tensor; the formal 256k mixed-domain D_stats contract is not satisfied."
        ),
        "phase_3_plus": (
            "not started: C1 requires D_block block covariance; C4 requires block-output perturbations; "
            "C5/C6 are backward/direct-KL high-cost work and were not run on CPU."
        ),
    }
    manifest["model"]["architecture"] = label_audit["architecture"]
    manifest["model"]["paired_static_q4_k_m"] = label_audit["paired_static_q4_k_m"]
    manifest["model"]["ud_gguf"]["gguf_dump"] = {
        "path": str(gguf_dump),
        "sha256": sha256_file(gguf_dump) if gguf_dump.is_file() else "unknown",
        "invocation": "gguf-dump --json <model> (read-only)",
        "version": "unknown: this local gguf-dump exposes no --version flag; binary SHA-256 is recorded instead",
    }
    manifest["measurement"] = {
        "execution": "CPU-only offline; CUDA_VISIBLE_DEVICES/HIP_VISIBLE_DEVICES/ROCR_VISIBLE_DEVICES were blank for collection and scoring.",
        "execution_code": {
            "collector_sha256": sha256_file(tools_dir / "collect-activation-stats.py"),
            "sampler_sha256": sha256_file(tools_dir / "run-aq-tensor-sample.py"),
            "summarizer_sha256": sha256_file(tools_dir / "summarize-importance-score-screen.py"),
            "merger_sha256": sha256_file(tools_dir / "merge-activation-stats.py"),
            "note": (
                "These per-tool source hashes identify the provisional CPU observations. "
                "The candidate manifest remains the pre-screen incomplete format contract and is not a final-storage Q_b revision."
            ),
        },
        "D_stats_cpu_pilot": {
            "samples_seen": int(combined_stats["samples_seen"]),
            "valid_tokens_seen": int(combined_stats["tokens_seen"]),
            "shard_count": int(combined_stats["shard_count"]),
            "module_count": int(combined_stats["module_count"]),
            "statistics_dtype": combined_stats["stat_dtype"],
            "output_sha256": combined_stats["output_sha256"],
            "scope": "32 repository-local smoke prompts, eight per shard; provisional only, not the formal 256k mixed-domain corpus.",
        },
        "C0": {
            "eligible_tensor_count": c0_ok,
            "status": "200/200 sampled evaluator rows succeeded",
            "sampling": "65,536 elements/tensor, seed 0, FP32 evaluator, weighted relative MSE from merged D_stats",
            "admission_use": "provisional sensitivity only; candidate storage contract is incomplete",
        },
        "C2_C3": {
            "eligible_tensor_count": int(metrics["n_eligible"]),
            "C2": "activation moment statistics over the CPU pilot",
            "C3": "activation statistics plus deterministic no-replacement 65,536-weight samples per tensor",
        },
        "ordinal_bootstrap": {
            "status": metrics["bootstrap"]["status"],
            "replicates": 10_000,
            "admission_use": "Qwen-only ordinal exploratory uncertainty; not a two-model admission decision",
        },
    }
    manifest["admission_status"] = (
        "HOLD: Qwen same-cohort Q4_K_M is absent, Gemma is intentionally unrun, formal corpus and "
        "serializable format contracts are incomplete; AUC/Precision@K/KL gates were therefore not computed."
    )
    manifest["format_status"] = (
        "AQ4 output is a sampled FP32-evaluator sensitivity screen only. AQ4 and AQ5 are excluded from "
        "gain/allocation because storage rounding, serialized bytes, and other contract fields remain incomplete."
    )
    manifest["artifact_sha256"] = {
        name: sha256_file(path)
        for name, path in paths.items()
        if name != "experiment"
    }
    paths["experiment"].write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"run_id": manifest["run_id"], "phase_status": manifest["phase_status"], "admission_status": manifest["admission_status"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
