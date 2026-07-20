#!/usr/bin/env python3
"""Freeze Qwen score candidates and gates before the Gemma label join."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any


FORMULAS = {
    "C0_I": {
        "definition": "L_C0(t, aq4_e4m3_g16_ts_flloyd16)",
        "binary_ranking": "C0_G=max(0,A_C0(aq4)-A_C0(aq5))",
        "format_dependent": True,
    },
    "C1_I": {
        "definition": "L_C1-block128(t, aq4_e4m3_g16_ts_flloyd16)",
        "binary_ranking": "C1_G=max(0,A_C1(aq4)-A_C1(aq5))",
        "format_dependent": True,
    },
    "S_AWQ_level": {
        "definition": "log(mean_j(E|x_j|)+1e-30)",
        "binary_ranking": "S_AWQ_level",
        "format_dependent": False,
    },
    "S_AWQ_tail": {
        "definition": "sum(top ceil(1%*d) E|x_j|)/sum_j(E|x_j|)",
        "binary_ranking": "S_AWQ_tail",
        "format_dependent": False,
    },
    "S_range": {
        "definition": "0.5*(log r_x + log r_w) with Q0.99 activation max and Q0.999 absolute weight",
        "binary_ranking": "S_range",
        "format_dependent": False,
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qwen-metrics", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--score-registry", type=Path, required=True)
    parser.add_argument("--corpus-manifest", type=Path, required=True)
    parser.add_argument("--subset-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = {
        "qwen_metrics": args.qwen_metrics.expanduser().resolve(),
        "candidate_manifest": args.candidate_manifest.expanduser().resolve(),
        "score_registry": args.score_registry.expanduser().resolve(),
        "corpus_manifest": args.corpus_manifest.expanduser().resolve(),
        "subset_manifest": args.subset_manifest.expanduser().resolve(),
    }
    metrics = json.loads(paths["qwen_metrics"].read_text(encoding="utf-8"))
    evaluated = [score for score in FORMULAS if score in metrics["scores"]]
    qwen_finalists = [
        score for score in evaluated if metrics["scores"][score]["admission_gate"]["pass"]
    ]
    freeze = {
        "schema_version": "importance-score-qwen-candidate-freeze-v0.1",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "sealed before any Gemma tensor-level score/label join",
        "development_model": "qwen3.5-9b",
        "lockbox_model": "gemma-4-E4B-it",
        "candidate_scores_transferred_unchanged": evaluated,
        "qwen_side_finalists": qwen_finalists,
        "no_qwen_finalist_rule": (
            "Gemma may still receive the complete frozen candidate table descriptively, but no candidate can "
            "become a two-model finalist if qwen_side_finalists is empty."
        ),
        "formulas": {score: FORMULAS[score] for score in evaluated},
        "thresholds": {
            "paired_coverage": 1.0,
            "nonconstant_repeated_families": 4,
            "mixed_positive_negative_families": 3,
            "within_family_macro_rho": 0.30,
            "within_family_macro_tau_b": 0.20,
            "cluster_bootstrap_rho_ci_lower_strictly_greater_than": 0.0,
            "positive_family_direction_fraction": 0.70,
            "major_family_min_tau_b": -0.20,
            "AUC_within": 0.65,
            "Precision_at_K": "p + 0.25*(1-p)",
            "KL_core_rho": 0.30,
            "KL_max_regression_from_C0": 0.05,
        },
        "aggregation": {
            "primary": "equal-weight macro mean of defined within-family layer Spearman/Kendall",
            "common_family_secondary": (
                "fixed attn_q/k/v/o and mlp_gate/up/down family set; reported for both models but "
                "never substituted for the architecture-specific admission primary"
            ),
            "bootstrap": "10,000 whole-layer cluster resamples; 95% percentile CI",
            "permutation": "10,000 common global-layer permutations restricted to each family; BH correction",
            "binary": "G_t for C0/C1; S_t itself for format-independent candidates",
            "KL": "score sensitivity versus same-tensor AQ4 C6 KL on score/label-independent KL-core",
        },
        "winner_rule": (
            "Only two-model admission passers; maximize min(Qwen rho,Gemma rho), then paired layer-cluster "
            "bootstrap comparisons in the frozen metric order; otherwise NO-GO."
        ),
        "forbidden_after_lockbox": [
            "formula changes",
            "threshold changes",
            "feature weights",
            "family exceptions",
            "candidate switching based on Gemma results",
        ],
        "input_hashes": {name: sha256_file(path) for name, path in paths.items()},
        "implementation_hashes": {
            name: sha256_file(Path(__file__).resolve().parent / name)
            for name in (
                "build-importance-score-prejoin.py",
                "report-importance-score-formal.py",
                "run-aq-tensor-sample.py",
                "score-block-covariance-c1.py",
                "run-importance-single-tensor-perturbation.py",
            )
        },
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, freeze)
    receipt = {
        "candidate_freeze_path": str(output),
        "candidate_freeze_sha256": sha256_file(output),
        "qwen_side_finalists": qwen_finalists,
        "candidate_scores_transferred_unchanged": evaluated,
    }
    write_json(output.with_suffix(".receipt.json"), receipt)
    print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
