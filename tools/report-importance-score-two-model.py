#!/usr/bin/env python3
"""Apply the frozen two-model admission and worst-model winner rules."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq


METRIC_ORDER = ("rho", "tau_b", "AUC_within", "Precision_at_K", "KL_rho")
IMPLEMENTED_PAIRED_METRICS = ("rho", "tau_b")
EXPECTED_FORMAL_SETTINGS = {
    "weight_sample_size": 65536,
    "seed": 0,
    "bootstrap_replicates": 10000,
    "permutation_replicates": 10000,
}
EXPECTED_BOOTSTRAP_REPLICATES = 10000
OUTPUT_FILENAMES = (
    "two-model-decision.json",
    "two-model-decision.md",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def score_point(metrics: dict[str, Any], score: str) -> dict[str, Any]:
    item = metrics["scores"][score]
    primary = item["primary_within_family_macro"]
    binary = item["binary"]
    kl = metrics["KL_core"]["scores"][score]
    common = item["common_family_secondary"]["primary_within_family_macro"]
    return {
        "admission_pass": bool(item["admission_gate"]["pass"]),
        "rho": primary["rho"],
        "tau_b": primary["tau_b"],
        "AUC_within": binary["AUC_within"],
        "Precision_at_K": binary["Precision_at_K"],
        "KL_rho": kl["rho_score_vs_AQ4_KL"],
        "common_family_rho_secondary": common["rho"],
        "common_family_tau_b_secondary": common["tau_b"],
    }


def bootstrap_arrays(
    path: Path,
    expected_scores: list[str],
    expected_replicates: int = EXPECTED_BOOTSTRAP_REPLICATES,
) -> dict[str, dict[str, np.ndarray]]:
    rows = pq.read_table(path).to_pylist()
    expected_score_set = set(expected_scores)
    result: dict[str, dict[str, np.ndarray]] = {
        score: {
            "rho": np.full(expected_replicates, np.nan, dtype=np.float64),
            "tau_b": np.full(expected_replicates, np.nan, dtype=np.float64),
        }
        for score in expected_scores
    }
    seen: dict[str, set[int]] = {score: set() for score in expected_scores}
    for row in rows:
        score = str(row["score_id"])
        if score not in expected_score_set:
            raise ValueError(f"bootstrap contains an unfrozen score: {score}")
        replicate = int(row["replicate"])
        if not 0 <= replicate < expected_replicates:
            raise ValueError(f"bootstrap replicate is outside the frozen range: {replicate}")
        if replicate in seen[score]:
            raise ValueError(f"duplicate bootstrap replicate for {score}: {replicate}")
        seen[score].add(replicate)
        if row.get("primary_rho") is not None:
            result[score]["rho"][replicate] = float(row["primary_rho"])
        if row.get("primary_tau_b") is not None:
            result[score]["tau_b"][replicate] = float(row["primary_tau_b"])
    expected_ids = set(range(expected_replicates))
    incomplete = {
        score: sorted(expected_ids - replicates)
        for score, replicates in seen.items()
        if replicates != expected_ids
    }
    if incomplete:
        counts = {score: len(missing) for score, missing in incomplete.items()}
        raise ValueError(
            f"bootstrap does not contain exactly {expected_replicates:,} rows per score: {counts}"
        )
    if len(rows) != len(expected_scores) * expected_replicates:
        raise ValueError(
            f"bootstrap row count differs from score_count * {expected_replicates:,}"
        )
    return result


def validate_report_receipt(
    model_name: str,
    metrics_path: Path,
    bootstrap_path: Path,
    receipt_path: Path,
    metrics: dict[str, Any],
    freeze: dict[str, Any],
    frozen_scores: list[str],
    winner_eligible_scores: list[str] | None = None,
    secondary_scores: list[str] | None = None,
) -> dict[str, Any]:
    winner_eligible_scores = winner_eligible_scores or frozen_scores
    secondary_scores = secondary_scores or []
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("schema_version") != "importance-score-formal-report-receipt-v0.1":
        raise SystemExit(f"{model_name} formal report receipt has an unexpected schema")
    if receipt.get("status") != "sealed formal report outputs":
        raise SystemExit(f"{model_name} formal report receipt is not sealed")
    if receipt.get("model_id") != metrics.get("model_id"):
        raise SystemExit(f"{model_name} metrics and report receipt model IDs differ")
    outputs = receipt.get("output_hashes", {})
    if outputs.get("metrics-by-model.json") != sha256_file(metrics_path):
        raise SystemExit(f"{model_name} metrics differ from their report receipt")
    if outputs.get("bootstrap-samples.parquet") != sha256_file(bootstrap_path):
        raise SystemExit(f"{model_name} bootstrap samples differ from their report receipt")
    if set(receipt.get("candidate_score_columns", [])) != set(frozen_scores):
        raise SystemExit(f"{model_name} report receipt score set differs from the freeze")
    if set(receipt.get("winner_eligible_score_columns", winner_eligible_scores)) != set(
        winner_eligible_scores
    ):
        raise SystemExit(f"{model_name} winner-eligible score set differs from the freeze")
    if set(receipt.get("secondary_score_columns", secondary_scores)) != set(
        secondary_scores
    ):
        raise SystemExit(f"{model_name} secondary score set differs from the freeze")
    formal_settings = receipt.get("execution_settings", {}).get("formal_report")
    if formal_settings != EXPECTED_FORMAL_SETTINGS:
        raise SystemExit(f"{model_name} formal report settings differ from the frozen contract")
    if metrics.get("execution_settings", {}).get("formal_report") != formal_settings:
        raise SystemExit(f"{model_name} metrics and report receipt settings differ")
    frozen_implementations = freeze.get("implementation_hashes", {})
    implementations = receipt.get("implementation_hashes", {})
    for name in (
        "report-importance-score-formal.py",
        "summarize-importance-score-screen.py",
    ):
        if implementations.get(name) != frozen_implementations.get(name):
            raise SystemExit(f"{model_name} {name} differs from the candidate freeze")
        if metrics.get("implementation_hashes", {}).get(name) != implementations.get(name):
            raise SystemExit(f"{model_name} metrics and report receipt disagree on {name}")
    bootstrap_contract = receipt.get("bootstrap_contract", {})
    if (
        bootstrap_contract.get("replicates_per_score") != EXPECTED_BOOTSTRAP_REPLICATES
        or bootstrap_contract.get("score_count") != len(frozen_scores)
        or bootstrap_contract.get("row_count")
        != EXPECTED_BOOTSTRAP_REPLICATES * len(frozen_scores)
        or bootstrap_contract.get("expected_row_count")
        != EXPECTED_BOOTSTRAP_REPLICATES * len(frozen_scores)
    ):
        raise SystemExit(f"{model_name} bootstrap receipt differs from the frozen contract")
    return receipt


def paired_worst_model_difference(
    qwen: dict[str, dict[str, np.ndarray]],
    gemma: dict[str, dict[str, np.ndarray]],
    left: str,
    right: str,
    metric: str,
) -> dict[str, Any]:
    arrays = (qwen[left][metric], gemma[left][metric], qwen[right][metric], gemma[right][metric])
    lengths = {len(array) for array in arrays}
    if len(lengths) != 1:
        raise ValueError(f"bootstrap replicate count mismatch for {left}/{right}/{metric}")
    finite = np.logical_and.reduce([np.isfinite(array) for array in arrays])
    if not bool(finite.any()):
        raise ValueError(f"no jointly defined bootstrap replicates for {left}/{right}/{metric}")
    difference = np.minimum(arrays[0][finite], arrays[1][finite]) - np.minimum(
        arrays[2][finite], arrays[3][finite]
    )
    ci = [float(np.quantile(difference, 0.025)), float(np.quantile(difference, 0.975))]
    return {
        "left": left,
        "right": right,
        "metric": metric,
        "replicates": len(difference),
        "mean_difference": float(difference.mean()),
        "median_difference": float(np.median(difference)),
        "paired_ci95_percentile": ci,
        "left_strictly_better": ci[0] > 0,
        "statistical_tie": ci[0] <= 0 <= ci[1],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qwen-metrics", type=Path, required=True)
    parser.add_argument("--gemma-metrics", type=Path, required=True)
    parser.add_argument("--qwen-bootstrap", type=Path, required=True)
    parser.add_argument("--gemma-bootstrap", type=Path, required=True)
    parser.add_argument("--qwen-report-receipt", type=Path, required=True)
    parser.add_argument("--gemma-report-receipt", type=Path, required=True)
    parser.add_argument("--qwen-candidate-freeze", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = {
        name: value.expanduser().resolve()
        for name, value in {
            "qwen_metrics": args.qwen_metrics,
            "gemma_metrics": args.gemma_metrics,
            "qwen_bootstrap": args.qwen_bootstrap,
            "gemma_bootstrap": args.gemma_bootstrap,
            "qwen_report_receipt": args.qwen_report_receipt,
            "gemma_report_receipt": args.gemma_report_receipt,
            "qwen_candidate_freeze": args.qwen_candidate_freeze,
        }.items()
    }
    qwen = json.loads(paths["qwen_metrics"].read_text(encoding="utf-8"))
    gemma = json.loads(paths["gemma_metrics"].read_text(encoding="utf-8"))
    freeze = json.loads(paths["qwen_candidate_freeze"].read_text(encoding="utf-8"))
    current_two_model_hash = sha256_file(Path(__file__).resolve())
    if freeze.get("implementation_hashes", {}).get(
        "report-importance-score-two-model.py"
    ) != current_two_model_hash:
        raise SystemExit("current two-model report implementation differs from the candidate freeze")
    if freeze["input_hashes"]["qwen_metrics"] != sha256_file(paths["qwen_metrics"]):
        raise SystemExit("Qwen metrics differ from the sealed candidate freeze")
    if freeze["input_hashes"].get("qwen_report_receipt") != sha256_file(
        paths["qwen_report_receipt"]
    ):
        raise SystemExit("Qwen formal report receipt differs from the candidate freeze")
    order = gemma.get("paired_cohort_audit", {}).get("lockbox_order_audit") or {}
    if order.get("qwen_candidate_freeze_sha256") != sha256_file(
        paths["qwen_candidate_freeze"]
    ):
        raise SystemExit("Gemma label opening was not authorized by this Qwen candidate freeze")
    prejoin = gemma.get("prejoin_audit", {})
    if not prejoin.get("label_join_performed_after_receipt"):
        raise SystemExit("Gemma metrics do not prove a score-before-label join")
    if order.get("sealed_score_table_sha256") != prejoin.get("score_table_sha256"):
        raise SystemExit("Gemma label authorization and joined score table hashes differ")
    frozen_scores = list(
        freeze.get("reported_score_columns", freeze["candidate_scores_transferred_unchanged"])
    )
    winner_eligible_scores = list(
        freeze.get("winner_eligible_score_columns", frozen_scores)
    )
    secondary_scores = list(freeze.get("secondary_score_columns", []))
    if set(winner_eligible_scores).intersection(secondary_scores):
        raise SystemExit("freeze overlaps winner-eligible and secondary score roles")
    if set(winner_eligible_scores).union(secondary_scores) != set(frozen_scores):
        raise SystemExit("freeze score roles do not cover the complete reported score table")
    if set(frozen_scores) != set(qwen["scores"]) or set(frozen_scores) != set(gemma["scores"]):
        raise SystemExit("Qwen/Gemma score sets differ from the sealed candidate table")
    qwen_receipt = validate_report_receipt(
        "Qwen",
        paths["qwen_metrics"],
        paths["qwen_bootstrap"],
        paths["qwen_report_receipt"],
        qwen,
        freeze,
        frozen_scores,
        winner_eligible_scores,
        secondary_scores,
    )
    gemma_receipt = validate_report_receipt(
        "Gemma",
        paths["gemma_metrics"],
        paths["gemma_bootstrap"],
        paths["gemma_report_receipt"],
        gemma,
        freeze,
        frozen_scores,
        winner_eligible_scores,
        secondary_scores,
    )
    q_bootstrap = bootstrap_arrays(paths["qwen_bootstrap"], frozen_scores)
    g_bootstrap = bootstrap_arrays(paths["gemma_bootstrap"], frozen_scores)
    rows = []
    finalists = []
    for score in frozen_scores:
        q_point = score_point(qwen, score)
        g_point = score_point(gemma, score)
        winner_eligible = score in winner_eligible_scores
        two_model_pass = (
            winner_eligible
            and q_point["admission_pass"]
            and g_point["admission_pass"]
        )
        if two_model_pass:
            finalists.append(score)
        q_rho = q_point["rho"]
        g_rho = g_point["rho"]
        rows.append(
            {
                "score_id": score,
                "winner_eligible": winner_eligible,
                "secondary_only": score in secondary_scores,
                "qwen": q_point,
                "gemma": g_point,
                "two_model_admission_pass": two_model_pass,
                "worst_model_rho": (
                    min(float(q_rho), float(g_rho))
                    if q_rho is not None and g_rho is not None
                    else None
                ),
            }
        )

    comparisons = []
    winner = None
    if not finalists:
        decision = "NO-GO"
        reason = "No frozen candidate passed every admission component on both Qwen and Gemma."
    elif len(finalists) == 1:
        winner = finalists[0]
        decision = "WINNER"
        reason = "Exactly one candidate passed the complete two-model admission gate."
    else:
        by_id = {row["score_id"]: row for row in rows}
        ordered = sorted(finalists, key=lambda score: (-by_id[score]["worst_model_rho"], score))
        provisional = ordered[0]
        rho_comparisons = [
            paired_worst_model_difference(
                q_bootstrap, g_bootstrap, provisional, other, "rho"
            )
            for other in ordered[1:]
        ]
        comparisons.extend(rho_comparisons)
        if all(item["left_strictly_better"] for item in rho_comparisons):
            winner = provisional
            decision = "WINNER"
            reason = "Worst-model rho is strictly better in every paired 95% bootstrap comparison."
        else:
            tied = [
                item["right"]
                for item in rho_comparisons
                if not item["left_strictly_better"]
            ]
            tau_comparisons = [
                paired_worst_model_difference(
                    q_bootstrap, g_bootstrap, provisional, other, "tau_b"
                )
                for other in tied
            ]
            comparisons.extend(tau_comparisons)
            if tau_comparisons and all(
                item["left_strictly_better"] for item in tau_comparisons
            ):
                winner = provisional
                decision = "WINNER"
                reason = "Rho tied, then worst-model tau-b was strictly better in paired comparisons."
            else:
                decision = "HOLD: statistical tie"
                reason = (
                    "The pre-frozen paired worst-model rho/tau comparisons did not separate every "
                    "finalist. AUC/Precision@K/KL remain reported point estimates only; no unfrozen "
                    "post-lockbox resampling or winner selection is permitted."
                )

    output = {
        "schema_version": "importance-score-two-model-decision-v0.1",
        "lockbox_validity": {
            "status": "valid one-shot Gemma lockbox",
            "qwen_candidate_freeze_sha256": sha256_file(paths["qwen_candidate_freeze"]),
            "gemma_prejoin_score_sha256": prejoin["score_table_sha256"],
            "formula_or_threshold_change_after_gemma": False,
            "third_model_required": False,
            "validation_scope": (
                "This receipt proves the sealed Qwen freeze, authorized Gemma prejoin/label order, "
                "formal report hashes, and frozen implementations used by this decision."
            ),
        },
        "candidate_rows": rows,
        "reported_score_columns": frozen_scores,
        "winner_eligible_score_columns": winner_eligible_scores,
        "secondary_score_columns": secondary_scores,
        "two_model_finalists": finalists,
        "paired_bootstrap_comparisons": comparisons,
        "preregistered_metric_order": list(METRIC_ORDER),
        "implemented_paired_metric_order": list(IMPLEMENTED_PAIRED_METRICS),
        "unimplemented_metric_policy": (
            "AUC_within, Precision_at_K, and KL_rho are not resampled by v0.1. If rho/tau-b do "
            "not separate finalists, the only authorized decision is HOLD: statistical tie."
        ),
        "decision": decision,
        "winner_candidate": winner,
        "reason": reason,
        "phase_6_authorized": decision == "WINNER",
        "report_receipts": {
            "qwen": {
                "path": str(paths["qwen_report_receipt"]),
                "sha256": sha256_file(paths["qwen_report_receipt"]),
                "model_id": qwen_receipt["model_id"],
            },
            "gemma": {
                "path": str(paths["gemma_report_receipt"]),
                "sha256": sha256_file(paths["gemma_report_receipt"]),
                "model_id": gemma_receipt["model_id"],
            },
        },
        "input_hashes": {name: sha256_file(path) for name, path in paths.items()},
    }
    output_dir = args.output_dir.expanduser().resolve()
    existing_outputs = [
        output_dir / name for name in OUTPUT_FILENAMES if (output_dir / name).exists()
    ]
    if existing_outputs:
        raise SystemExit(f"refusing to overwrite sealed two-model outputs: {existing_outputs}")
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "two-model-decision.json"
    json_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Two-model importance-score decision",
        "",
        f"- Decision: `{decision}`",
        f"- Winner: `{winner}`",
        f"- Gemma lockbox: valid one-shot evaluation",
        f"- Phase 6 authorized: `{output['phase_6_authorized']}`",
        "",
        "| Score | role | Qwen rho | Gemma rho | worst rho | Qwen gate | Gemma gate |",
        "|---|---|---:|---:|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['score_id']} | "
            f"{'winner-eligible' if row['winner_eligible'] else 'secondary'} | "
            f"{row['qwen']['rho']} | {row['gemma']['rho']} | "
            f"{row['worst_model_rho']} | {row['qwen']['admission_pass']} | "
            f"{row['gemma']['admission_pass']} |"
        )
    (output_dir / "two-model-decision.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"decision": decision, "winner": winner}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
