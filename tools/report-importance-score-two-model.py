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


def bootstrap_arrays(path: Path) -> dict[str, dict[str, np.ndarray]]:
    rows = pq.read_table(path).to_pylist()
    result: dict[str, dict[str, dict[int, float]]] = {}
    for row in rows:
        score = str(row["score_id"])
        result.setdefault(score, {"rho": {}, "tau_b": {}})
        if row.get("primary_rho") is not None:
            result[score]["rho"][int(row["replicate"])] = float(row["primary_rho"])
        if row.get("primary_tau_b") is not None:
            result[score]["tau_b"][int(row["replicate"])] = float(row["primary_tau_b"])
    arrays = {}
    for score, metrics in result.items():
        arrays[score] = {}
        for metric, values in metrics.items():
            size = max(values, default=-1) + 1
            array = np.full(size, np.nan, dtype=np.float64)
            for replicate, value in values.items():
                array[replicate] = value
            arrays[score][metric] = array
    return arrays


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
            "qwen_candidate_freeze": args.qwen_candidate_freeze,
        }.items()
    }
    qwen = json.loads(paths["qwen_metrics"].read_text(encoding="utf-8"))
    gemma = json.loads(paths["gemma_metrics"].read_text(encoding="utf-8"))
    freeze = json.loads(paths["qwen_candidate_freeze"].read_text(encoding="utf-8"))
    if freeze["input_hashes"]["qwen_metrics"] != sha256_file(paths["qwen_metrics"]):
        raise SystemExit("Qwen metrics differ from the sealed candidate freeze")
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
    frozen_implementations = freeze.get("implementation_hashes", {})
    for model_name, metrics in (("Qwen", qwen), ("Gemma", gemma)):
        report_hash = metrics.get("implementation_hashes", {}).get(
            "report-importance-score-formal.py"
        )
        if report_hash != frozen_implementations.get("report-importance-score-formal.py"):
            raise SystemExit(f"{model_name} report implementation differs from the candidate freeze")

    frozen_scores = list(freeze["candidate_scores_transferred_unchanged"])
    if set(frozen_scores) != set(qwen["scores"]) or set(frozen_scores) != set(gemma["scores"]):
        raise SystemExit("Qwen/Gemma score sets differ from the sealed candidate table")
    rows = []
    finalists = []
    for score in frozen_scores:
        q_point = score_point(qwen, score)
        g_point = score_point(gemma, score)
        two_model_pass = q_point["admission_pass"] and g_point["admission_pass"]
        if two_model_pass:
            finalists.append(score)
        q_rho = q_point["rho"]
        g_rho = g_point["rho"]
        rows.append(
            {
                "score_id": score,
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
        q_bootstrap = bootstrap_arrays(paths["qwen_bootstrap"])
        g_bootstrap = bootstrap_arrays(paths["gemma_bootstrap"])
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
                    "Frozen rho/tau paired comparisons did not separate every finalist; paired "
                    "AUC/Precision@K/KL tie-break resampling is required before selecting a winner."
                )

    output = {
        "schema_version": "importance-score-two-model-decision-v0.1",
        "lockbox_validity": {
            "status": "valid one-shot Gemma lockbox",
            "qwen_candidate_freeze_sha256": sha256_file(paths["qwen_candidate_freeze"]),
            "gemma_prejoin_score_sha256": prejoin["score_table_sha256"],
            "formula_or_threshold_change_after_gemma": False,
            "third_model_required": False,
        },
        "candidate_rows": rows,
        "two_model_finalists": finalists,
        "paired_bootstrap_comparisons": comparisons,
        "metric_tie_break_order": list(METRIC_ORDER),
        "decision": decision,
        "winner_candidate": winner,
        "reason": reason,
        "phase_6_authorized": decision == "WINNER",
        "input_hashes": {name: sha256_file(path) for name, path in paths.items()},
    }
    output_dir = args.output_dir.expanduser().resolve()
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
        "| Score | Qwen rho | Gemma rho | worst rho | Qwen gate | Gemma gate |",
        "|---|---:|---:|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['score_id']} | {row['qwen']['rho']} | {row['gemma']['rho']} | "
            f"{row['worst_model_rho']} | {row['qwen']['admission_pass']} | "
            f"{row['gemma']['admission_pass']} |"
        )
    (output_dir / "two-model-decision.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"decision": decision, "winner": winner}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
