#!/usr/bin/env python3
"""Produce the frozen Phase-4/5 statistical report for one model."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from safetensors import safe_open
from scipy.stats import kendalltau, rankdata, spearmanr
from sklearn.metrics import average_precision_score, ndcg_score, roc_auc_score


LOW = "aq4_e4m3_g16_ts_flloyd16"
HIGH = "aq5_e4m3_g16_ts_flloyd32"
EPSILON = 1e-30


def load_screen_module():
    path = Path(__file__).resolve().parent / "summarize-importance-score-screen.py"
    spec = importlib.util.spec_from_file_location("importance_formal_screen_helpers", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


SCREEN = load_screen_module()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_labels(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return [row for row in csv.DictReader(handle, delimiter="\t") if row["eligible"] == "true"]


def tensor_file_map(model_dir: Path) -> dict[str, Path]:
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        return {name: model_dir / filename for name, filename in index["weight_map"].items()}
    result = {}
    for path in sorted(model_dir.glob("*.safetensors")):
        with safe_open(path, framework="pt", device="cpu") as handle:
            for name in handle.keys():
                if name in result:
                    raise ValueError(f"duplicate safetensor key: {name}")
                result[name] = path
    return result


def parse_quantizer_rows(path: Path) -> dict[str, dict[str, dict[str, Any]]]:
    result: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("status") != "ok":
            continue
        names = row.get("scope", {}).get("tensor_names", [])
        if len(names) != 1:
            continue
        result[str(names[0])][str(row["candidate"]["candidate_id"])] = row
    return result


def parse_c1(path: Path | None) -> dict[str, dict[str, dict[str, Any]]]:
    result: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    if path is None:
        return result
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            result[str(row["hf_name"])][str(row["candidate_id"])] = row
    return result


def parse_perturbation(path: Path | None, mode: str) -> dict[str, dict[str, dict[str, Any]]]:
    result: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    if path is None or not path.is_file():
        return result
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("status") == "ok" and row.get("mode") == mode:
            result[str(row["tensor_name"])][str(row["candidate_id"])] = row
    return result


def full_loss(metrics: dict[str, Any]) -> float:
    value = metrics.get("weighted_sse_estimated_full_tensor")
    if value is None or not math.isfinite(float(value)):
        raise ValueError("formal C0 row lacks a finite full-tensor A estimate")
    return float(value)


def score_features(
    labels: list[dict[str, str]],
    model_dir: Path,
    combined_stats: dict[str, torch.Tensor],
    shard_stats: list[dict[str, torch.Tensor]],
    c0: dict[str, dict[str, dict[str, Any]]],
    c1: dict[str, dict[str, dict[str, Any]]],
    c4: dict[str, dict[str, dict[str, Any]]],
    weight_sample_size: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, dict[str, float]]]]:
    files = tensor_file_map(model_dir)
    rows = []
    per_shard: list[dict[str, dict[str, float]]] = [dict() for _ in shard_stats]
    for label in labels:
        name = label["hf_name"]
        if set(c0.get(name, {})) != {LOW, HIGH}:
            raise ValueError(f"formal C0 low/high coverage missing: {name}")
        module = SCREEN.module_key(name, combined_stats)
        second = combined_stats[module].to(torch.float64)
        mean_abs = combined_stats[f"{module}.mean_abs"].to(torch.float64)
        max_abs = combined_stats[f"{module}.max_abs"].to(torch.float64)
        sample, shape, n_params = SCREEN.deterministic_weight_sample(
            files[name], name, weight_sample_size, seed, 128
        )
        sample64 = sample.to(torch.float64)
        rms_w = float(sample64.square().mean().sqrt())
        q999_w = SCREEN.safe_quantile(sample64.abs(), 0.999)
        max_w = float(sample64.abs().max())
        rms_x = second.clamp_min(0).sqrt()
        q99_x = SCREEN.safe_quantile(max_abs, 0.99)
        q50_rms_x = SCREEN.safe_quantile(rms_x, 0.50)
        range_score = 0.5 * (
            math.log(q99_x / (q50_rms_x + EPSILON))
            + math.log(q999_w / (rms_w + EPSILON))
        )
        range_true_max = 0.5 * (
            math.log(float(max_abs.max()) / (q50_rms_x + EPSILON))
            + math.log(max_w / (rms_w + EPSILON))
        )
        c0_low = c0[name][LOW]["metrics"]
        c0_high = c0[name][HIGH]["metrics"]
        c0_a_low = full_loss(c0_low)
        c0_a_high = full_loss(c0_high)
        c0_raw_gain = c0_a_low - c0_a_high
        row: dict[str, Any] = {
            "model_id": label["model_id"],
            "architecture": label["architecture"],
            "layer_id": int(label["layer_id"]),
            "canonical_family": label["canonical_family"],
            "gguf_name": label["gguf_name"],
            "hf_name": name,
            "shape": label["shape"],
            "n_params": int(label["n_params"]),
            "qtype_ud": label["qtype_ud"],
            "qtype_static": label["qtype_static"],
            "ordinal_ud": float(label["ordinal_ud"]),
            "ordinal_static": float(label["ordinal_static"]),
            "packed_bpp_ud": float(label["packed_bpp_ud"]),
            "packed_bpp_static": float(label["packed_bpp_static"]),
            "promotion_delta_ordinal": float(label["promotion_delta_ordinal"]),
            "promotion_delta_bpp": float(label["promotion_delta_bpp"]),
            "promoted": label["promoted"] == "true",
            "C0_I": float(c0_low["weighted_relative_mse"]),
            "C0_A_low": c0_a_low,
            "C0_A_high": c0_a_high,
            "C0_raw_gain": c0_raw_gain,
            "C0_G": max(0.0, c0_raw_gain),
            "S_AWQ_level": math.log(float(mean_abs.mean()) + EPSILON),
            "S_AWQ_tail": float(
                mean_abs.topk(max(1, math.ceil(0.01 * mean_abs.numel()))).values.sum()
                / (mean_abs.sum() + EPSILON)
            ),
            "S_range": range_score,
            "S_range_true_max_diagnostic": range_true_max,
            "activation_rms_mean": float(rms_x.mean()),
            "activation_tail": float(
                mean_abs.topk(max(1, math.ceil(0.01 * mean_abs.numel()))).values.sum()
                / (mean_abs.sum() + EPSILON)
            ),
            "weight_rms_sample": rms_w,
            "weight_q999_abs_sample": q999_w,
            "weight_sample_count": int(sample.numel()),
            "weight_shape_audit": shape,
            "weight_n_params_audit": n_params,
        }
        if name in c1 and set(c1[name]) == {LOW, HIGH}:
            low = c1[name][LOW]
            high = c1[name][HIGH]
            raw_gain = float(low["C1_A_estimated_full_tensor"]) - float(
                high["C1_A_estimated_full_tensor"]
            )
            row.update(
                {
                    "C1_I": float(low["C1_L"]),
                    "C1_A_low": float(low["C1_A_estimated_full_tensor"]),
                    "C1_A_high": float(high["C1_A_estimated_full_tensor"]),
                    "C1_raw_gain": raw_gain,
                    "C1_G": max(0.0, raw_gain),
                }
            )
        if name in c4 and set(c4[name]) == {LOW, HIGH}:
            low = c4[name][LOW]["metrics"]
            high = c4[name][HIGH]["metrics"]
            raw_gain = float(low["C4_A"]) - float(high["C4_A"])
            row.update(
                {
                    "C4_I_subset": float(low["C4_L"]),
                    "C4_A_low_subset": float(low["C4_A"]),
                    "C4_A_high_subset": float(high["C4_A"]),
                    "C4_raw_gain_subset": raw_gain,
                    "C4_G_subset": max(0.0, raw_gain),
                }
            )
        rows.append(row)

        for index, stats in enumerate(shard_stats):
            shard_module = SCREEN.module_key(name, stats)
            shard_second = stats[shard_module].to(torch.float64)
            shard_abs = stats[f"{shard_module}.mean_abs"].to(torch.float64)
            shard_max = stats[f"{shard_module}.max_abs"].to(torch.float64)
            shard_rms = shard_second.clamp_min(0).sqrt()
            q99 = SCREEN.safe_quantile(shard_max, 0.99)
            q50 = SCREEN.safe_quantile(shard_rms, 0.50)
            per_shard[index][name] = {
                "S_AWQ_level": math.log(float(shard_abs.mean()) + EPSILON),
                "S_AWQ_tail": float(
                    shard_abs.topk(max(1, math.ceil(0.01 * shard_abs.numel()))).values.sum()
                    / (shard_abs.sum() + EPSILON)
                ),
                "S_range": 0.5 * (
                    math.log(q99 / (q50 + EPSILON))
                    + math.log(q999_w / (rms_w + EPSILON))
                ),
            }
    return rows, per_shard


def binary_metrics(rows: list[dict[str, Any]], score: str) -> dict[str, Any]:
    mixed = []
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_family[row["canonical_family"]].append(row)
    for family, members in sorted(by_family.items()):
        y = np.asarray([int(row["promoted"]) for row in members])
        values = np.asarray([float(row[score]) for row in members])
        if len(set(y.tolist())) == 2:
            mixed.append(
                {
                    "family": family,
                    "n": len(members),
                    "positives": int(y.sum()),
                    "roc_auc": float(roc_auc_score(y, values)),
                    "pr_auc": float(average_precision_score(y, values)),
                }
            )
    y = np.asarray([int(row["promoted"]) for row in rows])
    values = np.asarray([float(row[score]) for row in rows])
    k = int(y.sum())
    order = sorted(
        range(len(rows)),
        key=lambda i: (
            -float(rows[i][score]),
            hashlib.sha256(str(rows[i]["hf_name"]).encode()).digest(),
        ),
    )
    selected = order[:k]
    precision = float(y[selected].mean()) if selected else None
    prevalence = float(y.mean())
    threshold = prevalence + 0.25 * (1.0 - prevalence)
    global_mixed = len(set(y.tolist())) == 2
    return {
        "ranking_score": score,
        "mixed_family_count": len(mixed),
        "families": mixed,
        "AUC_within": float(np.mean([item["roc_auc"] for item in mixed])) if mixed else None,
        "PR_AUC_within": float(np.mean([item["pr_auc"] for item in mixed])) if mixed else None,
        "AUC_global_descriptive": float(roc_auc_score(y, values)) if global_mixed else None,
        "PR_AUC_global_descriptive": float(average_precision_score(y, values)) if global_mixed else None,
        "positive_count_K": k,
        "prevalence": prevalence,
        "Precision_at_K": precision,
        "Recall_at_K": float(y[selected].sum() / max(k, 1)),
        "NDCG_at_K": (
            float(ndcg_score(y.reshape(1, -1), values.reshape(1, -1), k=k)) if k > 0 else None
        ),
        "precision_gate_threshold": threshold,
        "precision_gate_pass": precision is not None and precision >= threshold,
        "tie_break": "descending score, then SHA-256(hf_name) ascending",
    }


def direction_gate(family_rows: list[dict[str, Any]]) -> dict[str, Any]:
    defined = [row for row in family_rows if row["defined"]]
    fraction = (
        sum(float(row["tau_b"]) > 0 for row in defined) / len(defined) if defined else None
    )
    negatives = [
        row["family"]
        for row in defined
        if int(row["n"]) >= 16 and float(row["tau_b"]) < -0.20
    ]
    return {
        "defined_family_count": len(defined),
        "positive_tau_fraction": fraction,
        "major_family_tau_b_below_minus_0_20": negatives,
        "pass": fraction is not None and fraction >= 0.70 and not negatives,
    }


def cluster_bootstrap(rows, score_columns, replicates, seed):
    return SCREEN.layer_cluster_bootstrap(rows, score_columns, replicates, seed)


def permutation_tests(rows: list[dict[str, Any]], score_columns: list[str], replicates: int, seed: int):
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_family[row["canonical_family"]].append(row)
    for members in by_family.values():
        members.sort(key=lambda row: int(row["layer_id"]))
    global_layers = sorted({int(row["layer_id"]) for row in rows})
    observed = {
        score: SCREEN.all_rank_metrics(rows, score)[1]["primary_within_family_macro"]["rho"]
        for score in score_columns
    }
    rng = np.random.default_rng(seed)
    null = {score: [] for score in score_columns}
    draw_rows = []
    for replicate in range(replicates):
        permutation = rng.permutation(global_layers).tolist()
        permuted = []
        for family, members in by_family.items():
            family_layers = {int(row["layer_id"]) for row in members}
            restricted = [layer for layer in permutation if layer in family_layers]
            labels_by_layer = {
                int(row["layer_id"]): float(row["ordinal_ud"]) for row in members
            }
            for row, source_layer in zip(members, restricted, strict=True):
                item = dict(row)
                item["ordinal_ud"] = labels_by_layer[source_layer]
                permuted.append(item)
        for score in score_columns:
            rho = SCREEN.all_rank_metrics(permuted, score)[1]["primary_within_family_macro"]["rho"]
            null[score].append(float(rho) if rho is not None else float("nan"))
            draw_rows.append({"score_id": score, "replicate": replicate, "primary_rho": rho})
    summary = {}
    raw_p = {}
    for score in score_columns:
        values = np.asarray([value for value in null[score] if math.isfinite(value)])
        obs = float(observed[score])
        p = (1 + int((np.abs(values) >= abs(obs)).sum())) / (1 + len(values))
        raw_p[score] = p
        summary[score] = {
            "observed_rho": obs,
            "replicates_defined": len(values),
            "two_sided_permutation_p": p,
        }
    ordered = sorted(raw_p, key=raw_p.get)
    adjusted = {}
    running = 1.0
    for reverse_index in range(len(ordered) - 1, -1, -1):
        score = ordered[reverse_index]
        rank = reverse_index + 1
        running = min(running, raw_p[score] * len(ordered) / rank)
        adjusted[score] = running
    for score in score_columns:
        summary[score]["benjamini_hochberg_adjusted_p"] = adjusted[score]
    return draw_rows, summary


def kl_metrics(
    rows: list[dict[str, Any]],
    score_columns: list[str],
    c6: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    low_kl = {}
    high_kl = {}
    for name, candidates in c6.items():
        if set(candidates) == {LOW, HIGH}:
            low_kl[name] = float(candidates[LOW]["metrics"]["C6_L"])
            high_kl[name] = float(candidates[HIGH]["metrics"]["C6_L"])
    by_name = {row["hf_name"]: row for row in rows}
    common = sorted(set(by_name) & set(low_kl))
    result = {
        "KL_core_tensor_count": len(common),
        "score_selection_independent": True,
        "scores": {},
    }
    for score in score_columns:
        values = [float(by_name[name][score]) for name in common]
        target = [low_kl[name] for name in common]
        rho = (
            float(spearmanr(values, target).statistic)
            if len(common) >= 4 and len(set(values)) > 1 and len(set(target)) > 1
            else None
        )
        result["scores"][score] = {"rho_score_vs_AQ4_KL": rho}
    gains = [max(0.0, low_kl[name] - high_kl[name]) for name in common]
    for prefix in ("C0", "C1"):
        column = f"{prefix}_G"
        if f"{prefix}_I" in score_columns:
            values = [float(by_name[name][column]) for name in common]
            result["scores"][column]["rho_gain_vs_KL_recovery_secondary"] = (
                float(spearmanr(values, gains).statistic)
                if len(set(values)) > 1 and len(set(gains)) > 1
                else None
            )
    baseline = result["scores"].get("C0_I", {}).get("rho_score_vs_AQ4_KL")
    for score in score_columns:
        rho = result["scores"][score]["rho_score_vs_AQ4_KL"]
        result["scores"][score]["gate"] = {
            "rho_at_least_0_30": rho is not None and rho >= 0.30,
            "not_more_than_0_05_below_C0": (
                rho is not None and baseline is not None and rho >= baseline - 0.05
            ),
            "pass": (
                rho is not None
                and rho >= 0.30
                and baseline is not None
                and rho >= baseline - 0.05
            ),
        }
    return result


def teacher_coverage(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_family[row["canonical_family"]].append(row)
    nonconstant = 0
    mixed = 0
    details = []
    for family, members in sorted(by_family.items()):
        ordinals = {row["ordinal_ud"] for row in members}
        promoted = {row["promoted"] for row in members}
        n = len(members)
        nonconstant_here = n >= 4 and len(ordinals) > 1
        mixed_here = n >= 4 and promoted == {False, True}
        nonconstant += nonconstant_here
        mixed += mixed_here
        details.append(
            {
                "family": family,
                "n": n,
                "ordinal_nonconstant": nonconstant_here,
                "promoted_mixed": mixed_here,
            }
        )
    return {
        "paired_coverage": sum(row["qtype_static"] != "unknown" for row in rows) / len(rows),
        "nonconstant_repeated_family_count": nonconstant,
        "mixed_positive_negative_family_count": mixed,
        "pass": nonconstant >= 4 and mixed >= 3,
        "families": details,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--combined-stats", type=Path, required=True)
    parser.add_argument("--shard-stats", type=Path, action="append", required=True)
    parser.add_argument("--c0-jsonl", type=Path, required=True)
    parser.add_argument("--c0-shard-jsonl", type=Path, action="append", default=[])
    parser.add_argument("--c1-jsonl", type=Path)
    parser.add_argument("--c4-jsonl", type=Path)
    parser.add_argument("--c6-jsonl", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--weight-sample-size", type=int, default=65536)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--permutation-replicates", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if len(args.shard_stats) != 4:
        raise SystemExit("formal report requires exactly four activation-stat shards")
    model_dir = args.model_dir.expanduser().resolve()
    labels_path = args.labels.expanduser().resolve()
    labels = read_labels(labels_path)
    combined = SCREEN.load_stats(args.combined_stats.expanduser().resolve())
    shards = [SCREEN.load_stats(path.expanduser().resolve()) for path in args.shard_stats]
    c0 = parse_quantizer_rows(args.c0_jsonl.expanduser().resolve())
    c1 = parse_c1(args.c1_jsonl.expanduser().resolve() if args.c1_jsonl else None)
    c4 = parse_perturbation(args.c4_jsonl.expanduser().resolve() if args.c4_jsonl else None, "c4")
    c6 = parse_perturbation(args.c6_jsonl.expanduser().resolve() if args.c6_jsonl else None, "c6")
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows, per_shard = score_features(
        labels, model_dir, combined, shards, c0, c1, c4, args.weight_sample_size, args.seed
    )
    full_scores = ["C0_I", "S_AWQ_level", "S_AWQ_tail", "S_range"]
    if all("C1_I" in row for row in rows):
        full_scores.append("C1_I")
    binary_score = {
        "C0_I": "C0_G",
        "C1_I": "C1_G",
        "S_AWQ_level": "S_AWQ_level",
        "S_AWQ_tail": "S_AWQ_tail",
        "S_range": "S_range",
    }
    family_rows = []
    metrics: dict[str, Any] = {
        "schema_version": "importance-score-formal-statistics-v0.1",
        "model_id": rows[0]["model_id"],
        "eligible_tensor_count": len(rows),
        "paired_labels_sha256": sha256_file(labels_path),
        "teacher_coverage": teacher_coverage(rows),
        "scores": {},
    }
    for score in full_scores:
        families, rank = SCREEN.all_rank_metrics(rows, score)
        family_rows.extend(families)
        rank["direction_gate"] = direction_gate(families)
        rank["binary"] = binary_metrics(rows, binary_score[score])
        metrics["scores"][score] = rank

    bootstrap_rows, bootstrap = cluster_bootstrap(
        rows, full_scores, args.bootstrap_replicates, args.seed
    )
    permutation_rows, permutation = permutation_tests(
        rows, full_scores, args.permutation_replicates, args.seed
    )
    kl = kl_metrics(rows, full_scores, c6)
    metrics["bootstrap"] = bootstrap
    metrics["permutation"] = permutation
    metrics["KL_core"] = kl

    c4_subset_rows = [row for row in rows if "C4_I_subset" in row]
    if c4_subset_rows:
        c4_families, c4_rank = SCREEN.all_rank_metrics(c4_subset_rows, "C4_I_subset")
        c4_rank["coverage"] = {
            "tensor_count": len(c4_subset_rows),
            "eligible_tensor_fraction": len(c4_subset_rows) / len(rows),
            "admission_use": "descriptive subset only; not a full-coverage finalist",
        }
        c4_rank["binary"] = binary_metrics(c4_subset_rows, "C4_G_subset")
        metrics["C4_subset"] = c4_rank
        family_rows.extend(c4_families)

    c6_oracle_rows = []
    by_name = {row["hf_name"]: row for row in rows}
    for name, candidates in c6.items():
        if name not in by_name or set(candidates) != {LOW, HIGH}:
            continue
        item = dict(by_name[name])
        low = float(candidates[LOW]["metrics"]["C6_L"])
        high = float(candidates[HIGH]["metrics"]["C6_L"])
        item["C6_I_oracle"] = low
        item["C6_G_oracle"] = max(0.0, low - high)
        c6_oracle_rows.append(item)
    if c6_oracle_rows:
        c6_families, c6_rank = SCREEN.all_rank_metrics(c6_oracle_rows, "C6_I_oracle")
        c6_rank["coverage"] = {
            "tensor_count": len(c6_oracle_rows),
            "eligible_tensor_fraction": len(c6_oracle_rows) / len(rows),
            "admission_use": "oracle KL-core only; not a production cheap-score winner",
        }
        c6_rank["binary"] = binary_metrics(c6_oracle_rows, "C6_G_oracle")
        metrics["C6_oracle_subset"] = c6_rank
        family_rows.extend(c6_families)

    for score in full_scores:
        primary = metrics["scores"][score]["primary_within_family_macro"]
        binary = metrics["scores"][score]["binary"]
        ci = bootstrap[score]["rho_ci95_percentile"]
        qwen_side = {
            "rho": primary["rho"] is not None and primary["rho"] >= 0.30,
            "tau_b": primary["tau_b"] is not None and primary["tau_b"] >= 0.20,
            "rho_ci_lower_positive": ci is not None and ci[0] > 0,
            "family_direction": metrics["scores"][score]["direction_gate"]["pass"],
            "AUC_within": binary["AUC_within"] is not None and binary["AUC_within"] >= 0.65,
            "Precision_at_K": binary["precision_gate_pass"],
            "KL": kl["scores"].get(score, {}).get("gate", {}).get("pass", False),
        }
        metrics["scores"][score]["admission_gate"] = {
            "components": qwen_side,
            "pass": all(qwen_side.values()) and metrics["teacher_coverage"]["pass"],
        }

    stability = []
    k = int(sum(row["promoted"] for row in rows))
    for score in ("S_AWQ_level", "S_AWQ_tail", "S_range"):
        stability.extend(SCREEN.rank_stability(per_shard, score, k))
    if len(args.c0_shard_jsonl) == 4:
        c0_shards = [parse_quantizer_rows(path.expanduser().resolve()) for path in args.c0_shard_jsonl]
        shard_maps = []
        for shard in c0_shards:
            shard_maps.append(
                {
                    name: {"C0_I": float(candidates[LOW]["metrics"]["weighted_relative_mse"])}
                    for name, candidates in shard.items()
                    if LOW in candidates
                }
            )
        stability.extend(SCREEN.rank_stability(shard_maps, "C0_I", k))

    pq.write_table(pa.Table.from_pylist(rows), output_dir / "scores.parquet", compression="zstd")
    pq.write_table(
        pa.Table.from_pylist(bootstrap_rows), output_dir / "bootstrap-samples.parquet", compression="zstd"
    )
    pq.write_table(
        pa.Table.from_pylist(permutation_rows), output_dir / "permutation-samples.parquet", compression="zstd"
    )
    SCREEN.write_tsv(
        output_dir / "metrics-by-family.tsv",
        family_rows,
        ["scope", "score_id", "family", "n", "label_nonconstant", "rho", "tau_b", "defined"],
    )
    SCREEN.write_tsv(
        output_dir / "shard-stability.tsv",
        stability,
        ["score_id", "shard_left", "shard_right", "n", "spearman_rho", "top_k", "top_k_jaccard"],
    )
    (output_dir / "metrics-by-model.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report = [
        f"# {rows[0]['model_id']} formal importance-score report",
        "",
        f"- Eligible paired tensors: {len(rows)}.",
        f"- Teacher coverage gate: {metrics['teacher_coverage']['pass']}.",
        "",
        "| Score | rho | tau-b | rho CI95 | AUC_within | Precision@K | KL rho | gate |",
        "|---|---:|---:|---|---:|---:|---:|---|",
    ]
    for score in full_scores:
        item = metrics["scores"][score]
        primary = item["primary_within_family_macro"]
        binary = item["binary"]
        kl_rho = kl["scores"].get(score, {}).get("rho_score_vs_AQ4_KL")
        report.append(
            f"| {score} | {primary['rho']} | {primary['tau_b']} | "
            f"{bootstrap[score]['rho_ci95_percentile']} | {binary['AUC_within']} | "
            f"{binary['Precision_at_K']} | {kl_rho} | {item['admission_gate']['pass']} |"
        )
    (output_dir / "final-report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
