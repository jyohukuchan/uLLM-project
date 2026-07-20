#!/usr/bin/env python3
"""Calculate Phase-2 C0/C2/C3 screen artifacts and ordinal correlations.

This is an offline, CPU-only post-processing step.  C0 is read from the
existing sampled AQ evaluator; C2 and C3 are computed from the frozen
activation statistics and a deterministic, bounded weight sample.  The tool
labels every resulting row as provisional when the format contract is
incomplete, so its correlations cannot be mistaken for admission evidence.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from safetensors import safe_open
from scipy.stats import kendalltau, rankdata, spearmanr


EPSILON = 1e-30
UNKNOWN = "unknown"


def parse_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            normalized = {}
            for field in fields:
                value = row.get(field, "")
                if value is None:
                    value = ""
                elif isinstance(value, bool):
                    value = "true" if value else "false"
                elif isinstance(value, (list, dict, tuple)):
                    value = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
                normalized[field] = value
            writer.writerow(normalized)


def load_stats(path: Path) -> dict[str, torch.Tensor]:
    if path.is_dir():
        path = path / "activation_second_moments.safetensors"
    result: dict[str, torch.Tensor] = {}
    with safe_open(path, framework="pt", device="cpu") as handle:
        for key in handle.keys():
            result[key] = handle.get_tensor(key)
    return result


def module_key(hf_name: str, stats: dict[str, torch.Tensor]) -> str:
    if not hf_name.endswith(".weight"):
        raise ValueError(f"expected weight name, got {hf_name}")
    name = hf_name.removesuffix(".weight")
    candidates = (name, name.removeprefix("model."), f"model.{name.removeprefix('model.')}")
    for candidate in candidates:
        if candidate in stats and f"{candidate}.mean_abs" in stats and f"{candidate}.max_abs" in stats:
            return candidate
    raise KeyError(f"activation statistics missing for {hf_name}; tried {candidates}")


def tensor_file_map(model_dir: Path) -> dict[str, Path]:
    index = json.loads((model_dir / "model.safetensors.index.json").read_text(encoding="utf-8"))
    return {name: model_dir / filename for name, filename in index["weight_map"].items()}


def deterministic_weight_sample(
    tensor_path: Path,
    tensor_name: str,
    sample_size: int,
    seed: int,
    rows_per_chunk: int,
) -> tuple[torch.Tensor, list[int], int]:
    """Take an explicit no-replacement systematic sample without full tensor materialization."""
    with safe_open(tensor_path, framework="pt", device="cpu") as handle:
        view = handle.get_slice(tensor_name)
        shape = [int(value) for value in view.get_shape()]
        if len(shape) != 2:
            raise ValueError(f"expected 2-D tensor, got {tensor_name}: {shape}")
        rows, columns = shape
        n_params = rows * columns
        count = min(sample_size, n_params)
        stride = max(1, n_params // count)
        digest = hashlib.sha256(f"{seed}:{tensor_name}".encode("utf-8")).digest()
        offset = int.from_bytes(digest[:8], "big") % stride
        indices = torch.arange(offset, n_params, stride, dtype=torch.long)[:count]
        chunks: list[torch.Tensor] = []
        for row_start in range(0, rows, rows_per_chunk):
            row_end = min(rows, row_start + rows_per_chunk)
            start = row_start * columns
            end = row_end * columns
            left = int(torch.searchsorted(indices, torch.tensor(start), right=False))
            right = int(torch.searchsorted(indices, torch.tensor(end), right=False))
            if right <= left:
                continue
            chunk = view[row_start:row_end].to(torch.float32).flatten()
            local = indices[left:right] - start
            chunks.append(chunk.index_select(0, local))
        sample = torch.cat(chunks) if chunks else torch.empty(0, dtype=torch.float32)
    if sample.numel() != count:
        raise RuntimeError(f"sample size mismatch for {tensor_name}: {sample.numel()} != {count}")
    return sample, shape, n_params


def safe_quantile(values: torch.Tensor, q: float) -> float:
    if values.numel() == 0:
        return float("nan")
    return float(torch.quantile(values.to(torch.float64), q))


def finite_metric(values: list[float]) -> tuple[float | None, float | None]:
    if len(values) < 2 or len(set(values)) < 2:
        return None, None
    return float(spearmanr(values[0], values[1]).statistic), float(kendalltau(values[0], values[1], variant="b").statistic)


def correlation(score: list[float], label: list[float]) -> dict[str, Any]:
    if len(score) < 4 or len(set(label)) < 2 or len(set(score)) < 2:
        return {"rho": None, "tau_b": None, "defined": False}
    return {
        "rho": float(spearmanr(score, label).statistic),
        "tau_b": float(kendalltau(score, label, variant="b").statistic),
        "defined": True,
    }


def all_rank_metrics(rows: list[dict[str, Any]], score_column: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    family_rows: list[dict[str, Any]] = []
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["canonical_family"])].append(row)
    valid_family_corrs = []
    family_score_medians = []
    family_label_medians = []
    residual_scores = []
    residual_labels = []
    raw_scores = []
    raw_labels = []
    for family, members in sorted(grouped.items()):
        score = [float(row[score_column]) for row in members]
        label = [float(row["ordinal_ud"]) for row in members]
        corr = correlation(score, label)
        family_rows.append(
            {
                "scope": "within_family",
                "score_id": score_column,
                "family": family,
                "n": len(members),
                "label_nonconstant": len(set(label)) > 1,
                **corr,
            }
        )
        if corr["defined"]:
            valid_family_corrs.append(corr)
        family_score_medians.append(float(np.median(score)))
        family_label_medians.append(float(np.median(label)))
        score_mid = (rankdata(score, method="average") - 0.5) / len(score)
        label_mid = (rankdata(label, method="average") - 0.5) / len(label)
        residual_scores.extend(float(item) for item in score_mid)
        residual_labels.extend(float(item) for item in label_mid)
        raw_scores.extend(score)
        raw_labels.extend(label)
    family_level = correlation(family_score_medians, family_label_medians)
    residualized = correlation(residual_scores, residual_labels)
    raw = correlation(raw_scores, raw_labels)
    summary = {
        "score_id": score_column,
        "primary_within_family_macro": {
            "rho": float(np.mean([item["rho"] for item in valid_family_corrs])) if valid_family_corrs else None,
            "tau_b": float(np.mean([item["tau_b"] for item in valid_family_corrs])) if valid_family_corrs else None,
            "defined_family_count": len(valid_family_corrs),
            "all_family_count": len(grouped),
        },
        "family_level_secondary": family_level,
        "residualized_whole_model_secondary": residualized,
        "raw_whole_model_descriptive": raw,
    }
    return family_rows, summary


def layer_cluster_bootstrap(
    rows: list[dict[str, Any]],
    score_columns: list[str],
    replicates: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Bootstrap primary macro rank metrics by resampling whole layers.

    A single draw sequence is shared by every score and every family.  This
    preserves same-layer Q/K/V and other tensor dependence, and leaves paired
    score comparisons possible in a later formal run.  The present Qwen-only
    output is ordinal exploratory evidence, not an admission decision.
    """
    layer_rows: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        layer_rows[int(row["layer_id"])].append(row)
    layer_ids = sorted(layer_rows)
    if not layer_ids:
        raise ValueError("no layer clusters available for bootstrap")
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(layer_ids), size=(replicates, len(layer_ids)), endpoint=False)
    samples: list[dict[str, Any]] = []
    by_score: dict[str, Any] = {}
    for score_column in score_columns:
        rho_values: list[float] = []
        tau_values: list[float] = []
        undefined = 0
        for replicate, draw in enumerate(draws):
            sampled_rows = [
                row
                for layer_index in draw
                for row in layer_rows[layer_ids[int(layer_index)]]
            ]
            _, summary = all_rank_metrics(sampled_rows, score_column)
            primary = summary["primary_within_family_macro"]
            rho = primary["rho"]
            tau = primary["tau_b"]
            if rho is None or tau is None:
                undefined += 1
            else:
                rho_values.append(float(rho))
                tau_values.append(float(tau))
            samples.append(
                {
                    "model_id": str(rows[0]["model_id"]),
                    "score_id": score_column,
                    "replicate": replicate,
                    "layer_cluster_count": len(layer_ids),
                    "primary_rho": rho,
                    "primary_tau_b": tau,
                    "defined_family_count": int(primary["defined_family_count"]),
                    "all_family_count": int(primary["all_family_count"]),
                }
            )
        by_score[score_column] = {
            "method": f"{replicates:,} layer-cluster bootstrap; one common layer-ID resample across all families and scores; 95% percentile CI",
            "replicates_requested": replicates,
            "replicates_defined": len(rho_values),
            "replicates_undefined": undefined,
            "rho_ci95_percentile": (
                [float(np.quantile(np.asarray(rho_values), 0.025)), float(np.quantile(np.asarray(rho_values), 0.975))]
                if rho_values
                else None
            ),
            "tau_b_ci95_percentile": (
                [float(np.quantile(np.asarray(tau_values), 0.025)), float(np.quantile(np.asarray(tau_values), 0.975))]
                if tau_values
                else None
            ),
        }
    return samples, by_score


def family_direction_consistency(family_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Report fixed direction-gate components without deciding admission."""
    defined = [row for row in family_rows if row["defined"]]
    positive = [row for row in defined if float(row["tau_b"]) > 0.0]
    major_negative = [
        row["family"]
        for row in defined
        if int(row["n"]) >= 16 and float(row["tau_b"]) < -0.20
    ]
    return {
        "defined_family_count": len(defined),
        "positive_tau_b_family_count": len(positive),
        "positive_tau_b_fraction": len(positive) / len(defined) if defined else None,
        "major_family_tau_b_below_minus_0_20": major_negative,
        "admission_thresholds_not_decided_here": {
            "positive_sign_fraction": ">= 0.70",
            "major_family_negative_tau_b": "none for n >= 16",
        },
    }


def rank_stability(per_shard: list[dict[str, dict[str, float]]], score_id: str, k: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for left in range(len(per_shard)):
        for right in range(left + 1, len(per_shard)):
            common = sorted(set(per_shard[left]) & set(per_shard[right]))
            a = [per_shard[left][name][score_id] for name in common]
            b = [per_shard[right][name][score_id] for name in common]
            rho = float(spearmanr(a, b).statistic) if len(set(a)) > 1 and len(set(b)) > 1 else None
            top_a = {name for name, _ in sorted(((name, per_shard[left][name][score_id]) for name in common), key=lambda item: item[1], reverse=True)[:k]}
            top_b = {name for name, _ in sorted(((name, per_shard[right][name][score_id]) for name in common), key=lambda item: item[1], reverse=True)[:k]}
            union = top_a | top_b
            rows.append(
                {
                    "score_id": score_id,
                    "shard_left": left,
                    "shard_right": right,
                    "n": len(common),
                    "spearman_rho": rho,
                    "top_k": k,
                    "top_k_jaccard": len(top_a & top_b) / len(union) if union else None,
                }
            )
    return rows


def parse_c0_rows(path: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        if item.get("status") != "ok":
            continue
        names = item.get("scope", {}).get("tensor_names", [])
        if len(names) != 1:
            continue
        result[str(names[0])] = item
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--combined-stats", type=Path, required=True)
    parser.add_argument("--shard-stats", type=Path, action="append", required=True)
    parser.add_argument("--c0-jsonl", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--weight-sample-size", type=int, default=65536)
    parser.add_argument("--rows-per-chunk", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--torch-threads", type=int, default=16)
    parser.add_argument("--torch-interop-threads", type=int, default=1)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if (
        args.weight_sample_size < 1
        or args.rows_per_chunk < 1
        or args.torch_threads < 1
        or args.torch_interop_threads < 1
        or args.bootstrap_replicates < 1
    ):
        raise SystemExit("sample/chunk/thread/bootstrap arguments must be >= 1")
    torch.set_num_threads(args.torch_threads)
    torch.set_num_interop_threads(args.torch_interop_threads)
    labels = [row for row in parse_tsv(args.labels.expanduser().resolve()) if row["eligible"] == "true"]
    if not labels:
        raise SystemExit("no eligible labels")
    combined = load_stats(args.combined_stats.expanduser().resolve())
    shard_stats = [load_stats(path.expanduser().resolve()) for path in args.shard_stats]
    if len(shard_stats) != 4:
        raise SystemExit("exactly four --shard-stats inputs are required")
    c0 = parse_c0_rows(args.c0_jsonl.expanduser().resolve())
    file_map = tensor_file_map(args.model_dir.expanduser().resolve())
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    scores: list[dict[str, Any]] = []
    per_shard: list[dict[str, dict[str, float]]] = [dict() for _ in shard_stats]
    for label in labels:
        hf_name = label["hf_name"]
        if hf_name not in c0:
            raise SystemExit(f"C0 missing eligible tensor: {hf_name}")
        if hf_name not in file_map:
            raise SystemExit(f"safetensors index missing eligible tensor: {hf_name}")
        module = module_key(hf_name, combined)
        h = combined[module].to(torch.float64)
        mean_abs = combined[f"{module}.mean_abs"].to(torch.float64)
        max_abs = combined[f"{module}.max_abs"].to(torch.float64)
        if not bool(torch.isfinite(h).all() and torch.isfinite(mean_abs).all() and torch.isfinite(max_abs).all()):
            raise SystemExit(f"non-finite activation statistic: {hf_name}")
        sample, shape, n_params = deterministic_weight_sample(
            file_map[hf_name], hf_name, args.weight_sample_size, args.seed, args.rows_per_chunk
        )
        sample64 = sample.to(torch.float64)
        rms_w = float(sample64.square().mean().sqrt())
        q999_w = safe_quantile(sample64.abs(), 0.999)
        max_w = float(sample64.abs().max())
        rms_x = h.clamp_min(0).sqrt()
        q99_x = safe_quantile(max_abs, 0.99)
        q50_rms_x = safe_quantile(rms_x, 0.50)
        max_x = float(max_abs.max())
        r_x = q99_x / (q50_rms_x + EPSILON)
        r_w = q999_w / (rms_w + EPSILON)
        r_x_max = max_x / (q50_rms_x + EPSILON)
        r_w_max = max_w / (rms_w + EPSILON)
        c0_metrics = c0[hf_name].get("metrics", {})
        c0_l = c0_metrics.get("weighted_relative_mse")
        if c0_l is None or not math.isfinite(float(c0_l)):
            raise SystemExit(f"non-finite C0 metric: {hf_name}")
        row = {
            "model_id": label["model_id"],
            "architecture": label["architecture"],
            "layer_id": int(label["layer_id"]),
            "canonical_family": label["canonical_family"],
            "gguf_name": label["gguf_name"],
            "hf_name": hf_name,
            "shape": label["shape"],
            "n_params": int(label["n_params"]),
            "ordinal_ud": float(label["ordinal_ud"]),
            "packed_bpp_ud": float(label["packed_bpp_ud"]),
            "qtype_ud": label["qtype_ud"],
            "promoted_vs_4bit_floor": label["promoted_vs_4bit_floor"] == "true",
            "C0_A_sample_normalized_weight": c0_metrics.get("weighted_sse"),
            "C0_reference_energy_sample_normalized_weight": c0_metrics.get("weighted_reference_sse"),
            "C0_L": float(c0_l),
            "S_AWQ_level": math.log(float(mean_abs.mean()) + EPSILON),
            "S_AWQ_tail": float(mean_abs.topk(max(1, math.ceil(0.01 * mean_abs.numel()))).values.sum() / (mean_abs.sum() + EPSILON)),
            "S_range": 0.5 * (math.log(r_x) + math.log(r_w)),
            "S_range_true_max": 0.5 * (math.log(r_x_max) + math.log(r_w_max)),
            "activation_rms_mean": float(rms_x.mean()),
            "activation_tail": float(mean_abs.topk(max(1, math.ceil(0.01 * mean_abs.numel()))).values.sum() / (mean_abs.sum() + EPSILON)),
            "activation_q99_channel_max": q99_x,
            "weight_rms_sample": rms_w,
            "weight_q999_abs_sample": q999_w,
            "weight_sample_count": int(sample.numel()),
            "weight_sample_method": "deterministic systematic no-replacement sample; q0.999 and RMS are sample estimates",
            "candidate_status": "provisional_sampler_sensitivity_only; not eligible for gain/allocation/admission",
        }
        scores.append(row)
        for shard_index, stats in enumerate(shard_stats):
            shard_module = module_key(hf_name, stats)
            shard_h = stats[shard_module].to(torch.float64)
            shard_a = stats[f"{shard_module}.mean_abs"].to(torch.float64)
            shard_max = stats[f"{shard_module}.max_abs"].to(torch.float64)
            shard_rms = shard_h.clamp_min(0).sqrt()
            shard_q99_x = safe_quantile(shard_max, 0.99)
            shard_q50_rms = safe_quantile(shard_rms, 0.50)
            shard_rx = shard_q99_x / (shard_q50_rms + EPSILON)
            shard_rx_max = float(shard_max.max()) / (shard_q50_rms + EPSILON)
            per_shard[shard_index][hf_name] = {
                "S_AWQ_level": math.log(float(shard_a.mean()) + EPSILON),
                "S_AWQ_tail": float(shard_a.topk(max(1, math.ceil(0.01 * shard_a.numel()))).values.sum() / (shard_a.sum() + EPSILON)),
                "S_range": 0.5 * (math.log(shard_rx) + math.log(r_w)),
                "S_range_true_max": 0.5 * (math.log(shard_rx_max) + math.log(r_w_max)),
            }

    score_columns = ["C0_L", "S_AWQ_level", "S_AWQ_tail", "S_range", "S_range_true_max"]
    family_metrics: list[dict[str, Any]] = []
    model_metrics: dict[str, Any] = {
        "schema_version": "importance-score-phase2-metrics-v0.1",
        "status": "exploratory Qwen-only provisional screen; no paired static baseline and no Gemma lockbox",
        "n_eligible": len(scores),
        "scores": {},
        "binary_retrieval": "not computed: paired promoted label unavailable; promoted_vs_4bit_floor is exploratory only",
        "bootstrap": {
            "status": "ordinal-only exploratory Qwen bootstrap; it cannot satisfy the two-model admission gate",
        },
    }
    for score_column in score_columns:
        by_family, summary = all_rank_metrics(scores, score_column)
        family_metrics.extend(by_family)
        summary["family_direction_consistency_exploratory"] = family_direction_consistency(by_family)
        model_metrics["scores"][score_column] = summary
    pq.write_table(pa.Table.from_pylist(scores), output_dir / "scores.parquet", compression="zstd")
    family_fields = ["scope", "score_id", "family", "n", "label_nonconstant", "rho", "tau_b"]
    write_tsv(output_dir / "metrics-by-family.tsv", family_metrics, family_fields)

    fallback_k = max(1, sum(1 for row in scores if row["promoted_vs_4bit_floor"]))
    stability: list[dict[str, Any]] = []
    for score_column in ("S_AWQ_level", "S_AWQ_tail", "S_range", "S_range_true_max"):
        stability.extend(rank_stability(per_shard, score_column, fallback_k))
    stability.append(
        {
            "score_id": "C0_L",
            "shard_left": "",
            "shard_right": "",
            "n": len(scores),
            "spearman_rho": "",
            "top_k": fallback_k,
            "top_k_jaccard": "",
            "status": "not_measured: C0 candidate fitting was run only on merged D_stats",
        }
    )
    write_tsv(
        output_dir / "shard-stability.tsv",
        stability,
        ["score_id", "shard_left", "shard_right", "n", "spearman_rho", "top_k", "top_k_jaccard", "status"],
    )

    # Disagreement rows remain explicitly exploratory because their binary
    # comparison uses the allowed fallback only.
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in scores:
        by_family[row["canonical_family"]].append(row)
    residuals = []
    for members in by_family.values():
        c0_rank = rankdata([row["C0_L"] for row in members], method="average") / len(members)
        ud_rank = rankdata([row["ordinal_ud"] for row in members], method="average") / len(members)
        for row, c_rank, u_rank in zip(members, c0_rank, ud_rank, strict=True):
            row["_c0_family_rank"] = float(c_rank)
            row["_ud_family_rank"] = float(u_rank)
            row["_rank_residual"] = float(c_rank - u_rank)
            residuals.append(abs(float(c_rank - u_rank)))
    c0_order = sorted(scores, key=lambda row: row["C0_L"])
    selected = {row["hf_name"] for row in c0_order[-fallback_k:] if not row["promoted_vs_4bit_floor"]}
    selected |= {row["hf_name"] for row in c0_order[:fallback_k] if row["promoted_vs_4bit_floor"]}
    threshold = float(np.quantile(np.asarray(residuals), 0.95)) if residuals else float("inf")
    selected |= {row["hf_name"] for row in scores if abs(row["_rank_residual"]) >= threshold}
    disagreement_rows = []
    for row in scores:
        if row["hf_name"] not in selected:
            continue
        disagreement_rows.append(
            {
                "model": row["model_id"], "layer": row["layer_id"], "family": row["canonical_family"],
                "gguf_name": row["gguf_name"], "shape": row["shape"], "n_params": row["n_params"],
                "ud_type": row["qtype_ud"], "static_type": UNKNOWN, "promotion_delta_ordinal": UNKNOWN,
                "promotion_delta_bpp": UNKNOWN, "score_raw": row["C0_L"], "score_family_rank": row["_c0_family_rank"],
                "ud_family_rank": row["_ud_family_rank"], "rank_residual": row["_rank_residual"],
                "activation_rms": row["activation_rms_mean"], "activation_tail": row["activation_tail"],
                "range_score": row["S_range"], "diag_mse": row["C0_L"], "block_cov_mse": "", "block_output_mse": "",
                "fisher": "", "kl": "", "flip_rate": "", "qualitative_class": "unknown",
                "notes": "exploratory fallback promoted_vs_4bit_floor; not paired-binary/admission evidence",
            }
        )
    disagreement_fields = [
        "model", "layer", "family", "gguf_name", "shape", "n_params", "ud_type", "static_type",
        "promotion_delta_ordinal", "promotion_delta_bpp", "score_raw", "score_family_rank", "ud_family_rank",
        "rank_residual", "activation_rms", "activation_tail", "range_score", "diag_mse", "block_cov_mse",
        "block_output_mse", "fisher", "kl", "flip_rate", "qualitative_class", "notes",
    ]
    write_tsv(output_dir / "disagreements.tsv", disagreement_rows, disagreement_fields)
    write_tsv(
        output_dir / "kl-subset.tsv",
        [{"status": "not_started", "reason": "C6 direct KL is GPU-required-near and excluded from this CPU-only run"}],
        ["status", "reason"],
    )
    bootstrap_rows, bootstrap_summary = layer_cluster_bootstrap(
        scores, score_columns, args.bootstrap_replicates, args.seed
    )
    model_metrics["bootstrap"].update(
        {
            "status": "ordinal-only exploratory Qwen bootstrap; paired binary and two-model admission remain HOLD",
            "scores": bootstrap_summary,
        }
    )
    pq.write_table(pa.Table.from_pylist(bootstrap_rows), output_dir / "bootstrap-samples.parquet", compression="zstd")
    (output_dir / "metrics-by-model.json").write_text(
        json.dumps(model_metrics, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    report_lines = [
        "# Qwen Phase-2 provisional importance-score screen",
        "",
        f"- Eligible core tensors: {len(scores)}.",
        "- C0 uses the sampled AQ evaluator with merged activation statistics; format storage semantics remain incomplete.",
        "- C2 is exact over collected activation moments; C3 weight q0.999/RMS uses a deterministic bounded sample and is therefore a screen estimate.",
        "- Static Q4_K_M is absent, so binary retrieval metrics and admission gates are HOLD; fallback labels are exploratory only.",
        "- Gemma and C1/C4/C5/C6 are not run.",
        "",
        "## Primary within-family macro correlations",
        "",
        "| Score | Spearman rho | Kendall tau-b | Defined families |",
        "|---|---:|---:|---:|",
    ]
    for score_column in score_columns:
        primary = model_metrics["scores"][score_column]["primary_within_family_macro"]
        bootstrap = model_metrics["bootstrap"]["scores"][score_column]
        report_lines.append(
            f"| {score_column} | {primary['rho']!s} | {primary['tau_b']!s} | {primary['defined_family_count']} |"
        )
        report_lines.append(
            f"  - layer-cluster bootstrap 95% CI: rho={bootstrap['rho_ci95_percentile']}, tau-b={bootstrap['tau_b_ci95_percentile']}."
        )
    (output_dir / "final-report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(json.dumps(model_metrics, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
