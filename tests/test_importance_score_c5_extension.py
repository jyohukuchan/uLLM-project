from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


def load_tool():
    path = Path(__file__).resolve().parents[1] / "tools" / "extend-importance-score-prejoin-c5.py"
    spec = importlib.util.spec_from_file_location("test_c5_prejoin_extension", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def make_gradient_receipt(
    tool,
    tmp_path: Path,
    mode: str,
    rows: list[dict],
    input_hashes: dict[str, str],
) -> Path:
    result_path = tmp_path / f"{mode}.jsonl"
    receipt_path = tmp_path / f"{mode}.receipt.json"
    write_jsonl(result_path, rows)
    names = sorted(row["hf_name"] for row in rows)
    write_json(
        receipt_path,
        {
            "schema_version": "importance-score-gradient-c5-receipt-v0.1",
            "status": "sealed C5 gradient score output",
            "mode": mode,
            "model_id": "m",
            "result_path": str(result_path),
            "result_sha256": tool.sha256_file(result_path),
            "tensor_count": len(rows),
            "name_set_sha256": tool.hashlib.sha256(
                tool.canonical_json(names).encode("utf-8")
            ).hexdigest(),
            "workspace_git_head": tool.git_revision(),
            "input_hashes": input_hashes,
            "implementation_hashes": {
                "gradient_runner": {
                    "sha256": tool.sha256_file(
                        Path(tool.__file__).resolve().parent
                        / "collect-importance-gradient-scores.py"
                    )
                }
            },
            "execution_settings": {"mode": mode, "smoke": False},
        },
    )
    return receipt_path


def extension_fixture(tool, tmp_path: Path, *, nonfinite: bool = False) -> list[str]:
    names = ["model.layers.0.mlp.up_proj.weight", "model.layers.1.mlp.up_proj.weight"]
    base_rows = []
    for index, name in enumerate(names):
        base_rows.append(
            {
                "model_id": "m",
                "architecture": "a",
                "layer_id": index,
                "canonical_family": "mlp_up",
                "hf_name": name,
                "shape": [2, 2],
                "n_params": 4,
                **{score: float(index + offset) for offset, score in enumerate(tool.BASE_SCORE_COLUMNS)},
            }
        )
    base_scores = tmp_path / "base.jsonl"
    base_shards = tmp_path / "base-shards.json"
    base_receipt = tmp_path / "base.receipt.json"
    write_jsonl(base_scores, base_rows)
    write_json(
        base_shards,
        [
            {name: {"S_range": float(index)} for index, name in enumerate(names)}
            for _ in range(4)
        ],
    )
    common_paths = {}
    for key in ("candidate_manifest", "score_registry", "corpus_manifest", "fisher_corpus_manifest"):
        path = tmp_path / f"{key}.json"
        write_json(path, {"key": key})
        common_paths[key] = path
    source_roster = tmp_path / "source-roster.jsonl"
    source_manifest = tmp_path / "source-roster-manifest.json"
    source_roster.write_text("source\n", encoding="utf-8")
    write_json(source_manifest, {"status": "frozen"})
    source_hashes = {
        "source_roster": tool.sha256_file(source_roster),
        "source_roster_manifest": tool.sha256_file(source_manifest),
    }
    write_json(
        base_receipt,
        {
            "schema_version": "importance-score-prejoin-receipt-v0.1",
            "status": tool.SEALED_STATUS,
            "score_table_sha256": tool.sha256_file(base_scores),
            "shard_scores_sha256": tool.sha256_file(base_shards),
            "tensor_count": len(names),
            "tensor_name_set_sha256": tool.tensor_name_set_sha256(set(names)),
            "candidate_score_columns": list(tool.BASE_SCORE_COLUMNS),
            "execution_settings": {"seed": 0},
            "input_hashes": {
                **source_hashes,
                **{
                    key: tool.sha256_file(path)
                    for key, path in common_paths.items()
                    if key != "fisher_corpus_manifest"
                },
            },
        },
    )
    expected_gradient_inputs = {
        **source_hashes,
        "candidate_manifest": tool.sha256_file(common_paths["candidate_manifest"]),
        "fisher_corpus_manifest": tool.sha256_file(
            common_paths["fisher_corpus_manifest"]
        ),
    }

    empirical_rows = []
    self_rows = []
    for index, base in enumerate(base_rows):
        empirical_scores = {
            key: float(index + offset + 1)
            for offset, key in enumerate(
                (*tool.C5A_ALL_KEYS, *tool.EMPIRICAL_FISHER_ALL_KEYS)
            )
        }
        if nonfinite and index == 0:
            empirical_scores[tool.C5A_ALL_KEYS[0]] = float("nan")
        self_scores = {
            key: float(index + offset + 20)
            for offset, key in enumerate(tool.SELF_FISHER_ALL_KEYS)
        }
        structure = {key: base[key] for key in tool.STRUCTURAL_KEYS}
        empirical_rows.append(
            {
                **structure,
                "scores": empirical_scores,
                "shard_scores": [
                    {"shard_index": shard, **empirical_scores} for shard in range(4)
                ],
            }
        )
        self_rows.append(
            {
                **structure,
                "scores": self_scores,
                "shard_scores": [
                    {"shard_index": shard, **self_scores} for shard in range(4)
                ],
            }
        )
    empirical_receipt = make_gradient_receipt(
        tool, tmp_path, "empirical", empirical_rows, expected_gradient_inputs
    )
    self_receipt = make_gradient_receipt(
        tool, tmp_path, "self_fisher", self_rows, expected_gradient_inputs
    )
    output_dir = tmp_path / "output"
    sys.argv = [
        "extend-importance-score-prejoin-c5.py",
        "--base-prejoin-scores",
        str(base_scores),
        "--base-prejoin-receipt",
        str(base_receipt),
        "--base-prejoin-shard-scores",
        str(base_shards),
        "--empirical-receipt",
        str(empirical_receipt),
        "--self-fisher-receipt",
        str(self_receipt),
        "--candidate-manifest",
        str(common_paths["candidate_manifest"]),
        "--score-registry",
        str(common_paths["score_registry"]),
        "--corpus-manifest",
        str(common_paths["corpus_manifest"]),
        "--fisher-corpus-manifest",
        str(common_paths["fisher_corpus_manifest"]),
        "--stage",
        "full",
        "--output-dir",
        str(output_dir),
    ]
    return names


def test_c5_prejoin_extension_preserves_legacy_rows_and_seals_roles(
    tmp_path: Path, monkeypatch
) -> None:
    tool = load_tool()
    old_argv = sys.argv
    try:
        names = extension_fixture(tool, tmp_path)
        assert tool.main() == 0
    finally:
        monkeypatch.setattr(sys, "argv", old_argv)
    rows = tool.read_jsonl(tmp_path / "output" / "scores-prejoin.jsonl")
    receipt = json.loads(
        (tmp_path / "output" / "scores-prejoin.receipt.json").read_text(
            encoding="utf-8"
        )
    )

    assert [row["hf_name"] for row in rows] == names
    assert receipt["c5_extension"]["legacy_base_preserved_exactly"] is True
    assert set(receipt["reported_score_columns"]) == {
        *tool.BASE_SCORE_COLUMNS,
        *tool.C5A_SCORE_COLUMNS,
        *tool.C5B_SCORE_COLUMNS,
    }
    assert set(receipt["secondary_score_columns"]) == set(tool.SECONDARY)
    assert set(receipt["implementation_hashes"]) == set(tool.CURRENT_IMPLEMENTATION_FILES)


def test_c5_prejoin_extension_rejects_nonfinite_scores(
    tmp_path: Path, monkeypatch
) -> None:
    tool = load_tool()
    old_argv = sys.argv
    try:
        extension_fixture(tool, tmp_path, nonfinite=True)
        with pytest.raises(ValueError, match="not finite"):
            tool.main()
    finally:
        monkeypatch.setattr(sys, "argv", old_argv)


def test_gradient_collector_compatibility_is_narrowly_scoped() -> None:
    tool = load_tool()
    receipt = {"architecture": "gemma4_text"}
    current = tool.sha256_file(
        Path(tool.__file__).resolve().parent
        / "collect-importance-gradient-scores.py"
    )

    accepted = tool.gradient_collector_compatibility(
        receipt,
        expected_mode="empirical",
        expected_model_id="gemma-4-E4B-it",
        sealed_runner_sha256=tool.GEMMA4_EMPIRICAL_BASE_RUNNER_SHA256,
        current_runner_sha256=current,
    )
    assert accepted["status"].startswith("accepted-score-equivalent")

    with pytest.raises(ValueError, match="implementation hash mismatch"):
        tool.gradient_collector_compatibility(
            receipt,
            expected_mode="self_fisher",
            expected_model_id="gemma-4-E4B-it",
            sealed_runner_sha256=tool.GEMMA4_EMPIRICAL_BASE_RUNNER_SHA256,
            current_runner_sha256=current,
        )
    with pytest.raises(ValueError, match="implementation hash mismatch"):
        tool.gradient_collector_compatibility(
            receipt,
            expected_mode="empirical",
            expected_model_id="another-model",
            sealed_runner_sha256=tool.GEMMA4_EMPIRICAL_BASE_RUNNER_SHA256,
            current_runner_sha256=current,
        )
