from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools/evaluate-qwen35-aq4-phase7-holdout.py"


def load_module():
    spec = importlib.util.spec_from_file_location("phase7_holdout_evaluator_test", TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def rows(logits_l2: float = 0.1):
    return [
        {
            "case_id": f"case-{index}",
            "metrics": {
                "token_agreement_rate": 1.0,
                "topk_overlap_rate_k10": 1.0,
                "logits_cosine": 1.0,
                "logits_relative_l2": logits_l2,
                "hidden_cosine": 1.0,
                "hidden_relative_l2": 0.1,
                "hidden_max_abs": 0.2,
                "bf16_top1_retained_in_aq4_top10_rate": 1.0,
            },
        }
        for index in range(24)
    ]


def receipt(module):
    result = {"derived_bounds": {}}
    for name, spec in module.PROTOCOL.METRICS.items():
        if spec["role"] == "diagnostic_only":
            result["derived_bounds"][name] = {"bound": None}
        elif spec["direction"] == "higher":
            result["derived_bounds"][name] = {"bound": 0.0}
        else:
            result["derived_bounds"][name] = {"bound": 1.0}
    return result


def test_frozen_policy_assessment_rejects_any_pathological_relative_l2_row() -> None:
    module = load_module()
    policy = module.PROTOCOL.policy()

    passed = module.assess_policy(policy, receipt(module), rows())
    assert passed["frozen_policy_passed"] is True

    rejected = module.assess_policy(policy, receipt(module), rows(logits_l2=1.01))
    assert rejected["frozen_policy_passed"] is False
    assert rejected["pathological_relative_l2_rejections"]["logits_relative_l2"] == [
        f"case-{index}" for index in range(24)
    ]
