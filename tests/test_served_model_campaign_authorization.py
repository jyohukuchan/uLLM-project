from __future__ import annotations

import importlib.util
import json
import os
import sys
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools/served_model_campaign_authorization.py"
SPEC = importlib.util.spec_from_file_location(
    "test_served_model_campaign_authorization_module", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
AUTH = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = AUTH
SPEC.loader.exec_module(AUTH)

NOW = datetime(2026, 7, 24, 12, 0, 0, tzinfo=timezone.utc)


def policy(tmp_path: Path) -> object:
    claims = tmp_path / "claims"
    outcomes = tmp_path / "outcomes"
    claims.mkdir(mode=0o700)
    outcomes.mkdir(mode=0o700)
    return AUTH.RegistryPolicy(
        claim_registry=claims,
        outcome_registry=outcomes,
        required_uid=os.geteuid(),
    )


def document(tmp_path: Path) -> dict[str, object]:
    outputs = tmp_path / "outputs"
    outputs.mkdir(mode=0o700)
    return {
        "schema_version": AUTH.AUTHORIZATION_SCHEMA,
        "authorization_id": "sq8-v2-window-20260724-001",
        "issued_at": AUTH.utc_timestamp(NOW - timedelta(minutes=1)),
        "expires_at": AUTH.utc_timestamp(NOW + timedelta(hours=2)),
        "max_attempts": 1,
        "authorization_note": "Reviewed candidate-only evidence window.",
        "purpose": "temporary_candidate_active_evidence_collection_only",
        "required_final_route": "restore_exact_aq4_then_bundle_v2_activation",
        "source": {"commit": "a" * 40, "tree": "b" * 40},
        "before": {
            "model_id": "ullm-qwen3.5-9b-aq4",
            "format_id": "AQ4_0",
            "manifest_sha256": "1" * 64,
            "worker_binary_sha256": "2" * 64,
            "promotion_source_commit": "c" * 40,
        },
        "candidate": {
            "model_id": "ullm-qwen3-14b-sq8",
            "format_id": "SQ8_0",
            "manifest_sha256": "3" * 64,
            "worker_protocol": "ullm.worker.v2",
            "worker_binary_sha256": "4" * 64,
            "promotion_source_commit": "a" * 40,
            "promotion_receipt_sha256": "5" * 64,
        },
        "campaigns": {
            "sq8_full": {
                "run_id": "sq8-full-20260724-001",
                "final_path": str(outputs / "sq8-full"),
            },
            "reasoning_release": {
                "run_id": "reasoning-release-20260724-001",
                "final_path": str(outputs / "reasoning-release"),
            },
            "reasoning_browser": {
                "run_id": "reasoning-browser-20260724-001",
                "final_path": str(outputs / "reasoning-browser"),
            },
        },
        "rollback": {
            "backup_path": str(outputs / "aq4-exact-backup.json"),
            "systemd_unit_sha256": "6" * 64,
            "environment_sha256": "7" * 64,
        },
        "prior_outcome": None,
    }


def issue(tmp_path: Path) -> tuple[object, Path, dict[str, object]]:
    selected_policy = policy(tmp_path)
    value = document(tmp_path)
    authorization_dir = tmp_path / "authorizations"
    authorization_dir.mkdir(mode=0o700)
    path = authorization_dir / "authorization.json"
    record = AUTH.issue_authorization(
        value,
        path,
        now=NOW,
        policy=selected_policy,
    )
    return selected_policy, path, value


def outcome_document(
    claim: object,
    *,
    status: str = "succeeded_restored",
    failure_stage: str | None = None,
) -> dict[str, object]:
    authorization = claim.authorization.document
    stages = {name: "passed" for name in AUTH.OUTCOME_STAGE_FIELDS}
    restoration = {
        "expected_manifest_sha256": authorization["before"]["manifest_sha256"],
        "observed_manifest_sha256": authorization["before"]["manifest_sha256"],
        "bytes_equal": True,
        "reverse_reconciliation_passed": True,
        "final_checks_passed": True,
        "model_id": "ullm-qwen3.5-9b-aq4",
        "format_id": "AQ4_0",
        "worker_binary_sha256": authorization["before"]["worker_binary_sha256"],
    }
    if failure_stage is not None:
        stages[failure_stage] = "failed"
    if status == "failed_restore":
        stages["aq4_restore"] = "failed"
        stages["reverse_reconciliation"] = "skipped"
        stages["final_checks"] = "skipped"
        restoration.update(
            observed_manifest_sha256=None,
            bytes_equal=False,
            reverse_reconciliation_passed=False,
            final_checks_passed=False,
            model_id=None,
            format_id=None,
            worker_binary_sha256=None,
        )
    campaigns = {}
    for name, value in authorization["campaigns"].items():
        campaigns[name] = {
            "run_id": value["run_id"],
            "path": value["final_path"],
            "kind": "directory",
            "sha256": "8" * 64,
            "artifact_count": 1,
            "total_bytes": 2,
            "selected_artifacts": {"SHA256SUMS": "9" * 64},
        }
    return {
        "schema_version": AUTH.OUTCOME_SCHEMA,
        "authorization_id": authorization["authorization_id"],
        "authorization_path": str(claim.authorization.snapshot.path),
        "authorization_sha256": claim.authorization.snapshot.sha256,
        "claim_path": str(claim.snapshot.path),
        "claim_sha256": claim.snapshot.sha256,
        "started_at": claim.document["claimed_at"],
        "completed_at": AUTH.utc_timestamp(NOW + timedelta(minutes=1)),
        "status": status,
        "failure_stage": failure_stage,
        "stages": stages,
        "candidate_observations": [
            {
                "stage": "candidate_checks",
                "active_manifest_sha256": authorization["candidate"][
                    "manifest_sha256"
                ],
                "bytes_equal": True,
            }
        ],
        "campaigns": campaigns,
        "restoration": restoration,
    }


def test_issue_and_claim_are_canonical_immutable_and_replay_safe(
    tmp_path: Path,
) -> None:
    selected_policy, path, value = issue(tmp_path)
    assert path.read_bytes() == AUTH.canonical_json_bytes(value)
    metadata = path.stat()
    assert metadata.st_mode & 0o777 == 0o444
    assert metadata.st_nlink == 1

    claim = AUTH.claim_authorization(path, now=NOW, policy=selected_policy)
    assert claim.document["attempt"] == 1
    assert claim.document["max_attempts"] == 1
    assert claim.snapshot.path == AUTH.claim_path(
        claim.authorization.snapshot.sha256,
        policy=selected_policy,
    ).resolve()
    assert claim.snapshot.mode == 0o444
    assert claim.snapshot.nlink == 1
    loaded = AUTH.load_claim(path, now=NOW, policy=selected_policy)
    assert loaded.snapshot.sha256 == claim.snapshot.sha256
    with pytest.raises(AUTH.AuthorizationConsumed):
        AUTH.claim_authorization(path, now=NOW, policy=selected_policy)


def test_concurrent_claim_has_exactly_one_winner(tmp_path: Path) -> None:
    selected_policy, path, _ = issue(tmp_path)
    barrier = threading.Barrier(8)
    outcomes: list[str] = []
    lock = threading.Lock()

    def compete() -> None:
        barrier.wait()
        try:
            AUTH.claim_authorization(path, now=NOW, policy=selected_policy)
        except AUTH.AuthorizationConsumed:
            result = "consumed"
        else:
            result = "claimed"
        with lock:
            outcomes.append(result)

    threads = [threading.Thread(target=compete) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert outcomes.count("claimed") == 1
    assert outcomes.count("consumed") == 7


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda value: value.update(max_attempts=2), "max_attempts"),
        (
            lambda value: value["candidate"].update(format_id="AQ4_0"),
            "candidate identity",
        ),
        (
            lambda value: value["candidate"].update(worker_protocol="ullm.worker.v1"),
            "candidate identity",
        ),
        (
            lambda value: value["source"].update(commit="8" * 40),
            "source/candidate commit",
        ),
        (
            lambda value: value["campaigns"]["sq8_full"].update(
                final_path=value["campaigns"]["reasoning_release"]["final_path"]
            ),
            "distinct",
        ),
        (
            lambda value: value.update(expires_at="2026-07-24T11:59:59Z"),
            "expired",
        ),
    ],
)
def test_semantic_mutations_are_rejected(
    tmp_path: Path, mutate: object, match: str
) -> None:
    value = document(tmp_path)
    mutate(value)
    with pytest.raises(AUTH.AuthorizationError, match=match):
        AUTH.validate_authorization_document(
            value,
            now=NOW,
            required_uid=os.geteuid(),
        )


def test_authorization_file_requires_canonical_bytes_mode_owner_and_nlink(
    tmp_path: Path,
) -> None:
    selected_policy = policy(tmp_path)
    value = document(tmp_path)
    authorization_dir = tmp_path / "authorizations"
    authorization_dir.mkdir(mode=0o700)
    path = authorization_dir / "authorization.json"
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    path.chmod(0o444)
    with pytest.raises(AUTH.AuthorizationError, match="not canonical"):
        AUTH.load_authorization(path, now=NOW, policy=selected_policy)

    path.chmod(0o644)
    with pytest.raises(AUTH.AuthorizationError, match="metadata"):
        AUTH.load_authorization(path, now=NOW, policy=selected_policy)
    path.chmod(0o444)
    sibling = authorization_dir / "second-link.json"
    os.link(path, sibling)
    with pytest.raises(AUTH.AuthorizationError, match="metadata"):
        AUTH.load_authorization(path, now=NOW, policy=selected_policy)


def test_claim_remains_valid_after_authorized_outputs_are_created(
    tmp_path: Path,
) -> None:
    selected_policy, path, value = issue(tmp_path)
    claim = AUTH.claim_authorization(path, now=NOW, policy=selected_policy)
    for campaign in value["campaigns"].values():
        Path(campaign["final_path"]).mkdir()
    Path(value["rollback"]["backup_path"]).write_bytes(b"aq4\n")
    loaded = AUTH.load_claim(path, now=NOW, policy=selected_policy)
    assert loaded.snapshot.sha256 == claim.snapshot.sha256


def test_window_and_campaign_bindings_are_exact(tmp_path: Path) -> None:
    selected_policy, path, value = issue(tmp_path)
    claim = AUTH.claim_authorization(path, now=NOW, policy=selected_policy)
    AUTH.require_window_binding(
        claim,
        source_commit=value["source"]["commit"],
        source_tree=value["source"]["tree"],
        before_manifest_sha256=value["before"]["manifest_sha256"],
        candidate_manifest_sha256=value["candidate"]["manifest_sha256"],
        candidate_worker_binary_sha256=value["candidate"]["worker_binary_sha256"],
        candidate_promotion_receipt_sha256=value["candidate"][
            "promotion_receipt_sha256"
        ],
        rollback_backup_path=Path(value["rollback"]["backup_path"]),
    )
    campaign = value["campaigns"]["sq8_full"]
    AUTH.require_campaign_binding(
        claim,
        campaign_name="sq8_full",
        run_id=campaign["run_id"],
        final_path=Path(campaign["final_path"]),
    )
    with pytest.raises(AUTH.AuthorizationError, match="window identity"):
        AUTH.require_window_binding(
            claim,
            source_commit=value["source"]["commit"],
            source_tree=value["source"]["tree"],
            before_manifest_sha256=value["before"]["manifest_sha256"],
            candidate_manifest_sha256="0" * 64,
            candidate_worker_binary_sha256=value["candidate"][
                "worker_binary_sha256"
            ],
            candidate_promotion_receipt_sha256=value["candidate"][
                "promotion_receipt_sha256"
            ],
            rollback_backup_path=Path(value["rollback"]["backup_path"]),
        )
    with pytest.raises(AUTH.AuthorizationError, match="run/output"):
        AUTH.require_campaign_binding(
            claim,
            campaign_name="sq8_full",
            run_id="wrong",
            final_path=Path(campaign["final_path"]),
        )


def test_prior_outcome_must_be_live_immutable_and_hash_bound(
    tmp_path: Path,
) -> None:
    prior_policy, prior_authorization, _ = issue(tmp_path)
    claim = AUTH.claim_authorization(
        prior_authorization,
        now=NOW,
        policy=prior_policy,
    )
    outcome = tmp_path / "previous-outcome.json"
    outcome_value = outcome_document(claim)
    outcome.write_bytes(AUTH.canonical_json_bytes(outcome_value))
    outcome.chmod(0o444)
    successor_root = tmp_path / "successor"
    successor_root.mkdir()
    value = document(successor_root)
    value["prior_outcome"] = {
        "path": str(outcome),
        "sha256": AUTH.hashlib.sha256(outcome.read_bytes()).hexdigest(),
    }
    AUTH.validate_authorization_document(
        value,
        now=NOW,
        required_uid=os.geteuid(),
    )
    value["prior_outcome"]["sha256"] = "0" * 64
    with pytest.raises(AUTH.AuthorizationError, match="SHA-256 differs"):
        AUTH.validate_authorization_document(
            value,
            now=NOW,
            required_uid=os.geteuid(),
        )


def test_publish_and_load_outcome_are_exact_claim_bound_and_no_replace(
    tmp_path: Path,
) -> None:
    selected_policy, path, _ = issue(tmp_path)
    claim = AUTH.claim_authorization(path, now=NOW, policy=selected_policy)
    value = outcome_document(claim)

    snapshot = AUTH.publish_outcome(
        claim,
        value,
        policy=selected_policy,
    )

    assert snapshot.path == AUTH.outcome_path(
        claim.authorization.snapshot.sha256,
        policy=selected_policy,
    ).resolve()
    assert snapshot.mode == 0o444
    assert snapshot.nlink == 1
    loaded_snapshot, loaded = AUTH.load_outcome(
        path,
        now=NOW,
        policy=selected_policy,
    )
    assert loaded_snapshot.sha256 == snapshot.sha256
    assert loaded == value
    with pytest.raises(AUTH.AuthorizationConsumed, match="already exists"):
        AUTH.publish_outcome(claim, value, policy=selected_policy)


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (
            lambda value: value.update(authorization_sha256="0" * 64),
            "claim identity",
        ),
        (
            lambda value: value["campaigns"]["sq8_full"].update(run_id="wrong"),
            "run/output",
        ),
        (
            lambda value: value["restoration"].update(bytes_equal=False),
            "byte result",
        ),
        (
            lambda value: value["stages"].update(sq8_full="pending"),
            "pending",
        ),
    ],
)
def test_outcome_mutations_are_rejected(
    tmp_path: Path, mutate: object, match: str
) -> None:
    selected_policy, path, _ = issue(tmp_path)
    claim = AUTH.claim_authorization(path, now=NOW, policy=selected_policy)
    value = outcome_document(claim)
    mutate(value)
    with pytest.raises(AUTH.AuthorizationError, match=match):
        AUTH.validate_outcome_document(value, claim=claim)


def test_failed_restored_and_failed_restore_outcomes_are_distinct(
    tmp_path: Path,
) -> None:
    selected_policy, path, _ = issue(tmp_path)
    claim = AUTH.claim_authorization(path, now=NOW, policy=selected_policy)
    failed_campaign = outcome_document(
        claim,
        status="failed_restored",
        failure_stage="sq8_full",
    )
    AUTH.validate_outcome_document(failed_campaign, claim=claim)

    failed_restore = outcome_document(
        claim,
        status="failed_restore",
        failure_stage="aq4_restore",
    )
    AUTH.validate_outcome_document(failed_restore, claim=claim)
