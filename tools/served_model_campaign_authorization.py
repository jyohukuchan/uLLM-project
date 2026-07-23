#!/usr/bin/env python3
"""Strict authorization and one-shot claim primitives for a v2 campaign window."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


AUTHORIZATION_SCHEMA = (
    "ullm.served_model.v2_cross_model_campaign_authorization.v1"
)
CLAIM_SCHEMA = "ullm.served_model.v2_cross_model_campaign_claim.v1"
OUTCOME_SCHEMA = "ullm.served_model.v2_cross_model_campaign_outcome.v1"
FIXED_CLAIM_REGISTRY = Path("/var/lib/ullm/served-model-campaign-claims")
FIXED_OUTCOME_REGISTRY = Path("/var/lib/ullm/served-model-campaign-outcomes")
MAX_DOCUMENT_BYTES = 1_048_576
HASH_RE = re.compile(r"[0-9a-f]{64}\Z")
GIT_OBJECT_RE = re.compile(r"[0-9a-f]{40}\Z")
IDENTIFIER_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")
TIMESTAMP_RE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z\Z")

AUTHORIZATION_FIELDS = {
    "schema_version",
    "authorization_id",
    "issued_at",
    "expires_at",
    "max_attempts",
    "authorization_note",
    "purpose",
    "required_final_route",
    "source",
    "before",
    "candidate",
    "campaigns",
    "rollback",
    "prior_outcome",
}
SOURCE_FIELDS = {"commit", "tree"}
BEFORE_FIELDS = {
    "model_id",
    "format_id",
    "manifest_sha256",
    "worker_binary_sha256",
    "promotion_source_commit",
}
CANDIDATE_FIELDS = BEFORE_FIELDS | {
    "worker_protocol",
    "promotion_receipt_sha256",
}
CAMPAIGN_FIELDS = {"sq8_full", "reasoning_release", "reasoning_browser"}
CAMPAIGN_IDENTITY_FIELDS = {"run_id", "final_path"}
ROLLBACK_FIELDS = {
    "backup_path",
    "systemd_unit_sha256",
    "environment_sha256",
}
PRIOR_OUTCOME_FIELDS = {"path", "sha256"}
CLAIM_FIELDS = {
    "schema_version",
    "authorization_id",
    "authorization_path",
    "authorization_sha256",
    "claimed_at",
    "attempt",
    "max_attempts",
}


class AuthorizationError(ValueError):
    """Raised when an authorization or claim is unsafe or semantically invalid."""


class AuthorizationConsumed(AuthorizationError):
    """Raised when the authorization-derived claim already exists."""


@dataclass(frozen=True, slots=True)
class FileSnapshot:
    path: Path
    raw: bytes
    sha256: str
    mode: int
    uid: int
    nlink: int


@dataclass(frozen=True, slots=True)
class AuthorizationRecord:
    snapshot: FileSnapshot
    document: dict[str, Any]
    issued_at: datetime
    expires_at: datetime


@dataclass(frozen=True, slots=True)
class ClaimRecord:
    snapshot: FileSnapshot
    document: dict[str, Any]
    authorization: AuthorizationRecord


@dataclass(frozen=True, slots=True)
class RegistryPolicy:
    claim_registry: Path = FIXED_CLAIM_REGISTRY
    outcome_registry: Path = FIXED_OUTCOME_REGISTRY
    required_uid: int = 0


def _without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise AuthorizationError("JSON contains a duplicate object key")
        result[key] = value
    return result


def _reject_constant(_value: str) -> None:
    raise AuthorizationError("JSON contains a non-finite number")


def canonical_json_bytes(document: dict[str, Any]) -> bytes:
    try:
        return (
            json.dumps(
                document,
                ensure_ascii=True,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("ascii")
            + b"\n"
        )
    except (TypeError, ValueError, UnicodeError) as error:
        raise AuthorizationError("document is not canonicalizable JSON") from error


def strict_json_bytes(raw: bytes, label: str) -> dict[str, Any]:
    try:
        document = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_without_duplicates,
            parse_constant=_reject_constant,
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise AuthorizationError(f"{label} is not strict JSON") from error
    if not isinstance(document, dict):
        raise AuthorizationError(f"{label} root must be an object")
    return document


def _reject_symlink_components(
    path: Path, label: str, *, leaf_may_absent: bool
) -> None:
    if not path.is_absolute():
        raise AuthorizationError(f"{label} path must be absolute")
    current = Path(path.anchor)
    components = path.parts[1:]
    for index, component in enumerate(components):
        if component in {"", ".", ".."}:
            raise AuthorizationError(f"{label} path is not canonical")
        current /= component
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            if leaf_may_absent and index == len(components) - 1:
                return
            raise AuthorizationError(f"{label} has an absent path component") from None
        if stat.S_ISLNK(metadata.st_mode):
            raise AuthorizationError(f"{label} traverses a symlink")


def _stable_read(
    path: Path,
    label: str,
    *,
    maximum: int = MAX_DOCUMENT_BYTES,
    required_mode: int | None = None,
    required_uid: int | None = None,
    required_nlink: int | None = None,
) -> FileSnapshot:
    _reject_symlink_components(path, label, leaf_may_absent=False)
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise AuthorizationError(f"{label} is unavailable") from error
    try:
        before = os.fstat(descriptor)
        mode = stat.S_IMODE(before.st_mode)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_size <= 0
            or before.st_size > maximum
            or (required_mode is not None and mode != required_mode)
            or (required_uid is not None and before.st_uid != required_uid)
            or (required_nlink is not None and before.st_nlink != required_nlink)
        ):
            raise AuthorizationError(f"{label} metadata is unsafe")
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, min(65_536, maximum - total + 1))
            if not chunk:
                break
            total += len(chunk)
            if total > maximum:
                raise AuthorizationError(f"{label} exceeds its size bound")
            chunks.append(chunk)
        after = os.fstat(descriptor)
        identity_before = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
            before.st_mode,
            before.st_uid,
            before.st_gid,
            before.st_nlink,
        )
        identity_after = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
            after.st_mode,
            after.st_uid,
            after.st_gid,
            after.st_nlink,
        )
        raw = b"".join(chunks)
        if identity_before != identity_after or len(raw) != before.st_size:
            raise AuthorizationError(f"{label} changed while being read")
        return FileSnapshot(
            path=path.resolve(strict=True),
            raw=raw,
            sha256=hashlib.sha256(raw).hexdigest(),
            mode=mode,
            uid=before.st_uid,
            nlink=before.st_nlink,
        )
    finally:
        os.close(descriptor)


def _exact_object(value: Any, fields: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise AuthorizationError(f"{label} fields differ")
    return value


def _hash(value: Any, label: str) -> str:
    if not isinstance(value, str) or HASH_RE.fullmatch(value) is None:
        raise AuthorizationError(f"{label} must be a lowercase SHA-256")
    return value


def _git_object(value: Any, label: str) -> str:
    if not isinstance(value, str) or GIT_OBJECT_RE.fullmatch(value) is None:
        raise AuthorizationError(f"{label} must be a full lowercase Git object ID")
    return value


def _identifier(value: Any, label: str) -> str:
    if not isinstance(value, str) or IDENTIFIER_RE.fullmatch(value) is None:
        raise AuthorizationError(f"{label} is invalid")
    return value


def _bounded_text(value: Any, label: str, maximum: int = 4_096) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or len(value.encode("utf-8")) > maximum
        or any(ord(character) < 0x20 or ord(character) == 0x7F for character in value)
    ):
        raise AuthorizationError(f"{label} is invalid")
    return value


def _timestamp(value: Any, label: str) -> datetime:
    if not isinstance(value, str) or TIMESTAMP_RE.fullmatch(value) is None:
        raise AuthorizationError(f"{label} must be a canonical UTC timestamp")
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc
        )
    except ValueError as error:
        raise AuthorizationError(f"{label} is invalid") from error
    if parsed.strftime("%Y-%m-%dT%H:%M:%SZ") != value:
        raise AuthorizationError(f"{label} is not canonical")
    return parsed


def utc_timestamp(value: datetime) -> str:
    if value.tzinfo is None:
        raise AuthorizationError("timestamp must be timezone-aware")
    normalized = value.astimezone(timezone.utc).replace(microsecond=0)
    return normalized.strftime("%Y-%m-%dT%H:%M:%SZ")


def _absolute_future_path(value: Any, label: str) -> Path:
    if not isinstance(value, str) or "\x00" in value:
        raise AuthorizationError(f"{label} is invalid")
    path = Path(value)
    _reject_symlink_components(path, label, leaf_may_absent=True)
    if path.exists() or path.is_symlink():
        raise AuthorizationError(f"{label} must name a fresh output")
    return path


def _absolute_bound_path(value: Any, label: str, *, require_fresh: bool) -> Path:
    if require_fresh:
        return _absolute_future_path(value, label)
    if not isinstance(value, str) or "\x00" in value:
        raise AuthorizationError(f"{label} is invalid")
    path = Path(value)
    if path.exists() or path.is_symlink():
        _reject_symlink_components(path, label, leaf_may_absent=False)
    else:
        _reject_symlink_components(path, label, leaf_may_absent=True)
    return path


def _validate_outcome_reference(
    value: Any, label: str, *, required_uid: int
) -> None:
    reference = _exact_object(value, PRIOR_OUTCOME_FIELDS, label)
    expected_hash = _hash(reference["sha256"], f"{label}.sha256")
    if not isinstance(reference["path"], str):
        raise AuthorizationError(f"{label}.path is invalid")
    snapshot = _stable_read(
        Path(reference["path"]),
        label,
        required_mode=0o444,
        required_uid=required_uid,
        required_nlink=1,
    )
    if snapshot.sha256 != expected_hash:
        raise AuthorizationError(f"{label} SHA-256 differs")
    outcome = strict_json_bytes(snapshot.raw, label)
    if outcome.get("schema_version") != OUTCOME_SCHEMA:
        raise AuthorizationError(f"{label} schema differs")


def validate_authorization_document(
    document: dict[str, Any],
    *,
    now: datetime,
    required_uid: int = 0,
    validate_prior_outcome: bool = True,
    require_fresh_outputs: bool = True,
) -> tuple[datetime, datetime]:
    _exact_object(document, AUTHORIZATION_FIELDS, "authorization")
    if document["schema_version"] != AUTHORIZATION_SCHEMA:
        raise AuthorizationError("authorization schema differs")
    _identifier(document["authorization_id"], "authorization_id")
    issued_at = _timestamp(document["issued_at"], "issued_at")
    expires_at = _timestamp(document["expires_at"], "expires_at")
    normalized_now = now.astimezone(timezone.utc)
    if issued_at > normalized_now:
        raise AuthorizationError("authorization is not yet valid")
    if expires_at <= issued_at or expires_at <= normalized_now:
        raise AuthorizationError("authorization is expired")
    if type(document["max_attempts"]) is not int or document["max_attempts"] != 1:
        raise AuthorizationError("authorization max_attempts must equal one")
    _bounded_text(document["authorization_note"], "authorization_note")
    if document["purpose"] != "temporary_candidate_active_evidence_collection_only":
        raise AuthorizationError("authorization purpose differs")
    if (
        document["required_final_route"]
        != "restore_exact_aq4_then_bundle_v2_activation"
    ):
        raise AuthorizationError("authorization final route differs")

    source = _exact_object(document["source"], SOURCE_FIELDS, "source")
    _git_object(source["commit"], "source.commit")
    _git_object(source["tree"], "source.tree")

    before = _exact_object(document["before"], BEFORE_FIELDS, "before")
    if before["model_id"] != "ullm-qwen3.5-9b-aq4" or before["format_id"] != "AQ4_0":
        raise AuthorizationError("authorization before identity is not AQ4_0")
    _hash(before["manifest_sha256"], "before.manifest_sha256")
    _hash(before["worker_binary_sha256"], "before.worker_binary_sha256")
    _git_object(before["promotion_source_commit"], "before.promotion_source_commit")

    candidate = _exact_object(
        document["candidate"], CANDIDATE_FIELDS, "candidate"
    )
    if (
        candidate["model_id"] != "ullm-qwen3-14b-sq8"
        or candidate["format_id"] != "SQ8_0"
        or candidate["worker_protocol"] != "ullm.worker.v2"
    ):
        raise AuthorizationError("authorization candidate identity is not SQ8_0 v2")
    _hash(candidate["manifest_sha256"], "candidate.manifest_sha256")
    _hash(
        candidate["worker_binary_sha256"], "candidate.worker_binary_sha256"
    )
    _git_object(
        candidate["promotion_source_commit"],
        "candidate.promotion_source_commit",
    )
    _hash(
        candidate["promotion_receipt_sha256"],
        "candidate.promotion_receipt_sha256",
    )
    if source["commit"] != candidate["promotion_source_commit"]:
        raise AuthorizationError("authorization source/candidate commit differs")
    if before["manifest_sha256"] == candidate["manifest_sha256"]:
        raise AuthorizationError("authorization before and candidate manifests are equal")

    campaigns = _exact_object(document["campaigns"], CAMPAIGN_FIELDS, "campaigns")
    final_paths: set[Path] = set()
    run_ids: set[str] = set()
    for name in sorted(CAMPAIGN_FIELDS):
        campaign = _exact_object(
            campaigns[name], CAMPAIGN_IDENTITY_FIELDS, f"campaigns.{name}"
        )
        run_id = _identifier(campaign["run_id"], f"campaigns.{name}.run_id")
        final_path = _absolute_bound_path(
            campaign["final_path"],
            f"campaigns.{name}.final_path",
            require_fresh=require_fresh_outputs,
        )
        if run_id in run_ids or final_path in final_paths:
            raise AuthorizationError("campaign run IDs and final paths must be distinct")
        run_ids.add(run_id)
        final_paths.add(final_path)

    rollback = _exact_object(document["rollback"], ROLLBACK_FIELDS, "rollback")
    backup_path = _absolute_bound_path(
        rollback["backup_path"],
        "rollback.backup_path",
        require_fresh=require_fresh_outputs,
    )
    if backup_path in final_paths:
        raise AuthorizationError("rollback backup collides with a campaign output")
    _hash(rollback["systemd_unit_sha256"], "rollback.systemd_unit_sha256")
    _hash(rollback["environment_sha256"], "rollback.environment_sha256")

    prior_outcome = document["prior_outcome"]
    if prior_outcome is not None:
        if validate_prior_outcome:
            _validate_outcome_reference(
                prior_outcome, "prior_outcome", required_uid=required_uid
            )
        else:
            reference = _exact_object(
                prior_outcome, PRIOR_OUTCOME_FIELDS, "prior_outcome"
            )
            if not isinstance(reference["path"], str):
                raise AuthorizationError("prior_outcome.path is invalid")
            _hash(reference["sha256"], "prior_outcome.sha256")
    return issued_at, expires_at


def load_authorization(
    path: Path,
    *,
    now: datetime,
    policy: RegistryPolicy = RegistryPolicy(),
    require_fresh_outputs: bool = True,
) -> AuthorizationRecord:
    snapshot = _stable_read(
        path,
        "campaign authorization",
        required_mode=0o444,
        required_uid=policy.required_uid,
        required_nlink=1,
    )
    document = strict_json_bytes(snapshot.raw, "campaign authorization")
    if canonical_json_bytes(document) != snapshot.raw:
        raise AuthorizationError("campaign authorization is not canonical JSON")
    issued_at, expires_at = validate_authorization_document(
        document,
        now=now,
        required_uid=policy.required_uid,
        require_fresh_outputs=require_fresh_outputs,
    )
    return AuthorizationRecord(snapshot, document, issued_at, expires_at)


def _validate_registry(path: Path, label: str, *, required_uid: int) -> Path:
    _reject_symlink_components(path, label, leaf_may_absent=False)
    try:
        metadata = path.stat()
    except OSError as error:
        raise AuthorizationError(f"{label} is unavailable") from error
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != required_uid
        or stat.S_IMODE(metadata.st_mode) & 0o022
    ):
        raise AuthorizationError(f"{label} metadata is unsafe")
    return path.resolve(strict=True)


def claim_path(
    authorization_sha256: str,
    *,
    policy: RegistryPolicy = RegistryPolicy(),
) -> Path:
    _hash(authorization_sha256, "authorization_sha256")
    return policy.claim_registry / f"{authorization_sha256}.claim.json"


def outcome_path(
    authorization_sha256: str,
    *,
    policy: RegistryPolicy = RegistryPolicy(),
) -> Path:
    _hash(authorization_sha256, "authorization_sha256")
    return policy.outcome_registry / f"{authorization_sha256}.outcome.json"


def _publish_no_replace(
    path: Path,
    raw: bytes,
    *,
    mode: int,
    required_uid: int,
    label: str,
) -> FileSnapshot:
    if os.geteuid() != required_uid:
        raise AuthorizationError(f"{label} publisher has the wrong effective UID")
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    parent = _validate_registry(path.parent, f"{label} directory", required_uid=required_uid)
    descriptor, temporary_raw = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=parent
    )
    temporary = Path(temporary_raw)
    published = False
    try:
        os.fchmod(descriptor, mode)
        view = memoryview(raw)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise AuthorizationError(f"{label} write made no progress")
            view = view[written:]
        os.fsync(descriptor)
        try:
            os.link(temporary, path)
        except FileExistsError:
            raise
        published = True
        temporary.unlink()
        directory_descriptor = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
        snapshot = _stable_read(
            path,
            label,
            required_mode=mode,
            required_uid=required_uid,
            required_nlink=1,
        )
        if snapshot.raw != raw:
            raise AuthorizationError(f"{label} bytes differ after publication")
        return snapshot
    except BaseException:
        temporary.unlink(missing_ok=True)
        # Once the destination link exists it is never removed here: publication
        # is the durable consume boundary even if a later verification fails.
        if published:
            pass
        raise
    finally:
        os.close(descriptor)


def issue_authorization(
    document: dict[str, Any],
    output: Path,
    *,
    now: datetime,
    policy: RegistryPolicy = RegistryPolicy(),
) -> AuthorizationRecord:
    validate_authorization_document(
        document,
        now=now,
        required_uid=policy.required_uid,
    )
    raw = canonical_json_bytes(document)
    if len(raw) > MAX_DOCUMENT_BYTES:
        raise AuthorizationError("campaign authorization exceeds its size bound")
    snapshot = _publish_no_replace(
        output,
        raw,
        mode=0o444,
        required_uid=policy.required_uid,
        label="campaign authorization",
    )
    return load_authorization(snapshot.path, now=now, policy=policy)


def claim_authorization(
    authorization_path: Path,
    *,
    now: datetime,
    policy: RegistryPolicy = RegistryPolicy(),
) -> ClaimRecord:
    authorization = load_authorization(
        authorization_path,
        now=now,
        policy=policy,
        require_fresh_outputs=True,
    )
    registry = _validate_registry(
        policy.claim_registry,
        "campaign claim registry",
        required_uid=policy.required_uid,
    )
    destination = registry / f"{authorization.snapshot.sha256}.claim.json"
    document = {
        "schema_version": CLAIM_SCHEMA,
        "authorization_id": authorization.document["authorization_id"],
        "authorization_path": os.fspath(authorization.snapshot.path),
        "authorization_sha256": authorization.snapshot.sha256,
        "claimed_at": utc_timestamp(now),
        "attempt": 1,
        "max_attempts": 1,
    }
    raw = canonical_json_bytes(document)
    try:
        snapshot = _publish_no_replace(
            destination,
            raw,
            mode=0o444,
            required_uid=policy.required_uid,
            label="campaign authorization claim",
        )
    except FileExistsError as error:
        raise AuthorizationConsumed("campaign authorization is already consumed") from error
    return ClaimRecord(snapshot, document, authorization)


def load_claim(
    authorization_path: Path,
    *,
    now: datetime,
    policy: RegistryPolicy = RegistryPolicy(),
) -> ClaimRecord:
    authorization = load_authorization(
        authorization_path,
        now=now,
        policy=policy,
        require_fresh_outputs=False,
    )
    expected = claim_path(authorization.snapshot.sha256, policy=policy)
    snapshot = _stable_read(
        expected,
        "campaign authorization claim",
        required_mode=0o444,
        required_uid=policy.required_uid,
        required_nlink=1,
    )
    document = strict_json_bytes(snapshot.raw, "campaign authorization claim")
    if canonical_json_bytes(document) != snapshot.raw:
        raise AuthorizationError("campaign authorization claim is not canonical JSON")
    _exact_object(document, CLAIM_FIELDS, "campaign authorization claim")
    if (
        document["schema_version"] != CLAIM_SCHEMA
        or document["authorization_id"]
        != authorization.document["authorization_id"]
        or document["authorization_path"]
        != os.fspath(authorization.snapshot.path)
        or document["authorization_sha256"] != authorization.snapshot.sha256
        or document["attempt"] != 1
        or document["max_attempts"] != 1
    ):
        raise AuthorizationError("campaign authorization claim identity differs")
    claimed_at = _timestamp(document["claimed_at"], "claim.claimed_at")
    if claimed_at < authorization.issued_at or claimed_at >= authorization.expires_at:
        raise AuthorizationError("campaign authorization claim time is out of range")
    return ClaimRecord(snapshot, document, authorization)


def require_window_binding(
    claim: ClaimRecord,
    *,
    source_commit: str,
    source_tree: str,
    before_manifest_sha256: str,
    candidate_manifest_sha256: str,
    candidate_worker_binary_sha256: str,
    candidate_promotion_receipt_sha256: str,
    rollback_backup_path: Path,
) -> None:
    """Bind an operational window to every authorization-owned identity."""

    authorization = claim.authorization.document
    source = authorization["source"]
    before = authorization["before"]
    candidate = authorization["candidate"]
    rollback = authorization["rollback"]
    if (
        source_commit != source["commit"]
        or source_tree != source["tree"]
        or before_manifest_sha256 != before["manifest_sha256"]
        or candidate_manifest_sha256 != candidate["manifest_sha256"]
        or candidate_worker_binary_sha256 != candidate["worker_binary_sha256"]
        or candidate_promotion_receipt_sha256
        != candidate["promotion_receipt_sha256"]
        or os.fspath(rollback_backup_path)
        != os.fspath(Path(rollback["backup_path"]))
    ):
        raise AuthorizationError("campaign window identity differs from authorization")


def require_campaign_binding(
    claim: ClaimRecord,
    *,
    campaign_name: str,
    run_id: str,
    final_path: Path,
) -> None:
    """Bind one campaign invocation to its reviewed run and output identity."""

    if campaign_name not in CAMPAIGN_FIELDS:
        raise AuthorizationError("campaign name is not authorized")
    campaign = claim.authorization.document["campaigns"][campaign_name]
    if (
        run_id != campaign["run_id"]
        or os.fspath(final_path) != os.fspath(Path(campaign["final_path"]))
    ):
        raise AuthorizationError("campaign run/output identity differs from authorization")
