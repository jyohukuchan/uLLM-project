#!/usr/bin/env python3
"""Verify pinned model acquisition files and freeze exact SHA-256 provenance."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_path = args.source_manifest.expanduser().resolve()
    manifest = json.loads(source_path.read_text(encoding="utf-8"))
    counted = 0
    verified_files = 0
    for artifact in manifest["artifacts"]:
        artifact_bytes = 0
        for item in artifact["files"]:
            path = Path(item["path"]).expanduser().resolve()
            if not path.is_file():
                raise SystemExit(f"missing provenance file: {path}")
            size = path.stat().st_size
            if size != int(item["size_bytes"]):
                raise SystemExit(f"size mismatch: {path}: {size} != {item['size_bytes']}")
            actual = sha256_file(path)
            if actual != item["file_sha256"]:
                raise SystemExit(f"SHA-256 mismatch: {path}: {actual} != {item['file_sha256']}")
            if item.get("lfs_sha256") is not None and actual != item["lfs_sha256"]:
                raise SystemExit(f"LFS SHA mismatch: {path}")
            item["path"] = str(path)
            item["verified_file_sha256"] = actual
            artifact_bytes += size
            verified_files += 1
        artifact["verified_file_bytes"] = artifact_bytes
        if artifact.get("counted_in_2026_07_21_download_budget"):
            counted += int(artifact["download_bytes"])
    incidental = int(manifest.get("incidental_metadata_download_bytes", 0))
    counted += incidental
    budget = int(manifest["download_budget_bytes"])
    if counted > budget:
        raise SystemExit(f"download budget exceeded: {counted} > {budget}")
    manifest.update(
        {
            "schema_version": "importance-score-model-provenance-v0.1",
            "verified_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "source_manifest_path": str(source_path),
            "source_manifest_sha256": sha256_file(source_path),
            "verified_file_count": verified_files,
            "cumulative_download_bytes": counted,
            "cumulative_download_gb_decimal": counted / 1e9,
            "cumulative_download_gib": counted / (2**30),
            "remaining_budget_bytes": budget - counted,
            "status": "all listed local files match pinned file SHA-256 and LFS SHA where applicable",
        }
    )
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "verified_files": verified_files,
                "cumulative_download_bytes": counted,
                "remaining_budget_bytes": budget - counted,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
