#!/usr/bin/env python3
"""Consume one pre-issued cross-model campaign authorization exactly once."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from served_model_campaign_authorization import (
    AuthorizationError,
    claim_authorization,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorization", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        record = claim_authorization(
            args.authorization,
            now=datetime.now(timezone.utc),
        )
    except AuthorizationError:
        print("campaign authorization claim failed", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "schema_version": record.document["schema_version"],
                "authorization_id": record.document["authorization_id"],
                "authorization_sha256": record.authorization.snapshot.sha256,
                "claim_sha256": record.snapshot.sha256,
                "claim_path": str(record.snapshot.path),
            },
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
