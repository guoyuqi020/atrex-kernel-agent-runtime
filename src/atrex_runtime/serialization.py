"""Canonical JSON encoding shared by persisted protocols and digest boundaries."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .domain.ids import ArtifactDigest, parse_artifact_digest


def canonical_json_text(value: object) -> str:
    """Encode deterministic, finite, compact JSON without ASCII escaping."""
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def canonical_json_bytes(value: object) -> bytes:
    """Encode canonical JSON as UTF-8 bytes."""
    return canonical_json_text(value).encode()


def canonical_json_digest(value: object) -> ArtifactDigest:
    """Return the Runtime Artifact-digest spelling for canonical JSON bytes."""
    hexdigest = hashlib.sha256(canonical_json_bytes(value)).hexdigest()
    return parse_artifact_digest(f"sha256:{hexdigest}")


def write_canonical_json(path: Path, value: object) -> None:
    """Create parent directories and write canonical JSON bytes."""
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.write_bytes(canonical_json_bytes(value))


__all__ = [
    "canonical_json_bytes",
    "canonical_json_digest",
    "canonical_json_text",
    "write_canonical_json",
]
