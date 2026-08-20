"""Canonical Session Trace retention policy applied at Runtime sealing boundaries."""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path
from uuid import uuid4

CLAUDE_THINKING_TOKEN_FILTER = "system/thinking_tokens"


def enforce_session_trace_retention(trace: Path) -> int:
    """Remove non-authoritative high-frequency telemetry before Artifact sealing.

    Provider usage remains authoritative in the normalized usage ledger. This policy also makes
    Runtime compatible with older Agent Bundles that captured Claude token-estimate events.
    """
    removed = 0
    for relative in ("provider/stdout.stream-json", "conversation.jsonl"):
        path = trace / relative
        if not path.exists():
            continue
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Session telemetry source must be a regular file: {relative}")
        original = path.read_bytes()
        filtered, file_removed = retained_session_file(relative, original)
        removed += file_removed
        if filtered != original:
            _atomic_replace(path, filtered)
    _declare_filter(trace / "session.json")
    return removed


def retained_session_file(relative: str, payload: bytes) -> tuple[bytes, int]:
    """Apply the retention policy to one named Session file without mutating it."""
    if relative not in {"provider/stdout.stream-json", "conversation.jsonl"}:
        return payload, 0
    lines = payload.splitlines(keepends=True)
    retained = [
        line
        for line in lines
        if not _is_claude_thinking_token_line(
            line,
            conversation=relative == "conversation.jsonl",
        )
    ]
    return b"".join(retained), len(lines) - len(retained)


def _is_claude_thinking_token_line(line: bytes, *, conversation: bool) -> bool:
    try:
        value = json.loads(line)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return False
    if not isinstance(value, dict):
        return False
    event: object = value.get("event") if conversation else value
    return (
        isinstance(event, dict)
        and event.get("type") == "system"
        and event.get("subtype") == "thinking_tokens"
    )


def _declare_filter(metadata: Path) -> None:
    if not metadata.exists():
        return
    if metadata.is_symlink() or not metadata.is_file():
        raise ValueError("Session metadata must be a regular file")
    try:
        value = json.loads(metadata.read_bytes())
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Session metadata must contain valid JSON") from error
    if not isinstance(value, dict):
        raise ValueError("Session metadata must contain a JSON object")
    raw_filters = value.get("provider_event_filters", [])
    if not isinstance(raw_filters, list) or not all(isinstance(item, str) for item in raw_filters):
        raise ValueError("Session provider_event_filters must be a string list")
    filters = list(dict.fromkeys((*raw_filters, CLAUDE_THINKING_TOKEN_FILTER)))
    if filters == raw_filters:
        return
    value["provider_event_filters"] = filters
    _atomic_replace(
        metadata,
        (json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True) + "\n").encode(),
    )


def _atomic_replace(path: Path, payload: bytes) -> None:
    mode = stat.S_IMODE(path.stat().st_mode)
    temporary = path.with_name(f".{path.name}.atrex-{uuid4().hex}.tmp")
    try:
        temporary.write_bytes(payload)
        temporary.chmod(mode)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


__all__ = [
    "CLAUDE_THINKING_TOKEN_FILTER",
    "enforce_session_trace_retention",
    "retained_session_file",
]
