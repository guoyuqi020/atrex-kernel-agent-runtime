"""Tests for the authoritative Runtime Session retention boundary."""

from __future__ import annotations

import json
from pathlib import Path

from atrex_runtime.workers.session_trace import enforce_session_trace_retention


def _line(value: object) -> str:
    return json.dumps(value) + "\n"


def test_retention_removes_only_claude_thinking_token_estimates(tmp_path: Path) -> None:
    trace = tmp_path / "trace"
    (trace / "provider").mkdir(parents=True)
    thinking = {
        "type": "system",
        "subtype": "thinking_tokens",
        "estimated_tokens": 121,
    }
    result = {"type": "result", "usage": {"input_tokens": 10, "output_tokens": 20}}
    (trace / "provider/stdout.stream-json").write_text(
        _line(thinking) + _line(result),
        encoding="utf-8",
    )
    (trace / "conversation.jsonl").write_text(
        _line({"type": "provider_event", "event": thinking})
        + _line({"type": "provider_event", "event": result}),
        encoding="utf-8",
    )
    (trace / "events.jsonl").write_text(
        _line({"kind": "usage", "total_tokens": 30}),
        encoding="utf-8",
    )
    (trace / "session.json").write_text(
        json.dumps({"schema_version": 1, "provider_event_filters": ["existing/filter"]}),
        encoding="utf-8",
    )

    assert enforce_session_trace_retention(trace) == 2
    assert "thinking_tokens" not in (trace / "provider/stdout.stream-json").read_text()
    assert "thinking_tokens" not in (trace / "conversation.jsonl").read_text()
    assert json.loads((trace / "provider/stdout.stream-json").read_text()) == result
    assert "total_tokens" in (trace / "events.jsonl").read_text()
    metadata = json.loads((trace / "session.json").read_text())
    assert metadata["provider_event_filters"] == [
        "existing/filter",
        "system/thinking_tokens",
    ]
    assert enforce_session_trace_retention(trace) == 0
