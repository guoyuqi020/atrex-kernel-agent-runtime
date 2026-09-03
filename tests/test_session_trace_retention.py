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


def test_retention_preserves_native_response_usage_and_attribution(tmp_path: Path) -> None:
    trace = tmp_path / "trace"
    (trace / "provider/claude-subagents").mkdir(parents=True)
    response = {
        "type": "assistant",
        "message": {"id": "message-1", "usage": {"input_tokens": 3, "output_tokens": 7}},
    }
    paths = ("provider/claude-session.raw-jsonl", "provider/claude-subagents/agent-child.jsonl")
    for path in paths:
        (trace / path).write_text(_line(response))
    (trace / "conversation.jsonl").write_text(
        "".join(
            _line({"type": "provider_event", "path": path, "event": response}) for path in paths
        )
    )
    event = {
        "kind": "usage_delta",
        "message_id": "message-1",
        "source_path": paths[0],
        "usage": {"output_tokens": 7},
    }
    (trace / "events.jsonl").write_text(_line(event))
    (trace / "session.json").write_text(json.dumps({"response_usage_complete": True}))

    assert enforce_session_trace_retention(trace) == 0
    for path in paths:
        assert (trace / path).read_text() == _line(response)
    assert json.loads((trace / "events.jsonl").read_text()) == event
    assert json.loads((trace / "session.json").read_text())["response_usage_complete"] is True
