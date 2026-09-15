"""Direction typo hints help recovery without crossing Journal visibility boundaries."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

import pytest
from test_gateway_journal_adoption import _History
from test_gateway_journal_adoption import history as history
from test_gateway_proxy import _service

from atrex_runtime.domain.errors import DirectionLookupError
from atrex_runtime.gateway import GatewayProxyAsgiApp, GatewayProxyLimits
from atrex_runtime.gateway.journals import RuntimeJournalService, _one_edit_apart
from atrex_runtime.gateway.proxy import _invalid_request_response


@pytest.mark.parametrize(
    ("first", "second", "expected"),
    [
        (
            "direction_d7462410254b4b8aa1ba96a0b93a3a58",
            "direction_d7462410254b4c8aa1ba96a0b93a3a58",
            True,
        ),
        ("abc", "abc", False),
        ("abc", "adc", True),
        ("abc", "ab", True),
        ("ab", "abc", True),
        ("abc", "ac", True),
        ("ac", "abc", True),
        ("abc", "xbc", True),
        ("abc", "xabc", True),
        ("abc", "abxyc", False),
        ("abc", "axd", False),
        ("abc", "acb", False),
        ("", "a", True),
        ("", "", False),
    ],
)
def test_direction_hint_requires_exactly_one_character_edit(
    first: str, second: str, expected: bool
) -> None:
    assert _one_edit_apart(first, second) is expected


def _typo(direction_id: str, edit: str = "substitution") -> str:
    if edit == "insertion":
        return direction_id + "a"
    if edit == "deletion":
        return direction_id[:-1]
    return direction_id[:-1] + ("0" if direction_id[-1] != "0" else "1")


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("operation", "edit"),
    [
        (operation, "substitution")
        for operation in ("direction_load", "direction_update", "experiment_record")
    ]
    + [
        (operation, edit)
        for operation in ("direction_update", "experiment_record")
        for edit in ("insertion", "deletion")
    ],
)
async def test_direction_typo_is_rejected_without_writes_and_can_be_corrected(
    history: _History, operation: str, edit: str
) -> None:
    direction_id = await history.start()
    bad_id = _typo(direction_id, edit)
    if operation == "direction_update":
        await history.journal("experiment_record", request=history.experiment(direction_id))
    fields: dict[str, object]
    if operation == "direction_load":
        fields = {"direction_id": bad_id}
        field_path = "direction_id"
    elif operation == "direction_update":
        fields = {
            "request": {"action": "defer", "direction_id": bad_id, "analysis": "pause exploration"}
        }
        field_path = "request.direction_id"
    else:
        fields = {"request": history.experiment(bad_id)}
        field_path = "request.direction_id"
    events_before = history.control.list_direction_events(history.current.id)
    experiments_before = history.control.list_experiments(history.current.id)
    with pytest.raises(DirectionLookupError) as rejected:
        await history.journal(operation, **fields)
    error = rejected.value
    assert error.requested_direction_id == bad_id
    assert error.suggested_direction_ids == (direction_id,)
    assert error.field_path == field_path
    assert "not found in the current Attempt's visible history" in str(error)
    assert "Did you mean" in str(error)
    assert history.control.list_direction_events(history.current.id) == events_before
    assert history.control.list_experiments(history.current.id) == experiments_before

    if operation == "direction_load":
        fields["direction_id"] = direction_id
    else:
        cast(dict[str, object], fields["request"])["direction_id"] = direction_id
    corrected = await history.journal(operation, **fields)
    if operation == "experiment_record":
        recorded = history.control.list_experiments(history.current.id)
        assert len(recorded) == 1
        assert recorded[0]["direction_id"] == direction_id
        assert recorded[0]["experiment_id"] == corrected.result["experiment_id"]
    else:
        assert corrected.result["direction_id"] == direction_id
    # Journal repairs do not repeat GPU measurements.
    assert len(history.adapter.requests) == history.initial_adapter_requests


@pytest.mark.anyio
@pytest.mark.parametrize("history", ["concurrent"], indirect=True)
async def test_near_match_hints_never_include_an_invisible_direction(history: _History) -> None:
    direction_id = await history.start()
    bad_id = _typo(direction_id)
    hidden_id = direction_id[:-1] + next(
        digit for digit in "0123456789abcdef" if digit not in {direction_id[-1], bad_id[-1]}
    )
    proposal = history.control.list_direction_events(history.current.id)[0]
    history.control.append_direction_event(
        history.historical_attempt.id,
        "invisible-near-match",
        {
            **proposal,
            "direction_id": hidden_id,
            "direction_event_id": f"directionevent_{uuid4().hex}",
        },
        recovery_generation=history.historical_attempt.recovery_generation,
    )
    assert history.control.list_direction_events(history.historical_attempt.id)
    assert _one_edit_apart(bad_id, hidden_id)
    assert hidden_id not in history.service._journals._direction_views(history.current.id)
    with pytest.raises(DirectionLookupError) as rejected:
        await history.journal("direction_load", direction_id=bad_id)
    assert rejected.value.suggested_direction_ids == (direction_id,)
    response = _invalid_request_response(
        b'{"operation":"direction_load"}', rejected.value, operation_scope="journal"
    )
    assert hidden_id not in json.dumps(response)


def test_near_match_hints_are_bounded_deterministic_and_do_not_replace_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry, control, attempt, _, service, _ = _service(tmp_path)
    try:
        requested = "direction_" + "0" * 32
        candidates = [requested[:-1] + digit for digit in "654321"]
        visible = {candidate: {"direction_id": candidate} for candidate in candidates}
        monkeypatch.setattr(RuntimeJournalService, "_direction_views", lambda self, _: visible)
        with pytest.raises(DirectionLookupError) as rejected:
            service._journals._require_direction(attempt.id, requested, field_path="direction_id")
        assert rejected.value.suggested_direction_ids == tuple(sorted(candidates)[:3])
        for candidate in candidates:
            assert (
                service._journals._require_direction(
                    attempt.id, candidate, field_path="direction_id"
                )
                is visible[candidate]
            )
        with pytest.raises(DirectionLookupError) as distant:
            service._journals._require_direction(
                attempt.id, "direction_" + "f" * 32, field_path="direction_id"
            )
        assert distant.value.suggested_direction_ids == ()
    finally:
        control.close()
        registry.close()


@pytest.mark.anyio
@pytest.mark.parametrize("near_match", [True, False])
async def test_http_lookup_error_includes_schema_and_actionable_recovery(
    history: _History, near_match: bool
) -> None:
    direction_id = await history.start()
    bad_id = _typo(direction_id) if near_match else "direction_" + "f" * 32
    body = json.dumps(
        {
            "schema_version": 2,
            "attempt_id": history.current.id,
            "idempotency_key": "bad-direction-http",
            "operation": "experiment_record",
            "request": history.experiment(bad_id),
        }
    ).encode()
    app = GatewayProxyAsgiApp(history.service, GatewayProxyLimits(64 * 1024, 8, 16 * 1024))
    sent: list[dict[str, Any]] = []

    async def receive() -> dict[str, object]:
        return {"type": "http.request", "body": body, "more_body": False}

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)

    await app(
        {
            "type": "http",
            "method": "POST",
            "path": "/v1/runtime/journals",
            "headers": [(b"authorization", f"Bearer {history.capability.token}".encode())],
        },
        receive,
        send,
    )

    assert sent[0]["status"] == 400
    response = json.loads(sent[1]["body"])
    assert response["error"] == "invalid_request"
    assert response["requested_direction_id"] == bad_id
    assert response["suggested_direction_ids"] == ([direction_id] if near_match else [])
    assert response["issues"] == [
        {
            "path": "request.direction_id",
            "code": "direction_not_found",
            "message": response["detail"],
        }
    ]
    assert set(response["request_schema"]["operations"]) == {"experiment_record"}
    recovery = response["recovery"]
    if near_match:
        assert recovery[0]["tool"] == "load-direction"
        assert recovery[0]["request"] == {"direction_id": direction_id}
    assert recovery[-2]["tool"] == "list-directions"
    assert recovery[-2]["request"] == {"file": "scratch/directions-index.json"}
    assert "correct" in recovery[-1]["instruction"].lower()
    assert history.control.list_experiments(history.current.id) == ()
