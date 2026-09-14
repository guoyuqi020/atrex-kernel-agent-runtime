"""Evidence selection is explicit; lifecycle closure never certifies an interpretation."""

from __future__ import annotations

import json
from typing import Any

import pytest
from test_gateway_journal_adoption import _History
from test_gateway_journal_adoption import history as history
from test_gateway_proxy import _request

from atrex_runtime.gateway.proxy import GatewayAdapterResult
from atrex_runtime.workers.attempt_report import AttemptExperimentV8


@pytest.mark.anyio
@pytest.mark.parametrize("operation_status", ["completed", "failed", "cancelled"])
@pytest.mark.parametrize("judgment", ["unresolved", "supported", "refuted"])
async def test_diagnostic_result_and_hypothesis_judgment_are_independent(
    history: _History,
    operation_status: str,
    judgment: str,
) -> None:
    direction = await history.start()
    history.adapter.result = GatewayAdapterResult(
        status=operation_status,
        result={"correct": False, "diagnostic": "compiler/check result"},
    )
    request = json.loads(_request(history.current))
    request.update(operation="check", idempotency_key="diagnostic")
    result = await history.service.execute(history.capability.token, json.dumps(request).encode())
    receipt = (
        await history.journal(
            "experiment_record",
            request={
                **history.experiment(direction, action="abandon_direction"),
                "before": None,
                "after": {"result_artifact_digest": result.result_artifact_digest},
            },
        )
    ).result
    closing = {
        "action": "abandon",
        "direction_id": direction,
        "analysis": "The tested claim",
        "hypothesis_status": judgment,
        "supporting_experiment_ids": [receipt["experiment_id"]],
    }
    events = history.control.list_direction_events(history.current.id)
    if operation_status != "completed" and judgment != "unresolved":
        with pytest.raises(ValueError, match="completed Gateway observation"):
            await history.journal("direction_update", request=closing)
        assert history.control.list_direction_events(history.current.id) == events
        return
    await history.journal("direction_update", request=closing)
    loaded = (await history.journal("direction_load", direction_id=direction)).result
    assert loaded["hypothesis_status"] == judgment
    assert loaded["supporting_experiment_ids"] == [receipt["experiment_id"]]
    await history.journal(
        "direction_update",
        request={
            "action": "start",
            "direction_id": direction,
            "analysis": "Revisit the evidence",
        },
    )
    loaded = (await history.journal("direction_load", direction_id=direction)).result
    assert loaded["hypothesis_status"] == "unresolved"
    assert loaded["supporting_experiment_ids"] == []
    assert loaded["associated_experiment_ids"] == [receipt["experiment_id"]]
    assert (
        history.control.list_direction_events(history.current.id)[-2]["hypothesis_status"]
        == judgment
    )


@pytest.mark.anyio
@pytest.mark.parametrize("invalid", ["missing", "duplicate", "invisible"])
async def test_rejected_support_selection_does_not_append(history: _History, invalid: str) -> None:
    direction = await history.start()
    receipt = (
        await history.journal("experiment_record", request=history.experiment(direction))
    ).result
    fields: dict[str, Any] = {
        "action": "complete",
        "direction_id": direction,
        "analysis": "Close",
        "hypothesis_status": "supported",
        "supporting_experiment_ids": [receipt["experiment_id"]],
    }
    if invalid == "missing":
        fields.pop("hypothesis_status")
    elif invalid == "duplicate":
        fields["supporting_experiment_ids"] *= 2
    else:
        fields["supporting_experiment_ids"] = ["experiment_" + "f" * 32]
    events = history.control.list_direction_events(history.current.id)
    # Use the raw endpoint, not a scenario helper supplying closure fields.
    with pytest.raises(ValueError):
        await history.service.execute(
            history.capability.token,
            json.dumps(
                {
                    "schema_version": 2,
                    "operation": "direction_update",
                    "attempt_id": history.current.id,
                    "idempotency_key": "invalid-close",
                    "request": fields,
                }
            ).encode(),
            operation_scope="journal",
        )
    assert history.control.list_direction_events(history.current.id) == events


@pytest.mark.anyio
async def test_legacy_unmeasured_notes_are_readable_but_cannot_support_new_closure(
    history: _History,
) -> None:
    direction = await history.start()
    legacy = {
        **history.experiment(direction, action="abandon_direction"),
        "before": None,
        "after": None,
        "sequence": 1,
        "experiment_id": "experiment_" + "d" * 32,
        "recorded_at": "2026-01-01T00:00:00Z",
    }
    with pytest.raises(ValueError, match="cannot both be null"):
        AttemptExperimentV8.model_validate(legacy)
    AttemptExperimentV8.model_validate(legacy, context={"trusted_experiment_history": True})
    history.control.append_experiment(history.current.id, "legacy", legacy, recovery_generation=0)
    loaded = (await history.journal("direction_load", direction_id=direction)).result
    assert loaded["associated_experiment_ids"] == [legacy["experiment_id"]]
    with pytest.raises(ValueError, match="historical unmeasured notes"):
        await history.journal(
            "direction_update",
            request={
                "action": "abandon",
                "direction_id": direction,
                "analysis": "Cannot certify this note",
                "hypothesis_status": "unresolved",
                "supporting_experiment_ids": [legacy["experiment_id"]],
            },
        )
