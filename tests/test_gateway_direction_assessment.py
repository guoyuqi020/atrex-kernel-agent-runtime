"""Evidence selection is explicit; lifecycle closure never certifies an interpretation."""

from __future__ import annotations

import json
from datetime import timedelta
from typing import Any

import pytest
from conftest import digest
from test_attempt_report import _value
from test_gateway_journal_adoption import _History
from test_gateway_journal_adoption import history as history
from test_gateway_proxy import NOW_DATETIME, _request

from atrex_runtime.domain.errors import SuggestedDirectionTransitionError
from atrex_runtime.domain.ids import new_attempt_id, parse_epoch_id
from atrex_runtime.gateway import GatewayCapabilityPolicy, GatewayOperation
from atrex_runtime.gateway.control import BootstrapGatewaySubject, BootstrapRunStatus
from atrex_runtime.gateway.proxy import GatewayAdapterResult
from atrex_runtime.workers.attempt_report import AttemptExperimentV8, AttemptReportV12


@pytest.mark.anyio
async def test_optimizer_report_cannot_submit_suggested_direction(history: _History) -> None:
    value = _value(str(history.current.id))
    events = value["direction_events"]
    assert isinstance(events, list)
    assert isinstance(events[0], dict)
    events[0]["action"] = "suggest"
    report = AttemptReportV12.model_validate(value)

    with pytest.raises(ValueError, match="only during Bootstrap"):
        history.service._journals.validate_report_journal(report)


@pytest.mark.anyio
async def test_bootstrap_suggestion_is_visible_but_not_startable_in_optimizer(
    history: _History,
) -> None:
    epoch = history.registry.get_epoch(history.current.epoch_id)
    lineage = history.registry.get_lineage(epoch.lineage_id)
    campaign = history.registry.get_campaign(lineage.campaign_id)
    bootstrap_id = new_attempt_id()
    capability = history.control.issue_bootstrap(
        BootstrapGatewaySubject(
            attempt_id=bootstrap_id,
            campaign_id=campaign.id,
            lineage_id=lineage.id,
            epoch_id=parse_epoch_id("epoch_" + str(bootstrap_id).removeprefix("attempt_")),
            kernel_agent_revision_id=history.current.kernel_agent_revision_id,
            operator=campaign.operator,
            hardware_target=campaign.hardware_target,
            dsl=lineage.dsl,
            evaluation_contract_digest=campaign.evaluation_contract_digest,
            input_kernel_digest=digest("bootstrap-input"),
            evidence_digest=digest("bootstrap-evidence"),
            created_at=NOW_DATETIME,
        ),
        GatewayCapabilityPolicy(
            frozenset({GatewayOperation.DIRECTION_UPDATE}),
            2,
            NOW_DATETIME + timedelta(hours=1),
        ),
    )
    suggestion = {
        "action": "suggest",
        "name": "Test a fused load schedule",
        "hypothesis": "Fusing loads may reduce memory traffic",
        "rationale": "Bootstrap found repeated loads but did not test this change",
        "plan": ["Implement fused loads", "Evaluate the new Kernel"],
        "success_criteria": "Correct and faster",
        "stop_conditions": "Incorrect or no measurable gain",
    }
    response = await history.service.execute(
        capability.token,
        json.dumps(
            {
                "schema_version": 2,
                "attempt_id": bootstrap_id,
                "idempotency_key": "bootstrap-suggestion",
                "operation": "direction_update",
                "request": suggestion,
            }
        ).encode(),
        operation_scope="journal",
    )
    suggested_id = response.result["direction_id"]
    history.control.finish_bootstrap_run(
        bootstrap_id,
        0,
        status=BootstrapRunStatus.COMPLETED,
        finish_reason="completed",
        failure_reason=None,
    )
    loaded = (await history.journal("direction_load", direction_id=suggested_id)).result
    assert loaded["status"] == "suggested"
    assert loaded["hypothesis"] == suggestion["hypothesis"]
    with pytest.raises(SuggestedDirectionTransitionError) as rejected:
        await history.journal(
            "direction_update",
            request={"action": "start", "direction_id": suggested_id, "analysis": "Try it"},
        )
    assert rejected.value.direction_id == suggested_id
    assert rejected.value.action == "start"
    with pytest.raises(ValueError, match="only during Bootstrap"):
        await history.journal("direction_update", request=suggestion)
    child = await history.journal(
        "direction_update",
        request={
            **suggestion,
            "action": "propose",
            "relationship": "adoption",
            "derived_from_direction_ids": [suggested_id],
        },
    )
    assert child.result["direction_id"] != suggested_id


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


@pytest.mark.anyio
async def test_optimizer_derives_from_suggested_direction_without_mutating_it(
    history: _History,
) -> None:
    epoch_id = history.current.epoch_id
    # The shared fixture begins in RUNNING; simulate the earlier Evolver-build phase
    # while recording its immutable suggestion, then restore the Attempt phase.
    history.registry._connection.execute(
        "UPDATE epochs SET status = 'building_challenger' WHERE id = ?", (epoch_id,)
    )
    suggestion = {
        "name": "Test a fused load schedule",
        "hypothesis": "Fusing loads may reduce traffic",
        "rationale": "Two prior approaches suggest the same bottleneck",
        "plan": ["Implement and evaluate"],
        "success_criteria": "Correct and faster",
        "stop_conditions": "Incorrect or no gain",
    }
    recorded = history.registry.record_epoch_suggested_directions(
        epoch_id,
        f"epoch:{epoch_id}:challenger:1",
        digest("evolver-direction-trace"),
        (suggestion,),
    )
    history.registry._connection.execute(
        "UPDATE epochs SET status = 'running' WHERE id = ?", (epoch_id,)
    )
    suggested_id = recorded[0]["direction_id"]
    loaded_suggestion = (await history.journal("direction_load", direction_id=suggested_id)).result
    assert loaded_suggestion["status"] == "suggested"
    with pytest.raises(ValueError, match="suggested Direction cannot"):
        await history.journal(
            "direction_update",
            request={"action": "start", "direction_id": suggested_id, "analysis": "Try it"},
        )
    with pytest.raises(ValueError, match="current status is suggested"):
        await history.journal("experiment_record", request=history.experiment(suggested_id))
    request = {
        "action": "propose",
        **suggestion,
        "relationship": "adoption",
        "derived_from_direction_ids": [suggested_id],
    }
    with pytest.raises(ValueError, match="not visible"):
        await history.journal(
            "direction_update",
            request={**request, "derived_from_direction_ids": ["direction_" + "f" * 32]},
        )
    response = await history.journal("direction_update", request=request)
    direction_id = response.result["direction_id"]
    loaded = (await history.journal("direction_load", direction_id=direction_id)).result
    assert loaded["relationship"] == "adoption"
    assert loaded["derived_from_direction_ids"] == [suggested_id]
    refined = await history.journal(
        "direction_update",
        request={
            **request,
            "name": "Test a staged load schedule",
            "hypothesis": "Staging loads may reduce traffic further",
            "relationship": "refinement",
        },
    )
    assert refined.result["direction_id"] != direction_id
    assert (
        await history.journal("direction_load", direction_id=suggested_id)
    ).result == loaded_suggestion


@pytest.mark.anyio
async def test_expired_suggestion_is_readable_but_requires_refinement(
    history: _History, monkeypatch: pytest.MonkeyPatch
) -> None:
    suggestion = {
        "direction_id": "direction_" + "e" * 32,
        "name": "Revisit staged loads",
        "hypothesis": "Staging might shorten reads",
        "rationale": "An older untested proposal",
        "plan": ["Implement", "Evaluate"],
        "success_criteria": "Correct and faster",
        "stop_conditions": "No gain",
        "status": "expired",
    }
    monkeypatch.setattr(
        history.control, "visible_suggested_directions", lambda _attempt_id: (suggestion,)
    )
    loaded = (
        await history.journal("direction_load", direction_id=suggestion["direction_id"])
    ).result
    assert loaded["status"] == "expired"
    with pytest.raises(SuggestedDirectionTransitionError, match="expired Direction"):
        await history.journal(
            "direction_update",
            request={
                "action": "start",
                "direction_id": suggestion["direction_id"],
                "analysis": "Try",
            },
        )
    proposal = {
        "action": "propose",
        **{
            key: value for key, value in suggestion.items() if key not in {"direction_id", "status"}
        },
        "derived_from_direction_ids": [suggestion["direction_id"]],
    }
    with pytest.raises(ValueError, match="unexpired suggested parent"):
        await history.journal("direction_update", request={**proposal, "relationship": "adoption"})
    response = await history.journal(
        "direction_update", request={**proposal, "relationship": "refinement"}
    )
    assert response.result["direction_id"] != suggestion["direction_id"]
