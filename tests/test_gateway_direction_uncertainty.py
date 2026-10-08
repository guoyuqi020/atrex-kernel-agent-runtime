"""Uncertainty can end work without becoming unsupported shared knowledge."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import pytest
from conftest import digest
from test_attempt_report import _value
from test_gateway_journal_adoption import _History
from test_gateway_journal_adoption import history as history
from test_gateway_proxy import _request

from atrex_runtime.gateway.proxy import GatewayAdapterResult
from atrex_runtime.workers.attempt_report import AttemptReportV12


def _modules(history: _History, experiments_enabled: bool) -> None:
    lineage = history.registry.get_epoch(history.current.epoch_id).lineage_id
    modules = ["directions", "experiments"] if experiments_enabled else ["directions"]
    history.registry._connection.execute(
        "UPDATE lineages SET tool_modules_json = ? WHERE id = ?", (json.dumps(modules), lineage)
    )


async def _close(history: _History, direction: str, **fields: object) -> dict[str, Any]:
    history.sequence += 1
    request = {
        "action": "defer",
        "direction_id": direction,
        "analysis": "Source inspection is only a reason to postpone this experiment",
        "hypothesis_status": "unresolved",
        **fields,
    }
    if "experiments" in history.control.tool_modules_for_attempt(
        history.current.id, ("directions", "experiments")
    ):
        request.setdefault("supporting_experiment_ids", [])
    response = await history.service.execute(
        history.capability.token,
        json.dumps(
            {
                "schema_version": 2,
                "attempt_id": history.current.id,
                "idempotency_key": f"uncertain-close-{history.sequence}",
                "operation": "direction_update",
                "request": request,
            }
        ).encode(),
        operation_scope="journal",
    )
    return response.result


@pytest.mark.anyio
@pytest.mark.parametrize("experiments_enabled", [True, False])
@pytest.mark.parametrize("action", ["complete", "block", "defer", "abandon"])
async def test_untested_direction_can_end_without_fake_experiment(
    history: _History, experiments_enabled: bool, action: str
) -> None:
    _modules(history, experiments_enabled)
    direction = await history.start()
    receipt = await _close(history, direction, action=action)
    assert receipt["hypothesis_status"] == "unresolved"
    assert history.control.list_experiments(history.current.id) == ()
    loaded = (await history.journal("direction_load", direction_id=direction)).result
    assert loaded["hypothesis_status"] == "unresolved"
    assert loaded["supporting_results"] == []
    assert loaded["scope"] is None
    assert "Agent interpretation" in loaded["interpretation_notice"]
    assert loaded["status"] != "in_progress(self)"


@pytest.mark.anyio
@pytest.mark.parametrize("experiments_enabled", [True, False])
@pytest.mark.parametrize("judgment", ["supported", "refuted"])
async def test_unmeasured_verdict_is_downgraded_without_blocking_work(
    history: _History, experiments_enabled: bool, judgment: str
) -> None:
    _modules(history, experiments_enabled)
    direction = await history.start()
    receipt = await _close(history, direction, hypothesis_status=judgment)
    assert receipt["hypothesis_status"] == "unresolved"
    assert receipt["assessment_notes"]
    recorded = history.control.list_direction_events(history.current.id)[-1]
    assert recorded["hypothesis_status"] == "unresolved"
    listed = (await history.journal("directions_list")).result["directions"]
    assert listed[0]["hypothesis_status"] == "unresolved"
    assert "Agent interpretation" in listed[0]["interpretation_notice"]


@pytest.mark.anyio
@pytest.mark.parametrize("experiments_enabled", [True, False])
async def test_exact_visible_result_can_support_direction_without_experiment(
    history: _History, experiments_enabled: bool
) -> None:
    _modules(history, experiments_enabled)
    direction = await history.start()
    references = [
        {
            "kernel_artifact_digest": history.after.kernel_artifact_digest,
            "result_artifact_digests": [history.after.result_artifact_digest],
        }
    ]
    receipt = await _close(
        history,
        direction,
        hypothesis_status="supported",
        claim_kind="implementation_outcome",
        scope="Only this kernel and the recorded evaluation contract",
        supporting_results=references,
    )
    assert receipt["hypothesis_status"] == "supported"
    loaded = (await history.journal("direction_load", direction_id=direction)).result
    assert loaded["supporting_results"] == references
    assert loaded["claim_kind"] == "implementation_outcome"
    assert loaded["scope"] == "Only this kernel and the recorded evaluation contract"
    assert history.control.list_experiments(history.current.id) == ()


@pytest.mark.anyio
@pytest.mark.parametrize("experiments_enabled", [True, False])
@pytest.mark.parametrize("invalid", ["invisible", "mismatch"])
async def test_invalid_result_references_are_hard_errors_even_for_unresolved(
    history: _History, experiments_enabled: bool, invalid: str
) -> None:
    _modules(history, experiments_enabled)
    direction = await history.start()
    reference = {
        "kernel_artifact_digest": history.after.kernel_artifact_digest,
        "result_artifact_digests": [history.after.result_artifact_digest],
    }
    if invalid == "invisible":
        reference["result_artifact_digests"] = [digest("not-visible")]
    else:
        reference["kernel_artifact_digest"] = history.before.kernel_artifact_digest
    previous = history.control.list_direction_events(history.current.id)
    with pytest.raises(ValueError, match="visible history"):
        await _close(history, direction, supporting_results=[reference])
    assert history.control.list_direction_events(history.current.id) == previous


@pytest.mark.anyio
@pytest.mark.parametrize("claim_kind", ["observation", "causal_hypothesis"])
async def test_compile_check_cannot_certify_causal_performance_explanation(
    history: _History, claim_kind: str
) -> None:
    _modules(history, False)
    direction = await history.start()
    history.adapter.result = GatewayAdapterResult(
        status="completed", result={"correct": True, "diagnostic": "compiled"}
    )
    request = json.loads(_request(history.current))
    request.update(operation="check", idempotency_key="check-diagnostic")
    request.pop("latency_prediction")
    result = await history.service.execute(history.capability.token, json.dumps(request).encode())
    receipt = await _close(
        history,
        direction,
        hypothesis_status="supported",
        claim_kind=claim_kind,
        scope="Only the tested compiler and kernel",
        supporting_results=[
            {
                "kernel_artifact_digest": result.kernel_artifact_digest,
                "result_artifact_digests": [result.result_artifact_digest],
            }
        ],
    )
    assert receipt["hypothesis_status"] == (
        "supported" if claim_kind == "observation" else "unresolved"
    )


@pytest.mark.anyio
@pytest.mark.parametrize("experiments_enabled", [True, False])
async def test_report_only_direction_cannot_bypass_evidence_gate(
    history: _History, experiments_enabled: bool
) -> None:
    _modules(history, experiments_enabled)
    value: dict[str, Any] = _value(str(history.current.id))
    value.update(
        status="blocked",
        final_candidate=None,
        blocker="No experiment was performed",
        findings=[],
        experiments=[],
        profile_evidence=None,
        tool_modules=["directions", "experiments"] if experiments_enabled else ["directions"],
    )
    for event in value["direction_events"]:
        event["supporting_experiment_ids"] = []
    value["direction_events"][-1]["hypothesis_status"] = "refuted"
    report = AttemptReportV12.model_validate(value)
    normalized, notes = history.service._journals_for_attempt(
        history.current.id
    ).validate_report_journal(report)
    assert normalized.direction_events[-1].hypothesis_status == "unresolved"
    assert notes
    assert report.direction_events[-1].hypothesis_status == "refuted"


@pytest.mark.anyio
@pytest.mark.parametrize("judgment", [None, "supported", "refuted"])
async def test_legacy_direction_judgment_is_read_as_unresolved_without_scope(
    history: _History, judgment: str | None
) -> None:
    direction = await history.start()
    event = {
        "direction_event_id": "directionevent_" + "f" * 32,
        "direction_id": direction,
        "recorded_at": datetime.now(UTC).isoformat(),
        "action": "defer",
        "name": None,
        "hypothesis": None,
        "rationale": None,
        "plan": [],
        "success_criteria": None,
        "stop_conditions": None,
        "analysis": "A legacy free-text conclusion",
        "supporting_experiment_ids": [],
        "hypothesis_status": judgment,
    }
    history.control.append_direction_event(
        history.current.id, "legacy-assertion", event, recovery_generation=0
    )
    loaded = (await history.journal("direction_load", direction_id=direction)).result
    assert loaded["hypothesis_status"] == "unresolved"
    # The stored append-only event is preserved even though readers lower certainty.
    assert (
        history.control.list_direction_events(history.current.id)[-1]["hypothesis_status"]
        == judgment
    )


@pytest.mark.anyio
async def test_report_normalization_cannot_disguise_an_edited_live_closure(
    history: _History,
) -> None:
    direction = await history.start()
    await _close(history, direction, hypothesis_status="refuted")
    value: dict[str, Any] = _value(str(history.current.id))
    value.update(
        status="blocked",
        final_candidate=None,
        blocker="Untested",
        findings=[],
        experiments=[],
        profile_evidence=None,
        direction_events=list(history.control.list_direction_events(history.current.id)),
    )
    # Both would normalize to unresolved, but the edit must still be rejected.
    value["direction_events"][-1]["hypothesis_status"] = "supported"
    with pytest.raises(ValueError, match="must match the Runtime-owned"):
        history.service._journals_for_attempt(history.current.id).validate_report_journal(
            AttemptReportV12.model_validate(value)
        )
