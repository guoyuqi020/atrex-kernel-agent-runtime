"""An adoption authorizes an exact nomination, never fabricated or incompatible evidence."""

from __future__ import annotations

import base64
import json
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from typing import Any, cast

import pytest
from conftest import digest
from test_attempt_report import _value
from test_gateway_finalization import (
    FakeClient,
    FakeContexts,
    FakeEvents,
    _builder,
)
from test_gateway_journal_adoption import _History
from test_gateway_journal_adoption import history as history
from test_gateway_proxy import NOW_DATETIME, _request

from atrex_runtime.artifacts.local import LocalArtifactStore
from atrex_runtime.domain.errors import DuplicateGatewayTaskError
from atrex_runtime.domain.ids import new_attempt_id, parse_artifact_digest
from atrex_runtime.domain.models import Dsl
from atrex_runtime.gateway.control import BootstrapGatewaySubject, GatewayCapabilityPolicy
from atrex_runtime.gateway.control_models import GatewayEvaluationSource, GatewayOperation
from atrex_runtime.gateway.finalization import (
    AgateAuthoritativeCandidateEvaluator,
    BootstrapEvaluationStage,
)
from atrex_runtime.gateway.protocol import EvaluationV2
from atrex_runtime.gateway.proxy import GatewayAdapterResult


def _candidate() -> dict[str, object]:
    return {
        "files": [
            {
                "path": "kernel.py",
                "content_base64": base64.b64encode(b"def kernel(): return None\n").decode(),
            }
        ]
    }


async def _adopt(history: _History) -> dict[str, Any]:
    direction = await history.start()
    await history.journal("experiment_record", request=history.experiment(direction))
    await history.journal(
        "direction_update",
        request={
            "action": "complete",
            "direction_id": direction,
            "analysis": "Adopted the exact measured source; no new measurement.",
        },
    )
    report: dict[str, Any] = _value(history.current.id)
    report.update(
        experiments=list(history.control.list_experiments(history.current.id)),
        direction_events=list(history.control.list_direction_events(history.current.id)),
        profile_evidence=None,
    )
    report["findings"][0]["supporting_experiment_ids"] = [report["experiments"][0]["experiment_id"]]
    return report


async def _submit(history: _History, report: dict[str, Any], *, key: str = "nominate") -> Any:
    return await history.service.execute(
        history.capability.token,
        json.dumps(
            {
                "schema_version": 2,
                "attempt_id": history.current.id,
                "idempotency_key": key,
                "operation": "attempt_report",
                "candidate": _candidate(),
                "report": report,
            }
        ).encode(),
        operation_scope="runtime",
    )


@pytest.mark.anyio
async def test_nomination_requires_persisted_adoption_not_report_claim(history: _History) -> None:
    forged: dict[str, Any] = _value(history.current.id)
    forged["experiments"][0]["action"] = "adopt"
    with pytest.raises(ValueError, match="record an adopt Experiment"):
        await _submit(history, forged, key="unrecorded-adoption")
    report = await _adopt(history)
    response = await _submit(history, report)
    assert response.result["status"] == "registered"
    assert history.control.list_evaluations(history.current.id) == ()
    assert len(history.adapter.requests) == history.initial_adapter_requests
    assert history.after.kernel_artifact_digest is not None
    resolved = history.control.find_candidate_evaluation(
        history.current.id,
        parse_artifact_digest(history.after.kernel_artifact_digest),
    )
    assert resolved is not None
    assert resolved.attempt_id == history.historical_attempt.id
    assert resolved == history.control.list_evaluations(history.historical_attempt.id)[-1]


@pytest.mark.anyio
async def test_adoption_does_not_cover_changed_candidate(history: _History) -> None:
    report = await _adopt(history)
    request = json.loads(_request(history.current))
    request.update(operation="attempt_report", report=report, idempotency_key="changed-candidate")
    # This is the other historical Kernel, not the one adopted in the Journal.
    with pytest.raises(ValueError, match="record an adopt Experiment"):
        await history.service.execute(history.capability.token, json.dumps(request).encode())


@pytest.mark.anyio
async def test_duplicate_evaluate_cannot_override_adopted_authority(history: _History) -> None:
    report = await _adopt(history)
    history.adapter.result = GatewayAdapterResult(
        "completed",
        {"correct": False},
        evaluation=EvaluationV2(correct=False, latency_us=None),
    )
    request = json.loads(_request(history.current))
    request.update(candidate=_candidate(), idempotency_key="fresh-failure")
    with pytest.raises(DuplicateGatewayTaskError) as duplicate:
        await history.service.execute(history.capability.token, json.dumps(request).encode())
    assert duplicate.value.previous_result_artifact_digest == history.after.result_artifact_digest
    await _submit(history, report)
    assert history.after.kernel_artifact_digest is not None
    resolved = history.control.find_candidate_evaluation(
        history.current.id,
        parse_artifact_digest(history.after.kernel_artifact_digest),
        gateway_result_digest=history.control.list_evaluations(history.historical_attempt.id)[
            -1
        ].gateway_result_digest,
    )
    assert resolved is not None and resolved.correct


@pytest.mark.anyio
@pytest.mark.parametrize("independent", [False, True])
async def test_adopted_nomination_reaches_unchanged_authoritative_gate(
    history: _History,
    tmp_path: Path,
    independent: bool,
) -> None:
    report = await _adopt(history)
    await _submit(history, report)
    client = FakeClient()
    events = FakeEvents()
    finalizer = AgateAuthoritativeCandidateEvaluator(
        client,  # type: ignore[arg-type]
        _builder,
        FakeContexts(),
        LocalArtifactStore(tmp_path / "artifacts"),
        history.control,
        events,
        wait_timeout_s=100,
        bootstrap_stages=(BootstrapEvaluationStage(1),),
        clock=lambda: NOW_DATETIME,
    )
    assert history.after.kernel_artifact_digest is not None
    outcome = await finalizer.finalize(
        history.current.id,
        parse_artifact_digest(history.after.kernel_artifact_digest),
        independent_evaluate=independent,
    )
    assert outcome.correct
    assert outcome.latency_us == (7.5 if independent else 12.0)
    assert len(client.submitted) == (1 if independent else 0)
    assert len(history.adapter.requests) == history.initial_adapter_requests
    assert [row.source for row in history.control.list_evaluations(history.current.id)] == (
        [GatewayEvaluationSource.RUNTIME_FINAL] if independent else []
    )
    provenance = cast(dict[str, Any], events.values[0][2])
    assert provenance["evaluation_attempt_id"] == history.historical_attempt.id


@pytest.mark.anyio
@pytest.mark.parametrize(
    "field", ["operator", "hardware_target", "dsl", "evaluation_contract_digest"]
)
async def test_visible_bootstrap_with_incompatible_context_cannot_be_adopted(
    history: _History,
    field: str,
) -> None:
    epoch = history.registry.get_epoch(history.current.epoch_id)
    lineage = history.registry.get_lineage(epoch.lineage_id)
    campaign = history.registry.get_campaign(lineage.campaign_id)
    subject = BootstrapGatewaySubject(
        new_attempt_id(),
        campaign.id,
        lineage.id,
        epoch.id,
        history.current.kernel_agent_revision_id,
        campaign.operator,
        campaign.hardware_target,
        lineage.dsl,
        campaign.evaluation_contract_digest,
        digest("seed"),
        digest("evidence"),
        NOW_DATETIME,
    )
    changes: dict[str, Any] = {
        "operator": "another_operator",
        "hardware_target": "another_gpu",
        "dsl": Dsl.CUDA if lineage.dsl != Dsl.CUDA else Dsl.TRITON,
        "evaluation_contract_digest": digest("different-contract"),
    }
    subject = replace(subject, **{field: changes[field]})
    capability = history.control.issue_bootstrap(
        subject,
        GatewayCapabilityPolicy(
            frozenset(GatewayOperation),
            4,
            NOW_DATETIME + timedelta(hours=1),
        ),
    )
    request = json.loads(_request(history.current))
    request.update(attempt_id=subject.attempt_id, candidate=_candidate())
    result = await history.service.execute(capability.token, json.dumps(request).encode())
    assert result.kernel_artifact_digest is not None
    with pytest.raises(ValueError, match="matching operator, hardware target, DSL"):
        history.control.validate_adoption_trial(
            history.current.id,
            next(
                t.id
                for t in history.control.list_kernel_trials((subject.attempt_id,))
                if t.kernel_artifact_digest == result.kernel_artifact_digest
            ),
        )


@pytest.mark.anyio
@pytest.mark.parametrize("status", ["blocked", "pivot"])
async def test_zero_experiment_report_is_accepted_without_gpu(
    history: _History,
    status: str,
) -> None:
    report: dict[str, Any] = _value(history.current.id)
    report.update(
        status=status,
        final_candidate=None,
        blocker="No safe experiment" if status == "blocked" else None,
        experiments=[],
        findings=[],
        direction_events=[],
        profile_evidence=None,
    )
    response = await _submit(history, report)
    assert response.result["status"] == "registered"
    assert history.control.list_experiments(history.current.id) == ()
    assert history.control.list_evaluations(history.current.id) == ()
    assert len(history.adapter.requests) == history.initial_adapter_requests


@pytest.mark.anyio
@pytest.mark.parametrize("status", ["blocked", "pivot"])
async def test_empty_report_cannot_omit_an_actual_in_progress_direction(
    history: _History,
    status: str,
) -> None:
    direction = await history.start()
    report: dict[str, Any] = _value(history.current.id)
    report.update(
        status=status,
        final_candidate=None,
        blocker="No safe experiment" if status == "blocked" else None,
        experiments=[],
        findings=[],
        direction_events=[],
        profile_evidence=None,
    )
    with pytest.raises(ValueError, match="Runtime-owned Direction in progress"):
        await _submit(history, report, key="omitted-direction")
    close_action = "block" if status == "blocked" else "defer"
    with pytest.raises(ValueError, match="requires at least one associated Experiment"):
        await history.journal(
            "direction_update",
            request={"action": close_action, "direction_id": direction, "analysis": "Blocked"},
        )
    await history.journal(
        "experiment_record",
        request={
            **history.experiment(direction, action="abandon_direction"),
            "before": {"result_artifact_digest": history.before.result_artifact_digest},
            "after": None,
            "evidence": "The planned check could not be run; no measurement was obtained.",
        },
    )
    await history.journal(
        "direction_update",
        request={
            "action": "block" if status == "blocked" else "defer",
            "direction_id": direction,
            "analysis": "No safe experiment was possible; preserve the actual direction state.",
        },
    )
    report["direction_events"] = list(history.control.list_direction_events(history.current.id))
    report["experiments"] = list(history.control.list_experiments(history.current.id))
    assert (await _submit(history, report)).result["status"] == "registered"


@pytest.mark.anyio
async def test_report_cannot_rewrite_or_omit_registered_adoption(history: _History) -> None:
    report = await _adopt(history)
    report["experiments"][0]["analysis"] = "Attempt to replace the immutable recorded decision."
    with pytest.raises(ValueError, match="must match the Runtime-owned"):
        await _submit(history, report)


@pytest.mark.anyio
async def test_adoption_survives_attempt_recovery_without_copying_measurement(
    history: _History,
) -> None:
    report = await _adopt(history)
    original = history.control.list_evaluations(history.historical_attempt.id)
    history.registry.record_infrastructure_failure(history.current.id, "interrupted after adoption")
    history.registry.retry_attempt(history.current.id)
    history.capability = history.control.issue(
        history.current.id,
        GatewayCapabilityPolicy(
            frozenset(GatewayOperation),
            4,
            NOW_DATETIME + timedelta(hours=1),
        ),
    )
    assert history.capability.recovery_generation == 1
    assert (await _submit(history, report)).result["status"] == "registered"
    assert history.control.list_evaluations(history.historical_attempt.id) == original
    assert history.control.list_evaluations(history.current.id) == ()
    assert len(history.adapter.requests) == history.initial_adapter_requests


@pytest.mark.anyio
async def test_empty_report_cannot_hide_inherited_in_progress_direction(history: _History) -> None:
    prior: dict[str, Any] = _value(history.historical_attempt.id)
    for index, event in enumerate(prior["direction_events"][:2]):
        history.control.append_direction_event(
            history.historical_attempt.id,
            f"old-direction-{index}",
            event,
            recovery_generation=0,
        )
    report: dict[str, Any] = _value(history.current.id)
    report.update(
        status="blocked",
        final_candidate=None,
        blocker="No safe experiment",
        experiments=[],
        findings=[],
        direction_events=[],
        profile_evidence=None,
    )
    assert history.control.list_direction_events(history.current.id) == ()
    with pytest.raises(ValueError, match="Runtime-owned Direction in progress"):
        await _submit(history, report, key="hide-inherited-direction")
    await history.journal(
        "experiment_record",
        request={
            **history.experiment(
                prior["direction_events"][0]["direction_id"], action="abandon_direction"
            ),
            "before": {"result_artifact_digest": history.before.result_artifact_digest},
            "after": None,
            "evidence": "The inherited investigation remains blocked; no measurement was made.",
        },
    )
    await history.journal(
        "direction_update",
        request={
            "action": "block",
            "direction_id": prior["direction_events"][0]["direction_id"],
            "analysis": "Acknowledge the inherited direction without inventing an experiment.",
        },
    )
    report["direction_events"] = list(history.control.list_direction_events(history.current.id))
    report["experiments"] = list(history.control.list_experiments(history.current.id))
    assert (await _submit(history, report)).result["status"] == "registered"
