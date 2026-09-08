"""Historical adoption reuses validated evidence without creating new authority."""

from __future__ import annotations

import base64
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, replace
from datetime import timedelta
from pathlib import Path
from typing import Any, cast

import pytest
from conftest import digest
from test_gateway_proxy import NOW_DATETIME, FakeGatewayAdapter, _request, _service

from atrex_runtime.domain.ids import new_attempt_id
from atrex_runtime.domain.models import Attempt
from atrex_runtime.gateway import (
    GatewayCapability,
    GatewayCapabilityPolicy,
    GatewayOperation,
    GatewayProxyService,
    SqliteGatewayControl,
)
from atrex_runtime.gateway.protocol import EvaluationV2, GatewayProxyResponseV2
from atrex_runtime.gateway.proxy import GatewayAdapterResult
from atrex_runtime.registry.sqlite import SqliteRegistry


@dataclass
class _History:
    registry: SqliteRegistry
    control: SqliteGatewayControl
    current: Attempt
    capability: GatewayCapability
    service: GatewayProxyService
    adapter: FakeGatewayAdapter
    before: GatewayProxyResponseV2
    after: GatewayProxyResponseV2
    historical_attempt: Attempt
    sequence: int = 0

    async def journal(self, operation: str, **fields: object) -> GatewayProxyResponseV2:
        self.sequence += 1
        return await self.service.execute(
            self.capability.token,
            json.dumps(
                {
                    "schema_version": 2,
                    "attempt_id": self.current.id,
                    "idempotency_key": f"adoption-journal-{self.sequence}",
                    "operation": operation,
                    **fields,
                }
            ).encode(),
            operation_scope="journal",
        )

    async def start(self) -> str:
        proposed = await self.journal(
            "direction_update",
            request={
                "action": "propose",
                "name": "adopt the validated historical candidate",
                "hypothesis": "the previous measured candidate already implements this direction",
                "rationale": "matching contract evidence is already available",
                "plan": ["review and adopt the exact historical candidate"],
                "success_criteria": "reuse eligible evidence without another measurement",
                "stop_conditions": "the evidence does not match this contract",
            },
        )
        direction_id = str(cast(dict[str, Any], proposed.result)["direction_id"])
        await self.journal(
            "direction_update",
            request={
                "action": "start",
                "direction_id": direction_id,
                "analysis": "review the recorded historical comparison",
            },
        )
        return direction_id

    def experiment(self, direction_id: str, *, action: str = "adopt") -> dict[str, object]:
        return {
            "direction_id": direction_id,
            "name": "historical candidate adoption",
            "hypothesis": "the historical candidate supplies the planned improvement",
            "change": "adopt the exact existing kernel; no new measurement was performed",
            "before": {"kernel_trial_id": self.before.kernel_trial_id},
            "after": {"kernel_trial_id": self.after.kernel_trial_id},
            "evidence": "the existing ordinary full-contract evaluation",
            "analysis": "reuse the matching measured evidence",
            "action": action,
        }


@pytest.fixture
async def history(tmp_path: Path, request: pytest.FixtureRequest) -> AsyncIterator[_History]:
    registry, control, first, capability, service, adapter = _service(
        tmp_path, attempts_per_trajectory=2
    )
    try:
        kind = getattr(request, "param", "full")
        payload = json.loads(_request(first))
        if kind == "incorrect":
            adapter.result = GatewayAdapterResult(
                status="completed",
                result={"correct": False},
                evaluation=EvaluationV2(correct=False, latency_us=None),
            )
        elif kind == "custom":
            payload["input_py"] = "def _make_inputs(): return {}"
        elif kind == "correctness_only":
            payload["mode"] = "correctness_only"
            adapter.result = GatewayAdapterResult(
                status="completed",
                result={"correct": True},
                worker_result={"correct": True, "correctness": {"status": "PASS"}},
            )
        elif kind == "profile":
            payload.update(operation="profile", level="sol")
            adapter.result = GatewayAdapterResult(
                status="completed",
                result={"status": "succeeded", "kernels": []},
                profile_result={"status": "succeeded", "kernels": []},
            )
        elif kind == "abba":
            payload.update(
                comparison={"method": "abba", "repeats": 2},
                baseline=json.loads(json.dumps(payload["candidate"])),
            )
            adapter.result = GatewayAdapterResult(
                status="completed",
                result={"correct": True},
                worker_result={"correct": True, "mode": "full", "input_scope": "contract"},
            )
        before = await service.execute(capability.token, json.dumps(payload).encode())
        payload["idempotency_key"] = "evaluate-historical-after"
        payload["candidate"]["files"][0]["content_base64"] = base64.b64encode(
            b"def kernel(): return None\n"
        ).decode()
        after = await service.execute(capability.token, json.dumps(payload).encode())
        if kind != "concurrent":
            registry.complete_attempt(
                first.id, None, accepted_as_branch_best=False, failure_reason=None
            )
        current = replace(first, id=new_attempt_id(), ordinal=2)
        registry.insert_attempt(current)
        current_capability = control.issue(
            current.id,
            GatewayCapabilityPolicy(
                frozenset(GatewayOperation), 4, NOW_DATETIME + timedelta(hours=1)
            ),
        )
        yield _History(
            registry, control, current, current_capability, service, adapter, before, after, first
        )
    finally:
        control.close()
        registry.close()


@pytest.mark.anyio
async def test_adoption_records_current_reasoning_without_new_measurements(
    history: _History,
) -> None:
    before_trials = history.control.list_kernel_trials((history.historical_attempt.id,))
    before_evaluations = history.control.list_evaluations(history.historical_attempt.id)
    direction_id = await history.start()

    response = await history.journal(
        "experiment_record", request=history.experiment(direction_id)
    )

    assert cast(dict[str, Any], response.result)["status"] == "recorded"
    recorded = history.control.list_experiments(history.current.id)
    assert len(recorded) == 1
    assert recorded[0]["action"] == "adopt"
    after = cast(dict[str, object], recorded[0]["after"])
    assert after == {
        "kernel_artifact_digest": history.after.kernel_artifact_digest,
        "kernel_trial_id": history.after.kernel_trial_id,
        "result_artifact_digests": [history.after.result_artifact_digest],
    }
    assert history.control.list_kernel_trials((history.historical_attempt.id,)) == before_trials
    assert history.control.list_evaluations(history.historical_attempt.id) == before_evaluations
    assert history.control.list_kernel_trials((history.current.id,)) == ()
    assert history.control.list_evaluations(history.current.id) == ()
    assert len(history.adapter.requests) == 2
    assert history.control.record_kernel_trial_annotations(history.current.id, recorded) == ()
    await history.journal(
        "direction_update",
        request={
            "action": "complete",
            "direction_id": direction_id,
            "analysis": "the matching historical candidate was adopted",
        },
    )


@pytest.mark.anyio
@pytest.mark.parametrize("action", ["keep_after", "restore_before", "abandon_direction"])
async def test_other_actions_still_reject_historical_after(history: _History, action: str) -> None:
    direction_id = await history.start()

    with pytest.raises(ValueError, match="after Kernel Trial must belong to this logical Attempt"):
        await history.journal(
            "experiment_record", request=history.experiment(direction_id, action=action)
        )

    assert history.control.list_experiments(history.current.id) == ()


@pytest.mark.anyio
@pytest.mark.parametrize(
    "history", ["incorrect", "custom", "correctness_only", "profile", "abba"], indirect=True
)
async def test_ineligible_adoption_is_rejected_before_journal_append(history: _History) -> None:
    direction_id = await history.start()

    with pytest.raises(ValueError):
        await history.journal("experiment_record", request=history.experiment(direction_id))

    assert history.control.list_experiments(history.current.id) == ()
    assert history.control.list_kernel_trials((history.current.id,)) == ()
    assert len(history.adapter.requests) == 2


@pytest.mark.anyio
@pytest.mark.parametrize("history", ["concurrent"], indirect=True)
async def test_adoption_rejects_another_running_attempt(history: _History) -> None:
    _, visible = history.control.visible_kernel_trial_attempt_ids(history.current.id)
    assert history.historical_attempt.id not in visible
    assert history.after.kernel_trial_id is not None
    with pytest.raises(ValueError, match="outside this Attempt's visible history"):
        history.control.validate_adoption_trial(history.current.id, history.after.kernel_trial_id)
    direction_id = await history.start()

    with pytest.raises(ValueError, match="Kernel Trial is outside visible history"):
        await history.journal("experiment_record", request=history.experiment(direction_id))

    assert history.control.list_experiments(history.current.id) == ()
    assert len(history.adapter.requests) == 2


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("side_name", "field"),
    [
        ("before", "kernel_artifact_digest"),
        ("after", "kernel_artifact_digest"),
        ("before", "result_artifact_digests"),
        ("after", "result_artifact_digests"),
        ("before", "kernel_trial_id"),
        ("after", "kernel_trial_id"),
    ],
)
async def test_adoption_annotation_still_validates_all_evidence(
    history: _History, side_name: str, field: str
) -> None:
    direction_id = await history.start()
    await history.journal("experiment_record", request=history.experiment(direction_id))
    experiment = dict(history.control.list_experiments(history.current.id)[0])
    subject = dict(cast(dict[str, object], experiment[side_name]))
    subject[field] = (
        [str(digest("unobserved-result"))]
        if field == "result_artifact_digests"
        else "gtrial_" + "0" * 32
        if field == "kernel_trial_id"
        else str(digest("wrong-kernel"))
    )
    experiment[side_name] = subject

    with pytest.raises(ValueError):
        history.control.record_kernel_trial_annotations(history.current.id, [experiment])

    assert all(
        not trial.annotations
        for trial in history.control.list_kernel_trials((history.historical_attempt.id,))
    )
