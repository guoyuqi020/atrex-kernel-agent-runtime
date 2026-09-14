"""Late evidence must not reopen a Direction or weaken Trial validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest
from test_attempt_report import _value as _report_value
from test_gateway_proxy import _request, _service

from atrex_runtime.artifacts.local import LocalArtifactStore
from atrex_runtime.gateway.journals import RuntimeJournalService
from atrex_runtime.gateway.proxy import GatewayAdapterResult
from atrex_runtime.workers.attempt_report import AttemptReportV12


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("close_action", "closed_status"),
    [
        ("complete", "completed"),
        ("abandon", "abandoned"),
        ("block", "blocked"),
        ("defer", "deferred"),
    ],
)
@pytest.mark.parametrize("other_in_progress", [False, True])
async def test_closed_direction_accepts_late_profile_evidence(
    tmp_path: Path,
    close_action: str,
    closed_status: str,
    other_in_progress: bool,
) -> None:
    registry, control, attempt, capability, service, adapter = _service(tmp_path)

    async def journal(operation: str, key: str, **fields: object) -> dict[str, Any]:
        response = await service.execute(
            capability.token,
            json.dumps(
                {
                    "schema_version": 2,
                    "attempt_id": attempt.id,
                    "idempotency_key": key,
                    "operation": operation,
                    **fields,
                }
            ).encode(),
            operation_scope="journal",
        )
        return cast(dict[str, Any], response.result)

    async def propose(key: str) -> str:
        response = await journal(
            "direction_update",
            key,
            request={
                "action": "propose",
                "name": key,
                "hypothesis": "vector loads reduce memory traffic",
                "rationale": "profile indicates excess transactions",
                "plan": ["measure the candidate"],
                "success_criteria": "correct and faster",
                "stop_conditions": "the hypothesis is resolved",
            },
        )
        return str(response["direction_id"])

    async def update(direction_id: str, action: str) -> None:
        await journal(
            "direction_update",
            f"{direction_id}-{action}",
            request={
                "direction_id": direction_id,
                "action": action,
                "analysis": f"{action} the research direction",
                **(
                    {
                        "hypothesis_status": "unresolved",
                        "supporting_experiment_ids": [
                            item["experiment_id"]
                            for item in control.list_experiments(attempt.id)
                            if item["direction_id"] == direction_id
                        ],
                    }
                    if action != "start"
                    else {}
                ),
            },
        )

    try:
        direction_id = await propose("late-evidence-direction")
        await update(direction_id, "start")
        evaluated = await service.execute(capability.token, _request(attempt))
        subject = {"result_artifact_digest": evaluated.result_artifact_digest}
        experiment = {
            "direction_id": direction_id,
            "name": "candidate evaluation",
            "hypothesis": "the candidate remains correct",
            "change": "validate the current candidate",
            "before": subject,
            "after": subject,
            "evidence": "the measured evaluation",
            "analysis": "retain the measured candidate",
            "action": "keep_after",
        }
        recorded = await journal("experiment_record", "first-experiment", request=experiment)

        # Profile was measured before closure, but its evidence was omitted from
        # the already-frozen Experiment. It is citable without a supplement now.
        adapter.result = GatewayAdapterResult(
            status="completed",
            result={"status": "succeeded", "result": {"kernels": []}},
            profile_result={"status": "succeeded", "kernels": []},
        )
        payload = json.loads(_request(attempt))
        payload.update(idempotency_key="late-evidence-profile", operation="profile", level="sol")
        profiled = await service.execute(capability.token, json.dumps(payload).encode())
        await update(direction_id, close_action)
        second = await propose("next-direction")
        if other_in_progress:
            await update(second, "start")

        prior_events = control.list_direction_events(attempt.id)
        prior_experiments = control.list_experiments(attempt.id)
        snapshot = await journal("journal_snapshot", "snapshot-before")
        binding = {
            "operation": "profile",
            "kernel_artifact_digest": profiled.kernel_artifact_digest,
            "result_artifact_digest": profiled.result_artifact_digest,
        }
        assert any(
            item["result_artifact_digest"] == profiled.result_artifact_digest
            for item in snapshot["citable_profile_results"]
        )
        control.record_kernel_trial_annotations(
            attempt.id,
            prior_experiments,
            profile_supporting_results=(binding,),
        )

        late_request = {**experiment, "name": "late profile diagnosis"}
        late = await journal("experiment_record", "late-experiment", request=late_request)
        assert (
            await journal(
                "experiment_record",
                "late-experiment",
                request=late_request,
            )
            == late
        )
        assert control.list_direction_events(attempt.id) == prior_events
        assert control.list_experiments(attempt.id)[:1] == prior_experiments
        assert len(control.list_experiments(attempt.id)) == 2
        assert len(adapter.requests) == 4  # Three Eval repetitions plus one Profile.

        snapshot = await journal("journal_snapshot", "snapshot-after")
        directions = {item["direction_id"]: item for item in snapshot["directions"]}
        assert directions[direction_id]["status"] == closed_status
        assert directions[direction_id]["supporting_experiment_ids"] == [
            recorded["experiment_id"],
        ]
        assert directions[direction_id]["associated_experiment_ids"] == [
            recorded["experiment_id"],
            late["experiment_id"],
        ]
        assert directions[second]["status"] == ("in_progress" if other_in_progress else "proposed")
        assert {key: value for key, value in binding.items() if key != "operation"} in (
            snapshot["citable_profile_results"]
        )

        # Closure never bypasses visibility or Trial identity validation.
        for ordinal, invalid in enumerate(
            (
                {**late_request, "direction_id": "direction_" + "f" * 32},
                {**late_request, "after": {"result_artifact_digest": "sha256:" + "f" * 64}},
            )
        ):
            with pytest.raises(ValueError, match="visible history"):
                await journal("experiment_record", f"invalid-late-{ordinal}", request=invalid)
        if not other_in_progress:
            with pytest.raises(ValueError, match="current status is proposed"):
                await journal(
                    "experiment_record",
                    "proposed-experiment",
                    request={
                        **late_request,
                        "direction_id": second,
                    },
                )
        else:
            await journal(
                "experiment_record",
                "pause-second-direction",
                request={
                    **experiment,
                    "direction_id": second,
                    "name": "second direction investigation paused",
                    "before": subject,
                    "after": None,
                    "action": "abandon_direction",
                    "evidence": "No measurement was made for this direction before the pause.",
                },
            )
            await update(second, "defer")
        assert len(control.list_experiments(attempt.id)) == (3 if other_in_progress else 2)

        snapshot = await journal("journal_snapshot", "terminal-snapshot")
        report = _report_value(attempt.id)
        report.update(
            experiments=snapshot["experiments"],
            direction_events=snapshot["direction_events"],
            contributing_result_artifact_digests=[evaluated.result_artifact_digest],
        )
        cast(dict[str, Any], report["profile_evidence"])["supporting_results"] = [binding]
        cast(list[dict[str, Any]], report["findings"])[0]["supporting_experiment_ids"] = [
            late["experiment_id"],
        ]
        validated = AttemptReportV12.model_validate(report)
        journals = RuntimeJournalService(control, LocalArtifactStore(tmp_path / "artifacts"))
        journals.validate_report_journal(validated)
        control.record_kernel_trial_annotations(
            attempt.id,
            control.list_experiments(attempt.id),
            profile_supporting_results=(binding,),
        )
    finally:
        control.close()
        registry.close()
