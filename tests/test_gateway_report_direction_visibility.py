"""Terminal reports preserve peer-bound Experiments without claiming peer Directions."""

from __future__ import annotations

import base64
import json
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest
from test_attempt_report import _value
from test_gateway_control import _insert_attempt
from test_gateway_proxy import NOW_DATETIME, FakeGatewayAdapter, _request, _service

from atrex_runtime.artifacts.local import LocalArtifactStore
from atrex_runtime.domain.errors import DirectionTrajectoryConflictError
from atrex_runtime.domain.ids import new_attempt_id
from atrex_runtime.gateway import (
    GatewayCapabilityPolicy,
    GatewayOperation,
    GatewayProxyLimits,
    GatewayProxyService,
    SqliteGatewayControl,
)
from atrex_runtime.gateway.journals import RuntimeJournalService
from atrex_runtime.gateway.protocol import EvaluationV2
from atrex_runtime.gateway.proxy import GatewayAdapterResult
from atrex_runtime.registry.sqlite import SqliteRegistry
from atrex_runtime.workers.attempt_report import AttemptReportV12


@pytest.mark.anyio
@pytest.mark.parametrize("status", ["candidate_ready", "blocked", "pivot"])
@pytest.mark.parametrize("visibility", ["broadcast", "isolated"])
async def test_report_uses_direction_visibility_instead_of_attempt_ownership(
    tmp_path: Path, status: str, visibility: str
) -> None:
    registry = SqliteRegistry(tmp_path / "registry.sqlite")
    peer = _insert_attempt(registry, trajectories=2)
    current = replace(peer, id=new_attempt_id(), trajectory_ordinal=2)
    registry.insert_attempt(current)
    lineage_id = registry.get_epoch(peer.epoch_id).lineage_id
    registry._connection.execute(
        "UPDATE lineages SET trajectory_visibility = 'broadcast' WHERE id = ?", (lineage_id,)
    )
    control = SqliteGatewayControl(
        tmp_path / "gateway.sqlite", registry, signing_key=b"v" * 32, clock=lambda: NOW_DATETIME
    )
    policy = GatewayCapabilityPolicy(
        frozenset(GatewayOperation), 8, NOW_DATETIME + timedelta(hours=1)
    )
    capabilities = {attempt.id: control.issue(attempt.id, policy) for attempt in (peer, current)}
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    adapter = FakeGatewayAdapter(
        GatewayAdapterResult(
            status="completed",
            result={"correct": True, "latency_us": 12.0, "latency_us_by_shape": {"0": 12.0}},
            evaluation=EvaluationV2(correct=True, latency_us=12.0),
        )
    )
    service = GatewayProxyService(
        control, artifacts, adapter, GatewayProxyLimits(64 * 1024, 8, 16 * 1024), registry
    )
    sequence = 0

    async def journal(attempt: Any, operation: str, **fields: object) -> dict[str, Any]:
        nonlocal sequence
        sequence += 1
        response = await service.execute(
            capabilities[attempt.id].token,
            json.dumps(
                {
                    "schema_version": 2,
                    "attempt_id": attempt.id,
                    "idempotency_key": f"journal-{sequence}",
                    "operation": operation,
                    **fields,
                }
            ).encode(),
            operation_scope="journal",
        )
        assert isinstance(response.result, dict)
        return response.result

    async def start(attempt: Any, name: str) -> str:
        proposal = await journal(
            attempt,
            "direction_update",
            request={
                "action": "propose",
                "name": name,
                "hypothesis": "A different schedule reduces latency",
                "rationale": "Inspect measured scheduling overhead",
                "plan": ["Evaluate a candidate"],
                "success_criteria": "Correct and faster",
                "stop_conditions": "The measurement resolves the hypothesis",
            },
        )
        direction_id = str(proposal["direction_id"])
        await journal(
            attempt,
            "direction_update",
            request={"action": "start", "direction_id": direction_id, "analysis": "Investigate"},
        )
        return direction_id

    try:
        peer_direction = await start(peer, "Peer schedule")
        own_direction = await start(current, "Own schedule")
        peer_payload = json.loads(_request(peer))
        peer_payload["candidate"]["files"][0]["content_base64"] = base64.b64encode(
            b"def kernel(): return None\n"
        ).decode()
        peer_eval = await service.execute(
            capabilities[peer.id].token, json.dumps(peer_payload).encode()
        )
        own_eval = await service.execute(capabilities[current.id].token, _request(current))

        def experiment(direction_id: str, result: Any, action: str) -> dict[str, object]:
            return {
                "direction_id": direction_id,
                "name": "Candidate comparison",
                "hypothesis": "The measured schedule is useful",
                "change": "Use the measured candidate",
                "before": {"result_artifact_digest": own_eval.result_artifact_digest},
                "after": {"result_artifact_digest": result.result_artifact_digest},
                "evidence": "The ordinary full-contract evaluation",
                "analysis": "Retain the recorded evidence",
                "action": action,
            }

        own_experiment = await journal(
            current, "experiment_record", request=experiment(own_direction, own_eval, "keep_after")
        )
        adoption = await journal(
            current, "experiment_record", request=experiment(peer_direction, peer_eval, "adopt")
        )
        await journal(
            current,
            "direction_update",
            request={
                "action": "complete",
                "direction_id": own_direction,
                "analysis": "Own measurement completed",
                "hypothesis_status": "unresolved",
                "supporting_experiment_ids": [own_experiment["experiment_id"]],
            },
        )
        # Reusing peer evidence does not authorize changing the peer's open Direction.
        with pytest.raises(DirectionTrajectoryConflictError):
            await journal(
                current,
                "direction_update",
                request={
                    "action": "defer",
                    "direction_id": peer_direction,
                    "analysis": "Try to close peer work",
                    "hypothesis_status": "unresolved",
                    "supporting_experiment_ids": [adoption["experiment_id"]],
                },
            )
        snapshot = await journal(current, "journal_snapshot")
        report_value = _value(current.id)
        report_value.update(
            status=status,
            experiments=snapshot["experiments"],
            direction_events=snapshot["direction_events"],
            profile_evidence=None,
            contributing_result_artifact_digests=[
                own_eval.result_artifact_digest,
                peer_eval.result_artifact_digest,
            ],
            blocker="No further safe change" if status == "blocked" else None,
        )
        report_value["findings"][0]["supporting_experiment_ids"] = [adoption["experiment_id"]]
        if status != "candidate_ready":
            report_value["final_candidate"] = None
        report = AttemptReportV12.model_validate(report_value)
        assert peer_direction not in {event.direction_id for event in report.direction_events}
        registry._connection.execute(
            "UPDATE lineages SET trajectory_visibility = ? WHERE id = ?", (visibility, lineage_id)
        )
        submission = json.dumps(
            {
                "schema_version": 2,
                "attempt_id": current.id,
                "operation": "attempt_report",
                "idempotency_key": "terminal-report",
                "candidate": peer_payload["candidate"],
                "report": report_value,
            }
        ).encode()
        if visibility == "isolated":
            with pytest.raises(
                ValueError,
                match=r"Direction outside this Attempt's visible history|"
                r"Kernel/Result Artifacts are outside the permitted visible history",
            ):
                await service.execute(
                    capabilities[current.id].token, submission, operation_scope="runtime"
                )
            return
        response = await service.execute(
            capabilities[current.id].token, submission, operation_scope="runtime"
        )
        assert response.result["status"] == "registered"
        assert response.result["report_status"] == status
        loaded = await journal(current, "direction_load", direction_id=peer_direction)
        assert loaded["status"] == "in_progress(other)"
        # Both the status endpoint and Worker file parser must accept the sealed report.
        accepted = await service.execute(
            capabilities[current.id].token,
            json.dumps(
                {
                    "schema_version": 2,
                    "attempt_id": current.id,
                    "operation": "attempt_report_status",
                    "idempotency_key": "report-status",
                }
            ).encode(),
            operation_scope="runtime",
        )
        assert accepted.result["status"] == "accepted"
        path = tmp_path / "report.json"
        path.write_text(json.dumps(accepted.result["report"]))
        assert (
            AttemptReportV12.from_file(path, expected_attempt_id=current.id, max_bytes=64 * 1024)
            == report
        )
        edited = report.model_copy(update={"experiments": report.experiments[:-1]})
        with pytest.raises(ValueError, match="must match the Runtime-owned"):
            RuntimeJournalService(control, artifacts).validate_report_journal(edited)
    finally:
        control.close()
        registry.close()


@pytest.mark.anyio
@pytest.mark.parametrize("status", ["blocked", "pivot"])
async def test_report_only_client_cannot_invent_an_external_direction(
    tmp_path: Path, status: str
) -> None:
    registry, control, attempt, _, _, _ = _service(tmp_path)
    try:
        value = _value(attempt.id)
        value.update(
            status=status,
            direction_events=[],
            final_candidate=None,
            blocker="Blocked" if status == "blocked" else None,
        )
        report = AttemptReportV12.model_validate(value)
        with pytest.raises(ValueError, match="Direction outside this Attempt's visible history"):
            RuntimeJournalService(
                control, LocalArtifactStore(tmp_path / "artifacts")
            ).validate_report_journal(report)
    finally:
        control.close()
        registry.close()
