"""Profile provenance depends on visible observations, not Experiment bookkeeping."""

from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path
from typing import Any, cast

import pytest
from conftest import digest
from test_attempt_report import _value as _report_value
from test_gateway_control import _insert_attempt
from test_gateway_proxy import NOW_DATETIME, _request, _service

from atrex_runtime.artifacts.local import LocalArtifactStore
from atrex_runtime.domain.models import Dsl
from atrex_runtime.gateway import GatewayCapabilityPolicy, GatewayOperation
from atrex_runtime.gateway.proxy import GatewayAdapterResult


@pytest.mark.anyio
async def test_profile_without_experiment_can_be_handed_off(tmp_path: Path) -> None:
    registry, control, attempt, capability, service, adapter = _service(tmp_path)
    try:
        # A normal Evaluate is not a citable Profile.
        evaluated = await service.execute(capability.token, _request(attempt))
        adapter.result = GatewayAdapterResult(
            status="completed",
            result={"status": "succeeded", "result": {"kernels": []}},
            profile_result={"status": "succeeded", "kernels": []},
        )
        payload = json.loads(_request(attempt))
        payload.update(idempotency_key="profile-1", operation="profile", level="sol")
        profiled = await service.execute(capability.token, json.dumps(payload).encode())
        identity = {
            "kernel_artifact_digest": profiled.kernel_artifact_digest,
            "kernel_trial_id": profiled.kernel_trial_id,
            "result_artifact_digest": profiled.result_artifact_digest,
        }
        reference = {"operation": "profile", **identity}

        # A pending Profile has a Kernel binding but no committed Result Artifact.
        control.authorize(
            capability, GatewayOperation.PROFILE,
            idempotency_key="pending-profile", request_digest=str(digest("pending-request")),
        )
        control.bind_operation_candidate(
            attempt.id, "pending-profile", GatewayOperation.PROFILE, digest("pending-kernel"),
        )
        snapshot = await service.execute(
            capability.token,
            json.dumps({
                "schema_version": 2, "attempt_id": attempt.id,
                "operation": "journal_snapshot", "idempotency_key": "snapshot",
            }).encode(),
            operation_scope="journal",
        )
        view = cast(dict[str, Any], snapshot.result)
        assert view["citable_profile_results"] == [identity]
        assert view["experiments"] == view["direction_events"] == []

        assert control.record_kernel_trial_annotations(
            attempt.id, (), profile_supporting_results=(reference,),
        ) == ()
        # Exact identities, operation kind and uniqueness are still mandatory.
        for invalid, message in (
            ({**reference, "kernel_artifact_digest": str(digest("wrong-kernel"))}, "Kernel"),
            ({**reference, "kernel_trial_id": "gtrial_" + "f" * 32}, "outside visible history"),
            ({**reference, "result_artifact_digest": str(digest("unrecorded"))}, "operation"),
            (
                {**reference, "result_artifact_digest": evaluated.result_artifact_digest},
                "operation",
            ),
            ({**reference, "operation": "evaluate"}, "must be profile"),
        ):
            with pytest.raises(ValueError, match=message):
                control.record_kernel_trial_annotations(
                    attempt.id, (), profile_supporting_results=(invalid,),
                )
        with pytest.raises(ValueError, match="must be unique"):
            control.record_kernel_trial_annotations(
                attempt.id, (), profile_supporting_results=(reference, reference),
            )

        # A diagnostic-only blocked handoff needs no invented optimization Experiment.
        report = _report_value(attempt.id)
        report.update(
            status="blocked", final_candidate=None, blocker="no viable optimization found",
            experiments=[], direction_events=[], findings=[], contributing_kernel_trial_ids=[],
        )
        cast(dict[str, Any], report["profile_evidence"])["supporting_results"] = [reference]
        payload = json.loads(_request(attempt))
        payload.update(operation="attempt_report", idempotency_key="report", report=report)
        accepted = await service.execute(
            capability.token, json.dumps(payload).encode(), operation_scope="runtime",
        )
        receipt = cast(dict[str, Any], accepted.result)
        assert receipt["status"] == "registered"
        sealed = LocalArtifactStore(tmp_path / "artifacts").verify(
            receipt["report_artifact_digest"],
        )
        stored = json.loads((sealed.payload_path / "value.json").read_text())
        assert stored["profile_evidence"]["supporting_results"] == [reference]
        assert stored["experiments"] == []
        assert len(adapter.requests) == 2
    finally:
        control.close()
        registry.close()


@pytest.mark.anyio
async def test_other_lineage_profile_is_neither_projected_nor_citable(tmp_path: Path) -> None:
    registry, control, current, capability, service, adapter = _service(tmp_path)
    try:
        foreign = _insert_attempt(registry, Dsl.CUDA)
        foreign_capability = control.issue(
            foreign.id,
            GatewayCapabilityPolicy(
                frozenset({GatewayOperation.PROFILE}), 1, NOW_DATETIME + timedelta(hours=1),
            ),
        )
        candidate, result = digest("foreign-kernel"), digest("foreign-profile")
        control.authorize(
            foreign_capability, GatewayOperation.PROFILE,
            idempotency_key="foreign-profile", request_digest=str(digest("foreign-request")),
        )
        control.bind_operation_candidate(
            foreign.id, "foreign-profile", GatewayOperation.PROFILE, candidate,
        )
        control.commit_operation_artifact(
            foreign.id, "foreign-profile", GatewayOperation.PROFILE, result,
        )
        reference = {
            "operation": "profile", "kernel_artifact_digest": str(candidate),
            "kernel_trial_id": control.list_kernel_trials((foreign.id,))[0].id,
            "result_artifact_digest": str(result),
        }
        assert control.record_kernel_trial_annotations(
            foreign.id, (), profile_supporting_results=(reference,),
        ) == ()
        snapshot = await service.execute(
            capability.token,
            json.dumps({
                "schema_version": 2, "attempt_id": current.id,
                "operation": "journal_snapshot", "idempotency_key": "snapshot",
            }).encode(), operation_scope="journal",
        )
        assert cast(dict[str, Any], snapshot.result)["citable_profile_results"] == []
        with pytest.raises(ValueError, match="outside visible history"):
            control.record_kernel_trial_annotations(
                current.id, (), profile_supporting_results=(reference,),
            )
        assert adapter.requests == []
    finally:
        control.close()
        registry.close()
