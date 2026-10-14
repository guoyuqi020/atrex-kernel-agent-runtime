"""Claims may cite separate visible measurements of one exact Kernel."""

from __future__ import annotations

import base64
import json
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import httpx
import pytest
from test_attempt_finding_evidence import _finding
from test_attempt_report import _value
from test_bootstrap_result_visibility import _seed_input
from test_gateway_proxy import NOW_DATETIME, _insert_attempt, _request, _service
from test_journal_measurement_summary import Case
from test_journal_measurement_summary import case as case

from atrex_runtime.artifacts.local import LocalArtifactStore
from atrex_runtime.domain.ids import new_attempt_id
from atrex_runtime.domain.models import Dsl
from atrex_runtime.gateway import (
    GatewayCapabilityPolicy,
    GatewayOperation,
    GatewayProxyAsgiApp,
    GatewayProxyLimits,
)
from atrex_runtime.gateway.proxy import GatewayAdapterResult


@pytest.mark.anyio
@pytest.mark.parametrize(
    "modules,read_enabled",
    [
        ((), True),
        (("experiments",), True),
        (("directions",), True),
        (("directions", "experiments"), True),
        ((), False),
    ],
)
@pytest.mark.parametrize("history", ["bootstrap", "previous_attempt", "previous_generation"])
async def test_report_and_direction_can_combine_visible_results(
    tmp_path: Path,
    modules: tuple[str, ...],
    read_enabled: bool,
    history: str,
) -> None:
    registry, control, first, capability, service, adapter = _service(
        tmp_path, attempts_per_trajectory=2, tool_modules=modules
    )
    try:
        baseline = _seed_input(
            registry, LocalArtifactStore(tmp_path / "artifacts"), first, tmp_path
        )
        first = registry.get_attempt(first.id)
        lineage = registry.get_epoch(first.epoch_id).lineage_id
        registry._connection.execute(
            "UPDATE lineages SET result_artifact_read_enabled=? WHERE id=?", (read_enabled, lineage)
        )
        candidate = {
            "files": [
                {
                    "path": "kernel.py",
                    "content_base64": base64.b64encode(b"# preserved bootstrap\n").decode(),
                }
            ]
        }
        app = GatewayProxyAsgiApp(service, GatewayProxyLimits(1_048_576, 8, 16 * 1024))
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://runtime",
            headers={"Authorization": f"Bearer {capability.token}"},
        ) as client:
            case = Case(client, first, adapter, control, modules)
            old_result = baseline.evaluation.gateway_result_digest
            if history != "bootstrap":
                measured = await case.call(
                    "evaluate", candidate=candidate, latency_prediction="retained"
                )
                old_result = measured["result_artifact_digest"]
                if history == "previous_attempt":
                    registry.complete_attempt(
                        first.id, None, accepted_as_branch_best=False, failure_reason=None
                    )
                    case.attempt = replace(first, id=new_attempt_id(), ordinal=2)
                    registry.insert_attempt(case.attempt)
                else:
                    registry.record_infrastructure_failure(first.id, "worker interruption")
                    registry.retry_attempt(first.id)
                    case.attempt = registry.get_attempt(first.id)
                capability = control.issue(
                    case.attempt.id,
                    GatewayCapabilityPolicy(
                        frozenset(GatewayOperation), 4, NOW_DATETIME + timedelta(hours=1)
                    ),
                )
                client.headers["Authorization"] = f"Bearer {capability.token}"

            adapter.result = GatewayAdapterResult(status="completed", result={"exit_code": 0})
            # Dev uses the operation endpoint; Case.call is deliberately limited to eval/check.
            response = await client.post(
                "/v1/operations",
                json={
                    "schema_version": 2,
                    "attempt_id": case.attempt.id,
                    "operation": "dev",
                    "idempotency_key": "current-dev",
                    "candidate": candidate,
                    "command": "python probe.py",
                },
            )
            assert response.status_code == 200, response.text
            dev = response.json()
            assert dev["kernel_artifact_digest"] == baseline.artifact_digest
            subject = {
                "kernel_artifact_digest": baseline.artifact_digest,
                "result_artifact_digests": [old_result, dev["result_artifact_digest"]],
            }
            before = control.visible_kernel_evidence(case.attempt.id)
            # The production failure: both edges exist, but no single Trial owns both.
            assert not any(
                set(subject["result_artifact_digests"])
                <= {o.result_artifact_digest for o in trial.observations}
                for trial in before
            )
            if "directions" in modules:
                direction = await case.direction()
                closed = (
                    await case.call(
                        "direction_update",
                        request={
                            "action": "complete",
                            "direction_id": direction,
                            "hypothesis_status": "supported",
                            "claim_kind": "causal_hypothesis",
                            "scope": "Historical evaluation and current Dev of the same Kernel",
                            "supporting_results": [subject],
                            **(
                                {"supporting_experiment_ids": []}
                                if "experiments" in modules
                                else {}
                            ),
                        },
                    )
                )["result"]
                assert closed["hypothesis_status"] == "supported"
            snapshot = (await case.call("journal_snapshot"))["result"]
            report = _value(case.attempt.id)
            report.update(
                status="pivot",
                final_candidate=None,
                profile_evidence=None,
                tool_modules=list(modules),
                experiments=snapshot["experiments"],
                direction_events=snapshot["direction_events"],
                findings=[
                    _finding(
                        claim="Both measurements concern this exact Kernel",
                        assessment="supported",
                        scope="Historical full evaluation and this Attempt's Dev probe",
                        supporting_results=[subject],
                    )
                ],
            )
            if history == "bootstrap" and "experiments" not in modules:
                # Citable Bootstrap evidence + Dev must not bypass the nomination gate.
                nomination = {
                    **report,
                    "status": "candidate_ready",
                    "final_candidate": {"change_summary": "Unchanged input"},
                }
                with pytest.raises(AssertionError, match="No Agent evaluate covers it"):
                    await case.call("attempt_report", candidate=candidate, report=nomination)
            submitted = (await case.call("attempt_report", candidate=candidate, report=report))[
                "result"
            ]
            assert submitted["status"] == "registered"
            recovered = (await case.call("attempt_report_status"))["result"]["report"]
            assert recovered["findings"][0]["supporting_results"] == [subject]
            assert recovered["findings"][0]["assessment"] == "supported"
            # Binding evidence must neither merge Trials nor fabricate current evaluations.
            after = control.visible_kernel_evidence(case.attempt.id)
            assert [trial.id for trial in after] == [trial.id for trial in before]
            for old, new in zip(before, after, strict=True):
                assert new.recovery_generation == old.recovery_generation
                assert all(observation in new.observations for observation in old.observations)
            assert not control.list_evaluations(case.attempt.id) or history == "previous_generation"
            assert len(adapter.requests) == (1 if history == "bootstrap" else 2)
    finally:
        control.close()
        registry.close()


@pytest.mark.anyio
@pytest.mark.parametrize("invalid", ["wrong_kernel", "typo", "isolated_peer"])
async def test_mixed_claim_references_still_reject_invalid_pairs(case: Case, invalid: str) -> None:
    measured, candidate = await case.measure({"0": 80.0})
    bad_result = measured["result_artifact_digest"][:-1] + (
        "0" if measured["result_artifact_digest"][-1] != "0" else "1"
    )
    if invalid == "wrong_kernel":
        other, _ = await case.measure({"0": 90.0})
        bad_result = other["result_artifact_digest"]
    elif invalid == "isolated_peer":
        peer = _insert_attempt(case.control._registry, dsl=Dsl.CUDA)
        capability = case.control.issue(
            peer.id,
            GatewayCapabilityPolicy(
                frozenset(GatewayOperation), 4, NOW_DATETIME + timedelta(hours=1)
            ),
        )
        request = json.loads(_request(peer))
        request["candidate"] = candidate  # Exact same Kernel in an invisible lineage.
        response = await case.client.post(
            "/v1/operations",
            json=request,
            headers={"Authorization": f"Bearer {capability.token}"},
        )
        assert response.status_code == 200, response.text
        bad_result = response.json()["result_artifact_digest"]
    subject = {
        "kernel_artifact_digest": measured["kernel_artifact_digest"],
        "result_artifact_digests": [measured["result_artifact_digest"], bad_result],
    }
    report = _value(case.attempt.id)
    report.update(
        status="pivot",
        final_candidate=None,
        tool_modules=list(case.modules),
        experiments=[],
        direction_events=[],
        profile_evidence=None,
        findings=[
            _finding(),
            _finding(
                claim="A scoped observation",
                assessment="supported",
                scope="This Kernel",
                supporting_results=[subject],
            ),
        ],
    )
    with pytest.raises(AssertionError) as error:
        await case.call("attempt_report", candidate=candidate, report=report)
    message = str(error.value)
    assert "findings[1]: supporting_results[0].result_artifact_digests[1]" in message
    assert bad_result in message and measured["kernel_artifact_digest"] in message
    assert "outside the permitted visible history" in message
    # Correcting just the bad reference permits the same report to be submitted.
    subject["result_artifact_digests"].pop()
    assert (await case.call("attempt_report", candidate=candidate, report=report))["result"][
        "status"
    ] == "registered"
