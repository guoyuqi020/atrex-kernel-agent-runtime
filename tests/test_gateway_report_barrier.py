"""Terminal reports must not overtake executing Gateway calls."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from typing import Any

import anyio
import pytest
from conftest import digest
from test_attempt_report import _value as report_value
from test_gateway_proxy import NOW_DATETIME, _request, _service

from atrex_runtime.domain.errors import GatewayOperationsInProgressError, InfrastructureError
from atrex_runtime.domain.ids import new_attempt_id
from atrex_runtime.gateway import (
    GatewayCapabilityPolicy,
    GatewayOperation,
    GatewayProxyAsgiApp,
    GatewayProxyLimits,
    SqliteGatewayControl,
)
from atrex_runtime.gateway.control import BootstrapGatewaySubject


def _report(attempt: Any, status: str = "blocked") -> bytes:
    value = json.loads(_request(attempt))
    report = report_value(attempt.id)
    report["status"] = status
    if status != "candidate_ready":
        report.update(
            final_candidate=None,
            profile_evidence=None,
            experiments=[],
            direction_events=[],
            findings=[],
            blocker="Unable to finish baseline" if status == "blocked" else None,
        )
    value.update(operation="attempt_report", idempotency_key="terminal-report", report=report)
    return json.dumps(value).encode()


async def _post_report(service: Any, token: str, payload: bytes) -> tuple[int, dict[str, Any]]:
    app = GatewayProxyAsgiApp(service, GatewayProxyLimits(64 * 1024, 8, 16 * 1024))
    messages: list[dict[str, Any]] = []

    async def receive() -> dict[str, Any]:
        return {"type": "http.request", "body": payload, "more_body": False}

    async def send(message: dict[str, Any]) -> None:
        messages.append(message)

    await app(
        {
            "type": "http",
            "method": "POST",
            "path": "/v1/runtime/queries",
            "headers": [(b"authorization", f"Bearer {token}".encode())],
        },
        receive,
        send,
    )
    return messages[0]["status"], json.loads(messages[1]["body"])


@pytest.mark.anyio
@pytest.mark.parametrize("bootstrap", [False, True])
@pytest.mark.parametrize("status", ["candidate_ready", "blocked", "pivot"])
async def test_report_waits_for_single_measurement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    bootstrap: bool,
    status: str,
) -> None:
    registry, control, attempt, capability, service, adapter = _service(tmp_path)
    try:
        if bootstrap:
            lineage = registry.get_lineage(registry.get_epoch(attempt.epoch_id).lineage_id)
            campaign = registry.get_campaign(lineage.campaign_id)
            subject = BootstrapGatewaySubject(
                attempt_id=new_attempt_id(),
                campaign_id=lineage.campaign_id,
                lineage_id=lineage.id,
                epoch_id=attempt.epoch_id,
                kernel_agent_revision_id=attempt.kernel_agent_revision_id,
                operator=campaign.operator,
                hardware_target=campaign.hardware_target,
                dsl=lineage.dsl,
                evaluation_contract_digest=campaign.evaluation_contract_digest,
                input_kernel_digest=digest("seed"),
                evidence_digest=digest("evidence"),
                created_at=NOW_DATETIME,
            )
            capability = control.issue_bootstrap(
                subject,
                GatewayCapabilityPolicy(
                    frozenset({GatewayOperation.EVALUATE}),
                    4,
                    NOW_DATETIME + timedelta(hours=1),
                ),
            )
            attempt = replace(attempt, id=subject.attempt_id)
        started, release = anyio.Event(), anyio.Event()
        original = adapter.execute

        async def delayed(request: Any) -> Any:
            started.set()
            await release.wait()
            return await original(request)

        monkeypatch.setattr(adapter, "execute", delayed)
        payload = _report(attempt, status)
        async with anyio.create_task_group() as tasks:
            tasks.start_soon(service.execute, capability.token, _request(attempt))
            with anyio.fail_after(5):
                await started.wait()
            code, error = await _post_report(service, capability.token, payload)
            assert code == 409
            assert error["error"] == "gateway_calls_in_progress"
            assert error["pending_operations"] == ["evaluate"]
            assert "background" in error["detail"]
            assert "already-started" in error["recovery"][0]["instruction"]
            assert (
                control.get_operation_artifact(
                    attempt.id,
                    "terminal-report",
                    GatewayOperation.ATTEMPT_REPORT,
                )
                is None
            )
            # Report-status reads remain available and cannot end the active call.
            status_request = json.dumps(
                {
                    "schema_version": 2,
                    "attempt_id": attempt.id,
                    "idempotency_key": "report-status",
                    "operation": "attempt_report_status",
                }
            ).encode()
            observed = await service.execute(capability.token, status_request)
            assert observed.result == {"status": "missing"}
            release.set()
        code, accepted = await _post_report(service, capability.token, payload)
        assert code == 200
        assert accepted["result"]["status"] == "registered"
        assert len(adapter.requests) == 1
    finally:
        control.close()
        registry.close()


@pytest.mark.anyio
@pytest.mark.parametrize(
    "operation, fields",
    [
        ("profile", {"level": "sol"}),
        ("dev", {"command": "true"}),
        ("check", {}),
        ("disassemble", {"fmt": "ptx"}),
        ("env", {"gpu": "L20D"}),
    ],
)
@pytest.mark.parametrize("ending", ["success", "exception", "cancel"])
async def test_other_gateway_calls_release_barrier_on_every_exit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
    fields: dict[str, Any],
    ending: str,
) -> None:
    registry, control, attempt, capability, service, adapter = _service(tmp_path)
    started, release = anyio.Event(), anyio.Event()
    original = adapter.execute

    async def delayed(request: Any) -> Any:
        started.set()
        await release.wait()
        if ending == "exception":
            raise InfrastructureError("upstream failed")
        return await original(request)

    async def call() -> None:
        try:
            await service.execute(capability.token, json.dumps(request).encode())
        except InfrastructureError:
            assert ending == "exception"

    request = json.loads(_request(attempt))
    request.update(operation=operation, **fields)
    if operation == "env":
        request.pop("candidate")
    monkeypatch.setattr(adapter, "execute", delayed)
    try:
        async with anyio.create_task_group() as tasks:
            tasks.start_soon(call)
            with anyio.fail_after(5):
                await started.wait()
            code, error = await _post_report(service, capability.token, _report(attempt))
            assert code == 409
            assert error["pending_operations"] == [operation]
            if ending == "cancel":
                tasks.cancel_scope.cancel()
            else:
                release.set()
        code, accepted = await _post_report(service, capability.token, _report(attempt))
        assert code == 200
        assert accepted["result"]["status"] == "registered"
    finally:
        control.close()
        registry.close()


def test_barrier_is_shared_and_failed_reservations_do_not_block(tmp_path: Path) -> None:
    registry, control, _attempt, capability, _service_value, _adapter = _service(tmp_path)
    other = SqliteGatewayControl(
        tmp_path / "gateway.sqlite",
        registry,
        signing_key=b"p" * 32,
        clock=lambda: NOW_DATETIME,
    )
    try:
        evaluate = control.authorize(
            capability,
            GatewayOperation.EVALUATE,
            idempotency_key="evaluate",
            request_digest=str(digest("evaluate")),
        )
        report = other.authorize(
            capability,
            GatewayOperation.ATTEMPT_REPORT,
            idempotency_key="report",
            request_digest=str(digest("report")),
        )
        # A reservation alone (e.g. a prior failed request) is not an active call.
        with (
            other.operation_execution(report),
            pytest.raises(GatewayOperationsInProgressError),
            control.operation_execution(evaluate),
        ):
            pytest.fail("new Gateway call bypassed report admission")
        # Two overlapping invocations of the same authorized call must both finish.
        with control.operation_execution(evaluate):
            with (
                other.operation_execution(evaluate),
                pytest.raises(GatewayOperationsInProgressError),
                other.operation_execution(report),
            ):
                pytest.fail("report bypassed active calls")
            with (
                pytest.raises(GatewayOperationsInProgressError),
                other.operation_execution(report),
            ):
                pytest.fail("one reconnect released another running call")
        with other.operation_execution(report):
            pass
    finally:
        other.close()
        control.close()
        registry.close()


def test_gateway_schema_13_adds_empty_active_call_tracker(tmp_path: Path) -> None:
    registry, control, _attempt, _capability, _service_value, _adapter = _service(tmp_path)
    control.close()
    with sqlite3.connect(tmp_path / "gateway.sqlite") as connection:
        connection.execute("DROP TABLE gateway_active_calls")
        connection.execute("UPDATE metadata SET value = 13 WHERE key = 'schema_version'")
    reopened = SqliteGatewayControl(
        tmp_path / "gateway.sqlite",
        registry,
        signing_key=b"p" * 32,
        clock=lambda: NOW_DATETIME,
    )
    try:
        with sqlite3.connect(tmp_path / "gateway.sqlite") as connection:
            assert connection.execute("SELECT count(*) FROM gateway_active_calls").fetchone() == (
                0,
            )
    finally:
        reopened.close()
        registry.close()


def test_barrier_is_scoped_to_attempt_and_recovery_generation(tmp_path: Path) -> None:
    registry, control, attempt, capability, _service_value, _adapter = _service(
        tmp_path,
        attempts_per_trajectory=2,
    )
    policy = GatewayCapabilityPolicy(
        frozenset({GatewayOperation.EVALUATE}),
        4,
        NOW_DATETIME + timedelta(hours=1),
    )
    try:
        evaluate = control.authorize(
            capability,
            GatewayOperation.EVALUATE,
            idempotency_key="evaluate",
            request_digest=str(digest("evaluate")),
        )
        other_attempt = replace(attempt, id=new_attempt_id(), ordinal=2)
        registry.insert_attempt(other_attempt)
        other_capability = control.issue(other_attempt.id, policy)
        other_report = control.authorize(
            other_capability,
            GatewayOperation.ATTEMPT_REPORT,
            idempotency_key="report",
            request_digest=str(digest("report")),
        )
        with control.operation_execution(evaluate):
            with control.operation_execution(other_report):
                pass
            registry.record_infrastructure_failure(attempt.id, "old worker lost")
            registry.retry_attempt(attempt.id)
            recovered = control.issue(attempt.id, policy)
            report = control.authorize(
                recovered,
                GatewayOperation.ATTEMPT_REPORT,
                idempotency_key="report",
                request_digest=str(digest("report")),
            )
            assert report.recovery_generation == evaluate.recovery_generation + 1
            with control.operation_execution(report):
                pass
    finally:
        control.close()
        registry.close()


@pytest.mark.anyio
async def test_barrier_remains_until_result_is_persisted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, control, attempt, capability, service, _adapter = _service(tmp_path)
    report = control.authorize(
        capability,
        GatewayOperation.ATTEMPT_REPORT,
        idempotency_key="report",
        request_digest=str(digest("report")),
    )
    commit = control.commit_operation_artifact
    checked = False

    def checked_commit(*args: Any, **kwargs: Any) -> Any:
        nonlocal checked
        if args[2] is GatewayOperation.EVALUATE:
            checked = True
            with (
                pytest.raises(GatewayOperationsInProgressError),
                control.operation_execution(report),
            ):
                pytest.fail("report was admitted before result persistence")
        return commit(*args, **kwargs)

    monkeypatch.setattr(control, "commit_operation_artifact", checked_commit)
    try:
        await service.execute(capability.token, _request(attempt))
        assert checked
        with control.operation_execution(report):
            pass
    finally:
        control.close()
        registry.close()
