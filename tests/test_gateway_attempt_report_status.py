"""Server-accepted terminal reports can be recovered without trusting local files or GPU jobs."""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass, replace
from datetime import timedelta
from pathlib import Path
from typing import Any, cast

import pytest
from pydantic import ValidationError
from test_attempt_report import _value as report_value
from test_gateway_proxy import NOW_DATETIME, FakeGatewayAdapter, _insert_attempt, _request

from atrex_runtime.artifacts.local import ArtifactKind, JsonValue, LocalArtifactStore
from atrex_runtime.domain.errors import InfrastructureError
from atrex_runtime.domain.ids import new_attempt_id
from atrex_runtime.domain.models import Attempt
from atrex_runtime.gateway.control import SqliteGatewayControl
from atrex_runtime.gateway.control_models import (
    GatewayCapability,
    GatewayCapabilityPolicy,
    GatewayOperation,
)
from atrex_runtime.gateway.protocol import gateway_agent_request_schema
from atrex_runtime.gateway.proxy import (
    _REQUEST_ADAPTER,
    GatewayAdapterResult,
    GatewayProxyAsgiApp,
    GatewayProxyLimits,
    GatewayProxyService,
)
from atrex_runtime.registry.sqlite import SqliteRegistry
from atrex_runtime.serialization import canonical_json_bytes, canonical_json_digest
from atrex_runtime.workers.attempt_report import AttemptReportV12


@dataclass
class Case:
    registry: SqliteRegistry
    control: SqliteGatewayControl
    attempt: Attempt
    capability: GatewayCapability
    artifacts: LocalArtifactStore
    adapter: FakeGatewayAdapter
    service: GatewayProxyService

    def request(self, operation: str, key: str, **fields: Any) -> bytes:
        return json.dumps(
            {
                "schema_version": 2,
                "attempt_id": self.attempt.id,
                "operation": operation,
                "idempotency_key": key,
                **fields,
            }
        ).encode()

    def report(self) -> dict[str, Any]:
        value = report_value(self.attempt.id)
        value.update(status="blocked", final_candidate=None, blocker="No correct Kernel yet")
        return value

    def submission(self, key: str = "report", report: dict[str, Any] | None = None) -> bytes:
        return self.request(
            "attempt_report",
            key,
            report=self.report() if report is None else report,
            candidate=json.loads(_request(self.attempt))["candidate"],
        )


@pytest.fixture
def case(tmp_path: Path) -> Iterator[Case]:
    registry = SqliteRegistry(tmp_path / "registry.sqlite")
    attempt = _insert_attempt(registry, attempts_per_trajectory=2)
    control = SqliteGatewayControl(
        tmp_path / "gateway.sqlite",
        registry,
        signing_key=b"r" * 32,
        clock=lambda: NOW_DATETIME,
    )
    # The report operations must be implicit and remain usable after this one call is spent.
    capability = control.issue(
        attempt.id,
        GatewayCapabilityPolicy(
            frozenset({GatewayOperation.EVALUATE}),
            1,
            NOW_DATETIME + timedelta(hours=1),
        ),
    )
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    adapter = FakeGatewayAdapter(GatewayAdapterResult("completed", {"unexpected": "GPU"}))
    service = GatewayProxyService(
        control,
        artifacts,
        adapter,
        GatewayProxyLimits(64 * 1024, 8, 16 * 1024),
        registry,
    )
    try:
        yield Case(registry, control, attempt, capability, artifacts, adapter, service)
    finally:
        control.close()
        registry.close()


@pytest.mark.anyio
async def test_report_status_refreshes_missing_to_accepted_without_quota_or_gpu(
    case: Case,
    tmp_path: Path,
) -> None:
    case.control.authorize(
        case.capability,
        GatewayOperation.EVALUATE,
        idempotency_key="spent-budget",
        request_digest=str(canonical_json_digest({"budget": "spent"})),
    )
    # Neither a local report nor an unreferenced sealed report proves server acceptance.
    (tmp_path / "report.json").write_text(json.dumps(case.report()), encoding="utf-8")
    case.artifacts.put_json(cast(JsonValue, case.report()), ArtifactKind.ATTEMPT_REPORT)
    query = case.request("attempt_report_status", "fixed-status-query")
    for _ in range(3):
        missing = await case.service.execute(
            case.capability.token,
            query,
            operation_scope="runtime",
        )
        assert missing.result == {"status": "missing"}
    assert (
        case.control.get_operation_artifact(
            case.attempt.id,
            "fixed-status-query",
            GatewayOperation.ATTEMPT_REPORT_STATUS,
        )
        is None
    )

    registered = await case.service.execute(
        case.capability.token,
        case.submission(),
        operation_scope="runtime",
    )
    expected = AttemptReportV12.model_validate(case.report()).model_dump(mode="json")
    assert isinstance(registered.result, dict)
    digest = registered.result["report_artifact_digest"]
    for _ in range(3):
        accepted = await case.service.execute(
            case.capability.token,
            query,
            operation_scope="runtime",
        )
        assert accepted.result == {
            "status": "accepted",
            "report": expected,
            "report_artifact_digest": digest,
        }
        assert accepted.evaluation is None and accepted.kernel_artifact_digest is None
    assert accepted.result_artifact_digest != missing.result_artifact_digest
    assert case.adapter.requests == []
    assert (
        len(
            case.control.list_operation_artifacts(
                (case.attempt.id,),
                GatewayOperation.ATTEMPT_REPORT,
                recovery_generation=0,
            )
        )
        == 1
    )


@pytest.mark.anyio
async def test_only_committed_acceptance_counts_and_interrupted_submission_replays(
    case: Case,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = case.control.commit_operation_artifact

    def interrupted(*_args: Any, **_kwargs: Any) -> Any:
        raise InfrastructureError("interrupted before acceptance commit")

    monkeypatch.setattr(case.control, "commit_operation_artifact", interrupted)
    with pytest.raises(InfrastructureError, match="before acceptance commit"):
        await case.service.execute(case.capability.token, case.submission())
    query = case.request("attempt_report_status", "status")
    missing = await case.service.execute(case.capability.token, query)
    assert missing.result == {"status": "missing"}
    monkeypatch.setattr(case.control, "commit_operation_artifact", original)
    first = await case.service.execute(case.capability.token, case.submission())
    assert await case.service.execute(case.capability.token, case.submission()) == first
    accepted = await case.service.execute(case.capability.token, query)
    assert isinstance(accepted.result, dict) and accepted.result["status"] == "accepted"
    assert case.adapter.requests == []


@pytest.mark.anyio
async def test_status_is_bound_to_attempt_and_current_authorized_generation(case: Case) -> None:
    await case.service.execute(case.capability.token, case.submission())
    query = case.request("attempt_report_status", "status")
    other = replace(case.attempt, id=new_attempt_id(), ordinal=2)
    case.registry.insert_attempt(other)
    policy = GatewayCapabilityPolicy(
        frozenset({GatewayOperation.EVALUATE}),
        1,
        NOW_DATETIME + timedelta(hours=1),
    )
    other_capability = case.control.issue(other.id, policy)
    other_query = json.loads(query)
    other_query["attempt_id"] = other.id
    with pytest.raises(PermissionError, match="invalid Gateway capability"):
        await case.service.execute(case.capability.token, json.dumps(other_query).encode())
    assert (
        await case.service.execute(
            other_capability.token,
            json.dumps(other_query).encode(),
        )
    ).result == {"status": "missing"}

    case.registry.record_infrastructure_failure(case.attempt.id, "session interrupted")
    case.registry.retry_attempt(case.attempt.id)
    with pytest.raises(PermissionError, match="stale recovery generation"):
        await case.service.execute(case.capability.token, query)
    current = case.control.issue(case.attempt.id, policy)
    assert current.recovery_generation == 1
    assert (await case.service.execute(current.token, query)).result == {"status": "missing"}
    assert (
        len(
            case.control.list_operation_artifacts(
                (case.attempt.id,),
                GatewayOperation.ATTEMPT_REPORT,
            )
        )
        == 1
    )
    assert (
        case.control.list_operation_artifacts(
            (case.attempt.id,),
            GatewayOperation.ATTEMPT_REPORT,
            recovery_generation=1,
        )
        == ()
    )
    await case.service.execute(current.token, case.submission())
    accepted = await case.service.execute(current.token, query)
    assert isinstance(accepted.result, dict) and accepted.result["status"] == "accepted"
    assert case.adapter.requests == []


@pytest.mark.anyio
@pytest.mark.parametrize("oversized", [False, True])
async def test_report_limit_counts_canonical_utf8_and_rejects_before_acceptance(
    case: Case,
    oversized: bool,
) -> None:
    report = case.report()
    report["analysis"] = "实测诊断结论" * 100
    expected = AttemptReportV12.model_validate(report).model_dump(mode="json")
    actual = len(canonical_json_bytes(expected))
    limit = actual - int(oversized)
    service = GatewayProxyService(
        case.control,
        case.artifacts,
        case.adapter,
        GatewayProxyLimits(64 * 1024, 8, 16 * 1024),
        case.registry,
        max_attempt_report_bytes=limit,
    )
    if oversized:
        with pytest.raises(ValueError, match=f"actual_bytes={actual}, max_bytes={limit}"):
            await service.execute(case.capability.token, case.submission(report=report))
    else:
        registered = await service.execute(case.capability.token, case.submission(report=report))
        assert isinstance(registered.result, dict) and registered.result["status"] == "registered"
    status = await service.execute(
        case.capability.token,
        case.request("attempt_report_status", "status"),
    )
    assert isinstance(status.result, dict)
    assert status.result["status"] == ("missing" if oversized else "accepted")
    assert case.adapter.requests == []


@pytest.mark.parametrize("limit", [0, -1, True, 1.5])
def test_report_byte_limit_must_be_a_positive_integer(case: Case, limit: Any) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        GatewayProxyService(
            case.control,
            case.artifacts,
            case.adapter,
            GatewayProxyLimits(64 * 1024, 8, 16 * 1024),
            case.registry,
            max_attempt_report_bytes=limit,
        )


@pytest.mark.anyio
@pytest.mark.parametrize("invalid", ["kind", "attempt", "model", "status", "digest"])
async def test_report_status_fails_closed_on_invalid_committed_artifacts(
    case: Case,
    invalid: str,
) -> None:
    report = case.report()
    kind = ArtifactKind.ATTEMPT_REPORT
    if invalid == "kind":
        kind = ArtifactKind.EVIDENCE
    elif invalid == "attempt":
        report["attempt_id"] = new_attempt_id()
    elif invalid == "model":
        report.pop("analysis")
    digest = case.artifacts.put_json(cast(JsonValue, report), kind)
    receipt = case.service._store_result_artifact(
        operation="attempt_report",
        status="completed",
        kernel_artifact_digest=None,
        authorization=case.control.authorize(
            case.capability,
            GatewayOperation.ATTEMPT_REPORT,
            idempotency_key="bad-receipt",
            request_digest=str(canonical_json_digest({"invalid": invalid})),
        ),
        job_id=None,
        evaluation=None,
        result={
            "status": "registered",
            "report_status": "pivot" if invalid == "status" else "blocked",
            "report_artifact_digest": "invalid" if invalid == "digest" else str(digest),
        },
    )
    case.control.authorize(
        case.capability,
        GatewayOperation.ATTEMPT_REPORT,
        idempotency_key="bad-receipt",
        request_digest=str(canonical_json_digest({"invalid": invalid})),
    )
    case.control.commit_operation_artifact(
        case.attempt.id,
        "bad-receipt",
        GatewayOperation.ATTEMPT_REPORT,
        receipt,
    )
    with pytest.raises(InfrastructureError, match="Accepted Attempt report Artifact is invalid"):
        await case.service.execute(
            case.capability.token,
            case.request("attempt_report_status", "status"),
        )
    assert case.adapter.requests == []


@pytest.mark.anyio
async def test_status_uses_runtime_query_endpoint_and_discovery_has_no_agent_fields(
    case: Case,
) -> None:
    query = case.request("attempt_report_status", "status")
    schema = gateway_agent_request_schema("attempt_report_status")
    assert schema["request_contract"] == "runtime-query"
    operations = cast(dict[str, Any], schema["operations"])
    assert operations["attempt_report_status"]["properties"] == {}
    assert operations["attempt_report_status"]["required"] == []
    with pytest.raises(ValidationError):
        _REQUEST_ADAPTER.validate_python({**json.loads(query), "report": case.report()})
    with pytest.raises(ValueError, match="requires /v1/runtime/queries"):
        await case.service.execute(case.capability.token, query, operation_scope="gateway")

    sent: list[dict[str, Any]] = []

    async def receive() -> dict[str, Any]:
        return {"type": "http.request", "body": query, "more_body": False}

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)

    app = GatewayProxyAsgiApp(case.service, GatewayProxyLimits(64 * 1024, 8, 16 * 1024))
    await app(
        {
            "type": "http",
            "method": "POST",
            "path": "/v1/runtime/queries",
            "headers": [(b"authorization", f"Bearer {case.capability.token}".encode())],
        },
        receive,
        send,
    )
    assert sent[0]["status"] == 200
    response = json.loads(sent[1]["body"])
    assert response["operation"] == "attempt_report_status"
    assert response["result"] == {"status": "missing"}
    assert case.adapter.requests == []


@pytest.mark.anyio
@pytest.mark.parametrize("completed,has_report_digest", [(False, True), (True, False)])
async def test_status_ignores_failed_and_historical_unsealed_receipts(
    case: Case,
    completed: bool,
    has_report_digest: bool,
) -> None:
    result: dict[str, JsonValue] = {"status": "registered", "report_status": "blocked"}
    if has_report_digest:
        result["report_artifact_digest"] = case.artifacts.put_json(
            cast(JsonValue, case.report()),
            ArtifactKind.ATTEMPT_REPORT,
        )
    receipt = case.service._store_result_artifact(
        operation="attempt_report",
        status="completed" if completed else "failed",
        kernel_artifact_digest=None,
        authorization=case.control.authorize(
            case.capability,
            GatewayOperation.ATTEMPT_REPORT,
            idempotency_key="old-receipt",
            request_digest=str(canonical_json_digest({"historical": True})),
        ),
        job_id=None,
        evaluation=None,
        result=result,
    )
    case.control.authorize(
        case.capability,
        GatewayOperation.ATTEMPT_REPORT,
        idempotency_key="old-receipt",
        request_digest=str(canonical_json_digest({"historical": True})),
    )
    case.control.commit_operation_artifact(
        case.attempt.id,
        "old-receipt",
        GatewayOperation.ATTEMPT_REPORT,
        receipt,
    )
    response = await case.service.execute(
        case.capability.token,
        case.request("attempt_report_status", "status"),
    )
    assert response.result == {"status": "missing"}
    assert case.adapter.requests == []
