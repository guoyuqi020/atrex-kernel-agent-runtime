"""Task-scoped environment queries through the authenticated Runtime boundary."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import timedelta
from http.client import InvalidURL
from pathlib import Path

import pytest
from atrex_gateway_client.errors import TransportError
from conftest import digest
from httpx import ASGITransport, AsyncClient
from test_agate_gateway_adapter import FakeAgateClient, _adapter, _contract
from test_gateway_proxy import NOW_DATETIME, _service
from test_gateway_report_barrier import _post_report, _report

from atrex_runtime.artifacts.local import ArtifactKind, LocalArtifactStore
from atrex_runtime.domain.ids import new_attempt_id
from atrex_runtime.gateway import (
    GatewayCapabilityPolicy,
    GatewayOperation,
    GatewayProxyAsgiApp,
    GatewayProxyLimits,
)
from atrex_runtime.gateway.contract import RegistryAgateEvaluationContextResolver
from atrex_runtime.gateway.control import BootstrapGatewaySubject
from atrex_runtime.gateway.retrying_client import RetryingAgateClient


def _env_request(attempt_id: str) -> dict:
    return {
        "schema_version": 2,
        "attempt_id": attempt_id,
        "idempotency_key": "task-environment",
        "operation": "env",
    }


@pytest.mark.anyio
@pytest.mark.parametrize("bootstrap", [False, True])
async def test_env_uses_task_selector_and_returns_both_details_and_capabilities(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    bootstrap: bool,
) -> None:
    registry, control, attempt, capability, service, _ = _service(tmp_path)
    client = FakeAgateClient({})
    adapter, _, jobs = _adapter(tmp_path, client)
    contract = _contract().model_copy(update={"agate_gpu": "L20D"})
    adapter._contexts.context = replace(
        adapter._contexts.context,
        hardware_target="sm_89",
        contract=contract,
    )
    if bootstrap:
        artifacts = LocalArtifactStore(tmp_path / "artifacts")
        contract_digest = artifacts.put_json(
            contract.model_dump(mode="json"),
            ArtifactKind.EVALUATION_CONTRACT,
        )
        lineage = registry.get_lineage(registry.get_epoch(attempt.epoch_id).lineage_id)
        subject = BootstrapGatewaySubject(
            attempt_id=new_attempt_id(),
            campaign_id=lineage.campaign_id,
            lineage_id=lineage.id,
            epoch_id=attempt.epoch_id,
            kernel_agent_revision_id=attempt.kernel_agent_revision_id,
            operator="vector_add",
            hardware_target="sm_89",
            dsl=lineage.dsl,
            evaluation_contract_digest=contract_digest,
            input_kernel_digest=digest("seed"),
            evidence_digest=digest("evidence"),
            created_at=NOW_DATETIME,
        )
        capability = control.issue_bootstrap(
            subject,
            GatewayCapabilityPolicy(
                frozenset({GatewayOperation.ENV}),
                4,
                NOW_DATETIME + timedelta(hours=1),
            ),
        )
        attempt = replace(attempt, id=subject.attempt_id)
        adapter._contexts = RegistryAgateEvaluationContextResolver(registry, artifacts, control)
    monkeypatch.setattr(service, "_adapter", adapter)
    try:
        async with AsyncClient(
            transport=ASGITransport(
                app=GatewayProxyAsgiApp(service, GatewayProxyLimits(65536, 8, 16384))
            ),
            base_url="http://runtime",
        ) as http:
            response = await http.post(
                "/v1/operations",
                json=_env_request(attempt.id),
                headers={"Authorization": f"Bearer {capability.token}"},
            )
        assert response.status_code == 200, response.text
        assert response.json()["result"] == {
            "gpu": "L20D",
            "env": {"gpu": "L20D", "arch": "sm_90"},
            "capabilities": {"gpu": "L20D", "frameworks": {"triton": {"available": True}}},
        }
        assert client.environment_calls == [
            ("detail", "L20D", False),
            ("capabilities", "L20D", False),
        ]
        assert not client.submitted
    finally:
        jobs.close()
        control.close()
        registry.close()


@pytest.mark.anyio
@pytest.mark.parametrize("extra", [{"gpu": "NVIDIA L20D"}, {"capabilities": True}, {"force": True}])
async def test_env_rejects_agent_parameters_before_adapter(
    tmp_path: Path,
    extra: dict,
) -> None:
    registry, control, attempt, capability, service, adapter = _service(tmp_path)
    try:
        async with AsyncClient(
            transport=ASGITransport(
                app=GatewayProxyAsgiApp(service, GatewayProxyLimits(65536, 8, 16384))
            ),
            base_url="http://runtime",
        ) as http:
            response = await http.post(
                "/v1/operations",
                json={**_env_request(attempt.id), **extra},
                headers={"Authorization": f"Bearer {capability.token}"},
            )
        assert response.status_code == 400, response.text
        assert response.json()["issues"][0]["code"] == "extra_forbidden"
        schema = response.json()["request_schema"]["operations"]["env"]
        assert not {"gpu", "capabilities", "force"}.intersection(schema["properties"])
        assert adapter.requests == []
    finally:
        control.close()
        registry.close()


@pytest.mark.anyio
@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("failing_method", ["get_env", "get_capabilities"])
async def test_invalid_url_fails_without_retry_and_releases_report_barrier(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    wrapped: bool,
    failing_method: str,
) -> None:
    registry, control, attempt, capability, service, _ = _service(tmp_path)
    client = FakeAgateClient({})
    adapter, _, jobs = _adapter(tmp_path, client)
    calls, delays = [], []

    def fail(gpu: str, force: bool = False) -> dict:
        calls.append((gpu, force))
        # A second call succeeds so a retry regression cannot hang the test.
        if len(calls) > 1:
            return {}
        if wrapped:
            raise TransportError(
                cause_class="InvalidURL",
                stage="request",
                errno=None,
                attempts=1,
                retryable=True,
            )
        raise InvalidURL("private URL containing secret query parameters")

    monkeypatch.setattr(client, failing_method, fail)
    adapter._client = RetryingAgateClient(client, sleeper=delays.append)
    monkeypatch.setattr(service, "_adapter", adapter)
    try:
        async with AsyncClient(
            transport=ASGITransport(
                app=GatewayProxyAsgiApp(service, GatewayProxyLimits(65536, 8, 16384))
            ),
            base_url="http://runtime",
        ) as http:
            response = await http.post(
                "/v1/operations",
                json=_env_request(attempt.id),
                headers={"Authorization": f"Bearer {capability.token}"},
            )
        assert response.status_code == 400, response.text
        assert "InvalidURL" in response.text and "agate_gpu" in response.text
        assert "not retryable" in response.text
        assert "private URL" not in response.text
        assert "gateway_unavailable" not in response.text
        assert len(calls) == 1 and not delays
        assert not control.list_evaluations(attempt.id)
        code, result = await _post_report(service, capability.token, _report(attempt))
        assert code == 200, json.dumps(result)
    finally:
        jobs.close()
        control.close()
        registry.close()
