"""Journal request repair diagnostics across the authenticated HTTP boundary."""
from __future__ import annotations

from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient
from test_gateway_proxy import _service

from atrex_runtime.gateway import GatewayProxyAsgiApp, GatewayProxyLimits


@pytest.mark.anyio
@pytest.mark.parametrize("experiments", [False, True])
@pytest.mark.parametrize(
    "agent_request,expected",
    [
        ({"direction_id": "direction_" + "a" * 32}, {"action": "missing"}),
        ({"action": None}, {"action": "string_type"}),
        ({"action": ["start"]}, {"action": "string_type"}),
        ({"action": "begin"}, {"action": "literal_error"}),
        ({"action": "start"}, {"direction_id": "missing"}),
        ({"action": "start", "typo": "private-value"},
         {"direction_id": "missing", "typo": "extra_forbidden"}),
        ({"action": "propose", "name": "private-value"},
         {key: "missing" for key in
          ("hypothesis", "rationale", "plan", "success_criteria", "stop_conditions")}),
    ],
)
async def test_direction_fields_reach_http_agent_without_losing_paths(
    tmp_path: Path, experiments: bool, agent_request: dict, expected: dict,
) -> None:
    modules = ("directions", "experiments") if experiments else ("directions",)
    registry, control, attempt, capability, service, _ = _service(tmp_path, tool_modules=modules)
    app = GatewayProxyAsgiApp(service, GatewayProxyLimits(65536, 8, 16384))
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://runtime") as http:
            response = await http.post(
                "/v1/runtime/journals",
                json={"schema_version": 2, "attempt_id": str(attempt.id),
                      "operation": "direction_update", "idempotency_key": "bad-direction",
                      "request": agent_request},
                headers={"Authorization": f"Bearer {capability.token}"},
            )
        assert response.status_code == 400, response.text
        body = response.json()
        assert body["error"] == "invalid_request"
        assert {x["path"]: x["code"] for x in body["issues"]} == {
            f"request.{key}": code for key, code in expected.items()
        }
        assert all(f"request.{key}" in body["detail"] for key in expected)
        assert "private-value" not in response.text
        assert not control.list_direction_events(attempt.id)
        assert not control.list_experiments(attempt.id)
    finally:
        control.close()
        registry.close()


@pytest.mark.anyio
@pytest.mark.parametrize("action", ["complete", "abandon", "block", "defer"])
@pytest.mark.parametrize("experiments", [False, True])
async def test_closure_missing_fields_respect_enabled_modules(
    tmp_path: Path, action: str, experiments: bool,
) -> None:
    modules = ("directions", "experiments") if experiments else ("directions",)
    registry, control, attempt, capability, service, _ = _service(tmp_path, tool_modules=modules)
    app = GatewayProxyAsgiApp(service, GatewayProxyLimits(65536, 8, 16384))
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://runtime") as http:
            response = await http.post(
                "/v1/runtime/journals",
                json={"schema_version": 2, "attempt_id": str(attempt.id),
                      "operation": "direction_update", "idempotency_key": "closure",
                      "request": {"action": action, "direction_id": "direction_" + "a" * 32}},
                headers={"Authorization": f"Bearer {capability.token}"},
            )
        assert response.status_code == 400
        expected = {"request.hypothesis_status"}
        if experiments:
            expected.add("request.supporting_experiment_ids")
        assert {x["path"] for x in response.json()["issues"]} == expected
        assert all(x["code"] == "missing" for x in response.json()["issues"])
        assert not control.list_direction_events(attempt.id)
    finally:
        control.close()
        registry.close()


@pytest.mark.anyio
@pytest.mark.parametrize(
    "agent_request,expected",
    [
        ({"action": "discard", "name": "n", "change": "c"},
         {"request.hypothesis", "request.before", "request.after"}),
        ({"action": "discard", "name": "n", "change": "c", "hypothesis": "h",
          "before": {}, "after": None}, {"request.before.result_artifact_digest"}),
        ({"action": "discard", "name": "n", "change": "c", "hypothesis": "h",
          "before": None, "after": {}}, {"request.after.result_artifact_digest"}),
    ],
)
async def test_experiment_missing_fields_include_nested_result_binding(
    tmp_path: Path, agent_request: dict, expected: set[str],
) -> None:
    registry, control, attempt, capability, service, _ = _service(
        tmp_path, tool_modules=("experiments",)
    )
    app = GatewayProxyAsgiApp(service, GatewayProxyLimits(65536, 8, 16384))
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://runtime") as http:
            response = await http.post(
                "/v1/runtime/journals",
                json={"schema_version": 2, "attempt_id": str(attempt.id),
                      "operation": "experiment_record", "idempotency_key": "experiment",
                      "request": agent_request},
                headers={"Authorization": f"Bearer {capability.token}"},
            )
        assert response.status_code == 400, response.text
        assert {x["path"] for x in response.json()["issues"]} == expected
        assert all(x["code"] == "missing" for x in response.json()["issues"])
        assert not control.list_experiments(attempt.id)
    finally:
        control.close()
        registry.close()


@pytest.mark.anyio
async def test_missing_action_can_be_repaired_and_started_without_closing(tmp_path: Path) -> None:
    registry, control, attempt, capability, service, _ = _service(
        tmp_path, tool_modules=("directions",)
    )
    app = GatewayProxyAsgiApp(service, GatewayProxyLimits(65536, 8, 16384))
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://runtime") as http:
            async def send(key, agent_request):
                return await http.post(
                    "/v1/runtime/journals",
                    json={"schema_version": 2, "attempt_id": str(attempt.id),
                          "operation": "direction_update", "idempotency_key": key,
                          "request": agent_request},
                    headers={"Authorization": f"Bearer {capability.token}"},
                )
            proposed = await send("propose", {
                "action": "propose", "name": "tuning", "hypothesis": "less traffic",
                "rationale": "bandwidth", "plan": ["measure"], "success_criteria": "faster",
                "stop_conditions": "no gain",
            })
            assert proposed.status_code == 200, proposed.text
            direction_id = proposed.json()["result"]["direction_id"]
            bad = await send("bad", {"direction_id": direction_id})
            assert bad.status_code == 400
            assert bad.json()["issues"][0]["path"] == "request.action"
            fixed = await send("start", {"action": "start", "direction_id": direction_id})
            assert fixed.status_code == 200, fixed.text
        assert [x["action"] for x in control.list_direction_events(attempt.id)] == [
            "propose", "start"
        ]
    finally:
        control.close()
        registry.close()
