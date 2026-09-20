"""Exploratory Evaluate options must not become trusted submission evidence."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError
from test_attempt_report import _value as _report_value
from test_gateway_proxy import _request, _service

from atrex_runtime.artifacts.local import LocalArtifactStore
from atrex_runtime.domain.errors import InfrastructureError
from atrex_runtime.gateway.control_models import GatewayOperation
from atrex_runtime.gateway.protocol import (
    EvaluateParametersV2,
    gateway_agent_request_schema,
)
from atrex_runtime.serialization import canonical_json_digest


@pytest.mark.parametrize(
    "parameters",
    [
        {"mode": "correctness"},
        {"mode": "performance_only"},
        {"input_py": ""},
        {"input_py": " \n"},
        {"input_py": "界" * 50_000},
        {"shapes": {}},
        {"shapes": {"not-an-id": {"N": 16}}},
        {"shapes": {"0": [16]}},
        {"shapes": {"0": None}},
        {"reference_py": "override is forbidden"},
    ],
)
def test_invalid_evaluate_parameters_are_rejected(parameters: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        EvaluateParametersV2.model_validate(parameters)


def test_agent_schema_has_file_helpers_and_resolvable_json_shape_definitions() -> None:
    document: Any = gateway_agent_request_schema("evaluate")
    schema = document["operations"]["evaluate"]
    assert schema["properties"]["mode"]["enum"] == ["full", "correctness_only"]
    assert schema["properties"]["mode"]["default"] == "full"
    assert schema["properties"]["input_path"]["type"] == "string"
    assert schema["properties"]["shapes_path"]["type"] == "string"
    assert "_make_inputs(**input_kwargs)" in schema["properties"]["input_py"]["description"]
    assert "Model.forward" in schema["properties"]["input_py"]["description"]
    assert "input_kwargs" in schema["properties"]["shapes"]["description"]
    assert "init_kwargs" in schema["properties"]["shapes"]["description"]
    assert {"not": {"required": ["input_py", "input_path"]}} in schema["allOf"]
    assert {"not": {"required": ["shapes", "shapes_path"]}} in schema["allOf"]
    assert "CandidateBundleV2" not in schema.get("$defs", {})

    def check_refs(value: Any) -> None:
        if isinstance(value, dict):
            if "$ref" in value:
                assert value["$ref"].removeprefix("#/$defs/") in schema["$defs"]
            for child in value.values():
                check_refs(child)
        elif isinstance(value, list):
            for child in value:
                check_refs(child)

    check_refs(schema)


@pytest.mark.anyio
@pytest.mark.parametrize(
    "parameters",
    [
        {"input_py": "def _make_inputs(N): return [N]\n"},
        {"shapes": {"7": {"N": 32}}},
        {"mode": "correctness_only"},
        {
            "mode": "correctness_only",
            "input_py": "def _make_inputs(N): return [N]\n",
            "shapes": {"7": {"N": 32}},
        },
    ],
)
async def test_exploration_is_recorded_but_cannot_authorize_candidate_submission(
    tmp_path: Path, parameters: dict[str, Any]
) -> None:
    registry, control, attempt, capability, service, adapter = _service(tmp_path)
    standard_result = adapter.result
    mode = parameters.get("mode", "full")
    scope = "custom" if "input_py" in parameters or "shapes" in parameters else "contract"
    if mode == "correctness_only":
        adapter.result = replace(
            standard_result,
            evaluation=None,
            worker_result={
                "correct": True,
                "correctness": {"max_abs_err": 0.001},
                # Even an upstream response with extra timing fields must not publish them.
                "latency_us_geomean": 12.0,
                "latency_us_by_shape": {"7": 12.0},
            },
        )
    try:
        request = {**json.loads(_request(attempt)), **parameters}
        payload = json.dumps(request).encode()
        response = await service.execute(capability.token, payload)
        replay = await service.execute(capability.token, payload)
        assert replay == response
        assert len(adapter.requests) == 1
        assert all(request.parameters == parameters for request in adapter.requests)
        assert response.evaluation is None
        assert control.list_evaluations(attempt.id) == ()
        assert isinstance(response.result, dict)
        assert response.result["correct"] is True
        assert response.result["mode"] == mode
        assert response.result["input_scope"] == scope
        assert response.result["latency_us_geomean"] == (
            None if mode == "correctness_only" else 12.0
        )
        if mode == "correctness_only":
            assert response.result["latency_us_arith_mean"] is None
            assert response.result["latency_us_by_shape"] == {}
            assert response.result["correctness"] == {
                "status": "PASS",
                "rel_err": None,
                "max_abs_err": 0.001,
                "max_rel_err": None,
            }

        measurements = control.list_measurements((attempt.id,))
        assert len(measurements) == (1 if mode == "correctness_only" else 2)
        measurement = next(
            item for item in measurements if item.point.metrics.get("aggregate") is True
        )
        point = measurement.point
        assert point.metrics["correct"] is True
        assert point.metrics["mode"] == mode
        assert point.metrics["input_scope"] == scope
        if mode == "correctness_only":
            assert "latency_us" not in point.metrics
        assert "input_py" not in point.metrics

        artifacts = LocalArtifactStore(tmp_path / "artifacts")
        private = artifacts.verify(measurement.gateway_result_digest)
        recorded = json.loads((private.payload_path / "evaluation-request.json").read_text())
        assert recorded == {"mode": "full", **parameters, "input_scope": scope}
        trials = control.list_kernel_trials((attempt.id,))
        assert len(trials) == 1
        assert trials[0].kernel_artifact_digest == response.kernel_artifact_digest
        assert any(
            o.result_artifact_digest == response.result_artifact_digest
            for o in trials[0].observations
        )

        read = await service.execute(
            capability.token,
            json.dumps(
                {
                    "schema_version": 2,
                    "attempt_id": attempt.id,
                    "idempotency_key": "read-exploratory-result",
                    "operation": "result_artifact_read",
                    "result_artifact_digest": response.result_artifact_digest,
                }
            ).encode(),
            operation_scope="runtime",
        )
        assert isinstance(read.result, dict)
        assert read.result["result"] == response.result

        report = {
            "schema_version": 2,
            "attempt_id": attempt.id,
            "idempotency_key": "report-before-full-evaluate",
            "operation": "attempt_report",
            "candidate": request["candidate"],
            "report": _report_value(attempt.id),
        }
        with pytest.raises(ValueError, match="custom inputs or correctness_only"):
            await service.execute(
                capability.token, json.dumps(report).encode(), operation_scope="runtime"
            )
        adapter.result = standard_result
        full_request = json.loads(_request(attempt))
        full_request["idempotency_key"] = "full-contract-evaluate"
        full = await service.execute(capability.token, json.dumps(full_request).encode())
        assert full.evaluation is not None
        assert len(control.list_evaluations(attempt.id)) == 1
        assert isinstance(full.result, dict)
        assert "input_scope" not in full.result
        report["idempotency_key"] = "report-after-full-evaluate"
        accepted = await service.execute(
            capability.token, json.dumps(report).encode(), operation_scope="runtime"
        )
        assert isinstance(accepted.result, dict)
        assert accepted.result["status"] == "registered"
    finally:
        control.close()
        registry.close()


@pytest.mark.anyio
async def test_default_evaluate_preserves_existing_request_identity(tmp_path: Path) -> None:
    registry, control, attempt, capability, service, adapter = _service(tmp_path)
    try:
        value = json.loads(_request(attempt))
        control.authorize(
            capability,
            GatewayOperation.EVALUATE,
            idempotency_key="evaluate-candidate-1",
            request_digest=str(canonical_json_digest(value)),
        )
        await service.execute(capability.token, _request(attempt))
        assert adapter.requests[0].parameters == {}
    finally:
        control.close()
        registry.close()


@pytest.mark.anyio
@pytest.mark.parametrize("status", ["completed", "queued"])
async def test_correctness_only_incomplete_result_is_retryable(tmp_path: Path, status: Any) -> None:
    registry, control, attempt, capability, service, adapter = _service(tmp_path)
    adapter.result = replace(adapter.result, status=status, result={}, evaluation=None)
    try:
        payload = json.loads(_request(attempt))
        payload["mode"] = "correctness_only"
        with pytest.raises(InfrastructureError, match=r"no correctness verdict|did not complete"):
            await service.execute(capability.token, json.dumps(payload).encode())
        assert control.list_evaluations(attempt.id) == ()
        adapter.result = replace(adapter.result, status="completed", result={"correct": False})
        response = await service.execute(capability.token, json.dumps(payload).encode())
        assert isinstance(response.result, dict)
        assert response.result["correct"] is False
        assert response.result["mode"] == "correctness_only"
        assert control.list_evaluations(attempt.id) == ()
        assert control.list_measurements((attempt.id,))[0].point.metrics["correct"] is False
    finally:
        control.close()
        registry.close()
