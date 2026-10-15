"""Native Agate evidence survives SDK -> batches -> HTTP -> sealed result reread."""
import json
from copy import deepcopy

import pytest
from httpx import ASGITransport, AsyncClient
from test_agate_gateway_adapter import FakeAgateClient, _adapter, _successful_job
from test_agent_abba import Case, case  # noqa: F401 -- shared native ABBA fixture
from test_gateway_abba import _abba_value
from test_gateway_proxy import _request, _service

from atrex_runtime.gateway.evaluation_feedback import bound_feedback, evaluation_feedback
from atrex_runtime.gateway.proxy import GatewayProxyAsgiApp, GatewayProxyLimits

ERROR = 'RuntimeError: Undeclared input mutation: "kv_cache"; CUDA illegal memory access'
TIMING = {"runs": 602, "median_ms": 0.11072, "mean_ms": 0.111,
          "stddev_ms": 0.002, "cv": 0.018, "p20_ms": 0.109, "p80_ms": 0.112,
          "min_ms": 0.108, "max_ms": 0.12}
KERNELS = [{"name": "fa_kernel<128>", "calls": 1, "device_time_us": 104.6724}]


def add_evidence(payload, *, failed=False):
    for sid, shape in payload["correctness"]["shapes"].items():
        shape["input_artifact"] = {"seed": "PRIVATE_SEED"}
        if failed:
            shape["error"] = ERROR
            payload["passed"]["correctness"][sid] = {"status": "failed", "reason": ERROR}
    for shape in payload.get("performance", {}).get("shapes", {}).values():
        shape.update(timing_stats=TIMING, kernel_events=KERNELS, benchmark_mode="cuda_graph",
                     cache_flush_mb=256, capture_time_ms=5.1, input_artifact="PRIVATE_SEED")


@pytest.mark.anyio
@pytest.mark.parametrize("mode", ["full", "correctness_only"])
@pytest.mark.parametrize("custom", [False, True])
@pytest.mark.parametrize("failed", [False, True])
async def test_eval_http_preserves_evidence_and_custom_scope(tmp_path, mode, custom, failed):
    registry, control, attempt, capability, service, _ = _service(tmp_path)
    job = _successful_job()
    add_evidence(job["result"], failed=failed)
    adapter, _, jobs = _adapter(tmp_path, FakeAgateClient(job))
    service._adapter = adapter
    value = json.loads(_request(attempt))
    value["mode"] = mode
    if mode == "correctness_only":
        value.pop("latency_prediction")
    if custom:
        value.update(input_py="def _make_inputs(N): return [N]", shapes={"0": {"N": 32}})
    app = GatewayProxyAsgiApp(service, GatewayProxyLimits(64 * 1024, 8, 16 * 1024))
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as http:
            headers = {"Authorization": f"Bearer {capability.token}"}
            response = await http.post("/v1/operations", json=value, headers=headers)
            assert response.status_code == 200, response.text
            body = response.json()
            result = body["result"]
            assert result["input_scope"] == ("custom" if custom else "contract")
            assert result["correct"] is not failed
            assert "PRIVATE_SEED" not in response.text
            if failed:
                assert any(d["error"] == ERROR and d["shape_id"] == "0"
                           for d in result["diagnostics"])
                assert "hidden evaluator cases failed" not in response.text
            if mode == "full":
                timing = result["performance_by_shape"]["0"]
                assert timing["timing_stats"] == TIMING
                assert timing["kernel_events"] == KERNELS
                assert timing["cache_flush_mb"] == 256
            read = await http.post("/v1/runtime/queries", headers=headers, json={
                "schema_version": 2, "attempt_id": attempt.id, "idempotency_key": "read-evidence",
                "operation": "result_artifact_read",
                "result_artifact_digest": body["result_artifact_digest"],
            })
            assert read.status_code == 200, read.text
            replay = read.json()["result"]["result"]
            for key in ("diagnostics", "performance_by_shape", "mode", "input_scope"):
                assert replay.get(key) == result.get(key)
    finally:
        jobs.close()
        control.close()
        registry.close()


@pytest.mark.anyio
async def test_abba_http_keeps_each_side_each_run_and_readback(tmp_path, case: Case):  # noqa: F811 -- pytest fixture
    original = case.client._blocks
    def blocks(request):
        values = original(request)
        for block in values:
            for row in block["abba"]["runs"]:
                add_evidence(row["result"], failed=row["revision"] == "candidate")
        return values
    case.client._blocks = blocks
    registry, control, attempt, capability, service, _ = _service(tmp_path)
    service._adapter = case.adapter
    app = GatewayProxyAsgiApp(service, GatewayProxyLimits(64 * 1024, 8, 16 * 1024))
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as http:
            headers = {"Authorization": f"Bearer {capability.token}"}
            response = await http.post("/v1/operations", json=_abba_value(attempt), headers=headers)
            assert response.status_code == 200, response.text
            result = response.json()["result"]
            for row in result["measurements"]:
                assert row["performance_by_shape"]["0"]["kernel_events"] == KERNELS
                if row["side"] == "B":
                    assert any(d["error"] == ERROR for d in row["diagnostics"])
            assert "PRIVATE_SEED" not in response.text
            read = await http.post("/v1/runtime/queries", headers=headers, json={
                "schema_version": 2, "attempt_id": attempt.id, "idempotency_key": "read-abba",
                "operation": "result_artifact_read",
                "result_artifact_digest": response.json()["result_artifact_digest"],
            })
            assert read.status_code == 200, read.text
            assert read.json()["result"]["result"] == result
    finally:
        control.close()
        registry.close()


def test_feedback_is_bounded_errors_precede_timings_and_secrets_are_scrubbed():
    job = _successful_job()
    add_evidence(job["result"], failed=True)
    job["result"]["correctness"]["shapes"]["0"]["error"] = "Authorization: Bearer secret-token"
    value = evaluation_feedback(job, ["0"])
    assert "secret-token" not in json.dumps(value)
    assert set(value["performance_by_shape"]) == {"0"}
    enormous = deepcopy(value)
    enormous["performance_by_shape"] = {str(i): {"name": "k" * 8192} for i in range(100)}
    bounded = bound_feedback(enormous)
    assert bounded["diagnostics"] == value["diagnostics"]
    assert bounded["feedback_truncation"]["omitted_performance_shapes"] > 0
    assert len(json.dumps(bounded).encode()) < 263000


@pytest.mark.anyio
async def test_agent_abba_sdk_validation_is_actionable_http_result(tmp_path, case: Case):  # noqa: F811
    class Rejection(Exception):
        def __init__(self):
            self.status = 400
            self.error_class = "validation"
            self.payload = {"reason": "source_archive_invalid", "message": ERROR}
    def submit(kind, payload):
        case.client.requests.append(payload)
        raise Rejection()
    case.client.submit_job = submit
    registry, control, attempt, capability, service, _ = _service(tmp_path)
    service._adapter = case.adapter
    app = GatewayProxyAsgiApp(service, GatewayProxyLimits(65536, 8, 16384))
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as http:
            response = await http.post("/v1/operations", json=_abba_value(attempt),
                                       headers={"Authorization": f"Bearer {capability.token}"})
            assert response.status_code == 200, response.text
            assert response.json()["status"] == "failed"
            assert "source_archive_invalid" in response.text
            assert ERROR in json.dumps(response.json(), ensure_ascii=False).replace('\\"', '"')
            assert len(case.client.requests) <= 2
    finally:
        control.close()
        registry.close()
