"""Regression for FA4's passing mutation checks being mistaken for violations.

The fixture retains one case from the first batch of archived raw Result
741db8da00eb7f49f336c1116184950d526fc1e6ea54b5d55e8c27f2a764569a
(2026-10-14 FA4 audit). Job metadata and private input seeds are omitted;
comparison records and verdicts are unchanged.
"""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient
from test_agate_gateway_adapter import FakeAgateClient, _adapter
from test_gateway_proxy import _request, _service

from atrex_runtime.gateway import GatewayProxyAsgiApp, GatewayProxyLimits
from atrex_runtime.gateway.agate import parse_agate_correctness
from atrex_runtime.gateway.batched_evaluate import ShapeBatchedEvaluateExecutor

FIXTURE = Path(__file__).parent / "fixtures" / "agate_correctness_mutation_checks.json"


def _job(shape_count: int = 1) -> dict:
    job = json.loads(FIXTURE.read_text())
    result = job["result"]
    for verdicts in result["passed"].values():
        verdict = verdicts["0"]
        verdicts.clear()
        verdicts.update({str(i): deepcopy(verdict) for i in range(shape_count)})
    shape = result["correctness"]["shapes"]["0"]
    result["correctness"]["shapes"] = {
        str(i): deepcopy(shape) for i in range(shape_count)
    }
    return job


def test_archived_passing_mutation_checks_are_not_violations() -> None:
    job = _job()
    case = job["result"]["correctness"]["shapes"]["0"]["cases"][0]
    assert case["unexpected_mutations"]
    assert all(record["passed"] is True for record in case["unexpected_mutations"])
    assert parse_agate_correctness(job, ("0",)) is True


@pytest.mark.parametrize("field", ["unexpected_mutations", "mutated_inputs", "outputs"])
@pytest.mark.parametrize("failure", ["passed", "error", "failed", "skipped"])
def test_comparison_failure_is_not_hidden_by_passing_shape_verdicts(
    field: str, failure: str,
) -> None:
    job = _job()
    case = job["result"]["correctness"]["shapes"]["0"]["cases"][0]
    record = case[field][-1]
    if failure == "passed":
        record["passed"] = False
    elif failure == "error":
        record["error"] = "input unexpectedly changed"
    else:
        record["status"] = failure
    assert parse_agate_correctness(job, ("0",)) is False


@pytest.mark.anyio
@pytest.mark.parametrize("batch_size", [1, 30])
@pytest.mark.parametrize("read_enabled", [False, True])
@pytest.mark.parametrize("failed_field", [None, "unexpected_mutations", "outputs"])
async def test_mutation_verdict_reaches_agent_across_all_shape_batches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, batch_size: int,
    read_enabled: bool, failed_field: str | None,
) -> None:
    registry, control, attempt, capability, service, _ = _service(tmp_path, tool_modules=())
    lineage_id = registry.get_epoch(attempt.epoch_id).lineage_id
    registry._connection.execute(
        "UPDATE lineages SET result_artifact_read_enabled = ?, tool_modules_json = '[]' "
        "WHERE id = ?", (int(read_enabled), lineage_id),
    )
    job = _job(30)
    shapes = job["result"]["correctness"]["shapes"]
    if failed_field is not None:
        # A failure in a later batch/case must survive aggregation, even if all
        # compile/shape verdicts and the first case still claim success.
        case = deepcopy(shapes["29"]["cases"][0])
        case[failed_field][-1]["passed"] = False
        shapes["29"]["cases"].append(case)
    client = FakeAgateClient(job)
    adapter, _, jobs = _adapter(tmp_path, client)
    context = adapter._contexts.context
    adapter._contexts.context = replace(
        context,
        contract=context.contract.model_copy(update={"shapes": {key: {} for key in shapes}}),
    )
    adapter._shape_batches = ShapeBatchedEvaluateExecutor(shape_batch_size=batch_size)
    monkeypatch.setattr(service, "_adapter", adapter)
    app = GatewayProxyAsgiApp(service, GatewayProxyLimits(65536, 8, 16384))
    request = json.loads(_request(attempt))
    request["mode"] = "correctness_only"
    request.pop("latency_prediction")
    headers = {"Authorization": f"Bearer {capability.token}"}
    correct = failed_field is None
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://runtime") as http:
            response = await http.post("/v1/operations", json=request, headers=headers)
            assert response.status_code == 200, response.text
            body = response.json()
            assert body["status"] == "completed"
            result = body["result"]
            assert result["correct"] is correct
            assert result["correctness"]["status"] == ("PASS" if correct else "FAIL")
            assert bool(result["failures"]) is not correct
            assert result["mode"] == "correctness_only"
            assert result["latency_us_geomean"] is None
            assert result["latency_us_by_shape"] == {}
            assert "unexpected_mutations" not in result  # No raw input/check collection.
            if failed_field is not None:
                assert any(d.get("check") == failed_field for d in result["diagnostics"])
            assert not control.list_evaluations(attempt.id)
            assert len(client.submitted) == 30 // batch_size
            assert all(kind == "eval" and payload["mode"] == "correctness_only"
                       for kind, payload in client.submitted)
            sent_shapes = [key for _, payload in client.submitted
                           for key in payload["reference"]["shapes"]]
            assert sorted(sent_shapes) == sorted(shapes)
            # Both the initial result and its durable projection carry the
            # verdict; turning result reads off must not affect correctness.
            read = await http.post(
                "/v1/runtime/queries", headers=headers, json={
                    "schema_version": 2, "attempt_id": attempt.id,
                    "idempotency_key": "read-mutation-verdict",
                    "operation": "result_artifact_read",
                    "result_artifact_digest": body["result_artifact_digest"],
                },
            )
            if read_enabled:
                assert read.status_code == 200, read.text
                assert read.json()["result"]["result"] == result
            else:
                assert read.status_code == 400, read.text
                assert "reads are disabled" in read.json()["detail"]
    finally:
        jobs.close()
        control.close()
        registry.close()
