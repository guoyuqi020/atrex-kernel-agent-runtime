"""HTTP Journal facts, minimal writing contracts and immutable report handoff."""

from __future__ import annotations

import base64
import json
from collections.abc import AsyncIterator
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import pytest
from test_attempt_report import _value
from test_gateway_proxy import FakeGatewayAdapter, _request, _service

from atrex_runtime.domain.models import Attempt
from atrex_runtime.gateway import GatewayProxyAsgiApp, GatewayProxyLimits, SqliteGatewayControl
from atrex_runtime.gateway.protocol import EvaluationV2
from atrex_runtime.gateway.proxy import GatewayAdapterResult


@dataclass
class Case:
    client: httpx.AsyncClient
    attempt: Attempt
    adapter: FakeGatewayAdapter
    control: SqliteGatewayControl
    modules: tuple[str, ...]
    sequence: int = 0

    async def call(self, operation: str, **fields: Any) -> dict[str, Any]:
        self.sequence += 1
        path = "/v1/runtime/journals"
        if operation in {"evaluate", "check"}:
            path = "/v1/operations"
        elif operation in {"attempt_report", "attempt_report_status"}:
            path = "/v1/runtime/queries"
        response = await self.client.post(
            path,
            json={
                "schema_version": 2,
                "attempt_id": self.attempt.id,
                "idempotency_key": f"summary-{self.sequence}",
                "operation": operation,
                **fields,
            },
        )
        assert response.status_code == 200, response.text
        return response.json()

    async def measure(
        self,
        values: dict[str, float],
        *,
        correct: bool = True,
        mode: str = "full",
        status: str = "completed",
        operation: str = "evaluate",
        **fields: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        candidate = json.loads(_request(self.attempt))["candidate"]
        candidate["files"][0]["content_base64"] = base64.b64encode(
            f"def kernel(): return {self.sequence}\n".encode()
        ).decode()
        self.adapter.result = GatewayAdapterResult(
            status=status,
            result={
                "correct": correct,
                "latency_us_by_shape": values,
                **(
                    {"error": "shape s2: numerical mismatch", "failures": ["shape s2"]}
                    if not correct
                    else {}
                ),
            },
            evaluation=EvaluationV2(correct=correct, latency_us=12.0 if correct else None),
        )
        options = (
            {"mode": mode, "latency_prediction": "improved"} if operation == "evaluate" else {}
        )
        result = await self.call(operation, candidate=candidate, **options, **fields)
        return result, candidate

    async def direction(self) -> str:
        value = (
            await self.call(
                "direction_update",
                request={
                    "action": "propose",
                    "name": "Memory requests",
                    "hypothesis": "Fewer requests may help",
                    "rationale": "Investigate memory layout",
                    "plan": ["Implement and measure"],
                    "success_criteria": "Correct and faster",
                    "stop_conditions": "Insufficient evidence",
                },
            )
        )["result"]
        direction = value["direction_id"]
        await self.call("direction_update", request={"action": "start", "direction_id": direction})
        return direction

    async def record(
        self,
        before: dict[str, Any] | None,
        after: dict[str, Any],
        **fields: Any,
    ) -> dict[str, Any]:
        request = {
            "name": "Request layout",
            "change": "Pair adjacent requests",
            "action": "keep_after",
            "before": {"result_artifact_digest": before["result_artifact_digest"]}
            if before
            else None,
            "after": {"result_artifact_digest": after["result_artifact_digest"]},
            **fields,
        }
        if "directions" not in self.modules:
            request["hypothesis"] = "Pairing requests may reduce traffic"
        return (await self.call("experiment_record", request=request))["result"]


@pytest.fixture
async def case(tmp_path: Path, request: pytest.FixtureRequest) -> AsyncIterator[Case]:
    modules = getattr(request, "param", ("experiments",))
    registry, control, attempt, capability, service, adapter = _service(
        tmp_path, tool_modules=modules
    )
    try:
        app = GatewayProxyAsgiApp(service, GatewayProxyLimits(1_048_576, 8, 16 * 1024))
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://runtime",
            headers={"Authorization": f"Bearer {capability.token}"},
        ) as client:
            yield Case(client, attempt, adapter, control, modules)
    finally:
        control.close()
        registry.close()


@pytest.mark.anyio
@pytest.mark.parametrize("case", [("experiments",), ("directions", "experiments")], indirect=True)
async def test_minimal_experiment_http_summary_snapshot_report_roundtrip(case: Case) -> None:
    direction = await case.direction() if "directions" in case.modules else None
    before, _ = await case.measure({"s0": 100.0, "s1": 100.0, "s2": 100.0})
    after, candidate = await case.measure({"s0": 80.0, "s1": 100.0, "s2": 110.0})
    extra = {"direction_id": direction} if direction else {}
    receipt = await case.record(before, after, **extra)
    summary = receipt["measurement_summary"]
    comparison = summary["comparison"]
    assert comparison["available"] is True
    assert comparison["method"] == "independent_evaluations"
    assert comparison["candidate_latency_us_geomean"] == pytest.approx((80 * 100 * 110) ** (1 / 3))
    assert comparison["baseline_latency_us_geomean"] == pytest.approx(100)
    assert comparison["per_shape_classification_counts"] == {
        "improved": 1,
        "retained": 1,
        "degraded": 1,
    }
    assert comparison["classification"] == "improved"
    assert "latency_prediction" not in json.dumps(summary)
    assert "input_py" not in json.dumps(summary)
    assert summary["after"][0]["result_artifact_digest"] == after["result_artifact_digest"]
    assert len(case.adapter.requests) == 2  # Journal operations never launch GPU work.
    loaded = (await case.call("experiment_load", experiment_id=receipt["experiment_id"]))["result"]
    assert loaded["measurement_summary"] == summary
    assert loaded["analysis"] is None and loaded["evidence"] is None
    assert loaded["hypothesis"] is None if direction else loaded["hypothesis"] is not None
    index = (await case.call("experiments_list"))["result"]["experiments"][0]
    assert "analysis" not in index and "evidence" not in index and "hypothesis" not in index
    assert index["measurement_summary"]["comparison"] == comparison
    if direction:
        await case.call(
            "direction_update",
            request={
                "action": "complete",
                "direction_id": direction,
                "hypothesis_status": "unresolved",
                "supporting_experiment_ids": [receipt["experiment_id"]],
            },
        )
    snapshot = (await case.call("journal_snapshot"))["result"]
    assert snapshot["experiments"][0]["measurement_summary"] == summary
    report = _value(case.attempt.id)
    report.update(
        experiments=snapshot["experiments"],
        direction_events=snapshot["direction_events"],
        findings=[],
        tool_modules=list(case.modules),
        profile_evidence=None,
    )
    forged = deepcopy(report)
    forged["experiments"][0]["measurement_summary"]["comparison"]["speedup"] = 100
    with pytest.raises(AssertionError, match="must match the Runtime-owned"):
        await case.call("attempt_report", report=forged, candidate=candidate)
    registered = (await case.call("attempt_report", report=report, candidate=candidate))["result"]
    assert registered["status"] == "registered"
    recovered = (await case.call("attempt_report_status"))["result"]["report"]
    assert recovered["findings"] == []
    assert recovered["experiments"][0]["measurement_summary"] == summary


@pytest.mark.anyio
@pytest.mark.parametrize(
    "variant,reason",
    [
        ("correctness_only", "requires_completed_full_evaluations"),
        ("incorrect", "both_sides_must_pass_correctness"),
        ("coverage", "shape_coverage_mismatch"),
        ("custom", "measurement_scope_mismatch"),
    ],
)
async def test_no_performance_verdict_for_incompatible_or_failed_evidence(
    case: Case,
    variant: str,
    reason: str,
) -> None:
    before, _ = await case.measure({"s0": 100.0, "s1": 200.0})
    after, _ = await case.measure(
        {"s0": 50.0} if variant == "coverage" else {"s0": 50.0, "s1": 100.0},
        correct=variant != "incorrect",
        mode="correctness_only" if variant == "correctness_only" else "full",
        **({"input_py": "def _make_inputs(): return {}"} if variant == "custom" else {}),
    )
    summary = (await case.record(before, after))["measurement_summary"]
    assert summary["comparison"]["available"] is False
    assert summary["comparison"]["reason"] == reason
    assert "classification" not in summary["comparison"]
    if variant == "incorrect":
        assert summary["after"][0]["error"] == "shape s2: numerical mismatch"


@pytest.mark.anyio
async def test_abba_summary_uses_own_a_b_not_experiment_before(case: Case) -> None:
    before, baseline = await case.measure({"s0": 1000})
    candidate = json.loads(_request(case.attempt))["candidate"]
    case.adapter.result = GatewayAdapterResult(
        status="completed",
        result={"correct": True},
        worker_result={
            "mode": "full",
            "input_scope": "contract",
            "correct": True,
            "baseline": {
                "correct": True,
                "latency_us_geomean": 100,
                "latency_us_by_shape": {"s0": 100},
            },
            "candidate": {
                "correct": True,
                "latency_us_geomean": 94,
                "latency_us_by_shape": {"s0": 94},
            },
        },
    )
    abba = await case.call(
        "evaluate",
        candidate=candidate,
        baseline=baseline,
        comparison={"method": "abba", "repeats": 2},
        latency_prediction="improved",
    )
    summary = (await case.record(before, abba))["measurement_summary"]
    comparison = summary["comparison"]
    assert summary["after"][0]["measured_shape_count"] == 1
    assert summary["after"][0]["latency_us_geomean"] == pytest.approx(94)
    assert comparison["available"] is True and comparison["method"] == "abba"
    assert comparison["baseline_latency_us_geomean"] == pytest.approx(100)
    assert comparison["latency_reduction_pct"] == pytest.approx(6)
    assert comparison["speedup"] == pytest.approx(100 / 94)
    assert comparison["repeats"] == 2
    assert comparison["baseline_kernel_artifact_digest"] == before["kernel_artifact_digest"]


@pytest.mark.anyio
@pytest.mark.parametrize("case", [("directions", "experiments")], indirect=True)
async def test_discard_never_closes_direction_or_promotes_diagnostic(case: Case) -> None:
    direction = await case.direction()
    result, _ = await case.measure({}, correct=False, operation="check", status="failed")
    summary = (await case.record(None, result, direction_id=direction, action="discard"))[
        "measurement_summary"
    ]
    assert summary["comparison"]["available"] is False
    assert summary["after"][0]["status"] == "failed"
    loaded = (await case.call("direction_load", direction_id=direction))["result"]
    assert loaded["status"] == "in_progress(self)"
    assert loaded["hypothesis_status"] == "unresolved"
    await case.call(
        "direction_update",
        request={
            "action": "defer",
            "direction_id": direction,
            "hypothesis_status": "supported",
            "supporting_experiment_ids": [],
        },
    )
    loaded = (await case.call("direction_load", direction_id=direction))["result"]
    assert loaded["hypothesis_status"] == "unresolved"


@pytest.mark.anyio
async def test_agent_cannot_forge_runtime_summary(case: Case) -> None:
    result, _ = await case.measure({"s0": 100})
    with pytest.raises(AssertionError, match=r"request\.measurement_summary.*extra_forbidden"):
        await case.record(result, result, measurement_summary={"comparison": {"speedup": 100}})
    assert not case.control.list_experiments(case.attempt.id)


@pytest.mark.anyio
async def test_boundary_one_percent_is_retained(case: Case) -> None:
    before, _ = await case.measure({"s0": 100, "s1": 100})
    after, _ = await case.measure({"s0": 99, "s1": 101})
    comparison = (await case.record(before, after))["measurement_summary"]["comparison"]
    assert comparison["per_shape_classification_counts"] == {
        "improved": 0,
        "retained": 2,
        "degraded": 0,
    }
    assert comparison["classification"] == "retained"


@pytest.mark.anyio
@pytest.mark.parametrize("case", [(), ("directions",)], indirect=True)
async def test_no_experiment_module_needs_no_duplicate_summary_or_finding(case: Case) -> None:
    direction = await case.direction() if "directions" in case.modules else None
    result, candidate = await case.measure({"s0": 10})
    if direction:
        await case.call(
            "direction_update",
            request={
                "action": "complete",
                "direction_id": direction,
                "hypothesis_status": "supported",
                "claim_kind": "implementation_outcome",
                "scope": "Exact evaluated Kernel and shapes",
                "supporting_results": [
                    {
                        "kernel_artifact_digest": result["kernel_artifact_digest"],
                        "result_artifact_digests": [result["result_artifact_digest"]],
                    }
                ],
            },
        )
        loaded = (await case.call("direction_load", direction_id=direction))["result"]
        assert loaded["hypothesis_status"] == "supported"
        assert loaded["analysis"] is None
    snapshot = (await case.call("journal_snapshot"))["result"]
    report = _value(case.attempt.id)
    report.update(
        experiments=[],
        direction_events=snapshot["direction_events"],
        findings=[],
        tool_modules=list(case.modules),
        profile_evidence=None,
    )
    receipt = (await case.call("attempt_report", report=report, candidate=candidate))["result"]
    assert receipt["status"] == "registered"


@pytest.mark.anyio
async def test_summary_reads_public_results_not_private_test_timings(case: Case) -> None:
    candidate = json.loads(_request(case.attempt))["candidate"]
    case.adapter.result = GatewayAdapterResult(
        status="completed",
        result={"correct": True, "latency_us_by_shape": {"hidden-test": 9000}},
        worker_result={"correct": True, "latency_us_by_shape": {"opaque-valid": 10}},
        evaluation=EvaluationV2(correct=True, latency_us=10),
    )
    result = await case.call("evaluate", candidate=candidate, latency_prediction="degraded")
    summary = (await case.record(result, result))["measurement_summary"]
    assert "hidden-test" not in json.dumps(summary) and "9000" not in json.dumps(summary)
    assert summary["after"][0]["shape_ids"] == ["opaque-valid"]
    assert summary["comparison"]["method"] == "same_result_reuse"
    assert summary["comparison"]["candidate_latency_us_geomean"] == pytest.approx(10)
    assert "latency_prediction" not in json.dumps(summary)
