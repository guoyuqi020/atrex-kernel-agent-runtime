"""Gateway accepts uncertain handoffs without promoting unsupported shared claims."""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from conftest import digest
from test_attempt_finding_evidence import MODULES, _finding
from test_attempt_report import _value
from test_gateway_journal_adoption import _History
from test_gateway_journal_adoption import history as history
from test_gateway_proxy import FakeGatewayAdapter, _request, _service

from atrex_runtime.domain.models import Attempt
from atrex_runtime.gateway import GatewayCapability, GatewayProxyService, SqliteGatewayControl
from atrex_runtime.gateway.protocol import GatewayProxyResponseV2
from atrex_runtime.gateway.proxy import GatewayAdapterResult
from atrex_runtime.registry.sqlite import SqliteRegistry


@dataclass
class ClaimEnvironment:
    registry: SqliteRegistry
    control: SqliteGatewayControl
    attempt: Attempt
    capability: GatewayCapability
    service: GatewayProxyService
    adapter: FakeGatewayAdapter
    modules: tuple[str, ...]


@pytest.fixture
def claim_environment(tmp_path: Path, request: pytest.FixtureRequest) -> Iterator[ClaimEnvironment]:
    modules = getattr(request, "param", ("directions", "experiments"))
    registry, control, attempt, capability, service, adapter = _service(
        tmp_path, tool_modules=modules
    )
    try:
        yield ClaimEnvironment(registry, control, attempt, capability, service, adapter, modules)
    finally:
        control.close()
        registry.close()


def _support(result: GatewayProxyResponseV2) -> dict[str, Any]:
    return {
        "kernel_artifact_digest": result.kernel_artifact_digest,
        "result_artifact_digests": [result.result_artifact_digest],
    }


async def _submit(
    service: GatewayProxyService,
    capability: GatewayCapability,
    attempt: Attempt,
    modules: tuple[str, ...],
    finding: dict[str, Any],
    *,
    experiments: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    report: dict[str, Any] = _value(attempt.id)
    report.update(
        status="pivot",
        final_candidate=None,
        profile_evidence=None,
        experiments=experiments or [],
        direction_events=[],
        tool_modules=list(modules),
        findings=[finding],
    )
    request = {
        "schema_version": 2,
        "attempt_id": attempt.id,
        "idempotency_key": "claim-report",
        "operation": "attempt_report",
        "candidate": json.loads(_request(attempt))["candidate"],
        "report": report,
    }
    response = await service.execute(
        capability.token, json.dumps(request).encode(), operation_scope="runtime"
    )
    receipt = response.result
    assert isinstance(receipt, dict)
    assert receipt["status"] == "registered"
    recovered = await service.execute(
        capability.token,
        json.dumps(
            {
                "schema_version": 2,
                "attempt_id": attempt.id,
                "idempotency_key": "claim-report-status",
                "operation": "attempt_report_status",
            }
        ).encode(),
        operation_scope="runtime",
    )
    assert isinstance(recovered.result, dict)
    assert recovered.result["status"] == "accepted"
    assert recovered.result.get("assessment_notes") == receipt.get("assessment_notes")
    sealed = recovered.result["report"]
    assert isinstance(sealed, dict)
    return receipt, sealed


async def _measure(
    env: ClaimEnvironment, operation: str, *, status: str = "completed"
) -> GatewayProxyResponseV2:
    request = json.loads(_request(env.attempt))
    request["operation"] = operation
    request["idempotency_key"] = f"claim-measurement-{operation}"
    if operation != "evaluate":
        request.pop("latency_prediction")
        env.adapter.result = GatewayAdapterResult(status=status, result={"diagnostic": "observed"})
    if operation == "dev":
        request.update(command="python -c 'print(12.0)'", intent="custom_harness")
    elif operation == "profile":
        request["level"] = "sol"
        env.adapter.result = GatewayAdapterResult(
            status=status,
            result={"status": "succeeded", "kernels": []},
            profile_result={"status": "succeeded", "kernels": []},
        )
    elif status != "completed":
        env.adapter.result = GatewayAdapterResult(status=status, result={"diagnostic": "failed"})
    return await env.service.execute(env.capability.token, json.dumps(request).encode())


@pytest.mark.anyio
@pytest.mark.parametrize("claim_environment", MODULES, indirect=True)
@pytest.mark.parametrize("assessment", ["unresolved", "supported", "refuted"])
async def test_missing_evidence_keeps_report_usable_but_never_certifies_a_claim(
    claim_environment: ClaimEnvironment, assessment: str
) -> None:
    env = claim_environment
    receipt, sealed = await _submit(
        env.service,
        env.capability,
        env.attempt,
        env.modules,
        _finding(
            claim="Output copies cannot account for the fixed overhead",
            scope="This candidate on the target GPU",
            assessment=assessment,
        ),
    )

    assert sealed["findings"][0]["assessment"] == "unresolved"
    assert sealed["findings"][0]["root_cause"] is None
    assert bool(receipt.get("assessment_notes")) is (assessment != "unresolved")
    assert env.adapter.requests == []


@pytest.mark.anyio
@pytest.mark.parametrize("claim_environment", MODULES, indirect=True)
async def test_real_result_can_support_a_scoped_claim_without_any_journal_module(
    claim_environment: ClaimEnvironment,
) -> None:
    env = claim_environment
    result = await _measure(env, "evaluate")
    receipt, sealed = await _submit(
        env.service,
        env.capability,
        env.attempt,
        env.modules,
        _finding(
            claim_kind="observation",
            claim="The candidate completed full evaluation",
            assessment="supported",
            scope="This exact kernel and this evaluation contract",
            supporting_results=[_support(result)],
        ),
    )

    assert sealed["findings"][0]["assessment"] == "supported"
    assert sealed["findings"][0]["supporting_experiment_ids"] == []
    assert not receipt.get("assessment_notes")


@pytest.mark.anyio
@pytest.mark.parametrize("missing", ["claim", "scope"])
async def test_real_measurement_does_not_make_an_unscoped_conclusion_supported(
    claim_environment: ClaimEnvironment, missing: str
) -> None:
    env = claim_environment
    result = await _measure(env, "evaluate")
    finding = _finding(
        claim_kind="implementation_outcome",
        claim="The tested candidate achieved the reported latency",
        assessment="supported",
        scope="This exact kernel and evaluation contract",
        supporting_results=[_support(result)],
    )
    finding.pop(missing)
    receipt, sealed = await _submit(env.service, env.capability, env.attempt, env.modules, finding)

    assert sealed["findings"][0]["assessment"] == "unresolved"
    assert receipt["assessment_notes"]


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("operation", "claim_kind", "eligible"),
    [
        ("check", "observation", True),
        ("check", "implementation_outcome", False),
        ("check", "causal_hypothesis", False),
        ("dev", "implementation_outcome", True),
        ("dev", "causal_hypothesis", True),
        ("profile", "observation", True),
        ("profile", "implementation_outcome", False),
        ("profile", "causal_hypothesis", True),
        ("evaluate", "implementation_outcome", True),
        ("evaluate", "causal_hypothesis", True),
    ],
)
async def test_evidence_eligibility_depends_on_claim_kind_and_operation(
    claim_environment: ClaimEnvironment, operation: str, claim_kind: str, eligible: bool
) -> None:
    env = claim_environment
    result = await _measure(env, operation)
    receipt, sealed = await _submit(
        env.service,
        env.capability,
        env.attempt,
        env.modules,
        _finding(
            claim_kind=claim_kind,
            claim="The scoped test supports this declared interpretation",
            assessment="refuted",
            scope="This candidate and this measured workload only",
            supporting_results=[_support(result)],
        ),
    )

    assert sealed["findings"][0]["assessment"] == ("refuted" if eligible else "unresolved")
    assert bool(receipt.get("assessment_notes")) is not eligible


@pytest.mark.anyio
@pytest.mark.parametrize("claim_kind", ["implementation_outcome", "causal_hypothesis"])
async def test_correctness_only_cannot_settle_performance_or_causal_claims(
    claim_environment: ClaimEnvironment, claim_kind: str
) -> None:
    env = claim_environment
    request = json.loads(_request(env.attempt))
    request.update(mode="correctness_only")
    request.pop("latency_prediction")
    env.adapter.result = GatewayAdapterResult(
        status="completed",
        result={"correct": True},
        worker_result={"correct": True, "correctness": {"status": "PASS"}},
    )
    result = await env.service.execute(env.capability.token, json.dumps(request).encode())
    receipt, sealed = await _submit(
        env.service,
        env.capability,
        env.attempt,
        env.modules,
        _finding(
            claim_kind=claim_kind,
            claim="The candidate improves the measured runtime",
            assessment="supported",
            scope="This measured workload",
            supporting_results=[_support(result)],
        ),
    )

    assert sealed["findings"][0]["assessment"] == "unresolved"
    assert receipt["assessment_notes"]


@pytest.mark.anyio
@pytest.mark.parametrize("status", ["failed", "cancelled"])
async def test_infrastructure_failure_cannot_refute_a_causal_hypothesis(
    claim_environment: ClaimEnvironment, status: str
) -> None:
    env = claim_environment
    result = await _measure(env, "dev", status=status)
    receipt, sealed = await _submit(
        env.service,
        env.capability,
        env.attempt,
        env.modules,
        _finding(
            claim="Output writes dominate this candidate's runtime",
            assessment="refuted",
            scope="This candidate and target GPU",
            supporting_results=[_support(result)],
        ),
    )

    assert sealed["findings"][0]["assessment"] == "unresolved"
    assert receipt["assessment_notes"]


@pytest.mark.anyio
@pytest.mark.parametrize("claim_environment", [("experiments",)], indirect=True)
async def test_successful_baseline_cannot_launder_a_failed_experiment_outcome(
    claim_environment: ClaimEnvironment,
) -> None:
    env = claim_environment
    before = await _measure(env, "evaluate")
    failed = await _measure(env, "dev", status="failed")
    recorded = await env.service.execute(
        env.capability.token,
        json.dumps(
            {
                "schema_version": 2,
                "attempt_id": env.attempt.id,
                "idempotency_key": "failed-experiment",
                "operation": "experiment_record",
                "request": {
                    "name": "Investigate output cost",
                    "hypothesis": "Output copies account for the measured overhead",
                    "change": "Run a targeted diagnostic",
                    "before": {"result_artifact_digest": before.result_artifact_digest},
                    "after": {"result_artifact_digest": failed.result_artifact_digest},
                    "evidence": "The baseline ran; the diagnostic failed before producing data",
                    "analysis": "The diagnostic has not resolved the hypothesis",
                    "action": "abandon_direction",
                },
            }
        ).encode(),
        operation_scope="journal",
    )
    assert isinstance(recorded.result, dict)
    receipt, sealed = await _submit(
        env.service,
        env.capability,
        env.attempt,
        env.modules,
        _finding(
            claim="The diagnostic refutes the output bottleneck hypothesis",
            assessment="refuted",
            scope="This candidate and measured workload",
            supporting_experiment_ids=[recorded.result["experiment_id"]],
        ),
        experiments=list(env.control.list_experiments(env.attempt.id)),
    )

    assert sealed["findings"][0]["assessment"] == "unresolved"
    assert receipt["assessment_notes"]


@pytest.mark.anyio
@pytest.mark.parametrize("corruption", ["missing_result", "wrong_kernel"])
async def test_invalid_evidence_references_are_rejected_even_for_unresolved_claims(
    claim_environment: ClaimEnvironment, corruption: str
) -> None:
    env = claim_environment
    result = await _measure(env, "evaluate")
    support = _support(result)
    if corruption == "missing_result":
        support["result_artifact_digests"] = [digest("nonexistent-result")]
    else:
        support["kernel_artifact_digest"] = digest("wrong-kernel")

    with pytest.raises(ValueError):
        await _submit(
            env.service,
            env.capability,
            env.attempt,
            env.modules,
            _finding(supporting_results=[support]),
        )


@pytest.mark.anyio
async def test_visible_historical_measurement_supports_claim_without_rerunning_it(
    history: _History,
) -> None:
    before_count = len(history.adapter.requests)
    receipt, sealed = await _submit(
        history.service,
        history.capability,
        history.current,
        ("directions", "experiments"),
        _finding(
            claim_kind="observation",
            claim="The historical kernel passed this recorded evaluation",
            assessment="supported",
            scope="The historical kernel and its sealed evaluation contract",
            supporting_results=[_support(history.after)],
        ),
    )

    assert sealed["findings"][0]["assessment"] == "supported"
    assert not receipt.get("assessment_notes")
    assert len(history.adapter.requests) == before_count


@pytest.mark.anyio
@pytest.mark.parametrize("history", ["concurrent"], indirect=True)
async def test_existing_but_invisible_measurement_is_rejected(history: _History) -> None:
    with pytest.raises(ValueError):
        await _submit(
            history.service,
            history.capability,
            history.current,
            ("directions", "experiments"),
            _finding(supporting_results=[_support(history.after)]),
        )
