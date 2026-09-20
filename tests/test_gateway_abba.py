"""Evaluate comparisons seal both uploads without creating submission authority."""

from __future__ import annotations

import base64
import json
import math
from copy import deepcopy
from dataclasses import dataclass, field, replace
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest
from pydantic import TypeAdapter, ValidationError
from test_attempt_report import _value as _report_value
from test_gateway_proxy import NOW_DATETIME, _request, _service

from atrex_runtime.artifacts.local import ArtifactKind, LocalArtifactStore
from atrex_runtime.domain.errors import (
    DuplicateGatewayTaskError,
    InfrastructureError,
    InvalidTransitionError,
)
from atrex_runtime.domain.ids import ArtifactDigest, AttemptId, new_attempt_id
from atrex_runtime.domain.models import Attempt
from atrex_runtime.gateway.control_models import GatewayCapabilityPolicy, GatewayOperation
from atrex_runtime.gateway.protocol import (
    EvaluateRequestV2,
    EvaluationV2,
    GatewayProxyRequestV2,
    gateway_agent_request_schema,
)
from atrex_runtime.gateway.proxy import GatewayAdapterResult
from atrex_runtime.serialization import canonical_json_digest

BASELINE_SOURCE = "def kernel(): return 'baseline'\n"
CANDIDATE_SOURCE = "def kernel(): pass\n"


def _bundle(source: str, *, path: str = "kernel.py") -> dict[str, Any]:
    return {"files": [{"path": path, "content_base64": base64.b64encode(source.encode()).decode()}]}


def _abba_value(attempt: Attempt | None = None) -> dict[str, Any]:
    value = (
        json.loads(_request(attempt))
        if attempt is not None
        else {"attempt_id": new_attempt_id(), "candidate": _bundle(CANDIDATE_SOURCE)}
    )
    return {
        **value,
        "operation": "evaluate",
        "idempotency_key": "abba-uploaded-pair",
        "baseline": _bundle(BASELINE_SOURCE),
        "comparison": {"method": "abba", "repeats": 2},
    }


def _result(*, repeats: int = 2, input_scope: str = "contract") -> GatewayAdapterResult:
    return GatewayAdapterResult(
        "completed",
        result={"private_raw_result": "must not reach the Agent"},
        evaluation=None,
        worker_result={
            "correct": True,
            "mode": "full",
            "input_scope": input_scope,
            "comparison": {"method": "abba", "repeats": repeats},
            "repeats": 99,  # An obsolete upstream field must not become public again.
            "baseline": {
                "correct": True,
                "latency_us_geomean": 20.0,
                "latency_us_by_shape": {"0": 20.0},
            },
            "candidate": {
                "correct": True,
                "latency_us_geomean": 10.0,
                "latency_us_by_shape": {"0": 10.0},
            },
            "speedup": 2.0,
            "improvement_pct": 50.0,
            "baseline_kernel_artifact_digest": "untrusted-upstream-baseline",
            "kernel_artifact_digest": "untrusted-upstream-candidate",
            "reference_py": "private source must not reach the Agent",
            "input_py": "private input source must not reach the Agent",
            "shapes": {"0": {"private_shape": True}},
        },
    )


@dataclass
class RecordingPolicy:
    artifacts: LocalArtifactStore
    reject_source: str | None = None
    calls: list[tuple[AttemptId, ArtifactDigest, str]] = field(default_factory=list)

    def validate(self, attempt_id: AttemptId, candidate_digest: ArtifactDigest) -> None:
        artifact = self.artifacts.verify(candidate_digest)
        assert artifact.kind is ArtifactKind.KERNEL
        source = (artifact.payload_path / "kernel.py").read_text()
        self.calls.append((attempt_id, candidate_digest, source))
        if source == self.reject_source:
            raise ValueError("production gate rejected uploaded ABBA source")


@pytest.mark.anyio
@pytest.mark.parametrize(
    "parameters",
    [
        {},
        {
            "comparison": {"method": "abba", "repeats": 3},
            "input_py": "def _make_inputs(N): return [N]\n",
            "shapes": {"7": {"N": 32}},
        },
    ],
)
async def test_abba_seals_both_sources_and_replays_the_bound_public_result(
    tmp_path: Path, parameters: dict[str, Any]
) -> None:
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    policy = RecordingPolicy(artifacts)
    registry, control, attempt, capability, service, adapter = _service(tmp_path, policy)
    adapter.result = _result(
        repeats=parameters.get("comparison", {}).get("repeats", 2),
        input_scope="custom" if parameters else "contract",
    )
    try:
        request = {**_abba_value(attempt), **parameters}
        payload = json.dumps(request).encode()
        response = await service.execute(capability.token, payload)
        assert response.operation == "evaluate"
        assert response.status == "completed"
        assert response.evaluation is None
        assert len(adapter.requests) == 1
        forwarded = adapter.requests[0]
        assert forwarded.operation is GatewayOperation.EVALUATE
        assert forwarded.attempt_id == attempt.id
        assert forwarded.recovery_generation == 0
        assert forwarded.parameters == {
            "comparison": {"method": "abba", "repeats": 2},
            **parameters,
        }
        assert forwarded.baseline_candidate_digest is not None
        assert forwarded.candidate_digest is not None
        assert forwarded.baseline_candidate_digest != forwarded.candidate_digest
        for artifact_digest, artifact_path, source in (
            (
                forwarded.baseline_candidate_digest,
                forwarded.baseline_candidate_path,
                BASELINE_SOURCE,
            ),
            (forwarded.candidate_digest, forwarded.candidate_path, CANDIDATE_SOURCE),
        ):
            sealed = artifacts.verify(artifact_digest)
            assert sealed.kind is ArtifactKind.KERNEL
            assert sealed.payload_path == artifact_path
            assert (sealed.payload_path / "kernel.py").read_text() == source
        assert policy.calls == [
            (attempt.id, forwarded.baseline_candidate_digest, BASELINE_SOURCE),
            (attempt.id, forwarded.candidate_digest, CANDIDATE_SOURCE),
        ]
        assert response.kernel_artifact_digest == forwarded.candidate_digest
        assert isinstance(response.result, dict)
        assert response.result["correct"] is True
        assert response.result["comparison"] == request["comparison"]
        assert "repeats" not in response.result
        assert response.result["speedup"] == pytest.approx(2.0)
        assert response.result["baseline"] == {
            "correct": True,
            "latency_us_geomean": 20.0,
            "latency_us_by_shape": {"0": 20.0},
        }
        assert response.result["candidate"] == {
            "correct": True,
            "latency_us_geomean": 10.0,
            "latency_us_by_shape": {"0": 10.0},
        }
        assert response.result["measurement_aggregation"] == {
            "repetitions": 1,
            "method": "single_measurement",
        }
        assert response.result["baseline_kernel_artifact_digest"] == (
            forwarded.baseline_candidate_digest
        )
        assert response.result["kernel_artifact_digest"] == forwarded.candidate_digest
        assert "reference_py" not in response.result
        assert "input_py" not in response.result
        assert "shapes" not in response.result
        assert "private_raw_result" not in response.result
        assert response.result_artifact_digest is not None
        public = artifacts.verify(response.result_artifact_digest)
        assert public.kind is ArtifactKind.RESULT_ARTIFACT
        assert json.loads((public.payload_path / "value.json").read_text()) == {
            "operation": "evaluate",
            "status": "completed",
            "result": response.result,
        }
        assert json.loads((public.payload_path / "metadata.json").read_text())["evaluation"] is None
        assert await service.execute(capability.token, payload) == response
        assert len(adapter.requests) == 1
        assert len(policy.calls) == 2

        changed = deepcopy(request)
        changed["baseline"] = _bundle("def kernel(): return 'changed baseline'\n")
        with pytest.raises(InvalidTransitionError, match="different request"):
            await service.execute(capability.token, json.dumps(changed).encode())
        assert len(adapter.requests) == 1
        assert control.list_evaluations(attempt.id) == ()
        assert len(control.list_measurements((attempt.id,))) == 2
        assert await control.get_outcome(attempt.id) is None
        events = [
            event
            for event in registry.list_runtime_events(after_sequence=0, limit=100)
            if event.kind.startswith("gateway.")
        ]
        assert [event.kind for event in events] == [
            "gateway.operation_submitted",
            "gateway.operation_completed",
        ]
        assert all(
            event.payload["baseline_kernel_artifact_digest"] == forwarded.baseline_candidate_digest
            and event.payload["kernel_artifact_digest"] == forwarded.candidate_digest
            for event in events
        )

        report = {
            "schema_version": 2,
            "attempt_id": attempt.id,
            "idempotency_key": "report-after-abba-only",
            "operation": "attempt_report",
            "candidate": request["candidate"],
            "report": _report_value(attempt.id),
        }
        with pytest.raises(ValueError, match=r"No Agent evaluate|comparison|non-authoritative"):
            await service.execute(
                capability.token, json.dumps(report).encode(), operation_scope="runtime"
            )
    finally:
        control.close()
        registry.close()


@pytest.mark.anyio
async def test_abba_result_exposes_both_sources_but_not_unrelated_artifacts(tmp_path: Path) -> None:
    registry, control, attempt, capability, service, adapter = _service(tmp_path)
    adapter.result = _result()
    try:
        compared = await service.execute(
            capability.token, json.dumps(_abba_value(attempt)).encode()
        )
        result_read = await service.execute(
            capability.token,
            json.dumps(
                {
                    "attempt_id": attempt.id,
                    "idempotency_key": "read-abba-result",
                    "operation": "result_artifact_read",
                    "result_artifact_digest": compared.result_artifact_digest,
                }
            ).encode(),
            operation_scope="runtime",
        )
        assert isinstance(result_read.result, dict)
        public = result_read.result["result"]
        assert public == compared.result
        assert isinstance(public, dict)
        for role, digest_field, source in (
            ("baseline", "baseline_kernel_artifact_digest", BASELINE_SOURCE),
            ("candidate", "kernel_artifact_digest", CANDIDATE_SOURCE),
        ):
            source_read = await service.execute(
                capability.token,
                json.dumps(
                    {
                        "attempt_id": attempt.id,
                        "idempotency_key": f"read-abba-{role}",
                        "operation": "kernel_artifact_read",
                        "kernel_artifact_digest": public[digest_field],
                        "file": "kernel.py",
                    }
                ).encode(),
                operation_scope="runtime",
            )
            assert isinstance(source_read.result, dict)
            assert source_read.result["content"] == source
            assert source_read.result["kernel_artifact_digest"] == public[digest_field]
        assert len(adapter.requests) == 1

        unrelated = tmp_path / "unrelated-kernel"
        unrelated.mkdir()
        (unrelated / "kernel.py").write_text("def kernel(): return 'unobserved'\n")
        artifacts = LocalArtifactStore(tmp_path / "artifacts")
        unrelated_digest = artifacts.put_directory(unrelated, ArtifactKind.KERNEL)
        with pytest.raises(ValueError, match="outside the visible Lineage history"):
            await service.execute(
                capability.token,
                json.dumps(
                    {
                        "attempt_id": attempt.id,
                        "idempotency_key": "read-unrelated-source",
                        "operation": "kernel_artifact_read",
                        "kernel_artifact_digest": unrelated_digest,
                        "file": "kernel.py",
                    }
                ).encode(),
                operation_scope="runtime",
            )
        assert len(adapter.requests) == 1
    finally:
        control.close()
        registry.close()


@pytest.mark.anyio
async def test_abba_uses_one_measurement_per_shape(tmp_path: Path) -> None:
    registry, control, attempt, capability, service, adapter = _service(tmp_path)

    def result(baseline: dict[str, float], candidate: dict[str, float]) -> GatewayAdapterResult:
        def side(values: dict[str, float]) -> dict[str, Any]:
            latency = math.exp(sum(math.log(value) for value in values.values()) / len(values))
            return {
                "correct": True,
                "latency_us_geomean": latency,
                "latency_us_by_shape": values,
            }

        a = side(baseline)
        b = side(candidate)
        return GatewayAdapterResult(
            "completed",
            result={"private": True},
            worker_result={
                "correct": True,
                "comparison": {"method": "abba", "repeats": 2},
                "baseline": a,
                "candidate": b,
                "speedup": a["latency_us_geomean"] / b["latency_us_geomean"],
                "measurements": [],
            },
        )

    adapter.queued_results = [
        result({"0": 10.0, "1": 100.0}, {"0": 8.0, "1": 80.0}),
        result({"0": 11.0, "1": 140.0}, {"0": 9.0, "1": 120.0}),
        result({"0": 9.0, "1": 105.0}, {"0": 7.0, "1": 84.0}),
    ]
    try:
        response = await service.execute(
            capability.token, json.dumps(_abba_value(attempt)).encode()
        )
        assert len(adapter.requests) == 1
        assert [request.measurement_repetition for request in adapter.requests] == [1]
        assert isinstance(response.result, dict)
        assert response.result["baseline"]["latency_us_by_shape"] == {
            "0": 10.0,
            "1": 100.0,
        }
        assert response.result["candidate"]["latency_us_by_shape"] == {
            "0": 8.0,
            "1": 80.0,
        }
        assert response.result["measurement_aggregation"] == {
            "repetitions": 1,
            "method": "single_measurement",
        }
        assert len(response.result["measurements"]) == 1
        records = control.list_measurements((attempt.id,), limit=50)
        assert len(records) == 4
        assert {record.point.shape_id for record in records} == {"0", "1"}
        assert len({record.kernel_artifact_digest for record in records}) == 2
    finally:
        control.close()
        registry.close()


@pytest.mark.anyio
async def test_failed_abba_with_a_false_verdict_is_persisted_and_replayed(tmp_path: Path) -> None:
    registry, control, attempt, capability, service, adapter = _service(tmp_path)
    adapter.result = GatewayAdapterResult(
        "failed",
        result={"private_diagnostics": "backend failure"},
        worker_result={
            "correct": False,
            "comparison": {"method": "abba", "repeats": 2},
            "error": {"category": "abba_execution_failed", "message": "ABBA run failed"},
        },
    )
    try:
        payload = json.dumps(_abba_value(attempt)).encode()
        response = await service.execute(capability.token, payload)
        assert response.status == "failed"
        assert response.operation == "evaluate"
        assert response.evaluation is None
        assert isinstance(response.result, dict)
        assert response.result["correct"] is False
        assert response.result["comparison"] == {"method": "abba", "repeats": 2}
        assert response.result["error"] == {
            "category": "abba_execution_failed",
            "message": "ABBA run failed",
        }
        assert response.result_artifact_digest is not None
        assert (
            control.get_operation_artifact(
                attempt.id, "abba-uploaded-pair", GatewayOperation.EVALUATE
            )
            == response.result_artifact_digest
        )
        assert await service.execute(capability.token, payload) == response
        assert len(adapter.requests) == 1
        assert control.list_evaluations(attempt.id) == ()
        assert control.list_measurements((attempt.id,)) == ()
        assert await control.get_outcome(attempt.id) is None
    finally:
        control.close()
        registry.close()


@pytest.mark.anyio
@pytest.mark.parametrize("source", [BASELINE_SOURCE, CANDIDATE_SOURCE])
async def test_abba_production_gate_rejects_either_upload_before_execution(
    tmp_path: Path, source: str
) -> None:
    policy = RecordingPolicy(LocalArtifactStore(tmp_path / "artifacts"), reject_source=source)
    registry, control, attempt, capability, service, adapter = _service(tmp_path, policy)
    try:
        with pytest.raises(ValueError, match="production gate rejected"):
            await service.execute(capability.token, json.dumps(_abba_value(attempt)).encode())
        assert policy.calls[-1][2] == source
        assert len(policy.calls) == (1 if source == BASELINE_SOURCE else 2)
        assert adapter.requests == []
        assert control.list_evaluations(attempt.id) == ()
    finally:
        control.close()
        registry.close()


@pytest.mark.anyio
async def test_completed_abba_is_not_reexecuted_after_attempt_recovery(tmp_path: Path) -> None:
    registry, control, attempt, capability, service, adapter = _service(tmp_path)
    adapter.result = _result()
    try:
        payload = json.dumps(_abba_value(attempt)).encode()
        await service.execute(capability.token, payload)
        registry.record_infrastructure_failure(attempt.id, "simulated worker interruption")
        registry.retry_attempt(attempt.id)
        recovered = control.issue(
            attempt.id,
            GatewayCapabilityPolicy(
                frozenset(GatewayOperation), 4, NOW_DATETIME + timedelta(hours=1)
            ),
        )
        with pytest.raises(DuplicateGatewayTaskError):
            await service.execute(recovered.token, payload)
        assert [request.recovery_generation for request in adapter.requests] == [0]
        assert control.list_evaluations(attempt.id) == ()
    finally:
        control.close()
        registry.close()


@pytest.mark.anyio
@pytest.mark.parametrize(
    "invalid_result",
    [
        replace(_result(), status="queued"),
        replace(_result(), evaluation=EvaluationV2(correct=True, latency_us=10)),
        replace(_result(), worker_result=None),
        replace(_result(), worker_result={}),
        replace(_result(), worker_result={"correct": "true"}),
    ],
)
async def test_abba_rejects_incomplete_or_authoritative_results_without_poisoning_replay(
    tmp_path: Path, invalid_result: GatewayAdapterResult
) -> None:
    registry, control, attempt, capability, service, adapter = _service(tmp_path)
    try:
        adapter.result = invalid_result
        payload = json.dumps(_abba_value(attempt)).encode()
        with pytest.raises(
            InfrastructureError,
            match=r"terminal exploratory comparison|Evaluate measurement",
        ):
            await service.execute(capability.token, payload)
        assert (
            control.get_operation_artifact(
                attempt.id, "abba-uploaded-pair", GatewayOperation.EVALUATE
            )
            is None
        )
        adapter.result = _result()
        response = await service.execute(capability.token, payload)
        assert response.status == "completed"
        assert len(adapter.requests) == 2
        assert control.list_evaluations(attempt.id) == ()
    finally:
        control.close()
        registry.close()


def test_abba_protocol_defaults_and_agent_schema_require_uploaded_sources() -> None:
    value = _abba_value()
    value["comparison"] = {"method": "abba"}
    request = TypeAdapter(GatewayProxyRequestV2).validate_python(value)
    assert isinstance(request, EvaluateRequestV2)
    assert request.comparison is not None
    assert request.comparison.method == "abba"
    assert request.comparison.repeats == 2
    assert request.mode == "full"
    assert request.is_contract_evaluation is False
    document: Any = gateway_agent_request_schema("evaluate")
    schema = document["operations"]["evaluate"]
    properties = schema["properties"]
    assert document["request_contract"] == "gateway-execute"
    assert "baseline" in document["runtime_owned_fields"]
    assert properties["operation"]["const"] == "evaluate"
    assert properties["mode"]["enum"] == ["full", "correctness_only"]
    assert "repeats" not in properties
    assert "baseline_path" not in properties
    comparison = properties["comparison"]
    if "anyOf" in comparison:
        comparison = next(item for item in comparison["anyOf"] if item.get("type") != "null")
    if "$ref" in comparison:
        comparison = schema["$defs"][comparison["$ref"].removeprefix("#/$defs/")]
    comparison_properties = comparison["properties"]
    assert comparison_properties["method"]["const"] == "abba"
    assert comparison_properties["repeats"]["default"] == 2
    assert comparison_properties["repeats"]["minimum"] == 2
    assert comparison_properties["repeats"]["maximum"] == 20
    assert comparison_properties["baseline_path"]["type"] == "string"
    assert "baseline_path" in comparison["required"]
    for path in ("candidate_path", "input_path", "shapes_path"):
        assert properties[path]["type"] == "string"
        assert properties[path]["minLength"] == 1
    assert "candidate_path" not in schema["required"]
    assert {"not": {"required": ["input_py", "input_path"]}} in schema["allOf"]
    assert {"not": {"required": ["shapes", "shapes_path"]}} in schema["allOf"]
    assert not {"baseline", "candidate", "attempt_id", "idempotency_key"} & properties.keys()
    assert "CandidateBundleV2" not in schema.get("$defs", {})
    assert schema["additionalProperties"] is False
    assert "abba" not in gateway_agent_request_schema()["operations"]


@pytest.mark.parametrize(
    "fields",
    [
        {"operation": "abba"},
        {"baseline_path": "baseline.py"},
        {"repeats": 2},
        {"repeats": 1},
        {"repeats": 21},
        {"comparison": {"method": "abba", "repeats": 1}},
        {"comparison": {"method": "abba", "repeats": 21}},
        {"comparison": {"method": "simple", "repeats": 2}},
        {"comparison": {"method": "abba", "baseline_path": "unexpanded.py"}},
        {"mode": "correctness_only"},
        {"mode": "performance_only"},
        {"input_py": " \n"},
        {"shapes": {}},
        {"shapes": {"not-a-shape-id": {}}},
        {"shapes": {"0": [32]}},
        {"trialId": "gtrial_" + "a" * 32},
        {"baseline_trial_id": "gtrial_" + "a" * 32},
        {"candidate_trial_id": "gtrial_" + "b" * 32},
        {"baseline": {"kernel_trial_id": "gtrial_" + "a" * 32}},
        {"reference_py": "overriding the trusted reference is forbidden"},
    ],
)
def test_abba_rejects_invalid_options_and_legacy_trial_inputs(fields: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        TypeAdapter(GatewayProxyRequestV2).validate_python({**_abba_value(), **fields})


@pytest.mark.parametrize("role", ["baseline", "candidate"])
@pytest.mark.parametrize(
    "path", ["", "/kernel.py", "../kernel.py", "a/../b.py", "a//b.py", "a\\b.py"]
)
def test_abba_rejects_unsafe_paths_in_either_bundle(role: str, path: str) -> None:
    value = _abba_value()
    value[role] = _bundle(BASELINE_SOURCE, path=path)
    with pytest.raises(ValidationError):
        EvaluateRequestV2.model_validate(value)


@pytest.mark.parametrize("role", ["baseline", "candidate"])
def test_abba_requires_both_nonempty_uploaded_bundles(role: str) -> None:
    value = _abba_value()
    del value[role]
    with pytest.raises(ValidationError):
        EvaluateRequestV2.model_validate(value)
    value[role] = {"files": []}
    with pytest.raises(ValidationError):
        EvaluateRequestV2.model_validate(value)


def test_baseline_upload_requires_an_explicit_comparison() -> None:
    value = _abba_value()
    del value["comparison"]
    with pytest.raises(ValidationError):
        EvaluateRequestV2.model_validate(value)


@pytest.mark.anyio
@pytest.mark.parametrize("explicit_defaults", [False, True])
async def test_plain_evaluate_keeps_the_old_request_hash_and_authority(
    tmp_path: Path, explicit_defaults: bool
) -> None:
    registry, control, attempt, capability, service, adapter = _service(tmp_path)
    try:
        legacy = json.loads(_request(attempt))
        control.authorize(
            capability,
            GatewayOperation.EVALUATE,
            idempotency_key=legacy["idempotency_key"],
            request_digest=str(canonical_json_digest(legacy)),
        )
        value = {
            **legacy,
            **({"comparison": None, "baseline": None} if explicit_defaults else {}),
        }
        response = await service.execute(capability.token, json.dumps(value).encode())
        assert response.evaluation is not None
        assert isinstance(response.result, dict)
        assert "comparison" not in response.result
        assert "baseline" not in response.result
        assert adapter.requests[0].parameters == {}
        assert adapter.requests[0].baseline_candidate_digest is None
        assert adapter.requests[0].baseline_candidate_path is None
        assert len(control.list_evaluations(attempt.id)) == 1
        assert control.list_measurements((attempt.id,))
    finally:
        control.close()
        registry.close()
