"""End-to-end tests for the trusted Gateway Proxy service and ASGI adapter."""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from conftest import NOW, digest, seed_lineage
from pydantic import TypeAdapter, ValidationError

from atrex_runtime.artifacts.local import ArtifactKind, LocalArtifactStore
from atrex_runtime.domain.ids import new_attempt_id, new_epoch_id
from atrex_runtime.domain.models import Attempt, AttemptStatus, BranchRole, Dsl, Epoch, EpochStatus
from atrex_runtime.gateway import (
    CandidateDiffPolicy,
    GatewayCapability,
    GatewayCapabilityPolicy,
    GatewayOperation,
    GatewayProxyAsgiApp,
    GatewayProxyLimits,
    GatewayProxyService,
    RegistryCandidateDiffValidator,
    SqliteGatewayControl,
)
from atrex_runtime.gateway.protocol import EvaluationV2, GatewayProxyRequestV2
from atrex_runtime.gateway.proxy import GatewayAdapterRequest, GatewayAdapterResult
from atrex_runtime.gateway.result_metrics import gateway_result_sol_summary
from atrex_runtime.registry.sqlite import SqliteRegistry

NOW_DATETIME = datetime(2026, 8, 14, tzinfo=UTC)
_PROTOCOL_ADAPTER: TypeAdapter[GatewayProxyRequestV2] = TypeAdapter(GatewayProxyRequestV2)


@dataclass
class FakeGatewayAdapter:
    result: GatewayAdapterResult
    requests: list[GatewayAdapterRequest] = field(default_factory=list)

    async def execute(self, request: GatewayAdapterRequest) -> GatewayAdapterResult:
        self.requests.append(request)
        return self.result


def _insert_attempt(registry: SqliteRegistry) -> Attempt:
    seeded = seed_lineage(
        registry,
        evidence_checkpoint=digest("evidence"),
        challenger_count=0,
        attempts_per_trajectory=1,
    )
    epoch = Epoch(
        id=new_epoch_id(),
        lineage_id=seeded.lineage_id,
        number=1,
        active_kernel_agent_revision_id=seeded.active_revision_id,
        challenger_kernel_agent_revision_ids=(),
        starting_kernel_revision_id=seeded.baseline.id,
        evidence_checkpoint=digest("evidence"),
        challenger_count=0,
        trajectories_per_branch=1,
        attempts_per_trajectory=1,
        status=EpochStatus.RUNNING,
        winner_kernel_agent_revision_id=None,
        best_kernel_revision_id=None,
        created_at=NOW,
        completed_at=None,
    )
    registry.insert_epoch(epoch)
    attempt = Attempt(
        id=new_attempt_id(),
        epoch_id=epoch.id,
        branch=BranchRole.ACTIVE,
        challenger_ordinal=0,
        trajectory_ordinal=1,
        ordinal=1,
        kernel_agent_revision_id=seeded.active_revision_id,
        input_kernel_revision_id=seeded.baseline.id,
        attempt_evidence_digest=digest("attempt-evidence"),
        output_kernel_revision_id=None,
        accepted_as_branch_best=False,
        status=AttemptStatus.RUNNING,
        infrastructure_failures=0,
        recovery_generation=0,
        authority_started_at=NOW,
        failure_reason=None,
        created_at=NOW,
        completed_at=None,
    )
    registry.insert_attempt(attempt)
    return attempt


def _request(attempt: Attempt, *, path: str = "kernel.py") -> bytes:
    return json.dumps(
        {
            "schema_version": 2,
            "attempt_id": attempt.id,
            "idempotency_key": "evaluate-candidate-1",
            "operation": "evaluate",
            "candidate": {
                "files": [
                    {
                        "path": path,
                        "content_base64": base64.b64encode(b"def kernel(): pass\n").decode(),
                    }
                ]
            },
        }
    ).encode()


@pytest.mark.parametrize(
    ("operation", "fields", "needs_candidate"),
    [
        ("submit", {"payload_path": "payload.json"}, True),
        ("profile", {"level": "deep", "kernel_name": "kernel"}, True),
        ("dev", {"command": "nvidia-smi", "intent": "inspect"}, True),
        ("check", {"sanitize": "memcheck"}, True),
        ("sol", {"solution_path": "solution.json", "subset": "L1"}, True),
        ("disassemble", {"fmt": "ptx"}, True),
        ("poll", {"job_id": "job-1", "wait": True, "include_spec": True}, False),
        ("jobs", {"kind": "dev", "status": "running", "limit": 10}, False),
        ("cancel", {"job_id": "job-1"}, False),
        ("env", {"gpu": "H20", "capabilities": True}, False),
        ("health", {}, False),
        ("config", {}, False),
    ],
)
def test_protocol_v2_parses_every_additional_agate_command(
    operation: str,
    fields: dict[str, object],
    needs_candidate: bool,
) -> None:
    payload: dict[str, object] = {
        "schema_version": 2,
        "attempt_id": new_attempt_id(),
        "idempotency_key": f"{operation}-1",
        "operation": operation,
        **fields,
    }
    if needs_candidate:
        payload["candidate"] = {
            "files": [
                {
                    "path": "kernel.py",
                    "content_base64": base64.b64encode(b"source").decode(),
                }
            ]
        }

    parsed = _PROTOCOL_ADAPTER.validate_python(payload)

    assert parsed.operation == operation
    if operation == "sol":
        assert parsed.lock_clocks is True


def _service(
    tmp_path: Path,
    candidate_production: Any = None,
) -> tuple[
    SqliteRegistry,
    SqliteGatewayControl,
    Attempt,
    GatewayCapability,
    GatewayProxyService,
    FakeGatewayAdapter,
]:
    registry = SqliteRegistry(tmp_path / "registry.sqlite")
    attempt = _insert_attempt(registry)
    control = SqliteGatewayControl(
        tmp_path / "gateway.sqlite",
        registry,
        signing_key=b"p" * 32,
        clock=lambda: NOW_DATETIME,
    )
    capability = control.issue(
        attempt.id,
        GatewayCapabilityPolicy(
            frozenset(GatewayOperation),
            4,
            NOW_DATETIME + timedelta(hours=1),
        ),
    )
    adapter = FakeGatewayAdapter(
        GatewayAdapterResult(
            status="completed",
            result={"correct": True, "latency_us": 12.0, "backend": "fake"},
            evaluation=EvaluationV2(correct=True, latency_us=12.0),
        )
    )
    limits = GatewayProxyLimits(64 * 1024, 8, 16 * 1024)
    service = GatewayProxyService(
        control,
        LocalArtifactStore(tmp_path / "artifacts"),
        adapter,
        limits,
        registry,
        candidate_production=candidate_production,
        clock=lambda: NOW_DATETIME,
    )
    return registry, control, attempt, capability, service, adapter


@pytest.mark.anyio
async def test_proxy_runs_production_gate_before_agate(tmp_path: Path) -> None:
    @dataclass
    class RejectingPolicy:
        calls: list[tuple[object, object]] = field(default_factory=list)

        def validate(self, attempt_id: object, candidate_digest: object) -> None:
            self.calls.append((attempt_id, candidate_digest))
            raise ValueError("production gate rejected candidate: torch.matmul")

    policy = RejectingPolicy()
    registry, control, attempt, capability_value, service, adapter = _service(
        tmp_path,
        policy,
    )

    with pytest.raises(ValueError, match="production gate rejected"):
        await service.execute(capability_value.token, _request(attempt))

    assert len(policy.calls) == 1
    assert adapter.requests == []
    control.close()
    registry.close()


@pytest.mark.anyio
async def test_proxy_records_agent_evaluation_without_committing_outcome(tmp_path: Path) -> None:
    registry, control, attempt, capability_value, service, adapter = _service(tmp_path)

    response = await service.execute(capability_value.token, _request(attempt))

    assert response.status == "completed"
    assert response.evaluation == EvaluationV2(correct=True, latency_us=12.0)
    assert response.result == {"correct": True, "latency_us": 12.0, "backend": "fake"}
    assert response.candidate_artifact_digest is not None
    assert len(adapter.requests) == 1
    adapter_request = adapter.requests[0]
    assert adapter_request.candidate_path is not None
    assert (adapter_request.candidate_path / "kernel.py").read_text() == "def kernel(): pass\n"
    assert await control.get_outcome(attempt.id) is None
    evaluations = control.list_evaluations(attempt.id)
    assert len(evaluations) == 1
    assert str(evaluations[0].candidate_artifact_digest) == response.candidate_artifact_digest
    assert evaluations[0].source.value == "agent"
    assert evaluations[0].correct is True
    assert evaluations[0].latency_us == 12.0

    replay = await service.execute(capability_value.token, _request(attempt))
    assert replay == response
    assert len(adapter.requests) == 1
    gateway_events = [
        event
        for event in registry.list_runtime_events(after_sequence=0, limit=100)
        if event.kind.startswith("gateway.")
    ]
    assert [event.kind for event in gateway_events] == [
        "gateway.operation_submitted",
        "gateway.operation_completed",
        "gateway.evaluation_recorded",
    ]
    assert all(event.aggregate_id == attempt.id for event in gateway_events)
    assert all(event.payload["schema_version"] == 1 for event in gateway_events)
    assert gateway_events[-1].payload["correlation"] == {
        "attempt_id": attempt.id,
        "epoch_id": attempt.epoch_id,
        "lineage_id": registry.get_epoch(attempt.epoch_id).lineage_id,
        "campaign_id": registry.get_lineage(
            registry.get_epoch(attempt.epoch_id).lineage_id
        ).campaign_id,
    }
    control.close()
    registry.close()


@pytest.mark.anyio
async def test_proxy_persists_raw_private_result_but_returns_only_worker_projection(
    tmp_path: Path,
) -> None:
    registry, control, attempt, capability_value, service, adapter = _service(tmp_path)
    adapter.result = GatewayAdapterResult(
        status="completed",
        result={"result": {"shapes": {"opaque-0": {"input_kwargs": {"secret_size": 1048576}}}}},
        evaluation=EvaluationV2(correct=True, latency_us=12.0),
        worker_result={
            "all_pass": True,
            "latency_us_geomean": 12.0,
            "shape_ids_are_opaque": True,
        },
    )

    response = await service.execute(capability_value.token, _request(attempt))

    assert response.result == {
        "all_pass": True,
        "latency_us_geomean": 12.0,
        "shape_ids_are_opaque": True,
    }
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    raw = artifacts.verify(response.gateway_result_digest).payload_path / "value.json"
    assert "secret_size" in raw.read_text(encoding="utf-8")
    assert "secret_size" not in json.dumps(response.model_dump(mode="json"))
    control.close()
    registry.close()


@pytest.mark.anyio
async def test_proxy_seals_paired_profile_with_agent_evaluation(tmp_path: Path) -> None:
    registry, control, attempt, capability_value, service, adapter = _service(tmp_path)
    adapter.result = GatewayAdapterResult(
        status="completed",
        result={"correct": True, "latency_us": 12.0},
        evaluation=EvaluationV2(correct=True, latency_us=12.0),
        profile_result={
            "status": "succeeded",
            "result": {
                "kernels": [
                    {
                        "compute_sol_pct": 25.0,
                        "mem_sol_pct": 65.0,
                        "duration": 100.0,
                        "duration_unit": "us",
                    }
                ]
            },
        },
    )

    response = await service.execute(capability_value.token, _request(attempt))

    evaluation = control.list_evaluations(attempt.id)[0]
    assert response.gateway_result_digest == evaluation.gateway_result_digest
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    summary = gateway_result_sol_summary(artifacts, evaluation.gateway_result_digest)
    assert summary.percent == 65.0
    assert summary.source == "ncu-profile"
    control.close()
    registry.close()


@pytest.mark.anyio
async def test_proxy_rejects_protocol_v1_after_v2_command_expansion(tmp_path: Path) -> None:
    registry, control, attempt, capability_value, service, adapter = _service(tmp_path)
    payload = json.loads(_request(attempt))
    payload["schema_version"] = 1

    with pytest.raises(ValidationError):
        await service.execute(capability_value.token, json.dumps(payload).encode())

    assert adapter.requests == []
    control.close()
    registry.close()


@pytest.mark.anyio
async def test_proxy_rejects_unsafe_candidate_path_before_adapter(tmp_path: Path) -> None:
    registry, control, attempt, capability_value, service, adapter = _service(tmp_path)

    with pytest.raises(ValueError):
        await service.execute(capability_value.token, _request(attempt, path="../kernel.py"))
    assert adapter.requests == []
    control.close()
    registry.close()


@pytest.mark.anyio
async def test_proxy_authorizes_profile_poll_and_cancel_worker_paths(tmp_path: Path) -> None:
    registry, control, attempt, capability_value, service, adapter = _service(tmp_path)
    candidate = {
        "files": [
            {
                "path": "kernel.py",
                "content_base64": base64.b64encode(b"def kernel(): return 1\n").decode(),
            }
        ]
    }
    operations = [
        (
            {
                "schema_version": 2,
                "attempt_id": attempt.id,
                "idempotency_key": "profile-1",
                "operation": "profile",
                "candidate": candidate,
                "level": "deep",
                "kernel_regex": "kernel",
            },
            GatewayAdapterResult("queued", {"status": "running"}, job_id="job-1"),
        ),
        (
            {
                "schema_version": 2,
                "attempt_id": attempt.id,
                "idempotency_key": "poll-1",
                "operation": "poll",
                "job_id": "job-1",
            },
            GatewayAdapterResult("completed", {"status": "succeeded"}, job_id="job-1"),
        ),
        (
            {
                "schema_version": 2,
                "attempt_id": attempt.id,
                "idempotency_key": "cancel-1",
                "operation": "cancel",
                "job_id": "job-1",
            },
            GatewayAdapterResult("cancelled", {"status": "cancelled"}, job_id="job-1"),
        ),
    ]

    for payload, result in operations:
        adapter.result = result
        response = await service.execute(
            capability_value.token,
            json.dumps(payload).encode(),
        )
        assert response.operation == payload["operation"]
        assert response.status == result.status

    profile, poll, cancel = adapter.requests
    assert profile.operation is GatewayOperation.PROFILE
    assert profile.profile_level == "deep"
    assert profile.kernel_regex == "kernel"
    assert profile.candidate_path is not None
    assert poll.operation is GatewayOperation.POLL
    assert poll.job_id == "job-1"
    assert poll.candidate_path is None
    assert cancel.operation is GatewayOperation.CANCEL
    assert cancel.job_id == "job-1"
    control.close()
    registry.close()


def test_candidate_diff_policy_rejects_disallowed_or_unchanged_files(tmp_path: Path) -> None:
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    before = tmp_path / "before"
    allowed = tmp_path / "allowed"
    disallowed = tmp_path / "disallowed"
    for path in (before, allowed, disallowed):
        path.mkdir()
    (before / "kernel.py").write_text("BLOCK = 64\n", encoding="utf-8")
    (allowed / "kernel.py").write_text("BLOCK = 128\n", encoding="utf-8")
    (disallowed / "kernel.py").write_text("BLOCK = 128\n", encoding="utf-8")
    (disallowed / "notes.md").write_text("exfiltration channel\n", encoding="utf-8")
    before_digest = artifacts.put_directory(before, ArtifactKind.KERNEL)
    allowed_digest = artifacts.put_directory(allowed, ArtifactKind.KERNEL)
    disallowed_digest = artifacts.put_directory(disallowed, ArtifactKind.KERNEL)
    attempt_id = new_attempt_id()
    registry = SimpleNamespace(
        get_attempt=lambda value: SimpleNamespace(
            epoch_id="epoch-1", input_kernel_revision_id="k-1"
        ),
        get_epoch=lambda value: SimpleNamespace(lineage_id="lineage-1"),
        get_lineage=lambda value: SimpleNamespace(dsl=Dsl.TRITON),
        get_kernel_revision=lambda value: SimpleNamespace(artifact_digest=before_digest),
    )
    policy = CandidateDiffPolicy(
        {
            Dsl.CUDA: ("*.cu",),
            Dsl.TRITON: ("*.py",),
            Dsl.CUTEDSL: ("*.py",),
        },
        True,
    )
    validator = RegistryCandidateDiffValidator(cast(Any, registry), artifacts, policy)

    validator.validate(attempt_id, allowed_digest)
    with pytest.raises(ValueError, match="does not change"):
        validator.validate(attempt_id, before_digest)
    with pytest.raises(ValueError, match=r"notes\.md"):
        validator.validate(attempt_id, disallowed_digest)


@pytest.mark.anyio
async def test_asgi_endpoint_requires_bearer_and_returns_canonical_json(tmp_path: Path) -> None:
    registry, control, attempt, capability_value, service, _adapter = _service(tmp_path)
    limits = GatewayProxyLimits(64 * 1024, 8, 16 * 1024)
    app = GatewayProxyAsgiApp(service, limits)
    sent: list[dict[str, object]] = []
    received = False

    async def receive() -> dict[str, object]:
        nonlocal received
        assert not received
        received = True
        return {"type": "http.request", "body": _request(attempt), "more_body": False}

    async def send(message: dict[str, object]) -> None:
        sent.append(message)

    await app(
        {
            "type": "http",
            "method": "POST",
            "path": "/v1/operations",
            "headers": [(b"authorization", f"Bearer {capability_value.token}".encode())],
        },
        receive,
        send,
    )

    assert sent[0]["status"] == 200
    body = sent[1]["body"]
    assert isinstance(body, bytes)
    assert json.loads(body)["operation"] == "evaluate"
    control.close()
    registry.close()
