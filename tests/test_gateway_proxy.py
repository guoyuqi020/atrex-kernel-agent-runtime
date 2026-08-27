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
from atrex_runtime.domain.errors import DirectionConcurrencyError, InvalidTransitionError
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
from atrex_runtime.gateway.protocol import (
    EvaluationV2,
    GatewayProxyRequestV2,
    gateway_agent_request_schema,
)
from atrex_runtime.gateway.proxy import (
    GatewayAdapterRequest,
    GatewayAdapterResult,
    _invalid_request_response,
)
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


def _insert_attempt(
    registry: SqliteRegistry,
    *,
    attempts_per_trajectory: int = 1,
) -> Attempt:
    seeded = seed_lineage(
        registry,
        evidence_checkpoint=digest("evidence"),
        challenger_count=0,
        attempts_per_trajectory=attempts_per_trajectory,
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
        attempts_per_trajectory=attempts_per_trajectory,
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


def test_attempt_runtime_state_checkpoint_tracks_latest_physical_session(tmp_path: Path) -> None:
    registry = SqliteRegistry(tmp_path / "runtime.db")
    attempt = _insert_attempt(registry)
    state = digest("attempt-runtime-state")

    registry.record_attempt_runtime_state(attempt.id, state)
    registry.record_attempt_runtime_state(attempt.id, state)

    assert registry.get_attempt(attempt.id).runtime_state_digest == state
    replacement = digest("other-runtime-state")
    registry.record_attempt_runtime_state(attempt.id, replacement)
    assert registry.get_attempt(attempt.id).runtime_state_digest == replacement
    registry.close()


def test_attempt_input_runtime_state_is_recorded_once(tmp_path: Path) -> None:
    registry = SqliteRegistry(tmp_path / "runtime.db")
    attempt = _insert_attempt(registry)
    input_state = digest("attempt-input-runtime-state")

    registry.record_attempt_input_runtime_state(attempt.id, input_state)
    registry.record_attempt_input_runtime_state(attempt.id, input_state)

    assert registry.get_attempt(attempt.id).input_runtime_state_digest == input_state
    with pytest.raises(InvalidTransitionError, match="cannot record Input Runtime State"):
        registry.record_attempt_input_runtime_state(attempt.id, digest("different-input-state"))
    registry.close()


@pytest.mark.parametrize(
    ("operation", "fields", "needs_candidate"),
    [
        ("profile", {"level": "deep", "kernel_name": "kernel"}, True),
        ("dev", {"command": "nvidia-smi", "intent": "inspect"}, True),
        ("check", {"sanitize": "memcheck"}, True),
        ("disassemble", {"fmt": "ptx"}, True),
        ("poll", {"job_id": "job-1", "wait": True, "include_spec": True}, False),
        ("jobs", {"kind": "dev", "status": "running", "limit": 10}, False),
        ("cancel", {"job_id": "job-1"}, False),
        ("env", {"gpu": "H20", "capabilities": True}, False),
        ("health", {}, False),
        ("config", {}, False),
        (
            "kernel_trial_show",
            {"kernel_trial_id": "gtrial_0123456789abcdef0123456789abcdef"},
            False,
        ),
        (
            "kernel_artifact_read",
            {"kernel_artifact_digest": "sha256:" + "a" * 64, "file": "kernel.py"},
            False,
        ),
        (
            "gateway_result_read",
            {"gateway_result_digest": "sha256:" + "b" * 64},
            False,
        ),
        ("direction_history", {}, False),
        ("experiment_history", {}, False),
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


def test_agent_request_schema_is_projected_from_live_gateway_model() -> None:
    document = gateway_agent_request_schema("profile")
    schema = cast(dict[str, Any], document["operations"])["profile"]
    properties = schema["properties"]

    assert document["gateway_protocol_version"] == 2
    assert document["runtime_owned_fields"] == ["attempt_id", "candidate", "schema_version"]
    assert "runtime_defaulted_fields" not in document
    assert properties["operation"]["const"] == "profile"
    assert properties["level"]["enum"] == ["survey", "sol", "deep"]
    assert properties["kernel_name"]["anyOf"][0]["type"] == "string"
    assert "candidate" not in properties
    assert "attempt_id" not in properties
    assert "idempotency_key" not in properties
    assert schema["additionalProperties"] is False

    runtime_document = gateway_agent_request_schema("kernel_trial_show")
    runtime_schema = cast(dict[str, Any], runtime_document["operations"])["kernel_trial_show"]
    assert runtime_document["request_contract"] == "runtime-query"
    assert runtime_document["runtime_owned_fields"] == [
        "attempt_id",
        "candidate",
        "operation",
        "schema_version",
    ]
    assert "operation" not in runtime_schema["properties"]
    journal_document = gateway_agent_request_schema("direction_update")
    journal_schema = cast(dict[str, Any], journal_document["operations"])["direction_update"]
    assert journal_document["request_contract"] == "runtime-journal"
    assert "operation" not in journal_schema["properties"]
    assert "request" in journal_schema["required"]
    all_operations = cast(dict[str, Any], gateway_agent_request_schema()["operations"])
    assert "submit" not in all_operations
    assert "sol" not in all_operations
    assert "measurements" not in all_operations
    assert "kernel_trials" not in all_operations


def _service(
    tmp_path: Path,
    candidate_production: Any = None,
    *,
    attempts_per_trajectory: int = 1,
) -> tuple[
    SqliteRegistry,
    SqliteGatewayControl,
    Attempt,
    GatewayCapability,
    GatewayProxyService,
    FakeGatewayAdapter,
]:
    registry = SqliteRegistry(tmp_path / "registry.sqlite")
    attempt = _insert_attempt(
        registry,
        attempts_per_trajectory=attempts_per_trajectory,
    )
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
    assert response.kernel_artifact_digest is not None
    assert response.kernel_trial_id is not None
    assert len(adapter.requests) == 1
    adapter_request = adapter.requests[0]
    assert adapter_request.candidate_path is not None
    assert (adapter_request.candidate_path / "kernel.py").read_text() == "def kernel(): pass\n"
    assert await control.get_outcome(attempt.id) is None
    evaluations = control.list_evaluations(attempt.id)
    assert len(evaluations) == 1
    assert str(evaluations[0].kernel_artifact_digest) == response.kernel_artifact_digest
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
    reread = await service.execute(
        capability_value.token,
        json.dumps(
            {
                "schema_version": 2,
                "attempt_id": attempt.id,
                "idempotency_key": "private-result-read-1",
                "operation": "gateway_result_read",
                "gateway_result_digest": response.gateway_result_digest,
            }
        ).encode(),
    )
    assert "secret_size" not in json.dumps(reread.result)
    assert reread.result == {
        "operation": "evaluate",
        "status": "completed",
        "result": {
            "correct": True,
            "correctness": {
                "status": "PASS",
                "rel_err": None,
                "max_abs_err": None,
                "max_rel_err": None,
            },
            "latency_us_geomean": 12.0,
            "latency_us_arith_mean": None,
            "latency_us_by_shape": {},
        },
    }
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
    trials = control.list_kernel_trials((attempt.id,))
    assert len(trials) == 1
    assert trials[0].observations[0].operation is GatewayOperation.PROFILE
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


def test_candidate_diff_policy_allows_unchanged_bootstrap_candidate(tmp_path: Path) -> None:
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    before = tmp_path / "before"
    disallowed = tmp_path / "disallowed"
    before.mkdir()
    disallowed.mkdir()
    (before / "kernel.py").write_text("BLOCK = 64\n", encoding="utf-8")
    (disallowed / "kernel.py").write_text("BLOCK = 64\n", encoding="utf-8")
    (disallowed / "notes.md").write_text("not a Kernel source\n", encoding="utf-8")
    before_digest = artifacts.put_directory(before, ArtifactKind.KERNEL)
    disallowed_digest = artifacts.put_directory(disallowed, ArtifactKind.KERNEL)
    attempt_id = new_attempt_id()

    def missing_attempt(_value: object) -> object:
        raise KeyError(attempt_id)

    registry = SimpleNamespace(get_attempt=missing_attempt)
    bootstrap_subjects = SimpleNamespace(
        get_bootstrap_subject=lambda value: SimpleNamespace(
            input_kernel_digest=before_digest,
            dsl=Dsl.TRITON,
        )
    )
    validator = RegistryCandidateDiffValidator(
        cast(Any, registry),
        artifacts,
        CandidateDiffPolicy(
            {
                Dsl.CUDA: ("*.cu",),
                Dsl.TRITON: ("*.py",),
                Dsl.CUTEDSL: ("*.py",),
            },
            True,
        ),
        cast(Any, bootstrap_subjects),
    )

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


@pytest.mark.anyio
async def test_runtime_queries_use_the_dedicated_http_endpoint(tmp_path: Path) -> None:
    registry, control, attempt, capability_value, service, adapter = _service(tmp_path)
    evaluated = await service.execute(capability_value.token, _request(attempt))
    assert evaluated.kernel_trial_id is not None
    payload = json.dumps(
        {
            "schema_version": 2,
            "attempt_id": attempt.id,
            "idempotency_key": "runtime-query-route",
            "operation": "kernel_trial_show",
            "kernel_trial_id": evaluated.kernel_trial_id,
        }
    ).encode()

    with pytest.raises(ValueError, match="requires /v1/runtime/queries"):
        await service.execute(
            capability_value.token,
            payload,
            operation_scope="gateway",
        )
    response = await service.execute(
        capability_value.token,
        payload,
        operation_scope="runtime",
    )

    assert response.operation == "kernel_trial_show"
    assert len(adapter.requests) == 1

    app = GatewayProxyAsgiApp(service, GatewayProxyLimits(64 * 1024, 8, 16 * 1024))
    routed_payload = payload.replace(b"runtime-query-route", b"runtime-query-http")
    sent: list[dict[str, object]] = []

    async def receive() -> dict[str, object]:
        return {"type": "http.request", "body": routed_payload, "more_body": False}

    async def send(message: dict[str, object]) -> None:
        sent.append(message)

    await app(
        {
            "type": "http",
            "method": "POST",
            "path": "/v1/runtime/queries",
            "headers": [(b"authorization", f"Bearer {capability_value.token}".encode())],
        },
        receive,
        send,
    )
    assert sent[0]["status"] == 200
    assert json.loads(cast(bytes, sent[1]["body"]))["operation"] == "kernel_trial_show"
    control.close()
    registry.close()


@pytest.mark.anyio
async def test_runtime_journal_history_reads_terminal_report_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, control, attempt, capability, service, adapter = _service(tmp_path)
    report_attempt_id = new_attempt_id()
    direction_events = [{"direction_event_id": "directionevent_" + "a" * 32}]
    experiments = [{"experiment_id": "experiment_" + "b" * 32}]
    report_digest = LocalArtifactStore(tmp_path / "artifacts").put_json(
        {
            "attempt_id": report_attempt_id,
            "direction_events": direction_events,
            "experiments": experiments,
        },
        ArtifactKind.ATTEMPT_REPORT,
    )
    monkeypatch.setattr(
        control,
        "visible_attempt_report_artifacts",
        lambda _attempt_id: ((report_attempt_id, report_digest),),
    )

    async def query(operation: str) -> Any:
        return await service.execute(
            capability.token,
            json.dumps(
                {
                    "schema_version": 2,
                    "attempt_id": attempt.id,
                    "idempotency_key": f"{operation}-1",
                    "operation": operation,
                }
            ).encode(),
            operation_scope="runtime",
        )

    directions = await query("direction_history")
    experiment_history = await query("experiment_history")

    assert directions.result == {"journals": [direction_events]}
    assert experiment_history.result == {"journals": [experiments]}
    assert adapter.requests == []
    control.close()
    registry.close()


@pytest.mark.anyio
async def test_runtime_journal_mutations_are_immediately_durable_and_queryable(
    tmp_path: Path,
) -> None:
    registry, control, attempt, capability, service, adapter = _service(tmp_path)

    async def journal(
        operation: str,
        idempotency_key: str,
        **fields: object,
    ) -> Any:
        return await service.execute(
            capability.token,
            json.dumps(
                {
                    "schema_version": 2,
                    "attempt_id": attempt.id,
                    "idempotency_key": idempotency_key,
                    "operation": operation,
                    **fields,
                }
            ).encode(),
            operation_scope="journal",
        )

    proposed = await journal(
        "direction_update",
        "direction-propose-1",
        request={
            "action": "propose",
            "name": "vectorize loads",
            "hypothesis": "one transaction replaces two",
            "rationale": "profile indicates excess memory transactions",
            "plan": ["replace scalar loads"],
            "success_criteria": "latency improves",
            "stop_conditions": "alignment cannot be preserved",
        },
    )
    direction_id = cast(dict[str, Any], proposed.result)["direction_id"]
    await journal(
        "direction_update",
        "direction-start-1",
        request={
            "action": "start",
            "direction_id": direction_id,
            "analysis": "begin the planned experiment",
        },
    )
    evaluated = await service.execute(capability.token, _request(attempt))
    assert evaluated.kernel_trial_id is not None
    assert evaluated.kernel_artifact_digest is not None
    assert evaluated.gateway_result_digest is not None
    subject = {
        "kernel_artifact_digest": evaluated.kernel_artifact_digest,
        "kernel_trial_id": evaluated.kernel_trial_id,
        "gateway_result_digests": [evaluated.gateway_result_digest],
    }
    recorded = await journal(
        "experiment_record",
        "experiment-record-1",
        request={
            "direction_id": direction_id,
            "name": "vectorize load",
            "hypothesis": "one transaction replaces two",
            "change": "use a vector load",
            "before": subject,
            "after": subject,
            "evidence": "the authoritative Evaluate result",
            "analysis": "the candidate remained correct",
            "action": "keep_after",
        },
    )
    experiment_id = cast(dict[str, Any], recorded.result)["experiment_id"]
    await journal(
        "direction_update",
        "direction-complete-1",
        request={
            "action": "complete",
            "direction_id": direction_id,
            "analysis": "the measured experiment completed",
        },
    )

    directions = await journal("directions_list", "directions-list-1")
    loaded_direction = await journal(
        "direction_load",
        "direction-load-1",
        direction_id=direction_id,
    )
    experiments = await journal("experiments_list", "experiments-list-1")
    loaded_experiment = await journal(
        "experiment_load",
        "experiment-load-1",
        experiment_id=experiment_id,
    )
    snapshot = await journal("journal_snapshot", "journal-snapshot-1")

    assert cast(dict[str, Any], directions.result)["directions"] == [
        {"direction_id": direction_id, "name": "vectorize loads", "status": "completed"}
    ]
    assert cast(dict[str, Any], loaded_direction.result)["supporting_experiment_ids"] == [
        experiment_id
    ]
    assert cast(dict[str, Any], experiments.result)["experiments"] == [
        {
            "experiment_id": experiment_id,
            "sequence": 1,
            "name": "vectorize load",
            "action": "keep_after",
        }
    ]
    assert cast(dict[str, Any], loaded_experiment.result)["after"] == subject
    assert len(cast(dict[str, Any], snapshot.result)["direction_events"]) == 3
    assert len(cast(dict[str, Any], snapshot.result)["experiments"]) == 1
    assert len(adapter.requests) == 1

    control.close()
    reopened = SqliteGatewayControl(
        tmp_path / "gateway.sqlite",
        registry,
        signing_key=b"p" * 32,
        clock=lambda: NOW_DATETIME,
    )
    assert len(reopened.list_direction_events(attempt.id)) == 3
    assert reopened.list_experiments(attempt.id)[0]["experiment_id"] == experiment_id
    reopened.close()
    registry.close()


@pytest.mark.anyio
async def test_runtime_rejects_concurrent_in_progress_directions(tmp_path: Path) -> None:
    registry, control, attempt, capability, service, _adapter = _service(tmp_path)

    async def update(idempotency_key: str, request: dict[str, object]) -> Any:
        return await service.execute(
            capability.token,
            json.dumps(
                {
                    "schema_version": 2,
                    "attempt_id": attempt.id,
                    "idempotency_key": idempotency_key,
                    "operation": "direction_update",
                    "request": request,
                }
            ).encode(),
            operation_scope="journal",
        )

    async def propose(idempotency_key: str, name: str) -> str:
        response = await update(
            idempotency_key,
            {
                "action": "propose",
                "name": name,
                "hypothesis": f"{name} may improve latency",
                "rationale": "the mechanism is worth testing",
                "plan": ["test the mechanism"],
                "success_criteria": "latency improves",
                "stop_conditions": "the mechanism is falsified",
            },
        )
        return str(cast(dict[str, Any], response.result)["direction_id"])

    first = await propose("direction-propose-first", "first direction")
    second = await propose("direction-propose-second", "second direction")
    await update(
        "direction-start-first",
        {"action": "start", "direction_id": first, "analysis": "begin first"},
    )

    with pytest.raises(DirectionConcurrencyError) as raised:
        await update(
            "direction-start-second",
            {"action": "start", "direction_id": second, "analysis": "begin second"},
        )
    assert raised.value.requested_direction_id == second
    assert raised.value.in_progress_direction_ids == (first,)
    assert len(control.list_direction_events(attempt.id)) == 3

    await update(
        "direction-defer-first",
        {"action": "defer", "direction_id": first, "analysis": "pause first"},
    )
    await update(
        "direction-start-second-after-close",
        {"action": "start", "direction_id": second, "analysis": "begin second alone"},
    )
    assert len(control.list_direction_events(attempt.id)) == 5
    control.close()
    registry.close()


def test_direction_concurrency_error_is_machine_readable_and_actionable() -> None:
    requested = "direction_" + "b" * 32
    active = "direction_" + "a" * 32
    payload = json.dumps(
        {
            "schema_version": 2,
            "attempt_id": "attempt_" + "c" * 32,
            "idempotency_key": "direction-start-second",
            "operation": "direction_update",
            "request": {
                "action": "start",
                "direction_id": requested,
                "analysis": "begin second",
            },
        }
    ).encode()

    response = _invalid_request_response(
        payload,
        DirectionConcurrencyError(requested, (active,)),
        operation_scope="journal",
    )

    assert response["issues"] == [
        {
            "path": "direction_id",
            "code": "direction_concurrency_conflict",
            "message": response["detail"],
        }
    ]
    assert response["conflict"] == {
        "requested_direction_id": requested,
        "in_progress_direction_ids": [active],
    }
    recovery = cast(list[dict[str, Any]], response["recovery"])
    assert recovery[0]["tool"] == "list-directions"
    assert "close it with update-direction" in recovery[1]["instruction"]
    assert "Retry start only after no other Direction is in progress" in recovery[2][
        "instruction"
    ]
    assert "request_schema" in response


@pytest.mark.anyio
async def test_runtime_journal_survives_attempt_recovery_generation(
    tmp_path: Path,
) -> None:
    registry, control, attempt, capability, service, _adapter = _service(tmp_path)

    async def journal(
        token: str,
        operation: str,
        idempotency_key: str,
        **fields: object,
    ) -> Any:
        return await service.execute(
            token,
            json.dumps(
                {
                    "schema_version": 2,
                    "attempt_id": attempt.id,
                    "idempotency_key": idempotency_key,
                    "operation": operation,
                    **fields,
                }
            ).encode(),
            operation_scope="journal",
        )

    proposed = await journal(
        capability.token,
        "direction_update",
        "recovery-direction-propose",
        request={
            "action": "propose",
            "name": "reuse recovered measurement",
            "hypothesis": "the measured candidate remains valid",
            "rationale": "Gateway evidence is already authoritative",
            "plan": ["reuse the exact measured Trial"],
            "success_criteria": "the Journal accepts the recovered evidence",
            "stop_conditions": "the Artifact is unavailable",
        },
    )
    direction_id = cast(dict[str, Any], proposed.result)["direction_id"]
    await journal(
        capability.token,
        "direction_update",
        "recovery-direction-start",
        request={
            "action": "start",
            "direction_id": direction_id,
            "analysis": "start before the infrastructure interruption",
        },
    )
    evaluated = await service.execute(capability.token, _request(attempt))

    registry.record_infrastructure_failure(attempt.id, "simulated worker interruption")
    registry.retry_attempt(attempt.id)
    recovered_capability = control.issue(
        attempt.id,
        GatewayCapabilityPolicy(
            frozenset(GatewayOperation),
            4,
            NOW_DATETIME + timedelta(hours=1),
        ),
    )
    assert recovered_capability.recovery_generation == 1
    subject = {
        "kernel_artifact_digest": evaluated.kernel_artifact_digest,
        "kernel_trial_id": evaluated.kernel_trial_id,
        "gateway_result_digests": [evaluated.gateway_result_digest],
    }
    recorded = await journal(
        recovered_capability.token,
        "experiment_record",
        "recovery-experiment-record",
        request={
            "direction_id": direction_id,
            "name": "preserve recovered evidence",
            "hypothesis": "the measured candidate remains valid",
            "change": "no source change after Runtime recovery",
            "before": subject,
            "after": subject,
            "evidence": "the generation-zero authoritative Evaluate result",
            "analysis": "the exact Trial remains visible to the logical Attempt",
            "action": "keep_after",
        },
    )

    assert str(cast(dict[str, Any], recorded.result)["experiment_id"]).startswith("experiment_")
    assert len(control.list_direction_events(attempt.id)) == 2
    assert len(control.list_experiments(attempt.id)) == 1
    trial = next(
        trial
        for trial in control.list_kernel_trials((attempt.id,))
        if trial.id == evaluated.kernel_trial_id
    )
    assert trial.recovery_generation == 0
    assert trial.annotations[0].experiment["name"] == "preserve recovered evidence"
    control.close()
    registry.close()


@pytest.mark.anyio
async def test_invalid_request_returns_corresponding_machine_readable_schema(
    tmp_path: Path,
) -> None:
    registry, control, attempt, capability_value, service, _adapter = _service(tmp_path)
    app = GatewayProxyAsgiApp(service, GatewayProxyLimits(64 * 1024, 8, 16 * 1024))
    payload = json.loads(_request(attempt))
    payload["operation"] = "dev"
    payload.pop("candidate")
    body = json.dumps(payload).encode()
    sent: list[dict[str, object]] = []

    async def receive() -> dict[str, object]:
        return {"type": "http.request", "body": body, "more_body": False}

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

    assert sent[0]["status"] == 400
    response_body = sent[1]["body"]
    assert isinstance(response_body, bytes)
    response = json.loads(response_body)
    assert response["error"] == "invalid_request"
    assert response["issues"] == [
        {"path": "command", "code": "missing", "message": "Field required"}
    ]
    assert set(response["request_schema"]["operations"]) == {"dev"}
    dev_schema = response["request_schema"]["operations"]["dev"]
    assert "command" in dev_schema["required"]
    assert "candidate" not in dev_schema["properties"]
    control.close()
    registry.close()


@pytest.mark.anyio
async def test_optimizer_reads_known_current_kernel_trial_without_agate(
    tmp_path: Path,
) -> None:
    registry, control, attempt, capability, service, adapter = _service(tmp_path)
    adapter.result = GatewayAdapterResult(
        status="completed",
        result={
            "all_pass": True,
            "latency_us_geomean": 12.0,
            "latency_us_by_shape": {"0": 10.0, "1": 14.0},
            "measured_by": "supervisor gateway re-measurement",
        },
        evaluation=EvaluationV2(correct=True, latency_us=12.0),
    )
    evaluated = await service.execute(capability.token, _request(attempt))
    assert evaluated.kernel_artifact_digest is not None
    control.record_kernel_trial_annotations(
        attempt.id,
        (
            {
                "experiment_id": "experiment_" + "a" * 32,
                "sequence": 1,
                "before": {
                    "kernel_artifact_digest": evaluated.kernel_artifact_digest,
                    "kernel_trial_id": evaluated.kernel_trial_id,
                    "gateway_result_digests": [evaluated.gateway_result_digest],
                },
                "after": {
                    "kernel_artifact_digest": evaluated.kernel_artifact_digest,
                    "kernel_trial_id": evaluated.kernel_trial_id,
                    "gateway_result_digests": [evaluated.gateway_result_digest],
                },
                "action": "restore_before",
                "hypothesis": "larger tile was slower",
                "recorded_at": NOW,
            },
        ),
    )

    assert len(adapter.requests) == 1
    trial_id = evaluated.kernel_trial_id
    assert isinstance(trial_id, str)

    shown = await service.execute(
        capability.token,
        json.dumps(
            {
                "schema_version": 2,
                "attempt_id": attempt.id,
                "idempotency_key": "kernel-trial-show-1",
                "operation": "kernel_trial_show",
                "kernel_trial_id": trial_id,
            }
        ).encode(),
    )

    assert len(adapter.requests) == 1
    assert isinstance(shown.result, dict)
    assert shown.result == {
        "kernel_artifact_digest": evaluated.kernel_artifact_digest,
        "gateway_results": [
            {
                "operation": "evaluate",
                "status": "completed",
                "result": {
                    "correct": True,
                    "correctness": {
                        "status": "PASS",
                        "rel_err": None,
                        "max_abs_err": None,
                        "max_rel_err": None,
                    },
                    "latency_us_geomean": 12.0,
                    "latency_us_arith_mean": 12.0,
                    "latency_us_by_shape": {"0": 10.0, "1": 14.0},
                },
            }
        ],
    }

    source = await service.execute(
        capability.token,
        json.dumps(
            {
                "schema_version": 2,
                "attempt_id": attempt.id,
                "idempotency_key": "kernel-artifact-read-1",
                "operation": "kernel_artifact_read",
                "kernel_artifact_digest": evaluated.kernel_artifact_digest,
                "file": "kernel.py",
            }
        ).encode(),
    )

    assert isinstance(source.result, dict)
    assert source.result["encoding"] == "utf-8"
    assert source.result["content"] == "def kernel(): pass\n"
    assert source.result["kernel_artifact_digest"] == evaluated.kernel_artifact_digest
    assert source.result["kernel_trial_ids"] == [trial_id]

    gateway_result = await service.execute(
        capability.token,
        json.dumps(
            {
                "schema_version": 2,
                "attempt_id": attempt.id,
                "idempotency_key": "gateway-result-read-1",
                "operation": "gateway_result_read",
                "gateway_result_digest": evaluated.gateway_result_digest,
            }
        ).encode(),
    )

    assert gateway_result.result == {
        "operation": "evaluate",
        "status": "completed",
        "result": {
            "correct": True,
            "correctness": {
                "status": "PASS",
                "rel_err": None,
                "max_abs_err": None,
                "max_rel_err": None,
            },
            "latency_us_geomean": 12.0,
            "latency_us_arith_mean": 12.0,
            "latency_us_by_shape": {"0": 10.0, "1": 14.0},
        },
    }

    control.close()
    registry.close()
