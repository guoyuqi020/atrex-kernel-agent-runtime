"""Tests for Runtime-owned independent final candidate evaluation."""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from conftest import digest

from atrex_runtime.artifacts.local import ArtifactKind, LocalArtifactStore
from atrex_runtime.domain.ids import (
    new_attempt_id,
    parse_campaign_id,
    parse_epoch_id,
    parse_kernel_agent_revision_id,
    parse_lineage_id,
)
from atrex_runtime.domain.models import Dsl
from atrex_runtime.gateway.contract import (
    AgateEvaluationContext,
    AgateEvaluationContractV1,
    AgateEvaluationOptionsV1,
)
from atrex_runtime.gateway.control import (
    BootstrapGatewaySubject,
    GatewayCapabilityPolicy,
    GatewayEvaluationSource,
    GatewayOperation,
    SqliteGatewayControl,
)
from atrex_runtime.gateway.finalization import (
    AgateAuthoritativeCandidateEvaluator,
    BootstrapEvaluationStage,
)
from atrex_runtime.gateway.result_metrics import gateway_result_sol_summary
from atrex_runtime.gateway.retrying_client import RetryingAgateClient
from atrex_runtime.ports import AttemptCandidateResult
from atrex_runtime.registry.sqlite import SqliteRegistry

NOW = datetime(2026, 8, 18, tzinfo=UTC)


@dataclass
class FakeClient:
    submitted: list[dict[str, object]] = field(default_factory=list)

    def submit_job(self, kind: str, request: dict[str, object]) -> dict[str, object]:
        assert kind == "eval"
        self.submitted.append(request)
        return {"job_id": "ev_final"}

    def get_job(
        self,
        job_id: str,
        wait: bool = False,
        timeout: float = 30.0,
        include_spec: bool = False,
    ) -> dict[str, object]:
        assert (job_id, wait, include_spec) == ("ev_final", True, False)
        assert timeout == 100.0
        return {
            "job_id": job_id,
            "status": "succeeded",
            "result": {"all_pass": True, "latency_us_geomean": 7.5},
        }


@dataclass
class RepeatedFinalClient:
    submitted: list[dict[str, object]] = field(default_factory=list)
    latency_by_job: dict[str, float] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def submit_job(self, kind: str, request: dict[str, object]) -> dict[str, object]:
        assert kind == "eval"
        with self.lock:
            ordinal = len(self.submitted)
            job_id = f"ev_final_{ordinal}"
            self.submitted.append(request)
            self.latency_by_job[job_id] = float(ordinal * 4 + 5)
        return {"job_id": job_id}

    def get_job(
        self,
        job_id: str,
        wait: bool = False,
        timeout: float = 30.0,
        include_spec: bool = False,
    ) -> dict[str, object]:
        assert wait is True
        assert timeout == 100.0
        assert include_spec is False
        return {
            "job_id": job_id,
            "status": "succeeded",
            "result": {
                "all_pass": True,
                "latency_us_geomean": self.latency_by_job[job_id],
            },
        }


@dataclass
class ProfilingClient:
    submitted: list[tuple[str, dict[str, object]]] = field(default_factory=list)

    def submit_job(self, kind: str, request: dict[str, object]) -> dict[str, object]:
        self.submitted.append((kind, request))
        return {"job_id": "ev_final" if kind == "eval" else "pr_final"}

    def get_job(
        self,
        job_id: str,
        wait: bool = False,
        timeout: float = 30.0,
        include_spec: bool = False,
    ) -> dict[str, object]:
        assert wait is True
        assert timeout == 100.0
        assert include_spec is False
        if job_id == "ev_final":
            return {
                "job_id": job_id,
                "status": "succeeded",
                "result": {"all_pass": True, "latency_us_geomean": 7.5},
            }
        assert job_id == "pr_final"
        return {
            "job_id": job_id,
            "status": "succeeded",
            "result": {
                "level": "sol",
                "profiler": "ncu",
                "kernels": [
                    {
                        "compute_sol_pct": 20.0,
                        "mem_sol_pct": 70.0,
                        "duration": 10_000.0,
                        "duration_unit": "ns",
                    }
                ],
            },
        }


class CandidateValidationError(Exception):
    def __init__(self) -> None:
        self.status = 400
        self.error_class = "validation"
        self.payload = {"message": "candidate source rejected"}
        super().__init__("validation rejected")


class TransientPollError(Exception):
    def __init__(self) -> None:
        self.status = 502
        super().__init__("temporary ingress failure")


@dataclass
class TransientPollingClient(FakeClient):
    poll_calls: int = 0

    def get_job(
        self,
        job_id: str,
        wait: bool = False,
        timeout: float = 30.0,
        include_spec: bool = False,
    ) -> dict[str, object]:
        assert (job_id, wait, include_spec) == ("ev_final", True, False)
        self.poll_calls += 1
        if self.poll_calls == 1:
            assert timeout == 100.0
            raise TransientPollError
        assert 0 < timeout <= 100.0
        return {
            "job_id": job_id,
            "status": "succeeded",
            "result": {"all_pass": True, "latency_us_geomean": 7.5},
        }


class RejectingClient:
    def submit_job(self, kind: str, request: dict[str, object]) -> dict[str, object]:
        del kind, request
        raise CandidateValidationError

    def get_job(
        self,
        job_id: str,
        wait: bool = False,
        timeout: float = 30.0,
        include_spec: bool = False,
    ) -> dict[str, object]:
        del job_id, wait, timeout, include_spec
        raise AssertionError("a validation rejection has no Agate job")


@dataclass
class FakeEvents:
    values: list[tuple[str, str, object]] = field(default_factory=list)

    def record_runtime_event(self, kind: str, aggregate_id: str, payload: object = None) -> None:
        self.values.append((kind, aggregate_id, payload))


class FakeContexts:
    def __init__(self, shape_count: int = 1) -> None:
        self._shape_count = shape_count

    def resolve(self, _attempt_id: object) -> AgateEvaluationContext:
        return AgateEvaluationContext(
            "vector_add",
            "L20N",
            Dsl.TRITON,
            AgateEvaluationContractV1(
                candidate_path="kernel.py",
                reference_py="reference",
                input_py="inputs",
                shapes={str(index): {} for index in range(self._shape_count)},
                options=AgateEvaluationOptionsV1(
                    num_correctness_cases=1,
                    bench_iters=1,
                    atol=0.0,
                    rtol=0.0,
                    timeout_s=60,
                ),
                lock_clocks=True,
            ),
        )


def _builder(
    candidate: str,
    reference: object,
    gpu: str,
    **values: object,
) -> dict[str, object]:
    return {"candidate": candidate, "reference": reference, "gpu": gpu, **values}


def _subject(attempt_id: object) -> BootstrapGatewaySubject:
    return BootstrapGatewaySubject(
        attempt_id,  # type: ignore[arg-type]
        parse_campaign_id("campaign_" + "1" * 32),
        parse_lineage_id("lineage_" + "2" * 32),
        parse_epoch_id("epoch_" + "3" * 32),
        parse_kernel_agent_revision_id("agentrev_" + "4" * 32),
        "vector_add",
        "L20N",
        Dsl.TRITON,
        digest("contract"),
        digest("seed"),
        digest("evidence"),
        NOW,
    )


@pytest.mark.anyio
async def test_finalizer_re_evaluates_nominated_kernel_and_commits_authority(
    tmp_path: Path,
) -> None:
    registry = SqliteRegistry(tmp_path / "registry.sqlite")
    control = SqliteGatewayControl(
        tmp_path / "gateway.sqlite",
        registry,
        signing_key=b"f" * 32,
        clock=lambda: NOW,
    )
    attempt_id = new_attempt_id()
    subject = _subject(attempt_id)
    control.issue_bootstrap(
        subject,
        GatewayCapabilityPolicy(
            frozenset({GatewayOperation.EVALUATE}),
            8,
            NOW + timedelta(hours=1),
        ),
    )
    candidate_root = tmp_path / "candidate"
    candidate_root.mkdir()
    (candidate_root / "kernel.py").write_text("def kernel(): pass\n")
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    candidate_digest = artifacts.put_directory(candidate_root, ArtifactKind.KERNEL)
    agent_result = artifacts.put_json({"agent": "pass"}, ArtifactKind.GATEWAY_RESULT)
    control.record_evaluation(
        attempt_id,
        source=GatewayEvaluationSource.AGENT,
        idempotency_key="agent-final",
        kernel_artifact_digest=candidate_digest,
        gateway_result_digest=agent_result,
        correct=True,
        latency_us=8.0,
        agate_job_id="ev_agent",
    )
    client = FakeClient()
    events = FakeEvents()
    finalizer = AgateAuthoritativeCandidateEvaluator(
        client,  # type: ignore[arg-type]
        _builder,
        FakeContexts(5),
        artifacts,
        control,
        events,
        wait_timeout_s=100.0,
        bootstrap_stages=(BootstrapEvaluationStage(1),),
        bootstrap_bench_iters=5,
        clock=lambda: NOW,
    )

    outcome = await finalizer.finalize(
        attempt_id,
        candidate_digest,
        nominated_gateway_result_digest=agent_result,
    )
    recovered = await finalizer.finalize(
        attempt_id,
        candidate_digest,
        nominated_gateway_result_digest=agent_result,
    )

    assert recovered == outcome
    assert outcome.correct is True
    assert outcome.latency_us == 7.5
    assert len(client.submitted) == 2
    assert sorted(
        len(request["reference"]["shapes"])
        for request in client.submitted  # type: ignore[index]
    ) == [1, 4]
    assert client.submitted[0]["gpu"] == "L20N"
    options = client.submitted[0]["options"]
    assert isinstance(options, dict)
    assert options["bench_iters"] == 5
    evaluations = control.list_evaluations(attempt_id)
    assert [item.source.value for item in evaluations] == ["agent", "runtime_final"]
    assert evaluations[-1].kernel_artifact_digest == candidate_digest
    assert control.get_committed_outcome(attempt_id) == outcome
    assert [kind for kind, _aggregate, _payload in events.values] == [
        "gateway.authoritative_evaluation_submitted",
        "gateway.authoritative_evaluation_submitted",
        "gateway.authoritative_evaluation_completed",
    ]
    control.close()
    registry.close()


@pytest.mark.anyio
async def test_finalizer_uses_shared_client_retry_for_transient_poll_failure(
    tmp_path: Path,
) -> None:
    registry = SqliteRegistry(tmp_path / "registry.sqlite")
    control = SqliteGatewayControl(
        tmp_path / "gateway.sqlite",
        registry,
        signing_key=b"t" * 32,
        clock=lambda: NOW,
    )
    attempt_id = new_attempt_id()
    control.issue_bootstrap(
        _subject(attempt_id),
        GatewayCapabilityPolicy(
            frozenset({GatewayOperation.EVALUATE}),
            8,
            NOW + timedelta(hours=1),
        ),
    )
    candidate_root = tmp_path / "candidate"
    candidate_root.mkdir()
    candidate_root.joinpath("kernel.py").write_text("def kernel(): pass\n")
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    candidate_digest = artifacts.put_directory(candidate_root, ArtifactKind.KERNEL)
    agent_result = artifacts.put_json({"agent": "pass"}, ArtifactKind.GATEWAY_RESULT)
    control.record_evaluation(
        attempt_id,
        source=GatewayEvaluationSource.AGENT,
        idempotency_key="agent-final",
        kernel_artifact_digest=candidate_digest,
        gateway_result_digest=agent_result,
        correct=True,
        latency_us=8.0,
        agate_job_id="ev_agent",
    )
    raw_client = TransientPollingClient()
    client = RetryingAgateClient(raw_client, sleeper=lambda _: None)
    events = FakeEvents()
    finalizer = AgateAuthoritativeCandidateEvaluator(
        client,  # type: ignore[arg-type]
        _builder,
        FakeContexts(),
        artifacts,
        control,
        events,
        wait_timeout_s=100.0,
        bootstrap_stages=(BootstrapEvaluationStage(1),),
        clock=lambda: NOW,
    )

    outcome = await finalizer.finalize(attempt_id, candidate_digest)

    assert outcome.correct is True
    assert raw_client.poll_calls == 2
    assert not [value for value in events.values if value[0].endswith("poll_retry")]
    control.close()
    registry.close()


@pytest.mark.anyio
async def test_finalizer_defers_authority_to_abba_without_an_extra_eval(tmp_path: Path) -> None:
    registry = SqliteRegistry(tmp_path / "registry.sqlite")
    control = SqliteGatewayControl(
        tmp_path / "gateway.sqlite",
        registry,
        signing_key=b"a" * 32,
        clock=lambda: NOW,
    )
    attempt_id = new_attempt_id()
    control.issue_bootstrap(
        _subject(attempt_id),
        GatewayCapabilityPolicy(
            frozenset({GatewayOperation.EVALUATE}),
            8,
            NOW + timedelta(hours=1),
        ),
    )
    candidate_root = tmp_path / "candidate"
    candidate_root.mkdir()
    candidate_root.joinpath("kernel.py").write_text("def kernel(): pass\n")
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    candidate_digest = artifacts.put_directory(candidate_root, ArtifactKind.KERNEL)
    agent_result = artifacts.put_json({"agent": "pass"}, ArtifactKind.GATEWAY_RESULT)
    control.record_evaluation(
        attempt_id,
        source=GatewayEvaluationSource.AGENT,
        idempotency_key="agent-final",
        kernel_artifact_digest=candidate_digest,
        gateway_result_digest=agent_result,
        correct=True,
        latency_us=8.0,
        agate_job_id="ev_agent",
    )
    client = FakeClient()
    events = FakeEvents()
    finalizer = AgateAuthoritativeCandidateEvaluator(
        client,  # type: ignore[arg-type]
        _builder,
        FakeContexts(),
        artifacts,
        control,
        events,
        wait_timeout_s=100.0,
        clock=lambda: NOW,
    )

    provisional = await finalizer.finalize(
        attempt_id,
        candidate_digest,
        independent_evaluate=False,
    )

    assert provisional == AttemptCandidateResult(candidate_digest, agent_result, True, 8.0)
    assert client.submitted == []
    assert control.get_committed_outcome(attempt_id) is None
    assert [kind for kind, _aggregate, _payload in events.values] == [
        "gateway.agent_evaluation_adopted_for_abba"
    ]
    control.close()
    registry.close()


@pytest.mark.anyio
async def test_finalizer_runs_ordered_atrex_bootstrap_stages_and_uses_second_latency(
    tmp_path: Path,
) -> None:
    registry = SqliteRegistry(tmp_path / "registry.sqlite")
    control = SqliteGatewayControl(
        tmp_path / "gateway.sqlite",
        registry,
        signing_key=b"m" * 32,
        clock=lambda: NOW,
    )
    attempt_id = new_attempt_id()
    control.issue_bootstrap(
        _subject(attempt_id),
        GatewayCapabilityPolicy(
            frozenset({GatewayOperation.EVALUATE}),
            8,
            NOW + timedelta(hours=1),
        ),
    )
    candidate_root = tmp_path / "candidate"
    candidate_root.mkdir()
    candidate_root.joinpath("kernel.py").write_text("def kernel(): pass\n")
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    candidate_digest = artifacts.put_directory(candidate_root, ArtifactKind.KERNEL)
    agent_result = artifacts.put_json({"agent": "pass"}, ArtifactKind.GATEWAY_RESULT)
    control.record_evaluation(
        attempt_id,
        source=GatewayEvaluationSource.AGENT,
        idempotency_key="agent-final",
        kernel_artifact_digest=candidate_digest,
        gateway_result_digest=agent_result,
        correct=True,
        latency_us=8.0,
        agate_job_id="ev_agent",
    )
    client = RepeatedFinalClient()
    finalizer = AgateAuthoritativeCandidateEvaluator(
        client,  # type: ignore[arg-type]
        _builder,
        FakeContexts(),
        artifacts,
        control,
        FakeEvents(),
        wait_timeout_s=100.0,
        bootstrap_stages=(BootstrapEvaluationStage(1), BootstrapEvaluationStage(5)),
        clock=lambda: NOW,
    )

    outcome = await finalizer.finalize(attempt_id, candidate_digest)

    assert outcome.correct is True
    assert outcome.latency_us == pytest.approx(9.0)
    assert len(client.submitted) == 2
    assert len({request["idempotency_key"] for request in client.submitted}) == 2
    raw = json.loads(
        (artifacts.verify(outcome.gateway_result_digest).payload_path / "value.json").read_text()
    )
    assert raw["all_pass"] is True
    assert raw["latency_source_stage"] == 1
    assert [stage["correctness_cases"] for stage in raw["completed_stages"]] == [1, 5]
    assert [request["options"]["num_correctness_cases"] for request in client.submitted] == [1, 5]
    control.close()
    registry.close()


@pytest.mark.anyio
async def test_finalizer_profiles_correct_kernel_when_roofline_is_missing(
    tmp_path: Path,
) -> None:
    registry = SqliteRegistry(tmp_path / "registry.sqlite")
    control = SqliteGatewayControl(
        tmp_path / "gateway.sqlite",
        registry,
        signing_key=b"p" * 32,
        clock=lambda: NOW,
    )
    attempt_id = new_attempt_id()
    control.issue_bootstrap(
        _subject(attempt_id),
        GatewayCapabilityPolicy(
            frozenset({GatewayOperation.EVALUATE}),
            8,
            NOW + timedelta(hours=1),
        ),
    )
    candidate_root = tmp_path / "candidate"
    candidate_root.mkdir()
    candidate_root.joinpath("kernel.py").write_text("def kernel(): pass\n")
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    candidate_digest = artifacts.put_directory(candidate_root, ArtifactKind.KERNEL)
    agent_result = artifacts.put_json({"agent": "pass"}, ArtifactKind.GATEWAY_RESULT)
    control.record_evaluation(
        attempt_id,
        source=GatewayEvaluationSource.AGENT,
        idempotency_key="agent-final",
        kernel_artifact_digest=candidate_digest,
        gateway_result_digest=agent_result,
        correct=True,
        latency_us=8.0,
        agate_job_id="ev_agent",
    )
    client = ProfilingClient()
    finalizer = AgateAuthoritativeCandidateEvaluator(
        client,  # type: ignore[arg-type]
        _builder,
        FakeContexts(),
        artifacts,
        control,
        FakeEvents(),
        wait_timeout_s=100.0,
        bootstrap_stages=(BootstrapEvaluationStage(1),),
        profile_without_roofline=True,
        clock=lambda: NOW,
    )

    outcome = await finalizer.finalize(attempt_id, candidate_digest)

    assert [kind for kind, _payload in client.submitted] == ["eval", "profile"]
    profile_request = client.submitted[1][1]
    assert profile_request["level"] == "sol"
    assert profile_request["top_kernels"] == 10
    summary = gateway_result_sol_summary(artifacts, outcome.gateway_result_digest)
    assert summary.percent == 70.0
    assert summary.source == "ncu-profile"
    payload = artifacts.verify(outcome.gateway_result_digest).payload_path
    assert payload.joinpath("value.json").is_file()
    assert payload.joinpath("profile.json").is_file()
    control.close()
    registry.close()


@pytest.mark.anyio
async def test_finalizer_records_validation_rejection_as_authoritative_outcome(
    tmp_path: Path,
) -> None:
    registry = SqliteRegistry(tmp_path / "registry.sqlite")
    control = SqliteGatewayControl(
        tmp_path / "gateway.sqlite",
        registry,
        signing_key=b"r" * 32,
        clock=lambda: NOW,
    )
    attempt_id = new_attempt_id()
    control.issue_bootstrap(
        _subject(attempt_id),
        GatewayCapabilityPolicy(
            frozenset({GatewayOperation.EVALUATE}),
            8,
            NOW + timedelta(hours=1),
        ),
    )
    candidate_root = tmp_path / "candidate"
    candidate_root.mkdir()
    (candidate_root / "kernel.py").write_text("invalid candidate\n")
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    candidate_digest = artifacts.put_directory(candidate_root, ArtifactKind.KERNEL)
    agent_result = artifacts.put_json({"agent": "pass"}, ArtifactKind.GATEWAY_RESULT)
    control.record_evaluation(
        attempt_id,
        source=GatewayEvaluationSource.AGENT,
        idempotency_key="agent-final",
        kernel_artifact_digest=candidate_digest,
        gateway_result_digest=agent_result,
        correct=True,
        latency_us=8.0,
        agate_job_id="ev_agent",
    )
    finalizer = AgateAuthoritativeCandidateEvaluator(
        RejectingClient(),  # type: ignore[arg-type]
        _builder,
        FakeContexts(),
        artifacts,
        control,
        FakeEvents(),
        wait_timeout_s=100.0,
        bootstrap_stages=(BootstrapEvaluationStage(1),),
        clock=lambda: NOW,
    )

    outcome = await finalizer.finalize(attempt_id, candidate_digest)

    assert outcome.correct is False
    assert outcome.latency_us is None
    final = control.list_evaluations(attempt_id)[-1]
    assert final.source is GatewayEvaluationSource.RUNTIME_FINAL
    assert final.agate_job_id is None
    raw = json.loads(
        (artifacts.verify(final.gateway_result_digest).payload_path / "value.json").read_text()
    )
    assert raw["all_pass"] is False
    assert raw["completed_stages"][0]["job"] == {
        "schema_version": 1,
        "operation": "shape_batched_evaluate",
        "status": "rejected",
        "error": {
            "category": "candidate_rejected",
            "detail": {"message": "candidate source rejected"},
        },
        "rejected_batch_index": 0,
        "shape_batch_count": 1,
    }
    assert control.get_committed_outcome(attempt_id) == outcome
    control.close()
    registry.close()
