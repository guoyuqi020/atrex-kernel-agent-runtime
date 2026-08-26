"""Tests for framework-neutral Core Optimizer orchestration."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import pytest
from conftest import NOW, digest

from atrex_runtime.domain.errors import InfrastructureError
from atrex_runtime.domain.ids import (
    ArtifactDigest,
    AttemptId,
    new_attempt_id,
    new_kernel_agent_revision_id,
    new_kernel_revision_id,
)
from atrex_runtime.domain.models import (
    AttemptReportStatus,
    AttemptSessionTrace,
    Dsl,
    TokenUsage,
    WorkerSessionStatus,
)
from atrex_runtime.ports import (
    AttemptCandidateResult,
    RunAttemptRequest,
    WorkerGatewayAuthority,
)
from atrex_runtime.registry.sqlite import SqliteRegistry
from atrex_runtime.workers.attempt_report import AttemptExperimentV8, AttemptReportV12
from atrex_runtime.workers.optimizer import (
    OptimizerSessionConfig,
    OptimizerSessionResult,
    SessionOptimizerRunner,
)
from atrex_runtime.workers.workspace import PreparedAttempt


@dataclass
class FakeWorkspaceAssembler:
    prepared: PreparedAttempt
    calls: int = 0

    def prepare(self, _request: RunAttemptRequest) -> PreparedAttempt:
        self.calls += 1
        return self.prepared


@dataclass
class FakeSessionDriver:
    result: OptimizerSessionResult
    calls: int = 0
    configs: list[OptimizerSessionConfig] | None = None

    async def run(
        self,
        _prepared: PreparedAttempt,
        config: OptimizerSessionConfig,
    ) -> OptimizerSessionResult:
        self.calls += 1
        if self.configs is not None:
            self.configs.append(config)
        return self.result


class TimeoutSessionDriver:
    async def run(
        self,
        _prepared: PreparedAttempt,
        _config: OptimizerSessionConfig,
    ) -> OptimizerSessionResult:
        raise InfrastructureError("Core Optimizer process exceeded its wall-time limit")


@dataclass
class SequencedOutcomes:
    values: list[AttemptCandidateResult | None]

    async def get_outcome(self, _attempt_id: object) -> AttemptCandidateResult | None:
        return self.values.pop(0)


@dataclass
class FakeFinalizer:
    result: AttemptCandidateResult
    calls: list[tuple[AttemptId, ArtifactDigest, ArtifactDigest | None, bool]]

    async def finalize(
        self,
        attempt_id: AttemptId,
        kernel_artifact_digest: ArtifactDigest,
        *,
        nominated_gateway_result_digest: ArtifactDigest | None = None,
        independent_evaluate: bool = True,
    ) -> AttemptCandidateResult:
        self.calls.append(
            (
                attempt_id,
                kernel_artifact_digest,
                nominated_gateway_result_digest,
                independent_evaluate,
            )
        )
        return self.result


@dataclass
class FakeGatewayAuthorities:
    calls: int = 0

    async def get_authority(self, _request: RunAttemptRequest) -> WorkerGatewayAuthority:
        self.calls += 1
        return WorkerGatewayAuthority("http://gateway-proxy", "attempt-capability")


@dataclass
class FakeSessionTraceRecorder:
    records: list[tuple[AttemptId, ArtifactDigest, str, int, TokenUsage]]
    runtime_states: list[tuple[AttemptId, ArtifactDigest]] | None = None

    def record_attempt_session_trace(
        self,
        attempt_id: AttemptId,
        artifact_digest: ArtifactDigest,
        finish_reason: str,
        token_budget: int,
        token_usage: TokenUsage,
    ) -> AttemptSessionTrace:
        self.records.append((attempt_id, artifact_digest, finish_reason, token_budget, token_usage))
        return AttemptSessionTrace(
            attempt_id,
            len(self.records),
            artifact_digest,
            finish_reason,
            token_budget,
            token_usage,
            NOW,
        )

    def record_attempt_runtime_state(
        self,
        attempt_id: AttemptId,
        runtime_state_digest: ArtifactDigest,
    ) -> None:
        if self.runtime_states is None:
            self.runtime_states = []
        self.runtime_states.append((attempt_id, runtime_state_digest))


@dataclass
class FakeRuntimeEventRecorder:
    records: list[tuple[str, str, object]]

    def record_runtime_event(
        self,
        kind: str,
        aggregate_id: str,
        payload: object = None,
    ) -> None:
        self.records.append((kind, aggregate_id, payload))


@dataclass
class FakeKernelTrialRecorder:
    records: list[
        tuple[
            AttemptId,
            Sequence[Mapping[str, object]],
            Sequence[Mapping[str, object]],
        ]
    ]

    def record_kernel_trial_annotations(
        self,
        attempt_id: AttemptId,
        experiments: Sequence[Mapping[str, object]],
        *,
        profile_supporting_results: Sequence[Mapping[str, object]] = (),
    ) -> object:
        self.records.append((attempt_id, experiments, profile_supporting_results))
        return ()


def _session_result(
    finish_reason: str,
    final_response: str,
    trace_digest: ArtifactDigest | None = None,
    *,
    with_report: bool = False,
) -> OptimizerSessionResult:
    report = (
        AttemptReportV12(
            schema_version=12,
            attempt_id=new_attempt_id(),
            status="candidate_ready",
            hypothesis="reduce memory traffic",
            diagnosis={
                "bottleneck": "memory bandwidth",
                "evidence": "survey localized memory traffic",
            },
            approach={
                "summary": "coalesce loads",
                "steps": ["vectorize aligned loads"],
                "expected_impact": "reduce memory transactions",
                "risks": [],
            },
            final_candidate={"change_summary": "coalesced global loads"},
            evidence_summary={
                "correctness": "Gateway evaluation completed",
                "performance": "candidate is faster",
            },
            profile_evidence={
                "tool_used": "gateway-execute/profile",
                "profiler": "ncu",
                "profile_level": "survey",
                "bottleneck_type": "memory_bound",
                "evidence_summary": "survey localized memory traffic",
                "evidence_chain": "survey counters support a memory-bound diagnosis",
                "supporting_results": [
                    {
                        "operation": "profile",
                        "kernel_artifact_digest": digest("candidate-experiment"),
                        "kernel_trial_id": "gtrial_" + "b" * 32,
                        "gateway_result_digest": digest("gateway-experiment"),
                    }
                ],
            },
            analysis="candidate is correct and faster",
            knowledge_used=(),
            findings=(
                {
                    "category": "performance",
                    "observation": "latency improved",
                    "root_cause": "coalesced global loads",
                    "resolution": "kept the coalesced-load candidate",
                    "lesson": "coalescing helped",
                    "supporting_experiment_ids": ["experiment_" + "a" * 32],
                },
            ),
            blocker=None,
            experiments=(
                AttemptExperimentV8(
                    experiment_id="experiment_" + "a" * 32,
                    direction_id="direction_" + "a" * 32,
                    sequence=1,
                    recorded_at="2026-08-16T00:00:00+00:00",
                    name="coalescing",
                    hypothesis="coalescing helps",
                    change="vectorized loads",
                    before={
                        "kernel_artifact_digest": digest("before-experiment"),
                        "kernel_trial_id": "gtrial_" + "a" * 32,
                        "gateway_result_digests": [digest("before-gateway")],
                    },
                    after={
                        "kernel_artifact_digest": digest("candidate-experiment"),
                        "kernel_trial_id": "gtrial_" + "b" * 32,
                        "gateway_result_digests": [digest("gateway-experiment")],
                    },
                    evidence="SOL memory traffic",
                    analysis="the hypothesis held and the candidate is faster",
                    action="keep_after",
                ),
            ),
            direction_events=(
                {
                    "direction_event_id": "directionevent_" + "1" * 32,
                    "direction_id": "direction_" + "a" * 32,
                    "recorded_at": "2026-08-16T00:00:00+00:00",
                    "action": "propose",
                    "name": "coalesced loads",
                    "hypothesis": "coalescing helps",
                    "rationale": "profile localized memory traffic",
                    "plan": ["vectorize aligned loads"],
                    "success_criteria": "latency improves",
                    "stop_conditions": "alignment is unsafe",
                    "analysis": None,
                    "supporting_experiment_ids": [],
                },
                {
                    "direction_event_id": "directionevent_" + "2" * 32,
                    "direction_id": "direction_" + "a" * 32,
                    "recorded_at": "2026-08-16T00:01:00+00:00",
                    "action": "complete",
                    "name": None,
                    "hypothesis": None,
                    "rationale": None,
                    "plan": [],
                    "success_criteria": None,
                    "stop_conditions": None,
                    "analysis": "latency improved",
                    "supporting_experiment_ids": ["experiment_" + "a" * 32],
                },
            ),
        )
        if with_report
        else None
    )
    return OptimizerSessionResult(
        finish_reason,
        final_response,
        TokenUsage(100, 20, 30, 10),
        1000,
        trace_digest,
        report,
        digest("attempt-report") if report is not None else None,
        None,
        digest("candidate") if report is not None else None,
        digest("runtime-state"),
    )


def _config(tmp_path: Path) -> OptimizerSessionConfig:
    return OptimizerSessionConfig(
        environment=(("MODEL_PROXY_TOKEN", "scoped-token"),),
    )


def _request() -> RunAttemptRequest:
    return RunAttemptRequest(
        attempt_id=new_attempt_id(),
        kernel_agent_revision_id=new_kernel_agent_revision_id(),
        input_kernel_revision_id=new_kernel_revision_id(),
        epoch_evidence_checkpoint=digest("evidence"),
        attempt_evidence_digest=digest("attempt-evidence"),
        dsl=Dsl.CUDA,
        model="optimizer-model",
    )


@pytest.mark.anyio
async def test_existing_gateway_outcome_skips_workspace_and_core_process(
    tmp_path: Path,
) -> None:
    candidate = AttemptCandidateResult(digest("candidate"), digest("gateway"), True, 10.0)
    prepared = PreparedAttempt(
        tmp_path, tmp_path / ".runtime/attempt.json", tmp_path / "sessions", "s"
    )
    workspaces = FakeWorkspaceAssembler(prepared)
    sessions = FakeSessionDriver(_session_result("completed", "done"))
    authorities = FakeGatewayAuthorities()
    traces = FakeSessionTraceRecorder([])
    runner = SessionOptimizerRunner(
        workspaces,
        sessions,
        SequencedOutcomes([candidate]),
        FakeFinalizer(candidate, []),
        authorities,
        traces,
        FakeRuntimeEventRecorder([]),
        _config(tmp_path),
    )

    result = await runner.run_attempt(_request())

    assert result.candidate == candidate
    assert workspaces.calls == 0
    assert sessions.calls == 0
    assert authorities.calls == 0
    assert traces.records == []


@pytest.mark.anyio
async def test_core_process_result_uses_only_gateway_authoritative_outcome(
    tmp_path: Path,
) -> None:
    candidate = AttemptCandidateResult(digest("candidate"), digest("gateway"), False, None)
    prepared = PreparedAttempt(
        tmp_path, tmp_path / ".runtime/attempt.json", tmp_path / "sessions", "s"
    )
    configs: list[OptimizerSessionConfig] = []
    trace_digest = digest("session-trace")
    traces = FakeSessionTraceRecorder([])
    events = FakeRuntimeEventRecorder([])
    request = _request()
    finalizer = FakeFinalizer(candidate, [])
    kernel_trials = FakeKernelTrialRecorder([])
    runner = SessionOptimizerRunner(
        FakeWorkspaceAssembler(prepared),
        FakeSessionDriver(
            _session_result("completed", "untrusted claim", trace_digest, with_report=True),
            configs=configs,
        ),
        SequencedOutcomes([None]),
        finalizer,
        FakeGatewayAuthorities(),
        traces,
        events,
        _config(tmp_path),
        independent_final_evaluation=False,
        wiki_enabled=True,
        kernel_trials=kernel_trials,
    )

    result = await runner.run_attempt(request)

    assert result.candidate == candidate
    assert configs[0].gateway_capability == "attempt-capability"
    assert configs[0].gateway_endpoint == "http://gateway-proxy"
    assert configs[0].wiki_capability == "attempt-capability"
    assert configs[0].wiki_endpoint == "http://gateway-proxy"
    assert configs[0].model == "optimizer-model"
    assert traces.records[0][0] == request.attempt_id
    assert finalizer.calls == [(request.attempt_id, digest("candidate"), None, False)]
    assert kernel_trials.records[0][2] == (
        {
            "operation": "profile",
            "kernel_artifact_digest": digest("candidate-experiment"),
            "kernel_trial_id": "gtrial_" + "b" * 32,
            "gateway_result_digest": digest("gateway-experiment"),
        },
    )
    assert [kind for kind, _aggregate, _payload in events.records] == [
        "worker.started",
        "worker.exited",
        "worker.cleaned",
        "attempt.reported",
    ]


@pytest.mark.anyio
async def test_optimizer_rejects_bootstrap_only_baseline_experiment(tmp_path: Path) -> None:
    session = _session_result("completed", "done", with_report=True)
    assert session.attempt_report is not None
    original = session.attempt_report.experiments[0]
    baseline = original.model_copy(update={"action": "baseline", "before": None})
    report = session.attempt_report.model_copy(update={"experiments": (baseline,)})
    session = replace(session, attempt_report=report)
    events = FakeRuntimeEventRecorder([])
    runner = SessionOptimizerRunner(
        FakeWorkspaceAssembler(
            PreparedAttempt(
                tmp_path,
                tmp_path / ".runtime/attempt.json",
                tmp_path / "sessions",
                "s",
            )
        ),
        FakeSessionDriver(session),
        SequencedOutcomes([None]),
        FakeFinalizer(
            AttemptCandidateResult(digest("unused"), digest("unused-result"), True, 1.0),
            [],
        ),
        FakeGatewayAuthorities(),
        FakeSessionTraceRecorder([]),
        events,
        _config(tmp_path),
    )

    result = await runner.run_attempt(_request())

    assert result.failure_reason == (
        "invalid Attempt report: Experiment action baseline is Bootstrap-only"
    )
    assert [kind for kind, _aggregate, _payload in events.records][-1] == (
        "attempt.report_rejected"
    )


@pytest.mark.anyio
async def test_missing_gateway_outcome_is_a_consumed_optimizer_failure(tmp_path: Path) -> None:
    prepared = PreparedAttempt(
        tmp_path, tmp_path / ".runtime/attempt.json", tmp_path / "sessions", "s"
    )
    runner = SessionOptimizerRunner(
        FakeWorkspaceAssembler(prepared),
        FakeSessionDriver(_session_result("max-tokens", "")),
        SequencedOutcomes([None]),
        FakeFinalizer(
            AttemptCandidateResult(digest("unused"), digest("unused-result"), True, 1.0),
            [],
        ),
        FakeGatewayAuthorities(),
        FakeSessionTraceRecorder([]),
        FakeRuntimeEventRecorder([]),
        _config(tmp_path),
    )

    result = await runner.run_attempt(_request())

    assert result.candidate is None
    assert result.failure_reason == "missing terminal Attempt report"


@pytest.mark.anyio
async def test_nonzero_core_exit_cannot_accept_a_gateway_candidate(tmp_path: Path) -> None:
    candidate = AttemptCandidateResult(digest("candidate"), digest("gateway"), True, 10.0)
    prepared = PreparedAttempt(
        tmp_path, tmp_path / ".runtime/attempt.json", tmp_path / "sessions", "s"
    )
    runner = SessionOptimizerRunner(
        FakeWorkspaceAssembler(prepared),
        FakeSessionDriver(
            _session_result(
                "process-exit-126",
                "untrusted partial output",
                digest("session-trace"),
                with_report=True,
            )
        ),
        SequencedOutcomes([None]),
        FakeFinalizer(candidate, []),
        FakeGatewayAuthorities(),
        FakeSessionTraceRecorder([]),
        FakeRuntimeEventRecorder([]),
        _config(tmp_path),
    )

    result = await runner.run_attempt(_request())

    assert result.candidate is None
    assert result.failure_reason == (
        "Optimizer session did not complete successfully: process-exit-126"
    )
    assert result.attempt_report_status is AttemptReportStatus.CANDIDATE_READY


@pytest.mark.anyio
async def test_optimizer_timeout_records_cleanup_lifecycle(tmp_path: Path) -> None:
    prepared = PreparedAttempt(
        tmp_path, tmp_path / ".runtime/attempt.json", tmp_path / "sessions", "s"
    )
    events = FakeRuntimeEventRecorder([])
    request = _request()
    runner = SessionOptimizerRunner(
        FakeWorkspaceAssembler(prepared),
        TimeoutSessionDriver(),
        SequencedOutcomes([None]),
        FakeFinalizer(
            AttemptCandidateResult(digest("unused"), digest("unused-result"), True, 1.0),
            [],
        ),
        FakeGatewayAuthorities(),
        FakeSessionTraceRecorder([]),
        events,
        _config(tmp_path),
    )

    with pytest.raises(InfrastructureError, match="wall-time limit"):
        await runner.run_attempt(request)

    assert [kind for kind, _aggregate, _payload in events.records] == [
        "worker.started",
        "worker.timeout",
        "worker.cleaned",
    ]


@pytest.mark.anyio
async def test_optimizer_persists_successful_worker_session_and_raw_trace(
    tmp_path: Path,
) -> None:
    prepared = PreparedAttempt(
        tmp_path / "run-success",
        tmp_path / ".runtime/attempt.json",
        tmp_path / "sessions",
        "optimizer-run-1",
    )
    trace_digest = digest("optimizer-raw-trace")
    request = _request()
    with SqliteRegistry(tmp_path / "registry.sqlite", clock=lambda: NOW) as registry:
        runner = SessionOptimizerRunner(
            FakeWorkspaceAssembler(prepared),
            FakeSessionDriver(_session_result("completed", "done", trace_digest, with_report=True)),
            SequencedOutcomes([None]),
            FakeFinalizer(
                AttemptCandidateResult(digest("candidate"), digest("gateway"), True, 1.0),
                [],
            ),
            FakeGatewayAuthorities(),
            FakeSessionTraceRecorder([]),
            FakeRuntimeEventRecorder([]),
            _config(tmp_path),
            worker_sessions=registry,
            backend="codex",
        )

        await runner.run_attempt(request)
        sessions = registry.list_worker_sessions(attempt_id=request.attempt_id)

    assert len(sessions) == 1
    assert sessions[0].status is WorkerSessionStatus.COMPLETED
    assert sessions[0].external_run_id == "optimizer-run-1"
    assert sessions[0].trace_digest == trace_digest
    assert sessions[0].token_budget == 1000
    assert sessions[0].backend == "codex"


@pytest.mark.anyio
async def test_optimizer_persists_timeout_without_waiting_for_trace(tmp_path: Path) -> None:
    prepared = PreparedAttempt(
        tmp_path / "run-timeout",
        tmp_path / ".runtime/attempt.json",
        tmp_path / "sessions",
        "optimizer-run-timeout",
    )
    request = _request()
    with SqliteRegistry(tmp_path / "registry.sqlite", clock=lambda: NOW) as registry:
        runner = SessionOptimizerRunner(
            FakeWorkspaceAssembler(prepared),
            TimeoutSessionDriver(),
            SequencedOutcomes([None]),
            FakeFinalizer(
                AttemptCandidateResult(digest("unused"), digest("unused-result"), True, 1.0),
                [],
            ),
            FakeGatewayAuthorities(),
            FakeSessionTraceRecorder([]),
            FakeRuntimeEventRecorder([]),
            _config(tmp_path),
            worker_sessions=registry,
        )

        with pytest.raises(InfrastructureError, match="wall-time limit"):
            await runner.run_attempt(request)
        sessions = registry.list_worker_sessions(attempt_id=request.attempt_id)

    assert len(sessions) == 1
    assert sessions[0].status is WorkerSessionStatus.TIMED_OUT
    assert sessions[0].trace_digest is None
    assert sessions[0].workspace_path == str(prepared.root)
