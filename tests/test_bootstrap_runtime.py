"""Composition tests for the Runtime-controlled Core lineage baseline phase."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
from conftest import digest

from atrex_runtime.artifacts import ArtifactKind, LocalArtifactStore
from atrex_runtime.composition.bootstrap import CoreLineageBaselineGenerator
from atrex_runtime.domain.errors import InfrastructureError
from atrex_runtime.domain.ids import (
    ArtifactDigest,
    AttemptId,
    parse_attempt_id,
    parse_campaign_id,
    parse_kernel_agent_revision_id,
    parse_lineage_id,
)
from atrex_runtime.domain.models import Dsl, TokenUsage, WorkerSessionStatus
from atrex_runtime.gateway.control import (
    GatewayEvaluationSource,
    GatewayOperation,
    SqliteGatewayControl,
)
from atrex_runtime.ports import AttemptCandidateResult
from atrex_runtime.registry.sqlite import SqliteRegistry
from atrex_runtime.workers import (
    LineageBootstrapManifestV1,
    LineageBootstrapReportV1,
    LineageBootstrapSessionConfig,
    LineageBootstrapSessionResult,
)

NOW = datetime(2026, 8, 17, tzinfo=UTC)


class CapturingWorkspaces:
    def __init__(self, root: Path) -> None:
        self._root = root
        self._calls = 0

    def prepare(self, manifest: LineageBootstrapManifestV1) -> SimpleNamespace:
        self._calls += 1
        root = self._root / f"run-test-{self._calls}"
        root.mkdir()
        return SimpleNamespace(
            root=root,
            bootstrap_attempt_id=manifest.bootstrap_attempt_id,
        )


class GatewayCommittingSessions:
    def __init__(self, control: SqliteGatewayControl) -> None:
        self._control = control
        self.calls = 0

    def run(
        self,
        prepared: LineageBootstrapManifestV1,
        config: LineageBootstrapSessionConfig,
    ) -> LineageBootstrapSessionResult:
        self.calls += 1
        del config
        outcome = AttemptCandidateResult(
            digest("generated-kernel"),
            digest("generated-gateway-result"),
            True,
            8.25,
        )
        self._control.record_evaluation(
            prepared.bootstrap_attempt_id,
            source=GatewayEvaluationSource.AGENT,
            idempotency_key="baseline-final",
            candidate_artifact_digest=outcome.artifact_digest,
            gateway_result_digest=outcome.gateway_result_digest,
            correct=True,
            latency_us=8.25,
            agate_job_id="ev_agent",
        )
        report = LineageBootstrapReportV1(
            bootstrap_attempt_id=prepared.bootstrap_attempt_id,
            status="baseline_ready",
            approach="simple tiled implementation",
            change_summary="implemented the first DSL kernel",
            correctness_evidence="all Gateway cases passed",
            latency_us=8.25,
            candidate_artifact_digest=outcome.artifact_digest,
            gateway_result_digest=outcome.gateway_result_digest,
            research_sources=(),
            lessons="the baseline is correct",
            next_directions=("profile the main kernel",),
            blocker=None,
        )
        return LineageBootstrapSessionResult(
            finish_reason="completed",
            final_response="done",
            token_usage=TokenUsage(10, 5, 0, 0),
            token_budget=100,
            report=report,
            report_digest=digest("baseline-report"),
            report_error=None,
            session_trace_digest=digest("baseline-trace"),
            candidate_artifact_digest=outcome.artifact_digest,
        )


class FakeBootstrapFinalizer:
    def __init__(self, control: SqliteGatewayControl) -> None:
        self._control = control

    async def finalize(
        self,
        attempt_id: object,
        candidate_artifact_digest: object,
        *,
        nominated_gateway_result_digest: object = None,
        nominated_recovery_generation: int | None = None,
    ) -> AttemptCandidateResult:
        from atrex_runtime.domain.ids import parse_artifact_digest, parse_attempt_id

        parsed_attempt = parse_attempt_id(str(attempt_id))
        candidate = parse_artifact_digest(str(candidate_artifact_digest))
        nominated = parse_artifact_digest(str(nominated_gateway_result_digest))
        assert (
            self._control.find_agent_evaluation(
                parsed_attempt,
                candidate,
                gateway_result_digest=nominated,
                recovery_generation=nominated_recovery_generation,
            )
            is not None
        )
        record = self._control.record_evaluation(
            parsed_attempt,
            source=GatewayEvaluationSource.RUNTIME_FINAL,
            idempotency_key="runtime-final",
            candidate_artifact_digest=candidate,
            gateway_result_digest=digest("runtime-final-result"),
            correct=True,
            latency_us=8.5,
            agate_job_id="ev_runtime_final",
        )
        return self._control.commit_authoritative_outcome(
            parsed_attempt,
            record.id,
            committed_at=NOW,
        )


class BlockedThenCommittingSessions(GatewayCommittingSessions):
    def __init__(self, control: SqliteGatewayControl) -> None:
        super().__init__(control)
        self.capabilities: list[str] = []

    def run(
        self,
        prepared: LineageBootstrapManifestV1,
        config: LineageBootstrapSessionConfig,
    ) -> LineageBootstrapSessionResult:
        self.capabilities.append(config.gateway_capability)
        if not self.calls:
            self.calls += 1
            return LineageBootstrapSessionResult(
                finish_reason="completed",
                final_response="blocked",
                token_usage=TokenUsage(20, 3, 10, 0),
                token_budget=100,
                report=LineageBootstrapReportV1(
                    bootstrap_attempt_id=prepared.bootstrap_attempt_id,
                    status="blocked",
                    approach="attempted a baseline",
                    change_summary="generated a candidate",
                    correctness_evidence="Gateway was unavailable",
                    latency_us=None,
                    candidate_artifact_digest=None,
                    gateway_result_digest=None,
                    research_sources=(),
                    lessons="retry with fresh authority",
                    next_directions=(),
                    blocker="Gateway capability was unavailable",
                ),
                report_digest=digest("blocked-report"),
                report_error=None,
                session_trace_digest=digest("blocked-trace"),
                candidate_artifact_digest=None,
            )
        return super().run(prepared, config)


class PersistingGatewayCommittingSessions:
    def __init__(
        self,
        control: SqliteGatewayControl,
        artifacts: LocalArtifactStore,
        root: Path,
    ) -> None:
        self._control = control
        self._artifacts = artifacts
        self._root = root
        self.calls = 0

    def run(
        self,
        prepared: LineageBootstrapManifestV1,
        config: LineageBootstrapSessionConfig,
    ) -> LineageBootstrapSessionResult:
        self.calls += 1
        del config
        candidate_root = self._root / f"candidate-{self.calls}"
        candidate_root.mkdir()
        candidate_root.joinpath("kernel.py").write_text("def kernel(): pass\n")
        candidate = self._artifacts.put_directory(candidate_root, ArtifactKind.KERNEL)
        gateway_result = self._artifacts.put_json(
            {"correct": True, "latency_us": 8.25}, ArtifactKind.GATEWAY_RESULT
        )
        self._control.record_evaluation(
            prepared.bootstrap_attempt_id,
            source=GatewayEvaluationSource.AGENT,
            idempotency_key="baseline-final",
            candidate_artifact_digest=candidate,
            gateway_result_digest=gateway_result,
            correct=True,
            latency_us=8.25,
            agate_job_id="ev_agent",
        )
        report = LineageBootstrapReportV1(
            bootstrap_attempt_id=prepared.bootstrap_attempt_id,
            status="baseline_ready",
            approach="simple tiled implementation",
            change_summary="implemented the first DSL kernel",
            correctness_evidence="all Gateway cases passed",
            latency_us=8.25,
            candidate_artifact_digest=candidate,
            gateway_result_digest=gateway_result,
            research_sources=(),
            lessons="the baseline is correct",
            next_directions=("profile the main kernel",),
            blocker=None,
        )
        report_digest = self._artifacts.put_json(
            report.model_dump(mode="json"), ArtifactKind.ATTEMPT_REPORT
        )
        trace_digest = self._artifacts.put_json(
            {"session": "completed"}, ArtifactKind.SESSION_LOG
        )
        return LineageBootstrapSessionResult(
            finish_reason="completed",
            final_response="done",
            token_usage=TokenUsage(10, 5, 0, 0),
            token_budget=100,
            report=report,
            report_digest=report_digest,
            report_error=None,
            session_trace_digest=trace_digest,
            candidate_artifact_digest=candidate,
        )


class TimeoutThenCommittingFinalizer:
    def __init__(
        self,
        control: SqliteGatewayControl,
        artifacts: LocalArtifactStore,
        *,
        failures: int = 1,
    ) -> None:
        self._control = control
        self._artifacts = artifacts
        self._failures = failures
        self.calls: list[int | None] = []

    async def finalize(
        self,
        attempt_id: AttemptId,
        candidate_artifact_digest: ArtifactDigest,
        *,
        nominated_gateway_result_digest: ArtifactDigest | None = None,
        nominated_recovery_generation: int | None = None,
        independent_evaluate: bool = True,
    ) -> AttemptCandidateResult:
        del independent_evaluate
        self.calls.append(nominated_recovery_generation)
        if len(self.calls) <= self._failures:
            raise InfrastructureError("authoritative Agate request timed out")
        assert nominated_gateway_result_digest is not None
        assert (
            self._control.find_agent_evaluation(
                attempt_id,
                candidate_artifact_digest,
                gateway_result_digest=nominated_gateway_result_digest,
                recovery_generation=nominated_recovery_generation,
            )
            is not None
        )
        final_result = self._artifacts.put_json(
            {"correct": True, "latency_us": 8.5}, ArtifactKind.GATEWAY_RESULT
        )
        record = self._control.record_evaluation(
            attempt_id,
            source=GatewayEvaluationSource.RUNTIME_FINAL,
            idempotency_key="runtime-final-stable",
            candidate_artifact_digest=candidate_artifact_digest,
            gateway_result_digest=final_result,
            correct=True,
            latency_us=8.5,
            agate_job_id="ev_runtime_final",
        )
        return self._control.commit_authoritative_outcome(
            attempt_id,
            record.id,
            committed_at=NOW,
        )


def test_core_lineage_baseline_uses_pre_lineage_gateway_subject_and_recovers(
    tmp_path: Path,
) -> None:
    registry = SqliteRegistry(tmp_path / "registry.sqlite")
    control = SqliteGatewayControl(
        tmp_path / "gateway.sqlite",
        registry,
        signing_key=b"b" * 32,
        clock=lambda: NOW,
    )
    sessions = GatewayCommittingSessions(control)
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    generator = CoreLineageBaselineGenerator(
        CapturingWorkspaces(tmp_path),  # type: ignore[arg-type]
        sessions,  # type: ignore[arg-type]
        control,
        registry,
        FakeBootstrapFinalizer(control),
        artifacts,
        gateway_endpoint="http://runtime.example.test",
        operations=frozenset({GatewayOperation.EVALUATE, GatewayOperation.WIKI_QUERY}),
        max_calls=4,
        capability_lifetime=timedelta(hours=1),
        environment=(),
        wiki_enabled=True,
    )
    values = {
        "bootstrap_attempt_id": parse_attempt_id("attempt_" + "1" * 32),
        "campaign_id": parse_campaign_id("campaign_" + "2" * 32),
        "lineage_id": parse_lineage_id("lineage_" + "3" * 32),
        "kernel_agent_revision_id": parse_kernel_agent_revision_id("agentrev_" + "4" * 32),
        "optimizer_digest": digest("optimizer"),
        "input_kernel_digest": digest("seed-kernel"),
        "evaluation_contract_digest": digest("contract"),
        "agent_problem_digest": digest("problem"),
        "evidence_digest": digest("evidence"),
        "dsl": Dsl.TRITON,
        "operator": "vector_add",
        "hardware_target": "nvidia-h100",
    }

    first = generator.generate(**values)  # type: ignore[arg-type]
    second = generator.generate(**values)  # type: ignore[arg-type]

    assert second == first
    assert first.kernel_digest == digest("generated-kernel")
    assert first.gateway_result_digest == digest("runtime-final-result")
    assert sessions.calls == 1
    subject = control.get_bootstrap_subject(values["bootstrap_attempt_id"])
    assert subject.input_kernel_digest == digest("seed-kernel")
    assert subject.evaluation_contract_digest == digest("contract")
    runs = control.list_bootstrap_runs(values["bootstrap_attempt_id"])  # type: ignore[arg-type]
    assert len(runs) == 1
    assert runs[0].status.value == "completed"
    assert runs[0].run_id == "run-test-1"
    assert runs[0].total_tokens == 15
    worker_sessions = registry.list_worker_sessions(
        attempt_id=values["bootstrap_attempt_id"]  # type: ignore[arg-type]
    )
    assert len(worker_sessions) == 1
    assert worker_sessions[0].role.value == "framework_baseline"
    assert worker_sessions[0].status.value == "completed"
    assert worker_sessions[0].trace_digest == digest("baseline-trace")
    events = registry.list_runtime_events(after_sequence=0, limit=10)
    assert [event.kind for event in events] == ["bootstrap.lineage_baseline_completed"]
    control.close()
    registry.close()


def test_core_lineage_baseline_records_failed_generation_before_retry(
    tmp_path: Path,
) -> None:
    registry = SqliteRegistry(tmp_path / "registry.sqlite")
    control = SqliteGatewayControl(
        tmp_path / "gateway.sqlite",
        registry,
        signing_key=b"f" * 32,
        clock=lambda: NOW,
    )
    sessions = BlockedThenCommittingSessions(control)
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    generator = CoreLineageBaselineGenerator(
        CapturingWorkspaces(tmp_path),  # type: ignore[arg-type]
        sessions,  # type: ignore[arg-type]
        control,
        registry,
        FakeBootstrapFinalizer(control),
        artifacts,
        gateway_endpoint="http://runtime.example.test",
        operations=frozenset({GatewayOperation.EVALUATE}),
        max_calls=4,
        capability_lifetime=timedelta(hours=1),
        environment=(),
        wiki_enabled=False,
    )
    values = {
        "bootstrap_attempt_id": parse_attempt_id("attempt_" + "5" * 32),
        "campaign_id": parse_campaign_id("campaign_" + "6" * 32),
        "lineage_id": parse_lineage_id("lineage_" + "7" * 32),
        "kernel_agent_revision_id": parse_kernel_agent_revision_id("agentrev_" + "8" * 32),
        "optimizer_digest": digest("optimizer-retry"),
        "input_kernel_digest": digest("seed-retry"),
        "evaluation_contract_digest": digest("contract-retry"),
        "agent_problem_digest": digest("problem-retry"),
        "evidence_digest": digest("evidence-retry"),
        "dsl": Dsl.TRITON,
        "operator": "vector_add",
        "hardware_target": "nvidia-h100",
    }

    with pytest.raises(RuntimeError, match="Gateway capability was unavailable"):
        generator.generate(**values)  # type: ignore[arg-type]
    generated = generator.generate(**values)  # type: ignore[arg-type]

    assert generated.kernel_digest == digest("generated-kernel")
    assert sessions.capabilities[0] != sessions.capabilities[1]
    runs = control.list_bootstrap_runs(values["bootstrap_attempt_id"])  # type: ignore[arg-type]
    assert [run.recovery_generation for run in runs] == [0, 1]
    assert [run.status.value for run in runs] == ["failed", "completed"]
    assert runs[0].finish_reason == "blocked"
    assert runs[0].failure_reason is not None
    assert runs[0].session_trace_digest == digest("blocked-trace")
    assert runs[0].report_digest == digest("blocked-report")
    assert runs[0].total_tokens == 33
    assert runs[1].run_id == "run-test-2"
    worker_sessions = registry.list_worker_sessions(
        attempt_id=values["bootstrap_attempt_id"]  # type: ignore[arg-type]
    )
    assert [session.status.value for session in worker_sessions] == ["failed", "completed"]
    assert [session.recovery_generation for session in worker_sessions] == [0, 1]
    assert [event.kind for event in registry.list_runtime_events(after_sequence=0, limit=10)] == [
        "bootstrap.lineage_baseline_failed",
        "bootstrap.lineage_baseline_completed",
    ]
    control.close()
    registry.close()


def test_core_lineage_baseline_retries_only_finalization_after_agent_success(
    tmp_path: Path,
) -> None:
    registry = SqliteRegistry(tmp_path / "registry.sqlite")
    control = SqliteGatewayControl(
        tmp_path / "gateway.sqlite",
        registry,
        signing_key=b"r" * 32,
        clock=lambda: NOW,
    )
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    sessions = PersistingGatewayCommittingSessions(control, artifacts, tmp_path)
    finalizer = TimeoutThenCommittingFinalizer(control, artifacts, failures=2)
    generator = CoreLineageBaselineGenerator(
        CapturingWorkspaces(tmp_path),  # type: ignore[arg-type]
        sessions,  # type: ignore[arg-type]
        control,
        registry,
        finalizer,
        artifacts,
        gateway_endpoint="http://runtime.example.test",
        operations=frozenset({GatewayOperation.EVALUATE}),
        max_calls=4,
        capability_lifetime=timedelta(hours=1),
        environment=(),
        wiki_enabled=False,
    )
    values = {
        "bootstrap_attempt_id": parse_attempt_id("attempt_" + "9" * 32),
        "campaign_id": parse_campaign_id("campaign_" + "a" * 32),
        "lineage_id": parse_lineage_id("lineage_" + "b" * 32),
        "kernel_agent_revision_id": parse_kernel_agent_revision_id("agentrev_" + "c" * 32),
        "optimizer_digest": digest("optimizer-finalization-retry"),
        "input_kernel_digest": digest("seed-finalization-retry"),
        "evaluation_contract_digest": digest("contract-finalization-retry"),
        "agent_problem_digest": digest("problem-finalization-retry"),
        "evidence_digest": digest("evidence-finalization-retry"),
        "dsl": Dsl.TRITON,
        "operator": "vector_add",
        "hardware_target": "nvidia-h100",
    }

    with pytest.raises(InfrastructureError, match="timed out"):
        generator.generate(**values)  # type: ignore[arg-type]
    with pytest.raises(InfrastructureError, match="timed out"):
        generator.generate(**values)  # type: ignore[arg-type]
    generated = generator.generate(**values)  # type: ignore[arg-type]

    assert generated.latency_us == 8.5
    assert sessions.calls == 1
    assert finalizer.calls == [0, 0, 0]
    runs = control.list_bootstrap_runs(values["bootstrap_attempt_id"])  # type: ignore[arg-type]
    assert [run.status.value for run in runs] == ["failed", "failed", "completed"]
    assert runs[0].finish_reason == "finalization-error"
    assert runs[0].candidate_digest is not None
    assert runs[1].run_id == "finalization-only-from-generation-0"
    assert runs[1].total_tokens is None
    assert runs[2].run_id == "finalization-only-from-generation-0"
    assert runs[2].total_tokens is None
    worker_sessions = registry.list_worker_sessions(
        attempt_id=values["bootstrap_attempt_id"]  # type: ignore[arg-type]
    )
    assert len(worker_sessions) == 1
    assert worker_sessions[0].status is WorkerSessionStatus.COMPLETED
    control.close()
    registry.close()
