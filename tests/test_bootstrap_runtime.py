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
from atrex_runtime.gateway.control_models import GatewayCapability, gateway_kernel_trial_id
from atrex_runtime.ports import AttemptCandidateResult
from atrex_runtime.registry.sqlite import SqliteRegistry
from atrex_runtime.workers import (
    LineageBootstrapManifestV2,
    LineageBootstrapSessionConfig,
    LineageBootstrapSessionResult,
)
from atrex_runtime.workers.attempt_report import AttemptReportV12

NOW = datetime(2026, 8, 17, tzinfo=UTC)


def _record_agent_evaluate(
    control: SqliteGatewayControl,
    *,
    capability_token: str,
    attempt_id: AttemptId,
    candidate: ArtifactDigest,
    gateway_result: ArtifactDigest,
    idempotency_key: str,
    latency_us: float,
) -> None:
    control.authorize(
        GatewayCapability(capability_token, attempt_id),
        GatewayOperation.EVALUATE,
        idempotency_key=idempotency_key,
        request_digest=str(digest(f"{idempotency_key}-request")),
    )
    control.bind_operation_candidate(
        attempt_id,
        idempotency_key,
        GatewayOperation.EVALUATE,
        candidate,
    )
    control.bind_operation_gateway_result(
        attempt_id,
        idempotency_key,
        GatewayOperation.EVALUATE,
        gateway_result,
    )
    control.commit_operation_artifact(
        attempt_id,
        idempotency_key,
        GatewayOperation.EVALUATE,
        gateway_result,
    )
    control.record_evaluation(
        attempt_id,
        source=GatewayEvaluationSource.AGENT,
        idempotency_key=idempotency_key,
        kernel_artifact_digest=candidate,
        gateway_result_digest=gateway_result,
        correct=True,
        latency_us=latency_us,
        agate_job_id="ev_agent",
    )


def _bootstrap_report(
    attempt_id: AttemptId,
    *,
    candidate: ArtifactDigest | None,
    gateway_result: ArtifactDigest | None,
    generation: int,
    blocked: bool = False,
) -> AttemptReportV12:
    experiment_id = "experiment_" + "1" * 32
    direction_id = "direction_" + "2" * 32
    subject = (
        None
        if candidate is None or gateway_result is None
        else {
            "kernel_artifact_digest": candidate,
            "kernel_trial_id": gateway_kernel_trial_id(
                attempt_id,
                generation,
                candidate,
            ),
            "result_artifact_digests": [gateway_result],
        }
    )
    terminal_action = "block" if blocked else "complete"
    return AttemptReportV12.model_validate(
        {
            "schema_version": 12,
            "attempt_id": attempt_id,
            "status": "blocked" if blocked else "candidate_ready",
            "hypothesis": "a direct implementation establishes the baseline",
            "diagnosis": {
                "bottleneck": "framework bring-up",
                "evidence": "seed inspection and Gateway validation",
            },
            "approach": {
                "summary": "implement a simple correct baseline",
                "steps": ["implement", "evaluate"],
                "expected_impact": "establish correctness",
                "risks": [],
            },
            "final_candidate": (
                None if blocked else {"change_summary": "implemented the first DSL kernel"}
            ),
            "evidence_summary": {
                "correctness": (
                    "Gateway unavailable" if blocked else "all Gateway cases passed"
                ),
                "performance": "no authoritative latency" if blocked else "positive latency",
            },
            "profile_evidence": None,
            "analysis": "retry with fresh authority" if blocked else "the baseline is correct",
            "knowledge_used": [],
            "findings": [
                {
                    "category": "infrastructure" if blocked else "correctness",
                    "observation": "Gateway unavailable" if blocked else "evaluation passed",
                    "root_cause": "authority unavailable" if blocked else "semantics preserved",
                    "resolution": "retry" if blocked else "retain the implementation",
                    "lesson": "use fresh authority" if blocked else "simple baseline is correct",
                    "supporting_experiment_ids": [experiment_id],
                }
            ],
            "blocker": "Gateway capability was unavailable" if blocked else None,
            "experiments": [
                {
                    "experiment_id": experiment_id,
                    "direction_id": direction_id,
                    "sequence": 1,
                    "recorded_at": NOW.isoformat(),
                    "name": "establish baseline",
                    "hypothesis": "the direct baseline is valid",
                    "change": "implemented the first DSL kernel",
                    "before": None if not blocked else subject,
                    "after": subject,
                    "evidence": "Gateway unavailable" if blocked else "evaluation passed",
                    "analysis": "blocked" if blocked else "hypothesis held",
                    "action": "abandon_direction" if blocked else "baseline",
                }
            ],
            "direction_events": [
                {
                    "direction_event_id": "directionevent_" + "3" * 32,
                    "direction_id": direction_id,
                    "recorded_at": NOW.isoformat(),
                    "action": "propose",
                    "name": "establish a correct baseline",
                    "hypothesis": "a direct implementation is sufficient",
                    "rationale": "bootstrap prioritizes correctness",
                    "plan": ["implement and evaluate"],
                    "success_criteria": "full correctness passes",
                    "stop_conditions": "the evaluator is unavailable",
                    "analysis": None,
                    "supporting_experiment_ids": [],
                },
                {
                    "direction_event_id": "directionevent_" + "4" * 32,
                    "direction_id": direction_id,
                    "recorded_at": NOW.isoformat(),
                    "action": terminal_action,
                    "name": None,
                    "hypothesis": None,
                    "rationale": None,
                    "plan": [],
                    "success_criteria": None,
                    "stop_conditions": None,
                    "analysis": "blocked by Gateway" if blocked else "evaluation passed",
                    "supporting_experiment_ids": [] if blocked else [experiment_id],
                },
            ],
        }
    )


class CapturingWorkspaces:
    def __init__(self, root: Path) -> None:
        self._root = root
        self._calls = 0

    def prepare(self, manifest: LineageBootstrapManifestV2) -> SimpleNamespace:
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
        prepared: LineageBootstrapManifestV2,
        config: LineageBootstrapSessionConfig,
    ) -> LineageBootstrapSessionResult:
        self.calls += 1
        outcome = AttemptCandidateResult(
            digest("generated-kernel"),
            digest("generated-gateway-result"),
            True,
            8.25,
        )
        _record_agent_evaluate(
            self._control,
            capability_token=config.gateway_capability,
            attempt_id=prepared.bootstrap_attempt_id,
            candidate=outcome.artifact_digest,
            gateway_result=outcome.gateway_result_digest,
            idempotency_key="baseline-final",
            latency_us=8.25,
        )
        report = _bootstrap_report(
            prepared.bootstrap_attempt_id,
            candidate=outcome.artifact_digest,
            gateway_result=outcome.gateway_result_digest,
            generation=self._control.current_generation(prepared.bootstrap_attempt_id),
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
            kernel_artifact_digest=outcome.artifact_digest,
        )


class FakeBootstrapFinalizer:
    def __init__(self, control: SqliteGatewayControl) -> None:
        self._control = control

    async def finalize(
        self,
        attempt_id: object,
        kernel_artifact_digest: object,
        *,
        nominated_gateway_result_digest: object = None,
        nominated_recovery_generation: int | None = None,
    ) -> AttemptCandidateResult:
        from atrex_runtime.domain.ids import parse_artifact_digest, parse_attempt_id

        parsed_attempt = parse_attempt_id(str(attempt_id))
        candidate = parse_artifact_digest(str(kernel_artifact_digest))
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
            kernel_artifact_digest=candidate,
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
        prepared: LineageBootstrapManifestV2,
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
                report=_bootstrap_report(
                    prepared.bootstrap_attempt_id,
                    candidate=None,
                    gateway_result=None,
                    generation=self._control.current_generation(prepared.bootstrap_attempt_id),
                    blocked=True,
                ),
                report_digest=digest("blocked-report"),
                report_error=None,
                session_trace_digest=digest("blocked-trace"),
                kernel_artifact_digest=None,
            )
        return super().run(prepared, config)


class ProcessExitThenCommittingSessions(GatewayCommittingSessions):
    def run(
        self,
        prepared: LineageBootstrapManifestV2,
        config: LineageBootstrapSessionConfig,
    ) -> LineageBootstrapSessionResult:
        if not self.calls:
            self.calls += 1
            return LineageBootstrapSessionResult(
                finish_reason="process-exit-1",
                final_response="API Error: Connection lost mid-response.",
                token_usage=TokenUsage(20, 3, 10, 0),
                token_budget=100,
                report=None,
                report_digest=None,
                report_error=None,
                session_trace_digest=digest("process-exit-trace"),
                kernel_artifact_digest=None,
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
        prepared: LineageBootstrapManifestV2,
        config: LineageBootstrapSessionConfig,
    ) -> LineageBootstrapSessionResult:
        self.calls += 1
        candidate_root = self._root / f"candidate-{self.calls}"
        candidate_root.mkdir()
        candidate_root.joinpath("kernel.py").write_text("def kernel(): pass\n")
        candidate = self._artifacts.put_directory(candidate_root, ArtifactKind.KERNEL)
        gateway_result = self._artifacts.put_json(
            {"correct": True, "latency_us": 8.25}, ArtifactKind.GATEWAY_RESULT
        )
        _record_agent_evaluate(
            self._control,
            capability_token=config.gateway_capability,
            attempt_id=prepared.bootstrap_attempt_id,
            candidate=candidate,
            gateway_result=gateway_result,
            idempotency_key="baseline-final",
            latency_us=8.25,
        )
        report = _bootstrap_report(
            prepared.bootstrap_attempt_id,
            candidate=candidate,
            gateway_result=gateway_result,
            generation=self._control.current_generation(prepared.bootstrap_attempt_id),
        )
        report_digest = self._artifacts.put_json(
            report.model_dump(mode="json"), ArtifactKind.ATTEMPT_REPORT
        )
        trace_digest = self._artifacts.put_json({"session": "completed"}, ArtifactKind.SESSION_LOG)
        return LineageBootstrapSessionResult(
            finish_reason="completed",
            final_response="done",
            token_usage=TokenUsage(10, 5, 0, 0),
            token_budget=100,
            report=report,
            report_digest=report_digest,
            report_error=None,
            session_trace_digest=trace_digest,
            kernel_artifact_digest=candidate,
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
        kernel_artifact_digest: ArtifactDigest,
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
                kernel_artifact_digest,
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
            kernel_artifact_digest=kernel_artifact_digest,
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
    assert first.report_digest == digest("baseline-report")
    assert first.session_trace_digest == digest("baseline-trace")
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


def test_core_lineage_baseline_automatically_retries_process_exit(
    tmp_path: Path,
) -> None:
    registry = SqliteRegistry(tmp_path / "registry.sqlite")
    control = SqliteGatewayControl(
        tmp_path / "gateway.sqlite",
        registry,
        signing_key=b"p" * 32,
        clock=lambda: NOW,
    )
    sessions = ProcessExitThenCommittingSessions(control)
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
        max_infrastructure_retries=1,
    )
    values = {
        "bootstrap_attempt_id": parse_attempt_id("attempt_" + "d" * 32),
        "campaign_id": parse_campaign_id("campaign_" + "e" * 32),
        "lineage_id": parse_lineage_id("lineage_" + "f" * 32),
        "kernel_agent_revision_id": parse_kernel_agent_revision_id("agentrev_" + "1" * 32),
        "optimizer_digest": digest("optimizer-process-exit-retry"),
        "input_kernel_digest": digest("seed-process-exit-retry"),
        "evaluation_contract_digest": digest("contract-process-exit-retry"),
        "agent_problem_digest": digest("problem-process-exit-retry"),
        "evidence_digest": digest("evidence-process-exit-retry"),
        "dsl": Dsl.CUTEDSL,
        "operator": "flash_attention",
        "hardware_target": "sm_120",
    }

    generated = generator.generate(**values)  # type: ignore[arg-type]

    assert generated.kernel_digest == digest("generated-kernel")
    assert sessions.calls == 2
    runs = control.list_bootstrap_runs(values["bootstrap_attempt_id"])  # type: ignore[arg-type]
    assert [run.recovery_generation for run in runs] == [0, 1]
    assert [run.status.value for run in runs] == ["failed", "completed"]
    assert runs[0].finish_reason == "process-exit-1"
    worker_sessions = registry.list_worker_sessions(
        attempt_id=values["bootstrap_attempt_id"]  # type: ignore[arg-type]
    )
    assert [session.status.value for session in worker_sessions] == ["failed", "completed"]
    assert [session.recovery_generation for session in worker_sessions] == [0, 1]
    assert [
        event.kind for event in registry.list_runtime_events(after_sequence=0, limit=10)
    ] == [
        "bootstrap.lineage_baseline_failed",
        "bootstrap.lineage_baseline_retrying",
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
