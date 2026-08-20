"""Framework-neutral orchestration for the Core-owned Optimizer process."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import Protocol

from ..domain.errors import InfrastructureError
from ..domain.ids import ArtifactDigest, new_worker_session_id
from ..domain.models import (
    AttemptReportStatus,
    TokenUsage,
    WorkerSession,
    WorkerSessionRole,
    WorkerSessionStatus,
)
from ..ports import (
    AttemptOutcomeSource,
    AttemptSessionTraceRecorder,
    AuthoritativeCandidateEvaluator,
    OptimizerRunner,
    RunAttemptRequest,
    RunAttemptResult,
    RuntimeEventRecorder,
    WorkerGatewayAuthorityProvider,
    WorkerSessionRecorder,
)
from .attempt_report import AttemptReportV2
from .launcher import validate_worker_environment
from .workspace import AttemptWorkspaceAssembler, PreparedAttempt

_RUNTIME_ENVIRONMENT_KEYS = {
    "ATREX_AGENT_BACKEND",
    "ATREX_DEV_SHELL_BACKENDS",
    "ATREX_AGENT_MODEL",
    "ATREX_AGENT_REASONING_EFFORT",
    "ATREX_AGENT_SESSION_SETTINGS",
    "ATREX_CORE_PHASE",
    "ATREX_ATTEMPT_MANIFEST",
    "ATREX_ATTEMPT_REPORT_PATH",
    "ATREX_GATEWAY_CAPABILITY",
    "ATREX_GATEWAY_PROXY_URL",
    "ATREX_OPTIMIZER_REPOSITORY",
    "ATREX_SESSION_TIMEOUT_SECONDS",
    "ATREX_SESSION_TRACE_PATH",
    "ATREX_USAGE_BUDGET",
    "ATREX_USAGE_UNIT",
    "ATREX_TOKEN_USAGE_REPORT",
    "ATREX_WIKI_CAPABILITY",
    "ATREX_WIKI_PROXY_URL",
}


@dataclass(frozen=True, slots=True)
class OptimizerSessionConfig:
    """Attempt-independent environment plus per-run Runtime authority."""

    environment: tuple[tuple[str, str], ...]
    model: str | None = None
    gateway_endpoint: str | None = None
    gateway_capability: str | None = None
    wiki_endpoint: str | None = None
    wiki_capability: str | None = None

    def __post_init__(self) -> None:
        if (self.gateway_endpoint is None) != (self.gateway_capability is None):
            raise ValueError("Gateway endpoint and capability must be set together")
        if (self.wiki_endpoint is None) != (self.wiki_capability is None):
            raise ValueError("Wiki endpoint and capability must be set together")
        keys = [key for key, _value in self.environment]
        if len(keys) != len(set(keys)):
            raise ValueError("Optimizer environment contains duplicate keys")
        overlap = _RUNTIME_ENVIRONMENT_KEYS.intersection(keys)
        if overlap:
            raise ValueError(
                f"Optimizer environment overrides Runtime-owned keys: {sorted(overlap)}"
            )
        validate_worker_environment(dict(self.environment))


@dataclass(frozen=True, slots=True)
class OptimizerSessionResult:
    """Small Core process result projection consumed by the Runtime."""

    finish_reason: str | None
    final_response: str
    token_usage: TokenUsage
    token_budget: int
    session_trace_digest: ArtifactDigest | None = None
    attempt_report: AttemptReportV2 | None = None
    attempt_report_digest: ArtifactDigest | None = None
    attempt_report_error: str | None = None
    candidate_artifact_digest: ArtifactDigest | None = None

    def __post_init__(self) -> None:
        if self.token_budget <= 0:
            raise ValueError("Optimizer session token budget must be positive")


class OptimizerSessionDriver(Protocol):
    """Execute one prepared workspace through the Core-owned entrypoint."""

    async def run(
        self,
        prepared: PreparedAttempt,
        config: OptimizerSessionConfig,
    ) -> OptimizerSessionResult:
        """Run and reap the Core process before returning."""
        ...


class SessionOptimizerRunner(OptimizerRunner):
    """Translate durable Attempts into isolated Core Optimizer executions."""

    def __init__(
        self,
        workspaces: AttemptWorkspaceAssembler,
        sessions: OptimizerSessionDriver,
        outcomes: AttemptOutcomeSource,
        finalizer: AuthoritativeCandidateEvaluator,
        gateway_authorities: WorkerGatewayAuthorityProvider,
        session_traces: AttemptSessionTraceRecorder,
        events: RuntimeEventRecorder,
        config: OptimizerSessionConfig,
        *,
        independent_final_evaluation: bool = True,
        wiki_enabled: bool = False,
        worker_sessions: WorkerSessionRecorder | None = None,
        backend: str | None = None,
    ) -> None:
        self._workspaces = workspaces
        self._sessions = sessions
        self._outcomes = outcomes
        self._finalizer = finalizer
        self._gateway_authorities = gateway_authorities
        self._session_traces = session_traces
        self._events = events
        self._config = config
        self._independent_final_evaluation = independent_final_evaluation
        self._wiki_enabled = wiki_enabled
        self._worker_sessions = worker_sessions
        self._backend = backend
        if config.gateway_endpoint is not None:
            raise ValueError("Optimizer base config cannot contain pre-issued Gateway authority")
        if config.wiki_endpoint is not None:
            raise ValueError("Optimizer base config cannot contain pre-issued Wiki authority")

    async def run_attempt(self, request: RunAttemptRequest) -> RunAttemptResult:
        """Return an existing authoritative result or run one fresh Core process."""
        existing = await self._outcomes.get_outcome(request.attempt_id)
        if existing is not None:
            return RunAttemptResult(candidate=existing)

        prepared = self._workspaces.prepare(request)
        worker_session_id = new_worker_session_id()
        if self._worker_sessions is not None:
            self._worker_sessions.start_worker_session(
                WorkerSession(
                    id=worker_session_id,
                    role=WorkerSessionRole.OPTIMIZER,
                    subject_id=str(request.attempt_id),
                    external_run_id=prepared.session_id,
                    workspace_path=str(prepared.root),
                    status=WorkerSessionStatus.RUNNING,
                    started_at=datetime.now(UTC).isoformat(),
                    attempt_id=request.attempt_id,
                    backend=self._backend,
                    model=request.model,
                )
            )
        event_base = {
            "worker_role": "optimizer",
            "worker_run_id": prepared.session_id,
            "kernel_agent_revision_id": request.kernel_agent_revision_id,
            "input_kernel_revision_id": request.input_kernel_revision_id,
            "dsl": request.dsl,
            "model": request.model,
        }
        self._events.record_runtime_event("worker.started", request.attempt_id, event_base)
        exit_kind = "failed"
        try:
            authority = await self._gateway_authorities.get_authority(request)
            result = await self._sessions.run(
                prepared,
                replace(
                    self._config,
                    model=request.model,
                    gateway_endpoint=authority.endpoint,
                    gateway_capability=authority.capability,
                    wiki_endpoint=authority.endpoint if self._wiki_enabled else None,
                    wiki_capability=authority.capability if self._wiki_enabled else None,
                ),
            )
        except InfrastructureError as error:
            exit_kind = "timeout" if "wall-time limit" in str(error) else "infrastructure_failed"
            if self._worker_sessions is not None:
                self._worker_sessions.finish_worker_session(
                    worker_session_id,
                    status=(
                        WorkerSessionStatus.TIMED_OUT
                        if exit_kind == "timeout"
                        else WorkerSessionStatus.FAILED
                    ),
                    finish_reason=exit_kind,
                    error_type=type(error).__name__,
                    error_message=str(error),
                )
            self._events.record_runtime_event(
                f"worker.{exit_kind}",
                request.attempt_id,
                {**event_base, "error_type": type(error).__name__},
            )
            raise
        except Exception as error:
            if self._worker_sessions is not None:
                self._worker_sessions.finish_worker_session(
                    worker_session_id,
                    status=WorkerSessionStatus.FAILED,
                    finish_reason="infrastructure-failed",
                    error_type=type(error).__name__,
                    error_message=str(error),
                )
            self._events.record_runtime_event(
                "worker.failed",
                request.attempt_id,
                {**event_base, "error_type": type(error).__name__},
            )
            raise
        else:
            exit_kind = "exited"
            reason = result.finish_reason or "no-turn-end"
            if self._worker_sessions is not None:
                self._worker_sessions.finish_worker_session(
                    worker_session_id,
                    status=(
                        WorkerSessionStatus.COMPLETED
                        if reason == "completed"
                        else WorkerSessionStatus.FAILED
                    ),
                    finish_reason=reason,
                    trace_digest=result.session_trace_digest,
                    token_budget=result.token_budget,
                    token_usage=result.token_usage,
                )
            self._events.record_runtime_event(
                "worker.exited",
                request.attempt_id,
                {
                    **event_base,
                    "finish_reason": reason,
                    "token_budget": result.token_budget,
                    "usage_unit": result.token_usage.usage_unit,
                    "usage_budget": result.token_budget,
                    "token_usage": {
                        "uncached_input_tokens": result.token_usage.uncached_input_tokens,
                        "output_tokens": result.token_usage.output_tokens,
                        "cache_read_tokens": result.token_usage.cache_read_tokens,
                        "cache_write_tokens": result.token_usage.cache_write_tokens,
                        "total_tokens": result.token_usage.total_tokens,
                        "credits": result.token_usage.credits,
                        "consumed": result.token_usage.consumed,
                    },
                    "session_trace_digest": result.session_trace_digest,
                },
            )
        finally:
            self._events.record_runtime_event(
                "worker.cleaned",
                request.attempt_id,
                {**event_base, "preceding_status": exit_kind},
            )
        reason = result.finish_reason or "no-turn-end"
        if result.session_trace_digest is not None:
            self._session_traces.record_attempt_session_trace(
                request.attempt_id,
                result.session_trace_digest,
                reason,
                result.token_budget,
                result.token_usage,
            )
        outcome = None
        if result.attempt_report is None:
            detail = (
                f"invalid Attempt report: {result.attempt_report_error}"
                if result.attempt_report_error is not None
                else "missing terminal Attempt report"
            )
            self._events.record_runtime_event(
                "attempt.report_rejected",
                request.attempt_id,
                {**event_base, "reason": detail, "gateway_outcome_present": outcome is not None},
            )
            return RunAttemptResult(failure_reason=detail)
        if result.attempt_report_digest is None:
            raise AssertionError("validated Attempt report has no sealed Artifact")
        report = result.attempt_report
        self._events.record_runtime_event(
            "attempt.reported",
            request.attempt_id,
            {
                **event_base,
                "attempt_report_artifact_digest": result.attempt_report_digest,
                "status": report.status,
                "decision": report.decision,
                "gateway_outcome_present": outcome is not None,
            },
        )
        if reason != "completed":
            return RunAttemptResult(
                failure_reason=f"Optimizer session did not complete successfully: {reason}",
                attempt_report_digest=result.attempt_report_digest,
                attempt_report_status=AttemptReportStatus(report.status),
            )
        if report.status == "candidate_ready":
            if result.candidate_artifact_digest is None:
                return RunAttemptResult(
                    failure_reason="Attempt report declared candidate_ready without a candidate",
                    attempt_report_digest=result.attempt_report_digest,
                    attempt_report_status=AttemptReportStatus(report.status),
                )
            try:
                outcome = await self._finalizer.finalize(
                    request.attempt_id,
                    result.candidate_artifact_digest,
                    independent_evaluate=self._independent_final_evaluation,
                )
            except ValueError as error:
                return RunAttemptResult(
                    failure_reason=f"Candidate nomination was rejected: {error}",
                    attempt_report_digest=result.attempt_report_digest,
                    attempt_report_status=AttemptReportStatus(report.status),
                )
            return RunAttemptResult(
                candidate=outcome,
                attempt_report_digest=result.attempt_report_digest,
                attempt_report_status=AttemptReportStatus(report.status),
            )
        return RunAttemptResult(
            failure_reason=f"Optimizer ended the engineering loop with status {report.status}",
            attempt_report_digest=result.attempt_report_digest,
            attempt_report_status=AttemptReportStatus(report.status),
        )


__all__ = [
    "OptimizerSessionConfig",
    "OptimizerSessionDriver",
    "OptimizerSessionResult",
    "SessionOptimizerRunner",
]
