"""Trusted Gateway Proxy service and minimal ASGI transport."""

from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import shutil
import statistics
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Protocol, cast

from pydantic import TypeAdapter, ValidationError

from ..artifacts.local import ArtifactKind, JsonValue, LocalArtifactStore
from ..asgi import AsgiReceive, AsgiSend, bearer_token, json_response, read_request_body
from ..domain.errors import (
    DirectionConcurrencyError,
    InfrastructureError,
    InvalidTransitionError,
    UpstreamGatewayError,
)
from ..domain.ids import ArtifactDigest, AttemptId, parse_artifact_digest
from ..ports import RuntimeEventRecorder
from ..serialization import canonical_json_digest
from .contract import AgateEvaluationContextResolver, candidate_path_for_attempt
from .control import SqliteGatewayControl
from .control_models import (
    GatewayCapability,
    GatewayEvaluationSource,
    GatewayKernelTrialRecord,
    GatewayMeasurementRecord,
    GatewayOperation,
    gateway_kernel_trial_id,
)
from .correctness import correctness_summary
from .diff_policy import RegistryCandidateDiffValidator
from .journals import RuntimeJournalService
from .measurement_history import normalized_measurement_points
from .production_policy import CandidateProductionValidator
from .protocol import (
    GATEWAY_PROXY_PROTOCOL_VERSION,
    AttemptReportRequestV2,
    CancelRequestV2,
    CandidateBundleV2,
    CandidateFileV2,
    CheckRequestV2,
    DevRequestV2,
    DirectionHistoryRequestV2,
    DisassembleRequestV2,
    EvaluateRequestV2,
    EvaluationV2,
    ExperimentHistoryRequestV2,
    GatewayProxyRequestV2,
    GatewayProxyResponseV2,
    KernelArtifactReadRequestV2,
    KernelTrialShowRequestV2,
    PollRequestV2,
    ProfileRequestV2,
    ResultArtifactReadRequestV2,
    gateway_agent_request_schema,
)

_REQUEST_ADAPTER: TypeAdapter[GatewayProxyRequestV2] = TypeAdapter(GatewayProxyRequestV2)
_CANDIDATE_REQUEST_TYPES = (
    AttemptReportRequestV2,
    EvaluateRequestV2,
    ProfileRequestV2,
    DevRequestV2,
    CheckRequestV2,
    DisassembleRequestV2,
)
# Nomination already fails closed on the same policy, so attempt_report needs no advisory.
_ADVISORY_PRODUCTION_GATE_TYPES = (
    ProfileRequestV2,
    DevRequestV2,
    CheckRequestV2,
    DisassembleRequestV2,
)
_RUNTIME_LOCAL_OPERATIONS = frozenset(
    {
        GatewayOperation.ATTEMPT_REPORT,
        GatewayOperation.KERNEL_TRIAL_SHOW,
        GatewayOperation.KERNEL_ARTIFACT_READ,
        GatewayOperation.RESULT_ARTIFACT_READ,
        GatewayOperation.DIRECTION_HISTORY,
        GatewayOperation.EXPERIMENT_HISTORY,
    }
)
_RUNTIME_JOURNAL_OPERATIONS = frozenset(
    {
        GatewayOperation.DIRECTION_UPDATE,
        GatewayOperation.DIRECTIONS_LIST,
        GatewayOperation.DIRECTION_LOAD,
        GatewayOperation.EXPERIMENT_RECORD,
        GatewayOperation.EXPERIMENTS_LIST,
        GatewayOperation.EXPERIMENT_LOAD,
        GatewayOperation.JOURNAL_SNAPSHOT,
    }
)
_AGENT_GATEWAY_OPERATIONS = frozenset(
    operation
    for operation in GatewayOperation
    if operation not in _RUNTIME_LOCAL_OPERATIONS
    and operation not in _RUNTIME_JOURNAL_OPERATIONS
    and operation
    not in {
        GatewayOperation.POLL,
        GatewayOperation.JOBS,
        GatewayOperation.CANCEL,
        GatewayOperation.HEALTH,
        GatewayOperation.CONFIG,
    }
)
# These answer "what is true right now", so replaying a committed response would pin a
# poll to the job state observed on its first call and never report completion.
_OBSERVATIONAL_OPERATIONS = frozenset(
    {
        GatewayOperation.POLL,
        GatewayOperation.JOBS,
        GatewayOperation.ENV,
        GatewayOperation.HEALTH,
        GatewayOperation.CONFIG,
    }
)
_RUNTIME_OWNED_ERROR_FIELDS = frozenset(
    {"schema_version", "attempt_id", "candidate", "idempotency_key"}
)
_GATEWAY_OPERATION_NAMES = frozenset(operation.value for operation in GatewayOperation)


def _dev_file_text(file: CandidateFileV2) -> str:
    """Decode one dev file, which the Agate payload carries as text."""
    try:
        return str(file.content().decode("utf-8"))
    except UnicodeDecodeError as error:
        raise ValueError(f"dev file must be UTF-8 text: {file.path}") from error


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _finite_number(value: object, *, positive: bool = False) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number) or (positive and number <= 0):
        return None
    return number


def _profile_duration_us(kernel: Mapping[str, object]) -> float | None:
    duration = _finite_number(kernel.get("duration"), positive=True)
    unit = kernel.get("duration_unit")
    if duration is None or not isinstance(unit, str):
        return _finite_number(kernel.get("duration_us"), positive=True)
    scale = {"ns": 0.001, "us": 1.0, "ms": 1_000.0, "s": 1_000_000.0}.get(unit)
    return None if scale is None else duration * scale


def _canonical_profile_kernel(raw: Mapping[str, object]) -> dict[str, JsonValue]:
    """Normalize one profiler Kernel before sealing the public Result Artifact."""
    kernel = cast(dict[str, JsonValue], dict(raw))
    name = raw.get("name", raw.get("kernel_name"))
    if isinstance(name, str) and name:
        kernel["name"] = name
    kernel.pop("kernel_name", None)

    duration_us = _profile_duration_us(raw)
    if duration_us is not None:
        kernel["duration_us"] = duration_us
    kernel.pop("duration", None)
    kernel.pop("duration_unit", None)

    memory_sol = _finite_number(raw.get("memory_sol_pct"))
    if memory_sol is None:
        memory_sol = _finite_number(raw.get("mem_sol_pct"))
    if memory_sol is not None:
        kernel["memory_sol_pct"] = memory_sol
    kernel.pop("mem_sol_pct", None)

    for source, target in (
        ("registers", "registers_per_thread"),
        ("smem_bytes", "shared_memory_bytes"),
    ):
        value = _finite_number(raw.get(target))
        if value is None:
            value = _finite_number(raw.get(source))
        if value is not None:
            kernel[target] = value
        kernel.pop(source, None)

    compute_sol = _finite_number(raw.get("compute_sol_pct"))
    bound = raw.get("bound")
    if (
        (not isinstance(bound, str) or not bound)
        and compute_sol is not None
        and memory_sol is not None
    ):
        kernel["bound"] = "compute" if compute_sol > memory_sol else "memory"
    return kernel


def _canonical_profile_result(
    raw: Mapping[str, object],
    request: GatewayAdapterRequest,
) -> dict[str, JsonValue]:
    kernels_value = raw.get("kernels")
    if not isinstance(kernels_value, list):
        return cast(dict[str, JsonValue], dict(raw))
    kernels = [
        _canonical_profile_kernel(item) for item in kernels_value if isinstance(item, Mapping)
    ]
    durations = [
        duration
        for kernel in kernels
        if (duration := _finite_number(kernel.get("duration_us"), positive=True)) is not None
    ]
    total_duration = sum(durations)
    if total_duration > 0:
        for kernel in kernels:
            duration = _finite_number(kernel.get("duration_us"), positive=True)
            if duration is not None:
                kernel["duration_share_pct"] = duration * 100.0 / total_duration

    weighted_sol = 0.0
    weighted_sol_duration = 0.0
    weighted_compute = 0.0
    weighted_memory = 0.0
    for kernel in kernels:
        duration = _finite_number(kernel.get("duration_us"), positive=True)
        compute = _finite_number(kernel.get("compute_sol_pct"))
        memory = _finite_number(kernel.get("memory_sol_pct"))
        if duration is None or compute is None or memory is None:
            continue
        weighted_sol += max(compute, memory) * duration
        weighted_sol_duration += duration
        weighted_compute += compute * duration
        weighted_memory += memory * duration

    normalized = cast(
        dict[str, JsonValue],
        {key: value for key, value in raw.items() if key not in {"kernels", "shape_id"}},
    )
    shape_id = request.parameters.get("shape_id", raw.get("shape_id"))
    if isinstance(shape_id, int) and not isinstance(shape_id, bool) and shape_id >= 0:
        normalized["shape_id"] = str(shape_id)
    elif isinstance(shape_id, str) and shape_id.isdecimal():
        normalized["shape_id"] = shape_id
    if request.profile_level:
        normalized["profile_level"] = request.profile_level
    normalized["kernel_count"] = len(kernels)
    if total_duration > 0:
        normalized["total_duration_us"] = total_duration
        dominant = max(
            kernels,
            key=lambda item: _finite_number(item.get("duration_us"), positive=True) or 0.0,
        )
        name = dominant.get("name")
        if isinstance(name, str) and name:
            normalized["dominant_kernel"] = name
    if weighted_sol_duration > 0:
        normalized["weighted_sol_pct"] = weighted_sol / weighted_sol_duration
        normalized["dominant_bound"] = (
            "compute" if weighted_compute > weighted_memory else "memory"
        )
    normalized["kernels"] = cast(JsonValue, kernels)
    return normalized


def _canonical_agent_result(
    request: GatewayAdapterRequest,
    adapter_result: GatewayAdapterResult,
    payload: JsonValue,
) -> JsonValue:
    """Create the one public result reused by initial execution and later reads."""
    if request.operation is GatewayOperation.EVALUATE and isinstance(payload, dict):
        by_shape = payload.get("latency_us_by_shape")
        shape_latencies = (
            [
                float(item)
                for item in by_shape.values()
                if isinstance(item, (int, float))
                and not isinstance(item, bool)
                and math.isfinite(float(item))
                and float(item) > 0
            ]
            if isinstance(by_shape, dict)
            else []
        )
        evaluation = adapter_result.evaluation
        correct = evaluation.correct if evaluation is not None else payload.get("correct")
        latency = _finite_number(
            None if evaluation is None else evaluation.latency_us,
            positive=True,
        )
        if not isinstance(correct, bool):
            correct = payload.get("all_pass")
        if not isinstance(latency, (int, float)) or isinstance(latency, bool):
            latency = _finite_number(payload.get("latency_us_geomean"), positive=True)
        normalized: dict[str, JsonValue] = {
            key: payload[key]
            for key in ("failures", "error", "production_gate")
            if key in payload
        }
        normalized.update(
            {
                "correct": correct,
                "correctness": correctness_summary(payload, passed=correct is True),
                "latency_us_geomean": latency,
                "latency_us_arith_mean": (
                    statistics.fmean(shape_latencies) if shape_latencies else None
                ),
                "latency_us_by_shape": (
                    by_shape if isinstance(by_shape, dict) else {}
                ),
            }
        )
        return cast(JsonValue, normalized)
    if request.operation is GatewayOperation.PROFILE and isinstance(payload, dict):
        normalized = cast(dict[str, JsonValue], dict(payload))
        profile = payload.get("result")
        if isinstance(profile, Mapping):
            normalized["result"] = _canonical_profile_result(profile, request)
        return cast(JsonValue, normalized)
    return payload


def _validate_canonical_agent_result(value: dict[str, object]) -> dict[str, JsonValue]:
    if set(value) != {"operation", "status", "result"}:
        raise InfrastructureError("Result Artifact has an invalid canonical result")
    operation = value.get("operation")
    status = value.get("status")
    if operation not in _GATEWAY_OPERATION_NAMES or status not in {
        "completed",
        "queued",
        "failed",
        "cancelled",
    }:
        raise InfrastructureError("Result Artifact has invalid operation status")
    return cast(dict[str, JsonValue], value)


@dataclass(frozen=True, slots=True)
class GatewayAdapterRequest:
    """Trusted normalized request passed to an external Gateway implementation."""

    attempt_id: AttemptId
    operation: GatewayOperation
    idempotency_key: str
    candidate_digest: ArtifactDigest | None
    candidate_path: Path | None
    profile_level: str | None
    kernel_regex: str | None
    job_id: str | None
    parameters: dict[str, JsonValue] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class GatewayAdapterResult:
    """Validated external Gateway result before Runtime persistence."""

    status: Literal["completed", "queued", "failed", "cancelled"]
    result: JsonValue
    job_id: str | None = None
    evaluation: EvaluationV2 | None = None
    profile_result: JsonValue | None = None
    worker_result: JsonValue | None = None


class GatewayAdapter(Protocol):
    """Execute normalized requests against one concrete Gateway deployment."""

    async def execute(self, request: GatewayAdapterRequest) -> GatewayAdapterResult:
        """Return a structured result; repeated idempotency keys must be safe."""
        ...


@dataclass(frozen=True, slots=True)
class GatewayProxyLimits:
    """Complete request and candidate acquisition bounds."""

    max_request_bytes: int
    max_candidate_files: int
    max_candidate_bytes: int

    def __post_init__(self) -> None:
        if min(self.max_request_bytes, self.max_candidate_files, self.max_candidate_bytes) <= 0:
            raise ValueError("Gateway Proxy limits must be positive")


class GatewayProxyService:
    """Authorize, seal, execute, and persist one Worker Gateway operation."""

    def __init__(
        self,
        control: SqliteGatewayControl,
        artifacts: LocalArtifactStore,
        adapter: GatewayAdapter,
        limits: GatewayProxyLimits,
        events: RuntimeEventRecorder,
        candidate_diff: RegistryCandidateDiffValidator | None = None,
        candidate_production: CandidateProductionValidator | None = None,
        *,
        contexts: AgateEvaluationContextResolver | None = None,
        clock: Callable[[], datetime] = _utc_now,
    ) -> None:
        self._control = control
        self._artifacts = artifacts
        self._adapter = adapter
        self._limits = limits
        self._events = events
        self._candidate_diff = candidate_diff
        self._candidate_production = candidate_production
        self._contexts = contexts
        self._clock = clock
        self._journals = RuntimeJournalService(control, artifacts)

    async def execute(
        self,
        token: str,
        payload: bytes,
        *,
        operation_scope: Literal["gateway", "runtime", "journal"] | None = None,
    ) -> GatewayProxyResponseV2:
        """Parse and execute one complete protocol request."""
        if len(payload) > self._limits.max_request_bytes:
            raise ValueError("Gateway Proxy request exceeds byte limit")
        request = _REQUEST_ADAPTER.validate_json(payload)
        operation = GatewayOperation(request.operation)
        is_runtime_query = operation in _RUNTIME_LOCAL_OPERATIONS
        is_runtime_journal = operation in _RUNTIME_JOURNAL_OPERATIONS
        if operation_scope == "gateway" and is_runtime_query:
            raise ValueError(
                f"Runtime-local operation {operation.value!r} requires /v1/runtime/queries"
            )
        if operation_scope == "gateway" and is_runtime_journal:
            raise ValueError(
                f"Runtime Journal operation {operation.value!r} requires /v1/runtime/journals"
            )
        if operation_scope == "gateway" and operation not in _AGENT_GATEWAY_OPERATIONS:
            raise ValueError(f"Gateway operation {operation.value!r} is not exposed to Agents")
        if operation_scope == "runtime" and is_runtime_journal:
            raise ValueError(
                f"Runtime Journal operation {operation.value!r} requires /v1/runtime/journals"
            )
        if operation_scope == "runtime" and not is_runtime_query:
            raise ValueError(f"Gateway operation {operation.value!r} requires /v1/operations")
        if operation_scope == "journal" and is_runtime_query:
            raise ValueError(
                f"Runtime-local operation {operation.value!r} requires /v1/runtime/queries"
            )
        if operation_scope == "journal" and not is_runtime_journal:
            raise ValueError(f"Gateway operation {operation.value!r} requires /v1/operations")
        request_digest = canonical_json_digest(request.model_dump(mode="json"))
        authorization = self._control.authorize(
            GatewayCapability(token, request.attempt_id),
            operation,
            idempotency_key=request.idempotency_key,
            request_digest=str(request_digest),
        )
        replayable = operation not in _OBSERVATIONAL_OPERATIONS
        existing_response = (
            self._control.get_operation_artifact(
                request.attempt_id,
                request.idempotency_key,
                operation,
            )
            if replayable
            else None
        )
        if existing_response is not None:
            return self._load_response(existing_response)

        candidate_digest: ArtifactDigest | None = None
        candidate_path: Path | None = None
        production_violations: tuple[str, ...] = ()
        if isinstance(request, _CANDIDATE_REQUEST_TYPES):
            if isinstance(request, DevRequestV2):
                self._validate_dev_files(request)
            candidate_digest = self._seal_candidate(request.candidate, request.attempt_id)
            self._control.bind_operation_candidate(
                request.attempt_id,
                request.idempotency_key,
                operation,
                candidate_digest,
                recovery_generation=authorization.recovery_generation,
            )
            if self._candidate_diff is not None and isinstance(request, EvaluateRequestV2):
                self._candidate_diff.validate(request.attempt_id, candidate_digest)
            if self._candidate_production is not None:
                if isinstance(request, EvaluateRequestV2):
                    self._candidate_production.validate(request.attempt_id, candidate_digest)
                elif isinstance(request, _ADVISORY_PRODUCTION_GATE_TYPES):
                    production_violations = self._candidate_production.violations(
                        request.attempt_id, candidate_digest
                    )
            candidate_path = self._artifacts.verify(candidate_digest).payload_path
        adapter_request = self._adapter_request(
            request,
            operation,
            candidate_digest,
            candidate_path,
        )
        event_base = {
            "operation": operation,
            "idempotency_key": request.idempotency_key,
            "request_digest": request_digest,
            "kernel_artifact_digest": candidate_digest,
        }
        self._events.record_runtime_event(
            "gateway.operation_submitted",
            request.attempt_id,
            event_base,
        )
        try:
            if isinstance(request, AttemptReportRequestV2):
                result = self._register_attempt_report(request, candidate_digest)
            elif isinstance(request, KernelTrialShowRequestV2):
                result = self._show_kernel_trial(request)
            elif isinstance(request, KernelArtifactReadRequestV2):
                result = self._read_kernel_artifact(request)
            elif isinstance(request, ResultArtifactReadRequestV2):
                result = self._read_result_artifact(request)
            elif isinstance(request, DirectionHistoryRequestV2):
                result = self._read_journal_history(request.attempt_id, "direction_events")
            elif isinstance(request, ExperimentHistoryRequestV2):
                result = self._read_journal_history(request.attempt_id, "experiments")
            elif operation in _RUNTIME_JOURNAL_OPERATIONS:
                result = GatewayAdapterResult(
                    "completed",
                    cast(JsonValue, self._journals.execute(request, authorization)),
                )
            else:
                result = await self._adapter.execute(adapter_request)
            agent_payload = result.result if result.worker_result is None else result.worker_result
            if production_violations:
                agent_payload = _with_production_gate_advisory(
                    agent_payload, production_violations
                )
            agent_payload = _canonical_agent_result(
                adapter_request,
                result,
                agent_payload,
            )
            result_digest = self._store_gateway_result(result)
            if replayable:
                self._control.bind_operation_gateway_result(
                    request.attempt_id,
                    request.idempotency_key,
                    operation,
                    result_digest,
                    recovery_generation=authorization.recovery_generation,
                )

            if isinstance(request, EvaluateRequestV2):
                if result.status != "completed" or result.evaluation is None:
                    raise InfrastructureError("evaluate did not return a completed evaluation")
                if candidate_digest is None:
                    raise AssertionError("evaluate candidate was not sealed")
                evaluation_record = self._control.record_evaluation(
                    request.attempt_id,
                    source=GatewayEvaluationSource.AGENT,
                    idempotency_key=request.idempotency_key,
                    kernel_artifact_digest=candidate_digest,
                    gateway_result_digest=result_digest,
                    correct=result.evaluation.correct,
                    latency_us=result.evaluation.latency_us,
                    agate_job_id=result.job_id,
                    recovery_generation=authorization.recovery_generation,
                )
            else:
                evaluation_record = None
            measurement_records: tuple[GatewayMeasurementRecord, ...] = ()
            if candidate_digest is not None and request.operation in {"evaluate", "profile"}:
                measurement_records = self._control.record_measurements(
                    request.attempt_id,
                    source_operation=operation,
                    idempotency_key=request.idempotency_key,
                    kernel_artifact_digest=candidate_digest,
                    gateway_result_digest=result_digest,
                    points=normalized_measurement_points(adapter_request, result),
                    recovery_generation=authorization.recovery_generation,
                )
        except Exception as error:
            self._events.record_runtime_event(
                "gateway.operation_failed",
                request.attempt_id,
                {**event_base, "error_type": type(error).__name__},
            )
            raise

        self._events.record_runtime_event(
            "gateway.operation_completed",
            request.attempt_id,
            {
                **event_base,
                "status": result.status,
                "gateway_result_digest": result_digest,
                "job_id": result.job_id,
                "correct": None if result.evaluation is None else result.evaluation.correct,
                "latency_us": (None if result.evaluation is None else result.evaluation.latency_us),
                "profile_status": (
                    None
                    if not isinstance(result.profile_result, dict)
                    else result.profile_result.get("status")
                ),
                "normalized_measurement_count": len(measurement_records),
            },
        )

        kernel_trial_id = (
            None
            if candidate_digest is None
            else gateway_kernel_trial_id(
                request.attempt_id,
                authorization.recovery_generation,
                candidate_digest,
            )
        )
        result_artifact_digest = self._store_result_artifact(
            operation=request.operation,
            status=result.status,
            kernel_artifact_digest=(None if candidate_digest is None else str(candidate_digest)),
            kernel_trial_id=kernel_trial_id,
            job_id=result.job_id,
            evaluation=result.evaluation,
            result=agent_payload,
        )
        if replayable:
            self._control.commit_operation_artifact(
                request.attempt_id,
                request.idempotency_key,
                operation,
                result_artifact_digest,
            )
        response = self._load_response(result_artifact_digest)
        if evaluation_record is not None:
            self._events.record_runtime_event(
                "gateway.evaluation_recorded",
                request.attempt_id,
                {
                    **event_base,
                    "evaluation_id": evaluation_record.id,
                    "evaluation_ordinal": evaluation_record.ordinal,
                    "source": evaluation_record.source.value,
                },
            )
        return response

    def _register_attempt_report(
        self,
        request: AttemptReportRequestV2,
        candidate_digest: ArtifactDigest | None,
    ) -> GatewayAdapterResult:
        if candidate_digest is None:
            raise InfrastructureError("Attempt report sealed no candidate")
        if request.report.status == "candidate_ready":
            evaluation = self._control.find_agent_evaluation(request.attempt_id, candidate_digest)
            if evaluation is None:
                raise ValueError(
                    "candidate_ready requires a completed Agent evaluate for the exact current "
                    f"work/kernel tree, sealed as {candidate_digest}. No Agent evaluate covers it; "
                    'run {"operation": "evaluate"} and submit this report again'
                )
            if not evaluation.correct:
                raise ValueError(
                    "candidate_ready requires a correct Agent evaluate for the exact current "
                    f"work/kernel tree, sealed as {candidate_digest}. Its evaluate reported "
                    "incorrect results; repair the candidate, re-evaluate, and submit again"
                )
        return GatewayAdapterResult(
            "completed",
            cast(
                JsonValue,
                {
                    "status": "registered",
                    "candidate_digest": str(candidate_digest),
                    "report_status": request.report.status,
                },
            ),
        )

    def _show_kernel_trial(self, request: KernelTrialShowRequestV2) -> GatewayAdapterResult:
        _, visible_attempt_ids = self._control.visible_kernel_trial_attempt_ids(request.attempt_id)
        trial = next(
            (
                value
                for value in self._control.list_kernel_trials(visible_attempt_ids, limit=5_000)
                if value.id == request.kernel_trial_id
            ),
            None,
        )
        if trial is None:
            raise ValueError("Kernel Trial is outside the visible Lineage history")
        return GatewayAdapterResult(
            "completed",
            cast(
                JsonValue,
                {
                    "kernel_artifact_digest": trial.kernel_artifact_digest,
                    "result_artifacts": self._trial_result_artifacts(trial),
                },
            ),
        )

    def _trial_result_artifacts(self, trial: GatewayKernelTrialRecord) -> list[JsonValue]:
        """Return a compact index; callers expand selected results explicitly."""
        values: list[JsonValue] = []
        seen: set[ArtifactDigest] = set()
        for observation in trial.observations:
            digest = observation.result_artifact_digest
            if digest is None or digest in seen:
                continue
            seen.add(digest)
            canonical = self._result_artifact_payload(digest)
            operation = canonical.get("operation")
            status = canonical.get("status")
            if operation != observation.operation.value or not isinstance(status, str):
                raise InfrastructureError(
                    "Kernel Trial Result Artifact disagrees with its recorded observation"
                )
            values.append(
                {
                    "result_artifact_digest": str(digest),
                    "operation": operation,
                    "status": status,
                }
            )
        return values

    def _read_kernel_artifact(
        self,
        request: KernelArtifactReadRequestV2,
    ) -> GatewayAdapterResult:
        lineage_id, visible_attempt_ids = self._control.visible_kernel_trial_attempt_ids(
            request.attempt_id
        )
        digest = parse_artifact_digest(request.kernel_artifact_digest)
        matching = tuple(
            trial
            for trial in self._control.list_kernel_trials(visible_attempt_ids, limit=5_000)
            if trial.kernel_artifact_digest == digest
        )
        if not matching:
            raise ValueError("Kernel Artifact is outside the visible Lineage history")
        response = self._kernel_artifact_payload(digest, request.file)
        response.update(
            {
                "lineage_id": lineage_id,
                "through_attempt_id": request.attempt_id,
                "kernel_trial_ids": [trial.id for trial in matching],
            }
        )
        return GatewayAdapterResult("completed", cast(JsonValue, response))

    def _read_result_artifact(
        self,
        request: ResultArtifactReadRequestV2,
    ) -> GatewayAdapterResult:
        _, visible_attempt_ids = self._control.visible_kernel_trial_attempt_ids(request.attempt_id)
        digest = parse_artifact_digest(request.result_artifact_digest)
        matching = tuple(
            (trial, observation)
            for trial in self._control.list_kernel_trials(visible_attempt_ids, limit=5_000)
            for observation in trial.observations
            if observation.result_artifact_digest == digest
        )
        if not matching:
            raise ValueError("Result Artifact is outside the visible Lineage history")
        response = self._result_artifact_payload(digest)
        return GatewayAdapterResult("completed", cast(JsonValue, response))

    def _read_journal_history(
        self,
        attempt_id: AttemptId,
        field: Literal["direction_events", "experiments"],
    ) -> GatewayAdapterResult:
        """Read frozen journals from terminal Attempt Report Artifacts."""
        journals: list[JsonValue] = []
        for report_attempt_id, digest in self._control.visible_attempt_report_artifacts(attempt_id):
            artifact = self._artifacts.verify(digest)
            if artifact.kind is not ArtifactKind.ATTEMPT_REPORT:
                raise InfrastructureError("Attempt Report history has an invalid Artifact kind")
            try:
                report = json.loads((artifact.payload_path / "value.json").read_bytes())
            except (FileNotFoundError, json.JSONDecodeError) as error:
                raise InfrastructureError("Attempt Report history is invalid JSON") from error
            if not isinstance(report, dict) or report.get("attempt_id") != report_attempt_id:
                raise InfrastructureError("Attempt Report history disagrees with Registry state")
            journal = report.get(field)
            if not isinstance(journal, list):
                raise InfrastructureError(f"Attempt Report has invalid {field}")
            if journal:
                journals.append(cast(JsonValue, journal))
        return GatewayAdapterResult(
            "completed",
            cast(JsonValue, {"journals": journals}),
        )

    def _kernel_artifact_payload(
        self,
        digest: ArtifactDigest,
        requested_file: str | None,
    ) -> dict[str, JsonValue]:
        artifact = self._artifacts.verify(digest)
        if artifact.kind is not ArtifactKind.KERNEL:
            raise ValueError("Kernel source Artifact has an invalid kind")
        files: list[JsonValue] = []
        for path in sorted(
            candidate for candidate in artifact.payload_path.rglob("*") if candidate.is_file()
        ):
            relative = path.relative_to(artifact.payload_path).as_posix()
            content = path.read_bytes()
            files.append(
                {
                    "path": relative,
                    "size_bytes": len(content),
                    "sha256": hashlib.sha256(content).hexdigest(),
                }
            )
        response: dict[str, JsonValue] = {
            "kernel_artifact_digest": digest,
            "files": files,
        }
        if requested_file is not None:
            source = artifact.payload_path.joinpath(*requested_file.split("/"))
            if not source.is_file():
                raise ValueError(f"Kernel Artifact file does not exist: {requested_file}")
            content = source.read_bytes()
            try:
                response["content"] = content.decode("utf-8")
                response["encoding"] = "utf-8"
            except UnicodeDecodeError:
                response["content"] = base64.b64encode(content).decode("ascii")
                response["encoding"] = "base64"
            response["file"] = requested_file
        return response

    def _result_artifact_payload(
        self,
        result_artifact_digest: ArtifactDigest,
    ) -> dict[str, JsonValue]:
        """Read the canonical Agent-facing value from one Result Artifact."""
        artifact = self._artifacts.verify(result_artifact_digest)
        if artifact.kind is ArtifactKind.RESULT_ARTIFACT:
            try:
                canonical = json.loads((artifact.payload_path / "value.json").read_bytes())
            except (json.JSONDecodeError, UnicodeDecodeError, OSError) as error:
                raise InfrastructureError("Result Artifact is invalid JSON") from error
            if not isinstance(canonical, dict):
                raise InfrastructureError("Result Artifact has no canonical result")
            return _validate_canonical_agent_result(canonical)
        if artifact.kind is not ArtifactKind.GATEWAY_RESULT:
            raise InfrastructureError("Result Artifact has an invalid kind")
        # Existing workspaces stored the full proxy response as a Gateway-result
        # Artifact. Treat it as the historical Result Artifact during recovery.
        try:
            value = json.loads((artifact.payload_path / "value.json").read_bytes())
        except (json.JSONDecodeError, UnicodeDecodeError, OSError) as error:
            raise InfrastructureError("Result Artifact contains invalid JSON") from error
        if not isinstance(value, dict):
            raise InfrastructureError("Result Artifact has no value.json")
        return self._normalize_recorded_response(value)

    @staticmethod
    def _normalize_recorded_response(value: dict[str, JsonValue]) -> dict[str, JsonValue]:
        operation = value.get("operation")
        status = value.get("status")
        result = value.get("result")
        if not isinstance(operation, str) or not isinstance(status, str):
            raise InfrastructureError("Gateway response Artifact has invalid operation status")
        if not isinstance(result, dict) or operation != GatewayOperation.EVALUATE.value:
            return {"operation": operation, "status": status, "result": result}

        normalized: dict[str, JsonValue] = {}
        for key in ("failures", "error"):
            if key in result:
                normalized[key] = result[key]
        evaluation = value.get("evaluation")
        correct = evaluation.get("correct") if isinstance(evaluation, dict) else None
        latency = evaluation.get("latency_us") if isinstance(evaluation, dict) else None
        if not isinstance(correct, bool):
            correct = result.get("correct", result.get("all_pass"))
        if not isinstance(latency, (int, float)) or isinstance(latency, bool):
            latency = result.get("latency_us_geomean")
        by_shape = result.get("latency_us_by_shape")
        shape_latencies = (
            [
                float(item)
                for item in by_shape.values()
                if isinstance(item, (int, float))
                and not isinstance(item, bool)
                and math.isfinite(float(item))
                and float(item) > 0
            ]
            if isinstance(by_shape, dict)
            else []
        )
        normalized["correct"] = correct
        normalized["correctness"] = correctness_summary(result, passed=correct is True)
        normalized["latency_us_geomean"] = latency
        normalized["latency_us_arith_mean"] = (
            statistics.fmean(shape_latencies) if shape_latencies else None
        )
        normalized["latency_us_by_shape"] = by_shape if isinstance(by_shape, dict) else {}
        return {
            "operation": operation,
            "status": status,
            "result": normalized,
        }

    def _store_gateway_result(self, result: GatewayAdapterResult) -> ArtifactDigest:
        """Seal the private authoritative Gateway result for Runtime consumers."""
        if result.profile_result is None:
            return self._artifacts.put_json(result.result, ArtifactKind.GATEWAY_RESULT)
        temporary = Path(tempfile.mkdtemp(prefix="gateway-result-"))
        try:
            for name, value in (
                ("value.json", result.result),
                ("profile.json", result.profile_result),
            ):
                temporary.joinpath(name).write_text(
                    json.dumps(
                        value,
                        ensure_ascii=False,
                        allow_nan=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    encoding="utf-8",
                )
            return self._artifacts.put_directory(temporary, ArtifactKind.GATEWAY_RESULT)
        finally:
            shutil.rmtree(temporary, ignore_errors=True)

    def _store_result_artifact(
        self,
        *,
        operation: str,
        status: str,
        kernel_artifact_digest: str | None,
        kernel_trial_id: str | None,
        job_id: str | None,
        evaluation: EvaluationV2 | None,
        result: JsonValue,
    ) -> ArtifactDigest:
        """Seal the exact normalized result and replay metadata exposed to the Agent."""
        temporary = Path(tempfile.mkdtemp(prefix="result-artifact-"))
        try:
            documents: tuple[tuple[str, JsonValue], ...] = (
                (
                    "value.json",
                    {"operation": operation, "status": status, "result": result},
                ),
                (
                    "metadata.json",
                    {
                        "schema_version": GATEWAY_PROXY_PROTOCOL_VERSION,
                        "kernel_artifact_digest": kernel_artifact_digest,
                        "kernel_trial_id": kernel_trial_id,
                        "job_id": job_id,
                        "evaluation": (
                            None if evaluation is None else evaluation.model_dump(mode="json")
                        ),
                    },
                ),
            )
            for name, value in documents:
                temporary.joinpath(name).write_text(
                    json.dumps(
                        value,
                        ensure_ascii=False,
                        allow_nan=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    encoding="utf-8",
                )
            return self._artifacts.put_directory(temporary, ArtifactKind.RESULT_ARTIFACT)
        finally:
            shutil.rmtree(temporary, ignore_errors=True)

    def _load_response(self, digest: ArtifactDigest) -> GatewayProxyResponseV2:
        stored = self._artifacts.verify(digest)
        try:
            value = json.loads((stored.payload_path / "value.json").read_bytes())
        except (json.JSONDecodeError, OSError) as error:
            raise InfrastructureError("cached Gateway operation response is invalid") from error
        if not isinstance(value, dict):
            raise InfrastructureError("cached Gateway operation response is invalid")
        if stored.kind is ArtifactKind.RESULT_ARTIFACT:
            try:
                metadata = json.loads((stored.payload_path / "metadata.json").read_bytes())
            except (json.JSONDecodeError, OSError) as error:
                raise InfrastructureError("cached Result Artifact metadata is invalid") from error
            if not isinstance(metadata, dict):
                raise InfrastructureError("cached Result Artifact metadata is invalid")
            response_value = {
                **metadata,
                **value,
                "result_artifact_digest": str(digest),
            }
        elif stored.kind is ArtifactKind.GATEWAY_RESULT:
            # Existing workspaces persisted the complete response under the old
            # Artifact kind and exposed its private Gateway Result Digest.
            response_value = dict(value)
            response_value.pop("gateway_result_digest", None)
            response_value["result_artifact_digest"] = str(digest)
        else:
            raise InfrastructureError("cached Gateway operation has the wrong Artifact kind")
        try:
            return GatewayProxyResponseV2.model_validate(response_value)
        except ValueError as error:
            raise InfrastructureError("cached Gateway operation response is invalid") from error

    def _validate_dev_files(self, request: DevRequestV2) -> None:
        """Bound extra dev files by the same limits that protect candidate acquisition."""
        if len(request.files) > self._limits.max_candidate_files:
            raise ValueError("dev files exceed file-count limit")
        total = sum(len(file.content()) for file in request.files)
        if total > self._limits.max_candidate_bytes:
            raise ValueError("dev files exceed decoded byte limit")

    def _seal_candidate(
        self,
        candidate: CandidateBundleV2,
        attempt_id: AttemptId,
    ) -> ArtifactDigest:
        if len(candidate.files) > self._limits.max_candidate_files:
            raise ValueError("candidate exceeds file-count limit")
        decoded = [(file.path, file.content()) for file in candidate.files]
        if sum(len(content) for _path, content in decoded) > self._limits.max_candidate_bytes:
            raise ValueError("candidate exceeds decoded byte limit")
        # Evaluation forwards only the contract's candidate file, so anything else in
        # the bundle would change this address without changing what was measured.
        selected = candidate_path_for_attempt(self._contexts, attempt_id)
        if selected is not None:
            decoded = [(path, content) for path, content in decoded if path == selected]
            if not decoded:
                raise ValueError(f"candidate bundle is missing {selected}")

        temporary = Path(tempfile.mkdtemp(prefix="gateway-candidate-"))
        try:
            for relative, content in decoded:
                target = temporary.joinpath(*relative.split("/"))
                target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                target.write_bytes(content)
                os.chmod(target, 0o600)
            return self._artifacts.put_directory(temporary, ArtifactKind.KERNEL)
        finally:
            shutil.rmtree(temporary, ignore_errors=True)

    @staticmethod
    def _adapter_request(
        request: GatewayProxyRequestV2,
        operation: GatewayOperation,
        candidate_digest: ArtifactDigest | None,
        candidate_path: Path | None,
    ) -> GatewayAdapterRequest:
        profile_level = request.level if isinstance(request, ProfileRequestV2) else None
        kernel_regex = request.kernel_regex if isinstance(request, ProfileRequestV2) else None
        job_id = request.job_id if isinstance(request, (PollRequestV2, CancelRequestV2)) else None
        excluded = {
            "schema_version",
            "attempt_id",
            "operation",
            "idempotency_key",
            "candidate",
            "job_id",
            "level",
            "kernel_regex",
        }
        parameters = cast(
            dict[str, JsonValue],
            request.model_dump(mode="json", exclude=excluded),
        )
        if isinstance(request, DevRequestV2):
            parameters["files"] = cast(
                JsonValue,
                {file.path: _dev_file_text(file) for file in request.files},
            )
        return GatewayAdapterRequest(
            attempt_id=request.attempt_id,
            operation=operation,
            idempotency_key=request.idempotency_key,
            candidate_digest=candidate_digest,
            candidate_path=candidate_path,
            profile_level=profile_level,
            kernel_regex=kernel_regex,
            job_id=job_id,
            parameters=parameters,
        )


def _with_production_gate_advisory(
    payload: JsonValue,
    violations: tuple[str, ...],
) -> JsonValue:
    advisory = cast(
        JsonValue,
        {
            "would_reject_at_seal": True,
            "violations": list(violations),
            "note": "this exploratory job still ran; evaluate would refuse this candidate",
        },
    )
    if isinstance(payload, dict):
        return cast(JsonValue, {**payload, "production_gate": advisory})
    return cast(JsonValue, {"result": payload, "production_gate": advisory})


class GatewayProxyAsgiApp:
    """ASGI adapter separating remote Gateway work from Runtime-local queries."""

    def __init__(self, service: GatewayProxyService, limits: GatewayProxyLimits) -> None:
        self._service = service
        self._limits = limits

    async def __call__(
        self,
        scope: Mapping[str, object],
        receive: AsgiReceive,
        send: AsgiSend,
    ) -> None:
        if scope.get("type") != "http":
            raise ValueError("Gateway Proxy supports only ASGI HTTP scopes")
        path = scope.get("path")
        if scope.get("method") != "POST" or path not in {
            "/v1/operations",
            "/v1/runtime/queries",
            "/v1/runtime/journals",
        }:
            await json_response(send, 404, {"error": "not_found"})
            return
        token = bearer_token(scope.get("headers"))
        if token is None:
            await json_response(send, 401, {"error": "missing_bearer_token"})
            return
        body = b""
        try:
            body = await read_request_body(
                receive,
                self._limits.max_request_bytes,
                oversized_message="Gateway Proxy request exceeds byte limit",
            )
            result = await self._service.execute(
                token,
                body,
                operation_scope=(
                    "runtime"
                    if path == "/v1/runtime/queries"
                    else "journal"
                    if path == "/v1/runtime/journals"
                    else "gateway"
                ),
            )
        except ValidationError as error:
            await json_response(
                send,
                400,
                _invalid_request_response(
                    body,
                    error,
                    operation_scope=(
                        "runtime"
                        if path == "/v1/runtime/queries"
                        else "journal"
                        if path == "/v1/runtime/journals"
                        else "gateway"
                    ),
                ),
            )
        except ValueError as error:
            await json_response(
                send,
                400,
                _invalid_request_response(
                    body,
                    error,
                    operation_scope=(
                        "runtime"
                        if path == "/v1/runtime/queries"
                        else "journal"
                        if path == "/v1/runtime/journals"
                        else "gateway"
                    ),
                ),
            )
        except PermissionError as error:
            await json_response(send, 403, {"error": "forbidden", "detail": str(error)})
        except InvalidTransitionError as error:
            await json_response(send, 409, {"error": "conflict", "detail": str(error)})
        except UpstreamGatewayError as error:
            await json_response(
                send,
                503,
                {"error": "gateway_unavailable", "detail": str(error)},
            )
        except InfrastructureError:
            await json_response(send, 503, {"error": "gateway_unavailable"})
        else:
            await json_response(send, 200, result.model_dump(mode="json"))


def _invalid_request_response(
    payload: bytes,
    error: Exception,
    *,
    operation_scope: Literal["gateway", "runtime", "journal"] | None = None,
) -> dict[str, JsonValue]:
    """Attach the exact Agent-facing schema for the operation that failed validation."""
    response: dict[str, JsonValue] = {
        "error": "invalid_request",
        "detail": str(error),
    }
    if isinstance(error, DirectionConcurrencyError):
        active_ids = list(error.in_progress_direction_ids)
        response["issues"] = cast(
            JsonValue,
            [
                {
                    "path": "direction_id",
                    "code": "direction_concurrency_conflict",
                    "message": str(error),
                }
            ],
        )
        response["conflict"] = cast(
            JsonValue,
            {
                "requested_direction_id": error.requested_direction_id,
                "in_progress_direction_ids": active_ids,
            },
        )
        response["recovery"] = cast(
            JsonValue,
            [
                {
                    "tool": "list-directions",
                    "request": {"file": "scratch/directions-index.json"},
                },
                {
                    "instruction": (
                        "Continue exploration only under the existing in-progress Direction, or "
                        "close it with update-direction action complete, abandon, defer, or block"
                    )
                },
                {
                    "instruction": (
                        "The requested Direction was not started. Retry start only after no other "
                        "Direction is in progress"
                    )
                },
            ],
        )
    if isinstance(error, ValidationError):
        issues = _agent_validation_issues(error)
        # str(error) leads with every tagged-union variant and echoes the request prefix,
        # so the cause it ends with costs far more context than the compacted issues.
        response["detail"] = (
            "; ".join(f"{issue['path']}: {issue['message']}" for issue in issues)
            if issues
            else "the request does not match the schema for this operation"
        )
        if issues:
            response["issues"] = cast(JsonValue, issues)
    operation: object = None
    try:
        value = json.loads(payload)
        if isinstance(value, dict):
            operation = value.get("operation")
    except (json.JSONDecodeError, UnicodeDecodeError):
        pass
    allowed_operations = _operations_for_scope(operation_scope)
    if isinstance(operation, str):
        try:
            response["request_schema"] = gateway_agent_request_schema(
                operation,
                allowed_operations=allowed_operations,
            )
        except (PermissionError, ValueError):
            response["supported_operations"] = cast(
                JsonValue,
                _supported_gateway_operations(operation_scope),
            )
    else:
        response["supported_operations"] = cast(
            JsonValue,
            _supported_gateway_operations(operation_scope),
        )
    return response


def _agent_validation_issues(error: ValidationError) -> list[dict[str, str]]:
    """Return compact repair hints without echoing request values or Runtime-owned fields."""
    issues: list[dict[str, str]] = []
    for issue in error.errors(include_url=False, include_context=False, include_input=False):
        location = [str(part) for part in issue.get("loc", ())]
        if location and location[0] in _GATEWAY_OPERATION_NAMES:
            location = location[1:]
        if location and location[0] in _RUNTIME_OWNED_ERROR_FIELDS:
            continue
        message = issue.get("msg")
        code = issue.get("type")
        if not isinstance(message, str) or not isinstance(code, str):
            continue
        issues.append(
            {
                "path": ".".join(location) if location else "$",
                "code": code,
                "message": message,
            }
        )
    return issues


def _operations_for_scope(
    operation_scope: Literal["gateway", "runtime", "journal"] | None,
) -> frozenset[str] | None:
    if operation_scope is None:
        return None
    runtime_operations = frozenset(operation.value for operation in _RUNTIME_LOCAL_OPERATIONS)
    if operation_scope == "runtime":
        return runtime_operations
    if operation_scope == "journal":
        return frozenset(operation.value for operation in _RUNTIME_JOURNAL_OPERATIONS)
    return frozenset(operation.value for operation in _AGENT_GATEWAY_OPERATIONS)


def _supported_gateway_operations(
    operation_scope: Literal["gateway", "runtime", "journal"] | None = None,
) -> list[str]:
    operations = gateway_agent_request_schema(
        allowed_operations=_operations_for_scope(operation_scope)
    ).get("operations")
    if not isinstance(operations, dict):
        raise TypeError("generated Gateway schema has no operation mapping")
    return sorted(operations)
