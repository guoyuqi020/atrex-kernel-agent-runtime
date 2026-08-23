"""Trusted Gateway Proxy service and minimal ASGI transport."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import shutil
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
    InfrastructureError,
    InvalidTransitionError,
    UpstreamGatewayError,
)
from ..domain.ids import ArtifactDigest, AttemptId, parse_artifact_digest
from ..ports import RuntimeEventRecorder
from ..serialization import canonical_json_digest
from .control import SqliteGatewayControl
from .control_models import (
    GatewayCapability,
    GatewayEvaluationSource,
    GatewayMeasurementRecord,
    GatewayOperation,
)
from .diff_policy import RegistryCandidateDiffValidator
from .measurement_history import normalized_measurement_points
from .production_policy import CandidateProductionValidator
from .protocol import (
    GATEWAY_PROXY_PROTOCOL_VERSION,
    CancelRequestV2,
    CandidateBundleV2,
    CheckRequestV2,
    DevRequestV2,
    DisassembleRequestV2,
    EvaluateRequestV2,
    EvaluationV2,
    GatewayProxyRequestV2,
    GatewayProxyResponseV2,
    KernelTrialReadRequestV2,
    KernelTrialsRequestV2,
    MeasurementsRequestV2,
    PollRequestV2,
    ProfileRequestV2,
    SolRequestV2,
    SubmitRequestV2,
    gateway_agent_request_schema,
)

_REQUEST_ADAPTER: TypeAdapter[GatewayProxyRequestV2] = TypeAdapter(GatewayProxyRequestV2)
_CANDIDATE_REQUEST_TYPES = (
    EvaluateRequestV2,
    SubmitRequestV2,
    ProfileRequestV2,
    DevRequestV2,
    CheckRequestV2,
    SolRequestV2,
    DisassembleRequestV2,
)


def _utc_now() -> datetime:
    return datetime.now(UTC)


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
        clock: Callable[[], datetime] = _utc_now,
    ) -> None:
        self._control = control
        self._artifacts = artifacts
        self._adapter = adapter
        self._limits = limits
        self._events = events
        self._candidate_diff = candidate_diff
        self._candidate_production = candidate_production
        self._clock = clock

    async def execute(self, token: str, payload: bytes) -> GatewayProxyResponseV2:
        """Parse and execute one complete protocol request."""
        if len(payload) > self._limits.max_request_bytes:
            raise ValueError("Gateway Proxy request exceeds byte limit")
        request = _REQUEST_ADAPTER.validate_json(payload)
        operation = GatewayOperation(request.operation)
        request_digest = canonical_json_digest(request.model_dump(mode="json"))
        authorization = self._control.authorize(
            GatewayCapability(token, request.attempt_id),
            operation,
            idempotency_key=request.idempotency_key,
            request_digest=str(request_digest),
        )
        existing_response = self._control.get_operation_artifact(
            request.attempt_id,
            request.idempotency_key,
            operation,
        )
        if existing_response is not None:
            return self._load_response(existing_response)

        candidate_digest: ArtifactDigest | None = None
        candidate_path: Path | None = None
        if isinstance(request, _CANDIDATE_REQUEST_TYPES):
            candidate_digest = self._seal_candidate(request.candidate)
            self._control.bind_operation_candidate(
                request.attempt_id,
                request.idempotency_key,
                operation,
                candidate_digest,
                recovery_generation=authorization.recovery_generation,
            )
            if self._candidate_diff is not None and isinstance(request, EvaluateRequestV2):
                self._candidate_diff.validate(request.attempt_id, candidate_digest)
            if self._candidate_production is not None and isinstance(request, EvaluateRequestV2):
                self._candidate_production.validate(request.attempt_id, candidate_digest)
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
            "candidate_artifact_digest": candidate_digest,
        }
        self._events.record_runtime_event(
            "gateway.operation_submitted",
            request.attempt_id,
            event_base,
        )
        try:
            if isinstance(request, MeasurementsRequestV2):
                result = self._query_measurements(request)
            elif isinstance(request, KernelTrialsRequestV2):
                result = self._query_kernel_trials(request)
            elif isinstance(request, KernelTrialReadRequestV2):
                result = self._read_kernel_trial(request)
            else:
                result = await self._adapter.execute(adapter_request)
            result_digest = self._store_gateway_result(result)

            if isinstance(request, EvaluateRequestV2):
                if result.status != "completed" or result.evaluation is None:
                    raise InfrastructureError("evaluate did not return a completed evaluation")
                if candidate_digest is None:
                    raise AssertionError("evaluate candidate was not sealed")
                evaluation_record = self._control.record_evaluation(
                    request.attempt_id,
                    source=GatewayEvaluationSource.AGENT,
                    idempotency_key=request.idempotency_key,
                    candidate_artifact_digest=candidate_digest,
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
                    candidate_artifact_digest=candidate_digest,
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

        response = GatewayProxyResponseV2(
            schema_version=GATEWAY_PROXY_PROTOCOL_VERSION,
            operation=request.operation,
            status=result.status,
            candidate_artifact_digest=(None if candidate_digest is None else str(candidate_digest)),
            gateway_result_digest=str(result_digest),
            job_id=result.job_id,
            evaluation=result.evaluation,
            result=result.result if result.worker_result is None else result.worker_result,
        )
        response_artifact = self._artifacts.put_json(
            cast(JsonValue, response.model_dump(mode="json")),
            ArtifactKind.GATEWAY_RESULT,
        )
        self._control.commit_operation_artifact(
            request.attempt_id,
            request.idempotency_key,
            operation,
            response_artifact,
        )
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

    def _query_measurements(self, request: MeasurementsRequestV2) -> GatewayAdapterResult:
        lineage_id, visible_attempt_ids = self._control.visible_measurement_attempt_ids(
            request.attempt_id
        )
        catalog = self._control.measurement_kernel_catalog(lineage_id)
        requested_digest = (
            None
            if request.kernel_artifact_digest is None
            else parse_artifact_digest(request.kernel_artifact_digest)
        )
        if request.kernel_revision_id is not None:
            revision_matches = [
                digest
                for digest, identity in catalog.items()
                if identity[0] == request.kernel_revision_id
            ]
            if not revision_matches:
                raise ValueError("kernel_revision_id is outside the visible Lineage")
            revision_digest = revision_matches[0]
            if requested_digest is not None and requested_digest != revision_digest:
                raise ValueError("Kernel revision and Artifact filters disagree")
            requested_digest = revision_digest
        records = self._control.list_measurements(
            visible_attempt_ids,
            kind=None if request.kind is None else GatewayOperation(request.kind),
            candidate_artifact_digest=requested_digest,
            shape_id=request.shape_id,
            kernel_name=request.kernel_name,
            metric=request.metric,
            limit=request.limit,
        )
        values: list[JsonValue] = []
        for record in records:
            identity = catalog.get(record.candidate_artifact_digest)
            values.append(
                {
                    "measurement_id": record.id,
                    "attempt_id": record.attempt_id,
                    "kernel_revision_id": None if identity is None else identity[0],
                    "kernel_revision_number": None if identity is None else identity[1],
                    "kernel_version": None if identity is None else identity[2],
                    "kernel_artifact_digest": record.candidate_artifact_digest,
                    "operation": record.point.kind.value,
                    "source_operation": record.source_operation.value,
                    "profile_level": record.point.profile_level,
                    "shape_id": record.point.shape_id,
                    "kernel_name": record.point.kernel_name,
                    "metrics": record.point.metrics,
                    "gateway_result_digest": record.gateway_result_digest,
                    "created_at": record.created_at,
                }
            )
        return GatewayAdapterResult(
            "completed",
            cast(
                JsonValue,
                {
                    "lineage_id": lineage_id,
                    "through_attempt_id": request.attempt_id,
                    "visible_attempt_count": len(visible_attempt_ids),
                    "measurements": values,
                },
            ),
        )

    def _query_kernel_trials(self, request: KernelTrialsRequestV2) -> GatewayAdapterResult:
        lineage_id, visible_attempt_ids = self._control.visible_kernel_trial_attempt_ids(
            request.attempt_id
        )
        trials = self._control.list_kernel_trials(visible_attempt_ids, limit=5_000)
        if request.decision is not None:
            trials = tuple(trial for trial in trials if trial.disposition == request.decision)
        trials = tuple(reversed(trials[-request.limit :]))
        values: list[JsonValue] = []
        for trial in trials:
            values.append(
                {
                    "kernel_trial_id": trial.id,
                    "attempt_id": trial.attempt_id,
                    "recovery_generation": trial.recovery_generation,
                    "ordinal": trial.ordinal,
                    "candidate_artifact_digest": trial.candidate_artifact_digest,
                    "disposition": trial.disposition,
                    "created_at": trial.created_at,
                    "observations": [
                        {
                            "operation": observation.operation.value,
                            "idempotency_key": observation.idempotency_key,
                            "gateway_result_digest": observation.result_artifact_digest,
                            "created_at": observation.created_at,
                        }
                        for observation in trial.observations
                    ],
                    "annotations": [
                        {
                            "sequence": annotation.sequence,
                            "decision": annotation.decision,
                            "experiment": annotation.experiment,
                            "recorded_at": annotation.recorded_at,
                        }
                        for annotation in trial.annotations
                    ],
                }
            )
        return GatewayAdapterResult(
            "completed",
            cast(
                JsonValue,
                {
                    "lineage_id": lineage_id,
                    "through_attempt_id": request.attempt_id,
                    "visible_attempt_count": len(visible_attempt_ids),
                    "kernel_trials": values,
                },
            ),
        )

    def _read_kernel_trial(self, request: KernelTrialReadRequestV2) -> GatewayAdapterResult:
        lineage_id, visible_attempt_ids = self._control.visible_kernel_trial_attempt_ids(
            request.attempt_id
        )
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
        artifact = self._artifacts.verify(trial.candidate_artifact_digest)
        if artifact.kind is not ArtifactKind.KERNEL:
            raise ValueError("Kernel Trial source Artifact has an invalid kind")
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
            "lineage_id": lineage_id,
            "kernel_trial_id": trial.id,
            "attempt_id": trial.attempt_id,
            "candidate_artifact_digest": trial.candidate_artifact_digest,
            "disposition": trial.disposition,
            "files": files,
        }
        if request.file is not None:
            source = artifact.payload_path.joinpath(*request.file.split("/"))
            if not source.is_file():
                raise ValueError(f"Kernel Trial file does not exist: {request.file}")
            content = source.read_bytes()
            try:
                response["content"] = content.decode("utf-8")
                response["encoding"] = "utf-8"
            except UnicodeDecodeError:
                response["content"] = base64.b64encode(content).decode("ascii")
                response["encoding"] = "base64"
            response["file"] = request.file
        return GatewayAdapterResult("completed", cast(JsonValue, response))

    def _store_gateway_result(self, result: GatewayAdapterResult) -> ArtifactDigest:
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

    def _load_response(self, digest: ArtifactDigest) -> GatewayProxyResponseV2:
        stored = self._artifacts.verify(digest)
        if stored.kind is not ArtifactKind.GATEWAY_RESULT:
            raise InfrastructureError("cached Gateway operation has the wrong Artifact kind")
        value = stored.payload_path / "value.json"
        if not value.is_file():
            raise InfrastructureError("cached Gateway operation has no JSON payload")
        try:
            return GatewayProxyResponseV2.model_validate_json(value.read_bytes())
        except (ValueError, OSError) as error:
            raise InfrastructureError("cached Gateway operation response is invalid") from error

    def _seal_candidate(self, candidate: CandidateBundleV2) -> ArtifactDigest:
        if len(candidate.files) > self._limits.max_candidate_files:
            raise ValueError("candidate exceeds file-count limit")
        decoded = [(file.path, file.content()) for file in candidate.files]
        if sum(len(content) for _path, content in decoded) > self._limits.max_candidate_bytes:
            raise ValueError("candidate exceeds decoded byte limit")

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


class GatewayProxyAsgiApp:
    """Small ASGI adapter for the single versioned Gateway Proxy endpoint."""

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
        if scope.get("method") != "POST" or scope.get("path") != "/v1/operations":
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
            result = await self._service.execute(token, body)
        except ValidationError as error:
            await json_response(send, 400, _invalid_request_response(body, error))
        except ValueError as error:
            await json_response(send, 400, _invalid_request_response(body, error))
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


def _invalid_request_response(payload: bytes, error: Exception) -> dict[str, JsonValue]:
    """Attach the exact Agent-facing schema for the operation that failed validation."""
    response: dict[str, JsonValue] = {
        "error": "invalid_request",
        "detail": str(error),
    }
    operation: object = None
    try:
        value = json.loads(payload)
        if isinstance(value, dict):
            operation = value.get("operation")
    except (json.JSONDecodeError, UnicodeDecodeError):
        pass
    if isinstance(operation, str):
        try:
            response["request_schema"] = gateway_agent_request_schema(operation)
        except ValueError:
            response["supported_operations"] = cast(JsonValue, _supported_gateway_operations())
    else:
        response["supported_operations"] = cast(JsonValue, _supported_gateway_operations())
    return response


def _supported_gateway_operations() -> list[str]:
    operations = gateway_agent_request_schema().get("operations")
    if not isinstance(operations, dict):
        raise TypeError("generated Gateway schema has no operation mapping")
    return sorted(operations)
