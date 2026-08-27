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
from .contract import AgateEvaluationContextResolver
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
    GatewayResultReadRequestV2,
    KernelArtifactReadRequestV2,
    KernelTrialShowRequestV2,
    PollRequestV2,
    ProfileRequestV2,
    gateway_agent_request_schema,
)

_REQUEST_ADAPTER: TypeAdapter[GatewayProxyRequestV2] = TypeAdapter(GatewayProxyRequestV2)
_CANDIDATE_REQUEST_TYPES = (
    EvaluateRequestV2,
    ProfileRequestV2,
    DevRequestV2,
    CheckRequestV2,
    DisassembleRequestV2,
)
_RUNTIME_LOCAL_OPERATIONS = frozenset(
    {
        GatewayOperation.KERNEL_TRIAL_SHOW,
        GatewayOperation.KERNEL_ARTIFACT_READ,
        GatewayOperation.GATEWAY_RESULT_READ,
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
            "kernel_artifact_digest": candidate_digest,
        }
        self._events.record_runtime_event(
            "gateway.operation_submitted",
            request.attempt_id,
            event_base,
        )
        try:
            if isinstance(request, KernelTrialShowRequestV2):
                result = self._show_kernel_trial(request)
            elif isinstance(request, KernelArtifactReadRequestV2):
                result = self._read_kernel_artifact(request)
            elif isinstance(request, GatewayResultReadRequestV2):
                result = self._read_gateway_result(request)
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
            result_digest = self._store_gateway_result(result)
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

        response = GatewayProxyResponseV2(
            schema_version=GATEWAY_PROXY_PROTOCOL_VERSION,
            operation=request.operation,
            status=result.status,
            kernel_artifact_digest=(None if candidate_digest is None else str(candidate_digest)),
            kernel_trial_id=(
                None
                if candidate_digest is None
                else gateway_kernel_trial_id(
                    request.attempt_id,
                    authorization.recovery_generation,
                    candidate_digest,
                )
            ),
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
                    "gateway_results": self._trial_gateway_results(trial),
                },
            ),
        )

    def _trial_gateway_results(self, trial: GatewayKernelTrialRecord) -> list[JsonValue]:
        values: list[JsonValue] = []
        seen: set[ArtifactDigest] = set()
        for observation in trial.observations:
            digest = observation.gateway_result_digest
            if digest is None or digest in seen:
                continue
            if observation.result_artifact_digest is None:
                raise InfrastructureError("Gateway Result has no Agent-visible response Artifact")
            seen.add(digest)
            values.append(self._gateway_result_payload(observation.result_artifact_digest))
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

    def _read_gateway_result(
        self,
        request: GatewayResultReadRequestV2,
    ) -> GatewayAdapterResult:
        _, visible_attempt_ids = self._control.visible_kernel_trial_attempt_ids(request.attempt_id)
        digest = parse_artifact_digest(request.gateway_result_digest)
        matching = tuple(
            (trial, observation)
            for trial in self._control.list_kernel_trials(visible_attempt_ids, limit=5_000)
            for observation in trial.observations
            if observation.gateway_result_digest == digest
        )
        if not matching:
            raise ValueError("Gateway Result is outside the visible Lineage history")
        response_artifact_digests = {
            observation.result_artifact_digest
            for _, observation in matching
            if observation.result_artifact_digest is not None
        }
        if len(response_artifact_digests) != 1:
            raise InfrastructureError("Gateway Result has no unique Agent-visible response")
        response = self._gateway_result_payload(next(iter(response_artifact_digests)))
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

    def _gateway_result_payload(
        self,
        response_artifact_digest: ArtifactDigest,
    ) -> dict[str, JsonValue]:
        """Return a minimal view of the recorded Agent-safe response."""
        artifact = self._artifacts.verify(response_artifact_digest)
        if artifact.kind is not ArtifactKind.GATEWAY_RESULT:
            raise InfrastructureError("Gateway response Artifact has an invalid kind")
        documents: dict[str, JsonValue] = {}
        for path in sorted(
            candidate for candidate in artifact.payload_path.rglob("*") if candidate.is_file()
        ):
            relative = path.relative_to(artifact.payload_path).as_posix()
            try:
                documents[relative] = cast(JsonValue, json.loads(path.read_bytes()))
            except (json.JSONDecodeError, UnicodeDecodeError) as error:
                raise InfrastructureError(
                    "Gateway Result Artifact contains invalid JSON"
                ) from error
        value = documents.get("value.json")
        if not isinstance(value, dict):
            raise InfrastructureError("Gateway Result Artifact has no value.json")
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
        selected = self._contract_candidate_path(attempt_id)
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

    def _contract_candidate_path(self, attempt_id: AttemptId) -> str | None:
        """Return the contract-declared candidate file, or None when unresolvable."""
        if self._contexts is None:
            return None
        try:
            return self._contexts.resolve(attempt_id).contract.candidate_path
        except (KeyError, LookupError, ValueError):
            return None

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
    return frozenset(
        operation.value
        for operation in GatewayOperation
        if operation not in _RUNTIME_LOCAL_OPERATIONS
        and operation not in _RUNTIME_JOURNAL_OPERATIONS
    )


def _supported_gateway_operations(
    operation_scope: Literal["gateway", "runtime", "journal"] | None = None,
) -> list[str]:
    operations = gateway_agent_request_schema(
        allowed_operations=_operations_for_scope(operation_scope)
    ).get("operations")
    if not isinstance(operations, dict):
        raise TypeError("generated Gateway schema has no operation mapping")
    return sorted(operations)
