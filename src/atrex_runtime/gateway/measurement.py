"""Trusted repeated-Evaluate measurements over the published Agate SDK."""

from __future__ import annotations

import math
import statistics
from collections.abc import Callable
from datetime import UTC, datetime
from typing import cast
from uuid import uuid4

from ..artifacts.local import ArtifactKind, JsonValue, LocalArtifactStore
from ..domain.errors import InfrastructureError
from ..domain.ids import ArtifactDigest
from ..domain.models import KernelMeasurement, KernelMeasurementPurpose, KernelRevision
from ..ports import KernelMeasurementJournal, KernelMeasurementRun, KernelMeasurementRunner
from .agate import AgateClient, AgateRequestBuilder, parse_agate_evaluation
from .batched_evaluate import (
    ShapeBatch,
    ShapeBatchedEvaluateExecutor,
    ShapeBatchOutcome,
)
from .candidate import resolve_kernel_candidate
from .contract import RegistryKernelEvaluationContextResolver
from .execution import build_evaluation_request, call_agate_json
from .protocol import EvaluationV2

_TERMINAL = frozenset({"succeeded", "failed", "cancelled"})


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _failed_evaluation() -> EvaluationV2:
    return EvaluationV2(correct=False, latency_us=None)


class AgateKernelMeasurementRunner(KernelMeasurementRunner):
    """Issue one fresh, single-job Agate evaluation for each requested repetition."""

    def __init__(
        self,
        client: AgateClient,
        request_builder: AgateRequestBuilder,
        contexts: RegistryKernelEvaluationContextResolver,
        artifacts: LocalArtifactStore,
        journal: KernelMeasurementJournal,
        *,
        wait_timeout_s: float,
        clock: Callable[[], str] = _utc_now,
    ) -> None:
        if wait_timeout_s <= 0:
            raise ValueError("Agate measurement wait timeout must be positive")
        self._client = client
        self._request_builder = request_builder
        self._contexts = contexts
        self._artifacts = artifacts
        self._journal = journal
        self._wait_timeout_s = wait_timeout_s
        self._clock = clock
        self._shape_batches = ShapeBatchedEvaluateExecutor()

    async def run(
        self,
        revision: KernelRevision,
        repeat: int,
        purpose: KernelMeasurementPurpose,
    ) -> KernelMeasurementRun:
        """Evaluate a sealed Kernel once without exposing Agate credentials to Core."""
        if repeat < 0:
            raise ValueError("Kernel measurement repeat cannot be negative")
        context = self._contexts.resolve(revision)
        candidate = resolve_kernel_candidate(
            self._artifacts,
            revision.artifact_digest,
            context.contract.candidate_path,
            error_type=InfrastructureError,
            kind_error="Kernel measurement Artifact has the wrong kind",
            missing_error="Kernel measurement candidate file is missing",
        ).source
        try:
            candidate_source = candidate.read_text(encoding="utf-8")
        except UnicodeDecodeError as error:
            raise InfrastructureError("Kernel measurement candidate is not UTF-8") from error

        measurement_id = uuid4().hex
        idempotency_key = (
            f"runtime-comparison:{purpose.value}:{revision.id}:{repeat}:{measurement_id}"
        )

        async def evaluate_batch(batch: ShapeBatch) -> ShapeBatchOutcome:
            payload = build_evaluation_request(
                self._request_builder,
                candidate_source=candidate_source,
                operator=context.operator,
                contract=batch.contract,
                hardware_target=context.hardware_target,
                dsl=context.dsl,
                name=f"{context.operator}_comparison_{measurement_id}_batch_{batch.index}",
                idempotency_key=batch.idempotency_key,
            )
            try:
                accepted = await self._call(lambda: self._client.submit_job("eval", payload))
            except InfrastructureError as error:
                source = error.__cause__
                fields = vars(source) if source is not None else {}
                if fields.get("status") in {400, 422}:
                    return ShapeBatchOutcome(
                        {"status": "rejected", "error": "candidate request rejected"},
                        evaluation=_failed_evaluation(),
                    )
                raise
            job_id = accepted.get("job_id")
            if not isinstance(job_id, str) or not job_id:
                raise InfrastructureError("Agate measurement acceptance has no job_id")
            self._journal.record_runtime_event(
                "comparison.measurement_submitted",
                revision.id,
                {
                    "kernel_revision_id": revision.id,
                    "repeat": repeat,
                    "measurement_id": measurement_id,
                    "agate_job_id": job_id,
                    "purpose": purpose.value,
                    "shape_batch": batch.index,
                    "shape_ids": list(batch.shape_ids),
                },
            )
            job = await self._call(
                lambda: self._client.get_job(
                    job_id,
                    wait=True,
                    timeout=self._wait_timeout_s,
                )
            )
            status = job.get("status")
            if status not in _TERMINAL:
                raise InfrastructureError("Agate measurement did not reach a terminal state")
            evaluation = (
                parse_agate_evaluation(job, batch.shape_ids)
                if status == "succeeded"
                else _failed_evaluation()
            )
            return ShapeBatchOutcome(job, evaluation, job_id)

        batched = await self._shape_batches.run(
            context.contract,
            idempotency_key,
            evaluate_batch,
        )
        evaluation = batched.evaluation
        job_id = batched.job_id
        result_digest = self._artifacts.put_json(batched.job, ArtifactKind.GATEWAY_RESULT)
        event_base = {
            "kernel_revision_id": revision.id,
            "repeat": repeat,
            "measurement_id": measurement_id,
            "agate_job_id": job_id,
            "purpose": purpose.value,
        }
        self._record(
            measurement_id,
            revision,
            purpose,
            repeat,
            correct=evaluation.correct,
            latency_us=evaluation.latency_us,
            gateway_result_digest=result_digest,
            agate_job_id=job_id,
        )
        self._journal.record_runtime_event(
            "comparison.measurement_completed",
            revision.id,
            {
                **event_base,
                "gateway_result_digest": result_digest,
                "correct": evaluation.correct,
                "latency_us": evaluation.latency_us,
            },
        )
        return KernelMeasurementRun(
            repeat,
            evaluation.correct,
            evaluation.latency_us,
            gateway_result_digest=result_digest,
            agate_job_id=job_id,
        )

    def aggregate(
        self,
        revision: KernelRevision,
        runs: tuple[KernelMeasurementRun, ...],
        purpose: KernelMeasurementPurpose,
    ) -> ArtifactDigest:
        """Seal the complete ordinary-Evaluate measurement set and arithmetic mean."""
        if not runs or tuple(run.repeat for run in runs) != tuple(range(len(runs))):
            raise ValueError("ordinary Evaluate aggregate requires contiguous repetitions")
        latencies = [run.latency_us for run in runs]
        correct = all(run.correct for run in runs) and all(
            value is not None and value > 0 and math.isfinite(value) for value in latencies
        )
        mean = statistics.fmean(cast(list[float], latencies)) if correct else None
        value = cast(
            JsonValue,
            {
                "schema_version": 1,
                "operation": "evaluate_comparison",
                "aggregation": "arithmetic_mean",
                "purpose": purpose.value,
                "kernel_revision_id": revision.id,
                "repeats": len(runs),
                "correct": correct,
                "latency_us": mean,
                "measurements": [
                    {
                        "repeat": run.repeat,
                        "correct": run.correct,
                        "latency_us": run.latency_us,
                        "gateway_result_digest": run.gateway_result_digest,
                        "agate_job_id": run.agate_job_id,
                    }
                    for run in runs
                ],
            },
        )
        digest = self._artifacts.put_json(value, ArtifactKind.GATEWAY_RESULT)
        self._journal.record_runtime_event(
            "comparison.evaluate_aggregate_completed",
            revision.id,
            {
                "purpose": purpose.value,
                "repeats": len(runs),
                "correct": correct,
                "latency_us": mean,
                "gateway_result_digest": digest,
            },
        )
        return digest

    def _record(
        self,
        measurement_id: str,
        revision: KernelRevision,
        purpose: KernelMeasurementPurpose,
        repeat: int,
        *,
        correct: bool,
        latency_us: float | None,
        gateway_result_digest: ArtifactDigest | None,
        agate_job_id: str | None,
    ) -> None:
        self._journal.record_kernel_measurement(
            KernelMeasurement(
                id=measurement_id,
                kernel_revision_id=revision.id,
                purpose=purpose,
                repeat=repeat,
                correct=correct,
                latency_us=latency_us,
                gateway_result_digest=gateway_result_digest,
                agate_job_id=agate_job_id,
                created_at=self._clock(),
            )
        )

    async def _call(self, operation: Callable[[], object]) -> dict[str, JsonValue]:
        return await call_agate_json(
            operation,
            request_error="Agate measurement request failed",
            invalid_response="Agate measurement returned invalid JSON",
            non_object_response="Agate measurement returned a non-object",
        )
