"""Independent Runtime evaluation of an Agent-nominated candidate Kernel."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import partial

import anyio

from ..artifacts.local import JsonValue, LocalArtifactStore
from ..domain.errors import InfrastructureError, InvalidTransitionError
from ..domain.ids import ArtifactDigest, AttemptId
from ..domain.models import Dsl
from ..ports import AttemptCandidateResult, RuntimeEventRecorder
from .agate import (
    AgateCandidateRejection,
    AgateClient,
    AgateRequestBuilder,
    parse_agate_evaluation,
)
from .batched_evaluate import ShapeBatch, ShapeBatchedEvaluateExecutor, ShapeBatchOutcome
from .candidate import resolve_kernel_candidate
from .contract import AgateEvaluationContextResolver, AgateEvaluationContractV1
from .control import SqliteGatewayControl
from .control_models import GatewayEvaluationSource
from .execution import (
    build_evaluation_request,
    call_agate_json,
    store_gateway_result,
    submit_agate_job,
)
from .production_policy import ProductionKernelPolicy
from .protocol import EvaluationV2
from .repeated_evaluate import aggregate_evaluations, repeated_evaluate_result

_TERMINAL = frozenset({"succeeded", "failed", "cancelled"})


def _utc_now() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True, slots=True)
class BootstrapEvaluationStage:
    """One ordered authoritative Bootstrap correctness stage."""

    correctness_cases: int
    evaluate_repeats: int = 1

    def __post_init__(self) -> None:
        if self.correctness_cases <= 0 or self.evaluate_repeats <= 0:
            raise ValueError("Bootstrap Gate stage values must be positive")


class AgateAuthoritativeCandidateEvaluator:
    """Re-evaluate one exact Agent nomination and commit only that independent result."""

    def __init__(
        self,
        client: AgateClient,
        request_builder: AgateRequestBuilder,
        contexts: AgateEvaluationContextResolver,
        artifacts: LocalArtifactStore,
        control: SqliteGatewayControl,
        events: RuntimeEventRecorder,
        *,
        wait_timeout_s: float,
        bootstrap_stages: tuple[BootstrapEvaluationStage, ...] = (
            BootstrapEvaluationStage(1),
            BootstrapEvaluationStage(5),
        ),
        bootstrap_bench_iters: int = 100,
        profile_without_roofline: bool = False,
        production_policy: ProductionKernelPolicy | None = None,
        clock: Callable[[], datetime] = _utc_now,
    ) -> None:
        if wait_timeout_s <= 0:
            raise ValueError("authoritative evaluation wait timeout must be positive")
        if not bootstrap_stages:
            raise ValueError("Bootstrap Gate requires at least one stage")
        if bootstrap_bench_iters <= 0:
            raise ValueError("Bootstrap bench iterations must be positive")
        self._client = client
        self._request_builder = request_builder
        self._contexts = contexts
        self._artifacts = artifacts
        self._control = control
        self._events = events
        self._wait_timeout_s = wait_timeout_s
        self._bootstrap_stages = bootstrap_stages
        self._bootstrap_bench_iters = bootstrap_bench_iters
        self._profile_without_roofline = profile_without_roofline
        self._production_policy = production_policy
        self._clock = clock
        self._shape_batches = ShapeBatchedEvaluateExecutor()

    async def finalize(
        self,
        attempt_id: AttemptId,
        kernel_artifact_digest: ArtifactDigest,
        *,
        nominated_gateway_result_digest: ArtifactDigest | None = None,
        nominated_recovery_generation: int | None = None,
        independent_evaluate: bool = True,
    ) -> AttemptCandidateResult:
        """Validate the nomination and optionally run an independent final Evaluate."""
        candidate_digest = ArtifactDigest(str(kernel_artifact_digest))
        agent_evaluation = self._control.find_agent_evaluation(
            attempt_id,
            candidate_digest,
            gateway_result_digest=nominated_gateway_result_digest,
            recovery_generation=nominated_recovery_generation,
        )
        if agent_evaluation is None:
            raise ValueError("nominated candidate has no matching Agent evaluation")
        if not agent_evaluation.correct:
            raise ValueError("nominated candidate's Agent evaluation is not correct")

        existing_outcome = self._control.get_committed_outcome(attempt_id)
        if existing_outcome is not None:
            if existing_outcome.artifact_digest != candidate_digest:
                raise InvalidTransitionError(
                    f"Attempt {attempt_id} already finalized a different candidate"
                )
            return existing_outcome

        generation = self._control.current_generation(attempt_id)
        # This identity deliberately excludes the recovery generation. A Runtime
        # restart must resume the same remote Agate job rather than submit another
        # authoritative measurement for an unchanged Attempt and candidate.
        idempotency_key = f"runtime-final:{attempt_id}:{candidate_digest}"
        recovered = self._control.find_runtime_final_evaluation(attempt_id, idempotency_key)
        if recovered is not None:
            return self._control.commit_authoritative_outcome(
                attempt_id,
                recovered.id,
                committed_at=self._clock(),
            )

        context = self._contexts.resolve(attempt_id)
        resolved = resolve_kernel_candidate(
            self._artifacts,
            candidate_digest,
            context.contract.candidate_path,
            error_type=ValueError,
            kind_error="nominated candidate Artifact is not a Kernel",
            missing_error="nominated candidate does not contain the contract candidate path",
        )
        candidate = resolved.source
        if context.contract.production_gate and self._production_policy is not None:
            self._production_policy.validate(
                resolved.root,
                context.contract.candidate_path,
                context.dsl,
            )
        if not independent_evaluate:
            self._events.record_runtime_event(
                "gateway.agent_evaluation_adopted_for_abba",
                attempt_id,
                {
                    "kernel_artifact_digest": candidate_digest,
                    "agent_evaluation_id": agent_evaluation.id,
                    "gateway_result_digest": agent_evaluation.gateway_result_digest,
                    "latency_us": agent_evaluation.latency_us,
                },
            )
            return AttemptCandidateResult(
                artifact_digest=candidate_digest,
                gateway_result_digest=agent_evaluation.gateway_result_digest,
                correct=agent_evaluation.correct,
                latency_us=agent_evaluation.latency_us,
            )
        try:
            candidate_source = candidate.read_text(encoding="utf-8")
        except UnicodeDecodeError as error:
            raise ValueError("nominated candidate source must be UTF-8") from error

        event_base = {
            "source": GatewayEvaluationSource.RUNTIME_FINAL.value,
            "kernel_artifact_digest": candidate_digest,
            "agent_evaluation_id": agent_evaluation.id,
            "recovery_generation": generation,
        }
        stage_results: list[JsonValue] = []
        evaluation = EvaluationV2(correct=False, latency_us=None)
        final_payload: dict[str, object] | None = None
        for stage_index, stage in enumerate(self._bootstrap_stages):
            stage_contract = context.contract.model_copy(
                update={
                    "options": context.contract.options.model_copy(
                        update={
                            "num_correctness_cases": stage.correctness_cases,
                            "bench_iters": self._bootstrap_bench_iters,
                        }
                    )
                }
            )
            stage_key = f"{idempotency_key}:stage:{stage_index}"
            payload = build_evaluation_request(
                self._request_builder,
                candidate_source=candidate_source,
                operator=context.operator,
                contract=stage_contract,
                hardware_target=context.agate_gpu,
                dsl=context.dsl,
                name=f"{context.operator}_{attempt_id}_bootstrap_stage_{stage_index}",
                idempotency_key=stage_key,
            )
            final_payload = payload
            stage_event = {
                **event_base,
                "stage": stage_index,
                "correctness_cases": stage.correctness_cases,
                "stage_repeats": stage.evaluate_repeats,
            }
            if stage.evaluate_repeats == 1:
                stage_job, _, evaluation = await self._evaluate_batched(
                    attempt_id,
                    candidate_source=candidate_source,
                    operator=context.operator,
                    contract=stage_contract,
                    hardware_target=context.agate_gpu,
                    dsl=context.dsl,
                    name=f"{context.operator}_{attempt_id}_bootstrap_stage_{stage_index}",
                    idempotency_key=stage_key,
                    event_base=stage_event,
                )
            else:
                repeated = await self._evaluate_repeated(
                    attempt_id,
                    candidate_source=candidate_source,
                    operator=context.operator,
                    contract=stage_contract,
                    hardware_target=context.agate_gpu,
                    dsl=context.dsl,
                    name=f"{context.operator}_{attempt_id}_bootstrap_stage_{stage_index}",
                    idempotency_key=stage_key,
                    event_base=stage_event,
                    repeats=stage.evaluate_repeats,
                )
                jobs = tuple(item[0] for item in repeated)
                evaluations = tuple(item[2] for item in repeated)
                stage_job = repeated_evaluate_result(jobs, evaluations)
                evaluation = aggregate_evaluations(evaluations)
            stage_results.append(
                {
                    "stage": stage_index,
                    "correctness_cases": stage.correctness_cases,
                    "evaluate_repeats": stage.evaluate_repeats,
                    "correct": evaluation.correct,
                    "latency_us": evaluation.latency_us,
                    "job": stage_job,
                }
            )
            if not evaluation.correct:
                break
        job: JsonValue = {
            "schema_version": 1,
            "operation": "bootstrap_staged_evaluate",
            "bench_iters": self._bootstrap_bench_iters,
            "completed_stages": stage_results,
            "all_pass": evaluation.correct and len(stage_results) == len(self._bootstrap_stages),
            "latency_source_stage": len(stage_results) - 1,
            "latency_us": evaluation.latency_us,
        }
        job_id = None
        profile_job: JsonValue | None = None
        if (
            evaluation.correct
            and len(stage_results) == len(self._bootstrap_stages)
            and context.contract.roofline is None
            and self._profile_without_roofline
            and final_payload is not None
        ):
            profile_job = await self._profile(
                attempt_id,
                generation,
                candidate_digest,
                final_payload,
            )
        return self._record_terminal(
            attempt_id,
            candidate_digest,
            idempotency_key=idempotency_key,
            generation=generation,
            agent_evaluation_id=agent_evaluation.id,
            job=job,
            job_id=job_id,
            correct=evaluation.correct and len(stage_results) == len(self._bootstrap_stages),
            latency_us=evaluation.latency_us,
            profile_job=profile_job,
        )

    async def _evaluate_repeated(
        self,
        attempt_id: AttemptId,
        *,
        candidate_source: str,
        operator: str,
        contract: AgateEvaluationContractV1,
        hardware_target: str,
        dsl: Dsl,
        name: str,
        idempotency_key: str,
        event_base: dict[str, object],
        repeats: int,
    ) -> tuple[tuple[JsonValue, str | None, EvaluationV2], ...]:
        """Run all authoritative ordinary Evaluate repetitions concurrently."""
        results: list[tuple[JsonValue, str | None, EvaluationV2] | None] = [
            None for _ in range(repeats)
        ]

        async def run_one(repeat: int) -> None:
            digest = hashlib.sha256(f"{attempt_id}:{idempotency_key}:{repeat}".encode()).hexdigest()
            results[repeat] = await self._evaluate_batched(
                attempt_id,
                candidate_source=candidate_source,
                operator=operator,
                contract=contract,
                hardware_target=hardware_target,
                dsl=dsl,
                name=f"{name}_repeat_{repeat}",
                idempotency_key=f"runtime-final-repeat:{digest}",
                event_base={
                    **event_base,
                    "repeat": repeat,
                    "repeats": repeats,
                },
            )

        async with anyio.create_task_group() as tasks:
            for repeat in range(repeats):
                tasks.start_soon(run_one, repeat)
        completed = tuple(result for result in results if result is not None)
        if len(completed) != repeats:
            raise AssertionError("authoritative ordinary Evaluate repetition produced no result")
        return completed

    async def _evaluate_batched(
        self,
        attempt_id: AttemptId,
        *,
        candidate_source: str,
        operator: str,
        contract: AgateEvaluationContractV1,
        hardware_target: str,
        dsl: Dsl,
        name: str,
        idempotency_key: str,
        event_base: dict[str, object],
    ) -> tuple[JsonValue, str | None, EvaluationV2]:
        """Execute one logical authoritative Evaluate through shared Shape batches."""

        async def evaluate_batch(batch: ShapeBatch) -> ShapeBatchOutcome:
            payload = build_evaluation_request(
                self._request_builder,
                candidate_source=candidate_source,
                operator=operator,
                contract=batch.contract,
                hardware_target=hardware_target,
                dsl=dsl,
                name=f"{name}_batch_{batch.index}",
                idempotency_key=batch.idempotency_key,
            )
            job, job_id, evaluation = await self._evaluate_once(
                attempt_id,
                payload,
                batch.shape_ids,
                {
                    **event_base,
                    "shape_batch": batch.index,
                    "shape_ids": list(batch.shape_ids),
                },
            )
            return ShapeBatchOutcome(job, evaluation, job_id)

        result = await self._shape_batches.run(contract, idempotency_key, evaluate_batch)
        return result.job, result.job_id, result.evaluation

    async def _evaluate_once(
        self,
        attempt_id: AttemptId,
        payload: dict[str, object],
        expected_shape_ids: tuple[str, ...],
        event_base: dict[str, object],
    ) -> tuple[JsonValue, str | None, EvaluationV2]:
        """Submit, await, and parse one authoritative ordinary Evaluate job."""
        try:
            accepted = await self._submit("eval", payload)
        except AgateCandidateRejection as rejection:
            return (
                {"status": "rejected", "error": rejection.payload},
                None,
                EvaluationV2(correct=False, latency_us=None),
            )
        job_id = accepted.get("job_id")
        if not isinstance(job_id, str) or not job_id:
            raise InfrastructureError("authoritative Agate acceptance has no job_id")
        self._events.record_runtime_event(
            "gateway.authoritative_evaluation_submitted",
            attempt_id,
            {**event_base, "agate_job_id": job_id},
        )
        job = await self._wait_for_job(job_id)
        status = job.get("status")
        if status not in _TERMINAL:
            raise InfrastructureError("authoritative Agate job did not reach a terminal state")
        evaluation = (
            parse_agate_evaluation(job, expected_shape_ids)
            if status == "succeeded"
            else EvaluationV2(correct=False, latency_us=None)
        )
        return job, job_id, evaluation

    async def _profile(
        self,
        attempt_id: AttemptId,
        generation: int,
        candidate_digest: ArtifactDigest,
        evaluation_payload: dict[str, object],
    ) -> JsonValue:
        """Run a non-authoritative NCU SOL Profile after a correct Roofline-free eval."""
        profile_key = f"runtime-final-profile:{attempt_id}:{generation}:{candidate_digest}"
        payload = dict(evaluation_payload)
        payload.update(
            {
                "idempotency_key": profile_key,
                "level": "sol",
                "top_kernels": 10,
            }
        )
        try:
            accepted = await self._submit("profile", payload)
            job_id = accepted.get("job_id")
            if not isinstance(job_id, str) or not job_id:
                raise InfrastructureError("authoritative Agate Profile acceptance has no job_id")
            self._events.record_runtime_event(
                "gateway.authoritative_profile_submitted",
                attempt_id,
                {
                    "kernel_artifact_digest": candidate_digest,
                    "agate_job_id": job_id,
                    "recovery_generation": generation,
                },
            )
            job = await self._wait_for_job(job_id)
            if job.get("status") not in _TERMINAL:
                raise InfrastructureError("authoritative Agate Profile did not terminate")
            self._events.record_runtime_event(
                "gateway.authoritative_profile_completed",
                attempt_id,
                {
                    "kernel_artifact_digest": candidate_digest,
                    "agate_job_id": job_id,
                    "recovery_generation": generation,
                    "status": job.get("status"),
                },
            )
            return job
        except Exception as error:
            message = f"{type(error).__name__}: {error}"[:1000]
            self._events.record_runtime_event(
                "gateway.authoritative_profile_failed",
                attempt_id,
                {
                    "kernel_artifact_digest": candidate_digest,
                    "recovery_generation": generation,
                    "error": message,
                },
            )
            return {"status": "failed", "error": {"message": message}}

    def _record_terminal(
        self,
        attempt_id: AttemptId,
        candidate_digest: ArtifactDigest,
        *,
        idempotency_key: str,
        generation: int,
        agent_evaluation_id: str,
        job: JsonValue,
        job_id: str | None,
        correct: bool,
        latency_us: float | None,
        profile_job: JsonValue | None = None,
    ) -> AttemptCandidateResult:
        result_digest = self._store_result(job, profile_job)
        record = self._control.record_evaluation(
            attempt_id,
            source=GatewayEvaluationSource.RUNTIME_FINAL,
            idempotency_key=idempotency_key,
            kernel_artifact_digest=candidate_digest,
            gateway_result_digest=result_digest,
            correct=correct,
            latency_us=latency_us,
            agate_job_id=job_id,
            recovery_generation=generation,
        )
        outcome = self._control.commit_authoritative_outcome(
            attempt_id,
            record.id,
            committed_at=self._clock(),
        )
        self._events.record_runtime_event(
            "gateway.authoritative_evaluation_completed",
            attempt_id,
            {
                "source": GatewayEvaluationSource.RUNTIME_FINAL.value,
                "kernel_artifact_digest": candidate_digest,
                "agent_evaluation_id": agent_evaluation_id,
                "agate_job_id": job_id,
                "recovery_generation": generation,
                "evaluation_id": record.id,
                "gateway_result_digest": result_digest,
                "correct": outcome.correct,
                "latency_us": outcome.latency_us,
            },
        )
        return outcome

    def _store_result(self, job: JsonValue, profile_job: JsonValue | None) -> ArtifactDigest:
        return store_gateway_result(
            self._artifacts,
            job,
            profile_job,
            temporary_prefix="gateway-result-",
        )

    async def _submit(self, kind: str, payload: dict[str, object]) -> dict[str, JsonValue]:
        return await submit_agate_job(
            self._client,
            kind,
            payload,
            request_error="authoritative Agate request failed",
            invalid_response="authoritative Agate response is invalid JSON",
            non_object_response="authoritative Agate response is not an object",
        )

    async def _call(self, operation: Callable[[], object]) -> dict[str, JsonValue]:
        return await call_agate_json(
            operation,
            request_error="authoritative Agate request failed",
            invalid_response="authoritative Agate response is invalid JSON",
            non_object_response="authoritative Agate response is not an object",
        )

    async def _wait_for_job(self, job_id: str) -> dict[str, JsonValue]:
        """Long-poll once; the shared Agate client owns persistent transport retries."""
        return await self._call(
            partial(
                self._client.get_job,
                job_id,
                wait=True,
                timeout=self._wait_timeout_s,
            )
        )
