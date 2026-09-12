"""Agent ABBA experiments over sealed source artifacts without promotion authority."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import cast

import anyio

from ..artifacts.local import JsonValue, LocalArtifactStore
from ..domain.errors import InfrastructureError
from ..domain.ids import ArtifactDigest
from ..kernel_sources import KernelSourceBundle, read_kernel_source
from ..serialization import canonical_json_text
from .abba import (
    AgateSameAllocationAbbaRunner,
    CommitPinnedAtrexBenchEvaluator,
    _parse_remote_payload,
    _schedule,
    build_abba_source_request,
)
from .agate import AgateClient, _nested_infrastructure_error
from .batched_evaluate import (
    EVALUATE_MAX_PARALLEL_BATCHES,
    sorted_shape_ids,
)
from .candidate import resolve_kernel_candidate
from .contract import AgateEvaluationContext, AgateEvaluationContextResolver
from .control_models import GatewayOperation
from .correctness import correctness_summary
from .execution import call_agate_json
from .job_recovery import JobExecution, run_with_log_recovery
from .private_results import project_private_job
from .protocol import AGATE_MAX_JOB_TIMEOUT_S, EvaluateParametersV2
from .proxy import GatewayAdapter, GatewayAdapterRequest, GatewayAdapterResult


@dataclass(frozen=True, slots=True)
class _BatchResult:
    job: dict[str, JsonValue]
    payload: dict[str, JsonValue] | None
    error: JsonValue | None = None


class AgentAbbaGatewayAdapter:
    """Execute comparative Agent measurements without creating authoritative evaluations."""

    def __init__(
        self,
        delegate: GatewayAdapter,
        client: AgateClient,
        contexts: AgateEvaluationContextResolver,
        artifacts: LocalArtifactStore,
        evaluator: CommitPinnedAtrexBenchEvaluator | None,
        *,
        wait_timeout_s: float,
        correctness_cases: int = 5,
        bench_iters: int = 100,
        per_run_timeout_seconds: float = 120,
        allocation_timeout_seconds: float = 600,
    ) -> None:
        for label, value in (
            ("wait timeout", wait_timeout_s),
            ("per-run timeout", per_run_timeout_seconds),
            ("allocation timeout", allocation_timeout_seconds),
        ):
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"Agent ABBA {label} must be positive and finite")
        if allocation_timeout_seconds > AGATE_MAX_JOB_TIMEOUT_S:
            raise ValueError("Agent ABBA allocation timeout exceeds the Agate job limit")
        if correctness_cases <= 0 or bench_iters <= 0:
            raise ValueError("Agent ABBA sampling counts must be positive")
        self._delegate = delegate
        self._client = client
        self._contexts = contexts
        self._artifacts = artifacts
        self._evaluator = evaluator
        self._wait_timeout_s = wait_timeout_s
        self._correctness_cases = correctness_cases
        self._bench_iters = bench_iters
        self._per_run_timeout_seconds = per_run_timeout_seconds
        self._allocation_timeout_seconds = allocation_timeout_seconds

    async def execute(self, request: GatewayAdapterRequest) -> GatewayAdapterResult:
        if (
            request.operation is not GatewayOperation.EVALUATE
            or request.parameters.get("comparison") is None
        ):
            return await self._delegate.execute(request)
        parameters = EvaluateParametersV2.model_validate(request.parameters)
        assert parameters.comparison is not None
        repeats = parameters.comparison.repeats
        schedule = _schedule(repeats)
        required_seconds = len(schedule) * self._per_run_timeout_seconds + 30
        if required_seconds > self._allocation_timeout_seconds:
            raise ValueError(
                f"ABBA repeats={repeats} requires a {required_seconds:g}s allocation "
                f"at the configured {self._per_run_timeout_seconds:g}s per-run budget; "
                f"the allocation limit is {self._allocation_timeout_seconds:g}s. "
                "Reduce repeats or ask the Runtime operator to adjust the time budget."
            )
        if self._evaluator is None:
            raise ValueError("Agent ABBA requires a configured commit-pinned Atrex Bench evaluator")
        if request.candidate_digest is None or request.baseline_candidate_digest is None:
            raise ValueError("Agent ABBA requires sealed baseline and current candidate artifacts")
        context = self._effective_context(request, parameters)
        baseline_source = self._source(request.baseline_candidate_digest, context)
        candidate_source = self._source(request.candidate_digest, context)
        evaluator_files = await anyio.to_thread.run_sync(self._evaluator.files)
        evaluator_digest = self._evaluator.bundle_digest()
        shape_ids = list(sorted_shape_ids(context.contract))
        comparison_id = hashlib.sha256(
            canonical_json_text(
                {
                    "attempt_id": request.attempt_id,
                    "recovery_generation": request.recovery_generation,
                    "idempotency_key": request.idempotency_key,
                    "baseline": request.baseline_candidate_digest,
                    "candidate": request.candidate_digest,
                    "parameters": parameters.model_dump(mode="json"),
                    "contract": context.contract.model_dump(mode="json"),
                    "evaluator_bundle_digest": evaluator_digest,
                }
            ).encode()
        ).hexdigest()
        batches: list[_BatchResult | None] = [None] * len(shape_ids)
        limiter = anyio.Semaphore(EVALUATE_MAX_PARALLEL_BATCHES)

        async def run_batch(index: int, shape_id: str) -> None:
            async with limiter:
                payload = build_abba_source_request(
                    hardware_target=context.agate_gpu,
                    contract=context.contract,
                    shape_ids=[shape_id],
                    schedule=schedule,
                    incumbent_source=baseline_source,
                    candidate_source=candidate_source,
                    evaluator_files=evaluator_files,
                    per_run_timeout_seconds=self._per_run_timeout_seconds,
                    allocation_timeout_seconds=self._allocation_timeout_seconds,
                )
                key = hashlib.sha256(f"{comparison_id}:{shape_id}".encode()).hexdigest()
                payload.update(
                    idempotency_key=f"agent-abba:{key}",
                    dev_note="Agent same-allocation ABBA experiment",
                )
                batches[index] = await self._run_batch(payload, schedule, [shape_id])

        try:
            async with anyio.create_task_group() as tasks:
                for index, shape_id in enumerate(shape_ids):
                    tasks.start_soon(run_batch, index, shape_id)
        except BaseExceptionGroup as errors:
            infrastructure = _nested_infrastructure_error(errors)
            if infrastructure is not None:
                raise infrastructure from errors
            raise
        completed = [batch for batch in batches if batch is not None]
        if len(completed) != len(shape_ids):
            raise InfrastructureError("Agent ABBA did not produce every requested Shape batch")
        complete_payloads = [batch.payload for batch in completed if batch.payload is not None]
        failed = len(complete_payloads) != len(completed)
        merged = (
            []
            if failed
            else AgateSameAllocationAbbaRunner._merge_payloads(
                complete_payloads, schedule, shape_ids
            )
        )
        baseline = AgateSameAllocationAbbaRunner._aggregate_revision_metrics(
            merged, "incumbent", repeats, shape_ids
        )
        candidate = AgateSameAllocationAbbaRunner._aggregate_revision_metrics(
            merged, "candidate", repeats, shape_ids
        )
        public = self._public_result(request, parameters, schedule, merged, baseline, candidate)
        public["shape_batch_count"] = len(shape_ids)
        public["max_parallel_shape_batches"] = EVALUATE_MAX_PARALLEL_BATCHES
        if failed:
            public["error"] = {
                "category": "abba_execution_failed",
                "message": "ABBA could not complete its scheduled measurements",
            }
        raw: dict[str, JsonValue] = {
            **public,
            "comparison_id": comparison_id,
            "atrex_bench_commit": self._evaluator.commit,
            "evaluator_bundle_digest": evaluator_digest,
            "evaluation_contract_digest": context.evaluation_contract_digest,
            "evaluation_parameters": cast(
                JsonValue, parameters.model_dump(mode="json", exclude_none=True)
            ),
            "shape_batches": [[shape_id] for shape_id in shape_ids],
            "jobs": [batch.job for batch in completed],
            "payloads": [batch.payload for batch in completed],
            "batch_errors": [batch.error for batch in completed],
        }
        return GatewayAdapterResult(
            status="failed" if failed else "completed", result=raw, worker_result=public
        )

    def _effective_context(
        self, request: GatewayAdapterRequest, parameters: EvaluateParametersV2
    ) -> AgateEvaluationContext:
        context = self._contexts.resolve(request.attempt_id)
        overrides = EvaluateParametersV2(
            input_py=parameters.input_py, shapes=parameters.shapes, mode="full"
        )
        updates: dict[str, object] = {
            "mode": "full",
            "options": context.contract.options.model_copy(
                update={
                    "num_correctness_cases": self._correctness_cases,
                    "bench_iters": self._bench_iters,
                }
            ),
        }
        if overrides.input_py is not None:
            updates["input_py"] = overrides.input_py
        if overrides.shapes is not None:
            updates["shapes"] = overrides.shapes
        if overrides.input_scope == "custom":
            updates.update(metadata=None, roofline=None)
        return replace(context, contract=context.contract.model_copy(update=updates, deep=True))

    def _source(
        self, digest: ArtifactDigest, context: AgateEvaluationContext
    ) -> str | KernelSourceBundle:
        source = resolve_kernel_candidate(
            self._artifacts,
            digest,
            context.contract.candidate_path,
            error_type=ValueError,
            kind_error="ABBA source Artifact must have Kernel kind",
            missing_error="ABBA source Artifact is missing its contract candidate file",
        )
        try:
            return read_kernel_source(
                source.root, context.contract.candidate_path, context.kernel_source
            )
        except UnicodeDecodeError as error:
            raise ValueError("ABBA source must be UTF-8") from error

    async def _run_batch(
        self,
        payload: dict[str, object],
        schedule: list[dict[str, int | str]],
        shape_ids: list[str],
    ) -> _BatchResult:
        async def execute(submission: dict[str, object]) -> JobExecution:
            accepted = await self._call(lambda: self._client.submit_job("dev", submission))
            job_id = accepted.get("job_id")
            if not isinstance(job_id, str) or not job_id:
                raise InfrastructureError("Agent ABBA acceptance has no job_id")
            job = await self._call(
                lambda: self._client.get_job(job_id, wait=True, timeout=self._wait_timeout_s)
            )
            return job_id, job

        _, job = await run_with_log_recovery(payload, execute)
        if job.get("status") not in {"succeeded", "failed", "cancelled"}:
            raise InfrastructureError("Agent ABBA job did not reach a terminal state")
        if job.get("status") != "succeeded" or job.get("command_ok") is False:
            return _BatchResult(job, None, project_private_job(job))
        command_result = job.get("result")
        if isinstance(command_result, dict) and command_result.get("exit_code") is not None:
            code = command_result["exit_code"]
            if type(code) is not int or code != 0:
                return _BatchResult(job, None, {"message": "ABBA command exited unsuccessfully"})
        try:
            parsed = _parse_remote_payload(job, schedule)
            self._validate_batch(parsed, shape_ids)
        except InfrastructureError as error:
            return _BatchResult(job, None, {"message": str(error)})
        return _BatchResult(job, parsed)

    @staticmethod
    def _validate_batch(payload: dict[str, JsonValue], shape_ids: list[str]) -> None:
        rows = payload.get("runs")
        if not isinstance(rows, list) or payload.get("error") is not None:
            raise InfrastructureError("Agent ABBA driver returned invalid measurements")
        for row in rows:
            if (
                not isinstance(row, dict)
                or type(row.get("repeat")) is not int
                or type(row.get("exit_code")) is not int
            ):
                raise InfrastructureError("Agent ABBA run has an invalid repeat or exit code")
            result = row.get("result")
            if not isinstance(result, dict) or result.get("all_pass") is not True:
                continue
            latencies = result.get("latency_us_by_shape")
            if (
                result.get("error") is not None
                or not isinstance(latencies, dict)
                or (
                    set(latencies) != set(shape_ids)
                    or any(
                        isinstance(value, bool)
                        or not isinstance(value, (int, float))
                        or not math.isfinite(value)
                        or value <= 0
                        for value in latencies.values()
                    )
                )
            ):
                raise InfrastructureError(
                    "Agent ABBA run is missing valid expected Shape latencies"
                )

    @staticmethod
    def _public_result(
        request: GatewayAdapterRequest,
        parameters: EvaluateParametersV2,
        schedule: list[dict[str, int | str]],
        measurements: list[dict[str, object]],
        baseline: dict[str, object],
        candidate: dict[str, object],
    ) -> dict[str, JsonValue]:
        assert parameters.comparison is not None
        sides = {"incumbent": "A", "candidate": "B"}

        def metrics(value: dict[str, object]) -> dict[str, JsonValue]:
            return {
                "correct": value.get("correct") is True,
                "correctness": correctness_summary(value, passed=value.get("correct") is True),
                "latency_us_geomean": cast(JsonValue, value.get("latency_us")),
                "latency_us_by_shape": cast(JsonValue, value.get("latency_us_by_shape", {})),
            }

        correct = baseline.get("correct") is True and candidate.get("correct") is True
        a = baseline.get("latency_us")
        b = candidate.get("latency_us")
        speedup = None
        improvement = None
        if correct and isinstance(a, (int, float)) and isinstance(b, (int, float)):
            speedup = a / b
            improvement = (a - b) / a * 100
        return {
            "baseline_kernel_artifact_digest": request.baseline_candidate_digest,
            "kernel_artifact_digest": request.candidate_digest,
            "mode": "full",
            "input_scope": parameters.input_scope,
            "comparison": {
                "method": parameters.comparison.method,
                "repeats": parameters.comparison.repeats,
            },
            "aggregation": "geometric_mean",
            "correct": correct,
            "baseline": metrics(baseline),
            "candidate": metrics(candidate),
            "speedup": speedup,
            "improvement_pct": improvement,
            "schedule": [
                {"side": sides[str(step["revision"])], "repeat": step["repeat"]}
                for step in schedule
            ],
            "measurements": [
                {
                    "side": sides[str(row["revision"])],
                    "repeat": cast(JsonValue, row["repeat"]),
                    "correct": row.get("correct") is True,
                    "correctness": correctness_summary(row, passed=row.get("correct") is True),
                    "latency_us": cast(JsonValue, row.get("latency_us")),
                    "latency_us_by_shape": cast(JsonValue, row.get("latency_us_by_shape", {})),
                }
                for row in measurements
            ],
            "hidden_case_details": "shape inputs and failure details withheld",
        }

    async def _call(self, operation: Callable[[], object]) -> dict[str, JsonValue]:
        return await call_agate_json(
            operation,
            request_error="Agent ABBA request failed",
            invalid_response="Agent ABBA returned invalid JSON",
            non_object_response="Agent ABBA returned a non-object",
        )
