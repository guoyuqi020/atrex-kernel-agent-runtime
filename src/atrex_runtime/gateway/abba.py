"""Trusted same-allocation ABBA measurements over Agate dev jobs."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
import tempfile
import threading
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import cast
from uuid import uuid4

import anyio

from ..artifacts.local import ArtifactKind, JsonValue, LocalArtifactStore
from ..domain.errors import InfrastructureError
from ..domain.ids import ArtifactDigest
from ..domain.models import KernelMeasurement, KernelMeasurementPurpose, KernelRevision
from ..git_import import SafeGitImporter
from ..ports import (
    KernelMeasurementJournal,
    KernelMeasurementRun,
    KernelPairMeasurementResult,
    KernelPairMeasurementRunner,
)
from ..roofline import strip_roofline_hardware_suffix
from . import abba_remote
from .agate import AgateClient
from .batched_evaluate import sorted_shape_ids, subset_shape_document
from .candidate import resolve_kernel_candidate
from .contract import AgateEvaluationContractV1, RegistryKernelEvaluationContextResolver
from .correctness import merge_correctness_summaries
from .execution import call_agate_json
from .job_recovery import JobExecution, run_with_log_recovery

_TERMINAL = frozenset({"succeeded", "failed", "cancelled"})
# One transient Agate batch used to fail the whole Epoch selection, discarding every sibling
# batch that had already succeeded. The batch Job is idempotent, so a transient failure is
# retried; the ceiling keeps a permanently unavailable Agate from hanging the selection.
_ABBA_BATCH_RETRIES = 10
_ABBA_RETRY_DELAY_SECONDS = 60.0


class AbbaBatchFailure(InfrastructureError):
    """One ABBA batch Job that produced no usable measurement, with Agate's own reason."""

    def __init__(self, job: dict[str, JsonValue]) -> None:
        error = job.get("error")
        detail = error if isinstance(error, dict) else {}
        self.error_class = detail.get("error_class")
        self.reason = detail.get("reason")
        self.trace_id = detail.get("trace_id") or job.get("trace_id")
        self.retryable = True
        super().__init__(
            "Agate ABBA batch produced no measurement: "
            f"job_id={job.get('job_id')} status={job.get('status')} "
            f"command_ok={job.get('command_ok')} error_class={self.error_class} "
            f"reason={self.reason} message={detail.get('message')} trace_id={self.trace_id}"
        )

    def event_payload(self) -> dict[str, JsonValue]:
        """Agate's failure identity, so a Runtime event can be reconciled against Agate."""
        return {
            "error_class": self.error_class,
            "reason": self.reason,
            "trace_id": self.trace_id,
            "retryable": self.retryable,
            "detail": str(self),
        }


_RUNTIME_PATHS = ("scripts/run_eval.py", "src/atrex_bench")
_PROTECTED_RUNNER_KEYS = frozenset(
    {
        "schema_version",
        "eval_mode",
        "input",
        "reference_dir",
        "output",
        "checkpoint_dir",
        "clock_locked",
        "require_clock_locked",
        "clock_lock_mode",
        "clock_lock_device",
        "gpu_clock_mhz",
        "memory_clock_mhz",
        "clock_lock_tolerance_mhz",
        "clock_lock_settle_seconds",
        "clock_lock_command_timeout_s",
        "clock_lock_require_idle",
        "clock_lock_monitor",
        "clock_lock_sample_interval_ms",
        "clock_lock_runtime_tolerance_mhz",
        "clock_lock_fail_on_deviation",
    }
)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _schedule(repeats: int) -> list[dict[str, int | str]]:
    schedule: list[dict[str, int | str]] = []
    for repeat in range(repeats):
        revisions = ("incumbent", "candidate") if repeat % 2 == 0 else ("candidate", "incumbent")
        schedule.extend({"revision": revision, "repeat": repeat} for revision in revisions)
    return schedule


class CommitPinnedAtrexBenchEvaluator:
    """Lazily export the evaluator-only subset of one exact Atrex Bench commit."""

    def __init__(
        self,
        *,
        repository: str,
        commit: str,
        git_executable: str | Path,
        fetch_timeout_seconds: float,
        max_archive_bytes: int,
        max_bundle_files: int,
        max_bundle_bytes: int,
    ) -> None:
        if not repository.strip() or not commit:
            raise ValueError("Atrex Bench comparison source cannot be empty")
        if max_bundle_files <= 0 or max_bundle_bytes <= 0:
            raise ValueError("Atrex Bench comparison Bundle limits must be positive")
        self._repository = repository
        self._commit = commit
        self._max_bundle_files = max_bundle_files
        self._max_bundle_bytes = max_bundle_bytes
        self._importer = SafeGitImporter(
            git_executable,
            timeout_seconds=fetch_timeout_seconds,
            max_archive_bytes=max_archive_bytes,
            label="Atrex Bench comparison evaluator",
        )
        self._lock = threading.Lock()
        self._files: dict[str, str] | None = None
        self._bundle_digest: str | None = None

    @property
    def commit(self) -> str:
        """Return the exact source commit used for every exported evaluator Bundle."""
        return self._commit

    def files(self) -> dict[str, str]:
        """Return an immutable source mapping suitable for an inline Agate dev payload."""
        with self._lock:
            if self._files is None:
                self._files = self._export()
                digest = hashlib.sha256()
                for relative, content in sorted(self._files.items()):
                    path = relative.encode("utf-8")
                    payload = content.encode("utf-8")
                    digest.update(len(path).to_bytes(8, "big"))
                    digest.update(path)
                    digest.update(len(payload).to_bytes(8, "big"))
                    digest.update(payload)
                self._bundle_digest = f"sha256:{digest.hexdigest()}"
            return dict(self._files)

    def bundle_digest(self) -> str:
        """Return the canonical digest of the exact uploaded evaluator-only Bundle."""
        self.files()
        if self._bundle_digest is None:
            raise AssertionError("Atrex Bench comparison Bundle digest was not initialized")
        return self._bundle_digest

    def _export(self) -> dict[str, str]:
        with tempfile.TemporaryDirectory(prefix="atrex-abba-evaluator-") as temporary:
            root = Path(temporary)
            repository = root / "repository"
            archive = root / "source.tar"
            export = root / "export"
            self._importer.run(("init", "--bare", str(repository)))
            self._importer.run(("-C", str(repository), "remote", "add", "origin", self._repository))
            self._importer.fetch_commit(repository, "origin", self._commit)
            resolved = self._importer.object_id(
                self._importer.run(("-C", str(repository), "rev-parse", "FETCH_HEAD^{commit}"))
            )
            if resolved != self._commit:
                raise ValueError("Git fetch resolved a different Atrex Bench comparison commit")
            self._importer.archive(
                repository,
                "FETCH_HEAD",
                archive,
                paths=_RUNTIME_PATHS,
            )
            export.mkdir(mode=0o700)
            self._importer.extract(archive, export)
            files: dict[str, str] = {}
            total_bytes = 0
            for path in sorted(export.rglob("*")):
                if not path.is_file() or "__pycache__" in path.parts:
                    continue
                relative = path.relative_to(export).as_posix()
                try:
                    content = path.read_text(encoding="utf-8")
                except UnicodeDecodeError as error:
                    raise ValueError(
                        f"Atrex Bench comparison source is not UTF-8: {relative}"
                    ) from error
                files[f"atrex-bench/{relative}"] = content
                total_bytes += len(content.encode("utf-8"))
                if len(files) > self._max_bundle_files:
                    raise ValueError("Atrex Bench comparison Bundle exceeds file limit")
                if total_bytes > self._max_bundle_bytes:
                    raise ValueError("Atrex Bench comparison Bundle exceeds byte limit")
            required = {
                "atrex-bench/scripts/run_eval.py",
                "atrex-bench/src/atrex_bench/__init__.py",
                "atrex-bench/src/atrex_bench/sdk.py",
            }
            if not required.issubset(files):
                raise ValueError("Atrex Bench commit lacks the required evaluation runtime")
            return files


class AgateSameAllocationAbbaRunner(KernelPairMeasurementRunner):
    """Run every A/B pair inside one Agate allocation for each private Shape batch."""

    def __init__(
        self,
        client: AgateClient,
        contexts: RegistryKernelEvaluationContextResolver,
        artifacts: LocalArtifactStore,
        journal: KernelMeasurementJournal,
        evaluator: CommitPinnedAtrexBenchEvaluator,
        *,
        wait_timeout_s: float,
        clock: Callable[[], str] = _utc_now,
    ) -> None:
        if wait_timeout_s <= 0:
            raise ValueError("Agate ABBA wait timeout must be positive")
        self._client = client
        self._contexts = contexts
        self._artifacts = artifacts
        self._journal = journal
        self._evaluator = evaluator
        self._wait_timeout_s = wait_timeout_s
        self._clock = clock

    async def run_pair(
        self,
        incumbent: KernelRevision,
        candidate: KernelRevision,
        *,
        repeats: int,
        purpose: KernelMeasurementPurpose,
        per_run_timeout_seconds: float,
        allocation_timeout_seconds: float,
        shape_batch_size: int,
        max_parallel_shape_batches: int,
    ) -> KernelPairMeasurementResult:
        """Submit exact interleaved schedules and persist every authoritative run."""
        if repeats <= 0 or shape_batch_size <= 0 or max_parallel_shape_batches <= 0:
            raise ValueError("ABBA repetition and batch limits must be positive")
        schedule = _schedule(repeats)
        if per_run_timeout_seconds * len(schedule) + 30 > allocation_timeout_seconds:
            raise ValueError("ABBA schedule cannot fit inside one allocation timeout")

        incumbent_context = self._contexts.resolve(incumbent)
        candidate_context = self._contexts.resolve(candidate)
        if incumbent_context != candidate_context:
            raise ValueError("ABBA Kernels do not share one immutable evaluation context")
        context = incumbent_context
        contract = context.contract
        if contract.mode != "full":
            raise ValueError("same-allocation ABBA requires a full evaluation contract")
        incumbent_source = self._kernel_source(incumbent, contract)
        candidate_source = self._kernel_source(candidate, contract)
        evaluator_files = await anyio.to_thread.run_sync(self._evaluator.files)
        evaluator_bundle_digest = self._evaluator.bundle_digest()
        shape_ids = list(sorted_shape_ids(contract))
        batches = tuple(
            shape_ids[offset : offset + shape_batch_size]
            for offset in range(0, len(shape_ids), shape_batch_size)
        )
        comparison_id = uuid4().hex
        payloads: list[dict[str, JsonValue] | None] = [None] * len(batches)
        jobs: list[dict[str, JsonValue] | None] = [None] * len(batches)
        limiter = anyio.Semaphore(max_parallel_shape_batches)

        async def run_batch(index: int, batch: list[str]) -> None:
            attempt = 0
            while True:
                try:
                    async with limiter:
                        job, payload = await self._run_batch(
                            comparison_id=comparison_id,
                            batch_index=index,
                            context_name=context.operator,
                            hardware_target=context.agate_gpu,
                            contract=contract,
                            shape_ids=batch,
                            schedule=schedule,
                            incumbent_source=incumbent_source,
                            candidate_source=candidate_source,
                            evaluator_files=evaluator_files,
                            per_run_timeout_seconds=per_run_timeout_seconds,
                            allocation_timeout_seconds=allocation_timeout_seconds,
                            purpose=purpose,
                            incumbent=incumbent,
                            candidate=candidate,
                        )
                except InfrastructureError as failure:
                    attempt += 1
                    if attempt > _ABBA_BATCH_RETRIES:
                        raise
                    failure_payload: dict[str, JsonValue]
                    if isinstance(failure, AbbaBatchFailure):
                        failure_payload = failure.event_payload()
                    else:
                        failure_payload = {
                            "error_class": None,
                            "reason": None,
                            "trace_id": None,
                            "retryable": True,
                            "failure_type": type(failure).__name__,
                            "detail": str(failure),
                        }
                    self._journal.record_runtime_event(
                        "comparison.abba_batch_retried",
                        candidate.id,
                        {
                            "comparison_id": comparison_id,
                            "batch_index": index,
                            "operator": context.operator,
                            "attempt": attempt,
                            "max_retries": _ABBA_BATCH_RETRIES,
                            "retry_delay_seconds": _ABBA_RETRY_DELAY_SECONDS,
                            **failure_payload,
                        },
                    )
                    await anyio.sleep(_ABBA_RETRY_DELAY_SECONDS)
                    continue
                jobs[index] = job
                payloads[index] = payload
                return

        async with anyio.create_task_group() as tasks:
            for index, batch in enumerate(batches):
                tasks.start_soon(run_batch, index, batch)

        complete_jobs = [job for job in jobs if job is not None]
        complete_payloads = [payload for payload in payloads if payload is not None]
        if len(complete_jobs) != len(batches) or len(complete_payloads) != len(batches):
            raise InfrastructureError("ABBA Shape batch execution was incomplete")
        merged = self._merge_payloads(complete_payloads, schedule, shape_ids)
        aggregate: dict[str, JsonValue] = {
            "schema_version": 1,
            "operation": "same_allocation_abba",
            "comparison_id": comparison_id,
            "atrex_bench_commit": self._evaluator.commit,
            "evaluator_bundle_digest": evaluator_bundle_digest,
            "evaluation_contract_digest": (
                None
                if context.evaluation_contract_digest is None
                else str(context.evaluation_contract_digest)
            ),
            "schedule": cast(list[JsonValue], schedule),
            "shape_batches": cast(list[JsonValue], [list(batch) for batch in batches]),
            "jobs": cast(list[JsonValue], complete_jobs),
            "payloads": cast(list[JsonValue], complete_payloads),
            "measurements": cast(list[JsonValue], merged),
            "incumbent": cast(
                dict[str, JsonValue],
                self._aggregate_revision_metrics(merged, "incumbent", repeats, shape_ids),
            ),
            "candidate": cast(
                dict[str, JsonValue],
                self._aggregate_revision_metrics(merged, "candidate", repeats, shape_ids),
            ),
        }
        result_digest = self._artifacts.put_json(aggregate, ArtifactKind.GATEWAY_RESULT)
        job_ids = [job.get("job_id") for job in complete_jobs]
        joined_job_ids = ",".join(value for value in job_ids if isinstance(value, str)) or None
        incumbent_runs, candidate_runs = self._record_runs(
            merged,
            incumbent,
            candidate,
            purpose,
            repeats,
            result_digest,
            joined_job_ids,
        )
        self._journal.record_runtime_event(
            "comparison.abba_completed",
            candidate.id,
            {
                "comparison_id": comparison_id,
                "atrex_bench_commit": self._evaluator.commit,
                "evaluator_bundle_digest": evaluator_bundle_digest,
                "purpose": purpose.value,
                "incumbent_kernel_revision_id": incumbent.id,
                "candidate_kernel_revision_id": candidate.id,
                "repeats": repeats,
                "shape_batch_count": len(batches),
                "agate_job_ids": [value for value in job_ids if isinstance(value, str)],
                "gateway_result_digest": result_digest,
            },
        )
        return KernelPairMeasurementResult(
            incumbent_runs,
            candidate_runs,
            gateway_result_digest=result_digest,
        )

    async def _run_batch(
        self,
        *,
        comparison_id: str,
        batch_index: int,
        context_name: str,
        hardware_target: str,
        contract: AgateEvaluationContractV1,
        shape_ids: list[str],
        schedule: list[dict[str, int | str]],
        incumbent_source: str,
        candidate_source: str,
        evaluator_files: dict[str, str],
        per_run_timeout_seconds: float,
        allocation_timeout_seconds: float,
        purpose: KernelMeasurementPurpose,
        incumbent: KernelRevision,
        candidate: KernelRevision,
    ) -> tuple[dict[str, JsonValue], dict[str, JsonValue]]:
        files = dict(evaluator_files)
        files.update(
            {
                "__atrex_abba.py": Path(abba_remote.__file__).read_text(encoding="utf-8"),
                "snapshots/incumbent.py": incumbent_source,
                "snapshots/candidate.py": candidate_source,
                "reference/reference.py": contract.reference_py,
                "reference/input.py": contract.input_py,
                "reference/shapes.json": _json_text(
                    {shape_id: contract.shapes[shape_id] for shape_id in shape_ids}
                ),
            }
        )
        metadata = subset_shape_document(contract.metadata, shape_ids, metadata=True)
        roofline = subset_shape_document(contract.roofline, shape_ids, metadata=False)
        if metadata is not None:
            files["reference/metadata.json"] = _json_text(metadata)
        if roofline is not None:
            files["reference/roofline.json"] = _json_text(strip_roofline_hardware_suffix(roofline))
        evaluator = dict(contract.runner_overrides)
        if _PROTECTED_RUNNER_KEYS.intersection(evaluator):
            raise ValueError("evaluation runner_overrides cannot replace ABBA-owned paths")
        evaluator.update(
            {
                "schema_version": "v1",
                "atol": contract.options.atol,
                "rtol": contract.options.rtol,
                "num_correctness_cases": contract.options.num_correctness_cases,
                "warmup_iters": _positive_runner_number(evaluator, "warmup_iters", 10),
                "bench_iters": contract.options.bench_iters,
                "candidate_timeout_s": _positive_runner_number(
                    evaluator,
                    "candidate_timeout_s",
                    min(float(contract.options.timeout_s), per_run_timeout_seconds),
                ),
                "perf_timeout_s": per_run_timeout_seconds,
                "validation_mode": contract.mode,
            }
        )
        # The outer Runtime driver owns one lock across the complete A/B
        # schedule. Each canonical evaluator subprocess either verifies the
        # inherited marker or explicitly stays off with the Contract policy.
        evaluator["clock_lock_mode"] = "external" if contract.lock_clocks else "off"
        request: dict[str, JsonValue] = {
            "schema_version": 1,
            "schedule": cast(list[JsonValue], schedule),
            "shape_ids": list(shape_ids),
            "sources": {
                "incumbent": "snapshots/incumbent.py",
                "candidate": "snapshots/candidate.py",
            },
            "evaluator": evaluator,
            "per_run_timeout_seconds": per_run_timeout_seconds,
            "lock_clocks": contract.lock_clocks,
        }
        files["request.json"] = _json_text(request)
        dev_request: dict[str, object] = {
            "spec": {"target_hardware": [hardware_target]},
            "command": "python3 __atrex_abba.py request.json",
            "timeout_s": math.ceil(allocation_timeout_seconds),
            "env_vars": contract.env_vars,
            "files": files,
            "recycle": True,
            "dev_intent": "custom_harness",
            "dev_note": "trusted same-allocation ABBA performance gate",
        }
        async def execute(submission: dict[str, object]) -> JobExecution:
            accepted = await self._call(lambda: self._client.submit_job("dev", submission))
            job_id = accepted.get("job_id")
            if not isinstance(job_id, str) or not job_id:
                raise InfrastructureError("Agate ABBA acceptance has no job_id")
            self._journal.record_runtime_event(
                "comparison.abba_batch_submitted",
                candidate.id,
                {
                    "comparison_id": comparison_id,
                    "batch_index": batch_index,
                    "purpose": purpose.value,
                    "incumbent_kernel_revision_id": incumbent.id,
                    "candidate_kernel_revision_id": candidate.id,
                    "shape_count": len(shape_ids),
                    "agate_job_id": job_id,
                    "operator": context_name,
                },
            )
            job = await self._call(
                lambda: self._client.get_job(job_id, wait=True, timeout=self._wait_timeout_s)
            )
            return job_id, job

        _, job = await run_with_log_recovery(dev_request, execute)
        if job.get("status") not in _TERMINAL:
            raise InfrastructureError("Agate ABBA job did not reach a terminal state")
        if job.get("status") != "succeeded" or job.get("command_ok") is False:
            raise AbbaBatchFailure(job)
        return job, _parse_remote_payload(job, schedule)

    def _kernel_source(
        self,
        revision: KernelRevision,
        contract: AgateEvaluationContractV1,
    ) -> str:
        candidate = resolve_kernel_candidate(
            self._artifacts,
            revision.artifact_digest,
            contract.candidate_path,
            error_type=InfrastructureError,
            kind_error="ABBA Kernel Artifact has the wrong kind",
            missing_error="ABBA Kernel candidate file is missing",
        ).source
        try:
            return candidate.read_text(encoding="utf-8")
        except UnicodeDecodeError as error:
            raise InfrastructureError("ABBA Kernel source is not UTF-8") from error

    @staticmethod
    def _merge_payloads(
        payloads: list[dict[str, JsonValue]],
        schedule: list[dict[str, int | str]],
        shape_ids: list[str],
    ) -> list[dict[str, object]]:
        merged: list[dict[str, object]] = []
        for index, step in enumerate(schedule):
            rows = []
            for payload in payloads:
                raw_rows = payload.get("runs")
                if not isinstance(raw_rows, list) or index >= len(raw_rows):
                    raise InfrastructureError("ABBA batch result is incomplete")
                row = raw_rows[index]
                if not isinstance(row, dict):
                    raise InfrastructureError("ABBA batch run is invalid")
                rows.append(row)
            latency_by_shape: dict[str, float] = {}
            sol_pct_by_shape: dict[str, float] = {}
            correctness_values: list[object] = []
            passed = True
            for row in rows:
                if row.get("exit_code") != 0:
                    passed = False
                result = row.get("result")
                if not isinstance(result, dict):
                    passed = False
                    continue
                correctness_value = result.get("correctness")
                if isinstance(correctness_value, dict):
                    correctness_values.append(correctness_value)
                if result.get("all_pass") is not True:
                    passed = False
                    continue
                by_shape = result.get("latency_us_by_shape")
                if not isinstance(by_shape, dict):
                    passed = False
                    continue
                for shape_id, latency in by_shape.items():
                    if (
                        isinstance(shape_id, str)
                        and isinstance(latency, (int, float))
                        and not isinstance(latency, bool)
                        and latency > 0
                        and math.isfinite(float(latency))
                    ):
                        latency_by_shape[shape_id] = float(latency)
                raw_sol = result.get("sol_pct_by_shape")
                if isinstance(raw_sol, dict):
                    for shape_id, percentage in raw_sol.items():
                        if (
                            isinstance(shape_id, str)
                            and isinstance(percentage, (int, float))
                            and not isinstance(percentage, bool)
                            and percentage >= 0
                            and math.isfinite(float(percentage))
                        ):
                            sol_pct_by_shape[shape_id] = float(percentage)
            passed = passed and set(latency_by_shape) == set(shape_ids)
            latency = (
                math.exp(
                    statistics.fmean(math.log(latency_by_shape[shape_id]) for shape_id in shape_ids)
                )
                if passed
                else None
            )
            merged.append(
                {
                    **step,
                    "correct": passed,
                    "correctness": merge_correctness_summaries(
                        correctness_values,
                        passed=passed,
                    ),
                    "latency_us": latency,
                    "latency_us_by_shape": latency_by_shape,
                    "sol_pct_by_shape": sol_pct_by_shape,
                }
            )
        return merged

    @staticmethod
    def _aggregate_revision_metrics(
        rows: list[dict[str, object]],
        revision: str,
        repeats: int,
        shape_ids: list[str],
    ) -> dict[str, object]:
        selected = sorted(
            (row for row in rows if row.get("revision") == revision),
            key=lambda row: int(cast(int, row["repeat"])),
        )
        expected_repeats = list(range(repeats))
        actual_repeats = [int(cast(int, row["repeat"])) for row in selected]
        correct = actual_repeats == expected_repeats and all(
            row.get("correct") is True for row in selected
        )
        latency_values = [
            float(value)
            for row in selected
            if isinstance((value := row.get("latency_us")), (int, float))
            and not isinstance(value, bool)
            and value > 0
            and math.isfinite(float(value))
        ]
        latency_us = (
            math.exp(statistics.fmean(math.log(value) for value in latency_values))
            if correct and len(latency_values) == repeats
            else None
        )

        latency_by_shape = AgateSameAllocationAbbaRunner._aggregate_shape_values(
            selected,
            "latency_us_by_shape",
            shape_ids,
            repeats,
        )
        sol_by_shape = AgateSameAllocationAbbaRunner._aggregate_shape_values(
            selected,
            "sol_pct_by_shape",
            shape_ids,
            repeats,
            allow_zero=True,
        )
        if len(sol_by_shape) == len(shape_ids) and shape_ids:
            sol_pct = (
                0.0
                if any(value == 0 for value in sol_by_shape.values())
                else math.exp(statistics.fmean(math.log(value) for value in sol_by_shape.values()))
            )
        else:
            sol_pct = None
        return {
            "correct": correct,
            "correctness": merge_correctness_summaries(
                [value for row in selected if isinstance((value := row.get("correctness")), dict)],
                passed=correct,
            ),
            "latency_us": latency_us,
            "latency_us_by_shape": latency_by_shape,
            "sol_pct": sol_pct,
            "sol_pct_by_shape": sol_by_shape,
        }

    @staticmethod
    def _aggregate_shape_values(
        rows: list[dict[str, object]],
        field: str,
        shape_ids: list[str],
        repeats: int,
        *,
        allow_zero: bool = False,
    ) -> dict[str, float]:
        aggregate: dict[str, float] = {}
        for shape_id in shape_ids:
            values: list[float] = []
            for row in rows:
                raw = row.get(field)
                value = raw.get(shape_id) if isinstance(raw, dict) else None
                if (
                    not isinstance(value, (int, float))
                    or isinstance(value, bool)
                    or value < 0
                    or (value == 0 and not allow_zero)
                    or not math.isfinite(float(value))
                ):
                    break
                values.append(float(value))
            if len(values) == repeats:
                aggregate[shape_id] = (
                    0.0
                    if any(value == 0 for value in values)
                    else math.exp(statistics.fmean(math.log(value) for value in values))
                )
        return aggregate

    def _record_runs(
        self,
        rows: list[dict[str, object]],
        incumbent: KernelRevision,
        candidate: KernelRevision,
        purpose: KernelMeasurementPurpose,
        repeats: int,
        result_digest: ArtifactDigest,
        job_ids: str | None,
    ) -> tuple[tuple[KernelMeasurementRun, ...], tuple[KernelMeasurementRun, ...]]:
        grouped: dict[str, list[KernelMeasurementRun]] = {"incumbent": [], "candidate": []}
        revisions = {"incumbent": incumbent, "candidate": candidate}
        for row in rows:
            label = str(row["revision"])
            repeat = int(cast(int, row["repeat"]))
            correct = bool(row["correct"])
            latency_value = row.get("latency_us")
            latency = float(latency_value) if isinstance(latency_value, (int, float)) else None
            run = KernelMeasurementRun(repeat, correct, latency if correct else None)
            grouped[label].append(run)
            self._journal.record_kernel_measurement(
                KernelMeasurement(
                    id=uuid4().hex,
                    kernel_revision_id=revisions[label].id,
                    purpose=purpose,
                    repeat=repeat,
                    correct=correct,
                    latency_us=run.latency_us,
                    gateway_result_digest=result_digest,
                    agate_job_id=job_ids,
                    created_at=self._clock(),
                )
            )
        incumbent_runs = tuple(sorted(grouped["incumbent"], key=lambda run: run.repeat))
        candidate_runs = tuple(sorted(grouped["candidate"], key=lambda run: run.repeat))
        if len(incumbent_runs) != repeats or len(candidate_runs) != repeats:
            raise InfrastructureError("ABBA merged result omitted a scheduled run")
        return incumbent_runs, candidate_runs

    async def _call(self, operation: Callable[[], object]) -> dict[str, JsonValue]:
        return await call_agate_json(
            operation,
            request_error="Agate ABBA request failed",
            invalid_response="Agate ABBA returned invalid JSON",
            non_object_response="Agate ABBA returned a non-object",
        )


def _json_text(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True)


def _positive_runner_number(
    evaluator: dict[str, JsonValue],
    key: str,
    default: float | int,
) -> float | int:
    value = evaluator.get(key, default)
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or value <= 0
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"evaluation runner override {key} must be positive")
    return value


def _parse_remote_payload(
    job: dict[str, JsonValue],
    schedule: list[dict[str, int | str]],
) -> dict[str, JsonValue]:
    result = job.get("result")
    if not isinstance(result, dict):
        raise InfrastructureError("Agate ABBA job has no result object")
    stdout = result.get("stdout")
    if not isinstance(stdout, str):
        raise InfrastructureError("Agate ABBA job has no command stdout")
    raw: object | None = None
    for line in reversed(stdout.splitlines()):
        if line.startswith(abba_remote.RESULT_PREFIX):
            try:
                raw = json.loads(line[len(abba_remote.RESULT_PREFIX) :])
            except json.JSONDecodeError as error:
                raise InfrastructureError("Agate ABBA result sentinel is invalid") from error
            break
    if not isinstance(raw, dict) or raw.get("schema_version") != 1:
        raise InfrastructureError("Agate ABBA result sentinel is missing")
    if raw.get("error"):
        raise InfrastructureError(f"Agate ABBA remote driver failed: {raw['error']}")
    rows = raw.get("runs")
    if not isinstance(rows, list) or len(rows) != len(schedule):
        raise InfrastructureError("Agate ABBA remote schedule is incomplete")
    actual = [
        {"revision": row.get("revision"), "repeat": row.get("repeat")}
        for row in rows
        if isinstance(row, dict)
    ]
    if actual != schedule:
        raise InfrastructureError("Agate ABBA remote schedule differs from the request")
    return cast(dict[str, JsonValue], raw)
