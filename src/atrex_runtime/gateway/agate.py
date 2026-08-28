"""Concrete Adapter for the published Agate Python SDK."""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import sqlite3
import statistics
import threading
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
from typing import Literal, Protocol, cast

import anyio
from pydantic import BaseModel, ConfigDict, Field, SecretStr, TypeAdapter, model_validator

from ..artifacts.local import JsonValue
from ..domain.errors import (
    InfrastructureError,
    InvalidTransitionError,
    UpstreamGatewayError,
)
from ..domain.ids import AttemptId, parse_attempt_id
from ..roofline import strip_roofline_hardware_suffix
from ..sqlite_support import configure_durable_sqlite
from .batched_evaluate import ShapeBatch, ShapeBatchedEvaluateExecutor, ShapeBatchOutcome
from .contract import (
    AgateEvaluationContext,
    AgateEvaluationContextResolver,
    AgateEvaluationContractV1,
)
from .control_models import GatewayOperation
from .private_results import (
    project_candidate_rejection,
    project_compile_job,
    project_private_evaluation,
    project_private_job,
)
from .protocol import EvaluationV2
from .proxy import GatewayAdapterRequest, GatewayAdapterResult
from .repeated_evaluate import (
    aggregate_evaluations,
    repeated_evaluate_result,
    repeated_evaluate_worker_result,
)
from .retrying_client import RETRYABLE_CLIENT_STATUSES, RetryingAgateClient

AGATE_JOB_SCHEMA_VERSION = 2
_JOB_KINDS = frozenset({"eval", "profile", "dev", "compile", "sol", "disassemble"})
_TERMINAL_STATUSES = frozenset({"succeeded", "failed", "cancelled"})
_COMPILE_OPERATIONS = frozenset({GatewayOperation.CHECK, GatewayOperation.DISASSEMBLE})
_JSON_VALUE_ADAPTER: TypeAdapter[JsonValue] = TypeAdapter(JsonValue)


def _worker_view(job: dict[str, JsonValue], operation: GatewayOperation) -> JsonValue | None:
    """Project the jobs whose Agent view carries no evaluation identity.

    `dev` is deliberately absent: its stdout is the Agent's own probe output, which the
    private-field strip would remove.
    """
    if operation is GatewayOperation.PROFILE:
        return project_private_job(job)
    if operation in _COMPILE_OPERATIONS:
        return project_compile_job(job)
    return None


def _nested_infrastructure_error(error: BaseException) -> InfrastructureError | None:
    if isinstance(error, InfrastructureError):
        return error
    if isinstance(error, BaseExceptionGroup):
        for nested in error.exceptions:
            if (found := _nested_infrastructure_error(nested)) is not None:
                return found
    return None


class AgateCandidateRejection(Exception):
    """Candidate request rejected by Agate validation before job creation."""

    def __init__(self, payload: JsonValue) -> None:
        self.payload = payload
        super().__init__("Agate rejected the candidate request")


class AgateClient(Protocol):
    """Published Agate client methods used by the Runtime."""

    def submit_job(self, kind: str, request: dict[str, object]) -> dict[str, object]:
        """Submit one typed Gateway job."""
        ...

    def get_job(
        self,
        job_id: str,
        wait: bool = False,
        timeout: float = 30.0,
        include_spec: bool = False,
    ) -> dict[str, object]:
        """Read or long-poll one Gateway job."""
        ...

    def cancel_job(self, job_id: str) -> dict[str, object]:
        """Request cancellation of one Gateway job."""
        ...

    def list_env(self, force: bool = False) -> list[dict[str, object]]:
        """List selectable Gateway GPU environments."""
        ...

    def get_env(self, gpu: str, force: bool = False) -> dict[str, object]:
        """Read one selectable GPU environment."""
        ...

    def get_capabilities(self, gpu: str, force: bool = False) -> dict[str, object]:
        """Read one environment's frameworks, profilers, and limits."""
        ...

    def health(self) -> bool:
        """Return whether the external Gateway liveness endpoint responds."""
        ...


class AgateRequestBuilder(Protocol):
    """Stable public Agate eval payload builder."""

    def __call__(
        self,
        candidate: str,
        reference: Mapping[str, object],
        gpu: str,
        *,
        name: str,
        spec_fields: Mapping[str, object] | None = None,
        options: Mapping[str, object] | None = None,
        env_vars: Mapping[str, str] | None = None,
        requirements: tuple[str, ...] | None = None,
        deps_mode: str | None = None,
        mode: str | None = None,
        lock_clocks: bool | None = None,
        harness: str | None = None,
        atrex_bench_version: str | None = None,
        runner_overrides: Mapping[str, object] | None = None,
        idempotency_key: str | None = None,
    ) -> dict[str, object]:
        """Build a JSON-ready eval or profile request."""
        ...


class AgateConnectionConfig(BaseModel):
    """Validated deployment-owned connection and wait policy for Agate."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    base_url: str
    auth_mode: Literal["none", "token", "ak_sk"]
    token: SecretStr | None = None
    access_key: SecretStr | None = None
    secret_key: SecretStr | None = None
    http_timeout_s: float = Field(gt=0)
    wait_timeout_s: float = Field(gt=0)

    @model_validator(mode="after")
    def _validate_connection(self) -> AgateConnectionConfig:
        if not self.base_url.startswith(("http://", "https://")):
            raise ValueError("Agate base_url must use HTTP or HTTPS")
        has_token = self.token is not None
        has_ak_sk = self.access_key is not None and self.secret_key is not None
        if (self.access_key is None) is not (self.secret_key is None):
            raise ValueError("Agate access_key and secret_key must be configured together")
        if self.auth_mode == "none" and (has_token or has_ak_sk):
            raise ValueError("Agate no-auth mode cannot include credentials")
        if self.auth_mode == "token" and (not has_token or has_ak_sk):
            raise ValueError("Agate token mode requires only token")
        if self.auth_mode == "ak_sk" and (has_token or not has_ak_sk):
            raise ValueError("Agate ak_sk mode requires only access_key and secret_key")
        return self


def load_agate_sdk(
    config: AgateConnectionConfig,
) -> tuple[AgateClient, AgateRequestBuilder]:
    """Load the published SDK and construct its sync client without exposing credentials."""
    try:
        package = importlib.import_module("atrex_gateway_client")
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "atrex-gateway-client is required for the Agate Gateway adapter"
        ) from error

    client_type = cast(Callable[..., AgateClient], package.__dict__["Client"])
    builder = cast(
        AgateRequestBuilder,
        package.__dict__["build_eval_request_from_content"],
    )
    if config.auth_mode == "none":
        no_auth_type = cast(Callable[[], object], package.__dict__["NoAuth"])
        auth = no_auth_type()
    elif config.auth_mode == "token":
        token_auth_type = cast(Callable[[str], object], package.__dict__["TokenAuth"])
        if config.token is None:
            raise AssertionError("validated token configuration is incomplete")
        auth = token_auth_type(config.token.get_secret_value())
    else:
        ak_sk_auth_type = cast(
            Callable[[str, str], object],
            package.__dict__["AkSkAuth"],
        )
        if config.access_key is None or config.secret_key is None:
            raise AssertionError("validated AK/SK configuration is incomplete")
        auth = ak_sk_auth_type(
            config.access_key.get_secret_value(),
            config.secret_key.get_secret_value(),
        )
    client = client_type(config.base_url, auth=auth, timeout=config.http_timeout_s)
    return cast(AgateClient, RetryingAgateClient(client)), builder


@dataclass(frozen=True, slots=True)
class AgateJobBinding:
    """Durable ownership of one external Agate job."""

    job_id: str
    attempt_id: AttemptId
    idempotency_key: str
    kind: Literal["eval", "profile", "dev", "compile", "sol", "disassemble"]
    operation: GatewayOperation = GatewayOperation.EVALUATE


class SqliteAgateJobStore:
    """Persist external job ownership so Worker poll and cancel remain Attempt-scoped."""

    def __init__(self, path: str | Path) -> None:
        database_path = Path(path)
        database_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._connection = sqlite3.connect(str(database_path), isolation_level=None)
        try:
            self._connection.row_factory = sqlite3.Row
            configure_durable_sqlite(self._connection)
            self._lock = threading.RLock()
            self._migrate()
            database_path.chmod(0o600)
        except BaseException:
            self._connection.close()
            raise

    def close(self) -> None:
        """Close the job binding database after request handling has stopped."""
        with self._lock:
            self._connection.close()

    def check_health(self) -> None:
        """Verify that the Agate job store can read and acquire a write transaction."""
        with self._transaction() as connection:
            connection.execute("SELECT 1").fetchone()

    def bind(self, binding: AgateJobBinding) -> AgateJobBinding:
        """Record an idempotent Attempt request to external job mapping."""
        if not binding.job_id:
            raise ValueError("Agate job_id cannot be empty")
        with self._transaction() as connection:
            by_request = connection.execute(
                "SELECT * FROM agate_jobs WHERE attempt_id = ? AND idempotency_key = ?",
                (binding.attempt_id, binding.idempotency_key),
            ).fetchone()
            by_job = connection.execute(
                "SELECT * FROM agate_jobs WHERE job_id = ?", (binding.job_id,)
            ).fetchone()
            for row in (by_request, by_job):
                if row is not None and self._row_binding(row) != binding:
                    raise InvalidTransitionError("Agate job binding conflicts with durable state")
            if by_request is None and by_job is None:
                connection.execute(
                    "INSERT INTO agate_jobs VALUES (?, ?, ?, ?, ?)",
                    (
                        binding.job_id,
                        binding.attempt_id,
                        binding.idempotency_key,
                        binding.kind,
                        binding.operation.value,
                    ),
                )
        return binding

    def require_owned(self, attempt_id: AttemptId, job_id: str) -> AgateJobBinding:
        """Return a job only when it was submitted by the same Attempt."""
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM agate_jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
        if row is None or row["attempt_id"] != attempt_id:
            raise PermissionError("Agate job is not owned by this Attempt")
        return self._row_binding(row)

    def list_owned(
        self,
        attempt_id: AttemptId,
        *,
        kind: str | None = None,
        limit: int = 50,
    ) -> tuple[AgateJobBinding, ...]:
        """List newest persisted jobs created through one Attempt capability."""
        if limit <= 0:
            raise ValueError("Agate job list limit must be positive")
        query = "SELECT * FROM agate_jobs WHERE attempt_id = ?"
        parameters: list[object] = [attempt_id]
        if kind is not None:
            if kind not in _JOB_KINDS:
                raise ValueError(f"unsupported Agate job kind: {kind}")
            query += " AND kind = ?"
            parameters.append(kind)
        query += " ORDER BY rowid DESC LIMIT ?"
        parameters.append(limit)
        with self._lock:
            rows = self._connection.execute(query, parameters).fetchall()
        return tuple(self._row_binding(row) for row in rows)

    @staticmethod
    def _row_binding(row: sqlite3.Row) -> AgateJobBinding:
        kind = row["kind"]
        if kind not in _JOB_KINDS:
            raise RuntimeError(f"invalid persisted Agate job kind: {kind!r}")
        try:
            operation = GatewayOperation(str(row["operation"]))
        except ValueError as error:
            invalid_operation = row["operation"]
            raise RuntimeError(
                f"invalid persisted Agate operation: {invalid_operation!r}"
            ) from error
        return AgateJobBinding(
            job_id=str(row["job_id"]),
            attempt_id=parse_attempt_id(str(row["attempt_id"])),
            idempotency_key=str(row["idempotency_key"]),
            kind=cast(
                Literal["eval", "profile", "dev", "compile", "sol", "disassemble"],
                kind,
            ),
            operation=operation,
        )

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                yield self._connection
            except BaseException:
                self._connection.rollback()
                raise
            else:
                self._connection.commit()

    def _migrate(self) -> None:
        with self._transaction() as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY, value INTEGER NOT NULL)"
            )
            row = connection.execute(
                "SELECT value FROM metadata WHERE key = 'schema_version'"
            ).fetchone()
            if row is None:
                connection.execute(
                    """
                    CREATE TABLE agate_jobs(
                        job_id TEXT PRIMARY KEY,
                        attempt_id TEXT NOT NULL,
                        idempotency_key TEXT NOT NULL,
                        kind TEXT NOT NULL CHECK(
                            kind IN ('eval', 'profile', 'dev', 'compile', 'sol', 'disassemble')
                        ),
                        operation TEXT NOT NULL,
                        UNIQUE(attempt_id, idempotency_key)
                    )
                    """
                )
                connection.execute(
                    "INSERT INTO metadata(key, value) VALUES ('schema_version', ?)",
                    (AGATE_JOB_SCHEMA_VERSION,),
                )
            elif row["value"] != AGATE_JOB_SCHEMA_VERSION:
                raise RuntimeError(f"unsupported Agate job schema version: {row['value']}")


class AgateGatewayAdapter:
    """Translate trusted Runtime operations to the synchronous Agate SDK."""

    def __init__(
        self,
        client: AgateClient,
        request_builder: AgateRequestBuilder,
        contexts: AgateEvaluationContextResolver,
        jobs: SqliteAgateJobStore,
        *,
        wait_timeout_s: float,
        optimizer_evaluate_repeats: int = 1,
        optimizer_correctness_cases: int = 1,
        optimizer_bench_iters: int = 100,
        profile_without_roofline: bool = False,
        connection_summary: Mapping[str, str] | None = None,
    ) -> None:
        if wait_timeout_s <= 0:
            raise ValueError("Agate wait timeout must be positive")
        if optimizer_evaluate_repeats <= 0:
            raise ValueError("Optimizer Evaluate repeats must be positive")
        if optimizer_correctness_cases <= 0 or optimizer_bench_iters <= 0:
            raise ValueError("Optimizer Gate sampling values must be positive")
        self._client = client
        self._request_builder = request_builder
        self._contexts = contexts
        self._jobs = jobs
        self._wait_timeout_s = wait_timeout_s
        self._optimizer_evaluate_repeats = optimizer_evaluate_repeats
        self._optimizer_correctness_cases = optimizer_correctness_cases
        self._optimizer_bench_iters = optimizer_bench_iters
        self._profile_without_roofline = profile_without_roofline
        self._connection_summary = dict(connection_summary or {"url": "managed", "auth": "managed"})
        self._shape_batches = ShapeBatchedEvaluateExecutor()

    async def execute(self, request: GatewayAdapterRequest) -> GatewayAdapterResult:
        """Execute one capability-authorized Agate command equivalent."""
        if request.operation in {
            GatewayOperation.EVALUATE,
            GatewayOperation.PROFILE,
            GatewayOperation.DEV,
            GatewayOperation.CHECK,
            GatewayOperation.DISASSEMBLE,
        }:
            return await self._submit(request)
        if request.operation is GatewayOperation.POLL:
            return await self._poll(request)
        if request.operation is GatewayOperation.CANCEL:
            return await self._cancel(request)
        if request.operation is GatewayOperation.JOBS:
            return await self._list_jobs(request)
        if request.operation is GatewayOperation.ENV:
            return await self._environment(request)
        if request.operation is GatewayOperation.HEALTH:
            ok = await anyio.to_thread.run_sync(self._client.health)
            return GatewayAdapterResult("completed", {"ok": ok})
        if request.operation is GatewayOperation.CONFIG:
            summary: dict[str, JsonValue] = dict(self._connection_summary)
            return GatewayAdapterResult("completed", summary)
        raise AssertionError(f"unsupported Gateway operation: {request.operation}")

    async def _submit(self, request: GatewayAdapterRequest) -> GatewayAdapterResult:
        context = self._contexts.resolve(request.attempt_id)
        payload = self._build_request(request, context)
        kind_by_operation: dict[
            GatewayOperation,
            Literal["eval", "profile", "dev", "compile", "sol", "disassemble"],
        ] = {
            GatewayOperation.EVALUATE: "eval",
            GatewayOperation.PROFILE: "profile",
            GatewayOperation.DEV: "dev",
            GatewayOperation.CHECK: "compile",
            GatewayOperation.DISASSEMBLE: "disassemble",
        }
        kind = kind_by_operation[request.operation]
        if request.operation is GatewayOperation.EVALUATE:
            if self._optimizer_evaluate_repeats > 1:
                mapped = await self._submit_repeated_evaluate(request, context)
            else:
                mapped = await self._submit_batched_evaluate(
                    request,
                    context,
                    idempotency_key=request.idempotency_key,
                )
        else:
            mapped = await self._submit_once(
                request,
                context,
                payload,
                kind,
                binding_key=request.idempotency_key,
            )
        if request.operation is GatewayOperation.PROFILE:
            mapped = self._label_profile_shape(
                mapped,
                self._profile_shape_id(context.contract, request.parameters),
            )
        if (
            request.operation is GatewayOperation.EVALUATE
            and mapped.evaluation is not None
            and mapped.evaluation.correct
            and context.contract.roofline is None
            and self._profile_without_roofline
        ):
            return GatewayAdapterResult(
                mapped.status,
                mapped.result,
                mapped.job_id,
                mapped.evaluation,
                await self._profile_evaluation(request, payload),
                mapped.worker_result,
            )
        return mapped

    async def _submit_once(
        self,
        request: GatewayAdapterRequest,
        context: AgateEvaluationContext,
        payload: dict[str, object],
        kind: Literal["eval", "profile", "dev", "compile", "sol", "disassemble"],
        *,
        binding_key: str,
        expected_shape_ids: tuple[str, ...] | None = None,
    ) -> GatewayAdapterResult:
        """Submit and await one concrete Agate job."""
        try:
            acceptance = await self._call(
                lambda: self._client.submit_job(kind, payload),
                validation_is_candidate=True,
            )
        except AgateCandidateRejection as rejection:
            rejected: JsonValue = {"status": "rejected", "error": rejection.payload}
            return GatewayAdapterResult(
                status="completed",
                result=rejected,
                evaluation=(
                    EvaluationV2(correct=False, latency_us=None)
                    if request.operation is GatewayOperation.EVALUATE
                    else None
                ),
                worker_result=project_candidate_rejection(rejection.payload),
            )
        job_id = acceptance.get("job_id")
        if not isinstance(job_id, str) or not job_id:
            raise InfrastructureError("Agate acceptance did not contain a valid job_id")
        binding = self._jobs.bind(
            AgateJobBinding(
                job_id,
                request.attempt_id,
                binding_key,
                kind,
                request.operation,
            )
        )
        job = await self._call(
            lambda: self._client.get_job(
                job_id,
                wait=True,
                timeout=self._wait_timeout_s,
            )
        )
        expected = expected_shape_ids
        if expected is None and binding.operation is GatewayOperation.EVALUATE:
            expected = tuple(context.contract.shapes)
        return self._map_job(job, binding.operation, expected)

    async def _submit_batched_evaluate(
        self,
        request: GatewayAdapterRequest,
        context: AgateEvaluationContext,
        *,
        idempotency_key: str,
    ) -> GatewayAdapterResult:
        """Execute one logical Optimizer Eval through shared Shape batches."""

        async def evaluate_batch(batch: ShapeBatch) -> ShapeBatchOutcome:
            payload = self._build_request(
                request,
                context,
                contract_override=batch.contract,
                idempotency_key=batch.idempotency_key,
            )
            mapped = await self._submit_once(
                request,
                context,
                payload,
                "eval",
                binding_key=batch.idempotency_key,
                expected_shape_ids=batch.shape_ids,
            )
            if mapped.status != "completed" or mapped.evaluation is None:
                raise InfrastructureError("Agate Eval batch did not complete")
            return ShapeBatchOutcome(
                mapped.result,
                mapped.evaluation,
                mapped.job_id,
                mapped.worker_result,
            )

        try:
            result = await self._shape_batches.run(
                context.contract,
                idempotency_key,
                evaluate_batch,
            )
        except BaseExceptionGroup as errors:
            infrastructure = _nested_infrastructure_error(errors)
            if infrastructure is not None:
                raise infrastructure from errors
            raise
        return GatewayAdapterResult(
            status="completed",
            result=result.job,
            job_id=result.job_id,
            evaluation=result.evaluation,
            worker_result=result.worker_result,
        )

    async def _submit_repeated_evaluate(
        self,
        request: GatewayAdapterRequest,
        context: AgateEvaluationContext,
    ) -> GatewayAdapterResult:
        """Run independent ordinary Eval jobs concurrently and average their latencies."""
        results: list[GatewayAdapterResult | None] = [
            None for _ in range(self._optimizer_evaluate_repeats)
        ]

        async def run_one(repeat: int) -> None:
            digest = hashlib.sha256(
                f"{request.attempt_id}:{request.idempotency_key}:{repeat}".encode()
            ).hexdigest()
            binding_key = f"ordinary-evaluate-repeat:{digest}"
            results[repeat] = await self._submit_batched_evaluate(
                request,
                context,
                idempotency_key=binding_key,
            )

        async with anyio.create_task_group() as tasks:
            for repeat in range(self._optimizer_evaluate_repeats):
                tasks.start_soon(run_one, repeat)
        completed = tuple(result for result in results if result is not None)
        if len(completed) != self._optimizer_evaluate_repeats:
            raise AssertionError("ordinary Evaluate repetition did not produce a result")
        if any(result.status != "completed" or result.evaluation is None for result in completed):
            raise InfrastructureError("ordinary Agate Eval repetition did not complete")
        evaluations = tuple(
            result.evaluation for result in completed if result.evaluation is not None
        )
        aggregate = aggregate_evaluations(evaluations)
        return GatewayAdapterResult(
            status="completed",
            result=repeated_evaluate_result(
                tuple(result.result for result in completed),
                evaluations,
            ),
            evaluation=aggregate,
            worker_result=repeated_evaluate_worker_result(evaluations),
        )

    async def _profile_evaluation(
        self,
        request: GatewayAdapterRequest,
        evaluation_payload: dict[str, object],
    ) -> JsonValue:
        """Attach best-effort NCU SOL evidence to one correct Agent Eval."""
        request_key_digest = hashlib.sha256(request.idempotency_key.encode()).hexdigest()
        profile_key = f"profile:{request.attempt_id}:{request_key_digest}"
        payload = dict(evaluation_payload)
        payload.update(
            {
                "idempotency_key": profile_key,
                "level": "sol",
                "top_kernels": 10,
            }
        )
        try:
            acceptance = await self._call(
                lambda: self._client.submit_job("profile", payload),
                validation_is_candidate=True,
            )
            job_id = acceptance.get("job_id")
            if not isinstance(job_id, str) or not job_id:
                raise InfrastructureError("Agate Profile acceptance has no job_id")
            self._jobs.bind(
                AgateJobBinding(
                    job_id,
                    request.attempt_id,
                    profile_key,
                    "profile",
                    GatewayOperation.PROFILE,
                )
            )
            job = await self._call(
                lambda: self._client.get_job(
                    job_id,
                    wait=True,
                    timeout=self._wait_timeout_s,
                )
            )
            if job.get("status") not in _TERMINAL_STATUSES:
                raise InfrastructureError("Agate Profile did not reach a terminal state")
            return job
        except Exception as error:
            return {
                "status": "failed",
                "error": {"message": f"{type(error).__name__}: {error}"[:1000]},
            }

    async def _poll(self, request: GatewayAdapterRequest) -> GatewayAdapterResult:
        if request.job_id is None:
            raise ValueError("poll requires job_id")
        binding = self._jobs.require_owned(request.attempt_id, request.job_id)
        wait = bool(request.parameters.get("wait", False))
        include_spec = bool(request.parameters.get("include_spec", False))
        job = await self._call(
            lambda: self._client.get_job(
                request.job_id or "",
                wait=wait,
                timeout=self._wait_timeout_s if wait else 30.0,
                include_spec=include_spec,
            )
        )
        expected_shape_ids = (
            tuple(self._contexts.resolve(request.attempt_id).contract.shapes)
            if binding.operation is GatewayOperation.EVALUATE
            else None
        )
        return self._map_job(job, binding.operation, expected_shape_ids)

    async def _cancel(self, request: GatewayAdapterRequest) -> GatewayAdapterResult:
        if request.job_id is None:
            raise ValueError("cancel requires job_id")
        binding = self._jobs.require_owned(request.attempt_id, request.job_id)
        job = await self._call(lambda: self._client.cancel_job(request.job_id or ""))
        return self._map_job(job, binding.operation)

    async def _list_jobs(self, request: GatewayAdapterRequest) -> GatewayAdapterResult:
        kind_value = request.parameters.get("kind")
        kind = kind_value if isinstance(kind_value, str) else None
        limit_value = request.parameters.get("limit", 50)
        if not isinstance(limit_value, int):
            raise ValueError("jobs limit must be an integer")
        status_value = request.parameters.get("status")
        status = status_value if isinstance(status_value, str) else None
        bindings = self._jobs.list_owned(request.attempt_id, kind=kind, limit=limit_value)
        rows: list[JsonValue] = []
        for binding in bindings:
            job = await self._call(partial(self._client.get_job, binding.job_id))
            if status is None or job.get("status") == status:
                expected_shape_ids = (
                    tuple(self._contexts.resolve(request.attempt_id).contract.shapes)
                    if binding.operation is GatewayOperation.EVALUATE
                    else None
                )
                mapped = self._map_job(job, binding.operation, expected_shape_ids)
                if mapped.worker_result is None:
                    rows.append(mapped.result)
                    continue
                projected = mapped.worker_result
                if isinstance(projected, dict):
                    projected = {
                        "job_id": binding.job_id,
                        "operation": binding.operation.value,
                        **projected,
                    }
                rows.append(projected)
        return GatewayAdapterResult("completed", {"jobs": rows[:limit_value]})

    async def _environment(self, request: GatewayAdapterRequest) -> GatewayAdapterResult:
        force = bool(request.parameters.get("force", False))
        gpu_value = request.parameters.get("gpu")
        gpu = gpu_value if isinstance(gpu_value, str) else None
        capabilities = bool(request.parameters.get("capabilities", False))
        if gpu is None:
            value = await self._call_json(lambda: self._client.list_env(force=force))
            return GatewayAdapterResult("completed", {"env": value})
        if capabilities:
            value = await self._call(lambda: self._client.get_capabilities(gpu, force=force))
        else:
            value = await self._call(lambda: self._client.get_env(gpu, force=force))
        return GatewayAdapterResult("completed", value)

    def _build_request(
        self,
        request: GatewayAdapterRequest,
        context: AgateEvaluationContext,
        *,
        contract_override: AgateEvaluationContractV1 | None = None,
        idempotency_key: str | None = None,
    ) -> dict[str, object]:
        if request.candidate_path is None:
            raise ValueError(f"{request.operation.value} requires a candidate")
        if request.operation is GatewayOperation.DEV:
            return self._build_dev(request, context)
        contract = contract_override or context.contract
        candidate = request.candidate_path.joinpath(*contract.candidate_path.split("/"))
        if candidate.is_symlink() or not candidate.is_file():
            raise ValueError(f"candidate file is missing: {contract.candidate_path}")
        try:
            candidate_source = candidate.read_text(encoding="utf-8")
        except UnicodeDecodeError as error:
            raise ValueError("candidate source must be UTF-8 text") from error

        if request.operation is GatewayOperation.CHECK:
            return self._build_source_job(
                request,
                context,
                candidate_source,
                "check",
                init_kwargs=self._private_constructor_kwargs(context.contract),
            )
        if request.operation is GatewayOperation.DISASSEMBLE:
            return self._build_source_job(
                request,
                context,
                candidate_source,
                "disassemble",
                init_kwargs=self._private_constructor_kwargs(context.contract),
            )

        shapes = contract.shapes
        metadata = contract.metadata
        roofline = contract.roofline
        if request.operation is GatewayOperation.PROFILE:
            shapes, metadata, roofline = self._private_profile_inputs(contract, request.parameters)
        reference: dict[str, object] = {
            "operator": context.operator,
            "reference_py": contract.reference_py,
            "input_py": contract.input_py,
            "shapes": shapes,
        }
        if metadata is not None:
            reference["metadata"] = metadata
        if roofline is not None:
            reference["roofline"] = strip_roofline_hardware_suffix(roofline)
        options = contract.options
        if request.operation is GatewayOperation.EVALUATE:
            options = options.model_copy(
                update={
                    "num_correctness_cases": self._optimizer_correctness_cases,
                    "bench_iters": self._optimizer_bench_iters,
                }
            )
        payload = self._request_builder(
            candidate_source,
            reference,
            context.agate_gpu,
            name=f"{context.operator}_{request.attempt_id}",
            spec_fields={"languages": [context.dsl.value]},
            options=cast(Mapping[str, object], options.model_dump(mode="python")),
            env_vars=contract.env_vars or None,
            requirements=contract.requirements or None,
            deps_mode=contract.deps_mode,
            mode=contract.mode,
            lock_clocks=contract.lock_clocks,
            harness=contract.harness,
            atrex_bench_version=contract.atrex_bench_version,
            runner_overrides=cast(Mapping[str, object], contract.runner_overrides) or None,
            idempotency_key=idempotency_key or request.idempotency_key,
        )
        if request.operation is GatewayOperation.PROFILE:
            if request.profile_level is None:
                raise ValueError("profile requires level")
            payload["level"] = request.profile_level
            if request.kernel_regex is not None:
                payload["kernel_regex"] = request.kernel_regex
            for source, target in (
                ("profiler", "profiler"),
                ("counters", "counters"),
                ("kernel_name", "kernel_name"),
                ("source", "source"),
                ("launch_skip", "launch_skip"),
                ("launch_count", "launch_count"),
                ("top_kernels", "top_kernels"),
            ):
                value = request.parameters.get(source)
                if value not in (None, False, [], ()):
                    payload[target] = value
            self._apply_dependencies(payload, request.parameters)
        return payload

    @staticmethod
    def _profile_shape_id(
        contract: AgateEvaluationContractV1,
        parameters: Mapping[str, JsonValue],
    ) -> str:
        requested = parameters.get("shape_id")
        shape_id = requested if isinstance(requested, str) else sorted(contract.shapes)[0]
        if shape_id not in contract.shapes:
            raise ValueError("profile shape_id is not an evaluator-owned opaque id")
        return shape_id

    @staticmethod
    def _label_profile_shape(
        result: GatewayAdapterResult,
        shape_id: str,
    ) -> GatewayAdapterResult:
        worker = result.worker_result
        if not isinstance(worker, dict):
            return result
        payload = worker.get("result")
        if not isinstance(payload, dict):
            return result
        labeled_payload = {**payload, "shape_id": shape_id}
        return replace(result, worker_result={**worker, "result": labeled_payload})

    @staticmethod
    def _private_profile_inputs(
        contract: AgateEvaluationContractV1,
        parameters: Mapping[str, JsonValue],
    ) -> tuple[
        dict[str, JsonValue],
        dict[str, JsonValue] | None,
        dict[str, JsonValue] | None,
    ]:
        """Select one opaque private case for profiling without exposing its values."""
        shape_id = AgateGatewayAdapter._profile_shape_id(contract, parameters)

        def subset(
            value: dict[str, JsonValue] | None,
            *,
            metadata: bool,
        ) -> dict[str, JsonValue] | None:
            if value is None:
                return None
            result = dict(value)
            per_shape = result.get("shapes")
            if isinstance(per_shape, dict):
                result["shapes"] = {shape_id: per_shape[shape_id]} if shape_id in per_shape else {}
            if metadata and "num_shapes" in result:
                result["num_shapes"] = 1
            return result

        return (
            {shape_id: contract.shapes[shape_id]},
            subset(contract.metadata, metadata=True),
            subset(contract.roofline, metadata=False),
        )

    def _build_dev(
        self,
        request: GatewayAdapterRequest,
        context: AgateEvaluationContext,
    ) -> dict[str, object]:
        if request.candidate_path is None:
            raise ValueError("dev requires a candidate bundle")
        files: dict[str, str] = {}
        for path in sorted(request.candidate_path.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(request.candidate_path).as_posix()
            try:
                files[relative] = path.read_text(encoding="utf-8")
            except UnicodeDecodeError as error:
                raise ValueError(f"dev inline file must be UTF-8 text: {relative}") from error
        command = request.parameters.get("command")
        if not isinstance(command, str) or not command:
            raise ValueError("dev requires command")
        extra = request.parameters.get("files")
        if extra is not None:
            if not isinstance(extra, Mapping):
                raise ValueError("dev files must be a mapping")
            for relative, content in extra.items():
                if not isinstance(relative, str) or not isinstance(content, str):
                    raise ValueError("dev file paths and contents must be strings")
                files.setdefault(relative, content)
        payload: dict[str, object] = {
            "spec": {"target_hardware": [context.agate_gpu]},
            "command": command,
            "env_vars": request.parameters.get("env_vars", {}),
            "files": files,
            "recycle": request.parameters.get("recycle", True),
        }
        for source, target in (
            ("job_timeout_s", "timeout_s"),
            ("intent", "dev_intent"),
            ("note", "dev_note"),
        ):
            value = request.parameters.get(source)
            if value is not None:
                payload[target] = value
        return payload

    def _build_source_job(
        self,
        request: GatewayAdapterRequest,
        context: AgateEvaluationContext,
        candidate_source: str,
        job: Literal["check", "disassemble"],
        *,
        init_kwargs: dict[str, JsonValue],
    ) -> dict[str, object]:
        payload: dict[str, object] = {
            "spec": {"target_hardware": [context.agate_gpu]},
            "candidate": candidate_source,
            "env_vars": request.parameters.get("env_vars", {}),
        }
        self._apply_dependencies(payload, request.parameters)
        # Both jobs construct Model before compiling, so a parameterized operator needs
        # its constructor arguments either way.
        payload["init_kwargs"] = init_kwargs
        if job == "check":
            for key in ("arch", "sanitize"):
                value = request.parameters.get(key)
                if value is not None:
                    payload[key] = value
        else:
            payload["fmt"] = request.parameters.get("fmt", "auto")
        return payload

    @staticmethod
    def _private_constructor_kwargs(
        contract: AgateEvaluationContractV1,
    ) -> dict[str, JsonValue]:
        """Resolve opaque Shape constructor arguments for an Agate compile job.

        Some operators declare `init_kwargs` on later Shapes only, so the first
        Shape that carries them wins rather than whichever Shape sorts first.
        """
        resolved: dict[str, JsonValue] | None = None
        for shape_id in sorted(contract.shapes):
            shape = contract.shapes[shape_id]
            if not isinstance(shape, dict):
                raise ValueError("check Shape must be an object")
            init_kwargs = shape.get("init_kwargs")
            if init_kwargs is None:
                continue
            if not isinstance(init_kwargs, dict):
                raise ValueError("check Shape init_kwargs must be an object or null")
            if resolved is None:
                resolved = dict(init_kwargs)
        return {} if resolved is None else resolved

    @staticmethod
    def _apply_dependencies(
        payload: dict[str, object],
        parameters: Mapping[str, JsonValue],
    ) -> None:
        env_vars = parameters.get("env_vars")
        if isinstance(env_vars, dict) and env_vars:
            payload["env_vars"] = env_vars
        requirements = parameters.get("requirements")
        if isinstance(requirements, (list, tuple)) and requirements:
            payload["requirements"] = list(requirements)
        deps_mode = parameters.get("deps_mode")
        if isinstance(deps_mode, str):
            payload["deps_mode"] = deps_mode

    @staticmethod
    def _candidate_file(request: GatewayAdapterRequest, relative: str) -> Path:
        if request.candidate_path is None:
            raise ValueError(f"{request.operation.value} requires a candidate bundle")
        path = request.candidate_path.joinpath(*relative.split("/"))
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"candidate file is missing: {relative}")
        return path

    def _read_json_object(
        self,
        request: GatewayAdapterRequest,
        relative: str,
        label: str,
    ) -> dict[str, JsonValue]:
        path = self._candidate_file(request, relative)
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"{label} must be a UTF-8 JSON object") from error
        normalized = _JSON_VALUE_ADAPTER.validate_python(value)
        if not isinstance(normalized, dict):
            raise ValueError(f"{label} must be a JSON object")
        return normalized

    async def _call(
        self,
        operation: Callable[[], dict[str, object]],
        *,
        validation_is_candidate: bool = False,
    ) -> dict[str, JsonValue]:
        normalized = await self._call_json(
            operation,
            validation_is_candidate=validation_is_candidate,
        )
        if not isinstance(normalized, dict):
            raise InfrastructureError("Agate SDK returned a non-object response")
        return normalized

    async def _call_json(
        self,
        operation: Callable[[], object],
        *,
        validation_is_candidate: bool = False,
    ) -> JsonValue:
        try:
            value = await anyio.to_thread.run_sync(operation)
        except Exception as error:
            error_fields = vars(error)
            if (
                validation_is_candidate
                and error_fields.get("status") in {400, 422}
                and error_fields.get("error_class") == "validation"
            ):
                try:
                    payload = _JSON_VALUE_ADAPTER.validate_python(error_fields.get("payload"))
                except ValueError:
                    pass
                else:
                    raise AgateCandidateRejection(payload) from error
            status = error_fields.get("status")
            if isinstance(status, int) and not isinstance(status, bool):
                message = f"Agate rejected the request: {error}"
                if 400 <= status < 500 and status not in RETRYABLE_CLIENT_STATUSES:
                    raise ValueError(message) from error
                raise UpstreamGatewayError(status, message) from error
            raise InfrastructureError(
                f"Agate SDK request failed: {type(error).__name__}: {error}"
            ) from error
        try:
            normalized = _JSON_VALUE_ADAPTER.validate_python(value)
        except ValueError as error:
            raise InfrastructureError("Agate SDK returned invalid JSON data") from error
        return normalized

    @staticmethod
    def _map_job(
        job: dict[str, JsonValue],
        operation: GatewayOperation,
        expected_shape_ids: tuple[str, ...] | None = None,
    ) -> GatewayAdapterResult:
        status = job.get("status")
        job_id_value = job.get("job_id")
        job_id = job_id_value if isinstance(job_id_value, str) else None
        if operation is GatewayOperation.EVALUATE and expected_shape_ids is not None:
            expected_shape_ids = _reported_shape_ids(job, expected_shape_ids)
        if status not in _TERMINAL_STATUSES:
            if status not in {"queued", "running"}:
                raise InfrastructureError(f"Agate returned an unknown job status: {status!r}")
            return GatewayAdapterResult(
                status="queued",
                result=job,
                job_id=job_id,
                worker_result=(
                    {
                        "status": status,
                        "hidden_case_details": "shape inputs and failure details withheld",
                    }
                    if operation is GatewayOperation.EVALUATE
                    else _worker_view(job, operation)
                ),
            )
        if status == "cancelled":
            return GatewayAdapterResult(
                status="cancelled",
                result=job,
                job_id=job_id,
                worker_result=(
                    {
                        "status": "cancelled",
                        "hidden_case_details": "shape inputs and failure details withheld",
                    }
                    if operation is GatewayOperation.EVALUATE
                    else _worker_view(job, operation)
                ),
            )
        if status == "failed":
            if operation is GatewayOperation.EVALUATE:
                return GatewayAdapterResult(
                    status="failed",
                    result=job,
                    job_id=job_id,
                    worker_result=project_private_job(job),
                )
            return GatewayAdapterResult(
                status="failed",
                result=job,
                job_id=job_id,
                worker_result=_worker_view(job, operation),
            )
        if operation is GatewayOperation.EVALUATE and expected_shape_ids is None:
            raise InfrastructureError("eval result mapping requires expected shape ids")
        evaluation = (
            parse_agate_evaluation(job, expected_shape_ids or ())
            if operation is GatewayOperation.EVALUATE
            else None
        )
        return GatewayAdapterResult(
            status="completed",
            result=job,
            job_id=job_id,
            evaluation=evaluation,
            worker_result=(
                project_private_evaluation(job, evaluation, expected_shape_ids or ())
                if operation is GatewayOperation.EVALUATE and evaluation is not None
                else _worker_view(job, operation)
            ),
        )


def parse_agate_evaluation(
    job: dict[str, JsonValue],
    expected_shape_ids: tuple[str, ...],
) -> EvaluationV2:
    """Map the current Atrex-Bench result and its normalized compatibility form."""
    result = job.get("result")
    if not isinstance(result, dict):
        raise InfrastructureError("successful Agate eval has no structured result")

    all_pass = result.get("all_pass")
    if isinstance(all_pass, bool):
        if not all_pass:
            return EvaluationV2(correct=False, latency_us=None)
        latency = _positive_number(result.get("latency_us_geomean"))
        if latency is None:
            raise InfrastructureError("passing Agate eval has no positive aggregate latency")
        return EvaluationV2(correct=True, latency_us=latency)

    passed = result.get("passed")
    correctness = result.get("correctness")
    performance = result.get("performance")
    if not expected_shape_ids:
        raise InfrastructureError("evaluation contract has no expected shape ids")
    if not isinstance(passed, dict):
        raise InfrastructureError("Agate eval result does not match the Atrex-Bench schema")
    if result.get("error") is not None:
        return EvaluationV2(correct=False, latency_us=None)
    if not _all_compile_passed(passed.get("compile"), expected_shape_ids):
        return EvaluationV2(correct=False, latency_us=None)
    if not _all_correctness_passed(passed.get("correctness"), expected_shape_ids):
        return EvaluationV2(correct=False, latency_us=None)
    if not isinstance(correctness, dict):
        raise InfrastructureError("passing Agate eval has no correctness results")
    correctness_shapes = correctness.get("shapes")
    if not isinstance(correctness_shapes, dict) or any(
        shape_id not in correctness_shapes for shape_id in expected_shape_ids
    ):
        raise InfrastructureError("passing Agate eval is missing expected correctness shapes")
    latency = _aggregate_latency(performance, expected_shape_ids)
    if latency is None:
        return EvaluationV2(correct=False, latency_us=None)
    return EvaluationV2(correct=True, latency_us=latency)


def _reported_shape_ids(
    job: dict[str, JsonValue],
    fallback: tuple[str, ...],
) -> tuple[str, ...]:
    """Recover opaque IDs from one physical batch for later jobs/poll inspection."""
    result = job.get("result")
    if not isinstance(result, dict):
        return fallback
    direct = result.get("latency_us_by_shape")
    candidates: object = direct
    if not isinstance(candidates, dict):
        performance = result.get("performance")
        candidates = performance.get("shapes") if isinstance(performance, dict) else None
    if not isinstance(candidates, dict):
        correctness = result.get("correctness")
        candidates = correctness.get("shapes") if isinstance(correctness, dict) else None
    if not isinstance(candidates, dict) or not candidates:
        return fallback
    return tuple(
        sorted(
            candidates,
            key=lambda value: (0, int(value)) if value.isdigit() else (1, value),
        )
    )


def _all_compile_passed(value: JsonValue | None, shape_ids: tuple[str, ...]) -> bool:
    if not isinstance(value, dict):
        return False
    aggregate = value.get("status")
    if aggregate is not None:
        return aggregate == "passed"
    return all(_status_passed(value.get(shape_id)) for shape_id in shape_ids)


def _all_correctness_passed(value: JsonValue | None, shape_ids: tuple[str, ...]) -> bool:
    if not isinstance(value, dict):
        return False
    return all(_status_passed(value.get(shape_id)) for shape_id in shape_ids)


def _status_passed(value: JsonValue | None) -> bool:
    return isinstance(value, dict) and value.get("status") == "passed"


def _aggregate_latency(value: JsonValue | None, shape_ids: tuple[str, ...]) -> float | None:
    if not isinstance(value, dict):
        return None
    shapes = value.get("shapes")
    if not isinstance(shapes, dict):
        return None
    latencies: list[float] = []
    for shape_id in shape_ids:
        shape = shapes.get(shape_id)
        if not isinstance(shape, dict) or shape.get("error") is not None:
            return None
        samples = shape.get("samples")
        if not isinstance(samples, list):
            return None
        values = [
            number
            for sample in samples
            if isinstance(sample, dict)
            and (number := _positive_number(sample.get("end_to_end_time_ms"))) is not None
        ]
        if not values:
            return None
        latencies.append(statistics.median(values) * 1000.0)
    if not latencies:
        return None
    return math.exp(sum(math.log(latency) for latency in latencies) / len(latencies))


def _positive_number(value: JsonValue | None) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if number > 0 and math.isfinite(number) else None
