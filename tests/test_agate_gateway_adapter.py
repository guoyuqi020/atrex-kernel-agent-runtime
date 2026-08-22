"""Agate SDK adapter, contract resolution, and job ownership tests."""

from __future__ import annotations

import json
import math
import sqlite3
import threading
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from conftest import NOW, digest, seed_lineage

from atrex_runtime.artifacts.local import ArtifactKind, LocalArtifactStore
from atrex_runtime.domain.errors import InfrastructureError, InvalidTransitionError
from atrex_runtime.domain.ids import AttemptId, new_attempt_id, new_epoch_id
from atrex_runtime.domain.models import (
    Attempt,
    AttemptStatus,
    BranchRole,
    Dsl,
    Epoch,
    EpochStatus,
)
from atrex_runtime.gateway.agate import (
    AgateConnectionConfig,
    AgateGatewayAdapter,
    AgateJobBinding,
    SqliteAgateJobStore,
    load_agate_sdk,
)
from atrex_runtime.gateway.contract import (
    AgateEvaluationContext,
    AgateEvaluationContractV1,
    AgateEvaluationOptionsV1,
    RegistryAgateEvaluationContextResolver,
)
from atrex_runtime.gateway.control import GatewayOperation
from atrex_runtime.gateway.proxy import GatewayAdapterRequest
from atrex_runtime.registry.sqlite import SqliteRegistry


def test_agate_job_store_closes_connection_when_initialization_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection = sqlite3.connect(tmp_path / "captured.sqlite")

    def connect(*_args: object, **_kwargs: object) -> sqlite3.Connection:
        return connection

    def fail_migration(_self: SqliteAgateJobStore) -> None:
        raise RuntimeError("broken")

    monkeypatch.setattr(sqlite3, "connect", connect)
    monkeypatch.setattr(SqliteAgateJobStore, "_migrate", fail_migration)
    with pytest.raises(RuntimeError, match="broken"):
        SqliteAgateJobStore(tmp_path / "agate.sqlite")

    with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
        connection.execute("SELECT 1")


def _contract() -> AgateEvaluationContractV1:
    return AgateEvaluationContractV1(
        candidate_path="kernel.py",
        reference_py="class Model: pass\n",
        input_py="def get_inputs(): return ()\n",
        shapes={"0": {}, "1": {}},
        options=AgateEvaluationOptionsV1(
            num_correctness_cases=2,
            bench_iters=50,
            atol=0.001,
            rtol=0.01,
            timeout_s=900,
        ),
        lock_clocks=True,
    )


@dataclass
class StaticContexts:
    context: AgateEvaluationContext

    def resolve(self, attempt_id: AttemptId) -> AgateEvaluationContext:
        del attempt_id
        return self.context


class FakeGatewayError(Exception):
    """Agate-compatible structured error for adapter tests."""

    def __init__(self, status: int, error_class: str, payload: dict[str, object]) -> None:
        self.status = status
        self.error_class = error_class
        self.payload = payload
        super().__init__("blocked candidate")


@dataclass
class FakeAgateClient:
    job: dict[str, object]
    submit_error: Exception | None = None
    acceptance_job_id: str = "ev_test"
    submitted: list[tuple[str, dict[str, object]]] = field(default_factory=list)
    fetched: list[tuple[str, bool, float]] = field(default_factory=list)
    cancelled: list[str] = field(default_factory=list)
    environment_calls: list[tuple[str, str | None, bool]] = field(default_factory=list)

    def submit_job(self, kind: str, request: dict[str, object]) -> dict[str, object]:
        self.submitted.append((kind, request))
        if self.submit_error is not None:
            raise self.submit_error
        return {"job_id": self.acceptance_job_id, "status": "queued"}

    def get_job(
        self,
        job_id: str,
        wait: bool = False,
        timeout: float = 30.0,
        include_spec: bool = False,
    ) -> dict[str, object]:
        del include_spec
        self.fetched.append((job_id, wait, timeout))
        return self.job

    def cancel_job(self, job_id: str) -> dict[str, object]:
        self.cancelled.append(job_id)
        return {"job_id": job_id, "status": "cancelled"}

    def list_env(self, force: bool = False) -> list[dict[str, object]]:
        self.environment_calls.append(("list", None, force))
        return [{"gpu": "H20"}]

    def get_env(self, gpu: str, force: bool = False) -> dict[str, object]:
        self.environment_calls.append(("detail", gpu, force))
        return {"gpu": gpu, "arch": "sm_90"}

    def get_capabilities(self, gpu: str, force: bool = False) -> dict[str, object]:
        self.environment_calls.append(("capabilities", gpu, force))
        return {"gpu": gpu, "frameworks": {"triton": {"available": True}}}

    def health(self) -> bool:
        return True


@dataclass
class EvalProfileAgateClient(FakeAgateClient):
    def submit_job(self, kind: str, request: dict[str, object]) -> dict[str, object]:
        self.submitted.append((kind, request))
        return {"job_id": "ev_test" if kind == "eval" else "pr_test", "status": "queued"}

    def get_job(
        self,
        job_id: str,
        wait: bool = False,
        timeout: float = 30.0,
        include_spec: bool = False,
    ) -> dict[str, object]:
        del include_spec
        self.fetched.append((job_id, wait, timeout))
        if job_id == "ev_test":
            return self.job
        assert job_id == "pr_test"
        return {
            "job_id": job_id,
            "status": "succeeded",
            "result": {
                "kernels": [
                    {
                        "compute_sol_pct": 30.0,
                        "mem_sol_pct": 70.0,
                        "duration": 10.0,
                        "duration_unit": "us",
                    }
                ]
            },
        }


@dataclass
class RepeatedEvalAgateClient(FakeAgateClient):
    """Assign a distinct deterministic result to each concurrently submitted Eval."""

    lock: threading.Lock = field(default_factory=threading.Lock)
    latency_ms_by_job: dict[str, float] = field(default_factory=dict)

    def submit_job(self, kind: str, request: dict[str, object]) -> dict[str, object]:
        assert kind == "eval"
        with self.lock:
            ordinal = len(self.submitted)
            job_id = f"ev_repeat_{ordinal}"
            self.submitted.append((kind, request))
            self.latency_ms_by_job[job_id] = float(ordinal * 2 + 1)
        return {"job_id": job_id, "status": "queued"}

    def get_job(
        self,
        job_id: str,
        wait: bool = False,
        timeout: float = 30.0,
        include_spec: bool = False,
    ) -> dict[str, object]:
        del include_spec
        self.fetched.append((job_id, wait, timeout))
        latency_ms = self.latency_ms_by_job[job_id]
        return {
            "job_id": job_id,
            "status": "succeeded",
            "result": {
                "passed": {
                    "compile": {"0": {"status": "passed"}, "1": {"status": "passed"}},
                    "correctness": {
                        "0": {"status": "passed"},
                        "1": {"status": "passed"},
                    },
                },
                "correctness": {"shapes": {"0": {"cases": []}, "1": {"cases": []}}},
                "performance": {
                    "shapes": {
                        "0": {"samples": [{"end_to_end_time_ms": latency_ms}]},
                        "1": {"samples": [{"end_to_end_time_ms": latency_ms}]},
                    }
                },
            },
        }


@dataclass
class BatchedEvalAgateClient(FakeAgateClient):
    latency_by_job: dict[str, float] = field(default_factory=dict)

    def submit_job(self, kind: str, request: dict[str, object]) -> dict[str, object]:
        assert kind == "eval"
        job_id = f"ev_batch_{len(self.submitted)}"
        self.submitted.append((kind, request))
        self.latency_by_job[job_id] = 2.0 if job_id.endswith("0") else 32.0
        return {"job_id": job_id, "status": "queued"}

    def get_job(
        self,
        job_id: str,
        wait: bool = False,
        timeout: float = 30.0,
        include_spec: bool = False,
    ) -> dict[str, object]:
        del include_spec
        self.fetched.append((job_id, wait, timeout))
        return {
            "job_id": job_id,
            "status": "succeeded",
            "result": {
                "all_pass": True,
                "latency_us_geomean": self.latency_by_job[job_id],
            },
        }


@dataclass
class CapturingBuilder:
    calls: list[dict[str, object]] = field(default_factory=list)

    def __call__(
        self,
        candidate: str,
        reference: object,
        gpu: str,
        **options: object,
    ) -> dict[str, object]:
        call = {
            "candidate": candidate,
            "reference": reference,
            "gpu": gpu,
            **options,
        }
        self.calls.append(call)
        return call


def _successful_job() -> dict[str, object]:
    return {
        "job_id": "ev_test",
        "status": "succeeded",
        "result": {
            "passed": {
                "compile": {
                    "0": {"status": "passed"},
                    "1": {"status": "passed"},
                },
                "correctness": {
                    "0": {"status": "passed"},
                    "1": {"status": "passed"},
                },
            },
            "correctness": {"shapes": {"0": {"cases": []}, "1": {"cases": []}}},
            "performance": {
                "shapes": {
                    "0": {
                        "samples": [
                            {"end_to_end_time_ms": 1.0},
                            {"end_to_end_time_ms": 3.0},
                        ]
                    },
                    "1": {"samples": [{"end_to_end_time_ms": 4.0}]},
                }
            },
        },
    }


def _adapter(
    tmp_path: Path,
    client: FakeAgateClient,
) -> tuple[AgateGatewayAdapter, CapturingBuilder, SqliteAgateJobStore]:
    builder = CapturingBuilder()
    jobs = SqliteAgateJobStore(tmp_path / "agate-jobs.sqlite")
    adapter = AgateGatewayAdapter(
        client,
        builder,
        StaticContexts(AgateEvaluationContext("vector_add", "H20", Dsl.TRITON, _contract())),
        jobs,
        wait_timeout_s=1200,
    )
    return adapter, builder, jobs


@pytest.mark.anyio
async def test_eval_uses_sealed_context_and_maps_raw_atrex_result(tmp_path: Path) -> None:
    client = FakeAgateClient(_successful_job())
    adapter, builder, jobs = _adapter(tmp_path, client)
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    (candidate / "kernel.py").write_text("class Model: pass\n")
    request = GatewayAdapterRequest(
        attempt_id=new_attempt_id(),
        operation=GatewayOperation.EVALUATE,
        idempotency_key="candidate-1",
        candidate_digest=digest("candidate"),
        candidate_path=candidate,
        profile_level=None,
        kernel_regex=None,
        job_id=None,
    )

    result = await adapter.execute(request)

    assert result.status == "completed"
    assert result.evaluation is not None
    assert result.evaluation.correct
    assert result.evaluation.latency_us == pytest.approx(math.sqrt(2000 * 4000))
    assert result.worker_result == {
        "all_pass": True,
        "failures": [],
        "latency_us_geomean": pytest.approx(math.sqrt(2000 * 4000)),
        "latency_us_by_shape": {"0": 2000.0, "1": 4000.0},
        "shape_ids_are_opaque": True,
        "hidden_case_details": "shape inputs and failure details withheld",
    }
    assert client.submitted[0][0] == "eval"
    assert client.fetched == [("ev_test", True, 1200)]
    assert builder.calls[0]["gpu"] == "H20"
    assert builder.calls[0]["idempotency_key"] == "candidate-1"
    assert builder.calls[0]["spec_fields"] == {"languages": ["triton"]}
    jobs.close()


@pytest.mark.anyio
async def test_failed_eval_job_remains_an_infrastructure_outcome(tmp_path: Path) -> None:
    client = FakeAgateClient(
        {
            "job_id": "ev_test",
            "status": "failed",
            "error": {
                "error_class": "infra",
                "reason": "deps_install_failed",
                "details": {"private_log": "must not reach the Agent"},
            },
        }
    )
    adapter, _builder, jobs = _adapter(tmp_path, client)
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    (candidate / "kernel.py").write_text("class Model: pass\n")

    with pytest.raises(InfrastructureError, match="Eval batch did not complete"):
        await adapter.execute(
            GatewayAdapterRequest(
                new_attempt_id(),
                GatewayOperation.EVALUATE,
                "candidate-infrastructure-failure",
                digest("candidate"),
                candidate,
                None,
                None,
                None,
            )
        )
    jobs.close()


@pytest.mark.anyio
async def test_eval_repetitions_run_as_independent_jobs_and_average(tmp_path: Path) -> None:
    client = RepeatedEvalAgateClient(_successful_job())
    builder = CapturingBuilder()
    jobs = SqliteAgateJobStore(tmp_path / "agate-jobs.sqlite")
    adapter = AgateGatewayAdapter(
        client,
        builder,
        StaticContexts(AgateEvaluationContext("vector_add", "H20", Dsl.TRITON, _contract())),
        jobs,
        wait_timeout_s=1200,
        optimizer_evaluate_repeats=2,
    )
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    candidate.joinpath("kernel.py").write_text("class Model: pass\n")

    result = await adapter.execute(
        GatewayAdapterRequest(
            new_attempt_id(),
            GatewayOperation.EVALUATE,
            "candidate-1",
            digest("candidate"),
            candidate,
            None,
            None,
            None,
        )
    )

    assert result.evaluation is not None
    assert result.evaluation.correct is True
    assert result.evaluation.latency_us == pytest.approx(2000.0)
    assert len(client.submitted) == 2
    child_keys = [payload["idempotency_key"] for _kind, payload in client.submitted]
    assert len(set(child_keys)) == 2
    assert isinstance(result.result, dict)
    assert result.result["repeats"] == 2
    assert len(result.result["jobs"]) == 2  # type: ignore[arg-type]
    jobs.close()


@pytest.mark.anyio
async def test_optimizer_eval_uses_shared_shape_batches(tmp_path: Path) -> None:
    contract = _contract().model_copy(update={"shapes": {str(index): {} for index in range(5)}})
    client = BatchedEvalAgateClient(_successful_job())
    builder = CapturingBuilder()
    jobs = SqliteAgateJobStore(tmp_path / "agate-jobs.sqlite")
    adapter = AgateGatewayAdapter(
        client,
        builder,
        StaticContexts(AgateEvaluationContext("vector_add", "H20", Dsl.TRITON, contract)),
        jobs,
        wait_timeout_s=1200,
    )
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    candidate.joinpath("kernel.py").write_text("class Model: pass\n")

    result = await adapter.execute(
        GatewayAdapterRequest(
            new_attempt_id(),
            GatewayOperation.EVALUATE,
            "candidate-batched",
            digest("candidate"),
            candidate,
            None,
            None,
            None,
        )
    )

    assert len(client.submitted) == 2
    references = [payload["reference"] for _kind, payload in client.submitted]
    assert sorted(len(reference["shapes"]) for reference in references) == [1, 4]  # type: ignore[index]
    assert result.evaluation is not None
    assert result.evaluation.latency_us == pytest.approx((2.0**4 * 32.0) ** (1 / 5))
    assert isinstance(result.result, dict)
    assert result.result["operation"] == "shape_batched_evaluate"
    assert result.job_id is None
    jobs.close()


@pytest.mark.anyio
async def test_eval_profiles_when_the_sealed_contract_has_no_roofline(tmp_path: Path) -> None:
    client = EvalProfileAgateClient(_successful_job())
    builder = CapturingBuilder()
    jobs = SqliteAgateJobStore(tmp_path / "agate-jobs.sqlite")
    adapter = AgateGatewayAdapter(
        client,
        builder,
        StaticContexts(AgateEvaluationContext("vector_add", "H20", Dsl.TRITON, _contract())),
        jobs,
        wait_timeout_s=1200,
        profile_without_roofline=True,
    )
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    candidate.joinpath("kernel.py").write_text("class Model: pass\n")

    result = await adapter.execute(
        GatewayAdapterRequest(
            new_attempt_id(),
            GatewayOperation.EVALUATE,
            "candidate-1",
            digest("candidate"),
            candidate,
            None,
            None,
            None,
        )
    )

    assert [kind for kind, _payload in client.submitted] == ["eval", "profile"]
    assert client.submitted[1][1]["level"] == "sol"
    assert client.submitted[1][1]["top_kernels"] == 10
    assert isinstance(result.profile_result, dict)
    assert result.profile_result["status"] == "succeeded"
    jobs.close()


@pytest.mark.anyio
async def test_compile_failure_is_a_candidate_outcome(tmp_path: Path) -> None:
    job = _successful_job()
    result_value = job["result"]
    assert isinstance(result_value, dict)
    passed = result_value["passed"]
    assert isinstance(passed, dict)
    passed["compile"] = {"0": {"status": "failed"}, "1": {"status": "passed"}}
    client = FakeAgateClient(job)
    adapter, _builder, jobs = _adapter(tmp_path, client)
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    (candidate / "kernel.py").write_text("class Model: pass\n")

    mapped = await adapter.execute(
        GatewayAdapterRequest(
            new_attempt_id(),
            GatewayOperation.EVALUATE,
            "candidate-1",
            digest("candidate"),
            candidate,
            None,
            None,
            None,
        )
    )

    assert mapped.evaluation is not None
    assert not mapped.evaluation.correct
    assert mapped.evaluation.latency_us is None
    assert isinstance(mapped.worker_result, dict)
    assert mapped.worker_result["all_pass"] is False
    assert mapped.worker_result["failures"] == [
        "one or more hidden evaluator cases failed; reproduce within the public shape_domain"
    ]
    assert "passed" not in mapped.worker_result
    jobs.close()


@pytest.mark.anyio
async def test_submission_validation_rejection_is_a_candidate_outcome(tmp_path: Path) -> None:
    rejection = FakeGatewayError(
        400,
        "validation",
        {
            "reason": "candidate_validation_failed",
            "details": {"forbidden_imports": ["subprocess"]},
        },
    )
    client = FakeAgateClient(_successful_job(), submit_error=rejection)
    adapter, _builder, jobs = _adapter(tmp_path, client)
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    (candidate / "kernel.py").write_text("import subprocess\n")

    mapped = await adapter.execute(
        GatewayAdapterRequest(
            new_attempt_id(),
            GatewayOperation.EVALUATE,
            "candidate-1",
            digest("candidate"),
            candidate,
            None,
            None,
            None,
        )
    )

    assert mapped.status == "completed"
    assert mapped.job_id is None
    assert mapped.evaluation is not None
    assert not mapped.evaluation.correct
    assert mapped.result == {
        "schema_version": 1,
        "operation": "shape_batched_evaluate",
        "status": "rejected",
        "error": {
            "category": "candidate_rejected",
            "detail": {
                "reason": "candidate_validation_failed",
                "details": {"forbidden_imports": ["subprocess"]},
            },
        },
        "rejected_batch_index": 0,
        "shape_batch_count": 1,
    }
    assert mapped.worker_result == {
        "status": "rejected",
        "error": {
            "category": "candidate_rejected",
            "message": "candidate request rejected before evaluation job creation",
        },
    }
    jobs.close()


@pytest.mark.anyio
async def test_profile_can_return_queued_job_for_later_poll(tmp_path: Path) -> None:
    client = FakeAgateClient({"job_id": "ev_test", "status": "running"})
    adapter, builder, jobs = _adapter(tmp_path, client)
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    (candidate / "kernel.py").write_text("class Model: pass\n")

    mapped = await adapter.execute(
        GatewayAdapterRequest(
            new_attempt_id(),
            GatewayOperation.PROFILE,
            "profile-1",
            digest("candidate"),
            candidate,
            "deep",
            "kernel_name",
            None,
            {
                "profiler": "ncu",
                "counters": ["sm__throughput.avg.pct_of_peak_sustained_elapsed"],
                "source": True,
                "launch_skip": 100,
                "launch_count": 24,
                "top_kernels": 3,
                "shape_id": "1",
            },
        )
    )

    assert mapped.status == "queued"
    assert mapped.job_id == "ev_test"
    assert client.submitted[0][0] == "profile"
    assert builder.calls[0]["level"] == "deep"
    assert builder.calls[0]["kernel_regex"] == "kernel_name"
    assert builder.calls[0]["profiler"] == "ncu"
    assert builder.calls[0]["source"] is True
    assert builder.calls[0]["launch_skip"] == 100
    reference = builder.calls[0]["reference"]
    assert isinstance(reference, dict)
    assert reference["shapes"] == {"1": {}}
    jobs.close()


@pytest.mark.anyio
async def test_profile_result_hides_the_privately_selected_case(tmp_path: Path) -> None:
    client = FakeAgateClient(
        {
            "job_id": "ev_test",
            "status": "succeeded",
            "spec": {"reference": {"shapes": {"1": {"input_kwargs": {"secret_size": 4096}}}}},
            "result": {"kernels": [{"name": "vector_add", "duration": 4.0}]},
        }
    )
    adapter, _builder, jobs = _adapter(tmp_path, client)
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    candidate.joinpath("kernel.py").write_text("class Model: pass\n")

    mapped = await adapter.execute(
        GatewayAdapterRequest(
            new_attempt_id(),
            GatewayOperation.PROFILE,
            "profile-private",
            digest("candidate"),
            candidate,
            "survey",
            None,
            None,
            {"shape_id": "1"},
        )
    )

    assert "secret_size" in json.dumps(mapped.result)
    assert "secret_size" not in json.dumps(mapped.worker_result)
    assert mapped.worker_result == {
        "job_id": "ev_test",
        "status": "succeeded",
        "result": {"kernels": [{"name": "vector_add", "duration": 4.0}]},
    }
    jobs.close()


@pytest.mark.anyio
async def test_attempt_can_poll_and_cancel_its_profile_job(tmp_path: Path) -> None:
    client = FakeAgateClient({"job_id": "ev_test", "status": "running"})
    adapter, _builder, jobs = _adapter(tmp_path, client)
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    (candidate / "kernel.py").write_text("class Model: pass\n")
    attempt_id = new_attempt_id()
    await adapter.execute(
        GatewayAdapterRequest(
            attempt_id,
            GatewayOperation.PROFILE,
            "profile-1",
            digest("candidate"),
            candidate,
            "sol",
            None,
            None,
        )
    )

    polled = await adapter.execute(
        GatewayAdapterRequest(
            attempt_id,
            GatewayOperation.POLL,
            "poll-1",
            None,
            None,
            None,
            None,
            "ev_test",
        )
    )
    cancelled = await adapter.execute(
        GatewayAdapterRequest(
            attempt_id,
            GatewayOperation.CANCEL,
            "cancel-1",
            None,
            None,
            None,
            None,
            "ev_test",
        )
    )

    assert polled.status == "queued"
    assert cancelled.status == "cancelled"
    assert client.fetched[-1] == ("ev_test", False, 30.0)
    assert client.cancelled == ["ev_test"]
    with pytest.raises(PermissionError):
        await adapter.execute(
            GatewayAdapterRequest(
                new_attempt_id(),
                GatewayOperation.POLL,
                "foreign-poll",
                None,
                None,
                None,
                None,
                "ev_test",
            )
        )
    jobs.close()


def test_job_store_enforces_attempt_ownership_and_idempotency(tmp_path: Path) -> None:
    store = SqliteAgateJobStore(tmp_path / "agate-jobs.sqlite")
    owner = new_attempt_id()
    binding = AgateJobBinding("ev_one", owner, "candidate-1", "eval")

    assert store.bind(binding) == binding
    assert store.bind(binding) == binding
    assert store.require_owned(owner, "ev_one") == binding
    with pytest.raises(PermissionError):
        store.require_owned(new_attempt_id(), "ev_one")
    with pytest.raises(InvalidTransitionError):
        store.bind(AgateJobBinding("ev_two", owner, "candidate-1", "eval"))
    store.close()


@pytest.mark.anyio
async def test_all_structured_agate_job_commands_use_attempt_context(tmp_path: Path) -> None:
    client = FakeAgateClient({"job_id": "job", "status": "succeeded", "command_ok": True})
    adapter, _builder, jobs = _adapter(tmp_path, client)
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    (candidate / "kernel.py").write_text("class Model: pass\n")
    (candidate / "helper.py").write_text("VALUE = 1\n")
    attempt_id = new_attempt_id()
    cases = (
        (
            GatewayOperation.DEV,
            "dv_test",
            {
                "command": "python3 kernel.py",
                "env_vars": {"TRITON_CACHE_DIR": "/tmp/triton"},
                "job_timeout_s": 300,
                "recycle": True,
                "intent": "compile",
                "note": "compile candidate",
            },
            "dev",
        ),
        (
            GatewayOperation.CHECK,
            "ck_test",
            {
                "arch": "sm_90",
                "sanitize": "memcheck",
                "requirements": ["custom-kernel-package==1"],
                "deps_mode": "no_deps",
            },
            "compile",
        ),
        (
            GatewayOperation.DISASSEMBLE,
            "da_test",
            {"fmt": "ptx"},
            "disassemble",
        ),
    )

    for operation, job_id, parameters, expected_kind in cases:
        client.acceptance_job_id = job_id
        client.job = {"job_id": job_id, "status": "succeeded", "command_ok": True}
        result = await adapter.execute(
            GatewayAdapterRequest(
                attempt_id,
                operation,
                f"{operation.value}-1",
                digest(f"{operation.value}-candidate"),
                candidate,
                None,
                None,
                None,
                parameters,
            )
        )
        assert result.status == "completed"
        kind, payload = client.submitted[-1]
        assert kind == expected_kind
        if operation is GatewayOperation.DEV:
            assert payload["spec"] == {"target_hardware": ["H20"]}
            assert payload["files"] == {
                "helper.py": "VALUE = 1\n",
                "kernel.py": "class Model: pass\n",
            }
            assert payload["dev_intent"] == "compile"
        else:
            assert payload["candidate"] == "class Model: pass\n"
    jobs.close()


@pytest.mark.anyio
async def test_submit_and_sol_are_non_authoritative_attempt_owned_jobs(tmp_path: Path) -> None:
    client = FakeAgateClient({"job_id": "job", "status": "succeeded"})
    adapter, _builder, jobs = _adapter(tmp_path, client)
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    (candidate / "kernel.py").write_text("class Model: pass\n")
    (candidate / "payload.json").write_text(
        '{"spec":{"target_hardware":["H20"]},"candidate":"source"}'
    )
    (candidate / "solution.json").write_text(
        '{"name":"custom","definition":"problem","sources":[]}'
    )
    attempt_id = new_attempt_id()

    client.acceptance_job_id = "ev_raw"
    client.job = {"job_id": "ev_raw", "status": "succeeded", "result": {"all_pass": True}}
    submitted = await adapter.execute(
        GatewayAdapterRequest(
            attempt_id,
            GatewayOperation.SUBMIT,
            "raw-1",
            digest("raw-candidate"),
            candidate,
            None,
            None,
            None,
            {"payload_path": "payload.json"},
        )
    )
    assert submitted.evaluation is None
    assert client.submitted[-1] == (
        "eval",
        {
            "spec": {"target_hardware": ["H20"]},
            "candidate": "source",
            "lock_clocks": True,
            "idempotency_key": "raw-1",
        },
    )

    client.acceptance_job_id = "sol_raw"
    client.job = {"job_id": "sol_raw", "status": "succeeded", "result": {"score": 1.0}}
    solved = await adapter.execute(
        GatewayAdapterRequest(
            attempt_id,
            GatewayOperation.SOL,
            "sol-1",
            digest("sol-candidate"),
            candidate,
            None,
            None,
            None,
            {
                "solution_path": "solution.json",
                "subset": "L1",
                "iterations": 10,
                "benchmark_reference": True,
            },
        )
    )
    assert solved.evaluation is None
    assert client.submitted[-1][0] == "sol"
    assert client.submitted[-1][1]["gpu"] == "H20"
    assert client.submitted[-1][1]["options"] == {
        "iterations": 10,
        "lock_clocks": True,
        "benchmark_reference": True,
    }
    jobs.close()


@pytest.mark.anyio
async def test_agate_query_commands_are_scoped_and_config_is_redacted(tmp_path: Path) -> None:
    client = FakeAgateClient({"job_id": "dv_query", "status": "running", "kind": "dev"})
    builder = CapturingBuilder()
    jobs = SqliteAgateJobStore(tmp_path / "agate-jobs.sqlite")
    attempt_id = new_attempt_id()
    jobs.bind(AgateJobBinding("dv_query", attempt_id, "dev-1", "dev", GatewayOperation.DEV))
    adapter = AgateGatewayAdapter(
        client,
        builder,
        StaticContexts(AgateEvaluationContext("vector_add", "H20", Dsl.TRITON, _contract())),
        jobs,
        wait_timeout_s=1200,
        connection_summary={"url": "https://gateway.example", "auth": "ak_sk"},
    )

    listed = await adapter.execute(
        GatewayAdapterRequest(
            attempt_id,
            GatewayOperation.JOBS,
            "jobs-1",
            None,
            None,
            None,
            None,
            None,
            {"kind": "dev", "status": "running", "limit": 10},
        )
    )
    environments = await adapter.execute(
        GatewayAdapterRequest(
            attempt_id,
            GatewayOperation.ENV,
            "env-1",
            None,
            None,
            None,
            None,
            None,
            {"gpu": "H20", "capabilities": True, "force": True},
        )
    )
    health = await adapter.execute(
        GatewayAdapterRequest(
            attempt_id,
            GatewayOperation.HEALTH,
            "health-1",
            None,
            None,
            None,
            None,
            None,
        )
    )
    config = await adapter.execute(
        GatewayAdapterRequest(
            attempt_id,
            GatewayOperation.CONFIG,
            "config-1",
            None,
            None,
            None,
            None,
            None,
        )
    )

    assert listed.result == {"jobs": [client.job]}
    assert environments.result["gpu"] == "H20"
    assert client.environment_calls == [("capabilities", "H20", True)]
    assert health.result == {"ok": True}
    assert config.result == {"url": "https://gateway.example", "auth": "ak_sk"}
    jobs.close()


def test_registry_context_resolver_reads_sealed_contract(tmp_path: Path) -> None:
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    contract_digest = artifacts.put_json(
        _contract().model_dump(mode="json"), ArtifactKind.EVALUATION_CONTRACT
    )
    registry = SqliteRegistry(tmp_path / "registry.sqlite")
    lineage = seed_lineage(
        registry,
        evaluation_contract_digest=contract_digest,
        evidence_checkpoint=digest("evidence"),
        challenger_count=0,
        attempts_per_trajectory=1,
    )
    epoch = Epoch(
        id=new_epoch_id(),
        lineage_id=lineage.lineage_id,
        number=1,
        active_kernel_agent_revision_id=lineage.active_revision_id,
        challenger_kernel_agent_revision_ids=(),
        starting_kernel_revision_id=lineage.baseline.id,
        evidence_checkpoint=digest("evidence"),
        challenger_count=0,
        trajectories_per_branch=1,
        attempts_per_trajectory=1,
        status=EpochStatus.RUNNING,
        winner_kernel_agent_revision_id=None,
        best_kernel_revision_id=None,
        created_at=NOW,
        completed_at=None,
    )
    registry.insert_epoch(epoch)
    attempt = Attempt(
        id=new_attempt_id(),
        epoch_id=epoch.id,
        branch=BranchRole.ACTIVE,
        challenger_ordinal=0,
        trajectory_ordinal=1,
        ordinal=1,
        kernel_agent_revision_id=lineage.active_revision_id,
        input_kernel_revision_id=lineage.baseline.id,
        attempt_evidence_digest=digest("attempt-evidence"),
        output_kernel_revision_id=None,
        accepted_as_branch_best=False,
        status=AttemptStatus.RUNNING,
        infrastructure_failures=0,
        recovery_generation=0,
        authority_started_at=NOW,
        failure_reason=None,
        created_at=NOW,
        completed_at=None,
    )
    registry.insert_attempt(attempt)

    context = RegistryAgateEvaluationContextResolver(registry, artifacts).resolve(attempt.id)

    assert context.operator == "vector_add"
    assert context.hardware_target == "nvidia-h100"
    assert context.contract.options.bench_iters == 50
    registry.close()


def test_published_agate_sdk_loads_through_production_factory() -> None:
    client, builder = load_agate_sdk(
        AgateConnectionConfig(
            base_url="http://127.0.0.1:8000",
            auth_mode="none",
            http_timeout_s=30,
            wait_timeout_s=600,
        )
    )

    assert type(client).__module__ == "atrex_runtime.gateway.retrying_client"
    assert type(client.wrapped_client).__module__ == "atrex_gateway_client.client"  # type: ignore[attr-defined]
    assert builder.__module__ == "atrex_gateway_client.payload"
