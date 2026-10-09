"""Trusted same-allocation ABBA runner tests."""

from __future__ import annotations

import json
import os
import subprocess
import threading
from pathlib import Path

import pytest
from atrex_gateway_client import build_eval_request_from_content
from conftest import NOW, digest

from atrex_runtime.artifacts.local import ArtifactKind, LocalArtifactStore
from atrex_runtime.domain.errors import InfrastructureError
from atrex_runtime.domain.ids import ArtifactDigest, new_kernel_revision_id
from atrex_runtime.domain.models import (
    Dsl,
    KernelEvaluation,
    KernelMeasurement,
    KernelMeasurementPurpose,
    KernelRevision,
)
from atrex_runtime.gateway.abba import (
    AgateSameAllocationAbbaRunner,
    CommitPinnedAtrexBenchEvaluator,
    _parse_native_abba_payload,
    _schedule,
    _uses_native_abba,
    build_native_abba_request,
)
from atrex_runtime.gateway.contract import (
    AgateEvaluationContext,
    AgateEvaluationContractV1,
    AgateEvaluationOptionsV1,
)
from atrex_runtime.gateway.result_metrics import gateway_result_sol_summary
from atrex_runtime.kernel_sources import KernelSourceBundle, KernelSourceContract


class FakeContextResolver:
    def __init__(self, context: AgateEvaluationContext) -> None:
        self.context = context

    def resolve(self, _revision: KernelRevision) -> AgateEvaluationContext:
        return self.context


class FakeEvaluator:
    commit = "f" * 40

    def files(self) -> dict[str, str]:
        pytest.fail("native ABBA must not export a custom Dev evaluator")

    def bundle_digest(self) -> str:
        pytest.fail("native ABBA must not inspect a custom Dev evaluator")


class FakeJournal:
    def __init__(self) -> None:
        self.measurements: list[KernelMeasurement] = []
        self.events: list[tuple[str, str, object]] = []
        self.abba_batches: dict[str, ArtifactDigest] = {}

    def record_kernel_measurement(self, measurement: KernelMeasurement) -> KernelMeasurement:
        existing = next((item for item in self.measurements if item.id == measurement.id), None)
        if existing is None:
            self.measurements.append(measurement)
        else:
            assert existing == measurement
        return measurement

    def get_authoritative_abba_batch(self, task_digest: ArtifactDigest) -> ArtifactDigest | None:
        return self.abba_batches.get(str(task_digest))

    def record_authoritative_abba_batch(
        self, task_digest: ArtifactDigest, result_digest: ArtifactDigest
    ) -> ArtifactDigest:
        return self.abba_batches.setdefault(str(task_digest), result_digest)

    def record_runtime_event(
        self,
        kind: str,
        aggregate_id: str,
        payload: object = None,
    ) -> None:
        self.events.append((kind, aggregate_id, payload))


class FakeAgateClient:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.jobs: dict[str, dict[str, object]] = {}
        self.requests: list[dict[str, object]] = []

    def submit_job(self, kind: str, request: dict[str, object]) -> dict[str, object]:
        assert kind == "eval"
        reference = request["reference"]
        abba = request["abba"]
        assert isinstance(reference, dict) and isinstance(abba, dict)
        shapes = reference["shapes"]
        assert isinstance(shapes, dict)
        shape_ids = list(shapes)
        repeats = abba["repeats"]
        assert type(repeats) is int
        expected = [
            {"index": 0, "revision": "baseline", "label": "A", "repeat": 0},
            {"index": 1, "revision": "candidate", "label": "B", "repeat": 0},
            {"index": 2, "revision": "candidate", "label": "B", "repeat": 1},
            {"index": 3, "revision": "baseline", "label": "A", "repeat": 1},
        ]
        blocks = []
        for _ in range(repeats):
            runs = [
                {
                    **step,
                    "result": _native_evaluation(
                        shape_ids, 100.0 if step["revision"] == "baseline" else 90.0
                    ),
                }
                for step in expected
            ]
            blocks.append(
                {
                    "eval_mode": "abba",
                    "error": None,
                    "passed": {"abba": {"status": "passed"}},
                    "abba": {"schedule": expected, "runs": runs, "comparison": {}},
                }
            )
        with self._lock:
            job_id = f"ev_abba_{len(self.requests)}"
            self.requests.append(request)
        self.jobs[job_id] = {
            "job_id": job_id,
            "status": "succeeded",
            "command_ok": True,
            "result": {"abba": {"sdk_results": blocks, "valid": True}},
        }
        return {"job_id": job_id, "status": "queued"}


    def get_job(
        self,
        job_id: str,
        wait: bool = False,
        timeout: float = 30.0,
        include_spec: bool = False,
    ) -> dict[str, object]:
        assert wait and timeout == 90 and not include_spec
        return self.jobs[job_id]


def _native_request_builder(
    candidate: str | dict[str, object],
    reference: dict[str, object],
    gpu: str,
    **fields: object,
) -> dict[str, object]:
    return {"candidate": candidate, "reference": reference, "gpu": gpu, **fields}


def _native_evaluation(shape_ids: list[str], latency_us: float) -> dict[str, object]:
    return {
        "error": None,
        "passed": {
            "compile": {"status": "passed"},
            "correctness": {shape_id: {"status": "passed"} for shape_id in shape_ids},
        },
        "correctness": {"shapes": {shape_id: {} for shape_id in shape_ids}},
        "performance": {
            "shapes": {
                shape_id: {
                    "error": None,
                    "samples": [{"end_to_end_time_ms": latency_us / 1000}],
                    "sol": {"pct": 50.0 if latency_us == 90 else 25.0},
                }
                for shape_id in shape_ids
            }
        },
    }




def test_commit_pinned_evaluator_exports_only_required_runtime(tmp_path: Path) -> None:
    repository = tmp_path / "atrex-bench"
    (repository / "scripts").mkdir(parents=True)
    (repository / "src/atrex_bench/eval").mkdir(parents=True)
    (repository / "scripts/run_eval.py").write_text("print('eval')\n", encoding="utf-8")
    (repository / "src/atrex_bench/__init__.py").write_text("", encoding="utf-8")
    (repository / "src/atrex_bench/sdk.py").write_text("def evaluate(config): return {}\n")
    (repository / "src/atrex_bench/eval/__init__.py").write_text("", encoding="utf-8")
    (repository / "unrelated.bin").write_bytes(b"not exported")
    subprocess.run(("git", "init", "-q", str(repository)), check=True)
    subprocess.run(("git", "-C", str(repository), "add", "."), check=True)
    environment = {
        **os.environ,
        "GIT_AUTHOR_NAME": "ATREX Test",
        "GIT_AUTHOR_EMAIL": "atrex@example.invalid",
        "GIT_COMMITTER_NAME": "ATREX Test",
        "GIT_COMMITTER_EMAIL": "atrex@example.invalid",
    }
    subprocess.run(
        ("git", "-C", str(repository), "commit", "-q", "-m", "evaluator"),
        check=True,
        env=environment,
    )
    commit = subprocess.run(
        ("git", "-C", str(repository), "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    files = CommitPinnedAtrexBenchEvaluator(
        repository=str(repository),
        commit=commit,
        git_executable="/usr/bin/git",
        fetch_timeout_seconds=30,
        max_archive_bytes=1024 * 1024,
        max_bundle_files=16,
        max_bundle_bytes=1024 * 1024,
    ).files()

    assert "atrex-bench/scripts/run_eval.py" in files
    assert "atrex-bench/src/atrex_bench/sdk.py" in files
    assert all("unrelated" not in path for path in files)


@pytest.mark.anyio
@pytest.mark.parametrize("lock_clocks", (False, True))
async def test_abba_runner_uses_one_allocation_per_shape_batch_and_records_runs(
    tmp_path: Path,
    lock_clocks: bool,
) -> None:
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    incumbent_dir = tmp_path / "incumbent"
    candidate_dir = tmp_path / "candidate"
    incumbent_dir.mkdir()
    candidate_dir.mkdir()
    (incumbent_dir / "kernel.py").write_text("INCUMBENT = True\n", encoding="utf-8")
    (candidate_dir / "kernel.py").write_text("CANDIDATE = True\n", encoding="utf-8")
    incumbent_digest = artifacts.put_directory(incumbent_dir, ArtifactKind.KERNEL)
    candidate_digest = artifacts.put_directory(candidate_dir, ArtifactKind.KERNEL)
    incumbent = KernelRevision(
        new_kernel_revision_id(),
        None,
        incumbent_digest,
        None,
        KernelEvaluation(True, 100, digest("incumbent-gateway")),
        NOW,
    )
    candidate = KernelRevision(
        new_kernel_revision_id(),
        incumbent.id,
        candidate_digest,
        None,
        KernelEvaluation(True, 90, digest("candidate-gateway")),
        NOW,
    )
    contract = AgateEvaluationContractV1(
        candidate_path="kernel.py",
        reference_py="def reference(): pass",
        input_py="def _make_inputs(): return ()",
        shapes={f"shape-{index}": [index] for index in range(5)},
        roofline={
            "shapes": {f"shape-{index}": {"SOL_time_ms": {"Test GPU": 0.045}} for index in range(5)}
        },
        options=AgateEvaluationOptionsV1(
            num_correctness_cases=1,
            bench_iters=10,
            atol=0.01,
            rtol=0.01,
            timeout_s=60,
        ),
        lock_clocks=lock_clocks,
    )
    contract_digest = digest("evaluation-contract")
    context = AgateEvaluationContext(
        "vecadd",
        "H20",
        Dsl.TRITON,
        contract,
        contract_digest,
    )
    client = FakeAgateClient()
    journal = FakeJournal()
    runner = AgateSameAllocationAbbaRunner(
        client,
        FakeContextResolver(context),  # type: ignore[arg-type]
        artifacts,
        journal,  # type: ignore[arg-type]
        FakeEvaluator(),  # type: ignore[arg-type]
        build_eval_request_from_content,
        wait_timeout_s=90,
    )

    result = await runner.run_pair(
        incumbent,
        candidate,
        repeats=2,
        purpose=KernelMeasurementPurpose.KERNEL_RETENTION,
        per_run_timeout_seconds=100,
        allocation_timeout_seconds=500,
        shape_batch_size=3,
        max_parallel_shape_batches=2,
    )

    assert len(client.requests) == 2
    assert len({request["idempotency_key"] for request in client.requests}) == 2
    assert all(request["lock_clocks"] is lock_clocks for request in client.requests)
    assert all(request["abba"]["repeats"] == 1 for request in client.requests)
    assert all("files" not in request and "command" not in request for request in client.requests)
    assert [run.latency_us for run in result.incumbent_runs] == pytest.approx([100] * 2)
    assert [run.latency_us for run in result.candidate_runs] == pytest.approx([90] * 2)
    assert result.incumbent_latency_us == pytest.approx(100)
    assert result.candidate_latency_us == pytest.approx(90)
    assert len(journal.measurements) == 4
    assert len({measurement.gateway_result_digest for measurement in journal.measurements}) == 1
    assert result.gateway_result_digest == journal.measurements[0].gateway_result_digest
    assert {
        frozenset(str(measurement.agate_job_id).split(",")) for measurement in journal.measurements
    } == {
        frozenset(("ev_abba_0", "ev_abba_1")),
    }
    assert any(kind == "comparison.abba_completed" for kind, _, _ in journal.events)
    assert result.gateway_result_digest is not None
    summary = gateway_result_sol_summary(artifacts, result.gateway_result_digest)
    assert summary.percent == pytest.approx(50.0)
    assert summary.source == "roofline"
    stored = artifacts.verify(result.gateway_result_digest)
    aggregate = json.loads((stored.payload_path / "value.json").read_text(encoding="utf-8"))
    assert aggregate["operation"] == "same_allocation_abba"
    assert aggregate["evaluation_contract_digest"] == str(contract_digest)
    assert aggregate["candidate"]["latency_us"] == pytest.approx(90.0)
    assert aggregate["candidate"]["sol_pct"] == pytest.approx(50.0)
    assert aggregate["measurement_aggregation"] == {
        "repetitions": 1,
        "method": "single_measurement",
    }
    assert len(aggregate["repetitions"]) == 1
    replay = await runner.run_pair(
        incumbent,
        candidate,
        repeats=2,
        purpose=KernelMeasurementPurpose.KERNEL_RETENTION,
        per_run_timeout_seconds=100,
        allocation_timeout_seconds=500,
        shape_batch_size=3,
        max_parallel_shape_batches=2,
    )
    assert len(client.requests) == 2
    assert len(journal.measurements) == 4
    assert replay.gateway_result_digest == result.gateway_result_digest
    another_candidate = KernelRevision(
        new_kernel_revision_id(),
        incumbent.id,
        candidate_digest,
        None,
        KernelEvaluation(True, 90, digest("another-candidate-gateway")),
        NOW,
    )
    await runner.run_pair(
        incumbent,
        another_candidate,
        repeats=2,
        purpose=KernelMeasurementPurpose.KERNEL_RETENTION,
        per_run_timeout_seconds=100,
        allocation_timeout_seconds=500,
        shape_batch_size=3,
        max_parallel_shape_batches=2,
    )
    assert len(client.requests) == 4  # The same bytes under a new revision are remeasured.


@pytest.mark.anyio
async def test_authoritative_abba_uses_its_single_measurement_without_a_median(
    tmp_path: Path,
) -> None:
    class OutlierClient(FakeAgateClient):
        def submit_job(self, kind: str, request: dict[str, object]) -> dict[str, object]:
            accepted = super().submit_job(kind, request)
            if len(self.requests) != 1:
                return accepted
            job = self.jobs[str(accepted["job_id"])]
            result = job["result"]
            assert isinstance(result, dict)
            for block in result["abba"]["sdk_results"]:
                for run in block["abba"]["runs"]:
                    for shape in run["result"]["performance"]["shapes"].values():
                        for sample in shape["samples"]:
                            sample["end_to_end_time_ms"] *= 2
            return accepted

    client = OutlierClient()
    result, _ = await _run_pair(client, tmp_path, shape_batch_size=5)

    assert len(client.requests) == 1
    assert result.incumbent_latency_us == pytest.approx(200.0)
    assert result.candidate_latency_us == pytest.approx(180.0)
    assert result.gateway_result_digest is not None
    stored = LocalArtifactStore(tmp_path / "artifacts").verify(result.gateway_result_digest)
    aggregate = json.loads((stored.payload_path / "value.json").read_text(encoding="utf-8"))
    assert [item["candidate"]["latency_us"] for item in aggregate["repetitions"]] == [
        pytest.approx(180),
    ]
    assert aggregate["candidate"]["latency_us_by_shape"] == {
        f"shape-{index}": 180.0 for index in range(5)
    }


class FlakyAgateClient(FakeAgateClient):
    """Fail the first N submissions with the exact payload a transient Agate batch returned."""

    def __init__(self, failures: int, *, error: dict[str, object] | None = None) -> None:
        super().__init__()
        self.remaining_failures = failures
        self.error = (
            error
            if error is not None
            else {
                "error_class": "infra",
                "reason": "no_result",
                "message": "result begin marker not found",
                "trace_id": "req-8567eddfc34a",
                "details": {"failure_origin": "unknown", "logs_tail": ""},
            }
        )

    def submit_job(self, kind: str, request: dict[str, object]) -> dict[str, object]:
        accepted = super().submit_job(kind, request)
        with self._lock:
            fail = self.remaining_failures > 0
            if fail:
                self.remaining_failures -= 1
        if fail:
            job_id = str(accepted["job_id"])
            self.jobs[job_id] = {
                "job_id": job_id,
                "kind": "eval",
                "status": "failed",
                "command_ok": None,
                "trace_id": "req-8567eddfc34a",
                "error": self.error,
            }
        return accepted


async def _run_pair(
    client: FakeAgateClient,
    tmp_path: Path,
    *,
    shape_batch_size: int = 3,
    repeats: int = 2,
    request_builder: object | None = build_eval_request_from_content,
    source_tree: bool = False,
    replay: bool = False,
    allocation_timeout_seconds: float = 500,
) -> tuple[object, FakeJournal]:
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    incumbent_dir = tmp_path / "incumbent"
    candidate_dir = tmp_path / "candidate"
    incumbent_dir.mkdir()
    candidate_dir.mkdir()
    (incumbent_dir / "kernel.py").write_text("INCUMBENT = True\n", encoding="utf-8")
    (candidate_dir / "kernel.py").write_text("CANDIDATE = True\n", encoding="utf-8")
    if source_tree:
        for root in (incumbent_dir, candidate_dir):
            (root / "helpers").mkdir()
            (root / "helpers/impl.py").write_text(f"SIDE = {root.name!r}\n", encoding="utf-8")
    incumbent = KernelRevision(
        new_kernel_revision_id(),
        None,
        artifacts.put_directory(incumbent_dir, ArtifactKind.KERNEL),
        None,
        KernelEvaluation(True, 100, digest("incumbent-gateway")),
        NOW,
    )
    candidate = KernelRevision(
        new_kernel_revision_id(),
        incumbent.id,
        artifacts.put_directory(candidate_dir, ArtifactKind.KERNEL),
        None,
        KernelEvaluation(True, 90, digest("candidate-gateway")),
        NOW,
    )
    contract = AgateEvaluationContractV1(
        candidate_path="kernel.py",
        reference_py="def reference(): pass",
        input_py="def _make_inputs(): return ()",
        shapes={f"shape-{index}": [index] for index in range(5)},
        options=AgateEvaluationOptionsV1(
            num_correctness_cases=1, bench_iters=10, atol=0.01, rtol=0.01, timeout_s=60
        ),
        kernel_sources={Dsl.TRITON: _source_contract()} if source_tree else {},
    )
    context = AgateEvaluationContext(
        "vecadd", "H20", Dsl.TRITON, contract, digest("evaluation-contract")
    )
    journal = FakeJournal()
    runner = AgateSameAllocationAbbaRunner(
        client,
        FakeContextResolver(context),  # type: ignore[arg-type]
        artifacts,
        journal,  # type: ignore[arg-type]
        FakeEvaluator(),  # type: ignore[arg-type]
        request_builder,  # type: ignore[arg-type]
        wait_timeout_s=90,
    )
    result = await runner.run_pair(
        incumbent,
        candidate,
        repeats=repeats,
        purpose=KernelMeasurementPurpose.KERNEL_RETENTION,
        per_run_timeout_seconds=100,
        allocation_timeout_seconds=allocation_timeout_seconds,
        shape_batch_size=shape_batch_size,
        max_parallel_shape_batches=2,
    )
    if replay:
        replayed = await runner.run_pair(
            incumbent,
            candidate,
            repeats=repeats,
            purpose=KernelMeasurementPurpose.KERNEL_RETENTION,
            per_run_timeout_seconds=100,
            allocation_timeout_seconds=allocation_timeout_seconds,
            shape_batch_size=shape_batch_size,
            max_parallel_shape_batches=2,
        )
        assert replayed.gateway_result_digest == result.gateway_result_digest
    return result, journal


def _source_contract() -> KernelSourceContract:
    return KernelSourceContract(
        source_revision="a" * 40,
        seed_digest=digest("native-source-seed"),
        package_root=".",
        editable_roots=("kernel.py", "helpers"),
        immutable_files={},
    )


@pytest.mark.anyio
async def test_authoritative_multi_file_abba_uses_native_eval_and_reuses_batches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    def no_dev_evaluator(_self: FakeEvaluator) -> dict[str, str]:
        pytest.fail("native source-tree ABBA must not export a Dev evaluator")

    monkeypatch.setattr(FakeEvaluator, "files", no_dev_evaluator)
    client = FakeAgateClient()
    result, journal = await _run_pair(
        client,
        tmp_path,
        repeats=4,
        request_builder=build_eval_request_from_content,
        source_tree=True,
        replay=True,
        allocation_timeout_seconds=900,
    )

    assert len(client.requests) == 2  # Five Shapes, batches of three; replay uses the cache.
    assert len({request["idempotency_key"] for request in client.requests}) == 2
    assert sorted(len(request["reference"]["shapes"]) for request in client.requests) == [2, 3]
    for request in client.requests:
        assert request["candidate"] == {
            "archive": "archives/candidate.tar.gz", "entry_point": "kernel.py"
        }
        assert request["abba"] == {
            "baseline": {"archive": "archives/baseline.tar.gz", "entry_point": "kernel.py"},
            "repeats": 2,
        }
        archives = request["__atrex_eval_archives"]
        assert archives["archives/baseline.tar.gz"]["helpers/impl.py"] == "SIDE = 'incumbent'\n"
        assert archives["archives/candidate.tar.gz"]["helpers/impl.py"] == "SIDE = 'candidate'\n"
        assert request["lock_clocks"] is True
        assert "command" not in request and "files" not in request
    assert [run.latency_us for run in result.incumbent_runs] == pytest.approx([100] * 4)
    assert [run.latency_us for run in result.candidate_runs] == pytest.approx([90] * 4)
    assert len(journal.measurements) == 8
    assert sum(kind == "comparison.abba_batch_reused" for kind, _, _ in journal.events) == 2


@pytest.mark.parametrize("tree_side", ("baseline", "candidate"))
def test_native_abba_supports_mixed_inline_and_archive_sources(tree_side: str) -> None:
    tree = KernelSourceBundle(
        {"kernel.py": "from helpers.impl import run\n", "helpers/impl.py": "def run(): pass\n"},
        _source_contract(),
        "kernel.py",
    )
    incumbent = tree if tree_side == "baseline" else "BASELINE = True\n"
    candidate = tree if tree_side == "candidate" else "CANDIDATE = True\n"
    assert _uses_native_abba(_native_request_builder, incumbent, candidate, 2)
    contract = AgateEvaluationContractV1(
        candidate_path="kernel.py", reference_py="REFERENCE = True\n", input_py="INPUT = True\n",
        shapes={"s": [1]},
        options=AgateEvaluationOptionsV1(
            num_correctness_cases=1, bench_iters=10, atol=0.01, rtol=0.01, timeout_s=60
        ),
    )
    request = build_native_abba_request(
        build_eval_request_from_content,
        hardware_target="L20D", operator="mixed", dsl="triton", contract=contract,
        shape_ids=["s"], repeats=2, incumbent_source=incumbent, candidate_source=candidate,
        allocation_timeout_seconds=600, name="native-mixed-abba",
    )
    assert request["__atrex_eval_archives"] == {f"archives/{tree_side}.tar.gz": tree.files}
    if tree_side == "baseline":
        assert request["candidate"] == candidate
        assert request["abba"]["baseline"]["entry_point"] == "kernel.py"
    else:
        assert request["abba"]["baseline"] == incumbent
        assert request["candidate"]["entry_point"] == "kernel.py"


@pytest.mark.parametrize("repeats", (0, 1, 3, 17, 18, 20, True, False, 2.0, "2", None))
@pytest.mark.parametrize("source_tree", (False, True))
@pytest.mark.parametrize("has_builder", (False, True))
def test_abba_never_falls_back_to_dev_for_unsupported_schedule(
    repeats: object, source_tree: bool, has_builder: bool,
) -> None:
    tree = KernelSourceBundle({"kernel.py": "SOURCE = True\n"}, _source_contract(), "kernel.py")
    source = tree if source_tree else "SOURCE = True\n"
    with pytest.raises(ValueError) as failure:
        _uses_native_abba(
            _native_request_builder if has_builder else None, source, "CANDIDATE = True\n", repeats
        )
    assert "2, 4, 6, 8, 10, 12, 14, 16" in str(failure.value)
    assert "measurements per side; 2 means A, B, B, A" in str(failure.value)
    assert f"got {repeats!r}" in str(failure.value)
    assert "not executed through Dev" in str(failure.value)


@pytest.mark.anyio
@pytest.mark.parametrize("repeats", (1, 3, 18, True, 2.0, "2"))
async def test_authoritative_abba_rejects_invalid_repeats_before_loading_sources(
    tmp_path: Path, repeats: object,
) -> None:
    class UnusedContext:
        def resolve(self, _revision: KernelRevision) -> AgateEvaluationContext:
            pytest.fail("invalid repeats must fail before resolving or loading a source")

    client = FakeAgateClient()
    runner = AgateSameAllocationAbbaRunner(
        client, UnusedContext(), LocalArtifactStore(tmp_path / "artifacts"),
        FakeJournal(), FakeEvaluator(), wait_timeout_s=90,
    )
    with pytest.raises(ValueError, match="not executed through Dev"):
        await runner.run_pair(
            None, None, repeats=repeats, purpose=KernelMeasurementPurpose.KERNEL_RETENTION,
            per_run_timeout_seconds=120, allocation_timeout_seconds=600,
            shape_batch_size=1, max_parallel_shape_batches=1,
        )
    assert client.requests == []


@pytest.mark.parametrize("source_tree", (False, True))
def test_abba_requires_native_request_builder_for_all_sources(source_tree: bool) -> None:
    tree = KernelSourceBundle({"kernel.py": "SOURCE = True\n"}, _source_contract(), "kernel.py")
    source = tree if source_tree else "SOURCE = True\n"
    with pytest.raises(ValueError) as failure:
        _uses_native_abba(None, source, source, 2)
    assert str(failure.value) == (
        "Native Agate Eval request builder is unavailable. ABBA requires native Eval; "
        "Dev fallback is disabled. Ask the Runtime operator to repair the Agate SDK configuration."
    )


@pytest.mark.anyio
async def test_authoritative_abba_rejects_missing_builder_before_loading_sources(
    tmp_path: Path,
) -> None:
    class UnusedContext:
        def resolve(self, _revision: KernelRevision) -> AgateEvaluationContext:
            pytest.fail("missing builder must fail before resolving or loading a source")

    client = FakeAgateClient()
    runner = AgateSameAllocationAbbaRunner(
        client, UnusedContext(), LocalArtifactStore(tmp_path / "artifacts"),
        FakeJournal(), FakeEvaluator(), wait_timeout_s=90,
    )
    with pytest.raises(ValueError, match="Dev fallback is disabled"):
        await runner.run_pair(
            None, None, repeats=2, purpose=KernelMeasurementPurpose.KERNEL_RETENTION,
            per_run_timeout_seconds=120, allocation_timeout_seconds=600,
            shape_batch_size=1, max_parallel_shape_batches=1,
        )
    assert client.requests == []


@pytest.mark.parametrize("invalid", ("order", "incomplete"))
def test_native_abba_parser_rejects_invalid_same_allocation_evidence(invalid: str) -> None:
    client = FakeAgateClient()
    accepted = client.submit_job(
        "eval", {"reference": {"shapes": {"s": [1]}}, "abba": {"repeats": 1}}
    )
    job = client.jobs[str(accepted["job_id"])]
    runs = job["result"]["abba"]["sdk_results"][0]["abba"]["runs"]
    if invalid == "order":
        runs[0], runs[1] = runs[1], runs[0]
    else:
        runs.pop()
    with pytest.raises(InfrastructureError, match=r"run order|incomplete runs"):
        _parse_native_abba_payload(job, _schedule(2), ["s"])


@pytest.mark.anyio
async def test_authoritative_single_file_abba_uses_native_agate_eval(tmp_path: Path) -> None:
    client = FakeAgateClient()

    result, journal = await _run_pair(
        client,
        tmp_path,
        shape_batch_size=5,
        repeats=2,
        request_builder=_native_request_builder,
    )

    assert len(client.requests) == 1
    request = client.requests[0]
    assert request["abba"] == {"baseline": "INCUMBENT = True\n", "repeats": 1}
    assert request["candidate"] == "CANDIDATE = True\n"
    assert request["lock_clocks"] is True
    assert request["options"]["timeout_s"] == 500
    assert [run.latency_us for run in result.incumbent_runs] == pytest.approx([100, 100])
    assert [run.latency_us for run in result.candidate_runs] == pytest.approx([90, 90])
    assert result.gateway_result_digest is not None
    aggregate = json.loads(
        (
            LocalArtifactStore(tmp_path / "artifacts")
            .verify(result.gateway_result_digest)
            .payload_path
            / "value.json"
        ).read_text(encoding="utf-8")
    )
    assert aggregate["execution_transport"] == "agate_native_eval_abba"
    assert aggregate["atrex_bench_commit"] is None
    assert any(kind == "comparison.abba_completed" for kind, _, _ in journal.events)


@pytest.mark.anyio
@pytest.mark.parametrize("source_tree", (False, True))
async def test_native_agate_abba_preserves_candidate_correctness_failure(
    tmp_path: Path, source_tree: bool,
) -> None:
    class IncorrectNativeClient(FakeAgateClient):
        def submit_job(self, kind: str, request: dict[str, object]) -> dict[str, object]:
            accepted = super().submit_job(kind, request)
            job = self.jobs[str(accepted["job_id"])]
            result = job["result"]
            assert isinstance(result, dict)
            comparison = result["abba"]
            assert isinstance(comparison, dict)
            comparison["valid"] = False
            for block in comparison["sdk_results"]:
                for row in block["abba"]["runs"]:
                    if row["revision"] == "candidate":
                        shape_id = next(iter(row["result"]["passed"]["correctness"]))
                        row["result"]["passed"]["correctness"][shape_id] = {
                            "status": "failed"
                        }
            return accepted

    client = IncorrectNativeClient()
    result, journal = await _run_pair(
        client,
        tmp_path,
        shape_batch_size=5,
        repeats=2,
        request_builder=_native_request_builder,
        source_tree=source_tree,
    )

    assert len(client.requests) == 1
    assert all(run.correct for run in result.incumbent_runs)
    assert all(not run.correct for run in result.candidate_runs)
    assert not any(kind == "comparison.abba_batch_retried" for kind, _, _ in journal.events)


@pytest.mark.anyio
async def test_terminal_infra_abba_batch_recovers_inside_shared_job_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    delays: list[float] = []

    async def sleep(delay: float) -> None:
        delays.append(delay)

    monkeypatch.setattr("atrex_runtime.gateway.job_recovery.anyio.sleep", sleep)
    client = FlakyAgateClient(10)

    result, journal = await _run_pair(client, tmp_path)

    assert client.remaining_failures == 0
    retries = [
        payload for kind, _, payload in journal.events if kind == "comparison.abba_batch_retried"
    ]
    assert retries == []  # The outer malformed-result retry budget is untouched.
    assert len(client.requests) == 12  # Two measured batches plus ten replacements.
    assert len(delays) == 10
    assert all(delay in {5, 10, 20, 40, 60} for delay in delays)
    assert any(kind == "comparison.abba_completed" for kind, _, _ in journal.events)
    assert result.gateway_result_digest is not None


@pytest.mark.anyio
async def test_terminal_infra_abba_batch_does_not_exhaust_outer_retry_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def sleep(_delay: float) -> None:
        return None

    monkeypatch.setattr("atrex_runtime.gateway.job_recovery.anyio.sleep", sleep)
    client = FlakyAgateClient(
        22,
        error={
            "error_class": "infra",
            "reason": "exec_failed",
            "message": "runtime_env setup failed: Could not create the actor",
        },
    )
    result, journal = await _run_pair(client, tmp_path)

    assert client.remaining_failures == 0
    assert len(client.requests) == 24
    assert all(run.correct for run in result.candidate_runs)
    assert not any(kind == "comparison.abba_batch_retried" for kind, _, _ in journal.events)


@pytest.mark.anyio
async def test_abba_negative_kernel_measurement_is_not_retried(tmp_path: Path) -> None:
    class IncorrectCandidateClient(FakeAgateClient):
        def submit_job(self, kind: str, request: dict[str, object]) -> dict[str, object]:
            accepted = super().submit_job(kind, request)
            job_id = str(accepted["job_id"])
            result = self.jobs[job_id]["result"]
            assert isinstance(result, dict)
            for block in result["abba"]["sdk_results"]:
                for run in block["abba"]["runs"]:
                    if run["revision"] == "candidate":
                        for shape in run["result"]["passed"]["correctness"].values():
                            shape["status"] = "failed"
            return accepted

    client = IncorrectCandidateClient()
    result, journal = await _run_pair(client, tmp_path)

    assert len(client.requests) == 2
    assert all(run.correct is False for run in result.candidate_runs)
    assert not any(kind == "comparison.abba_batch_retried" for kind, _, _ in journal.events)


@pytest.mark.anyio
async def test_abba_poll_error_retries_with_a_fresh_job(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("atrex_runtime.gateway.abba._ABBA_RETRY_DELAY_SECONDS", 0.0)

    class MissingAcceptedJobClient(FakeAgateClient):
        failed_job_id: str | None = None

        def submit_job(self, kind: str, request: dict[str, object]) -> dict[str, object]:
            accepted = super().submit_job(kind, request)
            if self.failed_job_id is None:
                self.failed_job_id = str(accepted["job_id"])
            return accepted

        def get_job(
            self,
            job_id: str,
            wait: bool = False,
            timeout: float = 30.0,
            include_spec: bool = False,
        ) -> dict[str, object]:
            if job_id == self.failed_job_id:
                raise RuntimeError(
                    "four transient 502 responses followed by 404 no job 'dv_f70805eb9398'"
                )
            return super().get_job(job_id, wait, timeout, include_spec)

    client = MissingAcceptedJobClient()
    result, journal = await _run_pair(client, tmp_path)

    assert len(client.requests) == 3
    assert client.failed_job_id == "ev_abba_0"
    retries = [
        payload for kind, _, payload in journal.events if kind == "comparison.abba_batch_retried"
    ]
    assert len(retries) == 1
    assert retries[0]["error_class"] is None
    assert retries[0]["reason"] is None
    assert retries[0]["trace_id"] is None
    assert retries[0]["retryable"] is True
    assert retries[0]["failure_type"] == "InfrastructureError"
    assert "404 no job 'dv_f70805eb9398'" in str(retries[0]["detail"])
    assert result.gateway_result_digest is not None


@pytest.mark.anyio
@pytest.mark.parametrize("reason", ["logs_unavailable", "exec_failed"])
async def test_abba_terminal_infra_resubmits_beyond_general_retry_ceiling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    reason: str,
) -> None:
    delays: list[float] = []
    polled: list[str] = []

    async def sleep(delay: float) -> None:
        delays.append(delay)

    monkeypatch.setattr("atrex_runtime.gateway.job_recovery.anyio.sleep", sleep)

    class LostLogsClient(FakeAgateClient):
        def get_job(
            self,
            job_id: str,
            wait: bool = False,
            timeout: float = 30,
            include_spec: bool = False,
        ) -> dict[str, object]:
            polled.append(job_id)
            if int(job_id.rsplit("_", 1)[1]) < 11:
                return {
                    "job_id": job_id,
                    "status": "failed",
                    "error": {
                        "error_class": "infra",
                        "reason": reason,
                        "details": {"backend_state": "succeeded"},
                    },
                }
            return super().get_job(job_id, wait, timeout, include_spec)

    client = LostLogsClient()
    result, journal = await _run_pair(client, tmp_path)
    assert len(client.requests) == 13  # Two successful batches and eleven lost executions.
    assert len(polled) == len(set(polled)) == 13
    assert len(delays) == 11 and max(delays) == 60
    replacements = [r for r in client.requests if "idempotency_key" in r]
    assert len({r["idempotency_key"] for r in replacements}) == 13
    assert len([e for e in journal.events if e[0] == "comparison.abba_batch_submitted"]) == 13
    assert not any(e[0] == "comparison.abba_batch_retried" for e in journal.events)
    assert all(run.correct for run in result.candidate_runs)


@pytest.mark.anyio
@pytest.mark.parametrize(
    "failure", ("submit_error", "missing_job_id", "nonterminal", "malformed_payload")
)
async def test_abba_batch_infrastructure_errors_are_retried(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    monkeypatch.setattr("atrex_runtime.gateway.abba._ABBA_RETRY_DELAY_SECONDS", 0.0)

    class OneFailureClient(FakeAgateClient):
        failed = False
        failure_lock = threading.Lock()

        def submit_job(self, kind: str, request: dict[str, object]) -> dict[str, object]:
            accepted = super().submit_job(kind, request)
            with self.failure_lock:
                fail = not self.failed
                if fail:
                    self.failed = True
            if not fail:
                return accepted
            if failure == "submit_error":
                raise RuntimeError("Agate submit response was lost after acceptance")
            job_id = str(accepted["job_id"])
            if failure == "missing_job_id":
                return {"status": "queued"}
            if failure == "nonterminal":
                self.jobs[job_id] = {"job_id": job_id, "status": "running"}
            else:
                self.jobs[job_id]["result"] = {
                    "abba": {"sdk_results": []},
                }
            return accepted

    client = OneFailureClient()
    result, journal = await _run_pair(client, tmp_path)

    assert len(client.requests) == 3
    retries = [
        payload for kind, _, payload in journal.events if kind == "comparison.abba_batch_retried"
    ]
    assert len(retries) == 1
    assert retries[0]["failure_type"] == "InfrastructureError"
    assert retries[0]["retryable"] is True
    assert result.gateway_result_digest is not None


@pytest.mark.anyio
async def test_abba_infrastructure_error_gives_up_past_the_ceiling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("atrex_runtime.gateway.abba._ABBA_RETRY_DELAY_SECONDS", 0.0)

    class UnavailableAgateClient(FakeAgateClient):
        def get_job(
            self,
            job_id: str,
            wait: bool = False,
            timeout: float = 30.0,
            include_spec: bool = False,
        ) -> dict[str, object]:
            raise RuntimeError(f"404 no job {job_id}")

    client = UnavailableAgateClient()
    with pytest.raises(BaseExceptionGroup) as caught:
        await _run_pair(client, tmp_path, shape_batch_size=5)

    assert len(client.requests) == 11
    assert all(isinstance(error, InfrastructureError) for error in caught.value.exceptions)


@pytest.mark.anyio
async def test_failed_abba_command_is_retried(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("atrex_runtime.gateway.abba._ABBA_RETRY_DELAY_SECONDS", 0.0)

    class CommandFailureClient(FakeAgateClient):
        failed = False

        def submit_job(self, kind: str, request: dict[str, object]) -> dict[str, object]:
            accepted = super().submit_job(kind, request)
            if not self.failed:
                self.failed = True
                job_id = str(accepted["job_id"])
                self.jobs[job_id] = {
                    "job_id": job_id,
                    "status": "succeeded",
                    "command_ok": False,
                    "error": {"error_class": "user", "reason": "nonzero_exit"},
                }
            return accepted

    client = CommandFailureClient()
    result, journal = await _run_pair(client, tmp_path)

    assert len(client.requests) == 3
    retries = [
        payload for kind, _, payload in journal.events if kind == "comparison.abba_batch_retried"
    ]
    assert len(retries) == 1
    assert retries[0]["error_class"] == "user"
    assert retries[0]["reason"] == "nonzero_exit"
    assert retries[0]["retryable"] is True
    assert result.gateway_result_digest is not None
