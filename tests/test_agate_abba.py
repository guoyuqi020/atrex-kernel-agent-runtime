"""Trusted same-allocation ABBA runner tests."""

from __future__ import annotations

import json
import os
import subprocess
import threading
from pathlib import Path

import pytest
from conftest import NOW, digest

from atrex_runtime.artifacts.local import ArtifactKind, LocalArtifactStore
from atrex_runtime.domain.ids import new_kernel_revision_id
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
)
from atrex_runtime.gateway.contract import (
    AgateEvaluationContext,
    AgateEvaluationContractV1,
    AgateEvaluationOptionsV1,
)
from atrex_runtime.gateway.result_metrics import gateway_result_sol_summary


class FakeContextResolver:
    def __init__(self, context: AgateEvaluationContext) -> None:
        self.context = context

    def resolve(self, _revision: KernelRevision) -> AgateEvaluationContext:
        return self.context


class FakeEvaluator:
    commit = "f" * 40

    def files(self) -> dict[str, str]:
        return {"atrex-bench/src/atrex_bench/__init__.py": ""}

    def bundle_digest(self) -> str:
        return "sha256:" + "e" * 64


class FakeJournal:
    def __init__(self) -> None:
        self.measurements: list[KernelMeasurement] = []
        self.events: list[tuple[str, str, object]] = []

    def record_kernel_measurement(self, measurement: KernelMeasurement) -> KernelMeasurement:
        self.measurements.append(measurement)
        return measurement

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
        assert kind == "dev"
        files = request["files"]
        assert isinstance(files, dict)
        abba_request = json.loads(files["request.json"])
        with self._lock:
            job_id = f"dv_abba_{len(self.requests)}"
            self.requests.append(request)
        runs = []
        for step in abba_request["schedule"]:
            revision = step["revision"]
            latency = 100.0 if revision == "incumbent" else 90.0
            runs.append(
                {
                    **step,
                    "exit_code": 0,
                    "result": {
                        "all_pass": True,
                        "latency_us_geomean": latency,
                        "latency_us_by_shape": {
                            shape_id: latency for shape_id in abba_request["shape_ids"]
                        },
                        "sol_pct_by_shape": {
                            shape_id: 50.0 if revision == "candidate" else 25.0
                            for shape_id in abba_request["shape_ids"]
                        },
                    },
                    "stdout_tail": "",
                    "stderr_tail": "",
                }
            )
        payload = {"schema_version": 1, "runs": runs, "error": None}
        self.jobs[job_id] = {
            "job_id": job_id,
            "status": "succeeded",
            "command_ok": True,
            "result": {
                "stdout": "__ATREX_RUNTIME_ABBA_RESULT__="
                + json.dumps(payload, separators=(",", ":")),
                "stderr": "",
                "exit_code": 0,
            },
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
async def test_abba_runner_uses_one_dev_allocation_per_shape_batch_and_records_runs(
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
    assert all(
        request["command"] == "python3 __atrex_abba.py request.json" for request in client.requests
    )
    assert all(
        json.loads(request["files"]["request.json"])["lock_clocks"] is lock_clocks
        for request in client.requests
    )
    assert all(
        json.loads(request["files"]["request.json"])["evaluator"]["clock_lock_mode"]
        == ("external" if lock_clocks else "off")
        for request in client.requests
    )
    assert [run.latency_us for run in result.incumbent_runs] == pytest.approx([100, 100])
    assert [run.latency_us for run in result.candidate_runs] == pytest.approx([90, 90])
    assert len(journal.measurements) == 4
    assert len({measurement.gateway_result_digest for measurement in journal.measurements}) == 1
    assert result.gateway_result_digest == journal.measurements[0].gateway_result_digest
    assert all(
        measurement.agate_job_id == "dv_abba_0,dv_abba_1" for measurement in journal.measurements
    )
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
