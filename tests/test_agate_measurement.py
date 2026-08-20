"""Trusted repeated-Evaluate runner tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from conftest import seed_lineage

from atrex_runtime.artifacts.local import ArtifactKind, LocalArtifactStore
from atrex_runtime.domain.models import KernelMeasurementPurpose
from atrex_runtime.gateway.contract import (
    AgateEvaluationContractV1,
    AgateEvaluationOptionsV1,
    RegistryKernelEvaluationContextResolver,
)
from atrex_runtime.gateway.measurement import AgateKernelMeasurementRunner
from atrex_runtime.registry.sqlite import SqliteRegistry


class FakeAgateClient:
    def __init__(self) -> None:
        self.submissions: list[tuple[str, dict[str, object]]] = []

    def submit_job(self, kind: str, request: dict[str, object]) -> dict[str, object]:
        self.submissions.append((kind, request))
        return {"job_id": "ev_measurement", "status": "queued"}

    def get_job(
        self,
        job_id: str,
        wait: bool = False,
        timeout: float = 30.0,
        include_spec: bool = False,
    ) -> dict[str, object]:
        assert (job_id, wait, timeout, include_spec) == (
            "ev_measurement",
            True,
            90.0,
            False,
        )
        return {
            "job_id": job_id,
            "status": "succeeded",
            "result": {"all_pass": True, "latency_us_geomean": 17.5},
        }


class CapturingBuilder:
    def __init__(self) -> None:
        self.payloads: list[dict[str, object]] = []

    def __call__(
        self, candidate: str, reference: object, gpu: str, **kwargs: object
    ) -> dict[str, object]:
        payload = {
            "candidate": candidate,
            "reference": reference,
            "gpu": gpu,
            **kwargs,
        }
        self.payloads.append(payload)
        return payload


@pytest.mark.anyio
async def test_repeated_evaluate_runner_uses_sealed_kernel_and_campaign_contract(
    tmp_path: Path,
) -> None:
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    contract = AgateEvaluationContractV1(
        candidate_path="kernel.py",
        reference_py="def reference(): pass",
        input_py="def get_inputs(): return ()",
        shapes={str(index): [index] for index in range(5)},
        options=AgateEvaluationOptionsV1(
            num_correctness_cases=1,
            bench_iters=10,
            atol=0.01,
            rtol=0.01,
            timeout_s=60,
        ),
        lock_clocks=True,
    )
    contract_digest = artifacts.put_json(
        contract.model_dump(mode="json"),
        ArtifactKind.EVALUATION_CONTRACT,
    )
    kernel_source = tmp_path / "kernel"
    kernel_source.mkdir()
    (kernel_source / "kernel.py").write_text("class Model: pass\n", encoding="utf-8")
    kernel_digest = artifacts.put_directory(kernel_source, ArtifactKind.KERNEL)
    registry = SqliteRegistry(tmp_path / "registry.sqlite")
    try:
        seeded = seed_lineage(
            registry,
            evaluation_contract_digest=contract_digest,
            kernel_artifact_digest=kernel_digest,
        )
        revision = seeded.baseline
        client = FakeAgateClient()
        builder = CapturingBuilder()
        runner = AgateKernelMeasurementRunner(
            client,
            builder,  # type: ignore[arg-type]
            RegistryKernelEvaluationContextResolver(registry, artifacts),
            artifacts,
            registry,
            wait_timeout_s=90,
        )

        first = await runner.run(
            revision,
            0,
            KernelMeasurementPurpose.KERNEL_RETENTION,
        )
        second = await runner.run(
            revision,
            1,
            KernelMeasurementPurpose.KERNEL_RETENTION,
        )
        aggregate = runner.aggregate(
            revision,
            (first, second),
            KernelMeasurementPurpose.KERNEL_RETENTION,
        )

        assert first.repeat == 0
        assert first.correct is True
        assert first.latency_us == 17.5
        assert first.gateway_result_digest is not None
        assert first.agate_job_id is None
        assert len(client.submissions) == 4
        references = [payload["reference"] for _kind, payload in client.submissions]
        assert sorted(len(reference["shapes"]) for reference in references) == [1, 1, 4, 4]  # type: ignore[index]
        assert client.submissions[0][0] == "eval"
        assert builder.payloads[0]["candidate"] == "class Model: pass\n"
        assert builder.payloads[0]["gpu"] == "nvidia-h100"
        assert builder.payloads[0]["spec_fields"] == {"languages": ["triton"]}
        events = registry.list_runtime_events(after_sequence=0, limit=100)
        assert [
            event.kind for event in events if event.kind.startswith("comparison.measurement_")
        ] == [
            "comparison.measurement_submitted",
            "comparison.measurement_submitted",
            "comparison.measurement_completed",
            "comparison.measurement_submitted",
            "comparison.measurement_submitted",
            "comparison.measurement_completed",
        ]
        measurements = registry.list_kernel_measurements(revision.id)
        assert len(measurements) == 2
        assert measurements[0].purpose is KernelMeasurementPurpose.KERNEL_RETENTION
        assert measurements[0].gateway_result_digest is not None
        aggregate_value = json.loads(
            (artifacts.verify(aggregate).payload_path / "value.json").read_text()
        )
        assert aggregate_value["aggregation"] == "arithmetic_mean"
        assert aggregate_value["latency_us"] == 17.5
        assert len(aggregate_value["measurements"]) == 2
    finally:
        registry.close()
