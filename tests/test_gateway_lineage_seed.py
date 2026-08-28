"""Trusted Agate evaluation used when an existing Kernel becomes Lineage v0."""

from __future__ import annotations

from pathlib import Path

import pytest
from conftest import seed_lineage

from atrex_runtime.artifacts.local import ArtifactKind, LocalArtifactStore
from atrex_runtime.domain.ids import new_lineage_id
from atrex_runtime.domain.models import Dsl
from atrex_runtime.gateway.contract import (
    AgateEvaluationContractV1,
    AgateEvaluationOptionsV1,
)
from atrex_runtime.gateway.lineage_seed import AgateLineageSeedEvaluator
from atrex_runtime.registry.sqlite import SqliteRegistry


class FakeAgateClient:
    def __init__(self) -> None:
        self.submissions: list[tuple[str, dict[str, object]]] = []

    def submit_job(self, kind: str, request: dict[str, object]) -> dict[str, object]:
        self.submissions.append((kind, request))
        return {"job_id": f"job-{kind}", "status": "queued"}

    def get_job(
        self,
        job_id: str,
        wait: bool = False,
        timeout: float = 30.0,
        include_spec: bool = False,
    ) -> dict[str, object]:
        assert wait is True
        assert timeout == 90
        assert include_spec is False
        if job_id == "job-profile":
            return {"job_id": job_id, "status": "succeeded", "result": {"sol": 71.0}}
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
async def test_lineage_seed_evaluation_uses_campaign_contract_and_profiles_sol(
    tmp_path: Path,
) -> None:
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    contract = AgateEvaluationContractV1(
        candidate_path="kernel.py",
        reference_py="def reference(): pass",
        input_py="def _make_inputs(): return ()",
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
    source = tmp_path / "kernel"
    source.mkdir()
    (source / "kernel.py").write_text("class Model: pass\n", encoding="utf-8")
    kernel_digest = artifacts.put_directory(source, ArtifactKind.KERNEL)
    with SqliteRegistry(tmp_path / "registry.sqlite") as registry:
        seeded = seed_lineage(registry, evaluation_contract_digest=contract_digest)
        campaign_id = registry.get_lineage(seeded.lineage_id).campaign_id
        client = FakeAgateClient()
        builder = CapturingBuilder()
        evaluator = AgateLineageSeedEvaluator(
            client,
            builder,  # type: ignore[arg-type]
            registry,
            artifacts,
            registry,
            wait_timeout_s=90,
        )

        result = await evaluator.evaluate(
            campaign_id=campaign_id,
            lineage_id=new_lineage_id(),
            dsl=Dsl.TRITON,
            kernel_artifact_digest=kernel_digest,
        )

        assert result.correct is True
        assert result.latency_us == 17.5
        assert [kind for kind, _ in client.submissions] == ["eval", "profile"]
        references = [
            payload["reference"] for kind, payload in client.submissions if kind == "eval"
        ]
        assert [len(reference["shapes"]) for reference in references] == [5]  # type: ignore[index]
        assert builder.payloads[0]["candidate"] == "class Model: pass\n"
        assert builder.payloads[0]["gpu"] == "nvidia-h100"
        assert builder.payloads[0]["spec_fields"] == {"languages": ["triton"]}
        stored = artifacts.verify(result.gateway_result_digest)
        assert (stored.payload_path / "value.json").is_file()
        assert (stored.payload_path / "profile.json").is_file()
