"""All-Valid inputs, legacy split compatibility, and safe result projections."""

from __future__ import annotations

import json
import random
from dataclasses import replace
from pathlib import Path

import pytest
from atrex_gateway_client import build_eval_request_from_content
from conftest import NOW, digest
from pydantic import ValidationError
from test_agate_abba import FakeAgateClient as AbbaClient
from test_agate_abba import FakeContextResolver, FakeEvaluator, FakeJournal
from test_agate_gateway_adapter import (
    CapturingBuilder,
    FakeAgateClient,
    StaticContexts,
    _successful_job,
)
from test_agent_abba import Case
from test_agent_abba import case as case
from test_problem_generalization_worker import _write_agent

from atrex_runtime.artifacts.local import ArtifactKind, LocalArtifactStore
from atrex_runtime.domain.ids import new_attempt_id, new_kernel_revision_id
from atrex_runtime.domain.models import (
    Dsl,
    KernelEvaluation,
    KernelMeasurementPurpose,
    KernelRevision,
)
from atrex_runtime.gateway.abba import AgateSameAllocationAbbaRunner
from atrex_runtime.gateway.agate import AgateGatewayAdapter, SqliteAgateJobStore
from atrex_runtime.gateway.contract import AgateEvaluationContext, AgateEvaluationContractV1
from atrex_runtime.gateway.control_models import GatewayOperation
from atrex_runtime.gateway.private_results import project_private_job
from atrex_runtime.gateway.proxy import GatewayAdapterRequest
from atrex_runtime.gateway.result_metrics import gateway_result_projection
from atrex_runtime.workers.evidence_view import _latest_evolver_epoch_facts
from atrex_runtime.workers.problem_generalization import (
    ProblemGeneralizationManifestV1,
    ProblemGeneralizationWorkspaceAssembler,
)


def _contract(count: int = 10) -> AgateEvaluationContractV1:
    return AgateEvaluationContractV1.model_validate(
        {
            "candidate_path": "kernel.py",
            "reference_py": "class Model: pass\n",
            "input_py": "def _make_inputs(n): return (n,)\n",
            "shapes": {str(i): {"input_kwargs": {"n": i + 11}} for i in range(count)},
            "metadata": {
                "num_shapes": count,
                "shapes": {str(i): {"meta": i} for i in range(count)},
                "trace": {"test_secret": "private-shape-trace"},
                "benchmark_contract": {"mutates_inputs": ["out"]},
            },
            "roofline": {"shapes": {str(i): {"bound": i} for i in range(count)}},
            "options": {
                "num_correctness_cases": 5,
                "bench_iters": 100,
                "atol": 0.01,
                "rtol": 0.05,
                "timeout_s": 120,
            },
        }
    )


@pytest.mark.parametrize("count", [1, 2, 3, 10, 11, 29, 30, 31, 32, 45, 100])
def test_all_shapes_are_stable_valid_and_preserved_in_sealed_contract(count: int) -> None:
    original = _contract(count)
    split = original.with_shape_holdout()
    reordered = original.model_copy(update={"shapes": dict(reversed(original.shapes.items()))})
    assert split == reordered.with_shape_holdout()
    assert split.with_shape_holdout() == split
    restored = AgateEvaluationContractV1.model_validate_json(split.model_dump_json())
    assert restored == split
    assert split.shape_split is not None
    record = split.shape_split
    assert record.algorithm == "all_valid"
    assert record.seed is None
    assert record.max_shapes_per_set is None
    assert record.source_shape_count == count
    assert record.source_shape_ids == tuple(sorted(original.shapes))
    assert set(record.valid_shape_ids) == set(split.validation_shape_ids)
    assert set(record.valid_shape_ids) | set(record.test_shape_ids) == set(split.shapes)
    assert record.valid_shape_ids == tuple(sorted(original.shapes))
    assert record.test_shape_ids == ()
    assert len(split.shapes) == count
    assert len(split.validation_shape_ids or ()) == count
    assert set(split.shapes) == set(original.shapes)
    assert split.metadata is not None and split.roofline is not None
    assert split.metadata["num_shapes"] == count
    assert set(split.metadata["shapes"]) == set(split.shapes)
    assert set(split.roofline["shapes"]) == set(split.shapes)
    assert record.agent_shape_id_map is not None
    assert set(record.agent_shape_id_map) == {str(index) for index in range(count)}
    assert set(record.agent_shape_id_map.values()) == set(record.valid_shape_ids)
    valid = split.for_agent()
    assert len(valid.shapes) == count
    assert set(valid.shapes) == set(record.agent_shape_id_map)
    assert valid.metadata is not None and valid.roofline is not None
    assert valid.metadata["num_shapes"] == count
    assert set(valid.metadata["shapes"]) == set(valid.shapes)
    assert set(valid.roofline["shapes"]) == set(valid.shapes)
    assert "private-shape-trace" not in json.dumps(valid.metadata)
    assert valid.metadata["benchmark_contract"] == {"mutates_inputs": ["out"]}
    assert valid.validation_shape_ids is None
    assert valid.shape_split is None
    assert "shape_split" not in valid.model_dump()
    assert "source_shape_ids" not in valid.model_dump_json()
    assert original.validation_shape_ids is None
    assert len(original.shapes) == count
    assert original.metadata["num_shapes"] == count


def test_all_valid_selection_does_not_modify_global_random_state() -> None:
    before = random.getstate()
    _contract(100).with_shape_holdout()
    assert random.getstate() == before


def test_legacy_fixed_seed_split_contract_remains_loadable() -> None:
    value = _contract(10).model_dump(mode="json")
    value["validation_shape_ids"] = [str(index) for index in range(5)]
    value["shape_split"] = {
        "algorithm": "python_random_shuffle_sample",
        "seed": 42,
        "max_shapes_per_set": 15,
        "source_shape_count": 10,
        "source_shape_ids": [str(index) for index in range(10)],
        "valid_shape_ids": [str(index) for index in range(5)],
        "test_shape_ids": [str(index) for index in range(5, 10)],
        "agent_shape_id_map": {str(index): str(index) for index in range(5)},
    }
    restored = AgateEvaluationContractV1.model_validate(value)
    assert restored.shape_split is not None
    assert restored.shape_split.algorithm == "python_random_shuffle_sample"


def test_split_archive_is_not_in_agent_result_projection() -> None:
    split = _contract(40).with_shape_holdout()
    assert split.shape_split is not None
    assert project_private_job(
        {
            "status": "completed",
            "shape_split": split.shape_split.model_dump(mode="json"),
            "test_observation": {
                "status": "completed",
                "candidate": {"correct": False, "latency_us": 123456},
            },
        }
    ) == {"status": "completed"}


def test_archived_selection_must_match_the_sealed_contract() -> None:
    split = _contract(40).with_shape_holdout()
    assert split.shape_split is not None
    value = split.model_dump(mode="json")
    moved = value["shape_split"]["valid_shape_ids"].pop()
    value["shape_split"]["test_shape_ids"] = [moved]
    with pytest.raises(ValidationError, match="every source Shape in Valid"):
        AgateEvaluationContractV1.model_validate(value)


def test_archived_agent_shape_id_map_must_be_contiguous_and_exact() -> None:
    split = _contract(40).with_shape_holdout()
    value = split.model_dump(mode="json")
    value["shape_split"]["agent_shape_id_map"]["0"] = value["shape_split"][
        "agent_shape_id_map"
    ]["1"]
    with pytest.raises(ValidationError, match="contiguous opaque Agent IDs"):
        AgateEvaluationContractV1.model_validate(value)


def test_sealed_all_valid_population_has_no_shape_cap() -> None:
    split = _contract(100).with_shape_holdout()
    restored = AgateEvaluationContractV1.model_validate_json(split.model_dump_json())
    assert len(restored.shapes) == 100
    assert restored.validation_shape_ids is not None
    assert len(restored.validation_shape_ids) == 100


def test_single_shape_is_valid() -> None:
    split = _contract(1).with_shape_holdout()
    assert split.validation_shape_ids == ("0",)
    assert split.shape_split is not None and split.shape_split.test_shape_ids == ()


@pytest.mark.parametrize("ids", [[], ["0", "0"], ["0", "1", "2"], ["unknown", "0"]])
def test_invalid_sealed_partition_is_rejected(ids: list[str]) -> None:
    value = _contract(4).model_dump()
    value["validation_shape_ids"] = ids
    with pytest.raises(ValidationError, match="validation_shape_ids"):
        AgateEvaluationContractV1.model_validate(value)


@pytest.mark.anyio
async def test_agent_eval_sends_and_returns_only_valid_shapes(tmp_path: Path) -> None:
    contract = _contract(2).with_shape_holdout()
    contexts = StaticContexts(AgateEvaluationContext("vecadd", "H20", Dsl.TRITON, contract))
    client = FakeAgateClient(_successful_job())
    builder = CapturingBuilder()
    jobs = SqliteAgateJobStore(tmp_path / "jobs.sqlite")
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    (candidate / "kernel.py").write_text("class Model: pass\n")
    adapter = AgateGatewayAdapter(client, builder, contexts, jobs, wait_timeout_s=90)
    request = GatewayAdapterRequest(
        new_attempt_id(),
        GatewayOperation.EVALUATE,
        "valid-only",
        digest("kernel"),
        candidate,
        None,
        None,
        None,
    )
    try:
        result = await adapter.execute(request)
        assert result.evaluation is not None and result.evaluation.correct
        sent = [payload["reference"]["shapes"] for _, payload in client.submitted]
        aliases = contract.agent_shape_id_map()
        assert aliases is not None
        assert len(sent) == len(aliases)
        assert {shape_id for batch in sent for shape_id in batch} == set(aliases)
        assert "private-shape-trace" not in json.dumps(client.submitted)
        assert set(result.worker_result["latency_us_by_shape"]) == set(aliases)
        invalid_id = "not-an-agent-shape"
        with pytest.raises(ValueError, match="not an evaluator-owned"):
            await adapter.execute(
                replace(
                    request,
                    operation=GatewayOperation.PROFILE,
                    parameters={"shape_id": invalid_id},
                    profile_level="sol",
                    idempotency_key="cannot-probe-test",
                )
            )
        assert len(client.submitted) == len(aliases)
    finally:
        jobs.close()


@pytest.mark.anyio
async def test_agent_profile_uses_opaque_valid_shape_id_end_to_end(tmp_path: Path) -> None:
    original = _contract(40)
    source_ids = {str(index): str(index + 100) for index in range(40)}
    metadata = dict(original.metadata or {})
    metadata["shapes"] = {
        source_ids[shape_id]: value
        for shape_id, value in (original.metadata or {})["shapes"].items()
    }
    roofline = {
        "shapes": {
            source_ids[shape_id]: value
            for shape_id, value in (original.roofline or {})["shapes"].items()
        }
    }
    contract = original.model_copy(
        update={
            "shapes": {source_ids[shape_id]: value for shape_id, value in original.shapes.items()},
            "metadata": metadata,
            "roofline": roofline,
        }
    ).with_shape_holdout()
    aliases = contract.agent_shape_id_map()
    assert aliases is not None
    alias, source_id = next(
        (alias, source_id)
        for alias, source_id in aliases.items()
        if alias != source_id
    )
    contexts = StaticContexts(AgateEvaluationContext("vecadd", "H20", Dsl.TRITON, contract))
    client = FakeAgateClient(
        {
            "job_id": "pf_opaque",
            "status": "succeeded",
            "spec": {
                "reference": {
                    "shapes": {
                        alias: {"input_kwargs": {"private_source_shape_id": source_id}}
                    }
                }
            },
            "result": {"kernels": [{"name": "vector_add", "duration": 4.0}]},
        }
    )
    builder = CapturingBuilder()
    jobs = SqliteAgateJobStore(tmp_path / "jobs.sqlite")
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    candidate.joinpath("kernel.py").write_text("class Model: pass\n")
    adapter = AgateGatewayAdapter(client, builder, contexts, jobs, wait_timeout_s=90)
    request = GatewayAdapterRequest(
        new_attempt_id(),
        GatewayOperation.PROFILE,
        "profile-opaque-valid-shape",
        digest("kernel"),
        candidate,
        "survey",
        None,
        None,
        {"shape_id": alias},
    )
    try:
        result = await adapter.execute(request)
        reference = builder.calls[0]["reference"]
        assert isinstance(reference, dict)
        assert reference["shapes"] == {alias: contract.shapes[source_id]}
        assert source_id not in reference["shapes"]
        assert result.worker_result == {
            "job_id": "pf_opaque",
            "status": "succeeded",
            "result": {
                "shape_id": alias,
                "kernels": [{"name": "vector_add", "duration": 4.0}],
            },
        }
        assert source_id not in json.dumps(result.worker_result)
        assert "private_source_shape_id" not in json.dumps(result.worker_result)
    finally:
        jobs.close()


@pytest.mark.anyio
async def test_agent_abba_uses_only_valid_inputs(case: Case) -> None:
    contract = case.contexts.context.contract.with_shape_holdout()
    case.contexts.context = replace(case.contexts.context, contract=contract)
    result = await case.adapter.execute(case.request)
    assert result.status == "completed"
    assert len(case.client.requests) == len(contract.for_agent().shapes)
    aliases = contract.agent_shape_id_map()
    assert aliases is not None
    assert {
        shape_id
        for request in case.client.requests
        for shape_id in request["reference"]["shapes"]
    } == set(aliases)
    assert set(result.worker_result["candidate"]["latency_us_by_shape"]) == set(
        aliases
    )


@pytest.mark.anyio
@pytest.mark.parametrize("shape_count", [10, 40])
async def test_authoritative_abba_uses_the_full_valid_population(
    tmp_path: Path,
    shape_count: int,
) -> None:
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    revisions = []
    for side in ("incumbent", "candidate"):
        source = tmp_path / side
        source.mkdir()
        (source / "kernel.py").write_text(f"SIDE = {side!r}\n")
        revisions.append(
            KernelRevision(
                new_kernel_revision_id(),
                None,
                artifacts.put_directory(source, ArtifactKind.KERNEL),
                None,
                KernelEvaluation(True, 100, digest(side)),
                NOW,
            )
        )
    contract = _contract(shape_count).with_shape_holdout()
    assert contract.shape_split is not None
    assert contract.shape_split.test_shape_ids == ()

    contract_digest = artifacts.put_json(
        contract.model_dump(mode="json"), ArtifactKind.EVALUATION_CONTRACT
    )
    context = AgateEvaluationContext(
        "vecadd",
        "H20",
        Dsl.TRITON,
        contract,
        evaluation_contract_digest=contract_digest,
    )
    client = AbbaClient()
    journal = FakeJournal()
    runner = AgateSameAllocationAbbaRunner(
        client,
        FakeContextResolver(context),
        artifacts,
        journal,
        FakeEvaluator(),
        build_eval_request_from_content,
        wait_timeout_s=90,
    )
    result = await runner.run_pair(
        *revisions,
        repeats=2,
        purpose=KernelMeasurementPurpose.KERNEL_RETENTION,
        per_run_timeout_seconds=100,
        allocation_timeout_seconds=500,
        shape_batch_size=1,
        max_parallel_shape_batches=16,
    )
    assert len(client.requests) == shape_count
    assert {
        sid
        for payload in client.requests
        for sid in payload["reference"]["shapes"]
    } == set(contract.shapes)
    assert len(contract.for_agent().shapes) == shape_count
    assert set(contract.validation_shape_ids or ()) == set(contract.shapes)
    raw_path = artifacts.verify(result.gateway_result_digest).payload_path / "value.json"
    raw = json.loads(raw_path.read_text())
    valid_ids = set(contract.validation_shape_ids)
    assert raw["promotion_domain"] == "valid"
    assert raw["candidate"]["correct"] is True
    assert all(run.correct for run in result.candidate_runs)
    assert set(raw["candidate"]["latency_us_by_shape"]) == valid_ids
    assert "test_observation" not in raw
    raw["candidate"]["latency_us_by_shape"] = {sid: 90 for sid in valid_ids}
    raw["candidate"]["correctness"]["max_abs_err"] = 123456789
    private_result = artifacts.put_json(raw, ArtifactKind.GATEWAY_RESULT)
    public = gateway_result_projection(artifacts, private_result, correct=True, latency_us=900)
    aliases = contract.agent_shape_id_map()
    assert aliases is not None
    assert set(public["latency_us_by_shape"]) == set(aliases)
    assert public["latency_us_geomean"] == pytest.approx(90)
    assert public["latency_us_arith_mean"] == pytest.approx(90)
    assert public["correctness"]["max_abs_err"] is None
    assert public["measurement_domain"] == "valid"
    assert public["shape_ids_are_opaque"] is True
    assert "123456789" not in json.dumps(public)

    lineage = tmp_path / "lineage"
    (lineage / "epochs").mkdir(parents=True)
    (lineage / "epochs/00000001.json").write_text(
        json.dumps(
            {
                "attempts": [
                    {
                        "attempt_id": "attempt_one",
                        "failure_reason": (
                            "candidate ABBA improvement 0.123456% did not exceed 0.5%"
                        ),
                        "output": {
                            "gateway_result_digest": private_result,
                            "correct": True,
                            "latency_us": 900,
                        },
                    }
                ],
            }
        )
    )
    facts = _latest_evolver_epoch_facts(lineage, 1, artifacts)
    assert facts["attempts"][0]["candidate"]["latency_us"] == pytest.approx(90)
    assert "0.123456" not in json.dumps(facts)


@pytest.mark.parametrize("shape_count", [10, 40])
def test_problem_generalizer_receives_every_shape_under_opaque_ids(
    tmp_path: Path, shape_count: int
) -> None:
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    contract = _contract(shape_count).with_shape_holdout()
    private_digest = artifacts.put_json(
        contract.model_dump(mode="json"), ArtifactKind.EVALUATION_CONTRACT
    )
    source = tmp_path / "agent"
    source.mkdir()
    _write_agent(source)
    manifest = ProblemGeneralizationManifestV1(
        generalization_id="generalize-valid",
        optimizer_digest=artifacts.put_directory(source, ArtifactKind.KERNEL_AGENT),
        evaluation_contract_digest=private_digest,
        dsl=Dsl.TRITON,
        operator="vecadd",
        hardware_target="sm_90",
    )
    prepared = ProblemGeneralizationWorkspaceAssembler(tmp_path / "workspaces", artifacts).prepare(
        manifest
    )
    root = prepared.root / manifest.paths.private_inputs
    aliases = contract.agent_shape_id_map()
    assert aliases is not None
    assert len(aliases) == shape_count
    assert set(json.loads((root / "shapes.json").read_text())) == set(aliases)
    assert set(json.loads((root / "metadata.json").read_text())["shapes"]) == set(
        aliases
    )
    assert "private-shape-trace" not in (root / "metadata.json").read_text()
    assert set(json.loads((root / "roofline.json").read_text())["shapes"]) == set(
        aliases
    )
