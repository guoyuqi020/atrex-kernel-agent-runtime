"""Fixed Valid/Test inputs and non-disclosing authoritative result projections."""

from __future__ import annotations

import json
import random
from dataclasses import replace
from pathlib import Path

import pytest
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


@pytest.mark.parametrize("count", [2, 3, 10, 11, 29, 30, 31, 32, 45, 100])
def test_holdout_is_balanced_stable_and_preserved_in_sealed_contract(count: int) -> None:
    original = _contract(count)
    split = original.with_shape_holdout()
    reordered = original.model_copy(update={"shapes": dict(reversed(original.shapes.items()))})
    assert split == reordered.with_shape_holdout()
    assert split.with_shape_holdout() == split
    restored = AgateEvaluationContractV1.model_validate_json(split.model_dump_json())
    assert restored == split
    assert split.shape_split is not None
    record = split.shape_split
    assert record.seed == 42
    assert record.source_shape_count == count
    assert record.source_shape_ids == tuple(sorted(original.shapes))
    assert set(record.valid_shape_ids) == set(split.validation_shape_ids)
    assert set(record.valid_shape_ids) | set(record.test_shape_ids) == set(split.shapes)
    # Replay only from the recorded population and seed, not source dict ordering.
    rng = random.Random(record.seed)
    replay = list(record.source_shape_ids)
    rng.shuffle(replay)
    midpoint = (count + 1) // 2
    assert record.valid_shape_ids == tuple(sorted(rng.sample(replay[:midpoint], min(15, midpoint))))
    assert record.test_shape_ids == tuple(
        sorted(rng.sample(replay[midpoint:], min(15, count // 2)))
    )
    valid_count, test_count = min(15, (count + 1) // 2), min(15, count // 2)
    assert len(split.shapes) == valid_count + test_count
    assert len(split.validation_shape_ids or ()) == valid_count
    assert len(set(split.shapes) - set(split.validation_shape_ids or ())) == test_count
    assert set(split.shapes) <= set(original.shapes)
    assert split.metadata is not None and split.roofline is not None
    assert split.metadata["num_shapes"] == valid_count + test_count
    assert set(split.metadata["shapes"]) == set(split.shapes)
    assert set(split.roofline["shapes"]) == set(split.shapes)
    assert record.agent_shape_id_map is not None
    assert set(record.agent_shape_id_map) == {str(index) for index in range(valid_count)}
    assert set(record.agent_shape_id_map.values()) == set(record.valid_shape_ids)
    valid = split.for_agent()
    assert len(valid.shapes) == valid_count
    assert set(valid.shapes) == set(record.agent_shape_id_map)
    assert valid.metadata is not None and valid.roofline is not None
    assert valid.metadata["num_shapes"] == valid_count
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


def test_fixed_seed_sampling_does_not_modify_global_random_state() -> None:
    before = random.getstate()
    _contract(100).with_shape_holdout()
    assert random.getstate() == before


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
    excluded = next(iter(set(split.shape_split.source_shape_ids) - set(split.shapes)))
    value["shape_split"]["test_shape_ids"][0] = excluded
    with pytest.raises(ValidationError, match="selections must match"):
        AgateEvaluationContractV1.model_validate(value)


def test_archived_agent_shape_id_map_must_be_contiguous_and_exact() -> None:
    split = _contract(40).with_shape_holdout()
    value = split.model_dump(mode="json")
    value["shape_split"]["agent_shape_id_map"]["0"] = value["shape_split"][
        "test_shape_ids"
    ][0]
    with pytest.raises(ValidationError, match="contiguous opaque Agent IDs"):
        AgateEvaluationContractV1.model_validate(value)


def test_sealed_partition_cannot_exceed_shape_cap() -> None:
    value = _contract(40).model_dump()
    value["validation_shape_ids"] = [str(i) for i in range(20)]
    with pytest.raises(ValidationError, match="at most 15 Shapes"):
        AgateEvaluationContractV1.model_validate(value)


def test_single_shape_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least 2 Shapes"):
        _contract(1).with_shape_holdout()


@pytest.mark.parametrize("ids", [[], ["0", "0"], ["0", "1", "2"], ["unknown", "0"]])
def test_invalid_sealed_partition_is_rejected(ids: list[str]) -> None:
    value = _contract(4).model_dump()
    value["validation_shape_ids"] = ids
    with pytest.raises(ValidationError, match="select half"):
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
        assert len(sent) == 1 and set(sent[0]) == set(aliases)
        assert "private-shape-trace" not in json.dumps(client.submitted)
        assert set(result.worker_result["latency_us_by_shape"]) == set(sent[0])
        test_id = "not-an-agent-shape"
        with pytest.raises(ValueError, match="not an evaluator-owned"):
            await adapter.execute(
                replace(
                    request,
                    operation=GatewayOperation.PROFILE,
                    parameters={"shape_id": test_id},
                    profile_level="sol",
                    idempotency_key="cannot-probe-test",
                )
            )
        assert len(client.submitted) == 1
    finally:
        jobs.close()


@pytest.mark.anyio
async def test_agent_profile_uses_opaque_valid_shape_id_end_to_end(tmp_path: Path) -> None:
    contract = _contract(40).with_shape_holdout()
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
    assert len(case.client.requests) == 1
    files = case.client.requests[0]["files"]
    aliases = contract.agent_shape_id_map()
    assert aliases is not None
    assert set(json.loads(files["reference/shapes.json"])) == set(aliases)
    assert set(result.worker_result["candidate"]["latency_us_by_shape"]) == set(
        aliases
    )


@pytest.mark.anyio
@pytest.mark.parametrize("shape_count", [10, 40])
@pytest.mark.parametrize("test_failure", [False, True])
async def test_authoritative_abba_gates_on_valid_and_observes_test_privately(
    tmp_path: Path,
    shape_count: int,
    test_failure: bool,
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
    failed_test_shape = contract.shape_split.test_shape_ids[0]

    class ObservationClient(AbbaClient):
        def submit_job(self, kind: str, request: dict[str, object]) -> dict[str, object]:
            accepted = super().submit_job(kind, request)
            files = request["files"]
            assert isinstance(files, dict)
            request_shapes = set(json.loads(files["reference/shapes.json"]))
            if test_failure and failed_test_shape in request_shapes:
                job = self.jobs[str(accepted["job_id"])]
                result_payload = job["result"]
                assert isinstance(result_payload, dict)
                stdout = result_payload["stdout"]
                assert isinstance(stdout, str)
                payload = json.loads(stdout.split("=", 1)[1])
                for run in payload["runs"]:
                    run["result"]["all_pass"] = False
                    run["result"]["latency_us_by_shape"] = {}
                result_payload["stdout"] = "__ATREX_RUNTIME_ABBA_RESULT__=" + json.dumps(
                    payload,
                    separators=(",", ":"),
                )
            return accepted

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
    client = ObservationClient()
    journal = FakeJournal()
    runner = AgateSameAllocationAbbaRunner(
        client,
        FakeContextResolver(context),
        artifacts,
        journal,
        FakeEvaluator(),
        wait_timeout_s=90,
    )
    result = await runner.run_pair(
        *revisions,
        repeats=1,
        purpose=KernelMeasurementPurpose.KERNEL_RETENTION,
        per_run_timeout_seconds=100,
        allocation_timeout_seconds=250,
        shape_batch_size=1,
        max_parallel_shape_batches=16,
    )
    assert len(client.requests) == min(shape_count, 30)
    assert {
        sid
        for payload in client.requests
        for sid in json.loads(payload["files"]["reference/shapes.json"])
    } == set(contract.shapes)
    assert len(contract.for_agent().shapes) <= 15
    assert len(set(contract.shapes) - set(contract.validation_shape_ids)) <= 15
    raw_path = artifacts.verify(result.gateway_result_digest).payload_path / "value.json"
    raw = json.loads(raw_path.read_text())
    valid_ids = set(contract.validation_shape_ids)
    test_ids = set(contract.shape_split.test_shape_ids)
    assert raw["promotion_domain"] == "valid"
    assert raw["candidate"]["correct"] is True
    assert all(run.correct for run in result.candidate_runs)
    assert set(raw["candidate"]["latency_us_by_shape"]) == valid_ids
    assert raw["test_observation"]["affects_promotion"] is False
    assert raw["test_observation"]["status"] == "completed"
    assert raw["test_observation"]["candidate"]["correct"] is (not test_failure)
    assert set(raw["test_observation"]["candidate"]["latency_us_by_shape"]) == (
        set() if test_failure else test_ids
    )
    # Synthetic distinctive Test measurements catch aggregate and scalar leakage.
    raw["candidate"]["latency_us_by_shape"] = {sid: 90 for sid in valid_ids}
    raw["test_observation"]["candidate"]["latency_us_by_shape"] = {sid: 9000 for sid in test_ids}
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
    assert not set(public["latency_us_by_shape"]).intersection(valid_ids - set(aliases))
    assert "9000" not in json.dumps(public) and "123456789" not in json.dumps(public)

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
def test_problem_generalizer_cannot_read_test_inputs(tmp_path: Path, shape_count: int) -> None:
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
    assert set(json.loads((root / "shapes.json").read_text())) == set(aliases)
    assert set(json.loads((root / "metadata.json").read_text())["shapes"]) == set(
        aliases
    )
    assert "private-shape-trace" not in (root / "metadata.json").read_text()
    assert set(json.loads((root / "roofline.json").read_text())["shapes"]) == set(
        aliases
    )
