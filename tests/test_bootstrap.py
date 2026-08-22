"""Commit-only Campaign bootstrap tests."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import cast

import pytest
from pydantic import ValidationError

from atrex_runtime.artifacts.local import ArtifactKind, LocalArtifactStore
from atrex_runtime.bootstrap import (
    CampaignBootstrapper,
    CampaignSpecV3,
    GeneratedLineageBaseline,
    load_campaign_spec,
)
from atrex_runtime.controller import EvidenceCheckpointV1
from atrex_runtime.domain.errors import InvalidTransitionError
from atrex_runtime.domain.ids import (
    ArtifactDigest,
    new_kernel_agent_revision_id,
    new_kernel_revision_id,
)
from atrex_runtime.domain.models import (
    Dsl,
    KernelAgentRevision,
    KernelEvaluation,
    KernelRevision,
    LineageStatus,
)
from atrex_runtime.gateway.contract import (
    AgateEvaluationOptionsV1,
    RuntimeGateContractPolicy,
)
from atrex_runtime.gateway.environment import ResolvedAgateEnvironment
from atrex_runtime.kernel_agents import GitOptimizerBaseResult
from atrex_runtime.ports import KernelAgentCandidate
from atrex_runtime.registry.sqlite import SqliteRegistry

NOW = "2026-08-15T00:00:00+00:00"


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _contract() -> dict[str, object]:
    return {
        "schema_version": 1,
        "candidate_path": "kernel.py",
        "reference_py": "class Model:\n    pass\n",
        "input_py": "def get_inputs():\n    return ()\n",
        "shapes": {"0": {}},
        "options": {
            "num_correctness_cases": 2,
            "bench_iters": 10,
            "atol": 0.001,
            "rtol": 0.01,
            "timeout_s": 60,
        },
        "lock_clocks": True,
        "runner_overrides": {
            "bench_iters": 999,
            "benchmark_mode": "cuda_graph_replay",
            "candidate_timeout_s": 999,
            "custom_non_gate_flag": True,
        },
    }


def _campaign_spec(
    tmp_path: Path,
    *,
    lineage_dsls: tuple[Dsl, ...] = tuple(Dsl),
    include_problem: bool = True,
    problem_generalization_model: str | None = None,
    optimizer_model: str | None = None,
    evolver_model: str | None = None,
) -> Path:
    _write_json(tmp_path / "evaluation.json", _contract())
    if include_problem:
        _write_json(
            tmp_path / "agent-problem.json",
            {
                "schema_version": "atrex.agent_problem.v1",
                "objective": "Optimize the operator without evaluator-private cases.",
                "evaluation": {
                    "exact_cases": "private",
                    "correctness_requirement": "all private cases pass",
                    "performance_requirement": "measure after correctness",
                    "development_cases_are_evaluation_cases": False,
                },
                "operator_contract": {},
                "workload_profile": {},
                "distribution_profile": {},
                "shape_domain": {},
                "invariants": ["preserve operator semantics"],
                "coverage_regimes": [],
                "development_cases": [],
            },
        )
    kernel = tmp_path / "baseline-kernel"
    kernel.mkdir(exist_ok=True)
    (kernel / "kernel.py").write_text("def kernel():\n    return None\n", encoding="utf-8")
    evidence = tmp_path / "evidence"
    evidence.mkdir(exist_ok=True)
    (evidence / "README.md").write_text("Initial evidence.\n", encoding="utf-8")
    value: dict[str, object] = {
        "schema_version": 3,
        "creation_key": "vector-add-h100",
        "operator": "vector_add",
        "hardware_target": "nvidia-h100",
        "evaluation_contract": "evaluation.json",
        "base_revision": {"commit": "a" * 40},
        "challenger_count": 1,
        "challenger_start_epoch": 2,
        "trajectories_per_branch": 1,
        "attempts_per_trajectory": 2,
        "lineages": {
            dsl.value: {
                "baseline_kernel": "baseline-kernel",
                "initial_evidence": "evidence",
                "models": {
                    "optimizer": optimizer_model,
                    "evolver": evolver_model,
                },
            }
            for dsl in lineage_dsls
        },
    }
    if include_problem:
        value["agent_problem"] = "agent-problem.json"
    else:
        value["problem_generalization_model"] = problem_generalization_model
    path = tmp_path / "campaign.json"
    _write_json(path, value)
    return path


class FakeGitLoader:
    def __init__(self, artifacts: LocalArtifactStore) -> None:
        self.calls: list[tuple[Dsl, str]] = []
        repository = artifacts._root.parent / "optimizer"
        repository.mkdir(exist_ok=True)
        (repository / "atrex-bundle.json").write_text("{}", encoding="utf-8")
        self.optimizer_digest = artifacts.put_directory(
            repository,
            ArtifactKind.KERNEL_AGENT,
        )
        self.provenance_digest = artifacts.put_json(
            {"schema_version": 1, "commit": "a" * 40},
            ArtifactKind.OPTIMIZER_SOURCE,
        )

    def build_candidate(self, dsl: Dsl, commit: str) -> GitOptimizerBaseResult:
        self.calls.append((dsl, commit))
        return GitOptimizerBaseResult(
            KernelAgentCandidate(dsl, self.optimizer_digest),
            self.provenance_digest,
        )


class FakeBaselineGenerator:
    def __init__(self, artifacts: LocalArtifactStore) -> None:
        self.calls: list[Dsl] = []
        self.models: list[str | None] = []
        self.hardware_targets: list[str] = []
        self.gateway_digest = artifacts.put_json(
            {"correct": True, "latency_us": 9.0},
            ArtifactKind.GATEWAY_RESULT,
        )

    def generate(self, **values: object) -> GeneratedLineageBaseline:
        dsl = values["dsl"]
        assert isinstance(dsl, Dsl)
        self.calls.append(dsl)
        self.models.append(cast(str | None, values["model"]))
        self.hardware_targets.append(str(values["hardware_target"]))
        return GeneratedLineageBaseline(
            cast(ArtifactDigest, values["input_kernel_digest"]),
            self.gateway_digest,
            9.0,
        )


def _bootstrapper(
    registry: SqliteRegistry,
    artifacts: LocalArtifactStore,
    loader: FakeGitLoader,
    baseline: FakeBaselineGenerator,
    *,
    problem_generator: object | None = None,
    roofline_builder: object | None = None,
    evolver_commit: str | None = None,
    gate_contract_policy: RuntimeGateContractPolicy | None = None,
    hardware_target_resolver: object | None = None,
    max_parallel_lineages: int = 1,
) -> CampaignBootstrapper:
    return CampaignBootstrapper(
        registry,
        artifacts,
        base_loader=loader,  # type: ignore[arg-type]
        problem_generator=problem_generator,  # type: ignore[arg-type]
        baseline_generator=baseline,
        roofline_builder=roofline_builder,  # type: ignore[arg-type]
        evolver_commit=evolver_commit,
        gate_contract_policy=gate_contract_policy,
        hardware_target_resolver=hardware_target_resolver,  # type: ignore[arg-type]
        max_parallel_lineages=max_parallel_lineages,
        clock=lambda: NOW,
    )


def test_campaign_bootstrap_resolves_agent_arch_from_agate_environment(tmp_path: Path) -> None:
    spec = CampaignSpecV3.from_file(_campaign_spec(tmp_path, lineage_dsls=(Dsl.TRITON,)))
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    baseline = FakeBaselineGenerator(artifacts)

    class Resolver:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def resolve(self, gpu: str) -> ResolvedAgateEnvironment:
            self.calls.append(gpu)
            return ResolvedAgateEnvironment("L20N", "sm_120")

    resolver = Resolver()
    with SqliteRegistry(tmp_path / "registry.sqlite") as registry:
        result = _bootstrapper(
            registry,
            artifacts,
            FakeGitLoader(artifacts),
            baseline,
            hardware_target_resolver=resolver,
        ).bootstrap_campaign(spec)
        campaign = registry.get_campaign(result.campaign_id)
        lineage = registry.get_lineage(result.lineages[0].lineage_id)

    contract_path = artifacts.verify(result.lineages[0].evaluation_contract_digest).payload_path
    contract = json.loads((contract_path / "value.json").read_text(encoding="utf-8"))
    assert resolver.calls == ["nvidia-h100"]
    assert campaign.hardware_target == "sm_120"
    assert lineage.hardware_target == "sm_120"
    assert baseline.hardware_targets == ["sm_120"]
    assert contract["agate_gpu"] == "L20N"


class ConcurrentBaselineGenerator(FakeBaselineGenerator):
    def __init__(self, artifacts: LocalArtifactStore) -> None:
        super().__init__(artifacts)
        self._lock = threading.Lock()
        self.active = 0
        self.max_active = 0

    def generate(self, **values: object) -> GeneratedLineageBaseline:
        with self._lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            time.sleep(0.05)
            return super().generate(**values)
        finally:
            with self._lock:
                self.active -= 1


def test_campaign_bootstrap_runs_dsl_lineages_in_configured_parallelism(
    tmp_path: Path,
) -> None:
    spec = CampaignSpecV3.from_file(_campaign_spec(tmp_path))
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    baseline = ConcurrentBaselineGenerator(artifacts)

    with SqliteRegistry(tmp_path / "registry.sqlite") as registry:
        result = _bootstrapper(
            registry,
            artifacts,
            FakeGitLoader(artifacts),
            baseline,
            max_parallel_lineages=3,
        ).bootstrap_campaign(spec)

    assert baseline.max_active == 3
    assert tuple(item.optimizer_model for item in result.lineages) == (None, None, None)
    assert set(baseline.calls) == set(Dsl)


def test_campaign_bootstrap_binds_runtime_evaluation_tolerances(tmp_path: Path) -> None:
    spec = CampaignSpecV3.from_file(_campaign_spec(tmp_path, lineage_dsls=(Dsl.TRITON,)))
    artifacts = LocalArtifactStore(tmp_path / "artifacts")

    with SqliteRegistry(tmp_path / "registry.sqlite") as registry:
        result = _bootstrapper(
            registry,
            artifacts,
            FakeGitLoader(artifacts),
            FakeBaselineGenerator(artifacts),
            gate_contract_policy=RuntimeGateContractPolicy(
                options=AgateEvaluationOptionsV1(
                    num_correctness_cases=1,
                    bench_iters=100,
                    atol=0.01,
                    rtol=0.05,
                    timeout_s=600,
                ),
                lock_clocks=False,
                atrex_bench_version="0.1.0",
                runner_overrides={
                    "warmup_iters": 10,
                    "candidate_timeout_s": 20,
                    "perf_timeout_s": 120,
                },
                production_gate=True,
            ),
        ).bootstrap_campaign(spec)

    contract_path = artifacts.verify(result.lineages[0].evaluation_contract_digest).payload_path
    contract = json.loads((contract_path / "value.json").read_text(encoding="utf-8"))
    assert contract["options"]["atol"] == 0.01
    assert contract["options"]["rtol"] == 0.05
    assert contract["options"]["num_correctness_cases"] == 1
    assert contract["options"]["bench_iters"] == 100
    assert contract["options"]["timeout_s"] == 600
    assert contract["lock_clocks"] is False
    assert contract["atrex_bench_version"] == "0.1.0"
    assert contract["production_gate"] is True
    assert contract["runner_overrides"] == {
        "benchmark_mode": "eager",
        "candidate_timeout_s": 20,
        "custom_non_gate_flag": True,
        "perf_timeout_s": 120,
        "warmup_iters": 10,
    }


def test_campaign_bootstrap_freezes_evolver_commit(tmp_path: Path) -> None:
    spec = CampaignSpecV3.from_file(_campaign_spec(tmp_path, lineage_dsls=(Dsl.TRITON,)))
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    with SqliteRegistry(tmp_path / "registry.sqlite") as registry:
        first = _bootstrapper(
            registry,
            artifacts,
            FakeGitLoader(artifacts),
            FakeBaselineGenerator(artifacts),
            evolver_commit="c" * 40,
        ).bootstrap_campaign(spec)

        assert first.evolver_commit == "c" * 40
        assert registry.get_campaign(first.campaign_id).evolver_commit == "c" * 40
        with pytest.raises(InvalidTransitionError, match="freezes Evolver commit"):
            _bootstrapper(
                registry,
                artifacts,
                FakeGitLoader(artifacts),
                FakeBaselineGenerator(artifacts),
                evolver_commit="d" * 40,
            ).bootstrap_campaign(spec)


def test_campaign_bootstrap_builds_and_recovers_one_shared_roofline(
    tmp_path: Path,
) -> None:
    spec = CampaignSpecV3.from_file(_campaign_spec(tmp_path, lineage_dsls=(Dsl.TRITON,)))
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    calls: list[tuple[str, str]] = []

    class Builder:
        def build(self, **values: object) -> dict[str, object]:
            calls.append((str(values["operator"]), str(values["hardware_target"])))
            return {
                "shapes": {
                    "0": {
                        "semantic_W_flops": {"fp32": 1},
                        "semantic_Q_read_bytes": 4,
                        "semantic_Q_write_bytes": 4,
                        "SOL_time_ms": {"test-gpu": 0.001},
                    }
                }
            }

    with SqliteRegistry(tmp_path / "registry.sqlite") as registry:
        bootstrapper = _bootstrapper(
            registry,
            artifacts,
            FakeGitLoader(artifacts),
            FakeBaselineGenerator(artifacts),
            roofline_builder=Builder(),
        )
        first = bootstrapper.bootstrap_campaign(spec)
        second = bootstrapper.bootstrap_campaign(spec)

    assert second == first
    assert first.roofline_mode == "generated"
    assert second.roofline_mode == "sealed-reuse"
    assert calls == [("vector_add", "nvidia-h100")]
    contract_path = artifacts.verify(first.lineages[0].evaluation_contract_digest).payload_path
    contract = json.loads((contract_path / "value.json").read_text(encoding="utf-8"))
    assert contract["roofline"]["shapes"]["0"]["SOL_time_ms"] == {"test-gpu": 0.001}


def test_campaign_bootstrap_falls_back_to_profile_when_roofline_build_fails(
    tmp_path: Path,
) -> None:
    spec = CampaignSpecV3.from_file(_campaign_spec(tmp_path, lineage_dsls=(Dsl.TRITON,)))
    artifacts = LocalArtifactStore(tmp_path / "artifacts")

    class Builder:
        def build(self, **values: object) -> dict[str, object]:
            del values
            raise RuntimeError("operator cost is unavailable")

    with SqliteRegistry(tmp_path / "registry.sqlite") as registry:
        result = _bootstrapper(
            registry,
            artifacts,
            FakeGitLoader(artifacts),
            FakeBaselineGenerator(artifacts),
            roofline_builder=Builder(),
        ).bootstrap_campaign(spec)

    assert result.roofline_mode == "profile-fallback"
    assert result.roofline_detail == "RuntimeError: operator cost is unavailable"
    contract_path = artifacts.verify(result.lineages[0].evaluation_contract_digest).payload_path
    contract = json.loads((contract_path / "value.json").read_text(encoding="utf-8"))
    assert contract["roofline"] is None


def test_campaign_bootstrap_rejects_public_problem_that_copies_a_private_case(
    tmp_path: Path,
) -> None:
    spec_path = _campaign_spec(tmp_path, lineage_dsls=(Dsl.TRITON,))
    evaluation = _contract()
    evaluation["shapes"] = {
        "opaque-0": {
            "init_kwargs": None,
            "input_kwargs": {"num_elements": 1048576},
        }
    }
    _write_json(tmp_path / "evaluation.json", evaluation)
    problem = json.loads((tmp_path / "agent-problem.json").read_text(encoding="utf-8"))
    problem["development_cases"] = [
        {
            "name": "copied-hidden-case",
            "init_kwargs": None,
            "input_kwargs": {"num_elements": 1048576},
        }
    ]
    _write_json(tmp_path / "agent-problem.json", problem)
    artifacts = LocalArtifactStore(tmp_path / "artifacts")

    with (
        SqliteRegistry(tmp_path / "registry.sqlite") as registry,
        pytest.raises(ValueError, match="duplicates a private evaluator case"),
    ):
        _bootstrapper(
            registry,
            artifacts,
            FakeGitLoader(artifacts),
            FakeBaselineGenerator(artifacts),
        ).bootstrap_campaign(CampaignSpecV3.from_file(spec_path))


def test_campaign_bootstrap_imports_core_once_and_initializes_selected_lineages(
    tmp_path: Path,
) -> None:
    spec = CampaignSpecV3.from_file(_campaign_spec(tmp_path))
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    loader = FakeGitLoader(artifacts)
    baseline = FakeBaselineGenerator(artifacts)
    with SqliteRegistry(tmp_path / "registry.sqlite") as registry:
        bootstrapper = _bootstrapper(registry, artifacts, loader, baseline)

        first = bootstrapper.bootstrap_campaign(spec)
        second = bootstrapper.bootstrap_campaign(spec)

        assert second == first
        assert loader.calls == [(Dsl.CUDA, "a" * 40), (Dsl.CUDA, "a" * 40)]
        assert baseline.calls == list(Dsl)
        assert [registry.get_lineage(item.lineage_id).dsl for item in first.lineages] == list(Dsl)
        assert len({item.evaluation_contract_digest for item in first.lineages}) == 1
        assert len({item.agent_problem_digest for item in first.lineages}) == 1
        for result in first.lineages:
            assert str(result.bootstrap_attempt_id).startswith("attempt_")
            lineage = registry.get_lineage(result.lineage_id)
            assert lineage.status is LineageStatus.READY
            assert lineage.challenger_start_epoch == 2
            checkpoint = EvidenceCheckpointV1.from_file(
                artifacts.verify(lineage.evidence_checkpoint).payload_path / "checkpoint.json"
            )
            assert checkpoint.lineage_id == lineage.id


def test_campaign_bootstrap_is_idempotent_after_lineage_has_evolved(
    tmp_path: Path,
) -> None:
    spec = CampaignSpecV3.from_file(_campaign_spec(tmp_path, lineage_dsls=(Dsl.TRITON,)))
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    loader = FakeGitLoader(artifacts)
    baseline = FakeBaselineGenerator(artifacts)
    with SqliteRegistry(tmp_path / "registry.sqlite") as registry:
        bootstrapper = _bootstrapper(registry, artifacts, loader, baseline)
        first = bootstrapper.bootstrap_campaign(spec)
        lineage = registry.get_lineage(first.lineages[0].lineage_id)
        evolved_optimizer = artifacts.put_json({"agent": "evolved"}, ArtifactKind.KERNEL_AGENT)
        evolved_agent = KernelAgentRevision(
            id=new_kernel_agent_revision_id(),
            parent_id=lineage.active_kernel_agent_revision_id,
            creation_key="evolved-agent",
            dsl=Dsl.TRITON,
            optimizer_digest=evolved_optimizer,
            created_by="evolver",
            created_at=NOW,
            evolution_trace_digest=artifacts.put_json(
                {"evolution": "trace"}, ArtifactKind.EVOLUTION
            ),
        )
        registry.register_kernel_agent_revision(evolved_agent)
        evolved_kernel = KernelRevision(
            id=new_kernel_revision_id(),
            parent_id=lineage.best_kernel_revision_id,
            artifact_digest=artifacts.put_json({"kernel": "evolved"}, ArtifactKind.KERNEL),
            produced_by_attempt_id=None,
            evaluation=KernelEvaluation(
                correct=True,
                latency_us=8.0,
                gateway_result_digest=artifacts.put_json(
                    {"correct": True, "latency_us": 8.0},
                    ArtifactKind.GATEWAY_RESULT,
                ),
            ),
            created_at=NOW,
        )
        registry.register_kernel_revision(evolved_kernel)
        registry._connection.execute(
            """UPDATE lineages SET active_kernel_agent_revision_id = ?,
               best_kernel_revision_id = ?, evidence_checkpoint = ?,
               next_epoch_number = 2 WHERE id = ?""",
            (
                evolved_agent.id,
                evolved_kernel.id,
                artifacts.put_json({"epoch": 2}, ArtifactKind.EVIDENCE),
                lineage.id,
            ),
        )
        registry._connection.commit()

        second = bootstrapper.bootstrap_campaign(spec)

        assert second == first
        assert registry.get_lineage(lineage.id).active_kernel_agent_revision_id == (
            evolved_agent.id
        )
        assert registry.get_lineage(lineage.id).best_kernel_revision_id == evolved_kernel.id
        assert evolved_optimizer == evolved_agent.optimizer_digest
        assert baseline.calls == [Dsl.TRITON]


def test_campaign_bootstrap_generates_one_shared_problem(tmp_path: Path) -> None:
    spec = CampaignSpecV3.from_file(
        _campaign_spec(
            tmp_path,
            include_problem=False,
            problem_generalization_model="problem-model",
            optimizer_model="optimizer-model",
            evolver_model="evolver-model",
        )
    )
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    generated = artifacts.put_json(
        {"schema_version": "atrex.agent_problem.v1", "objective": "shared"},
        ArtifactKind.AGENT_PROBLEM,
    )
    calls: list[dict[str, object]] = []

    class ProblemGenerator:
        def generate(self, **values: object) -> ArtifactDigest:
            calls.append(values)
            return generated

    baseline = FakeBaselineGenerator(artifacts)
    with SqliteRegistry(tmp_path / "registry.sqlite") as registry:
        result = _bootstrapper(
            registry,
            artifacts,
            FakeGitLoader(artifacts),
            baseline,
            problem_generator=ProblemGenerator(),
        ).bootstrap_campaign(spec)
        persisted_lineages = [registry.get_lineage(item.lineage_id) for item in result.lineages]

    assert len(calls) == 1
    assert calls[0]["dsl"] is Dsl.CUDA
    assert calls[0]["model"] == "problem-model"
    assert baseline.models == ["optimizer-model"] * len(Dsl)
    for lineage in persisted_lineages:
        assert lineage.optimizer_model == "optimizer-model"
        assert lineage.evolver_model == "evolver-model"
    assert {item.agent_problem_digest for item in result.lineages} == {generated}


def test_campaign_bootstrap_rejects_lineage_model_drift(tmp_path: Path) -> None:
    path = _campaign_spec(
        tmp_path,
        lineage_dsls=(Dsl.TRITON,),
        optimizer_model="optimizer-model-a",
        evolver_model="evolver-model-a",
    )
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    with SqliteRegistry(tmp_path / "registry.sqlite") as registry:
        bootstrapper = _bootstrapper(
            registry,
            artifacts,
            FakeGitLoader(artifacts),
            FakeBaselineGenerator(artifacts),
        )
        bootstrapper.bootstrap_campaign(CampaignSpecV3.from_file(path))

        value = json.loads(path.read_text(encoding="utf-8"))
        value["lineages"]["triton"]["models"]["optimizer"] = "optimizer-model-b"
        _write_json(path, value)

        with pytest.raises(ValueError, match="different lineage"):
            bootstrapper.bootstrap_campaign(CampaignSpecV3.from_file(path))


def test_campaign_bootstrap_recovers_after_partial_baseline_failure(tmp_path: Path) -> None:
    selected = (Dsl.CUDA, Dsl.TRITON)
    spec = CampaignSpecV3.from_file(_campaign_spec(tmp_path, lineage_dsls=selected))
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    baseline = FakeBaselineGenerator(artifacts)
    original = baseline.generate
    failed = False

    def fail_once(**values: object) -> GeneratedLineageBaseline:
        nonlocal failed
        if values["dsl"] is Dsl.TRITON and not failed:
            failed = True
            raise RuntimeError("simulated interruption")
        return original(**values)

    baseline.generate = fail_once  # type: ignore[method-assign]
    with SqliteRegistry(tmp_path / "registry.sqlite") as registry:
        bootstrapper = _bootstrapper(
            registry,
            artifacts,
            FakeGitLoader(artifacts),
            baseline,
        )
        with pytest.raises(RuntimeError, match="interruption"):
            bootstrapper.bootstrap_campaign(spec)

        result = bootstrapper.bootstrap_campaign(spec)

        assert [registry.get_lineage(item.lineage_id).dsl for item in result.lineages] == list(
            selected
        )
        assert baseline.calls == [Dsl.CUDA, Dsl.TRITON]


def test_campaign_bootstrap_requires_runtime_owned_generators(tmp_path: Path) -> None:
    spec = CampaignSpecV3.from_file(_campaign_spec(tmp_path, include_problem=False))
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    with SqliteRegistry(tmp_path / "registry.sqlite") as registry:
        with pytest.raises(ValueError, match="Agent Problem"):
            CampaignBootstrapper(
                registry,
                artifacts,
                base_loader=FakeGitLoader(artifacts),  # type: ignore[arg-type]
                baseline_generator=FakeBaselineGenerator(artifacts),
            ).bootstrap_campaign(spec)

        with pytest.raises(ValueError, match="Git Optimizer Base"):
            CampaignBootstrapper(
                registry,
                artifacts,
                baseline_generator=FakeBaselineGenerator(artifacts),
            ).bootstrap_campaign(spec)


def test_campaign_bootstrap_rejects_empty_baseline_candidate(tmp_path: Path) -> None:
    spec = CampaignSpecV3.from_file(_campaign_spec(tmp_path, lineage_dsls=(Dsl.TRITON,)))
    (spec.lineages[Dsl.TRITON].baseline_kernel / "kernel.py").write_text(" \n")
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    with (
        SqliteRegistry(tmp_path / "registry.sqlite") as registry,
        pytest.raises(ValueError, match="cannot be empty"),
    ):
        _bootstrapper(
            registry,
            artifacts,
            FakeGitLoader(artifacts),
            FakeBaselineGenerator(artifacts),
        ).bootstrap_campaign(spec)


def test_only_campaign_schema_and_commit_source_are_accepted(tmp_path: Path) -> None:
    path = _campaign_spec(tmp_path, lineage_dsls=(Dsl.TRITON,))
    value = json.loads(path.read_text())
    value["kernel_agent"] = "agent"
    with pytest.raises(ValidationError):
        CampaignSpecV3.model_validate(value)

    value = json.loads(path.read_text())
    value["dsls"] = ["triton"]
    with pytest.raises(ValidationError):
        CampaignSpecV3.model_validate(value)

    value = json.loads(path.read_text())
    value["lineages"] = {}
    with pytest.raises(ValidationError, match="at least one DSL Lineage"):
        CampaignSpecV3.model_validate(value)

    value = json.loads(path.read_text())
    value["lineages"]["triton"]["baseline_latency_us"] = 10.0
    with pytest.raises(ValidationError):
        CampaignSpecV3.model_validate(value)

    value = json.loads(path.read_text())
    value["problem_generalization_model"] = "unused-model"
    with pytest.raises(ValidationError, match="requires generated Agent Problem"):
        CampaignSpecV3.model_validate(value)

    value = json.loads(path.read_text())
    value["schema_version"] = 2
    _write_json(path, value)
    with pytest.raises(ValueError, match="unsupported Campaign spec"):
        load_campaign_spec(path)
