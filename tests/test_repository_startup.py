"""Repository-level smoke for the checked-in no-sandbox startup configuration."""

from __future__ import annotations

import base64
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import pytest
from conftest import with_local_interpreter

from atrex_runtime.api.app import build_runtime_application
from atrex_runtime.artifacts import LocalArtifactStore
from atrex_runtime.bootstrap import load_campaign_spec
from atrex_runtime.composition.campaign import build_campaign_runtime
from atrex_runtime.config import RuntimeSettings, RuntimeStorageSettings
from atrex_runtime.domain.models import Dsl
from atrex_runtime.gateway.abba import CommitPinnedAtrexBenchEvaluator
from atrex_runtime.gateway.contract import AgateEvaluationContractV1
from atrex_runtime.kernel_agents import GitOptimizerBaseLoader, KernelAgentRevisionBuilder
from atrex_runtime.lineage_seed import ArtifactLineageSeedV1, LineageSeedSpecV1
from atrex_runtime.workers.problem_generalization import AgentProblemV1

RUNNABLE_EXAMPLES = (
    "agate",
    "bootstrap",
    "evolution",
    "evolver-dev-shell",
    "lineage",
    "local-wiki",
    "optimizer-dev-shell",
)
RUNTIME_EXAMPLES = (
    "bootstrap",
    "evolution",
    "evolver-dev-shell",
    "lineage",
    "local-wiki",
    "optimizer-dev-shell",
)


def _settings(tmp_path: Path) -> RuntimeSettings:
    repository = Path(__file__).resolve().parents[1]
    settings = RuntimeSettings.from_file(repository / "runtime.example.json")
    assert settings.campaign is not None
    storage = RuntimeStorageSettings(
        registry_database=tmp_path / "registry.sqlite",
        gateway_database=tmp_path / "gateway.sqlite",
        agate_jobs_database=tmp_path / "agate.sqlite",
        artifacts_root=tmp_path / "artifacts",
    )
    campaign = with_local_interpreter(settings.campaign).model_copy(
        update={
            "attempt_workspaces_root": tmp_path / "attempts",
            "evolution_workspaces_root": tmp_path / "evolution",
            "problem_generalization_workspaces_root": tmp_path / "generalization",
            "lineage_bootstrap_workspaces_root": tmp_path / "bootstrap",
            "launcher": settings.campaign.launcher.model_copy(
                update={"mode": "development", "sandbox": None}
            ),
        }
    )
    return settings.model_copy(update={"storage": storage, "campaign": campaign})


def _environment() -> dict[str, str]:
    return {
        "ATREX_CAPABILITY_SIGNING_KEY": base64.urlsafe_b64encode(b"k" * 32).decode(),
        "ATREX_ADMIN_BEARER_TOKEN": "a" * 32,
        "ANTHROPIC_AUTH_TOKEN": "repository-startup-smoke-only",
        "PATH": os.environ["PATH"],
    }


def test_checked_in_configuration_builds_server_and_campaign_runtime(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    assert settings.kernel_agent.base_source is not None
    assert Path(settings.kernel_agent.base_source.repository).is_absolute()
    assert settings.campaign is not None
    assert Path(settings.campaign.evolver.repository).is_absolute()

    application = build_runtime_application(settings, _environment())
    application.close()
    with build_campaign_runtime(settings, _environment()):
        pass


def test_campaign_and_bootstrap_do_not_issue_wiki_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from atrex_runtime.composition import bootstrap, campaign
    from atrex_runtime.gateway.control import SqliteGatewayControl
    from atrex_runtime.gateway.control_models import GatewayOperation
    from atrex_runtime.registry.sqlite import SqliteRegistry

    settings = _settings(tmp_path)
    assert settings.gpu_wiki is not None  # Even a configured service does not expose an Agent tool.
    observed: list[dict[str, Any]] = []

    def watch(module: Any, name: str) -> None:
        original = getattr(module, name)

        def construct(*args: Any, **kwargs: Any) -> Any:
            observed.append(kwargs)
            return original(*args, **kwargs)

        monkeypatch.setattr(module, name, construct)

    watch(campaign, "AttemptTimedWorkerGatewayAuthorityProvider")
    watch(campaign, "SessionOptimizerRunner")
    watch(bootstrap, "CoreLineageBaselineGenerator")
    with build_campaign_runtime(settings, _environment()):
        pass
    registry = SqliteRegistry(settings.storage.registry_database)
    control = SqliteGatewayControl(
        settings.storage.gateway_database, registry, signing_key=b"k" * 32
    )
    try:
        bootstrap.build_core_lineage_baseline_generator(
            settings, LocalArtifactStore(settings.storage.artifacts_root),
            registry, control, _environment(),
        )
    finally:
        control.close()
        registry.close()
    policies = [item for item in observed if "operations" in item]
    assert len(policies) == 2
    assert all(GatewayOperation.WIKI_QUERY not in item["operations"] for item in policies)
    sessions = [item for item in observed if "wiki_enabled" in item]
    assert len(sessions) == 2
    assert all(item["wiki_enabled"] is False for item in sessions)


def test_checked_in_comparison_evaluator_is_the_pinned_submodule(tmp_path: Path) -> None:
    repository = Path(__file__).resolve().parents[1]
    settings = _settings(tmp_path)
    assert settings.campaign is not None
    evaluator = settings.campaign.gate_policy.evaluator
    evaluator_repository = Path(evaluator.repository)

    gate = settings.campaign.gate_policy
    assert (
        gate.optimizer.correctness_cases,
        gate.optimizer.bench_iters,
        gate.optimizer.evaluate_repeats,
    ) == (5, 100, 1)
    assert [
        (stage.correctness_cases, stage.evaluate_repeats) for stage in gate.bootstrap.stages
    ] == [(1, 1), (5, 1)]
    assert gate.bootstrap.bench_iters == gate.optimizer.bench_iters == 100
    production = json.loads((repository / "scripts/production/policy.json").read_text())
    assert production["gate_policy"]["bootstrap"]["bench_iters"] == 100
    assert production["comparison"]["max_parallel_shape_batches"] == 16
    assert (gate.retention.correctness_cases, gate.retention.bench_iters) == (1, 100)
    assert gate.production_gate is True
    assert (gate.warmup_iters, gate.atol, gate.rtol) == (10, 0.01, 0.05)
    assert (
        gate.evaluation_timeout_seconds,
        gate.candidate_timeout_seconds,
        gate.performance_timeout_seconds,
        gate.lock_clocks,
    ) == (600, 120, 120, True)
    for comparison in (
        settings.campaign.kernel_retention_comparison,
        settings.campaign.agent_promotion_comparison,
    ):
        assert comparison.method == "same_allocation_abba"
        assert comparison.repeats == 2
        assert comparison.minimum_improvement_percent == 0.0
        assert comparison.allocation_timeout_seconds == 600
        assert comparison.shape_batch_size == 1
        assert comparison.max_parallel_shape_batches == 16

    assert evaluator_repository == (repository / "third_party/atrex-bench").resolve()
    resolved = subprocess.run(
        (
            str(evaluator.git_executable),
            "-C",
            str(evaluator_repository),
            "rev-parse",
            "HEAD^{commit}",
        ),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert resolved == evaluator.commit
    assert (evaluator_repository / "scripts/run_eval.py").is_file()
    assert (evaluator_repository / "src/atrex_bench/sdk.py").is_file()
    bundle = CommitPinnedAtrexBenchEvaluator(
        repository=evaluator.repository,
        commit=evaluator.commit,
        git_executable=evaluator.git_executable,
        fetch_timeout_seconds=evaluator.fetch_timeout_seconds,
        max_archive_bytes=evaluator.max_archive_bytes,
        max_bundle_files=evaluator.max_bundle_files,
        max_bundle_bytes=evaluator.max_bundle_bytes,
    ).files()
    assert "atrex-bench/scripts/run_eval.py" in bundle
    assert "atrex-bench/src/atrex_bench/sdk.py" in bundle


def test_checked_in_campaign_defers_evolver_credentials_for_zero_challengers(
    tmp_path: Path,
) -> None:
    environment = _environment()
    del environment["ANTHROPIC_AUTH_TOKEN"]

    with build_campaign_runtime(_settings(tmp_path), environment):
        pass


def test_checked_in_bootstrap_commit_imports_current_core_bundle(tmp_path: Path) -> None:
    repository = Path(__file__).resolve().parents[1]
    settings = _settings(tmp_path)
    source = settings.kernel_agent.base_source
    assert source is not None
    spec = json.loads((repository / "examples/bootstrap/campaign.json").read_text())
    artifacts = LocalArtifactStore(tmp_path / "bootstrap-artifacts")
    loader = GitOptimizerBaseLoader(
        artifacts,
        KernelAgentRevisionBuilder(artifacts, limits=settings.kernel_agent.bundle_limits()),
        repository=source.repository,
        git_executable=source.git_executable,
        timeout_seconds=source.fetch_timeout_seconds,
        max_archive_bytes=source.max_archive_bytes,
        allowed_submodules=source.allowed_submodules,
    )

    result = loader.build_candidate(Dsl.TRITON, spec["base_revision"]["commit"])

    assert (
        artifacts.verify(result.candidate.optimizer_digest)
        .payload_path.joinpath("atrex-bundle.json")
        .is_file()
    )


def test_checked_in_bootstrap_campaign_resolves_complete_seed_inputs() -> None:
    repository = Path(__file__).resolve().parents[1]

    spec = load_campaign_spec(repository / "examples/bootstrap/campaign.json")
    lineage = spec.lineages[Dsl.TRITON]
    contract = AgateEvaluationContractV1.model_validate_json(spec.evaluation_contract.read_bytes())

    assert spec.agent_problem is not None and spec.agent_problem.is_file()
    problem = AgentProblemV1.model_validate_json(spec.agent_problem.read_bytes())
    assert problem.schema_version == "atrex.agent_problem.v1"
    assert lineage.baseline_kernel.joinpath(contract.candidate_path).is_file()
    assert any(lineage.initial_evidence.iterdir())


def test_checked_in_lineage_seed_template_is_strict_and_artifact_based() -> None:
    repository = Path(__file__).resolve().parents[1]

    spec = LineageSeedSpecV1.from_file(repository / "lineage-seed.example.json")

    assert isinstance(spec.seed, ArtifactLineageSeedV1)
    assert spec.dsl is Dsl.TRITON
    assert spec.attempts_per_trajectory == 3


def test_runtime_examples_own_their_configs_and_share_only_canonical_inputs() -> None:
    repository = Path(__file__).resolve().parents[1]
    shared = (repository / "examples/shared").resolve()

    for name in RUNTIME_EXAMPLES:
        example = repository / "examples" / name
        settings = RuntimeSettings.from_file(example / "runtime.json")
        assert settings.campaign is not None
        gate = settings.campaign.gate_policy
        assert gate.bootstrap.bench_iters == gate.optimizer.bench_iters == 100
        for comparison in (
            settings.campaign.kernel_retention_comparison,
            settings.campaign.agent_promotion_comparison,
        ):
            assert comparison.method == "same_allocation_abba"
            assert comparison.max_parallel_shape_batches == 16
        campaign = load_campaign_spec(example / "campaign.json")
        assert campaign.evaluation_contract.is_relative_to(shared)
        assert campaign.agent_problem is not None
        assert campaign.agent_problem.is_relative_to(shared)
        for lineage in campaign.lineages.values():
            assert lineage.baseline_kernel.is_relative_to(shared)
            assert lineage.initial_evidence.is_relative_to(shared)


def test_runnable_examples_never_import_another_runnable_example() -> None:
    repository = Path(__file__).resolve().parents[1]
    examples = repository / "examples"

    for current in RUNNABLE_EXAMPLES:
        for path in (examples / current).iterdir():
            if not path.is_file() or path.suffix not in {".py", ".sh"}:
                continue
            source = path.read_text(encoding="utf-8")
            assert "runtime.example.json" not in source
            for other in RUNNABLE_EXAMPLES:
                if other != current:
                    assert f"examples/{other}/" not in source
