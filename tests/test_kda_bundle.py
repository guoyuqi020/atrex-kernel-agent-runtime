"""KDA working-tree integration, without a provider call or production-state changes."""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from atrex_runtime.artifacts.local import LocalArtifactStore
from atrex_runtime.config import KernelAgentSettings
from atrex_runtime.domain.models import Dsl
from atrex_runtime.kernel_agents.git import GitOptimizerBaseLoader
from atrex_runtime.kernel_agents.revision import KernelAgentRevisionBuilder
from atrex_runtime.workers.extensions import install_optimizer_extensions
from atrex_runtime.workers.workspace import (
    REUSABLE_AGENT_DIRECTORIES,
    initialize_reusable_agent_state,
    remove_optimizer_state_seeds,
)

RUNTIME_ROOT = Path(__file__).resolve().parents[1]
KDA = RUNTIME_ROOT / "src/kernel-design-agents"
KDA_CONFIG = RUNTIME_ROOT / "examples/kernel-design-agents/kernel-agent.example.json"
pytestmark = pytest.mark.skipif(
    not (KDA / "atrex-bundle.json").is_file(),
    reason="KDA Optimizer submodule is not initialized",
)


@pytest.fixture(scope="module")
def exported_bundle(tmp_path_factory: pytest.TempPathFactory) -> Path:
    destination = tmp_path_factory.mktemp("kda") / "export"
    shutil.copytree(
        KDA,
        destination,
        ignore=shutil.ignore_patterns(
            ".git",
            "__pycache__",
            ".pytest_cache",
            ".ruff_cache",
            ".mypy_cache",
            "*.pyc",
        ),
    )
    return destination


def test_complete_kda_bundle_can_be_sealed(exported_bundle: Path, tmp_path: Path) -> None:
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    settings = KernelAgentSettings.model_validate_json(
        KDA_CONFIG.read_text(),
    )
    builder = KernelAgentRevisionBuilder(artifacts, limits=settings.bundle_limits())
    candidate = builder.build_candidate(exported_bundle, Dsl.TRITON)
    sealed = artifacts.verify(candidate.optimizer_digest).payload_path
    for path in (
        "src/main.py",
        "src/runtime_tools.py",
        "CLAUDE.md",
        "prompts/episode.md",
        "skills/README.md",
        "workflow/main.py",
        "workflow/runtime.py",
        "workflow/README.md",
    ):
        assert (sealed / path).is_file(), f"Bundle is missing {path}"
    for name in (
        "evolve_3.py",
        "isolated.py",
        "retained.py",
        "pool_3.py",
        "pool_retained_3.py",
    ):
        assert not (sealed / "workflow" / name).exists()
    assert not list(sealed.rglob(".git"))
    assert not list((sealed / "skills").rglob("SKILL.md"))
    assert not (sealed / ".gitmodules").exists()


def _respond(process: subprocess.Popen[str], call: dict[str, object], result: object) -> None:
    assert process.stdin is not None
    process.stdin.write(
        json.dumps({"request_id": call["request_id"], "ok": True, "result": result}) + "\n"
    )
    process.stdin.flush()


def _accept_trajectory(process: subprocess.Popen[str]) -> dict[str, object]:
    assert process.stdout is not None
    call = json.loads(process.stdout.readline())
    assert call["operation"] == "create_trajectory"
    arguments = call["arguments"]
    _respond(
        process,
        call,
        {
            "branch": arguments["branch"],
            "trajectory_ordinal": arguments["trajectory_ordinal"],
            "attempt_capacity": arguments["attempt_capacity"],
            "runtime_state_policy": arguments["runtime_state_policy"],
            "kernel_agent_revision_id": "agentrev_" + "0" * 32,
        },
    )
    return call


def _accept_attempt_batch(process: subprocess.Popen[str]) -> dict[str, object]:
    assert process.stdout is not None
    call = json.loads(process.stdout.readline())
    assert call["operation"] == "run_attempts_parallel"
    launches = call["arguments"]["launches"]
    _respond(process, call, {"attempts": [{"status": "completed"} for _ in launches]})
    return call


def _finish_workflow(process: subprocess.Popen[str], epoch_id: str) -> None:
    assert process.stdout is not None
    for operation, result in (
        ("select_best_kernel", {"kernel_revision_id": "kernelrev_" + "1" * 32}),
        ("compare_agents", {"kernel_agent_revision_id": "agentrev_" + "2" * 32}),
        ("complete_epoch", {"epoch_id": epoch_id, "status": "completed"}),
    ):
        call = json.loads(process.stdout.readline())
        assert call["operation"] == operation
        _respond(process, call, result)


@pytest.mark.parametrize("bundle_name", ("kernel-design-agents", "atrex-kernel-agent-core"))
def test_epoch_sdk_hides_attempt_bookkeeping_and_routes_between_rounds(
    bundle_name: str,
) -> None:
    runtime_path = RUNTIME_ROOT / "src" / bundle_name / "workflow/runtime.py"
    module_name = f"_test_{bundle_name.replace('-', '_')}_workflow_runtime"
    spec = importlib.util.spec_from_file_location(module_name, runtime_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(module_name, None)

    class FakeClient:
        def __init__(self) -> None:
            self.context = {"epoch_id": "epoch_" + "0" * 32}
            self.limits = {"optimizer_attempts": 4}
            self.launch_batches: list[tuple[object, ...]] = []

        def create_trajectory(self, **arguments: object) -> object:
            return module._Trajectory(
                branch=arguments["branch"],
                ordinal=arguments["ordinal"],
                attempt_capacity=arguments["attempt_capacity"],
                runtime_state_policy=arguments["runtime_state_policy"],
                kernel_agent_revision_id="agentrev_" + "0" * 32,
            )

        def run_attempts_parallel(self, launches: object) -> list[dict[str, object]]:
            batch = tuple(launches)
            self.launch_batches.append(batch)
            return [
                {
                    "attempt_id": f"attempt_{launch.ordinal:032d}",
                    "accepted": True,
                    "latency_us": 10.0 + launch.trajectory.ordinal,
                    "trajectory_kernel_revision_id": (
                        f"kernelrev_{launch.ordinal:016d}{launch.trajectory.ordinal:016d}"
                    ),
                }
                for launch in batch
            ]

        def select_best_kernel(self) -> str:
            return "kernelrev_" + "1" * 32

        def compare_agents(self) -> str:
            return "agentrev_" + "2" * 32

        def complete_epoch(self, **_arguments: object) -> dict[str, object]:
            return {"status": "completed"}

    client = FakeClient()
    epoch = module.EpochRuntime(client)
    pool = epoch.create_pool(
        branch="active",
        trajectories=2,
        rounds=2,
        runtime_state_policy="retain_across_attempts",
    )

    def broadcast_best(completed: object) -> None:
        best = completed.best_accepted_kernel(pool)
        if best is not None and completed.number < pool.rounds:
            completed.route_kernel(pool, best)

    rounds = epoch.run_pools([pool], after_round=broadcast_best)
    assert [item.number for item in rounds] == [1, 2]
    assert [launch.ordinal for launch in client.launch_batches[0]] == [1, 1]
    expected = "kernelrev_00000000000000010000000000000001"
    assert all(launch.input_kernel_revision_id == expected for launch in client.launch_batches[1])
    assert epoch.complete() == {"status": "completed"}
    assert "AttemptLaunch" not in module.__all__
    assert "WorkflowRuntime" not in module.__all__


@pytest.mark.parametrize("bundle_name", ("kernel-design-agents", "atrex-kernel-agent-core"))
def test_default_workflow_executes_complete_pool_epoch(bundle_name: str) -> None:
    bundle = RUNTIME_ROOT / "src" / bundle_name
    process = subprocess.Popen(
        (sys.executable, str(bundle / "workflow/main.py")),
        cwd=bundle / "workflow",
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert process.stdin is not None
    assert process.stdout is not None
    context = {
        "schema_version": 1,
        "operation": "run_epoch",
        "context": {
            "kernel_agent_revision_id": "agentrev_00000000000000000000000000000000",
            "dsl": "triton",
            "epoch_id": "epoch_00000000000000000000000000000000",
            "epoch_number": 2,
            "first_epoch_same_agent": False,
            "workflow_program_sha256": "a" * 64,
        },
        "limits": {
            "max_challengers": 0,
            "optimizer_attempts": 6,
            "default_trajectories": 2,
            "default_attempts_per_trajectory": 3,
            "default_runtime_state_policy": "retain_across_attempts",
        },
    }
    process.stdin.write(json.dumps(context) + "\n")
    process.stdin.flush()

    created = [_accept_trajectory(process), _accept_trajectory(process)]
    assert [call["arguments"]["trajectory_ordinal"] for call in created] == [1, 2]
    assert all(
        call["arguments"]
        == {
            "branch": "active",
            "trajectory_ordinal": ordinal,
            "trajectory_count": 2,
            "attempt_capacity": 3,
            "runtime_state_policy": "retain_across_attempts",
        }
        for ordinal, call in enumerate(created, start=1)
    )
    batches = [_accept_attempt_batch(process) for _ in range(3)]
    assert [
        [launch["attempt_ordinal"] for launch in call["arguments"]["launches"]] for call in batches
    ] == [[1, 1], [2, 2], [3, 3]]
    _finish_workflow(process, context["context"]["epoch_id"])
    process.stdin.close()
    assert process.wait(timeout=5) == 0


@pytest.mark.parametrize("bundle_name", ("kernel-design-agents", "atrex-kernel-agent-core"))
@pytest.mark.parametrize(
    ("program", "budget", "expected"),
    (
        ("isolated.py", 3, (1, 3, "reset_each_attempt")),
        ("retained.py", 3, (1, 3, "retain_across_attempts")),
        ("pool_3.py", 6, (2, 3, "reset_each_attempt")),
        ("pool_retained_3.py", 6, (2, 3, "retain_across_attempts")),
    ),
)
def test_control_workflow_program_owns_exact_topology(
    bundle_name: str,
    program: str,
    budget: int,
    expected: tuple[int, int, str],
) -> None:
    bundle = RUNTIME_ROOT / "src" / bundle_name
    program_path = RUNTIME_ROOT / "src/atrex_runtime/workflow_templates" / program
    process = subprocess.Popen(
        (sys.executable, str(program_path)),
        cwd=bundle / "workflow",
        env={**os.environ, "PYTHONPATH": str(bundle / "workflow")},
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert process.stdin is not None
    assert process.stdout is not None
    context = {
        "schema_version": 1,
        "operation": "run_epoch",
        "context": {
            "kernel_agent_revision_id": "agentrev_" + "0" * 32,
            "dsl": "triton",
            "epoch_id": "epoch_" + "0" * 32,
            "epoch_number": 1,
            "first_epoch_same_agent": False,
            "workflow_program_sha256": "a" * 64,
        },
        "limits": {
            "max_challengers": 0,
            "optimizer_attempts": budget,
            "default_trajectories": 9,
            "default_attempts_per_trajectory": 9,
            "default_runtime_state_policy": "retain_across_attempts",
        },
    }
    process.stdin.write(json.dumps(context) + "\n")
    process.stdin.flush()

    trajectories, attempts, policy = expected
    created = [_accept_trajectory(process) for _ in range(trajectories)]
    assert all(
        call["arguments"]["trajectory_count"] == trajectories
        and call["arguments"]["attempt_capacity"] == attempts
        and call["arguments"]["runtime_state_policy"] == policy
        for call in created
    )
    batches = [_accept_attempt_batch(process) for _ in range(attempts)]
    assert all(len(call["arguments"]["launches"]) == trajectories for call in batches)
    _finish_workflow(process, context["context"]["epoch_id"])
    process.stdin.close()
    assert process.wait(timeout=5) == 0


@pytest.mark.parametrize("bundle_name", ("kernel-design-agents", "atrex-kernel-agent-core"))
def test_evolve_3_workflow_owns_active_challenger_organization(bundle_name: str) -> None:
    bundle = RUNTIME_ROOT / "src" / bundle_name
    program_path = RUNTIME_ROOT / "src/atrex_runtime/workflow_templates/evolve_3.py"
    process = subprocess.Popen(
        (sys.executable, str(program_path)),
        cwd=bundle / "workflow",
        env={**os.environ, "PYTHONPATH": str(bundle / "workflow")},
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert process.stdin is not None
    assert process.stdout is not None
    context = {
        "schema_version": 1,
        "operation": "run_epoch",
        "context": {
            "kernel_agent_revision_id": "agentrev_" + "0" * 32,
            "dsl": "triton",
            "epoch_id": "epoch_" + "0" * 32,
            "epoch_number": 2,
            "first_epoch_same_agent": True,
            "workflow_program_sha256": "a" * 64,
        },
        "limits": {
            "max_challengers": 1,
            "optimizer_attempts": 6,
            "default_trajectories": 1,
            "default_attempts_per_trajectory": 3,
            "default_runtime_state_policy": "retain_across_attempts",
        },
    }
    process.stdin.write(json.dumps(context) + "\n")
    process.stdin.flush()
    evolve = json.loads(process.stdout.readline())
    assert evolve["operation"] == "evolve_agent"
    process.stdin.write(
        json.dumps(
            {
                "request_id": evolve["request_id"],
                "ok": True,
                "result": {"kernel_agent_revision_id": "agentrev_" + "3" * 32},
            }
        )
        + "\n"
    )
    process.stdin.flush()
    created = [_accept_trajectory(process), _accept_trajectory(process)]
    assert [call["arguments"]["branch"] for call in created] == [
        "active",
        "challenger-1",
    ]
    batches = [_accept_attempt_batch(process) for _ in range(3)]
    assert all(len(call["arguments"]["launches"]) == 2 for call in batches)
    _finish_workflow(process, context["context"]["epoch_id"])
    process.stdin.close()
    assert process.wait(timeout=5) == 0


@pytest.mark.parametrize("backend", ("claude", "codex"))
@pytest.mark.parametrize("phase", ("optimization_attempt", "framework_baseline"))
@pytest.mark.parametrize("custom_skill", (False, True))
def test_kda_skills_are_seeded_and_installed_per_session(
    exported_bundle: Path,
    tmp_path: Path,
    backend: str,
    phase: str,
    custom_skill: bool,
) -> None:
    workspace = tmp_path / "workspace"
    repository = workspace / "agent/optimizer"
    shutil.copytree(exported_bundle, repository)
    initialize_reusable_agent_state(workspace, repository)
    remove_optimizer_state_seeds(repository)
    if custom_skill:
        skill = workspace / "skills/example-method"
        skill.mkdir()
        (skill / "SKILL.md").write_text(
            "---\nname: example-method\ndescription: A reusable test procedure.\n---\n"
            "Use the supplied task and tool contracts.\n"
        )
    home = workspace / "sessions/agent-home"
    home.mkdir(parents=True)
    result = install_optimizer_extensions(
        workspace,
        {"HOME": str(home), "ATREX_CORE_PHASE": phase},
        (backend,),
    )
    for name in REUSABLE_AGENT_DIRECTORIES:
        assert (workspace / name / "README.md").is_file()
        assert not (repository / name).exists()
    assert (repository / "CLAUDE.md").is_file()
    assert not (workspace / "CLAUDE.md").exists()
    discovery = home / (".claude/skills" if backend == "claude" else ".agents/skills")
    for name in ("example-method",) if custom_skill else ():
        source = workspace / "skills" / name / "SKILL.md"
        installed = discovery / name / "SKILL.md"
        assert installed.read_bytes() == source.read_bytes()
        assert installed.stat().st_ino != source.stat().st_ino
    assert len(list(discovery.rglob("SKILL.md"))) == int(custom_skill)
    for name in ("KernelWiki", "ncu-report-skill"):
        assert not (workspace / "skills" / name).exists()
        assert not (discovery / name).exists()
    assert result["WORKSPACE_ROOT"] == str(workspace)
    config = json.loads((repository / "atrex-agent.json").read_text())
    assert config["prompts"]["optimization_attempt"] == "prompts/episode.md"


def test_git_import_seals_kda_without_skill_submodules(
    exported_bundle: Path,
    tmp_path: Path,
) -> None:
    executable = shutil.which("git")
    if executable is None:
        pytest.skip("Git is unavailable")
    repository = tmp_path / "repository"
    shutil.copytree(exported_bundle, repository)

    def git(*arguments: str, cwd: Path = repository) -> str:
        return subprocess.run(
            (executable, "-C", str(cwd), *arguments),
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout.strip()

    assert not (repository / ".gitmodules").exists()
    git("init")
    git("add", "--", ".")
    git(
        "-c",
        "user.name=Bundle Test",
        "-c",
        "user.email=bundle@example.test",
        "-c",
        "commit.gpgsign=false",
        "-c",
        "core.hooksPath=/dev/null",
        "commit",
        "-m",
        "Temporary migration test snapshot",
    )
    settings = KernelAgentSettings.model_validate_json(
        KDA_CONFIG.read_text(),
    )
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    loader = GitOptimizerBaseLoader(
        artifacts,
        KernelAgentRevisionBuilder(artifacts, limits=settings.bundle_limits()),
        repository=repository.as_uri(),
        git_executable=executable,
        timeout_seconds=30,
        max_archive_bytes=268435456,
    )
    result = loader.build_candidate(Dsl.TRITON, git("rev-parse", "HEAD"))
    provenance = artifacts.verify(result.source_provenance_digest).payload_path / "value.json"
    value = json.loads(provenance.read_text())
    assert value["submodules"] == []
    sealed = artifacts.verify(result.candidate.optimizer_digest).payload_path
    assert (sealed / "skills/README.md").read_bytes() == (KDA / "skills/README.md").read_bytes()
    assert not list((sealed / "skills").rglob("SKILL.md"))
    assert not list(sealed.rglob(".git"))
