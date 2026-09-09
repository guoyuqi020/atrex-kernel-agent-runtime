"""KDA working-tree integration, without a provider call or production-state changes."""

from __future__ import annotations

import json
import shutil
import subprocess
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
    not (KDA / "atrex-bundle.json").is_file(), reason="KDA Optimizer submodule is not initialized",
)


@pytest.fixture(scope="module")
def exported_bundle(tmp_path_factory: pytest.TempPathFactory) -> Path:
    destination = tmp_path_factory.mktemp("kda") / "export"
    shutil.copytree(
        KDA, destination,
        ignore=shutil.ignore_patterns(
            ".git", "__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache", "*.pyc",
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
        "src/main.py", "src/runtime_tools.py", "CLAUDE.md", "prompts/episode.md",
        "skills/README.md",
    ):
        assert (sealed / path).is_file(), f"Bundle is missing {path}"
    assert not list(sealed.rglob(".git"))
    assert not list((sealed / "skills").rglob("SKILL.md"))
    assert not (sealed / ".gitmodules").exists()


@pytest.mark.parametrize("backend", ("claude", "codex"))
@pytest.mark.parametrize("phase", ("optimization_attempt", "framework_baseline"))
@pytest.mark.parametrize("custom_skill", (False, True))
def test_kda_skills_are_seeded_and_installed_per_session(
    exported_bundle: Path, tmp_path: Path, backend: str, phase: str, custom_skill: bool,
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
        workspace, {"HOME": str(home), "ATREX_CORE_PHASE": phase}, (backend,),
    )
    for name in REUSABLE_AGENT_DIRECTORIES:
        assert (workspace / name / "README.md").is_file()
        assert not (repository / name).exists()
    assert (repository / "CLAUDE.md").is_file()
    assert not (workspace / "CLAUDE.md").exists()
    discovery = home / (".claude/skills" if backend == "claude" else ".agents/skills")
    for name in (("example-method",) if custom_skill else ()):
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
    exported_bundle: Path, tmp_path: Path,
) -> None:
    executable = shutil.which("git")
    if executable is None:
        pytest.skip("Git is unavailable")
    repository = tmp_path / "repository"
    shutil.copytree(exported_bundle, repository)

    def git(*arguments: str, cwd: Path = repository) -> str:
        return subprocess.run(
            (executable, "-C", str(cwd), *arguments), check=True,
            capture_output=True, text=True, timeout=30,
        ).stdout.strip()

    assert not (repository / ".gitmodules").exists()
    git("init")
    git("add", "--", ".")
    git(
        "-c", "user.name=Bundle Test", "-c", "user.email=bundle@example.test",
        "-c", "commit.gpgsign=false", "-c", "core.hooksPath=/dev/null",
        "commit", "-m", "Temporary migration test snapshot",
    )
    settings = KernelAgentSettings.model_validate_json(
        KDA_CONFIG.read_text(),
    )
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    loader = GitOptimizerBaseLoader(
        artifacts, KernelAgentRevisionBuilder(artifacts, limits=settings.bundle_limits()),
        repository=repository.as_uri(), git_executable=executable,
        timeout_seconds=30, max_archive_bytes=268435456,
    )
    result = loader.build_candidate(Dsl.TRITON, git("rev-parse", "HEAD"))
    provenance = artifacts.verify(result.source_provenance_digest).payload_path / "value.json"
    value = json.loads(provenance.read_text())
    assert value["submodules"] == []
    sealed = artifacts.verify(result.candidate.optimizer_digest).payload_path
    assert (sealed / "skills/README.md").read_bytes() == (KDA / "skills/README.md").read_bytes()
    assert not list((sealed / "skills").rglob("SKILL.md"))
    assert not list(sealed.rglob(".git"))
