"""KDA working-tree integration, without a provider call or production-state changes."""

from __future__ import annotations

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
        "skills/KernelWiki/SKILL.md", "skills/ncu-report-skill/SKILL.md",
    ):
        assert (sealed / path).is_file(), f"Bundle is missing {path}; initialize KDA submodules"
    assert not list(sealed.rglob(".git"))


@pytest.mark.parametrize("backend", ("claude", "codex"))
@pytest.mark.parametrize("phase", ("optimization_attempt", "framework_baseline"))
def test_kda_skills_are_seeded_and_installed_per_session(
    exported_bundle: Path, tmp_path: Path, backend: str, phase: str,
) -> None:
    workspace = tmp_path / "workspace"
    repository = workspace / "agent/optimizer"
    shutil.copytree(exported_bundle, repository)
    initialize_reusable_agent_state(workspace, repository)
    remove_optimizer_state_seeds(repository)
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
    for name in ("KernelWiki", "ncu-report-skill"):
        source = workspace / "skills" / name / "SKILL.md"
        installed = discovery / name / "SKILL.md"
        assert installed.read_bytes() == source.read_bytes()
        assert installed.stat().st_ino != source.stat().st_ino
    assert result["WORKSPACE_ROOT"] == str(workspace)
    config = json.loads((repository / "atrex-agent.json").read_text())
    assert config["prompts"]["optimization_attempt"] == "prompts/episode.md"


@pytest.mark.parametrize(
    "arguments", (
        ("scripts/query.py", "--tag", "tma", "--compact", "--limit", "1"),
        ("scripts/get_page.py", "kernel-flash-attention-4", "--body-only"),
    ),
)
def test_packaged_wiki_can_query_without_a_service(
    exported_bundle: Path, arguments: tuple[str, ...],
) -> None:
    pytest.importorskip("yaml", reason="install Runtime dependencies")
    result = subprocess.run(
        (sys.executable, *arguments),
        cwd=exported_bundle / "skills/KernelWiki",
        env={
            **{key: value for key, value in os.environ.items() if key != "BLACKWELL_WIKI_ROOT"},
            "PYTHONDONTWRITEBYTECODE": "1",
        },
        capture_output=True, text=True, timeout=30, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip()


def test_git_import_expands_both_real_pinned_skills(
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

    skills = ("skills/KernelWiki", "skills/ncu-report-skill")
    approved = {name: (KDA / name).as_uri() for name in skills}
    commits = {name: git("rev-parse", "HEAD", cwd=KDA / name) for name in skills}
    (repository / ".gitmodules").write_text("".join(
        f'[submodule "{name}"]\n\tpath = {name}\n\turl = {approved[name]}\n'
        for name in skills
    ))
    git("init")
    git("add", "--", ".", *(f":(exclude){name}" for name in skills))
    for name in skills:
        git("update-index", "--add", "--cacheinfo", f"160000,{commits[name]},{name}")
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
        timeout_seconds=30, max_archive_bytes=268435456, allowed_submodules=approved,
    )
    result = loader.build_candidate(Dsl.TRITON, git("rev-parse", "HEAD"))
    provenance = artifacts.verify(result.source_provenance_digest).payload_path / "value.json"
    value = json.loads(provenance.read_text())
    assert {item["path"]: item["commit"] for item in value["submodules"]} == commits
    sealed = artifacts.verify(result.candidate.optimizer_digest).payload_path
    for name in skills:
        assert (sealed / name / "SKILL.md").read_bytes() == (KDA / name / "SKILL.md").read_bytes()
    assert not list(sealed.rglob(".git"))
