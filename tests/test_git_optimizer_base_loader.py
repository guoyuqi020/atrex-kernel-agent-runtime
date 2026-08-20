"""Trusted Git importer tests for complete Optimizer Base Revisions."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest
from conftest import kernel_agent_limits

from atrex_runtime.artifacts.local import ArtifactKind, LocalArtifactStore
from atrex_runtime.domain.models import Dsl
from atrex_runtime.kernel_agents import GitOptimizerBaseLoader, KernelAgentRevisionBuilder


def _git() -> Path:
    executable = shutil.which("git")
    if executable is None:
        pytest.skip("git is unavailable")
    return Path(executable).resolve()


def _run(repository: Path, *arguments: str) -> str:
    result = subprocess.run(
        (str(_git()), "-C", str(repository), *arguments),
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _repository(tmp_path: Path) -> tuple[Path, str]:
    repository = tmp_path / "core"
    repository.mkdir()
    _run(repository, "init")
    prompt = repository / "prompts/episode.md"
    skill = repository / "skills/episode/SKILL.md"
    prompt.parent.mkdir(parents=True)
    skill.parent.mkdir(parents=True)
    (repository / "src").mkdir()
    prompt.write_text("Optimize through Runtime.\n", encoding="utf-8")
    skill.write_text("# Episode loop\n", encoding="utf-8")
    (repository / "atrex-bundle.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "bundle_format": "atrex-kernel-agent-bundle-v1",
                "entrypoint": {
                    "command": "src/main.py",
                },
            }
        ),
        encoding="utf-8",
    )
    (repository / "src/main.py").write_text("def optimize(): ...\n")
    _run(repository, "add", ".")
    _run(
        repository,
        "-c",
        "user.name=ATREX Test",
        "-c",
        "user.email=atrex@example.test",
        "commit",
        "-m",
        "base",
    )
    return repository, _run(repository, "rev-parse", "HEAD")


def _loader(
    tmp_path: Path,
    repository: Path,
    *,
    allowed_submodules: dict[str, str] | None = None,
) -> tuple[GitOptimizerBaseLoader, LocalArtifactStore]:
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    builder = KernelAgentRevisionBuilder(artifacts, limits=kernel_agent_limits())
    return (
        GitOptimizerBaseLoader(
            artifacts,
            builder,
            repository=repository.as_uri(),
            git_executable=_git(),
            timeout_seconds=10,
            max_archive_bytes=1_000_000,
            allowed_submodules=allowed_submodules,
        ),
        artifacts,
    )


def test_loader_fetches_exact_commit_and_seals_complete_tree(tmp_path: Path) -> None:
    repository, commit = _repository(tmp_path)
    loader, artifacts = _loader(tmp_path, repository)

    result = loader.build_candidate(Dsl.TRITON, commit)

    optimizer = artifacts.verify(result.candidate.optimizer_digest)
    assert optimizer.kind is ArtifactKind.KERNEL_AGENT
    assert (optimizer.payload_path / "src/main.py").is_file()
    provenance = artifacts.verify(result.source_provenance_digest)
    assert provenance.kind is ArtifactKind.OPTIMIZER_SOURCE
    value = json.loads((provenance.payload_path / "value.json").read_text())
    assert value["repository"] == repository.as_uri()
    assert value["commit"] == commit
    assert value["optimizer_digest"] == result.candidate.optimizer_digest


def test_loader_requires_full_lowercase_commit(tmp_path: Path) -> None:
    repository, commit = _repository(tmp_path)
    loader, _artifacts = _loader(tmp_path, repository)

    with pytest.raises(ValueError, match="full lowercase commit"):
        loader.build_candidate(Dsl.TRITON, commit[:12])


def test_loader_rejects_symlink_in_tracked_tree(tmp_path: Path) -> None:
    repository, _commit = _repository(tmp_path)
    try:
        os.symlink("src/main.py", repository / "linked.py")
    except (OSError, NotImplementedError):
        pytest.skip("symbolic links are unavailable")
    _run(repository, "add", "linked.py")
    _run(
        repository,
        "-c",
        "user.name=ATREX Test",
        "-c",
        "user.email=atrex@example.test",
        "commit",
        "-m",
        "add link",
    )
    commit = _run(repository, "rev-parse", "HEAD")
    loader, _artifacts = _loader(tmp_path, repository)

    with pytest.raises(ValueError, match="link or unresolved submodule"):
        loader.build_candidate(Dsl.TRITON, commit)


def _repository_with_submodule(tmp_path: Path) -> tuple[Path, str, Path, str]:
    submodule = tmp_path / "extension"
    submodule.mkdir()
    _run(submodule, "init")
    (submodule / "README.md").write_text("# Trusted extension\n", encoding="utf-8")
    _run(submodule, "add", ".")
    _run(
        submodule,
        "-c",
        "user.name=ATREX Test",
        "-c",
        "user.email=atrex@example.test",
        "commit",
        "-m",
        "extension",
    )
    submodule_commit = _run(submodule, "rev-parse", "HEAD")
    repository, _commit = _repository(tmp_path)
    _run(
        repository,
        "-c",
        "protocol.file.allow=always",
        "submodule",
        "add",
        submodule.as_uri(),
        "vendor/example-extension",
    )
    _run(repository, "add", ".")
    _run(
        repository,
        "-c",
        "user.name=ATREX Test",
        "-c",
        "user.email=atrex@example.test",
        "commit",
        "-m",
        "add extension",
    )
    return repository, _run(repository, "rev-parse", "HEAD"), submodule, submodule_commit


def test_loader_expands_only_approved_pinned_submodule(tmp_path: Path) -> None:
    repository, commit, submodule, submodule_commit = _repository_with_submodule(tmp_path)
    loader, artifacts = _loader(
        tmp_path,
        repository,
        allowed_submodules={"vendor/example-extension": submodule.as_uri()},
    )

    result = loader.build_candidate(Dsl.TRITON, commit)

    optimizer = artifacts.verify(result.candidate.optimizer_digest)
    expanded = optimizer.payload_path / "vendor/example-extension/README.md"
    assert expanded.read_text(encoding="utf-8") == "# Trusted extension\n"
    assert expanded.is_file()
    provenance = artifacts.verify(result.source_provenance_digest)
    value = json.loads((provenance.payload_path / "value.json").read_text())
    assert value["submodules"] == [
        {
            "path": "vendor/example-extension",
            "repository": submodule.as_uri(),
            "commit": submodule_commit,
            "tree": _run(submodule, "rev-parse", "HEAD^{tree}"),
        }
    ]


def test_loader_rejects_unapproved_submodule(tmp_path: Path) -> None:
    repository, commit, _submodule, _submodule_commit = _repository_with_submodule(tmp_path)
    loader, _artifacts = _loader(tmp_path, repository)

    with pytest.raises(ValueError, match="unapproved submodule"):
        loader.build_candidate(Dsl.TRITON, commit)


def test_loader_rejects_submodule_url_mismatch(tmp_path: Path) -> None:
    repository, commit, _submodule, _submodule_commit = _repository_with_submodule(tmp_path)
    loader, _artifacts = _loader(
        tmp_path,
        repository,
        allowed_submodules={"vendor/example-extension": "https://example.invalid/wrong.git"},
    )

    with pytest.raises(ValueError, match="URL is not approved"):
        loader.build_candidate(Dsl.TRITON, commit)
