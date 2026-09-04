"""Full-repository Kernel Agent revision validation and sealing tests."""

from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path

import pytest
from conftest import NOW, kernel_agent_limits

from atrex_runtime.artifacts.local import LocalArtifactStore
from atrex_runtime.domain.ids import new_kernel_agent_revision_id
from atrex_runtime.domain.models import Dsl, KernelAgentRevision
from atrex_runtime.kernel_agents import KernelAgentRevisionBuilder


def _source(tmp_path: Path) -> Path:
    root = tmp_path / "agent"
    prompt = root / "prompts/episode.md"
    prompt.parent.mkdir(parents=True)
    docs = root / "docs/design.md"
    docs.parent.mkdir(parents=True)
    (root / "src").mkdir()
    prompt.write_text("Optimize through Runtime tools.\n", encoding="utf-8")
    docs.write_text("# Agent design\n", encoding="utf-8")
    (root / "atrex-bundle.json").write_text(
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
    (root / "src/main.py").write_text("def optimize(): ...\n", encoding="utf-8")
    return root


def test_builder_seals_complete_repository(tmp_path: Path) -> None:
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    candidate = KernelAgentRevisionBuilder(
        artifacts,
        limits=kernel_agent_limits(),
    ).build_candidate(_source(tmp_path), Dsl.TRITON)

    assert candidate.dsl is Dsl.TRITON
    stored = artifacts.verify(candidate.optimizer_digest).payload_path
    assert (stored / "src/main.py").is_file()
    assert (stored / "prompts/episode.md").is_file()
    assert (stored / "docs/design.md").is_file()


@pytest.mark.parametrize("name", ("prompts", "memory", "knowledge", "skills", "tools", "hooks"))
def test_builder_seals_top_level_adaptive_state_seeds(
    tmp_path: Path,
    name: str,
) -> None:
    source = _source(tmp_path)
    seed = source / name
    seed.mkdir(exist_ok=True)
    (seed / "state.md").write_text("initial state\n", encoding="utf-8")
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    candidate = KernelAgentRevisionBuilder(
        artifacts, limits=kernel_agent_limits()
    ).build_candidate(source, Dsl.TRITON)
    stored = artifacts.verify(candidate.optimizer_digest).payload_path
    assert (stored / name / "state.md").read_text() == "initial state\n"


def test_builder_allows_non_entry_repository_content(tmp_path: Path) -> None:
    source = _source(tmp_path)
    (source / "docs").mkdir(exist_ok=True)
    (source / "docs/design.md").write_text("supporting documentation\n")
    candidate = KernelAgentRevisionBuilder(
        LocalArtifactStore(tmp_path / "artifacts"),
        limits=kernel_agent_limits(),
    ).build_candidate(source, Dsl.CUDA)

    assert candidate.dsl is Dsl.CUDA


def test_builder_excludes_generated_cache_content_from_revision(tmp_path: Path) -> None:
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    source = _source(tmp_path)
    baseline = KernelAgentRevisionBuilder(
        artifacts,
        limits=kernel_agent_limits(),
    ).build_candidate(source, Dsl.TRITON)
    (source / "src/__pycache__").mkdir()
    (source / "src/__pycache__/main.cpython-314.pyc").write_bytes(b"generated")
    (source / ".pytest_cache").mkdir()
    (source / ".pytest_cache/state").write_text("generated", encoding="utf-8")
    (source / ".coverage").write_text("generated", encoding="utf-8")

    with_cache = KernelAgentRevisionBuilder(
        artifacts,
        limits=kernel_agent_limits(),
    ).build_candidate(source, Dsl.TRITON)

    assert with_cache.optimizer_digest == baseline.optimizer_digest
    stored = artifacts.verify(with_cache.optimizer_digest).payload_path
    assert not (stored / "src/__pycache__").exists()
    assert not (stored / ".pytest_cache").exists()
    assert not (stored / ".coverage").exists()


def test_builder_rejects_runtime_selected_agent_adapter(tmp_path: Path) -> None:
    source = _source(tmp_path)
    manifest_path = source / "atrex-bundle.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["entrypoint"]["dsh"] = {
        "prompt": "prompts/episode.md",
        "skills": ["skills/example/SKILL.md"],
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        KernelAgentRevisionBuilder(
            LocalArtifactStore(tmp_path / "artifacts"),
            limits=kernel_agent_limits(),
        ).build_candidate(source, Dsl.TRITON)


def test_builder_requires_current_bundle_manifest_name_and_format(tmp_path: Path) -> None:
    source = _source(tmp_path)
    manifest_path = source / "atrex-bundle.json"
    legacy_path = source / "atrex-optimizer.json"
    manifest_path.rename(legacy_path)
    builder = KernelAgentRevisionBuilder(
        LocalArtifactStore(tmp_path / "artifacts"), limits=kernel_agent_limits()
    )

    with pytest.raises(ValueError, match="Bundle manifest is unavailable"):
        builder.build_candidate(source, Dsl.TRITON)

    legacy_path.rename(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["bundle_format"] = "atrex-optimizer-repository-v1"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="Bundle manifest is invalid"):
        builder.build_candidate(source, Dsl.TRITON)


def test_builder_rejects_missing_or_oversized_entry_file(tmp_path: Path) -> None:
    source = _source(tmp_path)
    command = source / "src/main.py"
    command.write_text("x" * 20)
    builder = KernelAgentRevisionBuilder(
        LocalArtifactStore(tmp_path / "artifacts"),
        limits=replace(kernel_agent_limits(), max_entrypoint_bytes=16),
    )

    with pytest.raises(ValueError, match="Optimizer command file exceeds byte limit"):
        builder.build_candidate(source, Dsl.TRITON)

    command.unlink()
    with pytest.raises(ValueError, match="Optimizer command file does not exist"):
        builder.build_candidate(source, Dsl.TRITON)


def test_builder_rejects_git_metadata_and_links(tmp_path: Path) -> None:
    source = _source(tmp_path)
    (source / ".git").write_text("gitdir: elsewhere\n")
    builder = KernelAgentRevisionBuilder(
        LocalArtifactStore(tmp_path / "artifacts"), limits=kernel_agent_limits()
    )

    with pytest.raises(ValueError, match="Git metadata"):
        builder.build_candidate(source, Dsl.TRITON)

    (source / ".git").unlink()
    try:
        os.symlink(source / "src/main.py", source / "linked.py")
    except (OSError, NotImplementedError):
        pytest.skip("symbolic links are unavailable")
    with pytest.raises(ValueError, match="symbolic links"):
        builder.build_candidate(source, Dsl.TRITON)


def test_challenger_requires_same_dsl_and_repository_change(tmp_path: Path) -> None:
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    builder = KernelAgentRevisionBuilder(artifacts, limits=kernel_agent_limits())
    first = builder.build_candidate(_source(tmp_path), Dsl.TRITON)
    parent = KernelAgentRevision(
        id=new_kernel_agent_revision_id(),
        parent_id=None,
        creation_key="bootstrap:test",
        dsl=first.dsl,
        optimizer_digest=first.optimizer_digest,
        created_by="bootstrap",
        created_at=NOW,
        source_provenance_digest=first.optimizer_digest,
    )

    with pytest.raises(ValueError, match="no Optimizer repository changes"):
        builder.validate_challenger(parent, first)

    changed_root = _source(tmp_path / "changed")
    (changed_root / "src/main.py").write_text("def optimize(): return 1\n")
    changed = builder.build_candidate(changed_root, Dsl.TRITON)
    builder.validate_challenger(parent, changed)

    wrong_dsl = builder.build_candidate(changed_root, Dsl.CUDA)
    with pytest.raises(ValueError, match="lineage DSL"):
        builder.validate_challenger(parent, wrong_dsl)
