"""Tests for fixed local Evolver Bundle identity validation."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from atrex_runtime.artifacts import ArtifactKind, LocalArtifactStore
from atrex_runtime.workers.evolver_bundle import (
    GitEvolverBundleResolver,
    LocalEvolverBundleResolver,
    evolver_bundle_sha256,
)


def _bundle(tmp_path: Path) -> Path:
    root = tmp_path / "evolver"
    (root / "src").mkdir(parents=True)
    (root / "src/main.py").write_text("raise SystemExit(0)\n", encoding="utf-8")
    (root / "atrex-evolver-bundle.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "bundle_format": "atrex-kernel-agent-evolver-bundle-v1",
                "entrypoint": {"command": "src/main.py"},
            }
        ),
        encoding="utf-8",
    )
    return root


def _commit_repository(root: Path) -> str:
    subprocess.run(("git", "init", "-q", str(root)), check=True)
    subprocess.run(("git", "-C", str(root), "add", "."), check=True)
    environment = {
        **os.environ,
        "GIT_AUTHOR_NAME": "ATREX Test",
        "GIT_AUTHOR_EMAIL": "atrex@example.invalid",
        "GIT_COMMITTER_NAME": "ATREX Test",
        "GIT_COMMITTER_EMAIL": "atrex@example.invalid",
    }
    subprocess.run(
        ("git", "-C", str(root), "commit", "-q", "-m", "test bundle"),
        check=True,
        env=environment,
    )
    return subprocess.run(
        ("git", "-C", str(root), "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def test_resolver_binds_launch_to_complete_bundle_digest(tmp_path: Path) -> None:
    root = _bundle(tmp_path)
    digest = evolver_bundle_sha256(root)

    resolved = LocalEvolverBundleResolver(
        root,
        expected_sha256=digest,
        command_prefix=(str(Path(sys.executable).resolve()),),
        max_files=16,
        max_bytes=65536,
    ).resolve()

    assert resolved.digest == digest
    assert resolved.command_argv[-1] == str(root / "src/main.py")


def test_resolver_rejects_bundle_changed_after_configuration(tmp_path: Path) -> None:
    root = _bundle(tmp_path)
    digest = evolver_bundle_sha256(root)
    (root / "src/main.py").write_text("raise SystemExit(1)\n", encoding="utf-8")

    with pytest.raises(ValueError, match="digest mismatch"):
        LocalEvolverBundleResolver(
            root,
            expected_sha256=digest,
            command_prefix=(str(Path(sys.executable).resolve()),),
            max_files=16,
            max_bytes=65536,
        ).resolve()


def test_cache_and_git_metadata_do_not_change_deployed_behavior_digest(tmp_path: Path) -> None:
    root = _bundle(tmp_path)
    digest = evolver_bundle_sha256(root)
    (root / ".git").mkdir()
    (root / ".git/HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    (root / "src/__pycache__").mkdir()
    (root / "src/__pycache__/main.pyc").write_bytes(b"cache")

    assert evolver_bundle_sha256(root) == digest


def test_git_resolver_fetches_exact_commit_into_sealed_artifact(tmp_path: Path) -> None:
    root = _bundle(tmp_path)
    commit = _commit_repository(root)
    artifacts = LocalArtifactStore(tmp_path / "artifacts")

    resolved = GitEvolverBundleResolver(
        artifacts,
        repository=str(root),
        commit=commit,
        git_executable="/usr/bin/git",
        fetch_timeout_seconds=10,
        max_archive_bytes=1048576,
        command_prefix=(str(Path(sys.executable).resolve()),),
        max_files=16,
        max_bytes=65536,
    ).resolve()

    stored = artifacts.verify(resolved.artifact_digest)
    assert resolved.commit == commit
    assert len(resolved.tree) == 40
    assert stored.kind is ArtifactKind.EVOLVER_BUNDLE
    assert resolved.command_argv[-1] == str(stored.payload_path / "src/main.py")


def test_git_resolver_ignores_uncommitted_checkout_changes(tmp_path: Path) -> None:
    root = _bundle(tmp_path)
    commit = _commit_repository(root)
    (root / "src/main.py").write_text("raise SystemExit(99)\n", encoding="utf-8")
    artifacts = LocalArtifactStore(tmp_path / "artifacts")

    resolved = GitEvolverBundleResolver(
        artifacts,
        repository=str(root),
        commit=commit,
        git_executable="/usr/bin/git",
        fetch_timeout_seconds=10,
        max_archive_bytes=1048576,
        command_prefix=(str(Path(sys.executable).resolve()),),
        max_files=16,
        max_bytes=65536,
    ).resolve()

    assert Path(resolved.command_argv[-1]).read_text(encoding="utf-8") == ("raise SystemExit(0)\n")


def test_git_resolver_rejects_tracked_symbolic_link(tmp_path: Path) -> None:
    root = _bundle(tmp_path)
    (root / "unsafe-link").symlink_to("src/main.py")
    commit = _commit_repository(root)

    with pytest.raises(ValueError, match="link or submodule"):
        GitEvolverBundleResolver(
            LocalArtifactStore(tmp_path / "artifacts"),
            repository=str(root),
            commit=commit,
            git_executable="/usr/bin/git",
            fetch_timeout_seconds=10,
            max_archive_bytes=1048576,
            command_prefix=(str(Path(sys.executable).resolve()),),
            max_files=16,
            max_bytes=65536,
        ).resolve()
