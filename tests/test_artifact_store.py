"""Behavior tests for the immutable local Artifact Store."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from atrex_runtime.artifacts import (
    ArtifactKind,
    LocalArtifactStore,
)


def test_directory_artifacts_are_deduplicated_and_materialized(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "kernel.py").write_text("def kernel():\n    return 1\n")
    (source / "nested").mkdir()
    (source / "nested" / "metadata.json").write_text('{"dsl":"triton"}')
    store = LocalArtifactStore(tmp_path / "artifacts")

    first = store.put_directory(source, ArtifactKind.KERNEL)
    second = store.put_directory(source, ArtifactKind.KERNEL)

    assert first == second
    verified = store.verify(first)
    assert verified.kind is ArtifactKind.KERNEL
    destination = store.materialize(first, tmp_path / "materialized")
    assert (destination / "kernel.py").read_text() == "def kernel():\n    return 1\n"
    assert (destination / "nested" / "metadata.json").read_text() == '{"dsl":"triton"}'


def test_one_artifact_file_can_be_materialized_to_a_flat_path(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path / "artifacts")
    digest = store.put_json({"objective": "vector add"}, ArtifactKind.AGENT_PROBLEM)

    destination = store.materialize_file(
        digest,
        "value.json",
        tmp_path / "workspace/.runtime/agent-problem.json",
    )

    assert destination.read_text() == '{"objective":"vector add"}'
    assert destination.stat().st_mode & 0o222 == 0


def test_artifact_store_rejects_symlinks(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    target = tmp_path / "outside.txt"
    target.write_text("secret")
    (source / "link").symlink_to(target)
    store = LocalArtifactStore(tmp_path / "artifacts")

    with pytest.raises(ValueError, match="symbolic link"):
        store.put_directory(source, ArtifactKind.KERNEL)


def test_verify_detects_payload_tampering(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "kernel.py").write_text("original")
    store = LocalArtifactStore(tmp_path / "artifacts")
    digest = store.put_directory(source, ArtifactKind.KERNEL)
    artifact = store.verify(digest)
    kernel = artifact.payload_path / "kernel.py"
    os.chmod(kernel, 0o600)
    kernel.write_text("tampered")

    with pytest.raises(ValueError, match="payload mismatch"):
        store.verify(digest)
