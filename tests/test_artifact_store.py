"""Behavior tests for the immutable local Artifact Store."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from atrex_runtime.artifacts import (
    ArtifactKind,
    LocalArtifactStore,
)


def _manifest_path(tmp_path: Path, digest: object) -> Path:
    address = str(digest).removeprefix("sha256:")
    return tmp_path / "artifacts/sha256" / address / "manifest.json"


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


def test_empty_directories_survive_the_seal_and_materialize_round_trip(tmp_path: Path) -> None:
    source = tmp_path / "source"
    (source / "skills").mkdir(parents=True)
    (source / "tools").mkdir()
    (source / "tools" / "README.md").write_text("reusable tools\n")
    store = LocalArtifactStore(tmp_path / "artifacts")

    digest = store.put_directory(source, ArtifactKind.KERNEL_AGENT_RUNTIME_STATE)

    verified = store.verify(digest)
    assert sorted(child.name for child in verified.payload_path.iterdir()) == ["skills", "tools"]
    destination = store.materialize(digest, tmp_path / "materialized")
    assert (destination / "skills").is_dir()
    assert list((destination / "skills").iterdir()) == []


def test_a_childless_directory_is_recorded_without_its_populated_parents(tmp_path: Path) -> None:
    source = tmp_path / "source"
    (source / "outer" / "inner").mkdir(parents=True)
    (source / "outer" / "kept.txt").write_text("kept\n")
    store = LocalArtifactStore(tmp_path / "artifacts")

    digest = store.put_directory(source, ArtifactKind.KERNEL)

    manifest = json.loads(_manifest_path(tmp_path, digest).read_bytes())
    assert manifest["directories"] == ["outer/inner"]
    destination = store.materialize(digest, tmp_path / "materialized")
    assert (destination / "outer/inner").is_dir()
    assert (destination / "outer/kept.txt").read_text() == "kept\n"


def test_a_tree_without_empty_directories_keeps_its_original_manifest(tmp_path: Path) -> None:
    source = tmp_path / "source"
    (source / "tools").mkdir(parents=True)
    (source / "tools" / "README.md").write_text("reusable tools\n")
    store = LocalArtifactStore(tmp_path / "artifacts")

    digest = store.put_directory(source, ArtifactKind.KERNEL_AGENT_RUNTIME_STATE)

    manifest = json.loads(_manifest_path(tmp_path, digest).read_bytes())
    assert sorted(manifest) == ["files", "kind", "version"]


def test_excluded_paths_are_dropped_so_a_live_tree_matches_its_clean_address(
    tmp_path: Path,
) -> None:
    store = LocalArtifactStore(tmp_path / "artifacts")
    clean = tmp_path / "clean"
    clean.mkdir()
    (clean / "kernel.py").write_text("KERNEL\n")
    clean_digest = store.put_directory(clean, ArtifactKind.KERNEL)
    live = tmp_path / "live"
    (live / "__pycache__").mkdir(parents=True)
    (live / "kernel.py").write_text("KERNEL\n")
    (live / "__pycache__/kernel.cpython-312.pyc").write_bytes(b"bytecode")

    sealed = store.put_directory(
        live,
        ArtifactKind.KERNEL,
        exclude=lambda relative, directory: relative.parts[0] == "__pycache__",
    )

    assert sealed == clean_digest
    payload = store.verify(sealed).payload_path
    assert [path.name for path in payload.rglob("*")] == ["kernel.py"]


def test_keeping_one_file_addresses_a_live_tree_like_its_single_file_seal(
    tmp_path: Path,
) -> None:
    store = LocalArtifactStore(tmp_path / "artifacts")
    declared = tmp_path / "declared"
    declared.mkdir()
    (declared / "kernel.py").write_text("KERNEL\n")
    declared_digest = store.put_directory(declared, ArtifactKind.KERNEL)
    live = tmp_path / "live"
    (live / "__pycache__").mkdir(parents=True)
    (live / "empty").mkdir()
    (live / "kernel.py").write_text("KERNEL\n")
    (live / "_devtest.py").write_text("probe\n")
    (live / "__pycache__/kernel.cpython-312.pyc").write_bytes(b"bytecode")

    sealed = store.put_directory(
        live,
        ArtifactKind.KERNEL,
        exclude=lambda relative, directory: directory or relative.as_posix() != "kernel.py",
    )

    assert sealed == declared_digest
    assert [path.name for path in store.verify(sealed).payload_path.rglob("*")] == ["kernel.py"]


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
