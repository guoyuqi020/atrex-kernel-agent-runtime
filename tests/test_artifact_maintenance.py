"""Offline Artifact retention and reference-safety tests."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from atrex_runtime.artifacts import ArtifactKind, LocalArtifactStore
from atrex_runtime.gateway.control import SqliteGatewayControl
from atrex_runtime.maintenance import ArtifactGarbageCollector, WorkspaceGarbageCollector
from atrex_runtime.registry.sqlite import SqliteRegistry


def test_gc_dry_run_and_apply_preserve_all_durable_references(tmp_path: Path) -> None:
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    nested = artifacts.put_json({"nested": True}, ArtifactKind.SESSION_LOG)
    retained = artifacts.put_json(
        {"retained": True, "nested_digest": str(nested)},
        ArtifactKind.EVIDENCE,
    )
    orphan = artifacts.put_json({"orphan": True}, ArtifactKind.EVIDENCE)
    registry = SqliteRegistry(tmp_path / "registry.sqlite")
    registry.record_runtime_event(
        "test.artifact_retained",
        "test",
        {"artifact_digest": retained},
    )
    control = SqliteGatewayControl(
        tmp_path / "gateway.sqlite",
        registry,
        signing_key=b"g" * 32,
    )
    collector = ArtifactGarbageCollector(registry, control, artifacts)

    dry_run = collector.run(minimum_age_seconds=0, limit=10, apply=False)
    assert dry_run.eligible == 1
    assert dry_run.deleted == 0
    assert artifacts.verify(orphan).digest == orphan

    applied = collector.run(minimum_age_seconds=0, limit=10, apply=True)
    assert applied.eligible == 1
    assert applied.deleted == 1
    assert applied.reclaimed_bytes > 0
    assert artifacts.verify(retained).digest == retained
    assert artifacts.verify(nested).digest == nested
    with pytest.raises(FileNotFoundError):
        artifacts.verify(orphan)
    assert any(
        event.kind == "artifact.gc_completed"
        for event in registry.list_runtime_events(after_sequence=0, limit=20)
    )
    control.close()
    registry.close()


def test_gc_obeys_age_and_batch_bounds(tmp_path: Path) -> None:
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    first = artifacts.put_json({"value": 1}, ArtifactKind.EVIDENCE)
    second = artifacts.put_json({"value": 2}, ArtifactKind.EVIDENCE)

    too_young = artifacts.collect_garbage(
        set(),
        minimum_age_seconds=60,
        limit=10,
        apply=True,
        clock=lambda: artifacts.verify(first).payload_path.parent.stat().st_mtime + 30,
    )
    assert too_young.deleted == 0

    bounded = artifacts.collect_garbage(
        set(),
        minimum_age_seconds=0,
        limit=1,
        apply=True,
    )
    assert bounded.eligible == 1
    assert bounded.deleted == 1
    remaining = 0
    for digest in (first, second):
        try:
            artifacts.verify(digest)
        except FileNotFoundError:
            continue
        remaining += 1
    assert remaining == 1


def test_workspace_gc_is_offline_bounded_and_removes_empty_subjects(tmp_path: Path) -> None:
    names = ("attempts", "evolution", "generalization", "bootstrap")
    roots = tuple(tmp_path / name for name in names)
    for root in roots:
        root.mkdir()
    first = roots[0] / "attempt-1/run-00000000000000000000000000000001"
    second = roots[1] / "revision-1/run-00000000000000000000000000000002"
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    (first / "payload").write_bytes(b"first")
    (second / "payload").write_bytes(b"second")
    os.utime(first, (100, 100))
    os.utime(second, (200, 200))
    registry = SqliteRegistry(tmp_path / "registry.sqlite")
    collector = WorkspaceGarbageCollector(roots, registry)

    dry_run = collector.run(minimum_age_seconds=0, limit=1, apply=False, clock=lambda: 300)
    assert dry_run.scanned == 2
    assert dry_run.eligible == 1
    assert dry_run.deleted == 0
    assert first.exists()

    applied = collector.run(minimum_age_seconds=0, limit=1, apply=True, clock=lambda: 300)
    assert applied.deleted == 1
    assert applied.reclaimed_bytes == len(b"first")
    assert not first.exists()
    assert not first.parent.exists()
    assert second.exists()
    assert any(
        event.kind == "workspace.gc_completed"
        for event in registry.list_runtime_events(after_sequence=0, limit=20)
    )
    registry.close()
