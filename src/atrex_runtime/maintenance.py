"""Offline maintenance operations over authoritative Runtime storage."""

from __future__ import annotations

import re
import shutil
import stat
import time
from collections.abc import Callable
from contextlib import suppress
from dataclasses import asdict, dataclass
from pathlib import Path

from .artifacts.local import ArtifactGarbageCollectionResult, LocalArtifactStore
from .gateway.control import SqliteGatewayControl
from .registry.sqlite import SqliteRegistry


class ArtifactGarbageCollector:
    """Collect CAS objects absent from every authoritative durable reference."""

    def __init__(
        self,
        registry: SqliteRegistry,
        gateway_control: SqliteGatewayControl,
        artifacts: LocalArtifactStore,
    ) -> None:
        self._registry = registry
        self._gateway_control = gateway_control
        self._artifacts = artifacts

    def run(
        self,
        *,
        minimum_age_seconds: float,
        limit: int,
        apply: bool,
    ) -> ArtifactGarbageCollectionResult:
        """Take one reference snapshot and execute a bounded CAS maintenance pass."""
        references = self._registry.list_referenced_artifact_digests()
        references.update(self._gateway_control.list_referenced_artifact_digests())
        references = self._artifacts.expand_reference_closure(references)
        result = self._artifacts.collect_garbage(
            references,
            minimum_age_seconds=minimum_age_seconds,
            limit=limit,
            apply=apply,
        )
        if apply:
            self._registry.record_runtime_event(
                "artifact.gc_completed",
                "artifact_store",
                asdict(result),
            )
        return result


@dataclass(frozen=True, slots=True)
class WorkspaceGarbageCollectionResult:
    """Bounded result from scanning append-only Worker run directories."""

    roots: int
    scanned: int
    eligible: int
    deleted: int
    reclaimed_bytes: int


class WorkspaceGarbageCollector:
    """Remove old append-only Worker runs during an explicit offline maintenance pass."""

    _RUN_NAME = re.compile(r"^run-[0-9a-f]{32}$")

    def __init__(self, roots: tuple[Path, ...], registry: SqliteRegistry) -> None:
        resolved = tuple(root.resolve() for root in roots)
        if not resolved or len(set(resolved)) != len(resolved):
            raise ValueError("Workspace GC roots must be non-empty and distinct")
        if any(root == Path(root.anchor) for root in resolved):
            raise ValueError("Workspace GC refuses a filesystem root")
        self._roots = resolved
        self._registry = registry

    def run(
        self,
        *,
        minimum_age_seconds: float,
        limit: int,
        apply: bool,
        clock: Callable[[], float] = time.time,
    ) -> WorkspaceGarbageCollectionResult:
        """Inspect or delete old exact-shape ``<subject>/run-<uuid>`` directories."""
        if minimum_age_seconds < 0 or limit <= 0:
            raise ValueError("Workspace GC age must be nonnegative and limit must be positive")
        cutoff = float(clock()) - minimum_age_seconds
        candidates: list[Path] = []
        scanned = 0
        for root in self._roots:
            if not root.exists():
                continue
            if root.is_symlink() or not root.is_dir():
                raise ValueError(f"Workspace GC root must be a real directory: {root}")
            for subject in sorted(root.iterdir()):
                if subject.is_symlink() or not subject.is_dir():
                    raise ValueError(f"Unexpected Workspace GC root entry: {subject}")
                for run in sorted(subject.iterdir()):
                    invalid = (
                        run.is_symlink()
                        or not run.is_dir()
                        or not self._RUN_NAME.fullmatch(run.name)
                    )
                    if invalid:
                        raise ValueError(f"Unexpected Worker workspace entry: {run}")
                    scanned += 1
                    if run.stat().st_mtime <= cutoff:
                        candidates.append(run)
        selected = sorted(candidates, key=lambda path: (path.stat().st_mtime, str(path)))[:limit]
        reclaimed_bytes = 0
        deleted = 0
        if apply:
            for run in selected:
                reclaimed_bytes += self._directory_bytes(run)
                parent = run.parent
                shutil.rmtree(run)
                deleted += 1
                with suppress(OSError):
                    parent.rmdir()
            self._registry.record_runtime_event(
                "workspace.gc_completed",
                "worker_workspaces",
                {
                    "roots": len(self._roots),
                    "scanned": scanned,
                    "eligible": len(selected),
                    "deleted": deleted,
                    "reclaimed_bytes": reclaimed_bytes,
                },
            )
        return WorkspaceGarbageCollectionResult(
            len(self._roots), scanned, len(selected), deleted, reclaimed_bytes
        )

    @staticmethod
    def _directory_bytes(root: Path) -> int:
        return sum(
            entry_stat.st_size
            for path in root.rglob("*")
            if stat.S_ISREG((entry_stat := path.lstat()).st_mode)
        )
