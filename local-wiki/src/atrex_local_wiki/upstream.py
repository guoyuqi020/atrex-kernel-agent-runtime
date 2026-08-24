"""Lifecycle adapter for the pinned upstream GPU Wiki."""

from __future__ import annotations

import hashlib
import os
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path

_SOURCE_MARKER = ".atrex-upstream-tree-digest"
_IGNORED_NAMES = {"__pycache__", ".DS_Store"}


@dataclass(frozen=True, slots=True)
class StoreSyncResult:
    """Result of reconciling one writable Store with its pinned source."""

    refreshed: bool


def synchronize_store(reference_root: Path, store_root: Path) -> StoreSyncResult:
    """Atomically refresh a writable upstream Store."""
    source = reference_root.resolve()
    target = store_root.resolve()
    if source == target or source in target.parents or target in source.parents:
        raise ValueError("reference_root and store_root must not overlap")
    digest = _tree_digest(source)
    marker = target / _SOURCE_MARKER
    if marker.is_file() and marker.read_text(encoding="utf-8").strip() == digest:
        return StoreSyncResult(refreshed=False)

    target.parent.mkdir(parents=True, exist_ok=True)
    stage = target.parent / f".{target.name}.stage-{uuid.uuid4().hex}"
    backup = target.parent / f".{target.name}.backup-{uuid.uuid4().hex}"
    try:
        shutil.copytree(source, stage, ignore=_ignore_generated)
        (stage / _SOURCE_MARKER).write_text(digest + "\n", encoding="utf-8")
        if target.exists():
            os.replace(target, backup)
        os.replace(stage, target)
        if backup.exists():
            shutil.rmtree(backup)
    except Exception:
        if stage.exists():
            shutil.rmtree(stage)
        if backup.exists() and not target.exists():
            os.replace(backup, target)
        raise
    return StoreSyncResult(refreshed=True)


def _tree_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        generated = any(part in _IGNORED_NAMES for part in path.parts)
        if not path.is_file() or path.is_symlink() or generated:
            continue
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return "sha256:" + digest.hexdigest()


def _ignore_generated(_directory: str, names: list[str]) -> set[str]:
    return {name for name in names if name in _IGNORED_NAMES or name.endswith((".pyc", ".pyo"))}


__all__ = [
    "StoreSyncResult",
    "synchronize_store",
]
