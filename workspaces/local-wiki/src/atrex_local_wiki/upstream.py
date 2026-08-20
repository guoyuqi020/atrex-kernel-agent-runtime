"""Lifecycle and feedback adapters for the pinned upstream GPU Wiki."""

from __future__ import annotations

import hashlib
import importlib.util
import os
import shutil
import subprocess
import threading
import uuid
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

from .models import WikiFeedbackReportV1

_SOURCE_MARKER = ".atrex-upstream-tree-digest"
_IGNORED_NAMES = {"__pycache__", ".DS_Store"}


class UpstreamGpuWikiError(RuntimeError):
    """The pinned upstream implementation could not complete an operation."""


@dataclass(frozen=True, slots=True)
class StoreSyncResult:
    """Result of reconciling one writable Store with its pinned source."""

    refreshed: bool
    feedback_preserved: bool


class _FeedbackModule(Protocol):
    def report(
        self, record_id: str, outcome: str, note: str, now: float
    ) -> list[dict[str, object]]: ...

    def append(self, events: list[dict[str, object]]) -> None: ...


def synchronize_store(reference_root: Path, store_root: Path) -> StoreSyncResult:
    """Atomically refresh a writable upstream Store while preserving its event log."""
    source = reference_root.resolve()
    target = store_root.resolve()
    if source == target or source in target.parents or target in source.parents:
        raise ValueError("reference_root and store_root must not overlap")
    digest = _tree_digest(source)
    marker = target / _SOURCE_MARKER
    if marker.is_file() and marker.read_text(encoding="utf-8").strip() == digest:
        return StoreSyncResult(refreshed=False, feedback_preserved=False)

    target.parent.mkdir(parents=True, exist_ok=True)
    stage = target.parent / f".{target.name}.stage-{uuid.uuid4().hex}"
    backup = target.parent / f".{target.name}.backup-{uuid.uuid4().hex}"
    feedback = target / "kernel_wiki" / "feedback" / "events.jsonl"
    preserved = feedback.read_bytes() if feedback.is_file() else None
    try:
        shutil.copytree(source, stage, ignore=_ignore_generated)
        if preserved is not None:
            staged_feedback = stage / "kernel_wiki" / "feedback" / "events.jsonl"
            staged_feedback.parent.mkdir(parents=True, exist_ok=True)
            staged_feedback.write_bytes(preserved)
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
    return StoreSyncResult(refreshed=True, feedback_preserved=preserved is not None)


class UpstreamFeedbackIngestor:
    """Translate Runtime feedback into upstream served events and rebuild ranking."""

    def __init__(self, root: Path, python_executable: Path, lock: threading.RLock) -> None:
        self._root = root.resolve()
        self._python = python_executable.resolve()
        self._lock = lock
        self._ingest_path = self._root / "tools" / "ingest_feedback.py"
        self._rebuild_path = self._root / "tools" / "rebuild_importance.py"
        self._kernel_root = self._root / "kernel_wiki"
        self._module: _FeedbackModule | None = None
        self._validate_layout()

    def check_health(self) -> None:
        """Require the complete upstream feedback and ranking interface."""
        self._validate_layout()

    def ingest(self, report: WikiFeedbackReportV1, received_at: float) -> int:
        """Record every public kernel Record serving and rebuild upstream importance."""
        with self._lock:
            module = self._feedback_module()
            events: list[dict[str, object]] = []
            try:
                for ordinal, (record_id, note) in enumerate(_served_kernel_records(report)):
                    timestamp = received_at + ordinal * 1e-6
                    events.extend(module.report(record_id, "served", note, timestamp))
            except (Exception, SystemExit) as error:
                raise UpstreamGpuWikiError(
                    f"upstream ingest_feedback.py rejected Runtime feedback: {error}"
                ) from error
            if not events:
                return 0
            try:
                module.append(events)
            except (Exception, SystemExit) as error:
                raise UpstreamGpuWikiError(
                    f"upstream ingest_feedback.py could not append events: {error}"
                ) from error
            self._rebuild()
            return len(events)

    def rebuild(self) -> None:
        """Run the pinned ranking rebuild after a source refresh preserved feedback."""
        with self._lock:
            self._rebuild()

    def _rebuild(self) -> None:
        process = subprocess.run(
            [str(self._python), str(self._rebuild_path), "--out", str(self._kernel_root)],
            check=False,
            capture_output=True,
            timeout=300,
        )
        if process.returncode != 0:
            detail = process.stderr.decode("utf-8", errors="replace").strip()[-1000:]
            raise UpstreamGpuWikiError(
                f"upstream rebuild_importance.py failed with exit {process.returncode}: {detail}"
            )

    def _feedback_module(self) -> _FeedbackModule:
        if self._module is not None:
            return self._module
        name = "atrex_local_wiki_upstream_ingest_" + hashlib.sha256(
            str(self._ingest_path).encode()
        ).hexdigest()
        spec = importlib.util.spec_from_file_location(name, self._ingest_path)
        if spec is None or spec.loader is None:
            raise UpstreamGpuWikiError("could not load upstream ingest_feedback.py")
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except Exception as error:
            raise UpstreamGpuWikiError(
                f"could not import upstream feedback implementation: {error}"
            ) from error
        self._module = cast(_FeedbackModule, module)
        return self._module

    def _validate_layout(self) -> None:
        required = (
            self._ingest_path,
            self._rebuild_path,
            self._kernel_root / "records" / "index.json",
        )
        missing = [str(path) for path in required if path.is_symlink() or not path.is_file()]
        if missing:
            raise ValueError(f"GPU Wiki feedback interface is incomplete: {missing}")


def _served_kernel_records(report: WikiFeedbackReportV1) -> Iterable[tuple[str, str]]:
    for attempt in report.attempts:
        for frozen in attempt.interactions:
            content = frozen.interaction.response.content
            if not isinstance(content, Mapping):
                continue
            records = content.get("records")
            if not isinstance(records, Mapping):
                continue
            for record_id, raw_record in records.items():
                if not isinstance(record_id, str) or not isinstance(raw_record, Mapping):
                    continue
                if raw_record.get("source") != "kernel_wiki":
                    continue
                store = raw_record.get("store")
                if store not in {None, "gpu_wiki"}:
                    continue
                note = (
                    f"runtime epoch feedback {report.epoch_id}; attempt {attempt.attempt_id}; "
                    f"interaction {frozen.artifact_digest}"
                )
                yield record_id, note


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
    "UpstreamFeedbackIngestor",
    "UpstreamGpuWikiError",
    "synchronize_store",
]
