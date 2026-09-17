"""Lineage-local native Evolver conversations, independent of Candidate revisions."""

from __future__ import annotations

import fcntl
import json
import os
import re
import shutil
import stat
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from ..domain.errors import InfrastructureError

CONTINUATION_METADATA = ".evolver-session.json"
_NATIVE_STATE = {
    "claude": (".claude/projects",),
    "codex": (".codex/sessions", ".codex/archived_sessions", ".codex/state_*.sqlite*"),
    "qodercli": (".qoder/projects", ".qoder/tasks"),
    "pi": (".atrex-pi",),
}


def _has_native(paths: list[Path]) -> bool:
    return any(
        path.is_file() or any(child.is_file() for child in path.rglob("*")) for path in paths
    )


def _no_links(path: Path, root: Path) -> None:
    for entry in (path, *path.parents):
        if entry.is_symlink():
            raise InfrastructureError("Evolver continuation contains unsafe native state")
        if entry == root:
            break


def _copy_native(source: Path, destination: Path) -> None:
    """Copy only regular native state; never follow an Agent-created link."""
    if destination.is_symlink() or (destination.is_file() and destination.stat().st_nlink > 1):
        raise InfrastructureError("Evolver continuation destination aliases another file")
    mode = source.lstat().st_mode
    if stat.S_ISDIR(mode):
        destination.mkdir(parents=True, exist_ok=True, mode=0o700)
        for child in source.iterdir():
            _copy_native(child, destination / child.name)
    elif stat.S_ISREG(mode) and source.stat().st_nlink == 1:
        destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        shutil.copyfile(source, destination)
        destination.chmod(0o600)
    else:
        raise InfrastructureError("Evolver continuation contains unsafe native state")


class EvolutionConversation:
    """Keep one conversation per Lineage and Backend, including infrastructure retries."""

    def __init__(self, workspace_root: Path, key: str, backend: str) -> None:
        self.workspace_root = workspace_root.resolve()
        self.backend = backend
        self.root = self.workspace_root / ".control/evolver-sessions" / key / backend
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.pointer = self.root / "latest.json"

    @contextmanager
    def owned(self) -> Iterator[None]:
        with (self.root / "session.lock").open("a+b") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def restore(self, workspace: Path, home: Path) -> str | None:
        """Restore transcripts/indexes, not credentials, Candidate files, or old scratch."""
        _no_links(home, self.workspace_root)
        if not self.pointer.exists():
            return None
        _no_links(self.pointer, self.workspace_root)
        value = json.loads(self.pointer.read_text(encoding="utf-8"))
        relative = Path(value["workspace"])
        if relative.is_absolute() or ".." in relative.parts:
            raise InfrastructureError("Invalid Evolver continuation workspace")
        previous = self.workspace_root / relative
        if not previous.is_dir():
            raise InfrastructureError(
                "Evolver continuation workspace is missing; retain previous Evolution workspaces "
                "to resume their conversations, or create a new Lineage"
            )
        if previous.is_symlink() or not previous.resolve().is_relative_to(self.workspace_root):
            raise InfrastructureError("Evolver continuation escapes its workspace root")
        source_home = previous / "scratch/agent-home"
        _no_links(source_home, self.workspace_root)
        metadata = source_home / CONTINUATION_METADATA
        native = [
            path for pattern in _NATIVE_STATE[self.backend] for path in source_home.glob(pattern)
        ]
        for path in native:
            _no_links(path, source_home)
        if not metadata.is_file():
            if _has_native(native):
                raise InfrastructureError("Evolver native history exists without session metadata")
            return None  # The preceding process never reached the Provider.
        if metadata.is_symlink():
            raise InfrastructureError("Unsafe Evolver continuation metadata")
        record = json.loads(metadata.read_text(encoding="utf-8"))
        if record.get("backend") != self.backend:
            raise InfrastructureError("Evolver continuation Backend mismatch")
        session_id = record.get("session_id")
        # Codex chooses its own ID; a hard kill can precede the wrapper's final update.
        if self.backend == "codex" and not session_id:
            for path in (source_home / ".codex/sessions").rglob("rollout-*.jsonl"):
                _no_links(path, source_home)
                with path.open(encoding="utf-8") as source:
                    first = json.loads(source.readline())
                if first.get("type") == "session_meta":
                    session_id = first.get("payload", {}).get("id")
                    break
        if self.backend == "pi" and not session_id:
            path = source_home / ".atrex-pi/evolver.jsonl"
            _no_links(path, source_home)
            if path.is_file():
                with path.open(encoding="utf-8") as source:
                    first = json.loads(source.readline())
                if first.get("type") == "session":
                    session_id = first.get("id")
        if not isinstance(session_id, str) or not session_id:
            return None
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", session_id):
            raise InfrastructureError("Evolver continuation has an invalid native session ID")
        if not _has_native(native):
            return None  # CLI setup failed before writing any conversation.
        if source_home.resolve() != home.resolve():
            for source in native:
                _copy_native(source, home / source.relative_to(source_home))
        return session_id

    def remember(self, workspace: Path) -> None:
        """Publish before launch so a killed driver can resume the live native files."""
        payload = json.dumps({"workspace": workspace.relative_to(self.workspace_root).as_posix()})
        with tempfile.NamedTemporaryFile(dir=self.root, delete=False) as output:
            temporary = Path(output.name)
            output.write(payload.encode())
            output.flush()
            os.fsync(output.fileno())
        temporary.replace(self.pointer)
        descriptor = os.open(self.root, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
