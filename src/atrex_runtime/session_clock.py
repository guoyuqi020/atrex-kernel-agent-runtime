"""Runtime-owned active-time accounting shared with read-only Agent workspaces."""

from __future__ import annotations

import fcntl
import json
import os
import tempfile
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from uuid import uuid4

CLOCK_ENVIRONMENT_KEY = "ATREX_SESSION_CLOCK_PATH"
CLOCK_RELATIVE_PATH = Path(".runtime/session-clock.json")


class SessionClock:
    """Subtract the union of authenticated Runtime request intervals from wall time.

    Only host-side Runtime processes write this file; .runtime is mounted read-only
    in Agent sandboxes. Each in-flight request also holds an OS lease, so a dead
    service cannot leave the budget permanently paused. New launches use a new ID
    to fence late completions from a previous launch in the same workspace.
    """

    def __init__(self, path: Path, *, clock: Callable[[], float] = time.monotonic) -> None:
        self.path = path
        self._clock = clock
        self._locks = path.parent / ".session-clock-locks"

    @contextmanager
    def _lock(self, name: str, *, blocking: bool = True) -> Iterator[bool]:
        self._locks.mkdir(parents=True, exist_ok=True, mode=0o700)
        descriptor = os.open(self._locks / name, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        try:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
            except BlockingIOError:
                yield False
            else:
                try:
                    yield True
                finally:
                    fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)

    def _read(self) -> dict[str, Any]:
        if self.path.is_symlink() or self.path.parent.is_symlink():
            raise ValueError("Session clock must be a Runtime-owned regular file")
        value = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(value, dict) or value.get("schema_version") != 1:
            raise ValueError("Invalid session clock")
        return value

    def _write(self, state: dict[str, Any]) -> None:
        with tempfile.NamedTemporaryFile(mode="w", dir=self.path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(state, stream)
        try:
            temporary.chmod(0o400)
            os.replace(temporary, self.path)
        finally:
            temporary.unlink(missing_ok=True)

    def start(self) -> None:
        if self.path.is_symlink() or self.path.parent.is_symlink() or self._locks.is_symlink():
            raise ValueError("Session clock must be a Runtime-owned regular file")
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with self._lock("state"):
            self._write({
                "schema_version": 1,
                "session_id": uuid4().hex,
                "wait_seconds": 0.0,
                "paused_since": None,
                "calls": [],
            })

    def _reap(self, state: dict[str, Any], now: float) -> bool:
        live = []
        for call in state["calls"]:
            with self._lock(call, blocking=False) as abandoned:
                if not abandoned:
                    live.append(call)
        if live == state["calls"]:
            return False
        state["calls"] = live
        if not live and state["paused_since"] is not None:
            state["wait_seconds"] += max(0.0, now - state["paused_since"])
            state["paused_since"] = None
        return True

    def now(self) -> float:
        """An advancing clock only while this Session is outside Runtime requests."""
        with self._lock("state"):
            state = self._read()
            now = self._clock()
            if self._reap(state, now):
                self._write(state)
            effective_now = now if state["paused_since"] is None else state["paused_since"]
            return float(effective_now - state["wait_seconds"])

    @contextmanager
    def waiting(self) -> Iterator[None]:
        call = uuid4().hex
        with self._lock(call):
            with self._lock("state"):
                state = self._read()
                self._reap(state, self._clock())
                session_id = state["session_id"]
                if not state["calls"]:
                    state["paused_since"] = self._clock()
                state["calls"].append(call)
                self._write(state)
            try:
                yield
            finally:
                with self._lock("state"):
                    state = self._read()
                    if state["session_id"] == session_id and call in state["calls"]:
                        state["calls"].remove(call)
                        if not state["calls"]:
                            state["wait_seconds"] += max(0.0, self._clock() - state["paused_since"])
                            state["paused_since"] = None
                        self._write(state)
