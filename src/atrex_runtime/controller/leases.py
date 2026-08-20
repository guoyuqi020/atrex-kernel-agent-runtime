"""Process-lifetime exclusion for local SQLite lineage scheduling."""

from __future__ import annotations

import os
import threading
from contextvars import Token
from datetime import UTC, datetime, timedelta
from types import TracebackType
from typing import Protocol

from ..domain.errors import InvalidTransitionError, LineageLeaseUnavailableError
from ..domain.ids import LineageId, parse_lineage_id
from ..registry.sqlite import SqliteRegistry


class LineageLease(Protocol):
    """A held lineage lease released on context exit."""

    def __enter__(self) -> None:
        """Keep the already acquired lease held."""
        ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Release the lease after scheduler work reaches quiescence."""
        ...


class LineageLeaseManager(Protocol):
    """Acquire exclusive ownership for one lineage schedule."""

    def acquire(self, lineage_id: LineageId) -> LineageLease:
        """Acquire immediately or fail when another scheduler owns the lineage."""
        ...


class _RegistryLineageLease:
    def __init__(
        self,
        registry: SqliteRegistry,
        lineage_id: LineageId,
        owner: str,
        generation: int,
        lease_seconds: float,
        heartbeat_seconds: float,
    ) -> None:
        self._registry = registry
        self._lineage_id = lineage_id
        self._owner = owner
        self._generation = generation
        self._lease_seconds = lease_seconds
        self._heartbeat_seconds = heartbeat_seconds
        self._stop = threading.Event()
        self._failure: BaseException | None = None
        self._thread = threading.Thread(target=self._heartbeat, daemon=True)
        self._token: Token[tuple[LineageId, int, str] | None] | None = None
        self._closed = False

    def __enter__(self) -> None:
        if self._closed or self._token is not None:
            raise RuntimeError("lineage fence cannot be entered twice")
        self._token = self._registry.activate_lineage_fence(
            self._lineage_id,
            self._generation,
            self._owner,
        )
        self._thread.start()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc_value, traceback
        if self._closed:
            return
        self._closed = True
        self._stop.set()
        self._thread.join()
        try:
            self._registry.release_lineage_fence(
                self._lineage_id,
                self._generation,
                self._owner,
            )
        finally:
            if self._token is not None:
                self._registry.deactivate_lineage_fence(self._token)
                self._token = None
        if self._failure is not None:
            raise RuntimeError("lineage fencing heartbeat failed") from self._failure

    def _heartbeat(self) -> None:
        while not self._stop.wait(self._heartbeat_seconds):
            try:
                self._registry.renew_lineage_fence(
                    self._lineage_id,
                    self._generation,
                    self._owner,
                    lease_expires_at=(
                        datetime.now(UTC) + timedelta(seconds=self._lease_seconds)
                    ).isoformat(),
                )
            except BaseException as error:
                self._failure = error
                self._stop.set()
                return


class RegistryLineageLeaseManager:
    """Renewable Registry lease whose generation fences every scheduler write."""

    def __init__(
        self,
        registry: SqliteRegistry,
        *,
        lease_seconds: float,
        heartbeat_seconds: float,
    ) -> None:
        if heartbeat_seconds <= 0 or lease_seconds <= heartbeat_seconds * 2:
            raise ValueError("fencing lease must exceed two positive heartbeat periods")
        self._registry = registry
        self._lease_seconds = lease_seconds
        self._heartbeat_seconds = heartbeat_seconds

    def acquire(self, lineage_id: LineageId) -> LineageLease:
        """Acquire a new generation immediately or fail while another remains live."""
        validated = parse_lineage_id(str(lineage_id))
        owner = f"pid-{os.getpid()}-{os.urandom(16).hex()}"
        now = datetime.now(UTC)
        try:
            generation = self._registry.acquire_lineage_fence(
                validated,
                owner,
                now=now.isoformat(),
                lease_expires_at=(now + timedelta(seconds=self._lease_seconds)).isoformat(),
            )
        except InvalidTransitionError as error:
            raise LineageLeaseUnavailableError(str(error)) from error
        return _RegistryLineageLease(
            self._registry,
            validated,
            owner,
            generation,
            self._lease_seconds,
            self._heartbeat_seconds,
        )
