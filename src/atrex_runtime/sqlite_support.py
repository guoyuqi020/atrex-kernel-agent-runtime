"""SQLite connection policy for durable Runtime state."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager

SQLITE_BUSY_TIMEOUT_MS = 30_000


@contextmanager
def immediate_transaction(
    connection: sqlite3.Connection,
    lock: AbstractContextManager[object],
) -> Iterator[sqlite3.Connection]:
    """Serialize one SQLite immediate transaction with rollback-on-error semantics."""
    with lock:
        connection.execute("BEGIN IMMEDIATE")
        try:
            yield connection
        except BaseException:
            connection.rollback()
            raise
        else:
            connection.commit()


def configure_durable_sqlite(
    connection: sqlite3.Connection,
    *,
    in_memory: bool = False,
) -> None:
    """Use a journal mode that is safe on VM shared filesystems.

    Runtime state is commonly stored below a Lima/virtiofs-mounted workspace and
    is opened by both the API process and Campaign CLI processes. SQLite WAL uses
    a shared-memory sidecar and explicitly requires local-filesystem locking
    semantics, which virtiofs does not reliably provide. Rollback journaling is
    slower but preserves SQLite's normal cross-process locking on these mounts.
    """
    connection.execute(f"PRAGMA busy_timeout = {SQLITE_BUSY_TIMEOUT_MS}")
    if not in_memory:
        row = connection.execute("PRAGMA journal_mode = DELETE").fetchone()
        if row is None or str(row[0]).lower() != "delete":
            mode = None if row is None else row[0]
            raise RuntimeError(f"SQLite refused rollback journal mode: {mode!r}")
        connection.execute("PRAGMA synchronous = FULL")
