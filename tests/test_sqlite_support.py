from __future__ import annotations

import sqlite3

from atrex_runtime.sqlite_support import SQLITE_BUSY_TIMEOUT_MS, configure_durable_sqlite


def test_durable_sqlite_uses_shared_filesystem_safe_journal(tmp_path) -> None:
    database = tmp_path / "runtime.sqlite"
    connection = sqlite3.connect(database, isolation_level=None)
    try:
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("CREATE TABLE values_table(value INTEGER NOT NULL)")
        connection.execute("INSERT INTO values_table VALUES (1)")

        configure_durable_sqlite(connection)

        assert connection.execute("PRAGMA journal_mode").fetchone() == ("delete",)
        assert connection.execute("PRAGMA synchronous").fetchone() == (2,)
        assert connection.execute("PRAGMA busy_timeout").fetchone() == (SQLITE_BUSY_TIMEOUT_MS,)
        assert connection.execute("SELECT value FROM values_table").fetchone() == (1,)
    finally:
        connection.close()


def test_durable_sqlite_leaves_memory_journal_unchanged() -> None:
    connection = sqlite3.connect(":memory:", isolation_level=None)
    try:
        configure_durable_sqlite(connection, in_memory=True)
        assert connection.execute("PRAGMA journal_mode").fetchone() == ("memory",)
        assert connection.execute("PRAGMA busy_timeout").fetchone() == (SQLITE_BUSY_TIMEOUT_MS,)
    finally:
        connection.close()
