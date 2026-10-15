from __future__ import annotations

import sqlite3
import threading

import pytest

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


@pytest.mark.parametrize('registry_transaction', [False, True])
def test_busy_commit_rolls_back_and_releases_writer_lock(tmp_path, registry_transaction):
    from atrex_runtime.registry.sqlite import SqliteRegistry
    from atrex_runtime.sqlite_support import immediate_transaction

    path = tmp_path / 'busy.sqlite'
    with SqliteRegistry(path) as registry:
        writer = registry._connection
        writer.execute('CREATE TABLE example(value INTEGER)')
        writer.execute('INSERT INTO example VALUES (1)')
        writer.execute('PRAGMA busy_timeout=5')
        reader = sqlite3.connect(path, isolation_level=None)
        try:
            reader.execute('BEGIN')
            assert reader.execute('SELECT value FROM example').fetchone() == (1,)
            transaction = (registry._transaction if registry_transaction else
                           lambda: immediate_transaction(writer, threading.RLock()))
            with pytest.raises(sqlite3.OperationalError, match='locked'), transaction():
                writer.execute('UPDATE example SET value=2')
            assert not writer.in_transaction
            reader.rollback()
            # A separate process/connection can write immediately after the failure.
            reader.execute('PRAGMA busy_timeout=5')
            reader.execute('BEGIN IMMEDIATE')
            assert reader.execute('SELECT value FROM example').fetchone() == (1,)
            reader.commit()
            with transaction():
                writer.execute('UPDATE example SET value=3')
            assert reader.execute('SELECT value FROM example').fetchone() == (3,)
        finally:
            reader.close()
