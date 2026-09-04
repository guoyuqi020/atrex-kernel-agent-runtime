"""Schema 33: resumable Epoch stops and interrupted execution records."""

from __future__ import annotations

import re
import sqlite3


def migrate_stops(connection: sqlite3.Connection) -> None:
    """Rebuild CHECK constraints without losing rows, indexes or foreign keys."""
    connection.execute("PRAGMA foreign_keys = OFF")
    connection.execute("BEGIN IMMEDIATE")
    try:
        for table, state in (
            ("epochs", "stopped"), ("attempts", "interrupted"),
            ("worker_sessions", "interrupted"),
        ):
            row = connection.execute(
                "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?", (table,)
            ).fetchone()
            if row is None:
                continue
            sql: str = row[0]
            if f"'{state}'" in sql:
                continue
            pattern = r"(CHECK\s*\(\s*status\s+IN\s*\()([^)]*)(\)\s*\))"
            updated, count = re.subn(
                pattern, r"\g<1>\g<2>" + f", '{state}'" + r"\g<3>", sql,
                flags=re.IGNORECASE,
            )
            if count == 0:  # Partial historical schemas used by migration probes.
                continue
            objects = connection.execute(
                "SELECT sql FROM sqlite_master WHERE tbl_name = ? "
                "AND type IN ('index', 'trigger') AND sql IS NOT NULL", (table,),
            ).fetchall()
            updated = re.sub(
                r'CREATE TABLE\s+(?:"\w+"|\w+)', f'CREATE TABLE "{table}_next"',
                updated, count=1, flags=re.IGNORECASE,
            )
            connection.execute(updated)
            connection.execute(f'INSERT INTO "{table}_next" SELECT * FROM "{table}"')
            connection.execute(f'DROP TABLE "{table}"')
            connection.execute(f'ALTER TABLE "{table}_next" RENAME TO "{table}"')
            for obj in objects:
                connection.execute(obj[0])
        connection.execute("""CREATE TABLE IF NOT EXISTS epoch_stops (
            epoch_id TEXT PRIMARY KEY REFERENCES epochs(id),
            previous_status TEXT NOT NULL,
            reason TEXT NOT NULL,
            stopped_at TEXT NOT NULL
        )""")
        if connection.execute("PRAGMA foreign_key_check").fetchall():
            raise RuntimeError("Registry stop-state migration violated foreign keys")
        connection.execute("PRAGMA user_version = 33")
        connection.execute("COMMIT")
    except BaseException:
        connection.execute("ROLLBACK")
        raise
    finally:
        connection.execute("PRAGMA foreign_keys = ON")
