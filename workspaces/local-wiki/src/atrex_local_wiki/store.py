"""SQLite observation store for local knowledge queries."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path


class LocalWikiStore:
    """Persist exact wire payloads without aggregating them into Agent memory."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(path, timeout=30)
        self._connection.execute("PRAGMA journal_mode=DELETE")
        self._connection.execute("PRAGMA synchronous=FULL")
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS schema_metadata (
                singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                version INTEGER NOT NULL
            );
            INSERT OR IGNORE INTO schema_metadata(singleton, version) VALUES (1, 3);
            CREATE TABLE IF NOT EXISTS queries (
                snapshot_id TEXT PRIMARY KEY,
                request_digest TEXT NOT NULL,
                request_json BLOB NOT NULL,
                response_json BLOB NOT NULL,
                received_at TEXT NOT NULL
            );
            """
        )
        version = self._connection.execute(
            "SELECT version FROM schema_metadata WHERE singleton = 1"
        ).fetchone()
        if version in {(1,), (2,)}:
            self._connection.execute("DROP TABLE IF EXISTS feedback")
            self._connection.execute(
                "UPDATE schema_metadata SET version = 3 WHERE singleton = 1"
            )
            self._connection.commit()
            version = (3,)
        if version != (3,):
            raise RuntimeError(f"unsupported local Wiki schema: {version!r}")

    def record_query(
        self,
        snapshot_id: str,
        request_digest: str,
        request_json: bytes,
        response_json: bytes,
    ) -> None:
        """Store one deterministic query observation idempotently."""
        self._connection.execute(
            "INSERT OR IGNORE INTO queries VALUES (?, ?, ?, ?, ?)",
            (snapshot_id, request_digest, request_json, response_json, _now()),
        )
        self._connection.commit()

    def check_health(self) -> None:
        """Verify that SQLite remains readable and writable."""
        self._connection.execute("SELECT version FROM schema_metadata").fetchone()

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        self._connection.close()


def _now() -> str:
    return datetime.now(UTC).isoformat()


__all__ = ["LocalWikiStore"]
