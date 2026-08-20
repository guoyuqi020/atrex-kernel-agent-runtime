"""SQLite observation store for local knowledge operations and Epoch feedback."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal


@dataclass(frozen=True, slots=True)
class FeedbackReservation:
    """Durable idempotency decision for one upstream feedback application."""

    status: Literal["pending", "complete", "conflict"]
    received_at: datetime


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
            INSERT OR IGNORE INTO schema_metadata(singleton, version) VALUES (1, 2);
            CREATE TABLE IF NOT EXISTS queries (
                snapshot_id TEXT PRIMARY KEY,
                request_digest TEXT NOT NULL,
                request_json BLOB NOT NULL,
                response_json BLOB NOT NULL,
                received_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS feedback (
                feedback_id TEXT PRIMARY KEY,
                request_digest TEXT NOT NULL,
                report_json BLOB NOT NULL,
                applied INTEGER NOT NULL CHECK (applied IN (0, 1)),
                received_at TEXT NOT NULL
            );
            """
        )
        version = self._connection.execute(
            "SELECT version FROM schema_metadata WHERE singleton = 1"
        ).fetchone()
        if version == (1,):
            self._connection.execute(
                "ALTER TABLE feedback ADD COLUMN applied INTEGER NOT NULL DEFAULT 0 "
                "CHECK (applied IN (0, 1))"
            )
            self._connection.execute(
                "UPDATE schema_metadata SET version = 2 WHERE singleton = 1"
            )
            self._connection.commit()
            version = (2,)
        if version != (2,):
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

    def reserve_feedback(
        self,
        feedback_id: str,
        request_digest: str,
        body: bytes,
    ) -> FeedbackReservation:
        """Persist a report before applying it through the upstream feedback implementation."""
        row = self._connection.execute(
            "SELECT request_digest, applied, received_at FROM feedback WHERE feedback_id = ?",
            (feedback_id,),
        ).fetchone()
        if row is not None:
            received_at = datetime.fromisoformat(str(row[2]))
            if row[0] != request_digest:
                return FeedbackReservation("conflict", received_at)
            return FeedbackReservation("complete" if row[1] else "pending", received_at)
        received_at = datetime.now(UTC)
        self._connection.execute(
            "INSERT INTO feedback VALUES (?, ?, ?, 0, ?)",
            (feedback_id, request_digest, body, received_at.isoformat()),
        )
        self._connection.commit()
        return FeedbackReservation("pending", received_at)

    def mark_feedback_complete(self, feedback_id: str) -> None:
        """Commit successful upstream ingestion for one reserved report."""
        cursor = self._connection.execute(
            "UPDATE feedback SET applied = 1 WHERE feedback_id = ?",
            (feedback_id,),
        )
        if cursor.rowcount != 1:
            raise RuntimeError(f"unknown feedback reservation: {feedback_id}")
        self._connection.commit()

    def pending_feedback(self) -> tuple[tuple[str, bytes, datetime], ...]:
        """Return reports interrupted before upstream ingestion completed."""
        rows = self._connection.execute(
            "SELECT feedback_id, report_json, received_at FROM feedback "
            "WHERE applied = 0 ORDER BY received_at, feedback_id"
        ).fetchall()
        return tuple(
            (str(feedback_id), bytes(body), datetime.fromisoformat(str(received_at)))
            for feedback_id, body, received_at in rows
        )

    def feedback_count(self) -> int:
        """Return the number of unique feedback reports for local assertions."""
        row = self._connection.execute("SELECT COUNT(*) FROM feedback").fetchone()
        assert row is not None
        return int(row[0])

    def check_health(self) -> None:
        """Verify that SQLite remains readable and writable."""
        self._connection.execute("SELECT version FROM schema_metadata").fetchone()

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        self._connection.close()


def _now() -> str:
    return datetime.now(UTC).isoformat()


__all__ = ["FeedbackReservation", "LocalWikiStore"]
