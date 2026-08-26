"""SQLite provider for authoritative Runtime state."""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from ..domain.errors import InvalidTransitionError
from ..domain.ids import (
    ArtifactDigest,
    AttemptId,
    CampaignId,
    CampaignTaskId,
    EpochId,
    KernelAgentRevisionId,
    KernelRevisionId,
    LineageId,
    WorkerSessionId,
    parse_artifact_digest,
    parse_attempt_id,
    parse_campaign_id,
    parse_campaign_task_id,
    parse_epoch_id,
    parse_kernel_agent_revision_id,
    parse_kernel_revision_id,
    parse_lineage_id,
    parse_worker_session_id,
)
from ..domain.models import (
    Attempt,
    AttemptReportStatus,
    AttemptSessionTrace,
    AttemptStatus,
    BranchRole,
    Campaign,
    CampaignStatus,
    CampaignTask,
    CampaignTaskStatus,
    ChallengerProposalType,
    Dsl,
    Epoch,
    EpochChallenger,
    EpochRecovery,
    EpochSelection,
    EpochStatus,
    KernelAgentCatalogEntry,
    KernelAgentRevision,
    KernelCatalogEntry,
    KernelEvaluation,
    KernelMeasurement,
    KernelMeasurementPurpose,
    KernelRevision,
    Lineage,
    LineageStatus,
    RuntimeEvent,
    RuntimeMetrics,
    TokenUsage,
    WorkerSession,
    WorkerSessionRole,
    WorkerSessionStatus,
)
from ..sqlite_support import configure_durable_sqlite

SCHEMA_VERSION = 29
_ACTIVE_FENCE: ContextVar[tuple[LineageId, int, str] | None] = ContextVar(
    "atrex_active_lineage_fence",
    default=None,
)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _required_text(row: sqlite3.Row, column: str) -> str:
    value = row[column]
    if not isinstance(value, str):
        raise TypeError(f"persisted {column} must be text")
    return value


def _optional_text(row: sqlite3.Row, column: str) -> str | None:
    value = row[column]
    if value is not None and not isinstance(value, str):
        raise TypeError(f"persisted {column} must be text or null")
    return value


def _required_int(row: sqlite3.Row, column: str) -> int:
    value = cast(object, row[column])
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"persisted {column} must be an integer")
    return value


def _optional_int(row: sqlite3.Row, column: str) -> int | None:
    value = cast(object, row[column])
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"persisted {column} must be an integer or null")
    return value


def _optional_float(row: sqlite3.Row, column: str) -> float | None:
    value = row[column]
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"persisted {column} must be numeric or null")
    return float(value)


def _append_unique_text(values: list[str], value: object) -> None:
    if isinstance(value, str) and value not in values:
        values.append(value)


def _append_agent_version(
    values: list[str],
    introduced_epochs: dict[str, str | None],
    value: object,
    epoch_id: str | None = None,
) -> None:
    if not isinstance(value, str):
        return
    if value not in values:
        values.append(value)
        introduced_epochs[value] = epoch_id
    elif epoch_id is not None and introduced_epochs[value] is None:
        introduced_epochs[value] = epoch_id


class SqliteRegistry:
    """SQLite implementation with explicit transitions and atomic promotion."""

    def __init__(
        self,
        path: str | Path,
        *,
        clock: Callable[[], str] = _utc_now,
        require_fencing: bool = False,
    ) -> None:
        self._path = str(path)
        if self._path != ":memory:":
            database_path = Path(self._path)
            database_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._connection = sqlite3.connect(
            self._path,
            isolation_level=None,
            check_same_thread=False,
        )
        try:
            self._connection.row_factory = sqlite3.Row
            self._lock = threading.RLock()
            self._clock = clock
            self._require_fencing = require_fencing
            configure_durable_sqlite(
                self._connection,
                in_memory=self._path == ":memory:",
            )
            self._connection.execute("PRAGMA foreign_keys = ON")
            self._migrate()
            if self._path != ":memory:":
                os.chmod(self._path, 0o600)
        except BaseException:
            self._connection.close()
            raise

    def close(self) -> None:
        """Close the database after all controller work has reached quiescence."""
        with self._lock:
            self._connection.close()

    def check_health(self) -> None:
        """Verify that the Registry connection can read and acquire a write transaction."""
        with self._transaction():
            self._connection.execute("SELECT 1").fetchone()

    def start_worker_session(self, session: WorkerSession) -> WorkerSession:
        """Create the durable running record before launching a Worker process."""
        if session.status is not WorkerSessionStatus.RUNNING:
            raise ValueError("a new Worker session must be running")
        with self._transaction():
            campaign_id = session.campaign_id
            lineage_id = session.lineage_id
            epoch_id = session.epoch_id
            if session.attempt_id is not None:
                row = self._connection.execute(
                    """SELECT a.epoch_id, e.lineage_id, l.campaign_id
                       FROM attempts a
                       JOIN epochs e ON e.id = a.epoch_id
                       JOIN lineages l ON l.id = e.lineage_id
                       WHERE a.id = ?""",
                    (session.attempt_id,),
                ).fetchone()
                if row is not None:
                    epoch_id = parse_epoch_id(_required_text(row, "epoch_id"))
                    lineage_id = parse_lineage_id(_required_text(row, "lineage_id"))
                    campaign_id = parse_campaign_id(_required_text(row, "campaign_id"))
            elif epoch_id is not None:
                row = self._connection.execute(
                    """SELECT e.lineage_id, l.campaign_id
                       FROM epochs e JOIN lineages l ON l.id = e.lineage_id
                       WHERE e.id = ?""",
                    (epoch_id,),
                ).fetchone()
                if row is not None:
                    lineage_id = parse_lineage_id(_required_text(row, "lineage_id"))
                    campaign_id = parse_campaign_id(_required_text(row, "campaign_id"))
            elif lineage_id is not None and campaign_id is None:
                row = self._connection.execute(
                    "SELECT campaign_id FROM lineages WHERE id = ?", (lineage_id,)
                ).fetchone()
                if row is not None:
                    campaign_id = parse_campaign_id(_required_text(row, "campaign_id"))
            enriched = replace(
                session,
                campaign_id=campaign_id,
                lineage_id=lineage_id,
                epoch_id=epoch_id,
            )
            usage = enriched.token_usage
            self._connection.execute(
                """INSERT INTO worker_sessions VALUES (
                       ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                   )""",
                (
                    enriched.id,
                    enriched.role,
                    enriched.subject_id,
                    enriched.external_run_id,
                    enriched.campaign_id,
                    enriched.lineage_id,
                    enriched.epoch_id,
                    enriched.attempt_id,
                    enriched.recovery_generation,
                    enriched.backend,
                    enriched.model,
                    enriched.workspace_path,
                    enriched.status,
                    enriched.finish_reason,
                    enriched.trace_digest,
                    enriched.token_budget,
                    None if usage is None else usage.uncached_input_tokens,
                    None if usage is None else usage.output_tokens,
                    None if usage is None else usage.cache_read_tokens,
                    None if usage is None else usage.cache_write_tokens,
                    None if usage is None else usage.credits,
                    enriched.process_returncode,
                    enriched.error_type,
                    enriched.error_message,
                    enriched.started_at,
                    enriched.completed_at,
                ),
            )
        return enriched

    def finish_worker_session(
        self,
        session_id: WorkerSessionId,
        *,
        status: WorkerSessionStatus,
        finish_reason: str,
        trace_digest: ArtifactDigest | None = None,
        token_budget: int | None = None,
        token_usage: TokenUsage | None = None,
        process_returncode: int | None = None,
        error_type: str | None = None,
        error_message: str | None = None,
    ) -> WorkerSession:
        """Atomically move a running Worker session to one terminal state."""
        if status is WorkerSessionStatus.RUNNING:
            raise ValueError("finish_worker_session requires a terminal status")
        with self._transaction():
            row = self._connection.execute(
                "SELECT * FROM worker_sessions WHERE id = ?", (session_id,)
            ).fetchone()
            if row is None:
                raise KeyError(f"Worker session not found: {session_id}")
            current = self._map_worker_session(row)
            if current.status is not WorkerSessionStatus.RUNNING:
                raise InvalidTransitionError(
                    f"Worker session {session_id} is already {current.status}"
                )
            completed = replace(
                current,
                status=status,
                finish_reason=finish_reason,
                trace_digest=trace_digest,
                token_budget=token_budget if token_budget is not None else current.token_budget,
                token_usage=token_usage,
                process_returncode=process_returncode,
                error_type=error_type,
                error_message=error_message,
                completed_at=self._clock(),
            )
            usage = completed.token_usage
            self._connection.execute(
                """UPDATE worker_sessions SET
                       status = ?, finish_reason = ?, trace_digest = ?,
                       token_budget = ?, uncached_input_tokens = ?, output_tokens = ?,
                       cache_read_tokens = ?, cache_write_tokens = ?, credits = ?,
                       process_returncode = ?, error_type = ?, error_message = ?,
                       completed_at = ? WHERE id = ? AND status = 'running'""",
                (
                    completed.status,
                    completed.finish_reason,
                    completed.trace_digest,
                    completed.token_budget,
                    None if usage is None else usage.uncached_input_tokens,
                    None if usage is None else usage.output_tokens,
                    None if usage is None else usage.cache_read_tokens,
                    None if usage is None else usage.cache_write_tokens,
                    None if usage is None else usage.credits,
                    completed.process_returncode,
                    completed.error_type,
                    completed.error_message,
                    completed.completed_at,
                    completed.id,
                ),
            )
        return completed

    def get_worker_session(self, session_id: WorkerSessionId) -> WorkerSession:
        """Return one Worker session lifecycle record."""
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM worker_sessions WHERE id = ?", (session_id,)
            ).fetchone()
        if row is None:
            raise KeyError(f"Worker session not found: {session_id}")
        return self._map_worker_session(row)

    def list_worker_sessions(
        self,
        *,
        campaign_id: CampaignId | None = None,
        lineage_id: LineageId | None = None,
        epoch_id: EpochId | None = None,
        attempt_id: AttemptId | None = None,
        subject_id: str | None = None,
        role: WorkerSessionRole | None = None,
        status: WorkerSessionStatus | None = None,
    ) -> list[WorkerSession]:
        """List Worker sessions using optional exact-match catalog filters."""
        filters: list[str] = []
        values: list[object] = []
        for column, value in (
            ("campaign_id", campaign_id),
            ("lineage_id", lineage_id),
            ("epoch_id", epoch_id),
            ("attempt_id", attempt_id),
            ("subject_id", subject_id),
            ("role", role),
            ("status", status),
        ):
            if value is not None:
                filters.append(f"{column} = ?")
                values.append(value)
        where = "" if not filters else " WHERE " + " AND ".join(filters)
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM worker_sessions" + where + " ORDER BY started_at, id",
                values,
            ).fetchall()
        return [self._map_worker_session(row) for row in rows]

    @staticmethod
    def _map_worker_session(row: sqlite3.Row) -> WorkerSession:
        usage_values = (
            _optional_int(row, "uncached_input_tokens"),
            _optional_int(row, "output_tokens"),
            _optional_int(row, "cache_read_tokens"),
            _optional_int(row, "cache_write_tokens"),
        )
        usage = None
        if all(value is not None for value in usage_values):
            usage = TokenUsage(
                *cast(tuple[int, int, int, int], usage_values),
                credits=_optional_float(row, "credits"),
            )
        return WorkerSession(
            id=parse_worker_session_id(_required_text(row, "id")),
            role=WorkerSessionRole(_required_text(row, "role")),
            subject_id=_required_text(row, "subject_id"),
            external_run_id=_required_text(row, "external_run_id"),
            campaign_id=(
                None
                if (value := _optional_text(row, "campaign_id")) is None
                else parse_campaign_id(value)
            ),
            lineage_id=(
                None
                if (value := _optional_text(row, "lineage_id")) is None
                else parse_lineage_id(value)
            ),
            epoch_id=(
                None
                if (value := _optional_text(row, "epoch_id")) is None
                else parse_epoch_id(value)
            ),
            attempt_id=(
                None
                if (value := _optional_text(row, "attempt_id")) is None
                else parse_attempt_id(value)
            ),
            recovery_generation=_optional_int(row, "recovery_generation"),
            backend=_optional_text(row, "backend"),
            model=_optional_text(row, "model"),
            workspace_path=_required_text(row, "workspace_path"),
            status=WorkerSessionStatus(_required_text(row, "status")),
            finish_reason=_optional_text(row, "finish_reason"),
            trace_digest=(
                None
                if (value := _optional_text(row, "trace_digest")) is None
                else parse_artifact_digest(value)
            ),
            token_budget=_optional_int(row, "token_budget"),
            token_usage=usage,
            process_returncode=_optional_int(row, "process_returncode"),
            error_type=_optional_text(row, "error_type"),
            error_message=_optional_text(row, "error_message"),
            started_at=_required_text(row, "started_at"),
            completed_at=_optional_text(row, "completed_at"),
        )

    def list_referenced_artifact_digests(self) -> set[ArtifactDigest]:
        """Return every CAS digest retained by Registry state or unpruned Events."""
        queries = (
            "SELECT evaluation_contract_digest AS digest FROM campaigns",
            "SELECT agent_problem_digest AS digest FROM campaigns",
            "SELECT optimizer_digest AS digest FROM kernel_agent_revisions",
            "SELECT source_provenance_digest AS digest FROM kernel_agent_revisions",
            "SELECT evolution_trace_digest AS digest FROM kernel_agent_revisions",
            "SELECT evolution_trace_digest AS digest FROM epoch_challengers",
            "SELECT artifact_digest AS digest FROM kernel_revisions",
            "SELECT gateway_result_digest AS digest FROM kernel_revisions",
            "SELECT gateway_result_digest AS digest FROM kernel_measurements",
            "SELECT evidence_checkpoint AS digest FROM lineages",
            "SELECT evidence_checkpoint AS digest FROM epochs",
            "SELECT attempt_evidence_digest AS digest FROM attempts",
            "SELECT attempt_report_digest AS digest FROM attempts",
            "SELECT artifact_digest AS digest FROM attempt_session_traces",
            "SELECT trace_digest AS digest FROM worker_sessions",
            """SELECT value AS digest FROM runtime_events, json_tree(payload_json)
               WHERE json_tree.type = 'text' AND value GLOB 'sha256:[0-9a-f]*'
                 AND (CAST(json_tree.key AS TEXT) LIKE '%artifact_digest'
                      OR json_tree.key IN (
                          'gateway_result_digest', 'evidence_checkpoint_digest',
                          'initial_evidence_digest', 'evolution_trace_digest',
                          'session_trace_digest'))""",
        )
        values: set[ArtifactDigest] = set()
        with self._lock:
            for query in queries:
                for row in self._connection.execute(query).fetchall():
                    value = row["digest"]
                    if value is not None:
                        if not isinstance(value, str):
                            raise TypeError("persisted Artifact Digest must be text")
                        values.add(parse_artifact_digest(value))
        return values

    def __enter__(self) -> SqliteRegistry:
        return self

    def __exit__(self, _exc_type: object, _exc: object, _traceback: object) -> None:
        self.close()

    @contextmanager
    def _transaction(self) -> Iterator[None]:
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                if self._require_fencing:
                    self._assert_active_fence()
                yield
            except BaseException:
                self._connection.execute("ROLLBACK")
                raise
            else:
                self._connection.execute("COMMIT")

    def _migrate(self) -> None:
        version_row = self._connection.execute("PRAGMA user_version").fetchone()
        if version_row is None:
            raise RuntimeError("SQLite did not return user_version")
        version = int(version_row[0])
        if version > SCHEMA_VERSION:
            raise RuntimeError(
                f"Registry schema {version} is newer than supported {SCHEMA_VERSION}"
            )
        if version == SCHEMA_VERSION:
            return
        if version == 23:
            with self._lock:
                self._connection.executescript(
                    """
                    BEGIN IMMEDIATE;
                    CREATE TABLE IF NOT EXISTS worker_sessions (
                        id TEXT PRIMARY KEY,
                        role TEXT NOT NULL CHECK (role IN (
                            'optimizer', 'framework_baseline',
                            'problem_generalization', 'evolver'
                        )),
                        subject_id TEXT NOT NULL,
                        external_run_id TEXT NOT NULL,
                        campaign_id TEXT,
                        lineage_id TEXT,
                        epoch_id TEXT,
                        attempt_id TEXT,
                        recovery_generation INTEGER CHECK (
                            recovery_generation IS NULL OR recovery_generation >= 0
                        ),
                        backend TEXT CHECK (
                            backend IS NULL OR backend IN ('claude', 'codex', 'qodercli', 'pi')
                        ),
                        model TEXT,
                        workspace_path TEXT NOT NULL,
                        status TEXT NOT NULL CHECK (status IN (
                            'running', 'completed', 'failed', 'timed_out'
                        )),
                        finish_reason TEXT,
                        trace_digest TEXT,
                        token_budget INTEGER CHECK (token_budget IS NULL OR token_budget > 0),
                        uncached_input_tokens INTEGER,
                        output_tokens INTEGER,
                        cache_read_tokens INTEGER,
                        cache_write_tokens INTEGER,
                        process_returncode INTEGER,
                        error_type TEXT,
                        error_message TEXT,
                        started_at TEXT NOT NULL,
                        completed_at TEXT,
                        CHECK ((status = 'running') =
                               (completed_at IS NULL AND finish_reason IS NULL)),
                        CHECK ((uncached_input_tokens IS NULL AND output_tokens IS NULL
                                AND cache_read_tokens IS NULL AND cache_write_tokens IS NULL)
                               OR
                               (uncached_input_tokens >= 0 AND output_tokens >= 0
                                AND cache_read_tokens >= 0 AND cache_write_tokens >= 0))
                    );
                    CREATE INDEX IF NOT EXISTS worker_sessions_by_subject
                        ON worker_sessions(subject_id, started_at, id);
                    CREATE INDEX IF NOT EXISTS worker_sessions_by_attempt
                        ON worker_sessions(attempt_id, started_at, id);
                    CREATE INDEX IF NOT EXISTS worker_sessions_by_lineage
                        ON worker_sessions(lineage_id, started_at, id);
                    CREATE INDEX IF NOT EXISTS worker_sessions_by_campaign
                        ON worker_sessions(campaign_id, started_at, id);
                    CREATE INDEX IF NOT EXISTS worker_sessions_by_epoch
                        ON worker_sessions(epoch_id, started_at, id);
                    PRAGMA user_version = 24;
                    COMMIT;
                    """
                )
            version = 24
        if version == 24:
            with self._lock:
                self._connection.execute("BEGIN IMMEDIATE")
                try:
                    if "credits" not in self._table_columns("worker_sessions"):
                        self._connection.execute(
                            "ALTER TABLE worker_sessions ADD COLUMN credits REAL "
                            "CHECK (credits IS NULL OR credits >= 0)"
                        )
                    if self._has_tables("attempt_session_traces") and "credits" not in (
                        self._table_columns("attempt_session_traces")
                    ):
                        self._connection.execute(
                            "ALTER TABLE attempt_session_traces ADD COLUMN credits REAL "
                            "CHECK (credits IS NULL OR credits >= 0)"
                        )
                    self._connection.execute("PRAGMA user_version = 25")
                except BaseException:
                    self._connection.execute("ROLLBACK")
                    raise
                else:
                    self._connection.execute("COMMIT")
            version = 25
        if version == 25:
            with self._lock:
                self._connection.executescript(
                    """
                    BEGIN IMMEDIATE;
                    DROP TABLE IF EXISTS wiki_feedback_outbox;
                    PRAGMA user_version = 26;
                    COMMIT;
                    """
                )
            version = 26
        if version == 26:
            with self._lock:
                self._connection.execute("BEGIN IMMEDIATE")
                try:
                    if self._has_tables("attempts") and "runtime_state_digest" not in (
                        self._table_columns("attempts")
                    ):
                        self._connection.execute(
                            "ALTER TABLE attempts ADD COLUMN runtime_state_digest TEXT"
                        )
                    self._connection.execute("PRAGMA user_version = 27")
                except BaseException:
                    self._connection.execute("ROLLBACK")
                    raise
                else:
                    self._connection.execute("COMMIT")
            version = 27
        if version == 27:
            with self._lock:
                self._connection.execute("BEGIN IMMEDIATE")
                try:
                    if self._has_tables("kernel_agent_revisions") and (
                        "runtime_state_digest"
                        not in self._table_columns("kernel_agent_revisions")
                    ):
                        self._connection.execute(
                            "ALTER TABLE kernel_agent_revisions "
                            "ADD COLUMN runtime_state_digest TEXT"
                        )
                    self._connection.execute("PRAGMA user_version = 28")
                except BaseException:
                    self._connection.execute("ROLLBACK")
                    raise
                else:
                    self._connection.execute("COMMIT")
            version = 28
        if version == 28:
            with self._lock:
                self._connection.execute("BEGIN IMMEDIATE")
                try:
                    if self._has_tables("attempts") and (
                        "input_runtime_state_digest" not in self._table_columns("attempts")
                    ):
                        self._connection.execute(
                            "ALTER TABLE attempts "
                            "ADD COLUMN input_runtime_state_digest TEXT"
                        )
                    self._connection.execute("PRAGMA user_version = 29")
                except BaseException:
                    self._connection.execute("ROLLBACK")
                    raise
                else:
                    self._connection.execute("COMMIT")
            return
        if version == 14:
            with self._lock:
                self._connection.executescript(
                    """
                    BEGIN IMMEDIATE;
                    CREATE TABLE kernel_measurements (
                        id TEXT PRIMARY KEY,
                        kernel_revision_id TEXT NOT NULL REFERENCES kernel_revisions(id),
                        purpose TEXT NOT NULL CHECK (purpose IN (
                            'kernel_retention', 'agent_promotion'
                        )),
                        repeat INTEGER NOT NULL CHECK (repeat >= 0),
                        correct INTEGER NOT NULL CHECK (correct IN (0, 1)),
                        latency_us REAL,
                        gateway_result_digest TEXT,
                        agate_job_id TEXT,
                        created_at TEXT NOT NULL,
                        CHECK ((correct = 1 AND latency_us > 0) OR
                               (correct = 0 AND latency_us IS NULL))
                    );
                    CREATE INDEX kernel_measurements_by_revision
                        ON kernel_measurements(kernel_revision_id, created_at, id);
                    PRAGMA user_version = 15;
                    COMMIT;
                    """
                )
            version = 15
        if version == 15:
            with self._lock:
                self._connection.execute("BEGIN IMMEDIATE")
                try:
                    self._connection.execute(
                        """CREATE TABLE lineage_kernel_versions (
                               kernel_revision_id TEXT PRIMARY KEY
                                   REFERENCES kernel_revisions(id),
                               lineage_id TEXT NOT NULL REFERENCES lineages(id),
                               revision_number INTEGER NOT NULL CHECK (revision_number >= 0),
                               linked_at TEXT NOT NULL,
                               UNIQUE(lineage_id, revision_number)
                           )"""
                    )
                    self._connection.execute(
                        """CREATE INDEX lineage_kernel_versions_by_lineage
                           ON lineage_kernel_versions(lineage_id, revision_number)"""
                    )
                    if self._has_tables("lineages", "epochs", "attempts"):
                        self._backfill_lineage_kernel_versions()
                    self._connection.execute("PRAGMA user_version = 16")
                except BaseException:
                    self._connection.execute("ROLLBACK")
                    raise
                else:
                    self._connection.execute("COMMIT")
            version = 16
        if version == 16:
            with self._lock:
                self._connection.execute("BEGIN IMMEDIATE")
                try:
                    self._connection.execute(
                        """CREATE TABLE lineage_agent_versions (
                               kernel_agent_revision_id TEXT PRIMARY KEY
                                   REFERENCES kernel_agent_revisions(id),
                               lineage_id TEXT NOT NULL REFERENCES lineages(id),
                               revision_number INTEGER NOT NULL CHECK (revision_number >= 0),
                               linked_at TEXT NOT NULL,
                               introduced_epoch_id TEXT REFERENCES epochs(id),
                               UNIQUE(lineage_id, revision_number)
                           )"""
                    )
                    self._connection.execute(
                        """CREATE INDEX lineage_agent_versions_by_lineage
                           ON lineage_agent_versions(lineage_id, revision_number)"""
                    )
                    if self._has_tables("lineages", "epochs", "kernel_agent_revisions"):
                        self._backfill_lineage_agent_versions()
                    self._connection.execute("PRAGMA user_version = 17")
                except BaseException:
                    self._connection.execute("ROLLBACK")
                    raise
                else:
                    self._connection.execute("COMMIT")
            version = 17
        if version == 17:
            with self._lock:
                self._connection.execute("BEGIN IMMEDIATE")
                try:
                    additions = {
                        "lineages": (
                            (
                                "challenger_count",
                                "INTEGER NOT NULL DEFAULT 1 CHECK (challenger_count >= 0)",
                            ),
                            (
                                "trajectories_per_branch",
                                "INTEGER NOT NULL DEFAULT 1 CHECK (trajectories_per_branch > 0)",
                            ),
                        ),
                        "epochs": (
                            (
                                "challenger_count",
                                "INTEGER NOT NULL DEFAULT 1 CHECK (challenger_count >= 0)",
                            ),
                            (
                                "trajectories_per_branch",
                                "INTEGER NOT NULL DEFAULT 1 CHECK (trajectories_per_branch > 0)",
                            ),
                        ),
                        "attempts": (
                            (
                                "challenger_ordinal",
                                "INTEGER NOT NULL DEFAULT 0 CHECK (challenger_ordinal >= 0)",
                            ),
                            (
                                "trajectory_ordinal",
                                "INTEGER NOT NULL DEFAULT 1 CHECK (trajectory_ordinal > 0)",
                            ),
                            (
                                "iteration_ordinal",
                                "INTEGER NOT NULL DEFAULT 1 CHECK (iteration_ordinal > 0)",
                            ),
                        ),
                    }
                    for table, columns in additions.items():
                        if not self._has_tables(table):
                            continue
                        existing = self._table_columns(table)
                        for column, declaration in columns:
                            if column not in existing:
                                self._connection.execute(
                                    f"ALTER TABLE {table} ADD COLUMN {column} {declaration}"
                                )
                    if self._has_tables("attempts"):
                        self._connection.execute(
                            "UPDATE attempts SET challenger_ordinal = 1 WHERE branch = 'challenger'"
                        )
                        self._connection.execute("UPDATE attempts SET iteration_ordinal = ordinal")
                    self._connection.execute(
                        """CREATE TABLE IF NOT EXISTS epoch_challengers (
                               epoch_id TEXT NOT NULL REFERENCES epochs(id),
                               challenger_ordinal INTEGER NOT NULL
                                   CHECK (challenger_ordinal > 0),
                               kernel_agent_revision_id TEXT NOT NULL
                                   REFERENCES kernel_agent_revisions(id),
                               PRIMARY KEY (epoch_id, challenger_ordinal),
                               UNIQUE (epoch_id, kernel_agent_revision_id)
                           )"""
                    )
                    if self._has_tables("epochs") and "challenger_kernel_agent_revision_id" in (
                        self._table_columns("epochs")
                    ):
                        self._connection.execute(
                            """INSERT OR IGNORE INTO epoch_challengers(
                                   epoch_id, challenger_ordinal, kernel_agent_revision_id
                               )
                               SELECT id, 1, challenger_kernel_agent_revision_id FROM epochs
                               WHERE challenger_kernel_agent_revision_id IS NOT NULL"""
                        )
                    if self._has_tables("attempts"):
                        self._connection.execute(
                            """CREATE UNIQUE INDEX IF NOT EXISTS
                                   attempts_by_trajectory_iteration
                               ON attempts(
                                   epoch_id, branch, challenger_ordinal,
                                   trajectory_ordinal, iteration_ordinal
                               )"""
                        )
                    self._connection.execute("PRAGMA user_version = 18")
                except BaseException:
                    self._connection.execute("ROLLBACK")
                    raise
                else:
                    self._connection.execute("COMMIT")
            version = 18
        if version == 18:
            with self._lock:
                self._connection.execute("BEGIN IMMEDIATE")
                try:
                    if self._has_tables("lineages") and "challenger_start_epoch" not in (
                        self._table_columns("lineages")
                    ):
                        self._connection.execute(
                            """ALTER TABLE lineages ADD COLUMN challenger_start_epoch
                               INTEGER NOT NULL DEFAULT 1
                               CHECK (challenger_start_epoch > 0)"""
                        )
                    self._connection.execute("PRAGMA user_version = 19")
                except BaseException:
                    self._connection.execute("ROLLBACK")
                    raise
                else:
                    self._connection.execute("COMMIT")
            version = 19
        if version == 19:
            with self._lock:
                self._connection.execute("BEGIN IMMEDIATE")
                try:
                    if self._has_tables("epoch_challengers"):
                        self._connection.execute(
                            "ALTER TABLE epoch_challengers RENAME TO epoch_challengers_v19"
                        )
                        self._connection.execute(
                            """CREATE TABLE epoch_challengers (
                                   epoch_id TEXT NOT NULL REFERENCES epochs(id),
                                   challenger_ordinal INTEGER NOT NULL
                                       CHECK (challenger_ordinal > 0),
                                   kernel_agent_revision_id TEXT NOT NULL
                                       REFERENCES kernel_agent_revisions(id),
                                   proposal_type TEXT NOT NULL CHECK (proposal_type IN (
                                       'evolved', 'reuse', 'evolve_from_history'
                                   )),
                                   base_kernel_agent_revision_id TEXT NOT NULL
                                       REFERENCES kernel_agent_revisions(id),
                                   evolution_trace_digest TEXT NOT NULL,
                                   PRIMARY KEY (epoch_id, challenger_ordinal),
                                   UNIQUE (epoch_id, kernel_agent_revision_id)
                               )"""
                        )
                        if self._has_tables("kernel_agent_revisions"):
                            self._connection.execute(
                                """INSERT INTO epoch_challengers(
                                   epoch_id, challenger_ordinal, kernel_agent_revision_id,
                                   proposal_type, base_kernel_agent_revision_id,
                                   evolution_trace_digest
                               )
                               SELECT ec.epoch_id, ec.challenger_ordinal,
                                      ec.kernel_agent_revision_id, 'evolved', r.parent_id,
                                      r.evolution_trace_digest
                               FROM epoch_challengers_v19 ec
                               JOIN kernel_agent_revisions r
                                 ON r.id = ec.kernel_agent_revision_id
                               WHERE r.parent_id IS NOT NULL
                                 AND r.evolution_trace_digest IS NOT NULL"""
                            )
                        source_count = self._connection.execute(
                            "SELECT COUNT(*) AS count FROM epoch_challengers_v19"
                        ).fetchone()
                        target_count = self._connection.execute(
                            "SELECT COUNT(*) AS count FROM epoch_challengers"
                        ).fetchone()
                        if (
                            source_count is None
                            or target_count is None
                            or (
                                _required_int(source_count, "count")
                                != _required_int(target_count, "count")
                            )
                        ):
                            raise RuntimeError("legacy Epoch Challenger lacks revision provenance")
                        self._connection.execute("DROP TABLE epoch_challengers_v19")
                    self._connection.execute("PRAGMA user_version = 20")
                except BaseException:
                    self._connection.execute("ROLLBACK")
                    raise
                else:
                    self._connection.execute("COMMIT")
            version = 20
        if version == 20:
            with self._lock:
                self._connection.execute("BEGIN IMMEDIATE")
                try:
                    if self._has_tables("campaigns") and "problem_generalization_model" not in (
                        self._table_columns("campaigns")
                    ):
                        self._connection.execute(
                            "ALTER TABLE campaigns ADD COLUMN problem_generalization_model TEXT"
                        )
                    if self._has_tables("lineages"):
                        lineage_columns = self._table_columns("lineages")
                        if "optimizer_model" not in lineage_columns:
                            self._connection.execute(
                                "ALTER TABLE lineages ADD COLUMN optimizer_model TEXT"
                            )
                        if "evolver_model" not in lineage_columns:
                            self._connection.execute(
                                "ALTER TABLE lineages ADD COLUMN evolver_model TEXT"
                            )
                    self._connection.execute("PRAGMA user_version = 21")
                except BaseException:
                    self._connection.execute("ROLLBACK")
                    raise
                else:
                    self._connection.execute("COMMIT")
            version = 21
        if version == 21:
            with self._lock:
                agent_columns = (
                    self._table_columns("kernel_agent_revisions")
                    if self._has_tables("kernel_agent_revisions")
                    else set()
                )
                complete_agent_columns = {
                    "id",
                    "parent_id",
                    "creation_key",
                    "dsl",
                    "optimizer_digest",
                    "created_by",
                    "created_at",
                    "source_provenance_digest",
                    "evolution_trace_digest",
                }
                if not complete_agent_columns <= agent_columns:
                    self._connection.execute("PRAGMA user_version = 22")
                    version = 22
                else:
                    self._connection.execute("PRAGMA foreign_keys = OFF")
                    try:
                        self._connection.executescript(
                            """
                            BEGIN IMMEDIATE;
                            CREATE TABLE kernel_agent_revisions_v22 (
                                id TEXT PRIMARY KEY,
                                parent_id TEXT REFERENCES kernel_agent_revisions_v22(id),
                                creation_key TEXT NOT NULL UNIQUE,
                                dsl TEXT NOT NULL CHECK (dsl IN ('cuda', 'triton', 'cutedsl')),
                                optimizer_digest TEXT NOT NULL,
                                created_by TEXT NOT NULL CHECK (created_by IN (
                                    'bootstrap', 'lineage_seed', 'evolver'
                                )),
                                created_at TEXT NOT NULL,
                                source_provenance_digest TEXT,
                                evolution_trace_digest TEXT,
                                CHECK ((created_by IN ('bootstrap', 'lineage_seed')
                                        AND source_provenance_digest IS NOT NULL
                                        AND evolution_trace_digest IS NULL) OR
                                       (created_by = 'evolver'
                                        AND source_provenance_digest IS NULL
                                        AND evolution_trace_digest IS NOT NULL))
                            );
                            INSERT INTO kernel_agent_revisions_v22 (
                                id, parent_id, creation_key, dsl, optimizer_digest,
                                created_by, created_at, source_provenance_digest,
                                evolution_trace_digest
                            )
                                SELECT id, parent_id, creation_key, dsl, optimizer_digest,
                                       created_by, created_at, source_provenance_digest,
                                       evolution_trace_digest
                                FROM kernel_agent_revisions;
                            DROP TABLE kernel_agent_revisions;
                            ALTER TABLE kernel_agent_revisions_v22
                                RENAME TO kernel_agent_revisions;
                            PRAGMA user_version = 22;
                            COMMIT;
                            """
                        )
                    except BaseException:
                        if self._connection.in_transaction:
                            self._connection.execute("ROLLBACK")
                        raise
                    finally:
                        self._connection.execute("PRAGMA foreign_keys = ON")
                    version = 22
        if version == 22:
            with self._lock:
                self._connection.execute("BEGIN IMMEDIATE")
                try:
                    if self._has_tables("campaigns") and "evolver_commit" not in (
                        self._table_columns("campaigns")
                    ):
                        self._connection.execute(
                            "ALTER TABLE campaigns ADD COLUMN evolver_commit TEXT"
                        )
                    self._connection.execute("PRAGMA user_version = 23")
                except BaseException:
                    self._connection.execute("ROLLBACK")
                    raise
                else:
                    self._connection.execute("COMMIT")
            version = 23
        if version == 23:
            with self._lock:
                self._connection.executescript(
                    """
                    BEGIN IMMEDIATE;
                    CREATE TABLE IF NOT EXISTS worker_sessions (
                        id TEXT PRIMARY KEY,
                        role TEXT NOT NULL CHECK (role IN (
                            'optimizer', 'framework_baseline',
                            'problem_generalization', 'evolver'
                        )),
                        subject_id TEXT NOT NULL,
                        external_run_id TEXT NOT NULL,
                        campaign_id TEXT,
                        lineage_id TEXT,
                        epoch_id TEXT,
                        attempt_id TEXT,
                        recovery_generation INTEGER CHECK (
                            recovery_generation IS NULL OR recovery_generation >= 0
                        ),
                        backend TEXT CHECK (
                            backend IS NULL OR backend IN ('claude', 'codex', 'qodercli', 'pi')
                        ),
                        model TEXT,
                        workspace_path TEXT NOT NULL,
                        status TEXT NOT NULL CHECK (status IN (
                            'running', 'completed', 'failed', 'timed_out'
                        )),
                        finish_reason TEXT,
                        trace_digest TEXT,
                        token_budget INTEGER CHECK (token_budget IS NULL OR token_budget > 0),
                        uncached_input_tokens INTEGER,
                        output_tokens INTEGER,
                        cache_read_tokens INTEGER,
                        cache_write_tokens INTEGER,
                        process_returncode INTEGER,
                        error_type TEXT,
                        error_message TEXT,
                        started_at TEXT NOT NULL,
                        completed_at TEXT,
                        CHECK ((status = 'running') =
                               (completed_at IS NULL AND finish_reason IS NULL)),
                        CHECK ((uncached_input_tokens IS NULL AND output_tokens IS NULL
                                AND cache_read_tokens IS NULL AND cache_write_tokens IS NULL)
                               OR
                               (uncached_input_tokens >= 0 AND output_tokens >= 0
                                AND cache_read_tokens >= 0 AND cache_write_tokens >= 0))
                    );
                    CREATE INDEX IF NOT EXISTS worker_sessions_by_subject
                        ON worker_sessions(subject_id, started_at, id);
                    CREATE INDEX IF NOT EXISTS worker_sessions_by_attempt
                        ON worker_sessions(attempt_id, started_at, id);
                    CREATE INDEX IF NOT EXISTS worker_sessions_by_lineage
                        ON worker_sessions(lineage_id, started_at, id);
                    CREATE INDEX IF NOT EXISTS worker_sessions_by_campaign
                        ON worker_sessions(campaign_id, started_at, id);
                    CREATE INDEX IF NOT EXISTS worker_sessions_by_epoch
                        ON worker_sessions(epoch_id, started_at, id);
                    PRAGMA user_version = 24;
                    COMMIT;
                    """
                )
            self._migrate()
            return
        if version != 0:
            raise RuntimeError(f"no migration path from Registry schema {version}")

        with self._lock:
            self._connection.executescript(
                """
                BEGIN IMMEDIATE;
                CREATE TABLE campaigns (
                    id TEXT PRIMARY KEY,
                    operator TEXT NOT NULL,
                    hardware_target TEXT NOT NULL,
                    evaluation_contract_digest TEXT NOT NULL,
                    agent_problem_digest TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    problem_generalization_model TEXT,
                    evolver_commit TEXT,
                    status TEXT NOT NULL CHECK (status IN (
                        'active', 'completed', 'cancelled', 'failed'
                    ))
                );
                CREATE TABLE campaign_tasks (
                    id TEXT PRIMARY KEY,
                    creation_key TEXT NOT NULL UNIQUE,
                    campaign_id TEXT NOT NULL REFERENCES campaigns(id),
                    target_epoch_number INTEGER NOT NULL CHECK (target_epoch_number > 0),
                    finalize INTEGER NOT NULL CHECK (finalize IN (0, 1)),
                    status TEXT NOT NULL CHECK (status IN (
                        'queued', 'running', 'cancelling', 'completed', 'failed', 'cancelled'
                    )),
                    attempt_count INTEGER NOT NULL DEFAULT 0 CHECK (attempt_count >= 0),
                    lease_owner TEXT,
                    lease_expires_at TEXT,
                    last_error TEXT,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    completed_at TEXT,
                    CHECK ((status IN ('running', 'cancelling')) =
                           (lease_owner IS NOT NULL AND lease_expires_at IS NOT NULL))
                );
                CREATE TABLE kernel_agent_revisions (
                    id TEXT PRIMARY KEY,
                    parent_id TEXT REFERENCES kernel_agent_revisions(id),
                    creation_key TEXT NOT NULL UNIQUE,
                    dsl TEXT NOT NULL CHECK (dsl IN ('cuda', 'triton', 'cutedsl')),
                    optimizer_digest TEXT NOT NULL,
                    created_by TEXT NOT NULL CHECK (created_by IN (
                        'bootstrap', 'lineage_seed', 'evolver'
                    )),
                    created_at TEXT NOT NULL,
                    source_provenance_digest TEXT,
                    evolution_trace_digest TEXT,
                    runtime_state_digest TEXT,
                    CHECK ((created_by IN ('bootstrap', 'lineage_seed')
                            AND source_provenance_digest IS NOT NULL
                            AND evolution_trace_digest IS NULL) OR
                           (created_by = 'evolver' AND source_provenance_digest IS NULL
                            AND evolution_trace_digest IS NOT NULL))
                );
                CREATE TABLE kernel_revisions (
                    id TEXT PRIMARY KEY,
                    parent_id TEXT REFERENCES kernel_revisions(id),
                    artifact_digest TEXT NOT NULL,
                    produced_by_attempt_id TEXT,
                    correct INTEGER NOT NULL CHECK (correct IN (0, 1)),
                    latency_us REAL,
                    gateway_result_digest TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    CHECK ((correct = 1 AND latency_us > 0) OR
                           (correct = 0 AND latency_us IS NULL))
                );
                CREATE UNIQUE INDEX one_kernel_per_attempt
                    ON kernel_revisions(produced_by_attempt_id)
                    WHERE produced_by_attempt_id IS NOT NULL;
                CREATE TABLE kernel_measurements (
                    id TEXT PRIMARY KEY,
                    kernel_revision_id TEXT NOT NULL REFERENCES kernel_revisions(id),
                    purpose TEXT NOT NULL CHECK (purpose IN (
                        'kernel_retention', 'agent_promotion'
                    )),
                    repeat INTEGER NOT NULL CHECK (repeat >= 0),
                    correct INTEGER NOT NULL CHECK (correct IN (0, 1)),
                    latency_us REAL,
                    gateway_result_digest TEXT,
                    agate_job_id TEXT,
                    created_at TEXT NOT NULL,
                    CHECK ((correct = 1 AND latency_us > 0) OR
                           (correct = 0 AND latency_us IS NULL))
                );
                CREATE INDEX kernel_measurements_by_revision
                    ON kernel_measurements(kernel_revision_id, created_at, id);
                CREATE TABLE lineages (
                    id TEXT PRIMARY KEY,
                    campaign_id TEXT NOT NULL REFERENCES campaigns(id),
                    dsl TEXT NOT NULL CHECK (dsl IN ('cuda', 'triton', 'cutedsl')),
                    hardware_target TEXT NOT NULL,
                    active_kernel_agent_revision_id TEXT NOT NULL
                        REFERENCES kernel_agent_revisions(id),
                    best_kernel_revision_id TEXT NOT NULL REFERENCES kernel_revisions(id),
                    evidence_checkpoint TEXT NOT NULL,
                    attempts_per_branch INTEGER NOT NULL CHECK (attempts_per_branch > 0),
                    next_epoch_number INTEGER NOT NULL CHECK (next_epoch_number > 0),
                    status TEXT NOT NULL
                        CHECK (status IN (
                            'ready', 'running', 'awaiting_evidence', 'completed', 'failed',
                            'cancelled'
                        )),
                    challenger_count INTEGER NOT NULL CHECK (challenger_count >= 0),
                    challenger_start_epoch INTEGER NOT NULL
                        CHECK (challenger_start_epoch > 0),
                    trajectories_per_branch INTEGER NOT NULL
                        CHECK (trajectories_per_branch > 0),
                    optimizer_model TEXT,
                    evolver_model TEXT
                );
                CREATE TABLE lineage_kernel_versions (
                    kernel_revision_id TEXT PRIMARY KEY REFERENCES kernel_revisions(id),
                    lineage_id TEXT NOT NULL REFERENCES lineages(id),
                    revision_number INTEGER NOT NULL CHECK (revision_number >= 0),
                    linked_at TEXT NOT NULL,
                    UNIQUE(lineage_id, revision_number)
                );
                CREATE INDEX lineage_kernel_versions_by_lineage
                    ON lineage_kernel_versions(lineage_id, revision_number);
                CREATE TABLE epochs (
                    id TEXT PRIMARY KEY,
                    lineage_id TEXT NOT NULL REFERENCES lineages(id),
                    number INTEGER NOT NULL CHECK (number > 0),
                    active_kernel_agent_revision_id TEXT NOT NULL
                        REFERENCES kernel_agent_revisions(id),
                    challenger_kernel_agent_revision_id TEXT
                        REFERENCES kernel_agent_revisions(id),
                    starting_kernel_revision_id TEXT NOT NULL REFERENCES kernel_revisions(id),
                    evidence_checkpoint TEXT NOT NULL,
                    attempts_per_branch INTEGER NOT NULL CHECK (attempts_per_branch > 0),
                    status TEXT NOT NULL CHECK (status IN (
                        'building_challenger', 'ready', 'running',
                        'selecting', 'completed', 'failed'
                    )),
                    winner_kernel_agent_revision_id TEXT
                        REFERENCES kernel_agent_revisions(id),
                    best_kernel_revision_id TEXT REFERENCES kernel_revisions(id),
                    failure_reason TEXT,
                    created_at TEXT NOT NULL,
                    completed_at TEXT,
                    challenger_count INTEGER NOT NULL CHECK (challenger_count >= 0),
                    trajectories_per_branch INTEGER NOT NULL
                        CHECK (trajectories_per_branch > 0),
                    UNIQUE (lineage_id, number)
                );
                CREATE UNIQUE INDEX one_open_epoch_per_lineage
                    ON epochs(lineage_id)
                    WHERE status NOT IN ('completed', 'failed');
                CREATE TABLE lineage_agent_versions (
                    kernel_agent_revision_id TEXT PRIMARY KEY
                        REFERENCES kernel_agent_revisions(id),
                    lineage_id TEXT NOT NULL REFERENCES lineages(id),
                    revision_number INTEGER NOT NULL CHECK (revision_number >= 0),
                    linked_at TEXT NOT NULL,
                    introduced_epoch_id TEXT REFERENCES epochs(id),
                    UNIQUE(lineage_id, revision_number)
                );
                CREATE INDEX lineage_agent_versions_by_lineage
                    ON lineage_agent_versions(lineage_id, revision_number);
                CREATE TABLE epoch_challengers (
                    epoch_id TEXT NOT NULL REFERENCES epochs(id),
                    challenger_ordinal INTEGER NOT NULL CHECK (challenger_ordinal > 0),
                    kernel_agent_revision_id TEXT NOT NULL
                        REFERENCES kernel_agent_revisions(id),
                    proposal_type TEXT NOT NULL CHECK (proposal_type IN (
                        'evolved', 'reuse', 'evolve_from_history'
                    )),
                    base_kernel_agent_revision_id TEXT NOT NULL
                        REFERENCES kernel_agent_revisions(id),
                    evolution_trace_digest TEXT NOT NULL,
                    PRIMARY KEY (epoch_id, challenger_ordinal),
                    UNIQUE (epoch_id, kernel_agent_revision_id)
                );
                CREATE TABLE attempts (
                    id TEXT PRIMARY KEY,
                    epoch_id TEXT NOT NULL REFERENCES epochs(id),
                    branch TEXT NOT NULL CHECK (branch IN ('active', 'challenger')),
                    ordinal INTEGER NOT NULL CHECK (ordinal > 0),
                    kernel_agent_revision_id TEXT NOT NULL
                        REFERENCES kernel_agent_revisions(id),
                    input_kernel_revision_id TEXT NOT NULL REFERENCES kernel_revisions(id),
                    attempt_evidence_digest TEXT NOT NULL,
                    output_kernel_revision_id TEXT REFERENCES kernel_revisions(id),
                    accepted_as_branch_best INTEGER NOT NULL DEFAULT 0
                        CHECK (accepted_as_branch_best IN (0, 1)),
                    status TEXT NOT NULL
                        CHECK (status IN ('running', 'completed', 'infrastructure_failed')),
                    infrastructure_failures INTEGER NOT NULL DEFAULT 0
                        CHECK (infrastructure_failures >= 0),
                    recovery_generation INTEGER NOT NULL DEFAULT 0
                        CHECK (recovery_generation >= 0),
                    authority_started_at TEXT NOT NULL,
                    failure_reason TEXT,
                    created_at TEXT NOT NULL,
                    completed_at TEXT,
                    attempt_report_digest TEXT,
                    attempt_report_status TEXT CHECK (attempt_report_status IN (
                        'candidate_ready', 'pivot', 'blocked'
                    )),
                    runtime_state_digest TEXT,
                    input_runtime_state_digest TEXT,
                    challenger_ordinal INTEGER NOT NULL CHECK (challenger_ordinal >= 0),
                    trajectory_ordinal INTEGER NOT NULL CHECK (trajectory_ordinal > 0),
                    iteration_ordinal INTEGER NOT NULL CHECK (iteration_ordinal > 0),
                    CHECK ((attempt_report_digest IS NULL) =
                           (attempt_report_status IS NULL)),
                    UNIQUE (epoch_id, branch, ordinal)
                );
                CREATE UNIQUE INDEX attempts_by_trajectory_iteration
                    ON attempts(
                        epoch_id, branch, challenger_ordinal,
                        trajectory_ordinal, iteration_ordinal
                    );
                CREATE TABLE epoch_recoveries (
                    epoch_id TEXT NOT NULL REFERENCES epochs(id),
                    recovery_key TEXT NOT NULL,
                    generation INTEGER NOT NULL CHECK (generation > 0),
                    attempt_ids_json TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (epoch_id, recovery_key),
                    UNIQUE (epoch_id, generation)
                );
                CREATE TABLE attempt_session_traces (
                    attempt_id TEXT NOT NULL REFERENCES attempts(id),
                    run_ordinal INTEGER NOT NULL CHECK (run_ordinal > 0),
                    artifact_digest TEXT NOT NULL,
                    finish_reason TEXT NOT NULL CHECK (length(finish_reason) > 0),
                    token_budget INTEGER NOT NULL CHECK (token_budget > 0),
                    uncached_input_tokens INTEGER NOT NULL
                        CHECK (uncached_input_tokens >= 0),
                    output_tokens INTEGER NOT NULL CHECK (output_tokens >= 0),
                    cache_read_tokens INTEGER NOT NULL CHECK (cache_read_tokens >= 0),
                    cache_write_tokens INTEGER NOT NULL CHECK (cache_write_tokens >= 0),
                    credits REAL CHECK (credits IS NULL OR credits >= 0),
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (attempt_id, run_ordinal)
                );
                CREATE TABLE worker_sessions (
                    id TEXT PRIMARY KEY,
                    role TEXT NOT NULL CHECK (role IN (
                        'optimizer', 'framework_baseline',
                        'problem_generalization', 'evolver'
                    )),
                    subject_id TEXT NOT NULL,
                    external_run_id TEXT NOT NULL,
                    campaign_id TEXT,
                    lineage_id TEXT,
                    epoch_id TEXT,
                    attempt_id TEXT,
                    recovery_generation INTEGER CHECK (
                        recovery_generation IS NULL OR recovery_generation >= 0
                    ),
                    backend TEXT CHECK (
                        backend IS NULL OR backend IN ('claude', 'codex', 'qodercli', 'pi')
                    ),
                    model TEXT,
                    workspace_path TEXT NOT NULL,
                    status TEXT NOT NULL CHECK (status IN (
                        'running', 'completed', 'failed', 'timed_out'
                    )),
                    finish_reason TEXT,
                    trace_digest TEXT,
                    token_budget INTEGER CHECK (token_budget IS NULL OR token_budget > 0),
                    uncached_input_tokens INTEGER,
                    output_tokens INTEGER,
                    cache_read_tokens INTEGER,
                    cache_write_tokens INTEGER,
                    credits REAL CHECK (credits IS NULL OR credits >= 0),
                    process_returncode INTEGER,
                    error_type TEXT,
                    error_message TEXT,
                    started_at TEXT NOT NULL,
                    completed_at TEXT,
                    CHECK ((status = 'running') =
                           (completed_at IS NULL AND finish_reason IS NULL)),
                    CHECK ((uncached_input_tokens IS NULL AND output_tokens IS NULL
                            AND cache_read_tokens IS NULL AND cache_write_tokens IS NULL)
                           OR
                           (uncached_input_tokens >= 0 AND output_tokens >= 0
                            AND cache_read_tokens >= 0 AND cache_write_tokens >= 0))
                );
                CREATE INDEX worker_sessions_by_subject
                    ON worker_sessions(subject_id, started_at, id);
                CREATE INDEX worker_sessions_by_attempt
                    ON worker_sessions(attempt_id, started_at, id);
                CREATE INDEX worker_sessions_by_lineage
                    ON worker_sessions(lineage_id, started_at, id);
                CREATE INDEX worker_sessions_by_campaign
                    ON worker_sessions(campaign_id, started_at, id);
                CREATE INDEX worker_sessions_by_epoch
                    ON worker_sessions(epoch_id, started_at, id);
                CREATE TABLE runtime_events (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    kind TEXT NOT NULL,
                    aggregate_id TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE lineage_fences (
                    lineage_id TEXT PRIMARY KEY REFERENCES lineages(id),
                    generation INTEGER NOT NULL CHECK (generation > 0),
                    owner TEXT NOT NULL,
                    lease_expires_at TEXT NOT NULL
                );
                PRAGMA user_version = 29;
                COMMIT;
                """
            )

    def _has_tables(self, *names: str) -> bool:
        placeholders = ",".join("?" for _ in names)
        rows = self._connection.execute(
            f"SELECT name FROM sqlite_master WHERE type = 'table' AND name IN ({placeholders})",
            names,
        ).fetchall()
        return len(rows) == len(names)

    def _table_columns(self, name: str) -> set[str]:
        return {
            str(row["name"])
            for row in self._connection.execute(f"PRAGMA table_info({name})").fetchall()
        }

    def _backfill_lineage_kernel_versions(self) -> None:
        """Assign deterministic lineage-local numbers to pre-v16 Kernel history."""
        lineages = self._connection.execute(
            "SELECT id, best_kernel_revision_id FROM lineages ORDER BY id"
        ).fetchall()
        for lineage in lineages:
            lineage_id = _required_text(lineage, "id")
            epochs = self._connection.execute(
                """SELECT id, starting_kernel_revision_id, best_kernel_revision_id
                   FROM epochs WHERE lineage_id = ? ORDER BY number""",
                (lineage_id,),
            ).fetchall()
            ordered: list[str] = []

            if epochs:
                _append_unique_text(ordered, epochs[0]["starting_kernel_revision_id"])
            else:
                _append_unique_text(ordered, lineage["best_kernel_revision_id"])
            for epoch in epochs:
                _append_unique_text(ordered, epoch["starting_kernel_revision_id"])
                attempts = self._connection.execute(
                    """SELECT input_kernel_revision_id, output_kernel_revision_id
                       FROM attempts WHERE epoch_id = ? ORDER BY branch, ordinal""",
                    (_required_text(epoch, "id"),),
                ).fetchall()
                for attempt in attempts:
                    _append_unique_text(ordered, attempt["input_kernel_revision_id"])
                    _append_unique_text(ordered, attempt["output_kernel_revision_id"])
                _append_unique_text(ordered, epoch["best_kernel_revision_id"])
            _append_unique_text(ordered, lineage["best_kernel_revision_id"])

            for revision_number, revision_id in enumerate(ordered):
                revision = self._connection.execute(
                    "SELECT created_at FROM kernel_revisions WHERE id = ?",
                    (revision_id,),
                ).fetchone()
                if revision is None:
                    raise RuntimeError(
                        f"Lineage {lineage_id} references missing Kernel {revision_id}"
                    )
                self._connection.execute(
                    "INSERT INTO lineage_kernel_versions VALUES (?, ?, ?, ?)",
                    (
                        revision_id,
                        lineage_id,
                        revision_number,
                        _required_text(revision, "created_at"),
                    ),
                )

    def _backfill_lineage_agent_versions(self) -> None:
        """Assign deterministic lineage-local numbers to pre-v17 Agent history."""
        lineages = self._connection.execute(
            "SELECT id, active_kernel_agent_revision_id FROM lineages ORDER BY id"
        ).fetchall()
        for lineage in lineages:
            lineage_id = _required_text(lineage, "id")
            epochs = self._connection.execute(
                """SELECT id, active_kernel_agent_revision_id,
                          challenger_kernel_agent_revision_id,
                          winner_kernel_agent_revision_id
                   FROM epochs WHERE lineage_id = ? ORDER BY number""",
                (lineage_id,),
            ).fetchall()
            ordered: list[str] = []
            introduced_epochs: dict[str, str | None] = {}

            if epochs:
                _append_agent_version(
                    ordered,
                    introduced_epochs,
                    epochs[0]["active_kernel_agent_revision_id"],
                )
            else:
                _append_agent_version(
                    ordered,
                    introduced_epochs,
                    lineage["active_kernel_agent_revision_id"],
                )
            for epoch in epochs:
                epoch_id = _required_text(epoch, "id")
                _append_agent_version(
                    ordered,
                    introduced_epochs,
                    epoch["active_kernel_agent_revision_id"],
                )
                _append_agent_version(
                    ordered,
                    introduced_epochs,
                    epoch["challenger_kernel_agent_revision_id"],
                    epoch_id,
                )
                _append_agent_version(
                    ordered,
                    introduced_epochs,
                    epoch["winner_kernel_agent_revision_id"],
                )
            _append_agent_version(
                ordered,
                introduced_epochs,
                lineage["active_kernel_agent_revision_id"],
            )

            for revision_number, revision_id in enumerate(ordered):
                revision = self._connection.execute(
                    "SELECT created_at FROM kernel_agent_revisions WHERE id = ?",
                    (revision_id,),
                ).fetchone()
                if revision is None:
                    raise RuntimeError(
                        f"Lineage {lineage_id} references missing Kernel Agent {revision_id}"
                    )
                self._connection.execute(
                    "INSERT INTO lineage_agent_versions VALUES (?, ?, ?, ?, ?)",
                    (
                        revision_id,
                        lineage_id,
                        revision_number,
                        _required_text(revision, "created_at"),
                        introduced_epochs[revision_id],
                    ),
                )

    def acquire_lineage_fence(
        self,
        lineage_id: LineageId,
        owner: str,
        *,
        now: str,
        lease_expires_at: str,
    ) -> int:
        """Acquire an expired/free lease and return its monotonic fencing generation."""
        if not owner:
            raise ValueError("lineage fence owner cannot be empty")
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                self.get_lineage(lineage_id)
                row = self._connection.execute(
                    "SELECT * FROM lineage_fences WHERE lineage_id = ?", (lineage_id,)
                ).fetchone()
                if row is not None and _required_text(row, "lease_expires_at") > now:
                    raise InvalidTransitionError(
                        f"lineage {lineage_id} is already fenced by another scheduler"
                    )
                generation = 1 if row is None else _required_int(row, "generation") + 1
                self._connection.execute(
                    """INSERT INTO lineage_fences VALUES (?, ?, ?, ?)
                       ON CONFLICT(lineage_id) DO UPDATE SET
                         generation = excluded.generation,
                         owner = excluded.owner,
                         lease_expires_at = excluded.lease_expires_at""",
                    (lineage_id, generation, owner, lease_expires_at),
                )
                self._connection.execute("COMMIT")
                return generation
            except BaseException:
                self._connection.execute("ROLLBACK")
                raise

    def renew_lineage_fence(
        self,
        lineage_id: LineageId,
        generation: int,
        owner: str,
        *,
        lease_expires_at: str,
    ) -> None:
        """Extend exactly the currently owned fencing generation."""
        with self._lock:
            changed = self._connection.execute(
                """UPDATE lineage_fences SET lease_expires_at = ?
                   WHERE lineage_id = ? AND generation = ? AND owner = ?""",
                (lease_expires_at, lineage_id, generation, owner),
            ).rowcount
        if changed != 1:
            raise InvalidTransitionError("lineage fencing generation was superseded")

    def release_lineage_fence(
        self,
        lineage_id: LineageId,
        generation: int,
        owner: str,
    ) -> None:
        """Expire a held generation without deleting its monotonic counter."""
        with self._lock:
            changed = self._connection.execute(
                """UPDATE lineage_fences SET lease_expires_at = ?
                   WHERE lineage_id = ? AND generation = ? AND owner = ?""",
                ("0001-01-01T00:00:00+00:00", lineage_id, generation, owner),
            ).rowcount
        if changed != 1:
            raise InvalidTransitionError("lineage fencing generation was superseded")

    @staticmethod
    def activate_lineage_fence(
        lineage_id: LineageId,
        generation: int,
        owner: str,
    ) -> Token[tuple[LineageId, int, str] | None]:
        """Attach one held generation to Registry writes in the current context."""
        return _ACTIVE_FENCE.set((lineage_id, generation, owner))

    @staticmethod
    def deactivate_lineage_fence(token: Token[tuple[LineageId, int, str] | None]) -> None:
        """Restore the previous write context after releasing a lineage."""
        _ACTIVE_FENCE.reset(token)

    def _assert_active_fence(self) -> None:
        active = _ACTIVE_FENCE.get()
        if active is None:
            raise InvalidTransitionError("Registry mutation requires a lineage fencing token")
        lineage_id, generation, owner = active
        row = self._connection.execute(
            "SELECT * FROM lineage_fences WHERE lineage_id = ?", (lineage_id,)
        ).fetchone()
        if (
            row is None
            or _required_int(row, "generation") != generation
            or _required_text(row, "owner") != owner
            or _required_text(row, "lease_expires_at") <= self._clock()
        ):
            raise InvalidTransitionError("Registry mutation uses a stale lineage fencing token")

    def _event(self, kind: str, aggregate_id: str, payload: object | None = None) -> None:
        correlation = self._event_correlation(aggregate_id, payload)
        if payload is None:
            versioned_payload: object = {
                "schema_version": 1,
                "correlation": correlation,
            }
        elif isinstance(payload, Mapping):
            reserved = {"schema_version", "correlation"}.intersection(payload)
            if reserved:
                raise ValueError(f"Runtime Event payload overrides reserved fields: {reserved}")
            versioned_payload = {
                "schema_version": 1,
                "correlation": correlation,
                **payload,
            }
        else:
            raise TypeError("Runtime Event payload must be a mapping")
        self._connection.execute(
            """INSERT INTO runtime_events(kind, aggregate_id, payload_json, created_at)
               VALUES (?, ?, ?, ?)""",
            (kind, aggregate_id, json.dumps(versioned_payload, sort_keys=True), self._clock()),
        )

    def _event_correlation(self, aggregate_id: str, payload: object | None) -> dict[str, str]:
        payload_values = payload if isinstance(payload, Mapping) else {}
        attempt_id = payload_values.get("attempt_id")
        epoch_id = payload_values.get("epoch_id")
        if isinstance(attempt_id, str):
            aggregate_id = attempt_id
        elif isinstance(epoch_id, str):
            aggregate_id = epoch_id

        if aggregate_id.startswith("attempt_"):
            row = self._connection.execute(
                """SELECT a.id AS attempt_id, e.id AS epoch_id,
                          l.id AS lineage_id, l.campaign_id AS campaign_id
                   FROM attempts a JOIN epochs e ON e.id = a.epoch_id
                   JOIN lineages l ON l.id = e.lineage_id WHERE a.id = ?""",
                (aggregate_id,),
            ).fetchone()
        elif aggregate_id.startswith("epoch_"):
            row = self._connection.execute(
                """SELECT e.id AS epoch_id, l.id AS lineage_id,
                          l.campaign_id AS campaign_id
                   FROM epochs e JOIN lineages l ON l.id = e.lineage_id WHERE e.id = ?""",
                (aggregate_id,),
            ).fetchone()
        elif aggregate_id.startswith("lineage_"):
            row = self._connection.execute(
                """SELECT id AS lineage_id, campaign_id
                   FROM lineages WHERE id = ?""",
                (aggregate_id,),
            ).fetchone()
        elif aggregate_id.startswith("task_"):
            row = self._connection.execute(
                """SELECT id AS campaign_task_id, campaign_id
                   FROM campaign_tasks WHERE id = ?""",
                (aggregate_id,),
            ).fetchone()
        elif aggregate_id.startswith("kernelrev_"):
            row = self._connection.execute(
                """SELECT k.id AS kernel_revision_id, a.id AS attempt_id,
                          e.id AS epoch_id, l.id AS lineage_id,
                          l.campaign_id AS campaign_id
                   FROM kernel_revisions k
                   LEFT JOIN attempts a ON a.id = k.produced_by_attempt_id
                   LEFT JOIN epochs e ON e.id = a.epoch_id
                   LEFT JOIN lineages l ON l.id = e.lineage_id WHERE k.id = ?""",
                (aggregate_id,),
            ).fetchone()
        elif aggregate_id.startswith("campaign_"):
            row = self._connection.execute(
                "SELECT id AS campaign_id FROM campaigns WHERE id = ?",
                (aggregate_id,),
            ).fetchone()
        else:
            row = None
        if row is None:
            return {}
        return {
            key: value
            for key in row.keys()  # noqa: SIM118 -- sqlite3.Row iteration yields values.
            if isinstance((value := row[key]), str)
        }

    def record_runtime_event(
        self,
        kind: str,
        aggregate_id: str,
        payload: Mapping[str, object] | None = None,
    ) -> None:
        """Append one versioned event from a trusted Runtime component."""
        if not kind or not aggregate_id:
            raise ValueError("Runtime Event kind and aggregate ID cannot be empty")
        with self._transaction():
            self._event(kind, aggregate_id, payload)

    def list_runtime_events(
        self,
        *,
        after_sequence: int,
        limit: int,
        kinds: tuple[str, ...] = (),
        correlation: Mapping[str, str] | None = None,
    ) -> list[RuntimeEvent]:
        """Return one ordered event page after an exclusive sequence cursor."""
        if after_sequence < 0:
            raise ValueError("event cursor cannot be negative")
        if limit <= 0:
            raise ValueError("event page limit must be positive")
        if any(not kind for kind in kinds):
            raise ValueError("event kinds cannot be empty")
        correlation = {} if correlation is None else dict(correlation)
        json_paths = {
            "campaign_id": "$.correlation.campaign_id",
            "lineage_id": "$.correlation.lineage_id",
            "epoch_id": "$.correlation.epoch_id",
            "attempt_id": "$.correlation.attempt_id",
            "campaign_task_id": "$.correlation.campaign_task_id",
            "kernel_revision_id": "$.correlation.kernel_revision_id",
        }
        unknown = set(correlation).difference(json_paths)
        if unknown:
            raise ValueError(f"unsupported Runtime Event correlations: {sorted(unknown)}")
        conditions = ["sequence > ?"]
        parameters: list[object] = [after_sequence]
        if kinds:
            conditions.append(f"kind IN ({','.join('?' for _kind in kinds)})")
            parameters.extend(kinds)
        for key, value in correlation.items():
            conditions.append("json_extract(payload_json, ?) = ?")
            parameters.extend((json_paths[key], value))
        parameters.append(limit)
        query = (
            "SELECT * FROM runtime_events WHERE "
            + " AND ".join(conditions)
            + " ORDER BY sequence LIMIT ?"
        )
        with self._lock:
            rows = self._connection.execute(query, parameters).fetchall()
        events: list[RuntimeEvent] = []
        for row in rows:
            payload = json.loads(_required_text(row, "payload_json"))
            if not isinstance(payload, dict) or not all(isinstance(key, str) for key in payload):
                raise TypeError("persisted Runtime Event payload must be a JSON object")
            events.append(
                RuntimeEvent(
                    sequence=_required_int(row, "sequence"),
                    kind=_required_text(row, "kind"),
                    aggregate_id=_required_text(row, "aggregate_id"),
                    payload=payload,
                    created_at=_required_text(row, "created_at"),
                )
            )
        return events

    def prune_runtime_events(self, *, before_sequence: int, limit: int) -> int:
        """Delete one bounded prefix and append an audit event after the removed range."""
        if before_sequence <= 0 or limit <= 0:
            raise ValueError("event prune sequence and limit must be positive")
        with self._transaction():
            rows = self._connection.execute(
                """SELECT sequence FROM runtime_events WHERE sequence < ?
                   ORDER BY sequence LIMIT ?""",
                (before_sequence, limit),
            ).fetchall()
            sequences = [_required_int(row, "sequence") for row in rows]
            if not sequences:
                return 0
            self._connection.executemany(
                "DELETE FROM runtime_events WHERE sequence = ?",
                ((sequence,) for sequence in sequences),
            )
            self._event(
                "runtime_events.pruned",
                "runtime_events",
                {
                    "deleted_count": len(sequences),
                    "first_deleted_sequence": sequences[0],
                    "last_deleted_sequence": sequences[-1],
                    "before_sequence": before_sequence,
                },
            )
            return len(sequences)

    def summarize_runtime_metrics(self) -> RuntimeMetrics:
        """Return bounded aggregate counters without scanning events through the API."""
        with self._lock:
            latest = self._connection.execute(
                "SELECT COALESCE(MAX(sequence), 0) AS value FROM runtime_events"
            ).fetchone()
            event_rows = self._connection.execute(
                "SELECT kind, COUNT(*) AS count FROM runtime_events GROUP BY kind ORDER BY kind"
            ).fetchall()
            task_rows = self._connection.execute(
                """SELECT status, COUNT(*) AS count FROM campaign_tasks
                   GROUP BY status ORDER BY status"""
            ).fetchall()
        if latest is None:
            raise AssertionError("SQLite aggregate did not return a row")
        return RuntimeMetrics(
            latest_event_sequence=_required_int(latest, "value"),
            event_counts=tuple(
                (_required_text(row, "kind"), _required_int(row, "count")) for row in event_rows
            ),
            campaign_task_counts=tuple(
                (_required_text(row, "status"), _required_int(row, "count")) for row in task_rows
            ),
        )

    def insert_campaign(self, campaign: Campaign) -> None:
        """Insert a new Campaign and its creation event."""
        with self._transaction():
            self._connection.execute(
                """INSERT INTO campaigns(
                       id, operator, hardware_target, evaluation_contract_digest,
                       agent_problem_digest, created_at, status,
                       problem_generalization_model, evolver_commit
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    campaign.id,
                    campaign.operator,
                    campaign.hardware_target,
                    campaign.evaluation_contract_digest,
                    campaign.agent_problem_digest,
                    campaign.created_at,
                    campaign.status,
                    campaign.problem_generalization_model,
                    campaign.evolver_commit,
                ),
            )
            self._event("campaign.created", campaign.id)

    def get_campaign(self, campaign_id: CampaignId) -> Campaign:
        """Load a Campaign or fail if it does not exist."""
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM campaigns WHERE id = ?", (campaign_id,)
            ).fetchone()
        if row is None:
            raise KeyError(f"Campaign not found: {campaign_id}")
        return Campaign(
            id=parse_campaign_id(_required_text(row, "id")),
            operator=_required_text(row, "operator"),
            hardware_target=_required_text(row, "hardware_target"),
            evaluation_contract_digest=parse_artifact_digest(
                _required_text(row, "evaluation_contract_digest")
            ),
            created_at=_required_text(row, "created_at"),
            status=CampaignStatus(_required_text(row, "status")),
            agent_problem_digest=parse_artifact_digest(_required_text(row, "agent_problem_digest")),
            problem_generalization_model=_optional_text(row, "problem_generalization_model"),
            evolver_commit=_optional_text(row, "evolver_commit"),
        )

    def ensure_campaign_evolver_commit(
        self,
        campaign_id: CampaignId,
        commit: str,
    ) -> Campaign:
        """Bind one migrated Campaign once or reject Evolver commit drift."""
        if len(commit) not in {40, 64} or any(
            character not in "0123456789abcdef" for character in commit
        ):
            raise ValueError("Campaign Evolver commit must be a full lowercase SHA")
        with self._transaction():
            row = self._connection.execute(
                "SELECT evolver_commit FROM campaigns WHERE id = ?",
                (campaign_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"Campaign not found: {campaign_id}")
            existing = _optional_text(row, "evolver_commit")
            if existing is None:
                self._connection.execute(
                    "UPDATE campaigns SET evolver_commit = ? WHERE id = ?",
                    (commit, campaign_id),
                )
                self._event(
                    "campaign.evolver_commit_bound",
                    campaign_id,
                    {"evolver_commit": commit},
                )
            elif existing != commit:
                raise InvalidTransitionError(
                    f"Campaign {campaign_id} freezes Evolver commit {existing}; "
                    f"configured commit is {commit}"
                )
        return self.get_campaign(campaign_id)

    def list_campaign_lineages(self, campaign_id: CampaignId) -> list[Lineage]:
        """Return all registered DSL lineages in deterministic DSL/id order."""
        self.get_campaign(campaign_id)
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM lineages WHERE campaign_id = ? ORDER BY dsl, id",
                (campaign_id,),
            ).fetchall()
        return [self._map_lineage(row) for row in rows]

    def complete_campaign(self, campaign_id: CampaignId, target_epoch: int) -> Campaign:
        """Mark a quiescent Campaign and every target-complete lineage terminal."""
        if target_epoch <= 0:
            raise ValueError("Campaign completion target must be positive")
        with self._transaction():
            campaign = self._connection.execute(
                "SELECT status FROM campaigns WHERE id = ?", (campaign_id,)
            ).fetchone()
            if campaign is None:
                raise KeyError(f"Campaign not found: {campaign_id}")
            status = CampaignStatus(_required_text(campaign, "status"))
            if status is CampaignStatus.COMPLETED:
                return self.get_campaign(campaign_id)
            if status is not CampaignStatus.ACTIVE:
                raise InvalidTransitionError(f"Campaign {campaign_id} is {status}")
            rows = self._connection.execute(
                "SELECT status, next_epoch_number FROM lineages WHERE campaign_id = ?",
                (campaign_id,),
            ).fetchall()
            if not rows:
                raise InvalidTransitionError("Campaign has no registered lineages")
            if any(
                LineageStatus(_required_text(row, "status")) is not LineageStatus.READY
                or _required_int(row, "next_epoch_number") <= target_epoch
                for row in rows
            ):
                raise InvalidTransitionError(
                    "Campaign completion requires quiescent lineages past the target"
                )
            self._connection.execute(
                "UPDATE lineages SET status = 'completed' WHERE campaign_id = ?",
                (campaign_id,),
            )
            self._connection.execute(
                "UPDATE campaigns SET status = 'completed' WHERE id = ?", (campaign_id,)
            )
            self._event("campaign.completed", campaign_id, {"target_epoch": target_epoch})
        return self.get_campaign(campaign_id)

    def cancel_campaign(self, campaign_id: CampaignId) -> Campaign:
        """Cancel a Campaign only when no lineage has in-flight or pending handoff work."""
        with self._transaction():
            campaign = self._connection.execute(
                "SELECT status FROM campaigns WHERE id = ?", (campaign_id,)
            ).fetchone()
            if campaign is None:
                raise KeyError(f"Campaign not found: {campaign_id}")
            status = CampaignStatus(_required_text(campaign, "status"))
            if status is CampaignStatus.CANCELLED:
                return self.get_campaign(campaign_id)
            if status is not CampaignStatus.ACTIVE:
                raise InvalidTransitionError(f"Campaign {campaign_id} is {status}")
            busy = self._connection.execute(
                """SELECT 1 FROM lineages WHERE campaign_id = ?
                   AND status NOT IN ('ready', 'completed', 'failed', 'cancelled') LIMIT 1""",
                (campaign_id,),
            ).fetchone()
            if busy is not None:
                raise InvalidTransitionError(
                    "Campaign cancellation requires all lineages to be quiescent"
                )
            self._connection.execute(
                """UPDATE lineages SET status = 'cancelled'
                   WHERE campaign_id = ? AND status = 'ready'""",
                (campaign_id,),
            )
            self._connection.execute(
                "UPDATE campaigns SET status = 'cancelled' WHERE id = ?", (campaign_id,)
            )
            self._event("campaign.cancelled", campaign_id)
        return self.get_campaign(campaign_id)

    def enqueue_campaign_task(self, task: CampaignTask) -> CampaignTask:
        """Insert one idempotent Campaign scheduling request."""
        if not task.creation_key.strip():
            raise ValueError("Campaign task creation key cannot be empty")
        if len(task.creation_key) > 256:
            raise ValueError("Campaign task creation key exceeds 256 characters")
        if task.status is not CampaignTaskStatus.QUEUED or task.attempt_count != 0:
            raise ValueError("a new Campaign task must be queued and unattempted")
        if task.lease_owner is not None or task.lease_expires_at is not None:
            raise ValueError("a new Campaign task cannot hold a lease")
        with self._transaction():
            campaign = self._connection.execute(
                "SELECT 1 FROM campaigns WHERE id = ?", (task.campaign_id,)
            ).fetchone()
            if campaign is None:
                raise KeyError(f"Campaign not found: {task.campaign_id}")
            existing_row = self._connection.execute(
                "SELECT * FROM campaign_tasks WHERE creation_key = ?",
                (task.creation_key,),
            ).fetchone()
            if existing_row is not None:
                existing = self._map_campaign_task(existing_row)
                if (
                    existing.campaign_id != task.campaign_id
                    or existing.target_epoch_number != task.target_epoch_number
                    or existing.finalize != task.finalize
                ):
                    raise InvalidTransitionError(
                        "Campaign task creation key was reused for different inputs"
                    )
                return existing
            self._connection.execute(
                """INSERT INTO campaign_tasks(
                       id, creation_key, campaign_id, target_epoch_number, finalize,
                       status, attempt_count, lease_owner, lease_expires_at, last_error,
                       created_at, started_at, completed_at
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    task.id,
                    task.creation_key,
                    task.campaign_id,
                    task.target_epoch_number,
                    int(task.finalize),
                    task.status,
                    task.attempt_count,
                    task.lease_owner,
                    task.lease_expires_at,
                    task.last_error,
                    task.created_at,
                    task.started_at,
                    task.completed_at,
                ),
            )
            self._event(
                "campaign_task.queued",
                task.id,
                {
                    "campaign_id": task.campaign_id,
                    "target_epoch_number": task.target_epoch_number,
                    "finalize": task.finalize,
                },
            )
        return task

    def get_campaign_task(self, task_id: CampaignTaskId) -> CampaignTask:
        """Load one durable Campaign task."""
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM campaign_tasks WHERE id = ?", (task_id,)
            ).fetchone()
        if row is None:
            raise KeyError(f"Campaign task not found: {task_id}")
        return self._map_campaign_task(row)

    def claim_campaign_task(
        self,
        owner: str,
        *,
        now: str,
        lease_expires_at: str,
    ) -> CampaignTask | None:
        """Claim ready work or finish an abandoned cooperative cancellation."""
        if not owner:
            raise ValueError("Campaign task owner cannot be empty")
        with self._transaction():
            row = self._connection.execute(
                """SELECT * FROM campaign_tasks
                   WHERE status = 'queued'
                      OR (status IN ('running', 'cancelling') AND lease_expires_at <= ?)
                   ORDER BY created_at, id LIMIT 1""",
                (now,),
            ).fetchone()
            if row is None:
                return None
            task_id = parse_campaign_task_id(_required_text(row, "id"))
            status = CampaignTaskStatus(_required_text(row, "status"))
            if status is CampaignTaskStatus.CANCELLING:
                self._connection.execute(
                    """UPDATE campaign_tasks SET status = 'cancelled', lease_owner = NULL,
                       lease_expires_at = NULL, completed_at = ? WHERE id = ?""",
                    (now, task_id),
                )
                self._event("campaign_task.cancelled", task_id, {"reason": "worker_lost"})
                cancelled = self._connection.execute(
                    "SELECT * FROM campaign_tasks WHERE id = ?", (task_id,)
                ).fetchone()
                if cancelled is None:
                    raise AssertionError("cancelled Campaign task disappeared")
                return self._map_campaign_task(cancelled)
            reclaimed = status is CampaignTaskStatus.RUNNING
            self._connection.execute(
                """UPDATE campaign_tasks SET status = 'running', attempt_count = attempt_count + 1,
                   lease_owner = ?, lease_expires_at = ?, last_error = NULL,
                   started_at = COALESCE(started_at, ?)
                   WHERE id = ?""",
                (owner, lease_expires_at, now, task_id),
            )
            self._event(
                "campaign_task.reclaimed" if reclaimed else "campaign_task.started",
                task_id,
                {"owner": owner},
            )
            claimed = self._connection.execute(
                "SELECT * FROM campaign_tasks WHERE id = ?", (task_id,)
            ).fetchone()
            if claimed is None:
                raise AssertionError("claimed Campaign task disappeared")
            return self._map_campaign_task(claimed)

    def renew_campaign_task(
        self,
        task_id: CampaignTaskId,
        owner: str,
        *,
        lease_expires_at: str,
    ) -> bool:
        """Renew one owned lease and report a cooperative cancellation request."""
        with self._transaction():
            changed = self._connection.execute(
                """UPDATE campaign_tasks SET lease_expires_at = ?
                   WHERE id = ? AND status IN ('running', 'cancelling')
                     AND lease_owner = ?""",
                (lease_expires_at, task_id, owner),
            ).rowcount
            if changed != 1:
                raise InvalidTransitionError("Campaign task lease was superseded")
            row = self._connection.execute(
                "SELECT status FROM campaign_tasks WHERE id = ?", (task_id,)
            ).fetchone()
            if row is None:
                raise AssertionError("renewed Campaign task disappeared")
            return CampaignTaskStatus(_required_text(row, "status")) is (
                CampaignTaskStatus.CANCELLING
            )

    def complete_campaign_task(
        self,
        task_id: CampaignTaskId,
        owner: str,
    ) -> CampaignTask:
        """Complete one task held by the current worker."""
        with self._transaction():
            row = self._connection.execute(
                "SELECT status FROM campaign_tasks WHERE id = ? AND lease_owner = ?",
                (task_id, owner),
            ).fetchone()
            if row is None:
                raise InvalidTransitionError("Campaign task cannot complete")
            if CampaignTaskStatus(_required_text(row, "status")) is (CampaignTaskStatus.CANCELLING):
                self._connection.execute(
                    """UPDATE campaign_tasks SET status = 'cancelled', lease_owner = NULL,
                       lease_expires_at = NULL, completed_at = ? WHERE id = ?""",
                    (self._clock(), task_id),
                )
                self._event("campaign_task.cancelled", task_id, {"reason": "requested"})
                return self.get_campaign_task(task_id)
            changed = self._connection.execute(
                """UPDATE campaign_tasks SET status = 'completed', lease_owner = NULL,
                   lease_expires_at = NULL, completed_at = ?
                   WHERE id = ? AND status = 'running' AND lease_owner = ?""",
                (self._clock(), task_id, owner),
            ).rowcount
            if changed != 1:
                raise InvalidTransitionError("Campaign task cannot complete")
            self._event("campaign_task.completed", task_id)
        return self.get_campaign_task(task_id)

    def fail_campaign_task(
        self,
        task_id: CampaignTaskId,
        owner: str,
        *,
        error: str,
    ) -> CampaignTask:
        """Record one inspected terminal task execution failure."""
        if not error:
            raise ValueError("Campaign task failure cannot be empty")
        with self._transaction():
            row = self._connection.execute(
                "SELECT status FROM campaign_tasks WHERE id = ? AND lease_owner = ?",
                (task_id, owner),
            ).fetchone()
            if row is not None and CampaignTaskStatus(_required_text(row, "status")) is (
                CampaignTaskStatus.CANCELLING
            ):
                self._connection.execute(
                    """UPDATE campaign_tasks SET status = 'cancelled', lease_owner = NULL,
                       lease_expires_at = NULL, completed_at = ? WHERE id = ?""",
                    (self._clock(), task_id),
                )
                self._event("campaign_task.cancelled", task_id, {"reason": "requested"})
                return self.get_campaign_task(task_id)
            changed = self._connection.execute(
                """UPDATE campaign_tasks SET status = 'failed', lease_owner = NULL,
                   lease_expires_at = NULL, last_error = ?, completed_at = ?
                   WHERE id = ? AND status = 'running' AND lease_owner = ?""",
                (error, self._clock(), task_id, owner),
            ).rowcount
            if changed != 1:
                raise InvalidTransitionError("Campaign task cannot fail")
            self._event("campaign_task.failed", task_id, {"error": error})
        return self.get_campaign_task(task_id)

    def cancel_campaign_task(self, task_id: CampaignTaskId) -> CampaignTask:
        """Cancel queued work or request a stop after the current bounded operation."""
        with self._transaction():
            row = self._connection.execute(
                "SELECT status FROM campaign_tasks WHERE id = ?", (task_id,)
            ).fetchone()
            if row is None:
                raise KeyError(f"Campaign task not found: {task_id}")
            status = CampaignTaskStatus(_required_text(row, "status"))
            if status is CampaignTaskStatus.CANCELLED:
                return self.get_campaign_task(task_id)
            if status is CampaignTaskStatus.CANCELLING:
                return self.get_campaign_task(task_id)
            if status is CampaignTaskStatus.QUEUED:
                self._connection.execute(
                    """UPDATE campaign_tasks SET status = 'cancelled', completed_at = ?
                       WHERE id = ?""",
                    (self._clock(), task_id),
                )
                self._event("campaign_task.cancelled", task_id, {"reason": "queued"})
            elif status is CampaignTaskStatus.RUNNING:
                self._connection.execute(
                    "UPDATE campaign_tasks SET status = 'cancelling' WHERE id = ?",
                    (task_id,),
                )
                self._event("campaign_task.cancellation_requested", task_id)
            else:
                raise InvalidTransitionError("Campaign task cannot be cancelled")
        return self.get_campaign_task(task_id)

    def requeue_campaign_task(self, task_id: CampaignTaskId) -> CampaignTask:
        """Return one inspected failed task to the queue without changing its identity."""
        with self._transaction():
            changed = self._connection.execute(
                """UPDATE campaign_tasks SET status = 'queued', last_error = NULL,
                   completed_at = NULL WHERE id = ? AND status = 'failed'""",
                (task_id,),
            ).rowcount
            if changed != 1:
                row = self._connection.execute(
                    "SELECT 1 FROM campaign_tasks WHERE id = ?", (task_id,)
                ).fetchone()
                if row is None:
                    raise KeyError(f"Campaign task not found: {task_id}")
                raise InvalidTransitionError("only a failed Campaign task can be requeued")
            self._event("campaign_task.requeued", task_id)
        return self.get_campaign_task(task_id)

    def register_kernel_agent_revision(self, revision: KernelAgentRevision) -> KernelAgentRevision:
        """Register a revision idempotently by its creation key."""
        with self._transaction():
            existing_row = self._connection.execute(
                "SELECT * FROM kernel_agent_revisions WHERE creation_key = ?",
                (revision.creation_key,),
            ).fetchone()
            if existing_row is not None:
                existing = self._map_kernel_agent_revision(existing_row)
                if self._agent_content(existing) != self._agent_content(revision):
                    raise RuntimeError(
                        f"creation key {revision.creation_key!r} resolved to different content"
                    )
                return existing
            self._connection.execute(
                """INSERT INTO kernel_agent_revisions (
                       id, parent_id, creation_key, dsl, optimizer_digest,
                       created_by, created_at, source_provenance_digest,
                       evolution_trace_digest, runtime_state_digest
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    revision.id,
                    revision.parent_id,
                    revision.creation_key,
                    revision.dsl,
                    revision.optimizer_digest,
                    revision.created_by,
                    revision.created_at,
                    revision.source_provenance_digest,
                    revision.evolution_trace_digest,
                    revision.runtime_state_digest,
                ),
            )
            self._event(
                "kernel_agent_revision.registered",
                revision.id,
                {"parent_id": revision.parent_id, "creation_key": revision.creation_key},
            )
        return revision

    @staticmethod
    def _agent_content(revision: KernelAgentRevision) -> tuple[object, ...]:
        return (
            revision.parent_id,
            revision.dsl,
            revision.optimizer_digest,
            revision.created_by,
            revision.source_provenance_digest,
            revision.evolution_trace_digest,
            revision.runtime_state_digest,
        )

    def get_kernel_agent_revision(self, revision_id: KernelAgentRevisionId) -> KernelAgentRevision:
        """Load a Kernel Agent revision or fail if it does not exist."""
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM kernel_agent_revisions WHERE id = ?", (revision_id,)
            ).fetchone()
        if row is None:
            raise KeyError(f"Kernel Agent revision not found: {revision_id}")
        return self._map_kernel_agent_revision(row)

    def find_kernel_agent_revision_by_creation_key(
        self, creation_key: str
    ) -> KernelAgentRevision | None:
        """Return the revision created for an idempotent operation, if present."""
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM kernel_agent_revisions WHERE creation_key = ?", (creation_key,)
            ).fetchone()
        return None if row is None else self._map_kernel_agent_revision(row)

    def _link_agent_version(
        self,
        lineage_id: LineageId,
        revision: KernelAgentRevision,
        *,
        expected_number: int | None,
        introduced_epoch_id: EpochId | None,
    ) -> int:
        existing = self._connection.execute(
            """SELECT lineage_id, revision_number, introduced_epoch_id
               FROM lineage_agent_versions WHERE kernel_agent_revision_id = ?""",
            (revision.id,),
        ).fetchone()
        if existing is not None:
            existing_lineage = parse_lineage_id(_required_text(existing, "lineage_id"))
            number = _required_int(existing, "revision_number")
            existing_epoch = _optional_text(existing, "introduced_epoch_id")
            if (
                existing_lineage != lineage_id
                or (expected_number is not None and number != expected_number)
                or existing_epoch != introduced_epoch_id
            ):
                raise InvalidTransitionError("Kernel Agent has conflicting lineage version")
            return number

        if revision.parent_id is not None:
            parent = self._connection.execute(
                """SELECT lineage_id FROM lineage_agent_versions
                   WHERE kernel_agent_revision_id = ?""",
                (revision.parent_id,),
            ).fetchone()
            if parent is None or _required_text(parent, "lineage_id") != lineage_id:
                raise InvalidTransitionError(
                    "Kernel Agent parent is absent from the target lineage version history"
                )
        row = self._connection.execute(
            """SELECT COALESCE(MAX(revision_number), -1) + 1 AS next_number
               FROM lineage_agent_versions WHERE lineage_id = ?""",
            (lineage_id,),
        ).fetchone()
        if row is None:
            raise RuntimeError("SQLite did not return the next Kernel Agent revision number")
        next_number = _required_int(row, "next_number")
        number = next_number if expected_number is None else expected_number
        if expected_number is not None and expected_number != next_number:
            raise InvalidTransitionError(
                "Kernel Agent revision number is not the next lineage version"
            )
        self._connection.execute(
            "INSERT INTO lineage_agent_versions VALUES (?, ?, ?, ?, ?)",
            (
                revision.id,
                lineage_id,
                number,
                revision.created_at,
                introduced_epoch_id,
            ),
        )
        return number

    def list_lineage_agent_revisions(self, lineage_id: LineageId) -> list[KernelAgentCatalogEntry]:
        """Return every versioned Agent revision attached to one Lineage."""
        lineage = self.get_lineage(lineage_id)
        with self._lock:
            rows = self._connection.execute(
                """SELECT * FROM lineage_agent_versions
                   WHERE lineage_id = ? ORDER BY revision_number""",
                (lineage_id,),
            ).fetchall()
        numbers = {
            parse_kernel_agent_revision_id(
                _required_text(row, "kernel_agent_revision_id")
            ): _required_int(row, "revision_number")
            for row in rows
        }
        entries: list[KernelAgentCatalogEntry] = []
        for row in rows:
            revision_id = parse_kernel_agent_revision_id(
                _required_text(row, "kernel_agent_revision_id")
            )
            revision = self.get_kernel_agent_revision(revision_id)
            parent_number = None
            if revision.parent_id is not None:
                try:
                    parent_number = numbers[revision.parent_id]
                except KeyError as error:
                    raise RuntimeError(
                        f"Kernel Agent {revision.id} parent has no lineage version"
                    ) from error
            introduced_value = _optional_text(row, "introduced_epoch_id")
            introduced_epoch = (
                None
                if introduced_value is None
                else self.get_epoch(parse_epoch_id(introduced_value))
            )
            with self._lock:
                participation_row = self._connection.execute(
                    """SELECT e.* FROM epoch_challengers ec
                       JOIN epochs e ON e.id = ec.epoch_id
                       WHERE e.lineage_id = ? AND ec.kernel_agent_revision_id = ?
                       ORDER BY e.number DESC LIMIT 1""",
                    (lineage_id, revision.id),
                ).fetchone()
            participation = (
                None if participation_row is None else self._map_epoch(participation_row)
            )
            disposition_epoch = participation or introduced_epoch
            if disposition_epoch is None:
                disposition = "baseline"
            elif disposition_epoch.status is EpochStatus.COMPLETED:
                disposition = (
                    "promoted"
                    if disposition_epoch.winner_kernel_agent_revision_id == revision.id
                    else "rejected"
                )
            elif disposition_epoch.status is EpochStatus.FAILED:
                disposition = "failed"
            else:
                disposition = "challenger"
            entries.append(
                KernelAgentCatalogEntry(
                    revision=revision,
                    revision_number=_required_int(row, "revision_number"),
                    parent_revision_number=parent_number,
                    campaign_id=lineage.campaign_id,
                    lineage_id=lineage.id,
                    introduced_epoch_id=(None if introduced_epoch is None else introduced_epoch.id),
                    introduced_epoch_number=(
                        None if introduced_epoch is None else introduced_epoch.number
                    ),
                    disposition=disposition,
                    active=lineage.active_kernel_agent_revision_id == revision.id,
                )
            )
        return entries

    def list_campaign_agent_revisions(
        self, campaign_id: CampaignId
    ) -> list[KernelAgentCatalogEntry]:
        """Return versioned Agent histories for every Campaign Lineage."""
        self.get_campaign(campaign_id)
        return [
            entry
            for lineage in self.list_campaign_lineages(campaign_id)
            for entry in self.list_lineage_agent_revisions(lineage.id)
        ]

    def find_kernel_agent_lineage(self, revision_id: KernelAgentRevisionId) -> Lineage:
        """Resolve the unique Lineage containing one versioned Agent revision."""
        with self._lock:
            row = self._connection.execute(
                """SELECT lineages.* FROM lineages
                   JOIN lineage_agent_versions
                     ON lineage_agent_versions.lineage_id = lineages.id
                   WHERE lineage_agent_versions.kernel_agent_revision_id = ?""",
                (revision_id,),
            ).fetchone()
        if row is None:
            raise KeyError(f"Kernel Agent lineage not found: {revision_id}")
        return self._map_lineage(row)

    @staticmethod
    def _map_kernel_agent_revision(row: sqlite3.Row) -> KernelAgentRevision:
        parent = _optional_text(row, "parent_id")
        return KernelAgentRevision(
            id=parse_kernel_agent_revision_id(_required_text(row, "id")),
            parent_id=None if parent is None else parse_kernel_agent_revision_id(parent),
            creation_key=_required_text(row, "creation_key"),
            dsl=Dsl(_required_text(row, "dsl")),
            optimizer_digest=parse_artifact_digest(_required_text(row, "optimizer_digest")),
            created_by=_required_text(row, "created_by"),
            created_at=_required_text(row, "created_at"),
            source_provenance_digest=(
                None
                if (source := _optional_text(row, "source_provenance_digest")) is None
                else parse_artifact_digest(source)
            ),
            evolution_trace_digest=(
                None
                if (trace := _optional_text(row, "evolution_trace_digest")) is None
                else parse_artifact_digest(trace)
            ),
            runtime_state_digest=(
                None
                if (state := _optional_text(row, "runtime_state_digest")) is None
                else parse_artifact_digest(state)
            ),
        )

    def register_kernel_revision(self, revision: KernelRevision) -> KernelRevision:
        """Register a Kernel idempotently by identity and its producing Attempt."""
        with self._transaction():
            identity_row = self._connection.execute(
                "SELECT * FROM kernel_revisions WHERE id = ?",
                (revision.id,),
            ).fetchone()
            if identity_row is not None:
                existing = self._map_kernel_revision(identity_row)
                if self._kernel_content(existing) != self._kernel_content(revision):
                    raise RuntimeError(f"Kernel identity {revision.id} resolved differently")
                if existing.produced_by_attempt_id is not None:
                    self._link_attempt_kernel_version(existing)
                return existing
            if revision.produced_by_attempt_id is not None:
                existing_row = self._connection.execute(
                    "SELECT * FROM kernel_revisions WHERE produced_by_attempt_id = ?",
                    (revision.produced_by_attempt_id,),
                ).fetchone()
                if existing_row is not None:
                    existing = self._map_kernel_revision(existing_row)
                    if self._kernel_content(existing) != self._kernel_content(revision):
                        raise RuntimeError(
                            f"Attempt {revision.produced_by_attempt_id} produced different Kernels"
                        )
                    self._link_attempt_kernel_version(existing)
                    return existing
            self._connection.execute(
                "INSERT INTO kernel_revisions VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    revision.id,
                    revision.parent_id,
                    revision.artifact_digest,
                    revision.produced_by_attempt_id,
                    int(revision.evaluation.correct),
                    revision.evaluation.latency_us,
                    revision.evaluation.gateway_result_digest,
                    revision.created_at,
                ),
            )
            self._event(
                "kernel_revision.registered",
                revision.id,
                {
                    "correct": revision.evaluation.correct,
                    "latency_us": revision.evaluation.latency_us,
                },
            )
            if revision.produced_by_attempt_id is not None:
                self._link_attempt_kernel_version(revision)
        return revision

    def finalize_kernel_revision_evaluation(
        self,
        revision_id: KernelRevisionId,
        evaluation: KernelEvaluation,
    ) -> KernelRevision:
        """Replace a running Attempt's provisional Eval with its ABBA authority."""
        with self._transaction():
            row = self._connection.execute(
                """SELECT k.*, a.status AS attempt_status
                   FROM kernel_revisions k
                   JOIN attempts a ON a.id = k.produced_by_attempt_id
                   WHERE k.id = ?""",
                (revision_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"Attempt Kernel revision not found: {revision_id}")
            revision = self._map_kernel_revision(row)
            if revision.evaluation == evaluation:
                return revision
            if _required_text(row, "attempt_status") != AttemptStatus.RUNNING.value:
                raise InvalidTransitionError(
                    f"Kernel revision {revision_id} evaluation is already immutable"
                )
            cursor = self._connection.execute(
                """UPDATE kernel_revisions SET correct = ?, latency_us = ?,
                   gateway_result_digest = ? WHERE id = ?""",
                (
                    int(evaluation.correct),
                    evaluation.latency_us,
                    evaluation.gateway_result_digest,
                    revision_id,
                ),
            )
            if cursor.rowcount != 1:
                raise InvalidTransitionError(
                    f"Kernel revision {revision_id} evaluation finalization was lost"
                )
            finalized = replace(revision, evaluation=evaluation)
            self._event(
                "kernel_revision.evaluation_finalized",
                revision_id,
                {
                    "correct": evaluation.correct,
                    "latency_us": evaluation.latency_us,
                    "gateway_result_digest": evaluation.gateway_result_digest,
                },
            )
            return finalized

    def _link_attempt_kernel_version(self, revision: KernelRevision) -> int:
        attempt_id = revision.produced_by_attempt_id
        if attempt_id is None:
            raise ValueError("Attempt Kernel version linking requires a producing Attempt")
        row = self._connection.execute(
            """SELECT epochs.lineage_id
               FROM attempts JOIN epochs ON epochs.id = attempts.epoch_id
               WHERE attempts.id = ?""",
            (attempt_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"Attempt not found: {attempt_id}")
        lineage_id = parse_lineage_id(_required_text(row, "lineage_id"))
        return self._link_kernel_version(lineage_id, revision, expected_number=None)

    def _link_kernel_version(
        self,
        lineage_id: LineageId,
        revision: KernelRevision,
        *,
        expected_number: int | None,
    ) -> int:
        existing = self._connection.execute(
            """SELECT lineage_id, revision_number FROM lineage_kernel_versions
               WHERE kernel_revision_id = ?""",
            (revision.id,),
        ).fetchone()
        if existing is not None:
            existing_lineage = parse_lineage_id(_required_text(existing, "lineage_id"))
            number = _required_int(existing, "revision_number")
            if existing_lineage != lineage_id or (
                expected_number is not None and number != expected_number
            ):
                raise InvalidTransitionError("Kernel revision has conflicting lineage version")
            return number

        if revision.parent_id is not None:
            parent = self._connection.execute(
                """SELECT lineage_id FROM lineage_kernel_versions
                   WHERE kernel_revision_id = ?""",
                (revision.parent_id,),
            ).fetchone()
            if parent is None or _required_text(parent, "lineage_id") != lineage_id:
                raise InvalidTransitionError(
                    "Kernel parent is absent from the target lineage version history"
                )
        row = self._connection.execute(
            """SELECT COALESCE(MAX(revision_number), -1) + 1 AS next_number
               FROM lineage_kernel_versions WHERE lineage_id = ?""",
            (lineage_id,),
        ).fetchone()
        if row is None:
            raise RuntimeError("SQLite did not return the next Kernel revision number")
        next_number = _required_int(row, "next_number")
        number = next_number if expected_number is None else expected_number
        if expected_number is not None and expected_number != next_number:
            raise InvalidTransitionError("Kernel revision number is not the next lineage version")
        self._connection.execute(
            "INSERT INTO lineage_kernel_versions VALUES (?, ?, ?, ?)",
            (revision.id, lineage_id, number, revision.created_at),
        )
        return number

    @staticmethod
    def _kernel_content(revision: KernelRevision) -> tuple[object, ...]:
        return (
            revision.parent_id,
            revision.artifact_digest,
            revision.produced_by_attempt_id,
            revision.evaluation,
        )

    def get_kernel_revision(self, revision_id: KernelRevisionId) -> KernelRevision:
        """Load a Kernel revision or fail if it does not exist."""
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM kernel_revisions WHERE id = ?", (revision_id,)
            ).fetchone()
        if row is None:
            raise KeyError(f"Kernel revision not found: {revision_id}")
        return self._map_kernel_revision(row)

    def find_kernel_revision_by_attempt(self, attempt_id: AttemptId) -> KernelRevision | None:
        """Return the Kernel already registered for an Attempt, if present."""
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM kernel_revisions WHERE produced_by_attempt_id = ?", (attempt_id,)
            ).fetchone()
        return None if row is None else self._map_kernel_revision(row)

    def list_lineage_kernels(self, lineage_id: LineageId) -> list[KernelCatalogEntry]:
        """Return the baseline and every terminal Attempt Kernel in lineage order."""
        lineage = self.get_lineage(lineage_id)
        epochs = self.list_epochs(lineage_id)
        entries: dict[KernelRevisionId, KernelCatalogEntry] = {}

        with self._lock:
            version_rows = self._connection.execute(
                """SELECT kernel_revision_id, revision_number
                   FROM lineage_kernel_versions
                   WHERE lineage_id = ? ORDER BY revision_number""",
                (lineage_id,),
            ).fetchall()
        version_numbers = {
            parse_kernel_revision_id(_required_text(row, "kernel_revision_id")): _required_int(
                row, "revision_number"
            )
            for row in version_rows
        }
        with self._lock:
            agent_version_rows = self._connection.execute(
                """SELECT kernel_agent_revision_id, revision_number
                   FROM lineage_agent_versions WHERE lineage_id = ?""",
                (lineage_id,),
            ).fetchall()
        agent_version_numbers = {
            parse_kernel_agent_revision_id(
                _required_text(row, "kernel_agent_revision_id")
            ): _required_int(row, "revision_number")
            for row in agent_version_rows
        }

        def agent_revision_number(revision_id: KernelAgentRevisionId) -> int:
            try:
                return agent_version_numbers[revision_id]
            except KeyError as error:
                raise RuntimeError(
                    f"Kernel Agent {revision_id} has no durable lineage version"
                ) from error

        def version_context(
            revision: KernelRevision,
        ) -> tuple[int, int | None, float | None]:
            try:
                revision_number = version_numbers[revision.id]
            except KeyError as error:
                raise RuntimeError(
                    f"Kernel {revision.id} has no durable lineage version"
                ) from error
            if revision.parent_id is None:
                return revision_number, None, None
            try:
                parent_number = version_numbers[revision.parent_id]
            except KeyError as error:
                raise RuntimeError(
                    f"Kernel {revision.id} parent has no durable lineage version"
                ) from error
            parent = self.get_kernel_revision(revision.parent_id)
            parent_latency = parent.evaluation.latency_us
            latency = revision.evaluation.latency_us
            improvement = (
                None
                if parent_latency is None or latency is None
                else ((parent_latency - latency) / parent_latency) * 100
            )
            return revision_number, parent_number, improvement

        def add_baseline(revision_id: KernelRevisionId, agent_id: KernelAgentRevisionId) -> None:
            if revision_id in entries:
                return
            revision = self.get_kernel_revision(revision_id)
            revision_number, parent_number, improvement = version_context(revision)
            entries[revision_id] = KernelCatalogEntry(
                revision=revision,
                revision_number=revision_number,
                parent_revision_number=parent_number,
                improvement_over_parent_percent=improvement,
                campaign_id=lineage.campaign_id,
                lineage_id=lineage.id,
                dsl=lineage.dsl,
                kernel_agent_revision_id=agent_id,
                kernel_agent_revision_number=agent_revision_number(agent_id),
                epoch_id=None,
                epoch_number=None,
                attempt_id=None,
                branch=None,
                challenger_ordinal=None,
                trajectory_ordinal=None,
                attempt_ordinal=None,
                accepted_as_branch_best=False,
            )

        if epochs:
            add_baseline(
                epochs[0].starting_kernel_revision_id,
                epochs[0].active_kernel_agent_revision_id,
            )
        else:
            add_baseline(lineage.best_kernel_revision_id, lineage.active_kernel_agent_revision_id)

        for epoch in epochs:
            if epoch.starting_kernel_revision_id not in entries:
                add_baseline(
                    epoch.starting_kernel_revision_id,
                    epoch.active_kernel_agent_revision_id,
                )
            for attempt in self.list_attempts(epoch.id):
                if attempt.output_kernel_revision_id is None:
                    continue
                revision = self.get_kernel_revision(attempt.output_kernel_revision_id)
                revision_number, parent_number, improvement = version_context(revision)
                entries[attempt.output_kernel_revision_id] = KernelCatalogEntry(
                    revision=revision,
                    revision_number=revision_number,
                    parent_revision_number=parent_number,
                    improvement_over_parent_percent=improvement,
                    campaign_id=lineage.campaign_id,
                    lineage_id=lineage.id,
                    dsl=lineage.dsl,
                    kernel_agent_revision_id=attempt.kernel_agent_revision_id,
                    kernel_agent_revision_number=agent_revision_number(
                        attempt.kernel_agent_revision_id
                    ),
                    epoch_id=epoch.id,
                    epoch_number=epoch.number,
                    attempt_id=attempt.id,
                    branch=attempt.branch,
                    challenger_ordinal=attempt.challenger_ordinal,
                    trajectory_ordinal=attempt.trajectory_ordinal,
                    attempt_ordinal=attempt.ordinal,
                    accepted_as_branch_best=attempt.accepted_as_branch_best,
                )
        if lineage.best_kernel_revision_id not in entries:
            raise RuntimeError("lineage best Kernel is absent from its durable history")
        return sorted(entries.values(), key=lambda entry: entry.revision_number)

    def list_campaign_kernels(self, campaign_id: CampaignId) -> list[KernelCatalogEntry]:
        """Return every selected DSL lineage Kernel in deterministic order."""
        self.get_campaign(campaign_id)
        return [
            entry
            for lineage in self.list_campaign_lineages(campaign_id)
            for entry in self.list_lineage_kernels(lineage.id)
        ]

    def record_kernel_measurement(self, measurement: KernelMeasurement) -> KernelMeasurement:
        """Store one repeated-Evaluate sample idempotently."""
        if measurement.purpose not in {
            KernelMeasurementPurpose.KERNEL_RETENTION,
            KernelMeasurementPurpose.AGENT_PROMOTION,
        }:
            raise ValueError("only comparison measurements are stored separately")
        with self._transaction():
            self.get_kernel_revision(measurement.kernel_revision_id)
            row = self._connection.execute(
                "SELECT * FROM kernel_measurements WHERE id = ?", (measurement.id,)
            ).fetchone()
            if row is not None:
                existing = self._map_kernel_measurement(row)
                if existing != measurement:
                    raise RuntimeError("Kernel measurement ID was reused with different content")
                return existing
            self._connection.execute(
                """INSERT INTO kernel_measurements
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    measurement.id,
                    measurement.kernel_revision_id,
                    measurement.purpose,
                    measurement.repeat,
                    int(measurement.correct),
                    measurement.latency_us,
                    measurement.gateway_result_digest,
                    measurement.agate_job_id,
                    measurement.created_at,
                ),
            )
            self._event(
                "kernel_measurement.recorded",
                measurement.kernel_revision_id,
                {
                    "measurement_id": measurement.id,
                    "purpose": measurement.purpose,
                    "repeat": measurement.repeat,
                    "correct": measurement.correct,
                    "latency_us": measurement.latency_us,
                    "gateway_result_digest": measurement.gateway_result_digest,
                    "agate_job_id": measurement.agate_job_id,
                },
            )
        return measurement

    def list_kernel_measurements(self, revision_id: KernelRevisionId) -> list[KernelMeasurement]:
        """Return durable repeated-Evaluate samples in creation order."""
        self.get_kernel_revision(revision_id)
        with self._lock:
            rows = self._connection.execute(
                """SELECT * FROM kernel_measurements
                   WHERE kernel_revision_id = ? ORDER BY created_at, id""",
                (revision_id,),
            ).fetchall()
        return [self._map_kernel_measurement(row) for row in rows]

    @staticmethod
    def _map_kernel_measurement(row: sqlite3.Row) -> KernelMeasurement:
        digest = _optional_text(row, "gateway_result_digest")
        return KernelMeasurement(
            id=_required_text(row, "id"),
            kernel_revision_id=parse_kernel_revision_id(_required_text(row, "kernel_revision_id")),
            purpose=KernelMeasurementPurpose(_required_text(row, "purpose")),
            repeat=_required_int(row, "repeat"),
            correct=bool(_required_int(row, "correct")),
            latency_us=_optional_float(row, "latency_us"),
            gateway_result_digest=(None if digest is None else parse_artifact_digest(digest)),
            agate_job_id=_optional_text(row, "agate_job_id"),
            created_at=_required_text(row, "created_at"),
        )

    @staticmethod
    def _map_kernel_revision(row: sqlite3.Row) -> KernelRevision:
        parent = _optional_text(row, "parent_id")
        attempt = _optional_text(row, "produced_by_attempt_id")
        return KernelRevision(
            id=parse_kernel_revision_id(_required_text(row, "id")),
            parent_id=None if parent is None else parse_kernel_revision_id(parent),
            artifact_digest=parse_artifact_digest(_required_text(row, "artifact_digest")),
            produced_by_attempt_id=None if attempt is None else parse_attempt_id(attempt),
            evaluation=KernelEvaluation(
                correct=bool(_required_int(row, "correct")),
                latency_us=_optional_float(row, "latency_us"),
                gateway_result_digest=parse_artifact_digest(
                    _required_text(row, "gateway_result_digest")
                ),
            ),
            created_at=_required_text(row, "created_at"),
        )

    def insert_lineage(self, lineage: Lineage) -> None:
        """Insert an initialized DSL lineage idempotently by its stable identity."""
        with self._transaction():
            existing_row = self._connection.execute(
                "SELECT * FROM lineages WHERE id = ?",
                (lineage.id,),
            ).fetchone()
            if existing_row is not None:
                existing = self._map_lineage(existing_row)
                if existing != lineage:
                    raise RuntimeError(f"Lineage identity {lineage.id} resolved differently")
                return
            self._connection.execute(
                """INSERT INTO lineages(
                       id, campaign_id, dsl, hardware_target,
                       active_kernel_agent_revision_id, best_kernel_revision_id,
                       evidence_checkpoint, attempts_per_branch, next_epoch_number, status,
                       challenger_count, challenger_start_epoch, trajectories_per_branch,
                       optimizer_model, evolver_model
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    lineage.id,
                    lineage.campaign_id,
                    lineage.dsl,
                    lineage.hardware_target,
                    lineage.active_kernel_agent_revision_id,
                    lineage.best_kernel_revision_id,
                    lineage.evidence_checkpoint,
                    lineage.attempts_per_trajectory,
                    lineage.next_epoch_number,
                    lineage.status,
                    lineage.challenger_count,
                    lineage.challenger_start_epoch,
                    lineage.trajectories_per_branch,
                    lineage.optimizer_model,
                    lineage.evolver_model,
                ),
            )
            baseline = self.get_kernel_revision(lineage.best_kernel_revision_id)
            self._link_kernel_version(lineage.id, baseline, expected_number=0)
            baseline_agent = self.get_kernel_agent_revision(lineage.active_kernel_agent_revision_id)
            self._link_agent_version(
                lineage.id,
                baseline_agent,
                expected_number=0,
                introduced_epoch_id=None,
            )
            self._event("lineage.created", lineage.id)

    def get_lineage(self, lineage_id: LineageId) -> Lineage:
        """Load a lineage or fail if it does not exist."""
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM lineages WHERE id = ?", (lineage_id,)
            ).fetchone()
        if row is None:
            raise KeyError(f"Lineage not found: {lineage_id}")
        return self._map_lineage(row)

    def find_kernel_lineage(self, kernel_revision_id: KernelRevisionId) -> Lineage:
        """Resolve the unique DSL lineage whose retained history contains a Kernel."""
        with self._lock:
            rows = self._connection.execute(
                """SELECT lineages.* FROM lineages
                   JOIN lineage_kernel_versions
                     ON lineage_kernel_versions.lineage_id = lineages.id
                   WHERE lineage_kernel_versions.kernel_revision_id = ?""",
                (kernel_revision_id,),
            ).fetchall()
        if not rows:
            raise KeyError(f"Kernel lineage not found: {kernel_revision_id}")
        if len(rows) != 1:
            raise InvalidTransitionError(
                f"Kernel {kernel_revision_id} belongs to multiple DSL lineages"
            )
        return self._map_lineage(rows[0])

    @staticmethod
    def _map_lineage(row: sqlite3.Row) -> Lineage:
        return Lineage(
            id=parse_lineage_id(_required_text(row, "id")),
            campaign_id=parse_campaign_id(_required_text(row, "campaign_id")),
            dsl=Dsl(_required_text(row, "dsl")),
            hardware_target=_required_text(row, "hardware_target"),
            active_kernel_agent_revision_id=parse_kernel_agent_revision_id(
                _required_text(row, "active_kernel_agent_revision_id")
            ),
            best_kernel_revision_id=parse_kernel_revision_id(
                _required_text(row, "best_kernel_revision_id")
            ),
            evidence_checkpoint=parse_artifact_digest(_required_text(row, "evidence_checkpoint")),
            challenger_count=_required_int(row, "challenger_count"),
            challenger_start_epoch=_required_int(row, "challenger_start_epoch"),
            trajectories_per_branch=_required_int(row, "trajectories_per_branch"),
            attempts_per_trajectory=_required_int(row, "attempts_per_branch"),
            next_epoch_number=_required_int(row, "next_epoch_number"),
            status=LineageStatus(_required_text(row, "status")),
            optimizer_model=_optional_text(row, "optimizer_model"),
            evolver_model=_optional_text(row, "evolver_model"),
        )

    def advance_lineage_evidence(
        self,
        lineage_id: LineageId,
        expected: ArtifactDigest,
        next_checkpoint: ArtifactDigest,
    ) -> None:
        """Publish the next cumulative checkpoint after a completed epoch."""
        with self._transaction():
            row = self._connection.execute(
                "SELECT next_epoch_number FROM lineages WHERE id = ?",
                (lineage_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"Lineage not found: {lineage_id}")
            cursor = self._connection.execute(
                """UPDATE lineages SET evidence_checkpoint = ?, status = 'ready'
                   WHERE id = ? AND status = 'awaiting_evidence'
                   AND evidence_checkpoint = ?""",
                (next_checkpoint, lineage_id, expected),
            )
            if cursor.rowcount != 1:
                raise InvalidTransitionError(
                    f"Lineage {lineage_id} cannot advance its evidence checkpoint"
                )
            self._event(
                "lineage.evidence_advanced",
                lineage_id,
                {"previous": expected, "next": next_checkpoint},
            )

    def insert_epoch(self, epoch: Epoch) -> None:
        """Atomically create an epoch and mark its lineage running."""
        with self._transaction():
            row = self._connection.execute(
                "SELECT * FROM lineages WHERE id = ?", (epoch.lineage_id,)
            ).fetchone()
            if row is None:
                raise KeyError(f"Lineage not found: {epoch.lineage_id}")
            lineage = self._map_lineage(row)
            if (
                lineage.status is not LineageStatus.READY
                or lineage.next_epoch_number != epoch.number
            ):
                raise InvalidTransitionError(
                    f"Lineage {lineage.id} cannot start epoch {epoch.number}"
                )
            expected_challenger_count = (
                0 if epoch.number < lineage.challenger_start_epoch else lineage.challenger_count
            )
            if (
                epoch.evidence_checkpoint != lineage.evidence_checkpoint
                or epoch.challenger_count != expected_challenger_count
                or epoch.trajectories_per_branch != lineage.trajectories_per_branch
                or epoch.attempts_per_trajectory != lineage.attempts_per_trajectory
            ):
                raise InvalidTransitionError(
                    f"Epoch {epoch.id} inputs disagree with lineage {lineage.id}"
                )
            self._connection.execute(
                """INSERT INTO epochs(
                       id, lineage_id, number, active_kernel_agent_revision_id,
                       challenger_kernel_agent_revision_id, starting_kernel_revision_id,
                       evidence_checkpoint, attempts_per_branch, status,
                       winner_kernel_agent_revision_id, best_kernel_revision_id,
                       failure_reason, created_at, completed_at,
                       challenger_count, trajectories_per_branch
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    epoch.id,
                    epoch.lineage_id,
                    epoch.number,
                    epoch.active_kernel_agent_revision_id,
                    (
                        None
                        if not epoch.challenger_kernel_agent_revision_ids
                        else epoch.challenger_kernel_agent_revision_ids[0]
                    ),
                    epoch.starting_kernel_revision_id,
                    epoch.evidence_checkpoint,
                    epoch.attempts_per_trajectory,
                    epoch.status,
                    epoch.winner_kernel_agent_revision_id,
                    epoch.best_kernel_revision_id,
                    None,
                    epoch.created_at,
                    epoch.completed_at,
                    epoch.challenger_count,
                    epoch.trajectories_per_branch,
                ),
            )
            for challenger_ordinal, revision_id in enumerate(
                epoch.challenger_kernel_agent_revision_ids,
                start=1,
            ):
                revision = self.get_kernel_agent_revision(revision_id)
                if revision.id == epoch.active_kernel_agent_revision_id:
                    proposal_type = ChallengerProposalType.REUSE
                    base_revision_id = revision.id
                else:
                    if revision.parent_id is None:
                        raise InvalidTransitionError(
                            "pre-attached Challenger lacks a revision parent"
                        )
                    base_revision_id = revision.parent_id
                    proposal_type = (
                        ChallengerProposalType.EVOLVED
                        if base_revision_id == epoch.active_kernel_agent_revision_id
                        else ChallengerProposalType.EVOLVE_FROM_HISTORY
                    )
                self._connection.execute(
                    """INSERT INTO epoch_challengers(
                           epoch_id, challenger_ordinal, kernel_agent_revision_id,
                           proposal_type, base_kernel_agent_revision_id,
                           evolution_trace_digest
                       ) VALUES (?, ?, ?, ?, ?, ?)""",
                    (
                        epoch.id,
                        challenger_ordinal,
                        revision_id,
                        proposal_type,
                        base_revision_id,
                        revision.evolution_trace_digest or epoch.evidence_checkpoint,
                    ),
                )
            self._connection.execute(
                "UPDATE lineages SET status = 'running' WHERE id = ?", (epoch.lineage_id,)
            )
            self._event(
                "epoch.created",
                epoch.id,
                {"lineage_id": epoch.lineage_id, "number": epoch.number},
            )

    def get_epoch(self, epoch_id: EpochId) -> Epoch:
        """Load an epoch or fail if it does not exist."""
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM epochs WHERE id = ?", (epoch_id,)
            ).fetchone()
        if row is None:
            raise KeyError(f"Epoch not found: {epoch_id}")
        return self._map_epoch(row)

    def list_epochs(self, lineage_id: LineageId) -> list[Epoch]:
        """Return every lineage Epoch in numeric order."""
        self.get_lineage(lineage_id)
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM epochs WHERE lineage_id = ? ORDER BY number",
                (lineage_id,),
            ).fetchall()
        return [self._map_epoch(row) for row in rows]

    def find_epoch(self, lineage_id: LineageId, number: int) -> Epoch | None:
        """Return a lineage epoch by its stable sequence number."""
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM epochs WHERE lineage_id = ? AND number = ?",
                (lineage_id, number),
            ).fetchone()
        return None if row is None else self._map_epoch(row)

    def find_open_epoch(self, lineage_id: LineageId) -> Epoch | None:
        """Return the single non-terminal epoch for a lineage, if present."""
        with self._lock:
            row = self._connection.execute(
                """SELECT * FROM epochs WHERE lineage_id = ?
                   AND status NOT IN ('completed', 'failed')""",
                (lineage_id,),
            ).fetchone()
        return None if row is None else self._map_epoch(row)

    def _map_epoch(self, row: sqlite3.Row) -> Epoch:
        winner = _optional_text(row, "winner_kernel_agent_revision_id")
        best = _optional_text(row, "best_kernel_revision_id")
        completed = _optional_text(row, "completed_at")
        epoch_id = parse_epoch_id(_required_text(row, "id"))
        with self._lock:
            challenger_rows = self._connection.execute(
                """SELECT kernel_agent_revision_id FROM epoch_challengers
                   WHERE epoch_id = ? ORDER BY challenger_ordinal""",
                (epoch_id,),
            ).fetchall()
        return Epoch(
            id=epoch_id,
            lineage_id=parse_lineage_id(_required_text(row, "lineage_id")),
            number=_required_int(row, "number"),
            active_kernel_agent_revision_id=parse_kernel_agent_revision_id(
                _required_text(row, "active_kernel_agent_revision_id")
            ),
            challenger_kernel_agent_revision_ids=tuple(
                parse_kernel_agent_revision_id(
                    _required_text(challenger_row, "kernel_agent_revision_id")
                )
                for challenger_row in challenger_rows
            ),
            starting_kernel_revision_id=parse_kernel_revision_id(
                _required_text(row, "starting_kernel_revision_id")
            ),
            evidence_checkpoint=parse_artifact_digest(_required_text(row, "evidence_checkpoint")),
            challenger_count=_required_int(row, "challenger_count"),
            trajectories_per_branch=_required_int(row, "trajectories_per_branch"),
            attempts_per_trajectory=_required_int(row, "attempts_per_branch"),
            status=EpochStatus(_required_text(row, "status")),
            winner_kernel_agent_revision_id=(
                None if winner is None else parse_kernel_agent_revision_id(winner)
            ),
            best_kernel_revision_id=None if best is None else parse_kernel_revision_id(best),
            created_at=_required_text(row, "created_at"),
            completed_at=completed,
        )

    @staticmethod
    def _map_epoch_challenger(row: sqlite3.Row) -> EpochChallenger:
        return EpochChallenger(
            epoch_id=parse_epoch_id(_required_text(row, "epoch_id")),
            challenger_ordinal=_required_int(row, "challenger_ordinal"),
            kernel_agent_revision_id=parse_kernel_agent_revision_id(
                _required_text(row, "kernel_agent_revision_id")
            ),
            proposal_type=ChallengerProposalType(_required_text(row, "proposal_type")),
            base_revision_id=parse_kernel_agent_revision_id(
                _required_text(row, "base_kernel_agent_revision_id")
            ),
            evolution_trace_digest=parse_artifact_digest(
                _required_text(row, "evolution_trace_digest")
            ),
        )

    def attach_challenger(self, challenger: EpochChallenger) -> None:
        """Attach one indexed Challenger and make the Epoch ready when its pool is full."""
        epoch_id = challenger.epoch_id
        challenger_ordinal = challenger.challenger_ordinal
        revision_id = challenger.kernel_agent_revision_id
        with self._transaction():
            row = self._connection.execute(
                "SELECT * FROM epochs WHERE id = ?", (epoch_id,)
            ).fetchone()
            if row is None:
                raise KeyError(f"Epoch not found: {epoch_id}")
            epoch = self._map_epoch(row)
            if challenger_ordinal > epoch.challenger_count:
                raise InvalidTransitionError(
                    f"Epoch {epoch_id} does not configure Challenger {challenger_ordinal}"
                )
            existing = self._connection.execute(
                """SELECT * FROM epoch_challengers
                   WHERE epoch_id = ? AND challenger_ordinal = ?""",
                (epoch_id, challenger_ordinal),
            ).fetchone()
            if existing is not None:
                persisted = self._map_epoch_challenger(existing)
                if persisted != challenger:
                    raise InvalidTransitionError(
                        f"Epoch {epoch_id} Challenger {challenger_ordinal} is already different"
                    )
                return
            if challenger_ordinal != len(epoch.challenger_kernel_agent_revision_ids) + 1:
                raise InvalidTransitionError("Epoch Challengers must be attached in order")
            revision = self.get_kernel_agent_revision(revision_id)
            base = self.get_kernel_agent_revision(challenger.base_revision_id)
            if revision.dsl is not base.dsl:
                raise InvalidTransitionError("Challenger and proposal base use different DSLs")
            base_version = self._connection.execute(
                """SELECT revision_number, introduced_epoch_id FROM lineage_agent_versions
                   WHERE lineage_id = ? AND kernel_agent_revision_id = ?""",
                (epoch.lineage_id, base.id),
            ).fetchone()
            if base_version is None:
                raise InvalidTransitionError(
                    "Challenger proposal base is absent from the Lineage history"
                )
            if (
                challenger.proposal_type
                in {
                    ChallengerProposalType.REUSE,
                    ChallengerProposalType.EVOLVE_FROM_HISTORY,
                }
                and _optional_text(base_version, "introduced_epoch_id") == epoch.id
            ):
                raise InvalidTransitionError(
                    "historical Challenger base was introduced in the current Epoch"
                )
            if challenger.proposal_type is ChallengerProposalType.REUSE:
                if revision.id == epoch.active_kernel_agent_revision_id:
                    raise InvalidTransitionError("The current Active Agent cannot be reused")
                version = self._connection.execute(
                    """SELECT revision_number FROM lineage_agent_versions
                       WHERE lineage_id = ? AND kernel_agent_revision_id = ?""",
                    (epoch.lineage_id, revision.id),
                ).fetchone()
                if version is None:
                    raise InvalidTransitionError("Reused Agent is absent from Lineage history")
                revision_number = _required_int(version, "revision_number")
            else:
                if revision.parent_id != base.id:
                    raise InvalidTransitionError(
                        "New Challenger revision parent disagrees with proposal base"
                    )
                if (
                    challenger.proposal_type is ChallengerProposalType.EVOLVED
                    and base.id != epoch.active_kernel_agent_revision_id
                ):
                    raise InvalidTransitionError("evolved proposal base is not Epoch Active")
                if (
                    challenger.proposal_type is ChallengerProposalType.EVOLVE_FROM_HISTORY
                    and base.id == epoch.active_kernel_agent_revision_id
                ):
                    raise InvalidTransitionError(
                        "evolve_from_history proposal base is the Epoch Active"
                    )
                revision_number = self._link_agent_version(
                    epoch.lineage_id,
                    revision,
                    expected_number=None,
                    introduced_epoch_id=epoch.id,
                )
            self._connection.execute(
                """INSERT INTO epoch_challengers(
                       epoch_id, challenger_ordinal, kernel_agent_revision_id,
                       proposal_type, base_kernel_agent_revision_id,
                       evolution_trace_digest
                   ) VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    epoch_id,
                    challenger_ordinal,
                    revision_id,
                    challenger.proposal_type,
                    challenger.base_revision_id,
                    challenger.evolution_trace_digest,
                ),
            )
            next_status = (
                EpochStatus.READY
                if challenger_ordinal == epoch.challenger_count
                else EpochStatus.BUILDING_CHALLENGER
            )
            cursor = self._connection.execute(
                """UPDATE epochs
                   SET challenger_kernel_agent_revision_id =
                           CASE WHEN ? = 1 THEN ? ELSE challenger_kernel_agent_revision_id END,
                       status = ?
                   WHERE id = ? AND status = 'building_challenger'""",
                (challenger_ordinal, revision_id, next_status, epoch_id),
            )
            if cursor.rowcount != 1:
                raise InvalidTransitionError(f"Epoch {epoch_id} cannot attach a Challenger")
            self._event(
                "epoch.challenger_ready",
                epoch_id,
                {
                    "challenger_ordinal": challenger_ordinal,
                    "revision_id": revision_id,
                    "agent_revision_number": revision_number,
                    "proposal_type": challenger.proposal_type,
                    "base_revision_id": challenger.base_revision_id,
                    "evolution_trace_digest": challenger.evolution_trace_digest,
                },
            )

    def list_epoch_challengers(self, epoch_id: EpochId) -> list[EpochChallenger]:
        """Return indexed Challenger participation and its proposal provenance."""
        with self._lock:
            rows = self._connection.execute(
                """SELECT * FROM epoch_challengers WHERE epoch_id = ?
                   ORDER BY challenger_ordinal""",
                (epoch_id,),
            ).fetchall()
        return [self._map_epoch_challenger(row) for row in rows]

    def transition_epoch(
        self,
        epoch_id: EpochId,
        expected: EpochStatus,
        next_status: EpochStatus,
    ) -> None:
        """Apply one allowed compare-and-swap epoch transition."""
        allowed = {
            (EpochStatus.READY, EpochStatus.RUNNING),
            (EpochStatus.RUNNING, EpochStatus.SELECTING),
        }
        if (expected, next_status) not in allowed:
            raise InvalidTransitionError(
                f"unsupported epoch transition {expected} -> {next_status}"
            )
        with self._transaction():
            cursor = self._connection.execute(
                "UPDATE epochs SET status = ? WHERE id = ? AND status = ?",
                (next_status, epoch_id, expected),
            )
            if cursor.rowcount != 1:
                raise InvalidTransitionError(
                    f"Epoch {epoch_id} cannot transition {expected} -> {next_status}"
                )
            self._event(
                "epoch.transitioned",
                epoch_id,
                {"expected": expected, "next": next_status},
            )

    def fail_epoch(self, epoch_id: EpochId, reason: str) -> None:
        """Atomically fail a non-terminal epoch and its lineage."""
        if not reason.strip():
            raise ValueError("epoch failure reason cannot be empty")
        with self._transaction():
            row = self._connection.execute(
                "SELECT lineage_id, status FROM epochs WHERE id = ?", (epoch_id,)
            ).fetchone()
            if row is None:
                raise KeyError(f"Epoch not found: {epoch_id}")
            status = EpochStatus(_required_text(row, "status"))
            if status in {EpochStatus.COMPLETED, EpochStatus.FAILED}:
                raise InvalidTransitionError(f"Epoch {epoch_id} is already terminal")
            lineage_id = parse_lineage_id(_required_text(row, "lineage_id"))
            self._connection.execute(
                "UPDATE epochs SET status = 'failed', failure_reason = ? WHERE id = ?",
                (reason, epoch_id),
            )
            self._connection.execute(
                "UPDATE lineages SET status = 'failed' WHERE id = ?", (lineage_id,)
            )
            self._connection.execute(
                """UPDATE campaigns SET status = 'failed'
                   WHERE id = (SELECT campaign_id FROM lineages WHERE id = ?)
                   AND status = 'active'""",
                (lineage_id,),
            )
            self._event("epoch.failed", epoch_id, {"reason": reason})

    def recover_failed_epoch(
        self,
        epoch_id: EpochId,
        *,
        recovery_key: str,
        reason: str,
    ) -> EpochRecovery:
        """Recover one failed epoch exactly once for an operator idempotency key."""
        recovery_key = recovery_key.strip()
        reason = reason.strip()
        if not recovery_key:
            raise ValueError("epoch recovery key cannot be empty")
        if not reason:
            raise ValueError("epoch recovery reason cannot be empty")
        if len(recovery_key) > 256:
            raise ValueError("epoch recovery key exceeds 256 characters")
        if len(reason) > 2048:
            raise ValueError("epoch recovery reason exceeds 2048 characters")

        with self._transaction():
            existing = self._connection.execute(
                """SELECT er.*, e.lineage_id, l.campaign_id
                   FROM epoch_recoveries er
                   JOIN epochs e ON e.id = er.epoch_id
                   JOIN lineages l ON l.id = e.lineage_id
                   WHERE er.epoch_id = ? AND er.recovery_key = ?""",
                (epoch_id, recovery_key),
            ).fetchone()
            if existing is not None:
                if _required_text(existing, "reason") != reason:
                    raise InvalidTransitionError(
                        "epoch recovery key was reused with a different reason"
                    )
                return self._map_epoch_recovery(existing)

            epoch = self._connection.execute(
                """SELECT e.*, l.campaign_id, l.status AS lineage_status
                   FROM epochs e JOIN lineages l ON l.id = e.lineage_id
                   WHERE e.id = ?""",
                (epoch_id,),
            ).fetchone()
            if epoch is None:
                raise KeyError(f"Epoch not found: {epoch_id}")
            if EpochStatus(_required_text(epoch, "status")) is not EpochStatus.FAILED:
                raise InvalidTransitionError(f"Epoch {epoch_id} is not failed")
            if LineageStatus(_required_text(epoch, "lineage_status")) is not LineageStatus.FAILED:
                raise InvalidTransitionError(f"Epoch {epoch_id} lineage is not failed")

            attempts = self._connection.execute(
                """SELECT a.id FROM attempts AS a
                   WHERE a.epoch_id = ? AND (
                       a.status = 'infrastructure_failed'
                       OR (
                           a.status = 'running'
                           AND a.output_kernel_revision_id IS NULL
                           AND EXISTS (
                               SELECT 1 FROM kernel_revisions AS kr
                               WHERE kr.produced_by_attempt_id = a.id
                           )
                       )
                   )
                   ORDER BY a.branch, a.ordinal, a.id""",
                (epoch_id,),
            ).fetchall()
            attempt_ids = tuple(parse_attempt_id(_required_text(row, "id")) for row in attempts)
            attached_challengers = self._connection.execute(
                "SELECT COUNT(*) AS count FROM epoch_challengers WHERE epoch_id = ?",
                (epoch_id,),
            ).fetchone()
            if attached_challengers is None:
                raise AssertionError("SQLite did not count attached Challengers")
            attached_count = _required_int(attached_challengers, "count")
            configured_count = _required_int(epoch, "challenger_count")
            if not attempts and attached_count >= configured_count:
                raise InvalidTransitionError(
                    f"Epoch {epoch_id} has no recoverable failed operation"
                )
            generation_row = self._connection.execute(
                """SELECT COALESCE(MAX(generation), 0) + 1 AS next_generation
                   FROM epoch_recoveries WHERE epoch_id = ?""",
                (epoch_id,),
            ).fetchone()
            if generation_row is None:
                raise AssertionError("SQLite did not compute a recovery generation")
            generation = _required_int(generation_row, "next_generation")
            created_at = self._clock()
            lineage_id = parse_lineage_id(_required_text(epoch, "lineage_id"))
            campaign_id = parse_campaign_id(_required_text(epoch, "campaign_id"))

            self._connection.execute(
                """UPDATE attempts SET status = 'running', infrastructure_failures = 0,
                   recovery_generation = recovery_generation + 1,
                   authority_started_at = ?, failure_reason = NULL
                   WHERE epoch_id = ? AND (
                       status = 'infrastructure_failed'
                       OR (
                           status = 'running'
                           AND output_kernel_revision_id IS NULL
                           AND EXISTS (
                               SELECT 1 FROM kernel_revisions AS kr
                               WHERE kr.produced_by_attempt_id = attempts.id
                           )
                       )
                   )""",
                (created_at, epoch_id),
            )
            recovered_epoch_status = (
                EpochStatus.RUNNING
                if attempts
                else (
                    EpochStatus.READY if configured_count == 0 else EpochStatus.BUILDING_CHALLENGER
                )
            )
            self._connection.execute(
                "UPDATE epochs SET status = ?, failure_reason = NULL WHERE id = ?",
                (recovered_epoch_status, epoch_id),
            )
            self._connection.execute(
                "UPDATE lineages SET status = 'running' WHERE id = ?",
                (lineage_id,),
            )
            self._connection.execute(
                """UPDATE lineage_fences
                   SET generation = generation + 1,
                       owner = ?, lease_expires_at = '0001-01-01T00:00:00+00:00'
                   WHERE lineage_id = ?""",
                (f"operator-recovery:{recovery_key}", lineage_id),
            )
            remaining_failed = self._connection.execute(
                """SELECT 1 FROM lineages
                   WHERE campaign_id = ? AND status = 'failed' LIMIT 1""",
                (campaign_id,),
            ).fetchone()
            reopened_campaign = remaining_failed is None
            if reopened_campaign:
                self._connection.execute(
                    "UPDATE campaigns SET status = 'active' WHERE id = ? AND status = 'failed'",
                    (campaign_id,),
                )
            self._connection.execute(
                """INSERT INTO epoch_recoveries(
                       epoch_id, recovery_key, generation, attempt_ids_json, reason, created_at
                   ) VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    epoch_id,
                    recovery_key,
                    generation,
                    json.dumps(attempt_ids),
                    reason,
                    created_at,
                ),
            )
            for attempt_id in attempt_ids:
                recovered_attempt = self._connection.execute(
                    "SELECT recovery_generation FROM attempts WHERE id = ?",
                    (attempt_id,),
                ).fetchone()
                if recovered_attempt is None:
                    raise AssertionError("recovered Attempt disappeared")
                self._event(
                    "attempt.recovered",
                    attempt_id,
                    {
                        "epoch_id": epoch_id,
                        "recovery_key": recovery_key,
                        "recovery_generation": _required_int(
                            recovered_attempt, "recovery_generation"
                        ),
                    },
                )
            event_payload = {
                "recovery_key": recovery_key,
                "generation": generation,
                "reason": reason,
                "attempt_ids": attempt_ids,
                "epoch_status": recovered_epoch_status,
            }
            self._event("epoch.recovered", epoch_id, event_payload)
            self._event("lineage.recovered", lineage_id, event_payload)
            if reopened_campaign:
                self._event(
                    "campaign.reopened",
                    campaign_id,
                    {"recovered_epoch_id": epoch_id, "recovery_key": recovery_key},
                )

        return EpochRecovery(
            epoch_id=epoch_id,
            lineage_id=lineage_id,
            campaign_id=campaign_id,
            recovery_key=recovery_key,
            generation=generation,
            attempt_ids=attempt_ids,
            reason=reason,
            created_at=created_at,
        )

    def insert_attempt(self, attempt: Attempt) -> None:
        """Persist an Attempt before launching its external Optimizer session."""
        with self._transaction():
            epoch = self.get_epoch(attempt.epoch_id)
            if attempt.trajectory_ordinal > epoch.trajectories_per_branch:
                raise InvalidTransitionError("Attempt Trajectory exceeds the Epoch budget")
            if attempt.ordinal > epoch.attempts_per_trajectory:
                raise InvalidTransitionError("Attempt iteration exceeds the Trajectory budget")
            if (
                attempt.branch is BranchRole.CHALLENGER
                and attempt.challenger_ordinal > epoch.challenger_count
            ):
                raise InvalidTransitionError("Attempt Challenger exceeds the Epoch pool")
            storage_ordinal = self._attempt_storage_ordinal(epoch, attempt)
            self._connection.execute(
                """INSERT INTO attempts(
                       id, epoch_id, branch, ordinal, kernel_agent_revision_id,
                       input_kernel_revision_id, attempt_evidence_digest,
                       output_kernel_revision_id, accepted_as_branch_best, status,
                       infrastructure_failures, recovery_generation, authority_started_at,
                       failure_reason, created_at, completed_at,
                       attempt_report_digest, attempt_report_status,
                       challenger_ordinal, trajectory_ordinal, iteration_ordinal,
                       input_runtime_state_digest
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    attempt.id,
                    attempt.epoch_id,
                    attempt.branch,
                    storage_ordinal,
                    attempt.kernel_agent_revision_id,
                    attempt.input_kernel_revision_id,
                    attempt.attempt_evidence_digest,
                    attempt.output_kernel_revision_id,
                    int(attempt.accepted_as_branch_best),
                    attempt.status,
                    attempt.infrastructure_failures,
                    attempt.recovery_generation,
                    attempt.authority_started_at,
                    attempt.failure_reason,
                    attempt.created_at,
                    attempt.completed_at,
                    attempt.attempt_report_digest,
                    attempt.attempt_report_status,
                    attempt.challenger_ordinal,
                    attempt.trajectory_ordinal,
                    attempt.ordinal,
                    attempt.input_runtime_state_digest,
                ),
            )
            self._event(
                "attempt.started",
                attempt.id,
                {
                    "epoch_id": attempt.epoch_id,
                    "branch": attempt.branch,
                    "challenger_ordinal": attempt.challenger_ordinal,
                    "trajectory_ordinal": attempt.trajectory_ordinal,
                    "iteration_ordinal": attempt.ordinal,
                    "attempt_evidence_digest": attempt.attempt_evidence_digest,
                },
            )

    def get_attempt(self, attempt_id: AttemptId) -> Attempt:
        """Load an Attempt or fail if it does not exist."""
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM attempts WHERE id = ?", (attempt_id,)
            ).fetchone()
        if row is None:
            raise KeyError(f"Attempt not found: {attempt_id}")
        return self._map_attempt(row)

    def find_attempt(
        self,
        epoch_id: EpochId,
        branch: BranchRole,
        challenger_ordinal: int,
        trajectory_ordinal: int,
        ordinal: int,
    ) -> Attempt | None:
        """Return one stable Branch/Trajectory iteration."""
        with self._lock:
            row = self._connection.execute(
                """SELECT * FROM attempts
                   WHERE epoch_id = ? AND branch = ? AND challenger_ordinal = ?
                     AND trajectory_ordinal = ? AND iteration_ordinal = ?""",
                (epoch_id, branch, challenger_ordinal, trajectory_ordinal, ordinal),
            ).fetchone()
        return None if row is None else self._map_attempt(row)

    def list_attempts(self, epoch_id: EpochId) -> list[Attempt]:
        """Return all epoch Attempts in deterministic order."""
        with self._lock:
            rows = self._connection.execute(
                """SELECT * FROM attempts WHERE epoch_id = ?
                   ORDER BY CASE branch WHEN 'active' THEN 0 ELSE 1 END,
                            challenger_ordinal, trajectory_ordinal, iteration_ordinal""",
                (epoch_id,),
            ).fetchall()
        return [self._map_attempt(row) for row in rows]

    def record_attempt_session_trace(
        self,
        attempt_id: AttemptId,
        artifact_digest: ArtifactDigest,
        finish_reason: str,
        token_budget: int,
        token_usage: TokenUsage,
    ) -> AttemptSessionTrace:
        """Append one session artifact using an Attempt-local run ordinal."""
        if not finish_reason:
            raise ValueError("Attempt session trace finish reason cannot be empty")
        with self._transaction():
            attempt = self._connection.execute(
                "SELECT 1 FROM attempts WHERE id = ?", (attempt_id,)
            ).fetchone()
            if attempt is None:
                raise KeyError(f"Attempt not found: {attempt_id}")
            row = self._connection.execute(
                """SELECT COALESCE(MAX(run_ordinal), 0) + 1 AS next_ordinal
                   FROM attempt_session_traces WHERE attempt_id = ?""",
                (attempt_id,),
            ).fetchone()
            if row is None:
                raise RuntimeError("SQLite did not return the next trace ordinal")
            trace = AttemptSessionTrace(
                attempt_id=attempt_id,
                run_ordinal=_required_int(row, "next_ordinal"),
                artifact_digest=artifact_digest,
                finish_reason=finish_reason,
                token_budget=token_budget,
                token_usage=token_usage,
                created_at=self._clock(),
            )
            self._connection.execute(
                """INSERT INTO attempt_session_traces
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    trace.attempt_id,
                    trace.run_ordinal,
                    trace.artifact_digest,
                    trace.finish_reason,
                    trace.token_budget,
                    trace.token_usage.uncached_input_tokens,
                    trace.token_usage.output_tokens,
                    trace.token_usage.cache_read_tokens,
                    trace.token_usage.cache_write_tokens,
                    trace.token_usage.credits,
                    trace.created_at,
                ),
            )
            self._event(
                "attempt.session_trace_recorded",
                attempt_id,
                {
                    "run_ordinal": trace.run_ordinal,
                    "artifact_digest": artifact_digest,
                    "finish_reason": finish_reason,
                    "token_budget": token_budget,
                    "token_usage": {
                        "uncached_input_tokens": token_usage.uncached_input_tokens,
                        "output_tokens": token_usage.output_tokens,
                        "cache_read_tokens": token_usage.cache_read_tokens,
                        "cache_write_tokens": token_usage.cache_write_tokens,
                        "total_tokens": token_usage.total_tokens,
                        "credits": token_usage.credits,
                        "usage_unit": token_usage.usage_unit,
                    },
                },
            )
        return trace

    def list_attempt_session_traces(self, attempt_id: AttemptId) -> list[AttemptSessionTrace]:
        """Return every recorded Optimizer session in launch order."""
        with self._lock:
            rows = self._connection.execute(
                """SELECT * FROM attempt_session_traces
                   WHERE attempt_id = ? ORDER BY run_ordinal""",
                (attempt_id,),
            ).fetchall()
        return [
            AttemptSessionTrace(
                attempt_id=parse_attempt_id(_required_text(row, "attempt_id")),
                run_ordinal=_required_int(row, "run_ordinal"),
                artifact_digest=parse_artifact_digest(_required_text(row, "artifact_digest")),
                finish_reason=_required_text(row, "finish_reason"),
                token_budget=_required_int(row, "token_budget"),
                token_usage=TokenUsage(
                    uncached_input_tokens=_required_int(row, "uncached_input_tokens"),
                    output_tokens=_required_int(row, "output_tokens"),
                    cache_read_tokens=_required_int(row, "cache_read_tokens"),
                    cache_write_tokens=_required_int(row, "cache_write_tokens"),
                    credits=_optional_float(row, "credits"),
                ),
                created_at=_required_text(row, "created_at"),
            )
            for row in rows
        ]

    def record_attempt_runtime_state(
        self,
        attempt_id: AttemptId,
        runtime_state_digest: ArtifactDigest,
    ) -> None:
        """Point a running Attempt at its latest immutable post-Session state."""
        with self._transaction():
            row = self._connection.execute(
                "SELECT status, runtime_state_digest FROM attempts WHERE id = ?",
                (attempt_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"Attempt not found: {attempt_id}")
            existing = _optional_text(row, "runtime_state_digest")
            if existing == runtime_state_digest:
                return
            if row["status"] != AttemptStatus.RUNNING:
                raise InvalidTransitionError(
                    f"Attempt {attempt_id} cannot record Runtime State"
                )
            cursor = self._connection.execute(
                """UPDATE attempts SET runtime_state_digest = ?
                   WHERE id = ? AND status = 'running'""",
                (runtime_state_digest, attempt_id),
            )
            if cursor.rowcount != 1:
                raise InvalidTransitionError(
                    f"Attempt {attempt_id} cannot record Runtime State"
                )
            self._event(
                "attempt.runtime_state_recorded",
                attempt_id,
                {
                    "runtime_state_digest": runtime_state_digest,
                    "previous_runtime_state_digest": existing,
                },
            )

    def record_attempt_input_runtime_state(
        self,
        attempt_id: AttemptId,
        runtime_state_digest: ArtifactDigest,
    ) -> None:
        """Attach the immutable logical input State to a running Attempt once."""
        with self._transaction():
            row = self._connection.execute(
                "SELECT status, input_runtime_state_digest FROM attempts WHERE id = ?",
                (attempt_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"Attempt not found: {attempt_id}")
            existing = _optional_text(row, "input_runtime_state_digest")
            if existing == runtime_state_digest:
                return
            if existing is not None or row["status"] != AttemptStatus.RUNNING:
                raise InvalidTransitionError(
                    f"Attempt {attempt_id} cannot record Input Runtime State"
                )
            cursor = self._connection.execute(
                """UPDATE attempts SET input_runtime_state_digest = ?
                   WHERE id = ? AND status = 'running'
                     AND input_runtime_state_digest IS NULL""",
                (runtime_state_digest, attempt_id),
            )
            if cursor.rowcount != 1:
                raise InvalidTransitionError(
                    f"Attempt {attempt_id} cannot record Input Runtime State"
                )
            self._event(
                "attempt.input_runtime_state_recorded",
                attempt_id,
                {"input_runtime_state_digest": runtime_state_digest},
            )

    def record_attempt_report(
        self,
        attempt_id: AttemptId,
        artifact_digest: ArtifactDigest,
        status: AttemptReportStatus,
    ) -> None:
        """Attach one immutable terminal report to a running Attempt idempotently."""
        with self._transaction():
            row = self._connection.execute(
                "SELECT attempt_report_digest, attempt_report_status FROM attempts WHERE id = ?",
                (attempt_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"Attempt not found: {attempt_id}")
            existing_digest = _optional_text(row, "attempt_report_digest")
            existing_status = _optional_text(row, "attempt_report_status")
            if existing_digest is not None or existing_status is not None:
                if existing_digest != artifact_digest or existing_status != status:
                    raise InvalidTransitionError(
                        f"Attempt {attempt_id} already has a different terminal report"
                    )
                return
            cursor = self._connection.execute(
                """UPDATE attempts SET attempt_report_digest = ?, attempt_report_status = ?
                   WHERE id = ? AND status = 'running'""",
                (artifact_digest, status, attempt_id),
            )
            if cursor.rowcount != 1:
                raise InvalidTransitionError(
                    f"Attempt {attempt_id} cannot record a terminal report"
                )
            self._event(
                "attempt.report_recorded",
                attempt_id,
                {"artifact_digest": artifact_digest, "status": status},
            )

    @staticmethod
    def _map_attempt(row: sqlite3.Row) -> Attempt:
        output = _optional_text(row, "output_kernel_revision_id")
        completed = _optional_text(row, "completed_at")
        return Attempt(
            id=parse_attempt_id(_required_text(row, "id")),
            epoch_id=parse_epoch_id(_required_text(row, "epoch_id")),
            branch=BranchRole(_required_text(row, "branch")),
            challenger_ordinal=_required_int(row, "challenger_ordinal"),
            trajectory_ordinal=_required_int(row, "trajectory_ordinal"),
            ordinal=_required_int(row, "iteration_ordinal"),
            kernel_agent_revision_id=parse_kernel_agent_revision_id(
                _required_text(row, "kernel_agent_revision_id")
            ),
            input_kernel_revision_id=parse_kernel_revision_id(
                _required_text(row, "input_kernel_revision_id")
            ),
            attempt_evidence_digest=parse_artifact_digest(
                _required_text(row, "attempt_evidence_digest")
            ),
            output_kernel_revision_id=(
                None if output is None else parse_kernel_revision_id(output)
            ),
            accepted_as_branch_best=bool(_required_int(row, "accepted_as_branch_best")),
            status=AttemptStatus(_required_text(row, "status")),
            infrastructure_failures=_required_int(row, "infrastructure_failures"),
            recovery_generation=_required_int(row, "recovery_generation"),
            authority_started_at=_required_text(row, "authority_started_at"),
            failure_reason=_optional_text(row, "failure_reason"),
            created_at=_required_text(row, "created_at"),
            completed_at=completed,
            attempt_report_digest=(
                None
                if (report := _optional_text(row, "attempt_report_digest")) is None
                else parse_artifact_digest(report)
            ),
            attempt_report_status=(
                None
                if (report_status := _optional_text(row, "attempt_report_status")) is None
                else AttemptReportStatus(report_status)
            ),
            runtime_state_digest=(
                None
                if (state := _optional_text(row, "runtime_state_digest")) is None
                else parse_artifact_digest(state)
            ),
            input_runtime_state_digest=(
                None
                if (
                    input_state := _optional_text(row, "input_runtime_state_digest")
                )
                is None
                else parse_artifact_digest(input_state)
            ),
        )

    @staticmethod
    def _attempt_storage_ordinal(epoch: Epoch, attempt: Attempt) -> int:
        """Encode a legacy branch-wide unique ordinal without exposing it as semantics."""
        branch_offset = (
            0
            if attempt.branch is BranchRole.ACTIVE
            else (attempt.challenger_ordinal - 1) * epoch.trajectories_per_branch
        )
        trajectory_offset = branch_offset + attempt.trajectory_ordinal - 1
        return trajectory_offset * epoch.attempts_per_trajectory + attempt.ordinal

    @staticmethod
    def _map_campaign_task(row: sqlite3.Row) -> CampaignTask:
        return CampaignTask(
            id=parse_campaign_task_id(_required_text(row, "id")),
            creation_key=_required_text(row, "creation_key"),
            campaign_id=parse_campaign_id(_required_text(row, "campaign_id")),
            target_epoch_number=_required_int(row, "target_epoch_number"),
            finalize=bool(_required_int(row, "finalize")),
            status=CampaignTaskStatus(_required_text(row, "status")),
            attempt_count=_required_int(row, "attempt_count"),
            lease_owner=_optional_text(row, "lease_owner"),
            lease_expires_at=_optional_text(row, "lease_expires_at"),
            last_error=_optional_text(row, "last_error"),
            created_at=_required_text(row, "created_at"),
            started_at=_optional_text(row, "started_at"),
            completed_at=_optional_text(row, "completed_at"),
        )

    @staticmethod
    def _map_epoch_recovery(row: sqlite3.Row) -> EpochRecovery:
        raw_attempt_ids = json.loads(_required_text(row, "attempt_ids_json"))
        if not isinstance(raw_attempt_ids, list) or not all(
            isinstance(value, str) for value in raw_attempt_ids
        ):
            raise TypeError("persisted recovery Attempt identifiers must be a JSON list")
        return EpochRecovery(
            epoch_id=parse_epoch_id(_required_text(row, "epoch_id")),
            lineage_id=parse_lineage_id(_required_text(row, "lineage_id")),
            campaign_id=parse_campaign_id(_required_text(row, "campaign_id")),
            recovery_key=_required_text(row, "recovery_key"),
            generation=_required_int(row, "generation"),
            attempt_ids=tuple(parse_attempt_id(value) for value in raw_attempt_ids),
            reason=_required_text(row, "reason"),
            created_at=_required_text(row, "created_at"),
        )

    def retry_attempt(self, attempt_id: AttemptId) -> None:
        """Start a fresh authority generation for an infrastructure-failed Attempt."""
        with self._transaction():
            authority_started_at = self._clock()
            cursor = self._connection.execute(
                """UPDATE attempts SET status = 'running', failure_reason = NULL,
                   recovery_generation = recovery_generation + 1,
                   authority_started_at = ?
                   WHERE id = ? AND status = 'infrastructure_failed'""",
                (authority_started_at, attempt_id),
            )
            if cursor.rowcount != 1:
                raise InvalidTransitionError(f"Attempt {attempt_id} cannot be retried")
            row = self._connection.execute(
                "SELECT recovery_generation FROM attempts WHERE id = ?", (attempt_id,)
            ).fetchone()
            if row is None:
                raise AssertionError("retried Attempt disappeared")
            self._event(
                "attempt.retried",
                attempt_id,
                {
                    "recovery_generation": _required_int(row, "recovery_generation"),
                    "authority_started_at": authority_started_at,
                },
            )

    def record_infrastructure_failure(self, attempt_id: AttemptId, reason: str) -> None:
        """Record a non-consuming infrastructure failure."""
        with self._transaction():
            cursor = self._connection.execute(
                """UPDATE attempts SET status = 'infrastructure_failed',
                   infrastructure_failures = infrastructure_failures + 1,
                   failure_reason = ? WHERE id = ? AND status = 'running'""",
                (reason, attempt_id),
            )
            if cursor.rowcount != 1:
                raise InvalidTransitionError(
                    f"Attempt {attempt_id} cannot record infrastructure failure"
                )
            self._event("attempt.infrastructure_failed", attempt_id, {"reason": reason})

    def complete_attempt(
        self,
        attempt_id: AttemptId,
        output_kernel_revision_id: KernelRevisionId | None,
        *,
        accepted_as_branch_best: bool,
        failure_reason: str | None,
    ) -> None:
        """Complete one opportunity after its output, if any, is registered."""
        with self._transaction():
            row = self._connection.execute(
                "SELECT input_kernel_revision_id FROM attempts WHERE id = ?",
                (attempt_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"Attempt not found: {attempt_id}")
            input_kernel_revision_id = _required_text(row, "input_kernel_revision_id")
            cursor = self._connection.execute(
                """UPDATE attempts SET status = 'completed', output_kernel_revision_id = ?,
                   accepted_as_branch_best = ?, failure_reason = ?, completed_at = ?
                   WHERE id = ? AND status = 'running'""",
                (
                    output_kernel_revision_id,
                    int(accepted_as_branch_best),
                    failure_reason,
                    self._clock(),
                    attempt_id,
                ),
            )
            if cursor.rowcount != 1:
                raise InvalidTransitionError(f"Attempt {attempt_id} cannot complete")
            self._event(
                "attempt.completed",
                attempt_id,
                {
                    "output_kernel_revision_id": output_kernel_revision_id,
                    "accepted_as_branch_best": accepted_as_branch_best,
                },
            )
            if output_kernel_revision_id is not None and not accepted_as_branch_best:
                self._event(
                    "kernel.rollback",
                    output_kernel_revision_id,
                    {
                        "attempt_id": attempt_id,
                        "restored_kernel_revision_id": input_kernel_revision_id,
                    },
                )

    def complete_epoch(self, epoch_id: EpochId, selection: EpochSelection) -> None:
        """Atomically commit Agent promotion, Kernel promotion, and epoch completion."""
        with self._transaction():
            row = self._connection.execute(
                "SELECT * FROM epochs WHERE id = ?", (epoch_id,)
            ).fetchone()
            if row is None:
                raise KeyError(f"Epoch not found: {epoch_id}")
            epoch = self._map_epoch(row)
            if epoch.status is not EpochStatus.SELECTING:
                raise InvalidTransitionError(f"Epoch {epoch.id} is not selecting")
            now = self._clock()
            cursor = self._connection.execute(
                """UPDATE epochs SET status = 'completed',
                   winner_kernel_agent_revision_id = ?, best_kernel_revision_id = ?,
                   completed_at = ? WHERE id = ? AND status = 'selecting'""",
                (
                    selection.winner_kernel_agent_revision_id,
                    selection.best_kernel_revision_id,
                    now,
                    epoch_id,
                ),
            )
            if cursor.rowcount != 1:
                raise InvalidTransitionError(f"Epoch {epoch.id} lost its selection lease")
            self._connection.execute(
                """UPDATE lineages SET status = 'awaiting_evidence',
                   active_kernel_agent_revision_id = ?, best_kernel_revision_id = ?,
                   next_epoch_number = next_epoch_number + 1 WHERE id = ?""",
                (
                    selection.winner_kernel_agent_revision_id,
                    selection.best_kernel_revision_id,
                    epoch.lineage_id,
                ),
            )
            self._event(
                "epoch.completed",
                epoch_id,
                {
                    "winner_kernel_agent_revision_id": selection.winner_kernel_agent_revision_id,
                    "best_kernel_revision_id": selection.best_kernel_revision_id,
                },
            )
            challengers = epoch.challenger_kernel_agent_revision_ids
            if selection.winner_kernel_agent_revision_id in challengers:
                self._event(
                    "kernel_agent.promoted",
                    selection.winner_kernel_agent_revision_id,
                    {
                        "epoch_id": epoch_id,
                        "replaced_revision_id": epoch.active_kernel_agent_revision_id,
                    },
                )
            for challenger_ordinal, challenger_id in enumerate(challengers, start=1):
                if challenger_id == selection.winner_kernel_agent_revision_id:
                    continue
                self._event(
                    "kernel_agent.rollback",
                    challenger_id,
                    {
                        "epoch_id": epoch_id,
                        "challenger_ordinal": challenger_ordinal,
                        "restored_revision_id": selection.winner_kernel_agent_revision_id,
                    },
                )
