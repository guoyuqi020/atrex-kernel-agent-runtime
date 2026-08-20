"""Unified Worker session lifecycle and trace-index tests."""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

import pytest
from conftest import NOW, digest

from atrex_runtime.domain.errors import InvalidTransitionError
from atrex_runtime.domain.ids import new_worker_session_id
from atrex_runtime.domain.models import (
    TokenUsage,
    WorkerSession,
    WorkerSessionRole,
    WorkerSessionStatus,
)
from atrex_runtime.registry.sqlite import SCHEMA_VERSION, SqliteRegistry


def test_worker_session_is_visible_while_running_and_retains_raw_trace(
    tmp_path: Path,
) -> None:
    registry = SqliteRegistry(tmp_path / "registry.sqlite", clock=lambda: NOW)
    session_id = new_worker_session_id()
    running = registry.start_worker_session(
        WorkerSession(
            id=session_id,
            role=WorkerSessionRole.PROBLEM_GENERALIZATION,
            subject_id="generalization-1",
            external_run_id="run-1",
            workspace_path=str(tmp_path / "run-1"),
            status=WorkerSessionStatus.RUNNING,
            started_at=NOW,
            backend="codex",
            model="gpt-5.6-codex",
        )
    )

    assert registry.get_worker_session(session_id) == running
    assert registry.list_worker_sessions(subject_id="generalization-1") == [running]

    trace_digest = digest("raw-session-trace")
    usage = TokenUsage(100, 20, 30, 0)
    completed = registry.finish_worker_session(
        session_id,
        status=WorkerSessionStatus.COMPLETED,
        finish_reason="completed",
        trace_digest=trace_digest,
        token_budget=20_000_000,
        token_usage=usage,
        process_returncode=0,
    )

    assert completed.trace_digest == trace_digest
    assert completed.token_budget == 20_000_000
    assert completed.token_usage == usage
    assert completed.completed_at == NOW
    assert trace_digest in registry.list_referenced_artifact_digests()
    with pytest.raises(InvalidTransitionError, match="already completed"):
        registry.finish_worker_session(
            session_id,
            status=WorkerSessionStatus.FAILED,
            finish_reason="late-failure",
        )
    registry.close()


def test_failed_worker_session_survives_without_a_trace(tmp_path: Path) -> None:
    with SqliteRegistry(tmp_path / "registry.sqlite", clock=lambda: NOW) as registry:
        session = registry.start_worker_session(
            WorkerSession(
                id=new_worker_session_id(),
                role=WorkerSessionRole.EVOLVER,
                subject_id="epoch_" + "1" * 32,
                external_run_id="run-timeout",
                workspace_path=str(tmp_path / "run-timeout"),
                status=WorkerSessionStatus.RUNNING,
                started_at=NOW,
            )
        )
        failed = registry.finish_worker_session(
            session.id,
            status=WorkerSessionStatus.TIMED_OUT,
            finish_reason="timeout",
            error_type="InfrastructureError",
            error_message="wall-time limit",
        )

        assert failed.trace_digest is None
        assert failed.status is WorkerSessionStatus.TIMED_OUT
        assert registry.list_worker_sessions(status=WorkerSessionStatus.TIMED_OUT) == [failed]


def test_qoder_credits_survive_worker_session_round_trip(tmp_path: Path) -> None:
    with SqliteRegistry(tmp_path / "registry.sqlite", clock=lambda: NOW) as registry:
        session = registry.start_worker_session(
            WorkerSession(
                id=new_worker_session_id(),
                role=WorkerSessionRole.OPTIMIZER,
                subject_id="attempt_" + "1" * 32,
                external_run_id="qoder-run",
                workspace_path=str(tmp_path / "qoder-run"),
                status=WorkerSessionStatus.RUNNING,
                started_at=NOW,
                backend="qodercli",
            )
        )
        usage = TokenUsage(0, 0, 0, 0, credits=13.75)
        registry.finish_worker_session(
            session.id,
            status=WorkerSessionStatus.COMPLETED,
            finish_reason="completed",
            token_budget=1000,
            token_usage=usage,
        )

        persisted = registry.get_worker_session(session.id)
        assert persisted.token_usage == usage
        assert persisted.token_usage is not None
        assert persisted.token_usage.usage_unit == "credits"
        assert persisted.token_usage.consumed == 13.75


def test_schema_23_migrates_to_worker_session_catalog(tmp_path: Path) -> None:
    path = tmp_path / "registry.sqlite"
    with closing(sqlite3.connect(path)) as connection:
        connection.execute("PRAGMA user_version = 23")

    with SqliteRegistry(path):
        pass

    with closing(sqlite3.connect(path)) as connection:
        version = connection.execute("PRAGMA user_version").fetchone()
        table = connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'worker_sessions'"
        ).fetchone()
    assert version == (SCHEMA_VERSION,)
    assert table == ("worker_sessions",)
