"""SQLite schema creation and migrations for Gateway Control."""

from __future__ import annotations

import sqlite3

GATEWAY_SCHEMA_VERSION = 7


def migrate_gateway_schema(connection: sqlite3.Connection) -> None:
    connection.execute(
        "CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY, value INTEGER NOT NULL)"
    )
    row = connection.execute("SELECT value FROM metadata WHERE key = 'schema_version'").fetchone()
    if row is None:
        connection.execute(
            """
            CREATE TABLE bootstrap_gateway_subjects(
                attempt_id TEXT PRIMARY KEY,
                campaign_id TEXT NOT NULL,
                lineage_id TEXT NOT NULL,
                epoch_id TEXT NOT NULL,
                kernel_agent_revision_id TEXT NOT NULL,
                operator TEXT NOT NULL,
                hardware_target TEXT NOT NULL,
                dsl TEXT NOT NULL CHECK(dsl IN ('cuda', 'triton', 'cutedsl')),
                evaluation_contract_digest TEXT NOT NULL,
                input_kernel_digest TEXT NOT NULL,
                evidence_digest TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE gateway_capabilities(
                attempt_id TEXT PRIMARY KEY,
                recovery_generation INTEGER NOT NULL
                    CHECK(recovery_generation >= 0),
                token_hash TEXT NOT NULL,
                operations_json TEXT NOT NULL,
                max_calls INTEGER NOT NULL CHECK(max_calls > 0),
                used_calls INTEGER NOT NULL CHECK(used_calls >= 0),
                expires_at TEXT NOT NULL,
                revoked INTEGER NOT NULL CHECK(revoked IN (0, 1))
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE gateway_operations(
                attempt_id TEXT NOT NULL,
                recovery_generation INTEGER NOT NULL
                    CHECK(recovery_generation >= 0),
                idempotency_key TEXT NOT NULL,
                operation TEXT NOT NULL,
                request_digest TEXT NOT NULL,
                result_artifact_digest TEXT,
                created_at TEXT NOT NULL,
                PRIMARY KEY(attempt_id, recovery_generation, idempotency_key),
                FOREIGN KEY(attempt_id) REFERENCES gateway_capabilities(attempt_id)
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE attempt_outcomes(
                attempt_id TEXT PRIMARY KEY,
                artifact_digest TEXT NOT NULL,
                gateway_result_digest TEXT NOT NULL,
                correct INTEGER NOT NULL CHECK(correct IN (0, 1)),
                latency_us REAL,
                committed_at TEXT NOT NULL,
                source_evaluation_id TEXT,
                FOREIGN KEY(attempt_id) REFERENCES gateway_capabilities(attempt_id)
            )
            """
        )
        _create_gateway_evaluations_table(connection)
        _create_bootstrap_runs_table(connection)
        connection.execute(
            "INSERT INTO metadata(key, value) VALUES ('schema_version', ?)",
            (GATEWAY_SCHEMA_VERSION,),
        )
    elif row["value"] == 3:
        connection.execute(
            """
            CREATE TABLE bootstrap_gateway_subjects(
                attempt_id TEXT PRIMARY KEY,
                campaign_id TEXT NOT NULL,
                lineage_id TEXT NOT NULL,
                epoch_id TEXT NOT NULL,
                kernel_agent_revision_id TEXT NOT NULL,
                operator TEXT NOT NULL,
                hardware_target TEXT NOT NULL,
                dsl TEXT NOT NULL CHECK(dsl IN ('cuda', 'triton', 'cutedsl')),
                evaluation_contract_digest TEXT NOT NULL,
                input_kernel_digest TEXT NOT NULL,
                evidence_digest TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        _migrate_gateway_v5(connection)
        _migrate_gateway_v6(connection)
        _migrate_gateway_v7(connection)
        connection.execute(
            "UPDATE metadata SET value = ? WHERE key = 'schema_version'",
            (GATEWAY_SCHEMA_VERSION,),
        )
    elif row["value"] == 4:
        _migrate_gateway_v5(connection)
        _migrate_gateway_v6(connection)
        _migrate_gateway_v7(connection)
        connection.execute(
            "UPDATE metadata SET value = ? WHERE key = 'schema_version'",
            (GATEWAY_SCHEMA_VERSION,),
        )
    elif row["value"] == 5:
        _migrate_gateway_v6(connection)
        _migrate_gateway_v7(connection)
        connection.execute(
            "UPDATE metadata SET value = ? WHERE key = 'schema_version'",
            (GATEWAY_SCHEMA_VERSION,),
        )
    elif row["value"] == 6:
        _migrate_gateway_v7(connection)
        connection.execute(
            "UPDATE metadata SET value = ? WHERE key = 'schema_version'",
            (GATEWAY_SCHEMA_VERSION,),
        )
    elif row["value"] != GATEWAY_SCHEMA_VERSION:
        raise RuntimeError(f"unsupported Gateway schema version: {row['value']}")


def _create_gateway_evaluations_table(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS gateway_evaluations(
            id TEXT PRIMARY KEY,
            attempt_id TEXT NOT NULL,
            recovery_generation INTEGER NOT NULL CHECK(recovery_generation >= 0),
            ordinal INTEGER NOT NULL CHECK(ordinal > 0),
            source TEXT NOT NULL CHECK(source IN ('agent', 'runtime_final')),
            idempotency_key TEXT NOT NULL,
            candidate_artifact_digest TEXT NOT NULL,
            gateway_result_digest TEXT NOT NULL,
            correct INTEGER NOT NULL CHECK(correct IN (0, 1)),
            latency_us REAL,
            agate_job_id TEXT,
            created_at TEXT NOT NULL,
            UNIQUE(attempt_id, recovery_generation, ordinal),
            UNIQUE(attempt_id, recovery_generation, source, idempotency_key),
            FOREIGN KEY(attempt_id) REFERENCES gateway_capabilities(attempt_id)
        )
        """
    )


def _migrate_gateway_v6(connection: sqlite3.Connection) -> None:
    _create_gateway_evaluations_table(connection)
    columns = {
        str(row["name"])
        for row in connection.execute("PRAGMA table_info(attempt_outcomes)").fetchall()
    }
    if "source_evaluation_id" not in columns:
        connection.execute("ALTER TABLE attempt_outcomes ADD COLUMN source_evaluation_id TEXT")


def _migrate_gateway_v7(connection: sqlite3.Connection) -> None:
    columns = {
        str(row["name"])
        for row in connection.execute("PRAGMA table_info(bootstrap_runs)").fetchall()
    }
    if "credits" not in columns:
        connection.execute(
            "ALTER TABLE bootstrap_runs ADD COLUMN credits REAL "
            "CHECK(credits IS NULL OR credits >= 0)"
        )


def _create_bootstrap_runs_table(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE bootstrap_runs(
            attempt_id TEXT NOT NULL,
            recovery_generation INTEGER NOT NULL CHECK(recovery_generation >= 0),
            status TEXT NOT NULL CHECK(status IN ('issued', 'running', 'completed', 'failed')),
            run_id TEXT,
            workspace_path TEXT,
            finish_reason TEXT,
            failure_reason TEXT,
            started_at TEXT NOT NULL,
            completed_at TEXT,
            session_trace_digest TEXT,
            token_budget INTEGER CHECK(token_budget > 0),
            uncached_input_tokens INTEGER CHECK(uncached_input_tokens >= 0),
            cache_read_tokens INTEGER CHECK(cache_read_tokens >= 0),
            cache_write_tokens INTEGER CHECK(cache_write_tokens >= 0),
            output_tokens INTEGER CHECK(output_tokens >= 0),
            credits REAL CHECK(credits IS NULL OR credits >= 0),
            report_digest TEXT,
            candidate_digest TEXT,
            gateway_result_digest TEXT,
            PRIMARY KEY(attempt_id, recovery_generation),
            FOREIGN KEY(attempt_id) REFERENCES bootstrap_gateway_subjects(attempt_id)
        )
        """
    )


def _migrate_gateway_v5(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE gateway_operations_v5(
            attempt_id TEXT NOT NULL,
            recovery_generation INTEGER NOT NULL CHECK(recovery_generation >= 0),
            idempotency_key TEXT NOT NULL,
            operation TEXT NOT NULL,
            request_digest TEXT NOT NULL,
            result_artifact_digest TEXT,
            created_at TEXT NOT NULL,
            PRIMARY KEY(attempt_id, recovery_generation, idempotency_key),
            FOREIGN KEY(attempt_id) REFERENCES gateway_capabilities(attempt_id)
        )
        """
    )
    connection.execute(
        """
        INSERT INTO gateway_operations_v5(
            attempt_id, recovery_generation, idempotency_key, operation,
            request_digest, result_artifact_digest, created_at
        )
        SELECT operations.attempt_id, capabilities.recovery_generation,
               operations.idempotency_key, operations.operation,
               operations.request_digest, operations.result_artifact_digest,
               operations.created_at
        FROM gateway_operations AS operations
        JOIN gateway_capabilities AS capabilities
          ON capabilities.attempt_id = operations.attempt_id
        """
    )
    connection.execute("DROP TABLE gateway_operations")
    connection.execute("ALTER TABLE gateway_operations_v5 RENAME TO gateway_operations")
    _create_bootstrap_runs_table(connection)
    connection.execute(
        """
        INSERT INTO bootstrap_runs(
            attempt_id, recovery_generation, status, finish_reason, failure_reason,
            started_at, completed_at, candidate_digest, gateway_result_digest
        )
        SELECT subjects.attempt_id, capabilities.recovery_generation,
               CASE
                 WHEN outcomes.attempt_id IS NOT NULL THEN 'completed'
                 ELSE 'failed'
               END,
               CASE
                 WHEN outcomes.attempt_id IS NOT NULL THEN 'migrated-completed'
                 ELSE 'migrated-without-outcome'
               END,
               CASE
                 WHEN outcomes.attempt_id IS NULL
                   THEN 'Legacy Bootstrap generation ended without an authoritative outcome'
                 ELSE NULL
               END,
               subjects.created_at,
               CASE
                 WHEN outcomes.attempt_id IS NOT NULL THEN outcomes.committed_at
                 ELSE subjects.created_at
               END,
               outcomes.artifact_digest,
               outcomes.gateway_result_digest
        FROM bootstrap_gateway_subjects AS subjects
        JOIN gateway_capabilities AS capabilities
          ON capabilities.attempt_id = subjects.attempt_id
        LEFT JOIN attempt_outcomes AS outcomes
          ON outcomes.attempt_id = subjects.attempt_id
        """
    )
