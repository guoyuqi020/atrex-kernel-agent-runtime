"""SQLite schema creation and migrations for Gateway Control."""

from __future__ import annotations

import sqlite3

GATEWAY_SCHEMA_VERSION = 12


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
                candidate_artifact_digest TEXT,
                gateway_result_digest TEXT,
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
        _create_gateway_measurements_table(connection)
        _create_gateway_trial_annotations_table(connection)
        _create_bootstrap_runs_table(connection)
        _migrate_gateway_v11(connection)
        _migrate_gateway_v12(connection)
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
        _migrate_gateway_v8(connection)
        _migrate_gateway_v9(connection)
        _migrate_gateway_v10(connection)
        _migrate_gateway_v11(connection)
        _migrate_gateway_v12(connection)
        connection.execute(
            "UPDATE metadata SET value = ? WHERE key = 'schema_version'",
            (GATEWAY_SCHEMA_VERSION,),
        )
    elif row["value"] == 4:
        _migrate_gateway_v5(connection)
        _migrate_gateway_v6(connection)
        _migrate_gateway_v7(connection)
        _migrate_gateway_v8(connection)
        _migrate_gateway_v9(connection)
        _migrate_gateway_v10(connection)
        _migrate_gateway_v11(connection)
        _migrate_gateway_v12(connection)
        connection.execute(
            "UPDATE metadata SET value = ? WHERE key = 'schema_version'",
            (GATEWAY_SCHEMA_VERSION,),
        )
    elif row["value"] == 5:
        _migrate_gateway_v6(connection)
        _migrate_gateway_v7(connection)
        _migrate_gateway_v8(connection)
        _migrate_gateway_v9(connection)
        _migrate_gateway_v10(connection)
        _migrate_gateway_v11(connection)
        _migrate_gateway_v12(connection)
        connection.execute(
            "UPDATE metadata SET value = ? WHERE key = 'schema_version'",
            (GATEWAY_SCHEMA_VERSION,),
        )
    elif row["value"] == 6:
        _migrate_gateway_v7(connection)
        _migrate_gateway_v8(connection)
        _migrate_gateway_v9(connection)
        _migrate_gateway_v10(connection)
        _migrate_gateway_v11(connection)
        _migrate_gateway_v12(connection)
        connection.execute(
            "UPDATE metadata SET value = ? WHERE key = 'schema_version'",
            (GATEWAY_SCHEMA_VERSION,),
        )
    elif row["value"] == 7:
        _migrate_gateway_v8(connection)
        _migrate_gateway_v9(connection)
        _migrate_gateway_v10(connection)
        _migrate_gateway_v11(connection)
        _migrate_gateway_v12(connection)
        connection.execute(
            "UPDATE metadata SET value = ? WHERE key = 'schema_version'",
            (GATEWAY_SCHEMA_VERSION,),
        )
    elif row["value"] == 8:
        _migrate_gateway_v9(connection)
        _migrate_gateway_v10(connection)
        _migrate_gateway_v11(connection)
        _migrate_gateway_v12(connection)
        connection.execute(
            "UPDATE metadata SET value = ? WHERE key = 'schema_version'",
            (GATEWAY_SCHEMA_VERSION,),
        )
    elif row["value"] == 9:
        _migrate_gateway_v10(connection)
        _migrate_gateway_v11(connection)
        _migrate_gateway_v12(connection)
        connection.execute(
            "UPDATE metadata SET value = ? WHERE key = 'schema_version'",
            (GATEWAY_SCHEMA_VERSION,),
        )
    elif row["value"] == 10:
        _migrate_gateway_v11(connection)
        _migrate_gateway_v12(connection)
        connection.execute(
            "UPDATE metadata SET value = ? WHERE key = 'schema_version'",
            (GATEWAY_SCHEMA_VERSION,),
        )
    elif row["value"] == 11:
        _migrate_gateway_v12(connection)
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


def _migrate_gateway_v10(connection: sqlite3.Connection) -> None:
    """Distinguish the upstream Gateway result from the cached Proxy response Artifact."""
    columns = {
        str(row["name"])
        for row in connection.execute("PRAGMA table_info(gateway_operations)").fetchall()
    }
    if "gateway_result_digest" not in columns:
        connection.execute("ALTER TABLE gateway_operations ADD COLUMN gateway_result_digest TEXT")


def _migrate_gateway_v11(connection: sqlite3.Connection) -> None:
    """Name sealed candidate Artifacts by their durable Kernel identity."""
    for table in (
        "gateway_operations",
        "gateway_evaluations",
        "gateway_measurements",
        "gateway_trial_annotations",
    ):
        columns = {
            str(row["name"]) for row in connection.execute(f"PRAGMA table_info({table})").fetchall()
        }
        if "candidate_artifact_digest" in columns:
            connection.execute(
                f"ALTER TABLE {table} RENAME COLUMN "
                "candidate_artifact_digest TO kernel_artifact_digest"
            )

    connection.executescript(
        """
        DROP INDEX IF EXISTS gateway_measurements_by_candidate;
        DROP INDEX IF EXISTS gateway_operations_by_candidate;
        DROP INDEX IF EXISTS gateway_trial_annotations_by_candidate;
        CREATE INDEX IF NOT EXISTS gateway_measurements_by_kernel
            ON gateway_measurements(kernel_artifact_digest, created_at, id);
        CREATE INDEX IF NOT EXISTS gateway_operations_by_kernel
            ON gateway_operations(
                attempt_id, recovery_generation, kernel_artifact_digest, created_at
            );
        CREATE INDEX IF NOT EXISTS gateway_trial_annotations_by_kernel
            ON gateway_trial_annotations(
                attempt_id, recovery_generation, kernel_artifact_digest, sequence
            );
        """
    )


def _migrate_gateway_v12(connection: sqlite3.Connection) -> None:
    """Persist live Direction and Experiment Journals at Runtime request boundaries."""
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS runtime_direction_events(
            attempt_id TEXT NOT NULL,
            sequence INTEGER NOT NULL CHECK(sequence > 0),
            recovery_generation INTEGER NOT NULL CHECK(recovery_generation >= 0),
            idempotency_key TEXT NOT NULL,
            direction_event_id TEXT NOT NULL UNIQUE,
            direction_id TEXT NOT NULL,
            event_json TEXT NOT NULL,
            recorded_at TEXT NOT NULL,
            PRIMARY KEY(attempt_id, sequence),
            UNIQUE(attempt_id, idempotency_key),
            FOREIGN KEY(attempt_id) REFERENCES gateway_capabilities(attempt_id)
        );
        CREATE INDEX IF NOT EXISTS runtime_direction_events_by_direction
            ON runtime_direction_events(attempt_id, direction_id, sequence);

        CREATE TABLE IF NOT EXISTS runtime_experiments(
            attempt_id TEXT NOT NULL,
            sequence INTEGER NOT NULL CHECK(sequence > 0),
            recovery_generation INTEGER NOT NULL CHECK(recovery_generation >= 0),
            idempotency_key TEXT NOT NULL,
            experiment_id TEXT NOT NULL UNIQUE,
            direction_id TEXT NOT NULL,
            experiment_json TEXT NOT NULL,
            recorded_at TEXT NOT NULL,
            PRIMARY KEY(attempt_id, sequence),
            UNIQUE(attempt_id, idempotency_key),
            FOREIGN KEY(attempt_id) REFERENCES gateway_capabilities(attempt_id)
        );
        CREATE INDEX IF NOT EXISTS runtime_experiments_by_direction
            ON runtime_experiments(attempt_id, direction_id, sequence);
        """
    )


def _create_gateway_measurements_table(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS gateway_measurements(
            id TEXT PRIMARY KEY,
            attempt_id TEXT NOT NULL,
            recovery_generation INTEGER NOT NULL CHECK(recovery_generation >= 0),
            ordinal INTEGER NOT NULL CHECK(ordinal > 0),
            source_operation TEXT NOT NULL CHECK(source_operation IN ('evaluate', 'profile')),
            idempotency_key TEXT NOT NULL,
            candidate_artifact_digest TEXT NOT NULL,
            gateway_result_digest TEXT NOT NULL,
            kind TEXT NOT NULL CHECK(kind IN ('evaluate', 'profile')),
            profile_level TEXT,
            shape_id TEXT,
            kernel_name TEXT,
            metrics_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            UNIQUE(attempt_id, recovery_generation, idempotency_key, ordinal),
            FOREIGN KEY(attempt_id) REFERENCES gateway_capabilities(attempt_id)
        );
        CREATE INDEX IF NOT EXISTS gateway_measurements_by_attempt
            ON gateway_measurements(attempt_id, recovery_generation, created_at, ordinal);
        """
    )
    columns = {
        str(row["name"])
        for row in connection.execute("PRAGMA table_info(gateway_measurements)").fetchall()
    }
    digest_column = (
        "kernel_artifact_digest"
        if "kernel_artifact_digest" in columns
        else "candidate_artifact_digest"
    )
    index_name = (
        "gateway_measurements_by_kernel"
        if digest_column == "kernel_artifact_digest"
        else "gateway_measurements_by_candidate"
    )
    connection.execute(
        f"CREATE INDEX IF NOT EXISTS {index_name} "
        f"ON gateway_measurements({digest_column}, created_at, id)"
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


def _migrate_gateway_v8(connection: sqlite3.Connection) -> None:
    _create_gateway_measurements_table(connection)


def _migrate_gateway_v9(connection: sqlite3.Connection) -> None:
    """Retain every candidate-bearing operation and its terminal Experiment disposition."""
    columns = {
        str(row["name"])
        for row in connection.execute("PRAGMA table_info(gateway_operations)").fetchall()
    }
    if "candidate_artifact_digest" not in columns:
        connection.execute(
            "ALTER TABLE gateway_operations ADD COLUMN candidate_artifact_digest TEXT"
        )
    evaluation_columns = {
        str(row["name"])
        for row in connection.execute("PRAGMA table_info(gateway_evaluations)").fetchall()
    }
    evaluation_digest_column = (
        "kernel_artifact_digest"
        if "kernel_artifact_digest" in evaluation_columns
        else "candidate_artifact_digest"
    )
    connection.execute(
        f"""
        UPDATE gateway_operations
           SET candidate_artifact_digest = (
               SELECT evaluations.{evaluation_digest_column}
                 FROM gateway_evaluations AS evaluations
                WHERE evaluations.attempt_id = gateway_operations.attempt_id
                  AND evaluations.recovery_generation = gateway_operations.recovery_generation
                  AND evaluations.idempotency_key = gateway_operations.idempotency_key
                LIMIT 1
           )
         WHERE candidate_artifact_digest IS NULL
           AND operation = 'evaluate'
        """
    )
    measurement_columns = {
        str(row["name"])
        for row in connection.execute("PRAGMA table_info(gateway_measurements)").fetchall()
    }
    measurement_digest_column = (
        "kernel_artifact_digest"
        if "kernel_artifact_digest" in measurement_columns
        else "candidate_artifact_digest"
    )
    connection.execute(
        f"""
        UPDATE gateway_operations
           SET candidate_artifact_digest = (
               SELECT measurements.{measurement_digest_column}
                 FROM gateway_measurements AS measurements
                WHERE measurements.attempt_id = gateway_operations.attempt_id
                  AND measurements.recovery_generation = gateway_operations.recovery_generation
                  AND measurements.idempotency_key = gateway_operations.idempotency_key
                LIMIT 1
           )
         WHERE candidate_artifact_digest IS NULL
           AND operation IN ('evaluate', 'profile')
        """
    )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS gateway_operations_by_candidate
            ON gateway_operations(
                attempt_id, recovery_generation, candidate_artifact_digest, created_at
            )
        """
    )
    _create_gateway_trial_annotations_table(connection)


def _create_gateway_trial_annotations_table(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS gateway_trial_annotations(
            attempt_id TEXT NOT NULL,
            recovery_generation INTEGER NOT NULL CHECK(recovery_generation >= 0),
            sequence INTEGER NOT NULL CHECK(sequence > 0),
            candidate_artifact_digest TEXT NOT NULL,
            decision TEXT NOT NULL CHECK(decision IN ('continue', 'revert', 'pivot')),
            experiment_json TEXT NOT NULL,
            recorded_at TEXT NOT NULL,
            PRIMARY KEY(attempt_id, recovery_generation, sequence),
            FOREIGN KEY(attempt_id) REFERENCES gateway_capabilities(attempt_id)
        )
        """
    )
    columns = {
        str(row["name"])
        for row in connection.execute("PRAGMA table_info(gateway_trial_annotations)").fetchall()
    }
    digest_column = (
        "kernel_artifact_digest"
        if "kernel_artifact_digest" in columns
        else "candidate_artifact_digest"
    )
    index_name = (
        "gateway_trial_annotations_by_kernel"
        if digest_column == "kernel_artifact_digest"
        else "gateway_trial_annotations_by_candidate"
    )
    connection.execute(
        f"CREATE INDEX IF NOT EXISTS {index_name} "
        "ON gateway_trial_annotations("
        f"attempt_id, recovery_generation, {digest_column}, sequence)"
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
