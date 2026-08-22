"""Persistent task-scoped Gateway capabilities and authoritative outcomes."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import sqlite3
import threading
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path

from ..domain.errors import (
    GatewayCapabilityPolicyChangedError,
    InfrastructureError,
    InvalidTransitionError,
)
from ..domain.ids import (
    ArtifactDigest,
    AttemptId,
    LineageId,
    parse_artifact_digest,
    parse_attempt_id,
    parse_campaign_id,
    parse_epoch_id,
    parse_kernel_agent_revision_id,
    parse_lineage_id,
)
from ..domain.models import AttemptStatus, Dsl, EpochStatus, TokenUsage
from ..ports import (
    AttemptCandidateResult,
    AttemptOutcomeSource,
    RunAttemptRequest,
    WorkerGatewayAuthority,
)
from ..registry.base import Registry
from ..sqlite_support import configure_durable_sqlite
from .control_models import (
    BootstrapGatewaySubject as BootstrapGatewaySubject,
)
from .control_models import (
    BootstrapRunOperationRecord as BootstrapRunOperationRecord,
)
from .control_models import (
    BootstrapRunRecord as BootstrapRunRecord,
)
from .control_models import (
    BootstrapRunStatus as BootstrapRunStatus,
)
from .control_models import (
    GatewayAuthorization as GatewayAuthorization,
)
from .control_models import (
    GatewayCapability as GatewayCapability,
)
from .control_models import (
    GatewayCapabilityPolicy as GatewayCapabilityPolicy,
)
from .control_models import (
    GatewayEvaluationRecord as GatewayEvaluationRecord,
)
from .control_models import (
    GatewayEvaluationSource as GatewayEvaluationSource,
)
from .control_models import (
    GatewayKernelTrialAnnotation as GatewayKernelTrialAnnotation,
)
from .control_models import (
    GatewayKernelTrialObservation as GatewayKernelTrialObservation,
)
from .control_models import (
    GatewayKernelTrialRecord as GatewayKernelTrialRecord,
)
from .control_models import (
    GatewayMeasurementPoint as GatewayMeasurementPoint,
)
from .control_models import (
    GatewayMeasurementRecord as GatewayMeasurementRecord,
)
from .control_models import (
    GatewayOperation as GatewayOperation,
)
from .control_schema import (
    GATEWAY_SCHEMA_VERSION as GATEWAY_SCHEMA_VERSION,
)
from .control_schema import migrate_gateway_schema


def _utc_now() -> datetime:
    return datetime.now(UTC)


_UNMETERED_OPERATIONS = frozenset(
    {
        GatewayOperation.DEV,
        GatewayOperation.MEASUREMENTS,
        GatewayOperation.KERNEL_TRIALS,
        GatewayOperation.KERNEL_TRIAL_READ,
        GatewayOperation.WIKI_QUERY,
    }
)

_IMPLICIT_READ_OPERATIONS = frozenset(
    {
        GatewayOperation.MEASUREMENTS,
        GatewayOperation.KERNEL_TRIALS,
        GatewayOperation.KERNEL_TRIAL_READ,
    }
)


class SqliteGatewayControl(AttemptOutcomeSource):
    """Persistent capability quota, idempotency, and final Attempt outcome store."""

    def __init__(
        self,
        path: str | Path,
        registry: Registry,
        *,
        signing_key: bytes,
        clock: Callable[[], datetime] = _utc_now,
    ) -> None:
        if len(signing_key) < 32:
            raise ValueError("Gateway capability signing key must contain at least 32 bytes")
        self._registry = registry
        self._signing_key = signing_key
        self._clock = clock
        database_path = Path(path)
        database_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._connection = sqlite3.connect(
            str(database_path),
            isolation_level=None,
            check_same_thread=False,
        )
        try:
            self._connection.row_factory = sqlite3.Row
            configure_durable_sqlite(self._connection)
            self._connection.execute("PRAGMA foreign_keys = ON")
            self._lock = threading.RLock()
            self._migrate()
            self._reconcile_call_usage()
            os.chmod(database_path, 0o600)
        except BaseException:
            self._connection.close()
            raise

    def close(self) -> None:
        """Close the Gateway database after request handling has stopped."""
        with self._lock:
            self._connection.close()

    def check_health(self) -> None:
        """Verify that the Gateway control store can read and acquire a write transaction."""
        with self._transaction() as connection:
            connection.execute("SELECT 1").fetchone()

    def _reconcile_call_usage(self) -> None:
        """Rebuild metered usage when operation accounting policy changes."""
        unmetered = tuple(operation.value for operation in _UNMETERED_OPERATIONS)
        placeholders = ", ".join("?" for _operation in unmetered)
        with self._transaction() as connection:
            connection.execute(
                f"""
                UPDATE gateway_capabilities
                   SET used_calls = (
                       SELECT COUNT(*)
                         FROM gateway_operations
                        WHERE gateway_operations.attempt_id =
                                  gateway_capabilities.attempt_id
                          AND gateway_operations.recovery_generation =
                                  gateway_capabilities.recovery_generation
                          AND gateway_operations.operation NOT IN ({placeholders})
                   )
                """,
                unmetered,
            )

    def list_referenced_artifact_digests(self) -> set[ArtifactDigest]:
        """Return Artifacts retained by operations, every evaluation, and final outcomes."""
        with self._lock:
            rows = self._connection.execute(
                "SELECT artifact_digest, gateway_result_digest FROM attempt_outcomes"
            ).fetchall()
            operation_rows = self._connection.execute(
                """SELECT candidate_artifact_digest, result_artifact_digest
                     FROM gateway_operations
                    WHERE candidate_artifact_digest IS NOT NULL
                       OR result_artifact_digest IS NOT NULL"""
            ).fetchall()
            annotation_rows = self._connection.execute(
                "SELECT candidate_artifact_digest FROM gateway_trial_annotations"
            ).fetchall()
            subject_rows = self._connection.execute(
                """SELECT evaluation_contract_digest, input_kernel_digest, evidence_digest
                   FROM bootstrap_gateway_subjects"""
            ).fetchall()
            bootstrap_run_rows = self._connection.execute(
                """SELECT session_trace_digest, report_digest, candidate_digest,
                          gateway_result_digest
                   FROM bootstrap_runs"""
            ).fetchall()
            evaluation_rows = self._connection.execute(
                """SELECT candidate_artifact_digest, gateway_result_digest
                   FROM gateway_evaluations"""
            ).fetchall()
            measurement_rows = self._connection.execute(
                """SELECT candidate_artifact_digest, gateway_result_digest
                   FROM gateway_measurements"""
            ).fetchall()
        values: set[ArtifactDigest] = set()
        for row in rows:
            for column in ("artifact_digest", "gateway_result_digest"):
                value = row[column]
                if not isinstance(value, str):
                    raise TypeError("persisted Gateway Artifact Digest must be text")
                values.add(parse_artifact_digest(value))
        for row in operation_rows:
            for column in ("candidate_artifact_digest", "result_artifact_digest"):
                value = row[column]
                if value is not None:
                    if not isinstance(value, str):
                        raise TypeError("persisted operation Artifact Digest must be text")
                    values.add(parse_artifact_digest(value))
        for row in annotation_rows:
            value = row["candidate_artifact_digest"]
            if not isinstance(value, str):
                raise TypeError("persisted Kernel Trial Artifact Digest must be text")
            values.add(parse_artifact_digest(value))
        for row in subject_rows:
            for column in (
                "evaluation_contract_digest",
                "input_kernel_digest",
                "evidence_digest",
            ):
                value = row[column]
                if not isinstance(value, str):
                    raise TypeError("persisted bootstrap subject Artifact Digest must be text")
                values.add(parse_artifact_digest(value))
        for row in bootstrap_run_rows:
            for column in (
                "session_trace_digest",
                "report_digest",
                "candidate_digest",
                "gateway_result_digest",
            ):
                value = row[column]
                if value is not None:
                    if not isinstance(value, str):
                        raise TypeError("persisted Bootstrap Run Artifact Digest must be text")
                    values.add(parse_artifact_digest(value))
        for row in evaluation_rows:
            for column in ("candidate_artifact_digest", "gateway_result_digest"):
                value = row[column]
                if not isinstance(value, str):
                    raise TypeError("persisted evaluation Artifact Digest must be text")
                values.add(parse_artifact_digest(value))
        for row in measurement_rows:
            for column in ("candidate_artifact_digest", "gateway_result_digest"):
                value = row[column]
                if not isinstance(value, str):
                    raise TypeError("persisted measurement Artifact Digest must be text")
                values.add(parse_artifact_digest(value))
        return values

    def issue(
        self,
        attempt_id: AttemptId,
        policy: GatewayCapabilityPolicy,
    ) -> GatewayCapability:
        """Issue or recover the deterministic capability for one durable Attempt."""
        attempt = self._registry.get_attempt(attempt_id)
        return self._issue(attempt_id, attempt.recovery_generation, policy)

    def issue_bootstrap(
        self,
        subject: BootstrapGatewaySubject,
        policy: GatewayCapabilityPolicy,
    ) -> GatewayCapability:
        """Register a pre-Lineage subject and rotate authority for each unfinished run."""
        values = (
            subject.campaign_id,
            subject.lineage_id,
            subject.epoch_id,
            subject.kernel_agent_revision_id,
            subject.operator,
            subject.hardware_target,
            subject.dsl.value,
            subject.evaluation_contract_digest,
            subject.input_kernel_digest,
            subject.evidence_digest,
            subject.created_at.astimezone(UTC).isoformat(),
        )
        operations_json = json.dumps(sorted(operation.value for operation in policy.operations))
        expires_at = policy.expires_at.astimezone(UTC).isoformat()
        with self._transaction() as connection:
            subject_row = connection.execute(
                "SELECT * FROM bootstrap_gateway_subjects WHERE attempt_id = ?",
                (subject.attempt_id,),
            ).fetchone()
            if subject_row is None:
                connection.execute(
                    """
                    INSERT INTO bootstrap_gateway_subjects(
                        attempt_id, campaign_id, lineage_id, epoch_id,
                        kernel_agent_revision_id, operator, hardware_target, dsl,
                        evaluation_contract_digest, input_kernel_digest, evidence_digest,
                        created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (subject.attempt_id, *values),
                )
            elif (
                tuple(
                    subject_row[column]
                    for column in (
                        "campaign_id",
                        "lineage_id",
                        "epoch_id",
                        "kernel_agent_revision_id",
                        "operator",
                        "hardware_target",
                        "dsl",
                        "evaluation_contract_digest",
                        "input_kernel_digest",
                        "evidence_digest",
                        "created_at",
                    )
                )
                != values
            ):
                raise InvalidTransitionError(
                    f"bootstrap Gateway subject changed for {subject.attempt_id}"
                )

            capability_row = connection.execute(
                "SELECT * FROM gateway_capabilities WHERE attempt_id = ?",
                (subject.attempt_id,),
            ).fetchone()
            outcome = connection.execute(
                "SELECT 1 FROM attempt_outcomes WHERE attempt_id = ?",
                (subject.attempt_id,),
            ).fetchone()
            if capability_row is None:
                generation = 0
                token = self._token(subject.attempt_id, generation)
                connection.execute(
                    """
                    INSERT INTO gateway_capabilities(
                        attempt_id, recovery_generation, token_hash, operations_json,
                        max_calls, used_calls, expires_at, revoked
                    ) VALUES (?, ?, ?, ?, ?, 0, ?, 0)
                    """,
                    (
                        subject.attempt_id,
                        generation,
                        self._token_hash(token),
                        operations_json,
                        policy.max_calls,
                        expires_at,
                    ),
                )
                create_run = True
            elif outcome is not None:
                generation = int(capability_row["recovery_generation"])
                token = self._token(subject.attempt_id, generation)
                create_run = False
            else:
                previous_generation = int(capability_row["recovery_generation"])
                generation = previous_generation + 1
                token = self._token(subject.attempt_id, generation)
                connection.execute(
                    """UPDATE bootstrap_runs SET status = ?, finish_reason = ?,
                       failure_reason = ?, completed_at = ?
                       WHERE attempt_id = ? AND recovery_generation = ?
                         AND status IN (?, ?)""",
                    (
                        BootstrapRunStatus.FAILED.value,
                        "superseded-by-retry",
                        "Runtime restarted before the Bootstrap generation recorded "
                        "a terminal result",
                        self._clock().astimezone(UTC).isoformat(),
                        subject.attempt_id,
                        previous_generation,
                        BootstrapRunStatus.ISSUED.value,
                        BootstrapRunStatus.RUNNING.value,
                    ),
                )
                connection.execute(
                    """UPDATE gateway_capabilities SET recovery_generation = ?,
                       token_hash = ?, operations_json = ?, max_calls = ?, used_calls = 0,
                       expires_at = ?, revoked = 0 WHERE attempt_id = ?""",
                    (
                        generation,
                        self._token_hash(token),
                        operations_json,
                        policy.max_calls,
                        expires_at,
                        subject.attempt_id,
                    ),
                )
                create_run = True
            if create_run:
                connection.execute(
                    """
                    INSERT INTO bootstrap_runs(
                        attempt_id, recovery_generation, status, started_at
                    ) VALUES (?, ?, ?, ?)
                    """,
                    (
                        subject.attempt_id,
                        generation,
                        BootstrapRunStatus.ISSUED.value,
                        self._clock().astimezone(UTC).isoformat(),
                    ),
                )
        return GatewayCapability(token, subject.attempt_id, generation)

    def get_bootstrap_subject(self, attempt_id: AttemptId) -> BootstrapGatewaySubject:
        """Return the immutable context for one pre-Lineage Gateway subject."""
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM bootstrap_gateway_subjects WHERE attempt_id = ?",
                (attempt_id,),
            ).fetchone()
        if row is None:
            raise KeyError(attempt_id)
        return BootstrapGatewaySubject(
            attempt_id=parse_attempt_id(str(row["attempt_id"])),
            campaign_id=parse_campaign_id(str(row["campaign_id"])),
            lineage_id=parse_lineage_id(str(row["lineage_id"])),
            epoch_id=parse_epoch_id(str(row["epoch_id"])),
            kernel_agent_revision_id=parse_kernel_agent_revision_id(
                str(row["kernel_agent_revision_id"])
            ),
            operator=str(row["operator"]),
            hardware_target=str(row["hardware_target"]),
            dsl=Dsl(str(row["dsl"])),
            evaluation_contract_digest=parse_artifact_digest(
                str(row["evaluation_contract_digest"])
            ),
            input_kernel_digest=parse_artifact_digest(str(row["input_kernel_digest"])),
            evidence_digest=parse_artifact_digest(str(row["evidence_digest"])),
            created_at=datetime.fromisoformat(str(row["created_at"])),
        )

    def _issue(
        self,
        attempt_id: AttemptId,
        generation: int,
        policy: GatewayCapabilityPolicy,
    ) -> GatewayCapability:
        token = self._token(attempt_id, generation)
        token_hash = self._token_hash(token)
        operations_json = json.dumps(sorted(operation.value for operation in policy.operations))
        expires_at = policy.expires_at.astimezone(UTC).isoformat()
        with self._transaction() as connection:
            row = connection.execute(
                "SELECT * FROM gateway_capabilities WHERE attempt_id = ?", (attempt_id,)
            ).fetchone()
            if row is None:
                connection.execute(
                    """
                    INSERT INTO gateway_capabilities(
                        attempt_id, recovery_generation, token_hash, operations_json,
                        max_calls, used_calls, expires_at, revoked
                    ) VALUES (?, ?, ?, ?, ?, 0, ?, 0)
                    """,
                    (
                        attempt_id,
                        generation,
                        token_hash,
                        operations_json,
                        policy.max_calls,
                        expires_at,
                    ),
                )
            else:
                stored_generation = int(row["recovery_generation"])
                if stored_generation > generation:
                    raise InvalidTransitionError(
                        f"Gateway capability is newer than Attempt {attempt_id}"
                    )
                if stored_generation < generation:
                    outcome = connection.execute(
                        "SELECT 1 FROM attempt_outcomes WHERE attempt_id = ?",
                        (attempt_id,),
                    ).fetchone()
                    if outcome is not None:
                        raise InvalidTransitionError(
                            f"Attempt {attempt_id} with a committed outcome cannot rotate"
                        )
                    connection.execute(
                        """UPDATE gateway_capabilities SET recovery_generation = ?,
                           token_hash = ?, operations_json = ?, max_calls = ?, used_calls = 0,
                           expires_at = ?, revoked = 0 WHERE attempt_id = ?""",
                        (
                            generation,
                            token_hash,
                            operations_json,
                            policy.max_calls,
                            expires_at,
                            attempt_id,
                        ),
                    )
                elif (
                    row["operations_json"] != operations_json
                    or row["max_calls"] != policy.max_calls
                    or row["expires_at"] != expires_at
                ):
                    raise GatewayCapabilityPolicyChangedError(
                        f"Gateway capability policy changed for Attempt {attempt_id}"
                    )
                elif row["token_hash"] != token_hash:
                    # The signing key is Runtime infrastructure, not part of the
                    # durable Attempt policy.  A restarted, exclusively fenced
                    # scheduler may rotate it without resetting quota or losing
                    # idempotency history.  The old token becomes invalid as soon
                    # as this transaction commits.
                    connection.execute(
                        """UPDATE gateway_capabilities SET token_hash = ?
                           WHERE attempt_id = ? AND recovery_generation = ?""",
                        (token_hash, attempt_id, generation),
                    )
        return GatewayCapability(token, attempt_id, generation)

    def authorize(
        self,
        capability: GatewayCapability,
        operation: GatewayOperation,
        *,
        idempotency_key: str,
        request_digest: str,
    ) -> GatewayAuthorization:
        """Authorize one idempotent operation and consume quota only on first reservation."""
        if not idempotency_key:
            raise ValueError("Gateway operation idempotency key cannot be empty")
        request_digest = str(parse_artifact_digest(request_digest))
        expected_hash = self._token_hash(capability.token)
        with self._transaction() as connection:
            row = connection.execute(
                "SELECT * FROM gateway_capabilities WHERE attempt_id = ?",
                (capability.attempt_id,),
            ).fetchone()
            if row is None or not hmac.compare_digest(row["token_hash"], expected_hash):
                raise PermissionError("invalid Gateway capability")
            generation = self._subject_generation(capability.attempt_id)
            if int(row["recovery_generation"]) != generation:
                raise PermissionError("Gateway capability belongs to a stale recovery generation")
            if row["revoked"]:
                raise PermissionError("Gateway capability is revoked")
            if self._clock() >= datetime.fromisoformat(row["expires_at"]):
                raise PermissionError("Gateway capability has expired")
            allowed = json.loads(row["operations_json"])
            if operation not in _IMPLICIT_READ_OPERATIONS and operation.value not in allowed:
                raise PermissionError(f"Gateway operation is not allowed: {operation.value}")

            existing = connection.execute(
                """
                SELECT operation, request_digest FROM gateway_operations
                WHERE attempt_id = ? AND recovery_generation = ? AND idempotency_key = ?
                """,
                (capability.attempt_id, generation, idempotency_key),
            ).fetchone()
            if existing is not None:
                if (
                    existing["operation"] != operation.value
                    or existing["request_digest"] != request_digest
                ):
                    raise InvalidTransitionError(
                        "Gateway idempotency key was reused for a different request"
                    )
            else:
                metered = operation not in _UNMETERED_OPERATIONS
                if metered and row["used_calls"] >= row["max_calls"]:
                    raise PermissionError("Gateway capability call budget is exhausted")
                connection.execute(
                    """
                    INSERT INTO gateway_operations(
                        attempt_id, recovery_generation, idempotency_key, operation,
                        request_digest, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        capability.attempt_id,
                        generation,
                        idempotency_key,
                        operation.value,
                        request_digest,
                        self._clock().isoformat(),
                    ),
                )
                if metered:
                    connection.execute(
                        """
                        UPDATE gateway_capabilities SET used_calls = used_calls + 1
                        WHERE attempt_id = ?
                        """,
                        (capability.attempt_id,),
                    )
        return GatewayAuthorization(
            capability.attempt_id,
            operation,
            idempotency_key,
            request_digest,
            generation,
        )

    def get_operation_artifact(
        self,
        attempt_id: AttemptId,
        idempotency_key: str,
        operation: GatewayOperation,
    ) -> ArtifactDigest | None:
        """Return the immutable result committed for one authorized operation."""
        generation = self._subject_generation(attempt_id)
        with self._lock:
            row = self._connection.execute(
                """SELECT result_artifact_digest FROM gateway_operations
                   WHERE attempt_id = ? AND recovery_generation = ?
                     AND idempotency_key = ? AND operation = ?""",
                (attempt_id, generation, idempotency_key, operation.value),
            ).fetchone()
        if row is None or row["result_artifact_digest"] is None:
            return None
        value = row["result_artifact_digest"]
        if not isinstance(value, str):
            raise TypeError("persisted operation result Artifact Digest must be text")
        return parse_artifact_digest(value)

    def bind_operation_candidate(
        self,
        attempt_id: AttemptId,
        idempotency_key: str,
        operation: GatewayOperation,
        candidate_artifact_digest: ArtifactDigest,
        *,
        recovery_generation: int | None = None,
    ) -> ArtifactDigest:
        """Bind an authorized operation to the exact candidate before external execution."""
        candidate = parse_artifact_digest(str(candidate_artifact_digest))
        generation = self._subject_generation(attempt_id)
        if recovery_generation is not None and recovery_generation != generation:
            raise InvalidTransitionError("Gateway candidate belongs to a stale generation")
        with self._transaction() as connection:
            row = connection.execute(
                """SELECT candidate_artifact_digest FROM gateway_operations
                   WHERE attempt_id = ? AND recovery_generation = ?
                     AND idempotency_key = ? AND operation = ?""",
                (attempt_id, generation, idempotency_key, operation.value),
            ).fetchone()
            if row is None:
                raise InvalidTransitionError("operation candidate has no authorization reservation")
            existing = row["candidate_artifact_digest"]
            if existing is None:
                connection.execute(
                    """UPDATE gateway_operations SET candidate_artifact_digest = ?
                       WHERE attempt_id = ? AND recovery_generation = ?
                         AND idempotency_key = ?""",
                    (candidate, attempt_id, generation, idempotency_key),
                )
                return candidate
            if existing != candidate:
                raise InvalidTransitionError(
                    "authorized operation already has a different candidate Artifact"
                )
            if not isinstance(existing, str):
                raise TypeError("persisted operation candidate Artifact Digest must be text")
            return parse_artifact_digest(existing)

    def commit_operation_artifact(
        self,
        attempt_id: AttemptId,
        idempotency_key: str,
        operation: GatewayOperation,
        artifact_digest: ArtifactDigest,
    ) -> ArtifactDigest:
        """Bind an authorized operation to exactly one immutable result Artifact."""
        digest = parse_artifact_digest(str(artifact_digest))
        generation = self._subject_generation(attempt_id)
        with self._transaction() as connection:
            row = connection.execute(
                """SELECT result_artifact_digest FROM gateway_operations
                   WHERE attempt_id = ? AND recovery_generation = ?
                     AND idempotency_key = ? AND operation = ?""",
                (attempt_id, generation, idempotency_key, operation.value),
            ).fetchone()
            if row is None:
                raise InvalidTransitionError("operation Artifact has no authorization reservation")
            existing = row["result_artifact_digest"]
            if existing is None:
                connection.execute(
                    """UPDATE gateway_operations SET result_artifact_digest = ?
                       WHERE attempt_id = ? AND recovery_generation = ?
                         AND idempotency_key = ?""",
                    (digest, attempt_id, generation, idempotency_key),
                )
                return digest
            if existing != digest:
                raise InvalidTransitionError(
                    "authorized operation already has a different result Artifact"
                )
            if not isinstance(existing, str):
                raise TypeError("persisted operation result Artifact Digest must be text")
            return parse_artifact_digest(existing)

    def record_kernel_trial_annotations(
        self,
        attempt_id: AttemptId,
        experiments: Sequence[Mapping[str, object]],
        *,
        recovery_generation: int | None = None,
    ) -> tuple[GatewayKernelTrialAnnotation, ...]:
        """Validate and retain Agent experiment decisions against observed candidates."""
        generation = self._subject_generation(attempt_id)
        if recovery_generation is not None and recovery_generation != generation:
            raise InvalidTransitionError("Kernel Trial annotations belong to a stale generation")
        records: list[GatewayKernelTrialAnnotation] = []
        with self._transaction() as connection:
            for expected_sequence, experiment in enumerate(experiments, 1):
                sequence = experiment.get("sequence")
                digest_value = experiment.get("candidate_artifact_digest")
                decision = experiment.get("decision")
                recorded_at = experiment.get("recorded_at")
                if sequence != expected_sequence:
                    raise ValueError("Kernel Trial annotation sequence must be contiguous")
                if digest_value is None:
                    continue
                if not isinstance(digest_value, str):
                    raise ValueError("Kernel Trial annotation requires a candidate Artifact Digest")
                candidate = parse_artifact_digest(digest_value)
                if decision not in {"continue", "revert", "pivot"}:
                    raise ValueError("Kernel Trial annotation decision is invalid")
                if not isinstance(recorded_at, str) or not recorded_at:
                    raise ValueError("Kernel Trial annotation requires recorded_at")
                observed = connection.execute(
                    """SELECT 1 FROM gateway_operations
                       WHERE attempt_id = ? AND recovery_generation = ?
                         AND candidate_artifact_digest = ? LIMIT 1""",
                    (attempt_id, generation, candidate),
                ).fetchone()
                if observed is None:
                    raise ValueError(
                        "Kernel Trial annotation references a candidate not observed "
                        "in this Attempt"
                    )
                payload = json.dumps(
                    dict(experiment),
                    ensure_ascii=False,
                    allow_nan=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                values = (str(candidate), str(decision), payload, recorded_at)
                existing = connection.execute(
                    """SELECT * FROM gateway_trial_annotations
                       WHERE attempt_id = ? AND recovery_generation = ? AND sequence = ?""",
                    (attempt_id, generation, sequence),
                ).fetchone()
                if existing is None:
                    connection.execute(
                        """INSERT INTO gateway_trial_annotations(
                               attempt_id, recovery_generation, sequence,
                               candidate_artifact_digest, decision, experiment_json, recorded_at
                           ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
                        (attempt_id, generation, sequence, *values),
                    )
                elif (
                    tuple(
                        existing[column]
                        for column in (
                            "candidate_artifact_digest",
                            "decision",
                            "experiment_json",
                            "recorded_at",
                        )
                    )
                    != values
                ):
                    raise InvalidTransitionError(
                        "Kernel Trial annotation sequence resolved to different evidence"
                    )
                records.append(
                    GatewayKernelTrialAnnotation(
                        sequence=sequence,
                        decision=str(decision),
                        experiment=json.loads(payload),
                        recorded_at=recorded_at,
                    )
                )
        return tuple(records)

    def list_kernel_trials(
        self,
        attempt_ids: tuple[AttemptId, ...],
        *,
        limit: int = 1_000,
    ) -> tuple[GatewayKernelTrialRecord, ...]:
        """List durable exact candidate snapshots and all observations, oldest first."""
        if limit <= 0 or limit > 5_000:
            raise ValueError("Kernel Trial query limit must be between 1 and 5000")
        if not attempt_ids:
            return ()
        placeholders = ",".join("?" for _attempt in attempt_ids)
        with self._lock:
            candidates = self._connection.execute(
                f"""SELECT attempt_id, recovery_generation, candidate_artifact_digest,
                            MIN(created_at) AS first_created_at
                       FROM gateway_operations
                      WHERE attempt_id IN ({placeholders})
                        AND candidate_artifact_digest IS NOT NULL
                      GROUP BY attempt_id, recovery_generation, candidate_artifact_digest
                      ORDER BY first_created_at, attempt_id, recovery_generation,
                               candidate_artifact_digest
                      LIMIT ?""",
                (*attempt_ids, limit),
            ).fetchall()
            records: list[GatewayKernelTrialRecord] = []
            ordinals: dict[tuple[str, int], int] = {}
            for candidate_row in candidates:
                attempt = parse_attempt_id(str(candidate_row["attempt_id"]))
                generation = int(candidate_row["recovery_generation"])
                candidate = parse_artifact_digest(str(candidate_row["candidate_artifact_digest"]))
                key = (str(attempt), generation)
                ordinal = ordinals.get(key, 0) + 1
                ordinals[key] = ordinal
                operation_rows = self._connection.execute(
                    """SELECT idempotency_key, operation, request_digest,
                              result_artifact_digest, created_at
                         FROM gateway_operations
                        WHERE attempt_id = ? AND recovery_generation = ?
                          AND candidate_artifact_digest = ?
                        ORDER BY created_at, idempotency_key""",
                    (attempt, generation, candidate),
                ).fetchall()
                annotation_rows = self._connection.execute(
                    """SELECT sequence, decision, experiment_json, recorded_at
                         FROM gateway_trial_annotations
                        WHERE attempt_id = ? AND recovery_generation = ?
                          AND candidate_artifact_digest = ?
                        ORDER BY sequence""",
                    (attempt, generation, candidate),
                ).fetchall()
                identity = hashlib.sha256(
                    f"{attempt}:{generation}:{candidate}".encode()
                ).hexdigest()[:32]
                records.append(
                    GatewayKernelTrialRecord(
                        id=f"gtrial_{identity}",
                        attempt_id=attempt,
                        recovery_generation=generation,
                        ordinal=ordinal,
                        candidate_artifact_digest=candidate,
                        observations=tuple(
                            GatewayKernelTrialObservation(
                                idempotency_key=str(row["idempotency_key"]),
                                operation=GatewayOperation(str(row["operation"])),
                                request_digest=parse_artifact_digest(str(row["request_digest"])),
                                result_artifact_digest=(
                                    None
                                    if row["result_artifact_digest"] is None
                                    else parse_artifact_digest(str(row["result_artifact_digest"]))
                                ),
                                created_at=str(row["created_at"]),
                            )
                            for row in operation_rows
                        ),
                        annotations=tuple(
                            GatewayKernelTrialAnnotation(
                                sequence=int(row["sequence"]),
                                decision=str(row["decision"]),
                                experiment=json.loads(str(row["experiment_json"])),
                                recorded_at=str(row["recorded_at"]),
                            )
                            for row in annotation_rows
                        ),
                        created_at=str(candidate_row["first_created_at"]),
                    )
                )
        return tuple(records)

    def get_kernel_trial(self, trial_id: str) -> GatewayKernelTrialRecord:
        """Return one Kernel Trial by deterministic identity."""
        if not trial_id.startswith("gtrial_"):
            raise KeyError(trial_id)
        with self._lock:
            attempts = tuple(
                parse_attempt_id(str(row["attempt_id"]))
                for row in self._connection.execute(
                    """SELECT DISTINCT attempt_id FROM gateway_operations
                       WHERE candidate_artifact_digest IS NOT NULL"""
                ).fetchall()
            )
        for trial in self.list_kernel_trials(attempts, limit=5_000):
            if trial.id == trial_id:
                return trial
        raise KeyError(trial_id)

    def list_operation_artifacts(
        self,
        attempt_ids: tuple[AttemptId, ...],
        operation: GatewayOperation,
    ) -> tuple[tuple[AttemptId, str, ArtifactDigest], ...]:
        """List committed operation Artifacts in deterministic Attempt and call order."""
        if not attempt_ids:
            return ()
        placeholders = ",".join("?" for _value in attempt_ids)
        parameters: tuple[object, ...] = (*attempt_ids, operation.value)
        with self._lock:
            rows = self._connection.execute(
                f"""SELECT attempt_id, idempotency_key, result_artifact_digest
                    FROM gateway_operations
                    WHERE attempt_id IN ({placeholders}) AND operation = ?
                      AND result_artifact_digest IS NOT NULL
                    ORDER BY attempt_id, recovery_generation, created_at, idempotency_key""",
                parameters,
            ).fetchall()
        values: list[tuple[AttemptId, str, ArtifactDigest]] = []
        for row in rows:
            attempt_id_value = row["attempt_id"]
            key = row["idempotency_key"]
            digest_value = row["result_artifact_digest"]
            if not all(isinstance(value, str) for value in (attempt_id_value, key, digest_value)):
                raise TypeError("persisted operation Artifact row must contain text")
            values.append(
                (
                    parse_attempt_id(attempt_id_value),
                    key,
                    parse_artifact_digest(digest_value),
                )
            )
        return tuple(values)

    def current_generation(self, attempt_id: AttemptId) -> int:
        """Return the control-plane generation currently authoritative for an Attempt."""
        return self._subject_generation(attempt_id)

    def record_evaluation(
        self,
        attempt_id: AttemptId,
        *,
        source: GatewayEvaluationSource,
        idempotency_key: str,
        candidate_artifact_digest: ArtifactDigest,
        gateway_result_digest: ArtifactDigest,
        correct: bool,
        latency_us: float | None,
        agate_job_id: str | None,
        recovery_generation: int | None = None,
    ) -> GatewayEvaluationRecord:
        """Append one evaluated Kernel/result pair, idempotently within one generation."""
        if not idempotency_key:
            raise ValueError("Gateway evaluation idempotency key cannot be empty")
        if correct and (latency_us is None or latency_us <= 0):
            raise ValueError("a correct Gateway evaluation requires positive latency")
        if not correct and latency_us is not None:
            raise ValueError("an incorrect Gateway evaluation cannot carry latency")
        candidate = parse_artifact_digest(str(candidate_artifact_digest))
        result = parse_artifact_digest(str(gateway_result_digest))
        generation = self._subject_generation(attempt_id)
        if recovery_generation is not None and recovery_generation != generation:
            raise InvalidTransitionError(
                "Gateway evaluation belongs to a stale recovery generation"
            )
        identity = hashlib.sha256(
            f"{attempt_id}:{generation}:{source.value}:{idempotency_key}".encode()
        ).hexdigest()[:32]
        evaluation_id = f"geval_{identity}"
        values = (
            source.value,
            idempotency_key,
            str(candidate),
            str(result),
            int(correct),
            latency_us,
            agate_job_id,
        )
        with self._transaction() as connection:
            existing = connection.execute(
                """SELECT * FROM gateway_evaluations
                   WHERE attempt_id = ? AND recovery_generation = ?
                     AND source = ? AND idempotency_key = ?""",
                (attempt_id, generation, source.value, idempotency_key),
            ).fetchone()
            if existing is None:
                ordinal_row = connection.execute(
                    """SELECT COALESCE(MAX(ordinal), 0) AS value
                       FROM gateway_evaluations
                       WHERE attempt_id = ? AND recovery_generation = ?""",
                    (attempt_id, generation),
                ).fetchone()
                if ordinal_row is None:
                    raise AssertionError("Gateway evaluation ordinal query returned no row")
                ordinal = int(ordinal_row["value"]) + 1
                connection.execute(
                    """INSERT INTO gateway_evaluations(
                           id, attempt_id, recovery_generation, ordinal, source,
                           idempotency_key, candidate_artifact_digest,
                           gateway_result_digest, correct, latency_us, agate_job_id,
                           created_at
                       ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        evaluation_id,
                        attempt_id,
                        generation,
                        ordinal,
                        *values,
                        self._clock().astimezone(UTC).isoformat(),
                    ),
                )
            else:
                persisted = tuple(
                    existing[column]
                    for column in (
                        "source",
                        "idempotency_key",
                        "candidate_artifact_digest",
                        "gateway_result_digest",
                        "correct",
                        "latency_us",
                        "agate_job_id",
                    )
                )
                if persisted != values or str(existing["id"]) != evaluation_id:
                    raise InvalidTransitionError(
                        "Gateway evaluation idempotency key resolved to different evidence"
                    )
        return self.get_evaluation(evaluation_id)

    def get_evaluation(self, evaluation_id: str) -> GatewayEvaluationRecord:
        """Return one immutable exploration or Runtime-final evaluation."""
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM gateway_evaluations WHERE id = ?", (evaluation_id,)
            ).fetchone()
        if row is None:
            raise KeyError(evaluation_id)
        return self._evaluation_from_row(row)

    def list_evaluations(
        self,
        attempt_id: AttemptId,
        *,
        recovery_generation: int | None = None,
    ) -> tuple[GatewayEvaluationRecord, ...]:
        """Return every retained evaluation in stable generation/ordinal order."""
        self._subject_generation(attempt_id)
        query = "SELECT * FROM gateway_evaluations WHERE attempt_id = ?"
        parameters: tuple[object, ...] = (attempt_id,)
        if recovery_generation is not None:
            if recovery_generation < 0:
                raise ValueError("recovery generation cannot be negative")
            query += " AND recovery_generation = ?"
            parameters = (attempt_id, recovery_generation)
        query += " ORDER BY recovery_generation, ordinal"
        with self._lock:
            rows = self._connection.execute(query, parameters).fetchall()
        return tuple(self._evaluation_from_row(row) for row in rows)

    def record_measurements(
        self,
        attempt_id: AttemptId,
        *,
        source_operation: GatewayOperation,
        idempotency_key: str,
        candidate_artifact_digest: ArtifactDigest,
        gateway_result_digest: ArtifactDigest,
        points: tuple[GatewayMeasurementPoint, ...],
        recovery_generation: int | None = None,
    ) -> tuple[GatewayMeasurementRecord, ...]:
        """Persist normalized measurement points idempotently with their raw evidence identity."""
        if source_operation not in {GatewayOperation.EVALUATE, GatewayOperation.PROFILE}:
            raise ValueError("measurement source operation must be evaluate or profile")
        if not idempotency_key:
            raise ValueError("measurement idempotency key cannot be empty")
        if not points:
            return ()
        generation = self._subject_generation(attempt_id)
        if recovery_generation is not None and recovery_generation != generation:
            raise InvalidTransitionError("Gateway measurements belong to a stale generation")
        candidate = parse_artifact_digest(str(candidate_artifact_digest))
        result = parse_artifact_digest(str(gateway_result_digest))
        created_at = self._clock().astimezone(UTC).isoformat()
        records: list[GatewayMeasurementRecord] = []
        with self._transaction() as connection:
            for ordinal, point in enumerate(points, 1):
                identity = hashlib.sha256(
                    f"{attempt_id}:{generation}:{idempotency_key}:{ordinal}".encode()
                ).hexdigest()[:32]
                measurement_id = f"gmeasure_{identity}"
                metrics_json = json.dumps(
                    point.metrics,
                    ensure_ascii=False,
                    allow_nan=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                values = (
                    measurement_id,
                    str(attempt_id),
                    generation,
                    ordinal,
                    source_operation.value,
                    idempotency_key,
                    str(candidate),
                    str(result),
                    point.kind.value,
                    point.profile_level,
                    point.shape_id,
                    point.kernel_name,
                    metrics_json,
                    created_at,
                )
                existing = connection.execute(
                    "SELECT * FROM gateway_measurements WHERE id = ?",
                    (measurement_id,),
                ).fetchone()
                if existing is None:
                    connection.execute(
                        """INSERT INTO gateway_measurements VALUES
                           (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        values,
                    )
                elif tuple(existing) != values:
                    raise InvalidTransitionError(
                        "Gateway measurement identity resolved to different evidence"
                    )
                records.append(
                    GatewayMeasurementRecord(
                        id=measurement_id,
                        attempt_id=attempt_id,
                        recovery_generation=generation,
                        ordinal=ordinal,
                        source_operation=source_operation,
                        idempotency_key=idempotency_key,
                        candidate_artifact_digest=candidate,
                        gateway_result_digest=result,
                        point=point,
                        created_at=created_at,
                    )
                )
        return tuple(records)

    def visible_measurement_attempt_ids(
        self,
        current_attempt_id: AttemptId,
    ) -> tuple[LineageId, tuple[AttemptId, ...]]:
        """Return exactly the promoted and same-trajectory history visible to an Optimizer."""
        current = self._registry.get_attempt(current_attempt_id)
        current_epoch = self._registry.get_epoch(current.epoch_id)
        lineage = self._registry.get_lineage(current_epoch.lineage_id)
        visible: list[AttemptId] = []
        for epoch in self._registry.list_epochs(lineage.id):
            if epoch.number > current_epoch.number:
                continue
            attempts = self._registry.list_attempts(epoch.id)
            if epoch.number < current_epoch.number:
                if (
                    epoch.status is not EpochStatus.COMPLETED
                    or epoch.winner_kernel_agent_revision_id is None
                ):
                    continue
                if epoch.winner_kernel_agent_revision_id == epoch.active_kernel_agent_revision_id:
                    selected_branch = "active"
                    selected_challenger = 0
                else:
                    try:
                        selected_challenger = (
                            epoch.challenger_kernel_agent_revision_ids.index(
                                epoch.winner_kernel_agent_revision_id
                            )
                            + 1
                        )
                    except ValueError as error:
                        raise RuntimeError(
                            "completed Epoch winner is outside its Branch pool"
                        ) from error
                    selected_branch = "challenger"
                visible.extend(
                    attempt.id
                    for attempt in attempts
                    if attempt.status is AttemptStatus.COMPLETED
                    and attempt.branch.value == selected_branch
                    and attempt.challenger_ordinal == selected_challenger
                )
            else:
                visible.extend(
                    attempt.id
                    for attempt in attempts
                    if attempt.status is AttemptStatus.COMPLETED
                    and attempt.branch is current.branch
                    and attempt.challenger_ordinal == current.challenger_ordinal
                    and attempt.trajectory_ordinal == current.trajectory_ordinal
                    and attempt.ordinal < current.ordinal
                )
        return lineage.id, tuple(visible)

    def visible_kernel_trial_attempt_ids(
        self,
        current_attempt_id: AttemptId,
    ) -> tuple[LineageId, tuple[AttemptId, ...]]:
        """Return visible history plus the caller's own in-progress Kernel Trials."""
        lineage_id, visible = self.visible_measurement_attempt_ids(current_attempt_id)
        return lineage_id, (*visible, current_attempt_id)

    def list_measurements(
        self,
        attempt_ids: tuple[AttemptId, ...],
        *,
        kind: GatewayOperation | None = None,
        candidate_artifact_digest: ArtifactDigest | None = None,
        shape_id: str | None = None,
        kernel_name: str | None = None,
        metric: str | None = None,
        limit: int = 50,
    ) -> tuple[GatewayMeasurementRecord, ...]:
        """Query a bounded caller-supplied visibility set, newest first."""
        if limit <= 0 or limit > 5_000:
            raise ValueError("measurement query limit must be between 1 and 5000")
        if kind is not None and kind not in {GatewayOperation.EVALUATE, GatewayOperation.PROFILE}:
            raise ValueError("measurement query kind must be evaluate or profile")
        if not attempt_ids:
            return ()
        placeholders = ",".join("?" for _attempt in attempt_ids)
        query = f"SELECT * FROM gateway_measurements WHERE attempt_id IN ({placeholders})"
        parameters: list[object] = [str(attempt_id) for attempt_id in attempt_ids]
        if kind is not None:
            query += " AND kind = ?"
            parameters.append(kind.value)
        if candidate_artifact_digest is not None:
            query += " AND candidate_artifact_digest = ?"
            parameters.append(str(parse_artifact_digest(str(candidate_artifact_digest))))
        if shape_id is not None:
            query += " AND shape_id = ?"
            parameters.append(shape_id)
        if kernel_name is not None:
            query += " AND kernel_name = ?"
            parameters.append(kernel_name)
        if metric is not None:
            query += " AND EXISTS (SELECT 1 FROM json_each(metrics_json) WHERE key = ?)"
            parameters.append(metric)
        query += " ORDER BY created_at DESC, attempt_id DESC, ordinal DESC LIMIT ?"
        parameters.append(limit)
        with self._lock:
            rows = self._connection.execute(query, tuple(parameters)).fetchall()
        return tuple(self._measurement_from_row(row) for row in rows)

    def measurement_kernel_catalog(
        self,
        lineage_id: LineageId,
    ) -> dict[ArtifactDigest, tuple[str, int, str]]:
        """Map sealed Kernel content to its lineage-local revision identity and version."""
        self._registry.get_lineage(lineage_id)
        return {
            entry.revision.artifact_digest: (
                str(entry.revision.id),
                entry.revision_number,
                f"v{entry.revision_number}",
            )
            for entry in self._registry.list_lineage_kernels(lineage_id)
        }

    def list_evidence_measurements(
        self,
        attempt_ids: tuple[AttemptId, ...],
        *,
        limit: int,
    ) -> tuple[GatewayMeasurementRecord, ...]:
        """Return bounded normalized rows for one frozen Evidence projection."""
        return self.list_measurements(attempt_ids, limit=limit)

    def find_agent_evaluation(
        self,
        attempt_id: AttemptId,
        candidate_artifact_digest: ArtifactDigest,
        *,
        gateway_result_digest: ArtifactDigest | None = None,
        recovery_generation: int | None = None,
    ) -> GatewayEvaluationRecord | None:
        """Find the newest Agent evaluation for an exact nominated Kernel/result."""
        current_generation = self._subject_generation(attempt_id)
        generation = current_generation if recovery_generation is None else recovery_generation
        if generation < 0 or generation > current_generation:
            raise InvalidTransitionError("Agent evaluation lookup uses an invalid generation")
        query = """SELECT * FROM gateway_evaluations
                   WHERE attempt_id = ? AND recovery_generation = ? AND source = ?
                     AND candidate_artifact_digest = ?"""
        parameters: list[object] = [
            attempt_id,
            generation,
            GatewayEvaluationSource.AGENT.value,
            str(parse_artifact_digest(str(candidate_artifact_digest))),
        ]
        if gateway_result_digest is not None:
            query += " AND gateway_result_digest = ?"
            parameters.append(str(parse_artifact_digest(str(gateway_result_digest))))
        query += " ORDER BY ordinal DESC LIMIT 1"
        with self._lock:
            row = self._connection.execute(query, tuple(parameters)).fetchone()
        return None if row is None else self._evaluation_from_row(row)

    def find_runtime_final_evaluation(
        self,
        attempt_id: AttemptId,
        idempotency_key: str,
    ) -> GatewayEvaluationRecord | None:
        """Recover one independently executed final evaluation by stable key."""
        generation = self._subject_generation(attempt_id)
        with self._lock:
            row = self._connection.execute(
                """SELECT * FROM gateway_evaluations
                   WHERE attempt_id = ? AND recovery_generation = ?
                     AND source = ? AND idempotency_key = ?""",
                (
                    attempt_id,
                    generation,
                    GatewayEvaluationSource.RUNTIME_FINAL.value,
                    idempotency_key,
                ),
            ).fetchone()
        return None if row is None else self._evaluation_from_row(row)

    def commit_authoritative_outcome(
        self,
        attempt_id: AttemptId,
        evaluation_id: str,
        *,
        committed_at: datetime,
    ) -> AttemptCandidateResult:
        """Commit only an independently executed Runtime-final evaluation as authority."""
        if committed_at.tzinfo is None:
            raise ValueError("Gateway outcome commit time must be timezone-aware")
        evaluation = self.get_evaluation(evaluation_id)
        generation = self._subject_generation(attempt_id)
        if (
            evaluation.attempt_id != attempt_id
            or evaluation.recovery_generation != generation
            or evaluation.source is not GatewayEvaluationSource.RUNTIME_FINAL
        ):
            raise InvalidTransitionError(
                "authoritative outcome requires a current Runtime-final evaluation"
            )
        outcome = AttemptCandidateResult(
            artifact_digest=evaluation.candidate_artifact_digest,
            gateway_result_digest=evaluation.gateway_result_digest,
            correct=evaluation.correct,
            latency_us=evaluation.latency_us,
        )
        values = (
            str(outcome.artifact_digest),
            str(outcome.gateway_result_digest),
            int(outcome.correct),
            outcome.latency_us,
            evaluation.id,
        )
        with self._transaction() as connection:
            existing = connection.execute(
                "SELECT * FROM attempt_outcomes WHERE attempt_id = ?", (attempt_id,)
            ).fetchone()
            if existing is None:
                connection.execute(
                    """INSERT INTO attempt_outcomes(
                           attempt_id, artifact_digest, gateway_result_digest, correct,
                           latency_us, committed_at, source_evaluation_id
                       ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (
                        attempt_id,
                        *values[:4],
                        committed_at.astimezone(UTC).isoformat(),
                        values[4],
                    ),
                )
            elif (
                tuple(
                    existing[column]
                    for column in (
                        "artifact_digest",
                        "gateway_result_digest",
                        "correct",
                        "latency_us",
                        "source_evaluation_id",
                    )
                )
                != values
            ):
                raise InvalidTransitionError(
                    f"Attempt {attempt_id} already has a different authoritative outcome"
                )
        return outcome

    @staticmethod
    def _evaluation_from_row(row: sqlite3.Row) -> GatewayEvaluationRecord:
        return GatewayEvaluationRecord(
            id=str(row["id"]),
            attempt_id=parse_attempt_id(str(row["attempt_id"])),
            recovery_generation=int(row["recovery_generation"]),
            ordinal=int(row["ordinal"]),
            source=GatewayEvaluationSource(str(row["source"])),
            idempotency_key=str(row["idempotency_key"]),
            candidate_artifact_digest=parse_artifact_digest(str(row["candidate_artifact_digest"])),
            gateway_result_digest=parse_artifact_digest(str(row["gateway_result_digest"])),
            correct=bool(row["correct"]),
            latency_us=None if row["latency_us"] is None else float(row["latency_us"]),
            agate_job_id=(None if row["agate_job_id"] is None else str(row["agate_job_id"])),
            created_at=str(row["created_at"]),
        )

    @staticmethod
    def _measurement_from_row(row: sqlite3.Row) -> GatewayMeasurementRecord:
        raw_metrics: object = json.loads(str(row["metrics_json"]))
        if not isinstance(raw_metrics, dict) or any(
            not isinstance(key, str) or isinstance(value, (dict, list))
            for key, value in raw_metrics.items()
        ):
            raise TypeError("persisted Gateway measurement metrics are invalid")
        metrics = {str(key): value for key, value in raw_metrics.items()}
        return GatewayMeasurementRecord(
            id=str(row["id"]),
            attempt_id=parse_attempt_id(str(row["attempt_id"])),
            recovery_generation=int(row["recovery_generation"]),
            ordinal=int(row["ordinal"]),
            source_operation=GatewayOperation(str(row["source_operation"])),
            idempotency_key=str(row["idempotency_key"]),
            candidate_artifact_digest=parse_artifact_digest(str(row["candidate_artifact_digest"])),
            gateway_result_digest=parse_artifact_digest(str(row["gateway_result_digest"])),
            point=GatewayMeasurementPoint(
                kind=GatewayOperation(str(row["kind"])),
                profile_level=(None if row["profile_level"] is None else str(row["profile_level"])),
                shape_id=None if row["shape_id"] is None else str(row["shape_id"]),
                kernel_name=None if row["kernel_name"] is None else str(row["kernel_name"]),
                metrics=metrics,
            ),
            created_at=str(row["created_at"]),
        )

    async def get_outcome(self, attempt_id: AttemptId) -> AttemptCandidateResult | None:
        """Return a durable authoritative outcome for Runner recovery."""
        return self.get_committed_outcome(attempt_id)

    def get_committed_outcome(self, attempt_id: AttemptId) -> AttemptCandidateResult | None:
        """Synchronously return an Attempt or bootstrap subject's committed outcome."""
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM attempt_outcomes WHERE attempt_id = ?", (attempt_id,)
            ).fetchone()
        if row is None:
            return None
        return AttemptCandidateResult(
            artifact_digest=parse_artifact_digest(row["artifact_digest"]),
            gateway_result_digest=parse_artifact_digest(row["gateway_result_digest"]),
            correct=bool(row["correct"]),
            latency_us=row["latency_us"],
        )

    def begin_bootstrap_run(
        self,
        attempt_id: AttemptId,
        recovery_generation: int,
        *,
        run_id: str,
        workspace_path: str,
    ) -> BootstrapRunRecord:
        """Bind one issued Bootstrap generation to its physical Session workspace."""
        if recovery_generation < 0:
            raise ValueError("Bootstrap recovery generation cannot be negative")
        if not run_id.strip() or not workspace_path.strip():
            raise ValueError("Bootstrap run identity and workspace path cannot be blank")
        with self._transaction() as connection:
            row = connection.execute(
                """SELECT * FROM bootstrap_runs
                   WHERE attempt_id = ? AND recovery_generation = ?""",
                (attempt_id, recovery_generation),
            ).fetchone()
            if row is None:
                raise KeyError((attempt_id, recovery_generation))
            if row["status"] == BootstrapRunStatus.ISSUED.value:
                connection.execute(
                    """UPDATE bootstrap_runs SET status = ?, run_id = ?, workspace_path = ?
                       WHERE attempt_id = ? AND recovery_generation = ?""",
                    (
                        BootstrapRunStatus.RUNNING.value,
                        run_id,
                        workspace_path,
                        attempt_id,
                        recovery_generation,
                    ),
                )
            elif (
                row["status"] != BootstrapRunStatus.RUNNING.value
                or row["run_id"] != run_id
                or row["workspace_path"] != workspace_path
            ):
                raise InvalidTransitionError("Bootstrap run cannot be rebound")
        return self.get_bootstrap_run(attempt_id, recovery_generation)

    def finish_bootstrap_run(
        self,
        attempt_id: AttemptId,
        recovery_generation: int,
        *,
        status: BootstrapRunStatus,
        finish_reason: str,
        failure_reason: str | None,
        session_trace_digest: ArtifactDigest | None = None,
        token_budget: int | None = None,
        token_usage: TokenUsage | None = None,
        report_digest: ArtifactDigest | None = None,
        candidate_digest: ArtifactDigest | None = None,
        gateway_result_digest: ArtifactDigest | None = None,
    ) -> BootstrapRunRecord:
        """Commit exactly one terminal result for a Bootstrap generation."""
        if status not in {BootstrapRunStatus.COMPLETED, BootstrapRunStatus.FAILED}:
            raise ValueError("Bootstrap run terminal status must be completed or failed")
        if not finish_reason.strip():
            raise ValueError("Bootstrap run finish reason cannot be blank")
        if status is BootstrapRunStatus.FAILED:
            if failure_reason is None or not failure_reason.strip():
                raise ValueError("Failed Bootstrap run requires a failure reason")
        elif failure_reason is not None:
            raise ValueError("Completed Bootstrap run cannot have a failure reason")
        if (token_budget is None) != (token_usage is None):
            raise ValueError("Bootstrap token budget and usage must be recorded together")
        if token_budget is not None and token_budget <= 0:
            raise ValueError("Bootstrap token budget must be positive")
        digests = (
            session_trace_digest,
            report_digest,
            candidate_digest,
            gateway_result_digest,
        )
        normalized = tuple(
            None if digest is None else parse_artifact_digest(str(digest)) for digest in digests
        )
        usage_values: tuple[int | float | None, ...]
        if token_usage is None:
            usage_values = (None, None, None, None, None)
        else:
            usage_values = (
                token_usage.uncached_input_tokens,
                token_usage.cache_read_tokens,
                token_usage.cache_write_tokens,
                token_usage.output_tokens,
                token_usage.credits,
            )
        values = (
            status.value,
            finish_reason,
            failure_reason,
            self._clock().astimezone(UTC).isoformat(),
            *normalized,
            token_budget,
            *usage_values,
        )
        with self._transaction() as connection:
            row = connection.execute(
                """SELECT * FROM bootstrap_runs
                   WHERE attempt_id = ? AND recovery_generation = ?""",
                (attempt_id, recovery_generation),
            ).fetchone()
            if row is None:
                raise KeyError((attempt_id, recovery_generation))
            if row["status"] in {
                BootstrapRunStatus.ISSUED.value,
                BootstrapRunStatus.RUNNING.value,
            }:
                connection.execute(
                    """UPDATE bootstrap_runs SET
                       status = ?, finish_reason = ?, failure_reason = ?, completed_at = ?,
                       session_trace_digest = ?, report_digest = ?, candidate_digest = ?,
                       gateway_result_digest = ?, token_budget = ?, uncached_input_tokens = ?,
                       cache_read_tokens = ?, cache_write_tokens = ?, output_tokens = ?,
                       credits = ?
                       WHERE attempt_id = ? AND recovery_generation = ?""",
                    (*values, attempt_id, recovery_generation),
                )
            else:
                raise InvalidTransitionError("Bootstrap run already has a terminal result")
        return self.get_bootstrap_run(attempt_id, recovery_generation)

    def get_bootstrap_run(
        self, attempt_id: AttemptId, recovery_generation: int
    ) -> BootstrapRunRecord:
        """Return one exact Bootstrap generation."""
        with self._lock:
            row = self._connection.execute(
                """SELECT * FROM bootstrap_runs
                   WHERE attempt_id = ? AND recovery_generation = ?""",
                (attempt_id, recovery_generation),
            ).fetchone()
        if row is None:
            raise KeyError((attempt_id, recovery_generation))
        return self._with_bootstrap_operations(self._bootstrap_run_from_row(row))

    def list_bootstrap_runs(self, attempt_id: AttemptId) -> tuple[BootstrapRunRecord, ...]:
        """Return every retained generation for one stable Bootstrap Attempt."""
        self.get_bootstrap_subject(attempt_id)
        with self._lock:
            rows = self._connection.execute(
                """SELECT * FROM bootstrap_runs WHERE attempt_id = ?
                   ORDER BY recovery_generation""",
                (attempt_id,),
            ).fetchall()
        return tuple(
            self._with_bootstrap_operations(self._bootstrap_run_from_row(row)) for row in rows
        )

    def _with_bootstrap_operations(self, run: BootstrapRunRecord) -> BootstrapRunRecord:
        with self._lock:
            rows = self._connection.execute(
                """SELECT idempotency_key, operation, request_digest,
                          result_artifact_digest, created_at
                   FROM gateway_operations
                   WHERE attempt_id = ? AND recovery_generation = ?
                   ORDER BY created_at, idempotency_key""",
                (run.attempt_id, run.recovery_generation),
            ).fetchall()
        operations = tuple(
            BootstrapRunOperationRecord(
                idempotency_key=str(row["idempotency_key"]),
                operation=GatewayOperation(str(row["operation"])),
                request_digest=parse_artifact_digest(str(row["request_digest"])),
                result_artifact_digest=(
                    None
                    if row["result_artifact_digest"] is None
                    else parse_artifact_digest(str(row["result_artifact_digest"]))
                ),
                created_at=str(row["created_at"]),
            )
            for row in rows
        )
        return BootstrapRunRecord(
            attempt_id=run.attempt_id,
            recovery_generation=run.recovery_generation,
            status=run.status,
            run_id=run.run_id,
            workspace_path=run.workspace_path,
            finish_reason=run.finish_reason,
            failure_reason=run.failure_reason,
            started_at=run.started_at,
            completed_at=run.completed_at,
            session_trace_digest=run.session_trace_digest,
            token_budget=run.token_budget,
            uncached_input_tokens=run.uncached_input_tokens,
            cache_read_tokens=run.cache_read_tokens,
            cache_write_tokens=run.cache_write_tokens,
            output_tokens=run.output_tokens,
            credits=run.credits,
            report_digest=run.report_digest,
            candidate_digest=run.candidate_digest,
            gateway_result_digest=run.gateway_result_digest,
            operations=operations,
        )

    @staticmethod
    def _bootstrap_run_from_row(row: sqlite3.Row) -> BootstrapRunRecord:
        def optional_digest(column: str) -> ArtifactDigest | None:
            value = row[column]
            return None if value is None else parse_artifact_digest(str(value))

        return BootstrapRunRecord(
            attempt_id=parse_attempt_id(str(row["attempt_id"])),
            recovery_generation=int(row["recovery_generation"]),
            status=BootstrapRunStatus(str(row["status"])),
            run_id=None if row["run_id"] is None else str(row["run_id"]),
            workspace_path=(None if row["workspace_path"] is None else str(row["workspace_path"])),
            finish_reason=(None if row["finish_reason"] is None else str(row["finish_reason"])),
            failure_reason=(None if row["failure_reason"] is None else str(row["failure_reason"])),
            started_at=str(row["started_at"]),
            completed_at=None if row["completed_at"] is None else str(row["completed_at"]),
            session_trace_digest=optional_digest("session_trace_digest"),
            token_budget=None if row["token_budget"] is None else int(row["token_budget"]),
            uncached_input_tokens=(
                None if row["uncached_input_tokens"] is None else int(row["uncached_input_tokens"])
            ),
            cache_read_tokens=(
                None if row["cache_read_tokens"] is None else int(row["cache_read_tokens"])
            ),
            cache_write_tokens=(
                None if row["cache_write_tokens"] is None else int(row["cache_write_tokens"])
            ),
            output_tokens=(None if row["output_tokens"] is None else int(row["output_tokens"])),
            credits=None if row["credits"] is None else float(row["credits"]),
            report_digest=optional_digest("report_digest"),
            candidate_digest=optional_digest("candidate_digest"),
            gateway_result_digest=optional_digest("gateway_result_digest"),
        )

    def _subject_generation(self, attempt_id: AttemptId) -> int:
        try:
            return self._registry.get_attempt(attempt_id).recovery_generation
        except KeyError:
            self.get_bootstrap_subject(attempt_id)
            with self._lock:
                row = self._connection.execute(
                    "SELECT recovery_generation FROM gateway_capabilities WHERE attempt_id = ?",
                    (attempt_id,),
                ).fetchone()
            if row is None:
                raise KeyError(attempt_id) from None
            return int(row["recovery_generation"])

    def revoke(self, attempt_id: AttemptId) -> None:
        """Permanently disable a capability after Attempt completion or quarantine."""
        with self._transaction() as connection:
            changed = connection.execute(
                "UPDATE gateway_capabilities SET revoked = 1 WHERE attempt_id = ?",
                (attempt_id,),
            ).rowcount
            if changed != 1:
                raise KeyError(attempt_id)

    def _token(self, attempt_id: AttemptId, recovery_generation: int) -> str:
        signature = hmac.new(
            self._signing_key,
            f"{attempt_id}:{recovery_generation}".encode(),
            hashlib.sha256,
        ).hexdigest()
        return f"gcap_{signature}"

    @staticmethod
    def _token_hash(token: str) -> str:
        return hashlib.sha256(token.encode()).hexdigest()

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                yield self._connection
            except BaseException:
                self._connection.rollback()
                raise
            else:
                self._connection.commit()

    def _migrate(self) -> None:
        with self._transaction() as connection:
            migrate_gateway_schema(connection)


class AttemptTimedWorkerGatewayAuthorityProvider:
    """Issue restart-stable policies relative to durable Attempt creation time."""

    def __init__(
        self,
        control: SqliteGatewayControl,
        registry: Registry,
        endpoint: str,
        *,
        operations: frozenset[GatewayOperation],
        max_calls: int,
        lifetime: timedelta,
    ) -> None:
        if not endpoint.startswith(("http://", "https://")):
            raise ValueError("Gateway Proxy endpoint must use HTTP or HTTPS")
        if not operations:
            raise ValueError("Gateway capability requires at least one operation")
        if max_calls <= 0:
            raise ValueError("Gateway capability max_calls must be positive")
        if lifetime <= timedelta(0):
            raise ValueError("Gateway capability lifetime must be positive")
        self._control = control
        self._registry = registry
        self._endpoint = endpoint
        self._operations = operations
        self._max_calls = max_calls
        self._lifetime = lifetime

    async def get_authority(self, request: RunAttemptRequest) -> WorkerGatewayAuthority:
        """Derive expiry from persisted Attempt time so restart cannot change policy."""
        attempt = self._registry.get_attempt(request.attempt_id)
        authority_started_at = datetime.fromisoformat(attempt.authority_started_at)
        if authority_started_at.tzinfo is None:
            raise ValueError("Attempt authority start time must be timezone-aware")
        try:
            capability = self._control.issue(
                request.attempt_id,
                GatewayCapabilityPolicy(
                    self._operations,
                    self._max_calls,
                    authority_started_at + self._lifetime,
                ),
            )
        except GatewayCapabilityPolicyChangedError as error:
            raise InfrastructureError(
                "Gateway capability policy changed; rotate the Attempt recovery generation"
            ) from error
        return WorkerGatewayAuthority(self._endpoint, capability.token)
