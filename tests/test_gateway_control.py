"""Tests for persistent Gateway capability and outcome control."""

from __future__ import annotations

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from conftest import NOW, digest, seed_lineage

from atrex_runtime.domain.errors import InfrastructureError, InvalidTransitionError
from atrex_runtime.domain.ids import (
    new_attempt_id,
    new_epoch_id,
    parse_campaign_id,
    parse_epoch_id,
    parse_kernel_agent_revision_id,
    parse_lineage_id,
)
from atrex_runtime.domain.models import (
    Attempt,
    AttemptStatus,
    BranchRole,
    Dsl,
    Epoch,
    EpochStatus,
)
from atrex_runtime.gateway import (
    AttemptTimedWorkerGatewayAuthorityProvider,
    GatewayCapability,
    GatewayCapabilityPolicy,
    GatewayOperation,
    SqliteGatewayControl,
)
from atrex_runtime.gateway.control import (
    BootstrapGatewaySubject,
    BootstrapRunStatus,
    GatewayEvaluationSource,
)
from atrex_runtime.ports import RunAttemptRequest
from atrex_runtime.registry.sqlite import SqliteRegistry

NOW_DATETIME = datetime(2026, 8, 14, tzinfo=UTC)


def test_kernel_trials_retain_exact_candidate_and_revert_annotation(tmp_path: Path) -> None:
    registry = SqliteRegistry(tmp_path / "registry.sqlite")
    attempt = _insert_attempt(registry)
    control = SqliteGatewayControl(
        tmp_path / "gateway.sqlite",
        registry,
        signing_key=b"t" * 32,
        clock=lambda: NOW_DATETIME,
    )
    capability = control.issue(
        attempt.id,
        GatewayCapabilityPolicy(
            frozenset({GatewayOperation.DEV}),
            2,
            NOW_DATETIME + timedelta(hours=1),
        ),
    )
    control.authorize(
        capability,
        GatewayOperation.DEV,
        idempotency_key="dev-reverted-1",
        request_digest=str(digest("dev-request")),
    )
    candidate = digest("reverted-candidate")
    control.bind_operation_candidate(
        attempt.id,
        "dev-reverted-1",
        GatewayOperation.DEV,
        candidate,
    )
    result = digest("dev-result")
    control.commit_operation_artifact(
        attempt.id,
        "dev-reverted-1",
        GatewayOperation.DEV,
        result,
    )
    experiment = {
        "sequence": 1,
        "recorded_at": NOW_DATETIME.isoformat(),
        "name": "failed tile",
        "hypothesis": "larger tile improves reuse",
        "change": "doubled block size",
        "candidate_artifact_digest": str(candidate),
        "evidence": "dev-reverted-1",
        "result": "latency regressed",
        "decision": "revert",
    }

    control.record_kernel_trial_annotations(attempt.id, (experiment,))
    trial = control.list_kernel_trials((attempt.id,))[0]

    assert trial.candidate_artifact_digest == candidate
    assert trial.disposition == "revert"
    assert trial.observations[0].result_artifact_digest == result
    assert trial.annotations[0].experiment["change"] == "doubled block size"
    assert candidate in control.list_referenced_artifact_digests()
    with pytest.raises(ValueError, match="not observed"):
        control.record_kernel_trial_annotations(
            attempt.id,
            ({**experiment, "candidate_artifact_digest": str(digest("foreign"))},),
        )
    control.close()
    registry.close()


def test_multiple_agent_evaluations_are_retained_before_runtime_final_outcome(
    tmp_path: Path,
) -> None:
    registry = SqliteRegistry(tmp_path / "registry.sqlite")
    attempt = _insert_attempt(registry)
    control = SqliteGatewayControl(
        tmp_path / "gateway.sqlite",
        registry,
        signing_key=b"e" * 32,
        clock=lambda: NOW_DATETIME,
    )
    control.issue(
        attempt.id,
        GatewayCapabilityPolicy(
            frozenset({GatewayOperation.EVALUATE}),
            8,
            NOW_DATETIME + timedelta(hours=1),
        ),
    )
    first = control.record_evaluation(
        attempt.id,
        source=GatewayEvaluationSource.AGENT,
        idempotency_key="agent-1",
        candidate_artifact_digest=digest("candidate-1"),
        gateway_result_digest=digest("result-1"),
        correct=False,
        latency_us=None,
        agate_job_id="ev_1",
    )
    second = control.record_evaluation(
        attempt.id,
        source=GatewayEvaluationSource.AGENT,
        idempotency_key="agent-2",
        candidate_artifact_digest=digest("candidate-2"),
        gateway_result_digest=digest("result-2"),
        correct=True,
        latency_us=11.0,
        agate_job_id="ev_2",
    )
    final = control.record_evaluation(
        attempt.id,
        source=GatewayEvaluationSource.RUNTIME_FINAL,
        idempotency_key="runtime-final",
        candidate_artifact_digest=second.candidate_artifact_digest,
        gateway_result_digest=digest("result-final"),
        correct=True,
        latency_us=12.0,
        agate_job_id="ev_final",
    )

    assert control.get_committed_outcome(attempt.id) is None
    outcome = control.commit_authoritative_outcome(
        attempt.id,
        final.id,
        committed_at=NOW_DATETIME,
    )

    assert [item.id for item in control.list_evaluations(attempt.id)] == [
        first.id,
        second.id,
        final.id,
    ]
    assert outcome.artifact_digest == second.candidate_artifact_digest
    assert outcome.gateway_result_digest == digest("result-final")
    with pytest.raises(InvalidTransitionError, match="Runtime-final"):
        control.commit_authoritative_outcome(
            attempt.id,
            second.id,
            committed_at=NOW_DATETIME,
        )
    control.close()
    registry.close()


def test_operation_artifacts_can_be_read_from_campaign_worker_thread(tmp_path: Path) -> None:
    registry = SqliteRegistry(tmp_path / "registry.sqlite")
    attempt = _insert_attempt(registry)
    control = SqliteGatewayControl(
        tmp_path / "gateway.sqlite",
        registry,
        signing_key=b"w" * 32,
        clock=lambda: NOW_DATETIME,
    )
    capability = control.issue(
        attempt.id,
        GatewayCapabilityPolicy(
            frozenset({GatewayOperation.WIKI_QUERY}),
            1,
            NOW_DATETIME + timedelta(hours=1),
        ),
    )
    control.authorize(
        capability,
        GatewayOperation.WIKI_QUERY,
        idempotency_key="wiki-1",
        request_digest=str(digest("wiki-request")),
    )
    control.commit_operation_artifact(
        attempt.id,
        "wiki-1",
        GatewayOperation.WIKI_QUERY,
        digest("wiki-result"),
    )

    with ThreadPoolExecutor(max_workers=1) as executor:
        rows = executor.submit(
            control.list_operation_artifacts,
            (attempt.id,),
            GatewayOperation.WIKI_QUERY,
        ).result()

    assert rows == ((attempt.id, "wiki-1", digest("wiki-result")),)
    control.close()
    registry.close()


def test_bootstrap_subject_can_use_gateway_before_registry_attempt_exists(
    tmp_path: Path,
) -> None:
    registry = SqliteRegistry(tmp_path / "registry.sqlite")
    control = SqliteGatewayControl(
        tmp_path / "gateway.sqlite",
        registry,
        signing_key=b"z" * 32,
        clock=lambda: NOW_DATETIME,
    )
    attempt_id = new_attempt_id()
    subject = BootstrapGatewaySubject(
        attempt_id=attempt_id,
        campaign_id=parse_campaign_id("campaign_" + "1" * 32),
        lineage_id=parse_lineage_id("lineage_" + "2" * 32),
        epoch_id=parse_epoch_id("epoch_" + "3" * 32),
        kernel_agent_revision_id=parse_kernel_agent_revision_id("agentrev_" + "4" * 32),
        operator="vector_add",
        hardware_target="nvidia-h100",
        dsl=Dsl.TRITON,
        evaluation_contract_digest=digest("contract"),
        input_kernel_digest=digest("seed"),
        evidence_digest=digest("evidence"),
        created_at=NOW_DATETIME,
    )
    capability = control.issue_bootstrap(
        subject,
        GatewayCapabilityPolicy(
            frozenset({GatewayOperation.EVALUATE}),
            1,
            NOW_DATETIME + timedelta(hours=1),
        ),
    )

    control.authorize(
        capability,
        GatewayOperation.EVALUATE,
        idempotency_key="baseline",
        request_digest=str(digest("request")),
    )
    final = control.record_evaluation(
        attempt_id,
        source=GatewayEvaluationSource.RUNTIME_FINAL,
        idempotency_key="runtime-final",
        candidate_artifact_digest=digest("candidate"),
        gateway_result_digest=digest("result"),
        correct=True,
        latency_us=9.0,
        agate_job_id="ev_final",
    )
    outcome = control.commit_authoritative_outcome(attempt_id, final.id, committed_at=NOW_DATETIME)

    assert control.get_bootstrap_subject(attempt_id) == subject
    assert control.get_committed_outcome(attempt_id) == outcome
    assert digest("contract") in control.list_referenced_artifact_digests()
    control.close()
    registry.close()


def test_unfinished_bootstrap_retry_rotates_revoked_capability(tmp_path: Path) -> None:
    registry = SqliteRegistry(tmp_path / "registry.sqlite")
    control = SqliteGatewayControl(
        tmp_path / "gateway.sqlite",
        registry,
        signing_key=b"r" * 32,
        clock=lambda: NOW_DATETIME,
    )
    attempt_id = new_attempt_id()
    subject = BootstrapGatewaySubject(
        attempt_id=attempt_id,
        campaign_id=parse_campaign_id("campaign_" + "1" * 32),
        lineage_id=parse_lineage_id("lineage_" + "2" * 32),
        epoch_id=parse_epoch_id("epoch_" + "3" * 32),
        kernel_agent_revision_id=parse_kernel_agent_revision_id("agentrev_" + "4" * 32),
        operator="vector_add",
        hardware_target="nvidia-h100",
        dsl=Dsl.TRITON,
        evaluation_contract_digest=digest("contract"),
        input_kernel_digest=digest("seed"),
        evidence_digest=digest("evidence"),
        created_at=NOW_DATETIME,
    )
    first = control.issue_bootstrap(
        subject,
        GatewayCapabilityPolicy(
            frozenset({GatewayOperation.EVALUATE}),
            1,
            NOW_DATETIME + timedelta(hours=1),
        ),
    )
    control.authorize(
        first,
        GatewayOperation.EVALUATE,
        idempotency_key="baseline",
        request_digest=str(digest("first-request")),
    )
    control.commit_operation_artifact(
        attempt_id,
        "baseline",
        GatewayOperation.EVALUATE,
        digest("first-operation-result"),
    )
    control.revoke(attempt_id)

    second = control.issue_bootstrap(
        subject,
        GatewayCapabilityPolicy(
            frozenset({GatewayOperation.EVALUATE}),
            2,
            NOW_DATETIME + timedelta(hours=2),
        ),
    )

    assert second.token != first.token
    with pytest.raises(PermissionError, match="invalid"):
        control.authorize(
            first,
            GatewayOperation.EVALUATE,
            idempotency_key="stale",
            request_digest=str(digest("stale-request")),
        )
    authorization = control.authorize(
        second,
        GatewayOperation.EVALUATE,
        idempotency_key="baseline",
        request_digest=str(digest("second-request")),
    )
    assert authorization.recovery_generation == 1
    assert control.list_operation_artifacts((attempt_id,), GatewayOperation.EVALUATE) == (
        (attempt_id, "baseline", digest("first-operation-result")),
    )
    runs = control.list_bootstrap_runs(attempt_id)
    assert [run.recovery_generation for run in runs] == [0, 1]
    assert [run.status for run in runs] == [
        BootstrapRunStatus.FAILED,
        BootstrapRunStatus.ISSUED,
    ]
    assert runs[0].finish_reason == "superseded-by-retry"
    assert len(runs[0].operations) == 1
    assert runs[0].operations[0].result_artifact_digest == digest("first-operation-result")
    assert len(runs[1].operations) == 1
    control.close()
    registry.close()


def test_gateway_control_closes_connection_when_initialization_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection = sqlite3.connect(tmp_path / "captured.sqlite")

    def connect(*_args: object, **_kwargs: object) -> sqlite3.Connection:
        return connection

    def fail_migration(_self: SqliteGatewayControl) -> None:
        raise RuntimeError("broken")

    with SqliteRegistry(":memory:") as registry:
        monkeypatch.setattr(sqlite3, "connect", connect)
        monkeypatch.setattr(SqliteGatewayControl, "_migrate", fail_migration)
        with pytest.raises(RuntimeError, match="broken"):
            SqliteGatewayControl(
                tmp_path / "gateway.sqlite",
                registry,
                signing_key=b"k" * 32,
            )

    with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
        connection.execute("SELECT 1")


def test_gateway_schema_v4_migrates_operations_and_legacy_bootstrap_run(
    tmp_path: Path,
) -> None:
    registry = SqliteRegistry(tmp_path / "registry.sqlite")
    database = tmp_path / "gateway.sqlite"
    control = SqliteGatewayControl(
        database,
        registry,
        signing_key=b"m" * 32,
        clock=lambda: NOW_DATETIME,
    )
    attempt_id = new_attempt_id()
    subject = BootstrapGatewaySubject(
        attempt_id=attempt_id,
        campaign_id=parse_campaign_id("campaign_" + "1" * 32),
        lineage_id=parse_lineage_id("lineage_" + "2" * 32),
        epoch_id=parse_epoch_id("epoch_" + "3" * 32),
        kernel_agent_revision_id=parse_kernel_agent_revision_id("agentrev_" + "4" * 32),
        operator="vector_add",
        hardware_target="nvidia-h100",
        dsl=Dsl.TRITON,
        evaluation_contract_digest=digest("contract"),
        input_kernel_digest=digest("seed"),
        evidence_digest=digest("evidence"),
        created_at=NOW_DATETIME,
    )
    capability = control.issue_bootstrap(
        subject,
        GatewayCapabilityPolicy(
            frozenset({GatewayOperation.WIKI_QUERY}),
            1,
            NOW_DATETIME + timedelta(hours=1),
        ),
    )
    control.authorize(
        capability,
        GatewayOperation.WIKI_QUERY,
        idempotency_key="wiki-1",
        request_digest=str(digest("wiki-request")),
    )
    control.commit_operation_artifact(
        attempt_id,
        "wiki-1",
        GatewayOperation.WIKI_QUERY,
        digest("wiki-result"),
    )
    control.revoke(attempt_id)
    control.close()

    with sqlite3.connect(database) as connection:
        connection.execute("DROP TABLE bootstrap_runs")
        connection.execute(
            """
            CREATE TABLE gateway_operations_v4(
                attempt_id TEXT NOT NULL,
                idempotency_key TEXT NOT NULL,
                operation TEXT NOT NULL,
                request_digest TEXT NOT NULL,
                result_artifact_digest TEXT,
                created_at TEXT NOT NULL,
                PRIMARY KEY(attempt_id, idempotency_key),
                FOREIGN KEY(attempt_id) REFERENCES gateway_capabilities(attempt_id)
            )
            """
        )
        connection.execute(
            """INSERT INTO gateway_operations_v4
               SELECT attempt_id, idempotency_key, operation, request_digest,
                      result_artifact_digest, created_at
               FROM gateway_operations"""
        )
        connection.execute("DROP TABLE gateway_operations")
        connection.execute("ALTER TABLE gateway_operations_v4 RENAME TO gateway_operations")
        connection.execute("UPDATE metadata SET value = 4 WHERE key = 'schema_version'")

    migrated = SqliteGatewayControl(
        database,
        registry,
        signing_key=b"m" * 32,
        clock=lambda: NOW_DATETIME,
    )
    runs = migrated.list_bootstrap_runs(attempt_id)
    assert len(runs) == 1
    assert runs[0].status is BootstrapRunStatus.FAILED
    assert runs[0].finish_reason == "migrated-without-outcome"
    assert runs[0].operations[0].idempotency_key == "wiki-1"
    assert migrated.list_operation_artifacts((attempt_id,), GatewayOperation.WIKI_QUERY) == (
        (attempt_id, "wiki-1", digest("wiki-result")),
    )
    with sqlite3.connect(database) as connection:
        version = connection.execute(
            "SELECT value FROM metadata WHERE key = 'schema_version'"
        ).fetchone()
        columns = {row[1] for row in connection.execute("PRAGMA table_info(gateway_operations)")}
        assert version == (9,)
        assert "candidate_artifact_digest" in columns
        measurement_columns = {
            row[1] for row in connection.execute("PRAGMA table_info(gateway_measurements)")
        }
        assert {"candidate_artifact_digest", "kind", "metrics_json"}.issubset(measurement_columns)
    assert "recovery_generation" in columns
    migrated.close()
    registry.close()


def _insert_attempt(registry: SqliteRegistry) -> Attempt:
    seeded = seed_lineage(
        registry,
        evidence_checkpoint=digest("evidence"),
        challenger_count=0,
        attempts_per_trajectory=1,
    )
    epoch = Epoch(
        id=new_epoch_id(),
        lineage_id=seeded.lineage_id,
        number=1,
        active_kernel_agent_revision_id=seeded.active_revision_id,
        challenger_kernel_agent_revision_ids=(),
        starting_kernel_revision_id=seeded.baseline.id,
        evidence_checkpoint=digest("evidence"),
        challenger_count=0,
        trajectories_per_branch=1,
        attempts_per_trajectory=1,
        status=EpochStatus.RUNNING,
        winner_kernel_agent_revision_id=None,
        best_kernel_revision_id=None,
        created_at=NOW,
        completed_at=None,
    )
    registry.insert_epoch(epoch)
    attempt = Attempt(
        id=new_attempt_id(),
        epoch_id=epoch.id,
        branch=BranchRole.ACTIVE,
        challenger_ordinal=0,
        trajectory_ordinal=1,
        ordinal=1,
        kernel_agent_revision_id=seeded.active_revision_id,
        input_kernel_revision_id=seeded.baseline.id,
        attempt_evidence_digest=digest("attempt-evidence"),
        output_kernel_revision_id=None,
        accepted_as_branch_best=False,
        status=AttemptStatus.RUNNING,
        infrastructure_failures=0,
        recovery_generation=0,
        authority_started_at=NOW,
        failure_reason=None,
        created_at=NOW,
        completed_at=None,
    )
    registry.insert_attempt(attempt)
    return attempt


@pytest.mark.anyio
async def test_capability_is_idempotent_scoped_and_persists_outcome(tmp_path: Path) -> None:
    registry = SqliteRegistry(tmp_path / "registry.sqlite")
    attempt = _insert_attempt(registry)
    control = SqliteGatewayControl(
        tmp_path / "gateway.sqlite",
        registry,
        signing_key=b"a" * 32,
        clock=lambda: NOW_DATETIME,
    )
    policy = GatewayCapabilityPolicy(
        frozenset(
            {
                GatewayOperation.DEV,
                GatewayOperation.EVALUATE,
                GatewayOperation.WIKI_QUERY,
            }
        ),
        max_calls=1,
        expires_at=NOW_DATETIME + timedelta(hours=1),
    )

    capability = control.issue(attempt.id, policy)
    assert control.issue(attempt.id, policy) == capability
    authorization = control.authorize(
        capability,
        GatewayOperation.EVALUATE,
        idempotency_key="candidate-1",
        request_digest=str(digest("request-1")),
    )
    assert (
        control.authorize(
            capability,
            GatewayOperation.EVALUATE,
            idempotency_key="candidate-1",
            request_digest=str(digest("request-1")),
        )
        == authorization
    )

    final = control.record_evaluation(
        attempt.id,
        source=GatewayEvaluationSource.RUNTIME_FINAL,
        idempotency_key="runtime-final",
        candidate_artifact_digest=digest("candidate"),
        gateway_result_digest=digest("gateway-result"),
        correct=True,
        latency_us=12.5,
        agate_job_id="ev_final",
    )
    outcome = control.commit_authoritative_outcome(attempt.id, final.id, committed_at=NOW_DATETIME)
    assert await control.get_outcome(attempt.id) == outcome
    assert control.list_referenced_artifact_digests() == {
        digest("candidate"),
        digest("gateway-result"),
    }

    with pytest.raises(PermissionError, match="budget"):
        control.authorize(
            capability,
            GatewayOperation.EVALUATE,
            idempotency_key="candidate-2",
            request_digest=str(digest("request-2")),
        )

    for ordinal in (1, 2):
        control.authorize(
            capability,
            GatewayOperation.WIKI_QUERY,
            idempotency_key=f"wiki-{ordinal}",
            request_digest=str(digest(f"wiki-request-{ordinal}")),
        )

    for ordinal in (1, 2):
        control.authorize(
            capability,
            GatewayOperation.DEV,
            idempotency_key=f"dev-{ordinal}",
            request_digest=str(digest(f"dev-request-{ordinal}")),
        )

    with pytest.raises(PermissionError, match="budget"):
        control.authorize(
            capability,
            GatewayOperation.EVALUATE,
            idempotency_key="candidate-after-unmetered-operations",
            request_digest=str(digest("request-after-unmetered-operations")),
        )

    forged = GatewayCapability(capability.token, new_attempt_id())
    with pytest.raises(PermissionError, match="invalid"):
        control.authorize(
            forged,
            GatewayOperation.EVALUATE,
            idempotency_key="forged",
            request_digest=str(digest("forged")),
        )
    control.close()
    registry.close()


def test_control_reconciles_historical_dev_calls_out_of_metered_usage(tmp_path: Path) -> None:
    registry = SqliteRegistry(tmp_path / "registry.sqlite")
    attempt = _insert_attempt(registry)
    database = tmp_path / "gateway.sqlite"
    policy = GatewayCapabilityPolicy(
        frozenset({GatewayOperation.DEV, GatewayOperation.EVALUATE}),
        max_calls=1,
        expires_at=NOW_DATETIME + timedelta(hours=1),
    )
    control = SqliteGatewayControl(
        database,
        registry,
        signing_key=b"u" * 32,
        clock=lambda: NOW_DATETIME,
    )
    capability = control.issue(attempt.id, policy)
    for ordinal in (1, 2):
        control.authorize(
            capability,
            GatewayOperation.DEV,
            idempotency_key=f"legacy-dev-{ordinal}",
            request_digest=str(digest(f"legacy-dev-request-{ordinal}")),
        )
    control.close()

    with sqlite3.connect(database) as connection:
        connection.execute(
            "UPDATE gateway_capabilities SET used_calls = 2 WHERE attempt_id = ?",
            (attempt.id,),
        )

    reconciled = SqliteGatewayControl(
        database,
        registry,
        signing_key=b"u" * 32,
        clock=lambda: NOW_DATETIME,
    )
    reconciled.authorize(
        capability,
        GatewayOperation.EVALUATE,
        idempotency_key="first-metered-call",
        request_digest=str(digest("first-metered-request")),
    )
    with pytest.raises(PermissionError, match="budget"):
        reconciled.authorize(
            capability,
            GatewayOperation.EVALUATE,
            idempotency_key="second-metered-call",
            request_digest=str(digest("second-metered-request")),
        )
    reconciled.close()
    registry.close()


def test_capability_rejects_policy_change_request_reuse_and_revocation(tmp_path: Path) -> None:
    registry = SqliteRegistry(tmp_path / "registry.sqlite")
    attempt = _insert_attempt(registry)
    control = SqliteGatewayControl(
        tmp_path / "gateway.sqlite",
        registry,
        signing_key=b"b" * 32,
        clock=lambda: NOW_DATETIME,
    )
    policy = GatewayCapabilityPolicy(
        frozenset({GatewayOperation.EVALUATE, GatewayOperation.PROFILE}),
        3,
        NOW_DATETIME + timedelta(hours=1),
    )
    capability = control.issue(attempt.id, policy)
    control.authorize(
        capability,
        GatewayOperation.PROFILE,
        idempotency_key="profile-1",
        request_digest=str(digest("profile-request")),
    )

    with pytest.raises(InvalidTransitionError, match="different request"):
        control.authorize(
            capability,
            GatewayOperation.PROFILE,
            idempotency_key="profile-1",
            request_digest=str(digest("changed-request")),
        )
    with pytest.raises(InvalidTransitionError, match="policy changed"):
        control.issue(
            attempt.id,
            GatewayCapabilityPolicy(
                frozenset({GatewayOperation.EVALUATE}),
                1,
                NOW_DATETIME + timedelta(hours=1),
            ),
        )
    control.revoke(attempt.id)
    with pytest.raises(PermissionError, match="revoked"):
        control.authorize(
            capability,
            GatewayOperation.PROFILE,
            idempotency_key="profile-2",
            request_digest=str(digest("profile-request-2")),
        )
    control.close()
    registry.close()


def test_capability_rotates_signing_key_without_resetting_policy_or_quota(
    tmp_path: Path,
) -> None:
    registry = SqliteRegistry(tmp_path / "registry.sqlite")
    attempt = _insert_attempt(registry)
    database = tmp_path / "gateway.sqlite"
    policy = GatewayCapabilityPolicy(
        frozenset({GatewayOperation.EVALUATE}),
        2,
        NOW_DATETIME + timedelta(hours=1),
    )
    original = SqliteGatewayControl(
        database,
        registry,
        signing_key=b"o" * 32,
        clock=lambda: NOW_DATETIME,
    )
    old_capability = original.issue(attempt.id, policy)
    original.authorize(
        old_capability,
        GatewayOperation.EVALUATE,
        idempotency_key="before-restart",
        request_digest=str(digest("before-restart")),
    )
    original.close()

    restarted = SqliteGatewayControl(
        database,
        registry,
        signing_key=b"n" * 32,
        clock=lambda: NOW_DATETIME,
    )
    new_capability = restarted.issue(attempt.id, policy)
    assert new_capability.token != old_capability.token
    with pytest.raises(PermissionError, match="invalid"):
        restarted.authorize(
            old_capability,
            GatewayOperation.EVALUATE,
            idempotency_key="old-token",
            request_digest=str(digest("old-token")),
        )
    restarted.authorize(
        new_capability,
        GatewayOperation.EVALUATE,
        idempotency_key="after-restart",
        request_digest=str(digest("after-restart")),
    )
    with pytest.raises(PermissionError, match="budget"):
        restarted.authorize(
            new_capability,
            GatewayOperation.EVALUATE,
            idempotency_key="quota-was-reset",
            request_digest=str(digest("quota-was-reset")),
        )
    restarted.close()
    registry.close()


@pytest.mark.anyio
async def test_attempt_timed_authority_is_stable_across_provider_restart(tmp_path: Path) -> None:
    registry = SqliteRegistry(tmp_path / "registry.sqlite")
    attempt = _insert_attempt(registry)
    control = SqliteGatewayControl(
        tmp_path / "gateway.sqlite",
        registry,
        signing_key=b"c" * 32,
        clock=lambda: NOW_DATETIME,
    )
    request = RunAttemptRequest(
        attempt_id=attempt.id,
        kernel_agent_revision_id=attempt.kernel_agent_revision_id,
        input_kernel_revision_id=attempt.input_kernel_revision_id,
        epoch_evidence_checkpoint=digest("evidence"),
        attempt_evidence_digest=attempt.attempt_evidence_digest,
        dsl=registry.get_kernel_agent_revision(attempt.kernel_agent_revision_id).dsl,
    )

    first = await AttemptTimedWorkerGatewayAuthorityProvider(
        control,
        registry,
        "https://runtime.example.test",
        operations=frozenset({GatewayOperation.EVALUATE}),
        max_calls=2,
        lifetime=timedelta(hours=2),
    ).get_authority(request)
    second = await AttemptTimedWorkerGatewayAuthorityProvider(
        control,
        registry,
        "https://runtime.example.test",
        operations=frozenset({GatewayOperation.EVALUATE}),
        max_calls=2,
        lifetime=timedelta(hours=2),
    ).get_authority(request)

    assert second == first
    with pytest.raises(InfrastructureError, match="policy changed"):
        await AttemptTimedWorkerGatewayAuthorityProvider(
            control,
            registry,
            "https://runtime.example.test",
            operations=frozenset({GatewayOperation.EVALUATE}),
            max_calls=3,
            lifetime=timedelta(hours=2),
        ).get_authority(request)
    control.close()
    registry.close()


@pytest.mark.anyio
async def test_attempt_timed_authority_rotates_policy_on_infrastructure_retry(
    tmp_path: Path,
) -> None:
    registry = SqliteRegistry(tmp_path / "registry.sqlite")
    attempt = _insert_attempt(registry)
    control = SqliteGatewayControl(
        tmp_path / "gateway.sqlite",
        registry,
        signing_key=b"r" * 32,
        clock=lambda: NOW_DATETIME,
    )
    request = RunAttemptRequest(
        attempt_id=attempt.id,
        kernel_agent_revision_id=attempt.kernel_agent_revision_id,
        input_kernel_revision_id=attempt.input_kernel_revision_id,
        epoch_evidence_checkpoint=digest("evidence"),
        attempt_evidence_digest=attempt.attempt_evidence_digest,
        dsl=registry.get_kernel_agent_revision(attempt.kernel_agent_revision_id).dsl,
    )
    original = AttemptTimedWorkerGatewayAuthorityProvider(
        control,
        registry,
        "http://runtime.test",
        operations=frozenset({GatewayOperation.EVALUATE}),
        max_calls=2,
        lifetime=timedelta(hours=1),
    )
    old_authority = await original.get_authority(request)

    changed = AttemptTimedWorkerGatewayAuthorityProvider(
        control,
        registry,
        "http://runtime.test",
        operations=frozenset({GatewayOperation.EVALUATE, GatewayOperation.WIKI_QUERY}),
        max_calls=2,
        lifetime=timedelta(hours=1),
    )
    with pytest.raises(InfrastructureError, match="recovery generation"):
        await changed.get_authority(request)

    registry.record_infrastructure_failure(attempt.id, "capability policy changed")
    registry.retry_attempt(attempt.id)
    recovered = registry.get_attempt(attempt.id)
    assert recovered.recovery_generation == 1
    new_authority = await changed.get_authority(request)
    assert new_authority.capability != old_authority.capability
    with pytest.raises(PermissionError, match="invalid Gateway capability"):
        control.authorize(
            GatewayCapability(old_authority.capability, attempt.id),
            GatewayOperation.EVALUATE,
            idempotency_key="old-authority",
            request_digest=str(digest("old-authority")),
        )
    control.authorize(
        GatewayCapability(new_authority.capability, attempt.id, 1),
        GatewayOperation.WIKI_QUERY,
        idempotency_key="wiki-after-recovery",
        request_digest=str(digest("wiki-after-recovery")),
    )
    control.close()
    registry.close()


@pytest.mark.anyio
async def test_failed_epoch_recovery_rotates_capability_and_resets_reservations(
    tmp_path: Path,
) -> None:
    recovery_time = "2026-08-15T00:00:00+00:00"
    registry = SqliteRegistry(tmp_path / "registry.sqlite", clock=lambda: recovery_time)
    attempt = _insert_attempt(registry)
    control = SqliteGatewayControl(
        tmp_path / "gateway.sqlite",
        registry,
        signing_key=b"d" * 32,
        clock=lambda: NOW_DATETIME,
    )
    provider = AttemptTimedWorkerGatewayAuthorityProvider(
        control,
        registry,
        "https://runtime.example.test",
        operations=frozenset({GatewayOperation.EVALUATE}),
        max_calls=1,
        lifetime=timedelta(hours=2),
    )
    request = RunAttemptRequest(
        attempt_id=attempt.id,
        kernel_agent_revision_id=attempt.kernel_agent_revision_id,
        input_kernel_revision_id=attempt.input_kernel_revision_id,
        epoch_evidence_checkpoint=digest("evidence"),
        attempt_evidence_digest=attempt.attempt_evidence_digest,
        dsl=registry.get_kernel_agent_revision(attempt.kernel_agent_revision_id).dsl,
    )
    old_authority = await provider.get_authority(request)
    old_capability = GatewayCapability(old_authority.capability, attempt.id)
    old_authorization = control.authorize(
        old_capability,
        GatewayOperation.EVALUATE,
        idempotency_key="candidate-1",
        request_digest=str(digest("old-request")),
    )

    registry.record_infrastructure_failure(attempt.id, "worker host lost")
    registry.fail_epoch(attempt.epoch_id, "retry budget exhausted")
    registry.recover_failed_epoch(
        attempt.epoch_id,
        recovery_key="incident-001",
        reason="worker host replaced",
    )
    with pytest.raises(PermissionError, match="stale recovery generation"):
        control.authorize(
            old_capability,
            GatewayOperation.EVALUATE,
            idempotency_key="old-after-recovery",
            request_digest=str(digest("old-after-recovery")),
        )
    with pytest.raises(InvalidTransitionError, match="stale recovery generation"):
        control.record_evaluation(
            attempt.id,
            source=GatewayEvaluationSource.RUNTIME_FINAL,
            idempotency_key="stale-runtime-final",
            candidate_artifact_digest=digest("stale-candidate"),
            gateway_result_digest=digest("stale-result"),
            correct=True,
            latency_us=10.0,
            agate_job_id="ev_stale",
            recovery_generation=old_authorization.recovery_generation,
        )

    new_authority = await provider.get_authority(request)
    new_capability = GatewayCapability(new_authority.capability, attempt.id)
    assert new_authority.capability != old_authority.capability
    authorization = control.authorize(
        new_capability,
        GatewayOperation.EVALUATE,
        idempotency_key="candidate-1",
        request_digest=str(digest("new-request")),
    )
    assert authorization.attempt_id == attempt.id
    assert registry.get_attempt(attempt.id).authority_started_at == recovery_time
    control.close()
    registry.close()


def test_visible_history_for_a_missing_attempt_is_a_conflict_not_a_crash(tmp_path: Path) -> None:
    registry = SqliteRegistry(tmp_path / "registry.sqlite")
    control = SqliteGatewayControl(
        tmp_path / "gateway.sqlite",
        registry,
        signing_key=b"t" * 32,
        clock=lambda: NOW_DATETIME,
    )
    vanished = new_attempt_id()

    with pytest.raises(InvalidTransitionError, match="Attempt not found") as raised:
        control.visible_kernel_trial_attempt_ids(vanished)

    assert str(vanished) in str(raised.value)
    control.close()
    registry.close()


def test_bootstrap_subject_sees_only_its_own_kernel_trials(tmp_path: Path) -> None:
    registry = SqliteRegistry(tmp_path / "registry.sqlite")
    control = SqliteGatewayControl(
        tmp_path / "gateway.sqlite",
        registry,
        signing_key=b"z" * 32,
        clock=lambda: NOW_DATETIME,
    )
    attempt_id = new_attempt_id()
    lineage_id = parse_lineage_id("lineage_" + "2" * 32)
    subject = BootstrapGatewaySubject(
        attempt_id=attempt_id,
        campaign_id=parse_campaign_id("campaign_" + "1" * 32),
        lineage_id=lineage_id,
        epoch_id=parse_epoch_id("epoch_" + "3" * 32),
        kernel_agent_revision_id=parse_kernel_agent_revision_id("agentrev_" + "4" * 32),
        operator="vector_add",
        hardware_target="sm_120",
        dsl=Dsl.CUDA,
        evaluation_contract_digest=digest("contract"),
        input_kernel_digest=digest("seed"),
        evidence_digest=digest("evidence"),
        created_at=NOW_DATETIME,
    )
    control.issue_bootstrap(
        subject,
        GatewayCapabilityPolicy(
            frozenset({GatewayOperation.KERNEL_TRIALS}),
            4,
            NOW_DATETIME + timedelta(hours=1),
        ),
    )

    assert control.visible_measurement_attempt_ids(attempt_id) == (lineage_id, ())
    assert control.visible_kernel_trial_attempt_ids(attempt_id) == (lineage_id, (attempt_id,))
    control.close()
    registry.close()
