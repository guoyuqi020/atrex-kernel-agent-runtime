"""Restore Bootstrap facts without restoring optimizer progress or execution authority."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from ..artifacts.local import LocalArtifactStore
from ..domain.ids import ArtifactDigest, LineageId, parse_artifact_digest
from ..registry.base import Registry
from .control_schema import migrate_gateway_schema

# Order follows foreign keys. Running jobs, dedup reservations, leases and capability
# tokens are deliberately absent: this is a history import, never execution recovery.
_HISTORY_TABLES = (
    "bootstrap_gateway_subjects",
    "bootstrap_runs",
    "gateway_operations",
    "gateway_evaluations",
    "gateway_measurements",
    "gateway_trial_annotations",
    "runtime_direction_events",
    "runtime_experiments",
    "attempt_outcomes",
)


def _bootstrap_lineage_ids(registry: Registry, lineage_id: LineageId) -> list[LineageId]:
    """Return one Bootstrap ancestry, rejecting a corrupt cycle."""
    lineage_ids: list[LineageId] = []
    current: LineageId | None = lineage_id
    while current is not None:
        if current in lineage_ids:
            raise ValueError("Bootstrap Lineage ancestry contains a cycle")
        lineage_ids.append(current)
        current = registry.get_lineage(current).bootstrap_source_lineage_id
    return lineage_ids


def _bootstrap_rows(
    source: sqlite3.Connection,
    lineage_ids: list[LineageId],
) -> tuple[list[sqlite3.Row], dict[str, list[dict[str, object]]]]:
    """Select only durable Bootstrap facts from one ancestry."""
    subjects = source.execute(
        "SELECT * FROM bootstrap_gateway_subjects WHERE lineage_id IN ("
        + ",".join("?" for _ in lineage_ids)
        + ")",
        lineage_ids,
    ).fetchall()
    ids = [str(row["attempt_id"]) for row in subjects]
    rows: dict[str, list[dict[str, object]]] = {table: [] for table in _HISTORY_TABLES}
    if not ids:
        return subjects, rows
    placeholders = ",".join("?" for _ in ids)
    rows = {}
    for table in _HISTORY_TABLES:
        condition = ""
        if table == "gateway_operations":
            condition = " AND result_artifact_digest IS NOT NULL"
        elif table == "bootstrap_runs":
            condition = " AND status IN ('completed', 'failed')"
        rows[table] = [
            dict(row)
            for row in source.execute(
                f"SELECT * FROM {table} WHERE attempt_id IN ({placeholders}){condition}", ids
            )
        ]
    return subjects, rows


def _bootstrap_artifact_roots(
    rows: dict[str, list[dict[str, object]]],
) -> set[ArtifactDigest]:
    """Collect every directly referenced Artifact in restored Bootstrap facts."""
    roots: set[ArtifactDigest] = set()
    for values in rows.values():
        for row in values:
            for key, value in row.items():
                if key.endswith("digest") and key != "request_digest" and value is not None:
                    roots.add(parse_artifact_digest(str(value)))
    for table, field in (
        ("runtime_experiments", "experiment_json"),
        ("runtime_direction_events", "event_json"),
    ):
        for row in rows[table]:

            def digests(value: object) -> None:
                if isinstance(value, str) and value.startswith("sha256:"):
                    roots.add(parse_artifact_digest(value))
                elif isinstance(value, dict):
                    for child in value.values():
                        digests(child)
                elif isinstance(value, list):
                    for child in value:
                        digests(child)

            digests(json.loads(str(row[field])))
    return roots


def verify_bootstrap_history_artifacts(
    source: sqlite3.Connection,
    registry: Registry,
    lineage_id: LineageId,
    artifacts: LocalArtifactStore,
) -> int:
    """Fail before arm creation when restored Bootstrap facts have dangling CAS roots."""
    lineage_ids = _bootstrap_lineage_ids(registry, lineage_id)
    _subjects, rows = _bootstrap_rows(source, lineage_ids)
    roots = _bootstrap_artifact_roots(rows)
    roots.add(registry.get_lineage(lineage_id).evidence_checkpoint)
    try:
        artifacts.expand_reference_closure(roots)
    except (FileNotFoundError, ValueError) as error:
        raise ValueError(
            "Bootstrap snapshot has an unavailable Artifact; export a consistent "
            "database and CAS snapshot before seeding an ablation arm: " + str(error)
        ) from error
    return len(roots)


def restore_bootstrap_history(
    destination: sqlite3.Connection,
    registry: Registry,
    lineage_id: LineageId,
    source_database: Path,
    source_artifacts: LocalArtifactStore,
    artifacts: LocalArtifactStore,
) -> dict[str, int]:
    """Import only trusted Bootstrap ancestry into an existing transaction.

    Call with a quiescent destination. IDs and measurements stay unchanged; only
    revoked capability placeholders are created to satisfy historical foreign keys.
    The source is snapshotted read-only and migrated in memory, never modified.
    Conflicting facts or missing Artifacts abort the database transaction.
    """
    lineage_ids = _bootstrap_lineage_ids(registry, lineage_id)
    source = sqlite3.connect(":memory:")
    source.row_factory = sqlite3.Row
    try:
        with sqlite3.connect(source_database.resolve().as_uri() + "?mode=ro", uri=True) as disk:
            disk.backup(source)
        migrate_gateway_schema(source)
        subjects, rows = _bootstrap_rows(source, lineage_ids)
        for subject in subjects:
            lineage = registry.get_lineage(subject["lineage_id"])
            campaign = registry.get_campaign(lineage.campaign_id)
            if (
                subject["campaign_id"] != campaign.id
                or subject["dsl"] != lineage.dsl.value
                or subject["operator"] != campaign.operator
                or subject["hardware_target"] != campaign.hardware_target
                or subject["evaluation_contract_digest"] != campaign.evaluation_contract_digest
            ):
                raise ValueError("Bootstrap history disagrees with its frozen source Campaign")
        ids = [str(row["attempt_id"]) for row in subjects]
        # A copied report is not authority to widen history. Require its Bootstrap
        # subject to exist in the explicitly restored ancestry instead of silently
        # creating an evidence pack whose IDs can never be read.
        checkpoint_roots: set[ArtifactDigest] = set()
        for ancestor_id in lineage_ids:
            checkpoint = registry.get_lineage(ancestor_id).evidence_checkpoint
            checkpoint_roots.add(checkpoint)
            try:
                evidence = artifacts.verify(checkpoint)
            except FileNotFoundError:
                evidence = source_artifacts.verify(checkpoint)
            report_path = evidence.payload_path / "bootstrap" / "report.json"
            if report_path.is_file():
                report = json.loads(report_path.read_text())
                reported_id = report.get("attempt_id") if isinstance(report, dict) else None
                if reported_id is not None and reported_id not in ids:
                    raise ValueError(
                        "Bootstrap evidence references an Attempt outside its restored ancestry: "
                        + str(reported_id)
                    )
        counts = dict.fromkeys(_HISTORY_TABLES, 0)
        if not ids:
            return counts
        roots = _bootstrap_artifact_roots(rows) | checkpoint_roots
        retained: set[ArtifactDigest] = set()
        pending = list(roots)
        while pending:
            digest = pending.pop()
            if digest in retained:
                continue
            try:
                stored = artifacts.verify(digest)
            except FileNotFoundError:
                try:
                    stored = source_artifacts.verify(digest)
                except FileNotFoundError:
                    if digest in roots:
                        raise
                    continue
                if artifacts.put_directory(stored.payload_path, stored.kind) != digest:
                    raise ValueError("Restored Bootstrap Artifact changed identity") from None
            retained.add(digest)
            for path in stored.payload_path.rglob("*"):
                if path.is_file():
                    pending.extend(LocalArtifactStore._file_digest_values(path) - retained)

        # Preserve no token hashes from the source. Old Bootstrap tokens must remain
        # unusable even when restoring into a store that already contains their rows.
        for attempt_id in ids:
            cap = source.execute(
                "SELECT * FROM gateway_capabilities WHERE attempt_id=?", (attempt_id,)
            ).fetchone()
            if cap is None:
                raise ValueError("Bootstrap history has no capability record")
            destination.execute(
                """INSERT INTO gateway_capabilities VALUES (?, ?, ?, '[]', 1, 0, ?, 1)
                   ON CONFLICT(attempt_id) DO UPDATE SET
                       token_hash=excluded.token_hash, operations_json='[]', revoked=1""",
                (attempt_id, cap["recovery_generation"], "0" * 64, cap["expires_at"]),
            )
        for table, values in rows.items():
            primary = [
                str(r["name"])
                for r in destination.execute(f"PRAGMA table_info({table})")
                if r["pk"]
            ]
            for row in values:
                existing = destination.execute(
                    f"SELECT * FROM {table} WHERE " + " AND ".join(f"{key}=?" for key in primary),
                    [row[key] for key in primary],
                ).fetchone()
                if existing is not None:
                    if dict(existing) != row:
                        raise ValueError(f"Conflicting Bootstrap history in {table}")
                    continue
                columns = list(row)
                destination.execute(
                    f"INSERT INTO {table} ({','.join(columns)}) VALUES "
                    f"({','.join('?' for _ in columns)})",
                    list(row.values()),
                )
                counts[table] += 1
        return counts
    finally:
        source.close()
