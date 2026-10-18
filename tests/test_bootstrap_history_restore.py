"""A reset preserves Bootstrap evidence and IDs, but cannot revive other work."""

from __future__ import annotations

import base64
import json
import sqlite3
from datetime import timedelta
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient
from test_gateway_proxy import NOW_DATETIME, _request, _service

from atrex_runtime.artifacts.local import ArtifactKind, LocalArtifactStore
from atrex_runtime.domain.ids import new_attempt_id, parse_epoch_id
from atrex_runtime.gateway import (
    BootstrapGatewaySubject,
    GatewayCapabilityPolicy,
    GatewayOperation,
    GatewayProxyAsgiApp,
    GatewayProxyLimits,
    GatewayProxyService,
    SqliteGatewayControl,
)
from atrex_runtime.gateway.control import BootstrapRunStatus


@pytest.mark.anyio
@pytest.mark.parametrize("legacy_schema", [False, True])
async def test_restore_bootstrap_queries_without_optimizer_history_or_authority(
    tmp_path: Path,
    legacy_schema: bool,
):
    registry, source, attempt, capability, service, adapter = _service(tmp_path)
    store = LocalArtifactStore(tmp_path / "artifacts")
    lineage = registry.get_lineage(registry.get_epoch(attempt.epoch_id).lineage_id)
    campaign = registry.get_campaign(lineage.campaign_id)
    contract = store.put_json({"test": "contract"}, ArtifactKind.EVALUATION_CONTRACT)
    registry._connection.execute(
        "UPDATE campaigns SET evaluation_contract_digest=? WHERE id=?", (contract, campaign.id)
    )
    input_kernel = store.put_json({"test": "source"}, ArtifactKind.KERNEL)
    evidence = store.put_json({"test": "bootstrap"}, ArtifactKind.EVIDENCE)
    boot_id = new_attempt_id()
    boot = source.issue_bootstrap(
        BootstrapGatewaySubject(
            attempt_id=boot_id,
            campaign_id=campaign.id,
            lineage_id=lineage.id,
            epoch_id=parse_epoch_id("epoch_" + str(boot_id).removeprefix("attempt_")),
            kernel_agent_revision_id=lineage.active_kernel_agent_revision_id,
            operator=campaign.operator,
            hardware_target=campaign.hardware_target,
            dsl=lineage.dsl,
            evaluation_contract_digest=contract,
            input_kernel_digest=input_kernel,
            evidence_digest=evidence,
            created_at=NOW_DATETIME,
        ),
        GatewayCapabilityPolicy(frozenset(GatewayOperation), 20, NOW_DATETIME + timedelta(hours=1)),
    )
    body = json.loads(_request(attempt))
    body["attempt_id"] = boot_id
    result = await service.execute(boot.token, json.dumps(body).encode())
    ordinal = 0

    async def journal(operation, **fields):
        nonlocal ordinal
        ordinal += 1
        return await service.execute(
            boot.token,
            json.dumps(
                {
                    "schema_version": 2,
                    "attempt_id": boot_id,
                    "operation": operation,
                    "idempotency_key": str(ordinal),
                    **fields,
                }
            ).encode(),
            operation_scope="journal",
        )

    direction = (
        await journal(
            "direction_update",
            request={
                "action": "propose",
                "name": "Original baseline",
                "hypothesis": "It is correct",
                "rationale": "Test the initial source",
                "plan": ["Evaluate"],
                "success_criteria": "Pass",
                "stop_conditions": "Incorrect",
            },
        )
    ).result["direction_id"]
    await journal(
        "direction_update",
        request={
            "action": "start",
            "direction_id": direction,
            "analysis": "Evaluate baseline",
        },
    )
    experiment = (
        await journal(
            "experiment_record",
            request={
                "direction_id": direction,
                "name": "Bootstrap evaluation",
                "change": "Initial implementation",
                "before": None,
                "after": {"result_artifact_digest": result.result_artifact_digest},
                "action": "baseline",
            },
        )
    ).result["experiment_id"]
    report = store.put_json(
        {"direction_events": [], "experiments": []}, ArtifactKind.ATTEMPT_REPORT
    )
    source.finish_bootstrap_run(
        boot_id,
        0,
        status=BootstrapRunStatus.COMPLETED,
        finish_reason="completed",
        failure_reason=None,
        report_digest=report,
    )
    optimizer_request = json.loads(_request(attempt))
    file = optimizer_request["candidate"]["files"][0]
    file["content_base64"] = base64.b64encode(
        base64.b64decode(file["content_base64"]) + b"\n# subsequent optimizer\n"
    ).decode()
    unrelated = await service.execute(capability.token, json.dumps(optimizer_request).encode())
    archive = tmp_path / "archive.sqlite"
    with sqlite3.connect(archive) as db:
        source._connection.backup(db)
        if legacy_schema:
            db.execute("UPDATE metadata SET value=14 WHERE key='schema_version'")
            # Older reset archives retained the subject and Journal but no run row.
            db.execute("DELETE FROM bootstrap_runs")
    # The retained checkpoint explicitly names the Bootstrap subject it came from.
    evidence_tree = tmp_path / "retained-evidence"
    (evidence_tree / "bootstrap").mkdir(parents=True)
    (evidence_tree / "bootstrap/report.json").write_text(json.dumps({"attempt_id": boot_id}))
    checkpoint = store.put_directory(evidence_tree, ArtifactKind.EVIDENCE)
    registry._connection.execute(
        "UPDATE lineages SET evidence_checkpoint=? WHERE id=?", (checkpoint, lineage.id)
    )
    before_bytes = archive.read_bytes()
    dest_store = LocalArtifactStore(tmp_path / "restored-artifacts")
    dest = SqliteGatewayControl(
        tmp_path / "restored.sqlite", registry, signing_key=b"z" * 32, clock=lambda: NOW_DATETIME
    )
    new_cap = dest.issue(
        attempt.id,
        GatewayCapabilityPolicy(frozenset(GatewayOperation), 20, NOW_DATETIME + timedelta(hours=1)),
    )
    dest_service = GatewayProxyService(
        dest,
        dest_store,
        adapter,
        GatewayProxyLimits(65536, 8, 16384),
        registry,
        clock=lambda: NOW_DATETIME,
    )
    app = GatewayProxyAsgiApp(dest_service, GatewayProxyLimits(65536, 8, 16384))
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://runtime") as http:

        async def query(op, **fields):
            nonlocal ordinal
            ordinal += 1
            return await http.post(
                "/v1/runtime/journals"
                if op in {"direction_load", "experiment_load"}
                else "/v1/runtime/queries",
                headers={"Authorization": f"Bearer {new_cap.token}"},
                json={
                    "schema_version": 2,
                    "attempt_id": attempt.id,
                    "operation": op,
                    "idempotency_key": f"{op}-{ordinal}",
                    **fields,
                },
            )

        denied = await query(
            "result_artifact_read", result_artifact_digest=result.result_artifact_digest
        )
        assert denied.status_code == 400
        # A broken ancestry must fail before any history is imported. A report alone
        # must not admit an unrelated subject into visibility.
        registry._connection.execute(
            "UPDATE lineages SET bootstrap_source_lineage_id=NULL WHERE id=?", (lineage.id,)
        )
        missing_subject = tmp_path / "missing-subject.sqlite"
        with sqlite3.connect(archive) as original, sqlite3.connect(missing_subject) as missing:
            original.backup(missing)
            missing.execute("DELETE FROM bootstrap_gateway_subjects")
        with pytest.raises(ValueError, match="outside its restored ancestry"):
            dest.restore_bootstrap_history(lineage.id, missing_subject, store, dest_store)
        assert not dest._connection.execute("SELECT * FROM bootstrap_gateway_subjects").fetchall()
        counts = dest.restore_bootstrap_history(lineage.id, archive, store, dest_store)
        assert counts["bootstrap_gateway_subjects"] == 1
        assert counts["runtime_experiments"] == 1
        assert counts["bootstrap_runs"] == (0 if legacy_schema else 1)
        assert archive.read_bytes() == before_bytes
        assert dest.verify_bootstrap_history_artifacts(lineage.id, dest_store) > 0
        retained_result = result.result_artifact_digest
        assert retained_result is not None
        report_path = (
            tmp_path
            / "restored-artifacts"
            / "sha256"
            / str(retained_result).removeprefix("sha256:")
        )
        hidden_report_path = report_path.with_name(report_path.name + ".missing")
        report_path.rename(hidden_report_path)
        try:
            with pytest.raises(ValueError, match="unavailable Artifact"):
                dest.verify_bootstrap_history_artifacts(lineage.id, dest_store)
        finally:
            hidden_report_path.rename(report_path)
        loaded = await query(
            "result_artifact_read", result_artifact_digest=result.result_artifact_digest
        )
        assert loaded.status_code == 200, loaded.text
        assert loaded.json()["result"]["result"]["correct"] is True
        loaded = await query("direction_load", direction_id=direction)
        assert loaded.status_code == 200, loaded.text
        loaded = await query("experiment_load", experiment_id=experiment)
        assert loaded.status_code == 200, loaded.text
        assert "Bootstrap evaluation" in loaded.text
        denied = await query(
            "result_artifact_read", result_artifact_digest=unrelated.result_artifact_digest
        )
        assert denied.status_code == 400
        assert dest.list_kernel_trials((attempt.id,), limit=100) == ()
        assert (
            dest._connection.execute("select count(*) from gateway_evaluate_tasks").fetchone()[0]
            == 0
        )
        assert (
            dest._connection.execute("select count(*) from gateway_active_calls").fetchone()[0] == 0
        )
        old_cap = dest._connection.execute(
            "select token_hash,revoked from gateway_capabilities where attempt_id=?", (boot_id,)
        ).fetchone()
        assert tuple(old_cap) == ("0" * 64, 1)
        assert not any(
            dest.restore_bootstrap_history(lineage.id, archive, store, dest_store).values()
        )
        # Disabled tools stay disabled even after their historical facts are restored.
        registry._connection.execute(
            "UPDATE lineages SET result_artifact_read_enabled=0 WHERE id=?", (lineage.id,)
        )
        denied = await query(
            "result_artifact_read", result_artifact_digest=result.result_artifact_digest
        )
        assert denied.status_code == 400 and "disabled" in denied.text
        # A conflicting replay rolls back; it never replaces a previously restored fact.
        with sqlite3.connect(archive) as db:
            db.execute(
                "UPDATE runtime_direction_events SET recorded_at='bad' WHERE attempt_id=?",
                (boot_id,),
            )
        with pytest.raises(ValueError, match="Conflicting Bootstrap history"):
            dest.restore_bootstrap_history(lineage.id, archive, store, dest_store)
        assert dest.list_direction_events(boot_id) == source.list_direction_events(boot_id)
    assert len(adapter.requests) == 2
    minimal = SqliteGatewayControl(
        tmp_path / "minimal.sqlite", registry, signing_key=b"m" * 32, clock=lambda: NOW_DATETIME
    )
    assert minimal.verify_bootstrap_history_artifacts(lineage.id, dest_store) == 1
    minimal.close()
    dest.close()
    source.close()
    registry.close()
