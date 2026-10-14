"""Restored Bootstrap measurements remain readable without widening peer history."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import pytest
from conftest import seed_lineage
from httpx import ASGITransport, AsyncClient
from test_gateway_proxy import NOW_DATETIME, _insert_attempt, _request, _service

from atrex_runtime.artifacts.local import ArtifactKind, LocalArtifactStore
from atrex_runtime.domain.ids import new_attempt_id, new_kernel_revision_id, parse_epoch_id
from atrex_runtime.domain.models import Dsl, KernelEvaluation
from atrex_runtime.gateway import (
    BootstrapGatewaySubject,
    GatewayCapabilityPolicy,
    GatewayOperation,
    GatewayProxyAsgiApp,
    GatewayProxyLimits,
)


def _seed_input(registry, artifacts, attempt, tmp_path, *, holdout=False):
    source = tmp_path / "baseline-source"
    source.mkdir()
    (source / "kernel.py").write_text("# preserved bootstrap\n")
    kernel = artifacts.put_directory(source, ArtifactKind.KERNEL)
    raw = {
        "job_id": "private-job-id",
        "status": "succeeded",
        "result": {
            "performance": {
                "shapes": {
                    "0": {
                        "samples": [
                            {"end_to_end_time_ms": 0.08},
                            {"end_to_end_time_ms": 0.10},
                            {"end_to_end_time_ms": 0.12},
                        ]
                    },
                    "1": {"samples": [{"end_to_end_time_ms": 0.4}]},
                }
            },
        },
    }
    if holdout:
        # A missing alias map must fail closed, not reveal source IDs or Test timing.
        raw["validation_shape_ids"] = ["0"]
    result = artifacts.put_json(raw, ArtifactKind.GATEWAY_RESULT)
    old = registry.get_kernel_revision(attempt.input_kernel_revision_id)
    revision = replace(
        old,
        id=new_kernel_revision_id(),
        artifact_digest=kernel,
        evaluation=KernelEvaluation(True, 200.0, result),
    )
    registry.register_kernel_revision(revision)
    registry._connection.execute(
        "UPDATE attempts SET input_kernel_revision_id=? WHERE id=?", (revision.id, attempt.id)
    )
    return revision


@pytest.mark.anyio
@pytest.mark.parametrize("holdout", [False, True])
async def test_restored_input_without_trials_is_readable_and_safe(tmp_path: Path, holdout: bool):
    registry, control, attempt, capability, service, adapter = _service(tmp_path)
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    baseline = _seed_input(registry, artifacts, attempt, tmp_path, holdout=holdout)
    app = GatewayProxyAsgiApp(service, GatewayProxyLimits(65536, 8, 16384))
    headers = {"Authorization": f"Bearer {capability.token}"}
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://runtime") as http:

        async def query(operation, **data):
            return await http.post(
                "/v1/runtime/queries",
                headers=headers,
                json={
                    "schema_version": 2,
                    "attempt_id": attempt.id,
                    "operation": operation,
                    "idempotency_key": operation,
                    **data,
                },
            )

        source = await query(
            "kernel_artifact_read",
            kernel_artifact_digest=baseline.artifact_digest,
            file="kernel.py",
        )
        assert source.status_code == 200, source.text
        assert source.json()["result"]["content"] == "# preserved bootstrap\n"
        assert source.json()["result"]["result_artifacts"] == [
            {
                "result_artifact_digest": baseline.evaluation.gateway_result_digest,
                "kernel_artifact_digest": baseline.artifact_digest,
                "operation": "evaluate",
                "status": "completed",
            }
        ]
        read = await query(
            "result_artifact_read", result_artifact_digest=baseline.evaluation.gateway_result_digest
        )
        assert read.status_code == 200, read.text
        result = read.json()["result"]
        assert result["kernel_artifact_digest"] == baseline.artifact_digest
        assert result["operation"] == "evaluate"
        assert result["status"] == "completed"
        assert result["result"]["correct"] is True
        if holdout:
            assert result["result"]["latency_us_by_shape"] == {}
            assert result["result"]["latency_us_geomean"] is None
        else:
            assert result["result"]["latency_us_by_shape"] == {"0": 100.0, "1": 400.0}
            assert result["result"]["latency_us_geomean"] == 200.0
            assert result["result"]["latency_us_arith_mean"] == 250.0
        assert "private-job-id" not in read.text
        assert "samples" not in read.text
        assert control.list_kernel_trials((attempt.id,), limit=100) == ()
        assert adapter.requests == []
        # Frozen no-read policy also covers the restored baseline path.
        lineage = registry.get_epoch(attempt.epoch_id).lineage_id
        registry._connection.execute(
            "UPDATE lineages SET result_artifact_read_enabled=0 WHERE id=?", (lineage,)
        )
        disabled = await query(
            "result_artifact_read", result_artifact_digest=baseline.evaluation.gateway_result_digest
        )
        assert disabled.status_code == 400
        assert "reads are disabled" in disabled.text
    control.close()
    registry.close()


@pytest.mark.anyio
async def test_input_ancestry_does_not_grant_source_future_or_unrelated_results(tmp_path: Path):
    registry, control, attempt, capability, service, adapter = _service(tmp_path)
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    baseline = _seed_input(registry, artifacts, attempt, tmp_path)
    # The source has already advanced. A clone may read its frozen input, not that best Kernel.
    other = artifacts.put_json(
        {"latency_us_by_shape": {"private": 1.0}}, ArtifactKind.GATEWAY_RESULT
    )
    peer = seed_lineage(registry, dsl=Dsl.CUDA, gateway_result_digest=other)
    registry._connection.execute(
        "UPDATE lineages SET bootstrap_source_lineage_id=? WHERE id=?",
        (peer.lineage_id, registry.get_epoch(attempt.epoch_id).lineage_id),
    )
    child = replace(
        baseline,
        id=new_kernel_revision_id(),
        parent_id=baseline.id,
        evaluation=KernelEvaluation(True, 1.0, other),
    )
    registry.register_kernel_revision(child)
    # Advancing the reader's input preserves visibility to its immutable parent as well.
    registry._connection.execute(
        "UPDATE attempts SET input_kernel_revision_id=? WHERE id=?", (child.id, attempt.id)
    )
    visible = control.visible_input_kernel_revisions(attempt.id)
    assert [k.id for k in visible] == [child.id, baseline.id]
    registry._connection.execute(
        "UPDATE attempts SET input_kernel_revision_id=? WHERE id=?", (baseline.id, attempt.id)
    )
    for operation, key, digest in (
        ("result_artifact_read", "result_artifact_digest", other),
        ("kernel_artifact_read", "kernel_artifact_digest", peer.baseline.artifact_digest),
    ):
        with pytest.raises(ValueError, match="outside the visible Lineage history"):
            await service.execute(
                capability.token,
                json.dumps(
                    {
                        "schema_version": 2,
                        "attempt_id": attempt.id,
                        "idempotency_key": operation,
                        "operation": operation,
                        key: digest,
                    }
                ).encode(),
                operation_scope="runtime",
            )
    assert adapter.requests == []
    control.close()
    registry.close()


@pytest.mark.anyio
async def test_restored_clone_reads_original_bootstrap_trials_but_not_source_optimizer(
    tmp_path: Path,
):
    registry, control, attempt, capability, service, adapter = _service(tmp_path)
    source = _insert_attempt(registry, dsl=Dsl.CUDA)
    lineage = registry.get_lineage(registry.get_epoch(source.epoch_id).lineage_id)
    campaign = registry.get_campaign(lineage.campaign_id)
    intermediate = seed_lineage(registry, dsl=Dsl.CUTEDSL, bootstrap_source_lineage_id=lineage.id)
    registry._connection.execute(
        "UPDATE lineages SET bootstrap_source_lineage_id=? WHERE id=?",
        (intermediate.lineage_id, registry.get_epoch(attempt.epoch_id).lineage_id),
    )
    bootstrap_id = new_attempt_id()
    boot = control.issue_bootstrap(
        BootstrapGatewaySubject(
            attempt_id=bootstrap_id,
            campaign_id=campaign.id,
            lineage_id=lineage.id,
            epoch_id=parse_epoch_id("epoch_" + str(bootstrap_id).removeprefix("attempt_")),
            kernel_agent_revision_id=lineage.active_kernel_agent_revision_id,
            operator=campaign.operator,
            hardware_target=campaign.hardware_target,
            dsl=lineage.dsl,
            evaluation_contract_digest=campaign.evaluation_contract_digest,
            input_kernel_digest=registry.get_kernel_revision(
                source.input_kernel_revision_id
            ).artifact_digest,
            evidence_digest=lineage.evidence_checkpoint,
            created_at=NOW_DATETIME,
        ),
        GatewayCapabilityPolicy(frozenset(GatewayOperation), 4, NOW_DATETIME + timedelta(hours=1)),
    )
    payload = json.loads(_request(source))
    payload["attempt_id"] = bootstrap_id
    evaluated = await service.execute(boot.token, json.dumps(payload).encode())
    result = await service.execute(
        capability.token,
        json.dumps(
            {
                "schema_version": 2,
                "attempt_id": attempt.id,
                "idempotency_key": "read-bootstrap",
                "operation": "result_artifact_read",
                "result_artifact_digest": evaluated.result_artifact_digest,
            }
        ).encode(),
        operation_scope="runtime",
    )
    assert result.result["result"]["correct"] is True
    assert bootstrap_id in control.visible_kernel_trial_attempt_ids(attempt.id)[1]
    assert source.id not in control.visible_kernel_trial_attempt_ids(attempt.id)[1]
    # A cycle must fail closed instead of widening scope or hanging.
    registry._connection.execute(
        "UPDATE lineages SET bootstrap_source_lineage_id=? WHERE id=?",
        (intermediate.lineage_id, lineage.id),
    )
    from atrex_runtime.domain.errors import InfrastructureError

    with pytest.raises(InfrastructureError, match="cycle"):
        control.visible_kernel_trial_attempt_ids(attempt.id)
    assert len(adapter.requests) == 1
    control.close()
    registry.close()
