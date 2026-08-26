"""Durable Campaign task worker and authenticated administration API tests."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import anyio
import pytest
from conftest import NOW, digest, seed_lineage

from atrex_runtime.api.administration import AdministrationAsgiApp
from atrex_runtime.artifacts.local import ArtifactKind, LocalArtifactStore
from atrex_runtime.bootstrap import (
    BootstrapResult,
    CampaignBootstrapResult,
    CampaignSpecV3,
)
from atrex_runtime.controller.tasks import CampaignTaskWorker
from atrex_runtime.domain.ids import (
    CampaignId,
    new_attempt_id,
    new_campaign_id,
    new_campaign_task_id,
    new_epoch_id,
    new_kernel_agent_revision_id,
    new_kernel_revision_id,
    new_lineage_id,
    new_worker_session_id,
)
from atrex_runtime.domain.models import (
    CampaignTask,
    CampaignTaskStatus,
    Dsl,
    Epoch,
    EpochSelection,
    EpochStatus,
    KernelMeasurement,
    KernelMeasurementPurpose,
    TokenUsage,
    WorkerSession,
    WorkerSessionRole,
    WorkerSessionStatus,
)
from atrex_runtime.gateway.control import (
    BootstrapGatewaySubject,
    BootstrapRunStatus,
    GatewayCapabilityPolicy,
    GatewayEvaluationSource,
    GatewayOperation,
    SqliteGatewayControl,
)
from atrex_runtime.lineage_seed import LineageSeedResult, LineageSeedSpecV1
from atrex_runtime.registry.sqlite import SqliteRegistry


def _task(campaign_id: CampaignId, key: str, target: int = 2) -> CampaignTask:
    return CampaignTask(
        id=new_campaign_task_id(),
        creation_key=key,
        campaign_id=campaign_id,
        target_epoch_number=target,
        finalize=False,
        status=CampaignTaskStatus.QUEUED,
        attempt_count=0,
        lease_owner=None,
        lease_expires_at=None,
        last_error=None,
        created_at=NOW,
        started_at=None,
        completed_at=None,
    )


@dataclass
class RecordingScheduler:
    """Record one registered Campaign execution and optionally fail."""

    calls: list[tuple[CampaignId, int, bool]]
    failure: Exception | None = None

    async def run_registered_campaign_through(
        self,
        campaign_id: CampaignId,
        target_epoch_number: int,
        *,
        finalize: bool = False,
    ) -> object:
        self.calls.append((campaign_id, target_epoch_number, finalize))
        await anyio.sleep(0.003)
        if self.failure is not None:
            raise self.failure
        return object()


@dataclass
class BlockingScheduler:
    started: anyio.Event

    async def run_registered_campaign_through(
        self,
        _campaign_id: CampaignId,
        _target_epoch_number: int,
        *,
        finalize: bool = False,
    ) -> object:
        del finalize
        self.started.set()
        await anyio.sleep(60)
        return object()


@dataclass
class CapturingBootstrapper:
    result: BootstrapResult
    calls: list[CampaignSpecV3]

    def bootstrap_campaign(self, spec: CampaignSpecV3) -> CampaignBootstrapResult:
        self.calls.append(spec)
        return CampaignBootstrapResult(
            self.result.campaign_id,
            (self.result,),
            hardware_target="sm_90",
            agate_gpu="nvidia-h100",
        )


@dataclass
class CapturingLineageSeeder:
    result: LineageSeedResult
    calls: list[tuple[CampaignId, LineageSeedSpecV1]]

    async def seed_lineage(
        self,
        campaign_id: CampaignId,
        spec: LineageSeedSpecV1,
    ) -> LineageSeedResult:
        self.calls.append((campaign_id, spec))
        return self.result


def _bootstrapper() -> CapturingBootstrapper:
    return CapturingBootstrapper(
        BootstrapResult(
            new_campaign_id(),
            new_lineage_id(),
            new_kernel_agent_revision_id(),
            new_kernel_revision_id(),
            digest("evaluation-contract"),
            digest("agent-problem"),
            digest("initial-evidence"),
            NOW,
            new_attempt_id(),
            NOW,
            digest("optimizer"),
        ),
        [],
    )


def test_campaign_task_registry_is_idempotent_reclaimable_and_cancellable(
    tmp_path: Path,
) -> None:
    with SqliteRegistry(tmp_path / "registry.sqlite") as registry:
        seeded = seed_lineage(registry)
        campaign_id = registry.get_lineage(seeded.lineage_id).campaign_id
        task = _task(campaign_id, "run-1")
        assert registry.enqueue_campaign_task(task) == task
        assert registry.enqueue_campaign_task(_task(campaign_id, "run-1")) == task

        claimed = registry.claim_campaign_task(
            "worker-1",
            now="2026-08-14T00:00:00+00:00",
            lease_expires_at="2026-08-14T00:01:00+00:00",
        )
        assert claimed is not None
        assert claimed.status is CampaignTaskStatus.RUNNING
        assert claimed.attempt_count == 1
        reclaimed = registry.claim_campaign_task(
            "worker-2",
            now="2026-08-14T00:02:00+00:00",
            lease_expires_at="2026-08-14T00:03:00+00:00",
        )
        assert reclaimed is not None
        assert reclaimed.id == task.id
        assert reclaimed.attempt_count == 2
        with pytest.raises(Exception, match="superseded"):
            registry.renew_campaign_task(
                task.id,
                "worker-1",
                lease_expires_at="2026-08-14T00:04:00+00:00",
            )
        completed = registry.complete_campaign_task(task.id, "worker-2")
        assert completed.status is CampaignTaskStatus.COMPLETED

        retryable = registry.enqueue_campaign_task(_task(campaign_id, "retry-1"))
        claimed_retry = registry.claim_campaign_task(
            "worker-3",
            now="2026-08-14T00:05:00+00:00",
            lease_expires_at="2026-08-14T00:06:00+00:00",
        )
        assert claimed_retry is not None and claimed_retry.id == retryable.id
        registry.fail_campaign_task(retryable.id, "worker-3", error="inspected failure")
        requeued = registry.requeue_campaign_task(retryable.id)
        assert requeued.status is CampaignTaskStatus.QUEUED
        assert requeued.last_error is None

        queued = registry.enqueue_campaign_task(_task(campaign_id, "cancel-1"))
        cancelled = registry.cancel_campaign_task(queued.id)
        assert cancelled.status is CampaignTaskStatus.CANCELLED
        assert registry.cancel_campaign_task(queued.id) == cancelled

        filtered = registry.list_runtime_events(
            after_sequence=0,
            limit=20,
            kinds=("campaign_task.requeued",),
            correlation={"campaign_id": campaign_id},
        )
        assert [event.aggregate_id for event in filtered] == [retryable.id]
        all_events = registry.list_runtime_events(after_sequence=0, limit=100)
        deleted = registry.prune_runtime_events(
            before_sequence=all_events[-1].sequence + 1,
            limit=2,
        )
        assert deleted == 2
        assert any(
            event.kind == "runtime_events.pruned"
            for event in registry.list_runtime_events(after_sequence=0, limit=100)
        )


def test_abandoned_cancelling_task_is_finalized_after_lease_expiry(tmp_path: Path) -> None:
    with SqliteRegistry(tmp_path / "registry.sqlite") as registry:
        seeded = seed_lineage(registry)
        campaign_id = registry.get_lineage(seeded.lineage_id).campaign_id
        task = registry.enqueue_campaign_task(_task(campaign_id, "abandoned-cancel"))
        claimed = registry.claim_campaign_task(
            "lost-worker",
            now="2026-08-14T00:00:00+00:00",
            lease_expires_at="2026-08-14T00:01:00+00:00",
        )
        assert claimed is not None and claimed.id == task.id
        assert registry.cancel_campaign_task(task.id).status is CampaignTaskStatus.CANCELLING

        finalized = registry.claim_campaign_task(
            "replacement-worker",
            now="2026-08-14T00:02:00+00:00",
            lease_expires_at="2026-08-14T00:03:00+00:00",
        )

        assert finalized is not None
        assert finalized.id == task.id
        assert finalized.status is CampaignTaskStatus.CANCELLED


@pytest.mark.anyio
async def test_campaign_task_worker_heartbeats_and_records_failure(tmp_path: Path) -> None:
    with SqliteRegistry(tmp_path / "registry.sqlite") as registry:
        seeded = seed_lineage(registry)
        campaign_id = registry.get_lineage(seeded.lineage_id).campaign_id
        successful = registry.enqueue_campaign_task(_task(campaign_id, "success"))
        scheduler = RecordingScheduler([])
        worker = CampaignTaskWorker(
            registry,
            scheduler,
            owner="worker-success",
            lease_seconds=0.02,
            heartbeat_seconds=0.005,
            max_error_bytes=64,
        )
        result = await worker.run_once()
        assert result is not None
        assert result.id == successful.id
        assert result.status is CampaignTaskStatus.COMPLETED
        assert scheduler.calls == [(campaign_id, 2, False)]

        failed = registry.enqueue_campaign_task(_task(campaign_id, "failure"))
        failing = CampaignTaskWorker(
            registry,
            RecordingScheduler([], RuntimeError("provider unavailable")),
            owner="worker-failure",
            lease_seconds=0.02,
            heartbeat_seconds=0.005,
            max_error_bytes=32,
        )
        failed_result = await failing.run_once()
        assert failed_result is not None
        assert failed_result.id == failed.id
        assert failed_result.status is CampaignTaskStatus.FAILED
        assert failed_result.last_error == "RuntimeError: provider unavailab"


@pytest.mark.anyio
async def test_running_task_cancellation_waits_for_cooperative_worker_cleanup(
    tmp_path: Path,
) -> None:
    with SqliteRegistry(tmp_path / "registry.sqlite") as registry:
        seeded = seed_lineage(registry)
        campaign_id = registry.get_lineage(seeded.lineage_id).campaign_id
        task = registry.enqueue_campaign_task(_task(campaign_id, "cancel-running"))
        scheduler = BlockingScheduler(anyio.Event())
        worker = CampaignTaskWorker(
            registry,
            scheduler,
            owner="worker-cancel",
            lease_seconds=0.02,
            heartbeat_seconds=0.005,
            max_error_bytes=64,
        )
        results: list[CampaignTask] = []

        async def run() -> None:
            result = await worker.run_once()
            assert result is not None
            results.append(result)

        async with anyio.create_task_group() as workers:
            workers.start_soon(run)
            await scheduler.started.wait()
            cancelling = registry.cancel_campaign_task(task.id)
            assert cancelling.status is CampaignTaskStatus.CANCELLING

        assert results[0].status is CampaignTaskStatus.CANCELLED
        kinds = [event.kind for event in registry.list_runtime_events(after_sequence=0, limit=100)]
        assert "campaign_task.cancellation_requested" in kinds
        assert "campaign_task.cancelled" in kinds


@pytest.mark.anyio
async def test_administration_seeds_lineage_from_registered_revisions(
    tmp_path: Path,
) -> None:
    campaign_id = new_campaign_id()
    result = LineageSeedResult(
        campaign_id=campaign_id,
        lineage_id=new_lineage_id(),
        dsl=Dsl.TRITON,
        kernel_agent_revision_id=new_kernel_agent_revision_id(),
        kernel_revision_id=new_kernel_revision_id(),
        agent_artifact_digest=digest("seed-agent"),
        kernel_artifact_digest=digest("seed-kernel"),
        gateway_result_digest=digest("seed-result"),
        latency_us=12.5,
        evidence_checkpoint=digest("seed-evidence"),
        source_provenance_digest=digest("seed-provenance"),
        source_agent_revision_id=new_kernel_agent_revision_id(),
        source_kernel_revision_id=new_kernel_revision_id(),
        optimizer_model="optimizer-model",
        evolver_model=None,
        created_at=NOW,
    )
    seeder = CapturingLineageSeeder(result, [])
    with SqliteRegistry(tmp_path / "registry.sqlite") as registry:
        app = AdministrationAsgiApp(
            registry,
            LocalArtifactStore(tmp_path / "artifacts"),
            _bootstrapper(),
            lineage_seeder=seeder,
            bearer_token="a" * 32,
            max_request_bytes=4096,
            event_page_limit=10,
            event_export_limit=100,
            event_prune_limit=10,
        )
        status, value = await _request(
            app,
            "POST",
            f"/v1/admin/campaigns/{campaign_id}/lineages",
            {
                "schema_version": 1,
                "creation_key": "from-history",
                "dsl": "triton",
                "seed": {
                    "source_type": "revisions",
                    "agent_revision_id": result.source_agent_revision_id,
                    "kernel_revision_id": result.source_kernel_revision_id,
                },
                "attempts_per_trajectory": 3,
            },
        )

    assert status == 200
    assert value["lineage_id"] == result.lineage_id
    assert value["kernel_agent"]["version"] == "agent-v0"
    assert value["kernel"]["version"] == "v0"
    assert seeder.calls[0][0] == campaign_id
    assert seeder.calls[0][1].attempts_per_trajectory == 3


@pytest.mark.anyio
async def test_administration_api_auth_task_status_cancel_and_event_pagination(
    tmp_path: Path,
) -> None:
    with SqliteRegistry(tmp_path / "registry.sqlite") as registry:
        seeded = seed_lineage(registry)
        campaign_id = registry.get_lineage(seeded.lineage_id).campaign_id
        app = AdministrationAsgiApp(
            registry,
            LocalArtifactStore(tmp_path / "artifacts"),
            _bootstrapper(),
            bearer_token="a" * 32,
            max_request_bytes=4096,
            event_page_limit=2,
            event_export_limit=100,
            event_prune_limit=10,
        )
        request = {
            "schema_version": 1,
            "creation_key": "api-task-1",
            "campaign_id": campaign_id,
            "target_epoch_number": 3,
            "finalize": False,
        }

        status, unauthorized = await _request(
            app,
            "POST",
            "/v1/admin/tasks",
            request,
            token="wrong" * 8,
        )
        assert status == 401
        assert unauthorized == {"error": "unauthorized"}

        status, created = await _request(app, "POST", "/v1/admin/tasks", request)
        assert status == 202
        task_id = created["task_id"]
        replay_status, replay = await _request(app, "POST", "/v1/admin/tasks", request)
        assert replay_status == 202
        assert replay["task_id"] == task_id

        status, loaded = await _request(app, "GET", f"/v1/admin/tasks/{task_id}")
        assert status == 200
        assert loaded["status"] == "queued"

        status, page = await _request(
            app,
            "GET",
            "/v1/admin/events",
            query=b"after=0&limit=2",
        )
        assert status == 200
        assert len(page["events"]) == 2
        assert page["next_after"] == page["events"][-1]["sequence"]
        assert all(event["payload"]["schema_version"] == 1 for event in page["events"])

        status, cancelled = await _request(
            app,
            "POST",
            f"/v1/admin/tasks/{task_id}/cancel",
        )
        assert status == 200
        assert cancelled["status"] == "cancelled"

        failed_task = registry.enqueue_campaign_task(_task(campaign_id, "api-requeue"))
        claimed = registry.claim_campaign_task(
            "api-worker",
            now="2026-08-14T00:00:00+00:00",
            lease_expires_at="2026-08-14T00:01:00+00:00",
        )
        assert claimed is not None and claimed.id == failed_task.id
        registry.fail_campaign_task(failed_task.id, "api-worker", error="inspected")
        status, requeued = await _request(
            app,
            "POST",
            f"/v1/admin/tasks/{failed_task.id}/requeue",
        )
        assert status == 200
        assert requeued["status"] == "queued"

        status, campaign = await _request(
            app,
            "GET",
            f"/v1/admin/campaigns/{campaign_id}",
        )
        assert status == 200
        assert campaign["lineages"][0]["lineage_id"] == seeded.lineage_id
        assert campaign["problem_generalization_model"] is None
        assert campaign["lineages"][0]["models"] == {
            "optimizer": None,
            "evolver": None,
        }

        status, campaign = await _request(
            app,
            "POST",
            f"/v1/admin/campaigns/{campaign_id}/cancel",
        )
        assert status == 200
        assert campaign["status"] == "cancelled"

        export_status, headers, body = await _raw_request(
            app,
            "GET",
            "/v1/admin/events/export",
            query=f"kind=campaign.cancelled&campaign_id={campaign_id}".encode(),
        )
        assert export_status == 200
        assert (b"content-type", b"application/x-ndjson") in headers
        exported = [json.loads(line) for line in body.splitlines()]
        assert [event["kind"] for event in exported] == ["campaign.cancelled"]

        latest = registry.list_runtime_events(after_sequence=0, limit=100)[-1].sequence
        status, pruned = await _request(
            app,
            "POST",
            "/v1/admin/events/prune",
            {"schema_version": 1, "before_sequence": latest, "limit": 1},
        )
        assert status == 200
        assert pruned == {"deleted": 1}

        status, metrics = await _request(app, "GET", "/v1/admin/metrics")
        assert status == 200
        assert metrics["latest_event_sequence"] > 0
        assert metrics["event_counts"]["runtime_events.pruned"] == 1
        assert metrics["campaign_task_counts"]["queued"] == 1


@pytest.mark.anyio
async def test_administration_kernel_catalog_measurements_and_source(tmp_path: Path) -> None:
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    source = tmp_path / "kernel-source"
    source.mkdir()
    (source / "kernel.py").write_text("class Model: pass\n", encoding="utf-8")
    kernel_digest = artifacts.put_directory(source, ArtifactKind.KERNEL)
    gateway_result = artifacts.put_json(
        {"result": {"performance": {"shapes": {"0": {"sol": {"pct": 71.25}}}}}},
        ArtifactKind.GATEWAY_RESULT,
    )
    with SqliteRegistry(tmp_path / "registry.sqlite") as registry:
        seeded = seed_lineage(
            registry,
            kernel_artifact_digest=kernel_digest,
            gateway_result_digest=gateway_result,
        )
        lineage = registry.get_lineage(seeded.lineage_id)
        registry.record_kernel_measurement(
            KernelMeasurement(
                id="measurement-api",
                kernel_revision_id=seeded.baseline.id,
                purpose=KernelMeasurementPurpose.AGENT_PROMOTION,
                repeat=0,
                correct=True,
                latency_us=99.0,
                gateway_result_digest=digest("measurement-api-result"),
                agate_job_id="agate-api",
                created_at=NOW,
            )
        )
        app = AdministrationAsgiApp(
            registry,
            artifacts,
            _bootstrapper(),
            bearer_token="a" * 32,
            max_request_bytes=8192,
            event_page_limit=10,
            event_export_limit=100,
            event_prune_limit=10,
        )

        status, campaign_catalog = await _request(
            app,
            "GET",
            f"/v1/admin/campaigns/{lineage.campaign_id}/kernels",
        )
        assert status == 200
        assert campaign_catalog["kernels"][0]["kernel_agent_revision_id"] == (
            seeded.active_revision_id
        )
        assert campaign_catalog["kernels"][0]["version"] == "v0"
        assert campaign_catalog["kernels"][0]["kernel_agent_version"] == "agent-v0"
        assert campaign_catalog["kernels"][0]["sol_percent"] == 71.25
        assert campaign_catalog["kernels"][0]["kernel_artifact"] == {
            "digest": kernel_digest,
            "kind": "kernel",
            "referenced_at": NOW,
        }

        status, lineage_catalog = await _request(
            app,
            "GET",
            f"/v1/admin/lineages/{lineage.id}/kernels",
        )
        assert status == 200
        assert lineage_catalog["kernels"] == campaign_catalog["kernels"]

        status, campaign_agents = await _request(
            app,
            "GET",
            f"/v1/admin/campaigns/{lineage.campaign_id}/agent-revisions",
        )
        assert status == 200
        assert campaign_agents["agent_revisions"][0]["agent_version"] == "agent-v0"

        status, lineage_agents = await _request(
            app,
            "GET",
            f"/v1/admin/lineages/{lineage.id}/agent-revisions",
        )
        assert status == 200
        assert lineage_agents == {
            "lineage_id": lineage.id,
            "agent_revisions": campaign_agents["agent_revisions"],
        }

        status, lineage_attempts = await _request(
            app,
            "GET",
            f"/v1/admin/lineages/{lineage.id}/attempts",
        )
        assert status == 200
        assert lineage_attempts == {"lineage_id": lineage.id, "attempts": []}

        status, campaign_attempts = await _request(
            app,
            "GET",
            f"/v1/admin/campaigns/{lineage.campaign_id}/attempts",
        )
        assert status == 200
        assert campaign_attempts == {
            "campaign_id": lineage.campaign_id,
            "attempts": [],
        }

        status, agent = await _request(
            app,
            "GET",
            f"/v1/admin/agent-revisions/{seeded.active_revision_id}",
        )
        assert status == 200
        assert agent["disposition"] == "baseline"
        assert agent["optimizer_artifact"]["referenced_at"] == NOW

        status, kernel = await _request(
            app,
            "GET",
            f"/v1/admin/kernels/{seeded.baseline.id}",
        )
        assert status == 200
        assert kernel["version"] == "v0"
        assert [item["purpose"] for item in kernel["measurements"]] == [
            "framework_baseline",
            "agent_promotion",
        ]

        status, measurements = await _request(
            app,
            "GET",
            f"/v1/admin/kernels/{seeded.baseline.id}/measurements",
        )
        assert status == 200
        assert measurements["measurements"] == kernel["measurements"]

        status, exported_source = await _request(
            app,
            "GET",
            f"/v1/admin/kernels/{seeded.baseline.id}/source",
        )
        assert status == 200
        assert exported_source["version"] == "v0"
        assert exported_source["referenced_at"] == NOW
        assert exported_source["files"] == [
            {
                "path": "kernel.py",
                "size": 18,
                "encoding": "utf-8",
                "content": "class Model: pass\n",
            }
        ]


@pytest.mark.anyio
async def test_administration_lists_epoch_winners_by_lineage_and_campaign(
    tmp_path: Path,
) -> None:
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    with SqliteRegistry(tmp_path / "registry.sqlite") as registry:
        seeded = seed_lineage(
            registry,
            challenger_count=0,
            attempts_per_trajectory=1,
        )
        lineage = registry.get_lineage(seeded.lineage_id)
        epoch = Epoch(
            id=new_epoch_id(),
            lineage_id=lineage.id,
            number=1,
            active_kernel_agent_revision_id=seeded.active_revision_id,
            challenger_kernel_agent_revision_ids=(),
            starting_kernel_revision_id=seeded.baseline.id,
            evidence_checkpoint=lineage.evidence_checkpoint,
            challenger_count=0,
            trajectories_per_branch=1,
            attempts_per_trajectory=1,
            status=EpochStatus.READY,
            winner_kernel_agent_revision_id=None,
            best_kernel_revision_id=None,
            created_at=NOW,
            completed_at=None,
        )
        registry.insert_epoch(epoch)
        registry.transition_epoch(epoch.id, EpochStatus.READY, EpochStatus.RUNNING)
        registry.transition_epoch(epoch.id, EpochStatus.RUNNING, EpochStatus.SELECTING)
        registry.complete_epoch(
            epoch.id,
            EpochSelection(seeded.active_revision_id, seeded.baseline.id),
        )
        app = AdministrationAsgiApp(
            registry,
            artifacts,
            _bootstrapper(),
            bearer_token="a" * 32,
            max_request_bytes=8192,
            event_page_limit=10,
            event_export_limit=100,
            event_prune_limit=10,
        )

        status, lineage_epochs = await _request(
            app,
            "GET",
            f"/v1/admin/lineages/{lineage.id}/epochs",
        )
        assert status == 200
        item = lineage_epochs["epochs"][0]
        assert item["active_agent_version"] == "agent-v0"
        assert item["challenger_agent_versions"] == []
        assert item["winner_agent_version"] == "agent-v0"
        assert item["winner_branch"] == "active"
        assert item["decision"] == "active_retained"

        status, campaign_epochs = await _request(
            app,
            "GET",
            f"/v1/admin/campaigns/{lineage.campaign_id}/epochs",
        )
        assert status == 200
        assert campaign_epochs == {
            "campaign_id": lineage.campaign_id,
            "epochs": lineage_epochs["epochs"],
        }


@pytest.mark.anyio
async def test_administration_accepts_campaign_bootstrap_request(tmp_path: Path) -> None:
    bootstrapper = _bootstrapper()
    with SqliteRegistry(tmp_path / "registry.sqlite") as registry:
        app = AdministrationAsgiApp(
            registry,
            LocalArtifactStore(tmp_path / "artifacts"),
            bootstrapper,
            bearer_token="a" * 32,
            max_request_bytes=8192,
            event_page_limit=10,
            event_export_limit=100,
            event_prune_limit=10,
        )
        lineage_inputs = {
            dsl: {
                "baseline_kernel": f"/inputs/{dsl}/kernel",
                "initial_evidence": f"/inputs/{dsl}/evidence",
            }
            for dsl in ("cuda", "triton", "cutedsl")
        }
        request: dict[str, object] = {
            "schema_version": 3,
            "creation_key": "api-campaign-bootstrap-1",
            "operator": "vector_add",
            "hardware_target": "nvidia-h100",
            "evaluation_contract": "/inputs/contract.json",
            "base_revision": {"commit": "a" * 40},
            "attempts_per_trajectory": 2,
            "lineages": lineage_inputs,
        }

        status, value = await _request(
            app,
            "POST",
            "/v1/admin/campaigns/bootstrap",
            request,
        )

        assert status == 200
        assert value["campaign_id"] == bootstrapper.result.campaign_id
        assert value["lineages"][0]["lineage_id"] == bootstrapper.result.lineage_id
        assert value["lineages"][0]["bootstrap_attempt_id"] == (
            bootstrapper.result.bootstrap_attempt_id
        )
        assert value["lineages"][0]["baseline_kernel"] == {
            "kernel_revision_id": bootstrapper.result.baseline_kernel_revision_id,
            "version": "v0",
            "created_at": NOW,
            "producer": {
                "kind": "bootstrap",
                "attempt_id": bootstrapper.result.bootstrap_attempt_id,
            },
        }
        assert value["lineages"][0]["kernel_agent"] == {
            "kernel_agent_revision_id": bootstrapper.result.kernel_agent_revision_id,
            "version": "agent-v0",
            "created_at": NOW,
            "producer": {"kind": "bootstrap"},
            "optimizer_artifact": {
                "digest": bootstrapper.result.optimizer_digest,
                "kind": "kernel_agent",
                "referenced_at": NOW,
            },
        }
        assert len(bootstrapper.calls) == 1
        assert isinstance(bootstrapper.calls[0], CampaignSpecV3)

        lineage_inputs["cuda"]["initial_evidence"] = "relative/evidence"
        status, value = await _request(
            app,
            "POST",
            "/v1/admin/campaigns/bootstrap",
            request,
        )
        assert status == 400
        assert "absolute" in str(value["error"])

        status, value = await _request(
            app,
            "POST",
            f"/v1/admin/epochs/{new_epoch_id()}/recover",
            {
                "schema_version": 1,
                "recovery_key": "incident-1",
                "reason": "worker host replaced",
            },
        )
        assert status == 404
        assert "not found" in str(value["error"]).lower()


@pytest.mark.anyio
async def test_administration_lists_exact_bootstrap_run_generations(tmp_path: Path) -> None:
    with SqliteRegistry(tmp_path / "registry.sqlite") as registry:
        control = SqliteGatewayControl(
            tmp_path / "gateway.sqlite",
            registry,
            signing_key=b"g" * 32,
            clock=lambda: datetime(2026, 8, 14, tzinfo=UTC),
        )
        attempt_id = new_attempt_id()
        subject = BootstrapGatewaySubject(
            attempt_id=attempt_id,
            campaign_id=new_campaign_id(),
            lineage_id=new_lineage_id(),
            epoch_id=new_epoch_id(),
            kernel_agent_revision_id=new_kernel_agent_revision_id(),
            operator="vector_add",
            hardware_target="nvidia-h100",
            dsl=Dsl.TRITON,
            evaluation_contract_digest=digest("admin-contract"),
            input_kernel_digest=digest("admin-kernel"),
            evidence_digest=digest("admin-evidence"),
            created_at=datetime(2026, 8, 14, tzinfo=UTC),
        )
        capability = control.issue_bootstrap(
            subject,
            GatewayCapabilityPolicy(
                frozenset({GatewayOperation.EVALUATE}),
                2,
                datetime(2026, 8, 14, tzinfo=UTC) + timedelta(hours=1),
            ),
        )
        control.begin_bootstrap_run(
            attempt_id,
            capability.recovery_generation,
            run_id="run-admin",
            workspace_path="/runtime/bootstrap/run-admin",
        )
        control.finish_bootstrap_run(
            attempt_id,
            capability.recovery_generation,
            status=BootstrapRunStatus.FAILED,
            finish_reason="token-budget-exhausted",
            failure_reason="Core lineage baseline token-budget-exhausted",
            session_trace_digest=digest("admin-trace"),
            token_budget=100,
            token_usage=TokenUsage(40, 5, 50, 0),
            report_digest=digest("admin-report"),
        )
        app = AdministrationAsgiApp(
            registry,
            LocalArtifactStore(tmp_path / "artifacts"),
            _bootstrapper(),
            bearer_token="a" * 32,
            max_request_bytes=4096,
            event_page_limit=10,
            event_export_limit=100,
            event_prune_limit=10,
            gateway_control=control,
        )

        status, listing = await _request(
            app,
            "GET",
            f"/v1/admin/bootstrap-attempts/{attempt_id}/runs",
        )
        assert status == 200
        assert len(listing["runs"]) == 1
        assert listing["runs"][0]["finish_reason"] == "token-budget-exhausted"
        assert listing["runs"][0]["token_usage"]["total_tokens"] == 95

        status, exact = await _request(
            app,
            "GET",
            f"/v1/admin/bootstrap-attempts/{attempt_id}/runs/0",
        )
        assert status == 200
        assert exact["run_id"] == "run-admin"
        assert exact["session_trace_digest"] == digest("admin-trace")
        control.close()


@pytest.mark.anyio
async def test_administration_exposes_every_evaluation_source_and_result(
    tmp_path: Path,
) -> None:
    with SqliteRegistry(tmp_path / "registry.sqlite") as registry:
        artifacts = LocalArtifactStore(tmp_path / "artifacts")
        control = SqliteGatewayControl(
            tmp_path / "gateway.sqlite",
            registry,
            signing_key=b"v" * 32,
            clock=lambda: datetime(2026, 8, 14, tzinfo=UTC),
        )
        attempt_id = new_attempt_id()
        subject = BootstrapGatewaySubject(
            attempt_id,
            new_campaign_id(),
            new_lineage_id(),
            new_epoch_id(),
            new_kernel_agent_revision_id(),
            "vector_add",
            "L20N",
            Dsl.TRITON,
            digest("contract"),
            digest("seed"),
            digest("evidence"),
            datetime(2026, 8, 14, tzinfo=UTC),
        )
        capability = control.issue_bootstrap(
            subject,
            GatewayCapabilityPolicy(
                frozenset({GatewayOperation.EVALUATE}),
                4,
                datetime(2026, 8, 14, tzinfo=UTC) + timedelta(hours=1),
            ),
        )
        source = tmp_path / "candidate"
        source.mkdir()
        (source / "kernel.py").write_text("def kernel(): return 1\n")
        candidate = artifacts.put_directory(source, ArtifactKind.KERNEL)
        agent_result = artifacts.put_json({"stage": "agent"}, ArtifactKind.GATEWAY_RESULT)
        final_result = artifacts.put_json({"stage": "runtime"}, ArtifactKind.GATEWAY_RESULT)
        control.authorize(
            capability,
            GatewayOperation.EVALUATE,
            idempotency_key="agent-1",
            request_digest=str(digest("agent-request")),
        )
        control.bind_operation_candidate(
            attempt_id,
            "agent-1",
            GatewayOperation.EVALUATE,
            candidate,
        )
        control.commit_operation_artifact(
            attempt_id,
            "agent-1",
            GatewayOperation.EVALUATE,
            agent_result,
        )
        control.bind_operation_gateway_result(
            attempt_id,
            "agent-1",
            GatewayOperation.EVALUATE,
            agent_result,
        )
        agent = control.record_evaluation(
            attempt_id,
            source=GatewayEvaluationSource.AGENT,
            idempotency_key="agent-1",
            kernel_artifact_digest=candidate,
            gateway_result_digest=agent_result,
            correct=True,
            latency_us=10.0,
            agate_job_id="ev_agent",
        )
        control.record_evaluation(
            attempt_id,
            source=GatewayEvaluationSource.RUNTIME_FINAL,
            idempotency_key="runtime-final",
            kernel_artifact_digest=candidate,
            gateway_result_digest=final_result,
            correct=True,
            latency_us=11.0,
            agate_job_id="ev_final",
        )
        control.record_kernel_trial_annotations(
            attempt_id,
            (
                {
                    "experiment_id": "experiment_" + "a" * 32,
                    "sequence": 1,
                    "recorded_at": NOW,
                    "name": "reverted experiment",
                    "hypothesis": "test",
                    "change": "test",
                    "before": {
                        "kernel_artifact_digest": str(candidate),
                        "kernel_trial_id": control.list_kernel_trials((attempt_id,))[0].id,
                        "gateway_result_digests": [str(agent_result)],
                    },
                    "after": {
                        "kernel_artifact_digest": str(candidate),
                        "kernel_trial_id": control.list_kernel_trials((attempt_id,))[0].id,
                        "gateway_result_digests": [str(agent_result)],
                    },
                    "evidence": "agent-1",
                    "analysis": "the hypothesis failed because latency regressed",
                    "action": "restore_before",
                },
            ),
        )
        app = AdministrationAsgiApp(
            registry,
            artifacts,
            _bootstrapper(),
            bearer_token="a" * 32,
            max_request_bytes=4096,
            event_page_limit=10,
            event_export_limit=100,
            event_prune_limit=10,
            gateway_control=control,
        )

        status, listing = await _request(app, "GET", f"/v1/admin/attempts/{attempt_id}/evaluations")
        assert status == 200
        assert [item["source"] for item in listing["evaluations"]] == [
            "agent",
            "runtime_final",
        ]
        assert [item["evaluation_label"] for item in listing["evaluations"]] == [
            "g0-e1",
            "g0-e2",
        ]
        assert listing["evaluations"][0]["candidate_artifact"]["referenced_at"] == NOW

        status, source_value = await _request(
            app,
            "GET",
            f"/v1/admin/attempts/{attempt_id}/evaluations/{agent.id}/source",
        )
        assert status == 200
        assert source_value["evaluation_label"] == "g0-e1"
        assert source_value["referenced_at"] == NOW
        assert source_value["files"][0]["content"] == "def kernel(): return 1\n"

        status, result_value = await _request(
            app,
            "GET",
            f"/v1/admin/attempts/{attempt_id}/evaluations/{agent.id}/result",
        )
        assert status == 200
        assert result_value["result"] == {"stage": "agent"}

        status, trials = await _request(
            app,
            "GET",
            f"/v1/admin/attempts/{attempt_id}/kernel-trials",
        )
        assert status == 200
        trial = trials["kernel_trials"][0]
        assert trial["disposition"] == "revert"
        status, trial_source = await _request(
            app,
            "GET",
            f"/v1/admin/attempts/{attempt_id}/kernel-trials/{trial['kernel_trial_id']}/source",
        )
        assert status == 200
        assert trial_source["files"][0]["content"] == "def kernel(): return 1\n"
        status, trial_results = await _request(
            app,
            "GET",
            f"/v1/admin/attempts/{attempt_id}/kernel-trials/{trial['kernel_trial_id']}/results",
        )
        assert status == 200
        assert trial_results["results"][0]["result"] == {"stage": "agent"}
        control.close()


@pytest.mark.anyio
async def test_administration_exposes_worker_session_lifecycle(tmp_path: Path) -> None:
    with SqliteRegistry(tmp_path / "registry.sqlite", clock=lambda: NOW) as registry:
        session = registry.start_worker_session(
            WorkerSession(
                id=new_worker_session_id(),
                role=WorkerSessionRole.PROBLEM_GENERALIZATION,
                subject_id="generalization-api",
                external_run_id="run-api",
                workspace_path="/runtime/generalization/run-api",
                status=WorkerSessionStatus.RUNNING,
                started_at=NOW,
            )
        )
        registry.finish_worker_session(
            session.id,
            status=WorkerSessionStatus.COMPLETED,
            finish_reason="completed",
            trace_digest=digest("api-session-trace"),
            token_budget=1000,
            token_usage=TokenUsage(10, 2, 3, 0),
        )
        app = AdministrationAsgiApp(
            registry,
            LocalArtifactStore(tmp_path / "artifacts"),
            _bootstrapper(),
            bearer_token="a" * 32,
            max_request_bytes=4096,
            event_page_limit=10,
            event_export_limit=100,
            event_prune_limit=10,
        )

        status, value = await _request(app, "GET", f"/v1/admin/worker-sessions/{session.id}")

    assert status == 200
    assert value["worker_session_id"] == session.id
    assert value["status"] == "completed"
    assert value["session_trace_digest"] == digest("api-session-trace")
    assert value["token_usage"]["total_tokens"] == 15


async def _request(
    app: AdministrationAsgiApp,
    method: str,
    path: str,
    value: dict[str, object] | None = None,
    *,
    token: str = "a" * 32,
    query: bytes = b"",
) -> tuple[int, dict[str, object]]:
    status, _headers, body = await _raw_request(
        app,
        method,
        path,
        value,
        token=token,
        query=query,
    )
    return status, json.loads(body)


async def _raw_request(
    app: AdministrationAsgiApp,
    method: str,
    path: str,
    value: dict[str, object] | None = None,
    *,
    token: str = "a" * 32,
    query: bytes = b"",
) -> tuple[int, list[tuple[bytes, bytes]], bytes]:
    body = b"" if value is None else json.dumps(value).encode()
    messages = iter([{"type": "http.request", "body": body, "more_body": False}])
    sent: list[dict[str, object]] = []

    async def receive() -> dict[str, object]:
        return next(messages)

    async def send(message: dict[str, object]) -> None:
        sent.append(message)

    await app(
        {
            "type": "http",
            "method": method,
            "path": path,
            "query_string": query,
            "headers": [(b"authorization", f"Bearer {token}".encode())],
        },
        receive,
        send,
    )
    return (
        int(sent[0]["status"]),
        list(sent[0]["headers"]),
        bytes(sent[-1]["body"]),
    )
