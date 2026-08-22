"""CLI execution commands for Campaign bootstrap, scheduling, and recovery."""

from __future__ import annotations

import json
import os
import sys
from datetime import UTC, datetime
from uuid import uuid4

import anyio

from ..artifacts.local import LocalArtifactStore
from ..bootstrap import (
    BootstrapResult,
    CampaignBootstrapResult,
    RooflineMode,
    load_campaign_spec,
)
from ..composition.bootstrap import build_campaign_bootstrapper, build_lineage_seeder
from ..composition.campaign import build_campaign_runtime
from ..config import RuntimeSettings
from ..controller.campaign import CampaignScheduleResult
from ..controller.tasks import CampaignTaskWorker
from ..domain.ids import parse_campaign_id, parse_epoch_id, parse_lineage_id
from ..domain.models import Attempt, CampaignTask
from ..gateway.agate import load_agate_sdk
from ..gateway.configuration import build_agate_connection
from ..gateway.control import SqliteGatewayControl
from ..gateway.result_metrics import GatewaySolSummary, gateway_result_sol_summary
from ..lineage_seed import LineageSeedSpecV1
from ..presentation import (
    bootstrap_result_value,
    lineage_bootstrap_value,
    lineage_seed_result_value,
)
from ..registry.sqlite import SqliteRegistry
from ..secrets import read_capability_signing_key
from .progress import AttemptProgressRenderer


def bootstrap_campaign(config_path: str, campaign_path: str) -> None:
    """Initialize one commit-anchored Campaign and print durable identities."""
    settings = RuntimeSettings.from_file(config_path)
    with SqliteRegistry(settings.storage.registry_database) as registry:
        artifacts = LocalArtifactStore(settings.storage.artifacts_root)
        control = (
            None
            if settings.campaign is None
            else SqliteGatewayControl(
                settings.storage.gateway_database,
                registry,
                signing_key=read_capability_signing_key(
                    os.environ,
                    settings.gateway_proxy.capability_signing_key_env,
                ),
            )
        )
        try:
            bootstrapper = build_campaign_bootstrapper(
                settings,
                artifacts,
                registry,
                os.environ,
                control=control,
                roofline_resolved=lambda mode, detail: print(
                    _roofline_resolution_line(mode, detail),
                    file=sys.stderr,
                    flush=True,
                ),
            )
            spec = load_campaign_spec(campaign_path)
            result = bootstrapper.bootstrap_campaign(spec)
            print(
                f"Agate hardware resolved: {result.agate_gpu} -> {result.hardware_target}",
                file=sys.stderr,
                flush=True,
            )
            for lineage in result.lineages:
                revision = registry.get_kernel_revision(lineage.baseline_kernel_revision_id)
                summary = gateway_result_sol_summary(
                    artifacts,
                    revision.evaluation.gateway_result_digest,
                )
                print(
                    f"[{datetime.now(UTC).isoformat()}] bootstrap "
                    f"{registry.get_lineage(lineage.lineage_id).dsl.value} v0 "
                    f"{_sol_execution_detail(summary)}",
                    file=sys.stderr,
                    flush=True,
                )
        finally:
            if control is not None:
                control.close()
    print(json.dumps(_bootstrap_cli_value(result), sort_keys=True))


def seed_lineage(config_path: str, campaign_value: str, spec_path: str) -> None:
    """Add an independently versioned Lineage rooted at sealed existing content."""
    settings = RuntimeSettings.from_file(config_path)
    spec = LineageSeedSpecV1.from_file(spec_path)
    connection = build_agate_connection(settings.agate, os.environ)
    client, request_builder = load_agate_sdk(connection)
    with SqliteRegistry(settings.storage.registry_database) as registry:
        artifacts = LocalArtifactStore(settings.storage.artifacts_root)
        seeder = build_lineage_seeder(
            settings,
            artifacts,
            registry,
            client,
            request_builder,
        )
        result = anyio.run(
            seeder.seed_lineage,
            parse_campaign_id(campaign_value),
            spec,
        )
    print(json.dumps(lineage_seed_result_value(result), sort_keys=True))


def _bootstrap_cli_value(
    result: BootstrapResult | CampaignBootstrapResult,
) -> dict[str, object]:
    if isinstance(result, BootstrapResult):
        return lineage_bootstrap_value(result)
    return bootstrap_result_value(result)


def _roofline_resolution_line(mode: RooflineMode, detail: str | None) -> str:
    labels = {
        "explicit": "using explicit Evaluation Contract Roofline",
        "sealed-reuse": "reusing Campaign-sealed Roofline",
        "generated": "successfully generated and sealed a Campaign Roofline",
        "profile-fallback": ("no usable Roofline; every correct eval will run NCU Profile"),
    }
    detail_suffix = "" if detail is None else f" ({detail})"
    return f"[{datetime.now(UTC).isoformat()}] Roofline: {labels[mode]}{detail_suffix}"


def _sol_execution_detail(summary: GatewaySolSummary) -> str:
    value = "unavailable" if summary.percent is None else f"{summary.percent:.3f}%"
    if summary.source == "roofline":
        return f"eval finished; SOL={value} source=roofline"
    if summary.source == "ncu-profile" and summary.percent is not None:
        return f"eval+profile finished; SOL={value} source=ncu-profile"
    if summary.source == "ncu-profile":
        return f"eval finished; profile failed; SOL={value} ({summary.detail})"
    return f"eval finished; profile unavailable; SOL={value} ({summary.detail})"


def run_campaign(
    config_path: str,
    lineage_values: tuple[str, ...] | None,
    campaign_value: str | None,
    target_epoch_number: int,
    *,
    finalize: bool,
) -> None:
    """Create or resume selected DSL lineages and print their durable final state."""
    settings = RuntimeSettings.from_file(config_path)
    lineage_ids = (
        None
        if lineage_values is None
        else tuple(parse_lineage_id(value) for value in lineage_values)
    )
    artifacts = LocalArtifactStore(settings.storage.artifacts_root)
    with SqliteRegistry(settings.storage.registry_database) as progress_registry:

        def attempt_detail(attempt: Attempt) -> str:
            if attempt.output_kernel_revision_id is None:
                return "no candidate evaluation was committed"
            revision = progress_registry.get_kernel_revision(attempt.output_kernel_revision_id)
            return _sol_execution_detail(
                gateway_result_sol_summary(
                    artifacts,
                    revision.evaluation.gateway_result_digest,
                )
            )

        progress = AttemptProgressRenderer(sys.stderr, attempt_detail=attempt_detail)
        with build_campaign_runtime(
            settings,
            os.environ,
            attempt_finished=progress,
        ) as runtime:
            if campaign_value is not None:
                campaign_id = parse_campaign_id(campaign_value)

                async def run_registered() -> CampaignScheduleResult:
                    return await runtime.scheduler.run_registered_campaign_through(
                        campaign_id,
                        target_epoch_number,
                        finalize=finalize,
                    )

                result = anyio.run(run_registered)
            else:
                if finalize:
                    raise ValueError("--finalize requires --campaign discovery")
                if lineage_ids is None:
                    raise AssertionError("validated run-campaign target is absent")
                result = anyio.run(
                    runtime.scheduler.run_campaign_through,
                    lineage_ids,
                    target_epoch_number,
                )
    print(
        json.dumps(
            {
                "campaign_id": result.campaign_id,
                "target_epoch_number": target_epoch_number,
                "lineages": [
                    {
                        "lineage_id": scheduled.lineage.id,
                        "dsl": scheduled.lineage.dsl.value,
                        "status": scheduled.lineage.status.value,
                        "next_epoch_number": scheduled.lineage.next_epoch_number,
                        "active_kernel_agent_revision_id": (
                            scheduled.lineage.active_kernel_agent_revision_id
                        ),
                        "best_kernel_revision_id": scheduled.lineage.best_kernel_revision_id,
                        "evidence_checkpoint": scheduled.lineage.evidence_checkpoint,
                        "completed_epochs": scheduled.completed_epochs,
                    }
                    for scheduled in result.lineages
                ],
            },
            sort_keys=True,
        )
    )


def cancel_campaign(config_path: str, campaign_value: str) -> None:
    """Cancel one quiescent Campaign and report its terminal status."""
    settings = RuntimeSettings.from_file(config_path)
    with SqliteRegistry(settings.storage.registry_database) as registry:
        campaign = registry.cancel_campaign(parse_campaign_id(campaign_value))
    print(json.dumps({"campaign_id": campaign.id, "status": campaign.status.value}, sort_keys=True))


def run_task_worker(config_path: str, *, watch: bool) -> None:
    """Run one durable task or continuously poll the trusted task queue."""
    settings = RuntimeSettings.from_file(config_path)
    administration = settings.administration
    if administration is None:
        raise ValueError("Runtime configuration does not define administration settings")
    with (
        build_campaign_runtime(settings, os.environ) as runtime,
        SqliteRegistry(settings.storage.registry_database) as registry,
    ):
        worker = CampaignTaskWorker(
            registry,
            runtime.scheduler,
            owner=f"task-worker:{uuid4().hex}",
            lease_seconds=administration.task_lease_seconds,
            heartbeat_seconds=administration.task_heartbeat_seconds,
            max_error_bytes=administration.max_error_bytes,
        )
        result = anyio.run(
            _watch_campaign_tasks if watch else _run_one_campaign_task,
            worker,
            administration.task_poll_seconds,
        )
    if result is not None:
        print(
            json.dumps(
                {
                    "task_id": result.id,
                    "campaign_id": result.campaign_id,
                    "status": result.status.value,
                    "attempt_count": result.attempt_count,
                    "last_error": result.last_error,
                },
                sort_keys=True,
            )
        )


async def _run_one_campaign_task(
    worker: CampaignTaskWorker,
    _poll_seconds: float,
) -> CampaignTask | None:
    return await worker.run_once()


async def _watch_campaign_tasks(
    worker: CampaignTaskWorker,
    poll_seconds: float,
) -> None:
    while True:
        task = await worker.run_once()
        if task is None:
            await anyio.sleep(poll_seconds)


def recover_epoch(
    config_path: str,
    epoch_value: str,
    recovery_key: str,
    reason: str,
) -> None:
    """Recover one failed epoch and print the durable recovery identity."""
    settings = RuntimeSettings.from_file(config_path)
    with SqliteRegistry(settings.storage.registry_database) as registry:
        recovery = registry.recover_failed_epoch(
            parse_epoch_id(epoch_value),
            recovery_key=recovery_key,
            reason=reason,
        )
    print(
        json.dumps(
            {
                "epoch_id": recovery.epoch_id,
                "lineage_id": recovery.lineage_id,
                "campaign_id": recovery.campaign_id,
                "recovery_key": recovery.recovery_key,
                "generation": recovery.generation,
                "attempt_ids": recovery.attempt_ids,
                "created_at": recovery.created_at,
            },
            sort_keys=True,
        )
    )
