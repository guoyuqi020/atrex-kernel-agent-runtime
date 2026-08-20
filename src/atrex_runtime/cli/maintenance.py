"""CLI maintenance, garbage collection, and Wiki outbox commands."""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path

import anyio

from ..artifacts.local import LocalArtifactStore
from ..composition.knowledge import build_wiki_feedback_runtime
from ..config import RuntimeSettings
from ..domain.ids import parse_wiki_feedback_id
from ..gateway.control import SqliteGatewayControl
from ..knowledge import WikiFeedbackDrainer, WikiFeedbackDrainResult
from ..maintenance import ArtifactGarbageCollector, WorkspaceGarbageCollector
from ..registry.sqlite import SqliteRegistry
from ..secrets import read_capability_signing_key
from ..workers.evolver_bundle import evolver_bundle_sha256


def drain_wiki_feedback(config_path: str, *, watch: bool) -> None:
    """Deliver one batch or continuously poll the independent feedback Outbox."""
    settings = RuntimeSettings.from_file(config_path)
    with build_wiki_feedback_runtime(settings, os.environ) as runtime:
        if watch:
            anyio.run(_watch_wiki_feedback, runtime.drainer, runtime.poll_seconds)
            return
        result = anyio.run(runtime.drainer.drain_once)
    print(json.dumps(_drain_result_value(result), sort_keys=True))


def requeue_wiki_feedback(config_path: str, item_value: str) -> None:
    settings = RuntimeSettings.from_file(config_path)
    with build_wiki_feedback_runtime(settings, os.environ) as runtime:
        item = runtime.requeue(parse_wiki_feedback_id(item_value))
    print(json.dumps({"item_id": item.id, "status": item.status.value}, sort_keys=True))


def maintain_wiki_feedback(config_path: str, *, compact: bool) -> None:
    settings = RuntimeSettings.from_file(config_path)
    with build_wiki_feedback_runtime(settings, os.environ) as runtime:
        pruned = runtime.maintain(compact=compact)
    print(json.dumps({"pruned": pruned, "compacted": compact}, sort_keys=True))


def gc_artifacts(
    config_path: str,
    *,
    minimum_age_seconds: float,
    limit: int,
    apply: bool,
    confirm_runtime_stopped: bool,
) -> None:
    """Run one dry-run or explicitly confirmed offline Artifact GC pass."""
    if apply and not confirm_runtime_stopped:
        raise ValueError("--apply requires --confirm-runtime-stopped")
    settings = RuntimeSettings.from_file(config_path)
    signing_key = read_capability_signing_key(
        os.environ,
        settings.gateway_proxy.capability_signing_key_env,
    )
    with SqliteRegistry(settings.storage.registry_database) as registry:
        control = SqliteGatewayControl(
            settings.storage.gateway_database,
            registry,
            signing_key=signing_key,
        )
        try:
            result = ArtifactGarbageCollector(
                registry,
                control,
                LocalArtifactStore(settings.storage.artifacts_root),
            ).run(
                minimum_age_seconds=minimum_age_seconds,
                limit=limit,
                apply=apply,
            )
        finally:
            control.close()
    print(json.dumps(asdict(result), sort_keys=True))


def gc_workspaces(
    config_path: str,
    *,
    minimum_age_seconds: float,
    limit: int,
    apply: bool,
    confirm_runtime_stopped: bool,
) -> None:
    """Run one dry-run or explicitly confirmed offline Worker workspace GC pass."""
    if apply and not confirm_runtime_stopped:
        raise ValueError("--apply requires --confirm-runtime-stopped")
    settings = RuntimeSettings.from_file(config_path)
    campaign = settings.campaign
    if campaign is None:
        raise ValueError("Workspace GC requires Campaign runtime configuration")
    roots = (
        campaign.attempt_workspaces_root,
        campaign.evolution_workspaces_root,
        campaign.problem_generalization_workspaces_root,
        campaign.lineage_bootstrap_workspaces_root,
    )
    with SqliteRegistry(settings.storage.registry_database) as registry:
        result = WorkspaceGarbageCollector(roots, registry).run(
            minimum_age_seconds=minimum_age_seconds,
            limit=limit,
            apply=apply,
        )
    print(json.dumps(asdict(result), sort_keys=True))


def digest_evolver_bundle(path: str, *, max_files: int, max_bytes: int) -> None:
    """Validate one local Bundle and print its canonical deployment identity."""
    digest = evolver_bundle_sha256(
        Path(path).expanduser().resolve(),
        max_files=max_files,
        max_bytes=max_bytes,
    )
    print(json.dumps({"bundle_sha256": digest}, sort_keys=True))


async def _watch_wiki_feedback(
    drainer: WikiFeedbackDrainer,
    poll_seconds: float,
) -> None:
    """Poll forever while allowing cancellation between bounded drain passes."""
    while True:
        result = await drainer.drain_once()
        if result.claimed == 0:
            await anyio.sleep(poll_seconds)


def _drain_result_value(result: WikiFeedbackDrainResult) -> dict[str, int]:
    return {
        "claimed": result.claimed,
        "completed": result.completed,
        "retried": result.retried,
        "failed": result.failed,
    }
