"""CLI maintenance and garbage collection commands."""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path

from ..artifacts.local import LocalArtifactStore
from ..config import RuntimeSettings
from ..gateway.control import SqliteGatewayControl
from ..maintenance import ArtifactGarbageCollector, WorkspaceGarbageCollector
from ..registry.sqlite import SqliteRegistry
from ..secrets import read_capability_signing_key
from ..workers.evolver_bundle import evolver_bundle_sha256


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
