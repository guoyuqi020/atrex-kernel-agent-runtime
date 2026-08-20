"""Registry-backed lineage fencing tests."""

from __future__ import annotations

from pathlib import Path

import pytest
from conftest import seed_lineage

from atrex_runtime.controller import RegistryLineageLeaseManager
from atrex_runtime.domain.errors import InvalidTransitionError
from atrex_runtime.domain.models import CampaignStatus
from atrex_runtime.registry.sqlite import SqliteRegistry


def test_registry_fence_is_required_for_campaign_scheduler_writes(tmp_path: Path) -> None:
    database = tmp_path / "registry.sqlite"
    with SqliteRegistry(database) as setup:
        seeded = seed_lineage(setup)
        campaign_id = setup.get_lineage(seeded.lineage_id).campaign_id
    with SqliteRegistry(database, require_fencing=True) as registry:
        with pytest.raises(InvalidTransitionError, match="requires a lineage fencing token"):
            registry.cancel_campaign(campaign_id)

        manager = RegistryLineageLeaseManager(
            registry,
            lease_seconds=10,
            heartbeat_seconds=1,
        )
        with manager.acquire(seeded.lineage_id):
            cancelled = registry.cancel_campaign(campaign_id)

        assert cancelled.status is CampaignStatus.CANCELLED


def test_registry_rejects_a_superseded_fencing_generation(tmp_path: Path) -> None:
    database = tmp_path / "registry.sqlite"
    with SqliteRegistry(database) as setup:
        seeded = seed_lineage(setup)
        campaign_id = setup.get_lineage(seeded.lineage_id).campaign_id
    with SqliteRegistry(database, require_fencing=True) as registry:
        first = registry.acquire_lineage_fence(
            seeded.lineage_id,
            "old-owner",
            now="2020-01-01T00:00:00+00:00",
            lease_expires_at="2020-01-01T00:00:01+00:00",
        )
        registry.acquire_lineage_fence(
            seeded.lineage_id,
            "new-owner",
            now="2020-01-01T00:00:02+00:00",
            lease_expires_at="2099-01-01T00:00:00+00:00",
        )
        token = registry.activate_lineage_fence(seeded.lineage_id, first, "old-owner")
        try:
            with pytest.raises(InvalidTransitionError, match="stale"):
                registry.cancel_campaign(campaign_id)
        finally:
            registry.deactivate_lineage_fence(token)
