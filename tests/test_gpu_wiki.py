"""Attempt-scoped live GPU Wiki query and freeze-before-return tests."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from conftest import NOW, digest, seed_lineage
from pydantic import SecretStr, ValidationError

from atrex_runtime.artifacts.local import ArtifactKind, JsonValue, LocalArtifactStore
from atrex_runtime.domain.ids import new_attempt_id, new_epoch_id
from atrex_runtime.domain.models import Attempt, AttemptStatus, BranchRole, Epoch, EpochStatus
from atrex_runtime.gateway import (
    GatewayCapabilityPolicy,
    GatewayOperation,
    SqliteGatewayControl,
)
from atrex_runtime.knowledge import (
    GpuWikiHttpResponse,
    HttpGpuWikiClient,
    KnowledgeInteractionV1,
    KnowledgeQueryV1,
    KnowledgeSnapshotResponseV1,
    KnowledgeUnavailableError,
    WikiProxyLimits,
    WikiProxyService,
)
from atrex_runtime.knowledge.client import GpuWikiClient, GpuWikiHttpTransport
from atrex_runtime.knowledge.models import canonical_json_bytes
from atrex_runtime.registry.sqlite import SqliteRegistry

NOW_DATETIME = datetime(2026, 8, 14, tzinfo=UTC)


def _snapshot_value() -> dict[str, JsonValue]:
    content: JsonValue = {
        "records": {
            "nvidia.hopper.triton.kernel-opt.reduction": {
                "store": "gpu_wiki",
                "source": "kernel_wiki",
                "type": "technique-card",
                "applies_to": {"arch": "hopper", "dsl": "triton"},
                "match": {"arch": "exact"},
                "payload": {"goal": "tile a reduction"},
            }
        },
        "notes": [],
    }
    content_digest = hashlib.sha256(canonical_json_bytes(content)).hexdigest()
    return {
        "schema_version": 1,
        "service_api_version": 1,
        "snapshot_id": "wiki-snapshot-1",
        "content_digest": f"sha256:{content_digest}",
        "content": content,
    }


@dataclass
class CapturingTransport(GpuWikiHttpTransport):
    response: GpuWikiHttpResponse
    calls: list[tuple[str, bytes, dict[str, str], float, int]] = field(default_factory=list)

    async def post(
        self,
        path: str,
        body: bytes,
        headers: dict[str, str],
        *,
        timeout_seconds: float,
        max_response_bytes: int,
    ) -> GpuWikiHttpResponse:
        self.calls.append((path, body, headers, timeout_seconds, max_response_bytes))
        return self.response


@dataclass
class StaticWikiClient(GpuWikiClient):
    response: KnowledgeSnapshotResponseV1
    calls: list[KnowledgeQueryV1] = field(default_factory=list)

    async def query(self, request: KnowledgeQueryV1) -> KnowledgeSnapshotResponseV1:
        self.calls.append(request)
        return self.response


@dataclass
class UnavailableWikiClient(GpuWikiClient):
    async def query(self, request: KnowledgeQueryV1) -> KnowledgeSnapshotResponseV1:
        del request
        raise KnowledgeUnavailableError("offline")


def _insert_attempt(registry: SqliteRegistry) -> Attempt:
    seeded = seed_lineage(
        registry,
        evidence_checkpoint=digest("epoch-evidence"),
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
        evidence_checkpoint=digest("epoch-evidence"),
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


def _query(attempt: Attempt, registry: SqliteRegistry) -> KnowledgeQueryV1:
    epoch = registry.get_epoch(attempt.epoch_id)
    lineage = registry.get_lineage(epoch.lineage_id)
    campaign = registry.get_campaign(lineage.campaign_id)
    return KnowledgeQueryV1(
        campaign_id=campaign.id,
        lineage_id=lineage.id,
        epoch_id=epoch.id,
        epoch_number=epoch.number,
        attempt_id=attempt.id,
        branch=attempt.branch,
        attempt_ordinal=attempt.ordinal,
        kernel_agent_revision_id=attempt.kernel_agent_revision_id,
        operator=campaign.operator,
        dsl=lineage.dsl,
        hardware_target=lineage.hardware_target,
        evaluation_contract_digest=campaign.evaluation_contract_digest,
        epoch_evidence_checkpoint_digest=epoch.evidence_checkpoint,
        attempt_evidence_digest=attempt.attempt_evidence_digest,
        query="How should this reduction be tiled?",
    )


@pytest.mark.anyio
async def test_http_client_sends_strict_versioned_live_query(tmp_path: Path) -> None:
    with SqliteRegistry(tmp_path / "registry.sqlite") as registry:
        query = _query(_insert_attempt(registry), registry)
    transport = CapturingTransport(
        GpuWikiHttpResponse(200, canonical_json_bytes(_snapshot_value()))
    )
    client = HttpGpuWikiClient(
        transport,
        bearer_token=SecretStr("wiki-secret"),
        timeout_seconds=12.5,
        max_response_bytes=4096,
    )

    response = await client.query(query)

    assert response.snapshot_id == "wiki-snapshot-1"
    assert transport.calls == [
        (
            "/v1/knowledge/query",
            query.canonical_json_bytes(),
            {
                "content-type": "application/json",
                "accept": "application/json",
                "authorization": "Bearer wiki-secret",
            },
            12.5,
            4096,
        )
    ]


def test_snapshot_rejects_content_digest_disagreement() -> None:
    value = _snapshot_value()
    value["content"] = {"recommendations": []}
    with pytest.raises(ValidationError, match="content digest"):
        KnowledgeSnapshotResponseV1.model_validate(value)


@pytest.mark.anyio
async def test_proxy_freezes_before_return_and_replays_idempotently(tmp_path: Path) -> None:
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
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
            4,
            NOW_DATETIME + timedelta(hours=1),
        ),
    )
    client = StaticWikiClient(KnowledgeSnapshotResponseV1.model_validate(_snapshot_value()))
    service = WikiProxyService(
        control,
        control,
        registry,
        artifacts,
        client,
        WikiProxyLimits(4096, 1024),
        registry,
    )
    payload = json.dumps(
        {
            "schema_version": 1,
            "attempt_id": attempt.id,
            "idempotency_key": "tiling-1",
            "query": "How should this reduction be tiled?",
        }
    ).encode()

    first = await service.query(capability.token, payload)
    second = await service.query(capability.token, payload)

    assert second == first
    assert len(client.calls) == 1
    assert client.calls[0].attempt_id == attempt.id
    assert client.calls[0].branch is BranchRole.ACTIVE
    stored = artifacts.verify(first.interaction_artifact_digest)
    assert stored.kind is ArtifactKind.WIKI_INTERACTION
    interaction = KnowledgeInteractionV1.model_validate_json(
        (stored.payload_path / "value.json").read_bytes()
    )
    assert interaction.query.query == "How should this reduction be tiled?"
    assert interaction.response.content == first.content
    assert control.list_operation_artifacts((attempt.id,), GatewayOperation.WIKI_QUERY) == (
        (attempt.id, "tiling-1", first.interaction_artifact_digest),
    )
    assert first.interaction_artifact_digest in control.list_referenced_artifact_digests()
    events = [
        event.kind
        for event in registry.list_runtime_events(after_sequence=0, limit=100)
        if event.kind.startswith("wiki.query_")
    ]
    assert events == ["wiki.query_submitted", "wiki.query_completed"]
    control.close()
    registry.close()


@pytest.mark.anyio
async def test_proxy_propagates_temporary_wiki_unavailability(tmp_path: Path) -> None:
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    registry = SqliteRegistry(tmp_path / "registry.sqlite")
    attempt = _insert_attempt(registry)
    control = SqliteGatewayControl(
        tmp_path / "gateway.sqlite",
        registry,
        signing_key=b"x" * 32,
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
    service = WikiProxyService(
        control,
        control,
        registry,
        artifacts,
        UnavailableWikiClient(),
        WikiProxyLimits(4096, 1024),
        registry,
    )

    with pytest.raises(KnowledgeUnavailableError, match="offline"):
        await service.query(
            capability.token,
            json.dumps(
                {
                    "schema_version": 1,
                    "attempt_id": attempt.id,
                    "idempotency_key": "offline-1",
                    "query": "lookup",
                }
            ).encode(),
        )
    assert (
        control.get_operation_artifact(attempt.id, "offline-1", GatewayOperation.WIKI_QUERY) is None
    )
    control.close()
    registry.close()
