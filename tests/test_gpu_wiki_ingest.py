"""Durable post-Epoch GPU Wiki feedback delivery tests."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path

import pytest
from conftest import digest
from pydantic import SecretStr

from atrex_runtime.artifacts.local import ArtifactKind, JsonValue, LocalArtifactStore
from atrex_runtime.domain.ids import (
    WikiFeedbackId,
    new_campaign_id,
    new_epoch_id,
    new_lineage_id,
    new_wiki_feedback_id,
)
from atrex_runtime.domain.models import WikiFeedbackOutboxItem, WikiFeedbackStatus
from atrex_runtime.knowledge import (
    HttpGpuWikiFeedbackClient,
    KnowledgeUnavailableError,
    WikiFeedbackAckV1,
    WikiFeedbackDrainer,
    WikiFeedbackDrainResult,
    WikiFeedbackReportV1,
)
from atrex_runtime.knowledge.client import GpuWikiHttpResponse, GpuWikiHttpTransport
from atrex_runtime.knowledge.models import canonical_json_bytes


def _report() -> WikiFeedbackReportV1:
    return WikiFeedbackReportV1(
        campaign_id=new_campaign_id(),
        lineage_id=new_lineage_id(),
        epoch_id=new_epoch_id(),
        epoch_number=1,
        operator="vector_add",
        dsl="triton",
        hardware_target="nvidia-h100",
        evaluation_contract_digest=digest("contract"),
        evidence_checkpoint_digest=digest("evidence"),
        attempts=(),
    )


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


@pytest.mark.anyio
async def test_http_feedback_client_uses_idempotency_key_and_strict_ack() -> None:
    feedback_id = new_wiki_feedback_id()
    report = _report()
    acknowledgement: dict[str, JsonValue] = {
        "schema_version": 1,
        "service_api_version": 1,
        "feedback_id": feedback_id,
        "accepted": True,
    }
    transport = CapturingTransport(GpuWikiHttpResponse(202, canonical_json_bytes(acknowledgement)))
    client = HttpGpuWikiFeedbackClient(
        transport,
        bearer_token=SecretStr("wiki-secret"),
        timeout_seconds=7,
        max_request_bytes=65536,
        max_response_bytes=4096,
    )

    result = await client.send(feedback_id, report)

    assert result.feedback_id == feedback_id
    assert transport.calls == [
        (
            "/v1/knowledge/epoch-feedback",
            report.canonical_json_bytes(),
            {
                "content-type": "application/json",
                "accept": "application/json",
                "idempotency-key": feedback_id,
                "authorization": "Bearer wiki-secret",
            },
            7,
            4096,
        )
    ]


@dataclass
class FakeOutbox:
    item: WikiFeedbackOutboxItem
    completed: list[WikiFeedbackId] = field(default_factory=list)
    retries: list[tuple[WikiFeedbackId, str, str]] = field(default_factory=list)
    failures: list[tuple[WikiFeedbackId, str]] = field(default_factory=list)

    def claim_wiki_feedback(
        self,
        owner: str,
        *,
        now: str,
        lease_expires_at: str,
        limit: int,
    ) -> list[WikiFeedbackOutboxItem]:
        del now, limit
        return [
            replace(
                self.item,
                status=WikiFeedbackStatus.LEASED,
                attempt_count=self.item.attempt_count + 1,
                lease_owner=owner,
                lease_expires_at=lease_expires_at,
            )
        ]

    def complete_wiki_feedback(self, item_id: WikiFeedbackId, owner: str) -> None:
        del owner
        self.completed.append(item_id)

    def retry_wiki_feedback(
        self,
        item_id: WikiFeedbackId,
        owner: str,
        *,
        available_at: str,
        error: str,
    ) -> None:
        del owner
        self.retries.append((item_id, available_at, error))

    def fail_wiki_feedback(
        self,
        item_id: WikiFeedbackId,
        owner: str,
        *,
        error: str,
    ) -> None:
        del owner
        self.failures.append((item_id, error))


@dataclass
class RecordingFeedbackClient:
    error: Exception | None = None
    calls: list[tuple[WikiFeedbackId, WikiFeedbackReportV1]] = field(default_factory=list)

    async def send(
        self,
        feedback_id: WikiFeedbackId,
        report: WikiFeedbackReportV1,
    ) -> WikiFeedbackAckV1:
        self.calls.append((feedback_id, report))
        if self.error is not None:
            raise self.error
        return WikiFeedbackAckV1(feedback_id=feedback_id, accepted=True)


def _outbox_with_report(
    tmp_path: Path,
) -> tuple[LocalArtifactStore, WikiFeedbackOutboxItem, WikiFeedbackReportV1]:
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    report = _report()
    report_digest = artifacts.put_json(
        report.model_dump(mode="json"),
        ArtifactKind.WIKI_FEEDBACK_REPORT,
    )
    item = WikiFeedbackOutboxItem(
        id=new_wiki_feedback_id(),
        lineage_id=report.lineage_id,
        epoch_number=report.epoch_number,
        report_artifact_digest=report_digest,
        status=WikiFeedbackStatus.PENDING,
        attempt_count=0,
        available_at="2026-08-15T00:00:00+00:00",
        lease_owner=None,
        lease_expires_at=None,
        last_error=None,
        created_at="2026-08-15T00:00:00+00:00",
        completed_at=None,
    )
    return artifacts, item, report


def _drainer(
    outbox: FakeOutbox,
    artifacts: LocalArtifactStore,
    client: RecordingFeedbackClient,
) -> WikiFeedbackDrainer:
    return WikiFeedbackDrainer(
        outbox,
        artifacts,
        client,
        batch_size=4,
        lease_seconds=30,
        retry_initial_seconds=5,
        retry_max_seconds=60,
        max_error_bytes=200,
        clock=lambda: datetime(2026, 8, 15, tzinfo=UTC),
    )


@pytest.mark.anyio
async def test_drainer_completes_an_acknowledged_report(tmp_path: Path) -> None:
    artifacts, item, report = _outbox_with_report(tmp_path)
    outbox = FakeOutbox(item)
    client = RecordingFeedbackClient()

    result = await _drainer(outbox, artifacts, client).drain_once()

    assert result == WikiFeedbackDrainResult(1, 1, 0, 0)
    assert client.calls == [(item.id, report)]
    assert outbox.completed == [item.id]


@pytest.mark.anyio
async def test_drainer_retries_only_temporary_failures_with_backoff(tmp_path: Path) -> None:
    artifacts, item, _report_value = _outbox_with_report(tmp_path)
    outbox = FakeOutbox(replace(item, attempt_count=2))
    client = RecordingFeedbackClient(KnowledgeUnavailableError("temporary"))

    result = await _drainer(outbox, artifacts, client).drain_once()

    assert (result.completed, result.retried, result.failed) == (0, 1, 0)
    assert outbox.retries[0][0] == item.id
    assert outbox.retries[0][1] == "2026-08-15T00:00:20+00:00"


@pytest.mark.anyio
async def test_drainer_marks_rejected_reports_permanently_failed(tmp_path: Path) -> None:
    artifacts, item, _report_value = _outbox_with_report(tmp_path)
    outbox = FakeOutbox(item)
    client = RecordingFeedbackClient(RuntimeError("rejected with status 400"))

    result = await _drainer(outbox, artifacts, client).drain_once()

    assert (result.completed, result.retried, result.failed) == (0, 0, 1)
    assert outbox.failures[0][0] == item.id
