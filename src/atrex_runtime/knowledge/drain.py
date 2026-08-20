"""Lease-based independent delivery of durable GPU Wiki feedback reports."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Protocol
from uuid import uuid4

from ..artifacts.local import ArtifactKind, LocalArtifactStore
from ..domain.ids import WikiFeedbackId
from ..domain.models import WikiFeedbackOutboxItem
from .client import GpuWikiFeedbackClient, KnowledgeUnavailableError
from .ingest_models import WikiFeedbackReportV1


def _utc_now() -> datetime:
    return datetime.now(UTC)


class WikiFeedbackOutbox(Protocol):
    """Durable leasing operations required by the independent Drainer."""

    def claim_wiki_feedback(
        self,
        owner: str,
        *,
        now: str,
        lease_expires_at: str,
        limit: int,
    ) -> list[WikiFeedbackOutboxItem]: ...

    def complete_wiki_feedback(self, item_id: WikiFeedbackId, owner: str) -> None: ...

    def retry_wiki_feedback(
        self,
        item_id: WikiFeedbackId,
        owner: str,
        *,
        available_at: str,
        error: str,
    ) -> None: ...

    def fail_wiki_feedback(
        self,
        item_id: WikiFeedbackId,
        owner: str,
        *,
        error: str,
    ) -> None: ...


@dataclass(frozen=True, slots=True)
class WikiFeedbackDrainResult:
    """Counts produced by one bounded Outbox drain pass."""

    claimed: int
    completed: int
    retried: int
    failed: int


class WikiFeedbackDrainer:
    """Claim reports, deliver them, and persist retry or terminal outcomes."""

    def __init__(
        self,
        registry: WikiFeedbackOutbox,
        artifacts: LocalArtifactStore,
        client: GpuWikiFeedbackClient,
        *,
        batch_size: int,
        lease_seconds: float,
        retry_initial_seconds: float,
        retry_max_seconds: float,
        max_error_bytes: int,
        clock: Callable[[], datetime] = _utc_now,
    ) -> None:
        values = (
            batch_size,
            lease_seconds,
            retry_initial_seconds,
            retry_max_seconds,
            max_error_bytes,
        )
        if any(value <= 0 for value in values):
            raise ValueError("Wiki feedback drain limits must be positive")
        if retry_max_seconds < retry_initial_seconds:
            raise ValueError("Wiki feedback maximum retry must cover initial retry")
        self._registry = registry
        self._artifacts = artifacts
        self._client = client
        self._batch_size = batch_size
        self._lease_seconds = lease_seconds
        self._retry_initial_seconds = retry_initial_seconds
        self._retry_max_seconds = retry_max_seconds
        self._max_error_bytes = max_error_bytes
        self._clock = clock

    async def drain_once(self) -> WikiFeedbackDrainResult:
        """Process at most one configured batch without waiting for future work."""
        owner = f"wiki-drainer-{uuid4().hex}"
        now = self._clock()
        items = self._registry.claim_wiki_feedback(
            owner,
            now=now.isoformat(),
            lease_expires_at=(now + timedelta(seconds=self._lease_seconds)).isoformat(),
            limit=self._batch_size,
        )
        completed = 0
        retried = 0
        failed = 0
        for item in items:
            try:
                report = self._load_report(item)
                await self._client.send(item.id, report)
            except KnowledgeUnavailableError as error:
                delay = min(
                    self._retry_max_seconds,
                    self._retry_initial_seconds * (2 ** min(max(item.attempt_count - 1, 0), 30)),
                )
                available_at = self._clock() + timedelta(seconds=delay)
                self._registry.retry_wiki_feedback(
                    item.id,
                    owner,
                    available_at=available_at.isoformat(),
                    error=self._error_text(error),
                )
                retried += 1
            except (FileNotFoundError, RuntimeError, ValueError) as error:
                self._registry.fail_wiki_feedback(
                    item.id,
                    owner,
                    error=self._error_text(error),
                )
                failed += 1
            else:
                self._registry.complete_wiki_feedback(item.id, owner)
                completed += 1
        return WikiFeedbackDrainResult(len(items), completed, retried, failed)

    def _load_report(self, item: WikiFeedbackOutboxItem) -> WikiFeedbackReportV1:
        artifact = self._artifacts.verify(item.report_artifact_digest)
        if artifact.kind is not ArtifactKind.WIKI_FEEDBACK_REPORT:
            raise ValueError("Wiki feedback Outbox item has the wrong Artifact kind")
        report = WikiFeedbackReportV1.model_validate_json(
            (artifact.payload_path / "value.json").read_bytes()
        )
        if report.lineage_id != item.lineage_id or report.epoch_number != item.epoch_number:
            raise ValueError("Wiki feedback report disagrees with its Outbox identity")
        return report

    def _error_text(self, error: Exception) -> str:
        text = f"{type(error).__name__}: {error}"
        encoded = text.encode()
        if len(encoded) <= self._max_error_bytes:
            return text
        return encoded[: self._max_error_bytes].decode(errors="ignore")
