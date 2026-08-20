"""Production composition for the independent GPU Wiki feedback worker."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime, timedelta

from ..artifacts.local import LocalArtifactStore
from ..config import RuntimeSettings
from ..domain.ids import WikiFeedbackId
from ..domain.models import WikiFeedbackOutboxItem
from ..knowledge import (
    HttpGpuWikiFeedbackClient,
    HttpxGpuWikiTransport,
    WikiFeedbackDrainer,
)
from ..registry.sqlite import SqliteRegistry
from ..secrets import required_secret


class WikiFeedbackRuntime:
    """Own the Registry connection used by one independent drain process."""

    def __init__(
        self,
        drainer: WikiFeedbackDrainer,
        registry: SqliteRegistry,
        *,
        poll_seconds: float,
        completed_retention_seconds: float,
        prune_batch_size: int,
    ) -> None:
        self.drainer = drainer
        self.poll_seconds = poll_seconds
        self.completed_retention_seconds = completed_retention_seconds
        self.prune_batch_size = prune_batch_size
        self._registry = registry
        self._closed = False

    def close(self) -> None:
        """Close the owned Registry connection once."""
        if self._closed:
            return
        self._closed = True
        self._registry.close()

    def __enter__(self) -> WikiFeedbackRuntime:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    def requeue(self, item_id: WikiFeedbackId) -> WikiFeedbackOutboxItem:
        """Administratively requeue one inspected permanent failure immediately."""
        return self._registry.requeue_wiki_feedback(
            item_id,
            available_at=datetime.now(UTC).isoformat(),
        )

    def maintain(self, *, compact: bool) -> int:
        """Prune one retained completed batch and optionally rebuild SQLite."""
        cutoff = datetime.now(UTC) - timedelta(seconds=self.completed_retention_seconds)
        pruned = self._registry.prune_wiki_feedback(
            completed_before=cutoff.isoformat(),
            limit=self.prune_batch_size,
        )
        if compact:
            self._registry.compact()
        return pruned


def build_wiki_feedback_runtime(
    settings: RuntimeSettings,
    environment: Mapping[str, str],
) -> WikiFeedbackRuntime:
    """Assemble the configured feedback Drainer or reject a disabled deployment."""
    wiki = settings.gpu_wiki
    if wiki is None or wiki.feedback is None:
        raise ValueError("Runtime configuration does not enable GPU Wiki feedback")
    feedback = wiki.feedback
    registry = SqliteRegistry(settings.storage.registry_database)
    try:
        artifacts = LocalArtifactStore(settings.storage.artifacts_root)
        client = HttpGpuWikiFeedbackClient(
            HttpxGpuWikiTransport(wiki.base_url),
            bearer_token=required_secret(environment, wiki.bearer_token_env),
            timeout_seconds=wiki.timeout_seconds,
            max_request_bytes=feedback.max_request_bytes,
            max_response_bytes=wiki.max_response_bytes,
        )
        drainer = WikiFeedbackDrainer(
            registry,
            artifacts,
            client,
            batch_size=feedback.batch_size,
            lease_seconds=feedback.lease_seconds,
            retry_initial_seconds=feedback.retry_initial_seconds,
            retry_max_seconds=feedback.retry_max_seconds,
            max_error_bytes=feedback.max_error_bytes,
        )
        return WikiFeedbackRuntime(
            drainer,
            registry,
            poll_seconds=feedback.poll_seconds,
            completed_retention_seconds=feedback.completed_retention_seconds,
            prune_batch_size=feedback.prune_batch_size,
        )
    except BaseException:
        registry.close()
        raise
