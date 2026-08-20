"""Assembly of post-Epoch GPU Wiki consumption and Trace feedback."""

from __future__ import annotations

from collections import defaultdict
from typing import Protocol, cast

from ..artifacts.local import ArtifactKind, JsonValue, LocalArtifactStore
from ..controller.projection import EvidenceArtifactProjector
from ..domain.ids import ArtifactDigest, AttemptId, LineageId
from ..domain.models import EpochStatus
from ..gateway.control_models import GatewayOperation
from ..registry.base import Registry
from .ingest_models import (
    WikiFeedbackAttemptV1,
    WikiFeedbackInteractionV1,
    WikiFeedbackReportV1,
    WikiFeedbackSessionTraceV1,
    WikiFeedbackTokenUsageV1,
)
from .models import KnowledgeInteractionV1


class WikiInteractionSource(Protocol):
    """Read frozen Wiki interaction Artifacts committed by the Runtime Proxy."""

    def list_operation_artifacts(
        self,
        attempt_ids: tuple[AttemptId, ...],
        operation: GatewayOperation,
    ) -> tuple[tuple[AttemptId, str, ArtifactDigest], ...]:
        """List committed operation Artifacts for the supplied Attempts."""
        ...


class WikiFeedbackPreparer(Protocol):
    """Seal one deterministic feedback report before Evidence publication."""

    def prepare(
        self,
        lineage_id: LineageId,
        epoch_number: int,
        evidence_checkpoint: ArtifactDigest,
    ) -> ArtifactDigest:
        """Return the sealed feedback report Artifact for the completed Epoch."""
        ...


class LocalWikiFeedbackPreparer:
    """Project frozen interactions and exact bounded Session traces into Epoch feedback."""

    def __init__(
        self,
        registry: Registry,
        artifacts: LocalArtifactStore,
        interactions: WikiInteractionSource,
        projector: EvidenceArtifactProjector,
    ) -> None:
        self._registry = registry
        self._artifacts = artifacts
        self._interactions = interactions
        self._projector = projector

    def prepare(
        self,
        lineage_id: LineageId,
        epoch_number: int,
        evidence_checkpoint: ArtifactDigest,
    ) -> ArtifactDigest:
        """Seal only Wiki consumption and bounded Trace observations for one Epoch."""
        lineage = self._registry.get_lineage(lineage_id)
        campaign = self._registry.get_campaign(lineage.campaign_id)
        epoch = self._registry.find_epoch(lineage_id, epoch_number)
        if epoch is None or epoch.status is not EpochStatus.COMPLETED:
            raise ValueError("Wiki feedback requires one completed Epoch")
        evidence = self._artifacts.verify(evidence_checkpoint)
        if evidence.kind is not ArtifactKind.EVIDENCE:
            raise ValueError("Wiki feedback Evidence has the wrong Artifact kind")

        attempts = tuple(self._registry.list_attempts(epoch.id))
        attempt_ids = tuple(attempt.id for attempt in attempts)
        interaction_rows = self._interactions.list_operation_artifacts(
            attempt_ids,
            GatewayOperation.WIKI_QUERY,
        )
        by_attempt: dict[AttemptId, list[WikiFeedbackInteractionV1]] = defaultdict(list)
        for attempt_id, _idempotency_key, digest in interaction_rows:
            artifact = self._artifacts.verify(digest)
            if artifact.kind is not ArtifactKind.WIKI_INTERACTION:
                raise ValueError("Wiki interaction index has the wrong Artifact kind")
            payload = (artifact.payload_path / "value.json").read_bytes()
            interaction = KnowledgeInteractionV1.model_validate_json(payload)
            if interaction.query.attempt_id != attempt_id:
                raise ValueError("Wiki interaction disagrees with its Attempt index")
            by_attempt[attempt_id].append(
                WikiFeedbackInteractionV1(artifact_digest=digest, interaction=interaction)
            )

        report = WikiFeedbackReportV1(
            campaign_id=campaign.id,
            lineage_id=lineage.id,
            epoch_id=epoch.id,
            epoch_number=epoch.number,
            operator=campaign.operator,
            dsl=lineage.dsl,
            hardware_target=lineage.hardware_target,
            evaluation_contract_digest=campaign.evaluation_contract_digest,
            evidence_checkpoint_digest=evidence_checkpoint,
            attempts=tuple(
                WikiFeedbackAttemptV1(
                    attempt_id=attempt.id,
                    branch=attempt.branch,
                    ordinal=attempt.ordinal,
                    kernel_agent_revision_id=attempt.kernel_agent_revision_id,
                    interactions=tuple(by_attempt[attempt.id]),
                    session_traces=tuple(
                        WikiFeedbackSessionTraceV1(
                            run_ordinal=trace.run_ordinal,
                            artifact_digest=trace.artifact_digest,
                            finish_reason=trace.finish_reason,
                            token_budget=trace.token_budget,
                            token_usage=WikiFeedbackTokenUsageV1(
                                uncached_input_tokens=trace.token_usage.uncached_input_tokens,
                                output_tokens=trace.token_usage.output_tokens,
                                cache_read_tokens=trace.token_usage.cache_read_tokens,
                                cache_write_tokens=trace.token_usage.cache_write_tokens,
                                total_tokens=trace.token_usage.total_tokens,
                            ),
                            projection=self._projector.raw_session_projection(
                                trace.artifact_digest
                            ),
                        )
                        for trace in self._registry.list_attempt_session_traces(attempt.id)
                    ),
                )
                for attempt in attempts
            ),
        )
        return self._artifacts.put_json(
            cast(JsonValue, report.model_dump(mode="json")),
            ArtifactKind.WIKI_FEEDBACK_REPORT,
        )
