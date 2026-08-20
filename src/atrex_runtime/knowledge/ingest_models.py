"""Versioned post-Epoch GPU Wiki consumption and Trace feedback records."""

from __future__ import annotations

from typing import Literal, cast

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..artifacts.local import JsonValue
from ..domain.ids import (
    ArtifactDigest,
    AttemptId,
    CampaignId,
    EpochId,
    KernelAgentRevisionId,
    LineageId,
    WikiFeedbackId,
    parse_artifact_digest,
    parse_attempt_id,
    parse_campaign_id,
    parse_epoch_id,
    parse_kernel_agent_revision_id,
    parse_lineage_id,
    parse_wiki_feedback_id,
)
from ..domain.models import BranchRole, Dsl
from .models import GPU_WIKI_API_VERSION, KnowledgeInteractionV1, canonical_json_bytes

WIKI_FEEDBACK_VERSION: Literal[1] = 1
WIKI_FEEDBACK_ACK_VERSION: Literal[1] = 1


class WikiFeedbackTokenUsageV1(BaseModel):
    """Provider token buckets associated with one sealed Optimizer Session."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    uncached_input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    cache_read_tokens: int = Field(ge=0)
    cache_write_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)

    @model_validator(mode="after")
    def _validate_total(self) -> WikiFeedbackTokenUsageV1:
        actual = (
            self.uncached_input_tokens
            + self.output_tokens
            + self.cache_read_tokens
            + self.cache_write_tokens
        )
        if self.total_tokens != actual:
            raise ValueError("Wiki feedback token total does not match its buckets")
        return self


class WikiFeedbackSessionTraceV1(BaseModel):
    """One immutable Session reference plus its exact bounded upload projection."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    run_ordinal: int = Field(gt=0)
    artifact_digest: ArtifactDigest
    finish_reason: str = Field(min_length=1, max_length=500)
    token_budget: int = Field(gt=0)
    token_usage: WikiFeedbackTokenUsageV1
    projection: JsonValue | None = None

    @field_validator("artifact_digest", mode="before")
    @classmethod
    def _validate_digest(cls, value: object) -> ArtifactDigest:
        if not isinstance(value, str):
            raise ValueError("Session Trace digest must be a string")
        return parse_artifact_digest(value)


class WikiFeedbackInteractionV1(BaseModel):
    """One frozen query/response Artifact and its complete strict payload."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    artifact_digest: ArtifactDigest
    interaction: KnowledgeInteractionV1

    @field_validator("artifact_digest", mode="before")
    @classmethod
    def _validate_digest(cls, value: object) -> ArtifactDigest:
        if not isinstance(value, str):
            raise ValueError("Wiki interaction digest must be a string")
        return parse_artifact_digest(value)


class WikiFeedbackAttemptV1(BaseModel):
    """Wiki consumption and Session Trace observations for one Epoch Attempt."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    attempt_id: AttemptId
    branch: BranchRole
    ordinal: int = Field(gt=0)
    kernel_agent_revision_id: KernelAgentRevisionId
    interactions: tuple[WikiFeedbackInteractionV1, ...]
    session_traces: tuple[WikiFeedbackSessionTraceV1, ...]

    @field_validator("attempt_id", mode="before")
    @classmethod
    def _validate_attempt_id(cls, value: object) -> AttemptId:
        if not isinstance(value, str):
            raise ValueError("Attempt ID must be a string")
        return parse_attempt_id(value)

    @field_validator("kernel_agent_revision_id", mode="before")
    @classmethod
    def _validate_agent_id(cls, value: object) -> KernelAgentRevisionId:
        if not isinstance(value, str):
            raise ValueError("Kernel Agent revision ID must be a string")
        return parse_kernel_agent_revision_id(value)


class WikiFeedbackReportV1(BaseModel):
    """Frozen Wiki consumption and Trace observations for one completed Epoch."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1] = WIKI_FEEDBACK_VERSION
    service_api_version: Literal[1] = GPU_WIKI_API_VERSION
    campaign_id: CampaignId
    lineage_id: LineageId
    epoch_id: EpochId
    epoch_number: int = Field(gt=0)
    operator: str = Field(min_length=1, max_length=500)
    dsl: Dsl
    hardware_target: str = Field(min_length=1, max_length=500)
    evaluation_contract_digest: ArtifactDigest
    evidence_checkpoint_digest: ArtifactDigest
    attempts: tuple[WikiFeedbackAttemptV1, ...]

    @field_validator("campaign_id", mode="before")
    @classmethod
    def _validate_campaign_id(cls, value: object) -> CampaignId:
        if not isinstance(value, str):
            raise ValueError("Campaign ID must be a string")
        return parse_campaign_id(value)

    @field_validator("lineage_id", mode="before")
    @classmethod
    def _validate_lineage_id(cls, value: object) -> LineageId:
        if not isinstance(value, str):
            raise ValueError("lineage ID must be a string")
        return parse_lineage_id(value)

    @field_validator("epoch_id", mode="before")
    @classmethod
    def _validate_epoch_id(cls, value: object) -> EpochId:
        if not isinstance(value, str):
            raise ValueError("Epoch ID must be a string")
        return parse_epoch_id(value)

    @field_validator("evaluation_contract_digest", "evidence_checkpoint_digest", mode="before")
    @classmethod
    def _validate_digest(cls, value: object) -> ArtifactDigest:
        if not isinstance(value, str):
            raise ValueError("Wiki feedback digest must be a string")
        return parse_artifact_digest(value)

    def canonical_json_bytes(self) -> bytes:
        """Serialize the exact HTTP feedback request body."""
        return canonical_json_bytes(cast(JsonValue, self.model_dump(mode="json")))


class WikiFeedbackAckV1(BaseModel):
    """Strict acknowledgement echoing the accepted feedback identity."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1] = WIKI_FEEDBACK_ACK_VERSION
    service_api_version: Literal[1] = GPU_WIKI_API_VERSION
    feedback_id: WikiFeedbackId
    accepted: Literal[True]

    @field_validator("feedback_id", mode="before")
    @classmethod
    def _validate_feedback_id(cls, value: object) -> WikiFeedbackId:
        if not isinstance(value, str):
            raise ValueError("Wiki feedback ID must be a string")
        return parse_wiki_feedback_id(value)
