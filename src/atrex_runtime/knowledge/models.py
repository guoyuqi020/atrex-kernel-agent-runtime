"""Versioned external GPU Wiki query, response, and frozen interaction records."""

from __future__ import annotations

import hashlib
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..artifacts.local import JsonValue
from ..domain.ids import (
    ArtifactDigest,
    AttemptId,
    CampaignId,
    EpochId,
    KernelAgentRevisionId,
    LineageId,
    parse_artifact_digest,
    parse_attempt_id,
    parse_campaign_id,
    parse_epoch_id,
    parse_kernel_agent_revision_id,
    parse_lineage_id,
)
from ..domain.models import BranchRole, Dsl
from ..serialization import canonical_json_bytes as canonical_json_bytes

GPU_WIKI_API_VERSION: Literal[1] = 1
KNOWLEDGE_QUERY_VERSION: Literal[1] = 1
KNOWLEDGE_SNAPSHOT_VERSION: Literal[1] = 1
KNOWLEDGE_INTERACTION_VERSION: Literal[1] = 1


class KnowledgeContextV1(BaseModel):
    """Trusted Attempt context shared by external knowledge operations."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1] = KNOWLEDGE_QUERY_VERSION
    service_api_version: Literal[1] = GPU_WIKI_API_VERSION
    campaign_id: CampaignId
    lineage_id: LineageId
    epoch_id: EpochId
    epoch_number: int = Field(gt=0)
    attempt_id: AttemptId
    branch: BranchRole
    attempt_ordinal: int = Field(gt=0)
    kernel_agent_revision_id: KernelAgentRevisionId
    operator: str = Field(min_length=1, max_length=500)
    dsl: Dsl
    hardware_target: str = Field(min_length=1, max_length=500)
    evaluation_contract_digest: ArtifactDigest
    epoch_evidence_checkpoint_digest: ArtifactDigest
    attempt_evidence_digest: ArtifactDigest

    @field_validator("campaign_id", mode="before")
    @classmethod
    def _validate_campaign_id(cls, value: object) -> CampaignId:
        if not isinstance(value, str):
            raise ValueError("campaign_id must be a string")
        return parse_campaign_id(value)

    @field_validator("lineage_id", mode="before")
    @classmethod
    def _validate_lineage_id(cls, value: object) -> LineageId:
        if not isinstance(value, str):
            raise ValueError("lineage_id must be a string")
        return parse_lineage_id(value)

    @field_validator("epoch_id", mode="before")
    @classmethod
    def _validate_epoch_id(cls, value: object) -> EpochId:
        if not isinstance(value, str):
            raise ValueError("epoch_id must be a string")
        return parse_epoch_id(value)

    @field_validator("attempt_id", mode="before")
    @classmethod
    def _validate_attempt_id(cls, value: object) -> AttemptId:
        if not isinstance(value, str):
            raise ValueError("attempt_id must be a string")
        return parse_attempt_id(value)

    @field_validator("kernel_agent_revision_id", mode="before")
    @classmethod
    def _validate_agent_revision_id(cls, value: object) -> KernelAgentRevisionId:
        if not isinstance(value, str):
            raise ValueError("kernel_agent_revision_id must be a string")
        return parse_kernel_agent_revision_id(value)

    @field_validator(
        "evaluation_contract_digest",
        "epoch_evidence_checkpoint_digest",
        "attempt_evidence_digest",
        mode="before",
    )
    @classmethod
    def _validate_digest(cls, value: object) -> ArtifactDigest:
        if not isinstance(value, str):
            raise ValueError("knowledge query digest must be a string")
        return parse_artifact_digest(value)

    def canonical_json_bytes(self) -> bytes:
        """Serialize the exact HTTP request body."""
        return canonical_json_bytes(self.model_dump(mode="json"))


class KnowledgeQueryV1(KnowledgeContextV1):
    """Trusted Attempt context plus one model-authored external knowledge query."""

    query: str = Field(min_length=1, max_length=65_536)


class KnowledgeSnapshotResponseV1(BaseModel):
    """Strict, digest-verified response returned by GPU Wiki API version 1."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1] = KNOWLEDGE_SNAPSHOT_VERSION
    service_api_version: Literal[1] = GPU_WIKI_API_VERSION
    snapshot_id: str = Field(min_length=1, max_length=300)
    content_digest: ArtifactDigest
    content: JsonValue

    @field_validator("content_digest", mode="before")
    @classmethod
    def _validate_digest(cls, value: object) -> ArtifactDigest:
        if not isinstance(value, str):
            raise ValueError("content_digest must be a string")
        return parse_artifact_digest(value)

    @model_validator(mode="after")
    def _validate_content_digest(self) -> KnowledgeSnapshotResponseV1:
        actual = hashlib.sha256(canonical_json_bytes(self.content)).hexdigest()
        if self.content_digest != f"sha256:{actual}":
            raise ValueError("GPU Wiki content digest does not match its content")
        return self


class KnowledgeInteractionV1(BaseModel):
    """One external query and response frozen before the response reaches a Worker."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1] = KNOWLEDGE_INTERACTION_VERSION
    idempotency_key: str = Field(min_length=1, max_length=200)
    query: KnowledgeQueryV1
    response: KnowledgeSnapshotResponseV1

    def canonical_json_bytes(self) -> bytes:
        """Serialize the exact frozen interaction payload."""
        return canonical_json_bytes(self.model_dump(mode="json"))
