"""Versioned Optimizer Worker-to-Wiki-Proxy protocol."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..artifacts.local import JsonValue
from ..domain.ids import ArtifactDigest, AttemptId, parse_artifact_digest, parse_attempt_id

WIKI_PROXY_PROTOCOL_VERSION: Literal[1] = 1


class WikiProxyRequestV1(BaseModel):
    """One idempotent external-knowledge query from an Optimizer Worker."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1] = WIKI_PROXY_PROTOCOL_VERSION
    attempt_id: AttemptId
    idempotency_key: str = Field(min_length=1, max_length=200)
    query: str = Field(min_length=1, max_length=65_536)

    @field_validator("attempt_id", mode="before")
    @classmethod
    def _validate_attempt_id(cls, value: object) -> AttemptId:
        if not isinstance(value, str):
            raise ValueError("attempt_id must be a string")
        return parse_attempt_id(value)


class WikiProxyResponseV1(BaseModel):
    """Frozen external knowledge returned to the Worker and its Session log."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1] = WIKI_PROXY_PROTOCOL_VERSION
    interaction_artifact_digest: ArtifactDigest
    snapshot_id: str = Field(min_length=1, max_length=300)
    content_digest: ArtifactDigest
    content: JsonValue

    @field_validator("interaction_artifact_digest", "content_digest", mode="before")
    @classmethod
    def _validate_digest(cls, value: object) -> ArtifactDigest:
        if not isinstance(value, str):
            raise ValueError("Wiki Proxy digest must be a string")
        return parse_artifact_digest(value)
