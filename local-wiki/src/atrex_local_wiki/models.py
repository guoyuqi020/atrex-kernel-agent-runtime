"""Independent version-1 wire models shared by the local Wiki endpoints."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

type JsonScalar = bool | int | float | str | None
type JsonValue = JsonScalar | list[JsonValue] | dict[str, JsonValue]
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_ID_SUFFIX = r"[0-9a-f]{32}"


def canonical_json_bytes(value: JsonValue) -> bytes:
    """Serialize JSON exactly as the Runtime digest protocol requires."""
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


class StrictModel(BaseModel):
    """Reject fields that version 1 does not define."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class KnowledgeContextV1(StrictModel):
    """Trusted Attempt context shared by external knowledge operations."""

    schema_version: Literal[1] = 1
    service_api_version: Literal[1] = 1
    campaign_id: str = Field(pattern=rf"^campaign_{_ID_SUFFIX}$")
    lineage_id: str = Field(pattern=rf"^lineage_{_ID_SUFFIX}$")
    epoch_id: str = Field(pattern=rf"^epoch_{_ID_SUFFIX}$")
    epoch_number: int = Field(gt=0)
    attempt_id: str = Field(pattern=rf"^attempt_{_ID_SUFFIX}$")
    branch: Literal["active", "challenger"]
    attempt_ordinal: int = Field(gt=0)
    kernel_agent_revision_id: str = Field(pattern=rf"^agentrev_{_ID_SUFFIX}$")
    operator: str = Field(min_length=1, max_length=500)
    dsl: Literal["cuda", "triton", "cutedsl"]
    hardware_target: str = Field(min_length=1, max_length=500)
    evaluation_contract_digest: str
    epoch_evidence_checkpoint_digest: str
    attempt_evidence_digest: str

    @field_validator(
        "evaluation_contract_digest",
        "epoch_evidence_checkpoint_digest",
        "attempt_evidence_digest",
    )
    @classmethod
    def _validate_digest(cls, value: str) -> str:
        if _DIGEST.fullmatch(value) is None:
            raise ValueError("invalid SHA-256 digest")
        return value


class KnowledgeQueryV1(KnowledgeContextV1):
    """Trusted context plus one model-authored search query."""

    query: str = Field(min_length=1, max_length=65_536)


class KnowledgeSnapshotResponseV1(StrictModel):
    """Digest-verifiable response consumed by Runtime API version 1."""

    schema_version: Literal[1] = 1
    service_api_version: Literal[1] = 1
    snapshot_id: str = Field(min_length=1, max_length=300)
    content_digest: str
    content: JsonValue

    @model_validator(mode="after")
    def _verify_digest(self) -> Self:
        expected = "sha256:" + hashlib.sha256(canonical_json_bytes(self.content)).hexdigest()
        if self.content_digest != expected:
            raise ValueError("content digest does not match content")
        return self
