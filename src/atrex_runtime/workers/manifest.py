"""Versioned manifest passed from the trusted Runtime to one Optimizer worker."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, field_validator

from ..domain.ids import (
    ArtifactDigest,
    AttemptId,
    CampaignId,
    EpochId,
    KernelAgentRevisionId,
    KernelRevisionId,
    LineageId,
    parse_artifact_digest,
    parse_attempt_id,
    parse_campaign_id,
    parse_epoch_id,
    parse_kernel_agent_revision_id,
    parse_kernel_revision_id,
    parse_lineage_id,
)
from ..domain.models import Dsl
from ..serialization import canonical_json_bytes

ATTEMPT_MANIFEST_VERSION: Literal[9] = 9
ATTEMPT_MANIFEST_RELATIVE_PATH = ".runtime/attempt.json"


@dataclass(frozen=True, slots=True)
class AttemptWorkspaceLayout:
    """Workspace-relative locations the Runtime assembles for an Optimizer worker.

    The layout is fixed in this Runtime and restated in the Agent Prompt, so the
    manifest does not carry it. Publishing it only let the worker compare one
    hardcoded table against another, which no deployment could ever fail.
    """

    input_kernel: str = "input/kernel"
    working_kernel: str = "work/kernel"
    evidence: str = "input/evidence"
    agent_problem: str = ".runtime/agent-problem.json"
    optimizer: str = "agent/optimizer"


ATTEMPT_WORKSPACE_LAYOUT = AttemptWorkspaceLayout()


class AttemptTaskContextV5(BaseModel):
    """Trusted Campaign and scheduling context rendered into the Agent prompt."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    campaign_id: CampaignId
    lineage_id: LineageId
    epoch_id: EpochId
    epoch_number: int
    attempt_ordinal: int
    operator: str
    hardware_target: str
    evaluation_contract_digest: ArtifactDigest
    agent_problem_digest: ArtifactDigest

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

    @field_validator("evaluation_contract_digest", mode="before")
    @classmethod
    def _validate_contract_digest(cls, value: object) -> ArtifactDigest:
        if not isinstance(value, str):
            raise ValueError("evaluation_contract_digest must be a string")
        return parse_artifact_digest(value)

    @field_validator("agent_problem_digest", mode="before")
    @classmethod
    def _validate_problem_digest(cls, value: object) -> ArtifactDigest:
        if not isinstance(value, str):
            raise ValueError("agent_problem_digest must be a string")
        return parse_artifact_digest(value)

    @field_validator("epoch_number", "attempt_ordinal")
    @classmethod
    def _validate_positive_integer(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("Attempt task ordinals must be positive")
        return value

    @field_validator("operator", "hardware_target")
    @classmethod
    def _validate_nonempty_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Attempt task context text cannot be blank")
        return value


class AttemptInputManifestV9(BaseModel):
    """Immutable inputs for exactly one fresh Optimizer session.

    Evolver implementation details are deliberately absent: this protocol does
    not let the Optimizer inspect or modify the separate Evolution worker.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[9] = ATTEMPT_MANIFEST_VERSION
    attempt_id: AttemptId
    kernel_agent_revision_id: KernelAgentRevisionId
    input_kernel_revision_id: KernelRevisionId
    input_kernel_digest: ArtifactDigest
    epoch_evidence_checkpoint: ArtifactDigest
    attempt_evidence_digest: ArtifactDigest
    optimizer_digest: ArtifactDigest
    dsl: Dsl
    context: AttemptTaskContextV5

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

    @field_validator("input_kernel_revision_id", mode="before")
    @classmethod
    def _validate_kernel_revision_id(cls, value: object) -> KernelRevisionId:
        if not isinstance(value, str):
            raise ValueError("input_kernel_revision_id must be a string")
        return parse_kernel_revision_id(value)

    @field_validator(
        "input_kernel_digest",
        "epoch_evidence_checkpoint",
        "attempt_evidence_digest",
        "optimizer_digest",
        mode="before",
    )
    @classmethod
    def _validate_digest(cls, value: object) -> ArtifactDigest:
        if not isinstance(value, str):
            raise ValueError("Attempt manifest digest must be a string")
        return parse_artifact_digest(value)

    @classmethod
    def from_json_bytes(cls, payload: bytes) -> Self:
        """Parse a manifest at the untrusted process/file boundary."""
        return cls.model_validate_json(payload)

    def canonical_json_bytes(self) -> bytes:
        """Serialize stable UTF-8 JSON for the worker workspace."""
        return canonical_json_bytes(self.model_dump(mode="json", exclude_none=False))
