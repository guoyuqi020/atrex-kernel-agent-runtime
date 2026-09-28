"""Ablation control arms, each isolated in its own Campaign."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .domain.ids import (
    ArtifactDigest,
    CampaignId,
    LineageId,
    parse_artifact_digest,
    parse_campaign_id,
    parse_lineage_id,
)
from .domain.models import Campaign, CampaignStatus
from .kernel_agents import KernelAgentBundleWorkflowV1
from .lineage_seed import (
    LineageBaselineSeedV1,
    LineageSeeder,
    LineageSeedModelsV1,
    LineageSeedResult,
    LineageSeedSpecV1,
)
from .registry.base import Registry

ABLATION_ARM_SPEC_VERSION: Literal[1] = 1


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


class AblationArmSpecV1(BaseModel):
    """One control arm cloned from an evolution Lineage's frozen Bootstrap baseline."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1] = ABLATION_ARM_SPEC_VERSION
    creation_key: str = Field(min_length=1, max_length=200)
    source_lineage_id: LineageId
    agent_artifact_digest: ArtifactDigest | None = None
    optimizer_attempt_budget: int = Field(gt=0)
    max_challengers: int = Field(default=0, ge=0)
    workflow_command: str | None = None
    optimizer_model: str | None = Field(default=None, min_length=1, max_length=200)
    evolver_observer_lineage_id: LineageId | None = None
    trajectory_visibility: Literal["isolated", "broadcast"] = "isolated"
    tool_modules: tuple[Literal["directions", "experiments"], ...] | None = None

    @field_validator("creation_key")
    @classmethod
    def _validate_creation_key(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized or "\x00" in normalized:
            raise ValueError("ablation arm creation_key is invalid")
        return normalized

    @field_validator("tool_modules")
    @classmethod
    def _validate_tool_modules(
        cls, value: tuple[Literal["directions", "experiments"], ...] | None
    ) -> tuple[Literal["directions", "experiments"], ...] | None:
        if value is not None and len(value) != len(set(value)):
            raise ValueError("ablation arm tool modules cannot repeat")
        return value

    @field_validator("source_lineage_id", mode="before")
    @classmethod
    def _validate_source_lineage_id(cls, value: object) -> LineageId:
        if not isinstance(value, str):
            raise ValueError("source Lineage ID must be a string")
        return parse_lineage_id(value)

    @field_validator("agent_artifact_digest", mode="before")
    @classmethod
    def _validate_agent_artifact_digest(cls, value: object) -> ArtifactDigest | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError("replacement Agent Artifact digest must be a string")
        return parse_artifact_digest(value)

    @field_validator("evolver_observer_lineage_id", mode="before")
    @classmethod
    def _validate_observer_lineage_id(cls, value: object) -> LineageId | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError("Evolver observer Lineage ID must be a string or null")
        return parse_lineage_id(value)

    @field_validator("optimizer_model")
    @classmethod
    def _validate_optimizer_model(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized or "\x00" in normalized:
            raise ValueError("ablation arm Optimizer model is invalid")
        return normalized

    @field_validator("workflow_command")
    @classmethod
    def _validate_workflow_command(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return KernelAgentBundleWorkflowV1(command=value).command

    @classmethod
    def from_file(cls, path: str | Path) -> Self:
        """Parse one strict ablation arm spec."""
        try:
            value = json.loads(Path(path).resolve().read_bytes())
        except json.JSONDecodeError as error:
            raise ValueError("ablation arm spec is not valid JSON") from error
        return cls.model_validate(value)


def parse_ablation_arm_spec_json(payload: bytes) -> AblationArmSpecV1:
    """Parse an HTTP ablation arm request."""
    try:
        value = json.loads(payload)
    except json.JSONDecodeError as error:
        raise ValueError("ablation arm request is not valid JSON") from error
    return AblationArmSpecV1.model_validate(value)


@dataclass(frozen=True, slots=True)
class AblationArmResult:
    """Identities and shape of one control arm, plus the evolution arm it mirrors."""

    campaign_id: CampaignId
    source_campaign_id: CampaignId
    source_lineage_id: LineageId
    optimizer_attempt_budget: int
    max_challengers: int
    workflow_command: str | None
    lineage: LineageSeedResult


class AblationArmSeeder:
    """Publish an isolated Campaign with its own optimization/evolution schedule."""

    def __init__(
        self,
        registry: Registry,
        seeder: LineageSeeder,
        *,
        clock: Callable[[], str] = _utc_now,
    ) -> None:
        self._registry = registry
        self._seeder = seeder
        self._clock = clock

    async def seed_arm(self, spec: AblationArmSpecV1) -> AblationArmResult:
        """Create or idempotently recover one control arm."""
        source_lineage = self._registry.get_lineage(spec.source_lineage_id)
        if source_lineage.bootstrap_source_lineage_id is not None:
            raise ValueError("an ablation arm cannot be cloned from another ablation arm")
        source_campaign = self._registry.get_campaign(source_lineage.campaign_id)
        campaign_id = parse_campaign_id(
            self._derived_id(
                "campaign",
                f"{source_campaign.id}:{spec.creation_key}",
            )
        )
        # The arm is only comparable if it is measured under the identical contract, so its
        # Campaign copies the evolution Campaign's identity instead of bootstrapping its own.
        self._ensure_campaign(
            Campaign(
                id=campaign_id,
                operator=source_campaign.operator,
                hardware_target=source_campaign.hardware_target,
                evaluation_contract_digest=source_campaign.evaluation_contract_digest,
                agent_problem_digest=source_campaign.agent_problem_digest,
                created_at=self._clock(),
                status=CampaignStatus.ACTIVE,
                problem_generalization_model=source_campaign.problem_generalization_model,
                evolver_commit=source_campaign.evolver_commit,
            )
        )
        lineage = await self._seeder.seed_lineage(
            campaign_id,
            LineageSeedSpecV1(
                creation_key=spec.creation_key,
                dsl=source_lineage.dsl,
                seed=LineageBaselineSeedV1(lineage_id=source_lineage.id),
                agent_artifact_digest=spec.agent_artifact_digest,
                models=LineageSeedModelsV1(
                    optimizer=spec.optimizer_model or source_lineage.optimizer_model,
                    evolver=source_lineage.evolver_model,
                ),
                max_challengers=spec.max_challengers,
                optimizer_attempt_budget=spec.optimizer_attempt_budget,
                workflow_command=spec.workflow_command,
                evolver_observer_lineage_id=spec.evolver_observer_lineage_id,
                trajectory_visibility=spec.trajectory_visibility,
                tool_modules=spec.tool_modules,
            ),
        )
        return AblationArmResult(
            campaign_id=campaign_id,
            source_campaign_id=source_campaign.id,
            source_lineage_id=source_lineage.id,
            optimizer_attempt_budget=spec.optimizer_attempt_budget,
            max_challengers=spec.max_challengers,
            workflow_command=spec.workflow_command,
            lineage=lineage,
        )

    def _ensure_campaign(self, expected: Campaign) -> None:
        try:
            existing = self._registry.get_campaign(expected.id)
        except KeyError:
            self._registry.insert_campaign(expected)
            return
        if (
            existing.operator,
            existing.hardware_target,
            existing.evaluation_contract_digest,
            existing.agent_problem_digest,
        ) != (
            expected.operator,
            expected.hardware_target,
            expected.evaluation_contract_digest,
            expected.agent_problem_digest,
        ):
            raise ValueError("ablation arm creation_key resolved to a different Campaign")

    @staticmethod
    def _derived_id(prefix: str, key: str) -> str:
        suffix = hashlib.sha256(f"atrex-ablation-arm:{prefix}:{key}".encode()).hexdigest()[:32]
        return f"{prefix}_{suffix}"


__all__ = [
    "ABLATION_ARM_SPEC_VERSION",
    "AblationArmResult",
    "AblationArmSeeder",
    "AblationArmSpecV1",
    "parse_ablation_arm_spec_json",
]
