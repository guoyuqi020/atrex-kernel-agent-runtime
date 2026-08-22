"""Validated terminal report emitted by one fresh Optimizer session."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..domain.ids import ArtifactDigest, AttemptId, parse_artifact_digest, parse_attempt_id


class AttemptExperimentV2(BaseModel):
    """One decisive within-session experiment recorded before terminal handoff."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    sequence: int = Field(gt=0)
    recorded_at: str = Field(min_length=1)
    name: str = Field(min_length=1)
    hypothesis: str = Field(min_length=1)
    change: str = Field(min_length=1)
    evidence: str = Field(min_length=1)
    result: str = Field(min_length=1)
    decision: Literal["continue", "revert", "pivot"]
    candidate_artifact_digest: ArtifactDigest | None

    @field_validator(
        "recorded_at",
        "name",
        "hypothesis",
        "change",
        "evidence",
        "result",
    )
    @classmethod
    def _validate_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Attempt experiment text cannot be blank")
        return value

    @field_validator("recorded_at")
    @classmethod
    def _validate_timestamp(cls, value: str) -> str:
        try:
            datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as error:
            raise ValueError("Attempt experiment timestamp must be ISO-8601") from error
        return value

    @field_validator("candidate_artifact_digest", mode="before")
    @classmethod
    def _validate_candidate_digest(cls, value: object) -> ArtifactDigest | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError("Experiment candidate Artifact Digest must be text")
        return parse_artifact_digest(value)


class AttemptReportV3(BaseModel):
    """Versioned engineering handoff reconciled with the Gateway outcome."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[3]
    attempt_id: AttemptId
    status: Literal["candidate_ready", "pivot", "blocked"]
    hypothesis: str = Field(min_length=1)
    bottleneck: str = Field(min_length=1)
    plan: tuple[str, ...] = Field(min_length=1)
    change_summary: str = Field(min_length=1)
    profile_evidence: str = Field(min_length=1)
    evaluation_evidence: str = Field(min_length=1)
    result_interpretation: str = Field(min_length=1)
    decision: Literal["keep", "pivot", "blocked"]
    research_sources: tuple[str, ...]
    lessons: tuple[str, ...] = Field(min_length=1)
    next_directions: tuple[str, ...]
    experiments: tuple[AttemptExperimentV2, ...] = Field(min_length=1)

    @field_validator(
        "hypothesis",
        "bottleneck",
        "change_summary",
        "profile_evidence",
        "evaluation_evidence",
        "result_interpretation",
    )
    @classmethod
    def _validate_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Attempt report text fields cannot be blank")
        return value

    @field_validator("plan", "research_sources", "lessons", "next_directions")
    @classmethod
    def _validate_text_lists(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not item.strip() for item in value):
            raise ValueError("Attempt report text lists cannot contain blank entries")
        return value

    @model_validator(mode="before")
    @classmethod
    def _parse_attempt_identifier(cls, value: object) -> object:
        if isinstance(value, dict) and isinstance(value.get("attempt_id"), str):
            value = {**value, "attempt_id": parse_attempt_id(value["attempt_id"])}
        return value

    @model_validator(mode="after")
    def _validate_terminal_state(self) -> AttemptReportV3:
        expected = {
            "candidate_ready": "keep",
            "pivot": "pivot",
            "blocked": "blocked",
        }[self.status]
        if self.decision != expected:
            raise ValueError(f"{self.status} requires decision={expected}")
        if tuple(experiment.sequence for experiment in self.experiments) != tuple(
            range(1, len(self.experiments) + 1)
        ):
            raise ValueError("Attempt experiment sequence must be contiguous")
        return self

    @classmethod
    def from_file(
        cls,
        path: Path,
        *,
        expected_attempt_id: AttemptId,
        max_bytes: int,
    ) -> AttemptReportV3:
        """Read one bounded regular JSON file and verify its Attempt identity."""
        if max_bytes <= 0:
            raise ValueError("Attempt report byte limit must be positive")
        stat = path.lstat()
        if path.is_symlink() or not path.is_file():
            raise ValueError("Attempt report must be a regular file")
        if stat.st_size > max_bytes:
            raise ValueError("Attempt report exceeds byte limit")
        report = cls.model_validate_json(path.read_bytes())
        if report.attempt_id != expected_attempt_id:
            raise ValueError("Attempt report belongs to a different Attempt")
        return report
