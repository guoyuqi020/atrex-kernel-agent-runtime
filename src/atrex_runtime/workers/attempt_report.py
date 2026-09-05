"""Validated terminal report emitted by one fresh Optimizer session."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..domain.ids import ArtifactDigest, AttemptId, parse_artifact_digest, parse_attempt_id


class AttemptExperimentSubjectV1(BaseModel):
    """One exact Kernel Trial snapshot materialized by Runtime when recorded."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kernel_artifact_digest: ArtifactDigest
    kernel_trial_id: str = Field(pattern=r"^gtrial_[0-9a-f]{32}$")
    result_artifact_digests: tuple[ArtifactDigest, ...] = Field(min_length=1, max_length=4_096)

    @field_validator("kernel_artifact_digest", mode="before")
    @classmethod
    def _validate_kernel_digest(cls, value: object) -> ArtifactDigest:
        if not isinstance(value, str):
            raise ValueError("Experiment Kernel Artifact Digest must be text")
        return parse_artifact_digest(value)

    @field_validator("result_artifact_digests", mode="before")
    @classmethod
    def _validate_result_digests(cls, value: object) -> tuple[ArtifactDigest, ...]:
        if not isinstance(value, (list, tuple)):
            raise ValueError("Experiment Result Artifact Digests must be an array")
        parsed_values: list[ArtifactDigest] = []
        for item in value:
            if not isinstance(item, str):
                raise ValueError("Experiment Result Artifact Digest must be text")
            parsed_values.append(parse_artifact_digest(item))
        parsed = tuple(parsed_values)
        if len(set(parsed)) != len(parsed):
            raise ValueError("Experiment Result Artifact Digests must be unique")
        return parsed


class AttemptExperimentV8(BaseModel):
    """One before/after experiment recorded before terminal handoff."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    experiment_id: str = Field(pattern=r"^experiment_[0-9a-f]{32}$")
    direction_id: str = Field(pattern=r"^direction_[0-9a-f]{32}$")
    sequence: int = Field(gt=0)
    recorded_at: str = Field(min_length=1)
    name: str = Field(min_length=1)
    hypothesis: str = Field(min_length=1)
    change: str = Field(min_length=1)
    before: AttemptExperimentSubjectV1 | None
    after: AttemptExperimentSubjectV1 | None
    evidence: str = Field(min_length=1)
    analysis: str = Field(min_length=1)
    action: Literal["keep_after", "restore_before", "abandon_direction", "baseline"]

    @field_validator(
        "recorded_at",
        "name",
        "hypothesis",
        "change",
        "evidence",
        "analysis",
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

    @model_validator(mode="after")
    def _validate_comparison(self) -> AttemptExperimentV8:
        if self.action == "baseline":
            if self.before is not None or self.after is None:
                raise ValueError(
                    "Experiment baseline requires before=null and complete after evidence"
                )
            return self
        if (self.before is None) != (self.after is None):
            raise ValueError("Experiment before and after must both be present or both be null")
        if self.action in {"keep_after", "restore_before"} and self.before is None:
            raise ValueError(
                "Experiment keep_after/restore_before requires before and after evidence"
            )
        return self


class AttemptDiagnosisV1(BaseModel):
    """Agent diagnosis supported by bounded profiling evidence."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    bottleneck: str = Field(min_length=1)
    evidence: str = Field(min_length=1)

    @field_validator("bottleneck", "evidence")
    @classmethod
    def _validate_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Attempt diagnosis text cannot be blank")
        return value


class AttemptApproachV1(BaseModel):
    """Planned optimization mechanism, expected effect, and known risks."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    summary: str = Field(min_length=1)
    steps: tuple[str, ...] = Field(min_length=1)
    expected_impact: str = Field(min_length=1)
    risks: tuple[str, ...]

    @field_validator("summary", "expected_impact")
    @classmethod
    def _validate_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Attempt approach text cannot be blank")
        return value

    @field_validator("steps", "risks")
    @classmethod
    def _validate_text_lists(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not item.strip() for item in value):
            raise ValueError("Attempt approach lists cannot contain blank entries")
        return value


class AttemptFinalCandidateV1(BaseModel):
    """Agent-authored description of the exact nominated working tree."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    change_summary: str = Field(min_length=1)

    @field_validator("change_summary")
    @classmethod
    def _validate_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Final Candidate summary cannot be blank")
        return value


class AttemptEvidenceSummaryV1(BaseModel):
    """Agent interpretation inputs, never the authoritative Runtime outcome."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    correctness: str = Field(min_length=1)
    performance: str = Field(min_length=1)

    @field_validator("correctness", "performance")
    @classmethod
    def _validate_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Attempt evidence summary cannot be blank")
        return value


class AttemptProfileEvidenceReferenceV1(BaseModel):
    """One exact profiling observation supporting the Agent's diagnosis."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    operation: Literal["profile"]
    kernel_artifact_digest: ArtifactDigest
    kernel_trial_id: str = Field(pattern=r"^gtrial_[0-9a-f]{32}$")
    result_artifact_digest: ArtifactDigest

    @field_validator("kernel_artifact_digest", "result_artifact_digest", mode="before")
    @classmethod
    def _validate_digest(cls, value: object) -> ArtifactDigest:
        if not isinstance(value, str):
            raise ValueError("Profile evidence Artifact Digest must be text")
        return parse_artifact_digest(value)


class AttemptProfileEvidenceV1(BaseModel):
    """Structured provenance and reasoning for an optional profiling investigation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    tool_used: str = Field(min_length=1)
    profiler: str = Field(min_length=1)
    profile_level: str = Field(min_length=1)
    bottleneck_type: str = Field(min_length=1)
    evidence_summary: str = Field(min_length=1)
    evidence_chain: str = Field(min_length=1)
    supporting_results: tuple[AttemptProfileEvidenceReferenceV1, ...] = Field(
        min_length=1,
        max_length=32,
    )

    @field_validator(
        "tool_used",
        "profiler",
        "profile_level",
        "bottleneck_type",
        "evidence_summary",
        "evidence_chain",
    )
    @classmethod
    def _validate_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Attempt profile evidence fields cannot be blank")
        return value

    @model_validator(mode="after")
    def _validate_supporting_results(self) -> AttemptProfileEvidenceV1:
        digests = tuple(item.result_artifact_digest for item in self.supporting_results)
        if len(set(digests)) != len(digests):
            raise ValueError("Profile evidence Result Artifact Digests must be unique")
        if not any(item.operation == "profile" for item in self.supporting_results):
            raise ValueError("Profile evidence requires at least one profile result")
        return self


class AttemptKnowledgeUseV1(BaseModel):
    """One external knowledge record that materially affected the work."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    record_id: str = Field(min_length=1)
    finding: str = Field(min_length=1)
    application: str = Field(min_length=1)

    @field_validator("record_id", "finding", "application")
    @classmethod
    def _validate_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Knowledge-use fields cannot be blank")
        return value


class AttemptFindingV1(BaseModel):
    """One reusable finding distilled from a positive or negative result."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    category: str = Field(min_length=1)
    observation: str = Field(min_length=1)
    root_cause: str = Field(min_length=1)
    resolution: str = Field(min_length=1)
    lesson: str = Field(min_length=1)
    supporting_experiment_ids: tuple[str, ...] = Field(min_length=1, max_length=32)

    @field_validator("category", "observation", "root_cause", "resolution", "lesson")
    @classmethod
    def _validate_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Attempt finding fields cannot be blank")
        return value

    @field_validator("supporting_experiment_ids")
    @classmethod
    def _validate_experiment_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(set(value)) != len(value):
            raise ValueError("Finding supporting Experiment IDs must be unique")
        for experiment_id in value:
            if (
                not experiment_id.startswith("experiment_")
                or len(experiment_id) != len("experiment_") + 32
                or any(character not in "0123456789abcdef" for character in experiment_id[11:])
            ):
                raise ValueError("Finding supporting Experiment ID is invalid")
        return value


class AttemptDirectionEventV1(BaseModel):
    """One append-only Direction definition or lifecycle update."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    direction_event_id: str = Field(pattern=r"^directionevent_[0-9a-f]{32}$")
    direction_id: str = Field(pattern=r"^direction_[0-9a-f]{32}$")
    recorded_at: str = Field(min_length=1)
    action: Literal["propose", "start", "complete", "abandon", "block", "defer"]
    name: str | None
    hypothesis: str | None
    rationale: str | None
    plan: tuple[str, ...]
    success_criteria: str | None
    stop_conditions: str | None
    analysis: str | None
    supporting_experiment_ids: tuple[str, ...] = Field(max_length=32)

    @field_validator("recorded_at")
    @classmethod
    def _validate_timestamp(cls, value: str) -> str:
        try:
            datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as error:
            raise ValueError("Direction timestamp must be ISO-8601") from error
        return value

    @model_validator(mode="after")
    def _validate_event(self) -> AttemptDirectionEventV1:
        definition = (
            self.name,
            self.hypothesis,
            self.rationale,
            self.success_criteria,
            self.stop_conditions,
        )
        if self.action == "propose":
            if any(value is None or not value.strip() for value in definition):
                raise ValueError("Direction proposal requires complete non-blank definition")
            if not self.plan or any(not value.strip() for value in self.plan):
                raise ValueError("Direction proposal requires a non-blank plan")
            if self.analysis is not None or self.supporting_experiment_ids:
                raise ValueError("Direction proposal cannot contain outcome evidence")
        else:
            if any(value is not None for value in definition) or self.plan:
                raise ValueError("Direction update cannot redefine its proposal")
            if self.analysis is None or not self.analysis.strip():
                raise ValueError("Direction update requires analysis")
            if self.action in {"complete", "abandon"} and not self.supporting_experiment_ids:
                raise ValueError(f"Direction {self.action} requires supporting Experiments")
        if len(set(self.supporting_experiment_ids)) != len(self.supporting_experiment_ids):
            raise ValueError("Direction supporting Experiment IDs must be unique")
        for experiment_id in self.supporting_experiment_ids:
            if (
                not experiment_id.startswith("experiment_")
                or len(experiment_id) != len("experiment_") + 32
                or any(character not in "0123456789abcdef" for character in experiment_id[11:])
            ):
                raise ValueError("Direction supporting Experiment ID is invalid")
        return self


class AttemptReportV12(BaseModel):
    """Versioned Agent engineering handoff awaiting Runtime reconciliation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[12]
    attempt_id: AttemptId
    status: Literal["candidate_ready", "pivot", "blocked"]
    hypothesis: str = Field(min_length=1)
    diagnosis: AttemptDiagnosisV1
    approach: AttemptApproachV1
    final_candidate: AttemptFinalCandidateV1 | None
    evidence_summary: AttemptEvidenceSummaryV1
    profile_evidence: AttemptProfileEvidenceV1 | None
    analysis: str = Field(min_length=1)
    knowledge_used: tuple[AttemptKnowledgeUseV1, ...]
    findings: tuple[AttemptFindingV1, ...] = Field(min_length=1)
    # Defaulted because sealed historical reports predating this field are re-parsed
    # strictly by composition/bootstrap.py.
    contributing_kernel_trial_ids: tuple[
        Annotated[str, Field(pattern=r"^gtrial_[0-9a-f]{32}$")], ...
    ] = Field(default=(), max_length=64)
    blocker: str | None
    experiments: tuple[AttemptExperimentV8, ...] = Field(min_length=1)
    direction_events: tuple[AttemptDirectionEventV1, ...] = Field(min_length=1)

    @field_validator("hypothesis", "analysis")
    @classmethod
    def _validate_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Attempt report text fields cannot be blank")
        return value

    @field_validator("blocker")
    @classmethod
    def _validate_optional_text(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("Attempt report blocker cannot be blank")
        return value

    @field_validator("contributing_kernel_trial_ids")
    @classmethod
    def _validate_contributing_kernel_trial_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(set(value)) != len(value):
            raise ValueError("Attempt report contributing Kernel Trial IDs must be unique")
        if list(value) != sorted(value):
            raise ValueError("Attempt report contributing Kernel Trial IDs must be sorted")
        return value

    @model_validator(mode="before")
    @classmethod
    def _parse_attempt_identifier(cls, value: object) -> object:
        if isinstance(value, dict) and isinstance(value.get("attempt_id"), str):
            value = {**value, "attempt_id": parse_attempt_id(value["attempt_id"])}
        return value

    @model_validator(mode="after")
    def _validate_terminal_state(self) -> AttemptReportV12:
        if self.status == "candidate_ready":
            if self.final_candidate is None:
                raise ValueError("candidate_ready requires final_candidate")
            if self.blocker is not None:
                raise ValueError("candidate_ready cannot declare a blocker")
        elif self.status == "pivot":
            if self.final_candidate is not None:
                raise ValueError("pivot cannot nominate final_candidate")
            if self.blocker is not None:
                raise ValueError("pivot cannot declare a blocker")
        else:
            if self.final_candidate is not None:
                raise ValueError("blocked cannot nominate final_candidate")
            if self.blocker is None:
                raise ValueError("blocked requires blocker")
        if tuple(experiment.sequence for experiment in self.experiments) != tuple(
            range(1, len(self.experiments) + 1)
        ):
            raise ValueError("Attempt experiment sequence must be contiguous")
        experiment_ids = {experiment.experiment_id for experiment in self.experiments}
        event_ids = [event.direction_event_id for event in self.direction_events]
        if len(set(event_ids)) != len(event_ids):
            raise ValueError("Direction Event IDs must be unique")
        direction_events: dict[str, list[AttemptDirectionEventV1]] = {}
        for event in self.direction_events:
            direction_events.setdefault(event.direction_id, []).append(event)
        advanced_direction_ids = {
            event.direction_id for event in self.direction_events if event.action == "start"
        }
        if len(advanced_direction_ids) > 3:
            raise ValueError(
                "Attempt Direction advancement limit exceeded: maximum=3; "
                f"started={len(advanced_direction_ids)}; "
                f"direction_ids={sorted(str(value) for value in advanced_direction_ids)}"
            )
        experiment_direction_ids = {experiment.direction_id for experiment in self.experiments}
        if not experiment_direction_ids.issubset(direction_events):
            raise ValueError("Experiment references a Direction absent from this Attempt")
        in_progress_direction_ids = sorted(
            str(direction_id)
            for direction_id, events in direction_events.items()
            if events[-1].action == "start"
        )
        if in_progress_direction_ids:
            raise ValueError(
                "Attempt report cannot leave any Direction in progress: "
                f"{in_progress_direction_ids}"
            )
        for finding in self.findings:
            if not set(finding.supporting_experiment_ids).issubset(experiment_ids):
                raise ValueError("Finding references an Experiment outside this Attempt report")
        return self

    @classmethod
    def from_file(
        cls,
        path: Path,
        *,
        expected_attempt_id: AttemptId,
        max_bytes: int,
    ) -> AttemptReportV12:
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
