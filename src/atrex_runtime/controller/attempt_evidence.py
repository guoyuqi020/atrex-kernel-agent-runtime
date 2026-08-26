"""Immutable same-branch Evidence snapshots for fresh Optimizer sessions."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Literal, Self, cast

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..artifacts.local import ArtifactKind, JsonValue, LocalArtifactStore
from ..attempt_reports import RuntimeAttemptReportProjector
from ..domain.ids import (
    ArtifactDigest,
    AttemptId,
    EpochId,
    parse_artifact_digest,
    parse_attempt_id,
    parse_epoch_id,
)
from ..domain.models import Attempt, AttemptStatus, BranchRole, KernelRevision
from ..ports import BuildAttemptEvidenceRequest
from ..registry.base import Registry
from ..serialization import write_canonical_json
from .projection import EvidenceArtifactProjector

ATTEMPT_EVIDENCE_VERSION: Literal[2] = 2


class AttemptEvidenceMetadataV2(BaseModel):
    """Identity and visibility range of one branch-local Evidence artifact."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[2] = ATTEMPT_EVIDENCE_VERSION
    epoch_id: EpochId
    attempt_id: AttemptId
    branch: BranchRole
    challenger_ordinal: int = Field(ge=0)
    trajectory_ordinal: int = Field(gt=0)
    ordinal: int = Field(gt=0)
    epoch_evidence_checkpoint: ArtifactDigest
    previous_attempt_ids: tuple[AttemptId, ...]

    @field_validator("epoch_id", mode="before")
    @classmethod
    def _validate_epoch_id(cls, value: object) -> EpochId:
        if not isinstance(value, str):
            raise ValueError("Attempt Evidence epoch_id must be a string")
        return parse_epoch_id(value)

    @field_validator("attempt_id", "previous_attempt_ids", mode="before")
    @classmethod
    def _validate_attempt_ids(cls, value: object) -> object:
        if isinstance(value, str):
            return parse_attempt_id(value)
        if isinstance(value, (list, tuple)):
            if any(not isinstance(item, str) for item in value):
                raise ValueError("Attempt Evidence IDs must be strings")
            return tuple(parse_attempt_id(item) for item in value)
        raise ValueError("Attempt Evidence IDs are invalid")

    @field_validator("epoch_evidence_checkpoint", mode="before")
    @classmethod
    def _validate_digest(cls, value: object) -> ArtifactDigest:
        if not isinstance(value, str):
            raise ValueError("Attempt Evidence checkpoint must be a string")
        return parse_artifact_digest(value)

    @classmethod
    def from_file(cls, path: Path) -> Self:
        """Parse metadata from a verified Attempt Evidence artifact."""
        return cls.model_validate_json(path.read_bytes())


class LocalAttemptEvidenceAssembler:
    """Derive one bounded snapshot from earlier completed same-branch Attempts."""

    def __init__(
        self,
        registry: Registry,
        artifacts: LocalArtifactStore,
        projector: EvidenceArtifactProjector,
    ) -> None:
        self._registry = registry
        self._artifacts = artifacts
        self._projector = projector
        self._reports = RuntimeAttemptReportProjector(registry, artifacts)

    def assemble(self, request: BuildAttemptEvidenceRequest) -> ArtifactDigest:
        """Seal authoritative facts and bounded projections before one Attempt starts."""
        previous = self._previous_attempts(request)

        staging = Path(tempfile.mkdtemp(prefix="atrex-attempt-evidence-"))
        try:
            (staging / "attempts").mkdir(mode=0o700)
            (staging / "traces").mkdir(mode=0o700)
            (staging / "diffs").mkdir(mode=0o700)
            (staging / "reports").mkdir(mode=0o700)
            metadata = self._metadata(request, previous)
            write_canonical_json(
                staging / "context.json",
                cast(JsonValue, metadata.model_dump(mode="json")),
            )
            annotations: list[JsonValue] = []
            for attempt in previous:
                self._append_attempt(staging, attempt, annotations)
            write_canonical_json(
                staging / "lessons.json",
                {
                    "schema_version": ATTEMPT_EVIDENCE_VERSION,
                    "annotations": annotations,
                },
            )
            return self._artifacts.put_directory(
                staging,
                ArtifactKind.ATTEMPT_EVIDENCE,
            )
        finally:
            shutil.rmtree(staging, ignore_errors=True)

    def validate(
        self,
        digest: ArtifactDigest,
        request: BuildAttemptEvidenceRequest,
    ) -> None:
        """Verify a recovered Attempt still names its exact branch-local snapshot."""
        stored = self._artifacts.verify(digest)
        if stored.kind is not ArtifactKind.ATTEMPT_EVIDENCE:
            raise ValueError("persisted Attempt Evidence has the wrong artifact kind")
        expected = self._metadata(request, self._previous_attempts(request))
        actual = AttemptEvidenceMetadataV2.from_file(stored.payload_path / "context.json")
        if actual != expected:
            raise ValueError("persisted Attempt Evidence disagrees with its Attempt")

    def _previous_attempts(self, request: BuildAttemptEvidenceRequest) -> list[Attempt]:
        epoch = self._registry.get_epoch(request.epoch_id)
        if epoch.evidence_checkpoint != request.epoch_evidence_checkpoint:
            raise ValueError("Attempt Evidence disagrees with the epoch checkpoint")
        if request.challenger_ordinal > epoch.challenger_count:
            raise ValueError("Attempt Evidence Challenger exceeds the Epoch pool")
        if request.trajectory_ordinal > epoch.trajectories_per_branch:
            raise ValueError("Attempt Evidence Trajectory exceeds the Branch budget")
        if request.ordinal > epoch.attempts_per_trajectory:
            raise ValueError("Attempt Evidence ordinal exceeds the Trajectory budget")
        base = self._artifacts.verify(request.epoch_evidence_checkpoint)
        if base.kind is not ArtifactKind.EVIDENCE:
            raise ValueError("Attempt Evidence base has the wrong artifact kind")
        previous = sorted(
            (
                attempt
                for attempt in self._registry.list_attempts(epoch.id)
                if attempt.branch is request.branch
                and attempt.challenger_ordinal == request.challenger_ordinal
                and attempt.trajectory_ordinal == request.trajectory_ordinal
                and attempt.ordinal < request.ordinal
            ),
            key=lambda attempt: attempt.ordinal,
        )
        if [attempt.ordinal for attempt in previous] != list(range(1, request.ordinal)):
            raise ValueError("Attempt Evidence has an incomplete same-branch history")
        if any(attempt.status is not AttemptStatus.COMPLETED for attempt in previous):
            raise ValueError("Attempt Evidence cannot include an unfinished Attempt")
        return previous

    @staticmethod
    def _metadata(
        request: BuildAttemptEvidenceRequest,
        previous: list[Attempt],
    ) -> AttemptEvidenceMetadataV2:
        return AttemptEvidenceMetadataV2(
            epoch_id=request.epoch_id,
            attempt_id=request.attempt_id,
            branch=request.branch,
            challenger_ordinal=request.challenger_ordinal,
            trajectory_ordinal=request.trajectory_ordinal,
            ordinal=request.ordinal,
            epoch_evidence_checkpoint=request.epoch_evidence_checkpoint,
            previous_attempt_ids=tuple(attempt.id for attempt in previous),
        )

    def _append_attempt(
        self,
        staging: Path,
        attempt: Attempt,
        annotations: list[JsonValue],
    ) -> None:
        input_kernel = self._registry.get_kernel_revision(attempt.input_kernel_revision_id)
        output_kernel = (
            None
            if attempt.output_kernel_revision_id is None
            else self._registry.get_kernel_revision(attempt.output_kernel_revision_id)
        )
        trace_files: list[JsonValue] = []
        for trace in self._registry.list_attempt_session_traces(attempt.id):
            projection = self._projector.session_projection(trace.artifact_digest)
            relative = f"traces/{attempt.ordinal:08d}-run-{trace.run_ordinal:04d}.json"
            write_canonical_json(staging / relative, projection)
            trace_files.append(relative)
            for annotation in self._final_annotations(projection):
                annotations.append(
                    {
                        "trusted": False,
                        "attempt_id": str(attempt.id),
                        "run_ordinal": trace.run_ordinal,
                        "source_session_log_digest": str(trace.artifact_digest),
                        "text": annotation,
                    }
                )

        diff_file: str | None = None
        if output_kernel is not None:
            diff_file = f"diffs/{attempt.ordinal:08d}.json"
            write_canonical_json(
                staging / diff_file,
                self._projector.kernel_diff(
                    input_kernel.artifact_digest,
                    output_kernel.artifact_digest,
                ),
            )
        report_file: str | None = None
        if attempt.attempt_report_digest is not None:
            report_value = self._reports.project(attempt)
            report_file = f"reports/{attempt.ordinal:08d}.json"
            write_canonical_json(staging / report_file, report_value)
            annotations.append(
                {
                    "trusted": False,
                    "attempt_id": str(attempt.id),
                    "source_attempt_report_digest": str(attempt.attempt_report_digest),
                    "report": report_value,
                }
            )
        write_canonical_json(
            staging / "attempts" / f"{attempt.ordinal:08d}.json",
            {
                "schema_version": ATTEMPT_EVIDENCE_VERSION,
                "attempt_id": str(attempt.id),
                "branch": attempt.branch.value,
                "challenger_ordinal": attempt.challenger_ordinal,
                "trajectory_ordinal": attempt.trajectory_ordinal,
                "ordinal": attempt.ordinal,
                "kernel_agent_revision_id": str(attempt.kernel_agent_revision_id),
                "input_kernel": self._kernel_value(input_kernel),
                "output_kernel": (
                    None if output_kernel is None else self._kernel_value(output_kernel)
                ),
                "accepted_as_branch_best": attempt.accepted_as_branch_best,
                "failure_reason": attempt.failure_reason,
                "trace_projections": trace_files,
                "kernel_diff": diff_file,
                "attempt_report": report_file,
            },
        )

    @staticmethod
    def _kernel_value(kernel: KernelRevision) -> dict[str, JsonValue]:
        return {
            "revision_id": str(kernel.id),
            "artifact_digest": str(kernel.artifact_digest),
            "correct": kernel.evaluation.correct,
            "latency_us": kernel.evaluation.latency_us,
            "gateway_result_digest": str(kernel.evaluation.gateway_result_digest),
        }

    @staticmethod
    def _final_annotations(projection: dict[str, JsonValue]) -> list[str]:
        sessions = projection.get("sessions")
        if not isinstance(sessions, list):
            raise ValueError("Attempt Evidence Session projection is invalid")
        annotations: list[str] = []
        for session in sessions:
            if not isinstance(session, dict):
                raise ValueError("Attempt Evidence Session entry is invalid")
            annotation = session.get("final_agent_annotation")
            if isinstance(annotation, str) and annotation.strip():
                annotations.append(annotation)
        return annotations
