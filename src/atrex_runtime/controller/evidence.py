"""Cumulative authoritative Evidence checkpoints between epochs."""

from __future__ import annotations

import os
import shutil
import stat
import tempfile
from pathlib import Path
from typing import Literal, Protocol, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..artifacts.local import ArtifactKind, JsonValue, LocalArtifactStore
from ..attempt_reports import RuntimeAttemptReportProjector
from ..domain.ids import (
    ArtifactDigest,
    AttemptId,
    EpochId,
    LineageId,
    parse_artifact_digest,
    parse_lineage_id,
)
from ..domain.models import Attempt, EpochStatus, KernelRevision, LineageStatus
from ..gateway.control_models import GatewayKernelTrialRecord, GatewayMeasurementRecord
from ..registry.base import Registry
from ..serialization import write_canonical_json
from ..workers.session_trace import retained_session_file
from .projection import EvidenceArtifactProjector

EVIDENCE_CHECKPOINT_VERSION: Literal[1] = 1


class HistoricalGatewayEvidenceSource(Protocol):
    """Read normalized Runtime measurements and exact experimental Kernel snapshots."""

    def list_evidence_measurements(
        self,
        attempt_ids: tuple[AttemptId, ...],
        *,
        limit: int,
    ) -> tuple[GatewayMeasurementRecord, ...]: ...

    def list_kernel_trials(
        self,
        attempt_ids: tuple[AttemptId, ...],
        *,
        limit: int,
    ) -> tuple[GatewayKernelTrialRecord, ...]: ...


class EvidenceCheckpointV1(BaseModel):
    """Identity of one flat cumulative Evidence bundle."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1] = EVIDENCE_CHECKPOINT_VERSION
    lineage_id: LineageId
    through_epoch: int = Field(ge=0)
    previous_checkpoint_digest: ArtifactDigest | None

    @field_validator("lineage_id", mode="before")
    @classmethod
    def _validate_lineage_id(cls, value: object) -> LineageId:
        if not isinstance(value, str):
            raise ValueError("lineage_id must be a string")
        return parse_lineage_id(value)

    @field_validator("previous_checkpoint_digest", mode="before")
    @classmethod
    def _validate_previous_digest(cls, value: object) -> ArtifactDigest | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError("previous_checkpoint_digest must be a string or null")
        return parse_artifact_digest(value)

    @classmethod
    def from_file(cls, path: Path) -> Self:
        """Parse checkpoint metadata from a verified Evidence artifact."""
        return cls.model_validate_json(path.read_bytes())


class LocalEvidenceAssembler:
    """Build flat cumulative Evidence bundles from authoritative Registry facts."""

    def __init__(
        self,
        registry: Registry,
        artifacts: LocalArtifactStore,
        projector: EvidenceArtifactProjector | None = None,
        measurements: HistoricalGatewayEvidenceSource | None = None,
    ) -> None:
        self._registry = registry
        self._artifacts = artifacts
        self._projector = projector
        self._measurements = measurements
        self._reports = RuntimeAttemptReportProjector(registry, artifacts)

    def create_initial(
        self,
        lineage_id: LineageId,
        source: str | Path,
        *,
        source_label: str = "trusted-bootstrap-input",
    ) -> ArtifactDigest:
        """Create a pre-Bootstrap seed checkpoint with the final two-file shape."""
        source_path = Path(source)
        try:
            source_stat = source_path.lstat()
        except FileNotFoundError as error:
            raise ValueError("initial evidence must be a real directory") from error
        if stat.S_ISLNK(source_stat.st_mode) or not stat.S_ISDIR(source_stat.st_mode):
            raise ValueError("initial evidence must be a real directory")
        staging = Path(tempfile.mkdtemp(prefix="atrex-initial-evidence-"))
        try:
            bootstrap = staging / "bootstrap"
            bootstrap.mkdir(mode=0o700)
            source_files = sorted(
                path.relative_to(source_path).as_posix()
                for path in source_path.rglob("*")
                if path.is_file() and not path.is_symlink()
            )
            write_canonical_json(
                bootstrap / "report.json",
                {
                    "schema_version": 1,
                    "status": "bootstrap_input",
                    "source": source_label,
                    "source_files": source_files,
                },
            )
            (bootstrap / "conversation.jsonl").write_bytes(b"")
            self._write_checkpoint(
                staging,
                EvidenceCheckpointV1(
                    lineage_id=lineage_id,
                    through_epoch=0,
                    previous_checkpoint_digest=None,
                ),
            )
            return self._artifacts.put_directory(staging, ArtifactKind.EVIDENCE)
        finally:
            shutil.rmtree(staging, ignore_errors=True)

    def create_bootstrap_input(
        self,
        lineage_id: LineageId,
        source: str | Path,
    ) -> ArtifactDigest:
        """Seal trusted initialization bytes for the Bootstrap control subject only."""
        source_path = Path(source)
        try:
            source_stat = source_path.lstat()
        except FileNotFoundError as error:
            raise ValueError("initial evidence must be a real directory") from error
        if stat.S_ISLNK(source_stat.st_mode) or not stat.S_ISDIR(source_stat.st_mode):
            raise ValueError("initial evidence must be a real directory")
        staging = Path(tempfile.mkdtemp(prefix="atrex-bootstrap-input-"))
        try:
            shutil.copytree(source_path, staging / "bootstrap-input", symlinks=True)
            self._write_checkpoint(
                staging,
                EvidenceCheckpointV1(
                    lineage_id=lineage_id,
                    through_epoch=0,
                    previous_checkpoint_digest=None,
                ),
            )
            return self._artifacts.put_directory(staging, ArtifactKind.EVIDENCE)
        finally:
            shutil.rmtree(staging, ignore_errors=True)

    def create_bootstrap(
        self,
        lineage_id: LineageId,
        *,
        report_digest: ArtifactDigest,
        session_trace_digest: ArtifactDigest,
    ) -> ArtifactDigest:
        """Seal one completed Bootstrap as report plus backend-neutral conversation."""
        report = self._artifacts.verify(report_digest)
        if report.kind is not ArtifactKind.ATTEMPT_REPORT:
            raise ValueError("Bootstrap report Artifact has the wrong kind")
        report_value = report.payload_path / "value.json"
        if report_value.is_symlink() or not report_value.is_file():
            raise ValueError("Bootstrap report Artifact has no regular value.json")
        session = self._artifacts.verify(session_trace_digest)
        if session.kind is not ArtifactKind.SESSION_LOG:
            raise ValueError("Bootstrap Session Artifact has the wrong kind")
        conversation = session.payload_path / "conversation.jsonl"
        if conversation.is_symlink() or not conversation.is_file():
            raise ValueError("Bootstrap Session Artifact has no regular conversation.jsonl")
        payload, _removed = retained_session_file(
            "conversation.jsonl",
            conversation.read_bytes(),
        )
        staging = Path(tempfile.mkdtemp(prefix="atrex-bootstrap-evidence-"))
        try:
            bootstrap = staging / "bootstrap"
            bootstrap.mkdir(mode=0o700)
            shutil.copyfile(report_value, bootstrap / "report.json")
            (bootstrap / "conversation.jsonl").write_bytes(payload)
            self._write_checkpoint(
                staging,
                EvidenceCheckpointV1(
                    lineage_id=lineage_id,
                    through_epoch=0,
                    previous_checkpoint_digest=None,
                ),
            )
            return self._artifacts.put_directory(staging, ArtifactKind.EVIDENCE)
        finally:
            shutil.rmtree(staging, ignore_errors=True)

    def clone_bootstrap(
        self,
        lineage_id: LineageId,
        source_checkpoint_digest: ArtifactDigest,
    ) -> ArtifactDigest:
        """Re-root another Lineage's sealed Bootstrap so an ablation arm starts identically."""
        source = self._artifacts.verify(source_checkpoint_digest)
        if source.kind is not ArtifactKind.EVIDENCE:
            raise ValueError("Bootstrap Evidence source has the wrong Artifact kind")
        report = source.payload_path / "bootstrap/report.json"
        conversation = source.payload_path / "bootstrap/conversation.jsonl"
        for path in (report, conversation):
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"Bootstrap Evidence source has no regular {path.name}")
        staging = Path(tempfile.mkdtemp(prefix="atrex-bootstrap-clone-"))
        try:
            bootstrap = staging / "bootstrap"
            bootstrap.mkdir(mode=0o700)
            shutil.copyfile(report, bootstrap / "report.json")
            shutil.copyfile(conversation, bootstrap / "conversation.jsonl")
            self._write_checkpoint(
                staging,
                EvidenceCheckpointV1(
                    lineage_id=lineage_id,
                    through_epoch=0,
                    previous_checkpoint_digest=None,
                ),
            )
            return self._artifacts.put_directory(staging, ArtifactKind.EVIDENCE)
        finally:
            shutil.rmtree(staging, ignore_errors=True)

    def assemble_next(
        self,
        lineage_id: LineageId,
    ) -> ArtifactDigest:
        """Append the completed epoch selected by the lineage handoff state."""
        lineage = self._registry.get_lineage(lineage_id)
        if lineage.status is not LineageStatus.AWAITING_EVIDENCE:
            raise ValueError(f"Lineage {lineage_id} is not awaiting evidence")
        epoch_number = lineage.next_epoch_number - 1
        epoch = self._registry.find_epoch(lineage_id, epoch_number)
        if epoch is None or epoch.status is not EpochStatus.COMPLETED:
            raise ValueError("lineage evidence handoff has no completed epoch")
        if epoch.evidence_checkpoint != lineage.evidence_checkpoint:
            raise ValueError("completed epoch input disagrees with lineage checkpoint")

        previous = self._artifacts.verify(lineage.evidence_checkpoint)
        if previous.kind is not ArtifactKind.EVIDENCE:
            raise ValueError("lineage checkpoint has the wrong artifact kind")
        actual_entries = {entry.name for entry in previous.payload_path.iterdir()}
        required_entries = {"bootstrap", "checkpoint.json"}
        if not required_entries.issubset(actual_entries) or not actual_entries.issubset(
            required_entries
            | {
                "bootstrap",
                "epochs",
                "traces",
                "lessons",
                "diffs",
                "reports",
                "measurements",
                "kernel-trials",
            }
        ):
            raise ValueError("lineage checkpoint has an unsupported file layout")
        metadata = EvidenceCheckpointV1.from_file(previous.payload_path / "checkpoint.json")
        if metadata.lineage_id != lineage_id or metadata.through_epoch != epoch_number - 1:
            raise ValueError("lineage checkpoint metadata is not the expected predecessor")

        staging = Path(tempfile.mkdtemp(prefix="atrex-epoch-evidence-"))
        try:
            previous_bootstrap = previous.payload_path / "bootstrap"
            expected_bootstrap = {"report.json", "conversation.jsonl"}
            if previous_bootstrap.is_symlink() or not previous_bootstrap.is_dir():
                raise ValueError("lineage checkpoint has no regular bootstrap directory")
            bootstrap_entries = tuple(previous_bootstrap.iterdir())
            if {entry.name for entry in bootstrap_entries} != expected_bootstrap or any(
                entry.is_symlink() or not entry.is_file() for entry in bootstrap_entries
            ):
                raise ValueError(
                    "lineage checkpoint bootstrap must contain only report.json "
                    "and conversation.jsonl"
                )
            shutil.copytree(previous_bootstrap, staging / "bootstrap")
            previous_epochs = previous.payload_path / "epochs"
            if previous_epochs.is_dir():
                shutil.copytree(previous_epochs, staging / "epochs")
            else:
                (staging / "epochs").mkdir(mode=0o700)
            os.chmod(staging / "epochs", 0o700)
            self._copy_derived_history(previous.payload_path, staging)
            epoch_value = self._epoch_value(epoch.id)
            if self._projector is not None:
                epoch_value["derived_evidence"] = self._append_derived(
                    staging,
                    epoch.id,
                    epoch_number,
                )
            write_canonical_json(staging / "epochs" / f"{epoch_number:08d}.json", epoch_value)
            self._write_checkpoint(
                staging,
                EvidenceCheckpointV1(
                    lineage_id=lineage_id,
                    through_epoch=epoch_number,
                    previous_checkpoint_digest=lineage.evidence_checkpoint,
                ),
            )
            return self._artifacts.put_directory(staging, ArtifactKind.EVIDENCE)
        finally:
            shutil.rmtree(staging, ignore_errors=True)

    def _append_derived(
        self,
        staging: Path,
        epoch_id: EpochId,
        epoch_number: int,
    ) -> dict[str, JsonValue]:
        if self._projector is None:
            raise AssertionError("derived evidence requires a projector")
        trace_root = staging / "traces" / f"{epoch_number:08d}"
        lesson_root = staging / "lessons"
        diff_root = staging / "diffs" / f"{epoch_number:08d}"
        report_root = staging / "reports" / f"{epoch_number:08d}"
        measurement_root = staging / "measurements"
        trial_root = staging / "kernel-trials"
        trace_root.mkdir(parents=True, mode=0o700)
        lesson_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        diff_root.mkdir(parents=True, mode=0o700)
        report_root.mkdir(parents=True, mode=0o700)
        measurement_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        trial_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        lessons: list[JsonValue] = []
        trace_files: list[JsonValue] = []
        diff_files: list[JsonValue] = []
        epoch = self._registry.get_epoch(epoch_id)
        for proposal in self._registry.list_epoch_challengers(epoch.id):
            from ..workers.evolution import EvolutionTraceV9

            evolution_artifact = self._artifacts.verify(proposal.evolution_trace_digest)
            if evolution_artifact.kind is not ArtifactKind.EVOLUTION:
                raise ValueError("Challenger Evolution trace has the wrong artifact kind")
            evolution = EvolutionTraceV9.model_validate_json(
                (evolution_artifact.payload_path / "value.json").read_bytes()
            )
            lessons.append(
                {
                    "kind": "evolver-authored-annotation",
                    "trusted": False,
                    "challenger_ordinal": proposal.challenger_ordinal,
                    "kernel_agent_revision_id": proposal.kernel_agent_revision_id,
                    "proposal_type": proposal.proposal_type.value,
                    "base_revision_id": proposal.base_revision_id,
                    "source_evolution_digest": proposal.evolution_trace_digest,
                    "hypothesis": evolution.output.hypothesis,
                    "expected_effect": evolution.output.expected_effect,
                    "changed_paths": list(getattr(evolution.output, "changed_paths", ())),
                    "unimplemented_capabilities": [
                        item.model_dump(mode="json")
                        for item in evolution.output.unimplemented_capabilities
                    ],
                }
            )
            if evolution.session_trace_digest is not None:
                projection = self._projector.session_projection(evolution.session_trace_digest)
                filename = f"evolver-{proposal.challenger_ordinal:04d}.json"
                relative = f"traces/{epoch_number:08d}/{filename}"
                write_canonical_json(trace_root / filename, projection)
                trace_files.append(relative)
                self._append_session_annotations(
                    lessons,
                    projection,
                    {
                        "kind": "evolver-session-annotation",
                        "challenger_ordinal": proposal.challenger_ordinal,
                        "kernel_agent_revision_id": proposal.kernel_agent_revision_id,
                        "proposal_type": proposal.proposal_type.value,
                        "base_revision_id": proposal.base_revision_id,
                        "source_session_log_digest": evolution.session_trace_digest,
                    },
                )
        for attempt in self._registry.list_attempts(epoch_id):
            if attempt.attempt_report_digest is not None:
                report = self._attempt_report_value(attempt)
                report_relative = f"reports/{epoch_number:08d}/{attempt.id}.json"
                write_canonical_json(report_root / f"{attempt.id}.json", report)
                lessons.append(
                    {
                        "kind": "agent-authored-attempt-report",
                        "trusted": False,
                        "attempt_id": attempt.id,
                        "branch": attempt.branch.value,
                        "challenger_ordinal": attempt.challenger_ordinal,
                        "trajectory_ordinal": attempt.trajectory_ordinal,
                        "iteration_ordinal": attempt.ordinal,
                        "source_attempt_report_digest": attempt.attempt_report_digest,
                        "report_path": report_relative,
                        "report": report,
                    }
                )
            for trace in self._registry.list_attempt_session_traces(attempt.id):
                projection = self._projector.session_projection(trace.artifact_digest)
                filename = f"{attempt.id}-run-{trace.run_ordinal:04d}.json"
                relative = f"traces/{epoch_number:08d}/{filename}"
                write_canonical_json(trace_root / filename, projection)
                trace_files.append(relative)
                self._append_session_annotations(
                    lessons,
                    projection,
                    {
                        "kind": "agent-authored-annotation",
                        "attempt_id": attempt.id,
                        "branch": attempt.branch.value,
                        "challenger_ordinal": attempt.challenger_ordinal,
                        "trajectory_ordinal": attempt.trajectory_ordinal,
                        "iteration_ordinal": attempt.ordinal,
                        "run_ordinal": trace.run_ordinal,
                        "source_session_log_digest": trace.artifact_digest,
                        "accepted_as_branch_best": attempt.accepted_as_branch_best,
                        "attempt_failure_reason": attempt.failure_reason,
                    },
                )
            if attempt.output_kernel_revision_id is None:
                continue
            before = self._registry.get_kernel_revision(attempt.input_kernel_revision_id)
            after = self._registry.get_kernel_revision(attempt.output_kernel_revision_id)
            diff = self._projector.kernel_diff(before.artifact_digest, after.artifact_digest)
            filename = f"{attempt.id}.json"
            relative = f"diffs/{epoch_number:08d}/{filename}"
            write_canonical_json(diff_root / filename, diff)
            diff_files.append(relative)
        lesson_relative = f"lessons/{epoch_number:08d}.json"
        write_canonical_json(
            lesson_root / f"{epoch_number:08d}.json",
            {
                "schema_version": 1,
                "epoch_id": epoch_id,
                "annotations": lessons,
            },
        )
        measurement_relative: str | None = None
        if self._measurements is not None:
            attempt_ids = tuple(attempt.id for attempt in self._registry.list_attempts(epoch_id))
            records = self._measurements.list_evidence_measurements(
                attempt_ids,
                limit=5_000,
            )
            measurement_relative = f"measurements/{epoch_number:08d}.json"
            write_canonical_json(
                measurement_root / f"{epoch_number:08d}.json",
                {
                    "schema_version": 1,
                    "epoch_id": epoch_id,
                    "measurements": [
                        self._measurement_value(record) for record in reversed(records)
                    ],
                },
            )
            trials = self._measurements.list_kernel_trials(attempt_ids, limit=5_000)
            write_canonical_json(
                trial_root / f"{epoch_number:08d}.json",
                {
                    "schema_version": 1,
                    "epoch_id": epoch_id,
                    "kernel_trials": [self._kernel_trial_value(trial) for trial in trials],
                },
            )
        return {
            "trace_projections": trace_files,
            "agent_annotations": lesson_relative,
            "kernel_diffs": diff_files,
            "measurements": measurement_relative,
        }

    @staticmethod
    def _append_session_annotations(
        lessons: list[JsonValue],
        projection: dict[str, JsonValue],
        fields: dict[str, JsonValue],
    ) -> None:
        sessions = projection["sessions"]
        if not isinstance(sessions, list):
            raise AssertionError("session projection has invalid sessions")
        for session in sessions:
            if not isinstance(session, dict):
                raise AssertionError("session projection entry is invalid")
            annotation = session.get("final_agent_annotation")
            if not isinstance(annotation, str) or not annotation.strip():
                continue
            lessons.append(
                {
                    **fields,
                    "trusted": False,
                    "session_id": session.get("session_id"),
                    "text": annotation,
                }
            )

    @staticmethod
    def _copy_derived_history(previous: Path, staging: Path) -> None:
        for name in (
            "traces",
            "lessons",
            "diffs",
            "reports",
            "measurements",
            "kernel-trials",
        ):
            source = previous / name
            if source.is_dir():
                destination = staging / name
                shutil.copytree(source, destination)
                LocalEvidenceAssembler._make_directories_writable(destination)

    @staticmethod
    def _measurement_value(record: GatewayMeasurementRecord) -> dict[str, JsonValue]:
        return {
            "measurement_id": record.id,
            "attempt_id": record.attempt_id,
            "kernel_artifact_digest": record.kernel_artifact_digest,
            "operation": record.point.kind.value,
            "source_operation": record.source_operation.value,
            "profile_level": record.point.profile_level,
            "shape_id": record.point.shape_id,
            "kernel_name": record.point.kernel_name,
            "metrics": record.point.metrics,
            "gateway_result_digest": record.gateway_result_digest,
            "created_at": record.created_at,
        }

    @staticmethod
    def _kernel_trial_value(record: GatewayKernelTrialRecord) -> dict[str, JsonValue]:
        return {
            "kernel_trial_id": record.id,
            "attempt_id": record.attempt_id,
            "recovery_generation": record.recovery_generation,
            "ordinal": record.ordinal,
            "kernel_artifact_digest": record.kernel_artifact_digest,
            "disposition": record.disposition,
            "observations": [
                {
                    "idempotency_key": item.idempotency_key,
                    "operation": item.operation.value,
                    "request_digest": item.request_digest,
                    "gateway_result_digest": item.gateway_result_digest,
                    "result_artifact_digest": item.result_artifact_digest,
                    "created_at": item.created_at,
                }
                for item in record.observations
            ],
            "annotations": [
                {
                    "sequence": item.sequence,
                    "disposition": item.disposition,
                    "experiment": item.experiment,
                    "recorded_at": item.recorded_at,
                }
                for item in record.annotations
            ],
            "created_at": record.created_at,
        }

    @staticmethod
    def _make_directories_writable(root: Path) -> None:
        os.chmod(root, 0o700)
        for path in root.rglob("*"):
            if path.is_dir():
                os.chmod(path, 0o700)

    def _epoch_value(self, epoch_id: EpochId) -> dict[str, JsonValue]:
        epoch = self._registry.get_epoch(epoch_id)
        starting_kernel = self._registry.get_kernel_revision(epoch.starting_kernel_revision_id)
        if epoch.best_kernel_revision_id is None:
            raise ValueError("completed Epoch has no best Kernel revision")
        best_kernel = self._registry.get_kernel_revision(epoch.best_kernel_revision_id)
        attempts: list[JsonValue] = []
        for attempt in self._registry.list_attempts(epoch.id):
            output: JsonValue = None
            if attempt.output_kernel_revision_id is not None:
                kernel = self._registry.get_kernel_revision(attempt.output_kernel_revision_id)
                output = {
                    "kernel_revision_id": kernel.id,
                    "artifact_digest": kernel.artifact_digest,
                    "correct": kernel.evaluation.correct,
                    "latency_us": kernel.evaluation.latency_us,
                    "gateway_result_digest": kernel.evaluation.gateway_result_digest,
                }
            attempts.append(
                {
                    "attempt_id": attempt.id,
                    "branch": attempt.branch.value,
                    "challenger_ordinal": attempt.challenger_ordinal,
                    "trajectory_ordinal": attempt.trajectory_ordinal,
                    "ordinal": attempt.ordinal,
                    "kernel_agent_revision_id": attempt.kernel_agent_revision_id,
                    "input_kernel_revision_id": attempt.input_kernel_revision_id,
                    "accepted_as_branch_best": attempt.accepted_as_branch_best,
                    "failure_reason": attempt.failure_reason,
                    "attempt_report_digest": attempt.attempt_report_digest,
                    "attempt_report_status": attempt.attempt_report_status,
                    "input_runtime_state_digest": attempt.input_runtime_state_digest,
                    "runtime_state_digest": attempt.runtime_state_digest,
                    "output": output,
                }
            )
        return {
            "schema_version": 1,
            "epoch_id": epoch.id,
            "lineage_id": epoch.lineage_id,
            "number": epoch.number,
            "active_kernel_agent_revision_id": epoch.active_kernel_agent_revision_id,
            "challenger_kernel_agent_revision_ids": list(
                epoch.challenger_kernel_agent_revision_ids
            ),
            "challenger_proposals": [
                {
                    "challenger_ordinal": item.challenger_ordinal,
                    "kernel_agent_revision_id": item.kernel_agent_revision_id,
                    "proposal_type": item.proposal_type.value,
                    "base_revision_id": item.base_revision_id,
                    "evolution_trace_digest": item.evolution_trace_digest,
                }
                for item in self._registry.list_epoch_challengers(epoch.id)
            ],
            "challenger_count": epoch.challenger_count,
            "trajectories_per_branch": epoch.trajectories_per_branch,
            "attempts_per_trajectory": epoch.attempts_per_trajectory,
            "winner_kernel_agent_revision_id": epoch.winner_kernel_agent_revision_id,
            "starting_kernel_revision_id": epoch.starting_kernel_revision_id,
            "starting_kernel": self._kernel_value(starting_kernel),
            "best_kernel_revision_id": epoch.best_kernel_revision_id,
            "best_kernel": self._kernel_value(best_kernel),
            "attempts": attempts,
        }

    @staticmethod
    def _kernel_value(kernel: KernelRevision) -> dict[str, JsonValue]:
        """Project one authoritative Kernel revision into cumulative Evidence."""
        return {
            "kernel_revision_id": kernel.id,
            "parent_kernel_revision_id": kernel.parent_id,
            "artifact_digest": kernel.artifact_digest,
            "correct": kernel.evaluation.correct,
            "latency_us": kernel.evaluation.latency_us,
            "gateway_result_digest": kernel.evaluation.gateway_result_digest,
            "produced_by_attempt_id": kernel.produced_by_attempt_id,
            "created_at": kernel.created_at,
        }

    def _attempt_report_value(self, attempt: Attempt) -> dict[str, JsonValue]:
        return self._reports.project(attempt)

    @staticmethod
    def _write_checkpoint(root: Path, checkpoint: EvidenceCheckpointV1) -> None:
        write_canonical_json(root / "checkpoint.json", checkpoint.model_dump(mode="json"))
