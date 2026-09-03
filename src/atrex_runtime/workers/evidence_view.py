"""Role-scoped, read-only Agent views over independently versioned Evidence artifacts."""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
from importlib.resources import files
from pathlib import Path
from typing import Literal, Self, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..artifacts.local import ArtifactKind, JsonValue, LocalArtifactStore
from ..domain.ids import (
    ArtifactDigest,
    KernelAgentRevisionId,
    parse_artifact_digest,
    parse_kernel_revision_id,
)
from ..domain.models import BranchRole
from ..filesystem import make_tree_read_only
from ..gateway.result_metrics import gateway_result_projection
from ..serialization import canonical_json_bytes, write_canonical_json
from .session_trace import enforce_session_trace_retention, retained_session_file

EVIDENCE_VIEW_VERSION: Literal[1] = 1
_AGENT_VERSION = re.compile(r"agent-v[0-9]+")
EVIDENCE_MANIFEST_RELATIVE_PATH = Path(".runtime/evidence-manifest.json")
EVIDENCE_PROMPT_RELATIVE_PATH = Path(".runtime/evidence-instructions.md")


def _evidence_template(name: str) -> str:
    """Load one Runtime-owned Prompt Fragment from the installed Python package."""
    return (
        files("atrex_runtime").joinpath("templates", "evidence", name).read_text(encoding="utf-8")
    )


OPTIMIZER_EVIDENCE_PROMPT_TEXT = _evidence_template("optimizer.md")
EVOLVER_EVIDENCE_PROMPT_TEXT = _evidence_template("evolver.md")
# Compatibility names denote the Optimizer fragment used by existing API consumers.
EVIDENCE_PROMPT_TEXT = OPTIMIZER_EVIDENCE_PROMPT_TEXT
EVIDENCE_PROMPT_SHA256 = hashlib.sha256(EVIDENCE_PROMPT_TEXT.encode()).hexdigest()


def _role_prompt(role: Literal["optimizer", "evolver"]) -> str:
    return OPTIMIZER_EVIDENCE_PROMPT_TEXT if role == "optimizer" else EVOLVER_EVIDENCE_PROMPT_TEXT


def _role_prompt_sha256(role: Literal["optimizer", "evolver"]) -> str:
    return hashlib.sha256(_role_prompt(role).encode()).hexdigest()


class CurrentEpochEvidenceViewV1(BaseModel):
    """Optional in-progress Epoch visible in addition to completed history."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    number: int = Field(gt=0)
    status: Literal["in_progress"] = "in_progress"
    snapshot_digest: ArtifactDigest
    trigger: None = None


class EvidenceVisibilityV1(BaseModel):
    """Explicit role-scoped visibility over completed and in-progress history."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    completed_epochs: Literal["all_completed_branches"] = "all_completed_branches"
    current_attempts_before: int | None = Field(default=None, gt=0)
    current_trajectory_ordinal: int | None = Field(default=None, gt=0)


class EvidenceViewManifestV1(BaseModel):
    """Trusted description of one assembled Agent-visible Evidence tree."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1] = EVIDENCE_VIEW_VERSION
    role: Literal["optimizer", "evolver"]
    lineage_checkpoint: ArtifactDigest
    prompt_fragment_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    through_completed_epoch: int = Field(ge=0)
    current_epoch: CurrentEpochEvidenceViewV1 | None
    visibility: EvidenceVisibilityV1

    @model_validator(mode="after")
    def _validate_role_scope(self) -> Self:
        before = self.visibility.current_attempts_before
        trajectory = self.visibility.current_trajectory_ordinal
        if self.current_epoch is None:
            if before is not None or trajectory is not None:
                raise ValueError("completed-only Evidence view cannot expose current Attempts")
            return self
        if self.current_epoch.number != self.through_completed_epoch + 1:
            raise ValueError("current Epoch must immediately follow completed history")
        if self.role == "evolver":
            raise ValueError("Evolver Evidence v1 cannot expose an in-progress Epoch")
        if before is None or trajectory is None:
            raise ValueError(
                "Optimizer Evidence view requires a bounded current Trajectory and Attempts"
            )
        return self

    def canonical_json_bytes(self) -> bytes:
        return canonical_json_bytes(self.model_dump(mode="json"))

    @classmethod
    def from_file(cls, path: Path) -> Self:
        return cls.model_validate_json(path.read_bytes())


def assemble_evolver_evidence_view(
    destination: Path,
    *,
    control_root: Path,
    lineage_payload: Path,
    lineage_checkpoint: ArtifactDigest,
    artifacts: LocalArtifactStore,
    agent_versions: dict[str, str] | None = None,
    pool_versions: frozenset[str] = frozenset(),
) -> EvidenceViewManifestV1:
    """Expose completed Lineage history and per-Agent runtime state to an Evolver."""
    through_epoch = _lineage_through_epoch(lineage_payload)
    manifest = EvidenceViewManifestV1(
        role="evolver",
        lineage_checkpoint=lineage_checkpoint,
        prompt_fragment_sha256=_role_prompt_sha256("evolver"),
        through_completed_epoch=through_epoch,
        current_epoch=None,
        visibility=EvidenceVisibilityV1(completed_epochs="all_completed_branches"),
    )
    _assemble_base(
        destination,
        control_root,
        lineage_payload,
        manifest,
        artifacts,
        write_prompt=False,
    )
    versions = agent_versions or {}
    if not pool_versions <= versions.keys():
        raise ValueError("the completed Epoch Branch pool is outside the visible Agent versions")
    for version, revision_id in sorted(versions.items()):
        if _AGENT_VERSION.fullmatch(version) is None:
            raise ValueError(f"invalid Evolver Agent version: {version}")
        entry = destination / version
        entry.mkdir(mode=0o700)
        write_canonical_json(
            entry / "optimization-summary.json",
            evolver_agent_optimization_summary(
                destination, revision_id, artifacts, version=version
            ),
        )
        if version not in pool_versions:
            continue
        _materialize_evolver_agent_sessions(destination, revision_id, entry / "sessions")
        _materialize_evolver_agent_reports(destination, revision_id, entry / "reports")
    shutil.rmtree(destination / "bootstrap")
    shutil.rmtree(destination / "epochs")
    make_tree_read_only(destination)
    return manifest


def evolver_agent_optimization_summary(
    evidence_root: Path,
    revision_id: str,
    artifacts: LocalArtifactStore,
    *,
    version: str,
) -> dict[str, JsonValue]:
    """Summarize the latest competition and career record of one Agent revision."""
    records = _evolver_agent_attempt_records(evidence_root, revision_id)
    latest_epoch: JsonValue = None
    if records:
        epoch_number, _label, latest_branch, _record_attempts = records[-1]
        shared_revision = sum(record[0] == epoch_number for record in records) > 1
        attempts = [
            attempt
            for record_epoch, _label, _branch, record_attempts in records
            if record_epoch == epoch_number
            for attempt in record_attempts
        ]
        outputs = [item.get("output") for item in attempts]
        correct_outputs = [
            item for item in outputs if isinstance(item, dict) and item.get("correct") is True
        ]
        incorrect_count = sum(
            isinstance(item, dict) and item.get("correct") is False for item in outputs
        )
        best_output = min(correct_outputs, key=_candidate_latency, default=None)
        raw_ordinal = latest_branch.get("challenger_ordinal")
        latest_epoch = {
            "epoch_number": epoch_number,
            "branch": "active_and_replica" if shared_revision else latest_branch.get("branch"),
            "challenger_ordinal": (
                None
                if shared_revision
                or not isinstance(raw_ordinal, int)
                or isinstance(raw_ordinal, bool)
                or raw_ordinal <= 0
                else raw_ordinal
            ),
            "outcome": "won" if latest_branch.get("selected") is True else "lost",
            "selection_reason": _evolver_epoch_selection_reason(evidence_root, epoch_number),
            "attempt_count": len(attempts),
            "correct_attempt_count": len(correct_outputs),
            "incorrect_attempt_count": incorrect_count,
            "no_candidate_attempt_count": sum(item is None for item in outputs),
            "best_kernel": (
                None if best_output is None else _evolver_best_kernel(best_output, artifacts)
            ),
        }
    participated_epochs = {epoch for epoch, _label, _branch, _attempts in records}
    won_epochs = {
        epoch for epoch, _label, branch, _attempts in records if branch.get("selected") is True
    }
    return {
        "kernel_agent_revision_id": revision_id,
        "version": version,
        "source_path": f"input/agents/{version}/source",
        "runtime_state_path": f"input/agents/{version}/runtime-state",
        "latest_epoch": latest_epoch,
        "career": {
            "epoch_participation_count": len(participated_epochs),
            "win_count": len(won_epochs),
            "loss_count": len(participated_epochs - won_epochs),
        },
    }


def _evolver_epoch_selection_reason(evidence_root: Path, epoch_number: int) -> JsonValue:
    """Read which rule resolved one completed Epoch's Agent comparison."""
    summary = _json_object(
        evidence_root / "epochs" / f"{epoch_number:08d}" / "summary.json",
        f"Evolver Epoch {epoch_number} summary",
    )
    reason = summary.get("selection_reason")
    if reason is None:
        return None
    if not isinstance(reason, str):
        raise ValueError("completed Epoch selection reason is invalid")
    return reason


def _candidate_latency(value: dict[str, JsonValue]) -> float:
    raw = value.get("latency_us")
    if (
        isinstance(raw, (int, float))
        and not isinstance(raw, bool)
        and math.isfinite(float(raw))
        and raw > 0
    ):
        return float(raw)
    return math.inf


def _evolver_best_kernel(
    output: dict[str, JsonValue],
    artifacts: LocalArtifactStore,
) -> dict[str, JsonValue]:
    raw_result_digest = output.get("gateway_result_digest")
    correct = output.get("correct")
    raw_latency = output.get("latency_us")
    if (
        not isinstance(raw_result_digest, str)
        or correct is not True
        or not isinstance(raw_latency, (int, float))
        or isinstance(raw_latency, bool)
    ):
        raise ValueError("Evolver best Kernel evaluation is invalid")
    gateway_result = gateway_result_projection(
        artifacts,
        parse_artifact_digest(raw_result_digest),
        correct=True,
        latency_us=float(raw_latency),
    )
    gateway_result.pop("operation", None)
    return {
        "gateway_result": gateway_result,
    }


def _materialize_evolver_agent_sessions(
    evidence_root: Path,
    revision_id: str,
    destination: Path,
) -> None:
    """Copy the latest completed Epoch's authoritative Attempt conversations."""
    destination.mkdir(mode=0o700)
    records = _evolver_agent_attempt_records(evidence_root, revision_id)
    if not records:
        return
    latest_epoch = records[-1][0]
    records = [record for record in records if record[0] == latest_epoch]
    stride = _shared_agent_trajectory_stride(records)
    for branch_index, (_epoch, _branch_label, _branch, attempts) in enumerate(records):
        for attempt in attempts:
            attempt_path = attempt.pop("_evidence_path", None)
            if not isinstance(attempt_path, str):
                continue
            attempt_root = evidence_root / attempt_path
            trajectory = attempt.get("trajectory_ordinal")
            ordinal = attempt.get("ordinal")
            if not isinstance(trajectory, int) or not isinstance(ordinal, int):
                raise ValueError("Evolver Attempt session coordinates are invalid")
            trajectory += branch_index * stride
            conversations = sorted(attempt_root.rglob("conversation.jsonl"))
            if not conversations:
                continue
            conversation = conversations[-1]
            if not conversation.is_file() or conversation.is_symlink():
                raise ValueError("Evolver Attempt conversation must be a regular file")
            trajectory_root = destination / f"trajectory-{trajectory:08d}"
            trajectory_root.mkdir(mode=0o700, exist_ok=True)
            target = trajectory_root / f"attempt-{ordinal:08d}.conversation.jsonl"
            if target.exists() or target.is_symlink():
                raise FileExistsError(target)
            shutil.copyfile(conversation, target)


def _materialize_evolver_agent_reports(
    evidence_root: Path,
    revision_id: str,
    destination: Path,
) -> None:
    """Copy the latest completed Epoch's Attempt reports for one Agent revision."""
    destination.mkdir(mode=0o700)
    records = _evolver_agent_attempt_records(evidence_root, revision_id)
    if not records:
        return
    latest_epoch = records[-1][0]
    records = [record for record in records if record[0] == latest_epoch]
    stride = _shared_agent_trajectory_stride(records)
    for branch_index, (_epoch, _branch_label, _branch, attempts) in enumerate(records):
        for attempt in attempts:
            attempt_path = attempt.get("_evidence_path")
            if not isinstance(attempt_path, str):
                continue
            report = evidence_root / attempt_path / "report.json"
            if not report.exists():
                continue
            trajectory = attempt.get("trajectory_ordinal")
            ordinal = attempt.get("ordinal")
            if not isinstance(trajectory, int) or not isinstance(ordinal, int):
                raise ValueError("Evolver Attempt report coordinates are invalid")
            trajectory += branch_index * stride
            if report.is_symlink() or not report.is_file():
                raise ValueError("Evolver Attempt report must be a regular file")
            trajectory_root = destination / f"trajectory-{trajectory:08d}"
            trajectory_root.mkdir(mode=0o700, exist_ok=True)
            target = trajectory_root / f"attempt-{ordinal:08d}.report.json"
            if target.exists() or target.is_symlink():
                raise FileExistsError(target)
            shutil.copyfile(report, target)


def _shared_agent_trajectory_stride(
    records: list[tuple[int, str, dict[str, JsonValue], list[dict[str, JsonValue]]]],
) -> int:
    """Give same-Agent Branches disjoint slots, matching their reusable State snapshots."""
    return max(
        (
            cast(int, attempt["trajectory_ordinal"])
            for _epoch, _label, _branch, attempts in records
            for attempt in attempts
            if isinstance(attempt.get("trajectory_ordinal"), int)
        ),
        default=1,
    )


def _evolver_agent_attempt_records(
    evidence_root: Path,
    revision_id: str,
) -> list[tuple[int, str, dict[str, JsonValue], list[dict[str, JsonValue]]]]:
    epochs_root = evidence_root / "epochs"
    if epochs_root.is_symlink() or not epochs_root.is_dir():
        raise ValueError("Evolver Evidence epochs are unavailable")
    records: list[tuple[int, str, dict[str, JsonValue], list[dict[str, JsonValue]]]] = []
    for epoch_root in sorted(epochs_root.iterdir()):
        if not epoch_root.is_dir() or not epoch_root.name.isdigit():
            continue
        summary = _json_object(epoch_root / "summary.json", "Evolver Epoch summary")
        raw_branches = summary.get("branches")
        if not isinstance(raw_branches, list):
            raise ValueError("Evolver Epoch summary has no branch catalog")
        for raw_branch in raw_branches:
            if (
                not isinstance(raw_branch, dict)
                or raw_branch.get("kernel_agent_revision_id") != revision_id
            ):
                continue
            branch = raw_branch
            role = branch.get("branch")
            ordinal = branch.get("challenger_ordinal")
            if role == BranchRole.ACTIVE.value:
                branch_label = BranchRole.ACTIVE.value
            elif role == BranchRole.CHALLENGER.value and isinstance(ordinal, int):
                branch_label = f"challenger-{ordinal:04d}"
            else:
                raise ValueError("Evolver Agent branch identity is invalid")
            branch_root = epoch_root / "branches" / branch_label / "trajectories"
            attempts: list[dict[str, JsonValue]] = []
            if branch_root.is_dir() and not branch_root.is_symlink():
                for attempt_path in sorted(branch_root.glob("*/attempts/*/summary.json")):
                    attempt = _json_object(attempt_path, "Evolver Attempt summary")
                    attempt["_evidence_path"] = str(attempt_path.parent.relative_to(evidence_root))
                    attempts.append(attempt)
            records.append((int(epoch_root.name), branch_label, branch, attempts))
    return records


def assemble_optimizer_evidence_view(
    destination: Path,
    *,
    control_root: Path,
    lineage_payload: Path,
    lineage_checkpoint: ArtifactDigest,
    attempt_payload: Path,
    attempt_snapshot: ArtifactDigest,
    current_epoch_number: int,
    branch: BranchRole,
    challenger_ordinal: int,
    trajectory_ordinal: int,
    selected_revision: KernelAgentRevisionId,
    attempt_ordinal: int,
    artifacts: LocalArtifactStore,
) -> EvidenceViewManifestV1:
    """Expose completed history plus earlier same-branch Attempts in one tree."""
    through_epoch = _lineage_through_epoch(lineage_payload)
    manifest = EvidenceViewManifestV1(
        role="optimizer",
        lineage_checkpoint=lineage_checkpoint,
        prompt_fragment_sha256=_role_prompt_sha256("optimizer"),
        through_completed_epoch=through_epoch,
        current_epoch=CurrentEpochEvidenceViewV1(
            number=current_epoch_number,
            snapshot_digest=attempt_snapshot,
        ),
        visibility=EvidenceVisibilityV1(
            current_attempts_before=attempt_ordinal,
            current_trajectory_ordinal=trajectory_ordinal,
        ),
    )
    _assemble_base(destination, control_root, lineage_payload, manifest, artifacts)
    _append_current_lineage_attempts(
        destination,
        attempt_payload,
        lineage_checkpoint=lineage_checkpoint,
        current_epoch_number=current_epoch_number,
        branch=branch,
        challenger_ordinal=challenger_ordinal,
        trajectory_ordinal=trajectory_ordinal,
        selected_revision=selected_revision,
        attempt_ordinal=attempt_ordinal,
        artifacts=artifacts,
    )
    make_tree_read_only(destination)
    return manifest


def _assemble_base(
    destination: Path,
    control_root: Path,
    lineage_payload: Path,
    manifest: EvidenceViewManifestV1,
    artifacts: LocalArtifactStore,
    *,
    write_prompt: bool = True,
) -> None:
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    destination.mkdir(parents=True, mode=0o700)
    bootstrap = destination / "bootstrap"
    bootstrap.mkdir(mode=0o700)
    source_bootstrap = lineage_payload / "bootstrap"
    if not source_bootstrap.is_dir() or source_bootstrap.is_symlink():
        raise ValueError("Lineage Evidence has no regular bootstrap directory")
    expected = {"report.json", "conversation.jsonl"}
    actual = {path.name for path in source_bootstrap.iterdir()}
    if actual != expected or any(
        path.is_symlink() or not path.is_file() for path in source_bootstrap.iterdir()
    ):
        raise ValueError(
            "Lineage bootstrap Evidence must contain only report.json and conversation.jsonl"
        )
    _copy_directory_contents(source_bootstrap, bootstrap)
    epochs = destination / "epochs"
    epochs.mkdir(mode=0o700)
    for number in range(1, manifest.through_completed_epoch + 1):
        if manifest.role == "evolver":
            _append_evolver_completed_epoch(lineage_payload, epochs, number, artifacts)
        else:
            _append_optimizer_completed_epoch(lineage_payload, epochs, number, artifacts)
    control_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    manifest_path = control_root / EVIDENCE_MANIFEST_RELATIVE_PATH.name
    if manifest_path.exists() or manifest_path.is_symlink():
        raise FileExistsError(manifest_path)
    prompt_path = control_root / EVIDENCE_PROMPT_RELATIVE_PATH.name
    if write_prompt:
        if prompt_path.exists() or prompt_path.is_symlink():
            raise FileExistsError(prompt_path)
        prompt_path.write_text(
            _role_prompt(manifest.role),
            encoding="utf-8",
        )
        prompt_path.chmod(0o400)
    manifest_path.write_bytes(manifest.canonical_json_bytes())
    manifest_path.chmod(0o400)


def _append_optimizer_completed_epoch(
    lineage: Path,
    epochs: Path,
    number: int,
    artifacts: LocalArtifactStore,
) -> None:
    """Project every branch's Attempt reports and latest conversation for one Epoch."""
    label = f"{number:08d}"
    source_summary = lineage / "epochs" / f"{label}.json"
    summary = _json_object(source_summary, f"completed Epoch {number} summary")
    raw_attempts = summary.get("attempts")
    active_revision = summary.get("active_kernel_agent_revision_id")
    raw_challenger_revisions = summary.get("challenger_kernel_agent_revision_ids")
    winner_revision = summary.get("winner_kernel_agent_revision_id")
    if (
        summary.get("number") != number
        or not isinstance(raw_attempts, list)
        or not isinstance(active_revision, str)
        or not isinstance(raw_challenger_revisions, list)
        or not all(isinstance(item, str) for item in raw_challenger_revisions)
        or not isinstance(winner_revision, str)
        or winner_revision not in {active_revision, *raw_challenger_revisions}
    ):
        raise ValueError(f"completed Epoch {number} summary is inconsistent")
    challenger_revisions = cast(list[str], raw_challenger_revisions)
    validated_attempts = _validated_completed_attempts(
        raw_attempts,
        number,
        active_revision=active_revision,
        challenger_revisions=challenger_revisions,
    )
    destination = epochs / label
    destination.mkdir(mode=0o700)
    branches: list[JsonValue] = [
        {
            "branch": BranchRole.ACTIVE.value,
            "challenger_ordinal": 0,
            "selected": winner_revision == active_revision,
        }
    ]
    branches.extend(
        {
            "branch": BranchRole.CHALLENGER.value,
            "challenger_ordinal": ordinal,
            "selected": winner_revision == revision,
        }
        for ordinal, revision in enumerate(challenger_revisions, start=1)
    )
    write_canonical_json(
        destination / "summary.json",
        {
            "schema_version": EVIDENCE_VIEW_VERSION,
            "number": number,
            "branches": branches,
        },
    )
    for raw_attempt in validated_attempts:
        attempt_id = cast(str, raw_attempt["attempt_id"])
        branch = cast(str, raw_attempt["branch"])
        challenger_ordinal = cast(int, raw_attempt["challenger_ordinal"])
        trajectory_ordinal = cast(int, raw_attempt["trajectory_ordinal"])
        ordinal = cast(int, raw_attempt["ordinal"])
        branch_label = (
            BranchRole.ACTIVE.value
            if branch == BranchRole.ACTIVE.value
            else f"challenger-{challenger_ordinal:04d}"
        )
        attempt_root = (
            destination
            / "branches"
            / branch_label
            / "trajectories"
            / f"{trajectory_ordinal:08d}"
            / "attempts"
            / f"{ordinal:08d}"
        )
        attempt_root.mkdir(parents=True, mode=0o700)
        _copy_optional_json(
            lineage / "reports" / label / f"{attempt_id}.json",
            attempt_root / "report.json",
        )
        _materialize_latest_conversation(
            lineage / "traces" / label,
            f"{attempt_id}-run-*.json",
            attempt_root / "conversation.jsonl",
            artifacts=artifacts,
        )


def _append_evolver_completed_epoch(
    lineage: Path,
    epochs: Path,
    number: int,
    artifacts: LocalArtifactStore,
) -> None:
    """Project one completed Epoch without hiding losing Agent or Kernel history."""
    label = f"{number:08d}"
    summary = _json_object(
        lineage / "epochs" / f"{label}.json",
        f"completed Epoch {number} summary",
    )
    raw_attempts = summary.get("attempts")
    active_revision = summary.get("active_kernel_agent_revision_id")
    raw_challenger_revisions = summary.get("challenger_kernel_agent_revision_ids")
    winner_revision = summary.get("winner_kernel_agent_revision_id")
    if (
        summary.get("number") != number
        or not isinstance(raw_attempts, list)
        or not isinstance(active_revision, str)
        or not isinstance(raw_challenger_revisions, list)
        or not all(isinstance(item, str) for item in raw_challenger_revisions)
        or not isinstance(winner_revision, str)
        or winner_revision not in {active_revision, *raw_challenger_revisions}
    ):
        raise ValueError(f"completed Epoch {number} summary is inconsistent")
    challenger_revisions = cast(list[str], raw_challenger_revisions)
    raw_proposals = summary.get("challenger_proposals", [])
    if not isinstance(raw_proposals, list):
        raise ValueError(f"completed Epoch {number} Challenger proposals are invalid")
    proposals: dict[int, dict[str, JsonValue]] = {}
    for raw in raw_proposals:
        if not isinstance(raw, dict):
            raise ValueError(f"completed Epoch {number} Challenger proposal is invalid")
        ordinal = raw.get("challenger_ordinal")
        revision = raw.get("kernel_agent_revision_id")
        proposal_type = raw.get("proposal_type")
        base_revision = raw.get("base_revision_id")
        if (
            not isinstance(ordinal, int)
            or ordinal <= 0
            or ordinal > len(challenger_revisions)
            or revision != challenger_revisions[ordinal - 1]
            or proposal_type not in {"evolved", "reuse", "evolve_from_history", "replica"}
            or not isinstance(base_revision, str)
            or ordinal in proposals
        ):
            raise ValueError(f"completed Epoch {number} Challenger proposal is inconsistent")
        if proposal_type == "replica" and (
            number != 1
            or revision != active_revision
            or base_revision != active_revision
            or raw.get("evolution_trace_digest") is not None
        ):
            raise ValueError(
                "Replica must reuse the initial Active Agent without an Evolution trace"
            )
        proposals[ordinal] = raw
    validated_attempts = _validated_completed_attempts(
        raw_attempts,
        number,
        active_revision=active_revision,
        challenger_revisions=challenger_revisions,
    )

    destination = epochs / label
    destination.mkdir(mode=0o700)
    projected_summary = {key: value for key, value in summary.items() if key != "attempts"}
    branches: list[JsonValue] = [
        {
            "branch": BranchRole.ACTIVE.value,
            "challenger_ordinal": 0,
            "kernel_agent_revision_id": active_revision,
            "selected": winner_revision == active_revision,
        }
    ]
    branches.extend(
        [
            {
                "branch": BranchRole.CHALLENGER.value,
                "challenger_ordinal": ordinal,
                "kernel_agent_revision_id": revision,
                "selected": winner_revision == revision,
                **(
                    {}
                    if ordinal not in proposals
                    else {
                        "proposal_type": proposals[ordinal]["proposal_type"],
                        "base_revision_id": proposals[ordinal]["base_revision_id"],
                        "evolution_trace_digest": proposals[ordinal]["evolution_trace_digest"],
                    }
                ),
            }
            for ordinal, revision in enumerate(challenger_revisions, start=1)
        ]
    )
    projected_summary["branches"] = branches
    write_canonical_json(destination / "summary.json", projected_summary)
    _copy_optional_json(
        lineage / "measurements" / f"{label}.json",
        destination / "measurements.json",
    )

    source_lessons = lineage / "lessons" / f"{label}.json"
    if source_lessons.is_file() and not source_lessons.is_symlink():
        write_canonical_json(
            destination / "lessons.json",
            _json_object(source_lessons, f"completed Epoch {number} lessons"),
        )

    for challenger_ordinal in range(1, len(challenger_revisions) + 1):
        source_evolver = lineage / "traces" / label / f"evolver-{challenger_ordinal:04d}.json"
        if source_evolver.is_file() and not source_evolver.is_symlink():
            _materialize_session_trace(
                source_evolver,
                destination / "evolution" / f"challenger-{challenger_ordinal:04d}" / "trace",
                artifacts,
            )

    for raw_attempt in validated_attempts:
        attempt_id = cast(str, raw_attempt["attempt_id"])
        branch = cast(str, raw_attempt["branch"])
        challenger_ordinal = cast(int, raw_attempt["challenger_ordinal"])
        trajectory_ordinal = cast(int, raw_attempt["trajectory_ordinal"])
        ordinal = cast(int, raw_attempt["ordinal"])
        branch_label = (
            BranchRole.ACTIVE.value
            if branch == BranchRole.ACTIVE.value
            else f"challenger-{challenger_ordinal:04d}"
        )
        attempt_root = (
            destination
            / "branches"
            / branch_label
            / "trajectories"
            / f"{trajectory_ordinal:08d}"
            / "attempts"
            / f"{ordinal:08d}"
        )
        attempt_root.mkdir(parents=True, mode=0o700)
        write_canonical_json(attempt_root / "summary.json", raw_attempt)
        _copy_optional_json(
            lineage / "reports" / label / f"{attempt_id}.json",
            attempt_root / "report.json",
        )
        _copy_optional_json(
            lineage / "diffs" / label / f"{attempt_id}.json",
            attempt_root / "kernel-diff.json",
        )
        _materialize_trace_matches(
            lineage / "traces" / label,
            f"{attempt_id}-run-*.json",
            attempt_root / "traces",
            prefix=f"{attempt_id}-",
            artifacts=artifacts,
        )

    _materialize_epoch_kernels(destination, summary, validated_attempts, artifacts)
    _materialize_epoch_kernel_trials(
        destination,
        lineage / "kernel-trials" / f"{label}.json",
        artifacts,
    )


def _validated_completed_attempts(
    raw_attempts: list[JsonValue],
    epoch_number: int,
    *,
    active_revision: str,
    challenger_revisions: list[str],
) -> list[dict[str, JsonValue]]:
    """Validate all branch identities once and return typed JSON objects."""
    selected: list[dict[str, JsonValue]] = []
    seen: set[tuple[str, int, int, int]] = set()
    valid_branches = {item.value for item in BranchRole}
    for raw_attempt in raw_attempts:
        if not isinstance(raw_attempt, dict):
            raise ValueError(f"completed Epoch {epoch_number} Attempt summary is invalid")
        attempt_id = raw_attempt.get("attempt_id")
        branch = raw_attempt.get("branch")
        challenger_ordinal = raw_attempt.get("challenger_ordinal")
        trajectory_ordinal = raw_attempt.get("trajectory_ordinal")
        ordinal = raw_attempt.get("ordinal")
        expected_revision = active_revision
        if branch == BranchRole.CHALLENGER.value and isinstance(challenger_ordinal, int):
            expected_revision = (
                challenger_revisions[challenger_ordinal - 1]
                if 0 < challenger_ordinal <= len(challenger_revisions)
                else ""
            )
        identity = (branch, challenger_ordinal, trajectory_ordinal, ordinal)
        if (
            not isinstance(attempt_id, str)
            or not isinstance(branch, str)
            or branch not in valid_branches
            or not isinstance(challenger_ordinal, int)
            or isinstance(challenger_ordinal, bool)
            or (
                (branch == BranchRole.ACTIVE.value and challenger_ordinal != 0)
                or (branch == BranchRole.CHALLENGER.value and challenger_ordinal <= 0)
            )
            or not isinstance(trajectory_ordinal, int)
            or isinstance(trajectory_ordinal, bool)
            or trajectory_ordinal <= 0
            or raw_attempt.get("kernel_agent_revision_id") != expected_revision
            or not isinstance(ordinal, int)
            or isinstance(ordinal, bool)
            or ordinal <= 0
            or identity in seen
        ):
            raise ValueError(f"completed Epoch {epoch_number} Attempt identity is invalid")
        seen.add(cast(tuple[str, int, int, int], identity))
        selected.append(raw_attempt)
    if not selected:
        raise ValueError(f"completed Epoch {epoch_number} has no Attempts")
    return selected


def _materialize_epoch_kernels(
    destination: Path,
    summary: dict[str, JsonValue],
    attempts: list[dict[str, JsonValue]],
    artifacts: LocalArtifactStore,
) -> None:
    """Materialize every Kernel referenced by a completed Epoch and index its roles."""
    records: dict[str, dict[str, JsonValue]] = {}

    def add(raw: JsonValue, role: str, *, attempt_id: str | None = None) -> None:
        if not isinstance(raw, dict):
            return
        revision_id = raw.get("kernel_revision_id")
        raw_digest = raw.get("artifact_digest")
        if not isinstance(revision_id, str) or not isinstance(raw_digest, str):
            return
        parse_kernel_revision_id(revision_id)
        digest = parse_artifact_digest(raw_digest)
        stored = artifacts.verify(digest)
        if stored.kind is not ArtifactKind.KERNEL:
            raise ValueError("Evidence Kernel source has the wrong Artifact kind")
        existing = records.get(revision_id)
        if existing is None:
            existing = {
                **raw,
                "roles": [],
                "attempt_ids": [],
                "path": f"{revision_id}/",
            }
            records[revision_id] = existing
            artifacts.materialize(digest, destination / "kernels" / revision_id)
        elif existing.get("artifact_digest") != raw_digest:
            raise ValueError("Kernel revision maps to conflicting Artifact digests")
        roles = cast(list[JsonValue], existing["roles"])
        if role not in roles:
            roles.append(role)
        if attempt_id is not None:
            attempt_ids = cast(list[JsonValue], existing["attempt_ids"])
            if attempt_id not in attempt_ids:
                attempt_ids.append(attempt_id)

    add(summary.get("starting_kernel"), "starting")
    add(summary.get("best_kernel"), "best")
    for attempt in attempts:
        attempt_id = attempt.get("attempt_id")
        add(
            attempt.get("output"),
            "attempt_output",
            attempt_id=attempt_id if isinstance(attempt_id, str) else None,
        )
    write_canonical_json(
        destination / "kernels" / "index.json",
        {
            "schema_version": EVIDENCE_VIEW_VERSION,
            "kernels": list(records.values()),
        },
    )


def _materialize_epoch_kernel_trials(
    destination: Path,
    source: Path,
    artifacts: LocalArtifactStore,
) -> None:
    """Materialize every observed experimental candidate without assigning a vN revision."""
    root = destination / "kernel-trials"
    root.mkdir(mode=0o700)
    values: list[JsonValue] = []
    if source.is_file() and not source.is_symlink():
        document = _json_object(source, "completed Epoch Kernel Trials")
        raw_trials = document.get("kernel_trials")
        if document.get("schema_version") != 1 or not isinstance(raw_trials, list):
            raise ValueError("completed Epoch Kernel Trial index is invalid")
        for raw in raw_trials:
            if not isinstance(raw, dict):
                raise ValueError("completed Epoch Kernel Trial record is invalid")
            trial_id = raw.get("kernel_trial_id")
            digest_value = raw.get("kernel_artifact_digest")
            if (
                not isinstance(trial_id, str)
                or not trial_id.startswith("gtrial_")
                or not trial_id.removeprefix("gtrial_").isalnum()
                or not isinstance(digest_value, str)
            ):
                raise ValueError("completed Epoch Kernel Trial identity is invalid")
            digest = parse_artifact_digest(digest_value)
            stored = artifacts.verify(digest)
            if stored.kind is not ArtifactKind.KERNEL:
                raise ValueError("Kernel Trial source has the wrong Artifact kind")
            artifacts.materialize(digest, root / trial_id / "source")
            observations = raw.get("observations")
            if not isinstance(observations, list):
                raise ValueError("completed Epoch Kernel Trial observations are invalid")
            materialized_results: set[str] = set()
            for observation in observations:
                if not isinstance(observation, dict):
                    raise ValueError("completed Epoch Kernel Trial observation is invalid")
                raw_result_digest = observation.get("gateway_result_digest")
                if raw_result_digest is None:
                    continue
                raw_response_digest = observation.get("result_artifact_digest")
                if not isinstance(raw_result_digest, str) or not isinstance(
                    raw_response_digest, str
                ):
                    raise ValueError("completed Epoch Gateway Result identity is invalid")
                result_digest = parse_artifact_digest(raw_result_digest)
                response_digest = parse_artifact_digest(raw_response_digest)
                if raw_result_digest in materialized_results:
                    continue
                response = artifacts.verify(response_digest)
                if response.kind is not ArtifactKind.GATEWAY_RESULT:
                    raise ValueError("Kernel Trial Gateway response has the wrong Artifact kind")
                artifacts.materialize(
                    response_digest,
                    root
                    / trial_id
                    / "gateway-results"
                    / str(result_digest).removeprefix("sha256:"),
                )
                materialized_results.add(raw_result_digest)
            values.append(
                {
                    **{key: value for key, value in raw.items() if key != "disposition"},
                    "source_path": f"{trial_id}/source/",
                    "gateway_results_path": f"{trial_id}/gateway-results/",
                }
            )
    write_canonical_json(
        root / "index.json",
        {
            "schema_version": EVIDENCE_VIEW_VERSION,
            "kernel_trials": values,
        },
    )


def _append_current_lineage_attempts(
    destination: Path,
    attempt_payload: Path,
    *,
    lineage_checkpoint: ArtifactDigest,
    current_epoch_number: int,
    branch: BranchRole,
    challenger_ordinal: int,
    trajectory_ordinal: int,
    selected_revision: KernelAgentRevisionId,
    attempt_ordinal: int,
    artifacts: LocalArtifactStore,
) -> None:
    context = _json_object(attempt_payload / "context.json", "Attempt Evidence context")
    if (
        context.get("branch") != branch.value
        or context.get("challenger_ordinal") != challenger_ordinal
        or context.get("trajectory_ordinal") != trajectory_ordinal
        or context.get("ordinal") != attempt_ordinal
        or context.get("epoch_evidence_checkpoint") != str(lineage_checkpoint)
        or not isinstance(context.get("previous_attempt_ids"), list)
    ):
        raise ValueError("Attempt Evidence context disagrees with the requested view")
    epoch_root = destination / "epochs" / f"{current_epoch_number:08d}"
    if epoch_root.exists():
        raise ValueError("in-progress Epoch collides with completed Evidence")
    epoch_root.mkdir(mode=0o700)
    attempts_root = epoch_root / "trajectories" / f"{trajectory_ordinal:08d}" / "attempts"
    attempts_root.mkdir(parents=True, mode=0o700)
    expected_ordinals = set(range(1, attempt_ordinal))
    actual_ordinals: set[int] = set()
    source_attempts = attempt_payload / "attempts"
    for source in sorted(source_attempts.glob("*.json")):
        raw = _json_object(source, "current-branch Attempt summary")
        ordinal = raw.get("ordinal")
        if (
            not isinstance(ordinal, int)
            or isinstance(ordinal, bool)
            or raw.get("branch") != branch.value
            or raw.get("challenger_ordinal") != challenger_ordinal
            or raw.get("trajectory_ordinal") != trajectory_ordinal
            or raw.get("kernel_agent_revision_id") != str(selected_revision)
            or ordinal in actual_ordinals
        ):
            raise ValueError("current-branch Attempt identity is invalid")
        actual_ordinals.add(ordinal)
        attempt_root = attempts_root / f"{ordinal:08d}"
        attempt_root.mkdir(mode=0o700)
        _project_optional_json(
            attempt_payload / "reports" / f"{ordinal:08d}.json",
            attempt_root / "report.json",
        )
        _materialize_latest_conversation(
            attempt_payload / "traces",
            f"{ordinal:08d}-run-*.json",
            attempt_root / "conversation.jsonl",
            artifacts=artifacts,
        )
    if actual_ordinals != expected_ordinals:
        raise ValueError("current-lineage Evidence has an incomplete Attempt sequence")


def _lineage_through_epoch(lineage_payload: Path) -> int:
    checkpoint = _json_object(lineage_payload / "checkpoint.json", "Lineage checkpoint")
    through = checkpoint.get("through_epoch")
    if (
        checkpoint.get("schema_version") != 1
        or not isinstance(checkpoint.get("lineage_id"), str)
        or not isinstance(through, int)
        or isinstance(through, bool)
        or through < 0
    ):
        raise ValueError("Lineage Evidence checkpoint is invalid")
    return through


def _copy_directory_contents(source: Path, destination: Path) -> None:
    for child in source.iterdir():
        target = destination / child.name
        if child.is_symlink():
            raise ValueError("Evidence view source cannot contain symbolic links")
        if child.is_dir():
            shutil.copytree(child, target)
        elif child.is_file():
            shutil.copyfile(child, target)
        else:
            raise ValueError("Evidence view source must contain only regular files")


def _project_optional_json(source: Path, destination: Path) -> None:
    if not source.exists():
        return
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"Evidence projection source is not a regular file: {source.name}")
    value = _json_object(source, "Evidence projection source")
    write_canonical_json(destination, _without_branch_identity(value))


def _copy_optional_json(source: Path, destination: Path) -> None:
    """Copy optional JSON while retaining control-plane identities for an Evolver."""
    if not source.exists():
        return
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"Evidence projection source is not a regular file: {source.name}")
    write_canonical_json(destination, _json_object(source, "Evidence projection source"))


def _materialize_trace_matches(
    source_root: Path,
    pattern: str,
    destination: Path,
    *,
    prefix: str,
    artifacts: LocalArtifactStore,
) -> None:
    if not source_root.is_dir():
        return
    for source in sorted(source_root.glob(pattern)):
        if source.is_symlink() or not source.is_file():
            raise ValueError("Evidence trace projection must be a regular file")
        name = source.name.removeprefix(prefix).removesuffix(".json")
        _materialize_session_trace(
            source,
            destination / name,
            artifacts,
        )


def _materialize_latest_conversation(
    source_root: Path,
    pattern: str,
    destination: Path,
    *,
    artifacts: LocalArtifactStore,
) -> None:
    """Project only the latest sealed backend-neutral transcript for one Attempt."""
    if not source_root.is_dir():
        return
    matches = sorted(source_root.glob(pattern))
    if not matches:
        return
    source = matches[-1]
    if source.is_symlink() or not source.is_file():
        raise ValueError("Evidence trace projection must be a regular file")
    projection = _json_object(source, "Evidence trace projection")
    raw_digest = projection.get("source_session_log_digest")
    if not isinstance(raw_digest, str):
        raise ValueError("Evidence trace projection has no source Session Log Digest")
    digest = parse_artifact_digest(raw_digest)
    stored = artifacts.verify(digest)
    if stored.kind is not ArtifactKind.SESSION_LOG:
        raise ValueError("Evidence trace source has the wrong Artifact kind")
    conversation = stored.payload_path / "conversation.jsonl"
    if conversation.is_symlink() or not conversation.is_file():
        raise ValueError("Session Artifact has no regular conversation.jsonl")
    payload, _removed = retained_session_file(
        "conversation.jsonl",
        conversation.read_bytes(),
    )
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    destination.write_bytes(payload)


def _materialize_session_trace(
    projection_path: Path,
    destination: Path,
    artifacts: LocalArtifactStore,
) -> None:
    projection = _json_object(projection_path, "Evidence trace projection")
    raw_digest = projection.get("source_session_log_digest")
    if not isinstance(raw_digest, str):
        raise ValueError("Evidence trace projection has no source Session Log Digest")
    digest = parse_artifact_digest(raw_digest)
    stored = artifacts.verify(digest)
    if stored.kind is not ArtifactKind.SESSION_LOG:
        raise ValueError("Evidence trace source has the wrong Artifact kind")
    artifacts.materialize(digest, destination)
    enforce_session_trace_retention(destination)


def _json_object(path: Path, label: str) -> dict[str, JsonValue]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular JSON file")
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{label} must contain a JSON object")
    return value


def _without_branch_identity(value: JsonValue) -> JsonValue:
    """Remove control-plane branch labels from a projected JSON value."""
    if isinstance(value, dict):
        return {
            key: _without_branch_identity(item)
            for key, item in value.items()
            if key
            not in {
                "branch",
                "visible_branch",
                "challenger_ordinal",
                "active_kernel_agent_revision_id",
                "challenger_kernel_agent_revision_id",
                "challenger_kernel_agent_revision_ids",
                "winner_kernel_agent_revision_id",
            }
        }
    if isinstance(value, list):
        return [_without_branch_identity(item) for item in value]
    return value
