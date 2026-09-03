"""Fixed, stateless Evolver execution and Challenger collection."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Literal, Protocol, Self, cast
from uuid import uuid4

import anyio
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..artifacts.local import ArtifactKind, JsonValue, LocalArtifactStore
from ..domain.errors import InfrastructureError
from ..domain.ids import (
    ArtifactDigest,
    KernelAgentRevisionId,
    new_worker_session_id,
    parse_artifact_digest,
    parse_kernel_agent_revision_id,
)
from ..domain.models import (
    Dsl,
    KernelAgentRevision,
    WorkerSession,
    WorkerSessionRole,
    WorkerSessionStatus,
)
from ..filesystem import make_tree_owner_writable, make_tree_read_only
from ..kernel_agents import (
    KernelAgentBundleLimits,
    KernelAgentRevisionBuilder,
    is_ignored_kernel_agent_path,
)
from ..ports import (
    BuildChallengerRequest,
    BuildChallengerResult,
    EvolverRunner,
    KernelAgentCandidate,
    KernelAgentCandidateProposal,
    KernelAgentReuseProposal,
    RuntimeEventRecorder,
    WorkerSessionRecorder,
)
from ..serialization import canonical_json_bytes
from .evidence_view import (
    EVOLVER_EVIDENCE_PROMPT_TEXT,
    assemble_evolver_evidence_view,
)
from .launcher import WorkerLauncher, validate_worker_environment
from .process import BoundedProcessConfig, BoundedProcessRunner
from .session_trace import enforce_session_trace_retention
from .state_selection import RuntimeStateAttempt, select_winning_trajectory_terminal_state
from .token_usage import ProviderUsageReportV2, UsageUnit
from .workspace import (
    copy_reusable_agent_state,
    ensure_reusable_directories,
    materialize_reusable_agent_state_snapshot,
    resolve_revision_runtime_state_seed,
    validate_reusable_agent_state_seed,
)

EVOLUTION_INPUT_VERSION: Literal[11] = 11
EVOLUTION_TRACE_VERSION: Literal[9] = 9
EVOLUTION_FAILURE_VERSION: Literal[5] = 5
EVOLVER_LAUNCH_INSTRUCTION = "Run the versioned Evolver Bundle once."
EVOLVER_TIMEOUT_EXIT_STATUS = 124
EVOLVER_WORKSPACE_RELATIVE_PATH = PurePosixPath("input/evolver")


def _last_completed_epoch_pool(
    evidence_root: Path,
) -> tuple[KernelAgentRevisionId, tuple[KernelAgentRevisionId, ...]] | None:
    """Resolve the most recent completed Epoch's Active and Challenger branch revisions."""
    epochs = evidence_root / "epochs"
    if epochs.is_symlink() or not epochs.is_dir():
        return None
    records = sorted(epochs.glob("*.json"), reverse=True)
    if not records:
        return None
    raw: object = json.loads(records[0].read_bytes())
    if not isinstance(raw, dict):
        raise ValueError("Completed Epoch Evidence record is invalid")
    raw_active = raw.get("active_kernel_agent_revision_id")
    raw_challengers = raw.get("challenger_kernel_agent_revision_ids")
    if (
        not isinstance(raw_active, str)
        or not isinstance(raw_challengers, list)
        or not all(isinstance(item, str) for item in raw_challengers)
    ):
        raise ValueError("Completed Epoch Branch pool is invalid")
    return (
        parse_kernel_agent_revision_id(raw_active),
        tuple(parse_kernel_agent_revision_id(cast(str, item)) for item in raw_challengers),
    )


def _active_next_epoch_runtime_state_seed(
    evidence_root: Path,
    active_revision_id: KernelAgentRevisionId,
    artifacts: LocalArtifactStore,
) -> Path | None:
    """Resolve the winning best-Kernel Trajectory's terminal State."""
    epochs = evidence_root / "epochs"
    if epochs.is_symlink() or not epochs.is_dir():
        return None
    for epoch_path in sorted(epochs.glob("*.json"), reverse=True):
        raw: object = json.loads(epoch_path.read_bytes())
        if not isinstance(raw, dict) or raw.get("winner_kernel_agent_revision_id") != str(
            active_revision_id
        ):
            continue
        raw_attempts = raw.get("attempts")
        raw_active = raw.get("active_kernel_agent_revision_id")
        raw_challengers = raw.get("challenger_kernel_agent_revision_ids")
        if (
            not isinstance(raw_attempts, list)
            or not isinstance(raw_active, str)
            or not isinstance(raw_challengers, list)
            or not all(isinstance(item, str) for item in raw_challengers)
        ):
            raise ValueError("Completed Epoch Runtime State catalog is invalid")
        attempts: list[RuntimeStateAttempt] = []
        for item in raw_attempts:
            if not isinstance(item, dict):
                raise ValueError("Completed Epoch Attempt State is invalid")
            required = (
                item.get("attempt_id"),
                item.get("branch"),
                item.get("challenger_ordinal"),
                item.get("trajectory_ordinal"),
                item.get("ordinal"),
                item.get("kernel_agent_revision_id"),
            )
            if (
                not isinstance(required[0], str)
                or required[1] not in {"active", "challenger"}
                or not isinstance(required[2], int)
                or isinstance(required[2], bool)
                or not isinstance(required[3], int)
                or isinstance(required[3], bool)
                or not isinstance(required[4], int)
                or isinstance(required[4], bool)
                or not isinstance(required[5], str)
            ):
                raise ValueError("Completed Epoch Attempt identity is invalid")
            output = item.get("output")
            latency: float | None = None
            if isinstance(output, dict):
                raw_latency = output.get("latency_us")
                if isinstance(raw_latency, (int, float)) and not isinstance(raw_latency, bool):
                    latency = float(raw_latency)
            raw_input_state = item.get("input_runtime_state_digest")
            raw_terminal_state = item.get("runtime_state_digest")
            attempts.append(
                RuntimeStateAttempt(
                    attempt_id=required[0],
                    branch=cast(str, required[1]),
                    challenger_ordinal=required[2],
                    trajectory_ordinal=required[3],
                    ordinal=required[4],
                    kernel_agent_revision_id=required[5],
                    accepted_as_branch_best=item.get("accepted_as_branch_best") is True,
                    output_latency_us=latency,
                    input_runtime_state_digest=(
                        None
                        if not isinstance(raw_input_state, str)
                        else parse_artifact_digest(raw_input_state)
                    ),
                    runtime_state_digest=(
                        None
                        if not isinstance(raw_terminal_state, str)
                        else parse_artifact_digest(raw_terminal_state)
                    ),
                )
            )
        producer_attempt_id: str | None = None
        best_kernel = raw.get("best_kernel")
        if isinstance(best_kernel, dict):
            raw_producer = best_kernel.get("produced_by_attempt_id")
            if isinstance(raw_producer, str):
                producer_attempt_id = raw_producer
        digest = select_winning_trajectory_terminal_state(
            attempts=tuple(attempts),
            winner_revision_id=active_revision_id,
            active_revision_id=parse_kernel_agent_revision_id(raw_active),
            challenger_revision_ids=tuple(
                parse_kernel_agent_revision_id(cast(str, item)) for item in raw_challengers
            ),
            best_kernel_producer_attempt_id=producer_attempt_id,
        )
        if digest is None:
            return None
        state = artifacts.verify(digest)
        if state.kind is not ArtifactKind.KERNEL_AGENT_RUNTIME_STATE:
            raise ValueError("Attempt Runtime State has the wrong Artifact kind")
        validate_reusable_agent_state_seed(state.payload_path)
        return state.payload_path
    return None


class EvolutionPathsV2(BaseModel):
    """Fixed paths exposed to an Evolver process."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    agents: Literal["input/agents"] = "input/agents"
    evidence: Literal["input/evidence"] = "input/evidence"
    candidate: Literal["candidate"] = "candidate"
    scratch: Literal["scratch"] = "scratch"
    output: Literal["scratch/evolution-report.json"] = "scratch/evolution-report.json"


class VisibleAgentRevisionV2(BaseModel):
    """One read-only Agent design exposed to an Evolver invocation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    revision_id: KernelAgentRevisionId
    version: str = Field(pattern=r"^agent-v[0-9]+$")
    optimizer_digest: ArtifactDigest
    path: str = Field(min_length=1, max_length=300)
    optimization_summary_path: str = Field(min_length=1, max_length=300)
    sessions_path: str | None = Field(default=None, min_length=1, max_length=300)
    reports_path: str | None = Field(default=None, min_length=1, max_length=300)
    runtime_state_path: str = Field(min_length=1, max_length=300)
    parent: bool
    relationship: Literal["active", "challenger", "current_epoch_challenger", "lineage_history"]
    challenger_ordinal: int | None = Field(default=None, gt=0)
    parent_revision_id: KernelAgentRevisionId | None
    created_by: str = Field(min_length=1, max_length=200)

    @field_validator("revision_id", mode="before")
    @classmethod
    def _validate_revision_id(cls, value: object) -> KernelAgentRevisionId:
        if not isinstance(value, str):
            raise ValueError("visible Agent revision_id must be a string")
        return parse_kernel_agent_revision_id(value)

    @field_validator("optimizer_digest", mode="before")
    @classmethod
    def _validate_optimizer_digest(cls, value: object) -> ArtifactDigest:
        if not isinstance(value, str):
            raise ValueError("visible Agent optimizer_digest must be a string")
        return parse_artifact_digest(value)

    @field_validator("parent_revision_id", mode="before")
    @classmethod
    def _validate_parent_revision_id(
        cls,
        value: object,
    ) -> KernelAgentRevisionId | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError("visible Agent parent_revision_id must be a string or null")
        return parse_kernel_agent_revision_id(value)

    @model_validator(mode="after")
    def _validate_relationship(self) -> Self:
        if self.parent and self.relationship not in {"active", "challenger"}:
            raise ValueError("the Parent must occupy the last completed Epoch's comparison pool")
        if (self.relationship in {"challenger", "current_epoch_challenger"}) != (
            self.challenger_ordinal is not None
        ):
            raise ValueError("only Challenger entries have a Challenger ordinal")
        competed = self.relationship in {"active", "challenger"}
        if competed != (self.sessions_path is not None) or competed != (
            self.reports_path is not None
        ):
            raise ValueError(
                "Sessions and Attempt reports belong to the last completed Epoch's branches"
            )
        return self


class EvolutionInputManifestV11(BaseModel):
    """Immutable parent, Agent pool, and lineage evidence for one Evolution session."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[11] = EVOLUTION_INPUT_VERSION
    parent_revision_id: KernelAgentRevisionId
    evidence_checkpoint: ArtifactDigest
    idempotency_key: str = Field(min_length=1, max_length=300)
    dsl: Dsl
    optimizer_digest: ArtifactDigest
    visible_agents: tuple[VisibleAgentRevisionV2, ...]
    paths: EvolutionPathsV2 = EvolutionPathsV2()

    @field_validator("parent_revision_id", mode="before")
    @classmethod
    def _validate_parent_id(cls, value: object) -> KernelAgentRevisionId:
        if not isinstance(value, str):
            raise ValueError("parent_revision_id must be a string")
        return parse_kernel_agent_revision_id(value)

    @field_validator(
        "evidence_checkpoint",
        "optimizer_digest",
        mode="before",
    )
    @classmethod
    def _validate_digest(cls, value: object) -> ArtifactDigest:
        if not isinstance(value, str):
            raise ValueError("Evolution artifact digest must be a string")
        return parse_artifact_digest(value)

    @model_validator(mode="after")
    def _validate_visible_agents(self) -> Self:
        if not self.visible_agents:
            raise ValueError("visible_agents must contain the parent Agent revision")
        revision_ids = [item.revision_id for item in self.visible_agents]
        if len(set(revision_ids)) != len(revision_ids):
            raise ValueError("visible_agents cannot contain duplicate Agent revisions")
        parents = [item for item in self.visible_agents if item.parent]
        if len(parents) != 1:
            raise ValueError("visible_agents must identify exactly one parent")
        parent = parents[0]
        if (
            parent.revision_id != self.parent_revision_id
            or parent.optimizer_digest != self.optimizer_digest
        ):
            raise ValueError("visible parent disagrees with the Evolution parent")
        return self

    def canonical_json_bytes(self) -> bytes:
        """Serialize stable UTF-8 JSON for the private Evolution workspace."""
        return canonical_json_bytes(self.model_dump(mode="json"))


class UnimplementedCapabilityV1(BaseModel):
    """Advisory Agent capability the Evolver could not add to this Candidate."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    capability: str = Field(min_length=1, max_length=2000)
    expected_benefit: str = Field(min_length=1, max_length=2000)
    reason_unimplemented: str = Field(min_length=1, max_length=2000)


class EvolutionOutput(BaseModel):
    """One uniform Agent-authored Challenger proposal."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    proposal_type: Literal["evolved", "evolve_from_history", "reuse"]
    kernel_agent_revision_id: KernelAgentRevisionId
    hypothesis: str = Field(min_length=1, max_length=4000)
    expected_effect: str = Field(min_length=1, max_length=4000)
    changed_paths: tuple[str, ...] = Field(max_length=512)
    # Defaulted because sealed historical traces predating this field are re-parsed
    # by _previous_evolution_output under extra="forbid".
    contributing_revision_ids: tuple[KernelAgentRevisionId, ...] = Field(
        default=(),
        max_length=64,
    )
    unimplemented_capabilities: tuple[UnimplementedCapabilityV1, ...] = Field(max_length=64)

    @field_validator("kernel_agent_revision_id", mode="before")
    @classmethod
    def _validate_source_reference(cls, value: object) -> KernelAgentRevisionId:
        if not isinstance(value, str):
            raise ValueError("kernel_agent_revision_id must be a string")
        return parse_kernel_agent_revision_id(value)

    @field_validator("contributing_revision_ids", mode="before")
    @classmethod
    def _validate_contributing_revisions(cls, value: object) -> tuple[str, ...]:
        if not isinstance(value, list | tuple):
            raise ValueError("contributing_revision_ids must be an array")
        normalized: list[str] = []
        for item in value:
            if not isinstance(item, str):
                raise ValueError("contributing_revision_ids entries must be strings")
            normalized.append(str(parse_kernel_agent_revision_id(item)))
        if len(set(normalized)) != len(normalized):
            raise ValueError("contributing_revision_ids cannot contain duplicates")
        if normalized != sorted(normalized):
            raise ValueError("contributing_revision_ids must be sorted")
        return tuple(normalized)

    @field_validator("changed_paths")
    @classmethod
    def _validate_changed_paths(
        cls,
        value: tuple[str, ...],
    ) -> tuple[str, ...]:
        normalized: list[str] = []
        for item in value:
            relative = PurePosixPath(item)
            if relative.is_absolute() or relative.as_posix() == "." or ".." in relative.parts:
                raise ValueError("changed_paths must contain safe Source-root-relative paths")
            normalized.append(relative.as_posix())
        if len(set(normalized)) != len(normalized):
            raise ValueError("changed_paths cannot contain duplicates")
        if normalized != sorted(normalized):
            raise ValueError("changed_paths must be sorted")
        return tuple(normalized)

    @model_validator(mode="after")
    def _validate_mode_fields(self) -> Self:
        if self.proposal_type == "reuse" and self.changed_paths:
            raise ValueError("reuse requires changed_paths to be empty")
        if self.proposal_type == "reuse" and self.contributing_revision_ids:
            raise ValueError("reuse requires contributing_revision_ids to be empty")
        if self.kernel_agent_revision_id in self.contributing_revision_ids:
            raise ValueError("contributing_revision_ids cannot repeat the selected Source base")
        return self


def read_evolution_output(path: Path, *, max_bytes: int) -> EvolutionOutput:
    """Parse a bounded regular output manifest after the Evolver is reaped."""
    try:
        path_stat = path.lstat()
    except FileNotFoundError as error:
        raise ValueError("Evolver did not produce ATREX_EVOLUTION_OUTPUT") from error
    if stat.S_ISLNK(path_stat.st_mode) or not stat.S_ISREG(path_stat.st_mode):
        raise ValueError("Evolution output manifest must be a regular file")
    if path_stat.st_size > max_bytes:
        raise ValueError("Evolution output manifest exceeds byte limit")
    return EvolutionOutput.model_validate_json(path.read_bytes())


class EvolutionAgentDescriptorV3(BaseModel):
    """Immutable Evolver Bundle identity and launch fingerprints for one run."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    bundle_commit: str = Field(pattern=r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
    bundle_tree: str = Field(pattern=r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
    bundle_artifact_digest: ArtifactDigest
    agent_backend: Literal["claude", "codex", "qodercli", "pi"]
    model: str | None
    reasoning_effort: Literal["low", "medium", "high", "max"]
    session_settings_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    command_executable: str = Field(min_length=1)
    command_argv_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    environment_keys: tuple[str, ...]
    isolated_home_environment_keys: tuple[str, ...]

    @field_validator("bundle_artifact_digest", mode="before")
    @classmethod
    def _validate_digest(cls, value: object) -> ArtifactDigest:
        if not isinstance(value, str):
            raise ValueError("Evolver Bundle Artifact digest must be a string")
        return parse_artifact_digest(value)


class EvolutionCandidateTraceV3(BaseModel):
    """Sealed source and runtime-state digests from one successful Evolution run."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    optimizer_digest: ArtifactDigest
    runtime_state_digest: ArtifactDigest | None

    @field_validator("optimizer_digest", mode="before")
    @classmethod
    def _validate_digest(cls, value: object) -> ArtifactDigest:
        if not isinstance(value, str):
            raise ValueError("Evolution Candidate digest must be a string")
        return parse_artifact_digest(value)

    @field_validator("runtime_state_digest", mode="before")
    @classmethod
    def _validate_optional_state_digest(cls, value: object) -> ArtifactDigest | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError("Evolution Candidate runtime-state digest must be a string")
        return parse_artifact_digest(value)


class EvolutionTraceV9(BaseModel):
    """Immutable provenance for one successful Kernel Agent Evolution run."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[9] = EVOLUTION_TRACE_VERSION
    input: EvolutionInputManifestV11
    agent: EvolutionAgentDescriptorV3
    process_returncode: Literal[0]
    stdout: str
    stderr: str
    session_trace_digest: ArtifactDigest | None
    token_usage: ProviderUsageReportV2
    output: EvolutionOutput
    candidate: EvolutionCandidateTraceV3 | None

    @field_validator("session_trace_digest", mode="before")
    @classmethod
    def _validate_optional_digest(cls, value: object) -> ArtifactDigest | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError("Evolution session trace digest must be a string")
        return parse_artifact_digest(value)


def _previous_evolution_output(
    artifacts: LocalArtifactStore,
    revision: KernelAgentRevision,
) -> EvolutionOutput | None:
    """Load only the Agent-authored report that created one visible revision."""
    if revision.evolution_trace_digest is None:
        return None
    try:
        stored = artifacts.verify(revision.evolution_trace_digest)
    except FileNotFoundError:
        # Old or externally imported catalogs may retain provenance whose Artifact
        # is unavailable locally. Absence is represented by no projected report.
        return None
    if stored.kind is not ArtifactKind.EVOLUTION:
        raise ValueError("Agent revision Evolution trace has the wrong Artifact kind")
    raw: object = json.loads((stored.payload_path / "value.json").read_bytes())
    if not isinstance(raw, dict):
        raise ValueError("Agent revision Evolution trace must be a JSON object")
    raw_output = raw.get("output")
    raw_candidate = raw.get("candidate")
    if raw_output is None:
        # Some imported or old traces retain only Candidate provenance.
        return None
    output = EvolutionOutput.model_validate(raw_output)
    if (
        not isinstance(raw_candidate, dict)
        or raw_candidate.get("optimizer_digest") != revision.optimizer_digest
    ):
        raise ValueError("Agent revision Evolution report does not produce its Source")
    return output


class EvolutionFailureProcessV2(BaseModel):
    """Bounded process evidence retained for a rejected Evolution result."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    agent: EvolutionAgentDescriptorV3
    returncode: int
    stdout: str
    stderr: str
    session_trace_digest: ArtifactDigest | None
    session_trace_retention_error_type: str | None
    token_usage: ProviderUsageReportV2

    @field_validator("session_trace_digest", mode="before")
    @classmethod
    def _validate_optional_digest(cls, value: object) -> ArtifactDigest | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError("Evolution failure session trace digest must be a string")
        return parse_artifact_digest(value)


class EvolutionFailureTraceV5(BaseModel):
    """Immutable bounded evidence for an Evolution run that produced no Challenger."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[5] = EVOLUTION_FAILURE_VERSION
    status: Literal["failed"] = "failed"
    input: EvolutionInputManifestV11
    phase: Literal["session", "candidate_validation"]
    error_type: str = Field(min_length=1, max_length=200)
    process: EvolutionFailureProcessV2 | None


@dataclass(frozen=True, slots=True)
class PreparedEvolution:
    """One private, append-only Evolution process allocation."""

    root: Path
    control_root: Path
    manifest_path: Path
    candidate_root: Path
    candidate_runtime_state_base_root: Path
    output_path: Path
    parent_revision: KernelAgentRevision
    model: str | None = None


class EvolutionWorkspaceAssembler:
    """Materialize frozen Agents and seed one writable source/runtime-state Candidate."""

    def __init__(
        self,
        root: str | Path,
        artifacts: LocalArtifactStore,
        *,
        evolver_bundle_digest: ArtifactDigest | None = None,
        attempt_workspaces_root: str | Path | None = None,
    ) -> None:
        self._root = Path(root).resolve()
        self._artifacts = artifacts
        self._evolver_bundle_digest = evolver_bundle_digest
        self._attempt_workspaces_root = attempt_workspaces_root
        if evolver_bundle_digest is not None:
            parse_artifact_digest(evolver_bundle_digest)
        self._root.mkdir(parents=True, exist_ok=True, mode=0o700)

    def prepare(self, request: BuildChallengerRequest) -> PreparedEvolution:
        """Create a fresh workspace containing one complete writable Agent Candidate."""
        run_id = f"run-{uuid4().hex}"
        run_root = self._root / request.parent_revision.id / run_id
        control_root = self._root / ".control" / request.parent_revision.id / run_id
        run_root.mkdir(parents=True, mode=0o700)
        control_root.mkdir(parents=True, mode=0o700)
        agents_root = run_root / "input/agents"
        evidence_root = run_root / "input/evidence"
        evolution_reports_root = run_root / "input/evolution-reports"
        candidate_root = run_root / "candidate"
        scratch_root = run_root / "scratch"
        scratch_root.mkdir(mode=0o700)
        output_path = scratch_root / "evolution-report.json"

        if self._evolver_bundle_digest is not None:
            evolver = self._artifacts.verify(self._evolver_bundle_digest)
            if evolver.kind is not ArtifactKind.EVOLVER_BUNDLE:
                raise ValueError("Evolution Evolver Bundle has the wrong Artifact kind")
            evolver_root = run_root.joinpath(*EVOLVER_WORKSPACE_RELATIVE_PATH.parts)
            self._artifacts.materialize(self._evolver_bundle_digest, evolver_root)
            make_tree_read_only(evolver_root)

        revision = request.parent_revision
        agents_root.mkdir(parents=True, mode=0o700)
        evolution_reports_root.mkdir(parents=True, mode=0o700)
        evidence = self._artifacts.verify(request.evidence_checkpoint)
        if evidence.kind is not ArtifactKind.EVIDENCE:
            raise ValueError("Evolution Evidence checkpoint has the wrong Artifact kind")
        catalog_by_id = {entry.revision.id: entry for entry in request.agent_catalog}
        visible_by_id = {
            revision_id: entry.revision for revision_id, entry in catalog_by_id.items()
        }
        visible_by_id[revision.id] = revision
        pool = _last_completed_epoch_pool(evidence.payload_path)
        pool_active, pool_challengers = (revision.id, ()) if pool is None else pool
        pool_ordinal_by_id = {
            challenger_id: ordinal
            for ordinal, challenger_id in enumerate(pool_challengers, start=1)
        }
        if (
            pool_active not in visible_by_id
            or not pool_ordinal_by_id.keys() <= visible_by_id.keys()
        ):
            raise ValueError("completed Epoch Branch pool is outside the visible Agent catalog")
        visible_agents: list[VisibleAgentRevisionV2] = []
        visible_paths: dict[KernelAgentRevisionId, tuple[str, str]] = {}
        previous_reports: list[tuple[KernelAgentRevision, EvolutionOutput, int | None]] = []
        agent_versions: dict[str, str] = {}
        pool_versions: set[str] = set()
        challenger_prefix = f"epoch:{request.epoch_id}:challenger:"
        for visible in visible_by_id.values():
            challenger_ordinal: int | None = None
            relationship: Literal[
                "active", "challenger", "current_epoch_challenger", "lineage_history"
            ] = "lineage_history"
            if visible.id == pool_active:
                relationship = "active"
            elif visible.id in pool_ordinal_by_id:
                relationship = "challenger"
                challenger_ordinal = pool_ordinal_by_id[visible.id]
            elif visible.creation_key.startswith(challenger_prefix):
                suffix = visible.creation_key.removeprefix(challenger_prefix)
                if suffix.isdigit() and int(suffix) > 0:
                    relationship = "current_epoch_challenger"
                    challenger_ordinal = int(suffix)
            catalog_entry = catalog_by_id.get(visible.id)
            if catalog_entry is None:
                raise ValueError("visible Agent revision has no lineage version")
            version = f"agent-v{catalog_entry.revision_number}"
            competed = relationship in {"active", "challenger"}
            relative = f"input/agents/{version}/source"
            runtime_state_path = f"input/agents/{version}/runtime-state"
            optimization_summary_path = f"input/evidence/{version}/optimization-summary.json"
            sessions_path = f"input/evidence/{version}/sessions" if competed else None
            reports_path = f"input/evidence/{version}/reports" if competed else None
            agent_versions[version] = visible.id
            if competed:
                pool_versions.add(version)
            self._artifacts.materialize(visible.optimizer_digest, run_root / relative)
            visible_paths[visible.id] = (relative, runtime_state_path)
            previous_output = _previous_evolution_output(self._artifacts, visible)
            if previous_output is not None:
                previous_reports.append((visible, previous_output, catalog_entry.revision_number))
            visible_agents.append(
                VisibleAgentRevisionV2(
                    revision_id=visible.id,
                    version=version,
                    optimizer_digest=visible.optimizer_digest,
                    path=relative,
                    optimization_summary_path=optimization_summary_path,
                    sessions_path=sessions_path,
                    reports_path=reports_path,
                    runtime_state_path=runtime_state_path,
                    parent=visible.id == revision.id,
                    relationship=relationship,
                    challenger_ordinal=challenger_ordinal,
                    parent_revision_id=visible.parent_id,
                    created_by=visible.created_by,
                )
            )
        used_evolution_numbers = {
            revision_number
            for _revision, _output, revision_number in previous_reports
            if revision_number is not None and revision_number > 0
        }
        next_evolution_number = 1
        for produced, previous_output, revision_number in sorted(
            previous_reports,
            key=lambda item: (item[0].created_at, item[0].id),
        ):
            evolution_number = revision_number
            if evolution_number is None or evolution_number <= 0:
                while next_evolution_number in used_evolution_numbers:
                    next_evolution_number += 1
                evolution_number = next_evolution_number
                used_evolution_numbers.add(evolution_number)
                next_evolution_number += 1
            base_paths = visible_paths.get(previous_output.kernel_agent_revision_id)
            produced_paths = visible_paths.get(produced.id)
            if base_paths is None or produced_paths is None:
                raise ValueError("Evolution report Agent paths are outside visible history")
            contributing_paths: list[str] = []
            for contributor_id in previous_output.contributing_revision_ids:
                contributor_paths = visible_paths.get(contributor_id)
                if contributor_paths is None:
                    raise ValueError("Evolution report Agent paths are outside visible history")
                contributing_paths.append(contributor_paths[0])
            report_path = evolution_reports_root / f"evo-{evolution_number}.json"
            if report_path.exists() or report_path.is_symlink():
                raise ValueError("Evolution report number is duplicated")
            report_path.write_bytes(
                canonical_json_bytes(
                    {
                        "evolution_number": evolution_number,
                        "parent": {
                            "source_path": base_paths[0],
                            "runtime_state_path": base_paths[1],
                        },
                        "generated_agent": {
                            "source_path": produced_paths[0],
                            "runtime_state_path": produced_paths[1],
                        },
                        "report": {
                            "proposal_type": previous_output.proposal_type,
                            "hypothesis": previous_output.hypothesis,
                            "expected_effect": previous_output.expected_effect,
                            "changed_paths": list(previous_output.changed_paths),
                            "contributing_source_paths": contributing_paths,
                            "unimplemented_capabilities": [
                                item.model_dump(mode="json")
                                for item in previous_output.unimplemented_capabilities
                            ],
                        },
                    }
                )
            )
        control_state_root = control_root / ".runtime"
        reusable_state_staging = control_root / "agent-state-staging"
        materialize_reusable_agent_state_snapshot(
            self._attempt_workspaces_root,
            reusable_state_staging,
            agent_lineages={
                revision_id: (
                    None
                    if revision_id not in catalog_by_id
                    else catalog_by_id[revision_id].lineage_id
                )
                for revision_id in visible_by_id
            },
            read_only=False,
        )
        assemble_evolver_evidence_view(
            evidence_root,
            control_root=control_state_root,
            lineage_payload=evidence.payload_path,
            lineage_checkpoint=request.evidence_checkpoint,
            artifacts=self._artifacts,
            agent_versions=agent_versions,
            pool_versions=frozenset(pool_versions),
        )
        for version, revision_id in agent_versions.items():
            source_state = reusable_state_staging / revision_id
            destination = agents_root / version
            if not source_state.is_dir() or source_state.is_symlink():
                raise ValueError(f"visible Agent runtime state is unavailable: {version}")
            shutil.move(source_state, destination / "runtime-state")
        if reusable_state_staging.exists():
            if any(reusable_state_staging.iterdir()):
                raise ValueError(
                    "unclassified Agent runtime state remains after Evolution assembly"
                )
            reusable_state_staging.rmdir()
        candidate_root.mkdir(mode=0o700)
        shutil.copytree(run_root / visible_paths[revision.id][0], candidate_root / "source")
        candidate_runtime_state = candidate_root / "runtime-state"
        active_runtime_state_seed = _active_next_epoch_runtime_state_seed(
            evidence.payload_path,
            revision.id,
            self._artifacts,
        ) or resolve_revision_runtime_state_seed(self._artifacts, revision)
        if active_runtime_state_seed is None:
            ensure_reusable_directories(candidate_runtime_state)
        else:
            copy_reusable_agent_state(active_runtime_state_seed, candidate_runtime_state)
        candidate_runtime_state_base = control_state_root / "candidate-runtime-state-base"
        shutil.copytree(candidate_runtime_state, candidate_runtime_state_base)
        make_tree_read_only(candidate_runtime_state_base)
        make_tree_owner_writable(candidate_root)
        make_tree_read_only(agents_root)
        make_tree_read_only(evolution_reports_root)

        manifest = EvolutionInputManifestV11(
            parent_revision_id=revision.id,
            evidence_checkpoint=request.evidence_checkpoint,
            idempotency_key=request.idempotency_key,
            dsl=revision.dsl,
            optimizer_digest=revision.optimizer_digest,
            visible_agents=tuple(visible_agents),
        )
        manifest_path = control_state_root / "evolution-input.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        manifest_path.write_bytes(manifest.canonical_json_bytes())
        os.chmod(manifest_path, 0o400)
        return PreparedEvolution(
            run_root,
            control_root,
            manifest_path,
            candidate_root,
            candidate_runtime_state_base,
            output_path,
            revision,
            request.model,
        )


@dataclass(frozen=True, slots=True)
class EvolutionSessionResult:
    """Bounded diagnostic output from one fully reaped Coding Agent process."""

    returncode: int
    stdout: str
    stderr: str
    agent: EvolutionAgentDescriptorV3
    session_trace_path: Path | None
    token_usage: ProviderUsageReportV2


@dataclass(frozen=True, slots=True)
class PreparedEvolutionLaunch:
    """Exact Evolver environment reusable by the process and dev-shell drivers."""

    environment: Mapping[str, str]
    token_usage_path: Path
    session_trace_path: Path | None


class EvolutionSessionDriver(Protocol):
    """Run one Coding Agent process and return only after its process is reaped."""

    async def run(self, prepared: PreparedEvolution) -> EvolutionSessionResult:
        """Mutate only the prepared Candidate and return bounded diagnostics."""
        ...


@dataclass(frozen=True, slots=True)
class EvolutionProcessConfig:
    """Immutable Bundle identity, command, environment, and process limits."""

    bundle_commit: str
    bundle_tree: str
    bundle_artifact_digest: ArtifactDigest
    command_argv: tuple[str, ...]
    isolated_home_environment_keys: tuple[str, ...]
    session_trace_relative_path: str | None
    token_usage_report_relative_path: str
    environment: tuple[tuple[str, str], ...]
    timeout_seconds: float
    terminate_grace_seconds: float
    max_diagnostic_bytes: int
    agent_backend: Literal["claude", "codex", "qodercli", "pi"] = "qodercli"
    reasoning_effort: Literal["low", "medium", "high", "max"] = "max"
    session_settings: str = ""

    def __post_init__(self) -> None:
        for label, value in (("commit", self.bundle_commit), ("tree", self.bundle_tree)):
            if len(value) not in {40, 64} or any(
                character not in "0123456789abcdef" for character in value
            ):
                raise ValueError(f"Evolver Bundle {label} must be a full lowercase object ID")
        parse_artifact_digest(self.bundle_artifact_digest)
        if not self.command_argv:
            raise ValueError("Evolution command cannot be empty")
        if self.agent_backend not in {"claude", "codex", "qodercli", "pi"}:
            raise ValueError("Evolver agent backend is unsupported")
        if self.reasoning_effort not in {"low", "medium", "high", "max"}:
            raise ValueError("Evolver reasoning effort is unsupported")
        if "\x00" in self.session_settings:
            raise ValueError("Evolver session settings cannot contain NUL")
        executable = Path(self.command_argv[0])
        if not executable.is_absolute():
            raise ValueError("Evolution command executable must be absolute")
        if self.timeout_seconds <= 0 or self.terminate_grace_seconds <= 0:
            raise ValueError("Evolution process timeouts must be positive")
        if self.max_diagnostic_bytes <= 0:
            raise ValueError("Evolution diagnostic byte limit must be positive")
        keys = [key for key, _value in self.environment]
        if len(keys) != len(set(keys)):
            raise ValueError("Evolution environment contains duplicate keys")
        if len(self.isolated_home_environment_keys) != len(
            set(self.isolated_home_environment_keys)
        ):
            raise ValueError("Evolution home environment keys cannot repeat")
        validate_worker_environment(
            {key: "validated" for key in self.isolated_home_environment_keys}
        )
        reserved = {
            "ATREX_AGENT_BACKEND",
            "ATREX_DEV_SHELL_BACKENDS",
            "ATREX_AGENT_REASONING_EFFORT",
            "ATREX_AGENT_SESSION_SETTINGS",
            "ATREX_AGENT_MODEL",
            "ATREX_EVOLUTION_INPUT",
            "ATREX_EVOLUTION_INPUT_JSON",
            "ATREX_EVOLUTION_WORKSPACE",
            "ATREX_EVOLUTION_CANDIDATE",
            "ATREX_EVIDENCE_PROMPT_PATH",
            "ATREX_EVIDENCE_PROMPT",
            "ATREX_EVOLUTION_OUTPUT",
            "ATREX_SESSION_TIMEOUT_SECONDS",
            "ATREX_USAGE_BUDGET",
            "ATREX_USAGE_UNIT",
            "ATREX_TOKEN_USAGE_REPORT",
            *self.isolated_home_environment_keys,
        }
        overlap = reserved.intersection(keys)
        if overlap:
            raise ValueError(
                f"Evolution environment overrides Runtime-owned keys: {sorted(overlap)}"
            )
        validate_worker_environment(dict(self.environment))
        if self.session_trace_relative_path is not None:
            relative = PurePosixPath(self.session_trace_relative_path)
            if relative.is_absolute() or relative.as_posix() == "." or ".." in relative.parts:
                raise ValueError("Evolution session trace path must be a safe relative path")
        usage_relative = PurePosixPath(self.token_usage_report_relative_path)
        if (
            usage_relative.is_absolute()
            or usage_relative.as_posix() == "."
            or ".." in usage_relative.parts
            or usage_relative.parts[0] != "scratch"
        ):
            raise ValueError(
                "Evolution token usage report path must be under the scratch directory"
            )

    def agent_descriptor(self, model: str | None) -> EvolutionAgentDescriptorV3:
        """Return the immutable Bundle and stable launch identity stored in provenance."""
        return EvolutionAgentDescriptorV3(
            bundle_commit=self.bundle_commit,
            bundle_tree=self.bundle_tree,
            bundle_artifact_digest=self.bundle_artifact_digest,
            agent_backend=self.agent_backend,
            model=model,
            reasoning_effort=self.reasoning_effort,
            session_settings_sha256=hashlib.sha256(
                self.session_settings.encode("utf-8")
            ).hexdigest(),
            command_executable=self.command_argv[0],
            command_argv_sha256=hashlib.sha256(
                json.dumps(
                    self.command_argv,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest(),
            environment_keys=tuple(sorted(key for key, _value in self.environment)),
            isolated_home_environment_keys=tuple(sorted(self.isolated_home_environment_keys)),
        )


class SubprocessEvolutionSessionDriver:
    """Run a stdin-prompted Coding Agent CLI through the configured sandbox."""

    def __init__(self, launcher: WorkerLauncher, config: EvolutionProcessConfig) -> None:
        self._launcher = launcher
        self._config = config
        self._processes = BoundedProcessRunner(
            BoundedProcessConfig(
                timeout_seconds=config.timeout_seconds,
                terminate_grace_seconds=config.terminate_grace_seconds,
                max_output_bytes=config.max_diagnostic_bytes,
            )
        )

    async def run(self, prepared: PreparedEvolution) -> EvolutionSessionResult:
        """Run blocking process ownership in a worker thread."""
        return await anyio.to_thread.run_sync(self._run_sync, prepared)

    def _run_sync(self, prepared: PreparedEvolution) -> EvolutionSessionResult:
        launch = self.prepare_launch(prepared)
        command_argv = self._config.command_argv
        argv = self._launcher.wrap(
            command_argv,
            workspace=prepared.root,
            environment=launch.environment,
        )
        try:
            result = self._processes.run(
                argv,
                cwd=prepared.root,
                stdin=EVOLVER_LAUNCH_INSTRUCTION.encode(),
            )
        except TimeoutError as error:
            raise InfrastructureError("Evolver timed out") from error
        except OSError as error:
            raise InfrastructureError(f"Evolution process could not start: {error}") from error
        # The Evolver Bundle owns the inner Agent process and translates its
        # wall-time expiration into the conventional timeout exit status. Check
        # that terminal condition before validating usage: a killed Provider
        # cannot emit its final authoritative usage event, so the resulting
        # partial report is evidence of the timeout rather than its root cause.
        if result.returncode == EVOLVER_TIMEOUT_EXIT_STATUS:
            raise InfrastructureError("Evolver timed out")
        try:
            expected_unit: UsageUnit = (
                "credits" if self._config.agent_backend == "qodercli" else "provider_tokens"
            )
            token_usage = ProviderUsageReportV2.from_file(
                launch.token_usage_path,
                expected_unit=expected_unit,
                expected_budget=None,
            )
        except ValueError as error:
            raise InfrastructureError(
                f"Invalid Evolution provider usage report: {error}"
            ) from error
        return EvolutionSessionResult(
            result.returncode,
            result.stdout,
            result.stderr,
            self._config.agent_descriptor(prepared.model),
            launch.session_trace_path,
            token_usage,
        )

    def prepare_launch(self, prepared: PreparedEvolution) -> PreparedEvolutionLaunch:
        """Prepare Evolver-owned paths and its exact environment without starting the Agent."""
        environment = dict(self._config.environment)
        agent_home = prepared.root / "scratch/agent-home"
        agent_home.mkdir(mode=0o700)
        environment.update(
            {
                "ATREX_AGENT_BACKEND": self._config.agent_backend,
                "ATREX_AGENT_REASONING_EFFORT": self._config.reasoning_effort,
                "ATREX_AGENT_SESSION_SETTINGS": self._config.session_settings,
                "ATREX_AGENT_MODEL": prepared.model or "",
                "ATREX_EVOLUTION_INPUT_JSON": prepared.manifest_path.read_text(encoding="utf-8"),
                "ATREX_EVOLUTION_WORKSPACE": str(prepared.root),
                "ATREX_EVOLUTION_CANDIDATE": str(prepared.candidate_root),
                "ATREX_EVIDENCE_PROMPT": EVOLVER_EVIDENCE_PROMPT_TEXT,
                "ATREX_EVOLUTION_OUTPUT": str(prepared.output_path),
                "ATREX_SESSION_TIMEOUT_SECONDS": str(self._config.timeout_seconds),
            }
        )
        token_usage_path = prepared.root.joinpath(
            *PurePosixPath(self._config.token_usage_report_relative_path).parts
        )
        token_usage_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        environment["ATREX_TOKEN_USAGE_REPORT"] = str(token_usage_path)
        environment.update(
            (key, str(agent_home)) for key in self._config.isolated_home_environment_keys
        )
        validate_worker_environment(environment)
        session_trace_path = (
            None
            if self._config.session_trace_relative_path is None
            else prepared.root.joinpath(
                *PurePosixPath(self._config.session_trace_relative_path).parts
            )
        )
        return PreparedEvolutionLaunch(
            environment=environment,
            token_usage_path=token_usage_path,
            session_trace_path=session_trace_path,
        )

    def wrap_command(
        self,
        prepared: PreparedEvolution,
        launch: PreparedEvolutionLaunch,
        runtime_argv: tuple[str, ...],
    ) -> tuple[str, ...]:
        """Apply the same production Sandbox boundary to an Evolver dev shell."""
        environment = dict(launch.environment)
        environment["ATREX_DEV_SHELL_BACKENDS"] = "claude,codex,qodercli,pi"
        return self._launcher.wrap(
            runtime_argv,
            workspace=prepared.root,
            environment=environment,
            interactive=True,
        )


class EvolverBundleRunner(EvolverRunner):
    """Validate and seal Challengers produced by one versioned Evolver Bundle."""

    def __init__(
        self,
        workspaces: EvolutionWorkspaceAssembler,
        sessions: EvolutionSessionDriver,
        artifacts: LocalArtifactStore,
        events: RuntimeEventRecorder,
        *,
        kernel_agent_limits: KernelAgentBundleLimits,
        max_output_manifest_bytes: int,
        worker_sessions: WorkerSessionRecorder | None = None,
        backend: str | None = None,
        max_infrastructure_retries: int = 0,
    ) -> None:
        if max_output_manifest_bytes <= 0:
            raise ValueError("max_output_manifest_bytes must be positive")
        if max_infrastructure_retries < 0:
            raise ValueError("Evolver infrastructure retries cannot be negative")
        self._workspaces = workspaces
        self._sessions = sessions
        self._artifacts = artifacts
        self._events = events
        self._builder = KernelAgentRevisionBuilder(
            artifacts,
            limits=kernel_agent_limits,
        )
        self._kernel_agent_limits = kernel_agent_limits
        self._max_output_manifest_bytes = max_output_manifest_bytes
        self._worker_sessions = worker_sessions
        self._backend = backend
        self._max_infrastructure_retries = max_infrastructure_retries

    async def build_challenger(self, request: BuildChallengerRequest) -> BuildChallengerResult:
        """Retry transient Evolver infrastructure failures in fresh Sessions."""
        failures = 0
        while True:
            try:
                return await self._build_challenger_once(request)
            except InfrastructureError as error:
                failures += 1
                if failures > self._max_infrastructure_retries:
                    if self._max_infrastructure_retries:
                        error.add_note(
                            "Evolver infrastructure retry budget exhausted: "
                            f"failures={failures}, "
                            f"max_retries={self._max_infrastructure_retries}"
                        )
                    raise
                self._events.record_runtime_event(
                    "evolution.retrying",
                    request.epoch_id,
                    {
                        "parent_revision_id": request.parent_revision.id,
                        "dsl": request.parent_revision.dsl.value,
                        "retry_number": failures,
                        "max_infrastructure_retries": self._max_infrastructure_retries,
                        "error_type": type(error).__name__,
                        "error_message": str(error),
                    },
                )

    async def _build_challenger_once(
        self,
        request: BuildChallengerRequest,
    ) -> BuildChallengerResult:
        """Run, validate, and seal one complete Challenger proposal."""
        prepared = self._workspaces.prepare(request)
        worker_session_id = new_worker_session_id()
        if self._worker_sessions is not None:
            self._worker_sessions.start_worker_session(
                WorkerSession(
                    id=worker_session_id,
                    role=WorkerSessionRole.EVOLVER,
                    subject_id=str(request.epoch_id),
                    external_run_id=prepared.root.name,
                    workspace_path=str(prepared.root),
                    status=WorkerSessionStatus.RUNNING,
                    started_at=datetime.now(UTC).isoformat(),
                    epoch_id=request.epoch_id,
                    backend=self._backend,
                    model=request.model,
                )
            )
        event_base = {
            "worker_role": "evolver",
            "worker_run_id": prepared.root.name,
            "parent_revision_id": request.parent_revision.id,
            "dsl": request.parent_revision.dsl,
            "model": request.model,
        }
        self._events.record_runtime_event("worker.started", request.epoch_id, event_base)
        exit_kind = "failed"
        try:
            result = await self._sessions.run(prepared)
        except InfrastructureError as error:
            exit_kind = (
                "timeout"
                if str(error) == "Evolver timed out" or "wall-time limit" in str(error)
                else "infrastructure_failed"
            )
            failure_digest, retention_error_type = self._seal_failure_trace(
                prepared,
                error,
                phase="session",
                result=None,
            )
            if self._worker_sessions is not None:
                self._worker_sessions.finish_worker_session(
                    worker_session_id,
                    status=(
                        WorkerSessionStatus.TIMED_OUT
                        if exit_kind == "timeout"
                        else WorkerSessionStatus.FAILED
                    ),
                    finish_reason=exit_kind,
                    error_type=type(error).__name__,
                    error_message=str(error),
                )
            self._events.record_runtime_event(
                f"worker.{exit_kind}",
                request.epoch_id,
                {
                    **event_base,
                    "error_type": type(error).__name__,
                    "failure_artifact_digest": failure_digest,
                    "failure_retention_error_type": retention_error_type,
                },
            )
            raise
        except Exception as error:
            failure_digest, retention_error_type = self._seal_failure_trace(
                prepared,
                error,
                phase="session",
                result=None,
            )
            if self._worker_sessions is not None:
                self._worker_sessions.finish_worker_session(
                    worker_session_id,
                    status=WorkerSessionStatus.FAILED,
                    finish_reason="infrastructure-failed",
                    error_type=type(error).__name__,
                    error_message=str(error),
                )
            self._events.record_runtime_event(
                "worker.failed",
                request.epoch_id,
                {
                    **event_base,
                    "error_type": type(error).__name__,
                    "failure_artifact_digest": failure_digest,
                    "failure_retention_error_type": retention_error_type,
                },
            )
            raise
        else:
            if result.returncode != 0:
                exit_kind = "infrastructure_failed"
                diagnostic = result.stderr.strip() or result.stdout.strip() or "no diagnostics"
                process_error = InfrastructureError(
                    f"Evolution process exited with {result.returncode}: {diagnostic[:1000]}"
                )
                failure_digest, retention_error_type = self._seal_failure_trace(
                    prepared,
                    process_error,
                    phase="session",
                    result=result,
                )
                session_trace_digest = self._session_trace_from_failure(failure_digest)
                if self._worker_sessions is not None:
                    self._worker_sessions.finish_worker_session(
                        worker_session_id,
                        status=WorkerSessionStatus.FAILED,
                        finish_reason=f"process-exit-{result.returncode}",
                        trace_digest=session_trace_digest,
                        token_usage=result.token_usage.to_domain(),
                        process_returncode=result.returncode,
                        error_type=type(process_error).__name__,
                        error_message=str(process_error),
                    )
                self._events.record_runtime_event(
                    "worker.infrastructure_failed",
                    request.epoch_id,
                    {
                        **event_base,
                        "process_returncode": result.returncode,
                        "error_type": type(process_error).__name__,
                        "failure_artifact_digest": failure_digest,
                        "failure_retention_error_type": retention_error_type,
                    },
                )
                raise process_error
            exit_kind = "exited"
            self._events.record_runtime_event(
                "worker.exited",
                request.epoch_id,
                {
                    **event_base,
                    "process_returncode": result.returncode,
                    "usage_unit": result.token_usage.usage_unit,
                    "usage_budget": result.token_usage.budget,
                    "usage": result.token_usage.model_dump(mode="json"),
                },
            )
        finally:
            self._events.record_runtime_event(
                "worker.cleaned",
                request.epoch_id,
                {**event_base, "preceding_status": exit_kind},
            )
        try:
            sealed, session_trace_digest = self._seal_result(request, prepared, result)
        except Exception as error:
            failure_digest, retention_error_type = self._seal_failure_trace(
                prepared,
                error,
                phase="candidate_validation",
                result=result,
            )
            session_trace_digest = self._session_trace_from_failure(failure_digest)
            if self._worker_sessions is not None:
                self._worker_sessions.finish_worker_session(
                    worker_session_id,
                    status=WorkerSessionStatus.FAILED,
                    finish_reason="candidate-validation-failed",
                    trace_digest=session_trace_digest,
                    token_usage=result.token_usage.to_domain(),
                    process_returncode=result.returncode,
                    error_type=type(error).__name__,
                    error_message=str(error),
                )
            self._events.record_runtime_event(
                "evolution.candidate_rejected",
                request.epoch_id,
                {
                    "parent_revision_id": request.parent_revision.id,
                    "worker_run_id": prepared.root.name,
                    "error_type": type(error).__name__,
                    "failure_artifact_digest": failure_digest,
                    "failure_retention_error_type": retention_error_type,
                },
            )
            raise
        if self._worker_sessions is not None:
            self._worker_sessions.finish_worker_session(
                worker_session_id,
                status=WorkerSessionStatus.COMPLETED,
                finish_reason="completed",
                trace_digest=session_trace_digest,
                token_usage=result.token_usage.to_domain(),
                process_returncode=result.returncode,
            )
        return sealed

    def _session_trace_from_failure(
        self, failure_digest: ArtifactDigest | None
    ) -> ArtifactDigest | None:
        if failure_digest is None:
            return None
        try:
            stored = self._artifacts.verify(failure_digest)
            if stored.kind is not ArtifactKind.EVOLUTION:
                return None
            payload = json.loads((stored.payload_path / "value.json").read_bytes())
            process = payload.get("process") if isinstance(payload, dict) else None
            value = process.get("session_trace_digest") if isinstance(process, dict) else None
            return None if value is None else parse_artifact_digest(str(value))
        except Exception:
            return None

    def _seal_failure_trace(
        self,
        prepared: PreparedEvolution,
        error: Exception,
        *,
        phase: Literal["session", "candidate_validation"],
        result: EvolutionSessionResult | None,
    ) -> tuple[ArtifactDigest | None, str | None]:
        """Best-effort seal bounded failure evidence without masking the primary error."""
        try:
            process: EvolutionFailureProcessV2 | None = None
            if result is not None:
                session_trace_digest: ArtifactDigest | None = None
                session_trace_retention_error_type: str | None = None
                if result.session_trace_path is not None:
                    try:
                        enforce_session_trace_retention(result.session_trace_path)
                        session_trace_digest = self._artifacts.put_directory(
                            result.session_trace_path,
                            ArtifactKind.SESSION_LOG,
                        )
                    except Exception as retention_error:
                        session_trace_retention_error_type = type(retention_error).__name__
                process = EvolutionFailureProcessV2(
                    agent=result.agent,
                    returncode=result.returncode,
                    stdout=result.stdout,
                    stderr=result.stderr,
                    session_trace_digest=session_trace_digest,
                    session_trace_retention_error_type=session_trace_retention_error_type,
                    token_usage=result.token_usage,
                )
            trace = EvolutionFailureTraceV5(
                input=EvolutionInputManifestV11.model_validate_json(
                    prepared.manifest_path.read_bytes()
                ),
                phase=phase,
                error_type=type(error).__name__,
                process=process,
            )
            digest = self._artifacts.put_json(
                cast(JsonValue, trace.model_dump(mode="json")),
                ArtifactKind.EVOLUTION,
            )
        except Exception as retention_error:
            return None, type(retention_error).__name__
        return digest, None

    def _seal_result(
        self,
        request: BuildChallengerRequest,
        prepared: PreparedEvolution,
        result: EvolutionSessionResult,
    ) -> tuple[BuildChallengerResult, ArtifactDigest | None]:
        output = read_evolution_output(
            prepared.output_path,
            max_bytes=self._max_output_manifest_bytes,
        )
        manifest = EvolutionInputManifestV11.model_validate_json(
            prepared.manifest_path.read_bytes()
        )
        repository_by_revision = {
            item.revision_id: prepared.root / item.path for item in manifest.visible_agents
        }
        visible = {entry.revision.id: entry.revision for entry in request.agent_catalog}
        visible[request.parent_revision.id] = request.parent_revision
        current_challenger_prefix = f"epoch:{request.epoch_id}:challenger:"
        candidate_trace: EvolutionCandidateTraceV3 | None = None
        proposal: KernelAgentCandidateProposal | KernelAgentReuseProposal
        runtime_state_changed_paths: set[str] = set()
        if output.proposal_type == "reuse":
            reused = visible.get(output.kernel_agent_revision_id)
            if reused is None:
                raise ValueError("Evolution reuse names an Agent revision outside frozen Evidence")
            if reused.id == request.parent_revision.id:
                raise ValueError("Evolution reuse cannot select the current Active revision")
            if reused.creation_key.startswith(current_challenger_prefix):
                raise ValueError("Evolution reuse must select Lineage history, not this Epoch")
            if reused.dsl is not request.parent_revision.dsl:
                raise ValueError("Evolution reuse changed the lineage DSL")
            active_root = repository_by_revision[request.parent_revision.id]
            if self._changed_paths(
                active_root,
                prepared.candidate_root / "source",
            ) or self._changed_paths(
                prepared.candidate_runtime_state_base_root,
                prepared.candidate_root / "runtime-state",
            ):
                raise ValueError("Evolution reuse must leave the writable Candidate unchanged")
            proposal = KernelAgentReuseProposal("reuse", reused.id)
            changed_paths: set[str] = set()
            base_revision_id = reused.id
        else:
            base = visible.get(output.kernel_agent_revision_id)
            if base is None:
                raise ValueError("Evolution base names an Agent revision outside frozen Evidence")
            if output.proposal_type == "evolved" and base.id != request.parent_revision.id:
                raise ValueError("An evolved proposal must derive from the current Active revision")
            if (
                output.proposal_type == "evolve_from_history"
                and base.id == request.parent_revision.id
            ):
                raise ValueError("An evolve_from_history proposal must use a historical revision")
            if output.proposal_type == "evolve_from_history" and base.creation_key.startswith(
                current_challenger_prefix
            ):
                raise ValueError(
                    "An evolve_from_history proposal must select completed Lineage history"
                )
            if base.dsl is not request.parent_revision.dsl:
                raise ValueError("Evolution base changed the lineage DSL")
            for contributor_id in output.contributing_revision_ids:
                contributor = visible.get(contributor_id)
                if contributor is None:
                    raise ValueError(
                        "Evolution credits an Agent revision outside frozen Evidence"
                    )
                if contributor.dsl is not request.parent_revision.dsl:
                    raise ValueError("Evolution credits an Agent revision from another DSL")
                if contributor.creation_key.startswith(current_challenger_prefix):
                    raise ValueError(
                        "Evolution can only credit completed Lineage history"
                    )
            candidate = self._builder.build_candidate(
                prepared.candidate_root / "source",
                request.parent_revision.dsl,
            )
            base_root = repository_by_revision[base.id]
            changed_paths = self._changed_paths(
                base_root,
                prepared.candidate_root / "source",
            )
            runtime_state_changed_paths = self._changed_paths(
                prepared.candidate_runtime_state_base_root,
                prepared.candidate_root / "runtime-state",
            )
            if not changed_paths and not runtime_state_changed_paths:
                raise ValueError("Evolver produced no Agent source or runtime-state changes")
            if set(output.changed_paths) != changed_paths:
                raise ValueError("Evolution changed_paths disagrees with sealed Agent Source")
            # Every new Agent revision is a complete logical Bundle. Even when
            # Evolver changes Source only, retain the exact Candidate State that
            # was paired with that Source in the writable workspace.
            runtime_state_digest = self._seal_runtime_state(
                prepared.candidate_root / "runtime-state"
            )
            candidate = KernelAgentCandidate(
                dsl=candidate.dsl,
                optimizer_digest=candidate.optimizer_digest,
                runtime_state_digest=runtime_state_digest,
            )
            proposal = KernelAgentCandidateProposal(
                output.proposal_type,
                base.id,
                candidate,
            )
            candidate_trace = EvolutionCandidateTraceV3(
                optimizer_digest=candidate.optimizer_digest,
                runtime_state_digest=runtime_state_digest,
            )
            base_revision_id = base.id
        session_trace_digest = (
            None
            if result.session_trace_path is None
            else self._seal_session_trace(result.session_trace_path)
        )
        trace = EvolutionTraceV9(
            input=EvolutionInputManifestV11.model_validate_json(
                prepared.manifest_path.read_bytes()
            ),
            agent=result.agent,
            process_returncode=0,
            stdout=result.stdout,
            stderr=result.stderr,
            session_trace_digest=session_trace_digest,
            token_usage=result.token_usage,
            output=output,
            candidate=candidate_trace,
        )
        evolution_trace_digest = self._artifacts.put_json(
            cast(JsonValue, trace.model_dump(mode="json")),
            ArtifactKind.EVOLUTION,
        )
        self._events.record_runtime_event(
            "evolution.proposal_sealed",
            request.epoch_id,
            {
                "active_revision_id": request.parent_revision.id,
                "proposal_type": output.proposal_type,
                "base_revision_id": base_revision_id,
                "evolution_trace_digest": evolution_trace_digest,
                "changed_paths": sorted(changed_paths),
                "contributing_revision_ids": list(output.contributing_revision_ids),
                "runtime_state_changed": bool(runtime_state_changed_paths),
                "unimplemented_capabilities": [
                    item.model_dump(mode="json") for item in output.unimplemented_capabilities
                ],
                "session_trace_digest": session_trace_digest,
            },
        )
        return BuildChallengerResult(proposal, evolution_trace_digest), session_trace_digest

    def _seal_session_trace(self, path: Path) -> ArtifactDigest:
        enforce_session_trace_retention(path)
        return self._artifacts.put_directory(path, ArtifactKind.SESSION_LOG)

    def _seal_runtime_state(self, path: Path) -> ArtifactDigest:
        validate_reusable_agent_state_seed(
            path,
            max_files=self._kernel_agent_limits.max_bundle_files,
            max_bytes=self._kernel_agent_limits.max_bundle_bytes,
            require_complete=True,
        )
        return self._artifacts.put_directory(
            path,
            ArtifactKind.KERNEL_AGENT_RUNTIME_STATE,
        )

    @staticmethod
    def _changed_paths(parent: Path, candidate: Path) -> set[str]:
        def snapshot(root: Path) -> dict[str, str]:
            values: dict[str, str] = {}
            for path in root.rglob("*"):
                mode = path.lstat().st_mode
                relative_path = PurePosixPath(*path.relative_to(root).parts)
                if stat.S_ISDIR(mode):
                    continue
                if is_ignored_kernel_agent_path(relative_path, directory=False):
                    continue
                if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
                    raise ValueError("Evolution repository contains a link or special file")
                relative = relative_path.as_posix()
                digest = hashlib.sha256()
                with path.open("rb") as source:
                    while chunk := source.read(1024 * 1024):
                        digest.update(chunk)
                values[relative] = digest.hexdigest()
            return values

        before = snapshot(parent)
        after = snapshot(candidate)
        return {
            path for path in before.keys() | after.keys() if before.get(path) != after.get(path)
        }
