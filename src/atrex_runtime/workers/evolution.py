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
from typing import Annotated, Literal, Protocol, Self, cast
from uuid import uuid4

import anyio
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, field_validator, model_validator

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
    KernelAgentCatalogEntry,
    KernelAgentRevision,
    KernelCatalogEntry,
    WorkerSession,
    WorkerSessionRole,
    WorkerSessionStatus,
)
from ..filesystem import make_tree_owner_writable, make_tree_read_only
from ..gateway.result_metrics import gateway_result_sol_percent
from ..kernel_agents import KernelAgentBundleLimits, KernelAgentRevisionBuilder
from ..ports import (
    BuildChallengerRequest,
    BuildChallengerResult,
    EvolverRunner,
    KernelAgentCandidateProposal,
    KernelAgentReuseProposal,
    RuntimeEventRecorder,
    WorkerSessionRecorder,
)
from ..serialization import canonical_json_bytes
from .evidence_view import EVIDENCE_PROMPT_FILENAME, assemble_evolver_evidence_view
from .launcher import WorkerLauncher, validate_worker_environment
from .process import BoundedProcessConfig, BoundedProcessRunner
from .session_trace import enforce_session_trace_retention
from .token_usage import ProviderUsageReportV2, UsageUnit

EVOLUTION_INPUT_VERSION: Literal[4] = 4
EVOLUTION_OUTPUT_VERSION: Literal[3] = 3
EVOLUTION_TRACE_VERSION: Literal[7] = 7
EVOLUTION_FAILURE_VERSION: Literal[3] = 3
EVOLVER_LAUNCH_INSTRUCTION = "Run the versioned Evolver Bundle once."
CANDIDATE_BASE_RECORD_MAX_BYTES = 4096


class EvolutionPathsV1(BaseModel):
    """Fixed paths exposed to an Evolver process."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    parent: Literal["input/parent"] = "input/parent"
    agents: Literal["input/agents"] = "input/agents"
    evidence: Literal["input/evidence"] = "input/evidence"
    runtime_tools: Literal["runtime-tools"] = "runtime-tools"
    candidate: Literal["candidate"] = "candidate"
    scratch: Literal["scratch"] = "scratch"
    output: Literal["scratch/evolution-output.json"] = "scratch/evolution-output.json"


class VisibleAgentRevisionV1(BaseModel):
    """One read-only Agent design exposed to an Evolver invocation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    revision_id: KernelAgentRevisionId
    optimizer_digest: ArtifactDigest
    path: str = Field(pattern=r"^input/agents/agentrev_[0-9a-f]{32}$")
    parent: bool
    relationship: Literal["active", "current_epoch_challenger", "lineage_history"]
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
        if self.parent != (self.relationship == "active"):
            raise ValueError("visible Agent parent marker disagrees with its relationship")
        if (self.relationship == "current_epoch_challenger") != (
            self.challenger_ordinal is not None
        ):
            raise ValueError("only current-Epoch Challengers have a Challenger ordinal")
        return self


class EvolutionInputManifestV4(BaseModel):
    """Immutable parent, Agent pool, and lineage evidence for one Evolution session."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[4] = EVOLUTION_INPUT_VERSION
    parent_revision_id: KernelAgentRevisionId
    evidence_checkpoint: ArtifactDigest
    idempotency_key: str = Field(min_length=1, max_length=300)
    dsl: Dsl
    optimizer_digest: ArtifactDigest
    visible_agents: tuple[VisibleAgentRevisionV1, ...]
    paths: EvolutionPathsV1 = EvolutionPathsV1()

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


class CandidateBaseRecordV1(BaseModel):
    """Runtime-seeded or Runtime-tool-selected Candidate repository base."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1] = 1
    base_revision_id: KernelAgentRevisionId
    selection: Literal["active_seed", "candidate_reset"]

    @field_validator("base_revision_id", mode="before")
    @classmethod
    def _validate_base_id(cls, value: object) -> KernelAgentRevisionId:
        if not isinstance(value, str):
            raise ValueError("Candidate base_revision_id must be a string")
        return parse_kernel_agent_revision_id(value)

    def canonical_json_bytes(self) -> bytes:
        """Serialize the stable Candidate-base record."""
        return canonical_json_bytes(self.model_dump(mode="json"))


def read_candidate_base_record(path: Path) -> CandidateBaseRecordV1:
    """Read the bounded regular Candidate-base record after Evolver exit."""
    try:
        metadata = path.lstat()
    except FileNotFoundError as error:
        raise ValueError("Evolver Candidate base record is missing") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise ValueError("Evolver Candidate base record must be a regular file")
    if metadata.st_size > CANDIDATE_BASE_RECORD_MAX_BYTES:
        raise ValueError("Evolver Candidate base record exceeds byte limit")
    return CandidateBaseRecordV1.model_validate_json(path.read_bytes())


class _EvolutionOutputBaseV3(BaseModel):
    """Fields common to every Agent-authored Challenger proposal."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[3] = EVOLUTION_OUTPUT_VERSION
    hypothesis: str = Field(min_length=1, max_length=4000)
    expected_effect: str = Field(min_length=1, max_length=4000)


class EvolutionCandidateOutputV3(_EvolutionOutputBaseV3):
    """A new revision derived from the Active or a historical revision."""

    proposal_type: Literal["evolved", "evolve_from_history"]
    base_revision_id: KernelAgentRevisionId
    changed_paths: tuple[str, ...] = Field(min_length=1, max_length=512)

    @field_validator("base_revision_id", mode="before")
    @classmethod
    def _validate_base_id(cls, value: object) -> KernelAgentRevisionId:
        if not isinstance(value, str):
            raise ValueError("base_revision_id must be a string")
        return parse_kernel_agent_revision_id(value)

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
                raise ValueError("changed_paths must contain safe repository-relative paths")
            normalized.append(relative.as_posix())
        if len(set(normalized)) != len(normalized):
            raise ValueError("changed_paths cannot contain duplicates")
        return tuple(normalized)


class EvolutionReuseOutputV3(_EvolutionOutputBaseV3):
    """Selection of one existing historical revision without repository changes."""

    proposal_type: Literal["reuse"]
    candidate_revision_id: KernelAgentRevisionId

    @field_validator("candidate_revision_id", mode="before")
    @classmethod
    def _validate_candidate_id(cls, value: object) -> KernelAgentRevisionId:
        if not isinstance(value, str):
            raise ValueError("candidate_revision_id must be a string")
        return parse_kernel_agent_revision_id(value)


EvolutionOutputV3 = Annotated[
    EvolutionCandidateOutputV3 | EvolutionReuseOutputV3,
    Field(discriminator="proposal_type"),
]
_EVOLUTION_OUTPUT_ADAPTER: TypeAdapter[EvolutionOutputV3] = TypeAdapter(EvolutionOutputV3)


def read_evolution_output(path: Path, *, max_bytes: int) -> EvolutionOutputV3:
    """Parse a bounded regular output manifest after the Evolver is reaped."""
    try:
        path_stat = path.lstat()
    except FileNotFoundError as error:
        raise ValueError("Evolver did not produce ATREX_EVOLUTION_OUTPUT") from error
    if stat.S_ISLNK(path_stat.st_mode) or not stat.S_ISREG(path_stat.st_mode):
        raise ValueError("Evolution output manifest must be a regular file")
    if path_stat.st_size > max_bytes:
        raise ValueError("Evolution output manifest exceeds byte limit")
    return _EVOLUTION_OUTPUT_ADAPTER.validate_json(path.read_bytes())


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


class EvolutionCandidateTraceV2(BaseModel):
    """Sealed repository digest produced by one successful Evolution run."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    optimizer_digest: ArtifactDigest

    @field_validator("*", mode="before")
    @classmethod
    def _validate_digest(cls, value: object) -> ArtifactDigest:
        if not isinstance(value, str):
            raise ValueError("Evolution Candidate digest must be a string")
        return parse_artifact_digest(value)


class EvolutionTraceV7(BaseModel):
    """Immutable provenance for one successful Kernel Agent Evolution run."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[7] = EVOLUTION_TRACE_VERSION
    input: EvolutionInputManifestV4
    agent: EvolutionAgentDescriptorV3
    process_returncode: Literal[0]
    stdout: str
    stderr: str
    session_trace_digest: ArtifactDigest | None
    token_usage: ProviderUsageReportV2
    output: EvolutionCandidateOutputV3 | EvolutionReuseOutputV3
    candidate: EvolutionCandidateTraceV2 | None

    @field_validator("session_trace_digest", mode="before")
    @classmethod
    def _validate_optional_digest(cls, value: object) -> ArtifactDigest | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError("Evolution session trace digest must be a string")
        return parse_artifact_digest(value)


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


class EvolutionFailureTraceV3(BaseModel):
    """Immutable bounded evidence for an Evolution run that produced no Challenger."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[3] = EVOLUTION_FAILURE_VERSION
    status: Literal["failed"] = "failed"
    input: EvolutionInputManifestV4
    phase: Literal["session", "candidate_validation"]
    error_type: str = Field(min_length=1, max_length=200)
    process: EvolutionFailureProcessV2 | None


@dataclass(frozen=True, slots=True)
class PreparedEvolution:
    """One private, append-only Evolution process allocation."""

    root: Path
    manifest_path: Path
    candidate_root: Path
    output_path: Path
    parent_revision: KernelAgentRevision
    model: str | None = None


class EvolutionWorkspaceAssembler:
    """Materialize a parent Optimizer repository and seed one writable full copy."""

    def __init__(self, root: str | Path, artifacts: LocalArtifactStore) -> None:
        self._root = Path(root).resolve()
        self._artifacts = artifacts
        self._root.mkdir(parents=True, exist_ok=True, mode=0o700)

    def prepare(self, request: BuildChallengerRequest) -> PreparedEvolution:
        """Create a fresh workspace containing one complete writable repository Candidate."""
        run_root = self._root / request.parent_revision.id / f"run-{uuid4().hex}"
        run_root.mkdir(parents=True, mode=0o700)
        parent_root = run_root / "input/parent"
        agents_root = run_root / "input/agents"
        evidence_root = run_root / "input/evidence"
        runtime_tools_root = run_root / "runtime-tools"
        candidate_root = run_root / "candidate"
        scratch_root = run_root / "scratch"
        scratch_root.mkdir(mode=0o700)
        output_path = scratch_root / "evolution-output.json"

        revision = request.parent_revision
        self._artifacts.materialize(revision.optimizer_digest, parent_root)
        agents_root.mkdir(parents=True, mode=0o700)
        visible_by_id = {entry.revision.id: entry.revision for entry in request.agent_catalog}
        visible_by_id[revision.id] = revision
        visible_agents: list[VisibleAgentRevisionV1] = []
        challenger_prefix = f"epoch:{request.epoch_id}:challenger:"
        for visible in visible_by_id.values():
            relative = f"input/agents/{visible.id}"
            self._artifacts.materialize(visible.optimizer_digest, run_root / relative)
            challenger_ordinal: int | None = None
            relationship: Literal["active", "current_epoch_challenger", "lineage_history"] = (
                "lineage_history"
            )
            if visible.id == revision.id:
                relationship = "active"
            elif visible.creation_key.startswith(challenger_prefix):
                suffix = visible.creation_key.removeprefix(challenger_prefix)
                if suffix.isdigit() and int(suffix) > 0:
                    relationship = "current_epoch_challenger"
                    challenger_ordinal = int(suffix)
            visible_agents.append(
                VisibleAgentRevisionV1(
                    revision_id=visible.id,
                    optimizer_digest=visible.optimizer_digest,
                    path=relative,
                    parent=visible.id == revision.id,
                    relationship=relationship,
                    challenger_ordinal=challenger_ordinal,
                    parent_revision_id=visible.parent_id,
                    created_by=visible.created_by,
                )
            )
        evidence = self._artifacts.verify(request.evidence_checkpoint)
        if evidence.kind is not ArtifactKind.EVIDENCE:
            raise ValueError("Evolution Evidence checkpoint has the wrong Artifact kind")
        assemble_evolver_evidence_view(
            evidence_root,
            lineage_payload=evidence.payload_path,
            lineage_checkpoint=request.evidence_checkpoint,
            artifacts=self._artifacts,
        )
        self._materialize_runtime_tools(runtime_tools_root, request)
        shutil.copytree(parent_root, candidate_root)
        make_tree_owner_writable(candidate_root)
        make_tree_read_only(parent_root)
        make_tree_read_only(agents_root)

        manifest = EvolutionInputManifestV4(
            parent_revision_id=revision.id,
            evidence_checkpoint=request.evidence_checkpoint,
            idempotency_key=request.idempotency_key,
            dsl=revision.dsl,
            optimizer_digest=revision.optimizer_digest,
            visible_agents=tuple(visible_agents),
        )
        manifest_path = run_root / "evolution-input.json"
        manifest_path.write_bytes(manifest.canonical_json_bytes())
        os.chmod(manifest_path, 0o400)
        candidate_base_path = scratch_root / "candidate-base.json"
        candidate_base_path.write_bytes(
            CandidateBaseRecordV1(
                base_revision_id=revision.id,
                selection="active_seed",
            ).canonical_json_bytes()
        )
        os.chmod(candidate_base_path, 0o600)
        return PreparedEvolution(
            run_root,
            manifest_path,
            candidate_root,
            output_path,
            revision,
            request.model,
        )

    def _materialize_runtime_tools(
        self,
        destination: Path,
        request: BuildChallengerRequest,
    ) -> None:
        """Freeze Evolver inspection/Candidate tools and their exact lineage catalog."""
        destination.mkdir(mode=0o700)
        source = Path(__file__).with_name("evolver_tools.py")
        if source.is_symlink() or not source.is_file():
            raise RuntimeError("Runtime Evolver tools implementation is unavailable")
        shutil.copyfile(source, destination / "evolver_tools.py")
        agents: tuple[KernelAgentCatalogEntry | KernelAgentRevision, ...] = (
            request.agent_catalog or (request.parent_revision,)
        )
        catalog = {
            "schema_version": 1,
            "evidence_checkpoint": request.evidence_checkpoint,
            "agents": [
                self._agent_catalog_value(
                    entry,
                    active_revision_id=request.parent_revision.id,
                )
                for entry in agents
            ],
            "kernels": [self._kernel_catalog_value(entry) for entry in request.kernel_catalog],
        }
        (destination / "catalog.json").write_bytes(canonical_json_bytes(catalog))
        kernels_root = destination / "kernels"
        kernels_root.mkdir(mode=0o700)
        for entry in request.kernel_catalog:
            artifact = self._artifacts.verify(entry.revision.artifact_digest)
            if artifact.kind is not ArtifactKind.KERNEL:
                raise ValueError("Evolver Runtime Tools Kernel has the wrong Artifact kind")
            self._artifacts.materialize(
                entry.revision.artifact_digest,
                kernels_root / str(entry.revision.id),
            )
        make_tree_read_only(destination)

    @staticmethod
    def _agent_catalog_value(
        entry: KernelAgentCatalogEntry | KernelAgentRevision,
        *,
        active_revision_id: KernelAgentRevisionId,
    ) -> dict[str, JsonValue]:
        if isinstance(entry, KernelAgentCatalogEntry):
            revision = entry.revision
            return {
                "revision_id": revision.id,
                "version": f"agent-v{entry.revision_number}",
                "revision_number": entry.revision_number,
                "parent_revision_id": revision.parent_id,
                "parent_version": (
                    None
                    if entry.parent_revision_number is None
                    else f"agent-v{entry.parent_revision_number}"
                ),
                "introduced_epoch_number": entry.introduced_epoch_number,
                "disposition": entry.disposition,
                "active": entry.active,
                "created_by": revision.created_by,
                "created_at": revision.created_at,
                "optimizer_digest": revision.optimizer_digest,
                "repository_path": f"input/agents/{revision.id}",
            }
        return {
            "revision_id": entry.id,
            "version": None,
            "revision_number": None,
            "parent_revision_id": entry.parent_id,
            "parent_version": None,
            "introduced_epoch_number": None,
            "disposition": "unknown",
            "active": entry.id == active_revision_id,
            "created_by": entry.created_by,
            "created_at": entry.created_at,
            "optimizer_digest": entry.optimizer_digest,
            "repository_path": f"input/agents/{entry.id}",
        }

    def _kernel_catalog_value(self, entry: KernelCatalogEntry) -> dict[str, JsonValue]:
        revision = entry.revision
        return {
            "revision_id": revision.id,
            "version": f"v{entry.revision_number}",
            "revision_number": entry.revision_number,
            "parent_revision_id": revision.parent_id,
            "parent_version": (
                None if entry.parent_revision_number is None else f"v{entry.parent_revision_number}"
            ),
            "artifact_digest": revision.artifact_digest,
            "correct": revision.evaluation.correct,
            "latency_us": revision.evaluation.latency_us,
            "sol_percent": gateway_result_sol_percent(
                self._artifacts,
                revision.evaluation.gateway_result_digest,
            ),
            "gateway_result_digest": revision.evaluation.gateway_result_digest,
            "created_at": revision.created_at,
            "kernel_agent_revision_id": entry.kernel_agent_revision_id,
            "kernel_agent_version": f"agent-v{entry.kernel_agent_revision_number}",
            "epoch_number": entry.epoch_number,
            "attempt_id": entry.attempt_id,
            "branch": None if entry.branch is None else entry.branch.value,
            "challenger_ordinal": entry.challenger_ordinal,
            "trajectory_ordinal": entry.trajectory_ordinal,
            "attempt_ordinal": entry.attempt_ordinal,
            "accepted_as_branch_best": entry.accepted_as_branch_best,
            "improvement_over_parent_percent": entry.improvement_over_parent_percent,
            "artifact_path": f"runtime-tools/kernels/{revision.id}",
        }


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
            "ATREX_EVOLUTION_CANDIDATE",
            "ATREX_EVIDENCE_PROMPT_PATH",
            "ATREX_EVOLUTION_OUTPUT",
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
        except OSError as error:
            raise InfrastructureError(f"Evolution process could not start: {error}") from error
        except TimeoutError as error:
            raise InfrastructureError("Evolution process exceeded its wall-time limit") from error
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
                "ATREX_EVOLUTION_INPUT": str(prepared.manifest_path),
                "ATREX_EVOLUTION_CANDIDATE": str(prepared.candidate_root),
                "ATREX_EVIDENCE_PROMPT_PATH": str(
                    prepared.root / "input/evidence" / EVIDENCE_PROMPT_FILENAME
                ),
                "ATREX_EVOLUTION_OUTPUT": str(prepared.output_path),
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
    ) -> None:
        if max_output_manifest_bytes <= 0:
            raise ValueError("max_output_manifest_bytes must be positive")
        self._workspaces = workspaces
        self._sessions = sessions
        self._artifacts = artifacts
        self._events = events
        self._builder = KernelAgentRevisionBuilder(
            artifacts,
            limits=kernel_agent_limits,
        )
        self._max_output_manifest_bytes = max_output_manifest_bytes
        self._worker_sessions = worker_sessions
        self._backend = backend

    async def build_challenger(self, request: BuildChallengerRequest) -> BuildChallengerResult:
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
            exit_kind = "timeout" if "wall-time limit" in str(error) else "infrastructure_failed"
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
            trace = EvolutionFailureTraceV3(
                input=EvolutionInputManifestV4.model_validate_json(
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
        if result.returncode != 0:
            diagnostic = result.stderr.strip() or result.stdout.strip() or "no diagnostics"
            raise RuntimeError(
                f"Evolution process exited with {result.returncode}: {diagnostic[:1000]}"
            )
        output = read_evolution_output(
            prepared.output_path,
            max_bytes=self._max_output_manifest_bytes,
        )
        candidate_base = read_candidate_base_record(prepared.root / "scratch/candidate-base.json")
        visible = {entry.revision.id: entry.revision for entry in request.agent_catalog}
        visible[request.parent_revision.id] = request.parent_revision
        current_challenger_prefix = f"epoch:{request.epoch_id}:challenger:"
        candidate_trace: EvolutionCandidateTraceV2 | None = None
        proposal: KernelAgentCandidateProposal | KernelAgentReuseProposal
        if isinstance(output, EvolutionReuseOutputV3):
            if (
                candidate_base.base_revision_id != request.parent_revision.id
                or candidate_base.selection != "active_seed"
            ):
                raise ValueError("Evolution reuse changed the Candidate base")
            reused = visible.get(output.candidate_revision_id)
            if reused is None:
                raise ValueError("Evolution reuse names an Agent revision outside frozen Evidence")
            if reused.id == request.parent_revision.id:
                raise ValueError("Evolution reuse cannot select the current Active revision")
            if reused.creation_key.startswith(current_challenger_prefix):
                raise ValueError("Evolution reuse must select Lineage history, not this Epoch")
            if reused.dsl is not request.parent_revision.dsl:
                raise ValueError("Evolution reuse changed the lineage DSL")
            if self._changed_paths(prepared.root / "input/parent", prepared.candidate_root):
                raise ValueError("Evolution reuse must leave the writable Candidate unchanged")
            proposal = KernelAgentReuseProposal("reuse", reused.id)
            changed_paths: set[str] = set()
            base_revision_id = reused.id
        else:
            base = visible.get(output.base_revision_id)
            if base is None:
                raise ValueError("Evolution base names an Agent revision outside frozen Evidence")
            if output.proposal_type == "evolved" and base.id != request.parent_revision.id:
                raise ValueError("An evolved proposal must derive from the current Active revision")
            if output.proposal_type == "evolved" and (
                candidate_base.base_revision_id != base.id
                or candidate_base.selection != "active_seed"
            ):
                raise ValueError("An evolved proposal changed the Candidate base")
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
            if output.proposal_type == "evolve_from_history" and (
                candidate_base.base_revision_id != base.id
                or candidate_base.selection != "candidate_reset"
            ):
                raise ValueError(
                    "An evolve_from_history proposal did not use candidate-reset for its base"
                )
            if base.dsl is not request.parent_revision.dsl:
                raise ValueError("Evolution base changed the lineage DSL")
            candidate = self._builder.build_candidate(
                prepared.candidate_root,
                request.parent_revision.dsl,
            )
            self._builder.validate_challenger(base, candidate)
            base_root = (
                prepared.root / "input/parent"
                if base.id == request.parent_revision.id
                else prepared.root / "input/agents" / base.id
            )
            changed_paths = self._changed_paths(base_root, prepared.candidate_root)
            if not changed_paths:
                raise ValueError("Evolver produced no repository changes")
            if set(output.changed_paths) != changed_paths:
                raise ValueError("Evolution changed_paths disagrees with sealed content")
            proposal = KernelAgentCandidateProposal(
                output.proposal_type,
                base.id,
                candidate,
            )
            candidate_trace = EvolutionCandidateTraceV2(
                optimizer_digest=candidate.optimizer_digest,
            )
            base_revision_id = base.id
        session_trace_digest = (
            None
            if result.session_trace_path is None
            else self._seal_session_trace(result.session_trace_path)
        )
        trace = EvolutionTraceV7(
            input=EvolutionInputManifestV4.model_validate_json(prepared.manifest_path.read_bytes()),
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
                "session_trace_digest": session_trace_digest,
            },
        )
        return BuildChallengerResult(proposal, evolution_trace_digest), session_trace_digest

    def _seal_session_trace(self, path: Path) -> ArtifactDigest:
        enforce_session_trace_retention(path)
        return self._artifacts.put_directory(path, ArtifactKind.SESSION_LOG)

    @staticmethod
    def _changed_paths(parent: Path, candidate: Path) -> set[str]:
        def snapshot(root: Path) -> dict[str, str]:
            values: dict[str, str] = {}
            for path in root.rglob("*"):
                mode = path.lstat().st_mode
                if stat.S_ISDIR(mode):
                    continue
                if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
                    raise ValueError("Evolution repository contains a link or special file")
                relative = path.relative_to(root).as_posix()
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
