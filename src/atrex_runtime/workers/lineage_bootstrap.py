"""Trusted workspace and process boundary for a Core lineage-baseline session."""

from __future__ import annotations

import os
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Self
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..artifacts.local import ArtifactKind, LocalArtifactStore
from ..domain.ids import (
    ArtifactDigest,
    AttemptId,
    KernelAgentRevisionId,
    parse_artifact_digest,
    parse_attempt_id,
    parse_kernel_agent_revision_id,
)
from ..domain.models import Dsl, TokenUsage
from ..filesystem import make_tree_owner_writable
from ..serialization import canonical_json_bytes
from .core import CoreOptimizerProcessConfig
from .core_phase import CorePhaseRunner, PreparedCorePhase
from .launcher import WorkerLauncher, validate_worker_environment

LINEAGE_BOOTSTRAP_MANIFEST_VERSION: Literal[1] = 1
LINEAGE_BOOTSTRAP_REPORT_VERSION: Literal[1] = 1
_RUNTIME_KEYS = {
    "ATREX_AGENT_BACKEND",
    "ATREX_AGENT_MODEL",
    "ATREX_AGENT_REASONING_EFFORT",
    "ATREX_AGENT_SESSION_SETTINGS",
    "ATREX_CORE_PHASE",
    "ATREX_GATEWAY_CAPABILITY",
    "ATREX_GATEWAY_PROXY_URL",
    "ATREX_LINEAGE_BOOTSTRAP_MANIFEST",
    "ATREX_LINEAGE_BOOTSTRAP_REPORT_PATH",
    "ATREX_OPTIMIZER_REPOSITORY",
    "ATREX_SESSION_TIMEOUT_SECONDS",
    "ATREX_SESSION_TRACE_PATH",
    "ATREX_USAGE_BUDGET",
    "ATREX_USAGE_UNIT",
    "ATREX_TOKEN_USAGE_REPORT",
    "ATREX_WIKI_CAPABILITY",
    "ATREX_WIKI_PROXY_URL",
}


class LineageBootstrapPathsV1(BaseModel):
    """Fixed filesystem contract visible to a framework-baseline session."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    input_kernel: Literal["input/kernel"] = "input/kernel"
    working_kernel: Literal["work/kernel"] = "work/kernel"
    agent_problem: Literal["input/agent-problem"] = "input/agent-problem"
    optimizer: Literal["agent/optimizer"] = "agent/optimizer"


class LineageBootstrapManifestV1(BaseModel):
    """Immutable input contract for one Core framework-baseline session."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    schema_version: Literal[1] = LINEAGE_BOOTSTRAP_MANIFEST_VERSION
    bootstrap_attempt_id: AttemptId
    kernel_agent_revision_id: KernelAgentRevisionId
    input_kernel_digest: ArtifactDigest
    optimizer_digest: ArtifactDigest
    evaluation_contract_digest: ArtifactDigest
    agent_problem_digest: ArtifactDigest
    dsl: Dsl
    operator: str = Field(min_length=1)
    hardware_target: str = Field(min_length=1)
    paths: LineageBootstrapPathsV1 = LineageBootstrapPathsV1()

    @field_validator("bootstrap_attempt_id", mode="before")
    @classmethod
    def _attempt_id(cls, value: object) -> AttemptId:
        if not isinstance(value, str):
            raise ValueError("bootstrap_attempt_id must be a string")
        return parse_attempt_id(value)

    @field_validator("kernel_agent_revision_id", mode="before")
    @classmethod
    def _agent_id(cls, value: object) -> KernelAgentRevisionId:
        if not isinstance(value, str):
            raise ValueError("kernel_agent_revision_id must be a string")
        return parse_kernel_agent_revision_id(value)

    @field_validator(
        "input_kernel_digest",
        "optimizer_digest",
        "evaluation_contract_digest",
        "agent_problem_digest",
        mode="before",
    )
    @classmethod
    def _digest(cls, value: object) -> ArtifactDigest:
        if not isinstance(value, str):
            raise ValueError("lineage bootstrap digest must be a string")
        return parse_artifact_digest(value)

    def canonical_json_bytes(self) -> bytes:
        return canonical_json_bytes(self.model_dump(mode="json"))


class LineageBootstrapReportV1(BaseModel):
    """Untrusted Core conclusion validated before Runtime lineage registration."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    schema_version: Literal[1] = LINEAGE_BOOTSTRAP_REPORT_VERSION
    bootstrap_attempt_id: AttemptId
    status: Literal["baseline_ready", "blocked"]
    approach: str = Field(min_length=1)
    change_summary: str = Field(min_length=1)
    correctness_evidence: str = Field(min_length=1)
    latency_us: float | None
    candidate_artifact_digest: ArtifactDigest | None
    gateway_result_digest: ArtifactDigest | None
    research_sources: tuple[str, ...]
    lessons: str = Field(min_length=1)
    next_directions: tuple[str, ...] = Field(max_length=3)
    blocker: str | None

    @field_validator("bootstrap_attempt_id", mode="before")
    @classmethod
    def _attempt_id(cls, value: object) -> AttemptId:
        if not isinstance(value, str):
            raise ValueError("bootstrap_attempt_id must be a string")
        return parse_attempt_id(value)

    @field_validator("candidate_artifact_digest", "gateway_result_digest", mode="before")
    @classmethod
    def _optional_digest(cls, value: object) -> ArtifactDigest | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError("lineage bootstrap result digest must be a string")
        return parse_artifact_digest(value)

    @field_validator("research_sources", "next_directions")
    @classmethod
    def _text_arrays(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not item.strip() for item in value):
            raise ValueError("lineage bootstrap report arrays cannot contain blank text")
        return value

    @model_validator(mode="after")
    def _status_fields(self) -> LineageBootstrapReportV1:
        results = (self.latency_us, self.candidate_artifact_digest, self.gateway_result_digest)
        if self.status == "baseline_ready":
            if (
                self.latency_us is None
                or self.latency_us <= 0
                or self.candidate_artifact_digest is None
                or self.gateway_result_digest is None
                or self.blocker is not None
            ):
                raise ValueError("ready lineage baseline requires only a complete result")
        elif any(value is not None for value in results) or not (
            self.blocker and self.blocker.strip()
        ):
            raise ValueError("blocked lineage baseline requires only a blocker")
        return self

    @classmethod
    def from_file(cls, path: Path, *, expected_attempt_id: AttemptId, max_bytes: int) -> Self:
        if path.is_symlink() or not path.is_file():
            raise ValueError("lineage bootstrap report must be a regular file")
        if path.stat().st_size > max_bytes:
            raise ValueError("lineage bootstrap report exceeds byte limit")
        report = cls.model_validate_json(path.read_bytes())
        if report.bootstrap_attempt_id != expected_attempt_id:
            raise ValueError("lineage bootstrap report belongs to a different session")
        return report


@dataclass(frozen=True, slots=True)
class PreparedLineageBootstrap:
    root: Path
    manifest_path: Path
    session_root: Path


class LineageBootstrapWorkspaceAssembler:
    """Materialize a fresh Core baseline workspace from immutable Artifacts."""

    def __init__(self, root: str | Path, artifacts: LocalArtifactStore) -> None:
        self._root = Path(root).resolve()
        self._artifacts = artifacts
        self._root.mkdir(parents=True, exist_ok=True, mode=0o700)

    def prepare(self, manifest: LineageBootstrapManifestV1) -> PreparedLineageBootstrap:
        root = self._root / str(manifest.bootstrap_attempt_id) / f"run-{uuid4().hex}"
        root.mkdir(parents=True, mode=0o700)
        artifacts = (
            (manifest.input_kernel_digest, ArtifactKind.KERNEL),
            (manifest.agent_problem_digest, ArtifactKind.AGENT_PROBLEM),
            (manifest.optimizer_digest, ArtifactKind.KERNEL_AGENT),
        )
        for digest, kind in artifacts:
            if self._artifacts.verify(digest).kind is not kind:
                raise ValueError(f"lineage bootstrap Artifact has wrong kind: {digest}")
        paths = manifest.paths
        self._artifacts.materialize(manifest.input_kernel_digest, root / paths.input_kernel)
        self._artifacts.materialize(manifest.agent_problem_digest, root / paths.agent_problem)
        self._artifacts.materialize(manifest.optimizer_digest, root / paths.optimizer)
        shutil.copytree(root / paths.input_kernel, root / paths.working_kernel)
        make_tree_owner_writable(root / paths.working_kernel)
        manifest_path = root / "lineage-bootstrap.json"
        manifest_path.write_bytes(manifest.canonical_json_bytes())
        os.chmod(manifest_path, 0o400)
        session_root = root / "sessions"
        session_root.mkdir(mode=0o700)
        (root / "scratch").mkdir(mode=0o700)
        return PreparedLineageBootstrap(root, manifest_path, session_root)


@dataclass(frozen=True, slots=True)
class LineageBootstrapSessionConfig:
    """Deployment environment and pre-issued authority for one baseline session."""

    environment: tuple[tuple[str, str], ...]
    gateway_endpoint: str
    gateway_capability: str
    model: str | None = None
    wiki_endpoint: str | None = None
    wiki_capability: str | None = None

    def __post_init__(self) -> None:
        if not self.gateway_endpoint.startswith(("http://", "https://")):
            raise ValueError("lineage bootstrap Gateway endpoint must use HTTP or HTTPS")
        if not self.gateway_capability:
            raise ValueError("lineage bootstrap Gateway capability cannot be empty")
        if (self.wiki_endpoint is None) != (self.wiki_capability is None):
            raise ValueError("lineage bootstrap Wiki endpoint and capability must be paired")
        environment = dict(self.environment)
        if len(environment) != len(self.environment):
            raise ValueError("lineage bootstrap environment contains duplicate keys")
        overlap = _RUNTIME_KEYS.intersection(environment)
        if overlap:
            raise ValueError(f"lineage bootstrap environment overrides Runtime keys: {overlap}")
        validate_worker_environment(environment)


@dataclass(frozen=True, slots=True)
class LineageBootstrapSessionResult:
    finish_reason: str
    final_response: str
    token_usage: TokenUsage
    token_budget: int
    report: LineageBootstrapReportV1 | None
    report_digest: ArtifactDigest | None
    report_error: str | None
    session_trace_digest: ArtifactDigest | None
    candidate_artifact_digest: ArtifactDigest | None


class CoreLineageBootstrapSessionDriver:
    """Execute the Core entrypoint in framework-baseline mode and seal its outputs."""

    def __init__(
        self,
        launcher: WorkerLauncher,
        config: CoreOptimizerProcessConfig,
        artifacts: LocalArtifactStore,
    ) -> None:
        self._config = config
        self._artifacts = artifacts
        self._phases = CorePhaseRunner(launcher, config, artifacts)

    def run(
        self, prepared: PreparedLineageBootstrap, config: LineageBootstrapSessionConfig
    ) -> LineageBootstrapSessionResult:
        phase = self._phases.prepare(prepared.root, prepared.session_root)
        report_path = prepared.root / "scratch/lineage-bootstrap-report.json"
        environment = self._environment(prepared, phase, report_path, config)
        result = self._phases.run(phase, environment, label="Core lineage bootstrap")
        manifest = LineageBootstrapManifestV1.model_validate_json(
            prepared.manifest_path.read_bytes()
        )
        report = None
        report_digest = None
        report_error = None
        candidate_digest = None
        if report_path.exists() or report_path.is_symlink():
            try:
                report = LineageBootstrapReportV1.from_file(
                    report_path,
                    expected_attempt_id=manifest.bootstrap_attempt_id,
                    max_bytes=self._config.max_attempt_report_bytes,
                )
            except ValueError as error:
                report_error = str(error)
            else:
                report_digest = self._artifacts.put_json(
                    report.model_dump(mode="json"), ArtifactKind.ATTEMPT_REPORT
                )
                if report.status == "baseline_ready":
                    candidate_digest = self._artifacts.put_directory(
                        prepared.root / manifest.paths.working_kernel,
                        ArtifactKind.KERNEL,
                    )
        return LineageBootstrapSessionResult(
            result.finish_reason,
            result.process.stdout,
            result.token_usage.to_domain(),
            int(result.token_usage.require_budget()),
            report,
            report_digest,
            report_error,
            result.session_trace_digest,
            candidate_digest,
        )

    def _environment(
        self,
        prepared: PreparedLineageBootstrap,
        phase: PreparedCorePhase,
        report_path: Path,
        config: LineageBootstrapSessionConfig,
    ) -> Mapping[str, str]:
        environment = dict(config.environment)
        environment.update(
            self._phases.runtime_environment(
                phase,
                phase="framework_baseline",
                model=config.model,
            )
        )
        environment.update(
            {
                "ATREX_LINEAGE_BOOTSTRAP_MANIFEST": str(prepared.manifest_path),
                "ATREX_LINEAGE_BOOTSTRAP_REPORT_PATH": str(report_path),
                "ATREX_GATEWAY_PROXY_URL": config.gateway_endpoint,
                "ATREX_GATEWAY_CAPABILITY": config.gateway_capability,
            }
        )
        if config.wiki_endpoint is not None and config.wiki_capability is not None:
            environment["ATREX_WIKI_PROXY_URL"] = config.wiki_endpoint
            environment["ATREX_WIKI_CAPABILITY"] = config.wiki_capability
        validate_worker_environment(environment)
        return environment
