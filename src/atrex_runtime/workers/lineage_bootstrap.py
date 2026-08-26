"""Trusted workspace and process boundary for a Core lineage-baseline session."""

from __future__ import annotations

import fcntl
import os
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..artifacts.local import ArtifactKind, LocalArtifactStore
from ..domain.ids import (
    ArtifactDigest,
    AttemptId,
    KernelAgentRevisionId,
    LineageId,
    parse_artifact_digest,
    parse_attempt_id,
    parse_kernel_agent_revision_id,
    parse_lineage_id,
)
from ..domain.models import Dsl, TokenUsage
from ..filesystem import make_tree_owner_writable
from ..serialization import canonical_json_bytes
from .attempt_report import AttemptReportV12
from .core import CoreOptimizerProcessConfig
from .core_phase import CorePhaseRunner, PreparedCorePhase
from .launcher import WorkerLauncher, validate_worker_environment
from .workspace import TOOLS_README, _copy_reusable_tree, _replace_reusable_tree

LINEAGE_BOOTSTRAP_MANIFEST_VERSION: Literal[2] = 2
_RUNTIME_KEYS = {
    "ATREX_AGENT_BACKEND",
    "ATREX_AGENT_MODEL",
    "ATREX_AGENT_REASONING_EFFORT",
    "ATREX_AGENT_SESSION_SETTINGS",
    "ATREX_CORE_PHASE",
    "ATREX_GATEWAY_CAPABILITY",
    "ATREX_GATEWAY_PROXY_URL",
    "ATREX_LINEAGE_BOOTSTRAP_MANIFEST",
    "ATREX_ATTEMPT_REPORT_PATH",
    "ATREX_OPTIMIZER_REPOSITORY",
    "ATREX_SESSION_TIMEOUT_SECONDS",
    "ATREX_SESSION_TRACE_PATH",
    "ATREX_USAGE_BUDGET",
    "ATREX_USAGE_UNIT",
    "ATREX_TOKEN_USAGE_REPORT",
    "ATREX_WIKI_CAPABILITY",
    "ATREX_WIKI_PROXY_URL",
}


class LineageBootstrapPathsV2(BaseModel):
    """Fixed filesystem contract visible to a framework-baseline session."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    input_kernel: Literal["input/kernel"] = "input/kernel"
    working_kernel: Literal["work/kernel"] = "work/kernel"
    agent_problem: Literal[".runtime/agent-problem.json"] = ".runtime/agent-problem.json"
    optimizer: Literal["agent/optimizer"] = "agent/optimizer"
    reference: Literal["reference"] = "reference"


class LineageBootstrapManifestV2(BaseModel):
    """Immutable input contract for one Core framework-baseline session."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    schema_version: Literal[2] = LINEAGE_BOOTSTRAP_MANIFEST_VERSION
    bootstrap_attempt_id: AttemptId
    lineage_id: LineageId
    kernel_agent_revision_id: KernelAgentRevisionId
    input_kernel_digest: ArtifactDigest
    optimizer_digest: ArtifactDigest
    evaluation_contract_digest: ArtifactDigest
    agent_problem_digest: ArtifactDigest
    dsl: Dsl
    operator: str = Field(min_length=1)
    hardware_target: str = Field(min_length=1)
    paths: LineageBootstrapPathsV2 = LineageBootstrapPathsV2()

    @field_validator("bootstrap_attempt_id", mode="before")
    @classmethod
    def _attempt_id(cls, value: object) -> AttemptId:
        if not isinstance(value, str):
            raise ValueError("bootstrap_attempt_id must be a string")
        return parse_attempt_id(value)

    @field_validator("lineage_id", mode="before")
    @classmethod
    def _lineage_id(cls, value: object) -> LineageId:
        if not isinstance(value, str):
            raise ValueError("lineage_id must be a string")
        return parse_lineage_id(value)

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


@dataclass(frozen=True, slots=True)
class PreparedLineageBootstrap:
    root: Path
    manifest_path: Path
    session_root: Path
    persistent_skills_root: Path | None = None
    persistent_tools_root: Path | None = None
    persistent_lock_path: Path | None = None

    def persist_reusable_directories(self) -> None:
        """Publish reusable Bootstrap state as the seed for later trajectories."""
        roots = (
            (self.root / "skills", self.persistent_skills_root),
            (self.root / "tools", self.persistent_tools_root),
        )
        if self.persistent_lock_path is None or any(target is None for _source, target in roots):
            return
        self.persistent_lock_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with self.persistent_lock_path.open("a+b") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            for source, raw_target in roots:
                assert raw_target is not None
                _replace_reusable_tree(
                    source,
                    raw_target,
                    ensure_tools_readme=source.name == "tools",
                )


class LineageBootstrapWorkspaceAssembler:
    """Materialize a fresh Core baseline workspace from immutable Artifacts."""

    def __init__(
        self,
        root: str | Path,
        artifacts: LocalArtifactStore,
        *,
        attempt_workspaces_root: str | Path | None = None,
    ) -> None:
        self._root = Path(root).resolve()
        self._artifacts = artifacts
        self._reusable_root = (
            None
            if attempt_workspaces_root is None
            else Path(attempt_workspaces_root).resolve() / ".reusable"
        )
        self._root.mkdir(parents=True, exist_ok=True, mode=0o700)

    def prepare(self, manifest: LineageBootstrapManifestV2) -> PreparedLineageBootstrap:
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
        self._artifacts.materialize_file(
            manifest.agent_problem_digest,
            "value.json",
            root / paths.agent_problem,
        )
        self._artifacts.materialize(manifest.optimizer_digest, root / paths.optimizer)
        shutil.copytree(root / paths.input_kernel, root / paths.working_kernel)
        make_tree_owner_writable(root / paths.working_kernel)
        manifest_path = root / ".runtime/lineage-bootstrap.json"
        manifest_path.write_bytes(manifest.canonical_json_bytes())
        os.chmod(manifest_path, 0o400)
        session_root = root / "sessions"
        session_root.mkdir(mode=0o700)
        (root / "scratch").mkdir(mode=0o700)
        # Stays empty on the host; the Sandbox binds the pinned reference tree over it.
        (root / paths.reference).mkdir(mode=0o500)
        persistent_skills = None
        persistent_tools = None
        persistent_lock = None
        if self._reusable_root is not None:
            scope = (
                self._reusable_root
                / str(manifest.lineage_id)
                / str(manifest.kernel_agent_revision_id)
                / "bootstrap"
            )
            persistent_lock = self._reusable_root / ".lock"
            persistent_lock.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            with persistent_lock.open("a+b") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                persistent_skills = scope / "skills"
                persistent_tools = scope / "tools"
                persistent_skills.mkdir(parents=True, exist_ok=True, mode=0o700)
                persistent_tools.mkdir(parents=True, exist_ok=True, mode=0o700)
                readme = persistent_tools / "README.md"
                if not readme.exists():
                    readme.write_text(TOOLS_README, encoding="utf-8")
                _copy_reusable_tree(persistent_skills, root / "skills")
                _copy_reusable_tree(persistent_tools, root / "tools")
        else:
            (root / "skills").mkdir(mode=0o700)
            (root / "tools").mkdir(mode=0o700)
            (root / "tools/README.md").write_text(TOOLS_README, encoding="utf-8")
        return PreparedLineageBootstrap(
            root,
            manifest_path,
            session_root,
            persistent_skills,
            persistent_tools,
            persistent_lock,
        )


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
    report: AttemptReportV12 | None
    report_digest: ArtifactDigest | None
    report_error: str | None
    session_trace_digest: ArtifactDigest | None
    kernel_artifact_digest: ArtifactDigest | None


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
        try:
            return self._run_once(prepared, config)
        finally:
            prepared.persist_reusable_directories()

    def _run_once(
        self, prepared: PreparedLineageBootstrap, config: LineageBootstrapSessionConfig
    ) -> LineageBootstrapSessionResult:
        phase = self._phases.prepare(prepared.root, prepared.session_root)
        report_path = prepared.root / "scratch/attempt-report.json"
        environment = self._environment(prepared, phase, report_path, config)
        result = self._phases.run(phase, environment, label="Core lineage bootstrap")
        manifest = LineageBootstrapManifestV2.model_validate_json(
            prepared.manifest_path.read_bytes()
        )
        report = None
        report_digest = None
        report_error = None
        candidate_digest = None
        if report_path.exists() or report_path.is_symlink():
            try:
                report = AttemptReportV12.from_file(
                    report_path,
                    expected_attempt_id=manifest.bootstrap_attempt_id,
                    max_bytes=self._config.max_attempt_report_bytes,
                )
                baseline_count = sum(
                    experiment.action == "baseline" for experiment in report.experiments
                )
                if baseline_count > 1:
                    raise ValueError(
                        "Bootstrap Attempt report may contain only one baseline Experiment"
                    )
                if report.status == "candidate_ready" and baseline_count != 1:
                    raise ValueError(
                        "Bootstrap candidate_ready report requires exactly one baseline Experiment"
                    )
                has_identity_bearing_experiment = any(
                    subject is not None
                    for experiment in report.experiments
                    for subject in (experiment.before, experiment.after)
                )
                if (
                    report.status == "blocked"
                    and baseline_count == 0
                    and has_identity_bearing_experiment
                ):
                    raise ValueError(
                        "Bootstrap blocked report may omit baseline only when no Experiment has "
                        "identity-bearing Gateway evidence"
                    )
            except ValueError as error:
                report_error = str(error)
            else:
                report_digest = self._artifacts.put_json(
                    report.model_dump(mode="json"), ArtifactKind.ATTEMPT_REPORT
                )
                if report.status == "candidate_ready":
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
                "ATREX_ATTEMPT_REPORT_PATH": str(report_path),
                "ATREX_GATEWAY_PROXY_URL": config.gateway_endpoint,
                "ATREX_GATEWAY_CAPABILITY": config.gateway_capability,
            }
        )
        if config.wiki_endpoint is not None and config.wiki_capability is not None:
            environment["ATREX_WIKI_PROXY_URL"] = config.wiki_endpoint
            environment["ATREX_WIKI_CAPABILITY"] = config.wiki_capability
        validate_worker_environment(environment)
        return environment
