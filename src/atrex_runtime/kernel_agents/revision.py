"""Validate and seal one full-repository Kernel Agent Bundle revision."""

from __future__ import annotations

import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

from ..artifacts.local import ArtifactKind, LocalArtifactStore
from ..domain.ids import ArtifactDigest
from ..domain.models import Dsl, KernelAgentRevision
from ..ports import KernelAgentCandidate

KERNEL_AGENT_BUNDLE_MANIFEST_VERSION: Literal[1] = 1
KERNEL_AGENT_BUNDLE_MANIFEST = "atrex-bundle.json"
KERNEL_AGENT_BUNDLE_FORMAT: Literal["atrex-kernel-agent-bundle-v1"] = "atrex-kernel-agent-bundle-v1"
KERNEL_AGENT_IGNORED_DIRECTORY_NAMES = frozenset(
    {".mypy_cache", ".pytest_cache", ".ruff_cache", "__pycache__"}
)
KERNEL_AGENT_IGNORED_FILE_NAMES = frozenset({".coverage", ".DS_Store"})
KERNEL_AGENT_IGNORED_FILE_SUFFIXES = frozenset({".pyc", ".pyo"})
KERNEL_AGENT_WORKFLOW_MAIN = "workflow/main.py"
KERNEL_AGENT_WORKFLOW_TEMPLATE_NAMES = frozenset(
    {"evolve_3.py", "isolated.py", "pool_3.py", "pool_retained_3.py", "retained.py"}
)
KERNEL_AGENT_WORKFLOW_TEMPLATES = Path(__file__).resolve().parents[1] / "workflow_templates"


def is_ignored_kernel_agent_path(relative: PurePosixPath, *, directory: bool) -> bool:
    """Return whether a generated path is excluded from Agent revisions and diffs."""
    directory_parts = relative.parts if directory else relative.parts[:-1]
    if any(part in KERNEL_AGENT_IGNORED_DIRECTORY_NAMES for part in directory_parts):
        return True
    if directory:
        return False
    return (
        relative.name in KERNEL_AGENT_IGNORED_FILE_NAMES
        or relative.suffix in KERNEL_AGENT_IGNORED_FILE_SUFFIXES
    )


def _safe_relative_file(value: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or path.as_posix() == "." or ".." in path.parts:
        raise ValueError("Optimizer entry paths must be normalized repository-relative files")
    return path.as_posix()


class KernelAgentBundleEntrypointV1(BaseModel):
    """Executable entry owned completely by the Core repository."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    command: str

    @field_validator("command")
    @classmethod
    def _validate_command(cls, value: str) -> str:
        return _safe_relative_file(value)


class KernelAgentBundleWorkflowV1(BaseModel):
    """Agent-owned Workflow program launched only through the trusted Runtime."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    command: str

    @field_validator("command", mode="before")
    @classmethod
    def _validate_command(cls, value: object) -> str:
        if not isinstance(value, str):
            raise ValueError("Workflow command must be a repository-relative path")
        normalized = _safe_relative_file(value)
        if PurePosixPath(normalized).parts[0] != "workflow":
            raise ValueError("Workflow command must be contained under workflow/")
        return normalized


class KernelAgentBundleManifestV1(BaseModel):
    """Strict Runtime entry manifest embedded in one complete Core repository."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1] = KERNEL_AGENT_BUNDLE_MANIFEST_VERSION
    bundle_format: Literal["atrex-kernel-agent-bundle-v1"] = KERNEL_AGENT_BUNDLE_FORMAT
    entrypoint: KernelAgentBundleEntrypointV1
    workflow: KernelAgentBundleWorkflowV1 | None = None

    @classmethod
    def from_file(cls, path: str | Path) -> Self:
        """Parse a Bundle manifest at a trusted import or Evolver output boundary."""
        manifest_path = Path(path)
        try:
            payload = manifest_path.read_bytes()
        except OSError as error:
            raise ValueError(
                f"Kernel Agent Bundle manifest is unavailable: {manifest_path}"
            ) from error
        try:
            return cls.model_validate_json(payload)
        except ValidationError as error:
            raise ValueError(
                f"Kernel Agent Bundle manifest is invalid: {manifest_path}: {error}"
            ) from error


@dataclass(frozen=True, slots=True)
class KernelAgentBundleLimits:
    """Deployment-owned limits for a complete Optimizer repository snapshot."""

    max_bundle_files: int
    max_bundle_bytes: int
    max_entrypoint_bytes: int

    def __post_init__(self) -> None:
        values = {
            "max_bundle_files": self.max_bundle_files,
            "max_bundle_bytes": self.max_bundle_bytes,
            "max_entrypoint_bytes": self.max_entrypoint_bytes,
        }
        invalid = sorted(name for name, value in values.items() if value <= 0)
        if invalid:
            raise ValueError(f"Optimizer repository limits must be positive: {invalid}")


class KernelAgentRevisionBuilder:
    """Validate a complete repository and seal it as one immutable Optimizer Artifact."""

    def __init__(self, artifacts: LocalArtifactStore, *, limits: KernelAgentBundleLimits) -> None:
        self._artifacts = artifacts
        self._limits = limits

    def build_candidate(self, source_root: str | Path, dsl: Dsl) -> KernelAgentCandidate:
        """Validate and seal a full-repository Optimizer candidate."""
        root = Path(source_root)
        try:
            root_stat = root.lstat()
        except FileNotFoundError as error:
            raise ValueError(f"Optimizer source does not exist: {root}") from error
        if stat.S_ISLNK(root_stat.st_mode) or not stat.S_ISDIR(root_stat.st_mode):
            raise ValueError(f"Optimizer source must be a real directory: {root}")
        if (root / ".git").exists() or (root / ".git").is_symlink():
            raise ValueError("Optimizer source cannot contain Git metadata")

        with tempfile.TemporaryDirectory(prefix="atrex-kernel-agent-") as temporary:
            normalized = Path(temporary) / "repository"
            normalized.mkdir(mode=0o700)
            self._copy_validated_tree(root, normalized)
            self._validate_no_workflow_alternatives(normalized)
            manifest = KernelAgentBundleManifestV1.from_file(
                normalized / KERNEL_AGENT_BUNDLE_MANIFEST
            )
            if (
                manifest.workflow is not None
                and manifest.workflow.command != KERNEL_AGENT_WORKFLOW_MAIN
            ):
                raise ValueError(
                    "New Optimizer Bundles must use workflow/main.py as their sole Workflow entry"
                )
            self._validate_entry_file(
                normalized,
                manifest.entrypoint.command,
                max_bytes=self._limits.max_entrypoint_bytes,
                label="Optimizer command",
            )
            if manifest.workflow is not None:
                self._validate_entry_file(
                    normalized,
                    manifest.workflow.command,
                    max_bytes=self._limits.max_entrypoint_bytes,
                    label="Agent Workflow command",
                )
            digest = self._artifacts.put_directory(normalized, ArtifactKind.KERNEL_AGENT)
        return KernelAgentCandidate(dsl=dsl, optimizer_digest=digest)

    def select_workflow(
        self,
        optimizer_digest: ArtifactDigest,
        dsl: Dsl,
        command: str,
    ) -> KernelAgentCandidate:
        """Derive a Bundle whose sole Workflow entry is materialized as ``main.py``.

        Controlled-arm templates are Runtime construction inputs, not Candidate Agent
        evidence.  The selected program replaces ``workflow/main.py`` and known
        alternative entry files are removed before sealing.  Supporting Workflow
        modules remain available, while Evolver sees only the program actually run.
        """
        selected_path = PurePosixPath(_safe_relative_file(command))
        if selected_path.parts[0] != "workflow":
            raise ValueError("Workflow command must be contained under workflow/")
        stored = self._artifacts.verify(optimizer_digest)
        if stored.kind is not ArtifactKind.KERNEL_AGENT:
            raise ValueError("Workflow can be selected only from a Kernel Agent Artifact")
        with tempfile.TemporaryDirectory(prefix="atrex-kernel-agent-workflow-") as temporary:
            root = Path(temporary) / "repository"
            self._artifacts.materialize(optimizer_digest, root)
            selected_source = root.joinpath(*selected_path.parts)
            if (
                not selected_source.is_file()
                and selected_path.name in KERNEL_AGENT_WORKFLOW_TEMPLATE_NAMES
            ):
                selected_source = KERNEL_AGENT_WORKFLOW_TEMPLATES / selected_path.name
            self._validate_entry_file(
                selected_source.parent,
                selected_source.name,
                max_bytes=self._limits.max_entrypoint_bytes,
                label="Selected Agent Workflow command",
            )
            selected_bytes = selected_source.read_bytes()
            workflow_main = root / KERNEL_AGENT_WORKFLOW_MAIN
            workflow_main.parent.mkdir(mode=0o700, exist_ok=True)
            if workflow_main.exists():
                workflow_main.chmod(0o600)
            workflow_main.write_bytes(selected_bytes)
            workflow_main.chmod(0o600)
            selected_in_bundle = root.joinpath(*selected_path.parts)
            if (
                selected_path.as_posix() != KERNEL_AGENT_WORKFLOW_MAIN
                and selected_in_bundle.is_file()
            ):
                selected_in_bundle.unlink()
            self._remove_workflow_alternatives(root)
            manifest_path = root / KERNEL_AGENT_BUNDLE_MANIFEST
            manifest = KernelAgentBundleManifestV1.from_file(manifest_path)
            updated = manifest.model_copy(
                update={"workflow": KernelAgentBundleWorkflowV1(command=KERNEL_AGENT_WORKFLOW_MAIN)}
            )
            manifest_path.chmod(0o600)
            manifest_path.write_text(
                updated.model_dump_json(indent=2) + "\n",
                encoding="utf-8",
            )
            return self.build_candidate(root, dsl)

    @staticmethod
    def _remove_workflow_alternatives(root: Path) -> None:
        workflow = root / "workflow"
        if not workflow.is_dir():
            return
        for name in KERNEL_AGENT_WORKFLOW_TEMPLATE_NAMES:
            path = workflow / name
            if path.is_file():
                path.unlink()

    @staticmethod
    def _validate_no_workflow_alternatives(root: Path) -> None:
        workflow = root / "workflow"
        alternatives = sorted(
            name for name in KERNEL_AGENT_WORKFLOW_TEMPLATE_NAMES if (workflow / name).is_file()
        )
        if alternatives:
            raise ValueError(
                "Optimizer Bundle must expose only workflow/main.py; "
                f"alternative Workflow entries are not allowed: {alternatives}"
            )

    @staticmethod
    def validate_challenger(
        parent: KernelAgentRevision,
        candidate: KernelAgentCandidate,
    ) -> None:
        """Require a same-DSL, content-changing full-repository proposal."""
        if candidate.dsl is not parent.dsl:
            raise ValueError("Challenger cannot change its lineage DSL")
        if candidate.optimizer_digest == parent.optimizer_digest:
            raise ValueError("Evolver produced no Optimizer repository changes")

    def _copy_validated_tree(self, root: Path, destination_root: Path) -> None:
        files = 0
        total_bytes = 0
        pending = [(root, destination_root)]
        while pending:
            directory, destination = pending.pop()
            for entry in directory.iterdir():
                entry_stat = entry.lstat()
                relative = PurePosixPath(*entry.relative_to(root).parts)
                if stat.S_ISLNK(entry_stat.st_mode):
                    raise ValueError("Optimizer repository cannot contain symbolic links")
                if stat.S_ISDIR(entry_stat.st_mode):
                    if is_ignored_kernel_agent_path(relative, directory=True):
                        continue
                    child_destination = destination / entry.name
                    child_destination.mkdir(mode=0o700)
                    pending.append((entry, child_destination))
                    continue
                if not stat.S_ISREG(entry_stat.st_mode):
                    raise ValueError("Optimizer repository can contain only regular files")
                if is_ignored_kernel_agent_path(relative, directory=False):
                    continue
                files += 1
                total_bytes += entry_stat.st_size
                if files > self._limits.max_bundle_files:
                    raise ValueError("Optimizer repository exceeds file limit")
                if total_bytes > self._limits.max_bundle_bytes:
                    raise ValueError("Optimizer repository exceeds byte limit")
                target = destination / entry.name
                with entry.open("rb") as source, target.open("xb") as output:
                    while chunk := source.read(1024 * 1024):
                        output.write(chunk)
                target.chmod(0o600)

    @staticmethod
    def _validate_entry_file(
        root: Path,
        relative: str,
        *,
        max_bytes: int,
        label: str,
    ) -> None:
        path = root.joinpath(*PurePosixPath(relative).parts)
        try:
            path_stat = path.lstat()
        except FileNotFoundError as error:
            raise ValueError(f"{label} file does not exist: {relative}") from error
        if stat.S_ISLNK(path_stat.st_mode) or not stat.S_ISREG(path_stat.st_mode):
            raise ValueError(f"{label} file must be regular: {relative}")
        if path_stat.st_size > max_bytes:
            raise ValueError(f"{label} file exceeds byte limit: {relative}")
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as error:
            raise ValueError(f"{label} file must be UTF-8: {relative}") from error
        if not text.strip():
            raise ValueError(f"{label} file cannot be empty: {relative}")
