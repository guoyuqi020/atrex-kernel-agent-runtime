"""Validate and seal one full-repository Kernel Agent Bundle revision."""

from __future__ import annotations

import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

from ..artifacts.local import ArtifactKind, LocalArtifactStore
from ..domain.models import Dsl, KernelAgentRevision
from ..ports import KernelAgentCandidate

KERNEL_AGENT_BUNDLE_MANIFEST_VERSION: Literal[1] = 1
KERNEL_AGENT_BUNDLE_MANIFEST = "atrex-bundle.json"
KERNEL_AGENT_BUNDLE_FORMAT: Literal["atrex-kernel-agent-bundle-v1"] = "atrex-kernel-agent-bundle-v1"


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


class KernelAgentBundleManifestV1(BaseModel):
    """Strict Runtime entry manifest embedded in one complete Core repository."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1] = KERNEL_AGENT_BUNDLE_MANIFEST_VERSION
    bundle_format: Literal["atrex-kernel-agent-bundle-v1"] = KERNEL_AGENT_BUNDLE_FORMAT
    entrypoint: KernelAgentBundleEntrypointV1

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

        manifest = KernelAgentBundleManifestV1.from_file(root / KERNEL_AGENT_BUNDLE_MANIFEST)
        self._validate_tree(root)
        self._validate_entry_file(
            root,
            manifest.entrypoint.command,
            max_bytes=self._limits.max_entrypoint_bytes,
            label="Optimizer command",
        )
        digest = self._artifacts.put_directory(root, ArtifactKind.KERNEL_AGENT)
        return KernelAgentCandidate(dsl=dsl, optimizer_digest=digest)

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

    def _validate_tree(self, root: Path) -> None:
        files = 0
        total_bytes = 0
        pending = [root]
        while pending:
            directory = pending.pop()
            for entry in directory.iterdir():
                entry_stat = entry.lstat()
                if stat.S_ISLNK(entry_stat.st_mode):
                    raise ValueError("Optimizer repository cannot contain symbolic links")
                if stat.S_ISDIR(entry_stat.st_mode):
                    pending.append(entry)
                    continue
                if not stat.S_ISREG(entry_stat.st_mode):
                    raise ValueError("Optimizer repository can contain only regular files")
                files += 1
                total_bytes += entry_stat.st_size
                if files > self._limits.max_bundle_files:
                    raise ValueError("Optimizer repository exceeds file limit")
                if total_bytes > self._limits.max_bundle_bytes:
                    raise ValueError("Optimizer repository exceeds byte limit")

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
