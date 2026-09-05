"""Trusted Git importer for complete Optimizer repository Base Revisions."""

from __future__ import annotations

import re
import tempfile
from collections.abc import Mapping
from configparser import ConfigParser
from configparser import Error as ConfigParserError
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal
from urllib.parse import unquote, urlparse

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..artifacts.local import ArtifactKind, LocalArtifactStore
from ..domain.ids import ArtifactDigest, parse_artifact_digest
from ..domain.models import Dsl
from ..git_import import SafeGitImporter
from ..ports import KernelAgentCandidate
from .revision import KernelAgentRevisionBuilder

_FULL_COMMIT = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")


class OptimizerSubmoduleProvenanceV1(BaseModel):
    """One deployment-approved submodule expanded into an Optimizer snapshot."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    path: str = Field(min_length=1)
    repository: str = Field(min_length=1)
    commit: str = Field(pattern=r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
    tree: str = Field(pattern=r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")


class OptimizerSourceProvenanceV1(BaseModel):
    """Immutable origin of one Git-imported Optimizer Base snapshot."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1] = 1
    source_type: Literal["git"] = "git"
    repository: str = Field(min_length=1)
    commit: str = Field(pattern=r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
    tree: str = Field(pattern=r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
    submodules: tuple[OptimizerSubmoduleProvenanceV1, ...] = ()
    optimizer_digest: ArtifactDigest

    @field_validator("optimizer_digest", mode="before")
    @classmethod
    def _validate_digest(cls, value: object) -> ArtifactDigest:
        if not isinstance(value, str):
            raise ValueError("optimizer_digest must be a string")
        return parse_artifact_digest(value)


@dataclass(frozen=True, slots=True)
class GitOptimizerBaseResult:
    """Sealed Base Candidate plus its separately sealed source provenance."""

    candidate: KernelAgentCandidate
    source_provenance_digest: ArtifactDigest


class GitOptimizerBaseLoader:
    """Seal one approved full commit from a complete local Git checkout."""

    def __init__(
        self,
        artifacts: LocalArtifactStore,
        builder: KernelAgentRevisionBuilder,
        *,
        repository: str,
        git_executable: str | Path,
        timeout_seconds: float,
        max_archive_bytes: int,
        allowed_submodules: Mapping[str, str] | None = None,
    ) -> None:
        if not repository.strip():
            raise ValueError("Optimizer Base repository cannot be empty")
        self._artifacts = artifacts
        self._builder = builder
        self._repository = repository
        self._checkout = self._local_checkout(repository)
        self._max_archive_bytes = max_archive_bytes
        self._importer = SafeGitImporter(
            git_executable,
            timeout_seconds=timeout_seconds,
            max_archive_bytes=max_archive_bytes,
            label="Optimizer",
        )
        self._allowed_submodules = self._validate_allowed_submodules(allowed_submodules or {})

    def build_candidate(self, dsl: Dsl, commit: str) -> GitOptimizerBaseResult:
        """Import an exact locally available commit without network access or code execution."""
        if _FULL_COMMIT.fullmatch(commit) is None:
            raise ValueError("Optimizer Base revision must be a full lowercase commit SHA")
        with tempfile.TemporaryDirectory(prefix="atrex-optimizer-git-") as temporary:
            root = Path(temporary)
            export = root / "export"
            archive_path = root / "source.tar"
            repository = self._checkout
            resolved = self._resolve_local_commit(
                repository,
                commit,
                label="Optimizer Base",
            )
            if resolved != commit:
                raise ValueError("Local Optimizer checkout resolved a different commit")
            tree = self._importer.object_id(
                self._importer.run(("-C", str(repository), "rev-parse", f"{commit}^{{tree}}"))
            )
            submodule_commits = self._validate_tree_entries(
                self._importer.run(("-C", str(repository), "ls-tree", "-rz", commit))
            )
            archive_bytes = self._importer.archive(repository, commit, archive_path)
            export.mkdir(mode=0o700)
            self._importer.extract(archive_path, export)
            declared_submodules = self._declared_submodules(export)
            if set(declared_submodules) != set(submodule_commits):
                raise ValueError("Git submodule declarations do not match tracked gitlinks")
            submodule_provenance: list[OptimizerSubmoduleProvenanceV1] = []
            for index, (path, submodule_commit) in enumerate(sorted(submodule_commits.items())):
                approved_repository = self._allowed_submodules[path]
                if declared_submodules[path] != approved_repository:
                    raise ValueError(f"Git submodule URL is not approved: {path}")
                submodule_repository = self._local_submodule_checkout(repository, path)
                submodule_archive = root / f"submodule-{index}.tar"
                resolved_submodule = self._resolve_local_commit(
                    submodule_repository,
                    submodule_commit,
                    label=f"Optimizer submodule {path}",
                )
                if resolved_submodule != submodule_commit:
                    raise ValueError(
                        f"Local Optimizer submodule resolved a different commit: {path}"
                    )
                submodule_tree = self._importer.object_id(
                    self._importer.run(
                        (
                            "-C",
                            str(submodule_repository),
                            "rev-parse",
                            f"{submodule_commit}^{{tree}}",
                        )
                    )
                )
                self._validate_plain_tree_entries(
                    self._importer.run(
                        (
                            "-C",
                            str(submodule_repository),
                            "ls-tree",
                            "-rz",
                            submodule_commit,
                        )
                    ),
                    label=f"Optimizer submodule {path}",
                )
                archive_bytes += self._importer.archive(
                    submodule_repository,
                    submodule_commit,
                    submodule_archive,
                )
                if archive_bytes > self._max_archive_bytes:
                    raise ValueError("Git Optimizer archives exceed total byte limit")
                submodule_destination = export.joinpath(*PurePosixPath(path).parts)
                submodule_destination.mkdir(parents=True, exist_ok=True, mode=0o700)
                self._importer.extract(submodule_archive, submodule_destination)
                submodule_provenance.append(
                    OptimizerSubmoduleProvenanceV1(
                        path=path,
                        repository=approved_repository,
                        commit=submodule_commit,
                        tree=submodule_tree,
                    )
                )
            candidate = self._builder.build_candidate(export, dsl)
            provenance = OptimizerSourceProvenanceV1(
                repository=self._repository,
                commit=commit,
                tree=tree,
                submodules=tuple(submodule_provenance),
                optimizer_digest=candidate.optimizer_digest,
            )
            provenance_digest = self._artifacts.put_json(
                provenance.model_dump(mode="json"),
                ArtifactKind.OPTIMIZER_SOURCE,
            )
            return GitOptimizerBaseResult(candidate, provenance_digest)

    @staticmethod
    def _local_checkout(repository: str) -> Path:
        parsed = urlparse(repository)
        if parsed.scheme:
            if parsed.scheme != "file" or parsed.netloc not in {"", "localhost"}:
                raise ValueError("Optimizer Base repository must be a local Git checkout")
            path = Path(unquote(parsed.path))
        else:
            path = Path(repository)
        if not path.is_absolute():
            raise ValueError("Optimizer Base repository must be a local Git checkout")
        try:
            resolved = path.resolve(strict=True)
        except OSError as error:
            raise ValueError("Optimizer Base local checkout does not exist") from error
        if not resolved.is_dir():
            raise ValueError("Optimizer Base local checkout must be a directory")
        return resolved

    @staticmethod
    def _local_submodule_checkout(repository: Path, path: str) -> Path:
        current = repository
        for part in PurePosixPath(path).parts:
            current /= part
            if current.is_symlink():
                raise ValueError(f"Optimizer submodule checkout cannot be a symlink: {path}")
        try:
            resolved = current.resolve(strict=True)
        except OSError as error:
            raise ValueError(
                f"Optimizer submodule checkout is not initialized: {path}; "
                "run git submodule update --init --recursive"
            ) from error
        if not resolved.is_dir() or not resolved.is_relative_to(repository):
            raise ValueError(f"Optimizer submodule checkout is unsafe: {path}")
        return resolved

    def _resolve_local_commit(self, repository: Path, commit: str, *, label: str) -> str:
        try:
            payload = self._importer.run(
                ("-C", str(repository), "rev-parse", "--verify", f"{commit}^{{commit}}")
            )
        except RuntimeError as error:
            raise ValueError(
                f"{label} local checkout does not contain pinned commit {commit}; "
                "initialize or update the checkout before starting Runtime"
            ) from error
        return self._importer.object_id(payload)

    @staticmethod
    def _validate_allowed_submodules(values: Mapping[str, str]) -> dict[str, str]:
        approved: dict[str, str] = {}
        for path, repository in values.items():
            relative = PurePosixPath(path)
            normalized = relative.as_posix()
            if (
                relative.is_absolute()
                or normalized in {"", "."}
                or ".." in relative.parts
                or normalized != path
                or ".git" in relative.parts
            ):
                raise ValueError(f"Approved Git submodule path is unsafe: {path}")
            if not repository.strip():
                raise ValueError(f"Approved Git submodule repository is empty: {path}")
            approved[path] = repository
        return approved

    def _validate_tree_entries(self, payload: bytes) -> dict[str, str]:
        submodules: dict[str, str] = {}
        for record in payload.split(b"\0"):
            if not record:
                continue
            try:
                metadata, raw_path = record.split(b"\t", 1)
                mode, kind, object_id = metadata.split(b" ", 2)
                path = raw_path.decode("utf-8")
            except (ValueError, UnicodeDecodeError) as error:
                raise ValueError("Git tree listing is malformed") from error
            relative = PurePosixPath(path)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Git tree contains an unsafe path")
            if mode == b"160000" and kind == b"commit":
                if path not in self._allowed_submodules:
                    raise ValueError(
                        f"Optimizer repository contains an unapproved submodule: {path}"
                    )
                commit = self._importer.object_id(object_id)
                if _FULL_COMMIT.fullmatch(commit) is None:
                    raise ValueError(f"Optimizer submodule has an invalid gitlink: {path}")
                submodules[path] = commit
                continue
            if mode == b"120000" or kind != b"blob":
                raise ValueError(
                    f"Optimizer repository contains a link or unresolved submodule: {path}"
                )
        return submodules

    @staticmethod
    def _validate_plain_tree_entries(payload: bytes, *, label: str) -> None:
        for record in payload.split(b"\0"):
            if not record:
                continue
            try:
                metadata, raw_path = record.split(b"\t", 1)
                mode, kind, _object_id = metadata.split(b" ", 2)
                path = raw_path.decode("utf-8")
            except (ValueError, UnicodeDecodeError) as error:
                raise ValueError("Git tree listing is malformed") from error
            relative = PurePosixPath(path)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Git tree contains an unsafe path")
            if mode in {b"120000", b"160000"} or kind != b"blob":
                raise ValueError(f"{label} contains a link or nested submodule: {path}")

    @staticmethod
    def _declared_submodules(export: Path) -> dict[str, str]:
        path = export / ".gitmodules"
        if not path.exists():
            return {}
        if path.is_symlink() or not path.is_file():
            raise ValueError("Git submodule declaration must be a regular file")
        parser = ConfigParser(interpolation=None, strict=True)
        try:
            with path.open(encoding="utf-8") as source:
                parser.read_file(source)
        except (ConfigParserError, UnicodeDecodeError) as error:
            raise ValueError("Git submodule declaration is malformed") from error
        declared: dict[str, str] = {}
        for section in parser.sections():
            if not section.startswith('submodule "') or not section.endswith('"'):
                continue
            if not parser.has_option(section, "path") or not parser.has_option(section, "url"):
                raise ValueError("Git submodule declaration requires path and URL")
            submodule_path = parser.get(section, "path")
            repository = parser.get(section, "url")
            if submodule_path in declared:
                raise ValueError(f"Git submodule path is declared more than once: {submodule_path}")
            declared[submodule_path] = repository
        return declared
