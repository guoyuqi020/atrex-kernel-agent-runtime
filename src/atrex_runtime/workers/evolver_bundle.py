"""Resolve and verify one immutable local Evolver Bundle snapshot."""

from __future__ import annotations

import hashlib
import re
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

from ..artifacts.local import ArtifactKind, LocalArtifactStore
from ..domain.ids import ArtifactDigest
from ..git_import import SafeGitImporter

EVOLVER_BUNDLE_MANIFEST = "atrex-evolver-bundle.json"
_IGNORED_DIRECTORY_NAMES = {".git", ".mypy_cache", ".pytest_cache", ".ruff_cache", "__pycache__"}
_IGNORED_FILE_NAMES = {".coverage", ".DS_Store"}
_FULL_COMMIT = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")


def _safe_relative_file(value: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or path.as_posix() == "." or ".." in path.parts:
        raise ValueError("Evolver entrypoint must be a normalized Bundle-relative file")
    return path.as_posix()


class EvolverBundleEntrypointV1(BaseModel):
    """Single executable entry declared by the fixed Evolver Bundle."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    command: str

    @field_validator("command")
    @classmethod
    def _validate_command(cls, value: str) -> str:
        return _safe_relative_file(value)


class EvolverBundleManifestV1(BaseModel):
    """Strict manifest at the Evolver trust boundary."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    schema_version: Literal[1] = 1
    bundle_format: Literal["atrex-kernel-agent-evolver-bundle-v1"] = (
        "atrex-kernel-agent-evolver-bundle-v1"
    )
    entrypoint: EvolverBundleEntrypointV1

    @classmethod
    def from_file(cls, path: Path) -> Self:
        try:
            payload = path.read_bytes()
        except OSError as error:
            raise ValueError(f"Evolver Bundle manifest is unavailable: {path}") from error
        try:
            return cls.model_validate_json(payload)
        except ValidationError as error:
            raise ValueError(f"Evolver Bundle manifest is invalid: {path}: {error}") from error


@dataclass(frozen=True, slots=True)
class ResolvedEvolverBundle:
    """Verified Bundle identity and launch argv."""

    digest: str
    command_argv: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ResolvedGitEvolverBundle:
    """Commit-anchored, sealed Evolver Bundle ready for process launch."""

    commit: str
    tree: str
    digest: str
    artifact_digest: ArtifactDigest
    command_argv: tuple[str, ...]


class LocalEvolverBundleResolver:
    """Hash all behavior-bearing files and resolve the manifest-owned entrypoint."""

    def __init__(
        self,
        root: Path,
        *,
        expected_sha256: str,
        command_prefix: tuple[str, ...],
        max_files: int,
        max_bytes: int,
    ) -> None:
        if len(expected_sha256) != 64 or any(c not in "0123456789abcdef" for c in expected_sha256):
            raise ValueError("Evolver Bundle SHA-256 must be 64 lowercase hexadecimal characters")
        if not command_prefix or any("\x00" in value for value in command_prefix):
            raise ValueError("Evolver command prefix must be non-empty and contain no NUL")
        if max_files <= 0 or max_bytes <= 0:
            raise ValueError("Evolver Bundle limits must be positive")
        self._root = root
        self._expected_sha256 = expected_sha256
        self._command_prefix = command_prefix
        self._max_files = max_files
        self._max_bytes = max_bytes

    def resolve(self) -> ResolvedEvolverBundle:
        root = self._root
        try:
            root_stat = root.lstat()
        except FileNotFoundError as error:
            raise ValueError(f"Evolver Bundle root does not exist: {root}") from error
        if stat.S_ISLNK(root_stat.st_mode) or not stat.S_ISDIR(root_stat.st_mode):
            raise ValueError("Evolver Bundle root must be a real directory")

        manifest = EvolverBundleManifestV1.from_file(root / EVOLVER_BUNDLE_MANIFEST)
        digest = self._tree_digest(root)
        if digest != self._expected_sha256:
            raise ValueError(
                f"Evolver Bundle digest mismatch: expected {self._expected_sha256}, got {digest}"
            )
        command = root.joinpath(*PurePosixPath(manifest.entrypoint.command).parts)
        try:
            command_stat = command.lstat()
        except FileNotFoundError as error:
            raise ValueError("Evolver Bundle entrypoint does not exist") from error
        if stat.S_ISLNK(command_stat.st_mode) or not stat.S_ISREG(command_stat.st_mode):
            raise ValueError("Evolver Bundle entrypoint must be a regular file")
        return ResolvedEvolverBundle(digest, (*self._command_prefix, str(command)))

    def _tree_digest(self, root: Path) -> str:
        files: list[tuple[str, Path]] = []
        total_bytes = 0
        pending = [root]
        while pending:
            directory = pending.pop()
            for entry in directory.iterdir():
                if entry.name in _IGNORED_DIRECTORY_NAMES and entry.is_dir():
                    continue
                if entry.name in _IGNORED_FILE_NAMES or entry.suffix in {".pyc", ".pyo"}:
                    continue
                entry_stat = entry.lstat()
                if stat.S_ISLNK(entry_stat.st_mode):
                    raise ValueError("Evolver Bundle cannot contain symbolic links")
                if stat.S_ISDIR(entry_stat.st_mode):
                    pending.append(entry)
                    continue
                if not stat.S_ISREG(entry_stat.st_mode):
                    raise ValueError("Evolver Bundle can contain only regular files")
                relative = entry.relative_to(root).as_posix()
                files.append((relative, entry))
                total_bytes += entry_stat.st_size
                if len(files) > self._max_files:
                    raise ValueError("Evolver Bundle exceeds file limit")
                if total_bytes > self._max_bytes:
                    raise ValueError("Evolver Bundle exceeds byte limit")

        digest = hashlib.sha256()
        for relative, path in sorted(files):
            relative_bytes = relative.encode()
            payload = path.read_bytes()
            digest.update(len(relative_bytes).to_bytes(8, "big"))
            digest.update(relative_bytes)
            digest.update(len(payload).to_bytes(8, "big"))
            digest.update(payload)
        return digest.hexdigest()


class GitEvolverBundleResolver:
    """Fetch one exact Evolver commit and launch only its sealed archive snapshot."""

    def __init__(
        self,
        artifacts: LocalArtifactStore,
        *,
        repository: str,
        commit: str,
        git_executable: str | Path,
        fetch_timeout_seconds: float,
        max_archive_bytes: int,
        command_prefix: tuple[str, ...],
        max_files: int,
        max_bytes: int,
    ) -> None:
        if not repository.strip():
            raise ValueError("Evolver repository cannot be empty")
        if _FULL_COMMIT.fullmatch(commit) is None:
            raise ValueError("Evolver revision must be a full lowercase commit SHA")
        self._artifacts = artifacts
        self._repository = repository
        self._commit = commit
        self._importer = SafeGitImporter(
            git_executable,
            timeout_seconds=fetch_timeout_seconds,
            max_archive_bytes=max_archive_bytes,
            label="Evolver",
        )
        self._command_prefix = command_prefix
        self._max_files = max_files
        self._max_bytes = max_bytes

    def resolve(self) -> ResolvedGitEvolverBundle:
        """Import, validate, seal, and resolve one immutable Git snapshot."""
        with tempfile.TemporaryDirectory(prefix="atrex-evolver-git-") as temporary:
            root = Path(temporary)
            repository = root / "repository"
            export = root / "export"
            archive_path = root / "source.tar"
            self._importer.run(("init", "--bare", str(repository)))
            self._importer.run(("-C", str(repository), "remote", "add", "origin", self._repository))
            self._importer.fetch_commit(repository, "origin", self._commit)
            resolved = self._importer.object_id(
                self._importer.run(("-C", str(repository), "rev-parse", "FETCH_HEAD^{commit}"))
            )
            if resolved != self._commit:
                raise ValueError("Git fetch resolved a different Evolver commit")
            tree = self._importer.object_id(
                self._importer.run(("-C", str(repository), "rev-parse", "FETCH_HEAD^{tree}"))
            )
            self._validate_tree_entries(
                self._importer.run(("-C", str(repository), "ls-tree", "-rz", "-r", "FETCH_HEAD"))
            )
            self._importer.archive(repository, "FETCH_HEAD", archive_path)
            export.mkdir(mode=0o700)
            self._importer.extract(archive_path, export)
            digest = evolver_bundle_sha256(
                export,
                max_files=self._max_files,
                max_bytes=self._max_bytes,
            )
            artifact_digest = self._artifacts.put_directory(
                export,
                ArtifactKind.EVOLVER_BUNDLE,
            )

        stored = self._artifacts.verify(artifact_digest)
        if stored.kind is not ArtifactKind.EVOLVER_BUNDLE:
            raise ValueError("sealed Evolver Bundle has the wrong Artifact kind")
        resolved_bundle = LocalEvolverBundleResolver(
            stored.payload_path,
            expected_sha256=digest,
            command_prefix=self._command_prefix,
            max_files=self._max_files,
            max_bytes=self._max_bytes,
        ).resolve()
        return ResolvedGitEvolverBundle(
            commit=self._commit,
            tree=tree,
            digest=digest,
            artifact_digest=artifact_digest,
            command_argv=resolved_bundle.command_argv,
        )

    @staticmethod
    def _validate_tree_entries(payload: bytes) -> None:
        for record in payload.split(b"\0"):
            if not record:
                continue
            try:
                metadata, raw_path = record.split(b"\t", 1)
                mode, kind, _object_id = metadata.split(b" ", 2)
                path = raw_path.decode("utf-8")
            except (ValueError, UnicodeDecodeError) as error:
                raise ValueError("Git Evolver tree listing is malformed") from error
            relative = PurePosixPath(path)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Git Evolver tree contains an unsafe path")
            if mode in {b"120000", b"160000"} or kind != b"blob":
                raise ValueError(f"Evolver repository contains a link or submodule: {path}")


def evolver_bundle_sha256(
    root: Path,
    *,
    max_files: int = 1024,
    max_bytes: int = 8388608,
) -> str:
    """Validate a Bundle and calculate its canonical deployment digest."""
    resolver = LocalEvolverBundleResolver(
        root,
        expected_sha256="0" * 64,
        command_prefix=("/unused",),
        max_files=max_files,
        max_bytes=max_bytes,
    )
    digest = resolver._tree_digest(root)
    LocalEvolverBundleResolver(
        root,
        expected_sha256=digest,
        command_prefix=("/unused",),
        max_files=max_files,
        max_bytes=max_bytes,
    ).resolve()
    return digest
