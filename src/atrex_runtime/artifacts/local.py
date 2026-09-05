"""Local filesystem content-addressed Artifact Store provider."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
import time
from collections.abc import Callable, Collection
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path, PurePosixPath

from ..domain.ids import ArtifactDigest, parse_artifact_digest
from ..filesystem import make_tree_owner_writable, make_tree_read_only
from ..serialization import canonical_json_bytes

type JsonScalar = bool | int | float | str | None
type JsonValue = JsonScalar | list[JsonValue] | dict[str, JsonValue]
_EMBEDDED_DIGEST = re.compile(rb"sha256:[0-9a-f]{64}")


class ArtifactKind(StrEnum):
    """Initial immutable artifact categories."""

    KERNEL_AGENT = "kernel_agent"
    KERNEL_AGENT_RUNTIME_STATE = "kernel_agent_runtime_state"
    OPTIMIZER_SOURCE = "optimizer_source"
    EVOLVER_BUNDLE = "evolver_bundle"
    KERNEL = "kernel"
    EVIDENCE = "evidence"
    ATTEMPT_EVIDENCE = "attempt_evidence"
    SESSION_LOG = "session_log"
    GATEWAY_RESULT = "gateway_result"
    RESULT_ARTIFACT = "result_artifact"
    EVOLUTION = "evolution"
    ATTEMPT_REPORT = "attempt_report"
    WIKI_INTERACTION = "wiki_interaction"
    EVALUATION_CONTRACT = "evaluation_contract"
    AGENT_PROBLEM = "agent_problem"


@dataclass(frozen=True, slots=True)
class StoredArtifact:
    """Verified local handle to one sealed artifact."""

    digest: ArtifactDigest
    kind: ArtifactKind
    payload_path: Path


@dataclass(frozen=True, slots=True)
class ArtifactGarbageCollectionResult:
    """Bounded CAS scan and deletion counts for one maintenance pass."""

    scanned: int
    eligible: int
    deleted: int
    reclaimed_bytes: int
    applied: bool


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


class LocalArtifactStore:
    """Immutable local CAS with manifest verification and symlink rejection."""

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root).resolve()
        self._objects = self._root / "sha256"
        self._temporary = self._root / ".tmp"
        self._objects.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._temporary.mkdir(parents=True, exist_ok=True, mode=0o700)

    def check_health(self) -> None:
        """Verify that the object root is readable and staging remains writable."""
        if not self._objects.is_dir() or not self._temporary.is_dir():
            raise RuntimeError("Artifact Store directories are unavailable")
        descriptor, temporary_name = tempfile.mkstemp(prefix="ready-", dir=self._temporary)
        try:
            os.write(descriptor, b"ready")
            os.close(descriptor)
            descriptor = -1
            if Path(temporary_name).read_bytes() != b"ready":
                raise RuntimeError("Artifact Store readiness probe could not read staging data")
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            Path(temporary_name).unlink(missing_ok=True)

    def put_json(self, value: JsonValue, kind: ArtifactKind) -> ArtifactDigest:
        """Seal one canonical JSON value and return its digest."""
        temporary = Path(tempfile.mkdtemp(prefix="json-", dir=self._temporary))
        try:
            value_path = temporary / "value.json"
            value_path.write_bytes(canonical_json_bytes(value))
            os.chmod(value_path, 0o600)
            return self.put_directory(temporary, kind)
        finally:
            shutil.rmtree(temporary, ignore_errors=True)

    def put_directory(
        self,
        source: str | Path,
        kind: ArtifactKind,
        *,
        exclude: Callable[[PurePosixPath, bool], bool] | None = None,
    ) -> ArtifactDigest:
        """Copy and seal a directory while rejecting links and special files.

        `exclude` receives each relative path and whether it is a directory, and
        drops it from the Artifact. An excluded directory drops its whole subtree,
        so a caller can seal a live workspace tree without build products.
        """
        source_path = Path(source)
        source_stat = source_path.lstat()
        if stat.S_ISLNK(source_stat.st_mode) or not stat.S_ISDIR(source_stat.st_mode):
            raise ValueError(f"artifact source must be a real directory: {source_path}")

        staging = Path(tempfile.mkdtemp(prefix="artifact-", dir=self._temporary))
        payload = staging / "payload"
        payload.mkdir(mode=0o700)
        files: list[JsonValue] = []
        directories: list[JsonValue] = []
        try:
            for candidate in sorted(source_path.rglob("*")):
                candidate_stat = candidate.lstat()
                if stat.S_ISLNK(candidate_stat.st_mode):
                    raise ValueError(f"artifact contains a symbolic link: {candidate}")
                relative = candidate.relative_to(source_path)
                relative_posix = PurePosixPath(*relative.parts).as_posix()
                is_directory = bool(stat.S_ISDIR(candidate_stat.st_mode))
                if exclude is not None and exclude(PurePosixPath(relative_posix), is_directory):
                    continue
                if is_directory:
                    if next(candidate.iterdir(), None) is None:
                        payload.joinpath(*relative.parts).mkdir(parents=True, mode=0o700)
                        directories.append(relative_posix)
                    continue
                if not stat.S_ISREG(candidate_stat.st_mode):
                    raise ValueError(f"artifact contains a non-regular file: {candidate}")
                destination = payload.joinpath(*relative.parts)
                destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                with candidate.open("rb") as reader, destination.open("xb") as writer:
                    while chunk := reader.read(1024 * 1024):
                        writer.write(chunk)
                os.chmod(destination, 0o600)
                files.append(
                    {
                        "path": relative_posix,
                        "size": destination.stat().st_size,
                        "sha256": _sha256(destination),
                    }
                )

            manifest: dict[str, JsonValue] = {
                "version": 1,
                "kind": kind.value,
                "files": files,
            }
            # Absent unless the tree really holds an empty directory, so every artifact
            # sealed before directories were recorded keeps its existing digest.
            if directories:
                manifest["directories"] = directories
            manifest_bytes = canonical_json_bytes(manifest)
            hexadecimal = hashlib.sha256(manifest_bytes).hexdigest()
            digest = parse_artifact_digest(f"sha256:{hexadecimal}")
            manifest_path = staging / "manifest.json"
            manifest_path.write_bytes(manifest_bytes)
            os.chmod(manifest_path, 0o400)
            make_tree_read_only(payload)

            destination = self._objects / hexadecimal
            try:
                os.rename(staging, destination)
            except OSError:
                if not destination.is_dir():
                    raise
                self._discard_staging(staging)
                self.verify(digest)
            return digest
        except BaseException:
            if staging.exists():
                self._discard_staging(staging)
            raise

    def verify(self, digest: ArtifactDigest) -> StoredArtifact:
        """Verify the manifest address, exact file set, and every payload hash."""
        hexadecimal = str(digest).removeprefix("sha256:")
        artifact_path = self._objects / hexadecimal
        manifest_path = artifact_path / "manifest.json"
        payload = artifact_path / "payload"
        if not manifest_path.is_file() or not payload.is_dir():
            raise FileNotFoundError(f"artifact is incomplete: {digest}")

        manifest_bytes = manifest_path.read_bytes()
        actual_digest = hashlib.sha256(manifest_bytes).hexdigest()
        if actual_digest != hexadecimal:
            raise ValueError(f"artifact manifest digest mismatch: {digest}")
        manifest_value: object = json.loads(manifest_bytes)
        if not isinstance(manifest_value, dict):
            raise ValueError(f"artifact manifest must be an object: {digest}")
        if manifest_value.get("version") != 1:
            raise ValueError(f"unsupported artifact manifest version: {digest}")
        try:
            kind = ArtifactKind(manifest_value["kind"])
        except (KeyError, ValueError) as error:
            raise ValueError(f"invalid artifact kind: {digest}") from error
        entries = manifest_value.get("files")
        if not isinstance(entries, list):
            raise ValueError(f"artifact files must be a list: {digest}")
        # Absent on every artifact sealed before empty directories were recorded.
        directory_entries = manifest_value.get("directories", [])
        if not isinstance(directory_entries, list):
            raise ValueError(f"artifact directories must be a list: {digest}")

        expected_directories: set[str] = set()
        for entry in directory_entries:
            if not isinstance(entry, str):
                raise ValueError(f"artifact directory entry must be a string: {digest}")
            relative = PurePosixPath(entry)
            if relative.is_absolute() or ".." in relative.parts or relative.as_posix() == ".":
                raise ValueError(f"unsafe artifact path: {entry!r}")
            expected_directories.add(relative.as_posix())
            path = payload.joinpath(*relative.parts)
            path_stat = path.lstat()
            if not stat.S_ISDIR(path_stat.st_mode):
                raise ValueError(f"artifact payload is not a directory: {entry}")
            if next(path.iterdir(), None) is not None:
                raise ValueError(f"artifact recorded directory is not empty: {entry}")

        expected_paths: set[str] = set()
        for entry in entries:
            if not isinstance(entry, dict):
                raise ValueError(f"artifact file entry must be an object: {digest}")
            relative_value = entry.get("path")
            size = entry.get("size")
            expected_hash = entry.get("sha256")
            if not isinstance(relative_value, str) or not isinstance(size, int):
                raise ValueError(f"invalid artifact file metadata: {digest}")
            if not isinstance(expected_hash, str):
                raise ValueError(f"invalid artifact file digest: {digest}")
            relative = PurePosixPath(relative_value)
            if relative.is_absolute() or ".." in relative.parts or relative.as_posix() == ".":
                raise ValueError(f"unsafe artifact path: {relative_value!r}")
            expected_paths.add(relative.as_posix())
            path = payload.joinpath(*relative.parts)
            path_stat = path.lstat()
            if not stat.S_ISREG(path_stat.st_mode):
                raise ValueError(f"artifact payload is not a regular file: {relative_value}")
            if path_stat.st_size != size or _sha256(path) != expected_hash:
                raise ValueError(f"artifact payload mismatch: {relative_value}")

        actual_paths: set[str] = set()
        actual_directories: set[str] = set()
        for path in payload.rglob("*"):
            path_stat = path.lstat()
            if stat.S_ISLNK(path_stat.st_mode):
                raise ValueError(f"artifact payload contains a symbolic link: {path}")
            if stat.S_ISREG(path_stat.st_mode):
                actual_paths.add(PurePosixPath(*path.relative_to(payload).parts).as_posix())
            elif stat.S_ISDIR(path_stat.st_mode):
                if next(path.iterdir(), None) is None:
                    actual_directories.add(
                        PurePosixPath(*path.relative_to(payload).parts).as_posix()
                    )
            else:
                raise ValueError(f"artifact payload contains a non-regular entry: {path}")
        if actual_paths != expected_paths:
            raise ValueError(f"artifact payload file set mismatch: {digest}")
        if actual_directories != expected_directories:
            raise ValueError(f"artifact payload directory set mismatch: {digest}")
        return StoredArtifact(digest=digest, kind=kind, payload_path=payload)

    def materialize(self, digest: ArtifactDigest, destination: str | Path) -> Path:
        """Copy a verified payload to a new destination without overwriting data."""
        artifact = self.verify(digest)
        destination_path = Path(destination)
        if destination_path.exists() or destination_path.is_symlink():
            raise FileExistsError(destination_path)
        destination_path.mkdir(parents=True, mode=0o700)
        try:
            for source in artifact.payload_path.rglob("*"):
                relative = source.relative_to(artifact.payload_path)
                target = destination_path / relative
                if source.is_dir():
                    target.mkdir(exist_ok=True, mode=0o700)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                    with source.open("rb") as reader, target.open("xb") as writer:
                        shutil.copyfileobj(reader, writer, length=1024 * 1024)
                    os.chmod(target, 0o400)
            return destination_path
        except BaseException:
            shutil.rmtree(destination_path, ignore_errors=True)
            raise

    def materialize_file(
        self,
        digest: ArtifactDigest,
        member: str,
        destination: str | Path,
    ) -> Path:
        """Copy one verified regular file from an Artifact to a new read-only path."""
        artifact = self.verify(digest)
        relative = PurePosixPath(member)
        if relative.is_absolute() or ".." in relative.parts or relative.as_posix() == ".":
            raise ValueError(f"unsafe artifact member path: {member!r}")
        source = artifact.payload_path.joinpath(*relative.parts)
        if source.is_symlink() or not source.is_file():
            raise ValueError(f"artifact member is not a regular file: {member!r}")
        destination_path = Path(destination)
        if destination_path.exists() or destination_path.is_symlink():
            raise FileExistsError(destination_path)
        destination_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        try:
            with source.open("rb") as reader, destination_path.open("xb") as writer:
                shutil.copyfileobj(reader, writer, length=1024 * 1024)
            os.chmod(destination_path, 0o400)
            return destination_path
        except BaseException:
            destination_path.unlink(missing_ok=True)
            raise

    def collect_garbage(
        self,
        referenced: Collection[ArtifactDigest],
        *,
        minimum_age_seconds: float,
        limit: int,
        apply: bool,
        clock: Callable[[], float] = time.time,
    ) -> ArtifactGarbageCollectionResult:
        """Verify and optionally remove a bounded set of old unreferenced objects.

        The caller must stop every Runtime process before an applying pass so the
        reference snapshot cannot race an Artifact seal followed by its database write.
        """
        if minimum_age_seconds < 0:
            raise ValueError("Artifact GC minimum age cannot be negative")
        if limit <= 0:
            raise ValueError("Artifact GC limit must be positive")
        referenced_values = {str(digest) for digest in referenced}
        now = clock()
        scanned = 0
        eligible = 0
        deleted = 0
        reclaimed_bytes = 0
        hexadecimal_digits = frozenset("0123456789abcdef")
        for artifact_path in sorted(self._objects.iterdir(), key=lambda path: path.name):
            if (
                len(artifact_path.name) != 64
                or any(character not in hexadecimal_digits for character in artifact_path.name)
                or not artifact_path.is_dir()
                or artifact_path.is_symlink()
            ):
                raise RuntimeError(f"unexpected Artifact Store object entry: {artifact_path.name}")
            scanned += 1
            digest = parse_artifact_digest(f"sha256:{artifact_path.name}")
            if str(digest) in referenced_values:
                continue
            age_seconds = now - artifact_path.stat().st_mtime
            if age_seconds < minimum_age_seconds:
                continue
            stored = self.verify(digest)
            size = (artifact_path / "manifest.json").stat().st_size + sum(
                path.stat().st_size for path in stored.payload_path.rglob("*") if path.is_file()
            )
            eligible += 1
            if apply:
                self._discard_staging(artifact_path)
                deleted += 1
                reclaimed_bytes += size
            if eligible >= limit:
                break
        return ArtifactGarbageCollectionResult(
            scanned=scanned,
            eligible=eligible,
            deleted=deleted,
            reclaimed_bytes=reclaimed_bytes,
            applied=apply,
        )

    def expand_reference_closure(
        self,
        roots: Collection[ArtifactDigest],
    ) -> set[ArtifactDigest]:
        """Return existing CAS objects reachable from verified durable roots."""
        retained: set[ArtifactDigest] = set()
        pending = list(roots)
        root_values = {str(digest) for digest in roots}
        while pending:
            digest = pending.pop()
            if digest in retained:
                continue
            try:
                stored = self.verify(digest)
            except FileNotFoundError:
                if str(digest) in root_values:
                    raise
                continue
            retained.add(digest)
            for path in stored.payload_path.rglob("*"):
                if not path.is_file():
                    continue
                for discovered in self._file_digest_values(path):
                    if discovered not in retained:
                        pending.append(discovered)
        return retained

    @staticmethod
    def _file_digest_values(path: Path) -> set[ArtifactDigest]:
        """Find exact digest tokens with bounded memory across chunk boundaries."""
        values: set[ArtifactDigest] = set()
        tail = b""
        with path.open("rb") as source:
            while chunk := source.read(1024 * 1024):
                data = tail + chunk
                values.update(
                    parse_artifact_digest(match.group().decode())
                    for match in _EMBEDDED_DIGEST.finditer(data)
                )
                tail = data[-70:]
        return values

    @staticmethod
    def _discard_staging(path: Path) -> None:
        """Restore owner write bits on a private staging tree, then remove it."""
        make_tree_owner_writable(path)
        shutil.rmtree(path)
