"""Trusted policy for changes submitted through the Kernel candidate channel."""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from ..artifacts.local import ArtifactKind, LocalArtifactStore
from ..domain.ids import ArtifactDigest, AttemptId
from ..domain.models import Dsl
from ..registry.base import Registry
from .control import SqliteGatewayControl


@dataclass(frozen=True, slots=True)
class CandidateDiffPolicy:
    """Allow only configured DSL paths to differ from an Attempt input Kernel."""

    allowed_paths: dict[Dsl, tuple[str, ...]]
    require_change: bool

    def __post_init__(self) -> None:
        if set(self.allowed_paths) != set(Dsl):
            raise ValueError("Candidate diff policy must define every DSL")
        for dsl, patterns in self.allowed_paths.items():
            if not patterns:
                raise ValueError(f"Candidate diff policy for {dsl.value} cannot be empty")
            for pattern in patterns:
                path = PurePosixPath(pattern)
                if path.is_absolute() or ".." in path.parts or path.as_posix() == ".":
                    raise ValueError(f"unsafe Candidate diff pattern: {pattern!r}")


class RegistryCandidateDiffValidator:
    """Compare a sealed candidate with the immutable input named by its Attempt."""

    def __init__(
        self,
        registry: Registry,
        artifacts: LocalArtifactStore,
        policy: CandidateDiffPolicy,
        bootstrap_subjects: SqliteGatewayControl | None = None,
    ) -> None:
        self._registry = registry
        self._artifacts = artifacts
        self._policy = policy
        self._bootstrap_subjects = bootstrap_subjects

    def validate(self, attempt_id: AttemptId, candidate_digest: ArtifactDigest) -> None:
        """Reject an unchanged candidate or any change outside its DSL allowlist."""
        try:
            attempt = self._registry.get_attempt(attempt_id)
        except KeyError:
            if self._bootstrap_subjects is None:
                raise
            subject = self._bootstrap_subjects.get_bootstrap_subject(attempt_id)
            input_digest = subject.input_kernel_digest
            dsl = subject.dsl
        else:
            epoch = self._registry.get_epoch(attempt.epoch_id)
            lineage = self._registry.get_lineage(epoch.lineage_id)
            baseline = self._registry.get_kernel_revision(attempt.input_kernel_revision_id)
            input_digest = baseline.artifact_digest
            dsl = lineage.dsl
        before = self._artifacts.verify(input_digest)
        after = self._artifacts.verify(candidate_digest)
        if before.kind is not ArtifactKind.KERNEL or after.kind is not ArtifactKind.KERNEL:
            raise ValueError("Candidate diff policy requires Kernel artifacts")
        before_files = self._files(before.payload_path)
        after_files = self._files(after.payload_path)
        changed = {
            path
            for path in set(before_files).union(after_files)
            if self._bytes(before_files.get(path)) != self._bytes(after_files.get(path))
        }
        if self._policy.require_change and not changed:
            raise ValueError("candidate does not change the input Kernel")
        patterns = self._policy.allowed_paths[dsl]
        rejected = sorted(
            path
            for path in changed
            if not any(fnmatch.fnmatchcase(path, pattern) for pattern in patterns)
        )
        if rejected:
            raise ValueError(f"candidate changes disallowed paths: {rejected}")

    @staticmethod
    def _files(root: Path) -> dict[str, Path]:
        return {
            PurePosixPath(*path.relative_to(root).parts).as_posix(): path
            for path in root.rglob("*")
            if path.is_file()
        }

    @staticmethod
    def _bytes(path: Path | None) -> bytes | None:
        return None if path is None else path.read_bytes()
