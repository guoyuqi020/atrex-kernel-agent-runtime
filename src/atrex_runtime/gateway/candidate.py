"""Resolve immutable Kernel Artifacts at trusted evaluation boundaries."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..artifacts.local import ArtifactKind, LocalArtifactStore
from ..domain.ids import ArtifactDigest


@dataclass(frozen=True, slots=True)
class ResolvedKernelCandidate:
    """Verified Kernel Artifact root and its contract-selected source file."""

    root: Path
    source: Path


def resolve_kernel_candidate(
    artifacts: LocalArtifactStore,
    digest: ArtifactDigest,
    candidate_path: str,
    *,
    error_type: type[Exception],
    kind_error: str,
    missing_error: str,
) -> ResolvedKernelCandidate:
    """Verify Artifact kind and resolve a regular, non-symlink candidate source."""
    stored = artifacts.verify(digest)
    if stored.kind is not ArtifactKind.KERNEL:
        raise error_type(kind_error)
    source = stored.payload_path.joinpath(*candidate_path.split("/"))
    if source.is_symlink() or not source.is_file():
        raise error_type(missing_error)
    return ResolvedKernelCandidate(stored.payload_path, source)


__all__ = ["ResolvedKernelCandidate", "resolve_kernel_candidate"]
