"""Immutable content-addressed artifact storage."""

from .local import (
    ArtifactGarbageCollectionResult,
    ArtifactKind,
    LocalArtifactStore,
    StoredArtifact,
)

__all__ = [
    "ArtifactGarbageCollectionResult",
    "ArtifactKind",
    "LocalArtifactStore",
    "StoredArtifact",
]
