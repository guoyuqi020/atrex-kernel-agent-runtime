"""Resolve an Agate scheduling environment into Agent-visible hardware metadata."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class AgateEnvironmentClient(Protocol):
    """Minimal Agate client surface required for hardware discovery."""

    def get_env(self, gpu: str, force: bool = False) -> dict[str, object]: ...


@dataclass(frozen=True, slots=True)
class ResolvedAgateEnvironment:
    """Canonical scheduler selector and architecture reported by Agate."""

    gpu: str
    arch: str

    def __post_init__(self) -> None:
        if not self.gpu.strip():
            raise ValueError("Agate GPU environment cannot be empty")
        if not self.arch.strip():
            raise ValueError("Agate GPU architecture cannot be empty")


class AgateHardwareTargetResolver:
    """Query Agate for the architecture corresponding to one GPU environment alias."""

    def __init__(self, client: AgateEnvironmentClient) -> None:
        self._client = client

    def resolve(self, gpu: str) -> ResolvedAgateEnvironment:
        requested = gpu.strip()
        if not requested:
            raise ValueError("Agate GPU environment cannot be empty")
        value = self._client.get_env(requested)
        canonical_gpu = value.get("gpu")
        arch = value.get("arch")
        if not isinstance(canonical_gpu, str) or not canonical_gpu.strip():
            raise ValueError("Agate environment response is missing canonical gpu")
        if not isinstance(arch, str) or not arch.strip():
            raise ValueError("Agate environment response is missing arch")
        return ResolvedAgateEnvironment(canonical_gpu.strip(), arch.strip())


__all__ = ["AgateHardwareTargetResolver", "ResolvedAgateEnvironment"]
