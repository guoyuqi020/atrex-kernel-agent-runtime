"""Resolve an Agate scheduling environment into Agent-visible hardware metadata."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal, Protocol, cast

type AcceleratorBackend = Literal["cuda", "rocm", "ppu"]


class AgateEnvironmentClient(Protocol):
    """Minimal Agate client surface required for hardware discovery."""

    def get_env(self, gpu: str, force: bool = False) -> dict[str, object]: ...


@dataclass(frozen=True, slots=True)
class ResolvedAgateEnvironment:
    """Canonical scheduler selector and architecture reported by Agate."""

    gpu: str
    arch: str
    accelerator_backend: AcceleratorBackend | None = None
    device_slug: str | None = None

    def __post_init__(self) -> None:
        if not self.gpu.strip():
            raise ValueError("Agate GPU environment cannot be empty")
        if not self.arch.strip():
            raise ValueError("Agate GPU architecture cannot be empty")
        if self.device_slug is not None and not self.device_slug.strip():
            raise ValueError("Agate device slug cannot be empty")

    @property
    def supports_clock_lock(self) -> bool:
        """Return whether the canonical evaluator may request managed clocks."""
        return self.accelerator_backend != "ppu"


def infer_accelerator_backend(gpu: str, arch: str) -> AcceleratorBackend | None:
    """Infer the backend when Agate has not yet published the explicit field."""
    values = tuple(value.strip().lower() for value in (gpu, arch) if value.strip())
    if any(
        value.startswith(("ppu-", "zw-")) or "ppu" in value
        for value in values
    ):
        return "ppu"
    if any(
        value.startswith("gfx") or "rocm" in value or re.search(r"(?:^|[-_])mi\d", value)
        for value in values
    ):
        return "rocm"
    if any(
        value.startswith("sm_") or "cuda" in value or "nvidia" in value
        for value in values
    ):
        return "cuda"
    return None


def _inferred_device_slug(arch: str, backend: AcceleratorBackend | None) -> str | None:
    if backend != "ppu":
        return None
    normalized = re.sub(r"[^a-z0-9]+", "-", arch.strip().lower()).strip("-")
    return normalized or None


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
        explicit_backend = value.get("accelerator_backend")
        if explicit_backend is not None and explicit_backend not in {"cuda", "rocm", "ppu"}:
            raise ValueError("Agate environment response has an invalid accelerator_backend")
        backend = (
            cast(AcceleratorBackend, explicit_backend)
            if explicit_backend is not None
            else infer_accelerator_backend(canonical_gpu, arch)
        )
        device_slug = value.get("device_slug")
        if device_slug is not None and not isinstance(device_slug, str):
            raise ValueError("Agate environment response has an invalid device_slug")
        normalized_slug = (
            device_slug.strip()
            if isinstance(device_slug, str)
            else _inferred_device_slug(arch, backend)
        )
        return ResolvedAgateEnvironment(
            canonical_gpu.strip(),
            arch.strip(),
            backend,
            normalized_slug,
        )


__all__ = [
    "AcceleratorBackend",
    "AgateHardwareTargetResolver",
    "ResolvedAgateEnvironment",
    "infer_accelerator_backend",
]
