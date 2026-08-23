"""Shared execution boundary for every phase of one Core Bundle."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Protocol

from ..artifacts.local import ArtifactKind, LocalArtifactStore
from ..domain.errors import InfrastructureError
from ..domain.ids import ArtifactDigest
from ..kernel_agents import KERNEL_AGENT_BUNDLE_MANIFEST, KernelAgentBundleManifestV1
from .launcher import WorkerLauncher, validate_worker_environment
from .process import BoundedProcessConfig, BoundedProcessResult, BoundedProcessRunner
from .session_trace import enforce_session_trace_retention
from .token_usage import ProviderUsageReportV2, UsageUnit

CORE_TIMEOUT_EXIT_STATUS = 124
_UNSEALED_SESSION_MARKER = ".runtime-live-session"


class CoreProcessPolicy(Protocol):
    @property
    def command_prefix(self) -> tuple[str, ...]: ...

    @property
    def agent_backend(self) -> str: ...

    @property
    def reasoning_effort(self) -> str: ...

    @property
    def session_settings(self) -> str: ...

    @property
    def isolated_home_environment_keys(self) -> tuple[str, ...]: ...

    @property
    def session_trace_relative_path(self) -> str | None: ...

    @property
    def token_usage_report_relative_path(self) -> str: ...

    @property
    def timeout_seconds(self) -> float: ...

    @property
    def terminate_grace_seconds(self) -> float: ...

    @property
    def max_diagnostic_bytes(self) -> int: ...

    @property
    def max_session_tokens(self) -> int: ...

    @property
    def max_session_credits(self) -> float: ...


@dataclass(frozen=True, slots=True)
class PreparedCorePhase:
    root: Path
    repository: Path
    command: Path
    token_usage_path: Path
    agent_home: Path


@dataclass(frozen=True, slots=True)
class CorePhaseResult:
    process: BoundedProcessResult
    token_usage: ProviderUsageReportV2
    session_trace_digest: ArtifactDigest | None

    @property
    def finish_reason(self) -> str:
        if self.process.returncode == 0:
            return "completed"
        if self.process.returncode == 125:
            return "usage-budget-exhausted"
        return f"process-exit-{self.process.returncode}"


class CorePhaseRunner:
    """Resolve the Bundle command and own process, token, and trace acquisition once."""

    def __init__(
        self,
        launcher: WorkerLauncher,
        policy: CoreProcessPolicy,
        artifacts: LocalArtifactStore,
    ) -> None:
        self._launcher = launcher
        self._policy = policy
        self._artifacts = artifacts
        self._processes = BoundedProcessRunner(
            BoundedProcessConfig(
                policy.timeout_seconds,
                policy.terminate_grace_seconds,
                policy.max_diagnostic_bytes,
            )
        )

    def prepare(self, root: Path, session_root: Path) -> PreparedCorePhase:
        repository = root / "agent/optimizer"
        manifest = KernelAgentBundleManifestV1.from_file(repository / KERNEL_AGENT_BUNDLE_MANIFEST)
        command = repository.joinpath(*PurePosixPath(manifest.entrypoint.command).parts)
        if command.is_symlink() or not command.is_file():
            raise InfrastructureError("Core Bundle command is unavailable")
        token_usage_path = root.joinpath(
            *PurePosixPath(self._policy.token_usage_report_relative_path).parts
        )
        token_usage_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        agent_home = session_root / "agent-home"
        agent_home.mkdir(mode=0o700)
        return PreparedCorePhase(root, repository, command, token_usage_path, agent_home)

    def runtime_environment(
        self,
        prepared: PreparedCorePhase,
        *,
        phase: str,
        model: str | None = None,
    ) -> dict[str, str]:
        usage_unit: UsageUnit = (
            "credits" if self._policy.agent_backend == "qodercli" else "provider_tokens"
        )
        usage_budget = (
            self._policy.max_session_credits
            if usage_unit == "credits"
            else float(self._policy.max_session_tokens)
        )
        environment = {
            "ATREX_AGENT_BACKEND": self._policy.agent_backend,
            "ATREX_AGENT_MODEL": model or "",
            "ATREX_AGENT_REASONING_EFFORT": self._policy.reasoning_effort,
            "ATREX_AGENT_SESSION_SETTINGS": self._policy.session_settings,
            "ATREX_CORE_PHASE": phase,
            "ATREX_OPTIMIZER_REPOSITORY": str(prepared.repository),
            "ATREX_SESSION_TIMEOUT_SECONDS": str(self._policy.timeout_seconds),
            "ATREX_USAGE_BUDGET": str(usage_budget),
            "ATREX_USAGE_UNIT": usage_unit,
            "ATREX_TOKEN_USAGE_REPORT": str(prepared.token_usage_path),
        }
        if self._policy.session_trace_relative_path is not None:
            environment["ATREX_SESSION_TRACE_PATH"] = str(
                prepared.root.joinpath(
                    *PurePosixPath(self._policy.session_trace_relative_path).parts
                )
            )
        environment.update(
            (key, str(prepared.agent_home)) for key in self._policy.isolated_home_environment_keys
        )
        return environment

    def run(
        self,
        prepared: PreparedCorePhase,
        environment: Mapping[str, str],
        *,
        label: str,
    ) -> CorePhaseResult:
        validate_worker_environment(environment)
        argv = self._launcher.wrap(
            (*self._policy.command_prefix, str(prepared.command)),
            workspace=prepared.root,
            environment=environment,
        )
        try:
            process = self._processes.run(argv, cwd=prepared.root, stdin=None)
        except OSError as error:
            raise InfrastructureError(f"{label} process could not start: {error}") from error
        except TimeoutError as error:
            raise InfrastructureError(f"{label} exceeded its wall-time limit") from error
        # The Bundle owns the inner Agent process and translates its wall-time
        # expiration into the conventional timeout exit status. Check that
        # terminal condition before validating usage: a killed Provider cannot
        # emit its final authoritative usage event, so the resulting partial
        # report is evidence of the timeout rather than its root cause.
        if process.returncode == CORE_TIMEOUT_EXIT_STATUS:
            raise InfrastructureError(f"{label} timed out")
        try:
            usage_unit: UsageUnit = (
                "credits" if self._policy.agent_backend == "qodercli" else "provider_tokens"
            )
            expected_budget = (
                self._policy.max_session_credits
                if usage_unit == "credits"
                else float(self._policy.max_session_tokens)
            )
            usage = ProviderUsageReportV2.from_file(
                prepared.token_usage_path,
                expected_unit=usage_unit,
                expected_budget=expected_budget,
            )
        except ValueError as error:
            raise InfrastructureError(f"Invalid {label} provider usage report: {error}") from error
        return CorePhaseResult(process, usage, self._seal_trace(prepared.root, label))

    def _seal_trace(self, root: Path, label: str) -> ArtifactDigest | None:
        relative = self._policy.session_trace_relative_path
        if relative is None:
            return None
        path = root.joinpath(*PurePosixPath(relative).parts)
        if not path.exists():
            return None
        if path.is_symlink() or not path.is_dir():
            raise InfrastructureError(f"{label} session trace must be a real directory")
        marker = path / _UNSEALED_SESSION_MARKER
        if marker.exists() or marker.is_symlink():
            return None
        enforce_session_trace_retention(path)
        return self._artifacts.put_directory(path, ArtifactKind.SESSION_LOG)
