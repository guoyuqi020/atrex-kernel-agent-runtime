"""Framework-neutral process adapter for a Core-owned Optimizer entrypoint."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath

import anyio

from ..artifacts.local import ArtifactKind, LocalArtifactStore
from .attempt_report import AttemptReportV12
from .core_phase import CorePhaseRunner, PreparedCorePhase
from .evidence_view import EVIDENCE_PROMPT_RELATIVE_PATH
from .launcher import WorkerLauncher, validate_worker_environment
from .manifest import AttemptInputManifestV9
from .optimizer import OptimizerSessionConfig, OptimizerSessionResult
from .process import BoundedProcessConfig
from .workspace import PreparedAttempt

_RUNTIME_KEYS = {
    "ATREX_AGENT_BACKEND",
    "ATREX_DEV_SHELL_BACKENDS",
    "ATREX_AGENT_MODEL",
    "ATREX_AGENT_REASONING_EFFORT",
    "ATREX_AGENT_SESSION_SETTINGS",
    "ATREX_CORE_PHASE",
    "ATREX_ATTEMPT_MANIFEST",
    "ATREX_ATTEMPT_REPORT_PATH",
    "ATREX_EVIDENCE_PROMPT_PATH",
    "ATREX_GATEWAY_CAPABILITY",
    "ATREX_GATEWAY_PROXY_URL",
    "ATREX_OPTIMIZER_REPOSITORY",
    "ATREX_SESSION_TIMEOUT_SECONDS",
    "ATREX_SESSION_TRACE_PATH",
    "ATREX_USAGE_BUDGET",
    "ATREX_USAGE_UNIT",
    "ATREX_TOKEN_USAGE_REPORT",
    "ATREX_WIKI_CAPABILITY",
    "ATREX_WIKI_PROXY_URL",
}


@dataclass(frozen=True, slots=True)
class CoreOptimizerProcessConfig:
    """Deployment process policy around a repository-declared Optimizer command."""

    command_prefix: tuple[str, ...]
    isolated_home_environment_keys: tuple[str, ...]
    session_trace_relative_path: str | None
    token_usage_report_relative_path: str
    max_attempt_report_bytes: int
    timeout_seconds: float
    terminate_grace_seconds: float
    max_diagnostic_bytes: int
    max_session_tokens: int
    max_session_credits: float = 1_000_000.0
    agent_backend: str = "qodercli"
    reasoning_effort: str = "max"
    session_settings: str = ""

    def __post_init__(self) -> None:
        if not self.command_prefix:
            raise ValueError("Optimizer command prefix cannot be empty")
        executable = Path(self.command_prefix[0])
        if not executable.is_absolute() or not executable.is_file():
            raise ValueError("Optimizer command-prefix executable must be an absolute file")
        if not os.access(executable, os.X_OK):
            raise ValueError("Optimizer command-prefix executable must be executable")
        if any("\x00" in argument for argument in self.command_prefix):
            raise ValueError("Optimizer command prefix cannot contain NUL")
        if self.agent_backend not in {"claude", "codex", "qodercli", "pi"}:
            raise ValueError("Optimizer agent backend is unsupported")
        if self.reasoning_effort not in {"low", "medium", "high", "max"}:
            raise ValueError("Optimizer reasoning effort is unsupported")
        if "\x00" in self.session_settings:
            raise ValueError("Optimizer session settings cannot contain NUL")
        if (
            self.max_attempt_report_bytes <= 0
            or self.max_session_tokens <= 0
            or self.max_session_credits <= 0
        ):
            raise ValueError("Optimizer report and provider-usage limits must be positive")
        validate_worker_environment(
            {key: "validated" for key in self.isolated_home_environment_keys}
        )
        if len(self.isolated_home_environment_keys) != len(
            set(self.isolated_home_environment_keys)
        ):
            raise ValueError("Optimizer isolated-home keys cannot repeat")
        self._validate_relative(self.token_usage_report_relative_path, "token usage report")
        if self.session_trace_relative_path is not None:
            self._validate_relative(self.session_trace_relative_path, "session trace")
        BoundedProcessConfig(
            self.timeout_seconds,
            self.terminate_grace_seconds,
            self.max_diagnostic_bytes,
        )

    @staticmethod
    def _validate_relative(value: str, label: str) -> None:
        relative = PurePosixPath(value)
        if relative.is_absolute() or relative.as_posix() == "." or ".." in relative.parts:
            raise ValueError(f"Optimizer {label} path must be safe and relative")


@dataclass(frozen=True, slots=True)
class PreparedOptimizerLaunch:
    """Fully prepared Core launch contract reusable by Agent and dev-shell drivers."""

    phase: PreparedCorePhase
    environment: Mapping[str, str]
    attempt_report_path: Path


class CoreOptimizerSessionDriver:
    """Execute Core code; Core itself owns the choice of Agent framework."""

    def __init__(
        self,
        launcher: WorkerLauncher,
        config: CoreOptimizerProcessConfig,
        artifacts: LocalArtifactStore,
    ) -> None:
        self._launcher = launcher
        self._config = config
        self._artifacts = artifacts
        self._phases = CorePhaseRunner(launcher, config, artifacts)

    async def run(
        self,
        prepared: PreparedAttempt,
        config: OptimizerSessionConfig,
    ) -> OptimizerSessionResult:
        """Run one repository-declared command and normalize its Runtime protocol files."""
        return await anyio.to_thread.run_sync(self._run_sync, prepared, config)

    def _run_sync(
        self,
        prepared: PreparedAttempt,
        config: OptimizerSessionConfig,
    ) -> OptimizerSessionResult:
        result: OptimizerSessionResult | None = None
        try:
            result = self._run_sync_once(prepared, config)
        finally:
            prepared.persist_reusable_directories()
        return replace(
            result,
            runtime_state_digest=prepared.seal_runtime_state(self._artifacts),
        )

    def _run_sync_once(
        self,
        prepared: PreparedAttempt,
        config: OptimizerSessionConfig,
    ) -> OptimizerSessionResult:
        launch = self.prepare_launch(prepared, config)
        result = self._phases.run(
            launch.phase,
            launch.environment,
            label="Optimizer repository",
        )
        report: AttemptReportV12 | None = None
        report_digest = None
        report_error: str | None = None
        candidate_digest = None
        if launch.attempt_report_path.exists() or launch.attempt_report_path.is_symlink():
            try:
                report = AttemptReportV12.from_file(
                    launch.attempt_report_path,
                    expected_attempt_id=AttemptInputManifestV9.from_json_bytes(
                        prepared.manifest_path.read_bytes()
                    ).attempt_id,
                    max_bytes=self._config.max_attempt_report_bytes,
                )
                if any(experiment.action == "baseline" for experiment in report.experiments):
                    raise ValueError(
                        "Experiment action baseline is only valid during Bootstrap"
                    )
            except ValueError as error:
                report_error = str(error)
            else:
                report_digest = self._artifacts.put_json(
                    report.model_dump(mode="json"),
                    ArtifactKind.ATTEMPT_REPORT,
                )
                if report.status == "candidate_ready":
                    candidate_digest = self._artifacts.put_directory(
                        prepared.root / "work/kernel",
                        ArtifactKind.KERNEL,
                    )
        return OptimizerSessionResult(
            finish_reason=result.finish_reason,
            final_response=result.process.stdout,
            token_usage=result.token_usage.to_domain(),
            token_budget=int(result.token_usage.require_budget()),
            session_trace_digest=result.session_trace_digest,
            attempt_report=report,
            attempt_report_digest=report_digest,
            attempt_report_error=report_error,
            kernel_artifact_digest=candidate_digest,
        )

    def prepare_launch(
        self,
        prepared: PreparedAttempt,
        config: OptimizerSessionConfig,
    ) -> PreparedOptimizerLaunch:
        """Prepare Core paths and exact Runtime environment without starting the Agent."""
        phase = self._phases.prepare(prepared.root, prepared.session_root)
        attempt_report_path = prepared.root / "scratch/attempt-report.json"
        return PreparedOptimizerLaunch(
            phase=phase,
            environment=self._environment(
                prepared,
                phase,
                attempt_report_path,
                config,
            ),
            attempt_report_path=attempt_report_path,
        )

    def wrap_command(
        self,
        launch: PreparedOptimizerLaunch,
        runtime_argv: tuple[str, ...],
    ) -> tuple[str, ...]:
        """Apply the same production Sandbox boundary to an alternate dev-shell command."""
        environment = dict(launch.environment)
        environment["ATREX_DEV_SHELL_BACKENDS"] = "claude,codex,qodercli,pi"
        return self._launcher.wrap(
            runtime_argv,
            workspace=launch.phase.root,
            environment=environment,
            interactive=True,
        )

    def _environment(
        self,
        prepared: PreparedAttempt,
        phase: PreparedCorePhase,
        attempt_report_path: Path,
        config: OptimizerSessionConfig,
    ) -> Mapping[str, str]:
        if config.gateway_endpoint is None or config.gateway_capability is None:
            raise ValueError("Optimizer launch requires Attempt-scoped Gateway authority")
        environment = dict(config.environment)
        overlap = _RUNTIME_KEYS.intersection(environment)
        if overlap:
            raise ValueError(f"Optimizer environment overrides Runtime keys: {sorted(overlap)}")
        environment.update(
            self._phases.runtime_environment(
                phase,
                phase="optimization_attempt",
                model=config.model,
            )
        )
        environment.update(
            {
                "ATREX_ATTEMPT_MANIFEST": str(prepared.manifest_path),
                "ATREX_ATTEMPT_REPORT_PATH": str(attempt_report_path),
                "ATREX_EVIDENCE_PROMPT_PATH": str(
                    prepared.root / EVIDENCE_PROMPT_RELATIVE_PATH
                ),
                "ATREX_GATEWAY_CAPABILITY": config.gateway_capability,
                "ATREX_GATEWAY_PROXY_URL": config.gateway_endpoint,
            }
        )
        if config.wiki_endpoint is not None and config.wiki_capability is not None:
            environment.update(
                {
                    "ATREX_WIKI_CAPABILITY": config.wiki_capability,
                    "ATREX_WIKI_PROXY_URL": config.wiki_endpoint,
                }
            )
        validate_worker_environment(environment)
        return environment
