"""Strict deployment configuration for the trusted Runtime process."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Annotated, Literal, Self
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .domain.models import Dsl
from .kernel_agents import KernelAgentBundleLimits

RUNTIME_CONFIG_VERSION: Literal[1] = 1
_ENVIRONMENT_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
AgentBackend = Literal["claude", "codex", "qodercli", "pi"]
ReasoningEffort = Literal["low", "medium", "high", "max"]


class RuntimeServerSettings(BaseModel):
    """Socket settings for the Runtime ASGI server."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    host: str = Field(min_length=1)
    port: int = Field(ge=1, le=65535)


class RuntimeStorageSettings(BaseModel):
    """Durable Runtime state locations."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    registry_database: Path
    gateway_database: Path
    agate_jobs_database: Path
    artifacts_root: Path

    @model_validator(mode="after")
    def _validate_distinct_paths(self) -> RuntimeStorageSettings:
        paths = (
            self.registry_database.resolve(),
            self.gateway_database.resolve(),
            self.agate_jobs_database.resolve(),
            self.artifacts_root.resolve(),
        )
        if len(set(paths)) != len(paths):
            raise ValueError("Runtime storage locations must be distinct")
        database_parents = paths[:3]
        if any(path.parent == Path("/") for path in database_parents):
            raise ValueError("Runtime databases require a dedicated non-root parent directory")
        return self

    def resolve_from(self, base: Path) -> RuntimeStorageSettings:
        """Resolve all relative storage paths against the configuration directory."""
        values = {
            name: value if value.is_absolute() else (base / value).resolve()
            for name, value in (
                ("registry_database", self.registry_database),
                ("gateway_database", self.gateway_database),
                ("agate_jobs_database", self.agate_jobs_database),
                ("artifacts_root", self.artifacts_root),
            )
        }
        return RuntimeStorageSettings.model_validate(values)


class GatewayProxySettings(BaseModel):
    """Trusted Worker request acquisition limits."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_request_bytes: int = Field(gt=0)
    max_candidate_files: int = Field(gt=0)
    max_candidate_bytes: int = Field(gt=0)
    capability_signing_key_env: str = Field(min_length=1)
    candidate_diff_allowed_paths: dict[Dsl, tuple[str, ...]]
    candidate_diff_require_change: bool

    @model_validator(mode="after")
    def _validate_diff_policy(self) -> GatewayProxySettings:
        if set(self.candidate_diff_allowed_paths) != set(Dsl):
            raise ValueError("Gateway Candidate diff policy must define every DSL")
        if any(not patterns for patterns in self.candidate_diff_allowed_paths.values()):
            raise ValueError("Gateway Candidate diff allowlists cannot be empty")
        return self


class AgateSettings(BaseModel):
    """Agate endpoint, auth source names, and network wait policy."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    base_url: str
    auth_mode: Literal["none", "token", "ak_sk"]
    token_env: str | None = None
    access_key_env: str | None = None
    secret_key_env: str | None = None
    http_timeout_s: float = Field(gt=0)
    wait_timeout_s: float = Field(gt=0)
    health_check_interval_s: float = Field(default=30.0, gt=0)

    @model_validator(mode="after")
    def _validate_auth_sources(self) -> AgateSettings:
        if not self.base_url.startswith(("http://", "https://")):
            raise ValueError("Agate base_url must use HTTP or HTTPS")
        sources = (self.token_env, self.access_key_env, self.secret_key_env)
        if self.auth_mode == "none" and any(sources):
            raise ValueError("Agate no-auth mode cannot name credential environment variables")
        if self.auth_mode == "token" and (
            not self.token_env or self.access_key_env or self.secret_key_env
        ):
            raise ValueError("Agate token mode requires only token_env")
        if self.auth_mode == "ak_sk" and (
            self.token_env or not self.access_key_env or not self.secret_key_env
        ):
            raise ValueError("Agate ak_sk mode requires access_key_env and secret_key_env")
        return self


class GitOptimizerBaseSettings(BaseModel):
    """Approved Core repository and trusted Git import process settings."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    repository: str = Field(min_length=1)
    git_executable: Path
    fetch_timeout_seconds: float = Field(gt=0)
    max_archive_bytes: int = Field(gt=0)
    allowed_submodules: dict[str, str] = Field(default_factory=dict)

    def resolve_from(self, base: Path) -> GitOptimizerBaseSettings:
        """Resolve local repositories and Git executable from the configuration directory."""
        executable = self.git_executable
        repository = self.repository
        if repository.startswith(("./", "../", "/")):
            repository_path = Path(repository)
            repository = str(
                repository_path
                if repository_path.is_absolute()
                else (base / repository_path).resolve()
            )
        return self.model_copy(
            update={
                "repository": repository,
                "git_executable": (
                    executable if executable.is_absolute() else (base / executable).resolve()
                ),
            }
        )


class KernelAgentSettings(BaseModel):
    """Trusted validation and loader limits for evolvable Kernel Agent Bundles."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_bundle_files: int = Field(gt=0)
    max_bundle_bytes: int = Field(gt=0)
    max_entrypoint_bytes: int = Field(gt=0)
    max_agent_problem_bytes: int = Field(gt=0)
    base_source: GitOptimizerBaseSettings | None = None

    def bundle_limits(self) -> KernelAgentBundleLimits:
        """Create the revision-layer value without coupling that layer to configuration."""
        return KernelAgentBundleLimits(
            max_bundle_files=self.max_bundle_files,
            max_bundle_bytes=self.max_bundle_bytes,
            max_entrypoint_bytes=self.max_entrypoint_bytes,
        )

    def resolve_from(self, base: Path) -> KernelAgentSettings:
        """Resolve the optional trusted Git importer executable."""
        return self.model_copy(
            update={
                "base_source": (
                    None if self.base_source is None else self.base_source.resolve_from(base)
                )
            }
        )


class GpuWikiSettings(BaseModel):
    """External knowledge endpoint plus Worker proxy limits."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    base_url: str
    bearer_token_env: str | None = None
    timeout_seconds: float = Field(gt=0)
    max_proxy_request_bytes: int = Field(gt=0)
    max_query_bytes: int = Field(gt=0)
    max_response_bytes: int = Field(gt=0)

    @model_validator(mode="after")
    def _validate_endpoint(self) -> GpuWikiSettings:
        if not self.base_url.startswith(("http://", "https://")):
            raise ValueError("GPU Wiki base_url must use HTTP or HTTPS")
        if (
            self.bearer_token_env is not None
            and _ENVIRONMENT_KEY.fullmatch(self.bearer_token_env) is None
        ):
            raise ValueError("GPU Wiki bearer token environment name is invalid")
        return self


class WorkerEnvironmentSettings(BaseModel):
    """Exact static, required-inherited, and optional-inherited worker values."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    values: dict[str, str] = Field(default_factory=dict)
    inherit: tuple[str, ...] = ()
    inherit_optional: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _validate_entries(self) -> WorkerEnvironmentSettings:
        keys = (*self.values, *self.inherit, *self.inherit_optional)
        invalid = sorted(key for key in keys if _ENVIRONMENT_KEY.fullmatch(key) is None)
        if invalid:
            raise ValueError(f"invalid worker environment keys: {invalid}")
        if len(self.inherit) != len(set(self.inherit)):
            raise ValueError("worker inherited environment keys cannot repeat")
        if len(self.inherit_optional) != len(set(self.inherit_optional)):
            raise ValueError("worker optional inherited environment keys cannot repeat")
        overlap = set(self.values).intersection((*self.inherit, *self.inherit_optional))
        overlap.update(set(self.inherit).intersection(self.inherit_optional))
        if overlap:
            raise ValueError(
                f"worker environment keys cannot be static and inherited: {sorted(overlap)}"
            )
        nul_keys = sorted(key for key, value in self.values.items() if "\x00" in value)
        if nul_keys:
            raise ValueError(f"worker environment values contain NUL: {nul_keys}")
        return self

    def resolve(self, ambient: Mapping[str, str]) -> tuple[tuple[str, str], ...]:
        """Resolve only named ambient variables and return deterministic pairs."""
        missing = sorted(key for key in self.inherit if not ambient.get(key))
        if missing:
            raise ValueError(f"required worker environment variables are missing: {missing}")
        resolved = dict(self.values)
        resolved.update((key, ambient[key]) for key in self.inherit)
        resolved.update((key, ambient[key]) for key in self.inherit_optional if ambient.get(key))
        return tuple(sorted(resolved.items()))


class CoreOptimizerWorkerSettings(BaseModel):
    """Launcher for the sole Optimizer entrypoint declared by the Core repository."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    command_prefix: tuple[str, ...] = Field(min_length=1)
    agent_backend: AgentBackend = "qodercli"
    reasoning_effort: ReasoningEffort = "max"
    session_settings: str = ""
    environment: WorkerEnvironmentSettings = Field(default_factory=WorkerEnvironmentSettings)
    isolated_home_environment_keys: tuple[str, ...] = ("HOME",)
    session_trace_relative_path: str | None = None
    token_usage_report_relative_path: str
    max_attempt_report_bytes: int = Field(gt=0)
    timeout_seconds: float = Field(gt=0)
    bootstrap_timeout_seconds: float = Field(default=10_800.0, gt=0)
    terminate_grace_seconds: float = Field(gt=0)
    max_diagnostic_bytes: int = Field(gt=0)
    max_session_tokens: int = Field(gt=0)
    max_session_credits: float = Field(default=1_000_000.0, gt=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def _validate_launcher(self) -> CoreOptimizerWorkerSettings:
        if not self.command_prefix[0]:
            raise ValueError("Optimizer command-prefix executable cannot be empty")
        if any("\x00" in argument for argument in self.command_prefix):
            raise ValueError("Optimizer command prefix cannot contain NUL")
        if "\x00" in self.session_settings:
            raise ValueError("Optimizer session settings cannot contain NUL")
        invalid = sorted(
            key
            for key in self.isolated_home_environment_keys
            if _ENVIRONMENT_KEY.fullmatch(key) is None
        )
        if invalid or len(self.isolated_home_environment_keys) != len(
            set(self.isolated_home_environment_keys)
        ):
            raise ValueError("Optimizer isolated-home environment keys are invalid")
        configured = set(self.environment.values).union(
            self.environment.inherit,
            self.environment.inherit_optional,
        )
        overlap = configured.intersection(self.isolated_home_environment_keys)
        if overlap:
            raise ValueError(
                f"Optimizer environment overrides isolated home keys: {sorted(overlap)}"
            )
        for label, value in (
            ("session trace", self.session_trace_relative_path),
            ("token usage report", self.token_usage_report_relative_path),
        ):
            if value is None:
                continue
            relative = PurePosixPath(value)
            if relative.is_absolute() or relative.as_posix() == "." or ".." in relative.parts:
                raise ValueError(f"Optimizer {label} path must be safe and relative")
            if relative.parts[0] not in {"sessions", "scratch"}:
                raise ValueError(f"Optimizer {label} path must be under sessions or scratch")
        return self

    def resolve_from(self, base: Path) -> CoreOptimizerWorkerSettings:
        """Resolve only the deployment-owned command executable."""
        executable = Path(self.command_prefix[0])
        if not executable.is_absolute():
            executable = (base / executable).resolve()
        return self.model_copy(
            update={"command_prefix": (str(executable), *self.command_prefix[1:])}
        )


class EvolverWorkerSettings(BaseModel):
    """Commit-anchored Evolver repository and process limits."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    repository: str = Field(min_length=1)
    commit: str = Field(pattern=r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
    git_executable: Path
    fetch_timeout_seconds: float = Field(gt=0)
    max_archive_bytes: int = Field(gt=0)
    command_prefix: tuple[str, ...] = Field(min_length=1)
    agent_backend: AgentBackend = "qodercli"
    reasoning_effort: ReasoningEffort = "max"
    session_settings: str = ""
    max_bundle_files: int = Field(default=1024, gt=0)
    max_bundle_bytes: int = Field(default=8388608, gt=0)
    isolated_home_environment_keys: tuple[str, ...] = ()
    session_trace_relative_path: str | None = None
    token_usage_report_relative_path: str
    environment: WorkerEnvironmentSettings = Field(default_factory=WorkerEnvironmentSettings)
    timeout_seconds: float = Field(gt=0)
    terminate_grace_seconds: float = Field(gt=0)
    max_diagnostic_bytes: int = Field(gt=0)
    max_output_manifest_bytes: int = Field(gt=0)

    @model_validator(mode="after")
    def _validate_command(self) -> EvolverWorkerSettings:
        if not self.command_prefix[0]:
            raise ValueError("Evolver command prefix executable cannot be empty")
        if any("\x00" in argument for argument in self.command_prefix):
            raise ValueError("Evolver command arguments cannot contain NUL")
        if "\x00" in self.session_settings:
            raise ValueError("Evolver session settings cannot contain NUL")
        invalid = sorted(
            key
            for key in self.isolated_home_environment_keys
            if _ENVIRONMENT_KEY.fullmatch(key) is None
        )
        if invalid:
            raise ValueError(f"invalid Evolver home environment keys: {invalid}")
        if len(self.isolated_home_environment_keys) != len(
            set(self.isolated_home_environment_keys)
        ):
            raise ValueError("Evolver home environment keys cannot repeat")
        if self.session_trace_relative_path is not None:
            relative = PurePosixPath(self.session_trace_relative_path)
            if relative.is_absolute() or relative.as_posix() == "." or ".." in relative.parts:
                raise ValueError("Evolver session trace path must be a safe relative path")
        usage_relative = PurePosixPath(self.token_usage_report_relative_path)
        if (
            usage_relative.is_absolute()
            or usage_relative.as_posix() == "."
            or ".." in usage_relative.parts
            or usage_relative.parts[0] != "scratch"
        ):
            raise ValueError("Evolver token usage report path must be under the scratch directory")
        configured = set(self.environment.values).union(
            self.environment.inherit,
            self.environment.inherit_optional,
        )
        overlap = configured.intersection(self.isolated_home_environment_keys)
        if overlap:
            raise ValueError(f"Evolver environment overrides isolated home keys: {sorted(overlap)}")
        return self

    def resolve_from(self, base: Path) -> EvolverWorkerSettings:
        """Resolve a local repository plus trusted executables from the config directory."""
        executable = Path(self.command_prefix[0])
        if not executable.is_absolute():
            executable = (base / executable).resolve()
        git_executable = self.git_executable
        if not git_executable.is_absolute():
            git_executable = (base / git_executable).resolve()
        repository = self.repository
        if repository.startswith(("./", "../", "/")):
            repository_path = Path(repository)
            repository = str(
                repository_path
                if repository_path.is_absolute()
                else (base / repository_path).resolve()
            )
        return self.model_copy(
            update={
                "repository": repository,
                "git_executable": git_executable,
                "command_prefix": (str(executable), *self.command_prefix[1:]),
            }
        )


class CgroupResourceSettings(BaseModel):
    """Per-Session cgroup v2 resource limits applied through systemd."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    memory_max_bytes: int = Field(gt=0)
    memory_swap_max_bytes: int = Field(ge=0)
    cpu_quota_percent: int = Field(gt=0)
    tasks_max: int = Field(gt=0)


class BwrapSandboxSettings(BaseModel):
    """Strict Linux mount namespace and cgroup policy with host networking."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    bwrap_executable: Path
    systemd_run_executable: Path
    systemd_user: Literal[False] = False
    worker_user: str = Field(
        min_length=1,
        pattern=r"^[a-z_][a-z0-9_-]*[$]?$",
    )
    sandbox_home: PurePosixPath = PurePosixPath("/home/agent")
    workspace_mount: PurePosixPath = PurePosixPath("/home/agent/workspace")
    resolv_conf: Path = Path("/etc/resolv.conf")
    read_only_bind_paths: tuple[Path, ...] = ()
    hidden_host_paths: tuple[Path, ...] = ()
    resources: CgroupResourceSettings

    @model_validator(mode="after")
    def _validate_sandbox(self) -> BwrapSandboxSettings:
        if not self.sandbox_home.is_absolute() or self.sandbox_home.as_posix() == "/":
            raise ValueError("Sandbox home must be an absolute non-root path")
        if not self.workspace_mount.is_absolute():
            raise ValueError("Sandbox workspace mount must be absolute")
        try:
            self.workspace_mount.relative_to(self.sandbox_home)
        except ValueError as error:
            raise ValueError("Sandbox workspace must be mounted below sandbox home") from error
        if self.workspace_mount.name != "workspace":
            raise ValueError("Sandbox workspace must be exposed as ~/workspace")
        for label, paths in (
            ("read-only bind", self.read_only_bind_paths),
            ("hidden host", self.hidden_host_paths),
        ):
            if any(not path.is_absolute() for path in paths):
                raise ValueError(f"Sandbox {label} paths must be absolute")
            if len(paths) != len(set(paths)):
                raise ValueError(f"Sandbox {label} paths cannot repeat")
            if any(path.is_relative_to("/run") for path in paths):
                raise ValueError(f"Sandbox {label} paths cannot be below /run")
        return self

    def resolve_from(self, base: Path) -> BwrapSandboxSettings:
        """Resolve trusted host executable and mount paths from the config directory."""

        def resolve(path: Path) -> Path:
            return path if path.is_absolute() else (base / path).resolve()

        return self.model_copy(
            update={
                "bwrap_executable": resolve(self.bwrap_executable),
                "systemd_run_executable": resolve(self.systemd_run_executable),
                "resolv_conf": resolve(self.resolv_conf),
                "read_only_bind_paths": tuple(resolve(path) for path in self.read_only_bind_paths),
                "hidden_host_paths": tuple(resolve(path) for path in self.hidden_host_paths),
            }
        )


class BackendCredentialSettings(BaseModel):
    """Host login-state mounts shared read-only with coding-agent CLIs."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    enabled: bool = True
    host_home: Path | None = None
    development_bwrap_executable: Path = Path("/usr/bin/bwrap")

    def resolve_from(self, base: Path) -> BackendCredentialSettings:
        def resolve(path: Path) -> Path:
            return path if path.is_absolute() else (base / path).resolve()

        return self.model_copy(
            update={
                "host_home": (None if self.host_home is None else resolve(self.host_home)),
                "development_bwrap_executable": resolve(self.development_bwrap_executable),
            }
        )


class WorkerLaunchSettings(BaseModel):
    """Explicit development launcher or mandatory Linux production sandbox."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    mode: Literal["development", "sandbox"]
    env_executable: Path
    backend_credentials: BackendCredentialSettings = Field(
        default_factory=BackendCredentialSettings
    )
    sandbox: BwrapSandboxSettings | None = None

    @model_validator(mode="after")
    def _validate_mode(self) -> WorkerLaunchSettings:
        if (self.mode == "sandbox") != (self.sandbox is not None):
            raise ValueError("launcher.sandbox must be set exactly in sandbox mode")
        return self

    def resolve_from(self, base: Path) -> WorkerLaunchSettings:
        """Resolve trusted launcher executables and Sandbox host paths."""

        def resolve(path: Path) -> Path:
            return path if path.is_absolute() else (base / path).resolve()

        return self.model_copy(
            update={
                "env_executable": resolve(self.env_executable),
                "backend_credentials": self.backend_credentials.resolve_from(base),
                "sandbox": None if self.sandbox is None else self.sandbox.resolve_from(base),
            }
        )


class EvidenceProjectionSettings(BaseModel):
    """Limits and deployment-specific redactions for derived Evidence."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_trace_files: int = Field(gt=0)
    max_trace_bytes: int = Field(gt=0)
    max_trace_events: int = Field(gt=0)
    max_projection_text_bytes: int = Field(gt=0)
    max_diff_files: int = Field(gt=0)
    max_diff_bytes: int = Field(gt=0)
    redaction_patterns: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _validate_patterns(self) -> EvidenceProjectionSettings:
        for pattern in self.redaction_patterns:
            try:
                re.compile(pattern)
            except re.error as error:
                raise ValueError(f"invalid Evidence redaction pattern: {pattern!r}") from error
        return self


class EvaluateComparisonSettings(BaseModel):
    """Comparison over the authoritative aggregate from ordinary Gateway evaluate."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    method: Literal["evaluate"]
    repeats: int = Field(default=1, gt=0)
    measurement_uncertainty_us: float = Field(ge=0)


class SameAllocationAbbaComparisonSettings(BaseModel):
    """Exact interleaved comparison executed inside one allocation per Shape batch."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    method: Literal["same_allocation_abba"]
    repeats: int = Field(default=2, gt=0)
    minimum_improvement_percent: float = Field(default=0.0, ge=0, lt=100)
    allocation_timeout_seconds: int = Field(default=600, gt=0)
    shape_batch_size: int = Field(default=1, gt=0)
    max_parallel_shape_batches: int = Field(default=4, gt=0)


ComparisonSettings = Annotated[
    EvaluateComparisonSettings | SameAllocationAbbaComparisonSettings,
    Field(discriminator="method"),
]


class AtrexBenchComparisonEvaluatorSettings(BaseModel):
    """Commit-pinned Atrex Bench runtime uploaded into trusted comparison allocations."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    repository: str = Field(min_length=1)
    commit: str = Field(pattern=r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
    git_executable: Path
    fetch_timeout_seconds: float = Field(gt=0)
    max_archive_bytes: int = Field(gt=0)
    max_bundle_files: int = Field(default=128, gt=0)
    max_bundle_bytes: int = Field(default=4194304, gt=0)
    agate_package_version: str | None = Field(default=None, min_length=1)

    def resolve_from(self, base: Path) -> AtrexBenchComparisonEvaluatorSettings:
        """Resolve local repository and Git executable paths from the config file."""
        repository = self.repository
        if repository.startswith(("./", "../", "/")):
            path = Path(repository)
            repository = str(path if path.is_absolute() else (base / path).resolve())
        git = self.git_executable
        return self.model_copy(
            update={
                "repository": repository,
                "git_executable": git if git.is_absolute() else (base / git).resolve(),
            }
        )


class EvaluationGateSettings(BaseModel):
    """Correctness and performance sampling for one evaluation role."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    correctness_cases: int = Field(gt=0)
    bench_iters: int = Field(gt=0)


class OptimizerGateSettings(EvaluationGateSettings):
    """Optimizer exploratory evaluation policy."""

    evaluate_repeats: int = Field(default=1, gt=0)


class BootstrapGateStageSettings(BaseModel):
    """One sequential correctness stage in authoritative Bootstrap validation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    correctness_cases: int = Field(gt=0)
    evaluate_repeats: int = Field(default=1, gt=0)


class BootstrapGateSettings(BaseModel):
    """Ordered Bootstrap validation stages and their small timing sample."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    stages: tuple[BootstrapGateStageSettings, ...] = Field(min_length=1)
    bench_iters: int = Field(gt=0)


class GatePolicySettings(BaseModel):
    """Single trusted source of all correctness and performance Gate semantics."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    optimizer: OptimizerGateSettings
    bootstrap: BootstrapGateSettings
    retention: EvaluationGateSettings
    production_gate: bool = False
    warmup_iters: int = Field(default=10, gt=0)
    atol: float = Field(default=1e-2, ge=0)
    rtol: float = Field(default=0.05, ge=0)
    evaluation_timeout_seconds: int = Field(default=600, gt=0)
    candidate_timeout_seconds: float = Field(default=20.0, gt=0)
    performance_timeout_seconds: float = Field(default=120.0, gt=0)
    lock_clocks: bool = True
    evaluator: AtrexBenchComparisonEvaluatorSettings

    def resolve_from(self, base: Path) -> GatePolicySettings:
        return self.model_copy(update={"evaluator": self.evaluator.resolve_from(base)})


class AtrexBenchRooflineBuilderSettings(BaseModel):
    """Commit-pinned trusted Atrex Bench analytical Roofline generator."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    repository: str = Field(min_length=1)
    commit: str = Field(pattern=r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
    git_executable: Path
    python_executable: Path
    fetch_timeout_seconds: float = Field(gt=0)
    execution_timeout_seconds: float = Field(gt=0)
    max_archive_bytes: int = Field(gt=0)
    max_output_bytes: int = Field(gt=0)
    sku_by_hardware_target: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_skus(self) -> AtrexBenchRooflineBuilderSettings:
        if any(
            not key.strip() or not value.strip()
            for key, value in self.sku_by_hardware_target.items()
        ):
            raise ValueError("Roofline hardware target and SKU mappings cannot be empty")
        return self

    def resolve_from(self, base: Path) -> AtrexBenchRooflineBuilderSettings:
        """Resolve local repositories and trusted executables from the config directory."""
        repository = self.repository
        if repository.startswith(("./", "../", "/")):
            path = Path(repository)
            repository = str(path if path.is_absolute() else (base / path).resolve())

        def resolve(path: Path) -> Path:
            return path if path.is_absolute() else (base / path).resolve()

        return self.model_copy(
            update={
                "repository": repository,
                "git_executable": resolve(self.git_executable),
                "python_executable": resolve(self.python_executable),
            }
        )


class CampaignRuntimeSettings(BaseModel):
    """Trusted composition policy for Campaign scheduling and both worker roles."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    attempt_workspaces_root: Path
    evolution_workspaces_root: Path
    problem_generalization_workspaces_root: Path
    lineage_bootstrap_workspaces_root: Path
    fencing_lease_seconds: float = Field(gt=0)
    fencing_heartbeat_seconds: float = Field(gt=0)
    gateway_proxy_url: str
    gateway_operations: tuple[
        Literal[
            "evaluate",
            "submit",
            "profile",
            "dev",
            "check",
            "sol",
            "disassemble",
            "poll",
            "jobs",
            "cancel",
            "env",
            "health",
            "config",
        ],
        ...,
    ] = ("evaluate",)
    gateway_max_calls: int = Field(gt=0)
    gateway_capability_lifetime_seconds: float = Field(gt=0)
    gate_policy: GatePolicySettings
    max_infrastructure_retries: int = Field(ge=0)
    bootstrap_max_parallel_lineages: int = Field(default=1, gt=0)
    max_parallel_branches: int = Field(default=4, gt=0)
    roofline_builder: AtrexBenchRooflineBuilderSettings | None = None
    kernel_retention_comparison: ComparisonSettings
    agent_promotion_comparison: ComparisonSettings
    evidence: EvidenceProjectionSettings
    optimizer: CoreOptimizerWorkerSettings
    evolver: EvolverWorkerSettings
    launcher: WorkerLaunchSettings

    @model_validator(mode="after")
    def _validate_gateway(self) -> CampaignRuntimeSettings:
        if not self.gateway_proxy_url.startswith(("http://", "https://")):
            raise ValueError("Campaign Gateway Proxy URL must use HTTP or HTTPS")
        if not self.gateway_operations:
            raise ValueError("Campaign Gateway operations cannot be empty")
        if len(self.gateway_operations) != len(set(self.gateway_operations)):
            raise ValueError("Campaign Gateway operations cannot repeat")
        if "evaluate" not in self.gateway_operations:
            raise ValueError("Campaign Gateway operations must include evaluate")
        comparisons = (
            self.kernel_retention_comparison,
            self.agent_promotion_comparison,
        )
        for comparison in comparisons:
            if not isinstance(comparison, SameAllocationAbbaComparisonSettings):
                continue
            schedule_runs = comparison.repeats * 2
            required = self.gate_policy.performance_timeout_seconds * schedule_runs + 30
            if required > comparison.allocation_timeout_seconds:
                raise ValueError(
                    "same-allocation ABBA schedule cannot fit allocation_timeout_seconds"
                )
        if self.fencing_lease_seconds <= self.fencing_heartbeat_seconds * 2:
            raise ValueError("Campaign fencing lease must exceed two heartbeat periods")
        roots = (
            self.attempt_workspaces_root.resolve(),
            self.evolution_workspaces_root.resolve(),
            self.problem_generalization_workspaces_root.resolve(),
            self.lineage_bootstrap_workspaces_root.resolve(),
        )
        if any(root.resolve().is_relative_to("/run") for root in roots):
            raise ValueError("Sandbox Worker roots cannot be below /run")
        if len(set(roots)) != len(roots):
            raise ValueError("Campaign workspace roots must be distinct")
        return self

    def resolve_from(self, base: Path) -> CampaignRuntimeSettings:
        """Resolve all local paths against the configuration directory."""

        def resolve(path: Path) -> Path:
            return path if path.is_absolute() else (base / path).resolve()

        return self.model_copy(
            update={
                "attempt_workspaces_root": resolve(self.attempt_workspaces_root),
                "evolution_workspaces_root": resolve(self.evolution_workspaces_root),
                "problem_generalization_workspaces_root": resolve(
                    self.problem_generalization_workspaces_root
                ),
                "lineage_bootstrap_workspaces_root": resolve(
                    self.lineage_bootstrap_workspaces_root
                ),
                "optimizer": self.optimizer.resolve_from(base),
                "evolver": self.evolver.resolve_from(base),
                "launcher": self.launcher.resolve_from(base),
                "gate_policy": self.gate_policy.resolve_from(base),
                "roofline_builder": (
                    None
                    if self.roofline_builder is None
                    else self.roofline_builder.resolve_from(base)
                ),
            }
        )


class AdministrationSettings(BaseModel):
    """Authenticated task API and independent task-worker policy."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    bearer_token_env: str = Field(min_length=1)
    max_request_bytes: int = Field(gt=0)
    event_page_limit: int = Field(gt=0)
    event_export_limit: int = Field(gt=0)
    event_prune_limit: int = Field(gt=0)
    task_lease_seconds: float = Field(gt=0)
    task_heartbeat_seconds: float = Field(gt=0)
    task_poll_seconds: float = Field(gt=0)
    max_error_bytes: int = Field(gt=0)

    @model_validator(mode="after")
    def _validate_administration(self) -> AdministrationSettings:
        if _ENVIRONMENT_KEY.fullmatch(self.bearer_token_env) is None:
            raise ValueError("administration bearer token environment name is invalid")
        if self.task_lease_seconds <= self.task_heartbeat_seconds * 2:
            raise ValueError("task lease must exceed two heartbeat periods")
        return self


class RuntimeSettings(BaseModel):
    """Complete versioned Runtime server configuration."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1] = RUNTIME_CONFIG_VERSION
    server: RuntimeServerSettings
    storage: RuntimeStorageSettings
    gateway_proxy: GatewayProxySettings
    agate: AgateSettings
    kernel_agent: KernelAgentSettings
    administration: AdministrationSettings | None = None
    gpu_wiki: GpuWikiSettings | None = None
    gate_policy: GatePolicySettings | None = None
    campaign: CampaignRuntimeSettings | None = None

    @model_validator(mode="after")
    def _validate_sandbox(self) -> RuntimeSettings:
        campaign = self.campaign
        if (
            campaign is not None
            and self.gate_policy is not None
            and campaign.gate_policy != self.gate_policy
        ):
            raise ValueError("Runtime and Campaign Gate policies must match")
        if campaign is None:
            return self
        if campaign.launcher.mode == "development":
            return self
        sandbox = campaign.launcher.sandbox
        if sandbox is None:
            raise ValueError("Sandbox launcher requires Sandbox settings")

        gateway = urlsplit(campaign.gateway_proxy_url)
        if gateway.hostname is None:
            raise ValueError("Campaign Gateway URL has no host")
        if gateway.hostname != self.server.host or gateway.port != self.server.port:
            raise ValueError("Campaign Gateway URL must identify this Runtime API socket")
        roots = (
            campaign.attempt_workspaces_root,
            campaign.evolution_workspaces_root,
            campaign.problem_generalization_workspaces_root,
            campaign.lineage_bootstrap_workspaces_root,
        )
        if any(
            bind.resolve().is_relative_to(root.resolve())
            for bind in sandbox.read_only_bind_paths
            for root in roots
        ):
            raise ValueError("Sandbox read-only bind paths cannot expose Worker roots")
        runtime_storage_roots = (
            self.storage.artifacts_root,
            self.storage.registry_database.parent,
            self.storage.gateway_database.parent,
            self.storage.agate_jobs_database.parent,
        )
        if any(
            bind.resolve().is_relative_to(root.resolve())
            for bind in sandbox.read_only_bind_paths
            for root in runtime_storage_roots
        ):
            raise ValueError("Sandbox read-only bind paths cannot expose Runtime storage")
        return self

    @classmethod
    def from_file(cls, path: str | Path) -> Self:
        """Parse strict JSON and resolve storage paths relative to its directory."""
        config_path = Path(path).resolve()
        try:
            value = json.loads(config_path.read_bytes())
        except json.JSONDecodeError as error:
            raise ValueError(f"Runtime configuration is not valid JSON: {config_path}") from error
        settings = cls.model_validate(value)
        campaign = settings.campaign
        resolved = settings.model_copy(
            update={
                "storage": settings.storage.resolve_from(config_path.parent),
                "kernel_agent": settings.kernel_agent.resolve_from(config_path.parent),
                "gate_policy": (
                    settings.gate_policy.resolve_from(config_path.parent)
                    if settings.gate_policy is not None
                    else None
                ),
                "campaign": (
                    campaign.resolve_from(config_path.parent) if campaign is not None else None
                ),
            }
        )
        return cls.model_validate(resolved.model_dump(mode="python"))
