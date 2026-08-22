"""Versioned Worker-to-Gateway-Proxy JSON protocol."""

from __future__ import annotations

import base64
from copy import deepcopy
from typing import Annotated, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..artifacts.local import JsonValue
from ..domain.ids import (
    AttemptId,
    KernelRevisionId,
    parse_artifact_digest,
    parse_attempt_id,
    parse_kernel_revision_id,
)

GATEWAY_PROXY_PROTOCOL_VERSION: Literal[2] = 2


class CandidateFileV2(BaseModel):
    """One regular candidate file with a safe workspace-relative path."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    path: str
    content_base64: str

    @field_validator("path")
    @classmethod
    def _validate_path(cls, value: str) -> str:
        return _safe_relative_path(value, "candidate path")

    def content(self) -> bytes:
        """Decode strict Base64 at the wire boundary."""
        try:
            return base64.b64decode(self.content_base64, validate=True)
        except ValueError as error:
            raise ValueError(f"candidate file is not valid Base64: {self.path}") from error


class CandidateBundleV2(BaseModel):
    """Complete candidate source bundle uploaded by one Worker tool call."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    files: tuple[CandidateFileV2, ...]

    @model_validator(mode="after")
    def _validate_unique_paths(self) -> CandidateBundleV2:
        paths = [file.path for file in self.files]
        if not paths:
            raise ValueError("candidate bundle cannot be empty")
        if len(paths) != len(set(paths)):
            raise ValueError("candidate bundle contains duplicate paths")
        return self


class _GatewayRequestV2(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[2] = GATEWAY_PROXY_PROTOCOL_VERSION
    attempt_id: AttemptId
    idempotency_key: str = Field(min_length=1, max_length=200)

    @field_validator("attempt_id", mode="before")
    @classmethod
    def _validate_attempt_id(cls, value: object) -> AttemptId:
        if not isinstance(value, str):
            raise ValueError("attempt_id must be a string")
        return parse_attempt_id(value)


class _CandidateRequestV2(_GatewayRequestV2):
    candidate: CandidateBundleV2


class _DependencyRequestV2(_CandidateRequestV2):
    env_vars: dict[str, str] = Field(default_factory=dict)
    requirements: tuple[str, ...] = ()
    deps_mode: Literal["freeze_installed", "no_deps"] | None = None


class EvaluateRequestV2(_CandidateRequestV2):
    """Seal and evaluate one candidate against the trusted Evaluation Contract."""

    operation: Literal["evaluate"]


class SubmitRequestV2(_CandidateRequestV2):
    """Submit a complete EvalRequest document from the candidate bundle."""

    operation: Literal["submit"]
    payload_path: str

    @field_validator("payload_path")
    @classmethod
    def _validate_payload_path(cls, value: str) -> str:
        return _safe_relative_path(value, "payload_path")


class ProfileRequestV2(_DependencyRequestV2):
    """Run an Agate profiler over the current candidate."""

    operation: Literal["profile"]
    level: Literal["survey", "sol", "deep"] = "sol"
    profiler: Literal["ncu", "rocprofv3"] | None = None
    counters: tuple[str, ...] = ()
    kernel_regex: str | None = Field(default=None, max_length=500)
    kernel_name: str | None = Field(default=None, max_length=500)
    source: bool = False
    launch_skip: int | None = Field(default=None, ge=0)
    launch_count: int | None = Field(default=None, gt=0)
    top_kernels: int | None = Field(default=None, gt=0)
    shape_id: str | None = Field(default=None, min_length=1, max_length=200)

    @model_validator(mode="after")
    def _validate_kernel_selection(self) -> ProfileRequestV2:
        if self.kernel_regex is not None and self.kernel_name is not None:
            raise ValueError("profile kernel_regex and kernel_name are mutually exclusive")
        if self.level == "deep" and self.kernel_regex is None and self.kernel_name is None:
            raise ValueError("deep profile requires kernel_regex or kernel_name")
        return self


class DevRequestV2(_CandidateRequestV2):
    """Run an arbitrary command with the candidate bundle in a recycled GPU pod."""

    operation: Literal["dev"]
    command: str = Field(min_length=1, max_length=16_384)
    env_vars: dict[str, str] = Field(default_factory=dict)
    job_timeout_s: int | None = Field(default=None, gt=0, le=10_800)
    recycle: bool = True
    intent: (
        Literal[
            "workspace",
            "scratch_exec",
            "inspect",
            "compile",
            "profile_adhoc",
            "sanitize",
            "custom_harness",
            "other",
        ]
        | None
    ) = None
    note: str | None = Field(default=None, max_length=200)


class CheckRequestV2(_DependencyRequestV2):
    """Compile or sanitize the current candidate on the target GPU."""

    operation: Literal["check"]
    arch: str | None = Field(default=None, max_length=100)
    sanitize: Literal["memcheck", "racecheck", "initcheck", "synccheck"] | None = None


class SolRequestV2(_DependencyRequestV2):
    """Evaluate a SOL-ExecBench solution document from the candidate bundle."""

    operation: Literal["sol"]
    solution_path: str
    subset: Literal["L1", "L2", "Quant", "FlashInfer-Bench"] | None = None
    definition_path: str | None = None
    workload_path: str | None = None
    job_timeout_s: int | None = Field(default=None, gt=0, le=10_800)
    workload_timeout_s: int | None = Field(default=None, gt=0)
    compile_timeout_s: int | None = Field(default=None, gt=0)
    iterations: int | None = Field(default=None, gt=0)
    warmup_runs: int | None = Field(default=None, ge=0)
    lock_clocks: bool = True
    benchmark_reference: bool = False

    @field_validator("solution_path", "definition_path", "workload_path")
    @classmethod
    def _validate_source_path(cls, value: str | None) -> str | None:
        return None if value is None else _safe_relative_path(value, "SOL source path")

    @model_validator(mode="after")
    def _validate_problem(self) -> SolRequestV2:
        if (self.definition_path is None) is not (self.workload_path is None):
            raise ValueError("SOL custom problem requires definition_path and workload_path")
        if self.subset is not None and self.definition_path is not None:
            raise ValueError("SOL subset cannot be combined with a custom problem")
        return self


class DisassembleRequestV2(_DependencyRequestV2):
    """Compile and return GPU assembly for the current candidate."""

    operation: Literal["disassemble"]
    fmt: Literal["sass", "ptx", "auto"] = "auto"


class PollRequestV2(_GatewayRequestV2):
    """Read or long-poll a Gateway job owned by this Attempt."""

    operation: Literal["poll"]
    job_id: str = Field(min_length=1, max_length=200)
    wait: bool = False
    include_spec: bool = False


class JobsRequestV2(_GatewayRequestV2):
    """List jobs submitted by this Attempt."""

    operation: Literal["jobs"]
    kind: Literal["eval", "profile", "dev", "compile", "sol", "disassemble"] | None = None
    status: Literal["queued", "running", "succeeded", "failed", "cancelled"] | None = None
    limit: int = Field(default=50, gt=0, le=200)


class CancelRequestV2(_GatewayRequestV2):
    """Cancel a Gateway job owned by this Attempt."""

    operation: Literal["cancel"]
    job_id: str = Field(min_length=1, max_length=200)


class EnvRequestV2(_GatewayRequestV2):
    """Read selectable GPU environments or one environment's capabilities."""

    operation: Literal["env"]
    gpu: str | None = Field(default=None, min_length=1, max_length=200)
    capabilities: bool = False
    force: bool = False

    @model_validator(mode="after")
    def _validate_capabilities_target(self) -> EnvRequestV2:
        if self.capabilities and self.gpu is None:
            raise ValueError("env capabilities requires gpu")
        return self


class HealthRequestV2(_GatewayRequestV2):
    """Read external Agate Gateway liveness."""

    operation: Literal["health"]


class ConfigRequestV2(_GatewayRequestV2):
    """Read the Runtime-owned non-secret Agate connection configuration."""

    operation: Literal["config"]


class MeasurementsRequestV2(_GatewayRequestV2):
    """Query normalized measurements visible to this Attempt's lineage position."""

    operation: Literal["measurements"]
    kind: Literal["evaluate", "profile"] | None = None
    kernel_revision_id: KernelRevisionId | None = None
    kernel_artifact_digest: str | None = Field(default=None, max_length=80)
    shape_id: str | None = Field(default=None, min_length=1, max_length=200)
    kernel_name: str | None = Field(default=None, min_length=1, max_length=500)
    metric: str | None = Field(default=None, min_length=1, max_length=200)
    limit: int = Field(default=50, gt=0, le=200)

    @field_validator("kernel_revision_id", mode="before")
    @classmethod
    def _validate_kernel_revision_id(cls, value: object) -> KernelRevisionId | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError("kernel_revision_id must be a string")
        return parse_kernel_revision_id(value)

    @field_validator("kernel_artifact_digest")
    @classmethod
    def _validate_kernel_artifact_digest(cls, value: str | None) -> str | None:
        return None if value is None else str(parse_artifact_digest(value))


class KernelTrialsRequestV2(_GatewayRequestV2):
    """List exact experimental Kernel snapshots visible to this Attempt."""

    operation: Literal["kernel_trials"]
    decision: Literal["observed", "continue", "revert", "pivot"] | None = None
    limit: int = Field(default=50, gt=0, le=200)


class KernelTrialReadRequestV2(_GatewayRequestV2):
    """Read the file index or one exact source file from a visible Kernel Trial."""

    operation: Literal["kernel_trial_read"]
    kernel_trial_id: str = Field(pattern=r"^gtrial_[0-9a-f]{32}$")
    file: str | None = None

    @field_validator("file")
    @classmethod
    def _validate_file(cls, value: str | None) -> str | None:
        return None if value is None else _safe_relative_path(value, "Kernel Trial file")


type GatewayProxyRequestV2 = Annotated[
    EvaluateRequestV2
    | SubmitRequestV2
    | ProfileRequestV2
    | DevRequestV2
    | CheckRequestV2
    | SolRequestV2
    | DisassembleRequestV2
    | PollRequestV2
    | JobsRequestV2
    | CancelRequestV2
    | EnvRequestV2
    | HealthRequestV2
    | ConfigRequestV2
    | MeasurementsRequestV2
    | KernelTrialsRequestV2
    | KernelTrialReadRequestV2,
    Field(discriminator="operation"),
]


GATEWAY_AGENT_SCHEMA_VERSION: Literal[1] = 1
_RUNTIME_OWNED_REQUEST_FIELDS = frozenset({"schema_version", "attempt_id", "candidate"})
_RUNTIME_DEFAULTED_REQUEST_FIELDS = frozenset({"idempotency_key"})
_GATEWAY_REQUEST_MODELS: dict[str, type[_GatewayRequestV2]] = {
    "evaluate": EvaluateRequestV2,
    "submit": SubmitRequestV2,
    "profile": ProfileRequestV2,
    "dev": DevRequestV2,
    "check": CheckRequestV2,
    "sol": SolRequestV2,
    "disassemble": DisassembleRequestV2,
    "poll": PollRequestV2,
    "jobs": JobsRequestV2,
    "cancel": CancelRequestV2,
    "env": EnvRequestV2,
    "health": HealthRequestV2,
    "config": ConfigRequestV2,
    "measurements": MeasurementsRequestV2,
    "kernel_trials": KernelTrialsRequestV2,
    "kernel_trial_read": KernelTrialReadRequestV2,
}


def gateway_agent_request_schema(
    operation: str | None = None,
    *,
    allowed_operations: frozenset[str] | None = None,
) -> dict[str, JsonValue]:
    """Project the live wire models into the request shape accepted from an Agent.

    Runtime-owned identity and Candidate fields are attached by ``gateway-execute`` and are
    deliberately absent. Every remaining field, default, enum, bound, and extra-field policy is
    generated from the exact Pydantic model used by the Gateway Proxy.
    """
    if operation is not None and operation not in _GATEWAY_REQUEST_MODELS:
        raise ValueError(f"unsupported Gateway operation: {operation}")
    names = (operation,) if operation is not None else tuple(_GATEWAY_REQUEST_MODELS)
    if allowed_operations is not None:
        names = tuple(name for name in names if name in allowed_operations)
        if operation is not None and not names:
            raise PermissionError(f"Gateway operation is not allowed: {operation}")
    operations = {name: _agent_operation_schema(_GATEWAY_REQUEST_MODELS[name]) for name in names}
    return cast(
        dict[str, JsonValue],
        {
            "schema_version": GATEWAY_AGENT_SCHEMA_VERSION,
            "gateway_protocol_version": GATEWAY_PROXY_PROTOCOL_VERSION,
            "request_contract": "gateway-execute",
            "runtime_owned_fields": sorted(_RUNTIME_OWNED_REQUEST_FIELDS),
            "runtime_defaulted_fields": sorted(_RUNTIME_DEFAULTED_REQUEST_FIELDS),
            "operations": operations,
        },
    )


def _agent_operation_schema(model: type[_GatewayRequestV2]) -> dict[str, object]:
    schema = deepcopy(model.model_json_schema(mode="validation"))
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        raise TypeError(f"Gateway request schema has no properties: {model.__name__}")
    for field_name in _RUNTIME_OWNED_REQUEST_FIELDS:
        properties.pop(field_name, None)
    required = schema.get("required")
    if isinstance(required, list):
        schema["required"] = [
            field_name
            for field_name in required
            if field_name not in (_RUNTIME_OWNED_REQUEST_FIELDS | _RUNTIME_DEFAULTED_REQUEST_FIELDS)
        ]
    # CandidateBundleV2 is the only nested definition and becomes unreachable after projection.
    schema.pop("$defs", None)
    schema["title"] = model.__name__.removesuffix("RequestV2") + " gateway-execute request"
    return schema


class EvaluationV2(BaseModel):
    """Correctness and comparable latency returned by the external Gateway."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    correct: bool
    latency_us: float | None

    @model_validator(mode="after")
    def _validate_latency(self) -> EvaluationV2:
        if self.correct and (self.latency_us is None or self.latency_us <= 0):
            raise ValueError("a correct evaluation requires a positive latency")
        if not self.correct and self.latency_us is not None:
            raise ValueError("an incorrect evaluation cannot carry latency")
        return self


class GatewayProxyResponseV2(BaseModel):
    """Canonical result returned to an Optimizer binding and recorded in its trace."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[2] = GATEWAY_PROXY_PROTOCOL_VERSION
    operation: Literal[
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
        "measurements",
        "kernel_trials",
        "kernel_trial_read",
    ]
    status: Literal["completed", "queued", "failed", "cancelled"]
    candidate_artifact_digest: str | None = None
    gateway_result_digest: str
    job_id: str | None = None
    evaluation: EvaluationV2 | None = None
    result: JsonValue


def _safe_relative_path(value: str, label: str) -> str:
    if not value or value.startswith("/") or "\\" in value or "\x00" in value:
        raise ValueError(f"{label} must be non-empty, normalized, and relative")
    parts = value.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError(f"{label} contains an unsafe component")
    return value
