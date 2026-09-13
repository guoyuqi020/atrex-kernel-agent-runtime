"""Versioned Worker-to-Gateway-Proxy JSON protocol."""

from __future__ import annotations

import base64
from copy import deepcopy
from typing import Annotated, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..artifacts.local import JsonValue
from ..domain.ids import (
    AttemptId,
    parse_artifact_digest,
    parse_attempt_id,
)
from ..workers.attempt_report import AttemptReportV12

GATEWAY_PROXY_PROTOCOL_VERSION: Literal[2] = 2

# Agate rejects a larger per-job timeout with an unrecoverable 422.
AGATE_MAX_JOB_TIMEOUT_S = 600


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


class EvaluateComparisonV2(BaseModel):
    """Optional paired measurement strategy; source uploads stay at the wire boundary."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    method: Literal["abba"]
    repeats: int = Field(default=2, ge=2, le=20)


class EvaluateParametersV2(BaseModel):
    """Optional exploratory overrides; the Campaign contract remains immutable."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    input_py: str | None = Field(
        default=None,
        min_length=1,
        max_length=131_072,
        description=(
            "Python source defining _make_inputs(**input_kwargs); return a dictionary whose keys "
            "match Model.forward arguments. Shape input_kwargs supply the generator parameters."
        ),
    )
    shapes: dict[str, JsonValue] | None = Field(
        default=None,
        description=(
            "Non-empty object keyed by numeric case IDs. Each record has input_kwargs for "
            "_make_inputs(**input_kwargs) and init_kwargs for Model construction (null or {} "
            "when no constructor arguments are needed). These are custom cases, not hidden IDs."
        ),
    )
    mode: Literal["full", "correctness_only"] = "full"
    comparison: EvaluateComparisonV2 | None = None

    @model_validator(mode="after")
    def _validate_comparison_mode(self) -> EvaluateParametersV2:
        if self.comparison is not None and self.mode != "full":
            raise ValueError("comparison requires mode='full'; correctness_only has no timing")
        return self

    @field_validator("input_py")
    @classmethod
    def _validate_input_py(cls, value: str | None) -> str | None:
        if value is not None and (not value.strip() or len(value.encode("utf-8")) > 131_072):
            raise ValueError("input_py must be nonblank UTF-8 source of at most 128 KiB")
        return value

    @field_validator("shapes")
    @classmethod
    def _validate_shapes(cls, value: dict[str, JsonValue] | None) -> dict[str, JsonValue] | None:
        if value is None:
            return None
        if not value:
            raise ValueError("custom shapes must be non-empty")
        for shape_id, shape in value.items():
            try:
                int(shape_id)
            except ValueError as error:
                raise ValueError("custom shape IDs must be integer-parseable strings") from error
            if not isinstance(shape, dict):
                raise ValueError("each custom shape must be a JSON object")
        return value

    @property
    def input_scope(self) -> Literal["contract", "custom"]:
        return "custom" if self.input_py is not None or self.shapes is not None else "contract"

    @property
    def is_contract_evaluation(self) -> bool:
        return self.mode == "full" and self.input_scope == "contract" and self.comparison is None


class EvaluateRequestV2(_CandidateRequestV2, EvaluateParametersV2):
    """Evaluate a sealed candidate with contract inputs or explicit exploratory overrides."""

    operation: Literal["evaluate"]
    baseline: CandidateBundleV2 | None = None

    @model_validator(mode="after")
    def _validate_baseline(self) -> EvaluateRequestV2:
        if self.comparison is not None and self.baseline is None:
            raise ValueError(
                "comparison requires a baseline source uploaded from comparison.baseline_path"
            )
        if self.comparison is None and self.baseline is not None:
            raise ValueError("baseline is only allowed when comparison is specified")
        return self


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
    files: tuple[CandidateFileV2, ...] = ()
    env_vars: dict[str, str] = Field(default_factory=dict)
    job_timeout_s: int | None = Field(default=None, gt=0, le=AGATE_MAX_JOB_TIMEOUT_S)
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

    @model_validator(mode="after")
    def _validate_files(self) -> DevRequestV2:
        paths = [file.path for file in self.files]
        if len(paths) != len(set(paths)):
            raise ValueError("dev files contain duplicate paths")
        collisions = sorted({file.path for file in self.candidate.files}.intersection(paths))
        if collisions:
            raise ValueError(f"dev files cannot replace candidate paths: {collisions}")
        return self


class CheckRequestV2(_DependencyRequestV2):
    """Compile or sanitize the current candidate on the target GPU."""

    operation: Literal["check"]
    arch: str | None = Field(default=None, max_length=100)
    sanitize: Literal["memcheck", "racecheck", "initcheck", "synccheck"] | None = None


class DisassembleRequestV2(_DependencyRequestV2):
    """Compile and return GPU assembly for the current candidate."""

    operation: Literal["disassemble"]
    fmt: Literal["sass", "ptx", "isa", "auto"] = "auto"


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


class AttemptReportRequestV2(_CandidateRequestV2):
    """Register one terminal Attempt handoff against the exact current candidate."""

    operation: Literal["attempt_report"]
    report: AttemptReportV12

    @model_validator(mode="after")
    def _validate_subject(self) -> AttemptReportRequestV2:
        if self.report.attempt_id != self.attempt_id:
            raise ValueError("Attempt report names a different Attempt")
        return self


class AttemptReportStatusRequestV2(_GatewayRequestV2):
    """Read this Attempt generation's server-accepted terminal report for local recovery."""

    operation: Literal["attempt_report_status"]


class KernelArtifactReadRequestV2(_GatewayRequestV2):
    """Read the file index or one source file from a visible Kernel Artifact."""

    operation: Literal["kernel_artifact_read"]
    kernel_artifact_digest: str = Field(max_length=80)
    file: str | None = None

    @field_validator("kernel_artifact_digest")
    @classmethod
    def _validate_kernel_artifact_digest(cls, value: str) -> str:
        return str(parse_artifact_digest(value))

    @field_validator("file")
    @classmethod
    def _validate_file(cls, value: str | None) -> str | None:
        return None if value is None else _safe_relative_path(value, "Kernel Artifact file")


class ResultArtifactReadRequestV2(_GatewayRequestV2):
    """Read one exact visible Agent-facing Result Artifact."""

    operation: Literal["result_artifact_read"]
    result_artifact_digest: str = Field(max_length=80)

    @field_validator("result_artifact_digest")
    @classmethod
    def _validate_result_artifact_digest(cls, value: str) -> str:
        return str(parse_artifact_digest(value))


class DirectionHistoryRequestV2(_GatewayRequestV2):
    """Read frozen Direction journals visible to one optimization Attempt."""

    operation: Literal["direction_history"]


class ExperimentHistoryRequestV2(_GatewayRequestV2):
    """Read frozen Experiment journals visible to one optimization Attempt."""

    operation: Literal["experiment_history"]


class DirectionUpdateRequestV2(_GatewayRequestV2):
    """Append one validated Direction proposal or lifecycle update."""

    operation: Literal["direction_update"]
    request: dict[str, JsonValue]


class DirectionsListRequestV2(_GatewayRequestV2):
    """List normalized Directions visible to one optimization Attempt."""

    operation: Literal["directions_list"]


class DirectionLoadRequestV2(_GatewayRequestV2):
    """Load one normalized Direction visible to one optimization Attempt."""

    operation: Literal["direction_load"]
    direction_id: str = Field(pattern=r"^direction_[0-9a-f]{32}$")


class ExperimentRecordRequestV2(_GatewayRequestV2):
    """Append one validated Experiment to the authoritative Attempt Journal."""

    operation: Literal["experiment_record"]
    request: dict[str, JsonValue]


class ExperimentsListRequestV2(_GatewayRequestV2):
    """List Experiments visible to one optimization Attempt."""

    operation: Literal["experiments_list"]


class ExperimentLoadRequestV2(_GatewayRequestV2):
    """Load one Experiment visible to one optimization Attempt."""

    operation: Literal["experiment_load"]
    experiment_id: str = Field(pattern=r"^experiment_[0-9a-f]{32}$")


class JournalSnapshotRequestV2(_GatewayRequestV2):
    """Read the current Attempt's authoritative Journal for terminal reporting."""

    operation: Literal["journal_snapshot"]


type GatewayProxyRequestV2 = Annotated[
    EvaluateRequestV2
    | ProfileRequestV2
    | DevRequestV2
    | CheckRequestV2
    | DisassembleRequestV2
    | PollRequestV2
    | JobsRequestV2
    | CancelRequestV2
    | EnvRequestV2
    | HealthRequestV2
    | ConfigRequestV2
    | AttemptReportRequestV2
    | AttemptReportStatusRequestV2
    | KernelArtifactReadRequestV2
    | ResultArtifactReadRequestV2
    | DirectionHistoryRequestV2
    | ExperimentHistoryRequestV2
    | DirectionUpdateRequestV2
    | DirectionsListRequestV2
    | DirectionLoadRequestV2
    | ExperimentRecordRequestV2
    | ExperimentsListRequestV2
    | ExperimentLoadRequestV2
    | JournalSnapshotRequestV2,
    Field(discriminator="operation"),
]


GATEWAY_AGENT_SCHEMA_VERSION: Literal[1] = 1
_RUNTIME_OWNED_REQUEST_FIELDS = frozenset({"schema_version", "attempt_id", "candidate"})
_AGENT_HIDDEN_REQUEST_FIELDS = _RUNTIME_OWNED_REQUEST_FIELDS | {"idempotency_key"}
_RUNTIME_QUERY_OPERATION_NAMES = frozenset(
    {
        "attempt_report",
        "attempt_report_status",
        "kernel_artifact_read",
        "result_artifact_read",
        "direction_history",
        "experiment_history",
    }
)
_RUNTIME_JOURNAL_OPERATION_NAMES = frozenset(
    {
        "direction_update",
        "directions_list",
        "direction_load",
        "experiment_record",
        "experiments_list",
        "experiment_load",
        "journal_snapshot",
    }
)
_GATEWAY_REQUEST_MODELS: dict[str, type[_GatewayRequestV2]] = {
    "evaluate": EvaluateRequestV2,
    "profile": ProfileRequestV2,
    "dev": DevRequestV2,
    "check": CheckRequestV2,
    "disassemble": DisassembleRequestV2,
    "poll": PollRequestV2,
    "jobs": JobsRequestV2,
    "cancel": CancelRequestV2,
    "env": EnvRequestV2,
    "health": HealthRequestV2,
    "config": ConfigRequestV2,
    "attempt_report": AttemptReportRequestV2,
    "attempt_report_status": AttemptReportStatusRequestV2,
    "kernel_artifact_read": KernelArtifactReadRequestV2,
    "result_artifact_read": ResultArtifactReadRequestV2,
    "direction_history": DirectionHistoryRequestV2,
    "experiment_history": ExperimentHistoryRequestV2,
    "direction_update": DirectionUpdateRequestV2,
    "directions_list": DirectionsListRequestV2,
    "direction_load": DirectionLoadRequestV2,
    "experiment_record": ExperimentRecordRequestV2,
    "experiments_list": ExperimentsListRequestV2,
    "experiment_load": ExperimentLoadRequestV2,
    "journal_snapshot": JournalSnapshotRequestV2,
}


def gateway_agent_request_schema(
    operation: str | None = None,
    *,
    allowed_operations: frozenset[str] | None = None,
) -> dict[str, JsonValue]:
    """Project the live wire models into the request shape accepted from an Agent.

    Runtime-owned identity, operation, and Candidate fields are attached by the selected Core tool
    binding and are deliberately absent. Every remaining field, default, enum, bound, and
    extra-field policy is generated from the exact Pydantic model used by Runtime.
    """
    if operation is not None and operation not in _GATEWAY_REQUEST_MODELS:
        raise ValueError(f"unsupported Gateway operation: {operation}")
    names = (operation,) if operation is not None else tuple(_GATEWAY_REQUEST_MODELS)
    if allowed_operations is not None:
        names = tuple(name for name in names if name in allowed_operations)
        if operation is not None and not names:
            raise PermissionError(f"Gateway operation is not allowed: {operation}")
    operations = {
        name: _agent_operation_schema(
            _GATEWAY_REQUEST_MODELS[name],
            runtime_bound=(
                name in _RUNTIME_QUERY_OPERATION_NAMES or name in _RUNTIME_JOURNAL_OPERATION_NAMES
            ),
        )
        for name in names
    }
    request_contract = (
        "runtime-query"
        if operation in _RUNTIME_QUERY_OPERATION_NAMES
        else "runtime-journal"
        if operation in _RUNTIME_JOURNAL_OPERATION_NAMES
        else "gateway-execute"
        if operation is not None
        else "runtime-operation"
    )
    runtime_owned_fields = _RUNTIME_OWNED_REQUEST_FIELDS | (
        {"operation"}
        if operation in _RUNTIME_QUERY_OPERATION_NAMES
        or operation in _RUNTIME_JOURNAL_OPERATION_NAMES
        else set()
    )
    if operation == "evaluate":
        runtime_owned_fields = runtime_owned_fields | {"baseline"}
    return cast(
        dict[str, JsonValue],
        {
            "schema_version": GATEWAY_AGENT_SCHEMA_VERSION,
            "gateway_protocol_version": GATEWAY_PROXY_PROTOCOL_VERSION,
            "request_contract": request_contract,
            "runtime_owned_fields": sorted(runtime_owned_fields),
            "operations": operations,
        },
    )


def _agent_operation_schema(
    model: type[_GatewayRequestV2],
    *,
    runtime_bound: bool,
) -> dict[str, object]:
    schema = deepcopy(model.model_json_schema(mode="validation"))
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        raise TypeError(f"Gateway request schema has no properties: {model.__name__}")
    hidden_fields = _AGENT_HIDDEN_REQUEST_FIELDS | ({"operation"} if runtime_bound else set())
    if model is EvaluateRequestV2:
        hidden_fields = hidden_fields | {"baseline"}
    for field_name in hidden_fields:
        properties.pop(field_name, None)
    required = schema.get("required")
    if isinstance(required, list):
        schema["required"] = [
            field_name for field_name in required if field_name not in hidden_fields
        ]
    if model is EvaluateRequestV2:
        for source, path in (("input_py", "input_path"), ("shapes", "shapes_path")):
            properties[path] = {
                "type": "string",
                "minLength": 1,
                "description": f"Workspace-relative file expanded by the Core tool into {source}.",
            }
        schema["allOf"] = [
            {"not": {"required": ["input_py", "input_path"]}},
            {"not": {"required": ["shapes", "shapes_path"]}},
        ]
        properties["candidate_path"] = {
            "type": "string",
            "minLength": 1,
            "description": "Workspace Kernel file or directory; defaults to current Kernel.",
        }
        comparison = schema["$defs"]["EvaluateComparisonV2"]
        comparison["properties"]["baseline_path"] = {
            "type": "string",
            "minLength": 1,
            "description": "Workspace-relative Kernel file or directory for comparison baseline A.",
        }
        comparison["required"].append("baseline_path")
        schema["allOf"].append(
            {
                "if": {
                    "properties": {"comparison": {"type": "object"}},
                    "required": ["comparison"],
                },
                "then": {"properties": {"mode": {"const": "full"}}},
            }
        )
    # Keep definitions reachable from Agent fields (such as recursive JsonValue), not
    # the hidden Candidate bundle and Runtime-owned request fields.
    definitions = schema.pop("$defs", {})
    if isinstance(definitions, dict):
        reachable: set[str] = set()

        def visit(value: object) -> None:
            if isinstance(value, dict):
                reference = value.get("$ref")
                if isinstance(reference, str) and reference.startswith("#/$defs/"):
                    name = reference.removeprefix("#/$defs/")
                    if name not in reachable and name in definitions:
                        reachable.add(name)
                        visit(definitions[name])
                for nested in value.values():
                    visit(nested)
            elif isinstance(value, list):
                for nested in value:
                    visit(nested)

        visit(schema)
        if reachable:
            schema["$defs"] = {name: definitions[name] for name in sorted(reachable)}
    suffix = " Runtime request" if runtime_bound else " gateway-execute request"
    schema["title"] = model.__name__.removesuffix("RequestV2") + suffix
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
        "profile",
        "dev",
        "check",
        "disassemble",
        "poll",
        "jobs",
        "cancel",
        "env",
        "health",
        "config",
        "attempt_report",
        "attempt_report_status",
        "kernel_artifact_read",
        "result_artifact_read",
        "direction_history",
        "experiment_history",
        "direction_update",
        "directions_list",
        "direction_load",
        "experiment_record",
        "experiments_list",
        "experiment_load",
        "journal_snapshot",
    ]
    status: Literal["completed", "queued", "failed", "cancelled"]
    kernel_artifact_digest: str | None = None
    result_artifact_digest: str
    job_id: str | None = None
    evaluation: EvaluationV2 | None = None
    result: JsonValue

    @field_validator("result_artifact_digest")
    @classmethod
    def _validate_result_artifact_digest(cls, value: str) -> str:
        return str(parse_artifact_digest(value))


def _safe_relative_path(value: str, label: str) -> str:
    if not value or value.startswith("/") or "\\" in value or "\x00" in value:
        raise ValueError(f"{label} must be non-empty, normalized, and relative")
    parts = value.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError(f"{label} contains an unsafe component")
    return value
