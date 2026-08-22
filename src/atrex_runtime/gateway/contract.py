"""Sealed evaluation contracts resolved for one Gateway operation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..artifacts.local import ArtifactKind, JsonValue, LocalArtifactStore
from ..domain.ids import ArtifactDigest, AttemptId
from ..domain.models import Dsl, KernelRevision
from ..registry.base import Registry
from .control import SqliteGatewayControl

EVALUATION_CONTRACT_VERSION: Literal[1] = 1
_GATE_OWNED_RUNNER_KEYS = frozenset(
    {
        "atol",
        "rtol",
        "num_correctness_cases",
        "warmup_iters",
        "bench_iters",
        "benchmark_mode",
        "candidate_timeout_s",
        "perf_timeout_s",
        "validation_mode",
        "clock_locked",
        "require_clock_locked",
        "clock_lock_mode",
        "clock_lock_device",
        "gpu_clock_mhz",
        "memory_clock_mhz",
        "clock_lock_tolerance_mhz",
        "clock_lock_settle_seconds",
        "clock_lock_command_timeout_s",
        "clock_lock_require_idle",
        "clock_lock_monitor",
        "clock_lock_sample_interval_ms",
        "clock_lock_runtime_tolerance_mhz",
        "clock_lock_fail_on_deviation",
    }
)


class AgateEvaluationOptionsV1(BaseModel):
    """Atrex-Bench options fixed for every evaluation in one Campaign."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    num_correctness_cases: int = Field(ge=1)
    bench_iters: int = Field(ge=1)
    atol: float = Field(ge=0)
    rtol: float = Field(ge=0)
    timeout_s: int = Field(gt=0)


class AgateEvaluationContractV1(BaseModel):
    """Complete trusted Agate request inputs shared by a Campaign's DSL lineages."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1] = EVALUATION_CONTRACT_VERSION
    agate_gpu: str | None = Field(default=None, min_length=1)
    candidate_path: str
    reference_py: str = Field(min_length=1)
    input_py: str = Field(min_length=1)
    shapes: dict[str, JsonValue]
    metadata: dict[str, JsonValue] | None = None
    roofline: dict[str, JsonValue] | None = None
    options: AgateEvaluationOptionsV1
    env_vars: dict[str, str] = Field(default_factory=dict)
    requirements: tuple[str, ...] = ()
    deps_mode: Literal["freeze_installed", "no_deps"] | None = None
    mode: Literal["full", "correctness_only"] = "full"
    lock_clocks: bool = True
    harness: Literal["atrex_bench"] | None = None
    atrex_bench_version: str | None = None
    runner_overrides: dict[str, JsonValue] = Field(default_factory=dict)
    production_gate: bool = False

    @field_validator("candidate_path")
    @classmethod
    def _validate_candidate_path(cls, value: str) -> str:
        if not value or value.startswith("/"):
            raise ValueError("candidate_path must be non-empty and relative")
        if any(part in {"", ".", ".."} for part in value.split("/")):
            raise ValueError("candidate_path contains an unsafe component")
        return value

    @field_validator("shapes")
    @classmethod
    def _validate_shapes(cls, value: dict[str, JsonValue]) -> dict[str, JsonValue]:
        if not value:
            raise ValueError("shapes must be non-empty")
        return value

    @field_validator("requirements")
    @classmethod
    def _validate_requirements(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not requirement.strip() for requirement in value):
            raise ValueError("requirements cannot contain empty entries")
        return value


@dataclass(frozen=True, slots=True)
class RuntimeGateContractPolicy:
    """Trusted Gate fields applied before an Evaluation Contract is sealed."""

    options: AgateEvaluationOptionsV1
    lock_clocks: bool
    runner_overrides: dict[str, JsonValue]
    atrex_bench_version: str | None = None
    production_gate: bool = False

    def apply(self, contract: AgateEvaluationContractV1) -> AgateEvaluationContractV1:
        overrides = {
            key: value
            for key, value in contract.runner_overrides.items()
            if key not in _GATE_OWNED_RUNNER_KEYS
        }
        overrides.update(self.runner_overrides)
        overrides["benchmark_mode"] = "eager"
        return contract.model_copy(
            update={
                "options": self.options,
                "mode": "full",
                "lock_clocks": self.lock_clocks,
                "harness": "atrex_bench",
                "atrex_bench_version": self.atrex_bench_version,
                "runner_overrides": overrides,
                "production_gate": self.production_gate,
            }
        )


@dataclass(frozen=True, slots=True)
class AgateEvaluationContext:
    """Campaign identity and its validated immutable evaluation contract."""

    operator: str
    hardware_target: str
    dsl: Dsl
    contract: AgateEvaluationContractV1
    evaluation_contract_digest: ArtifactDigest | None = None

    @property
    def agate_gpu(self) -> str:
        """Return the Agate scheduler selector, distinct from Agent-visible architecture."""
        return self.contract.agate_gpu or self.hardware_target


class AgateEvaluationContextResolver(Protocol):
    """Resolve trusted Agate inputs from an Attempt identity."""

    def resolve(self, attempt_id: AttemptId) -> AgateEvaluationContext:
        """Return the immutable Campaign evaluation context for ``attempt_id``."""
        ...


class RegistryAgateEvaluationContextResolver:
    """Resolve and verify a Campaign evaluation contract through the Registry and CAS."""

    def __init__(
        self,
        registry: Registry,
        artifacts: LocalArtifactStore,
        bootstrap_subjects: SqliteGatewayControl | None = None,
    ) -> None:
        self._registry = registry
        self._artifacts = artifacts
        self._bootstrap_subjects = bootstrap_subjects

    def resolve(self, attempt_id: AttemptId) -> AgateEvaluationContext:
        """Follow Attempt ownership and parse the sealed versioned contract."""
        try:
            attempt = self._registry.get_attempt(attempt_id)
        except KeyError:
            if self._bootstrap_subjects is None:
                raise
            subject = self._bootstrap_subjects.get_bootstrap_subject(attempt_id)
            operator = subject.operator
            hardware_target = subject.hardware_target
            dsl = subject.dsl
            contract_digest = subject.evaluation_contract_digest
        else:
            epoch = self._registry.get_epoch(attempt.epoch_id)
            lineage = self._registry.get_lineage(epoch.lineage_id)
            campaign = self._registry.get_campaign(lineage.campaign_id)
            operator = campaign.operator
            hardware_target = campaign.hardware_target
            dsl = lineage.dsl
            contract_digest = campaign.evaluation_contract_digest
        contract = load_evaluation_contract(self._artifacts, contract_digest)
        return AgateEvaluationContext(
            operator=operator,
            hardware_target=hardware_target,
            dsl=dsl,
            contract=contract,
            evaluation_contract_digest=contract_digest,
        )


class RegistryKernelEvaluationContextResolver:
    """Resolve the immutable evaluation context for an already registered Kernel."""

    def __init__(self, registry: Registry, artifacts: LocalArtifactStore) -> None:
        self._registry = registry
        self._artifacts = artifacts

    def resolve(self, revision: KernelRevision) -> AgateEvaluationContext:
        """Follow the Kernel's unique retained lineage to its sealed Campaign contract."""
        registered = self._registry.get_kernel_revision(revision.id)
        if registered.artifact_digest != revision.artifact_digest:
            raise ValueError("Kernel revision disagrees with the Registry")
        lineage = self._registry.find_kernel_lineage(revision.id)
        campaign = self._registry.get_campaign(lineage.campaign_id)
        return AgateEvaluationContext(
            operator=campaign.operator,
            hardware_target=lineage.hardware_target,
            dsl=lineage.dsl,
            contract=load_evaluation_contract(
                self._artifacts,
                campaign.evaluation_contract_digest,
            ),
            evaluation_contract_digest=campaign.evaluation_contract_digest,
        )


def load_evaluation_contract(
    artifacts: LocalArtifactStore,
    digest: ArtifactDigest,
) -> AgateEvaluationContractV1:
    stored = artifacts.verify(digest)
    if stored.kind is not ArtifactKind.EVALUATION_CONTRACT:
        raise ValueError("Campaign evaluation contract has the wrong artifact kind")
    value_path = stored.payload_path / "value.json"
    if not value_path.is_file():
        raise ValueError("Campaign evaluation contract must contain value.json")
    try:
        value = json.loads(value_path.read_bytes())
    except json.JSONDecodeError as error:
        raise ValueError("Campaign evaluation contract is not valid JSON") from error
    return AgateEvaluationContractV1.model_validate(value)
