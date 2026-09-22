"""Sealed evaluation contracts resolved for one Gateway operation."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from typing import Final, Literal, Protocol, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..artifacts.local import ArtifactKind, JsonValue, LocalArtifactStore
from ..domain.ids import ArtifactDigest, AttemptId
from ..domain.models import Dsl, KernelRevision
from ..kernel_sources import KernelSourceContract
from ..registry.base import Registry
from .control import SqliteGatewayControl
from .environment import AcceleratorBackend

EVALUATION_CONTRACT_VERSION: Literal[1] = 1
MAX_HOLDOUT_SHAPES: Final = 15
SHAPE_SPLIT_SEED: Final = 42
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


class AgentOutputToleranceV1(BaseModel):
    """One evaluator-owned elementwise tolerance safe to disclose to the Agent."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    atol: float = Field(ge=0)
    rtol: float = Field(ge=0)


class AgentCorrectnessPolicyV1(BaseModel):
    """Trusted Agent-facing projection of the sealed correctness policy."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    comparison: Literal["elementwise"] = "elementwise"
    formula: Literal[
        "abs(candidate - reference) <= atol + rtol * abs(reference)"
    ] = "abs(candidate - reference) <= atol + rtol * abs(reference)"
    default_tolerance: AgentOutputToleranceV1
    output_tolerances: dict[str, AgentOutputToleranceV1] = Field(default_factory=dict)


class ShapeSplitRecordV1(BaseModel):
    """Private replay record for the exact fixed-seed split and capped sampling."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    algorithm: Literal["python_random_shuffle_sample"] = "python_random_shuffle_sample"
    seed: int = Field(ge=0, strict=True)
    max_shapes_per_set: Literal[15] = 15
    source_shape_count: int = Field(ge=2, strict=True)
    source_shape_ids: tuple[str, ...]
    valid_shape_ids: tuple[str, ...]
    test_shape_ids: tuple[str, ...]
    agent_shape_id_map: dict[str, str] | None = Field(
        default=None,
        description="Private Agent Shape ID to evaluator Shape ID mapping",
    )

    @model_validator(mode="after")
    def _validate_selection(self) -> Self:
        source = set(self.source_shape_ids)
        valid, test = set(self.valid_shape_ids), set(self.test_shape_ids)
        aliases = self.agent_shape_id_map
        if (
            len(source) != self.source_shape_count
            or len(self.source_shape_ids) != self.source_shape_count
            or len(self.valid_shape_ids) != len(valid)
            or len(self.test_shape_ids) != len(test)
            or len(valid) != min(self.max_shapes_per_set, (self.source_shape_count + 1) // 2)
            or len(test) != min(self.max_shapes_per_set, self.source_shape_count // 2)
            or valid & test
            or not (valid | test) <= source
        ):
            raise ValueError(
                "shape_split must record unique, disjoint, capped Valid/Test selections"
            )
        if aliases is not None and (
            set(aliases) != {str(index) for index in range(len(valid))}
            or len(set(aliases.values())) != len(aliases)
            or set(aliases.values()) != valid
        ):
            raise ValueError(
                "shape_split.agent_shape_id_map must map contiguous opaque Agent IDs "
                "exactly once onto every Valid Shape"
            )
        return self


class AgateEvaluationContractV1(BaseModel):
    """Complete trusted Agate request inputs shared by a Campaign's DSL lineages."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1] = EVALUATION_CONTRACT_VERSION
    agate_gpu: str | None = Field(default=None, min_length=1)
    accelerator_backend: AcceleratorBackend | None = None
    device_slug: str | None = Field(default=None, min_length=1)
    candidate_path: str
    reference_py: str = Field(min_length=1)
    input_py: str = Field(min_length=1)
    shapes: dict[str, JsonValue]
    validation_shape_ids: tuple[str, ...] | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    shape_split: ShapeSplitRecordV1 | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
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
    kernel_sources: dict[Dsl, KernelSourceContract] = Field(
        default_factory=dict, exclude_if=lambda value: not value
    )

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

    @model_validator(mode="after")
    def _validate_holdout(self) -> Self:
        if self.validation_shape_ids is not None:
            ids = self.validation_shape_ids
            if (
                not 2 <= len(self.shapes) <= 2 * MAX_HOLDOUT_SHAPES
                or len(ids) != (len(self.shapes) + 1) // 2
                or len(set(ids)) != len(ids)
                or not set(ids) <= self.shapes.keys()
            ):
                raise ValueError(
                    "validation_shape_ids must select half of retained shapes (odd extra: Valid); "
                    f"Valid and Test must each contain at most {MAX_HOLDOUT_SHAPES} Shapes"
                )
        if self.shape_split is not None and (
            self.validation_shape_ids is None
            or set(self.shape_split.valid_shape_ids) != set(self.validation_shape_ids)
            or set(self.shape_split.valid_shape_ids) | set(self.shape_split.test_shape_ids)
            != self.shapes.keys()
        ):
            raise ValueError("shape_split selections must match the sealed evaluation Shapes")
        return self

    def with_shape_holdout(self) -> AgateEvaluationContractV1:
        """Randomly split and sample with a fixed local RNG, then seal the replay record."""
        if len(self.shapes) < 2:
            raise ValueError(
                "Valid/Test splitting requires at least 2 Shapes; "
                "single-Shape tasks are not supported"
            )
        if self.validation_shape_ids is not None:
            return self
        source_ids = tuple(sorted(self.shapes))
        ordered = list(source_ids)
        rng = random.Random(SHAPE_SPLIT_SEED)
        rng.shuffle(ordered)
        midpoint = (len(ordered) + 1) // 2
        valid_ids = tuple(sorted(rng.sample(ordered[:midpoint], min(midpoint, MAX_HOLDOUT_SHAPES))))
        test_ids = tuple(
            sorted(rng.sample(ordered[midpoint:], min(len(ordered) - midpoint, MAX_HOLDOUT_SHAPES)))
        )
        agent_shape_id_map = {
            str(index): shape_id
            for index, shape_id in enumerate(sorted(valid_ids, key=_shape_id_sort_key))
        }
        retained = self
        if len(ordered) > 2 * MAX_HOLDOUT_SHAPES:
            from .batched_evaluate import subset_evaluation_contract

            retained = subset_evaluation_contract(self, tuple(sorted((*valid_ids, *test_ids))))
        return retained.model_copy(
            update={
                "validation_shape_ids": valid_ids,
                "shape_split": ShapeSplitRecordV1(
                    seed=SHAPE_SPLIT_SEED,
                    source_shape_count=len(source_ids),
                    source_shape_ids=source_ids,
                    valid_shape_ids=valid_ids,
                    test_shape_ids=test_ids,
                    agent_shape_id_map=agent_shape_id_map,
                ),
            }
        )

    def for_agent(self) -> AgateEvaluationContractV1:
        """Materialize only Valid inputs; Test Shapes and per-Shape auxiliary data stay private."""
        if self.validation_shape_ids is None:
            return self
        # Local import avoids the batching module's contract import cycle.
        from .batched_evaluate import remap_evaluation_contract, subset_evaluation_contract

        aliases = None if self.shape_split is None else self.shape_split.agent_shape_id_map
        valid = (
            subset_evaluation_contract(self, self.validation_shape_ids)
            if aliases is None
            else remap_evaluation_contract(self, aliases)
        )
        if valid.metadata is not None:
            # Dataset-wide traces can enumerate Test cases outside metadata.shapes.
            # Preserve evaluator semantics and the filtered Valid map, not provenance.
            valid = valid.model_copy(
                update={
                    "metadata": {
                        key: value
                        for key, value in valid.metadata.items()
                        if key
                        in {
                            "benchmark_contract",
                            "category",
                            "dtype",
                            "input_dtypes",
                            "output_dtypes",
                            "shapes",
                            "num_shapes",
                        }
                    }
                }
            )
        return valid

    def agent_shape_id_map(self) -> dict[str, str] | None:
        """Return the sealed private Agent-ID to evaluator-ID map, if this Campaign has one."""
        if self.shape_split is None or self.shape_split.agent_shape_id_map is None:
            return None
        return dict(self.shape_split.agent_shape_id_map)

    def agent_correctness_policy(self) -> AgentCorrectnessPolicyV1:
        """Project exact Gate tolerances without exposing evaluator cases or inputs."""
        output_tolerances: dict[str, AgentOutputToleranceV1] = {}
        metadata = self.metadata
        benchmark = metadata.get("benchmark_contract") if isinstance(metadata, dict) else None
        raw_tolerances = (
            benchmark.get("correctness_tolerances")
            if isinstance(benchmark, dict)
            else None
        )
        if isinstance(raw_tolerances, dict):
            for output, value in raw_tolerances.items():
                if not isinstance(output, str) or not output.strip():
                    raise ValueError(
                        "benchmark_contract.correctness_tolerances keys must be "
                        "non-empty output names"
                    )
                output_tolerances[output] = AgentOutputToleranceV1.model_validate(value)
        return AgentCorrectnessPolicyV1(
            default_tolerance=AgentOutputToleranceV1(
                atol=self.options.atol,
                rtol=self.options.rtol,
            ),
            output_tolerances=output_tolerances,
        )


def load_agent_correctness_policy(
    artifacts: LocalArtifactStore,
    digest: ArtifactDigest,
) -> AgentCorrectnessPolicyV1:
    """Load the sealed contract and return only its Agent-safe correctness projection."""
    artifact = artifacts.verify(digest)
    if artifact.kind is not ArtifactKind.EVALUATION_CONTRACT:
        raise ValueError("Agent correctness policy requires an Evaluation Contract")
    contract = AgateEvaluationContractV1.model_validate_json(
        (artifact.payload_path / "value.json").read_bytes()
    )
    return contract.agent_correctness_policy()


def _shape_id_sort_key(value: str) -> tuple[int, int | str]:
    return (0, int(value)) if value.isdigit() else (1, value)


@dataclass(frozen=True, slots=True)
class RuntimeGateContractPolicy:
    """Trusted Gate fields applied before an Evaluation Contract is sealed."""

    options: AgateEvaluationOptionsV1
    lock_clocks: bool
    runner_overrides: dict[str, JsonValue]
    atrex_bench_version: str | None = None
    production_gate: bool = False

    def apply(
        self,
        contract: AgateEvaluationContractV1,
        *,
        accelerator_backend: AcceleratorBackend | None = None,
    ) -> AgateEvaluationContractV1:
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
                # PPU exposes a CUDA-compatible execution surface but its PPU-SMI
                # shim does not implement managed graphics clocks.
                "lock_clocks": self.lock_clocks and accelerator_backend != "ppu",
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
    def kernel_source(self) -> KernelSourceContract | None:
        return self.contract.kernel_sources.get(self.dsl)

    @property
    def agate_gpu(self) -> str:
        """Return the Agate scheduler selector, distinct from Agent-visible architecture."""
        return self.contract.agate_gpu or self.hardware_target


class AgateEvaluationContextResolver(Protocol):
    """Resolve trusted Agate inputs from an Attempt identity."""

    def resolve(self, attempt_id: AttemptId) -> AgateEvaluationContext:
        """Return the immutable Campaign evaluation context for ``attempt_id``."""
        ...


def candidate_path_for_attempt(
    resolver: AgateEvaluationContextResolver | None,
    attempt_id: AttemptId,
) -> str | None:
    """Return the contract Candidate path when the Attempt context is available."""
    if resolver is None:
        return None
    try:
        return resolver.resolve(attempt_id).contract.candidate_path
    except (LookupError, ValueError):
        return None


class RegistryAgateEvaluationContextResolver:
    """Resolve the Agent-visible Valid-only evaluation context."""

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
        """Follow Attempt ownership and project the sealed contract for an Agent."""
        return self._resolve(attempt_id, agent_visible=True)

    def _resolve(
        self,
        attempt_id: AttemptId,
        *,
        agent_visible: bool,
    ) -> AgateEvaluationContext:
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
        if agent_visible:
            contract = contract.for_agent()
        return AgateEvaluationContext(
            operator=operator,
            hardware_target=hardware_target,
            dsl=dsl,
            contract=contract,
            evaluation_contract_digest=contract_digest,
        )


class RegistryAuthoritativeEvaluationContextResolver(RegistryAgateEvaluationContextResolver):
    """Resolve the complete private Valid+Test contract for Runtime-owned Gates."""

    def resolve(self, attempt_id: AttemptId) -> AgateEvaluationContext:
        """Follow Attempt ownership without applying the Agent-visible projection."""
        return self._resolve(attempt_id, agent_visible=False)


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
