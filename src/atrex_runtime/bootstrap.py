"""Idempotent trusted Bootstrap for Campaigns and their selected DSL Lineages."""

from __future__ import annotations

import hashlib
import json
import stat
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Protocol, Self

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, field_validator, model_validator

from .artifacts.local import ArtifactKind, JsonValue, LocalArtifactStore
from .controller.evidence import EvidenceCheckpointV1, LocalEvidenceAssembler
from .domain.ids import (
    ArtifactDigest,
    AttemptId,
    CampaignId,
    KernelAgentRevisionId,
    KernelRevisionId,
    LineageId,
    parse_attempt_id,
    parse_campaign_id,
    parse_kernel_agent_revision_id,
    parse_kernel_revision_id,
    parse_lineage_id,
)
from .domain.models import (
    Campaign,
    Dsl,
    KernelAgentRevision,
    KernelEvaluation,
    KernelRevision,
    Lineage,
    LineageStatus,
)
from .gateway.contract import AgateEvaluationContractV1, RuntimeGateContractPolicy
from .gateway.environment import ResolvedAgateEnvironment
from .kernel_agents import GitOptimizerBaseLoader
from .ports import KernelAgentCandidate
from .registry.base import Registry
from .roofline import RooflineBuilder
from .workers.problem_generalization import AgentProblemV1

CAMPAIGN_SPEC_VERSION: Literal[3] = 3
_JSON_VALUE_ADAPTER: TypeAdapter[JsonValue] = TypeAdapter(JsonValue)
type RooflineMode = Literal["explicit", "sealed-reuse", "generated", "profile-fallback"]


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


class GitBaseRevisionV1(BaseModel):
    """One explicit immutable commit from the deployment-approved Core repository."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    commit: str = Field(pattern=r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")


class AgentProblemGenerator(Protocol):
    """Run an isolated Core phase and return one validated public problem Artifact."""

    def generate(
        self,
        *,
        generalization_id: str,
        optimizer_digest: ArtifactDigest,
        evaluation_contract_digest: ArtifactDigest,
        dsl: Dsl,
        operator: str,
        hardware_target: str,
        model: str | None,
    ) -> ArtifactDigest: ...


@dataclass(frozen=True, slots=True)
class GeneratedLineageBaseline:
    """Authoritative output of one Runtime-controlled Core baseline phase."""

    kernel_digest: ArtifactDigest
    gateway_result_digest: ArtifactDigest
    latency_us: float
    report_digest: ArtifactDigest
    session_trace_digest: ArtifactDigest

    def __post_init__(self) -> None:
        if self.latency_us <= 0:
            raise ValueError("generated lineage baseline latency must be positive")


class LineageBaselineGenerator(Protocol):
    """Run the Core framework-baseline phase before registering a Lineage."""

    def generate(
        self,
        *,
        bootstrap_attempt_id: AttemptId,
        campaign_id: CampaignId,
        lineage_id: LineageId,
        kernel_agent_revision_id: KernelAgentRevisionId,
        optimizer_digest: ArtifactDigest,
        input_kernel_digest: ArtifactDigest,
        evaluation_contract_digest: ArtifactDigest,
        agent_problem_digest: ArtifactDigest,
        evidence_digest: ArtifactDigest,
        dsl: Dsl,
        operator: str,
        hardware_target: str,
        model: str | None,
    ) -> GeneratedLineageBaseline: ...


class HardwareTargetResolver(Protocol):
    """Resolve an Agate environment selector into canonical GPU architecture metadata."""

    def resolve(self, gpu: str) -> ResolvedAgateEnvironment: ...


class _LineageBootstrapSpec(BaseModel):
    """Resolved common and DSL-specific inputs for one internal lineage step."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    creation_key: str = Field(min_length=1, max_length=200)
    operator: str = Field(min_length=1)
    hardware_target: str = Field(min_length=1)
    dsl: Dsl
    evaluation_contract: Path
    baseline_kernel: Path
    initial_evidence: Path
    challenger_count: int = Field(ge=0)
    challenger_start_epoch: int = Field(gt=0)
    trajectories_per_branch: int = Field(gt=0)
    attempts_per_trajectory: int = Field(gt=0)
    optimizer_model: str | None
    evolver_model: str | None
    problem_generalization_model: str | None


class LineageAgentModelsV1(BaseModel):
    """Optional model identities selected independently for one DSL Lineage."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    optimizer: str | None = Field(default=None, min_length=1, max_length=200)
    evolver: str | None = Field(default=None, min_length=1, max_length=200)

    @field_validator("optimizer", "evolver")
    @classmethod
    def _validate_model(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized or "\x00" in normalized:
            raise ValueError("Agent model must be non-empty and cannot contain NUL")
        return normalized


class CampaignLineageSpecV2(BaseModel):
    """DSL-specific baseline inputs inside one Campaign definition."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    baseline_kernel: Path
    initial_evidence: Path
    models: LineageAgentModelsV1 = LineageAgentModelsV1()


class CampaignSpecV3(BaseModel):
    """One Campaign request whose hardware selector is resolved through Agate."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[3] = CAMPAIGN_SPEC_VERSION
    creation_key: str = Field(min_length=1, max_length=200)
    operator: str = Field(min_length=1)
    hardware_target: str = Field(min_length=1)
    evaluation_contract: Path
    agent_problem: Path | None = None
    problem_generalization_model: str | None = Field(
        default=None,
        min_length=1,
        max_length=200,
    )
    base_revision: GitBaseRevisionV1
    challenger_count: int = Field(default=1, ge=0)
    challenger_start_epoch: int = Field(default=1, gt=0)
    trajectories_per_branch: int = Field(default=1, gt=0)
    attempts_per_trajectory: int = Field(gt=0)
    lineages: dict[Dsl, CampaignLineageSpecV2]

    @field_validator("problem_generalization_model")
    @classmethod
    def _validate_problem_model(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized or "\x00" in normalized:
            raise ValueError("Problem Generalization model cannot be empty or contain NUL")
        return normalized

    @model_validator(mode="after")
    def _validate_campaign(self) -> CampaignSpecV3:
        if not self.lineages:
            raise ValueError("Campaign requires at least one DSL Lineage")
        if self.agent_problem is not None and self.problem_generalization_model is not None:
            raise ValueError("problem_generalization_model requires generated Agent Problem")
        return self

    def selected_dsls(self) -> tuple[Dsl, ...]:
        """Return the Lineage keys in the canonical DSL order."""
        return tuple(dsl for dsl in Dsl if dsl in self.lineages)

    def lineage_spec(self, dsl: Dsl) -> _LineageBootstrapSpec:
        """Project one selected DSL into the recoverable internal lineage operation."""
        lineage = self.lineages[dsl]
        return _LineageBootstrapSpec(
            creation_key=self.creation_key,
            operator=self.operator,
            hardware_target=self.hardware_target,
            dsl=dsl,
            evaluation_contract=self.evaluation_contract,
            baseline_kernel=lineage.baseline_kernel,
            initial_evidence=lineage.initial_evidence,
            challenger_count=self.challenger_count,
            challenger_start_epoch=self.challenger_start_epoch,
            trajectories_per_branch=self.trajectories_per_branch,
            attempts_per_trajectory=self.attempts_per_trajectory,
            optimizer_model=lineage.models.optimizer,
            evolver_model=lineage.models.evolver,
            problem_generalization_model=self.problem_generalization_model,
        )

    @classmethod
    def from_file(cls, path: str | Path) -> Self:
        """Parse strict JSON and resolve common and per-DSL source paths."""
        spec_path = Path(path).resolve()
        try:
            value = json.loads(spec_path.read_bytes())
        except json.JSONDecodeError as error:
            raise ValueError(f"Campaign spec is not valid JSON: {spec_path}") from error
        spec = cls.model_validate(value)

        def resolve(source: Path) -> Path:
            return source if source.is_absolute() else (spec_path.parent / source).resolve()

        lineages = {
            dsl: lineage.model_copy(
                update={
                    "baseline_kernel": resolve(lineage.baseline_kernel),
                    "initial_evidence": resolve(lineage.initial_evidence),
                }
            )
            for dsl, lineage in spec.lineages.items()
        }
        return spec.model_copy(
            update={
                "evaluation_contract": resolve(spec.evaluation_contract),
                "agent_problem": (
                    None if spec.agent_problem is None else resolve(spec.agent_problem)
                ),
                "lineages": lineages,
            }
        )


def load_campaign_spec(path: str | Path) -> CampaignSpecV3:
    """Load the sole supported Campaign-definition protocol."""
    spec_path = Path(path).resolve()
    try:
        value = json.loads(spec_path.read_bytes())
    except json.JSONDecodeError as error:
        raise ValueError(f"Campaign spec is not valid JSON: {spec_path}") from error
    if not isinstance(value, dict):
        raise ValueError("Campaign spec must be a JSON object")
    version = value.get("schema_version")
    if version == CAMPAIGN_SPEC_VERSION:
        return CampaignSpecV3.from_file(spec_path)
    raise ValueError(f"unsupported Campaign spec schema_version: {version!r}")


def parse_campaign_spec_json(payload: bytes) -> CampaignSpecV3:
    """Parse an HTTP Campaign definition without resolving deployment-host paths."""
    try:
        value = json.loads(payload)
    except json.JSONDecodeError as error:
        raise ValueError("Campaign request is not valid JSON") from error
    if not isinstance(value, dict):
        raise ValueError("Campaign request must be a JSON object")
    version = value.get("schema_version")
    if version == CAMPAIGN_SPEC_VERSION:
        return CampaignSpecV3.model_validate(value)
    raise ValueError(f"unsupported Campaign schema_version: {version!r}")


@dataclass(frozen=True, slots=True)
class BootstrapResult:
    """Durable identities and evidence checkpoint created or recovered by bootstrap."""

    campaign_id: CampaignId
    lineage_id: LineageId
    kernel_agent_revision_id: KernelAgentRevisionId
    baseline_kernel_revision_id: KernelRevisionId
    evaluation_contract_digest: ArtifactDigest
    agent_problem_digest: ArtifactDigest
    initial_evidence_digest: ArtifactDigest
    baseline_kernel_created_at: str
    bootstrap_attempt_id: AttemptId
    kernel_agent_created_at: str
    optimizer_digest: ArtifactDigest
    optimizer_model: str | None = None
    evolver_model: str | None = None


@dataclass(frozen=True, slots=True)
class CampaignBootstrapResult:
    """One Campaign identity and the ordered Lineages created or recovered for it."""

    campaign_id: CampaignId
    lineages: tuple[BootstrapResult, ...]
    hardware_target: str
    agate_gpu: str
    roofline_mode: RooflineMode = field(default="profile-fallback", compare=False)
    roofline_detail: str | None = field(default=None, compare=False)
    problem_generalization_model: str | None = None
    evolver_commit: str | None = None

    def __post_init__(self) -> None:
        if not self.lineages:
            raise ValueError("Campaign bootstrap result requires at least one Lineage")
        if any(result.campaign_id != self.campaign_id for result in self.lineages):
            raise ValueError("Campaign bootstrap results disagree on Campaign identity")


class CampaignBootstrapper:
    """Seal trusted inputs and recoverably initialize Campaign DSL Lineages."""

    def __init__(
        self,
        registry: Registry,
        artifacts: LocalArtifactStore,
        *,
        base_loader: GitOptimizerBaseLoader | None = None,
        problem_generator: AgentProblemGenerator | None = None,
        baseline_generator: LineageBaselineGenerator | None = None,
        roofline_builder: RooflineBuilder | None = None,
        roofline_resolved: Callable[[RooflineMode, str | None], None] | None = None,
        hardware_target_resolver: HardwareTargetResolver | None = None,
        evolver_commit: str | None = None,
        gate_contract_policy: RuntimeGateContractPolicy | None = None,
        max_parallel_lineages: int = 1,
        clock: Callable[[], str] = _utc_now,
    ) -> None:
        if max_parallel_lineages <= 0:
            raise ValueError("maximum parallel bootstrap Lineages must be positive")
        self._registry = registry
        self._artifacts = artifacts
        self._base_loader = base_loader
        self._problem_generator = problem_generator
        self._baseline_generator = baseline_generator
        self._roofline_builder = roofline_builder
        self._roofline_resolved = roofline_resolved
        self._hardware_target_resolver = hardware_target_resolver
        self._evolver_commit = evolver_commit
        self._gate_contract_policy = gate_contract_policy
        self._max_parallel_lineages = max_parallel_lineages
        self._clock = clock

    def bootstrap_campaign(self, spec: CampaignSpecV3) -> CampaignBootstrapResult:
        """Import shared Core once, then initialize selected Lineages recoverably."""
        requested_gpu = spec.hardware_target
        environment = (
            ResolvedAgateEnvironment(requested_gpu, requested_gpu)
            if self._hardware_target_resolver is None
            else self._hardware_target_resolver.resolve(requested_gpu)
        )
        spec = spec.model_copy(update={"hardware_target": environment.arch})
        selected = spec.selected_dsls()
        campaign_id = parse_campaign_id(self._derived_id("campaign", spec.creation_key))
        try:
            existing_campaign = self._registry.get_campaign(campaign_id)
        except KeyError:
            existing_campaign = None
        if existing_campaign is not None and self._evolver_commit is not None:
            existing_campaign = self._registry.ensure_campaign_evolver_commit(
                campaign_id,
                self._evolver_commit,
            )
        if (
            existing_campaign is not None
            and existing_campaign.problem_generalization_model != spec.problem_generalization_model
        ):
            raise ValueError(
                "bootstrap creation_key resolved to a Campaign with a different "
                "Problem Generalization model"
            )
        input_contract = AgateEvaluationContractV1.model_validate(
            self._read_json(spec.evaluation_contract, "evaluation contract")
        )
        if input_contract.agate_gpu not in {None, environment.gpu}:
            raise ValueError(
                "evaluation contract Agate GPU disagrees with the resolved environment"
            )
        input_contract = input_contract.model_copy(update={"agate_gpu": environment.gpu})
        if self._gate_contract_policy is not None:
            input_contract = self._gate_contract_policy.apply(input_contract)
        contract, roofline_mode, roofline_detail = self._resolve_evaluation_contract(
            input_contract,
            existing_contract_digest=(
                None if existing_campaign is None else existing_campaign.evaluation_contract_digest
            ),
            operator=spec.operator,
            hardware_target=environment.gpu,
        )
        if self._roofline_resolved is not None:
            self._roofline_resolved(roofline_mode, roofline_detail)
        if self._base_loader is None:
            raise ValueError("Git Optimizer Base loader is not configured")
        base = self._base_loader.build_candidate(selected[0], spec.base_revision.commit)
        shared_contract = self._artifacts.put_json(
            contract.model_dump(mode="json"),
            ArtifactKind.EVALUATION_CONTRACT,
        )
        if spec.agent_problem is not None:
            public_problem = AgentProblemV1.from_value(
                self._read_json(spec.agent_problem, "agent problem"),
                private_shapes=contract.shapes,
            )
            shared_problem = self._artifacts.put_json(
                public_problem.model_dump(mode="json"),
                ArtifactKind.AGENT_PROBLEM,
            )
        elif self._problem_generator is not None:
            shared_problem = self._problem_generator.generate(
                generalization_id=self._derived_id("generalize", spec.creation_key),
                optimizer_digest=base.candidate.optimizer_digest,
                evaluation_contract_digest=shared_contract,
                dsl=selected[0],
                operator=spec.operator,
                hardware_target=spec.hardware_target,
                model=spec.problem_generalization_model,
            )
        else:
            raise ValueError("campaign bootstrap requires an Agent Problem or Core generator")
        if self._artifacts.verify(shared_problem).kind is not ArtifactKind.AGENT_PROBLEM:
            raise ValueError("bootstrap Agent Problem generator returned the wrong Artifact kind")
        results: list[BootstrapResult] = []
        if existing_campaign is not None and (
            shared_contract != existing_campaign.evaluation_contract_digest
            or shared_problem != existing_campaign.agent_problem_digest
            or spec.problem_generalization_model != existing_campaign.problem_generalization_model
        ):
            raise ValueError("bootstrap creation_key resolved to a different Campaign")
        self._ensure_campaign(
            Campaign(
                campaign_id,
                spec.operator,
                spec.hardware_target,
                shared_contract,
                shared_problem,
                self._clock(),
                problem_generalization_model=spec.problem_generalization_model,
                evolver_commit=self._evolver_commit,
            )
        )

        def bootstrap_one(dsl: Dsl) -> BootstrapResult:
            return self._bootstrap_lineage(
                spec.lineage_spec(dsl),
                campaign_contract=contract,
                campaign_contract_digest=shared_contract,
                campaign_problem_digest=shared_problem,
                agent_candidate=KernelAgentCandidate(
                    dsl=dsl,
                    optimizer_digest=base.candidate.optimizer_digest,
                ),
                source_provenance_digest=base.source_provenance_digest,
            )

        if len(selected) == 1 or self._max_parallel_lineages == 1:
            results.extend(bootstrap_one(dsl) for dsl in selected)
        else:
            worker_count = min(len(selected), self._max_parallel_lineages)
            with ThreadPoolExecutor(
                max_workers=worker_count,
                thread_name_prefix="atrex-bootstrap",
            ) as executor:
                futures: dict[Dsl, Future[BootstrapResult]] = {
                    dsl: executor.submit(bootstrap_one, dsl) for dsl in selected
                }
                results.extend(futures[dsl].result() for dsl in selected)
        if results[0].campaign_id != campaign_id:
            raise AssertionError("Campaign bootstrap derived inconsistent identity")
        return CampaignBootstrapResult(
            campaign_id=campaign_id,
            lineages=tuple(results),
            hardware_target=environment.arch,
            agate_gpu=environment.gpu,
            roofline_mode=roofline_mode,
            roofline_detail=roofline_detail,
            problem_generalization_model=spec.problem_generalization_model,
            evolver_commit=self._evolver_commit,
        )

    def _resolve_evaluation_contract(
        self,
        contract: AgateEvaluationContractV1,
        *,
        existing_contract_digest: ArtifactDigest | None,
        operator: str,
        hardware_target: str,
    ) -> tuple[
        AgateEvaluationContractV1,
        RooflineMode,
        str | None,
    ]:
        """Build a missing Roofline once or recover the Campaign-sealed result."""
        if existing_contract_digest is not None and contract.roofline is None:
            stored = self._artifacts.verify(existing_contract_digest)
            if stored.kind is not ArtifactKind.EVALUATION_CONTRACT:
                raise ValueError("Campaign Evaluation Contract Artifact has the wrong kind")
            stored_contract = AgateEvaluationContractV1.model_validate(
                self._read_json(stored.payload_path / "value.json", "stored evaluation contract")
            )
            without_roofline = {"roofline": None}
            if contract == stored_contract.model_copy(update=without_roofline):
                if stored_contract.roofline is not None:
                    return stored_contract, "sealed-reuse", None
                return stored_contract, "profile-fallback", "Campaign has no sealed Roofline"
        if contract.roofline is not None:
            return contract, "explicit", None
        if contract.roofline is None and self._roofline_builder is not None:
            try:
                roofline = self._roofline_builder.build(
                    operator=operator,
                    hardware_target=hardware_target,
                    contract=contract,
                )
            except Exception as error:
                detail = f"{type(error).__name__}: {error}"[:1000]
                return contract, "profile-fallback", detail
            return contract.model_copy(update={"roofline": roofline}), "generated", None
        return contract, "profile-fallback", "Roofline Builder is not configured"

    def _bootstrap_lineage(
        self,
        spec: _LineageBootstrapSpec,
        *,
        campaign_contract: AgateEvaluationContractV1,
        campaign_contract_digest: ArtifactDigest,
        campaign_problem_digest: ArtifactDigest,
        agent_candidate: KernelAgentCandidate,
        source_provenance_digest: ArtifactDigest,
    ) -> BootstrapResult:
        """Execute one idempotent Lineage step, optionally reusing Campaign artifacts."""
        contract = campaign_contract
        self._require_real_directory(spec.baseline_kernel, "baseline Kernel")
        candidate_path = spec.baseline_kernel.joinpath(*contract.candidate_path.split("/"))
        if candidate_path.is_symlink() or not candidate_path.is_file():
            raise ValueError("baseline Kernel does not contain the contract candidate_path")
        try:
            candidate_text = candidate_path.read_text(encoding="utf-8")
        except UnicodeDecodeError as error:
            raise ValueError("baseline Kernel candidate must be UTF-8") from error
        if not candidate_text.strip():
            raise ValueError("baseline Kernel candidate cannot be empty")
        self._require_real_directory(spec.initial_evidence, "initial evidence")

        contract_digest = self._artifacts.put_json(
            contract.model_dump(mode="json"),
            ArtifactKind.EVALUATION_CONTRACT,
        )
        if contract_digest != campaign_contract_digest:
            raise ValueError("Campaign Lineages must share one Evaluation Contract")
        problem_digest = campaign_problem_digest
        if self._artifacts.verify(problem_digest).kind is not ArtifactKind.AGENT_PROBLEM:
            raise ValueError("bootstrap Agent Problem generator returned the wrong Artifact kind")
        if agent_candidate.dsl is not spec.dsl:
            raise ValueError("Campaign Lineage DSL disagrees with Kernel Agent manifest")
        input_kernel_digest = self._artifacts.put_directory(
            spec.baseline_kernel,
            ArtifactKind.KERNEL,
        )
        campaign_id = parse_campaign_id(self._derived_id("campaign", spec.creation_key))
        agent_id = parse_kernel_agent_revision_id(
            self._derived_id("agentrev", f"{spec.creation_key}:{spec.dsl.value}")
        )
        kernel_id = parse_kernel_revision_id(
            self._derived_id("kernelrev", f"{spec.creation_key}:{spec.dsl.value}:baseline")
        )
        lineage_id = parse_lineage_id(
            self._derived_id("lineage", f"{spec.creation_key}:{spec.dsl.value}")
        )
        bootstrap_attempt_id = parse_attempt_id(
            self._derived_id("attempt", f"{spec.creation_key}:{spec.dsl.value}:baseline")
        )
        evidence_assembler = LocalEvidenceAssembler(
            self._registry,
            self._artifacts,
        )
        bootstrap_input_digest = evidence_assembler.create_bootstrap_input(
            lineage_id,
            spec.initial_evidence,
        )
        created_at = self._clock()
        campaign = Campaign(
            campaign_id,
            spec.operator,
            spec.hardware_target,
            contract_digest,
            problem_digest,
            created_at,
            problem_generalization_model=spec.problem_generalization_model,
            evolver_commit=self._evolver_commit,
        )
        self._ensure_campaign(campaign)
        agent = self._registry.register_kernel_agent_revision(
            KernelAgentRevision(
                id=agent_id,
                parent_id=None,
                creation_key=f"bootstrap:{spec.creation_key}:{spec.dsl.value}",
                dsl=spec.dsl,
                optimizer_digest=agent_candidate.optimizer_digest,
                created_by="bootstrap",
                created_at=created_at,
                source_provenance_digest=source_provenance_digest,
            )
        )
        try:
            existing_lineage = self._registry.get_lineage(lineage_id)
        except KeyError:
            pass
        else:
            if (
                existing_lineage.campaign_id != campaign_id
                or existing_lineage.dsl is not spec.dsl
                or existing_lineage.hardware_target != spec.hardware_target
                or existing_lineage.challenger_count != spec.challenger_count
                or existing_lineage.challenger_start_epoch != spec.challenger_start_epoch
                or existing_lineage.trajectories_per_branch != spec.trajectories_per_branch
                or existing_lineage.attempts_per_trajectory != spec.attempts_per_trajectory
                or existing_lineage.optimizer_model != spec.optimizer_model
                or existing_lineage.evolver_model != spec.evolver_model
            ):
                raise ValueError("bootstrap creation_key resolved to a different lineage")
            if existing_lineage.next_epoch_number == 1 and (
                existing_lineage.active_kernel_agent_revision_id != agent.id
                or existing_lineage.best_kernel_revision_id != kernel_id
            ):
                raise ValueError(
                    "bootstrap creation_key resolved to different initial lineage state"
                )
            existing_kernel = self._registry.get_kernel_revision(kernel_id)
            initial_checkpoint = self._initial_evidence_checkpoint(
                existing_lineage.evidence_checkpoint
            )
            return BootstrapResult(
                campaign_id,
                lineage_id,
                agent.id,
                kernel_id,
                contract_digest,
                problem_digest,
                initial_checkpoint,
                existing_kernel.created_at,
                bootstrap_attempt_id,
                agent.created_at,
                agent.optimizer_digest,
                spec.optimizer_model,
                spec.evolver_model,
            )

        if self._baseline_generator is None:
            raise ValueError("campaign bootstrap requires the Core baseline generator")
        generated = self._baseline_generator.generate(
            bootstrap_attempt_id=bootstrap_attempt_id,
            campaign_id=campaign_id,
            lineage_id=lineage_id,
            kernel_agent_revision_id=agent.id,
            optimizer_digest=agent.optimizer_digest,
            input_kernel_digest=input_kernel_digest,
            evaluation_contract_digest=contract_digest,
            agent_problem_digest=problem_digest,
            evidence_digest=bootstrap_input_digest,
            dsl=spec.dsl,
            operator=spec.operator,
            hardware_target=spec.hardware_target,
            model=spec.optimizer_model,
        )
        kernel_digest = generated.kernel_digest
        gateway_digest = generated.gateway_result_digest
        latency_us = generated.latency_us
        evidence_digest = evidence_assembler.create_bootstrap(
            lineage_id,
            report_digest=generated.report_digest,
            session_trace_digest=generated.session_trace_digest,
        )
        if self._artifacts.verify(kernel_digest).kind is not ArtifactKind.KERNEL:
            raise ValueError("bootstrap baseline generator returned the wrong Kernel kind")
        if self._artifacts.verify(gateway_digest).kind is not ArtifactKind.GATEWAY_RESULT:
            raise ValueError("bootstrap baseline generator returned the wrong Gateway result kind")
        baseline = KernelRevision(
            kernel_id,
            None,
            kernel_digest,
            None,
            KernelEvaluation(True, latency_us, gateway_digest),
            created_at,
        )
        self._ensure_kernel(baseline)
        lineage = Lineage(
            id=lineage_id,
            campaign_id=campaign_id,
            dsl=spec.dsl,
            hardware_target=spec.hardware_target,
            active_kernel_agent_revision_id=agent.id,
            best_kernel_revision_id=baseline.id,
            evidence_checkpoint=evidence_digest,
            challenger_count=spec.challenger_count,
            challenger_start_epoch=spec.challenger_start_epoch,
            trajectories_per_branch=spec.trajectories_per_branch,
            attempts_per_trajectory=spec.attempts_per_trajectory,
            next_epoch_number=1,
            status=LineageStatus.READY,
            optimizer_model=spec.optimizer_model,
            evolver_model=spec.evolver_model,
        )
        self._ensure_lineage(lineage)
        return BootstrapResult(
            campaign_id,
            lineage_id,
            agent.id,
            baseline.id,
            contract_digest,
            problem_digest,
            evidence_digest,
            baseline.created_at,
            bootstrap_attempt_id,
            agent.created_at,
            agent.optimizer_digest,
            spec.optimizer_model,
            spec.evolver_model,
        )

    def _initial_evidence_checkpoint(
        self,
        digest: ArtifactDigest,
    ) -> ArtifactDigest:
        """Resolve the immutable through-Epoch-0 checkpoint from a cumulative chain."""
        visited: set[ArtifactDigest] = set()
        current = digest
        while current not in visited:
            visited.add(current)
            artifact = self._artifacts.verify(current)
            if artifact.kind is not ArtifactKind.EVIDENCE:
                raise ValueError("Lineage Evidence checkpoint has the wrong Artifact kind")
            checkpoint = EvidenceCheckpointV1.from_file(
                artifact.payload_path / "checkpoint.json"
            )
            if checkpoint.through_epoch == 0:
                return current
            if checkpoint.previous_checkpoint_digest is None:
                raise ValueError("Lineage Evidence chain has no Epoch-0 checkpoint")
            current = checkpoint.previous_checkpoint_digest
        raise ValueError("Lineage Evidence checkpoint chain contains a cycle")

    def _ensure_campaign(self, expected: Campaign) -> None:
        try:
            existing = self._registry.get_campaign(expected.id)
        except KeyError:
            self._registry.insert_campaign(expected)
            return
        if (
            existing.operator,
            existing.hardware_target,
            existing.evaluation_contract_digest,
            existing.agent_problem_digest,
            existing.problem_generalization_model,
            existing.evolver_commit,
        ) != (
            expected.operator,
            expected.hardware_target,
            expected.evaluation_contract_digest,
            expected.agent_problem_digest,
            expected.problem_generalization_model,
            expected.evolver_commit,
        ):
            raise ValueError("bootstrap creation_key resolved to a different Campaign")

    def _ensure_kernel(self, expected: KernelRevision) -> None:
        try:
            existing = self._registry.get_kernel_revision(expected.id)
        except KeyError:
            self._registry.register_kernel_revision(expected)
            return
        if (
            existing.parent_id,
            existing.artifact_digest,
            existing.produced_by_attempt_id,
            existing.evaluation,
        ) != (
            expected.parent_id,
            expected.artifact_digest,
            expected.produced_by_attempt_id,
            expected.evaluation,
        ):
            raise ValueError("bootstrap creation_key resolved to a different baseline Kernel")

    def _ensure_lineage(self, expected: Lineage) -> None:
        try:
            existing = self._registry.get_lineage(expected.id)
        except KeyError:
            self._registry.insert_lineage(expected)
            return
        if (
            existing.campaign_id,
            existing.dsl,
            existing.hardware_target,
            existing.challenger_count,
            existing.challenger_start_epoch,
            existing.trajectories_per_branch,
            existing.attempts_per_trajectory,
            existing.optimizer_model,
            existing.evolver_model,
        ) != (
            expected.campaign_id,
            expected.dsl,
            expected.hardware_target,
            expected.challenger_count,
            expected.challenger_start_epoch,
            expected.trajectories_per_branch,
            expected.attempts_per_trajectory,
            expected.optimizer_model,
            expected.evolver_model,
        ):
            raise ValueError("bootstrap creation_key resolved to a different lineage")
        if existing.next_epoch_number == 1 and (
            existing.evidence_checkpoint != expected.evidence_checkpoint
        ):
            raise ValueError("bootstrap creation_key resolved to different initial evidence")

    @staticmethod
    def _derived_id(prefix: str, key: str) -> str:
        suffix = hashlib.sha256(f"atrex-bootstrap:{prefix}:{key}".encode()).hexdigest()[:32]
        return f"{prefix}_{suffix}"

    @staticmethod
    def _require_real_directory(path: Path, label: str) -> None:
        try:
            path_stat = path.lstat()
        except FileNotFoundError as error:
            raise ValueError(f"{label} must be a real directory") from error
        if stat.S_ISLNK(path_stat.st_mode) or not stat.S_ISDIR(path_stat.st_mode):
            raise ValueError(f"{label} must be a real directory")

    @staticmethod
    def _read_json(path: Path, label: str) -> JsonValue:
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"{label} must be a regular file")
        try:
            value = json.loads(path.read_bytes())
        except json.JSONDecodeError as error:
            raise ValueError(f"{label} is not valid JSON") from error
        return _JSON_VALUE_ADAPTER.validate_python(value)
