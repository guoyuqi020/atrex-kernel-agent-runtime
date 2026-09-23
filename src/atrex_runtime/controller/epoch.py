"""Recoverable configurable Active-versus-Challenger-pool Epoch controller."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime

import anyio

from ..domain.errors import InfrastructureError, InvalidTransitionError
from ..domain.ids import (
    AttemptId,
    EpochId,
    KernelAgentRevisionId,
    KernelRevisionId,
    LineageId,
    new_attempt_id,
    new_epoch_id,
    new_kernel_agent_revision_id,
    new_kernel_revision_id,
    parse_attempt_id,
    parse_kernel_agent_revision_id,
    parse_kernel_revision_id,
)
from ..domain.models import (
    AgentSelectionReason,
    Attempt,
    AttemptReportStatus,
    AttemptStatus,
    BranchRole,
    BranchScore,
    ChallengerProposalType,
    Epoch,
    EpochBranchWorkflow,
    EpochChallenger,
    EpochSelection,
    EpochStatus,
    KernelAgentRevision,
    KernelEvaluation,
    KernelRevision,
    LineageStatus,
)
from ..ports import (
    AgentWorkflowOperationHandler,
    AgentWorkflowRunner,
    AttemptEvidenceAssembler,
    BuildAttemptEvidenceRequest,
    BuildChallengerRequest,
    EvolverRunner,
    KernelAgentNoChangeProposal,
    KernelAgentReuseProposal,
    KernelComparator,
    OptimizerRunner,
    RunAgentWorkflowRequest,
    RunAttemptRequest,
)
from ..registry.base import Registry
from ..selection import TrustedLatencyKernelComparator, select_best_kernel, select_kernel_agent


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True, slots=True)
class EpochRunResult:
    """Completed Epoch and every Agent Branch score used for selection."""

    epoch: Epoch
    scores: tuple[BranchScore, ...]

    @property
    def active_score(self) -> BranchScore:
        """Return the first and only Active score."""
        for score in self.scores:
            if score.branch is BranchRole.ACTIVE:
                return score
        raise InvalidTransitionError(f"Epoch {self.epoch.id} did not execute an Active Branch")

    @property
    def challenger_scores(self) -> tuple[BranchScore, ...]:
        """Return Challenger scores in configured ordinal order."""
        return tuple(score for score in self.scores if score.branch is BranchRole.CHALLENGER)


class _EpochWorkflowOperations(AgentWorkflowOperationHandler):
    """Capability surface exposed to one untrusted executable Epoch Workflow."""

    def __init__(
        self,
        controller: EpochController,
        epoch: Epoch,
        *,
        optimizer_attempt_budget: int,
    ) -> None:
        self._controller = controller
        self._epoch_id = epoch.id
        self._optimizer_attempt_budget = optimizer_attempt_budget
        self._program_sha256: str | None = None
        self._selected_kernel_revision_id: str | None = None
        self._selected_agent_revision_id: str | None = None
        self._selection_reason: AgentSelectionReason | None = None

    async def execute_workflow_operation(
        self,
        operation: str,
        arguments: Mapping[str, object],
    ) -> Mapping[str, object]:
        program_sha256 = arguments.get("_runtime_workflow_program_sha256")
        if not isinstance(program_sha256, str):
            raise ValueError("Runtime omitted trusted Workflow identity")
        if self._program_sha256 is None:
            self._program_sha256 = program_sha256
        elif self._program_sha256 != program_sha256:
            raise InvalidTransitionError("Workflow program identity changed during one Epoch")
        public = {
            key: value
            for key, value in arguments.items()
            if key != "_runtime_workflow_program_sha256"
        }
        if operation == "replicate_active":
            ordinal = self._positive_int(public, "challenger_ordinal")
            agent_revision_id = self._controller._workflow_replicate_active(
                self._epoch_id,
                ordinal,
            )
            return {"kernel_agent_revision_id": agent_revision_id}
        if operation == "evolve_agent":
            ordinal = self._positive_int(public, "challenger_ordinal")
            evolved_revision_id = await self._controller._workflow_evolve_agent(
                self._epoch_id,
                ordinal,
            )
            return {
                "kernel_agent_revision_id": evolved_revision_id,
                "created": evolved_revision_id is not None,
            }
        if operation == "create_trajectory":
            return self._controller._workflow_create_trajectory(
                self._epoch_id,
                public,
                optimizer_attempt_budget=self._optimizer_attempt_budget,
                program_sha256=program_sha256,
            )
        if operation == "run_attempts_parallel":
            launches = public.get("launches")
            if not isinstance(launches, list):
                raise ValueError("run_attempts_parallel requires a launches array")
            return await self._controller._workflow_run_attempts_parallel(
                self._epoch_id,
                tuple(launches),
                optimizer_attempt_budget=self._optimizer_attempt_budget,
                program_sha256=program_sha256,
            )
        if operation == "trajectory_status":
            return self._controller._workflow_trajectory_status(
                self._epoch_id,
                public,
            )
        if operation == "select_best_kernel":
            self._controller._workflow_validate_selection_ready(
                self._epoch_id,
                optimizer_attempt_budget=self._optimizer_attempt_budget,
                program_sha256=program_sha256,
            )
            if self._selected_kernel_revision_id is not None:
                selected = self._controller._registry.get_kernel_revision(
                    parse_kernel_revision_id(self._selected_kernel_revision_id)
                )
                return {
                    "kernel_revision_id": selected.id,
                    "latency_us": selected.evaluation.latency_us,
                }
            kernel_revision = self._controller._workflow_select_best_kernel(self._epoch_id)
            self._selected_kernel_revision_id = str(kernel_revision.id)
            return {
                "kernel_revision_id": kernel_revision.id,
                "latency_us": kernel_revision.evaluation.latency_us,
            }
        if operation == "compare_agents":
            self._controller._workflow_validate_selection_ready(
                self._epoch_id,
                optimizer_attempt_budget=self._optimizer_attempt_budget,
                program_sha256=program_sha256,
            )
            if self._selected_agent_revision_id is not None:
                return {
                    "kernel_agent_revision_id": self._selected_agent_revision_id,
                    "reason": (
                        None if self._selection_reason is None else self._selection_reason.value
                    ),
                }
            score, reason = await self._controller._workflow_compare_agents(self._epoch_id)
            self._selected_agent_revision_id = str(score.kernel_agent_revision_id)
            self._selection_reason = reason
            return {
                "kernel_agent_revision_id": score.kernel_agent_revision_id,
                "reason": None if reason is None else reason.value,
            }
        if operation == "complete_epoch":
            kernel_revision_id = public.get("kernel_revision_id")
            requested_agent_revision_id = public.get("kernel_agent_revision_id")
            if not isinstance(kernel_revision_id, str) or not isinstance(
                requested_agent_revision_id, str
            ):
                raise ValueError(
                    "complete_epoch requires kernel_revision_id and kernel_agent_revision_id"
                )
            if kernel_revision_id != self._selected_kernel_revision_id:
                raise InvalidTransitionError(
                    "Workflow must complete with select_best_kernel's exact result"
                )
            if requested_agent_revision_id != self._selected_agent_revision_id:
                raise InvalidTransitionError(
                    "Workflow must complete with compare_agents' exact result"
                )
            epoch = self._controller._workflow_complete_epoch(
                self._epoch_id,
                kernel_revision_id,
                requested_agent_revision_id,
                self._selection_reason,
            )
            return {"epoch_id": epoch.id, "status": epoch.status.value}
        raise ValueError(f"unsupported Workflow Runtime operation: {operation}")

    @staticmethod
    def _positive_int(arguments: Mapping[str, object], name: str) -> int:
        value = arguments.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
        return value


class EpochController:
    """Persist every intent and resume an epoch from its last committed transition."""

    def __init__(
        self,
        registry: Registry,
        evolver: EvolverRunner,
        optimizer: OptimizerRunner,
        attempt_evidence: AttemptEvidenceAssembler,
        *,
        kernel_retention_comparator: KernelComparator | None = None,
        agent_promotion_comparator: KernelComparator | None = None,
        max_infrastructure_retries: int = 2,
        kernel_measurement_uncertainty_us: float = 0.0,
        agent_measurement_uncertainty_us: float = 0.0,
        max_parallel_attempts: int = 4,
        workflow_runner: AgentWorkflowRunner | None = None,
        clock: Callable[[], str] = _utc_now,
        attempt_finished: Callable[[Epoch, Attempt], None] | None = None,
    ) -> None:
        if max_infrastructure_retries < 0:
            raise ValueError("max infrastructure retries cannot be negative")
        if kernel_measurement_uncertainty_us < 0:
            raise ValueError("Kernel measurement uncertainty cannot be negative")
        if agent_measurement_uncertainty_us < 0:
            raise ValueError("Agent measurement uncertainty cannot be negative")
        if max_parallel_attempts <= 0:
            raise ValueError("maximum parallel Attempts must be positive")
        self._registry = registry
        self._evolver = evolver
        self._optimizer = optimizer
        self._attempt_evidence = attempt_evidence
        self._kernel_retention_comparator = (
            kernel_retention_comparator
            or TrustedLatencyKernelComparator(kernel_measurement_uncertainty_us)
        )
        self._agent_promotion_comparator = agent_promotion_comparator
        self._max_infrastructure_retries = max_infrastructure_retries
        self._agent_measurement_uncertainty_us = agent_measurement_uncertainty_us
        self._max_parallel_attempts = max_parallel_attempts
        self._workflow_runner = workflow_runner
        self._clock = clock
        self._attempt_finished = attempt_finished

    async def run_epoch(
        self,
        lineage_id: LineageId,
        epoch_number: int,
    ) -> EpochRunResult:
        """Create or resume one stable epoch number until its selection is committed."""
        if epoch_number <= 0:
            raise ValueError("epoch number must be positive")

        epoch = self._registry.find_epoch(lineage_id, epoch_number)
        if epoch is None:
            lineage = self._registry.get_lineage(lineage_id)
            epoch = Epoch(
                id=new_epoch_id(),
                lineage_id=lineage.id,
                number=epoch_number,
                active_kernel_agent_revision_id=lineage.active_kernel_agent_revision_id,
                challenger_kernel_agent_revision_ids=(),
                starting_kernel_revision_id=lineage.best_kernel_revision_id,
                evidence_checkpoint=lineage.evidence_checkpoint,
                max_challengers=lineage.max_challengers,
                optimizer_attempt_budget=lineage.optimizer_attempt_budget,
                status=(
                    EpochStatus.READY
                    if lineage.max_challengers == 0
                    else EpochStatus.BUILDING_CHALLENGER
                ),
                winner_kernel_agent_revision_id=None,
                best_kernel_revision_id=None,
                created_at=self._clock(),
                completed_at=None,
            )
            self._registry.insert_epoch(epoch)

        if epoch.status is EpochStatus.STOPPED:
            epoch = self._registry.resume_stopped_epoch(epoch.id)
        if epoch.status is EpochStatus.FAILED:
            raise InvalidTransitionError(f"Epoch {epoch.id} has failed")
        if self._workflow_runner is None:
            raise RuntimeError("Epoch execution requires an Agent-owned Workflow")
        return await self._run_executable_workflow_epoch(epoch)

    async def _run_executable_workflow_epoch(self, epoch: Epoch) -> EpochRunResult:
        """Let the Active Agent Revision orchestrate one Epoch through Runtime services."""
        if epoch.status is EpochStatus.COMPLETED:
            return EpochRunResult(epoch, self._scores(epoch))
        if epoch.status not in {
            EpochStatus.BUILDING_CHALLENGER,
            EpochStatus.READY,
            EpochStatus.RUNNING,
            EpochStatus.SELECTING,
        }:
            raise InvalidTransitionError(
                f"Epoch {epoch.id} cannot run executable Workflow from {epoch.status}"
            )
        active = self._registry.get_kernel_agent_revision(epoch.active_kernel_agent_revision_id)
        operations = _EpochWorkflowOperations(
            self,
            epoch,
            optimizer_attempt_budget=epoch.optimizer_attempt_budget,
        )
        workflow_runner = self._workflow_runner
        if workflow_runner is None:
            raise AssertionError("executable Workflow runner disappeared")
        await workflow_runner.run(
            RunAgentWorkflowRequest(
                revision=active,
                epoch_id=epoch.id,
                epoch_number=epoch.number,
                max_challengers=epoch.max_challengers,
                optimizer_attempt_budget=epoch.optimizer_attempt_budget,
            ),
            operations,
        )
        epoch = self._registry.get_epoch(epoch.id)
        if epoch.status is not EpochStatus.COMPLETED:
            raise InvalidTransitionError(
                f"Agent Workflow exited without completing Epoch {epoch.id}"
            )
        return EpochRunResult(epoch, self._scores(epoch))

    async def _ensure_challengers(
        self,
        epoch: Epoch,
        *,
        through_ordinal: int | None = None,
    ) -> None:
        lineage = self._registry.get_lineage(epoch.lineage_id)
        parent = self._registry.get_kernel_agent_revision(epoch.active_kernel_agent_revision_id)
        target_ordinal = epoch.max_challengers
        if through_ordinal is not None:
            if through_ordinal <= 0 or through_ordinal > epoch.max_challengers:
                raise ValueError("Workflow Challenger ordinal exceeds the Runtime limit")
            target_ordinal = through_ordinal
        for challenger_ordinal in range(
            len(epoch.challenger_kernel_agent_revision_ids) + 1,
            target_ordinal + 1,
        ):
            creation_key = f"epoch:{epoch.id}:challenger:{challenger_ordinal}"
            revision = self._registry.find_kernel_agent_revision_by_creation_key(creation_key)
            proposal_type: ChallengerProposalType
            base_revision_id: KernelAgentRevisionId
            evolution_trace_digest = None
            if revision is None:
                agent_catalog = tuple(self._registry.list_lineage_agent_revisions(epoch.lineage_id))
                catalog_by_id = {entry.revision.id: entry for entry in agent_catalog}
                visible_by_id = {entry.revision.id: entry.revision for entry in agent_catalog}
                visible_by_id[parent.id] = parent
                observer_id = lineage.evolver_observer_lineage_id
                observer = (
                    None if observer_id is None else self._registry.get_lineage(observer_id)
                )
                if observer is not None:
                    if observer.dsl is not lineage.dsl:
                        raise InvalidTransitionError(
                            "Evolver observer disagrees with the Challenger DSL"
                        )
                    if (
                        observer.bootstrap_source_lineage_id
                        != lineage.bootstrap_source_lineage_id
                    ):
                        raise InvalidTransitionError(
                            "Evolver observer and Challenger do not share a Bootstrap Lineage"
                        )
                    observer_campaign = self._registry.get_campaign(observer.campaign_id)
                    challenger_campaign = self._registry.get_campaign(lineage.campaign_id)
                    if (
                        observer_campaign.operator,
                        observer_campaign.hardware_target,
                        observer_campaign.evaluation_contract_digest,
                        observer_campaign.agent_problem_digest,
                    ) != (
                        challenger_campaign.operator,
                        challenger_campaign.hardware_target,
                        challenger_campaign.evaluation_contract_digest,
                        challenger_campaign.agent_problem_digest,
                    ):
                        raise InvalidTransitionError(
                            "Evolver observer disagrees with the Challenger Campaign contract"
                        )
                observer_epoch = (
                    None
                    if observer is None
                    else self._registry.find_epoch(observer.id, epoch.number)
                )
                # The external Isolated control may run ahead. Its numbered Epoch input is
                # immutable and therefore gives the exact through-(N-1) view without leaking
                # any later observer work into this Evolution.
                observer_checkpoint = (
                    None
                    if observer is None
                    else (
                        observer_epoch.evidence_checkpoint
                        if observer_epoch is not None
                        else observer.evidence_checkpoint
                    )
                )
                if observer is not None and epoch.number > 1:
                    preceding = self._registry.find_epoch(observer.id, epoch.number - 1)
                    if (
                        preceding is None
                        or preceding.status is not EpochStatus.COMPLETED
                        or (
                            observer_epoch is None
                            and (
                                observer.next_epoch_number != epoch.number
                                or observer.status is not LineageStatus.READY
                            )
                        )
                    ):
                        raise InvalidTransitionError(
                            "Evolver observer has not published its preceding-Epoch checkpoint"
                        )
                observer_agent_catalog = (
                    ()
                    if observer is None
                    else tuple(
                        entry
                        for entry in self._registry.list_lineage_agent_revisions(observer.id)
                        if entry.introduced_epoch_number is None
                        or entry.introduced_epoch_number < epoch.number
                    )
                )
                observer_kernel_catalog = (
                    ()
                    if observer is None
                    else tuple(
                        entry
                        for entry in self._registry.list_lineage_kernels(observer.id)
                        if entry.epoch_number is None or entry.epoch_number < epoch.number
                    )
                )
                build = await self._evolver.build_challenger(
                    BuildChallengerRequest(
                        parent_revision=parent,
                        epoch_id=epoch.id,
                        evidence_checkpoint=epoch.evidence_checkpoint,
                        idempotency_key=creation_key,
                        agent_catalog=agent_catalog,
                        kernel_catalog=tuple(self._registry.list_lineage_kernels(epoch.lineage_id)),
                        model=lineage.evolver_model,
                        hardware_target=lineage.hardware_target,
                        epoch_number=epoch.number,
                        max_challengers=epoch.max_challengers,
                        optimizer_attempt_budget=epoch.optimizer_attempt_budget,
                        observer_lineage_id=observer_id,
                        observer_evidence_checkpoint=observer_checkpoint,
                        observer_agent_catalog=observer_agent_catalog,
                        observer_kernel_catalog=observer_kernel_catalog,
                    )
                )
                evolution_trace_digest = build.evolution_trace_digest
                if isinstance(build.proposal, KernelAgentNoChangeProposal):
                    self._registry.close_challenger_pool(
                        epoch.id,
                        challenger_ordinal - 1,
                        evolution_trace_digest,
                    )
                    return
                if isinstance(build.proposal, KernelAgentReuseProposal):
                    revision = visible_by_id.get(build.proposal.candidate_revision_id)
                    if revision is None:
                        raise ValueError("Evolver reused an Agent outside frozen lineage history")
                    if revision.id == parent.id:
                        raise ValueError("Evolver cannot reuse the current Active Agent")
                    if catalog_by_id[revision.id].introduced_epoch_id == epoch.id:
                        raise ValueError("Evolver cannot reuse a current-Epoch Challenger")
                    proposal_type = ChallengerProposalType.REUSE
                    base_revision_id = revision.id
                else:
                    proposal = build.proposal
                    base = visible_by_id.get(proposal.base_revision_id)
                    if base is None:
                        raise ValueError("Evolver used a base outside frozen lineage history")
                    proposal_type = ChallengerProposalType(proposal.proposal_type)
                    if proposal_type is ChallengerProposalType.EVOLVED and base.id != parent.id:
                        raise ValueError("evolved proposal base is not the current Active Agent")
                    if (
                        proposal_type is ChallengerProposalType.EVOLVE_FROM_HISTORY
                        and base.id == parent.id
                    ):
                        raise ValueError(
                            "evolve_from_history proposal used the current Active Agent"
                        )
                    if (
                        proposal_type is ChallengerProposalType.EVOLVE_FROM_HISTORY
                        and catalog_by_id[base.id].introduced_epoch_id == epoch.id
                    ):
                        raise ValueError(
                            "evolve_from_history proposal used a current-Epoch Challenger"
                        )
                    candidate = proposal.candidate
                    if candidate.dsl is not parent.dsl or base.dsl is not parent.dsl:
                        raise ValueError("Evolver changed the lineage DSL")
                    if candidate.runtime_state_digest is None:
                        raise ValueError(
                            "Evolver produced an incomplete Agent Bundle without Runtime State"
                        )
                    if (
                        candidate.optimizer_digest == base.optimizer_digest
                        and candidate.runtime_state_digest == base.runtime_state_digest
                    ):
                        raise ValueError(
                            "Evolver produced no Agent source or runtime-state changes"
                        )
                    base_revision_id = base.id
                    revision = self._registry.register_kernel_agent_revision(
                        KernelAgentRevision(
                            id=new_kernel_agent_revision_id(),
                            parent_id=base.id,
                            creation_key=creation_key,
                            dsl=candidate.dsl,
                            optimizer_digest=candidate.optimizer_digest,
                            created_by="evolver",
                            created_at=self._clock(),
                            evolution_trace_digest=build.evolution_trace_digest,
                            runtime_state_digest=candidate.runtime_state_digest,
                        )
                    )
            else:
                if revision.parent_id is None or revision.evolution_trace_digest is None:
                    raise InvalidTransitionError("Recovered Evolver revision lacks provenance")
                base_revision_id = revision.parent_id
                proposal_type = (
                    ChallengerProposalType.EVOLVED
                    if base_revision_id == parent.id
                    else ChallengerProposalType.EVOLVE_FROM_HISTORY
                )
                evolution_trace_digest = revision.evolution_trace_digest
            assert evolution_trace_digest is not None
            self._registry.attach_challenger(
                EpochChallenger(
                    epoch_id=epoch.id,
                    challenger_ordinal=challenger_ordinal,
                    kernel_agent_revision_id=revision.id,
                    proposal_type=proposal_type,
                    base_revision_id=base_revision_id,
                    evolution_trace_digest=evolution_trace_digest,
                )
            )
            epoch = self._registry.get_epoch(epoch.id)

    def _workflow_replicate_active(
        self,
        epoch_id: EpochId,
        challenger_ordinal: int,
    ) -> KernelAgentRevisionId:
        """Idempotently attach the Active Agent as an isolated Workflow Branch."""
        epoch = self._registry.get_epoch(epoch_id)
        if challenger_ordinal > epoch.max_challengers:
            raise ValueError("Replica Challenger exceeds the Runtime limit")
        existing = self._registry.list_epoch_challengers(epoch.id)
        if challenger_ordinal <= len(existing):
            challenger = existing[challenger_ordinal - 1]
            if (
                challenger.proposal_type is not ChallengerProposalType.REPLICA
                or challenger.kernel_agent_revision_id != epoch.active_kernel_agent_revision_id
            ):
                raise InvalidTransitionError(
                    "Recovered Workflow Challenger is not the requested Active replica"
                )
            return challenger.kernel_agent_revision_id
        if challenger_ordinal != len(existing) + 1:
            raise ValueError("Workflow must create Challenger replicas in ordinal order")
        parent = self._registry.get_kernel_agent_revision(epoch.active_kernel_agent_revision_id)
        self._registry.attach_challenger(
            EpochChallenger(
                epoch_id=epoch.id,
                challenger_ordinal=challenger_ordinal,
                kernel_agent_revision_id=parent.id,
                base_revision_id=parent.id,
                proposal_type=ChallengerProposalType.REPLICA,
                evolution_trace_digest=None,
            )
        )
        return parent.id

    async def _workflow_evolve_agent(
        self,
        epoch_id: EpochId,
        challenger_ordinal: int,
    ) -> KernelAgentRevisionId | None:
        """Idempotently invoke Evolver for exactly one indexed Challenger."""
        epoch = self._registry.get_epoch(epoch_id)
        existing = self._registry.list_epoch_challengers(epoch.id)
        if challenger_ordinal <= len(existing):
            challenger = existing[challenger_ordinal - 1]
            if challenger.proposal_type is ChallengerProposalType.REPLICA:
                raise InvalidTransitionError(
                    "Recovered Workflow Challenger is a replica, not an Evolver result"
                )
            return challenger.kernel_agent_revision_id
        if challenger_ordinal != len(existing) + 1:
            raise ValueError("Workflow must evolve Challengers in ordinal order")
        await self._ensure_challengers(
            epoch,
            through_ordinal=challenger_ordinal,
        )
        epoch = self._registry.get_epoch(epoch.id)
        if challenger_ordinal > len(epoch.challenger_kernel_agent_revision_ids):
            return None
        return epoch.challenger_kernel_agent_revision_ids[challenger_ordinal - 1]

    def _workflow_create_trajectory(
        self,
        epoch_id: EpochId,
        arguments: Mapping[str, object],
        *,
        optimizer_attempt_budget: int,
        program_sha256: str,
    ) -> Mapping[str, object]:
        """Freeze one Branch capacity and return one stable Trajectory handle.

        The operation is intentionally idempotent. A restarted Workflow can recreate
        the same handles, but it cannot mutate a Branch plan after the first Attempt.
        Execution is separate and happens through ``run_attempts_parallel``.
        """
        epoch = self._registry.get_epoch(epoch_id)
        label = arguments.get("branch")
        if not isinstance(label, str):
            raise ValueError("create_trajectory requires a branch label")
        branch, challenger_ordinal = self._workflow_branch_identity(label)
        trajectory_ordinal = self._positive_workflow_int(arguments, "trajectory_ordinal")
        trajectory_count = self._positive_workflow_int(arguments, "trajectory_count")
        attempt_capacity = self._positive_workflow_int(arguments, "attempt_capacity")
        if trajectory_ordinal > trajectory_count:
            raise ValueError("trajectory_ordinal exceeds trajectory_count")
        attached = self._registry.list_epoch_challengers(epoch.id)
        if branch is BranchRole.ACTIVE:
            revision_id = epoch.active_kernel_agent_revision_id
        else:
            if challenger_ordinal > len(attached):
                raise ValueError(f"Workflow Branch {label} has no attached Challenger Agent")
            revision_id = attached[challenger_ordinal - 1].kernel_agent_revision_id

        existing = self._registry.get_epoch_branch_workflow(
            epoch.id,
            branch,
            challenger_ordinal,
        )
        expected = EpochBranchWorkflow(
            epoch_id=epoch.id,
            branch=branch,
            challenger_ordinal=challenger_ordinal,
            kernel_agent_revision_id=revision_id,
            kind=("executable_epoch_workflow_v2" if existing is None else existing.kind),
            program_sha256=program_sha256,
            trajectories=trajectory_count,
            attempts_per_trajectory=attempt_capacity,
            created_at=(self._clock() if existing is None else existing.created_at),
        )
        if existing is not None:
            if existing != expected:
                raise InvalidTransitionError(
                    "Recovered Trajectory differs from its frozen Branch plan"
                )
            workflow = existing
        else:
            if epoch.status not in {
                EpochStatus.BUILDING_CHALLENGER,
                EpochStatus.READY,
            }:
                raise InvalidTransitionError(
                    "Workflow cannot add a new Trajectory after Attempt execution starts"
                )
            planned = sum(
                item.attempt_budget for item in self._registry.list_epoch_branch_workflows(epoch.id)
            )
            requested = trajectory_count * attempt_capacity
            if planned + requested > optimizer_attempt_budget:
                raise ValueError(
                    "Workflow Branch capacities exceed the Runtime Optimizer Attempt budget: "
                    f"requested {planned + requested}, allowed {optimizer_attempt_budget}"
                )
            workflow = self._registry.ensure_epoch_branch_workflow(expected)

        self._registry.freeze_epoch_workflow_challengers(
            epoch.id,
            len(attached),
            program_sha256,
        )
        return {
            "branch": label,
            "trajectory_ordinal": trajectory_ordinal,
            "attempt_capacity": workflow.attempts_per_trajectory,
            "kernel_agent_revision_id": workflow.kernel_agent_revision_id,
        }

    async def _workflow_run_attempts_parallel(
        self,
        epoch_id: EpochId,
        raw_launches: tuple[object, ...],
        *,
        optimizer_attempt_budget: int,
        program_sha256: str,
    ) -> Mapping[str, object]:
        """Execute explicit logical Attempts, possibly across Trajectories in parallel."""
        if not raw_launches:
            raise ValueError("run_attempts_parallel requires at least one launch")
        epoch = self._registry.get_epoch(epoch_id)
        self._workflow_validate_plan(
            epoch,
            optimizer_attempt_budget=optimizer_attempt_budget,
            program_sha256=program_sha256,
        )
        if epoch.status is EpochStatus.READY:
            self._registry.transition_epoch(
                epoch.id,
                EpochStatus.READY,
                EpochStatus.RUNNING,
            )
            epoch = self._registry.get_epoch(epoch.id)
        if epoch.status is not EpochStatus.RUNNING:
            raise InvalidTransitionError(
                f"Workflow cannot run Attempts while Epoch is {epoch.status}"
            )

        launches: list[
            tuple[BranchRole, int, int, int, KernelRevisionId | None, AttemptId | None]
        ] = []
        trajectory_identities: set[tuple[BranchRole, int, int]] = set()
        attempt_identities: set[tuple[BranchRole, int, int, int]] = set()
        for raw in raw_launches:
            if not isinstance(raw, dict):
                raise ValueError("each Workflow Attempt launch must be an object")
            label = raw.get("branch")
            if not isinstance(label, str):
                raise ValueError("Attempt launch requires a branch label")
            branch, challenger_ordinal = self._workflow_branch_identity(label)
            trajectory_ordinal = self._positive_workflow_int(raw, "trajectory_ordinal")
            attempt_ordinal = self._positive_workflow_int(raw, "attempt_ordinal")
            workflow = self._registry.get_epoch_branch_workflow(
                epoch.id,
                branch,
                challenger_ordinal,
            )
            if workflow is None:
                raise ValueError(f"Attempt launch names an unregistered Branch: {label}")
            if trajectory_ordinal > workflow.trajectories:
                raise ValueError("Attempt launch Trajectory exceeds its registered capacity")
            if attempt_ordinal > workflow.attempts_per_trajectory:
                raise ValueError("Attempt launch ordinal exceeds its registered capacity")
            trajectory_identity = (branch, challenger_ordinal, trajectory_ordinal)
            if trajectory_identity in trajectory_identities:
                raise ValueError(
                    "one parallel batch cannot launch two Attempts on the same Trajectory"
                )
            trajectory_identities.add(trajectory_identity)
            attempt_identity = (*trajectory_identity, attempt_ordinal)
            if attempt_identity in attempt_identities:
                raise ValueError("Workflow Attempt launch is duplicated")
            attempt_identities.add(attempt_identity)

            raw_kernel = raw.get("input_kernel_revision_id")
            input_kernel_id = None
            if raw_kernel is not None:
                if not isinstance(raw_kernel, str):
                    raise ValueError("input_kernel_revision_id must be a string or null")
                input_kernel_id = parse_kernel_revision_id(raw_kernel)
            raw_state_attempt = raw.get("input_state_from_attempt_id")
            state_attempt_id = None
            if raw_state_attempt is not None:
                if not isinstance(raw_state_attempt, str):
                    raise ValueError("input_state_from_attempt_id must be a string or null")
                state_attempt_id = parse_attempt_id(raw_state_attempt)
            launches.append(
                (
                    branch,
                    challenger_ordinal,
                    trajectory_ordinal,
                    attempt_ordinal,
                    input_kernel_id,
                    state_attempt_id,
                )
            )

        limiter = anyio.Semaphore(self._max_parallel_attempts)
        outcomes: list[tuple[int, Mapping[str, object]]] = []
        failures: list[BaseException] = []

        async def run_one(
            index: int,
            launch: tuple[
                BranchRole,
                int,
                int,
                int,
                KernelRevisionId | None,
                AttemptId | None,
            ],
        ) -> None:
            async with limiter:
                try:
                    outcome = await self._workflow_run_attempt(epoch, *launch)
                    outcomes.append((index, outcome))
                except BaseException as error:
                    failures.append(error)

        async with anyio.create_task_group() as tasks:
            for index, launch in enumerate(launches):
                tasks.start_soon(run_one, index, launch)
        if failures:
            infrastructure = next(
                (
                    self._first_infrastructure_error(error)
                    for error in failures
                    if self._first_infrastructure_error(error) is not None
                ),
                None,
            )
            if infrastructure is not None:
                raise infrastructure
            raise failures[0]
        return {"attempts": [value for _, value in sorted(outcomes)]}

    async def _workflow_run_attempt(
        self,
        epoch: Epoch,
        branch: BranchRole,
        challenger_ordinal: int,
        trajectory_ordinal: int,
        attempt_ordinal: int,
        requested_kernel_id: KernelRevisionId | None,
        state_from_attempt_id: AttemptId | None,
    ) -> Mapping[str, object]:
        workflow = self._registry.get_epoch_branch_workflow(
            epoch.id,
            branch,
            challenger_ordinal,
        )
        if workflow is None:
            raise ValueError("Attempt launch has no registered Branch Workflow")
        input_kernel_id = epoch.starting_kernel_revision_id
        for previous_ordinal in range(1, attempt_ordinal):
            previous = self._registry.find_attempt(
                epoch.id,
                branch,
                challenger_ordinal,
                trajectory_ordinal,
                previous_ordinal,
            )
            if previous is None or previous.status is not AttemptStatus.COMPLETED:
                raise InvalidTransitionError(
                    "Workflow must complete earlier Trajectory Attempts before a later ordinal"
                )
        if requested_kernel_id is not None:
            self._workflow_validate_kernel_input(epoch, requested_kernel_id)
            input_kernel_id = requested_kernel_id

        attempt = self._registry.find_attempt(
            epoch.id,
            branch,
            challenger_ordinal,
            trajectory_ordinal,
            attempt_ordinal,
        )
        if attempt is None:
            attempt_id = new_attempt_id()
            evidence_digest = self._attempt_evidence.assemble(
                BuildAttemptEvidenceRequest(
                    attempt_id=attempt_id,
                    epoch_id=epoch.id,
                    branch=branch,
                    challenger_ordinal=challenger_ordinal,
                    trajectory_ordinal=trajectory_ordinal,
                    ordinal=attempt_ordinal,
                    epoch_evidence_checkpoint=epoch.evidence_checkpoint,
                )
            )
            created_at = self._clock()
            attempt = Attempt(
                id=attempt_id,
                epoch_id=epoch.id,
                branch=branch,
                challenger_ordinal=challenger_ordinal,
                trajectory_ordinal=trajectory_ordinal,
                ordinal=attempt_ordinal,
                kernel_agent_revision_id=workflow.kernel_agent_revision_id,
                input_kernel_revision_id=input_kernel_id,
                attempt_evidence_digest=evidence_digest,
                output_kernel_revision_id=None,
                accepted_as_branch_best=False,
                status=AttemptStatus.RUNNING,
                infrastructure_failures=0,
                recovery_generation=0,
                authority_started_at=created_at,
                failure_reason=None,
                created_at=created_at,
                completed_at=None,
            )
            self._registry.insert_attempt(attempt)
        elif (
            attempt.input_kernel_revision_id != input_kernel_id
            or attempt.kernel_agent_revision_id != workflow.kernel_agent_revision_id
        ):
            raise InvalidTransitionError(
                "Recovered Workflow Attempt differs from its original Kernel or Agent input"
            )

        if state_from_attempt_id is not None:
            source = self._registry.get_attempt(state_from_attempt_id)
            if source.epoch_id != epoch.id or source.status is not AttemptStatus.COMPLETED:
                raise ValueError("Runtime State source must be a completed Attempt in this Epoch")
            if source.kernel_agent_revision_id != attempt.kernel_agent_revision_id:
                raise ValueError("Runtime State cannot cross Kernel Agent revisions")
            if source.runtime_state_digest is None:
                raise ValueError("Runtime State source Attempt has no State checkpoint")
            if attempt.input_runtime_state_digest is None:
                self._registry.record_attempt_input_runtime_state(
                    attempt.id,
                    source.runtime_state_digest,
                )
                self._registry.record_runtime_event(
                    "epoch.workflow_state_routed",
                    epoch.id,
                    {
                        "source_attempt_id": source.id,
                        "target_attempt_id": attempt.id,
                        "runtime_state_digest": source.runtime_state_digest,
                    },
                )
                attempt = self._registry.get_attempt(attempt.id)
            elif attempt.input_runtime_state_digest != source.runtime_state_digest:
                raise InvalidTransitionError(
                    "Recovered Workflow Attempt names a different Runtime State source"
                )

        evidence_request = BuildAttemptEvidenceRequest(
            attempt_id=attempt.id,
            epoch_id=epoch.id,
            branch=branch,
            challenger_ordinal=challenger_ordinal,
            trajectory_ordinal=trajectory_ordinal,
            ordinal=attempt_ordinal,
            epoch_evidence_checkpoint=epoch.evidence_checkpoint,
        )
        self._attempt_evidence.validate(attempt.attempt_evidence_digest, evidence_request)
        was_completed = attempt.status is AttemptStatus.COMPLETED
        attempt = await self._finish_attempt(epoch, attempt)
        if not was_completed and self._attempt_finished is not None:
            with suppress(Exception):
                self._attempt_finished(epoch, attempt)
        return self._workflow_attempt_projection(attempt)

    def _workflow_trajectory_status(
        self,
        epoch_id: EpochId,
        arguments: Mapping[str, object],
    ) -> Mapping[str, object]:
        epoch = self._registry.get_epoch(epoch_id)
        label = arguments.get("branch")
        if not isinstance(label, str):
            raise ValueError("trajectory_status requires a branch label")
        branch, challenger_ordinal = self._workflow_branch_identity(label)
        trajectory_ordinal = self._positive_workflow_int(arguments, "trajectory_ordinal")
        workflow = self._registry.get_epoch_branch_workflow(
            epoch.id,
            branch,
            challenger_ordinal,
        )
        if workflow is None or trajectory_ordinal > workflow.trajectories:
            raise ValueError("trajectory_status names an unregistered Trajectory")
        attempts = [
            attempt
            for attempt in self._registry.list_attempts(epoch.id)
            if attempt.branch is branch
            and attempt.challenger_ordinal == challenger_ordinal
            and attempt.trajectory_ordinal == trajectory_ordinal
        ]
        current_kernel_id = epoch.starting_kernel_revision_id
        for attempt in attempts:
            if attempt.accepted_as_branch_best and attempt.output_kernel_revision_id is not None:
                current_kernel_id = attempt.output_kernel_revision_id
        current = self._registry.get_kernel_revision(current_kernel_id)
        return {
            "branch": label,
            "trajectory_ordinal": trajectory_ordinal,
            "attempt_capacity": workflow.attempts_per_trajectory,
            "completed_attempts": sum(
                attempt.status is AttemptStatus.COMPLETED for attempt in attempts
            ),
            "current_kernel_revision_id": current.id,
            "current_latency_us": current.evaluation.latency_us,
            "attempts": [self._workflow_attempt_projection(attempt) for attempt in attempts],
        }

    def _workflow_validate_plan(
        self,
        epoch: Epoch,
        *,
        optimizer_attempt_budget: int,
        program_sha256: str,
    ) -> None:
        workflows = self._registry.list_epoch_branch_workflows(epoch.id)
        identities = {(item.branch, item.challenger_ordinal) for item in workflows}
        if not identities:
            raise ValueError("Workflow must register at least one Branch")
        attached = len(self._registry.list_epoch_challengers(epoch.id))
        if any(
            branch is BranchRole.CHALLENGER and ordinal > attached
            for branch, ordinal in identities
        ):
            raise ValueError("Workflow registered a Challenger Branch without an attached Agent")
        if any(item.program_sha256 != program_sha256 for item in workflows):
            raise InvalidTransitionError("Workflow Branch was frozen by another program")
        planned = sum(item.attempt_budget for item in workflows)
        if planned != optimizer_attempt_budget:
            raise ValueError(
                "Workflow must allocate the exact Runtime Optimizer Attempt budget: "
                f"allocated {planned}, required {optimizer_attempt_budget}"
            )

    def _workflow_validate_selection_ready(
        self,
        epoch_id: EpochId,
        *,
        optimizer_attempt_budget: int,
        program_sha256: str,
    ) -> None:
        epoch = self._registry.get_epoch(epoch_id)
        self._workflow_validate_plan(
            epoch,
            optimizer_attempt_budget=optimizer_attempt_budget,
            program_sha256=program_sha256,
        )
        workflows = self._registry.list_epoch_branch_workflows(epoch.id)
        planned = sum(item.attempt_budget for item in workflows)
        attempts = self._registry.list_attempts(epoch.id)
        if len(attempts) != planned:
            raise InvalidTransitionError(
                "Workflow cannot select before every allocated Attempt has run: "
                f"completed or present {len(attempts)}, required {planned}"
            )
        if any(attempt.status is not AttemptStatus.COMPLETED for attempt in attempts):
            raise InvalidTransitionError("Workflow cannot select with unfinished Attempts")

    def _workflow_validate_kernel_input(
        self,
        epoch: Epoch,
        kernel_revision_id: KernelRevisionId,
    ) -> None:
        if kernel_revision_id == epoch.starting_kernel_revision_id:
            return
        for attempt in self._registry.list_attempts(epoch.id):
            if (
                attempt.status is AttemptStatus.COMPLETED
                and attempt.accepted_as_branch_best
                and attempt.output_kernel_revision_id == kernel_revision_id
            ):
                return
        raise ValueError(
            "input_kernel_revision_id must be the Epoch start or an accepted Kernel "
            "from a completed Attempt in this Epoch"
        )

    def _workflow_attempt_projection(self, attempt: Attempt) -> Mapping[str, object]:
        output = (
            None
            if attempt.output_kernel_revision_id is None
            else self._registry.get_kernel_revision(attempt.output_kernel_revision_id)
        )
        trajectory_kernel_id = (
            attempt.output_kernel_revision_id
            if attempt.accepted_as_branch_best and attempt.output_kernel_revision_id is not None
            else attempt.input_kernel_revision_id
        )
        return {
            "attempt_id": attempt.id,
            "branch": (
                "active"
                if attempt.branch is BranchRole.ACTIVE
                else f"challenger-{attempt.challenger_ordinal}"
            ),
            "trajectory_ordinal": attempt.trajectory_ordinal,
            "attempt_ordinal": attempt.ordinal,
            "status": attempt.status.value,
            "input_kernel_revision_id": attempt.input_kernel_revision_id,
            "output_kernel_revision_id": attempt.output_kernel_revision_id,
            "trajectory_kernel_revision_id": trajectory_kernel_id,
            "accepted": attempt.accepted_as_branch_best,
            "correct": None if output is None else output.evaluation.correct,
            "latency_us": None if output is None else output.evaluation.latency_us,
            "failure_reason": attempt.failure_reason,
            "runtime_state_available": attempt.runtime_state_digest is not None,
            "output_state_from_attempt_id": (
                attempt.id if attempt.runtime_state_digest is not None else None
            ),
        }

    @staticmethod
    def _positive_workflow_int(arguments: Mapping[str, object], name: str) -> int:
        value = arguments.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
        return value

    def _workflow_select_best_kernel(self, epoch_id: EpochId) -> KernelRevision:
        epoch = self._registry.get_epoch(epoch_id)
        self._scores(epoch)
        return select_best_kernel(self._retained_kernels(epoch))

    async def _workflow_compare_agents(
        self,
        epoch_id: EpochId,
    ) -> tuple[BranchScore, AgentSelectionReason | None]:
        epoch = self._registry.get_epoch(epoch_id)
        scores = self._scores(epoch)
        return await self._select_kernel_agent(epoch, scores)

    def _workflow_complete_epoch(
        self,
        epoch_id: EpochId,
        kernel_revision_id: str,
        agent_revision_id: str,
        selection_reason: AgentSelectionReason | None,
    ) -> Epoch:
        epoch = self._registry.get_epoch(epoch_id)
        kernel_id = parse_kernel_revision_id(kernel_revision_id)
        agent_id = parse_kernel_agent_revision_id(agent_revision_id)
        if epoch.status is EpochStatus.COMPLETED:
            if (
                epoch.best_kernel_revision_id != kernel_id
                or epoch.winner_kernel_agent_revision_id != agent_id
            ):
                raise InvalidTransitionError(
                    "Recovered Epoch was completed with different Workflow selections"
                )
            return epoch
        trusted_kernel = select_best_kernel(self._retained_kernels(epoch))
        if kernel_id != trusted_kernel.id:
            raise InvalidTransitionError("Workflow selected a non-best Kernel")
        valid_agents = {score.kernel_agent_revision_id for score in self._scores(epoch)}
        if agent_id not in valid_agents:
            raise InvalidTransitionError("Workflow selected an Agent outside completed Branches")
        if epoch.status is EpochStatus.RUNNING:
            self._registry.transition_epoch(
                epoch.id,
                EpochStatus.RUNNING,
                EpochStatus.SELECTING,
            )
            epoch = self._registry.get_epoch(epoch.id)
        if epoch.status is not EpochStatus.SELECTING:
            raise InvalidTransitionError(f"Workflow cannot complete Epoch from {epoch.status}")
        self._registry.complete_epoch(
            epoch.id,
            EpochSelection(
                winner_kernel_agent_revision_id=agent_id,
                best_kernel_revision_id=kernel_id,
                selection_reason=selection_reason,
            ),
        )
        return self._registry.get_epoch(epoch.id)

    @staticmethod
    def _workflow_branch_identity(label: str) -> tuple[BranchRole, int]:
        if label == "active":
            return BranchRole.ACTIVE, 0
        prefix = "challenger-"
        if not label.startswith(prefix):
            raise ValueError(
                "Workflow Branch label must be active or challenger-<positive ordinal>"
            )
        raw = label[len(prefix) :]
        if not raw.isdigit() or int(raw) <= 0:
            raise ValueError("Workflow Challenger label has an invalid ordinal")
        return BranchRole.CHALLENGER, int(raw)

    @classmethod
    def _first_infrastructure_error(
        cls,
        error: BaseException,
    ) -> InfrastructureError | None:
        if isinstance(error, InfrastructureError):
            return error
        if isinstance(error, BaseExceptionGroup):
            for item in error.exceptions:
                found = cls._first_infrastructure_error(item)
                if found is not None:
                    return found
        return None

    async def _finish_attempt(self, epoch: Epoch, attempt: Attempt) -> Attempt:
        while attempt.status is not AttemptStatus.COMPLETED:
            if attempt.status is AttemptStatus.INFRASTRUCTURE_FAILED:
                if attempt.infrastructure_failures > self._max_infrastructure_retries:
                    reason = (
                        f"Attempt {attempt.id} exceeded infrastructure retry budget: "
                        f"{attempt.failure_reason}"
                    )
                    raise InfrastructureError(reason)
                self._registry.retry_attempt(attempt.id)
                attempt = self._registry.get_attempt(attempt.id)

            registered = self._registry.find_kernel_revision_by_attempt(attempt.id)
            if registered is not None:
                try:
                    await self._complete_registered_attempt(attempt, registered)
                except Exception as error:
                    infrastructure = self._first_infrastructure_error(error)
                    if infrastructure is None:
                        raise
                    self._registry.record_infrastructure_failure(
                        attempt.id,
                        str(infrastructure),
                    )
                attempt = self._registry.get_attempt(attempt.id)
                continue

            try:
                result = await self._optimizer.run_attempt(
                    RunAttemptRequest(
                        attempt_id=attempt.id,
                        kernel_agent_revision_id=attempt.kernel_agent_revision_id,
                        input_kernel_revision_id=attempt.input_kernel_revision_id,
                        epoch_evidence_checkpoint=epoch.evidence_checkpoint,
                        attempt_evidence_digest=attempt.attempt_evidence_digest,
                        dsl=self._registry.get_kernel_agent_revision(
                            attempt.kernel_agent_revision_id
                        ).dsl,
                        model=self._registry.get_lineage(epoch.lineage_id).optimizer_model,
                    )
                )
            except InfrastructureError as error:
                self._registry.record_infrastructure_failure(attempt.id, str(error))
                attempt = self._registry.get_attempt(attempt.id)
                continue

            if result.attempt_report_digest is not None:
                if result.attempt_report_status is None:
                    raise AssertionError("Attempt report Digest has no status")
                self._registry.record_attempt_report(
                    attempt.id,
                    result.attempt_report_digest,
                    result.attempt_report_status,
                )
                attempt = self._registry.get_attempt(attempt.id)

            if result.candidate is None:
                self._registry.complete_attempt(
                    attempt.id,
                    None,
                    accepted_as_branch_best=False,
                    failure_reason=result.failure_reason or "Optimizer produced no candidate",
                )
            else:
                evaluation = KernelEvaluation(
                    correct=result.candidate.correct,
                    latency_us=result.candidate.latency_us,
                    gateway_result_digest=result.candidate.gateway_result_digest,
                )
                self._registry.register_kernel_revision(
                    KernelRevision(
                        id=new_kernel_revision_id(),
                        parent_id=attempt.input_kernel_revision_id,
                        artifact_digest=result.candidate.artifact_digest,
                        produced_by_attempt_id=attempt.id,
                        evaluation=evaluation,
                        created_at=self._clock(),
                    )
                )
            attempt = self._registry.get_attempt(attempt.id)
        return attempt

    async def _complete_registered_attempt(
        self,
        attempt: Attempt,
        output: KernelRevision,
    ) -> None:
        input_kernel = self._registry.get_kernel_revision(attempt.input_kernel_revision_id)
        report_ready = attempt.attempt_report_status is AttemptReportStatus.CANDIDATE_READY
        if not report_ready:
            self._registry.complete_attempt(
                attempt.id,
                output.id,
                accepted_as_branch_best=False,
                failure_reason="candidate lacks a candidate_ready Attempt report",
            )
            return

        # The Agent-authored terminal Report is immutable and deliberately does not
        # own Kernel retention. Only after that handoff is durably registered may
        # Runtime execute ordinary comparison or same-allocation ABBA and replace
        # the provisional Candidate evaluation with the authoritative result.
        comparison = await self._kernel_retention_comparator.compare(input_kernel, output)
        authoritative = comparison.authoritative_candidate
        if authoritative is not None:
            if authoritative.artifact_digest != output.artifact_digest:
                raise InvalidTransitionError("Kernel comparator finalized a different Candidate")
            output = self._registry.finalize_kernel_revision_evaluation(
                output.id,
                KernelEvaluation(
                    correct=authoritative.correct,
                    latency_us=authoritative.latency_us,
                    gateway_result_digest=authoritative.gateway_result_digest,
                ),
            )
        improved = comparison.accepted
        report_failure = comparison.reason if not comparison.accepted else None
        self._registry.complete_attempt(
            attempt.id,
            output.id,
            accepted_as_branch_best=improved,
            failure_reason=report_failure,
        )

    def _scores(self, epoch: Epoch) -> tuple[BranchScore, ...]:
        workflows = self._registry.list_epoch_branch_workflows(epoch.id)
        if workflows:
            return tuple(
                self._score_branch(
                    epoch,
                    workflow.branch,
                    workflow.challenger_ordinal,
                    workflow.kernel_agent_revision_id,
                )
                for workflow in workflows
            )
        return (
            self._score_branch(
                epoch,
                BranchRole.ACTIVE,
                0,
                epoch.active_kernel_agent_revision_id,
            ),
            *(
                self._score_branch(
                    epoch,
                    BranchRole.CHALLENGER,
                    challenger_ordinal,
                    revision_id,
                )
                for challenger_ordinal, revision_id in enumerate(
                    epoch.challenger_kernel_agent_revision_ids,
                    start=1,
                )
            ),
        )

    def _score_branch(
        self,
        epoch: Epoch,
        branch: BranchRole,
        challenger_ordinal: int,
        revision_id: KernelAgentRevisionId,
    ) -> BranchScore:
        starting = self._registry.get_kernel_revision(epoch.starting_kernel_revision_id)
        if starting.evaluation.latency_us is None:
            raise InvalidTransitionError(f"Epoch {epoch.id} has no valid starting latency")
        best_latency = starting.evaluation.latency_us
        first_best = 0
        strict_improvements = 0
        valid_candidates = 0
        failed_candidates = 0
        attempts = [
            attempt
            for attempt in self._registry.list_attempts(epoch.id)
            if attempt.branch is branch and attempt.challenger_ordinal == challenger_ordinal
        ]
        workflow = self._registry.get_epoch_branch_workflow(
            epoch.id,
            branch,
            challenger_ordinal,
        )
        if workflow is None:
            raise InvalidTransitionError(f"Branch {branch} has no Agent Workflow plan")
        expected_attempts = workflow.attempt_budget
        if len(attempts) != expected_attempts:
            raise InvalidTransitionError(f"Branch {branch} has an incomplete Attempt set")
        for attempt in attempts:
            if attempt.status is not AttemptStatus.COMPLETED:
                raise InvalidTransitionError(f"Attempt {attempt.id} is not completed")
            if attempt.output_kernel_revision_id is None:
                failed_candidates += 1
                continue
            output = self._registry.get_kernel_revision(attempt.output_kernel_revision_id)
            if not output.evaluation.correct:
                failed_candidates += 1
                continue
            valid_candidates += 1
            if attempt.accepted_as_branch_best:
                if output.evaluation.latency_us is None:
                    raise InvalidTransitionError(f"Correct Kernel {output.id} has no latency")
                if output.evaluation.latency_us < best_latency:
                    best_latency = output.evaluation.latency_us
                    first_best = (
                        (attempt.trajectory_ordinal - 1)
                        * workflow.attempts_per_trajectory
                        + attempt.ordinal
                    )
                strict_improvements += 1
        return BranchScore(
            branch=branch,
            challenger_ordinal=challenger_ordinal,
            kernel_agent_revision_id=revision_id,
            best_latency_us=best_latency,
            first_best_attempt=first_best,
            strict_improvements=strict_improvements,
            valid_candidates=valid_candidates,
            failed_candidates=failed_candidates,
        )

    async def _select_kernel_agent(
        self,
        epoch: Epoch,
        scores: tuple[BranchScore, ...],
    ) -> tuple[BranchScore, AgentSelectionReason | None]:
        if not scores:
            raise InvalidTransitionError(f"Epoch {epoch.id} has no Agent scores")
        winner = scores[0]
        reason: AgentSelectionReason | None = None
        for candidate in scores[1:]:
            if winner.kernel_agent_revision_id == candidate.kernel_agent_revision_id:
                # Replica branches optimize independently, but cannot promote an Agent
                # over itself. Kernel selection still covers both branches below.
                reason = None
                continue
            if self._agent_promotion_comparator is None:
                selection = select_kernel_agent(
                    winner,
                    candidate,
                    measurement_uncertainty_us=self._agent_measurement_uncertainty_us,
                )
                winner = selection.winner
                reason = selection.reason
                continue
            incumbent_kernel = self._branch_best_kernel(
                epoch,
                winner.branch,
                winner.challenger_ordinal,
            )
            candidate_kernel = self._branch_best_kernel(
                epoch,
                candidate.branch,
                candidate.challenger_ordinal,
            )
            if incumbent_kernel.id == candidate_kernel.id:
                reason = AgentSelectionReason.IDENTICAL_KERNEL
                continue
            comparison = await self._agent_promotion_comparator.compare(
                incumbent_kernel,
                candidate_kernel,
            )
            self._registry.record_runtime_event(
                "epoch.agent_comparison_completed",
                epoch.id,
                {
                    "incumbent_kernel_agent_revision_id": winner.kernel_agent_revision_id,
                    "candidate_kernel_agent_revision_id": candidate.kernel_agent_revision_id,
                    "incumbent_kernel_revision_id": incumbent_kernel.id,
                    "candidate_kernel_revision_id": candidate_kernel.id,
                    "candidate_accepted": comparison.accepted,
                    "reason": comparison.reason,
                },
            )
            reason = AgentSelectionReason.AUTHORITATIVE_COMPARISON
            if comparison.accepted:
                winner = candidate
        return winner, reason

    def _branch_best_kernel(
        self,
        epoch: Epoch,
        branch: BranchRole,
        challenger_ordinal: int,
    ) -> KernelRevision:
        retained = [self._registry.get_kernel_revision(epoch.starting_kernel_revision_id)]
        for attempt in self._registry.list_attempts(epoch.id):
            if (
                attempt.branch is not branch
                or attempt.challenger_ordinal != challenger_ordinal
                or not attempt.accepted_as_branch_best
            ):
                continue
            if attempt.output_kernel_revision_id is None:
                raise InvalidTransitionError(f"Attempt {attempt.id} accepted a missing Kernel")
            retained.append(self._registry.get_kernel_revision(attempt.output_kernel_revision_id))
        return select_best_kernel(retained)

    def _retained_kernels(self, epoch: Epoch) -> list[KernelRevision]:
        revisions = [self._registry.get_kernel_revision(epoch.starting_kernel_revision_id)]
        for attempt in self._registry.list_attempts(epoch.id):
            if not attempt.accepted_as_branch_best or attempt.output_kernel_revision_id is None:
                continue
            revision = self._registry.get_kernel_revision(attempt.output_kernel_revision_id)
            if revision.evaluation.correct:
                revisions.append(revision)
        return revisions
