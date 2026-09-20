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
    RuntimeStatePolicy,
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
        return self.scores[0]

    @property
    def challenger_scores(self) -> tuple[BranchScore, ...]:
        """Return Challenger scores in configured ordinal order."""
        return self.scores[1:]


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
        if operation == "run_branches":
            branches = public.get("branches")
            if not isinstance(branches, list):
                raise ValueError("run_branches requires a branches array")
            return await self._controller._workflow_run_branches_legacy(
                self._epoch_id,
                tuple(branches),
                optimizer_attempt_budget=self._optimizer_attempt_budget,
                program_sha256=program_sha256,
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
        max_parallel_branches: int = 4,
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
        if max_parallel_branches <= 0:
            raise ValueError("maximum parallel Branches must be positive")
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
        self._max_parallel_branches = max_parallel_branches
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
            challenger_count = lineage.challengers_for_epoch(epoch_number)
            epoch = Epoch(
                id=new_epoch_id(),
                lineage_id=lineage.id,
                number=epoch_number,
                active_kernel_agent_revision_id=lineage.active_kernel_agent_revision_id,
                challenger_kernel_agent_revision_ids=(),
                starting_kernel_revision_id=lineage.best_kernel_revision_id,
                evidence_checkpoint=lineage.evidence_checkpoint,
                challenger_count=challenger_count,
                trajectories_per_branch=lineage.trajectories_per_branch,
                attempts_per_trajectory=lineage.attempts_per_trajectory,
                status=(
                    EpochStatus.READY if challenger_count == 0 else EpochStatus.BUILDING_CHALLENGER
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
        if self._workflow_runner is not None:
            return await self._run_executable_workflow_epoch(epoch)
        if epoch.status is EpochStatus.BUILDING_CHALLENGER:
            try:
                await self._ensure_challengers(epoch)
            except Exception as error:
                reason = (
                    f"Evolver failed while building Challenger: {type(error).__name__}: {error}"
                )
                self._registry.fail_epoch(epoch.id, reason[:2048])
                raise
            epoch = self._registry.get_epoch(epoch.id)
        if epoch.status is EpochStatus.READY:
            self._registry.transition_epoch(epoch.id, EpochStatus.READY, EpochStatus.RUNNING)
            epoch = self._registry.get_epoch(epoch.id)
        if epoch.status is EpochStatus.RUNNING:
            await self._run_all_attempts(epoch)
            self._registry.transition_epoch(epoch.id, EpochStatus.RUNNING, EpochStatus.SELECTING)
            epoch = self._registry.get_epoch(epoch.id)
        if epoch.status is EpochStatus.SELECTING:
            scores = self._scores(epoch)
            winner, selection_reason = await self._select_kernel_agent(epoch, scores)
            best_kernel = select_best_kernel(self._retained_kernels(epoch))
            self._registry.complete_epoch(
                epoch.id,
                EpochSelection(
                    winner_kernel_agent_revision_id=winner.kernel_agent_revision_id,
                    best_kernel_revision_id=best_kernel.id,
                    selection_reason=selection_reason,
                ),
            )
            epoch = self._registry.get_epoch(epoch.id)
        if epoch.status is not EpochStatus.COMPLETED:
            raise InvalidTransitionError(f"Epoch {epoch.id} stopped in {epoch.status}")
        return EpochRunResult(epoch, self._scores(epoch))

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
        lineage = self._registry.get_lineage(epoch.lineage_id)
        configured_challengers = lineage.challengers_for_epoch(epoch.number)
        # The resource envelope is immutable Lineage input, not a projection of
        # partially registered Trajectories. A Workflow may be interrupted after
        # freezing only its first Branch; deriving the budget from that prefix would
        # shrink the resumed Epoch and make deterministic replay impossible.
        optimizer_attempt_budget = (
            (1 + configured_challengers)
            * epoch.trajectories_per_branch
            * epoch.attempts_per_trajectory
        )
        active = self._registry.get_kernel_agent_revision(epoch.active_kernel_agent_revision_id)
        operations = _EpochWorkflowOperations(
            self,
            epoch,
            optimizer_attempt_budget=optimizer_attempt_budget,
        )
        workflow_runner = self._workflow_runner
        if workflow_runner is None:
            raise AssertionError("executable Workflow runner disappeared")
        await workflow_runner.run(
            RunAgentWorkflowRequest(
                revision=active,
                epoch_id=epoch.id,
                epoch_number=epoch.number,
                max_challengers=(
                    epoch.challenger_count
                    if epoch.status is not EpochStatus.BUILDING_CHALLENGER
                    else configured_challengers
                ),
                optimizer_attempt_budget=optimizer_attempt_budget,
                default_trajectories=epoch.trajectories_per_branch,
                default_attempts_per_trajectory=epoch.attempts_per_trajectory,
                default_runtime_state_policy=(
                    RuntimeStatePolicy.RESET_EACH_ATTEMPT.value
                    if lineage.ephemeral_agent_state
                    else RuntimeStatePolicy.RETAIN_ACROSS_ATTEMPTS.value
                ),
                first_epoch_same_agent=lineage.first_epoch_same_agent,
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
        honor_first_epoch_replica: bool = True,
    ) -> None:
        lineage = self._registry.get_lineage(epoch.lineage_id)
        parent = self._registry.get_kernel_agent_revision(epoch.active_kernel_agent_revision_id)
        if honor_first_epoch_replica and epoch.number == 1 and lineage.first_epoch_same_agent:
            self._registry.attach_challenger(
                EpochChallenger(
                    epoch_id=epoch.id,
                    challenger_ordinal=1,
                    kernel_agent_revision_id=parent.id,
                    base_revision_id=parent.id,
                    proposal_type=ChallengerProposalType.REPLICA,
                    evolution_trace_digest=None,
                )
            )
            return
        target_ordinal = epoch.challenger_count
        if through_ordinal is not None:
            if through_ordinal <= 0 or through_ordinal > epoch.challenger_count:
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
                build = await self._evolver.build_challenger(
                    BuildChallengerRequest(
                        parent_revision=parent,
                        epoch_id=epoch.id,
                        evidence_checkpoint=epoch.evidence_checkpoint,
                        idempotency_key=creation_key,
                        agent_catalog=agent_catalog,
                        kernel_catalog=tuple(self._registry.list_lineage_kernels(epoch.lineage_id)),
                        model=lineage.evolver_model,
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
        if challenger_ordinal > epoch.challenger_count:
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
            honor_first_epoch_replica=False,
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
        policy_value = arguments.get("runtime_state_policy")
        if not isinstance(policy_value, str):
            raise ValueError("runtime_state_policy must be a string")
        try:
            policy = RuntimeStatePolicy(policy_value)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "runtime_state_policy must be reset_each_attempt or retain_across_attempts"
            ) from error

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
            kind=(
                "executable_epoch_workflow_v2" if existing is None else existing.kind
            ),
            program_sha256=program_sha256,
            trajectories=trajectory_count,
            attempts_per_trajectory=attempt_capacity,
            runtime_state_policy=policy,
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
                item.attempt_budget
                for item in self._registry.list_epoch_branch_workflows(epoch.id)
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
            "runtime_state_policy": workflow.runtime_state_policy.value,
            "kernel_agent_revision_id": workflow.kernel_agent_revision_id,
        }

    async def _workflow_run_branches_legacy(
        self,
        epoch_id: EpochId,
        raw_branches: tuple[object, ...],
        *,
        optimizer_attempt_budget: int,
        program_sha256: str,
    ) -> Mapping[str, object]:
        """Resume pre-v2 Agent revisions without exposing this coarse API to new Bundles."""
        if not raw_branches:
            raise ValueError("Workflow must run at least the Active Branch")
        plans: list[tuple[str, int, int]] = []
        for raw in raw_branches:
            if not isinstance(raw, dict):
                raise ValueError("each Workflow Branch must be an object")
            label = raw.get("branch")
            if not isinstance(label, str):
                raise ValueError("Workflow Branch requires a branch label")
            trajectories = self._positive_workflow_int(raw, "trajectories")
            attempts = self._positive_workflow_int(raw, "attempts_per_trajectory")
            policy = raw.get("runtime_state_policy")
            if not isinstance(policy, str):
                raise ValueError("runtime_state_policy must be a string")
            for trajectory_ordinal in range(1, trajectories + 1):
                self._workflow_create_trajectory(
                    epoch_id,
                    {
                        "branch": label,
                        "trajectory_ordinal": trajectory_ordinal,
                        "trajectory_count": trajectories,
                        "attempt_capacity": attempts,
                        "runtime_state_policy": policy,
                    },
                    optimizer_attempt_budget=optimizer_attempt_budget,
                    program_sha256=program_sha256,
                )
            plans.append((label, trajectories, attempts))

        for attempt_ordinal in range(1, max(attempts for _, _, attempts in plans) + 1):
            launches = [
                {
                    "branch": label,
                    "trajectory_ordinal": trajectory_ordinal,
                    "attempt_ordinal": attempt_ordinal,
                }
                for label, trajectories, attempts in plans
                if attempt_ordinal <= attempts
                for trajectory_ordinal in range(1, trajectories + 1)
            ]
            await self._workflow_run_attempts_parallel(
                epoch_id,
                tuple(launches),
                optimizer_attempt_budget=optimizer_attempt_budget,
                program_sha256=program_sha256,
            )
        scores = self._scores(self._registry.get_epoch(epoch_id))
        return {
            "branches": [
                {
                    "branch": (
                        "active"
                        if score.branch is BranchRole.ACTIVE
                        else f"challenger-{score.challenger_ordinal}"
                    ),
                    "kernel_agent_revision_id": score.kernel_agent_revision_id,
                    "best_latency_us": score.best_latency_us,
                    "valid_candidates": score.valid_candidates,
                    "failed_candidates": score.failed_candidates,
                }
                for score in scores
            ]
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

        limiter = anyio.Semaphore(self._max_parallel_branches)
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
            if previous.accepted_as_branch_best:
                if previous.output_kernel_revision_id is None:
                    raise InvalidTransitionError("accepted Attempt has no output Kernel")
                input_kernel_id = previous.output_kernel_revision_id
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
            if workflow.runtime_state_policy is not RuntimeStatePolicy.RETAIN_ACROSS_ATTEMPTS:
                raise ValueError(
                    "input_state_from_attempt_id requires retain_across_attempts"
                )
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
        required = {
            (BranchRole.ACTIVE, 0),
            *(
                (BranchRole.CHALLENGER, ordinal)
                for ordinal in range(1, len(self._registry.list_epoch_challengers(epoch.id)) + 1)
            ),
        }
        if identities != required:
            raise ValueError(
                "Workflow must register Active and every attached Challenger before Attempts"
            )
        if any(item.program_sha256 != program_sha256 for item in workflows):
            raise InvalidTransitionError("Workflow Branch was frozen by another program")
        planned = sum(item.attempt_budget for item in workflows)
        if planned != optimizer_attempt_budget:
            raise ValueError(
                "Workflow must allocate the Runtime Optimizer Attempt budget exactly: "
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
        attempts = self._registry.list_attempts(epoch.id)
        if len(attempts) != optimizer_attempt_budget:
            raise InvalidTransitionError(
                "Workflow cannot select before every allocated Attempt has run: "
                f"completed or present {len(attempts)}, required {optimizer_attempt_budget}"
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

    async def _run_all_attempts(self, epoch: Epoch) -> None:
        if len(epoch.challenger_kernel_agent_revision_ids) != epoch.challenger_count:
            raise InvalidTransitionError(f"Epoch {epoch.id} has an incomplete Challenger pool")
        branches = (
            (BranchRole.ACTIVE, 0, epoch.active_kernel_agent_revision_id),
            *(
                (BranchRole.CHALLENGER, ordinal, revision_id)
                for ordinal, revision_id in enumerate(
                    epoch.challenger_kernel_agent_revision_ids,
                    start=1,
                )
            ),
        )
        planned = tuple(
            (
                branch,
                challenger_ordinal,
                revision_id,
                self._ensure_branch_workflow(
                    epoch,
                    branch,
                    challenger_ordinal,
                    revision_id,
                ),
            )
            for branch, challenger_ordinal, revision_id in branches
        )
        limiter = anyio.Semaphore(self._max_parallel_branches)
        failures: dict[int, Exception] = {}

        async def run_branch(
            branch: BranchRole,
            challenger_ordinal: int,
            revision_id: KernelAgentRevisionId,
            workflow: EpochBranchWorkflow,
        ) -> None:
            async with limiter:
                try:
                    await self._run_branch(
                        epoch,
                        branch,
                        challenger_ordinal,
                        revision_id,
                        workflow,
                    )
                except Exception as error:
                    failures[challenger_ordinal] = error

        async with anyio.create_task_group() as tasks:
            for branch, challenger_ordinal, revision_id, workflow in planned:
                tasks.start_soon(
                    run_branch,
                    branch,
                    challenger_ordinal,
                    revision_id,
                    workflow,
                )
        if failures:
            first_ordinal = min(failures)
            error = failures[first_ordinal]
            infrastructure = next(
                (
                    (ordinal, failure)
                    for ordinal, failure in sorted(failures.items())
                    if self._contains_infrastructure_error(failure)
                ),
                None,
            )
            if infrastructure is not None:
                failed_ordinal, failure = infrastructure
                reason = f"Branch {failed_ordinal} failed: {failure}"
                self._registry.fail_epoch(epoch.id, reason[:2048])
                error = failure
            raise error

    async def _run_branch(
        self,
        epoch: Epoch,
        branch: BranchRole,
        challenger_ordinal: int,
        revision_id: KernelAgentRevisionId,
        workflow: EpochBranchWorkflow,
    ) -> None:
        """Run one isolated Branch while allowing sibling Branches to proceed."""
        if workflow.trajectories == 1:
            await self._run_trajectory(
                epoch,
                branch,
                challenger_ordinal,
                1,
                revision_id,
                workflow.attempts_per_trajectory,
            )
            return
        async with anyio.create_task_group() as tasks:
            for trajectory_ordinal in range(1, workflow.trajectories + 1):
                tasks.start_soon(
                    self._run_trajectory,
                    epoch,
                    branch,
                    challenger_ordinal,
                    trajectory_ordinal,
                    revision_id,
                    workflow.attempts_per_trajectory,
                )

    def _ensure_branch_workflow(
        self,
        epoch: Epoch,
        branch: BranchRole,
        challenger_ordinal: int,
        revision_id: KernelAgentRevisionId,
    ) -> EpochBranchWorkflow:
        """Resolve once and durably freeze the Workflow carried by an Agent Revision."""
        existing = self._registry.get_epoch_branch_workflow(
            epoch.id,
            branch,
            challenger_ordinal,
        )
        if existing is not None:
            if existing.kernel_agent_revision_id != revision_id:
                raise InvalidTransitionError("Recovered Branch Workflow names a different Agent")
            return existing
        return self._registry.ensure_epoch_branch_workflow(
            EpochBranchWorkflow(
                epoch_id=epoch.id,
                branch=branch,
                challenger_ordinal=challenger_ordinal,
                kernel_agent_revision_id=revision_id,
                kind="campaign_topology",
                program_sha256=None,
                trajectories=epoch.trajectories_per_branch,
                attempts_per_trajectory=epoch.attempts_per_trajectory,
                runtime_state_policy=(
                    RuntimeStatePolicy.RESET_EACH_ATTEMPT
                    if self._registry.get_lineage(epoch.lineage_id).ephemeral_agent_state
                    else RuntimeStatePolicy.RETAIN_ACROSS_ATTEMPTS
                ),
                created_at=self._clock(),
            )
        )

    @classmethod
    def _contains_infrastructure_error(cls, error: BaseException) -> bool:
        return cls._first_infrastructure_error(error) is not None

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

    async def _run_trajectory(
        self,
        epoch: Epoch,
        branch: BranchRole,
        challenger_ordinal: int,
        trajectory_ordinal: int,
        agent_revision_id: KernelAgentRevisionId,
        attempts_per_trajectory: int,
    ) -> None:
        input_kernel_id = epoch.starting_kernel_revision_id
        for ordinal in range(1, attempts_per_trajectory + 1):
            attempt = self._registry.find_attempt(
                epoch.id,
                branch,
                challenger_ordinal,
                trajectory_ordinal,
                ordinal,
            )
            if attempt is None:
                attempt_id = new_attempt_id()
                attempt_evidence_digest = self._attempt_evidence.assemble(
                    BuildAttemptEvidenceRequest(
                        attempt_id=attempt_id,
                        epoch_id=epoch.id,
                        branch=branch,
                        challenger_ordinal=challenger_ordinal,
                        trajectory_ordinal=trajectory_ordinal,
                        ordinal=ordinal,
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
                    ordinal=ordinal,
                    kernel_agent_revision_id=agent_revision_id,
                    input_kernel_revision_id=input_kernel_id,
                    attempt_evidence_digest=attempt_evidence_digest,
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
            elif attempt.input_kernel_revision_id != input_kernel_id:
                raise InvalidTransitionError(
                    f"Attempt {attempt.id} input disagrees with recovered branch state"
                )

            self._attempt_evidence.validate(
                attempt.attempt_evidence_digest,
                BuildAttemptEvidenceRequest(
                    attempt_id=attempt.id,
                    epoch_id=epoch.id,
                    branch=branch,
                    challenger_ordinal=challenger_ordinal,
                    trajectory_ordinal=trajectory_ordinal,
                    ordinal=ordinal,
                    epoch_evidence_checkpoint=epoch.evidence_checkpoint,
                ),
            )

            was_completed = attempt.status is AttemptStatus.COMPLETED
            attempt = await self._finish_attempt(epoch, attempt)
            if not was_completed and self._attempt_finished is not None:
                with suppress(Exception):
                    self._attempt_finished(epoch, attempt)
            if attempt.accepted_as_branch_best:
                if attempt.output_kernel_revision_id is None:
                    raise InvalidTransitionError(f"Attempt {attempt.id} accepted a missing Kernel")
                input_kernel_id = attempt.output_kernel_revision_id

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
        expected_attempts = (
            epoch.trajectories_per_branch * epoch.attempts_per_trajectory
            if workflow is None
            else workflow.attempt_budget
        )
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
                    first_best = (attempt.trajectory_ordinal - 1) * (
                        epoch.attempts_per_trajectory
                        if workflow is None
                        else workflow.attempts_per_trajectory
                    ) + attempt.ordinal
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
