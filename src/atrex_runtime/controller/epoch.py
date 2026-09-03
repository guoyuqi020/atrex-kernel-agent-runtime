"""Recoverable configurable Active-versus-Challenger-pool Epoch controller."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime

import anyio

from ..domain.errors import InfrastructureError, InvalidTransitionError
from ..domain.ids import (
    KernelAgentRevisionId,
    LineageId,
    new_attempt_id,
    new_epoch_id,
    new_kernel_agent_revision_id,
    new_kernel_revision_id,
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
    EpochChallenger,
    EpochSelection,
    EpochStatus,
    KernelAgentRevision,
    KernelEvaluation,
    KernelRevision,
)
from ..ports import (
    AttemptEvidenceAssembler,
    BuildAttemptEvidenceRequest,
    BuildChallengerRequest,
    EvolverRunner,
    KernelAgentReuseProposal,
    KernelComparator,
    OptimizerRunner,
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

        if epoch.status is EpochStatus.FAILED:
            raise InvalidTransitionError(f"Epoch {epoch.id} has failed")
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

    async def _ensure_challengers(self, epoch: Epoch) -> None:
        lineage = self._registry.get_lineage(epoch.lineage_id)
        parent = self._registry.get_kernel_agent_revision(epoch.active_kernel_agent_revision_id)
        if epoch.number == 1 and lineage.first_epoch_same_agent:
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
        for challenger_ordinal in range(
            len(epoch.challenger_kernel_agent_revision_ids) + 1,
            epoch.challenger_count + 1,
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
        limiter = anyio.Semaphore(self._max_parallel_branches)
        failures: dict[int, Exception] = {}

        async def run_branch(
            branch: BranchRole,
            challenger_ordinal: int,
            revision_id: KernelAgentRevisionId,
        ) -> None:
            async with limiter:
                try:
                    await self._run_branch(
                        epoch,
                        branch,
                        challenger_ordinal,
                        revision_id,
                    )
                except Exception as error:
                    failures[challenger_ordinal] = error

        async with anyio.create_task_group() as tasks:
            for branch, challenger_ordinal, revision_id in branches:
                tasks.start_soon(
                    run_branch,
                    branch,
                    challenger_ordinal,
                    revision_id,
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
    ) -> None:
        """Run one isolated Branch while allowing sibling Branches to proceed."""
        if epoch.trajectories_per_branch == 1:
            await self._run_trajectory(
                epoch,
                branch,
                challenger_ordinal,
                1,
                revision_id,
            )
            return
        async with anyio.create_task_group() as tasks:
            for trajectory_ordinal in range(1, epoch.trajectories_per_branch + 1):
                tasks.start_soon(
                    self._run_trajectory,
                    epoch,
                    branch,
                    challenger_ordinal,
                    trajectory_ordinal,
                    revision_id,
                )

    @classmethod
    def _contains_infrastructure_error(cls, error: BaseException) -> bool:
        if isinstance(error, InfrastructureError):
            return True
        if isinstance(error, BaseExceptionGroup):
            return any(cls._contains_infrastructure_error(item) for item in error.exceptions)
        return False

    async def _run_trajectory(
        self,
        epoch: Epoch,
        branch: BranchRole,
        challenger_ordinal: int,
        trajectory_ordinal: int,
        agent_revision_id: KernelAgentRevisionId,
    ) -> None:
        input_kernel_id = epoch.starting_kernel_revision_id
        for ordinal in range(1, epoch.attempts_per_trajectory + 1):
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
                await self._complete_registered_attempt(attempt, registered)
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
                registered = self._registry.register_kernel_revision(
                    KernelRevision(
                        id=new_kernel_revision_id(),
                        parent_id=attempt.input_kernel_revision_id,
                        artifact_digest=result.candidate.artifact_digest,
                        produced_by_attempt_id=attempt.id,
                        evaluation=evaluation,
                        created_at=self._clock(),
                    )
                )
                await self._complete_registered_attempt(attempt, registered)
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
        expected_attempts = epoch.trajectories_per_branch * epoch.attempts_per_trajectory
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
                        attempt.trajectory_ordinal - 1
                    ) * epoch.attempts_per_trajectory + attempt.ordinal
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
