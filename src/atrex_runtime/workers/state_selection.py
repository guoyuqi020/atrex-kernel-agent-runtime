"""Pure selection of the canonical Runtime State shared at an Evolution boundary."""

from __future__ import annotations

from dataclasses import dataclass

from ..domain.ids import ArtifactDigest, KernelAgentRevisionId


@dataclass(frozen=True, slots=True)
class RuntimeStateAttempt:
    """Minimal Attempt facts needed to select one winning Trajectory checkpoint."""

    attempt_id: str
    branch: str
    challenger_ordinal: int
    trajectory_ordinal: int
    ordinal: int
    kernel_agent_revision_id: str
    accepted_as_branch_best: bool
    output_latency_us: float | None
    input_runtime_state_digest: ArtifactDigest | None
    runtime_state_digest: ArtifactDigest | None


def select_winning_trajectory_terminal_state(
    *,
    attempts: tuple[RuntimeStateAttempt, ...],
    winner_revision_id: KernelAgentRevisionId,
    active_revision_id: KernelAgentRevisionId,
    challenger_revision_ids: tuple[KernelAgentRevisionId, ...],
    best_kernel_producer_attempt_id: str | None,
) -> ArtifactDigest | None:
    """Select the winner's best-Kernel Trajectory State at the end of an Epoch.

    The authoritative best Kernel may belong to a different Agent when Kernel retention
    and Agent promotion use different comparators. In that case, use the fastest retained
    Kernel produced by the winning Agent. If that Agent retained no candidate, select its
    lowest-numbered Trajectory deterministically.
    """
    if winner_revision_id == active_revision_id:
        branch = "active"
        challenger_ordinal = 0
    else:
        try:
            challenger_ordinal = challenger_revision_ids.index(winner_revision_id) + 1
        except ValueError as error:
            raise ValueError("Epoch winner is absent from its Agent catalog") from error
        branch = "challenger"

    selected = tuple(
        attempt
        for attempt in attempts
        if attempt.branch == branch
        and attempt.challenger_ordinal == challenger_ordinal
        and attempt.kernel_agent_revision_id == str(winner_revision_id)
    )
    if not selected:
        raise ValueError("Winning Agent has no Attempt Runtime State")

    producing = next(
        (
            attempt
            for attempt in selected
            if attempt.attempt_id == best_kernel_producer_attempt_id
        ),
        None,
    )
    if producing is not None:
        trajectory_ordinal = producing.trajectory_ordinal
    else:
        retained = tuple(
            attempt
            for attempt in selected
            if attempt.accepted_as_branch_best and attempt.output_latency_us is not None
        )
        if retained:
            trajectory_ordinal = min(
                retained,
                key=lambda attempt: (
                    attempt.output_latency_us,
                    attempt.trajectory_ordinal,
                    attempt.ordinal,
                ),
            ).trajectory_ordinal
        else:
            trajectory_ordinal = min(attempt.trajectory_ordinal for attempt in selected)

    trajectory = tuple(
        attempt
        for attempt in selected
        if attempt.trajectory_ordinal == trajectory_ordinal
    )
    terminal = max(
        (
            attempt
            for attempt in trajectory
            if attempt.runtime_state_digest is not None
        ),
        key=lambda attempt: attempt.ordinal,
        default=None,
    )
    if terminal is not None:
        return terminal.runtime_state_digest

    # A completed Epoch normally has a terminal checkpoint. This fallback keeps old
    # Campaigns created before terminal State capture recoverable without inventing State.
    epoch_start = min(
        (
            attempt
            for attempt in trajectory
            if attempt.input_runtime_state_digest is not None
        ),
        key=lambda attempt: attempt.ordinal,
        default=None,
    )
    return None if epoch_start is None else epoch_start.input_runtime_state_digest
