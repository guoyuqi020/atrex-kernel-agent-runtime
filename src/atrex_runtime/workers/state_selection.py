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
    branches = {
        ("challenger", ordinal)
        for ordinal, revision_id in enumerate(challenger_revision_ids, start=1)
        if revision_id == winner_revision_id
    }
    if winner_revision_id == active_revision_id:
        branches.add(("active", 0))
    if not branches:
        raise ValueError("Epoch winner is absent from its Agent catalog")

    selected = tuple(
        attempt
        for attempt in attempts
        if (attempt.branch, attempt.challenger_ordinal) in branches
        and attempt.kernel_agent_revision_id == str(winner_revision_id)
    )
    if not selected:
        raise ValueError("Winning Agent has no Attempt Runtime State")

    producing = next(
        (attempt for attempt in selected if attempt.attempt_id == best_kernel_producer_attempt_id),
        None,
    )
    if producing is not None:
        anchor = producing
    else:
        retained = tuple(
            attempt
            for attempt in selected
            if attempt.accepted_as_branch_best and attempt.output_latency_us is not None
        )
        if retained:
            anchor = min(
                retained,
                key=lambda attempt: (
                    attempt.output_latency_us,
                    attempt.branch,
                    attempt.challenger_ordinal,
                    attempt.trajectory_ordinal,
                    attempt.ordinal,
                ),
            )
        else:
            anchor = min(
                selected,
                key=lambda attempt: (
                    attempt.branch,
                    attempt.challenger_ordinal,
                    attempt.trajectory_ordinal,
                ),
            )

    trajectory = tuple(
        attempt
        for attempt in selected
        if (attempt.branch, attempt.challenger_ordinal, attempt.trajectory_ordinal)
        == (
            anchor.branch,
            anchor.challenger_ordinal,
            anchor.trajectory_ordinal,
        )
    )
    terminal = max(
        (attempt for attempt in trajectory if attempt.runtime_state_digest is not None),
        key=lambda attempt: attempt.ordinal,
        default=None,
    )
    if terminal is not None:
        return terminal.runtime_state_digest

    # A completed Epoch normally has a terminal checkpoint. This fallback keeps old
    # Campaigns created before terminal State capture recoverable without inventing State.
    epoch_start = min(
        (attempt for attempt in trajectory if attempt.input_runtime_state_digest is not None),
        key=lambda attempt: attempt.ordinal,
        default=None,
    )
    return None if epoch_start is None else epoch_start.input_runtime_state_digest
