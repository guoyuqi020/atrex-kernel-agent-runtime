"""Runtime State handoff when two isolated branches share the same Agent."""

from dataclasses import replace

import pytest
from conftest import digest

from atrex_runtime.domain.ids import parse_kernel_agent_revision_id
from atrex_runtime.workers.state_selection import (
    RuntimeStateAttempt,
    select_winning_trajectory_terminal_state,
)


@pytest.mark.parametrize("producer", ["challenger-1", None])
def test_same_agent_uses_best_branch_terminal_state_not_same_numbered_active_trajectory(
    producer: str | None,
) -> None:
    agent = parse_kernel_agent_revision_id("agentrev_" + "a" * 32)
    active = RuntimeStateAttempt(
        attempt_id="active-1",
        branch="active",
        challenger_ordinal=0,
        trajectory_ordinal=1,
        ordinal=1,
        kernel_agent_revision_id=agent,
        accepted_as_branch_best=True,
        output_latency_us=90,
        input_runtime_state_digest=digest("start"),
        runtime_state_digest=digest("active-first"),
    )
    challenger = replace(
        active,
        attempt_id="challenger-1",
        branch="challenger",
        challenger_ordinal=1,
        output_latency_us=80,
        runtime_state_digest=digest("challenger-first"),
    )
    attempts = (
        active,
        replace(
            active,
            attempt_id="active-2",
            ordinal=2,
            accepted_as_branch_best=False,
            runtime_state_digest=digest("active-terminal"),
        ),
        challenger,
        replace(
            challenger,
            attempt_id="challenger-2",
            ordinal=2,
            accepted_as_branch_best=False,
            runtime_state_digest=digest("challenger-terminal"),
        ),
    )
    result = select_winning_trajectory_terminal_state(
        attempts=attempts,
        winner_revision_id=agent,
        active_revision_id=agent,
        challenger_revision_ids=(agent,),
        best_kernel_producer_attempt_id=producer,
    )
    assert result == digest("challenger-terminal")
