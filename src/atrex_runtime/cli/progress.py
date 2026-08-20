"""Interactive and plain-text Campaign Attempt progress rendering."""

from __future__ import annotations

import sys
from collections.abc import Callable
from typing import TextIO

from ..domain.models import Attempt, BranchRole, Epoch


def _attempt_finished_line(epoch: Epoch, attempt: Attempt) -> str:
    """Render one durable Attempt completion as a stable plain-text event."""
    del epoch
    if attempt.completed_at is None:
        raise ValueError("Attempt progress requires a durable completion timestamp")
    branch_label = (
        "active"
        if attempt.branch is BranchRole.ACTIVE
        else f"challenger-{attempt.challenger_ordinal}"
    )
    return (
        f"[{attempt.completed_at}] {branch_label} trajectory {attempt.trajectory_ordinal} "
        f"attempt {attempt.ordinal} finished"
    )


def print_attempt_finished(epoch: Epoch, attempt: Attempt) -> None:
    """Print one durable Attempt completion without contaminating JSON stdout."""
    print(_attempt_finished_line(epoch, attempt), file=sys.stderr, flush=True)


class AttemptProgressRenderer:
    """Maintain an in-place Branch progress chart on interactive terminals."""

    def __init__(
        self,
        stream: TextIO,
        *,
        interactive: bool | None = None,
        attempt_detail: Callable[[Attempt], str] | None = None,
    ) -> None:
        self._stream = stream
        self._interactive = stream.isatty() if interactive is None else interactive
        self._completed: dict[tuple[str, int, int, int], int] = {}
        self._epochs: dict[tuple[str, int], Epoch] = {}
        self._rendered_lines = 0
        self._attempt_detail = attempt_detail

    def __call__(self, epoch: Epoch, attempt: Attempt) -> None:
        line = _attempt_finished_line(epoch, attempt)
        if self._attempt_detail is not None:
            try:
                line = f"{line}; {self._attempt_detail(attempt)}"
            except (KeyError, OSError, ValueError) as error:
                line = f"{line}; SOL status unavailable ({error})"
        if not self._interactive:
            print(line, file=self._stream, flush=True)
            return
        epoch_key = (str(epoch.lineage_id), epoch.number)
        self._epochs[epoch_key] = epoch
        progress_key = (
            *epoch_key,
            attempt.challenger_ordinal,
            attempt.trajectory_ordinal,
        )
        self._completed[progress_key] = max(
            attempt.ordinal,
            self._completed.get(progress_key, 0),
        )
        self._clear()
        print(line, file=self._stream)
        lines = self._dashboard_lines()
        self._stream.write("".join(f"{item}\n" for item in lines))
        self._stream.flush()
        self._rendered_lines = len(lines)

    def _clear(self) -> None:
        for _ in range(self._rendered_lines):
            self._stream.write("\x1b[1A\r\x1b[2K")
        self._rendered_lines = 0

    def _dashboard_lines(self) -> list[str]:
        lines: list[str] = []
        for epoch_key, epoch in self._epochs.items():
            lines.append(f"Epoch {epoch.number} branch progress ({epoch.lineage_id!s})")
            for challenger_ordinal in range(epoch.challenger_count + 1):
                branch_label = (
                    "active" if challenger_ordinal == 0 else f"challenger-{challenger_ordinal}"
                )
                lines.append(f"  {branch_label}")
                for trajectory_ordinal in range(1, epoch.trajectories_per_branch + 1):
                    completed = self._completed.get(
                        (*epoch_key, challenger_ordinal, trajectory_ordinal),
                        0,
                    )
                    lines.append(
                        f"    trajectory {trajectory_ordinal:<3} "
                        f"{_attempt_progress_bar(completed, epoch.attempts_per_trajectory)} "
                        f"{completed}/{epoch.attempts_per_trajectory}"
                    )
        return lines


def _attempt_progress_bar(completed: int, total: int) -> str:
    """Render one bounded Unicode bar while preserving exact numeric progress."""
    width = min(total, 24)
    filled = 0 if completed == 0 else max(1, round(completed * width / total))
    return f"[{'█' * min(filled, width)}{'░' * max(width - filled, 0)}]"
