"""Terminal reports may describe honest blocked/pivot outcomes without experiments."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError
from test_attempt_report import _value

from atrex_runtime.domain.ids import new_attempt_id
from atrex_runtime.workers.attempt_report import (
    AttemptDirectionEventV1,
    AttemptExperimentV8,
    AttemptReportV12,
)


def _report(status: str) -> dict[str, Any]:
    value = _value(new_attempt_id())
    value.update(
        status=status,
        final_candidate=None,
        blocker="No safe experiment was possible" if status == "blocked" else None,
        profile_evidence=None,
        experiments=[],
        findings=[],
        direction_events=[],
    )
    return value


@pytest.mark.parametrize("status", ["blocked", "pivot"])
def test_terminal_report_accepts_empty_journals_and_findings(status: str) -> None:
    report = AttemptReportV12.model_validate(_report(status))

    assert report.status == status
    assert report.experiments == ()
    assert report.findings == ()
    assert report.direction_events == ()


@pytest.mark.parametrize("field", ["experiments", "findings", "direction_events"])
def test_candidate_ready_still_requires_each_journal_collection(field: str) -> None:
    value = _value(new_attempt_id())
    value[field] = []

    with pytest.raises(ValidationError, match="candidate_ready requires non-empty"):
        AttemptReportV12.model_validate(value)


@pytest.mark.parametrize("status", ["blocked", "pivot"])
@pytest.mark.parametrize("action", ["propose", "complete", "abandon", "block", "defer"])
def test_empty_experiment_report_only_accepts_proposals(status: str, action: str) -> None:
    value = _report(status)
    original = _value(value["attempt_id"])
    events = original["direction_events"]
    assert isinstance(events, list)
    value["direction_events"] = events[:1]
    if action != "propose":
        closing = dict(events[1])
        closing["action"] = action
        closing["analysis"] = "No experiment was performed; retain this direction for later."
        value["direction_events"].append(closing)
        with pytest.raises(ValidationError, match=f"Direction {action} requires supporting"):
            AttemptReportV12.model_validate(value)
        return

    assert AttemptReportV12.model_validate(value).experiments == ()


@pytest.mark.parametrize("action", ["complete", "abandon", "block", "defer"])
def test_history_context_only_preserves_old_block_and_defer(action: str) -> None:
    event = dict(_value(new_attempt_id())["direction_events"][1], action=action)
    assert event["supporting_experiment_ids"] == []
    with pytest.raises(ValidationError, match="requires supporting Experiments"):
        AttemptDirectionEventV1.model_validate(event)
    if action in {"block", "defer"}:
        assert AttemptDirectionEventV1.model_validate(
            event, context={"trusted_direction_history": True}
        ).supporting_experiment_ids == ()
    else:
        with pytest.raises(ValidationError, match="requires supporting Experiments"):
            AttemptDirectionEventV1.model_validate(
                event, context={"trusted_direction_history": True}
            )
    # The Agent cannot enable the internal read context through request fields.
    with pytest.raises(ValidationError):
        AttemptDirectionEventV1.model_validate({**event, "trusted_direction_history": True})


@pytest.mark.parametrize("status", ["blocked", "pivot"])
def test_empty_report_cannot_leave_direction_in_progress(status: str) -> None:
    value = _report(status)
    original = _value(value["attempt_id"])
    events = original["direction_events"]
    assert isinstance(events, list)
    value["direction_events"] = events[:2]

    with pytest.raises(ValidationError, match="cannot leave any Direction in progress"):
        AttemptReportV12.model_validate(value)


@pytest.mark.parametrize("status", ["blocked", "pivot"])
def test_empty_report_cannot_fabricate_finding_support(status: str) -> None:
    value = _report(status)
    value["findings"] = _value(value["attempt_id"])["findings"]

    with pytest.raises(ValidationError, match="Finding references an Experiment outside"):
        AttemptReportV12.model_validate(value)


@pytest.mark.parametrize("status", ["blocked", "pivot"])
def test_nonempty_report_still_requires_experiment_direction(status: str) -> None:
    value = _report(status)
    value["experiments"] = _value(value["attempt_id"])["experiments"]

    with pytest.raises(ValidationError, match="Direction absent from this Attempt"):
        AttemptReportV12.model_validate(value)


@pytest.mark.parametrize("missing", ["before", "after", "both"])
def test_adoption_requires_complete_before_after_evidence(missing: str) -> None:
    experiments = _value(new_attempt_id())["experiments"]
    assert isinstance(experiments, list)
    value = dict(experiments[0], action="adopt")
    for side in ("before", "after"):
        if missing in {side, "both"}:
            value[side] = None

    with pytest.raises(ValidationError, match="before and after"):
        AttemptExperimentV8.model_validate(value)


def test_adoption_accepts_complete_evidence_without_changing_trial_identity() -> None:
    experiments = _value(new_attempt_id())["experiments"]
    assert isinstance(experiments, list)
    value = dict(experiments[0], action="adopt")

    experiment = AttemptExperimentV8.model_validate(value)

    assert experiment.action == "adopt"
    assert experiment.model_dump(mode="json")["after"] == value["after"]
