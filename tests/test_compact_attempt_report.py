"""Compact handoff end-to-end: preserve safety without a third analysis record."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from test_attempt_evidence import _complete_attempt, _seed_epoch
from test_attempt_report import _value
from test_journal_measurement_summary import Case
from test_journal_measurement_summary import case as case

from atrex_runtime.artifacts.local import ArtifactKind, LocalArtifactStore
from atrex_runtime.attempt_reports import RuntimeAttemptReportProjector
from atrex_runtime.domain.ids import new_attempt_id
from atrex_runtime.domain.models import BranchRole
from atrex_runtime.registry.sqlite import SqliteRegistry
from atrex_runtime.workers.attempt_report import REPORT_ANALYSIS_FIELDS, AttemptReportV12


def compact(value: dict[str, Any]) -> dict[str, Any]:
    return {
        **{key: item for key, item in value.items() if key not in REPORT_ANALYSIS_FIELDS},
        "schema_version": 13,
    }


@pytest.mark.parametrize("field", sorted(REPORT_ANALYSIS_FIELDS))
def test_v13_rejects_removed_fields_even_when_empty(field: str) -> None:
    value = compact(_value(new_attempt_id()))
    value[field] = None
    with pytest.raises(ValueError, match="cannot contain analysis fields"):
        AttemptReportV12.model_validate(value)


def test_v13_file_roundtrip_does_not_restore_legacy_defaults(tmp_path: Path) -> None:
    value = compact(_value(new_attempt_id()))
    report = AttemptReportV12.model_validate(value)
    path = tmp_path / "report.json"
    path.write_text(report.model_dump_json())
    restored = AttemptReportV12.from_file(
        path, expected_attempt_id=report.attempt_id, max_bytes=100_000
    )
    assert restored == report
    assert not REPORT_ANALYSIS_FIELDS.intersection(json.loads(path.read_text()))
    assert not REPORT_ANALYSIS_FIELDS.intersection(report.model_dump(mode="json"))


@pytest.mark.anyio
@pytest.mark.parametrize(
    "case", [(), ("directions",), ("experiments",), ("directions", "experiments")], indirect=True
)
async def test_compact_http_handoff_keeps_journals_evidence_and_write_once(case: Case) -> None:
    direction = await case.direction() if "directions" in case.modules else None
    before, _ = await case.measure({"s0": 100})
    after, candidate = await case.measure({"s0": 90})
    experiment = None
    if "experiments" in case.modules:
        experiment = await case.record(
            before,
            after,
            **({"direction_id": direction} if direction else {}),
            analysis="This measured variant improved; the causal explanation remains untested.",
        )
    if direction:
        await case.call(
            "direction_update",
            request={
                "action": "defer",
                "direction_id": direction,
                "hypothesis_status": "unresolved",
                **(
                    {
                        "supporting_experiment_ids": [experiment["experiment_id"]]
                        if experiment
                        else []
                    }
                    if "experiments" in case.modules
                    else {}
                ),
            },
        )
    snapshot = (await case.call("journal_snapshot"))["result"]
    report = compact(_value(case.attempt.id))
    report.update(
        tool_modules=list(case.modules),
        experiments=snapshot["experiments"],
        direction_events=snapshot["direction_events"],
        contributing_result_artifact_digests=[],
    )
    accepted = (await case.call("attempt_report", report=report, candidate=candidate))["result"]
    assert accepted["status"] == "registered"
    sealed = accepted["report"]
    assert sealed["schema_version"] == 13
    assert not REPORT_ANALYSIS_FIELDS.intersection(sealed)
    if experiment:
        assert sealed["experiments"][0]["analysis"] == report["experiments"][0]["analysis"]
        assert sealed["experiments"][0]["measurement_summary"]["source"] == "runtime"
    recovered = (await case.call("attempt_report_status"))["result"]
    assert recovered["report"] == sealed
    assert len(case.adapter.requests) == 2  # Handoff/status never submit new GPU work.
    modified = {**report, "final_candidate": {"change_summary": "rewritten after acceptance"}}
    with pytest.raises(AssertionError, match="already accepted"):
        await case.call("attempt_report", report=modified, candidate=candidate)
    assert (await case.call("attempt_report_status"))["result"]["report"] == sealed


@pytest.mark.anyio
@pytest.mark.parametrize("case", [()], indirect=True)
async def test_compact_nomination_without_evaluation_is_still_rejected(case: Case) -> None:
    from test_gateway_proxy import _request

    report = compact(_value(case.attempt.id))
    report.update(tool_modules=[], experiments=[], direction_events=[])
    candidate = json.loads(_request(case.attempt))["candidate"]
    with pytest.raises(AssertionError):
        await case.call("attempt_report", report=report, candidate=candidate)
    assert (await case.call("attempt_report_status"))["result"]["status"] == "missing"


@pytest.mark.anyio
@pytest.mark.parametrize("case", [("directions",)], indirect=True)
async def test_compact_blocked_report_cannot_hide_open_direction(case: Case) -> None:
    from test_gateway_proxy import _request

    await case.direction()
    report = compact(_value(case.attempt.id))
    report.update(
        status="blocked",
        final_candidate=None,
        blocker="Budget ended",
        tool_modules=["directions"],
        experiments=[],
        direction_events=[],
    )
    candidate = json.loads(_request(case.attempt))["candidate"]
    with pytest.raises(AssertionError, match="in progress"):
        await case.call("attempt_report", report=report, candidate=candidate)


def test_compact_history_projects_ids_and_runtime_outcomes_not_journal_prose(
    tmp_path: Path,
) -> None:
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    with SqliteRegistry(tmp_path / "registry.sqlite") as registry:
        epoch, baseline, _ = _seed_epoch(registry, artifacts, tmp_path)
        attempt = _complete_attempt(
            registry,
            artifacts,
            tmp_path,
            epoch,
            baseline,
            BranchRole.ACTIVE,
            "compact",
            "a local experimental interpretation",
        )
        assert attempt.attempt_report_digest is not None
        original = artifacts.verify(attempt.attempt_report_digest).payload_path / "value.json"
        old_bytes = original.read_bytes()
        handoff = compact(json.loads(old_bytes))
        digest = artifacts.put_json(handoff, ArtifactKind.ATTEMPT_REPORT)
        projected = RuntimeAttemptReportProjector(registry, artifacts).project(
            replace(attempt, attempt_report_digest=digest)
        )
        assert not REPORT_ANALYSIS_FIELDS.intersection(projected)
        assert "experiments" not in projected and "direction_events" not in projected
        assert projected["experiment_ids"] == [e["experiment_id"] for e in handoff["experiments"]]
        assert set(projected["direction_ids"]) >= {
            e["direction_id"] for e in handoff["experiments"] if e.get("direction_id")
        }
        assert projected["candidate_kernel"]["gateway_result"]["correct"] is True
        assert projected["candidate_kernel"]["comparison_with_parent"] is not None
        assert original.read_bytes() == old_bytes
