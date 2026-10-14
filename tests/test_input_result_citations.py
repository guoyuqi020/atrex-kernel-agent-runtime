"""Sealed input results can be cited wherever they can be read, without new evaluations."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
from test_attempt_finding_evidence import _finding
from test_attempt_report import _value
from test_bootstrap_result_visibility import _seed_input
from test_journal_measurement_summary import Case
from test_journal_measurement_summary import case as case

from atrex_runtime.artifacts.local import LocalArtifactStore
from atrex_runtime.domain.ids import new_kernel_revision_id

MODULES = [(), ("experiments",), ("directions",), ("directions", "experiments")]


@pytest.mark.anyio
@pytest.mark.parametrize("case", MODULES, indirect=True)
@pytest.mark.parametrize("holdout", [False, True])
async def test_input_read_experiment_direction_and_report_use_same_evidence(
    case: Case,
    tmp_path: Path,
    holdout: bool,
) -> None:
    registry = case.control._registry
    baseline = _seed_input(
        registry,
        LocalArtifactStore(tmp_path / "artifacts"),
        case.attempt,
        tmp_path,
        holdout=holdout,
    )
    digest = baseline.evaluation.gateway_result_digest
    subject = {
        "kernel_artifact_digest": baseline.artifact_digest,
        "result_artifact_digests": [digest],
    }
    direction = await case.direction() if "directions" in case.modules else None
    measured, candidate = await case.measure({"0": 80.0, "1": 300.0})
    receipt = None
    if "experiments" in case.modules:
        receipt = await case.record(
            {"result_artifact_digest": digest},
            measured,
            **({"direction_id": direction} if direction else {}),
        )
        before = receipt["measurement_summary"]["before"][0]
        assert before["kernel_artifact_digest"] == baseline.artifact_digest
        assert before["result_artifact_digest"] == digest
        assert before["status"] == "completed"
        assert before["correct"] is True
        assert before["measured_shape_count"] == (0 if holdout else 2)
        assert before["latency_us_geomean"] == (None if holdout else pytest.approx(200))
        loaded = (await case.call("experiment_load", experiment_id=receipt["experiment_id"]))[
            "result"
        ]
        assert loaded["measurement_summary"] == receipt["measurement_summary"]
        assert "private-job-id" not in json.dumps(loaded)
        assert "samples" not in json.dumps(loaded)
        # Evidence binding never becomes a current measurement or mutates its source.
        assert len(case.control.list_evaluations(case.attempt.id)) == 1
        assert len(case.control.list_kernel_trials((case.attempt.id,))) == 1
    if direction:
        closed = (
            await case.call(
                "direction_update",
                request={
                    "action": "complete",
                    "direction_id": direction,
                    "hypothesis_status": "supported",
                    "claim_kind": "implementation_outcome",
                    "scope": "The sealed input on the published validation workload",
                    "supporting_results": [subject],
                    **({"supporting_experiment_ids": []} if "experiments" in case.modules else {}),
                },
            )
        )["result"]
        assert closed["hypothesis_status"] == ("unresolved" if holdout else "supported")
    snapshot = (await case.call("journal_snapshot"))["result"]
    report = _value(case.attempt.id)
    report.update(
        experiments=snapshot["experiments"],
        direction_events=snapshot["direction_events"],
        tool_modules=list(case.modules),
        profile_evidence=None,
        findings=[
            _finding(
                claim="The sealed input passed and has the cited latency",
                assessment="supported",
                claim_kind="implementation_outcome",
                scope="Published validation workload",
                supporting_results=[subject],
            )
        ],
        contributing_result_artifact_digests=[digest, measured["result_artifact_digest"]],
    )
    registered = (await case.call("attempt_report", candidate=candidate, report=report))["result"]
    assert registered["status"] == "registered"
    recovered = (await case.call("attempt_report_status"))["result"]["report"]
    assert recovered["findings"][0]["assessment"] == ("unresolved" if holdout else "supported")
    assert len(case.adapter.requests) == 1  # No GPU re-evaluation to obtain citation access.


@pytest.mark.anyio
async def test_input_citation_cannot_claim_fresh_measurement_or_foreign_evidence(
    case: Case,
    tmp_path: Path,
) -> None:
    registry = case.control._registry
    baseline = _seed_input(
        registry, LocalArtifactStore(tmp_path / "artifacts"), case.attempt, tmp_path
    )
    before = {"result_artifact_digest": baseline.evaluation.gateway_result_digest}
    measured, _ = await case.measure({"0": 80.0, "1": 300.0})
    # A genuine full measurement is still required for ordinary after evidence.
    with pytest.raises(AssertionError, match="for this logical Attempt"):
        await case.record(measured, before)
    # Merely having an immutable revision in the Registry does not grant visibility.
    foreign = replace(
        baseline,
        id=new_kernel_revision_id(),
        artifact_digest=measured["kernel_artifact_digest"],
        evaluation=replace(baseline.evaluation, gateway_result_digest="sha256:" + "d" * 64),
    )
    registry.register_kernel_revision(foreign)
    with pytest.raises(AssertionError, match="outside the permitted visible history"):
        await case.record(
            {"result_artifact_digest": foreign.evaluation.gateway_result_digest}, measured
        )
    assert not case.control.list_experiments(case.attempt.id)
    assert len(case.control.list_evaluations(case.attempt.id)) == 1
    # The actual sealed input remains usable after both rejections.
    assert (await case.record(before, measured))["status"] == "recorded"
