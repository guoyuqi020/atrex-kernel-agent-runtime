"""Attempt report file protocol validation tests."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from atrex_runtime.domain.ids import new_attempt_id
from atrex_runtime.workers.attempt_report import AttemptReportV12


def _value(attempt_id: str) -> dict[str, object]:
    return {
        "schema_version": 12,
        "attempt_id": attempt_id,
        "status": "candidate_ready",
        "hypothesis": "coalesced loads reduce memory transactions",
        "diagnosis": {
            "bottleneck": "memory bandwidth",
            "evidence": "SOL localized memory traffic",
        },
        "approach": {
            "summary": "vectorize aligned loads",
            "steps": ["replace scalar loads"],
            "expected_impact": "reduce memory transactions",
            "risks": ["alignment must be preserved"],
        },
        "final_candidate": {"change_summary": "replaced scalar loads with vector loads"},
        "evidence_summary": {
            "correctness": "Gateway reported correct",
            "performance": "Gateway reported faster",
        },
        "profile_evidence": {
            "tool_used": "gateway-execute/profile",
            "profiler": "ncu",
            "profile_level": "sol",
            "bottleneck_type": "memory_bound",
            "evidence_summary": "SOL localized memory traffic",
            "evidence_chain": "SOL metrics identified memory traffic as the limiting term",
            "supporting_results": [
                {
                    "operation": "profile",
                    "kernel_artifact_digest": "sha256:" + "d" * 64,
                    "kernel_trial_id": "gtrial_" + "e" * 32,
                    "result_artifact_digest": "sha256:" + "f" * 64,
                }
            ],
        },
        "analysis": "the targeted change improved latency",
        "knowledge_used": [
            {
                "record_id": "wiki://coalescing",
                "finding": "aligned vector loads coalesce",
                "application": "selected a four-element vector load",
            }
        ],
        "findings": [
            {
                "category": "performance",
                "observation": "latency improved",
                "root_cause": "fewer memory transactions",
                "resolution": "kept the aligned vector-load candidate",
                "lesson": "preserve vector alignment",
                "supporting_experiment_ids": ["experiment_" + "d" * 32],
            }
        ],
        "blocker": None,
        "experiments": [
            {
                "experiment_id": "experiment_" + "d" * 32,
                "direction_id": "direction_" + "a" * 32,
                "sequence": 1,
                "recorded_at": "2026-08-16T00:00:00+00:00",
                "name": "aligned loads",
                "hypothesis": "coalescing helps",
                "change": "vectorized loads",
                "before": {
                    "kernel_artifact_digest": "sha256:" + "a" * 64,
                    "kernel_trial_id": "gtrial_" + "b" * 32,
                    "result_artifact_digests": ["sha256:" + "c" * 64],
                },
                "after": {
                    "kernel_artifact_digest": "sha256:" + "d" * 64,
                    "kernel_trial_id": "gtrial_" + "e" * 32,
                    "result_artifact_digests": [
                        "sha256:" + "f" * 64,
                        "sha256:" + "1" * 64,
                    ],
                },
                "evidence": "SOL memory traffic",
                "analysis": "the hypothesis held and the candidate is faster",
                "action": "keep_after",
            }
        ],
        "direction_events": [
            {
                "direction_event_id": "directionevent_" + "1" * 32,
                "direction_id": "direction_" + "a" * 32,
                "recorded_at": "2026-08-16T00:00:00+00:00",
                "action": "propose",
                "name": "aligned vector loads",
                "hypothesis": "coalescing helps",
                "rationale": "profile localized memory traffic",
                "plan": ["replace scalar loads"],
                "success_criteria": "latency improves by two percent",
                "stop_conditions": "alignment cannot be preserved",
                "analysis": None,
                "supporting_experiment_ids": [],
            },
            {
                "direction_event_id": "directionevent_" + "2" * 32,
                "direction_id": "direction_" + "a" * 32,
                "recorded_at": "2026-08-16T00:01:00+00:00",
                "action": "start",
                "name": None,
                "hypothesis": None,
                "rationale": None,
                "plan": [],
                "success_criteria": None,
                "stop_conditions": None,
                "analysis": "starting the planned experiment",
                "supporting_experiment_ids": [],
            },
            {
                "direction_event_id": "directionevent_" + "3" * 32,
                "direction_id": "direction_" + "a" * 32,
                "recorded_at": "2026-08-16T00:02:00+00:00",
                "action": "complete",
                "name": None,
                "hypothesis": None,
                "rationale": None,
                "plan": [],
                "success_criteria": None,
                "stop_conditions": None,
                "analysis": "the success criterion was met",
                "supporting_experiment_ids": ["experiment_" + "d" * 32],
            },
        ],
    }


def test_attempt_report_is_bound_to_expected_attempt(tmp_path: Path) -> None:
    attempt_id = new_attempt_id()
    path = tmp_path / "report.json"
    path.write_text(json.dumps(_value(attempt_id)), encoding="utf-8")

    report = AttemptReportV12.from_file(path, expected_attempt_id=attempt_id, max_bytes=8192)

    assert report.status == "candidate_ready"
    assert report.experiments[0].experiment_id == "experiment_" + "d" * 32


def test_attempt_report_models_one_sided_baseline_experiment(tmp_path: Path) -> None:
    attempt_id = new_attempt_id()
    value = _value(attempt_id)
    experiments = value["experiments"]
    assert isinstance(experiments, list)
    experiments[0]["action"] = "baseline"
    experiments[0]["before"] = None
    path = tmp_path / "baseline-report.json"
    path.write_text(json.dumps(value), encoding="utf-8")

    report = AttemptReportV12.from_file(path, expected_attempt_id=attempt_id, max_bytes=8192)

    assert report.experiments[0].action == "baseline"
    assert report.experiments[0].before is None
    assert report.experiments[0].after is not None


def test_attempt_report_rejects_baseline_with_before_evidence(tmp_path: Path) -> None:
    attempt_id = new_attempt_id()
    value = _value(attempt_id)
    experiments = value["experiments"]
    assert isinstance(experiments, list)
    experiments[0]["action"] = "baseline"
    path = tmp_path / "invalid-baseline-report.json"
    path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError, match="baseline requires before=null"):
        AttemptReportV12.from_file(path, expected_attempt_id=attempt_id, max_bytes=8192)


def test_attempt_report_rejects_legacy_top_level_decision(tmp_path: Path) -> None:
    attempt_id = new_attempt_id()
    value = _value(attempt_id)
    value["decision"] = "keep"
    path = tmp_path / "report.json"
    path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        AttemptReportV12.from_file(path, expected_attempt_id=attempt_id, max_bytes=8192)


def test_attempt_report_allows_more_than_three_continuable_directions(tmp_path: Path) -> None:
    attempt_id = new_attempt_id()
    value = _value(attempt_id)
    events = value["direction_events"]
    assert isinstance(events, list)
    for ordinal in range(4):
        marker = f"{ordinal + 10:032x}"
        events.append(
            {
                "direction_event_id": f"directionevent_{marker}",
                "direction_id": f"direction_{marker}",
                "recorded_at": "2026-08-16T00:03:00+00:00",
                "action": "propose",
                "name": f"follow-up {ordinal}",
                "hypothesis": "a follow-up may improve latency",
                "rationale": "the completed experiment exposed another opportunity",
                "plan": ["test the follow-up"],
                "success_criteria": "latency improves",
                "stop_conditions": "the mechanism is falsified",
                "analysis": None,
                "supporting_experiment_ids": [],
            }
        )
    path = tmp_path / "report.json"
    path.write_text(json.dumps(value), encoding="utf-8")

    report = AttemptReportV12.from_file(
        path,
        expected_attempt_id=attempt_id,
        max_bytes=16384,
    )

    assert sum(event.action == "propose" for event in report.direction_events) == 5


def test_attempt_report_rejects_more_than_three_advanced_directions(tmp_path: Path) -> None:
    attempt_id = new_attempt_id()
    value = _value(attempt_id)
    events = value["direction_events"]
    assert isinstance(events, list)
    for ordinal in range(3):
        direction_marker = f"{ordinal + 10:032x}"
        event_base = 20 + ordinal * 3
        events.extend(
            [
                {
                    "direction_event_id": f"directionevent_{event_base:032x}",
                    "direction_id": f"direction_{direction_marker}",
                    "recorded_at": "2026-08-16T00:03:00+00:00",
                    "action": "propose",
                    "name": f"additional direction {ordinal}",
                    "hypothesis": "an independent mechanism may improve latency",
                    "rationale": "the mechanism has distinct supporting evidence",
                    "plan": ["test the independent mechanism"],
                    "success_criteria": "latency improves",
                    "stop_conditions": "the mechanism is falsified",
                    "analysis": None,
                    "supporting_experiment_ids": [],
                },
                {
                    "direction_event_id": f"directionevent_{event_base + 1:032x}",
                    "direction_id": f"direction_{direction_marker}",
                    "recorded_at": "2026-08-16T00:04:00+00:00",
                    "action": "start",
                    "name": None,
                    "hypothesis": None,
                    "rationale": None,
                    "plan": [],
                    "success_criteria": None,
                    "stop_conditions": None,
                    "analysis": "start the independent direction",
                    "supporting_experiment_ids": [],
                },
                {
                    "direction_event_id": f"directionevent_{event_base + 2:032x}",
                    "direction_id": f"direction_{direction_marker}",
                    "recorded_at": "2026-08-16T00:05:00+00:00",
                    "action": "defer",
                    "name": None,
                    "hypothesis": None,
                    "rationale": None,
                    "plan": [],
                    "success_criteria": None,
                    "stop_conditions": None,
                    "analysis": "defer further work",
                    "supporting_experiment_ids": [],
                },
            ]
        )
    path = tmp_path / "report.json"
    path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="Direction advancement limit exceeded: maximum=3; started=4",
    ):
        AttemptReportV12.from_file(path, expected_attempt_id=attempt_id, max_bytes=32768)


def test_attempt_report_rejects_unexperimented_direction_left_in_progress(
    tmp_path: Path,
) -> None:
    attempt_id = new_attempt_id()
    value = _value(attempt_id)
    events = value["direction_events"]
    assert isinstance(events, list)
    events.extend(
        [
            {
                "direction_event_id": "directionevent_" + "4" * 32,
                "direction_id": "direction_" + "b" * 32,
                "recorded_at": "2026-08-16T00:03:00+00:00",
                "action": "propose",
                "name": "unmeasured follow-up",
                "hypothesis": "another mechanism may improve latency",
                "rationale": "the mechanism is distinct",
                "plan": ["investigate the mechanism"],
                "success_criteria": "latency improves",
                "stop_conditions": "the mechanism is falsified",
                "analysis": None,
                "supporting_experiment_ids": [],
            },
            {
                "direction_event_id": "directionevent_" + "5" * 32,
                "direction_id": "direction_" + "b" * 32,
                "recorded_at": "2026-08-16T00:04:00+00:00",
                "action": "start",
                "name": None,
                "hypothesis": None,
                "rationale": None,
                "plan": [],
                "success_criteria": None,
                "stop_conditions": None,
                "analysis": "start the follow-up",
                "supporting_experiment_ids": [],
            },
        ]
    )
    path = tmp_path / "report.json"
    path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError, match="cannot leave any Direction in progress"):
        AttemptReportV12.from_file(path, expected_attempt_id=attempt_id, max_bytes=16384)


def test_attempt_report_requires_finding_resolution(tmp_path: Path) -> None:
    attempt_id = new_attempt_id()
    value = _value(attempt_id)
    value["findings"][0].pop("resolution")  # type: ignore[index]
    path = tmp_path / "report.json"
    path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError, match="resolution"):
        AttemptReportV12.from_file(path, expected_attempt_id=attempt_id, max_bytes=8192)


def test_attempt_report_rejects_finding_from_another_experiment(tmp_path: Path) -> None:
    attempt_id = new_attempt_id()
    value = _value(attempt_id)
    value["findings"][0]["supporting_experiment_ids"] = [  # type: ignore[index]
        "experiment_" + "9" * 32
    ]
    path = tmp_path / "report.json"
    path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError, match="outside this Attempt report"):
        AttemptReportV12.from_file(path, expected_attempt_id=attempt_id, max_bytes=8192)


def test_attempt_report_rejects_duplicate_finding_experiment_ids(tmp_path: Path) -> None:
    attempt_id = new_attempt_id()
    value = _value(attempt_id)
    experiment_id = "experiment_" + "d" * 32
    value["findings"][0]["supporting_experiment_ids"] = [  # type: ignore[index]
        experiment_id,
        experiment_id,
    ]
    path = tmp_path / "report.json"
    path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError, match="must be unique"):
        AttemptReportV12.from_file(path, expected_attempt_id=attempt_id, max_bytes=8192)


def test_attempt_report_rejects_incomplete_profile_evidence(tmp_path: Path) -> None:
    attempt_id = new_attempt_id()
    value = _value(attempt_id)
    value["profile_evidence"].pop("evidence_chain")  # type: ignore[union-attr]
    path = tmp_path / "report.json"
    path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError, match="evidence_chain"):
        AttemptReportV12.from_file(path, expected_attempt_id=attempt_id, max_bytes=8192)


def test_attempt_report_allows_explicitly_absent_profile_evidence(tmp_path: Path) -> None:
    attempt_id = new_attempt_id()
    value = _value(attempt_id)
    value["profile_evidence"] = None
    path = tmp_path / "report.json"
    path.write_text(json.dumps(value), encoding="utf-8")

    report = AttemptReportV12.from_file(
        path,
        expected_attempt_id=attempt_id,
        max_bytes=8192,
    )

    assert report.profile_evidence is None


def test_attempt_report_profile_evidence_requires_a_profile_result(tmp_path: Path) -> None:
    attempt_id = new_attempt_id()
    value = _value(attempt_id)
    value["profile_evidence"]["supporting_results"][0]["operation"] = "dev"  # type: ignore[index]
    path = tmp_path / "report.json"
    path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError, match="Input should be 'profile'"):
        AttemptReportV12.from_file(path, expected_attempt_id=attempt_id, max_bytes=8192)


def test_attempt_report_requires_complete_before_after_comparison(tmp_path: Path) -> None:
    attempt_id = new_attempt_id()
    value = _value(attempt_id)
    value["experiments"][0]["after"] = None  # type: ignore[index]
    path = tmp_path / "report.json"
    path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError, match="both be present or both be null"):
        AttemptReportV12.from_file(path, expected_attempt_id=attempt_id, max_bytes=8192)


def test_attempt_report_rejects_legacy_experiment_result_field(tmp_path: Path) -> None:
    attempt_id = new_attempt_id()
    value = _value(attempt_id)
    experiment = value["experiments"][0]  # type: ignore[index]
    experiment["result"] = experiment.pop("analysis")
    path = tmp_path / "report.json"
    path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        AttemptReportV12.from_file(path, expected_attempt_id=attempt_id, max_bytes=8192)


def test_attempt_report_rejects_legacy_experiment_decision_field(tmp_path: Path) -> None:
    attempt_id = new_attempt_id()
    value = _value(attempt_id)
    experiment = value["experiments"][0]  # type: ignore[index]
    experiment["decision"] = "continue"
    experiment.pop("action")
    path = tmp_path / "report.json"
    path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        AttemptReportV12.from_file(path, expected_attempt_id=attempt_id, max_bytes=8192)


def _load(tmp_path: Path, value: dict[str, object], attempt_id: str) -> AttemptReportV12:
    path = tmp_path / "report.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    return AttemptReportV12.from_file(path, expected_attempt_id=attempt_id, max_bytes=8192)


def test_sealed_report_without_contributing_trials_defaults_to_none(tmp_path: Path) -> None:
    """Reports sealed before the field existed are still re-parsed by the Bootstrap path."""
    attempt_id = new_attempt_id()
    value = _value(attempt_id)
    assert "contributing_kernel_trial_ids" not in value

    assert _load(tmp_path, value, attempt_id).contributing_kernel_trial_ids == ()


def test_attempt_report_accepts_sorted_unique_contributing_trials(tmp_path: Path) -> None:
    attempt_id = new_attempt_id()
    trials = ["gtrial_" + "a" * 32, "gtrial_" + "b" * 32]
    value = {**_value(attempt_id), "contributing_kernel_trial_ids": trials}

    report = _load(tmp_path, value, attempt_id)

    assert report.contributing_kernel_trial_ids == tuple(trials)


@pytest.mark.parametrize(
    ("trials", "message"),
    [
        (["kerneltrial_" + "a" * 32], "String should match pattern"),
        (["gtrial_" + "a" * 32, "gtrial_" + "a" * 32], "must be unique"),
        (["gtrial_" + "b" * 32, "gtrial_" + "a" * 32], "must be sorted"),
    ],
)
def test_attempt_report_rejects_invalid_contributing_trials(
    tmp_path: Path, trials: list[str], message: str
) -> None:
    attempt_id = new_attempt_id()
    value = {**_value(attempt_id), "contributing_kernel_trial_ids": trials}

    with pytest.raises(ValueError, match=message):
        _load(tmp_path, value, attempt_id)


def test_report_field_set_matches_the_pinned_core_contract() -> None:
    """A drift between the two definitions rejects valid reports at one boundary only."""
    core_src = Path(__file__).resolve().parents[1] / "src/atrex-kernel-agent-core/src"
    sys.path.insert(0, str(core_src))
    try:
        from runtime_tools import _REPORT_FIELDS
    finally:
        sys.path.remove(str(core_src))

    # Runtime attaches these four itself; every other field is Agent-supplied.
    runtime_owned = {"schema_version", "attempt_id", "experiments", "direction_events"}
    assert set(AttemptReportV12.model_fields) - runtime_owned == set(_REPORT_FIELDS)
