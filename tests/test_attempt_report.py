"""Attempt report file protocol validation tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from atrex_runtime.domain.ids import new_attempt_id
from atrex_runtime.workers.attempt_report import AttemptReportV3


def _value(attempt_id: str) -> dict[str, object]:
    return {
        "schema_version": 3,
        "attempt_id": attempt_id,
        "status": "candidate_ready",
        "hypothesis": "coalesced loads reduce memory transactions",
        "bottleneck": "memory bandwidth",
        "plan": ["vectorize aligned loads"],
        "change_summary": "replaced scalar loads with vector loads",
        "profile_evidence": "SOL localized memory traffic",
        "evaluation_evidence": "Gateway reported correct and faster",
        "result_interpretation": "the targeted change improved latency",
        "decision": "keep",
        "research_sources": ["wiki://coalescing"],
        "lessons": ["preserve vector alignment"],
        "next_directions": [],
        "experiments": [
            {
                "sequence": 1,
                "recorded_at": "2026-08-16T00:00:00+00:00",
                "name": "aligned loads",
                "hypothesis": "coalescing helps",
                "change": "vectorized loads",
                "candidate_artifact_digest": "sha256:" + "a" * 64,
                "evidence": "SOL memory traffic",
                "result": "correct and faster",
                "decision": "continue",
            }
        ],
    }


def test_attempt_report_is_bound_to_expected_attempt(tmp_path: Path) -> None:
    attempt_id = new_attempt_id()
    path = tmp_path / "report.json"
    path.write_text(json.dumps(_value(attempt_id)), encoding="utf-8")

    report = AttemptReportV3.from_file(path, expected_attempt_id=attempt_id, max_bytes=4096)

    assert report.status == "candidate_ready"
    assert report.decision == "keep"


def test_attempt_report_rejects_inconsistent_terminal_decision(tmp_path: Path) -> None:
    attempt_id = new_attempt_id()
    value = _value(attempt_id)
    value["decision"] = "pivot"
    path = tmp_path / "report.json"
    path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError, match="candidate_ready requires decision=keep"):
        AttemptReportV3.from_file(path, expected_attempt_id=attempt_id, max_bytes=4096)
