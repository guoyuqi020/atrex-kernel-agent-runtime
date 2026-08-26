"""Agent-safe Gateway rejection and hidden-result projection tests."""

from __future__ import annotations

from atrex_runtime.gateway.private_results import (
    project_candidate_rejection,
    project_private_job,
)


def test_candidate_source_validation_keeps_actionable_safe_details() -> None:
    projected = project_candidate_rejection(
        {
            "message": "source validation failed",
            "details": {
                "forbidden_imports": ["os"],
                "input_py": "hidden evaluator source",
                "shape": {"m": 4096},
            },
        }
    )

    assert projected == {
        "status": "rejected",
        "error": {
            "category": "candidate_rejected",
            "message": "candidate request rejected before evaluation job creation",
            "details": {
                "message": "source validation failed",
                "details": {"forbidden_imports": ["os"]},
            },
        },
    }


def test_post_job_failure_continues_to_hide_case_diagnostics() -> None:
    projected = project_private_job(
        {
            "status": "failed",
            "job_id": "ev_hidden",
            "stderr": "failure on secret shape m=4096",
            "shape": {"m": 4096},
        }
    )

    assert projected == {
        "status": "failed",
        "job_id": "ev_hidden",
        "error": "gateway job did not complete; hidden-case details withheld",
    }
