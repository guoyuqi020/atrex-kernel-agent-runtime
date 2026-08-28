"""Agent-safe Gateway rejection and hidden-result projection tests."""

from __future__ import annotations

import json

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
                "shape_valid": {"0": {"input_kwargs": {"m": 4096}}},
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


def _nested_log_tail(*, levels: int = 2, clip: bool = True) -> str:
    """Build a log tail the way Agate does: an eval document escaped into nested JSON."""
    document = {
        "passed": {
            "compile": {
                "0": {
                    "status": "failed",
                    "reason": "1/1 correctness cases failed: candidate raised exception",
                }
            },
            "correctness": {
                "0": {
                    "status": "failed",
                    "reason": "1/1 correctness cases failed: candidate raised exception",
                }
            },
            "performance": {
                "0": {
                    "status": "skipped",
                    "reason": "Skipped because correctness stage did not pass.",
                }
            },
        },
        "correctness": {
            "shapes": {
                "0": {
                    "cases": [
                        {
                            "input_artifact": {"seed": 151904033, "format": "manual_seed"},
                            "error": (
                                "Traceback (most recent call last):\n"
                                '  File "/tmp/agate-pf_x/deps/atrex_bench/eval/correctness.py",'
                                " line 403, in check_correctness\n"
                                "    candidate_output = loaded_models.candidate_model(\n"
                                '  File "/venv/lib/python3.12/site-packages/torch/nn/modules'
                                '/module.py", line 1775, in _wrapped_call_impl\n'
                                "    return self._call_impl(*args, **kwargs)\n"
                                '  File "/tmp/agate-pf_x/candidate/kernel.py", line 365,'
                                " in forward\n"
                                "    return self._ext.fa_forward(\n"
                                "RuntimeError: baseline kernel supports page_size == 32\n"
                            ),
                        }
                    ]
                }
            }
        },
    }
    tail = json.dumps(document)
    for _ in range(levels):
        tail = json.dumps(tail)[1:-1]
    # Agate keeps only the last window of the log, which clips the first stage key.
    return tail[tail.index("ile") :] if clip else tail


def test_a_failed_job_reports_why_without_revealing_evaluator_cases() -> None:
    projected = project_private_job(
        {
            "status": "failed",
            "job_id": "pf_e29fe46c7793",
            "result": None,
            "error": {
                "reason": "code_execution_failed",
                "trace_id": "req-30bea68ed7a2",
                "details": {
                    "failure_origin": "code",
                    "failure_rule": "python_exception",
                    "logs_tail": _nested_log_tail(),
                },
            },
        }
    )

    error = projected["error"]
    assert isinstance(error, dict)
    assert error["failure_origin"] == "code"
    assert error["failure_rule"] == "python_exception"
    assert error["reason"] == "code_execution_failed"
    assert error["trace_id"] == "req-30bea68ed7a2"
    assert error["stages"] == {
        "compile": {
            "status": "failed",
            "reason": "1/1 correctness cases failed: candidate raised exception",
            "key_truncated": True,
        },
        "correctness": {
            "status": "failed",
            "reason": "1/1 correctness cases failed: candidate raised exception",
        },
        "performance": {
            "status": "skipped",
            "reason": "Skipped because correctness stage did not pass.",
        },
    }
    assert error["candidate_traceback"] == (
        'File "candidate/kernel.py", line 365, in forward\n'
        "    return self._ext.fa_forward(\n"
        "RuntimeError: baseline kernel supports page_size == 32"
    )

    rendered = repr(projected)
    for private in (
        "151904033",
        "manual_seed",
        "input_artifact",
        "atrex_bench/eval",
        "/tmp/agate-",
        "site-packages",
        "torch/nn",
        "check_correctness",
    ):
        assert private not in rendered
