"""Agent-safe Gateway rejection and hidden-result projection tests."""

from __future__ import annotations

import json

import pytest

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


SHIM_MESSAGE = (
    "candidate: source validation failed: kernel.py: Blocked import: sys "
    "(module 'sys' is not permitted); vendor/flash_attn/cache_utils.py: "
    "Blocked import: pickle (module 'pickle' is not permitted); "
    "vendor/算子.py: Forbidden name: getattr"
)


def _shim_log_tail(*, levels: int = 0, payload: dict | None = None) -> str:
    value = (
        "private evaluator output\n===ATREX_SHIM_ERROR_BEGIN===\n"
        + json.dumps(payload if payload is not None else {
            "error_class": "source_archive_invalid", "message": SHIM_MESSAGE,
            "input_py": "private-input", "shapes": {"secret": 100},
        }, ensure_ascii=False)
        + "\n===ATREX_SHIM_ERROR_END===\nprivate evaluator output"
    )
    for _ in range(levels):
        value = json.dumps(value, ensure_ascii=False)[1:-1]
    return value


@pytest.mark.parametrize("levels", range(5))
@pytest.mark.parametrize("role", ["candidate", "baseline"])
def test_source_archive_shim_retains_files_and_reasons(levels: int, role: str) -> None:
    message = SHIM_MESSAGE.replace("candidate:", f"{role}:")
    result = project_private_job({
        "status": "failed", "job_id": "ev_shim",
        "error": {"reason": "source_archive_invalid", "error_class": "validation",
                  "details": {"logs_tail": _shim_log_tail(levels=levels, payload={
                      "error_class": "source_archive_invalid", "message": message,
                      "input_py": "private-input", "shapes": {"secret": 100},
                  })}},
    })
    shim = result["error"]["shim_error"]
    assert shim["message"] == message
    assert shim["error_class"] == "source_archive_invalid"
    assert shim["retryable"] is False
    assert "Modify" in shim["resolution"]
    assert "private" not in json.dumps(result)
    assert "secret" not in json.dumps(result)


@pytest.mark.parametrize("payload", [
    {"error_class": "correctness_failed", "message": "private shape failed"},
    {"error_class": "source_archive_invalid", "message": "reference: private-input"},
    {"error_class": "source_archive_invalid", "message": "input: private-input"},
    {"message": SHIM_MESSAGE},
    {"error_class": "source_archive_invalid", "message": {"private": "input"}},
])
def test_shim_does_not_expose_unqualified_evaluator_errors(payload: dict) -> None:
    result = project_private_job({
        "status": "failed", "error": {"details": {"logs_tail": _shim_log_tail(payload=payload)}},
    })
    assert "shim_error" not in json.dumps(result)
    assert "private" not in json.dumps(result)


@pytest.mark.parametrize("tail", [
    "===ATREX_SHIM_ERROR_BEGIN===not-json===ATREX_SHIM_ERROR_END===",
    "===ATREX_SHIM_ERROR_BEGIN===[]===ATREX_SHIM_ERROR_END===",
    "===ATREX_SHIM_ERROR_BEGIN===", "===ATREX_SHIM_ERROR_END===", "", None,
])
def test_truncated_or_malformed_shim_is_safe(tail: str | None) -> None:
    result = project_private_job({"status": "failed", "error": {"details": {"logs_tail": tail}}})
    assert "shim_error" not in json.dumps(result)


def test_source_archive_shim_has_explicit_diagnostic_size_limit() -> None:
    result = project_private_job({"status": "failed", "error": {"details": {
        "logs_tail": _shim_log_tail(payload={"error_class": "source_archive_invalid",
                                            "message": SHIM_MESSAGE + "x" * 20_000}),
    }}})
    shim = result["error"]["shim_error"]
    assert len(shim["message"]) == 16_000
    assert shim["message_truncated"] is True


@pytest.mark.parametrize("levels", range(1, 5))
def test_shim_parses_escaped_body_inside_literal_log_lines(levels: int) -> None:
    result = project_private_job({"status": "failed", "error": {"details": {
        "logs_tail": "literal log\n" + _shim_log_tail(levels=levels) + "\nmore log",
    }}})
    assert result["error"]["shim_error"]["message"] == SHIM_MESSAGE
