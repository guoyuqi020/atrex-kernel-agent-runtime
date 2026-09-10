"""Strict Worker token-usage report tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from atrex_runtime.domain.models import TokenUsage
from atrex_runtime.workers.token_usage import ProviderUsageReportV2


def _write_report(path: Path, *, total: int = 105, budget: int = 100) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "usage_unit": "provider_tokens",
                "budget": budget,
                "consumed": total,
                "token_usage": {
                    "uncached_input_tokens": 50,
                    "output_tokens": 25,
                    "cache_read_tokens": 20,
                    "cache_write_tokens": 10,
                },
                "credits": None,
                "budget_exhausted": total >= budget,
                "session_count": 1,
                "model_request_count": 1,
                "usage_complete": True,
            }
        ),
        encoding="utf-8",
    )


def test_report_preserves_actual_consumption_above_budget(tmp_path: Path) -> None:
    path = tmp_path / "usage.json"
    _write_report(path)

    report = ProviderUsageReportV2.from_file(
        path, expected_unit="provider_tokens", expected_budget=100
    )

    assert report.to_domain() == TokenUsage(50, 25, 20, 10)
    assert report.consumed == 105
    assert report.budget_exhausted


def test_report_rejects_inconsistent_derived_total(tmp_path: Path) -> None:
    path = tmp_path / "usage.json"
    _write_report(path, total=104)

    with pytest.raises(ValueError, match="consumed amount is inconsistent"):
        ProviderUsageReportV2.from_file(path, expected_unit="provider_tokens", expected_budget=100)


def test_report_rejects_a_different_deployment_budget(tmp_path: Path) -> None:
    path = tmp_path / "usage.json"
    _write_report(path, budget=200)

    with pytest.raises(ValueError, match="different budget"):
        ProviderUsageReportV2.from_file(path, expected_unit="provider_tokens", expected_budget=100)


def test_report_accepts_complete_usage_without_a_budget(tmp_path: Path) -> None:
    path = tmp_path / "usage.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "usage_unit": "provider_tokens",
                "budget": None,
                "consumed": 105,
                "token_usage": {
                    "uncached_input_tokens": 50,
                    "output_tokens": 25,
                    "cache_read_tokens": 20,
                    "cache_write_tokens": 10,
                },
                "credits": None,
                "budget_exhausted": False,
                "session_count": 1,
                "model_request_count": 1,
                "usage_complete": True,
            }
        ),
        encoding="utf-8",
    )

    report = ProviderUsageReportV2.from_file(
        path, expected_unit="provider_tokens", expected_budget=None
    )

    assert report.budget is None
    assert report.consumed == 105
    assert report.budget_exhausted is False


def test_report_accepts_qoder_native_credits(tmp_path: Path) -> None:
    path = tmp_path / "usage.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "usage_unit": "credits",
                "budget": 100.0,
                "consumed": 13.75,
                "token_usage": {
                    "uncached_input_tokens": 0,
                    "output_tokens": 0,
                    "cache_read_tokens": 0,
                    "cache_write_tokens": 0,
                },
                "credits": 13.75,
                "budget_exhausted": False,
                "session_count": 1,
                "model_request_count": 1,
                "usage_complete": True,
            }
        ),
        encoding="utf-8",
    )

    report = ProviderUsageReportV2.from_file(path, expected_unit="credits", expected_budget=100.0)

    assert report.to_domain() == TokenUsage(0, 0, 0, 0, credits=13.75)
    assert report.consumed == 13.75


@pytest.mark.parametrize("allow", [False, True])
def test_known_claude_accounting_gap_requires_explicit_backend_permission(
    tmp_path: Path,
    allow: bool,
) -> None:
    path = tmp_path / "usage.json"
    _write_report(path)
    value = json.loads(path.read_text())
    value.update(
        usage_complete=False,
        usage_warnings=["claude_response_usage_incomplete_or_unreconciled"],
    )
    path.write_text(json.dumps(value))
    if not allow:
        with pytest.raises(ValueError, match="without provider-reported"):
            ProviderUsageReportV2.from_file(
                path,
                expected_unit="provider_tokens",
                expected_budget=100,
            )
        return
    report = ProviderUsageReportV2.from_file(
        path,
        expected_unit="provider_tokens",
        expected_budget=100,
        allow_claude_accounting_gap=True,
    )
    assert not report.usage_complete
    assert report.consumed == 105 and report.budget_exhausted
    assert report.to_domain() == TokenUsage(50, 25, 20, 10)


@pytest.mark.parametrize("mutation", ["warning", "empty", "requests", "sessions", "unknown"])
def test_partial_permission_does_not_accept_unidentified_or_empty_usage(
    tmp_path: Path,
    mutation: str,
) -> None:
    path = tmp_path / "usage.json"
    _write_report(path)
    value = json.loads(path.read_text())
    value.update(
        usage_complete=False,
        usage_warnings=["claude_response_usage_incomplete_or_unreconciled"],
    )
    if mutation == "warning":
        value["usage_warnings"] = []
    elif mutation == "empty":
        value.update(consumed=0, budget_exhausted=False)
        value["token_usage"] = dict.fromkeys(value["token_usage"], 0)
    elif mutation == "requests":
        value["model_request_count"] = 0
    elif mutation == "sessions":
        value["session_count"] = 0
    else:
        value["usage_warnings"] = ["unknown_error"]
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        ProviderUsageReportV2.from_file(
            path,
            expected_unit="provider_tokens",
            expected_budget=100,
            allow_claude_accounting_gap=True,
        )
