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
