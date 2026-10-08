"""Reusable claims retain uncertainty independently of optional Journal modules."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError
from test_attempt_report import _value

from atrex_runtime.domain.ids import new_attempt_id
from atrex_runtime.workers.attempt_report import AttemptFindingV1, AttemptReportV12

MODULES = [(), ("directions",), ("experiments",), ("directions", "experiments")]


def _finding(**overrides: Any) -> dict[str, Any]:
    return {
        "category": "performance",
        "observation": "The output path uses a vectorized copy",
        "root_cause": None,
        "resolution": "Prioritize another candidate for this attempt",
        "lesson": "The output-path cost is still unknown",
        "supporting_experiment_ids": [],
        **overrides,
    }


def _report(modules: tuple[str, ...], finding: dict[str, Any]) -> dict[str, Any]:
    value: dict[str, Any] = _value(str(new_attempt_id()))
    value.update(
        status="pivot",
        final_candidate=None,
        tool_modules=list(modules),
        experiments=[],
        direction_events=[],
        profile_evidence=None,
        findings=[finding],
    )
    return value


def test_legacy_finding_does_not_become_a_certified_causal_conclusion() -> None:
    legacy: dict[str, Any] = _value(str(new_attempt_id()))
    finding = AttemptReportV12.model_validate(legacy).findings[0]

    assert finding.claim_kind == "causal_hypothesis"
    assert finding.assessment == "unresolved"
    assert finding.scope is None
    assert finding.supporting_results == ()


@pytest.mark.parametrize("modules", MODULES)
def test_uninvestigated_finding_can_remain_unresolved_without_an_experiment(
    modules: tuple[str, ...],
) -> None:
    report = AttemptReportV12.model_validate(_report(modules, _finding()))

    assert report.findings[0].root_cause is None
    assert report.findings[0].assessment == "unresolved"
    assert report.findings[0].supporting_experiment_ids == ()


@pytest.mark.parametrize("modules", MODULES)
def test_direct_result_references_do_not_require_an_experiment_module(
    modules: tuple[str, ...],
) -> None:
    finding = _finding(
        claim_kind="observation",
        claim="The candidate completed this check",
        assessment="supported",
        scope="This kernel and target GPU only",
        supporting_results=[
            {
                "kernel_artifact_digest": "sha256:" + "a" * 64,
                "result_artifact_digests": ["sha256:" + "b" * 64],
            }
        ],
    )

    report = AttemptReportV12.model_validate(_report(modules, finding))

    assert report.findings[0].supporting_results[0].kernel_artifact_digest == "sha256:" + "a" * 64
    assert report.findings[0].supporting_experiment_ids == ()


@pytest.mark.parametrize(
    "support",
    [
        {
            "kernel_artifact_digest": "not-a-kernel",
            "result_artifact_digests": ["sha256:" + "b" * 64],
        },
        {
            "kernel_artifact_digest": "sha256:" + "a" * 64,
            "result_artifact_digests": ["not-a-result"],
        },
        {"kernel_artifact_digest": "sha256:" + "a" * 64, "result_artifact_digests": []},
        {
            "kernel_artifact_digest": "sha256:" + "a" * 64,
            "result_artifact_digests": ["sha256:" + "b" * 64] * 2,
        },
    ],
)
def test_direct_evidence_has_exact_nonempty_artifact_identity(support: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        AttemptFindingV1.model_validate(_finding(supporting_results=[support]))
