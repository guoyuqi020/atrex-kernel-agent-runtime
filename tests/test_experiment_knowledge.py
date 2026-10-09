"""Experiment knowledge attribution is optional, durable, and Agent-declared."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pytest
from test_attempt_report import _value as _report_value
from test_gateway_proxy import NOW_DATETIME, _request, _service

from atrex_runtime.artifacts.local import ArtifactKind, LocalArtifactStore
from atrex_runtime.domain.errors import InvalidTransitionError
from atrex_runtime.domain.models import Attempt
from atrex_runtime.gateway import GatewayCapability, GatewayProxyService, SqliteGatewayControl
from atrex_runtime.gateway.journals import RuntimeJournalService
from atrex_runtime.workers.attempt_report import AttemptReportV12

KNOWLEDGE = [
    {
        "record_id": "internal_gpu_wiki::kernel_wiki::aligned-vector-loads",
        "finding": "The historical record recommends aligned vector loads for this layout.",
        "application": "Use four-element vector loads in this candidate.",
    }
]


@dataclass
class _Case:
    control: SqliteGatewayControl
    attempt: Attempt
    capability: GatewayCapability
    service: GatewayProxyService
    request: dict[str, Any]
    result_artifact_digest: str
    kernel_artifact_digest: str
    sequence: int = 0

    async def journal(self, operation: str, **fields: object) -> dict[str, Any]:
        self.sequence += 1
        response = await self.service.execute(
            self.capability.token,
            json.dumps(
                {
                    "schema_version": 2,
                    "attempt_id": self.attempt.id,
                    "idempotency_key": f"knowledge-journal-{self.sequence}",
                    "operation": operation,
                    **fields,
                }
            ).encode(),
            operation_scope="runtime" if operation == "experiment_history" else "journal",
        )
        return cast(dict[str, Any], response.result)


@asynccontextmanager
async def _case(tmp_path: Path, modules: tuple[str, ...]) -> AsyncIterator[_Case]:
    registry, control, attempt, capability, service, _adapter = _service(
        tmp_path, tool_modules=modules
    )
    try:
        measured = await service.execute(capability.token, _request(attempt))
        subject = {"result_artifact_digest": measured.result_artifact_digest}
        case = _Case(
            control,
            attempt,
            capability,
            service,
            {
                "name": "aligned vector loads",
                "hypothesis": "vector loads reduce memory traffic",
                "change": "use four-element vector loads",
                "before": subject,
                "after": subject,
                "evidence": "the recorded full evaluation",
                "analysis": "retain the correct candidate; causal benefit remains unverified",
                "action": "keep_after",
            },
            str(measured.result_artifact_digest),
            str(measured.kernel_artifact_digest),
        )
        if "directions" in modules:
            proposed = await case.journal(
                "direction_update",
                request={
                    "action": "propose",
                    "name": "aligned vector loads",
                    "hypothesis": "vector loads reduce memory traffic",
                    "rationale": "investigate the historical recommendation",
                    "plan": ["measure the vectorized candidate"],
                    "success_criteria": "correct and faster",
                    "stop_conditions": "the recommendation does not apply to this layout",
                },
            )
            case.request["direction_id"] = proposed["direction_id"]
            await case.journal(
                "direction_update",
                request={
                    "action": "start",
                    "direction_id": proposed["direction_id"],
                    "analysis": "test the recommendation",
                },
            )
        yield case
    finally:
        control.close()
        registry.close()


@pytest.mark.anyio
@pytest.mark.parametrize("modules", [("experiments",), ("directions", "experiments")])
@pytest.mark.parametrize("declare_knowledge", [False, True])
async def test_experiment_knowledge_roundtrip_without_wiki_service(
    tmp_path: Path, modules: tuple[str, ...], declare_knowledge: bool
) -> None:
    async with _case(tmp_path, modules) as case:
        request = dict(case.request)
        expected = KNOWLEDGE if declare_knowledge else []
        if declare_knowledge:
            request["knowledge_used"] = KNOWLEDGE
        recorded = await case.journal("experiment_record", request=request)
        experiment_id = recorded["experiment_id"]

        listed = await case.journal("experiments_list")
        loaded = await case.journal("experiment_load", experiment_id=experiment_id)
        snapshot = await case.journal("journal_snapshot")
        assert listed["experiments"][0]["knowledge_used"] == expected
        assert loaded["knowledge_used"] == expected
        assert snapshot["experiments"][0]["knowledge_used"] == expected
        assert case.control.list_experiments(case.attempt.id)[0]["knowledge_used"] == expected
        trial = case.control.list_kernel_trials((case.attempt.id,), limit=10)[0]
        assert trial.annotations[0].experiment["knowledge_used"] == expected

        if "directions" in modules:
            await case.journal(
                "direction_update",
                request={
                    "action": "defer",
                    "direction_id": request["direction_id"],
                    "analysis": "further causal investigation is deferred",
                    "hypothesis_status": "unresolved",
                    "supporting_experiment_ids": [experiment_id],
                },
            )
        snapshot = await case.journal("journal_snapshot")
        report_value = _report_value(case.attempt.id)
        report_value.update(
            status="pivot",
            final_candidate=None,
            findings=[],
            profile_evidence=None,
            tool_modules=list(modules),
            experiments=snapshot["experiments"],
            direction_events=snapshot["direction_events"],
        )
        report = AttemptReportV12.model_validate(report_value)
        journals = RuntimeJournalService(
            case.control, LocalArtifactStore(tmp_path / "artifacts"), tool_modules=modules
        )
        journals.validate_report_journal(report)
        assert report.model_dump(mode="json")["experiments"][0]["knowledge_used"] == expected
        snapshot["experiments"][0]["knowledge_used"] = [] if declare_knowledge else KNOWLEDGE
        altered_report = AttemptReportV12.model_validate(report_value)
        with pytest.raises(ValueError, match="must match the Runtime-owned"):
            journals.validate_report_journal(altered_report)

        reopened = SqliteGatewayControl(
            tmp_path / "gateway.sqlite",
            case.control._registry,
            signing_key=b"p" * 32,
            clock=lambda: NOW_DATETIME,
        )
        try:
            assert reopened.list_experiments(case.attempt.id)[0]["knowledge_used"] == expected
        finally:
            reopened.close()


@pytest.mark.anyio
@pytest.mark.parametrize(
    "invalid",
    [
        None,
        "wiki-record",
        {},
        ["wiki-record"],
        [{}],
        [{"record_id": "wiki-record", "finding": "example"}],
        [{**KNOWLEDGE[0], "record_id": 42}],
        [{**KNOWLEDGE[0], "finding": "  "}],
        [{**KNOWLEDGE[0], "application": ""}],
        [{**KNOWLEDGE[0], "unknown": "extra field"}],
    ],
)
async def test_malformed_knowledge_is_rejected_before_persistence(
    tmp_path: Path, invalid: object
) -> None:
    async with _case(tmp_path, ("experiments",)) as case:
        with pytest.raises(ValueError, match=r"knowledge_used|Knowledge-use"):
            await case.journal(
                "experiment_record", request={**case.request, "knowledge_used": invalid}
            )
        assert case.control.list_experiments(case.attempt.id) == ()
        assert case.control.list_kernel_trials((case.attempt.id,), limit=10)[0].annotations == ()


@pytest.mark.anyio
@pytest.mark.parametrize("modules", [(), ("directions",)])
async def test_knowledge_does_not_enable_disabled_experiment_tools(
    tmp_path: Path, modules: tuple[str, ...]
) -> None:
    async with _case(tmp_path, modules) as case:
        with pytest.raises(ValueError, match="Experiment tools are disabled"):
            await case.journal(
                "experiment_record", request={**case.request, "knowledge_used": KNOWLEDGE}
            )
        assert case.control.list_experiments(case.attempt.id) == ()


@pytest.mark.anyio
async def test_old_durable_annotations_reconcile_with_default_knowledge(tmp_path: Path) -> None:
    async with _case(tmp_path, ("experiments",)) as case:
        experiment = cast(list[dict[str, Any]], _report_value(case.attempt.id)["experiments"])[0]
        subject = {
            "kernel_artifact_digest": case.kernel_artifact_digest,
            "result_artifact_digests": [case.result_artifact_digest],
        }
        experiment.update(direction_id=None, before=subject, after=subject)
        assert "knowledge_used" not in experiment
        case.control.append_experiment(
            case.attempt.id, "old-client-record", experiment, recovery_generation=0
        )
        case.control.record_kernel_trial_annotations(case.attempt.id, [experiment])
        snapshot = await case.journal("journal_snapshot")
        normalized = snapshot["experiments"]
        assert normalized[0]["knowledge_used"] == []
        case.control.record_kernel_trial_annotations(case.attempt.id, normalized)
        normalized[0]["knowledge_used"] = KNOWLEDGE
        with pytest.raises(InvalidTransitionError, match="different evidence"):
            case.control.record_kernel_trial_annotations(case.attempt.id, normalized)


@pytest.mark.anyio
@pytest.mark.parametrize("declare_knowledge", [False, True])
async def test_frozen_report_knowledge_is_available_in_history_and_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, declare_knowledge: bool
) -> None:
    async with _case(tmp_path, ("experiments",)) as case:
        report_value = _report_value(case.attempt.id)
        experiment = cast(list[dict[str, Any]], report_value["experiments"])[0]
        if declare_knowledge:
            experiment["knowledge_used"] = KNOWLEDGE
        experiment["direction_id"] = None
        report_digest = LocalArtifactStore(tmp_path / "artifacts").put_json(
            cast(Any, report_value), ArtifactKind.ATTEMPT_REPORT
        )
        monkeypatch.setattr(
            case.control,
            "visible_attempt_report_artifacts",
            lambda _attempt_id: ((case.attempt.id, report_digest),),
        )
        loaded = await case.journal("experiment_load", experiment_id=experiment["experiment_id"])
        assert loaded["knowledge_used"] == (KNOWLEDGE if declare_knowledge else [])
        history = await case.journal("experiment_history")
        assert history["journals"][0][0]["knowledge_used"] == (
            KNOWLEDGE if declare_knowledge else []
        )
