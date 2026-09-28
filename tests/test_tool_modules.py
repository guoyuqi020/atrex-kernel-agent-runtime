"""Direction and Experiment tools can be enabled independently."""

from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

import pytest
from test_attempt_report import _value as _report_value
from test_gateway_proxy import NOW_DATETIME, _insert_attempt, _request, _service

from atrex_runtime.domain.models import Dsl
from atrex_runtime.gateway import GatewayCapabilityPolicy, GatewayOperation


def _journal_payload(
    attempt_id: str,
    operation: str,
    key: str,
    request: dict | None = None,
    kernel_artifact_digest: str | None = None,
) -> bytes:
    return json.dumps(
        {
            "schema_version": 2,
            "attempt_id": attempt_id,
            "idempotency_key": key,
            "operation": operation,
            **({"request": request} if request is not None else {}),
            **(
                {"kernel_artifact_digest": kernel_artifact_digest}
                if kernel_artifact_digest is not None
                else {}
            ),
        }
    ).encode()


async def _submit_report(service, attempt, capability, modules: tuple[str, ...]) -> None:
    snapshot = await service.execute(
        capability.token,
        _journal_payload(attempt.id, "journal_snapshot", "report-snapshot"),
        operation_scope="journal",
    )
    report = _report_value(attempt.id)
    report["tool_modules"] = modules
    report["profile_evidence"] = None
    report["direction_events"] = snapshot.result["direction_events"]
    report["experiments"] = snapshot.result["experiments"]
    if "experiments" in modules:
        report["findings"][0]["supporting_experiment_ids"] = [
            item["experiment_id"] for item in snapshot.result["experiments"]
        ]
    else:
        report["findings"][0].pop("supporting_experiment_ids")
    payload = json.loads(_request(attempt))
    payload.pop("latency_prediction")
    payload["operation"] = "attempt_report"
    payload["idempotency_key"] = "modular-report"
    payload["report"] = report
    accepted = await service.execute(
        capability.token, json.dumps(payload).encode(), operation_scope="runtime"
    )
    assert accepted.result["status"] == "registered"


@pytest.mark.anyio
async def test_one_gateway_serves_different_lineage_tool_modules(tmp_path: Path) -> None:
    registry, control, first, first_capability, service, _ = _service(tmp_path)
    second = _insert_attempt(registry, dsl=Dsl.CUDA)
    second_capability = control.issue(
        second.id,
        GatewayCapabilityPolicy(
            frozenset(GatewayOperation), 4, NOW_DATETIME + timedelta(hours=1)
        ),
    )
    for attempt, modules in ((first, []), (second, ["experiments"])):
        lineage_id = registry.get_epoch(attempt.epoch_id).lineage_id
        registry._connection.execute(
            "UPDATE lineages SET tool_modules_json = ? WHERE id = ?",
            (json.dumps(modules), lineage_id),
        )
    assert control.tool_modules_for_attempt(first.id, ("directions", "experiments")) == ()
    assert control.tool_modules_for_attempt(second.id, ("directions", "experiments")) == (
        "experiments",
    )
    with pytest.raises(ValueError, match="Experiment tools are disabled"):
        await service.execute(
            first_capability.token,
            _journal_payload(first.id, "experiments_list", "first-experiments"),
            operation_scope="journal",
        )
    response = await service.execute(
        second_capability.token,
        _journal_payload(second.id, "experiments_list", "second-experiments"),
        operation_scope="journal",
    )
    assert response.result["experiments"] == []
    with pytest.raises(ValueError, match="Direction tools are disabled"):
        await service.execute(
            second_capability.token,
            _journal_payload(second.id, "directions_list", "second-directions"),
            operation_scope="journal",
        )


@pytest.mark.anyio
async def test_experiments_only_records_without_direction(tmp_path: Path) -> None:
    _, control, attempt, capability, service, _ = _service(tmp_path, tool_modules=("experiments",))
    evaluated = await service.execute(capability.token, _request(attempt))
    subject = {"result_artifact_digest": evaluated.result_artifact_digest}
    recorded = await service.execute(
        capability.token,
        _journal_payload(
            attempt.id,
            "experiment_record",
            "standalone-experiment",
            {
                "name": "standalone result",
                "hypothesis": "the candidate is correct",
                "change": "measure the candidate",
                "before": subject,
                "after": subject,
                "evidence": "a completed Evaluate",
                "analysis": "the candidate remains correct",
                "action": "keep_after",
            },
        ),
        operation_scope="journal",
    )
    assert recorded.result["status"] == "recorded"
    assert control.list_experiments(attempt.id)[0]["direction_id"] is None
    experiments = await service.execute(
        capability.token,
        _journal_payload(
            attempt.id,
            "kernel_experiments_find",
            "experiments-only-kernel-lookup",
            kernel_artifact_digest=evaluated.kernel_artifact_digest,
        ),
        operation_scope="journal",
    )
    assert experiments.result["experiment_ids"] == [recorded.result["experiment_id"]]
    with pytest.raises(ValueError, match="Direction tools are disabled"):
        await service.execute(
            capability.token,
            _journal_payload(
                attempt.id,
                "kernel_directions_find",
                "experiments-only-direction-lookup",
                kernel_artifact_digest=evaluated.kernel_artifact_digest,
            ),
            operation_scope="journal",
        )
    await _submit_report(service, attempt, capability, ("experiments",))
    with pytest.raises(ValueError, match="Direction tools are disabled"):
        await service.execute(
            capability.token,
            _journal_payload(attempt.id, "directions_list", "disabled-list"),
            operation_scope="journal",
        )


@pytest.mark.anyio
async def test_directions_only_closes_without_experiment(tmp_path: Path) -> None:
    _, _, attempt, capability, service, _ = _service(tmp_path, tool_modules=("directions",))
    proposed = await service.execute(
        capability.token,
        _journal_payload(
            attempt.id,
            "direction_update",
            "propose",
            {
                "action": "propose",
                "name": "standalone direction",
                "hypothesis": "coalescing might help",
                "rationale": "public operator layout",
                "plan": ["evaluate the candidate"],
                "success_criteria": "correct and faster",
                "stop_conditions": "no improvement",
            },
        ),
        operation_scope="journal",
    )
    direction_id = proposed.result["direction_id"]
    await service.execute(
        capability.token,
        _journal_payload(
            attempt.id,
            "direction_update",
            "start",
            {"action": "start", "direction_id": direction_id, "analysis": "begin"},
        ),
        operation_scope="journal",
    )
    closed = await service.execute(
        capability.token,
        _journal_payload(
            attempt.id,
            "direction_update",
            "defer",
            {
                "action": "defer",
                "direction_id": direction_id,
                "analysis": "time exhausted",
                "hypothesis_status": "unresolved",
            },
        ),
        operation_scope="journal",
    )
    assert closed.result["status"] == "recorded"
    evaluated = await service.execute(capability.token, _request(attempt))
    directions = await service.execute(
        capability.token,
        _journal_payload(
            attempt.id,
            "kernel_directions_find",
            "directions-only-kernel-lookup",
            kernel_artifact_digest=evaluated.kernel_artifact_digest,
        ),
        operation_scope="journal",
    )
    assert directions.result["direction_ids"] == []
    with pytest.raises(ValueError, match="Experiment tools are disabled"):
        await service.execute(
            capability.token,
            _journal_payload(
                attempt.id,
                "kernel_experiments_find",
                "directions-only-experiment-lookup",
                kernel_artifact_digest=evaluated.kernel_artifact_digest,
            ),
            operation_scope="journal",
        )
    await _submit_report(service, attempt, capability, ("directions",))
    with pytest.raises(ValueError, match="Experiment tools are disabled"):
        await service.execute(
            capability.token,
            _journal_payload(attempt.id, "experiments_list", "disabled-list"),
            operation_scope="journal",
        )


@pytest.mark.anyio
async def test_no_modules_rejects_both_journals(tmp_path: Path) -> None:
    _, _, attempt, capability, service, _ = _service(tmp_path, tool_modules=())
    frontier = await service.execute(
        capability.token,
        _journal_payload(attempt.id, "kernel_pareto_frontier", "no-modules-frontier"),
        operation_scope="runtime",
    )
    assert frontier.result["best_by_shape"] == {}
    await service.execute(capability.token, _request(attempt))
    await _submit_report(service, attempt, capability, ())
    for operation, scope, message in (
        ("directions_list", "journal", "Direction tools are disabled"),
        ("experiments_list", "journal", "Experiment tools are disabled"),
        ("direction_history", "runtime", "Direction tools are disabled"),
        ("experiment_history", "runtime", "Experiment tools are disabled"),
    ):
        with pytest.raises(ValueError, match=message):
            await service.execute(
                capability.token,
                _journal_payload(attempt.id, operation, operation),
                operation_scope=scope,
            )
