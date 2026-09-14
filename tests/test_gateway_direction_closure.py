"""Direction closures select bound evidence, with lifecycle separate from hypothesis verdicts."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest
from conftest import digest
from test_attempt_report import _value
from test_gateway_proxy import NOW_DATETIME, _request, _service

from atrex_runtime.domain.ids import new_attempt_id
from atrex_runtime.gateway import GatewayCapabilityPolicy, GatewayOperation
from atrex_runtime.gateway.control import BootstrapGatewaySubject


@pytest.mark.anyio
@pytest.mark.parametrize("bootstrap", [False, True])
@pytest.mark.parametrize("action", ["complete", "abandon", "block", "defer"])
async def test_closure_requires_its_own_experiment_and_a_gateway_result(
    tmp_path: Path, bootstrap: bool, action: str
) -> None:
    registry, control, attempt, capability, service, adapter = _service(tmp_path)
    try:
        if bootstrap:
            lineage = registry.get_lineage(registry.get_epoch(attempt.epoch_id).lineage_id)
            campaign = registry.get_campaign(lineage.campaign_id)
            subject = BootstrapGatewaySubject(
                attempt_id=new_attempt_id(),
                campaign_id=lineage.campaign_id,
                lineage_id=lineage.id,
                epoch_id=attempt.epoch_id,
                kernel_agent_revision_id=attempt.kernel_agent_revision_id,
                operator=campaign.operator,
                hardware_target=campaign.hardware_target,
                dsl=lineage.dsl,
                evaluation_contract_digest=campaign.evaluation_contract_digest,
                input_kernel_digest=digest("seed"),
                evidence_digest=digest("evidence"),
                created_at=NOW_DATETIME,
            )
            capability = control.issue_bootstrap(
                subject,
                GatewayCapabilityPolicy(
                    frozenset({GatewayOperation.EVALUATE}),
                    1,
                    NOW_DATETIME + timedelta(hours=1),
                ),
            )
            attempt = replace(attempt, id=subject.attempt_id)

        async def query(operation: str, key: str, **fields: Any) -> dict[str, Any]:
            response = await service.execute(
                capability.token,
                json.dumps(
                    {
                        "schema_version": 2,
                        "attempt_id": attempt.id,
                        "idempotency_key": key,
                        "operation": operation,
                        **fields,
                    }
                ).encode(),
                operation_scope="journal",
            )
            return response.result

        async def start(key: str) -> str:
            proposed = await query(
                "direction_update",
                key,
                request={
                    "action": "propose",
                    "name": key,
                    "hypothesis": "the toolchain supports the intended change",
                    "rationale": "check feasibility before measurement",
                    "plan": ["inspect the required toolchain"],
                    "success_criteria": "the required toolchain is available",
                    "stop_conditions": "a required tool is missing",
                },
            )
            direction_id = proposed["direction_id"]
            await query(
                "direction_update",
                key + "-start",
                request={
                    "action": "start",
                    "direction_id": direction_id,
                    "analysis": "Inspect tools",
                },
            )
            return direction_id

        async def record(direction_id: str, *, unbound: bool = False) -> dict[str, Any]:
            return await query(
                "experiment_record",
                direction_id + ("-unbound" if unbound else "-record"),
                request={
                    "direction_id": direction_id,
                    "name": "toolchain feasibility check",
                    "hypothesis": "the required toolchain is available",
                    "change": "None; no candidate was measured",
                    "before": None
                    if unbound
                    else {"result_artifact_digest": diagnostic.result_artifact_digest},
                    "after": None,
                    "evidence": "The required compiler executable is missing",
                    "analysis": "Unable to proceed; no performance claim is made",
                    "action": "abandon_direction",
                },
            )

        unrelated = await start("unrelated")
        diagnostic = await service.execute(capability.token, _request(attempt))
        unrelated_receipt = await record(unrelated)
        await query(
            "direction_update",
            "unrelated-close",
            request={
                "action": "block",
                "direction_id": unrelated,
                "analysis": "Recorded blocker",
                "hypothesis_status": "unresolved",
                "supporting_experiment_ids": [unrelated_receipt["experiment_id"]],
            },
        )
        direction_id = await start("target")
        events = control.list_direction_events(attempt.id)
        close = {
            "action": action,
            "direction_id": direction_id,
            "analysis": "Finish investigation",
            "hypothesis_status": "unresolved",
            "supporting_experiment_ids": [],
        }
        with pytest.raises(
            ValueError, match="requires at least one associated Experiment"
        ) as error:
            await query("direction_update", "target-empty", request=close)
        assert "record-experiment" in str(error.value)
        assert "real Kernel-bound Gateway Result" in str(error.value)
        assert control.list_direction_events(attempt.id) == events
        assert len(control.list_experiments(attempt.id)) == 1

        with pytest.raises(ValueError, match="before and after cannot both be null"):
            await record(direction_id, unbound=True)
        assert len(control.list_experiments(attempt.id)) == 1
        with pytest.raises(ValueError, match="must belong to the Direction"):
            await query(
                "direction_update",
                "target-foreign",
                request={
                    **close,
                    "supporting_experiment_ids": [unrelated_receipt["experiment_id"]],
                },
            )
        receipt = await record(direction_id)
        close["supporting_experiment_ids"] = [receipt["experiment_id"]]
        for status in ("supported", "refuted"):
            with pytest.raises(ValueError, match="Gateway Result evidence for every selected"):
                await query(
                    "direction_update",
                    "target-" + status,
                    request={
                        **close,
                        "hypothesis_status": status,
                    },
                )
        closed = await query("direction_update", "target-close", request=close)
        assert await query("direction_update", "target-close", request=close) == closed
        assert len(control.list_direction_events(attempt.id)) == len(events) + 1
        assert control.list_direction_events(attempt.id)[-1]["supporting_experiment_ids"] == [
            receipt["experiment_id"]
        ]
        report = _value(attempt.id)
        report.update(
            status="blocked",
            final_candidate=None,
            profile_evidence=None,
            blocker="Compiler unavailable",
            findings=[],
            direction_events=list(control.list_direction_events(attempt.id)),
            experiments=list(control.list_experiments(attempt.id)),
        )
        request = json.loads(_request(attempt))
        request.update(operation="attempt_report", idempotency_key="handoff", report=report)
        published = await service.execute(capability.token, json.dumps(request).encode())
        assert published.result["status"] == "registered"
        assert adapter.requests  # A real proxy Result was produced by the fake Gateway.
    finally:
        control.close()
        registry.close()
