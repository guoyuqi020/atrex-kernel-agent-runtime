"""Dev-only policy through migration, HTTP, nomination, trusted gating, and Agent prompts."""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
from contextlib import closing
from pathlib import Path

import pytest
from conftest import digest, seed_lineage
from httpx import ASGITransport, AsyncClient
from test_attempt_report import _value as report_value
from test_gateway_finalization import FakeClient, FakeContexts, FakeEvents, _builder
from test_gateway_proxy import _request, _service

from atrex_runtime.ablation import AblationArmSpecV1
from atrex_runtime.gateway import GatewayOperation, GatewayProxyAsgiApp, GatewayProxyLimits
from atrex_runtime.gateway.finalization import (
    AgateAuthoritativeCandidateEvaluator,
    BootstrapEvaluationStage,
)
from atrex_runtime.gateway.proxy import GatewayAdapterResult
from atrex_runtime.registry.sqlite import SCHEMA_VERSION, SqliteRegistry
from atrex_runtime.workers.evidence_view import optimizer_evidence_prompt
from atrex_runtime.workers.session_contract import materialize_session_contract

ROOT = Path(__file__).resolve().parents[1]


def enable_dev_only(registry, attempt):
    lineage_id = registry.get_epoch(attempt.epoch_id).lineage_id
    registry._connection.execute(
        "UPDATE lineages SET dev_only=1, tool_modules_json='[]', "
        "result_artifact_read_enabled=0 WHERE id=?",
        (lineage_id,),
    )
    return lineage_id


def test_migrate_preserves_standard_lineages_and_reopens_dev_only(tmp_path):
    path = tmp_path / "registry.sqlite"
    with SqliteRegistry(path) as registry:
        lineage_id = seed_lineage(registry).lineage_id
    with closing(sqlite3.connect(path)) as db:
        db.execute("ALTER TABLE lineages DROP COLUMN dev_only")
        db.execute("PRAGMA user_version=46")
        db.commit()
    with SqliteRegistry(path) as registry:
        assert registry.get_lineage(lineage_id).dev_only is False
        registry._connection.execute(
            "UPDATE lineages SET dev_only=1, tool_modules_json='[]', "
            "result_artifact_read_enabled=0 WHERE id=?",
            (lineage_id,),
        )
    with SqliteRegistry(path) as registry:
        assert registry.get_lineage(lineage_id).dev_only is True
        assert registry._connection.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION


@pytest.mark.parametrize(
    "patch",
    [
        {"dev_only": "true"},
        {"dev_only": 1},
        {"tool_modules": None},
        {"tool_modules": ["experiments"]},
        {"result_artifact_read_enabled": True},
    ],
)
def test_rejects_ambiguous_or_conflicting_policy(patch):
    with pytest.raises(ValueError, match="dev_only"):
        AblationArmSpecV1.model_validate(
            {
                "creation_key": "dev-only",
                "source_lineage_id": "lineage_" + "0" * 32,
                "optimizer_attempt_budget": 3,
                "dev_only": True,
                "tool_modules": [],
                "result_artifact_read_enabled": False,
                **patch,
            }
        )


@pytest.mark.anyio
async def test_dev_only_enforced_on_http_and_shared_authorization(tmp_path):
    registry, control, attempt, capability, service, adapter = _service(tmp_path)
    enable_dev_only(registry, attempt)
    adapter.result = GatewayAdapterResult("completed", {"stdout": "probe passed", "exit_code": 0})
    base = {"schema_version": 2, "attempt_id": attempt.id}
    candidate = json.loads(_request(attempt))["candidate"]
    headers = {"Authorization": f"Bearer {capability.token}"}
    try:
        async with AsyncClient(
            transport=ASGITransport(
                app=GatewayProxyAsgiApp(service, GatewayProxyLimits(65536, 8, 16384))
            ),
            base_url="http://runtime",
        ) as http:
            for operation, fields in [
                ("evaluate", {"candidate": candidate, "latency_prediction": "improved"}),
                ("evaluate", {"candidate": candidate, "mode": "correctness_only"}),
                (
                    "evaluate",
                    {
                        "candidate": candidate,
                        "latency_prediction": "retained",
                        "baseline": candidate,
                        "comparison": {"method": "abba", "repeats": 2},
                    },
                ),
                ("profile", {"candidate": candidate}),
                ("check", {"candidate": candidate}),
                ("disassemble", {"candidate": candidate}),
                ("env", {}),
                ("kernel_pareto_frontier", {}),
            ]:
                url = "/v1/runtime/queries" if operation.startswith("kernel_") else "/v1/operations"
                response = await http.post(
                    url,
                    headers=headers,
                    json={
                        **base,
                        "operation": operation,
                        "idempotency_key": operation,
                        **fields,
                    },
                )
                assert response.status_code == 403, response.text
                assert "dev-only" in response.text
            assert not adapter.requests
            request = {
                **base,
                "operation": "dev",
                "candidate": candidate,
                "idempotency_key": "probe",
                "command": "python3 probe.py",
            }
            response = await http.post("/v1/operations", headers=headers, json=request)
            assert response.status_code == 200, response.text
            assert response.json()["result"]["stdout"] == "probe passed"
            replay = await http.post("/v1/operations", headers=headers, json=request)
            assert replay.json() == response.json()
            assert len(adapter.requests) == 1
            assert not control.list_evaluations(attempt.id)
            source_request = {
                **base,
                "operation": "kernel_artifact_read",
                "idempotency_key": "read-source",
                "kernel_artifact_digest": response.json()["kernel_artifact_digest"],
                "file": "kernel.py",
            }
            source = await http.post("/v1/runtime/queries", headers=headers, json=source_request)
            assert source.status_code == 200, source.text
            assert source.json()["result"]["content"] == "def kernel(): pass\n"
            assert "correct" not in source.json()["result"]
            assert "latency_us_geomean" not in source.json()["result"]
            hidden = await http.post(
                "/v1/runtime/queries",
                headers=headers,
                json={
                    **source_request,
                    "idempotency_key": "hidden-source",
                    "kernel_artifact_digest": digest("invisible-source"),
                },
            )
            assert hidden.status_code == 400, hidden.text
            assert "outside the visible Lineage history" in hidden.text
            read_result = await http.post(
                "/v1/runtime/queries",
                headers=headers,
                json={
                    **base,
                    "operation": "result_artifact_read",
                    "idempotency_key": "hidden-result",
                    "result_artifact_digest": response.json()["result_artifact_digest"],
                },
            )
            assert read_result.status_code == 400, read_result.text
            assert "reads are disabled" in read_result.text
            assert len(adapter.requests) == 1
        # Same authorization path protects Wiki, hidden legacy endpoints and forged broad tokens.
        for operation in GatewayOperation:
            if operation in {
                GatewayOperation.DEV,
                GatewayOperation.KERNEL_ARTIFACT_READ,
                GatewayOperation.ATTEMPT_REPORT,
                GatewayOperation.ATTEMPT_REPORT_STATUS,
            }:
                continue
            with pytest.raises(PermissionError, match="dev-only"):
                control.authorize(
                    capability,
                    operation,
                    idempotency_key="bypass-" + operation,
                    request_digest=digest("bypass"),
                )
    finally:
        control.close()
        registry.close()


@pytest.mark.anyio
@pytest.mark.parametrize("passing", [True, False])
@pytest.mark.parametrize("independent_evaluate", [True, False])
async def test_nomination_without_agent_eval_still_runs_authoritative_gate(
    tmp_path,
    passing,
    independent_evaluate,
):
    registry, control, attempt, capability, service, adapter = _service(tmp_path)
    lineage_id = enable_dev_only(registry, attempt)
    report = report_value(attempt.id)
    report.update(
        tool_modules=[],
        experiments=[],
        direction_events=[],
        findings=[],
        profile_evidence=None,
        knowledge_used=[],
        evidence_summary={
            "correctness": "Not independently validated",
            "performance": "Exploratory only",
        },
    )
    payload = json.loads(_request(attempt))
    payload.pop("latency_prediction")
    payload.update(operation="attempt_report", report=report, idempotency_key="nominate")
    try:
        # Check the ordinary gate before accepting a report: accepted reports are
        # immutable within a recovery generation, regardless of the arm policy.
        registry._connection.execute("UPDATE lineages SET dev_only=0 WHERE id=?", (lineage_id,))
        payload["idempotency_key"] = "standard-nomination"
        with pytest.raises(ValueError, match="completed Agent evaluate"):
            await service.execute(
                capability.token, json.dumps(payload).encode(), operation_scope="runtime"
            )
        registry._connection.execute("UPDATE lineages SET dev_only=1 WHERE id=?", (lineage_id,))
        payload["idempotency_key"] = "nominate"
        response = await service.execute(
            capability.token, json.dumps(payload).encode(), operation_scope="runtime"
        )
        assert response.result["status"] == "registered"
        assert not control.list_evaluations(attempt.id) and not adapter.requests

        class Client(FakeClient):
            def get_job(self, *args, **kwargs):
                job = super().get_job(*args, **kwargs)
                job["result"] = {"all_pass": passing, "latency_us_geomean": 7.5}
                return job

        client = Client()
        finalizer = AgateAuthoritativeCandidateEvaluator(
            client,
            _builder,
            FakeContexts(),
            service._artifacts,
            control,
            FakeEvents(),
            wait_timeout_s=100,
            bootstrap_stages=(BootstrapEvaluationStage(1),),
        )
        result = await finalizer.finalize(
            attempt.id, response.kernel_artifact_digest, independent_evaluate=independent_evaluate
        )
        assert len(client.submitted) == 1
        assert result.correct is passing
        assert result.latency_us == (7.5 if passing else None)
        assert control.get_committed_outcome(attempt.id) == result
        assert all(
            row.source.value == "runtime_final" for row in control.list_evaluations(attempt.id)
        )
        recovered = await finalizer.finalize(
            attempt.id, response.kernel_artifact_digest, independent_evaluate=independent_evaluate
        )
        assert recovered == result and len(client.submitted) == 1
    finally:
        control.close()
        registry.close()


@pytest.mark.parametrize("bundle", ["atrex-kernel-agent-core", "kernel-design-agents"])
def test_cli_schema_normal_and_recovery_prompts_only_offer_dev(tmp_path, bundle):
    contract = materialize_session_contract(
        tmp_path,
        phase="optimization_attempt",
        dsl="cuda",
        hardware_target="L20D",
        agent_backend="claude",
        model=None,
        session_timeout_seconds=3600,
        usage_unit="provider_tokens",
        usage_budget=1000,
        max_attempt_report_bytes=100_000,
        wiki_available=True,
        tool_modules=(),
        result_artifact_read_enabled=False,
        dev_only=True,
    )
    script = """
import json, sys
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0, sys.argv[1])
from runtime_contract import project_contract
from runtime_tools import _available_attempt_commands
from sessions.tool_module_prompts import modular_tool_instructions, modular_workflow
from sessions.report_completion import _missing_prompt
root = Path(sys.argv[1]).parent
_, contract = project_contract(allow_baseline=False)
ctx = SimpleNamespace(manifest={}, session_trace_path=None)
print(json.dumps({"contract":contract,"commands":_available_attempt_commands(),
 "tools":modular_tool_instructions((root/'prompts/attempt-tools.md').read_text(),'cuda',frozenset()),
 "workflow":modular_workflow('',bootstrap=False,modules=frozenset()),
 "recovery":_missing_prompt(ctx,None)}))
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(ROOT / "src" / bundle / "src")],
        env={**os.environ, "ATREX_RUNTIME_CONTRACT_PATH": str(contract)},
        check=True,
        capture_output=True,
        text=True,
    )
    value = json.loads(result.stdout)
    assert set(value["commands"]) == {"gateway-execute", "kernel-artifact-read", "attempt-report"}
    assert set(value["contract"]["tools"]) == set(value["commands"]) | {"runtime-contract"}
    assert set(value["contract"]["tools"]["gateway-execute"]["operations"]) == {"dev"}
    assert value["contract"]["environment"]["services"]["wiki"] is False
    assert "file_paths" in value["tools"] and "private" in value["tools"]
    assert "kernel-artifact-read" in value["tools"] and "artifact_file" in value["tools"]
    assert "does not require an Agent Evaluate" in value["recovery"]
    for visibility in ("isolated", "broadcast"):
        evidence = optimizer_evidence_prompt(
            (), trajectory_visibility=visibility, result_artifact_read_enabled=False, dev_only=True
        )
        assert "report.json" in evidence and "conversation.jsonl" in evidence
        assert "without requiring an Agent Evaluate" in evidence
        assert "kernel-artifact-read" in evidence and "artifact_file" in evidence
        for disabled in ("result-artifact-read", "kernel-pareto-frontier", "record-experiment"):
            assert disabled not in evidence + value["tools"] + value["workflow"]


@pytest.mark.parametrize(
    "path",
    [
        "data/FA4/independent-ablation.json",
        "data/GDN/native-eval-bootstrap/independent-ablation.json",
        "data/qwen35-fa-l20n/independent-ablation.json",
    ],
)
def test_independent_plans_add_three_dev_only_repeats(path):
    plan = json.loads((ROOT / path).read_text())
    arms = [arm for arm in plan["arms"] if arm.get("dev_only")]
    assert len(plan["arms"]) == 21
    assert {arm["replica"] for arm in arms} == {1, 2, 3}
    assert all(
        arm["configuration"] == "dev-only"
        and arm["tool_modules"] == []
        and arm["result_artifact_read_enabled"] is False
        and arm["optimizer_attempt_budget_total"] == 15
        for arm in arms
    )
