"""Frozen result-read ablation through storage, HTTP, live CLI, and rendered Prompts."""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
from contextlib import closing
from datetime import timedelta
from pathlib import Path

import anyio
import pytest
from conftest import seed_lineage
from httpx import ASGITransport, AsyncClient
from test_gateway_adoption_nomination import _candidate
from test_gateway_module_independent_nomination import peer_history as peer_history
from test_gateway_proxy import NOW_DATETIME, _insert_attempt, _request, _service
from test_tool_modules import _submit_report

from atrex_runtime.ablation import AblationArmSpecV1
from atrex_runtime.domain.models import Dsl
from atrex_runtime.gateway import (
    GatewayCapabilityPolicy,
    GatewayOperation,
    GatewayProxyAsgiApp,
    GatewayProxyLimits,
)
from atrex_runtime.registry.sqlite import SCHEMA_VERSION, SqliteRegistry
from atrex_runtime.workers.evidence_view import optimizer_evidence_prompt
from atrex_runtime.workers.session_contract import materialize_session_contract

REPOSITORY = Path(__file__).resolve().parents[1]


def test_migrate_45_keeps_existing_reads_and_persists_disabled_policy(tmp_path: Path) -> None:
    path = tmp_path / "registry.sqlite"
    with SqliteRegistry(path) as registry:
        lineage_id = seed_lineage(registry).lineage_id
    with closing(sqlite3.connect(path)) as db:
        db.execute("ALTER TABLE lineages DROP COLUMN result_artifact_read_enabled")
        db.execute("PRAGMA user_version = 45")
        db.commit()
    with SqliteRegistry(path) as registry:
        assert registry.get_lineage(lineage_id).result_artifact_read_enabled is True
        registry._connection.execute(
            "UPDATE lineages SET result_artifact_read_enabled = 0 WHERE id = ?", (lineage_id,)
        )
    with SqliteRegistry(path) as registry:
        assert registry.get_lineage(lineage_id).result_artifact_read_enabled is False
        assert registry._connection.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION


@pytest.mark.parametrize("invalid", [0, 1, "false", None])
def test_seed_rejects_ambiguous_result_read_switch(invalid: object) -> None:
    with pytest.raises(ValueError, match="result_artifact_read_enabled"):
        AblationArmSpecV1.model_validate(
            {
                "creation_key": "no-result-read",
                "source_lineage_id": "lineage_" + "0" * 32,
                "optimizer_attempt_budget": 3,
                "result_artifact_read_enabled": invalid,
            }
        )


@pytest.mark.anyio
async def test_no_result_read_arm_returns_live_results_but_rejects_readback(tmp_path: Path) -> None:
    registry, control, attempt, capability, service, adapter = _service(tmp_path)
    lineage_id = registry.get_epoch(attempt.epoch_id).lineage_id
    registry._connection.execute(
        "UPDATE lineages SET result_artifact_read_enabled = 0, tool_modules_json = '[]' "
        "WHERE id = ?",
        (lineage_id,),
    )
    app = GatewayProxyAsgiApp(service, GatewayProxyLimits(65536, 8, 16384))
    headers = {"Authorization": f"Bearer {capability.token}"}
    base = {"schema_version": 2, "attempt_id": attempt.id}
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://runtime") as http:
            request = json.loads(_request(attempt))
            evaluated = await http.post("/v1/operations", json=request, headers=headers)
            assert evaluated.status_code == 200, evaluated.text
            result = evaluated.json()
            assert result["result"]["correct"] is True
            assert result["result"]["latency_us_geomean"] == 12.0
            assert control.list_evaluations(attempt.id)
            # Delivery recovery reuses the same request and does not rerun the GPU job.
            replay = await http.post("/v1/operations", json=request, headers=headers)
            assert replay.json() == result
            for scope in ("/v1/runtime/queries", "/v1/operations"):
                read = await http.post(
                    scope,
                    headers=headers,
                    json={
                        **base,
                        "operation": "result_artifact_read",
                        "idempotency_key": "hidden-read",
                        "result_artifact_digest": result["result_artifact_digest"],
                    },
                )
                assert read.status_code == 400, read.text
                assert "reads are disabled" in read.json()["detail"]
                assert "result" not in read.json()
            duplicate = await http.post(
                "/v1/operations", headers=headers, json={**request, "idempotency_key": "new-key"}
            )
            assert duplicate.status_code == 200, duplicate.text
            assert duplicate.json() == result
            assert len(control.list_evaluations(attempt.id)) == 1
            duplicate_replay = await http.post(
                "/v1/operations",
                headers=headers,
                json={**request, "idempotency_key": "new-key"},
            )
            assert duplicate_replay.json() == result
            assert len(adapter.requests) == 1
            # Other queries remain usable, including the explicitly permitted frontier.
            source = await http.post(
                "/v1/runtime/queries",
                headers=headers,
                json={
                    **base,
                    "operation": "kernel_artifact_read",
                    "idempotency_key": "source",
                    "kernel_artifact_digest": result["kernel_artifact_digest"],
                    "file": "kernel.py",
                },
            )
            assert source.status_code == 200, source.text
            assert source.json()["result"]["content"] == "def kernel(): pass\n"
            frontier = await http.post(
                "/v1/runtime/queries",
                headers=headers,
                json={
                    **base,
                    "operation": "kernel_pareto_frontier",
                    "idempotency_key": "frontier",
                },
            )
            assert frontier.status_code == 200, frontier.text
            for legacy in ("kernel_trials", "kernel_trial_show"):
                response = await http.post(
                    "/v1/runtime/queries",
                    headers=headers,
                    json={
                        **base,
                        "operation": legacy,
                        "idempotency_key": legacy,
                    },
                )
                assert response.status_code == 400
        await _submit_report(service, attempt, capability, ())
        # A second Lineage on the same service still has the normal tools.
        second = _insert_attempt(registry, dsl=Dsl.CUDA)
        normal = control.issue(
            second.id,
            GatewayCapabilityPolicy(
                frozenset(GatewayOperation),
                4,
                NOW_DATETIME + timedelta(hours=1),
            ),
        )
        evaluated2 = await service.execute(normal.token, _request(second))
        read2 = await service.execute(
            normal.token,
            json.dumps(
                {
                    "schema_version": 2,
                    "attempt_id": second.id,
                    "idempotency_key": "normal-read",
                    "operation": "result_artifact_read",
                    "result_artifact_digest": evaluated2.result_artifact_digest,
                }
            ).encode(),
            operation_scope="runtime",
        )
        assert read2.result["result"]["correct"] is True
    finally:
        control.close()
        registry.close()


@pytest.mark.parametrize("bundle", ["atrex-kernel-agent-core", "kernel-design-agents"])
@pytest.mark.parametrize("enabled", [True, False])
def test_live_contract_cli_and_prompt_agree(tmp_path: Path, bundle: str, enabled: bool) -> None:
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
        wiki_available=False,
        tool_modules=(),
        result_artifact_read_enabled=enabled,
    )
    bindings = json.loads((contract / "tools.json").read_text())["bindings"]
    assert ("result-artifact-read" in bindings) is enabled
    assert set(bindings) == {
        "gateway-execute",
        "kernel-artifact-read",
        "kernel-pareto-frontier",
        "attempt-report",
        "runtime-contract",
        *(["result-artifact-read"] if enabled else []),
    }
    root = REPOSITORY / "src" / bundle
    env = {**os.environ, "ATREX_RUNTIME_CONTRACT_PATH": str(contract)}
    script = """
import json, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from runtime_contract import project_contract
from runtime_tools import _available_attempt_commands
from sessions.tool_module_prompts import modular_tool_instructions
root = Path(sys.argv[1]).parent
_, contract = project_contract(allow_baseline=False)
print(json.dumps({"contract": contract, "commands": _available_attempt_commands(),
    "prompt": modular_tool_instructions((root / "prompts/attempt-tools.md").read_text(),
                                        "cuda", frozenset())}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script, str(root / "src")],
        env=env,
        check=True,
        text=True,
        capture_output=True,
    )
    value = json.loads(completed.stdout)
    assert ("result-artifact-read" in value["commands"]) is enabled
    assert ("result-artifact-read" in json.dumps(value["contract"])) is enabled
    assert ("`result-artifact-read`" in value["prompt"]) is enabled
    for visibility in ("isolated", "broadcast"):
        evidence = optimizer_evidence_prompt(
            (), trajectory_visibility=visibility, result_artifact_read_enabled=enabled
        )
        assert ("`result-artifact-read`" in evidence) is enabled
        assert "`report.json`" in evidence and "`conversation.jsonl`" in evidence
        assert "`kernel-pareto-frontier`" in evidence
        if not enabled:
            assert "Result Artifact reads are disabled" in evidence
            assert "original tool response" in evidence
            assert "own notes" in value["prompt"]


def test_fa4_independent_plan_has_three_repeats_of_all_seven_settings() -> None:
    plan = json.loads((REPOSITORY / "data/FA4/independent-ablation.json").read_text())
    assert len(plan["arms"]) == 21
    for config in (
        "no-modules",
        "experiments",
        "directions",
        "all",
        "no-result-read",
        "dev-only",
        "raw",
    ):
        arms = [arm for arm in plan["arms"] if arm["configuration"] == config]
        assert {arm["replica"] for arm in arms} == {1, 2, 3}
        assert all(arm["trajectory_visibility"] == "isolated" for arm in arms)
        assert all(arm["optimizer_attempt_budget_total"] == 15 for arm in arms)
        assert all(
            arm.get("result_artifact_read_enabled", True)
            == (config not in {"no-result-read", "dev-only", "raw"})
            for arm in arms
        )


def test_focused_no_result_read_sharing_plans_hold_equal_per_trajectory_budgets() -> None:
    paths = (
        REPOSITORY / "data/FA4/no-result-read-sharing-ablation.json",
        REPOSITORY / "data/qwen35-fa-l20n/no-result-read-sharing-ablation.json",
        REPOSITORY / "data/GDN/native-eval-bootstrap/no-result-read-sharing-ablation.json",
    )
    plans = [json.loads(path.read_text()) for path in paths]
    assert plans[1:] == plans[:-1]
    plan = plans[0]
    assert plan["main_evolve_enabled"] is False
    assert plan["optimizer_attempt_budget_per_trajectory"] == 15
    assert [
        (
            arm["kind"],
            arm["target_epoch_number"],
            arm["workflow_command"],
            arm["optimizer_attempt_budget"],
            arm["optimizer_attempt_budget_total"],
            arm["trajectory_visibility"],
        )
        for arm in plan["arms"]
    ] == [
        ("epoch-shared", 5, "workflow/epoch_shared_3.py", 9, 45, "isolated"),
        ("broadcast", 15, "workflow/broadcast_epoch_1.py", 3, 45, "broadcast"),
    ]
    assert all(
        arm["tool_modules"] == [] and arm["result_artifact_read_enabled"] is False
        for arm in plan["arms"]
    )


@pytest.mark.anyio
@pytest.mark.parametrize("visibility", ["isolated", "broadcast"])
async def test_duplicate_reuse_respects_peer_visibility(peer_history, visibility):
    h = peer_history
    lineage = h.registry.get_epoch(h.current.epoch_id).lineage_id
    h.registry._connection.execute(
        "UPDATE lineages SET result_artifact_read_enabled = 0, trajectory_visibility = ? "
        "WHERE id = ?",
        (visibility, lineage),
    )
    payload = json.loads(_request(h.current))
    payload["candidate"] = _candidate()
    result = await h.service.execute(h.capability.token, json.dumps(payload).encode())
    expected_calls = 1 if visibility == "broadcast" else 2
    assert len(h.adapter.requests) == expected_calls
    if visibility == "broadcast":
        assert result.result_artifact_digest == h.after.result_artifact_digest
    payload["idempotency_key"] = "read-cached-peer-again"
    repeated = await h.service.execute(h.capability.token, json.dumps(payload).encode())
    assert repeated == result
    assert len(h.adapter.requests) == expected_calls


@pytest.mark.anyio
@pytest.mark.parametrize("enabled", [False, True])
async def test_running_duplicate_waits_then_replays_or_reports_completed_duplicate(
    tmp_path,
    monkeypatch,
    enabled,
):
    registry, control, attempt, capability, service, adapter = _service(tmp_path)
    lineage = registry.get_epoch(attempt.epoch_id).lineage_id
    registry._connection.execute(
        "UPDATE lineages SET result_artifact_read_enabled = ? WHERE id = ?",
        (int(enabled), lineage),
    )
    started, release = anyio.Event(), anyio.Event()
    original = adapter.execute

    async def delayed(request):
        started.set()
        await release.wait()
        return await original(request)

    monkeypatch.setattr(adapter, "execute", delayed)
    app = GatewayProxyAsgiApp(service, GatewayProxyLimits(65536, 8, 16384))
    results = []

    async def submit(key):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://runtime") as http:
            payload = json.loads(_request(attempt))
            payload["idempotency_key"] = key
            results.append(
                await http.post(
                    "/v1/operations",
                    json=payload,
                    headers={"Authorization": f"Bearer {capability.token}"},
                )
            )

    try:
        with anyio.fail_after(4):
            async with anyio.create_task_group() as group:
                group.start_soon(submit, "owner")
                await started.wait()
                group.start_soon(submit, "follower")
                await anyio.sleep(0.35)
                assert not results
                release.set()
        assert [r.status_code for r in results] == [200, 400 if enabled else 200]
        assert len(adapter.requests) == 1
        if enabled:
            assert results[1].json()["recovery"][0]["tool"] == "result-artifact-read"
        else:
            assert results[0].json() == results[1].json()
    finally:
        control.close()
        registry.close()
