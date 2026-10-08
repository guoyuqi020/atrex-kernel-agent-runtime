"""Disabling Experiment tools must not deadlock reuse of visible measured kernels."""

from __future__ import annotations

import base64
import json
from collections.abc import AsyncIterator
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from typing import Any, cast

import pytest
from conftest import digest
from test_attempt_report import _value
from test_gateway_adoption_nomination import _candidate, _submit
from test_gateway_control import _insert_attempt
from test_gateway_finalization import FakeClient, FakeContexts, FakeEvents, _builder
from test_gateway_journal_adoption import _History
from test_gateway_journal_adoption import history as history
from test_gateway_proxy import NOW_DATETIME, FakeGatewayAdapter, _request

from atrex_runtime.artifacts.local import LocalArtifactStore
from atrex_runtime.domain.errors import DuplicateGatewayTaskError
from atrex_runtime.domain.ids import new_attempt_id, parse_artifact_digest
from atrex_runtime.domain.models import Dsl
from atrex_runtime.gateway import GatewayProxyLimits, GatewayProxyService, SqliteGatewayControl
from atrex_runtime.gateway.control import BootstrapGatewaySubject, GatewayCapabilityPolicy
from atrex_runtime.gateway.control_models import GatewayEvaluationSource, GatewayOperation
from atrex_runtime.gateway.finalization import (
    AgateAuthoritativeCandidateEvaluator,
    BootstrapEvaluationStage,
)
from atrex_runtime.gateway.protocol import EvaluationV2
from atrex_runtime.gateway.proxy import GatewayAdapterResult
from atrex_runtime.registry.sqlite import SqliteRegistry


@pytest.fixture
async def peer_history(tmp_path: Path) -> AsyncIterator[_History]:
    registry = SqliteRegistry(tmp_path / "registry.sqlite")
    control = SqliteGatewayControl(
        tmp_path / "gateway.sqlite", registry, signing_key=b"p" * 32, clock=lambda: NOW_DATETIME
    )
    adapter = FakeGatewayAdapter(
        GatewayAdapterResult(
            "completed",
            {"correct": True, "latency_us": 12.0, "latency_us_by_shape": {"0": 12.0}},
            evaluation=EvaluationV2(correct=True, latency_us=12.0),
        )
    )
    service = GatewayProxyService(
        control,
        LocalArtifactStore(tmp_path / "artifacts"),
        adapter,
        GatewayProxyLimits(64 * 1024, 8, 16 * 1024),
        registry,
        clock=lambda: NOW_DATETIME,
    )
    try:
        peer = _insert_attempt(registry, trajectories=2)
        current = replace(peer, id=new_attempt_id(), trajectory_ordinal=2)
        registry.insert_attempt(current)
        policy = GatewayCapabilityPolicy(
            frozenset(GatewayOperation), 4, NOW_DATETIME + timedelta(hours=1)
        )
        peer_capability = control.issue(peer.id, policy)
        current_capability = control.issue(current.id, policy)
        payload = json.loads(_request(peer))
        payload["candidate"] = _candidate()
        evaluated = await service.execute(peer_capability.token, json.dumps(payload).encode())
        yield _History(
            registry,
            control,
            current,
            current_capability,
            service,
            adapter,
            evaluated,
            evaluated,
            peer,
            len(adapter.requests),
        )
    finally:
        control.close()
        registry.close()


async def _module_report(history: _History, modules: tuple[str, ...]) -> dict[str, Any]:
    lineage_id = history.registry.get_epoch(history.current.epoch_id).lineage_id
    history.registry._connection.execute(
        "UPDATE lineages SET tool_modules_json = ? WHERE id = ?",
        (json.dumps(modules), lineage_id),
    )
    if "directions" in modules:
        direction = await history.start()
        await history.service.execute(
            history.capability.token,
            json.dumps(
                {
                    "schema_version": 2,
                    "attempt_id": history.current.id,
                    "idempotency_key": "close-module-independent-direction",
                    "operation": "direction_update",
                    "request": {
                        "action": "complete",
                        "direction_id": direction,
                        "hypothesis_status": "unresolved",
                        "analysis": "Reuse the exact visible measured kernel; no new measurement.",
                    },
                }
            ).encode(),
            operation_scope="journal",
        )
    report: dict[str, Any] = _value(history.current.id)
    report.update(
        tool_modules=list(modules),
        experiments=[],
        direction_events=list(history.control.list_direction_events(history.current.id)),
        profile_evidence=None,
    )
    report["findings"][0]["supporting_experiment_ids"] = []
    return report


@pytest.mark.anyio
@pytest.mark.parametrize("modules", [(), ("directions",)])
@pytest.mark.parametrize("independent", [False, True])
@pytest.mark.parametrize("service_defaults", [False, True])
async def test_disabled_experiment_reuses_visible_evidence_through_finalization(
    history: _History,
    tmp_path: Path,
    modules: tuple[str, ...],
    independent: bool,
    service_defaults: bool,
) -> None:
    report = await _module_report(history, modules)
    configured_modules = ("directions", "experiments")
    if service_defaults:
        lineage_id = history.registry.get_epoch(history.current.epoch_id).lineage_id
        history.registry._connection.execute(
            "UPDATE lineages SET tool_modules_json = NULL WHERE id = ?", (lineage_id,)
        )
        configured_modules = modules
        history.service = GatewayProxyService(
            history.control,
            LocalArtifactStore(tmp_path / "artifacts"),
            history.adapter,
            GatewayProxyLimits(64 * 1024, 8, 16 * 1024),
            history.registry,
            tool_modules=modules,
            clock=lambda: NOW_DATETIME,
        )
    original_evaluations = history.control.list_evaluations(history.historical_attempt.id)
    original_trials = history.control.list_kernel_trials((history.historical_attempt.id,))
    request = json.loads(_request(history.current))
    request.update(candidate=_candidate(), idempotency_key="redundant-full-evaluate")

    # Reproduce the deadlock: the exact candidate's full Evaluate is already deduplicated.
    with pytest.raises(DuplicateGatewayTaskError) as duplicate:
        await history.service.execute(history.capability.token, json.dumps(request).encode())
    assert duplicate.value.previous_result_artifact_digest == history.after.result_artifact_digest
    assert (await _submit(history, report)).result["status"] == "registered"
    assert history.control.list_experiments(history.current.id) == ()
    assert history.control.list_evaluations(history.current.id) == ()
    assert history.control.list_measurements((history.current.id,)) == ()
    assert history.after.kernel_artifact_digest is not None
    candidate_digest = parse_artifact_digest(history.after.kernel_artifact_digest)
    resolved = history.control.find_candidate_evaluation(
        history.current.id, candidate_digest, default_tool_modules=configured_modules
    )
    assert resolved == original_evaluations[-1]

    client, events = FakeClient(), FakeEvents()
    finalizer = AgateAuthoritativeCandidateEvaluator(
        client,  # type: ignore[arg-type]
        _builder,
        FakeContexts(),
        LocalArtifactStore(tmp_path / "artifacts"),
        history.control,
        events,
        wait_timeout_s=100,
        bootstrap_stages=(BootstrapEvaluationStage(1),),
        tool_modules=configured_modules,
        clock=lambda: NOW_DATETIME,
    )
    outcome = await finalizer.finalize(
        history.current.id,
        candidate_digest,
        nominated_gateway_result_digest=resolved.gateway_result_digest,
        independent_evaluate=independent,
    )
    assert outcome.correct
    assert outcome.latency_us == (7.5 if independent else 12.0)
    assert len(client.submitted) == (1 if independent else 0)
    assert len(history.adapter.requests) == history.initial_adapter_requests
    assert history.control.list_evaluations(history.historical_attempt.id) == original_evaluations
    assert history.control.list_kernel_trials((history.historical_attempt.id,)) == original_trials
    assert [row.source for row in history.control.list_evaluations(history.current.id)] == (
        [GatewayEvaluationSource.RUNTIME_FINAL] if independent else []
    )
    provenance = cast(dict[str, Any], events.values[0][2])
    assert provenance["evaluation_attempt_id"] == history.historical_attempt.id


@pytest.mark.anyio
@pytest.mark.parametrize("broadcast", [False, True])
async def test_running_peer_trajectory_evidence_still_requires_visibility(
    peer_history: _History, broadcast: bool
) -> None:
    history = peer_history
    assert history.current.trajectory_ordinal != history.historical_attempt.trajectory_ordinal
    report = await _module_report(history, ("directions",))
    if broadcast:
        lineage_id = history.registry.get_epoch(history.current.epoch_id).lineage_id
        history.registry._connection.execute(
            "UPDATE lineages SET trajectory_visibility = 'broadcast' WHERE id = ?", (lineage_id,)
        )
        assert (await _submit(history, report)).result["status"] == "registered"
    else:
        with pytest.raises(ValueError, match="candidate_ready requires"):
            await _submit(history, report)
    assert history.control.list_evaluations(history.current.id) == ()
    assert len(history.adapter.requests) == history.initial_adapter_requests


@pytest.mark.anyio
@pytest.mark.parametrize(
    "history", ["incorrect", "custom", "correctness_only", "profile", "abba"], indirect=True
)
@pytest.mark.parametrize("modules", [(), ("directions",)])
async def test_module_independent_reuse_rejects_ineligible_evidence(
    history: _History, modules: tuple[str, ...]
) -> None:
    report = await _module_report(history, modules)
    with pytest.raises(ValueError, match="candidate_ready requires"):
        await _submit(history, report)
    assert history.control.list_evaluations(history.current.id) == ()
    assert len(history.adapter.requests) == history.initial_adapter_requests


@pytest.mark.anyio
async def test_module_independent_reuse_requires_exact_kernel_and_result(history: _History) -> None:
    report = await _module_report(history, ())
    assert history.after.kernel_artifact_digest is not None
    candidate_digest = parse_artifact_digest(history.after.kernel_artifact_digest)
    assert (
        history.control.find_candidate_evaluation(
            history.current.id, candidate_digest, gateway_result_digest=digest("different-result")
        )
        is None
    )
    payload = json.loads(_request(history.current))
    payload.update(operation="attempt_report", report=report, idempotency_key="changed-kernel")
    payload.pop("latency_prediction")
    payload["candidate"]["files"][0]["content_base64"] = base64.b64encode(
        b"def kernel(): return 'unmeasured'\n"
    ).decode()
    with pytest.raises(ValueError, match="candidate_ready requires"):
        await history.service.execute(history.capability.token, json.dumps(payload).encode())


@pytest.mark.anyio
async def test_own_failed_evaluate_cannot_be_hidden_by_visible_success(history: _History) -> None:
    report = await _module_report(history, ("directions",))
    assert history.after.kernel_artifact_digest is not None
    candidate_digest = parse_artifact_digest(history.after.kernel_artifact_digest)
    failed = history.control.record_evaluation(
        history.current.id,
        source=GatewayEvaluationSource.AGENT,
        idempotency_key="own-later-failure",
        kernel_artifact_digest=candidate_digest,
        gateway_result_digest=digest("failed-result"),
        correct=False,
        latency_us=None,
        agate_job_id=None,
    )
    assert history.control.find_candidate_evaluation(history.current.id, candidate_digest) == failed
    with pytest.raises(ValueError, match="evaluate reported incorrect"):
        await _submit(history, report)


@pytest.mark.anyio
async def test_recovery_cannot_reuse_success_superseded_by_failed_generation(
    history: _History,
) -> None:
    payload = json.loads(_request(history.current))
    payload["candidate"]["files"][0]["content_base64"] = base64.b64encode(
        b"def kernel(): return 'recovered candidate'\n"
    ).decode()
    evaluated = await history.service.execute(
        history.capability.token, json.dumps(payload).encode()
    )
    assert evaluated.kernel_artifact_digest is not None
    candidate_digest = parse_artifact_digest(evaluated.kernel_artifact_digest)
    history.registry.record_infrastructure_failure(history.current.id, "first interruption")
    history.registry.retry_attempt(history.current.id)
    failed = history.control.record_evaluation(
        history.current.id,
        source=GatewayEvaluationSource.AGENT,
        idempotency_key="failed-in-generation-one",
        kernel_artifact_digest=candidate_digest,
        gateway_result_digest=digest("generation-one-failure"),
        correct=False,
        latency_us=None,
        agate_job_id=None,
    )
    assert failed.recovery_generation == 1
    history.registry.record_infrastructure_failure(history.current.id, "second interruption")
    history.registry.retry_attempt(history.current.id)
    history.capability = history.control.issue(
        history.current.id,
        GatewayCapabilityPolicy(frozenset(GatewayOperation), 4, NOW_DATETIME + timedelta(hours=1)),
    )
    assert history.capability.recovery_generation == 2
    report = await _module_report(history, ())
    resolved = history.control.find_candidate_evaluation(history.current.id, candidate_digest)
    assert resolved is None or not resolved.correct
    payload.update(operation="attempt_report", report=report, idempotency_key="stale-success")
    payload.pop("latency_prediction")
    with pytest.raises(ValueError, match="candidate_ready requires"):
        await history.service.execute(history.capability.token, json.dumps(payload).encode())


@pytest.mark.anyio
@pytest.mark.parametrize(
    "field", ["operator", "hardware_target", "dsl", "evaluation_contract_digest"]
)
async def test_module_independent_reuse_requires_matching_evaluation_context(
    history: _History, field: str
) -> None:
    report = await _module_report(history, ())
    epoch = history.registry.get_epoch(history.current.epoch_id)
    lineage = history.registry.get_lineage(epoch.lineage_id)
    campaign = history.registry.get_campaign(lineage.campaign_id)
    subject = BootstrapGatewaySubject(
        new_attempt_id(),
        campaign.id,
        lineage.id,
        epoch.id,
        history.current.kernel_agent_revision_id,
        campaign.operator,
        campaign.hardware_target,
        lineage.dsl,
        campaign.evaluation_contract_digest,
        digest("seed"),
        digest("evidence"),
        NOW_DATETIME,
    )
    changes: dict[str, Any] = {
        "operator": "another_operator",
        "hardware_target": "another_gpu",
        "dsl": Dsl.CUDA if lineage.dsl != Dsl.CUDA else Dsl.TRITON,
        "evaluation_contract_digest": digest("another-contract"),
    }
    subject = replace(subject, **{field: changes[field]})
    capability = history.control.issue_bootstrap(
        subject,
        GatewayCapabilityPolicy(frozenset(GatewayOperation), 4, NOW_DATETIME + timedelta(hours=1)),
    )
    payload = json.loads(_request(history.current))
    payload["attempt_id"] = subject.attempt_id
    payload["candidate"]["files"][0]["content_base64"] = base64.b64encode(
        b"def kernel(): return 'only measured in incompatible context'\n"
    ).decode()
    evaluated = await history.service.execute(capability.token, json.dumps(payload).encode())
    assert evaluated.kernel_artifact_digest is not None
    _, visible = history.control.visible_kernel_trial_attempt_ids(history.current.id)
    assert subject.attempt_id in visible
    assert (
        history.control.find_candidate_evaluation(
            history.current.id, parse_artifact_digest(evaluated.kernel_artifact_digest)
        )
        is None
    )
    payload.update(
        attempt_id=history.current.id,
        operation="attempt_report",
        report=report,
        idempotency_key="incompatible-nomination",
    )
    payload.pop("latency_prediction")
    with pytest.raises(ValueError, match="candidate_ready requires"):
        await history.service.execute(history.capability.token, json.dumps(payload).encode())
