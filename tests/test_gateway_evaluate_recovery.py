"""Restart recovery through the authenticated HTTP API and real Agate adapters."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from copy import deepcopy
from datetime import timedelta
from pathlib import Path

import anyio
import pytest
from atrex_gateway_client import build_eval_request_from_content
from httpx import ASGITransport, AsyncClient
from test_agate_gateway_adapter import (
    CapturingBuilder,
    FakeAgateClient,
    FakeGatewayError,
    _contract,
    _successful_job,
)
from test_agent_abba import FakeContexts, NativeFakeAgate
from test_gateway_abba import _abba_value
from test_gateway_proxy import NOW_DATETIME, _request, _service
from test_gateway_report_barrier import _report

from atrex_runtime.artifacts.local import LocalArtifactStore
from atrex_runtime.domain.errors import InfrastructureError
from atrex_runtime.domain.models import Dsl
from atrex_runtime.gateway import GatewayProxyAsgiApp, GatewayProxyLimits, SqliteGatewayControl
from atrex_runtime.gateway.agate import AgateGatewayAdapter, SqliteAgateJobStore
from atrex_runtime.gateway.agent_abba import AgentAbbaGatewayAdapter
from atrex_runtime.gateway.contract import AgateEvaluationContext
from atrex_runtime.gateway.proxy import GatewayProxyService


class RecoveryClient:
    """Control the old jobs independently of new submissions and track every RPC."""

    def __init__(self, abba):
        self.backend = NativeFakeAgate() if abba else FakeAgateClient(_successful_job())
        self.submissions = []
        self.fetches = []
        self.cancellations = []
        self.overrides = {}
        self.by_key = {}
        self.crash_after_acceptance = False

    def submit_job(self, kind, request):
        key = request['idempotency_key']
        if key not in self.by_key:
            accepted = self.backend.submit_job(kind, request)
            self.by_key[key] = accepted
            self.submissions.append((kind, deepcopy(request), accepted['job_id']))
        if self.crash_after_acceptance:
            raise OSError('connection lost after accepting replacement')
        return self.by_key[key]

    def get_job(self, job_id, wait=False, timeout=30, include_spec=False):
        self.fetches.append((job_id, wait))
        if job_id in self.overrides:
            value = self.overrides[job_id]
            if isinstance(value, Exception):
                raise value
            return {'job_id': job_id, **value}
        if isinstance(self.backend, NativeFakeAgate) and not wait:
            return deepcopy(self.backend.jobs[job_id])
        return self.backend.get_job(job_id, wait=wait, timeout=timeout)

    def cancel_job(self, job_id):
        self.cancellations.append(job_id)
        self.overrides[job_id] = {'status': 'cancelled', 'result': None}
        return {'job_id': job_id, 'status': 'cancelled'}


class Scenario:
    def __init__(self, root: Path, abba: bool):
        self.root = root
        self.abba = abba
        self.registry, self.control, self.attempt, self.capability, _, _ = _service(root)
        self.jobs = SqliteAgateJobStore(root / 'agate-jobs.sqlite')
        self.client = RecoveryClient(abba)
        self.request = _abba_value(self.attempt) if abba else json.loads(_request(self.attempt))
        self._service()

    def _service(self):
        artifacts = LocalArtifactStore(self.root / 'artifacts')
        contexts = FakeContexts(
            AgateEvaluationContext('vector_add', 'H20', Dsl.TRITON, _contract())
        )
        adapter = AgateGatewayAdapter(
            self.client, CapturingBuilder(), contexts, self.jobs, wait_timeout_s=1200,
        )
        if self.abba:
            adapter = AgentAbbaGatewayAdapter(
                adapter, self.client, contexts, artifacts, None, build_eval_request_from_content,
                jobs=self.jobs, wait_timeout_s=90,
            )
        self.service = GatewayProxyService(
            self.control, artifacts, adapter, GatewayProxyLimits(65536, 8, 16384), self.registry,
            clock=lambda: NOW_DATETIME,
        )

    def restart(self, *, legacy=False):
        self.control.close()
        self.jobs.close()
        if legacy:
            with sqlite3.connect(self.root / 'gateway.sqlite') as db:
                for column in ('execution_attempt_id', 'execution_generation', 'execution_key'):
                    db.execute(f'ALTER TABLE gateway_evaluate_tasks DROP COLUMN {column}')
                db.execute("UPDATE metadata SET value = 14 WHERE key = 'schema_version'")
        self.control = SqliteGatewayControl(
            self.root / 'gateway.sqlite', self.registry, signing_key=b'p' * 32,
            clock=lambda: NOW_DATETIME,
        )
        self.jobs = SqliteAgateJobStore(self.root / 'agate-jobs.sqlite')
        self._service()

    async def post(self, request=None, *, service=None, path='/v1/operations'):
        app = GatewayProxyAsgiApp(service or self.service, GatewayProxyLimits(65536, 8, 16384))
        async with AsyncClient(transport=ASGITransport(app=app), base_url='http://runtime') as http:
            return await http.post(
                path, json=request or self.request,
                headers={'Authorization': f'Bearer {self.capability.token}'},
            )

    async def interrupt(self, monkeypatch):
        def fail(*_args, **_kwargs):
            raise InfrastructureError('simulated Runtime exit before result persistence')

        with monkeypatch.context() as patch:
            patch.setattr(self.service, '_store_gateway_result', fail)
            response = await self.post()
            assert response.status_code == 503, response.text
            assert 'simulated Runtime exit' in response.text
        assert len(self.client.submissions) == 2
        assert len(self.jobs.list_owned(self.attempt.id)) == 2
        self.inject_dead_call()
        self.client.fetches.clear()

    def inject_dead_call(self):
        # A hard process exit does not run operation_execution's finally block.
        with sqlite3.connect(self.root / 'gateway.sqlite') as db:
            db.execute(
                """INSERT INTO gateway_active_calls
                   SELECT 'dead-http-call', attempt_id, recovery_generation, idempotency_key,
                          operation, created_at FROM gateway_operations
                   WHERE idempotency_key = ?""", (self.request['idempotency_key'],),
            )

    def rows(self, table):
        with sqlite3.connect(self.root / 'gateway.sqlite') as db:
            db.row_factory = sqlite3.Row
            return [dict(row) for row in db.execute(f'SELECT * FROM {table}')]

    def assert_completed(self):
        tasks = self.rows('gateway_evaluate_tasks')
        assert len(tasks) == 1
        assert tasks[0]['status'] == 'completed'
        assert tasks[0]['result_artifact_digest']
        assert not self.rows('gateway_active_calls')
        assert len(self.rows('gateway_evaluations')) == (0 if self.abba else 1)
        points = self.rows('gateway_measurements')
        assert sum(row['shape_id'] is not None for row in points) == (4 if self.abba else 2)

    def close(self):
        self.jobs.close()
        self.control.close()
        self.registry.close()


def reconnect_service(scenario):
    control = SqliteGatewayControl(
        scenario.root / 'gateway.sqlite', scenario.registry, signing_key=b'p' * 32,
        clock=lambda: NOW_DATETIME,
    )
    service = GatewayProxyService(
        control, LocalArtifactStore(scenario.root / 'artifacts'), scenario.service._adapter,
        GatewayProxyLimits(65536, 8, 16384), scenario.registry, clock=lambda: NOW_DATETIME,
    )
    return control, service


@pytest.mark.anyio
@pytest.mark.parametrize('mode', ['full', 'abba', 'correctness_only'])
@pytest.mark.parametrize('owner_fails', [False, True])
async def test_identical_http_reconnect_waits_and_returns_original_result(
    tmp_path, monkeypatch, mode, owner_fails,
):
    scenario = Scenario(tmp_path, mode == 'abba')
    second = None
    try:
        if mode == 'correctness_only':
            scenario.request['mode'] = mode
            scenario.request.pop('latency_prediction', None)
        started, release, waiting = anyio.Event(), anyio.Event(), anyio.Event()
        original = scenario.service._adapter.execute
        calls = []

        async def blocked(request):
            calls.append(request)
            result = await original(request)
            if len(calls) == 1:
                started.set()
                await release.wait()
                if owner_fails:
                    raise InfrastructureError('original HTTP executor exited before commit')
            return result

        monkeypatch.setattr(scenario.service._adapter, 'execute', blocked)
        second, service = reconnect_service(scenario)
        original_lease = second.evaluate_request_execution

        @contextmanager
        def observe_reconnect(authorization):
            with original_lease(authorization) as acquired:
                if not acquired:
                    waiting.set()
                yield acquired

        monkeypatch.setattr(second, 'evaluate_request_execution', observe_reconnect)
        responses = {}

        async def post(label, target):
            responses[label] = await scenario.post(service=target)

        with anyio.fail_after(10):
            async with anyio.create_task_group() as group:
                group.start_soon(post, 'original', scenario.service)
                await started.wait()
                jobs = {item[2] for item in scenario.client.submissions}
                group.start_soon(post, 'reconnect', service)
                await waiting.wait()
                assert not responses  # No intermediate running/duplicate result reaches Core.
                assert len(calls) == 1
                assert len(scenario.rows('gateway_active_calls')) == 1
                assert scenario.rows('gateway_capabilities')[0]['used_calls'] == 1
                release.set()
        assert responses['original'].status_code == (503 if owner_fails else 200)
        response = responses['reconnect']
        assert response.status_code == 200, response.text
        assert response.json()['result']['correct'] is True
        if not owner_fails:
            assert response.json() == responses['original'].json()
            assert len(calls) == 1
        assert {item[2] for item in scenario.client.submissions} == jobs
        assert not scenario.client.cancellations
        assert not scenario.rows('gateway_active_calls')
        replay = await scenario.post(service=service)
        assert replay.json() == response.json()
        # A new key is a new submission, so completed semantic duplicates stay rejected.
        if mode != 'correctness_only':
            duplicate = await scenario.post(
                {**scenario.request, 'idempotency_key': 'completed-new-submission'},
                service=service,
            )
            assert duplicate.status_code == 400
            assert duplicate.json()['error'] == 'duplicate_gateway_task'
            assert duplicate.json()['previous_result_artifact_digest'] == (
                response.json()['result_artifact_digest']
            )
            scenario.assert_completed()
    finally:
        if second is not None:
            second.close()
        scenario.close()


@pytest.mark.anyio
@pytest.mark.parametrize('abba', [False, True])
async def test_cancelling_a_waiting_reconnect_does_not_cancel_the_original(
    tmp_path, monkeypatch, abba,
):
    scenario = Scenario(tmp_path, abba)
    second = None
    try:
        started, release, waiting, waiter_cancelled = (
            anyio.Event(), anyio.Event(), anyio.Event(), anyio.Event()
        )
        original = scenario.service._adapter.execute

        async def blocked(request):
            result = await original(request)
            started.set()
            await release.wait()
            return result

        monkeypatch.setattr(scenario.service._adapter, 'execute', blocked)
        second, service = reconnect_service(scenario)
        original_lease = second.evaluate_request_execution

        @contextmanager
        def observe_reconnect(authorization):
            with original_lease(authorization) as acquired:
                if not acquired:
                    waiting.set()
                yield acquired

        monkeypatch.setattr(second, 'evaluate_request_execution', observe_reconnect)
        responses = []

        async def first():
            responses.append(await scenario.post())

        async def reconnect(*, task_status=anyio.TASK_STATUS_IGNORED):
            with anyio.CancelScope() as scope:
                task_status.started(scope)
                await scenario.post(service=service)
            waiter_cancelled.set()

        with anyio.fail_after(10):
            async with anyio.create_task_group() as group:
                group.start_soon(first)
                await started.wait()
                scope = await group.start(reconnect)
                await waiting.wait()
                scope.cancel()
                await waiter_cancelled.wait()
                assert not responses
                assert len(scenario.rows('gateway_active_calls')) == 1
                assert not scenario.client.cancellations
                release.set()
        assert responses[0].status_code == 200, responses[0].text
        replay = await scenario.post(service=service)
        assert replay.json() == responses[0].json()
        assert len(scenario.client.submissions) == 2
        scenario.assert_completed()
    finally:
        if second is not None:
            second.close()
        scenario.close()


@pytest.mark.anyio
async def test_reused_http_key_with_changed_content_is_rejected_without_waiting(
    tmp_path, monkeypatch,
):
    scenario = Scenario(tmp_path, False)
    try:
        started, release = anyio.Event(), anyio.Event()
        original = scenario.service._adapter.execute

        async def blocked(request):
            started.set()
            await release.wait()
            return await original(request)

        monkeypatch.setattr(scenario.service._adapter, 'execute', blocked)
        responses = []

        async def first():
            responses.append(await scenario.post())

        with anyio.fail_after(10):
            async with anyio.create_task_group() as group:
                group.start_soon(first)
                await started.wait()
                changed = {**scenario.request, 'latency_prediction': 'improved'}
                response = await scenario.post(changed)
                assert response.status_code == 409, response.text
                assert 'reused for a different request' in response.json()['detail']
                assert not responses
                release.set()
        assert responses[0].status_code == 200
        assert len(scenario.client.submissions) == 2
        scenario.assert_completed()
    finally:
        scenario.close()


@pytest.mark.anyio
@pytest.mark.parametrize('abba', [False, True])
@pytest.mark.parametrize('legacy', [False, True])
async def test_restart_gets_finished_jobs_and_commits_result_without_resubmitting(
    tmp_path, monkeypatch, abba, legacy,
):
    scenario = Scenario(tmp_path, abba)
    try:
        await scenario.interrupt(monkeypatch)
        old_jobs = {item[2] for item in scenario.client.submissions}
        scenario.restart(legacy=legacy)
        # Agents often change the HTTP key when retrying a stuck measurement.
        retry = {**scenario.request, 'idempotency_key': 'retry-after-runtime-restart'}
        response = await scenario.post(retry)
        assert response.status_code == 200, response.text
        assert response.json()['result']['correct'] is True
        assert len(scenario.client.submissions) == 2
        assert set(scenario.client.fetches) == {(job, False) for job in old_jobs}
        scenario.assert_completed()
        task = scenario.rows('gateway_evaluate_tasks')[0]
        assert task['execution_key'] == scenario.request['idempotency_key']
        assert task['idempotency_key'] == retry['idempotency_key']
        # Same-key replay is immutable and performs no additional Agate calls.
        fetched = list(scenario.client.fetches)
        replay = await scenario.post(retry)
        assert replay.json() == response.json()
        assert scenario.client.fetches == fetched
        report = await scenario.post(
            json.loads(_report(scenario.attempt)), path='/v1/runtime/queries',
        )
        assert report.status_code == 200, report.text
    finally:
        scenario.close()


@pytest.mark.anyio
@pytest.mark.parametrize('abba', [False, True])
@pytest.mark.parametrize(
    'state', ['missing', 'gone', 'running', 'queued', 'cancelled', 'failed', 'empty'],
)
async def test_only_unfinished_or_missing_batch_is_replaced(tmp_path, monkeypatch, abba, state):
    scenario = Scenario(tmp_path, abba)
    try:
        await scenario.interrupt(monkeypatch)
        stale, completed = [item[2] for item in scenario.client.submissions]
        scenario.client.overrides[stale] = (
            FakeGatewayError(404 if state == 'missing' else 410, 'not_found', {})
            if state in {'missing', 'gone'}
            else {'status': 'succeeded' if state == 'empty' else state, 'result': None}
        )
        scenario.restart()
        response = await scenario.post()
        assert response.status_code == 200, response.text
        assert response.json()['result']['correct'] is True
        assert len(scenario.client.submissions) == 3
        assert (completed, False) in scenario.client.fetches
        assert (stale, False) in scenario.client.fetches
        _, new_request, replacement = scenario.client.submissions[-1]
        assert new_request['idempotency_key'].startswith('orphan-retry:')
        assert (replacement, True) in scenario.client.fetches
        assert scenario.client.cancellations == ([stale] if state in {'running', 'queued'} else [])
        assert {binding.job_id for binding in scenario.jobs.list_owned(scenario.attempt.id)} == {
            completed, replacement,
        }
        scenario.assert_completed()
    finally:
        scenario.close()


@pytest.mark.anyio
@pytest.mark.parametrize('abba', [False, True])
@pytest.mark.parametrize('status', [401, 503])
async def test_get_failure_preserves_old_job_identity_for_next_restart(
    tmp_path, monkeypatch, abba, status,
):
    scenario = Scenario(tmp_path, abba)
    try:
        await scenario.interrupt(monkeypatch)
        old_job = scenario.client.submissions[0][2]
        scenario.client.overrides[old_job] = FakeGatewayError(status, 'auth_or_transport', {})
        scenario.restart()
        retry = {**scenario.request, 'idempotency_key': 'first-recovery'}
        response = await scenario.post(retry)
        assert response.status_code != 200
        assert len(scenario.client.submissions) == 2
        assert not scenario.client.cancellations
        assert scenario.rows('gateway_evaluate_tasks')[0]['status'] == 'running'
        scenario.restart()
        del scenario.client.overrides[old_job]
        retry['idempotency_key'] = 'second-recovery'
        response = await scenario.post(retry)
        assert response.status_code == 200, response.text
        assert len(scenario.client.submissions) == 2
        scenario.assert_completed()
    finally:
        scenario.close()


@pytest.mark.anyio
@pytest.mark.parametrize('abba', [False, True])
async def test_crash_after_replacement_acceptance_reuses_same_submission_key(
    tmp_path, monkeypatch, abba,
):
    scenario = Scenario(tmp_path, abba)
    try:
        await scenario.interrupt(monkeypatch)
        stale = scenario.client.submissions[0][2]
        scenario.client.overrides[stale] = {'status': 'running', 'result': None}
        scenario.restart()
        scenario.client.crash_after_acceptance = True
        response = await scenario.post()
        assert response.status_code == 503, response.text
        assert len(scenario.client.submissions) == 3
        assert stale in {row.job_id for row in scenario.jobs.list_owned(scenario.attempt.id)}
        scenario.restart()
        scenario.client.crash_after_acceptance = False
        retry = {**scenario.request, 'idempotency_key': 'retry-after-second-crash'}
        response = await scenario.post(retry)
        assert response.status_code == 200, response.text
        assert len(scenario.client.submissions) == 3
        scenario.assert_completed()
    finally:
        scenario.close()


@pytest.mark.anyio
@pytest.mark.parametrize('recovering', [False, True])
async def test_live_executor_blocks_concurrent_retry_from_another_control_instance(
    tmp_path, monkeypatch, recovering,
):
    scenario = Scenario(tmp_path, False)
    second = None
    try:
        if recovering:
            await scenario.interrupt(monkeypatch)
            scenario.restart()
        started = anyio.Event()
        release = anyio.Event()
        original = scenario.service._adapter.execute

        async def blocked(request):
            started.set()
            await release.wait()
            return await original(request)

        monkeypatch.setattr(scenario.service._adapter, 'execute', blocked)
        second = SqliteGatewayControl(
            tmp_path / 'gateway.sqlite', scenario.registry, signing_key=b'p' * 32,
            clock=lambda: NOW_DATETIME,
        )
        service = GatewayProxyService(
            second, LocalArtifactStore(tmp_path / 'artifacts'), scenario.service._adapter,
            GatewayProxyLimits(65536, 8, 16384), scenario.registry, clock=lambda: NOW_DATETIME,
        )
        completed = []

        async def first():
            completed.append(await scenario.post())

        with anyio.fail_after(10):
            async with anyio.create_task_group() as group:
                group.start_soon(first)
                await started.wait()
                retry = {**scenario.request, 'idempotency_key': 'concurrent-retry'}
                response = await scenario.post(retry, service=service)
                assert response.json()['error'] == 'duplicate_gateway_task'
                assert len(scenario.rows('gateway_active_calls')) == 1
                assert not scenario.client.cancellations
                release.set()
        assert completed[0].status_code == 200, completed[0].text
        assert len(scenario.client.submissions) == 2
        scenario.assert_completed()
    finally:
        if second is not None:
            second.close()
        scenario.close()


@pytest.mark.anyio
@pytest.mark.parametrize('abba', [False, True])
@pytest.mark.parametrize('stage', ['after_private_result', 'after_response'])
async def test_restart_finishes_partially_committed_result(tmp_path, monkeypatch, abba, stage):
    scenario = Scenario(tmp_path, abba)
    try:
        def fail(*_args, **_kwargs):
            raise InfrastructureError('simulated crash during DB persistence')

        with monkeypatch.context() as patch:
            if stage == 'after_response':
                patch.setattr(scenario.control, 'complete_evaluate_task', fail)
            else:
                patch.setattr(scenario.service, '_store_result_artifact', fail)
            response = await scenario.post()
            assert response.status_code == 503, response.text
            assert 'simulated crash during DB persistence' in response.text
        scenario.inject_dead_call()
        assert scenario.rows('gateway_evaluate_tasks')[0]['status'] == 'running'
        scenario.restart()
        # Wall time changes across a real restart; persisted measurement times must not.
        monkeypatch.setattr(scenario.control, '_clock', lambda: NOW_DATETIME + timedelta(minutes=1))
        scenario.client.fetches.clear()
        response = await scenario.post()
        assert response.status_code == 200, response.text
        assert len(scenario.client.submissions) == 2
        if stage == 'after_response':
            assert not scenario.client.fetches
        scenario.assert_completed()
    finally:
        scenario.close()


def test_execution_lease_is_exclusive_and_released_after_process_death(tmp_path):
    import subprocess
    import sys

    from atrex_runtime.gateway.execution_lock import execution_lock

    code = (
        'import sys; from pathlib import Path; '
        'from atrex_runtime.gateway.execution_lock import execution_lock\n'
        'with execution_lock(Path(sys.argv[1]), "task") as acquired:\n'
        ' print(acquired, flush=True)\n'
        ' sys.stdin.read()\n'
    )
    with subprocess.Popen(
        [sys.executable, '-c', code, str(tmp_path)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    ) as process:
        try:
            assert process.stdout.readline().strip() == 'True'
            with execution_lock(tmp_path, 'task') as acquired:
                assert not acquired
            process.kill()
            process.wait(timeout=10)
            with execution_lock(tmp_path, 'task') as acquired:
                assert acquired
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=10)
