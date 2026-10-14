"""OSS placement failures never become negative evidence or completed-task dedup."""

from __future__ import annotations

import base64
import json
import threading
from pathlib import Path

import pytest
import test_kernel_sources
from atrex_gateway_client import build_eval_request_from_content
from atrex_gateway_client.errors import GatewayError
from httpx import ASGITransport, AsyncClient
from test_agate_gateway_adapter import FakeAgateClient, _adapter, _contract, _successful_job
from test_agent_abba import FakeContexts, NativeFakeAgate
from test_gateway_abba import _abba_value
from test_gateway_proxy import _request, _service

from atrex_runtime.artifacts.local import LocalArtifactStore
from atrex_runtime.domain.errors import InfrastructureError
from atrex_runtime.domain.models import Dsl
from atrex_runtime.gateway import GatewayOperation, GatewayProxyAsgiApp, GatewayProxyLimits
from atrex_runtime.gateway.agate import SqliteAgateJobStore
from atrex_runtime.gateway.agent_abba import AgentAbbaGatewayAdapter
from atrex_runtime.gateway.contract import AgateEvaluationContext
from atrex_runtime.gateway.execution import submit_agate_job
from atrex_runtime.gateway.oss_client import OssAgateClient
from atrex_runtime.gateway.retrying_client import RetryingAgateClient
from atrex_runtime.gateway.source_tree import SourceTreeAgateClient

source_seed = test_kernel_sources.source_seed


def _oss_error():
    return GatewayError(400, {
        "reason": "oss_unavailable", "error_class": "validation",
        "message": "no OSS-capable cluster matches the requested hardware",
        "trace_id": "req-oss-outage",
        "details": {"input_py": "private-input", "put_url": "private-presigned-url"},
    })


class UploadClient:
    """Exercise real packing/SDK composition, with failure at /v1/uploads."""

    def __init__(self, jobs):
        self.jobs = jobs
        self.unavailable = True
        self.prepared = []
        self.uploaded = []
        self.lock = threading.Lock()

    def __getattr__(self, name):
        return getattr(self.jobs, name)

    def prepare_uploads(self, gpu, files, *, kind):
        assert kind == "eval"
        with self.lock:
            self.prepared.append((gpu, files, kind))
            reservation = len(self.prepared)
            if self.unavailable:
                raise _oss_error()
        return {
            "job_id": f"ev-reservation-{reservation}",
            "uploads": [
                {"path": f["path"], "put_url": "https://oss.example.test/put",
                 "upload_ref": f"ref-{reservation}-{i}"}
                for i, f in enumerate(files)
            ],
        }

    def upload_file(self, url, path):
        assert Path(path).read_bytes()
        self.uploaded.append(path)


def _setup(tmp_path, monkeypatch, source_seed, mode, *, retry=False):
    registry, control, attempt, capability, service, delegate = _service(tmp_path)
    jobs_client = NativeFakeAgate() if mode == "abba" else FakeAgateClient(_successful_job())
    raw = UploadClient(jobs_client)
    sleeps = []

    async def recovered_after_backoff(delay):
        # This callback executes inside the still-pending HTTP tool call.
        assert not control.list_evaluations(attempt.id)
        assert control.get_operation_artifact(
            attempt.id, request["idempotency_key"], GatewayOperation.EVALUATE,
        ) is None
        states = control._connection.execute("SELECT status FROM gateway_evaluate_tasks").fetchall()
        assert all(row["status"] == "running" for row in states)
        sleeps.append(delay)
        raw.unavailable = False

    client = SourceTreeAgateClient(OssAgateClient(
        RetryingAgateClient(raw, async_sleeper=recovered_after_backoff) if retry else raw,
    ), None)
    context = AgateEvaluationContext("vector_add", "L20D", Dsl.TRITON, _contract().model_copy(
        update={"kernel_sources": {Dsl.TRITON: source_seed.contract}},
    ))
    if mode == "abba":
        store = SqliteAgateJobStore(tmp_path / "agate-jobs.sqlite")
        adapter = AgentAbbaGatewayAdapter(
            delegate, client, FakeContexts(context), LocalArtifactStore(tmp_path / "artifacts"),
            None, build_eval_request_from_content, wait_timeout_s=90, jobs=store,
        )
        request = _abba_value(attempt)
    else:
        adapter, _, store = _adapter(tmp_path, client)
        adapter._request_builder = build_eval_request_from_content
        adapter._contexts.context = context
        request = json.loads(_request(attempt))
        if mode == "correctness_only":
            request["mode"] = mode
            request.pop("latency_prediction")
    request["candidate"]["files"] = [
        {"path": path, "content_base64": base64.b64encode(text.encode()).decode()}
        for path, text in source_seed.contract.validate_tree(source_seed.working).items()
    ]
    if mode == "abba":
        request["baseline"] = request["candidate"]
    monkeypatch.setattr(service, "_adapter", adapter)
    app = GatewayProxyAsgiApp(service, GatewayProxyLimits(65536, 8, 16384))
    return registry, control, attempt, capability, store, raw, request, app, sleeps


@pytest.mark.anyio
@pytest.mark.parametrize("mode", ["full", "correctness_only", "abba"])
@pytest.mark.parametrize("new_key", [False, True])
async def test_oss_rejection_has_no_verdict_and_identical_source_can_retry(
    tmp_path, monkeypatch, source_seed, mode, new_key,
):
    registry, control, attempt, capability, store, raw, request, app, _ = _setup(
        tmp_path, monkeypatch, source_seed, mode,
    )
    headers = {"Authorization": f"Bearer {capability.token}"}
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://runtime") as http:
            response = await http.post("/v1/operations", json=request, headers=headers)
            assert response.status_code == 503, response.text
            assert response.json()["error"] == "gateway_unavailable"
            assert "oss_unavailable" in response.text and "req-oss-outage" in response.text
            assert "private-input" not in response.text
            assert "private-presigned-url" not in response.text
            assert "candidate_rejected" not in response.text
            assert not control.list_evaluations(attempt.id)
            assert control.get_operation_artifact(
                attempt.id, request["idempotency_key"], GatewayOperation.EVALUATE,
            ) is None
            assert not control._connection.execute(
                "SELECT 1 FROM gateway_evaluate_tasks WHERE status='completed'",
            ).fetchall()
            raw.unavailable = False
            if new_key:
                request = {**request, "idempotency_key": "retry-after-oss-recovery"}
            response = await http.post("/v1/operations", json=request, headers=headers)
            assert response.status_code == 200, response.text
            assert response.json()["result"]["correct"] is True
            assert len(control.list_evaluations(attempt.id)) == (1 if mode == "full" else 0)
            assert raw.uploaded
            # A real completed measurement still has the normal duplicate guard.
            if mode != "correctness_only":
                duplicate = await http.post("/v1/operations", json={
                    **request, "idempotency_key": "repeat-completed-measurement",
                }, headers=headers)
                assert duplicate.status_code == 400, duplicate.text
                assert duplicate.json()["error"] == "duplicate_gateway_task"
    finally:
        if store is not None:
            store.close()
        control.close()
        registry.close()


@pytest.mark.anyio
@pytest.mark.parametrize("mode", ["full", "correctness_only", "abba"])
async def test_native_oss_recovers_inside_one_http_call(tmp_path, monkeypatch, source_seed, mode):
    registry, control, attempt, capability, store, raw, request, app, sleeps = _setup(
        tmp_path, monkeypatch, source_seed, mode, retry=True,
    )
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://runtime") as http:
            response = await http.post("/v1/operations", json=request, headers={
                "Authorization": f"Bearer {capability.token}",
            })
        assert response.status_code == 200, response.text
        assert response.json()["result"]["correct"] is True
        assert sleeps and all(delay == 5 for delay in sleeps)
        assert len(control.list_evaluations(attempt.id)) == (1 if mode == "full" else 0)
        assert all(e.correct for e in control.list_evaluations(attempt.id))
        assert {kind for _, _, kind in raw.prepared} == {"eval"}
    finally:
        if store is not None:
            store.close()
        control.close()
        registry.close()


@pytest.mark.anyio
async def test_authoritative_submit_does_not_convert_oss_error_to_candidate_rejection():
    client = FakeAgateClient(_successful_job(), submit_error=_oss_error())
    with pytest.raises(InfrastructureError) as caught:
        await submit_agate_job(
            client, "eval", {}, request_error="submit failed", invalid_response="invalid",
            non_object_response="not an object",
        )
    assert "oss_unavailable" in caught.value.public_detail


@pytest.mark.anyio
@pytest.mark.parametrize("mode", ["full", "correctness_only", "abba"])
async def test_upload_and_worker_download_retries_stay_inside_native_eval_http_call(
    tmp_path, monkeypatch, source_seed, mode,
):
    """A failed worker download gets fresh OSS refs, never a negative Kernel verdict."""
    registry, control, attempt, capability, store, raw, request, app, sleeps = _setup(
        tmp_path, monkeypatch, source_seed, mode, retry=True,
    )
    raw.unavailable = False
    upload = raw.upload_file
    get_job = raw.jobs.get_job
    failed_jobs = set()
    upload_attempts = []
    recovery_delays = []

    def flaky_upload(url, path):
        data = Path(path).read_bytes()
        with raw.lock:
            upload_attempts.append(data)
            first = len(upload_attempts) == 1
        if first:
            raise GatewayError(502, {
                "reason": "oss_upload_failed",
                "message": "OSS upload transport failed (TimeoutError)",
            })
        upload(url, path)

    def download_failure_once(job_id, **kwargs):
        with raw.lock:
            if not failed_jobs:
                failed_jobs.add(job_id)
            failed = job_id in failed_jobs
        if failed:
            return {"job_id": job_id, "status": "failed", "error": {
                "error_class": "infra", "reason": "oss_input_download_failed",
                "message": "worker could not download an OSS input", "trace_id": "req-download",
            }}
        return get_job(job_id, **kwargs)

    async def recovery_sleep(delay):
        recovery_delays.append(delay)
        assert not control.list_evaluations(attempt.id)
        assert control.get_operation_artifact(
            attempt.id, request["idempotency_key"], GatewayOperation.EVALUATE,
        ) is None
        assert not control._connection.execute(
            "SELECT 1 FROM gateway_evaluate_tasks WHERE status='completed'",
        ).fetchall()

    monkeypatch.setattr(raw, "upload_file", flaky_upload)
    monkeypatch.setattr(raw.jobs, "get_job", download_failure_once)
    monkeypatch.setattr("atrex_runtime.gateway.job_recovery.anyio.sleep", recovery_sleep)
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://runtime") as http:
            response = await http.post("/v1/operations", json=request, headers={
                "Authorization": f"Bearer {capability.token}",
            })
        assert response.status_code == 200, response.text
        assert response.json()["result"]["correct"] is True
        assert "oss_input_download_failed" not in response.text
        assert sleeps == [5] and recovery_delays == [5]
        sent = raw.jobs.requests if mode == "abba" else [r for _, r in raw.jobs.submitted]
        replacements = [r for r in sent if r["idempotency_key"].startswith("infra-retry:")]
        assert len(replacements) == 1
        references = [tuple(f["upload_ref"] for f in r["oss_files"]) for r in sent]
        assert len(set(references)) == len(sent)  # Replacement re-reserves, not stale URL reuse.
        assert all(descriptors == raw.prepared[0][1] for _, descriptors, _ in raw.prepared)
        assert all(kind == "eval" for _, _, kind in raw.prepared)
        assert len(control.list_evaluations(attempt.id)) == (1 if mode == "full" else 0)
        assert all(e.correct for e in control.list_evaluations(attempt.id))
    finally:
        if store is not None:
            store.close()
        control.close()
        registry.close()
