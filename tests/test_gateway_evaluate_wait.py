"""Evaluate/ABBA wait inside Runtime across Agate long-poll expiry."""

from collections import Counter
from pathlib import Path

import anyio
import pytest
from test_gateway_error_end_to_end import _assert_diagnostic, _failed_job
from test_gateway_evaluate_recovery import Scenario

from atrex_runtime.gateway import GatewayOperation


def _scenario(tmp_path, mode):
    scenario = Scenario(tmp_path, mode == "abba")
    if mode == "correctness_only":
        scenario.request["mode"] = mode
        scenario.request.pop("latency_prediction")
    return scenario


@pytest.mark.anyio
@pytest.mark.parametrize("mode", ["full", "correctness_only", "abba"])
@pytest.mark.parametrize("terminal_status", ["succeeded", "failed", "cancelled"])
async def test_evaluate_waits_for_terminal_jobs_before_http_reply_and_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str, terminal_status: str,
) -> None:
    scenario = _scenario(tmp_path, mode)
    polls = Counter()
    finished = set()
    original_get = scenario.client.get_job

    def get_job(job_id, **kwargs):
        assert kwargs == {"wait": False, "timeout": 30.0}
        assert scenario.control.get_operation_artifact(
            scenario.attempt.id, scenario.request["idempotency_key"], GatewayOperation.EVALUATE,
        ) is None
        polls[job_id] += 1
        if polls[job_id] <= 2:
            return {"job_id": job_id, "status": "queued" if polls[job_id] == 1 else "running"}
        finished.add(job_id)
        if terminal_status != "succeeded":
            return {**_failed_job(terminal_status), "job_id": job_id}
        return original_get(job_id, **kwargs)

    monkeypatch.setattr(scenario.client, "get_job", get_job)
    store_result = scenario.service._store_gateway_result

    def store_terminal_only(*args, **kwargs):
        assert finished == {row[2] for row in scenario.client.submissions}
        return store_result(*args, **kwargs)

    monkeypatch.setattr(scenario.service, "_store_gateway_result", store_terminal_only)
    try:
        with anyio.fail_after(10):
            response = await scenario.post()
        assert len(scenario.client.submissions) == 2
        assert not scenario.client.cancellations
        assert finished
        assert not scenario.rows("gateway_active_calls")
        assert response.status_code == 200, response.text
        body = response.json()
        expected_status = (
            "completed" if terminal_status == "succeeded"
            else "failed" if mode == "abba" else terminal_status
        )
        assert body["status"] == expected_status
        if terminal_status == "succeeded":
            assert body["result"]["correct"] is True
        elif mode == "abba":
            assert body["result"]["correct"] is False
        else:
            assert body["result"]["correct"] is None
            assert body["result"]["correctness"] is None
            assert not scenario.control.list_evaluations(scenario.attempt.id)
        assert polls == {row[2]: 3 for row in scenario.client.submissions}
        if terminal_status != "succeeded":
            _assert_diagnostic(response.text)
        # Only the final result is replayable. Reading it does not query Agate again.
        replay = await scenario.post()
        assert replay.json() == body
        read = await scenario.post({
            "schema_version": 2,
            "attempt_id": scenario.attempt.id,
            "idempotency_key": "read-result",
            "operation": "result_artifact_read",
            "result_artifact_digest": body["result_artifact_digest"],
        }, path="/v1/runtime/queries")
        assert read.status_code == 200, read.text
        assert read.json()["result"]["result"] == body["result"]
        assert sum(polls.values()) == 6
        assert len(scenario.client.submissions) == 2
    finally:
        scenario.close()


@pytest.mark.anyio
@pytest.mark.parametrize("mode", ["full", "correctness_only", "abba"])
async def test_interrupted_evaluate_wait_resumes_original_jobs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str,
) -> None:
    scenario = _scenario(tmp_path, mode)
    original_get = scenario.client.get_job
    polls = Counter()

    def running(job_id, **kwargs):
        polls[job_id] += 1
        return {"job_id": job_id, "status": "running"}

    monkeypatch.setattr(scenario.client, "get_job", running)
    try:
        with anyio.move_on_after(0.2) as scope:
            await scenario.post()
            pytest.fail("Evaluate returned before the job finished")
        assert scope.cancel_called
        assert len(scenario.client.submissions) == 2
        job_ids = {row[2] for row in scenario.client.submissions}
        assert polls == {job_id: 1 for job_id in job_ids}
        assert scenario.control.get_operation_artifact(
            scenario.attempt.id, scenario.request["idempotency_key"], GatewayOperation.EVALUATE,
        ) is None
        assert {job.job_id for job in scenario.jobs.list_owned(scenario.attempt.id)} == job_ids
        monkeypatch.setattr(scenario.client, "get_job", original_get)
        with anyio.fail_after(5):
            response = await scenario.post()
        assert response.status_code == 200, response.text
        assert response.json()["result"]["correct"] is True
        assert {row[0] for row in scenario.client.fetches} == job_ids
        # Full/ABBA recovery probes durable bindings first; completed jobs are reused.
        assert {wait for _, wait in scenario.client.fetches} == {False}
        assert len(scenario.client.submissions) == 2
        assert not scenario.client.cancellations
    finally:
        scenario.close()


@pytest.mark.anyio
@pytest.mark.parametrize("mode", ["full", "correctness_only", "abba"])
async def test_evaluate_rejects_unknown_status_without_sealing_or_polling_forever(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str,
) -> None:
    scenario = _scenario(tmp_path, mode)
    polls = Counter()

    def get_job(job_id, **kwargs):
        polls[job_id] += 1
        return {"job_id": job_id, "status": "unexpected"}

    monkeypatch.setattr(scenario.client, "get_job", get_job)
    try:
        with anyio.fail_after(5):
            response = await scenario.post()
        assert response.status_code == 503, response.text
        assert "unknown job status" in response.json()["detail"]
        assert polls and all(count == 1 for count in polls.values())
        assert scenario.control.get_operation_artifact(
            scenario.attempt.id, scenario.request["idempotency_key"], GatewayOperation.EVALUATE,
        ) is None
    finally:
        scenario.close()
