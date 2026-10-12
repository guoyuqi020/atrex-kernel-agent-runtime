"""Only Runtime request intervals, unioned per Session, suspend the Agent budget."""

import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest
from test_gateway_proxy import _request, _service
from test_gpu_wiki import _snapshot_value

from atrex_runtime.artifacts.local import LocalArtifactStore
from atrex_runtime.domain.errors import InfrastructureError
from atrex_runtime.domain.ids import new_worker_session_id
from atrex_runtime.domain.models import WorkerSession, WorkerSessionRole, WorkerSessionStatus
from atrex_runtime.knowledge import (
    KnowledgeSnapshotResponseV1,
    WikiProxyLimits,
    WikiProxyService,
)
from atrex_runtime.session_clock import CLOCK_RELATIVE_PATH, SessionClock
from atrex_runtime.workers.process import BoundedProcessConfig, BoundedProcessRunner


def test_wait_intervals_are_unioned_and_other_sessions_are_independent(tmp_path: Path) -> None:
    now = [100.0]
    clock = SessionClock(tmp_path / "a" / CLOCK_RELATIVE_PATH, clock=lambda: now[0])
    other = SessionClock(tmp_path / "b" / CLOCK_RELATIVE_PATH, clock=lambda: now[0])
    clock.start()
    other.start()
    peer = SessionClock(clock.path, clock=lambda: now[0])
    now[0] += 2
    with clock.waiting():
        now[0] += 10
        assert clock.now() == 102
        assert other.now() == 112
        with peer.waiting():
            now[0] += 30
            assert peer.now() == 102
        now[0] += 20
        assert clock.now() == 102
    now[0] += 3
    assert clock.now() == 105
    assert json.loads(clock.path.read_text())["wait_seconds"] == 60
    with pytest.raises(RuntimeError), clock.waiting():
        now[0] += 5
        raise RuntimeError("tool failed")
    now[0] += 4
    assert clock.now() == 109


def test_new_launch_fences_old_call_completions(tmp_path: Path) -> None:
    now = [100.0]
    clock = SessionClock(tmp_path / CLOCK_RELATIVE_PATH, clock=lambda: now[0])
    clock.start()
    with clock.waiting():
        now[0] += 10
        clock.start()
        now[0] += 20
    assert clock.now() == 130
    state = json.loads(clock.path.read_text())
    assert state["wait_seconds"] == 0 and state["calls"] == []


def test_dead_service_cannot_leave_session_clock_permanently_paused(tmp_path: Path) -> None:
    clock = SessionClock(tmp_path / CLOCK_RELATIVE_PATH)
    clock.start()
    subprocess.run([
        sys.executable, "-c",
        "from pathlib import Path; import os, sys; "
        "from atrex_runtime.session_clock import SessionClock; "
        "scope = SessionClock(Path(sys.argv[1])).waiting(); "
        "scope.__enter__(); os._exit(0)", str(clock.path),
    ], check=True, timeout=5)
    assert json.loads(clock.path.read_text())["paused_since"] is not None
    before = clock.now()
    time.sleep(0.02)
    assert clock.now() > before
    assert json.loads(clock.path.read_text())["paused_since"] is None


@pytest.mark.parametrize("role", [
    WorkerSessionRole.OPTIMIZER, WorkerSessionRole.FRAMEWORK_BASELINE,
])
@pytest.mark.parametrize("fail", [False, True])
@pytest.mark.anyio
async def test_authenticated_runtime_request_pauses_only_its_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, role: WorkerSessionRole, fail: bool,
) -> None:
    registry, control, attempt, capability, service, adapter = _service(tmp_path)
    registry.start_worker_session(WorkerSession(
        id=new_worker_session_id(), role=role, subject_id=attempt.id,
        external_run_id="session-clock-test", workspace_path=str(tmp_path / "worker"),
        status=WorkerSessionStatus.RUNNING, started_at=attempt.created_at,
        attempt_id=attempt.id, recovery_generation=attempt.recovery_generation,
    ))
    now = [100.0]
    clock = SessionClock(tmp_path / "worker" / CLOCK_RELATIVE_PATH, clock=lambda: now[0])
    clock.start()
    monkeypatch.setattr("atrex_runtime.gateway.control.SessionClock", lambda _path: clock)

    async def execute(request):
        assert clock.now() == 100
        now[0] += 8 * 3600 + 60
        assert clock.now() == 100
        if fail:
            raise InfrastructureError("terminal tool failure")
        return adapter.result

    monkeypatch.setattr(adapter, "execute", execute)
    try:
        if fail:
            with pytest.raises(InfrastructureError, match="terminal tool failure"):
                await service.execute(capability.token, _request(attempt))
        else:
            response = await service.execute(capability.token, _request(attempt))
            assert response.status == "completed"
        assert json.loads(clock.path.read_text())["calls"] == []
        now[0] += 10
        assert clock.now() == 110
    finally:
        control.close()
        registry.close()


def test_outer_process_survives_tool_wait_longer_than_budget_then_times_out_on_work(
    tmp_path: Path,
) -> None:
    clock = SessionClock(tmp_path / CLOCK_RELATIVE_PATH)
    clock.start()
    active = threading.Event()
    release = threading.Event()

    def wait_for_runtime():
        with clock.waiting():
            active.set()
            release.wait(timeout=5)

    waiter = threading.Thread(target=wait_for_runtime)
    waiter.start()
    assert active.wait(timeout=5)
    runner = BoundedProcessRunner(BoundedProcessConfig(0.15, 0.1, 4096))
    try:
        # Four times the entire active budget, all spent waiting for Runtime.
        result = runner.run(
            (sys.executable, "-c", "import time; time.sleep(0.6); print('finished')"),
            cwd=tmp_path, stdin=None, clock=clock.now,
        )
        assert result.returncode == 0 and result.stdout.strip() == "finished"
    finally:
        release.set()
        waiter.join(timeout=5)
    with pytest.raises(TimeoutError, match="active-time"):
        runner.run(
            (sys.executable, "-c", "import time; time.sleep(5)"),
            cwd=tmp_path, stdin=None, clock=clock.now,
        )


@pytest.mark.anyio
async def test_wiki_wait_also_stops_the_session_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, control, attempt, capability, _, _ = _service(tmp_path)
    registry.start_worker_session(WorkerSession(
        id=new_worker_session_id(), role=WorkerSessionRole.OPTIMIZER, subject_id=attempt.id,
        external_run_id="wiki-clock-test", workspace_path=str(tmp_path / "worker"),
        status=WorkerSessionStatus.RUNNING, started_at=attempt.created_at, attempt_id=attempt.id,
    ))
    now = [100.0]
    clock = SessionClock(tmp_path / "worker" / CLOCK_RELATIVE_PATH, clock=lambda: now[0])
    clock.start()
    monkeypatch.setattr("atrex_runtime.gateway.control.SessionClock", lambda _path: clock)

    class Client:
        async def query(self, request):
            now[0] += 600
            assert clock.now() == 100
            return KnowledgeSnapshotResponseV1.model_validate(_snapshot_value())

    wiki = WikiProxyService(
        control, control, registry, LocalArtifactStore(tmp_path / "artifacts"), Client(),
        WikiProxyLimits(4096, 1024), registry,
    )
    try:
        result = await wiki.query(capability.token, json.dumps({
            "schema_version": 1, "attempt_id": attempt.id,
            "idempotency_key": "wiki-wait", "query": "How to tile?",
        }).encode())
        assert result.snapshot_id == "wiki-snapshot-1"
        now[0] += 2
        assert clock.now() == 102
    finally:
        control.close()
        registry.close()
