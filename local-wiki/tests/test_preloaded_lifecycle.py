"""A resident native index must never mix revisions or silently lose failures."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest

from atrex_local_wiki import retrieval
from atrex_local_wiki.models import KnowledgeQueryV1
from atrex_local_wiki.preloaded_worker import PreloadedWorker, PreloadedWorkerError
from atrex_local_wiki.retrieval import CorpusIndex, GpuWikiQueryError


@pytest.fixture
def native_store(tmp_path: Path) -> Path:
    root = tmp_path / "store"
    (root / "tools").mkdir(parents=True)
    (root / "search_index").mkdir()
    for name in retrieval._INDEXED_TOOLS:
        (root / "tools" / name).write_text("# test native implementation\n")
    (root / "search_index/index.json").write_text(
        json.dumps(
            {
                "schema": "gpu-search-1.0",
                "manifest_schema": "gpu-search-manifest-1.1",
                "store_id": "test_store",
                "index_digest": "sha256:" + "a" * 64,
                "shards": {"kernel_wiki": "kernel_wiki.json"},
            }
        )
    )
    (root / "search_index/kernel_wiki.json").write_text('{"records": {}}')
    (root / "search_index/wiki_governance.json").write_text('{"records": {}}')
    return root


class FakeWorker:
    def __init__(self, root: Path, _python: Path, **_kwargs: Any) -> None:
        self.root = root
        self.closed = 0
        self.healthy = True
        self.query_hook: Callable[[], None] | None = None
        self.calls: list[tuple[list[str], float | None]] = []

    def check_health(self) -> None:
        if self.closed or not self.healthy:
            raise PreloadedWorkerError("fake worker unavailable")

    def query(self, argv: list[str], timeout: float | None) -> subprocess.CompletedProcess[bytes]:
        self.check_health()
        self.calls.append((argv, timeout))
        if self.query_hook:
            self.query_hook()
        return subprocess.CompletedProcess(
            argv, 0, b'{"query_id":"native-id","records":{},"notes":[]}', b""
        )

    def close(self) -> None:
        self.closed += 1


@pytest.fixture
def workers(monkeypatch: pytest.MonkeyPatch) -> list[FakeWorker]:
    created: list[FakeWorker] = []

    def make(*args: Any, **kwargs: Any) -> FakeWorker:
        worker = FakeWorker(*args, **kwargs)
        created.append(worker)
        return worker

    def no_subprocess(*_args: Any, **_kwargs: Any) -> None:
        pytest.fail("a preloaded query must not fall back to subprocess retrieval")

    monkeypatch.setattr(retrieval, "PreloadedWorker", make)
    monkeypatch.setattr(retrieval.subprocess, "run", no_subprocess)
    return created


def make_index(root: Path) -> CorpusIndex:
    return CorpusIndex(
        root,
        python_executable=Path(sys.executable),
        agent_cli="claude",
        query_timeout_seconds=60,
        max_concurrent_queries=3,
        max_results=7,
        max_response_bytes=100_000,
        indexed_execution="preloaded",
    )


@pytest.fixture
def index(native_store: Path, workers: list[FakeWorker]) -> Iterator[CorpusIndex]:
    value = make_index(native_store)
    try:
        yield value
    finally:
        value.close()


def query_request() -> KnowledgeQueryV1:
    return KnowledgeQueryV1(
        campaign_id="campaign_" + "1" * 32,
        lineage_id="lineage_" + "2" * 32,
        epoch_id="epoch_" + "3" * 32,
        epoch_number=1,
        attempt_id="attempt_" + "4" * 32,
        branch="active",
        attempt_ordinal=1,
        kernel_agent_revision_id="agentrev_" + "5" * 32,
        operator="gated_residual_combine",
        dsl="cuda",
        hardware_target="ZW-M890P",
        evaluation_contract_digest="sha256:" + "6" * 64,
        epoch_evidence_checkpoint_digest="sha256:" + "7" * 64,
        attempt_evidence_digest="sha256:" + "8" * 64,
        query="What are the resource constraints?",
    )


def mutate(root: Path) -> None:
    target = root / "search_index/kernel_wiki.json"
    target.write_text(target.read_text() + "\n")


def test_preheats_before_first_query_and_reuses_same_revision(
    index: CorpusIndex, workers: list[FakeWorker]
) -> None:
    assert len(workers) == 1
    assert not workers[0].calls
    first = index.query(query_request())
    second = index.query(query_request())
    assert first == second
    assert len(workers) == 1
    argv, timeout = workers[0].calls[0]
    assert timeout == 90
    assert argv[-6:] == ["--agent-cli", "claude", "--timeout", "60", "--max-records", "7"]
    assert argv[0] == query_request().query


def test_revision_change_rebuilds_once_and_closes_unused_generation(
    native_store: Path, index: CorpusIndex, workers: list[FakeWorker]
) -> None:
    before = index.query(query_request()).revision
    mutate(native_store)
    after = index.query(query_request()).revision
    assert before != after
    assert len(workers) == 2
    assert workers[0].closed == 1
    assert workers[1].closed == 0
    index.query(query_request())
    assert len(workers) == 2


def test_old_generation_stays_alive_until_its_last_query_releases_it(
    native_store: Path, index: CorpusIndex, workers: list[FakeWorker]
) -> None:
    revision = index._query_revision()
    with index._lease_worker(revision) as previous:
        mutate(native_store)
        index.query(query_request())
        assert previous is workers[0]
        assert workers[0].closed == 0
        assert workers[1].closed == 0
    assert workers[0].closed == 1
    assert workers[1].closed == 0


def test_close_stops_current_and_in_flight_retired_generation_once(
    native_store: Path, index: CorpusIndex, workers: list[FakeWorker]
) -> None:
    with index._lease_worker(index._query_revision()):
        mutate(native_store)
        index.query(query_request())
        index.close()
        index.close()
        assert [worker.closed for worker in workers] == [1, 1]
        with pytest.raises(GpuWikiQueryError, match="closed"):
            index.query(query_request())
    assert [worker.closed for worker in workers] == [1, 1]


def test_unhealthy_worker_is_replaced_without_using_legacy_execution(
    index: CorpusIndex, workers: list[FakeWorker]
) -> None:
    workers[0].healthy = False
    with pytest.raises(PreloadedWorkerError, match="unavailable"):
        index.check_health()
    index.query(query_request())
    assert len(workers) == 2
    assert workers[0].closed == 1


def test_preload_failure_does_not_fall_back_or_publish_partial_generation(
    native_store: Path,
    index: CorpusIndex,
    workers: list[FakeWorker],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mutate(native_store)

    def fail(*_args: Any, **_kwargs: Any) -> None:
        raise PreloadedWorkerError("bad native revision")

    monkeypatch.setattr(retrieval, "PreloadedWorker", fail)
    with pytest.raises(GpuWikiQueryError, match="bad native revision"):
        index.query(query_request())
    assert len(workers) == 1
    assert workers[0].closed == 0


def test_mutation_during_preload_discards_the_new_worker(
    native_store: Path,
    index: CorpusIndex,
    workers: list[FakeWorker],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mutate(native_store)

    def unstable(*args: Any, **kwargs: Any) -> FakeWorker:
        worker = FakeWorker(*args, **kwargs)
        workers.append(worker)
        mutate(native_store)
        return worker

    monkeypatch.setattr(retrieval, "PreloadedWorker", unstable)
    with pytest.raises(GpuWikiQueryError, match="changed during preloading"):
        index.query(query_request())
    assert workers[0].closed == 0
    assert workers[1].closed == 1


def test_store_change_during_query_rejects_the_old_result(
    native_store: Path, index: CorpusIndex, workers: list[FakeWorker]
) -> None:
    workers[0].query_hook = lambda: mutate(native_store)
    with pytest.raises(GpuWikiQueryError, match="Store changed while"):
        index.query(query_request())
    result = index.query(query_request())
    assert result.revision == index._query_revision()
    assert workers[0].closed == 1


def test_timeout_returns_failure_without_using_subprocess_fallback(
    index: CorpusIndex, workers: list[FakeWorker]
) -> None:
    def timeout() -> None:
        raise PreloadedWorkerError("query timed out")

    workers[0].query_hook = timeout
    with pytest.raises(GpuWikiQueryError, match="timed out"):
        index.query(query_request())
    assert len(workers) == 1


def test_concurrent_queries_share_one_new_generation(
    native_store: Path, index: CorpusIndex, workers: list[FakeWorker]
) -> None:
    mutate(native_store)
    barrier = threading.Barrier(3)

    def run() -> str:
        barrier.wait(timeout=5)
        return index.query(query_request()).revision

    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(run) for _ in range(3)]
        revisions = [future.result(timeout=10) for future in futures]
    assert len(set(revisions)) == 1
    assert len(workers) == 2
    assert len(workers[1].calls) == 3
    assert workers[0].closed == 1


@pytest.fixture
def fork_worker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[PreloadedWorker]:
    if not hasattr(os, "fork"):
        pytest.skip("native preload workers require POSIX fork")
    driver = tmp_path / "fake_native_server.py"
    source_root = Path(retrieval.__file__).resolve().parents[1]
    # Exercise the actual daemon, transport and NativePreload.run cleanup. Only
    # replace the corpus-dependent constructor and the model with a sleeping CLI.
    driver.write_text(
        "import json, subprocess, sys, time\n"
        "from pathlib import Path\n"
        "from types import SimpleNamespace\n"
        f"sys.path.insert(0, {str(source_root)!r})\n"
        "from atrex_local_wiki import preloaded_server\n"
        "from atrex_local_wiki.native_preload import NativePreload\n"
        "def query(argv):\n"
        "    if argv[0] == 'quick':\n"
        "        print(json.dumps({'query_id': 'ok', 'records': {}, 'notes': []}))\n"
        "        return 0\n"
        "    model = subprocess.Popen(\n"
        "        [sys.executable, '-c', 'import time; time.sleep(120)'],\n"
        "        start_new_session=True)\n"
        "    Path(argv[1]).write_text(str(model.pid))\n"
        "    while True:\n"
        "        time.sleep(10)\n"
        "class FakeNative(NativePreload):\n"
        "    def __init__(self, root):\n"
        "        self.modules = {'query_nl': SimpleNamespace(main=query)}\n"
        "preloaded_server.NativePreload = FakeNative\n"
        "preloaded_server.serve(\n"
        "    Path(sys.argv[1]), sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))\n"
    )
    real_popen = subprocess.Popen

    def start_driver(command: list[str], **kwargs: Any) -> subprocess.Popen[bytes]:
        assert Path(command[1]).name == "preloaded_server.py"
        return real_popen([*command[:1], str(driver), *command[2:]], **kwargs)

    monkeypatch.setattr(subprocess, "Popen", start_driver)
    worker = PreloadedWorker(
        tmp_path, Path(sys.executable), concurrency=3, max_response_bytes=100_000
    )
    try:
        yield worker
    finally:
        worker.close()


def wait_until(predicate: Callable[[], bool], *, seconds: float = 5) -> None:
    deadline = time.monotonic() + seconds
    while not predicate():
        if time.monotonic() >= deadline:
            pytest.fail("process lifecycle did not reach the expected state")
        time.sleep(0.02)


def process_is_dead(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    return False


def test_query_timeout_reaps_model_in_an_independent_process_group(
    fork_worker: PreloadedWorker, tmp_path: Path
) -> None:
    pid_file = tmp_path / "model.pid"
    with pytest.raises(PreloadedWorkerError, match="timed out"):
        fork_worker.query(["slow", str(pid_file)], timeout=0.5)
    wait_until(pid_file.exists)
    pid = int(pid_file.read_text())
    wait_until(lambda: process_is_dead(pid))
    fork_worker.check_health()
    assert fork_worker.query(["quick"], timeout=2).returncode == 0


def test_daemon_close_reaps_active_model_and_releases_query_client(
    fork_worker: PreloadedWorker, tmp_path: Path
) -> None:
    pid_file = tmp_path / "model.pid"
    with ThreadPoolExecutor(max_workers=1) as pool:
        query = pool.submit(fork_worker.query, ["slow", str(pid_file)], 10)
        wait_until(pid_file.exists)
        pid = int(pid_file.read_text())
        fork_worker.close()
        wait_until(lambda: process_is_dead(pid))
        # Closing an active worker may return the native cancellation envelope
        # or an explicit transport error; it must never claim query success.
        try:
            result = query.result(timeout=3)
        except PreloadedWorkerError:
            pass
        else:
            assert result.returncode != 0


@pytest.mark.parametrize("missing", [False, True])
def test_real_server_boots_without_site_packages_and_reports_native_tool_error(
    native_store: Path, tmp_path: Path, missing: bool
) -> None:
    if missing:
        (native_store / "tools/wiki_profile.py").unlink()
    server = Path(retrieval.__file__).with_name("preloaded_server.py")
    process = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            str(server),
            str(native_store),
            str(tmp_path / "query.sock"),
            "1",
            "100000",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert process.returncode == 1
    envelope = json.loads(process.stdout)
    assert envelope["status"] == "error"
    expected = (
        "GPU Wiki native tool is missing or unsafe: wiki_profile"
        if missing
        else "Unsupported GPU Wiki native tool revision: wiki_profile"
    )
    assert envelope["error"] == expected
    assert process.stderr == ""


def test_startup_eof_preserves_native_interpreter_diagnostics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_popen = subprocess.Popen

    def fail_before_readiness(_command: list[str], **kwargs: Any) -> subprocess.Popen[bytes]:
        return real_popen(
            [sys.executable, "-S", "-c", "raise RuntimeError('startup-regression-sentinel')"],
            **kwargs,
        )

    monkeypatch.setattr(subprocess, "Popen", fail_before_readiness)
    with pytest.raises(PreloadedWorkerError, match="startup-regression-sentinel"):
        PreloadedWorker(tmp_path, Path(sys.executable), concurrency=1, max_response_bytes=100_000)
