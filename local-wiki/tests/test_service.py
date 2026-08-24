"""HTTP behavior and Runtime wire-compatibility tests for the local Wiki."""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import anyio
import pytest
from atrex_runtime.domain.ids import (
    ArtifactDigest,
    new_attempt_id,
    new_campaign_id,
    new_epoch_id,
    new_kernel_agent_revision_id,
    new_lineage_id,
    parse_artifact_digest,
)
from atrex_runtime.domain.models import BranchRole, Dsl
from atrex_runtime.knowledge import (
    KnowledgeQueryV1 as RuntimeKnowledgeQueryV1,
)
from atrex_runtime.knowledge import (
    KnowledgeSnapshotResponseV1 as RuntimeKnowledgeSnapshotResponseV1,
)

from atrex_local_wiki.app import LocalWikiApplication, build_application
from atrex_local_wiki.config import LocalWikiSettings


def _digest(label: str) -> ArtifactDigest:
    return parse_artifact_digest("sha256:" + (label.encode().hex() + "0" * 64)[:64])


def _settings(tmp_path: Path, *, authenticated: bool = False) -> LocalWikiSettings:
    return LocalWikiSettings(
        host="127.0.0.1",
        port=8091,
        reference_root=Path(__file__).parent / "fixtures" / "corpus",
        store_root=tmp_path / "gpu-wiki",
        database=tmp_path / "wiki.sqlite",
        python_executable=Path(sys.executable),
        bearer_token_env="LOCAL_WIKI_TOKEN" if authenticated else None,
        max_request_bytes=1_000_000,
        max_response_bytes=100_000,
    )


def test_settings_default_to_current_python() -> None:
    settings = LocalWikiSettings(
        host="127.0.0.1",
        port=8091,
        reference_root=Path(__file__).parent / "fixtures" / "corpus",
        store_root=Path("gpu-wiki-state"),
        database=Path("wiki.sqlite"),
        max_request_bytes=1_000_000,
        max_response_bytes=100_000,
    )

    assert settings.python_executable == Path(sys.executable).resolve()
    assert settings.max_concurrent_queries == 16


def _query() -> RuntimeKnowledgeQueryV1:
    return RuntimeKnowledgeQueryV1(
        campaign_id=new_campaign_id(),
        lineage_id=new_lineage_id(),
        epoch_id=new_epoch_id(),
        epoch_number=1,
        attempt_id=new_attempt_id(),
        branch=BranchRole.ACTIVE,
        attempt_ordinal=1,
        kernel_agent_revision_id=new_kernel_agent_revision_id(),
        operator="reduction",
        dsl=Dsl.TRITON,
        hardware_target="nvidia-h100",
        evaluation_contract_digest=_digest("contract"),
        epoch_evidence_checkpoint_digest=_digest("epoch"),
        attempt_evidence_digest=_digest("attempt"),
        query="How should I tile this reduction?",
    )


async def _request(
    app: LocalWikiApplication,
    path: str,
    body: bytes = b"",
    *,
    headers: Mapping[str, str] | None = None,
    method: str = "POST",
) -> tuple[int, bytes]:
    messages = [{"type": "http.request", "body": body, "more_body": False}]
    sent: list[dict[str, Any]] = []

    async def receive() -> dict[str, Any]:
        return messages.pop(0)

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)

    raw_headers = [
        (name.encode("latin-1"), value.encode("latin-1"))
        for name, value in (headers or {}).items()
    ]
    await app(
        {"type": "http", "method": method, "path": path, "headers": raw_headers},
        receive,
        send,
    )
    return int(sent[0]["status"]), bytes(sent[1].get("body", b""))


@pytest.mark.anyio
async def test_query_response_is_runtime_compatible_and_architecture_scoped(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record_id = "nvidia.hopper.triton.kernel-opt.reduction"
    commands: list[list[str]] = []

    def run_query(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[bytes]:
        commands.append(command)
        return subprocess.CompletedProcess(
            (),
            0,
            json.dumps(
                {
                    "records": {
                        record_id: {
                            "store": "gpu_wiki",
                            "source": "kernel_wiki",
                            "type": "technique-card",
                            "applies_to": {"arch": "hopper", "dsl": "triton"},
                            "match": {"arch": "exact"},
                            "payload": {"goal": "tile a reduction"},
                        }
                    },
                    "notes": [],
                }
            ).encode(),
            b"",
        )

    monkeypatch.setattr(subprocess, "run", run_query)
    app = build_application(_settings(tmp_path), {})
    query = _query()

    status, body = await _request(app, "/v1/knowledge/query", query.canonical_json_bytes())

    assert status == 200
    response = RuntimeKnowledgeSnapshotResponseV1.model_validate_json(body)
    assert response.snapshot_id.startswith("localwiki_")
    assert isinstance(response.content, dict)
    records = response.content["records"]
    assert isinstance(records, dict)
    assert list(records) == [record_id]
    assert Path(commands[0][1]).name == "query_nl.py"
    assert "Target hardware reported by the runtime: nvidia-h100" in commands[0][2]
    assert "--agent-cli" not in commands[0]
    assert "--timeout" not in commands[0]
    assert "--max-records" not in commands[0]
    app.close()


@pytest.mark.anyio
async def test_absent_internal_store_note_is_dropped_without_hiding_a_public_outage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def run_query(_command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[bytes]:
        return subprocess.CompletedProcess(
            (),
            0,
            json.dumps(
                {
                    "records": {},
                    "notes": [
                        "[internal_gpu_wiki] store unavailable; module returned empty",
                        "[gpu_wiki] store unavailable; module returned empty",
                        "[internal_gpu_wiki] retrieval failed; module returned empty: boom",
                        "[gpu_wiki] widened to any dsl",
                    ],
                }
            ).encode(),
            b"",
        )

    monkeypatch.setattr(subprocess, "run", run_query)
    app = build_application(_settings(tmp_path), {})

    status, body = await _request(
        app,
        "/v1/knowledge/query",
        _query().canonical_json_bytes(),
    )

    assert status == 200
    response = RuntimeKnowledgeSnapshotResponseV1.model_validate_json(body)
    assert isinstance(response.content, dict)
    assert response.content["notes"] == [
        "[gpu_wiki] store unavailable; module returned empty",
        "[internal_gpu_wiki] retrieval failed; module returned empty: boom",
        "[gpu_wiki] widened to any dsl",
    ]
    app.close()


@pytest.mark.anyio
async def test_query_execution_uses_bounded_concurrency(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(tmp_path).model_copy(update={"max_concurrent_queries": 2})
    state_lock = threading.Lock()
    two_running = threading.Event()
    active = 0
    peak = 0

    def run_query(_command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[bytes]:
        nonlocal active, peak
        with state_lock:
            active += 1
            peak = max(peak, active)
            if active == 2:
                two_running.set()
        try:
            if not two_running.wait(timeout=2):
                raise AssertionError("queries remained globally serialized")
            time.sleep(0.05)
            return subprocess.CompletedProcess(
                (),
                0,
                json.dumps({"records": {}, "notes": []}).encode(),
                b"",
            )
        finally:
            with state_lock:
                active -= 1

    monkeypatch.setattr(subprocess, "run", run_query)
    app = build_application(settings, {})
    statuses: list[int] = []

    async def query_once() -> None:
        status, _body = await _request(
            app,
            "/v1/knowledge/query",
            _query().canonical_json_bytes(),
        )
        statuses.append(status)

    async with anyio.create_task_group() as tasks:
        for _ in range(4):
            tasks.start_soon(query_once)

    assert statuses == [200, 200, 200, 200]
    assert peak == 2
    app.close()


@pytest.mark.anyio
async def test_browser_client_is_served_without_changing_versioned_api(tmp_path: Path) -> None:
    app = build_application(_settings(tmp_path), {})

    status, body = await _request(app, "/", method="GET")

    assert status == 200
    assert b"Local GPU Wiki" in body
    assert b"/v1/knowledge/query" in body
    assert b"/v1/knowledge/read" not in body
    assert b"local test double" in body
    app.close()


@pytest.mark.anyio
async def test_feedback_endpoint_is_not_exposed(tmp_path: Path) -> None:
    app = build_application(_settings(tmp_path), {})

    status, body = await _request(
        app,
        "/v1/knowledge/epoch-feedback",
        b"{}",
    )

    assert status == 404
    assert json.loads(body) == {"error": "not_found"}
    app.close()


@pytest.mark.anyio
async def test_optional_bearer_auth_and_readiness(tmp_path: Path) -> None:
    app = build_application(_settings(tmp_path, authenticated=True), {"LOCAL_WIKI_TOKEN": "x"})

    unauthorized, _body = await _request(
        app,
        "/v1/knowledge/query",
        _query().canonical_json_bytes(),
    )
    ready, body = await _request(app, "/readyz", method="GET")

    assert unauthorized == 401
    assert ready == 200
    assert json.loads(body) == {"status": "ready"}
    app.close()
