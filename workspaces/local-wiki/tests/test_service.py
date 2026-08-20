"""HTTP behavior and Runtime wire-compatibility tests for the local Wiki."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
from atrex_runtime.artifacts.local import JsonValue
from atrex_runtime.domain.ids import (
    ArtifactDigest,
    new_attempt_id,
    new_campaign_id,
    new_epoch_id,
    new_kernel_agent_revision_id,
    new_lineage_id,
    new_wiki_feedback_id,
    parse_artifact_digest,
)
from atrex_runtime.domain.models import BranchRole, Dsl
from atrex_runtime.knowledge import (
    KnowledgeInteractionV1 as RuntimeKnowledgeInteractionV1,
)
from atrex_runtime.knowledge import (
    KnowledgeQueryV1 as RuntimeKnowledgeQueryV1,
)
from atrex_runtime.knowledge import (
    KnowledgeSnapshotResponseV1 as RuntimeKnowledgeSnapshotResponseV1,
)
from atrex_runtime.knowledge import (
    WikiFeedbackAckV1 as RuntimeWikiFeedbackAckV1,
)
from atrex_runtime.knowledge import (
    WikiFeedbackReportV1 as RuntimeWikiFeedbackReportV1,
)
from atrex_runtime.knowledge.ingest_models import (
    WikiFeedbackAttemptV1 as RuntimeWikiFeedbackAttemptV1,
)
from atrex_runtime.knowledge.ingest_models import (
    WikiFeedbackInteractionV1 as RuntimeWikiFeedbackInteractionV1,
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
async def test_feedback_ack_is_runtime_compatible_and_idempotent(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    app = build_application(settings, {})
    query = _query()
    report = RuntimeWikiFeedbackReportV1(
        campaign_id=query.campaign_id,
        lineage_id=query.lineage_id,
        epoch_id=query.epoch_id,
        epoch_number=1,
        operator=query.operator,
        dsl=query.dsl,
        hardware_target=query.hardware_target,
        evaluation_contract_digest=query.evaluation_contract_digest,
        evidence_checkpoint_digest=query.epoch_evidence_checkpoint_digest,
        attempts=(),
    )
    feedback_id = new_wiki_feedback_id()
    headers = {"idempotency-key": feedback_id}

    first = await _request(
        app,
        "/v1/knowledge/epoch-feedback",
        report.canonical_json_bytes(),
        headers=headers,
    )
    second = await _request(
        app,
        "/v1/knowledge/epoch-feedback",
        report.canonical_json_bytes(),
        headers=headers,
    )

    assert first[0] == second[0] == 202
    assert RuntimeWikiFeedbackAckV1.model_validate_json(first[1]).feedback_id == feedback_id
    with sqlite3.connect(settings.database) as connection:
        assert connection.execute("SELECT COUNT(*) FROM feedback").fetchone() == (1,)
    app.close()


@pytest.mark.anyio
async def test_feedback_uses_upstream_served_event_and_is_idempotent(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    app = build_application(settings, {})
    query = _query()
    content: JsonValue = {
        "records": {
            "nvidia.hopper.triton.kernel-opt.reduction": {
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
    content_digest = parse_artifact_digest(
        "sha256:"
        + hashlib.sha256(
            json.dumps(content, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    )
    response = RuntimeKnowledgeSnapshotResponseV1(
        snapshot_id="snapshot",
        content_digest=content_digest,
        content=content,
    )
    interaction = RuntimeKnowledgeInteractionV1(
        idempotency_key="query-1",
        query=query,
        response=response,
    )
    frozen = RuntimeWikiFeedbackInteractionV1(
        artifact_digest=_digest("interaction"),
        interaction=interaction,
    )
    attempt = RuntimeWikiFeedbackAttemptV1(
        attempt_id=query.attempt_id,
        branch=query.branch,
        ordinal=1,
        kernel_agent_revision_id=query.kernel_agent_revision_id,
        interactions=(frozen,),
        session_traces=(),
    )
    report = RuntimeWikiFeedbackReportV1(
        campaign_id=query.campaign_id,
        lineage_id=query.lineage_id,
        epoch_id=query.epoch_id,
        epoch_number=1,
        operator=query.operator,
        dsl=query.dsl,
        hardware_target=query.hardware_target,
        evaluation_contract_digest=query.evaluation_contract_digest,
        evidence_checkpoint_digest=query.epoch_evidence_checkpoint_digest,
        attempts=(attempt,),
    )
    feedback_id = new_wiki_feedback_id()
    headers = {"idempotency-key": feedback_id}

    first = await _request(
        app,
        "/v1/knowledge/epoch-feedback",
        report.canonical_json_bytes(),
        headers=headers,
    )
    second = await _request(
        app,
        "/v1/knowledge/epoch-feedback",
        report.canonical_json_bytes(),
        headers=headers,
    )

    assert first[0] == second[0] == 202
    events_path = settings.store_root / "kernel_wiki" / "feedback" / "events.jsonl"
    events = [json.loads(line) for line in events_path.read_text().splitlines()]
    assert len(events) == 1
    assert events[0]["record_id"] == "nvidia.hopper.triton.kernel-opt.reduction"
    assert events[0]["counts"] == {"served_count": 1}
    app.close()


@pytest.mark.anyio
async def test_feedback_rejects_changed_body_for_same_identity(tmp_path: Path) -> None:
    app = build_application(_settings(tmp_path), {})
    query = _query()
    report = RuntimeWikiFeedbackReportV1(
        campaign_id=query.campaign_id,
        lineage_id=query.lineage_id,
        epoch_id=query.epoch_id,
        epoch_number=1,
        operator=query.operator,
        dsl=query.dsl,
        hardware_target=query.hardware_target,
        evaluation_contract_digest=query.evaluation_contract_digest,
        evidence_checkpoint_digest=query.epoch_evidence_checkpoint_digest,
        attempts=(),
    )
    feedback_id = new_wiki_feedback_id()
    headers = {"idempotency-key": feedback_id}
    await _request(
        app,
        "/v1/knowledge/epoch-feedback",
        report.canonical_json_bytes(),
        headers=headers,
    )
    changed = report.model_copy(update={"epoch_number": 2})

    status, body = await _request(
        app,
        "/v1/knowledge/epoch-feedback",
        changed.canonical_json_bytes(),
        headers=headers,
    )

    assert status == 409
    assert json.loads(body) == {"error": "idempotency_conflict"}
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
