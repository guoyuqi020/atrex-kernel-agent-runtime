"""Real Agent CLI to HTTP Wiki Proxy, with frozen internal-corpus responses."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType

import pytest

from atrex_runtime.artifacts.local import ArtifactKind, JsonValue, LocalArtifactStore
from atrex_runtime.config import RuntimeSettings
from atrex_runtime.domain.ids import (
    new_attempt_id,
    new_campaign_id,
    new_epoch_id,
    new_kernel_agent_revision_id,
    new_lineage_id,
    parse_artifact_digest,
)
from atrex_runtime.domain.models import Dsl
from atrex_runtime.gateway.control import BootstrapGatewaySubject, SqliteGatewayControl
from atrex_runtime.gateway.control_models import GatewayOperation
from atrex_runtime.knowledge import (
    KnowledgeInteractionV1,
    KnowledgeQueryV1,
    KnowledgeSnapshotResponseV1,
)
from atrex_runtime.knowledge.models import canonical_json_bytes
from atrex_runtime.registry.sqlite import SqliteRegistry

REPOSITORY = Path(__file__).resolve().parents[1]
EXAMPLE = REPOSITORY / "examples/local-wiki"
RECORD_ID = "internal_gpu_wiki::alibaba.zwm890p.cuda.flash-attn-fp8.ppu15-resources-lowering"


def _helper() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "temporary_wiki_shell_integration", EXAMPLE / "temporary_wiki_shell.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _snapshot() -> KnowledgeSnapshotResponseV1:
    # Keep the internal native record's nested identity and evidence fields.
    # The proxy and Agent tool must not flatten it to the legacy public schema.
    content: JsonValue = {
        "query_id": "internal-ppu-query-001",
        "records": {
            RECORD_ID: {
                "source": "kernel_wiki",
                "wiki_identity": {
                    "wiki_id": RECORD_ID,
                    "store_id": "internal_gpu_wiki",
                    "wiki_revision": "sha256:" + "a" * 64,
                    "record_digest": "sha256:" + "b" * 64,
                },
                "management": {"tier": "core", "status": "current"},
                "payload": {
                    "confidence": "reported",
                    "toolchain": "SDK 2.1.2-a828f9",
                    "limitations": ["historical measurements; not rerun for this task"],
                },
            },
        },
        "notes": ["generation-reference; validate on the current product and workload"],
    }
    return KnowledgeSnapshotResponseV1(
        schema_version=1,
        service_api_version=1,
        snapshot_id="internal-ppu-snapshot-001",
        content_digest=parse_artifact_digest(
            "sha256:" + hashlib.sha256(canonical_json_bytes(content)).hexdigest()
        ),
        content=content,
    )


class RecordingWikiClient:
    def __init__(self) -> None:
        self.calls: list[KnowledgeQueryV1] = []
        self.response = _snapshot()

    async def query(self, query: KnowledgeQueryV1) -> KnowledgeSnapshotResponseV1:
        self.calls.append(query)
        return self.response


@pytest.mark.parametrize("bundle", ("atrex-kernel-agent-core", "kernel-design-agents"))
def test_agent_wiki_query_preserves_internal_records_and_replays_frozen_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    bundle: str,
) -> None:
    helper = _helper()
    settings = RuntimeSettings.from_file(EXAMPLE / "runtime.json")
    assert settings.gpu_wiki is not None
    settings = settings.model_copy(
        update={
            "gpu_wiki": settings.gpu_wiki.model_copy(update={"enabled": True}),
        }
    )
    client = RecordingWikiClient()
    monkeypatch.setattr(helper, "HttpGpuWikiClient", lambda *_args, **_kwargs: client)
    monkeypatch.setattr(helper, "HttpxGpuWikiTransport", lambda _url: object())
    digest = parse_artifact_digest("sha256:" + "0" * 64)
    subject = BootstrapGatewaySubject(
        attempt_id=new_attempt_id(),
        campaign_id=new_campaign_id(),
        lineage_id=new_lineage_id(),
        epoch_id=new_epoch_id(),
        kernel_agent_revision_id=new_kernel_agent_revision_id(),
        operator="gated_residual_combine",
        hardware_target="ZW-M890P",
        dsl=Dsl.CUDA,
        evaluation_contract_digest=digest,
        input_kernel_digest=digest,
        evidence_digest=digest,
        created_at=datetime.now(UTC),
    )
    signing_key = b"w" * 32
    server, thread, listener, port, capability = helper._serve(
        tmp_path,
        settings,
        subject,
        signing_key,
    )
    try:
        workspace, injected = helper._prepare_workspace(
            tmp_path,
            REPOSITORY / "src" / bundle,
            subject=subject,
            dsl=Dsl.CUDA,
        )
        query = "ZW-M890P 上 CUDA 寄存器资源和 lowering 有什么限制?"
        (workspace / "scratch/wiki-query.json").write_text(
            json.dumps({"query": query}, ensure_ascii=False),
            encoding="utf-8",
        )
        endpoint = f"http://127.0.0.1:{port}"
        # A deliberately small environment proves that provider/GPU credentials
        # and access to the internal corpus are not required in the Agent process.
        environment = {
            "PATH": os.environ["PATH"],
            **injected,
            "ATREX_GATEWAY_PROXY_URL": endpoint,
            "ATREX_GATEWAY_CAPABILITY": capability,
            "ATREX_WIKI_PROXY_URL": endpoint,
            "ATREX_WIKI_CAPABILITY": capability,
            "NO_PROXY": "127.0.0.1,localhost",
        }
        results = []
        for _ in range(2):
            result = subprocess.run(
                (
                    sys.executable,
                    "agent/optimizer/src/runtime_tools.py",
                    "wiki-query",
                    "--request",
                    "scratch/wiki-query.json",
                ),
                cwd=workspace,
                env=environment,
                check=False,
                capture_output=True,
                text=True,
                timeout=20,
            )
            assert result.returncode == 0, result.stderr or result.stdout
            results.append(json.loads(result.stdout))
        assert results == [client.response.content, client.response.content]
        assert len(client.calls) == 1
        trusted = client.calls[0]
        assert trusted.query == query
        assert trusted.attempt_id == subject.attempt_id
        assert trusted.operator == subject.operator
        assert trusted.dsl is Dsl.CUDA
        assert trusted.hardware_target == "ZW-M890P"
    finally:
        server.should_exit = True
        thread.join(timeout=10)
        listener.close()
        assert not thread.is_alive()

    # Confirm both CLI calls share the same persisted interaction, rather than
    # being served from a process-local fake or an Agent-side response cache.
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    with SqliteRegistry(tmp_path / "registry.sqlite") as registry:
        control = SqliteGatewayControl(
            tmp_path / "gateway.sqlite",
            registry,
            signing_key=signing_key,
        )
        try:
            interactions = control.list_operation_artifacts(
                (subject.attempt_id,),
                GatewayOperation.WIKI_QUERY,
            )
            assert len(interactions) == 1
            frozen = artifacts.verify(interactions[0][2])
            assert frozen.kind is ArtifactKind.WIKI_INTERACTION
            interaction = KnowledgeInteractionV1.model_validate_json(
                (frozen.payload_path / "value.json").read_bytes()
            )
            assert interaction.query == client.calls[0]
            assert interaction.response == client.response
            events = [
                event.kind
                for event in registry.list_runtime_events(after_sequence=0, limit=100)
                if event.kind.startswith("wiki.query_")
            ]
            assert events == ["wiki.query_submitted", "wiki.query_completed"]
        finally:
            control.close()


def test_temporary_wiki_shell_rejects_disabled_switch_before_credentials_or_launch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    helper = _helper()
    config = json.loads((EXAMPLE / "runtime.json").read_text())
    config["gpu_wiki"]["enabled"] = False
    path = tmp_path / "disabled-runtime.json"
    path.write_text(json.dumps(config), encoding="utf-8")

    def unexpected(*_args, **_kwargs):
        pytest.fail("disabled Wiki must fail before credential loading or shell preparation")

    monkeypatch.setattr(helper, "read_capability_signing_key", unexpected)
    monkeypatch.setattr(helper, "_prepare_workspace", unexpected)
    monkeypatch.setattr(helper, "_serve", unexpected)
    with pytest.raises(ValueError, match=r"gpu_wiki\.enabled=true"):
        helper.main(["--config", str(path)])
