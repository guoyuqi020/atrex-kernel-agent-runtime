"""The internal search-index corpus uses its own tools and wire representation."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import pytest
from atrex_runtime.knowledge import KnowledgeSnapshotResponseV1

from atrex_local_wiki.app import build_application
from atrex_local_wiki.config import LocalWikiSettings
from atrex_local_wiki.models import KnowledgeQueryV1
from atrex_local_wiki.retrieval import CorpusIndex, GpuWikiQueryError

_TOOLS = (
    "query_nl.py",
    "query.py",
    "agent_launch.py",
    "hardware_identity.py",
    "operator_scope.py",
    "wiki_profile.py",
)
_RECORD_ID = "internal_gpu_wiki::alibaba.zwm890p.cuda.flash-attn-fp8.ppu15-resources-lowering"
_RECORD = {
    "source": "kernel_wiki",
    "wiki_identity": {
        "wiki_id": _RECORD_ID,
        "store_id": "internal_gpu_wiki",
        "wiki_revision": "sha256:" + "a" * 64,
        "record_digest": "sha256:" + "b" * 64,
    },
    "management": {"tier": "core", "status": "current"},
    "payload": {"confidence": "reported", "toolchain": "SDK 2.1.2-a828f9"},
}


@pytest.fixture
def indexed_corpus(tmp_path: Path) -> Path:
    root = tmp_path / "reference"
    (root / "tools").mkdir(parents=True)
    (root / "search_index").mkdir()
    for name in _TOOLS:
        (root / "tools" / name).write_text("# native tool\n")
    (root / "tools/query_nl.py").write_text(
        "import argparse, json\n"
        "from pathlib import Path\n"
        "parser = argparse.ArgumentParser()\n"
        "parser.add_argument('question')\n"
        "for name in ['store-root', 'max-bytes', 'max-records', 'timeout', 'agent-cli']:\n"
        "    parser.add_argument('--' + name)\n"
        "args = parser.parse_args()\n"
        "root = Path(args.store_root)\n"
        "(root / 'invocation.json').write_text(json.dumps(vars(args)))\n"
        "records = json.loads((root / 'search_index/kernel_wiki.json').read_text())['records']\n"
        "print(json.dumps({'query_id': 'native-query-001', 'records': records,\n"
        "                  'notes': ['generation-reference; not a measured product result']}))\n"
    )
    manifest = {
        "schema": "gpu-search-1.0",
        "manifest_schema": "gpu-search-manifest-1.1",
        "store_id": "internal_gpu_wiki",
        "index_digest": "sha256:" + "a" * 64,
        "shards": {"kernel_wiki": "kernel_wiki.json"},
    }
    (root / "search_index/index.json").write_text(json.dumps(manifest))
    (root / "search_index/kernel_wiki.json").write_text(
        json.dumps({"records": {_RECORD_ID: _RECORD}})
    )
    (root / "search_index/wiki_governance.json").write_text('{"records": {}}')
    return root


def _index(root: Path) -> CorpusIndex:
    return CorpusIndex(
        root,
        python_executable=Path(sys.executable),
        agent_cli=None,
        query_timeout_seconds=None,
        max_concurrent_queries=2,
        max_results=None,
        max_response_bytes=100_000,
    )


def _query() -> KnowledgeQueryV1:
    return KnowledgeQueryV1(
        campaign_id="campaign_" + "1" * 32,
        lineage_id="lineage_" + "2" * 32,
        epoch_id="epoch_" + "3" * 32,
        epoch_number=1,
        attempt_id="attempt_" + "4" * 32,
        branch="active",
        attempt_ordinal=1,
        kernel_agent_revision_id="agentrev_" + "5" * 32,
        operator="flash-attn-fp8",
        dsl="cuda",
        hardware_target="ZW-M890P (compatibility arch=sm_89)",
        evaluation_contract_digest="sha256:" + "6" * 64,
        epoch_evidence_checkpoint_digest="sha256:" + "7" * 64,
        attempt_evidence_digest="sha256:" + "8" * 64,
        query="N32 interleave 的适用边界是什么? Do not conflate PPU with NVIDIA Ada.",
    )


@pytest.mark.anyio
async def test_native_envelope_is_runtime_compatible_and_full_question_is_preserved(
    indexed_corpus: Path,
    tmp_path: Path,
) -> None:
    settings = LocalWikiSettings(
        host="127.0.0.1",
        port=8091,
        reference_root=indexed_corpus,
        store_root=tmp_path / "state",
        database=tmp_path / "wiki.sqlite",
        python_executable=Path(sys.executable),
        agent_cli="claude",
        query_timeout_seconds=20,
        max_results=5,
        max_request_bytes=100_000,
        max_response_bytes=100_000,
    )
    app = build_application(settings, {})
    sent: list[dict[str, Any]] = []
    request = _query()

    async def receive() -> dict[str, Any]:
        return {"type": "http.request", "body": request.model_dump_json().encode()}

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)

    try:
        await app(
            {"type": "http", "method": "POST", "path": "/v1/knowledge/query", "headers": []},
            receive,
            send,
        )
        assert sent[0]["status"] == 200
        response = KnowledgeSnapshotResponseV1.model_validate_json(sent[1]["body"])
        assert response.content == {
            "query_id": "native-query-001",
            "records": {_RECORD_ID: _RECORD},
            "notes": ["generation-reference; not a measured product result"],
        }
        invocation = json.loads((settings.store_root / "invocation.json").read_text())
        assert invocation["question"] == request.query
        assert invocation["agent_cli"] == "claude"
        assert invocation["timeout"] == "20"
        assert invocation["max_records"] == "5"
        assert not (settings.store_root / "kernel_wiki/records/index.json").exists()
        app._index.check_health()
    finally:
        app.close()


@pytest.mark.parametrize(
    "missing",
    [
        *(f"tools/{name}" for name in _TOOLS),
        "search_index/index.json",
        "search_index/kernel_wiki.json",
        "search_index/wiki_governance.json",
    ],
)
def test_native_readiness_detects_missing_dependencies_without_legacy_fallback(
    indexed_corpus: Path,
    missing: str,
) -> None:
    index = _index(indexed_corpus)
    for source in ("kernel_wiki", "hardware_wiki"):
        legacy = indexed_corpus / source / "records/index.json"
        legacy.parent.mkdir(parents=True)
        legacy.write_text("{}")
    (indexed_corpus / missing).unlink()
    with pytest.raises(ValueError, match="GPU Wiki"):
        index.check_health()
    with pytest.raises(ValueError, match="GPU Wiki"):
        _index(indexed_corpus)


@pytest.mark.parametrize("relative", ["../outside.json", "/tmp/outside.json", ""])
def test_manifest_cannot_reference_shards_outside_the_store(
    indexed_corpus: Path,
    relative: str,
) -> None:
    manifest = indexed_corpus / "search_index/index.json"
    content = json.loads(manifest.read_text())
    content["shards"]["kernel_wiki"] = relative
    manifest.write_text(json.dumps(content))
    with pytest.raises(ValueError, match="unsafe shard path"):
        _index(indexed_corpus)


def test_symlinked_shard_is_not_ready(indexed_corpus: Path, tmp_path: Path) -> None:
    shard = indexed_corpus / "search_index/kernel_wiki.json"
    outside = tmp_path / "outside.json"
    shard.rename(outside)
    shard.symlink_to(outside)
    with pytest.raises(ValueError, match="incomplete"):
        _index(indexed_corpus)


def test_unsupported_index_manifest_is_not_ready(indexed_corpus: Path) -> None:
    manifest = indexed_corpus / "search_index/index.json"
    content = json.loads(manifest.read_text())
    content["manifest_schema"] = "future-schema"
    manifest.write_text(json.dumps(content))
    with pytest.raises(ValueError, match="incompatible"):
        _index(indexed_corpus)


@pytest.mark.parametrize(
    "changed",
    [
        *(f"tools/{name}" for name in _TOOLS),
        "search_index/index.json",
        "search_index/kernel_wiki.json",
        "search_index/wiki_governance.json",
    ],
)
def test_native_revision_tracks_query_code_and_data_without_trusting_manifest_digest(
    indexed_corpus: Path,
    changed: str,
) -> None:
    index = _index(indexed_corpus)
    original = index._revision()
    path = indexed_corpus / changed
    path.write_text(path.read_text() + "\n")
    assert index._revision() != original


@pytest.mark.parametrize(
    "relative",
    ["profile_evidence.json", "governance_revisions/revision-1/governance.json"],
)
def test_native_revision_tracks_optional_evidence_and_governance_archives(
    indexed_corpus: Path,
    relative: str,
) -> None:
    index = _index(indexed_corpus)
    before = index._revision()
    path = indexed_corpus / "search_index" / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{}")
    added = index._revision()
    assert added != before
    path.write_text('{"revision": 2}')
    assert index._revision() != added
    path.unlink()
    assert index._revision() == before


def test_archived_governance_keeps_native_store_ready(indexed_corpus: Path) -> None:
    archive = indexed_corpus / "search_index/governance_revisions/revision-1/governance.json"
    archive.parent.mkdir(parents=True)
    (indexed_corpus / "search_index/wiki_governance.json").rename(archive)
    _index(indexed_corpus).check_health()


def test_unchanged_native_files_are_not_rehashed(
    indexed_corpus: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index = _index(indexed_corpus)
    original = index._revision()
    actual_file_digest = hashlib.file_digest
    hashed: list[str] = []

    def track_digest(fileobj: Any, digest: str) -> Any:
        hashed.append(fileobj.name)
        return actual_file_digest(fileobj, digest)

    monkeypatch.setattr(hashlib, "file_digest", track_digest)
    assert index._revision() == original
    assert hashed == []
    shard = indexed_corpus / "search_index/kernel_wiki.json"
    shard.write_text(shard.read_text() + "\n")
    assert index._revision() != original
    assert hashed == [str(shard)]


def test_native_query_rejects_snapshot_if_store_changes_during_execution(
    indexed_corpus: Path,
) -> None:
    tool = indexed_corpus / "tools/query_nl.py"
    tool.write_text(
        tool.read_text()
        + "(root / 'search_index/wiki_governance.json').write_text('{\"updated\": true}')\n"
    )
    with pytest.raises(GpuWikiQueryError, match="Store changed"):
        _index(indexed_corpus).query(_query())


def test_unavailable_native_dependency_is_reported_as_upstream_failure(
    indexed_corpus: Path,
) -> None:
    index = _index(indexed_corpus)
    (indexed_corpus / "search_index/kernel_wiki.json").unlink()
    with pytest.raises(GpuWikiQueryError, match="Store is unavailable"):
        index.query(_query())
