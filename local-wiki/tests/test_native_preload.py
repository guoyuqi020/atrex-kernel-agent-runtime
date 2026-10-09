"""Resident caches preserve native contracts without model or network calls."""

from __future__ import annotations

import contextlib
import io
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from atrex_local_wiki import native_preload
from atrex_local_wiki.native_preload import NativePreload, NativePreloadError


@pytest.fixture
def synthetic_native(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Small stand-in exposes the reviewed callable seams, not retrieval logic."""
    root = tmp_path / "store"
    (root / "search_index").mkdir(parents=True)
    calls: Counter[str] = Counter()
    manifest = {
        "store_id": "internal_gpu_wiki",
        "index_digest": "sha256:test",
        "shards": {"kernel_wiki": "kernel.json", "docs": "docs.json"},
    }
    entries = [
        {"id": "kernel", "source": "kernel_wiki", "served": {"text": "kernel body"}},
        {"id": "doc", "source": "docs", "served": {"text": "doc body"}},
    ]
    query = ModuleType("query")
    profile = ModuleType("wiki_profile")
    scopes = ModuleType("operator_scope")
    nl = ModuleType("query_nl")

    def manifest_loader(_root: Path) -> tuple[Path, dict[str, Any]]:
        calls["manifest"] += 1
        return root / "search_index/index.json", dict(manifest)

    def index_loader(_root: Path) -> dict[str, Any]:
        calls["index"] += 1
        return {**manifest, "records": entries}

    def identity(entry: dict[str, Any], store: dict[str, Any]) -> dict[str, Any]:
        calls["identity"] += 1
        assert "_management" not in entry
        return {"record_digest": entry["id"], **store}

    def management(entry: dict[str, Any], store: dict[str, Any], _governance: Any) -> Any:
        calls["management"] += 1
        return {"tier": "core", "identity": profile.record_identity(entry, store)}

    def searchable(entry: dict[str, Any]) -> tuple[str, str, str]:
        calls["searchable"] += 1
        return entry["id"], entry["id"], entry["served"]["text"]

    def die(message: str, code: int = 2) -> None:
        print(message, file=sys.stderr)
        raise SystemExit(code)

    query.load_manifest = manifest_loader
    query.load_index = index_loader
    query.load_governance = lambda *_: {"revision": "governance-1"}
    query.management_for_entry = management
    query.searchable = searchable
    query.vocab = lambda rows: {"source": {entry["source"] for entry in rows}}
    query.MANAGEMENT_TIERS = {"core", "faded", "isolated"}
    query.configured_management_tiers = lambda: {"core", "faded"}
    query.die = die
    profile.record_identity = identity
    profile.store_identity = lambda doc: {
        "store_id": doc["store_id"],
        "wiki_revision": doc["index_digest"],
    }
    profile._read_json = lambda _: None
    profile.QUERY_STAGE_ENV = "ATREX_WIKI_QUERY_STAGE"

    class Resolver:
        @classmethod
        def from_store(cls, _root: Path, management_tiers: set[str]) -> Resolver:
            return cls()

        def _ensure_documents(self) -> None:
            pass

    scopes.OperatorScopeResolver = Resolver

    def query_main(argv: list[str]) -> int:
        source = argv[0] if argv else "kernel_wiki"
        doc = query.load_index(root, {source})
        store = profile.store_identity(manifest)
        governance = query.load_governance(root / "search_index", store)
        records = {}
        for entry in doc["records"]:
            entry["_management"] = query.management_for_entry(entry, store, governance)
            records[entry["id"]] = {
                "identity": profile.record_identity(entry, store),
                "text": query.searchable(entry),
                "management": entry["_management"],
            }
        print(json.dumps({"records": records, "stage": os.environ.get(profile.QUERY_STAGE_ENV)}))
        return 0

    def nl_main(argv: list[str]) -> int:
        code, value, error = nl._run_json(
            [sys.executable, str(root / "tools/query.py"), *argv], "kernel:exact"
        )
        if error:
            print(error, file=sys.stderr)
        print(json.dumps(value))
        return code

    query.main = query_main
    nl.main = nl_main
    modules = {"query": query, "wiki_profile": profile, "operator_scope": scopes, "query_nl": nl}
    monkeypatch.setattr(native_preload, "_load_modules", lambda _: modules)
    return SimpleNamespace(root=root, calls=calls, entries=entries, modules=modules)


def test_preload_avoids_repeating_body_work_and_source_contamination(
    synthetic_native: SimpleNamespace,
) -> None:
    cached = NativePreload(synthetic_native.root)
    counts = synthetic_native.calls.copy()
    for source, expected in [("kernel_wiki", "kernel"), ("docs", "doc"), ("kernel_wiki", "kernel")]:
        code, output, errors = cached.run([source])
        assert code == 0, errors
        payload = json.loads(output)
        assert set(payload["records"]) == {expected}
        assert payload["stage"] == "kernel:exact"
    assert synthetic_native.calls == counts
    assert counts["index"] == 1
    assert counts["identity"] == counts["management"] == counts["searchable"] == 2
    assert all("_management" in entry for entry in synthetic_native.entries)
    _, manifest = cached.modules["query"].load_manifest(synthetic_native.root)
    assert "records" not in manifest


def test_unknown_source_preserves_native_failure_and_stage_environment(
    synthetic_native: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ATREX_WIKI_QUERY_STAGE", "parent-stage")
    cached = NativePreload(synthetic_native.root)
    code, _, error = cached.run(["missing"])
    assert code == 2
    assert "unknown-source" in error
    assert os.environ["ATREX_WIKI_QUERY_STAGE"] == "parent-stage"


def test_preload_rejects_another_store_or_subprocess(synthetic_native: SimpleNamespace) -> None:
    cached = NativePreload(synthetic_native.root)
    with pytest.raises(NativePreloadError, match="different Store"):
        cached.modules["query"].load_index(synthetic_native.root.parent)
    with pytest.raises(NativePreloadError, match="unsupported query subprocess"):
        cached._run_json([sys.executable, "/other/tools/query.py"])


def test_stage_environment_restored_on_native_exception(
    synthetic_native: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ATREX_WIKI_QUERY_STAGE", raising=False)
    cached = NativePreload(synthetic_native.root)

    def failed(_argv: list[str]) -> int:
        assert os.environ["ATREX_WIKI_QUERY_STAGE"] == "test-stage"
        raise RuntimeError("native failure")

    cached.modules["query"].main = failed
    with pytest.raises(RuntimeError, match="native failure"):
        cached._run_json(
            [sys.executable, str(synthetic_native.root / "tools/query.py")], "test-stage"
        )
    assert "ATREX_WIKI_QUERY_STAGE" not in os.environ


def test_query_cleanup_reaps_independent_model_process(synthetic_native: SimpleNamespace) -> None:
    cached = NativePreload(synthetic_native.root)
    children = []
    original_popen = subprocess.Popen

    def abandoned(_argv: list[str]) -> int:
        children.append(
            subprocess.Popen(
                [sys.executable, "-c", "import time; time.sleep(60)"],
                start_new_session=True,
            )
        )
        raise RuntimeError("query cancelled")

    cached.modules["query_nl"].main = abandoned
    code, _, error = cached.run([])
    assert code == 1
    assert "query cancelled" in error
    assert children[0].poll() is not None
    assert subprocess.Popen is original_popen


def test_unknown_native_revision_fails_before_execution(tmp_path: Path) -> None:
    root = tmp_path / "unknown"
    (root / "tools").mkdir(parents=True)
    marker = tmp_path / "must-not-exist"
    (root / "tools/wiki_profile.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).touch()\n"
    )
    with pytest.raises(NativePreloadError, match=r"Unsupported.*revision"):
        NativePreload(root)
    assert not marker.exists()


def test_optional_real_corpus_matches_native_with_no_model(monkeypatch: pytest.MonkeyPatch) -> None:
    root = Path(__file__).resolve().parents[1] / "corpus/internal_gpu_wiki"
    if not (root / "tools/query_nl.py").is_file():
        pytest.skip("private indexed corpus is not installed")
    cached = NativePreload(root)

    def no_model(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("this deterministic regression must not call a model")

    monkeypatch.setattr(cached.modules["agent_launch"], "run_json", no_model)
    request = (
        "Target hardware ZW-M890P, DSL cuda. Optimize operator flash_attention "
        "and retrieve techniques and pitfalls."
    )
    args = [request, "--store-root", str(root), "--max-bytes", "131072"]
    code, output, error = cached.run(args)
    assert code == 0, error
    accelerated = json.loads(output)
    # Restore only the native subprocess boundary for an independent low-level
    # retrieval comparison; a child process starts with uncached native code.
    native_nl = native_preload._load_modules(root)["query_nl"]
    monkeypatch.setattr(native_nl.agent_launch, "run_json", no_model)
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        assert native_nl.main(args) == 0
    baseline = json.loads(stdout.getvalue())
    assert accelerated["records"] == baseline["records"]
    assert accelerated["notes"] == baseline["notes"]
