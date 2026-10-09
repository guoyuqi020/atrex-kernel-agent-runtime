"""Preload a reviewed native Wiki implementation for isolated forked queries.

The owner must bind this object to one immutable Store revision, fork before
``run``, and replace the whole preload process when that revision changes.
Native code remains responsible for filtering, ranking, governance and output.
Only pure projections over the pinned snapshot are cached here.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import itertools
import json
import os
import signal
import subprocess
import sys
import traceback
import warnings
from pathlib import Path
from types import ModuleType
from typing import Any, cast

# internal_source/gpu-wiki at 2076c865cc6618d810cfe2bf09b4fc395536693e.
# Re-review the adapter before accepting a different native implementation.
PINNED_TOOL_DIGESTS = {
    "wiki_profile": "da3d09d0b6ce891d5036bba6203f55b311a3f97f741c301f0ca0aa32810c1011",
    "hardware_identity": "ae8d43618a66a195884729f4ba99c303e8babd1880f0b02f8dcf4f86a244341f",
    "agent_launch": "0c6bf92ef3f7b2b99dd4dfa99ac9c66b8041e527e1ccf1f12c4ceed7ebc7f7b7",
    "query": "cf079ae9a2528bed87483d3ff869a0ca28a830bcbfc8a710161414499069bf14",
    "operator_scope": "a9f2a17771be1a61831ceec52e1a2656870cd8f01b746493a7dcbf349bde2646",
    "query_nl": "9e59451669f0d6132071de0e22d83e2337b5fa285fee734c52e1fa4a2cd48467",
}


class NativePreloadError(ValueError):
    """The native implementation or a query does not match the pinned snapshot."""


def _load_modules(root: Path) -> dict[str, ModuleType]:
    """Load verified bytes with private references to all native dependencies."""
    sources: dict[str, tuple[Path, bytes]] = {}
    for name, expected in PINNED_TOOL_DIGESTS.items():
        path = root / "tools" / f"{name}.py"
        if path.is_symlink() or not path.is_file():
            raise NativePreloadError(f"GPU Wiki native tool is missing or unsafe: {name}")
        source = path.read_bytes()
        if hashlib.sha256(source).hexdigest() != expected:
            raise NativePreloadError(f"Unsupported GPU Wiki native tool revision: {name}")
        sources[name] = path, source
    modules: dict[str, ModuleType] = {}
    previous = {name: sys.modules.get(name) for name in sources}
    try:
        for name, (path, source) in sources.items():
            module = ModuleType(name)
            module.__file__ = str(path)
            modules[name] = module
            sys.modules[name] = module
            exec(compile(source, str(path), "exec"), module.__dict__)
    finally:
        for name, original in previous.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original
    return modules


def _store_key(store: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(
        store.get(key)
        for key in ("store_id", "wiki_revision", "index_schema", "shard_schema", "record_count")
    )


class NativePreload:
    """Resident immutable caches; ``run`` is only safe in an isolated process."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        # Exact hashes above are the API boundary for these dynamically loaded
        # modules; their private callable additions have no importable stubs.
        self.modules: dict[str, Any] = _load_modules(self.root)
        query = self.modules["query"]
        profile = self.modules["wiki_profile"]
        scopes = self.modules["operator_scope"]
        self._manifest_path, self._manifest = query.load_manifest(self.root)
        loaded = query.load_index(self.root)
        self._entries = loaded["records"]
        self._store = profile.store_identity(self._manifest)
        self._store_key = _store_key(self._store)
        self._governance: dict[str, Any] = query.load_governance(
            self._manifest_path.parent, self._store
        )
        self._known_sources = frozenset(self._manifest["shards"])
        self._indexes: dict[frozenset[str], dict[str, Any]] = {}

        # Native governance hashes full record bodies. Compute these exactly
        # once, before it adds its explicitly digest-excluded _management field.
        original_identity = profile.record_identity
        identities = {id(entry): original_identity(entry, self._store) for entry in self._entries}

        def identity(entry: dict[str, Any], store: dict[str, Any]) -> dict[str, Any]:
            self._require_store(store)
            return dict(self._cached(identities, entry))

        profile.__dict__["record_identity"] = identity
        original_management = query.management_for_entry
        management = {
            id(entry): original_management(entry, self._store, self._governance)
            for entry in self._entries
        }
        searchable = {id(entry): query.searchable(entry) for entry in self._entries}

        def managed(
            entry: dict[str, Any], store: dict[str, Any], governance: dict[str, Any]
        ) -> dict[str, Any]:
            self._require_store(store)
            if governance is not self._governance:
                raise NativePreloadError("GPU Wiki governance is outside the preloaded snapshot")
            # Native consumers only read this value or deepcopy it into output.
            return cast(dict[str, Any], self._cached(management, entry))

        query.__dict__.update(
            {
                "load_manifest": self._load_manifest,
                "load_index": self._load_index,
                "load_governance": self._load_governance,
                "management_for_entry": managed,
                "searchable": lambda entry: self._cached(searchable, entry),
            }
        )

        original_vocab = query.vocab
        vocabularies: dict[tuple[int, ...], dict[str, set[str]]] = {}

        def vocabulary(entries: list[dict[str, Any]]) -> dict[str, set[str]]:
            key = tuple(id(entry) for entry in entries)
            if key not in vocabularies:
                vocabularies[key] = original_vocab(entries)
            return vocabularies[key]

        query.__dict__["vocab"] = vocabulary
        # A query may select any governance zone. Prebuild every reviewed tier
        # combination so an unusual query also avoids repeating body digests.
        original_resolver = scopes.OperatorScopeResolver.from_store
        resolvers: dict[frozenset[str], Any] = {}
        resolver_warnings: dict[frozenset[str], list[warnings.WarningMessage]] = {}
        tiers = sorted(query.MANAGEMENT_TIERS)
        for count in range(len(tiers) + 1):
            for combination in itertools.combinations(tiers, count):
                selected = frozenset(combination)
                with warnings.catch_warnings(record=True) as emitted:
                    warnings.simplefilter("always")
                    resolver = original_resolver(self.root, management_tiers=set(selected))
                    resolver._ensure_documents()
                resolvers[selected] = resolver
                resolver_warnings[selected] = list(emitted)

        def from_store(_cls: type, root: Path, management_tiers: set[str] | None = None) -> Any:
            self._require_root(root)
            selected = frozenset(
                query.configured_management_tiers()
                if management_tiers is None
                else management_tiers
            )
            if selected not in resolvers:
                raise NativePreloadError("GPU Wiki requested an unsupported governance zone")
            for warning in resolver_warnings[selected]:
                warnings.warn(str(warning.message), warning.category, stacklevel=2)
            return resolvers[selected]

        scopes.OperatorScopeResolver.from_store = classmethod(from_store)
        # Freeze optional evidence too; other profile reads are operational
        # state (run identity, immutable query events), not corpus cache data.
        evidence_path = self.root / "search_index" / "profile_evidence.json"
        original_read_json = profile._read_json
        evidence = original_read_json(evidence_path)

        def read_json(path: Path) -> Any:
            if Path(path).resolve() == evidence_path:
                return evidence
            return original_read_json(path)

        profile.__dict__["_read_json"] = read_json
        self.modules["query_nl"].__dict__["_run_json"] = self._run_json

        # Warm the common source subsets and their default governance vocab.
        default_tiers = query.configured_management_tiers()
        source_sets = [self._known_sources, self._known_sources - {"microarch_wiki"}]
        source_sets.extend(frozenset({source}) for source in self._known_sources)
        for selected_sources in source_sets:
            document = self._load_index(self.root, set(selected_sources))
            vocabulary(
                [
                    entry
                    for entry in document["records"]
                    if management[id(entry)]["tier"] in default_tiers
                ]
            )

    @staticmethod
    def _cached(cache: dict[int, Any], entry: dict[str, Any]) -> Any:
        try:
            return cache[id(entry)]
        except KeyError as error:
            raise NativePreloadError("GPU Wiki record is outside the preloaded snapshot") from error

    def _require_store(self, store: dict[str, Any]) -> None:
        if _store_key(store) != self._store_key:
            raise NativePreloadError("GPU Wiki Store identity differs from the preloaded snapshot")

    def _require_root(self, root: Path) -> None:
        candidate = Path(root).resolve()
        if candidate.name in {"kernel_wiki", "hardware_wiki", "3rd_repo_wiki", "microarch_wiki"}:
            candidate = candidate.parent
        if candidate != self.root:
            raise NativePreloadError("GPU Wiki query targets a different Store")

    def _load_manifest(self, root: Path) -> tuple[Path, dict[str, Any]]:
        self._require_root(root)
        return self._manifest_path, dict(self._manifest)

    def _load_index(self, root: Path, sources: set[str] | None = None) -> dict[str, Any]:
        self._require_root(root)
        selected = self._known_sources if sources is None else frozenset(sources)
        unknown = selected - self._known_sources
        if unknown:
            self.modules["query"].die(f"unknown-source {', '.join(sorted(unknown))!r}", 2)
        if selected not in self._indexes:
            records = [entry for entry in self._entries if entry["source"] in selected]
            self._indexes[selected] = {
                **self._manifest,
                "records": records,
                "loaded_sources": sorted(selected),
                "loaded_count": len(records),
            }
        # Native load_index decorates the manifest. Returning a fresh shallow
        # envelope keeps source selections separate without copying large bodies.
        return dict(self._indexes[selected])

    def _load_governance(self, root: Path, store: dict[str, Any]) -> dict[str, Any]:
        if Path(root).resolve() != self.root / "search_index":
            raise NativePreloadError("GPU Wiki governance targets a different Store")
        self._require_store(store)
        return self._governance

    def _run_json(self, argv: list[str], profile_stage: str = "") -> tuple[int, object | None, str]:
        if len(argv) < 2 or Path(argv[1]).resolve() != self.root / "tools" / "query.py":
            raise NativePreloadError("GPU Wiki attempted an unsupported query subprocess")
        stage_name = self.modules["wiki_profile"].QUERY_STAGE_ENV
        previous = os.environ.get(stage_name)
        stdout, stderr = io.StringIO(), io.StringIO()
        try:
            if profile_stage:
                os.environ[stage_name] = profile_stage
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                try:
                    code = int(self.modules["query"].main(argv[2:]))
                except SystemExit as error:
                    code = self._exit_code(error, stderr)
        finally:
            self._restore_environment(stage_name, previous)
        payload = None
        if stdout.getvalue().strip():
            with contextlib.suppress(json.JSONDecodeError):
                payload = json.loads(stdout.getvalue())
        return code, payload, stderr.getvalue().strip()

    @staticmethod
    def _restore_environment(name: str, previous: str | None) -> None:
        if previous is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = previous

    @staticmethod
    def _exit_code(error: SystemExit, stderr: io.StringIO) -> int:
        if error.code is None:
            return 0
        if isinstance(error.code, int):
            return error.code
        print(error.code, file=stderr)
        return 1

    def run(self, argv: list[str]) -> tuple[int, str, str]:
        """Run native query_nl arguments, excluding interpreter/script prefixes."""
        stdout, stderr = io.StringIO(), io.StringIO()
        original_popen = subprocess.Popen
        children: list[tuple[subprocess.Popen[Any], bool]] = []

        def tracked_popen(*args: Any, **kwargs: Any) -> subprocess.Popen[Any]:
            process = original_popen(*args, **kwargs)
            children.append((process, bool(kwargs.get("start_new_session"))))
            return process

        # Native agent_launch deliberately creates a separate session. Track it
        # without changing its communicate/timeout protocol, so cancellation of
        # the forked query cannot orphan the model CLI in that separate group.
        subprocess.Popen = tracked_popen  # type: ignore[misc, assignment]
        try:
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                try:
                    code = int(self.modules["query_nl"].main(argv))
                except SystemExit as error:
                    code = self._exit_code(error, stderr)
                except Exception:
                    traceback.print_exc(file=stderr)
                    code = 1
        finally:
            subprocess.Popen = original_popen  # type: ignore[misc]
            for process, independent_group in reversed(children):
                self._reap_child(process, independent_group)
        return code, stdout.getvalue(), stderr.getvalue()

    @staticmethod
    def _reap_child(process: subprocess.Popen[Any], independent_group: bool) -> None:
        def send(sig: signal.Signals) -> None:
            try:
                if independent_group:
                    os.killpg(process.pid, sig)
                elif process.poll() is None:
                    process.send_signal(sig)
            except ProcessLookupError:
                pass

        send(signal.SIGTERM)
        with contextlib.suppress(subprocess.TimeoutExpired):
            process.wait(timeout=0.25)
        # Also remove descendants whose process-group leader has already exited.
        send(signal.SIGKILL)
        with contextlib.suppress(subprocess.TimeoutExpired):
            process.wait(timeout=1)
