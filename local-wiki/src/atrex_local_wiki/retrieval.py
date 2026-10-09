"""Adapter over the selected GPU Wiki's native natural-language query interface."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import threading
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from .models import JsonValue, KnowledgeQueryV1
from .preloaded_worker import PreloadedWorker, PreloadedWorkerError

_INDEXED_TOOLS = (
    "query_nl.py",
    "query.py",
    "agent_launch.py",
    "hardware_identity.py",
    "operator_scope.py",
    "wiki_profile.py",
)
type _FileSignature = tuple[int, int, int, int, int]


class GpuWikiQueryError(ValueError):
    """The pinned GPU Wiki rejected or failed one natural-language query."""


@dataclass(frozen=True, slots=True)
class GpuWikiQueryResult:
    """One upstream query result paired with the exact mutable Store revision."""

    content: dict[str, JsonValue]
    revision: str


@dataclass
class _Generation:
    revision: str
    worker: PreloadedWorker
    active: int = 0


class CorpusIndex:
    """Execute the pinned GPU Wiki implementation without reimplementing retrieval."""

    def __init__(
        self,
        root: Path,
        *,
        python_executable: Path,
        agent_cli: str | None,
        query_timeout_seconds: int | None,
        max_concurrent_queries: int,
        max_results: int | None,
        max_response_bytes: int,
        indexed_execution: str = "subprocess",
    ) -> None:
        self._root = root.resolve()
        self._python = python_executable.resolve()
        self._agent_cli = agent_cli
        self._query_timeout_seconds = query_timeout_seconds
        self._query_slots = threading.BoundedSemaphore(max_concurrent_queries)
        self._max_results = max_results
        self._max_response_bytes = max_response_bytes
        self._max_concurrent_queries = max_concurrent_queries
        self._preloaded = indexed_execution == "preloaded"
        self._generation: _Generation | None = None
        self._retired: list[_Generation] = []
        self._worker_lock = threading.Lock()
        self._closed = False
        self._query_tool = self._root / "tools" / "query_nl.py"
        self._kernel_index = self._root / "kernel_wiki" / "records" / "index.json"
        self._hardware_index = self._root / "hardware_wiki" / "records" / "index.json"
        self._search_index = self._root / "search_index"
        # An incomplete indexed Store must fail rather than silently querying its
        # older kernel/hardware record directories with the public contract.
        self._indexed = self._search_index.exists() or (self._root / "tools/query.py").exists()
        self._digest_cache: dict[Path, tuple[_FileSignature, bytes]] = {}
        self._revision_lock = threading.Lock()
        self._validate_layout()
        if self._preloaded:
            if not self._indexed:
                raise ValueError("GPU Wiki preloading requires an indexed native Store")
            # Build all derived data before the HTTP service advertises readiness.
            with self._lease_worker(self._query_revision()):
                pass

    def check_health(self) -> None:
        """Fail when a dependency of the selected native interface disappears."""
        self._validate_layout()
        if self._preloaded:
            with self._worker_lock:
                if self._closed or self._generation is None:
                    raise ValueError("GPU Wiki preloaded index is unavailable")
                self._generation.worker.check_health()

    def query(self, request: KnowledgeQueryV1) -> GpuWikiQueryResult:
        """Return the native ``query_nl.py`` envelope without rewriting its contents."""
        description = (
            f"Target hardware reported by the runtime: {request.hardware_target}. "
            f"Required DSL: {request.dsl}. Operator: {request.operator}. "
            f"Optimization question: {request.query}"
        )
        command: list[str] = [
            str(self._python),
            str(self._query_tool),
            description,
            "--store-root",
            str(self._root),
            "--max-bytes",
            str(self._max_response_bytes),
        ]
        if self._agent_cli is not None:
            command.extend(("--agent-cli", self._agent_cli))
        if self._query_timeout_seconds is not None:
            command.extend(("--timeout", str(self._query_timeout_seconds)))
        if self._max_results is not None:
            command.extend(("--max-records", str(self._max_results)))
        outer_timeout = (
            self._query_timeout_seconds + 30 if self._query_timeout_seconds is not None else None
        )
        with self._query_slots:
            before = self._query_revision() if self._indexed else None
            try:
                if self._preloaded:
                    assert before is not None
                    with self._lease_worker(before) as worker:
                        process = worker.query(command[2:], outer_timeout)
                else:
                    process = subprocess.run(
                        command,
                        check=False,
                        capture_output=True,
                        timeout=outer_timeout,
                    )
            except PreloadedWorkerError as error:
                raise GpuWikiQueryError(str(error)) from error
            except subprocess.TimeoutExpired as error:
                raise GpuWikiQueryError("GPU Wiki natural-language query timed out") from error
            stdout = process.stdout[: self._max_response_bytes + 1]
            if len(stdout) > self._max_response_bytes:
                raise GpuWikiQueryError("GPU Wiki response exceeded the configured byte limit")
            if process.returncode != 0:
                detail = process.stderr.decode("utf-8", errors="replace").strip()[-1000:]
                raise GpuWikiQueryError(
                    f"GPU Wiki query failed with exit {process.returncode}: {detail}"
                )
            try:
                value: object = json.loads(stdout)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise GpuWikiQueryError("GPU Wiki query returned invalid JSON") from error
            if not isinstance(value, dict) or set(value) != {"query_id", "records", "notes"}:
                raise GpuWikiQueryError("GPU Wiki query returned an incompatible envelope")
            if not isinstance(value["query_id"], str) or not value["query_id"]:
                raise GpuWikiQueryError("GPU Wiki query_id must be a nonempty string")
            if not isinstance(value.get("records"), dict) or not isinstance(
                value.get("notes"), list
            ):
                raise GpuWikiQueryError("GPU Wiki records/notes have incompatible types")
            revision = self._query_revision()
            if before is not None and before != revision:
                raise GpuWikiQueryError("GPU Wiki Store changed while the query was running; retry")
            return GpuWikiQueryResult(_json_object(value), revision)

    @contextmanager
    def _lease_worker(self, revision: str) -> Iterator[PreloadedWorker]:
        with self._worker_lock:
            if self._closed:
                raise GpuWikiQueryError("GPU Wiki index is closed")
            generation = self._generation
            if generation is not None:
                try:
                    generation.worker.check_health()
                except PreloadedWorkerError:
                    generation = None
            if generation is None or generation.revision != revision:
                worker = PreloadedWorker(
                    self._root,
                    self._python,
                    concurrency=self._max_concurrent_queries,
                    max_response_bytes=self._max_response_bytes,
                )
                try:
                    if self._query_revision() != revision:
                        raise GpuWikiQueryError("GPU Wiki Store changed during preloading; retry")
                except BaseException:
                    worker.close()
                    raise
                if self._generation is not None:
                    self._retired.append(self._generation)
                generation = _Generation(revision, worker)
                self._generation = generation
                self._close_retired()
            generation.active += 1
        try:
            yield generation.worker
        finally:
            with self._worker_lock:
                generation.active -= 1
                self._close_retired()

    def _close_retired(self) -> None:
        remaining = []
        for generation in self._retired:
            if generation.active:
                remaining.append(generation)
            else:
                generation.worker.close()
        self._retired = remaining

    def close(self) -> None:
        """Stop the native preload server and all of its active query children."""
        with self._worker_lock:
            self._closed = True
            for generation in [*self._retired, *([self._generation] if self._generation else [])]:
                generation.worker.close()
            self._retired.clear()
            self._generation = None

    def _validate_layout(self) -> None:
        required = self._interface_files()
        missing = [str(path) for path in required if not self._regular_store_file(path)]
        if missing:
            raise ValueError(f"GPU Wiki native interface is incomplete: {missing}")
        if not self._python.is_absolute() or not self._python.is_file():
            raise ValueError("GPU Wiki Python executable must be an existing absolute file")

    def _revision(self) -> str:
        if self._indexed:
            with self._revision_lock:
                digest = hashlib.sha256()
                for path in sorted(self._interface_files()):
                    digest.update(path.relative_to(self._root).as_posix().encode())
                    digest.update(b"\0")
                    digest.update(self._cached_file_digest(path))
                    digest.update(b"\0")
                return "sha256:" + digest.hexdigest()
        digest = hashlib.sha256()
        for path in (self._query_tool, self._kernel_index, self._hardware_index):
            digest.update(path.relative_to(self._root).as_posix().encode())
            digest.update(b"\0")
            digest.update(path.read_bytes())
            digest.update(b"\0")
        return "sha256:" + digest.hexdigest()

    def _query_revision(self) -> str:
        try:
            self._validate_layout()
            return self._revision()
        except (OSError, ValueError) as error:
            raise GpuWikiQueryError(f"GPU Wiki Store is unavailable: {error}") from error

    def _interface_files(self) -> tuple[Path, ...]:
        if not self._indexed:
            return self._query_tool, self._kernel_index, self._hardware_index
        manifest_path = self._search_index / "index.json"
        if not self._regular_store_file(manifest_path):
            raise ValueError(f"GPU Wiki search index is missing or unsafe: {manifest_path}")
        try:
            manifest = json.loads(manifest_path.read_bytes())
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("GPU Wiki search index manifest is unreadable") from error
        if (
            not isinstance(manifest, dict)
            or manifest.get("schema") != "gpu-search-1.0"
            or manifest.get("manifest_schema") != "gpu-search-manifest-1.1"
            or not isinstance(manifest.get("store_id"), str)
            or not manifest["store_id"].strip()
            or re.fullmatch(r"sha256:[0-9a-f]{64}", str(manifest.get("index_digest"))) is None
        ):
            raise ValueError("GPU Wiki search index manifest is incompatible")
        shards = manifest.get("shards")
        if not isinstance(shards, dict) or not shards:
            raise ValueError("GPU Wiki search index manifest has no shards")
        files = {self._root / "tools" / name for name in _INDEXED_TOOLS}
        files.add(manifest_path)
        for relative in shards.values():
            if (
                not isinstance(relative, str)
                or not relative
                or Path(relative).is_absolute()
                or ".." in Path(relative).parts
            ):
                raise ValueError("GPU Wiki search index has an unsafe shard path")
            files.add(self._search_index / relative)
        governance = self._search_index / "wiki_governance.json"
        archives = set((self._search_index / "governance_revisions").glob("*/governance.json"))
        if governance.exists() or governance.is_symlink():
            files.add(governance)
        elif not archives:
            # Without a governance projection the native implementation hides all
            # records. Do not advertise that empty Store as ready.
            raise ValueError("GPU Wiki search index has no governance projection")
        files.update(archives)
        evidence = self._search_index / "profile_evidence.json"
        if evidence.exists() or evidence.is_symlink():
            files.add(evidence)
        return tuple(files)

    def _regular_store_file(self, path: Path) -> bool:
        return path.is_file() and not any(
            parent.is_symlink() for parent in (path, *path.parents) if parent != self._root
        )

    def _cached_file_digest(self, path: Path) -> bytes:
        stat = path.stat()
        signature = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        cached = self._digest_cache.get(path)
        if cached is not None and cached[0] == signature:
            return cached[1]
        with path.open("rb") as stream:
            value = hashlib.file_digest(stream, "sha256").digest()
        after = path.stat()
        if signature != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise ValueError(f"GPU Wiki file changed while computing its revision: {path}")
        self._digest_cache[path] = signature, value
        return value


def _json_object(value: object) -> dict[str, JsonValue]:
    """Validate a JSON object without importing Runtime implementation models."""
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ValueError("GPU Wiki value must be a JSON object")
    try:
        encoded = json.dumps(value, ensure_ascii=False, allow_nan=False)
        normalized: object = json.loads(encoded)
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError("GPU Wiki value is not JSON-compatible") from error
    if not isinstance(normalized, dict):
        raise AssertionError("normalized GPU Wiki object changed type")
    return normalized


__all__ = ["CorpusIndex", "GpuWikiQueryError", "GpuWikiQueryResult"]
