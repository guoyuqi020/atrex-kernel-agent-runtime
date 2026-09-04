"""Adapter over the pinned GPU Wiki's public query and record interfaces."""

from __future__ import annotations

import hashlib
import json
import subprocess
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from .models import JsonValue, KnowledgeQueryV1


class GpuWikiQueryError(ValueError):
    """The pinned GPU Wiki rejected or failed one natural-language query."""


@dataclass(frozen=True, slots=True)
class GpuWikiQueryResult:
    """One upstream query result paired with the exact mutable Store revision."""

    content: dict[str, JsonValue]
    revision: str


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
    ) -> None:
        self._root = root.resolve()
        self._python = python_executable.resolve()
        self._agent_cli = agent_cli
        self._query_timeout_seconds = query_timeout_seconds
        self._query_slots = threading.BoundedSemaphore(max_concurrent_queries)
        self._max_results = max_results
        self._max_response_bytes = max_response_bytes
        self._query_tool = self._root / "tools" / "query_nl.py"
        self._kernel_index = self._root / "kernel_wiki" / "records" / "index.json"
        self._hardware_index = self._root / "hardware_wiki" / "records" / "index.json"
        self._validate_layout()

    def check_health(self) -> None:
        """Fail when the pinned tools or either public record store disappear."""
        self._validate_layout()

    def query(self, request: KnowledgeQueryV1) -> GpuWikiQueryResult:
        """Return the public ``query_nl.py`` envelope without rewriting its contents."""
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
            try:
                process = subprocess.run(
                    command,
                    check=False,
                    capture_output=True,
                    timeout=outer_timeout,
                )
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
            return GpuWikiQueryResult(_json_object(value), self._revision())

    def _validate_layout(self) -> None:
        required = (self._query_tool, self._kernel_index, self._hardware_index)
        missing = [str(path) for path in required if path.is_symlink() or not path.is_file()]
        if missing:
            raise ValueError(f"GPU Wiki public interface is incomplete: {missing}")
        if not self._python.is_absolute() or not self._python.is_file():
            raise ValueError("GPU Wiki Python executable must be an existing absolute file")

    def _revision(self) -> str:
        digest = hashlib.sha256()
        for path in (self._query_tool, self._kernel_index, self._hardware_index):
            digest.update(path.relative_to(self._root).as_posix().encode())
            digest.update(b"\0")
            digest.update(path.read_bytes())
            digest.update(b"\0")
        return "sha256:" + digest.hexdigest()


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
