"""Minimal ASGI service implementing the external Atrex GPU Wiki API."""

from __future__ import annotations

import hashlib
import hmac
import json
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

import anyio
from pydantic import ValidationError

from .config import LocalWikiSettings
from .models import (
    JsonValue,
    KnowledgeQueryV1,
    KnowledgeSnapshotResponseV1,
    canonical_json_bytes,
)
from .retrieval import CorpusIndex, GpuWikiQueryError
from .store import LocalWikiStore
from .ui import BROWSER_UI
from .upstream import synchronize_store

AsgiReceive = Callable[[], Awaitable[dict[str, Any]]]
AsgiSend = Callable[[dict[str, Any]], Awaitable[None]]


class LocalWikiApplication:
    """Own the local corpus projection and query observation store."""

    def __init__(
        self,
        index: CorpusIndex,
        store: LocalWikiStore,
        *,
        bearer_token: str | None,
        max_request_bytes: int,
    ) -> None:
        self._index = index
        self._store = store
        self._bearer_token = bearer_token
        self._max_request_bytes = max_request_bytes
        self._closed = False

    async def __call__(
        self,
        scope: Mapping[str, object],
        receive: AsgiReceive,
        send: AsgiSend,
    ) -> None:
        if scope.get("type") == "lifespan":
            await self._lifespan(receive, send)
            return
        path = scope.get("path")
        if scope.get("type") != "http":
            await _json_response(send, 404, {"error": "not_found"})
            return
        if path in {"/", "/ui"}:
            if scope.get("method") != "GET":
                await _json_response(send, 405, {"error": "method_not_allowed"})
                return
            await _html_response(send, 200, BROWSER_UI)
            return
        if path == "/healthz":
            await self._health(scope, send)
            return
        if path == "/readyz":
            await self._ready(scope, send)
            return
        if path != "/v1/knowledge/query":
            await _json_response(send, 404, {"error": "not_found"})
            return
        if scope.get("method") != "POST":
            await _json_response(send, 405, {"error": "method_not_allowed"})
            return
        headers = _headers(scope)
        if not self._authorized(headers):
            await _json_response(send, 401, {"error": "unauthorized"})
            return
        try:
            body = await _read_body(receive, self._max_request_bytes)
            await self._query(body, send)
        except GpuWikiQueryError as error:
            await _json_response(send, 503, {"error": "upstream_unavailable", "detail": str(error)})
        except (json.JSONDecodeError, UnicodeDecodeError, ValidationError, ValueError) as error:
            await _json_response(send, 400, {"error": "invalid_request", "detail": str(error)})

    async def _query(self, body: bytes, send: AsgiSend) -> None:
        request = KnowledgeQueryV1.model_validate_json(body)
        result = await anyio.to_thread.run_sync(self._index.query, request)
        content_digest = "sha256:" + hashlib.sha256(
            canonical_json_bytes(result.content)
        ).hexdigest()
        request_json = canonical_json_bytes(request.model_dump(mode="json"))
        request_digest = hashlib.sha256(request_json).hexdigest()
        snapshot_material = f"{request_digest}\0{result.revision}\0{content_digest}".encode()
        snapshot_id = "localwiki_" + hashlib.sha256(snapshot_material).hexdigest()
        response = KnowledgeSnapshotResponseV1(
            snapshot_id=snapshot_id,
            content_digest=content_digest,
            content=result.content,
        )
        response_body = canonical_json_bytes(response.model_dump(mode="json"))
        self._store.record_query(snapshot_id, request_digest, body, response_body)
        await _bytes_response(send, 200, response_body)

    def _authorized(self, headers: Mapping[str, str]) -> bool:
        if self._bearer_token is None:
            return True
        return hmac.compare_digest(
            headers.get("authorization", ""),
            f"Bearer {self._bearer_token}",
        )

    @staticmethod
    async def _health(scope: Mapping[str, object], send: AsgiSend) -> None:
        if scope.get("method") != "GET":
            await _json_response(send, 405, {"error": "method_not_allowed"})
            return
        await _json_response(send, 200, {"status": "ok"})

    async def _ready(self, scope: Mapping[str, object], send: AsgiSend) -> None:
        if scope.get("method") != "GET":
            await _json_response(send, 405, {"error": "method_not_allowed"})
            return
        try:
            self._index.check_health()
            self._store.check_health()
        except Exception:
            await _json_response(send, 503, {"status": "unavailable"})
            return
        await _json_response(send, 200, {"status": "ready"})

    async def _lifespan(self, receive: AsgiReceive, send: AsgiSend) -> None:
        while True:
            message = await receive()
            if message.get("type") == "lifespan.startup":
                await send({"type": "lifespan.startup.complete"})
            elif message.get("type") == "lifespan.shutdown":
                self.close()
                await send({"type": "lifespan.shutdown.complete"})
                return
            else:
                raise ValueError("unexpected ASGI lifespan message")

    def close(self) -> None:
        """Close owned resources exactly once."""
        if not self._closed:
            self._closed = True
            self._store.close()


def build_application(
    settings: LocalWikiSettings,
    environment: Mapping[str, str],
) -> LocalWikiApplication:
    """Build the configured local Wiki and resolve its optional credential."""
    token = None
    if settings.bearer_token_env is not None:
        token = environment.get(settings.bearer_token_env)
        if not token:
            raise ValueError(f"missing environment variable: {settings.bearer_token_env}")
    synchronize_store(settings.reference_root, settings.store_root)
    store = LocalWikiStore(settings.database)
    index = CorpusIndex(
        settings.store_root,
        python_executable=settings.python_executable,
        agent_cli=settings.agent_cli,
        query_timeout_seconds=settings.query_timeout_seconds,
        max_concurrent_queries=settings.max_concurrent_queries,
        max_results=settings.max_results,
        max_response_bytes=settings.max_response_bytes,
        operator_families=settings.operator_families,
    )
    return LocalWikiApplication(
        index,
        store,
        bearer_token=token,
        max_request_bytes=settings.max_request_bytes,
    )


async def _read_body(receive: AsgiReceive, limit: int) -> bytes:
    chunks: list[bytes] = []
    size = 0
    more = True
    while more:
        message = await receive()
        if message.get("type") != "http.request":
            raise ValueError("unexpected ASGI request message")
        chunk = message.get("body", b"")
        if not isinstance(chunk, bytes):
            raise ValueError("ASGI body must be bytes")
        size += len(chunk)
        if size > limit:
            raise ValueError("request exceeds byte limit")
        chunks.append(chunk)
        more = bool(message.get("more_body", False))
    return b"".join(chunks)


def _headers(scope: Mapping[str, object]) -> dict[str, str]:
    raw = scope.get("headers", [])
    if not isinstance(raw, list):
        return {}
    return {
        key.decode("latin-1").casefold(): value.decode("latin-1")
        for key, value in raw
        if isinstance(key, bytes) and isinstance(value, bytes)
    }


async def _json_response(send: AsgiSend, status: int, value: JsonValue) -> None:
    await _bytes_response(send, status, canonical_json_bytes(value))


async def _bytes_response(send: AsgiSend, status: int, body: bytes) -> None:
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": [(b"content-type", b"application/json")],
        }
    )
    await send({"type": "http.response.body", "body": body})


async def _html_response(send: AsgiSend, status: int, body: bytes) -> None:
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": [
                (b"content-type", b"text/html; charset=utf-8"),
                (
                    b"content-security-policy",
                    b"default-src 'none'; style-src 'unsafe-inline'; "
                    b"script-src 'unsafe-inline'; connect-src 'self'",
                ),
                (b"x-content-type-options", b"nosniff"),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})
