"""HTTP client for the independent Atrex GPU Wiki service."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import httpx
from pydantic import SecretStr

from .models import KnowledgeQueryV1, KnowledgeSnapshotResponseV1


class KnowledgeUnavailableError(RuntimeError):
    """GPU Wiki is temporarily unavailable and the caller may retry explicitly."""


@dataclass(frozen=True, slots=True)
class GpuWikiHttpResponse:
    """Bounded HTTP response returned by a replaceable transport."""

    status_code: int
    body: bytes


class GpuWikiHttpTransport(Protocol):
    """Transport seam used by the strict GPU Wiki client."""

    async def post(
        self,
        path: str,
        body: bytes,
        headers: dict[str, str],
        *,
        timeout_seconds: float,
        max_response_bytes: int,
    ) -> GpuWikiHttpResponse:
        """Submit one bounded JSON request without interpreting its payload."""
        ...


class GpuWikiClient(Protocol):
    """Query the versioned knowledge service without exposing credentials to Workers."""

    async def query(self, request: KnowledgeQueryV1) -> KnowledgeSnapshotResponseV1:
        """Return one strict digest-verified knowledge snapshot."""
        ...


class HttpxGpuWikiTransport:
    """HTTPX transport that bounds the streamed response before JSON parsing."""

    def __init__(self, base_url: str) -> None:
        self._base_url = base_url.rstrip("/")

    async def post(
        self,
        path: str,
        body: bytes,
        headers: dict[str, str],
        *,
        timeout_seconds: float,
        max_response_bytes: int,
    ) -> GpuWikiHttpResponse:
        """POST bytes and stop reading as soon as the response bound is exceeded."""
        try:
            async with (
                httpx.AsyncClient(timeout=timeout_seconds) as client,
                client.stream(
                    "POST",
                    f"{self._base_url}{path}",
                    content=body,
                    headers=headers,
                ) as response,
            ):
                chunks: list[bytes] = []
                size = 0
                async for chunk in response.aiter_bytes():
                    size += len(chunk)
                    if size > max_response_bytes:
                        raise ValueError("GPU Wiki response exceeds byte limit")
                    chunks.append(chunk)
                return GpuWikiHttpResponse(response.status_code, b"".join(chunks))
        except httpx.TransportError as error:
            raise KnowledgeUnavailableError(f"GPU Wiki transport failed: {error}") from error


class HttpGpuWikiClient:
    """Strict API-version owner over a replaceable bounded HTTP transport."""

    def __init__(
        self,
        transport: GpuWikiHttpTransport,
        *,
        bearer_token: SecretStr | None,
        timeout_seconds: float,
        max_response_bytes: int,
    ) -> None:
        if timeout_seconds <= 0 or max_response_bytes <= 0:
            raise ValueError("GPU Wiki HTTP limits must be positive")
        self._transport = transport
        self._bearer_token = bearer_token
        self._timeout_seconds = timeout_seconds
        self._max_response_bytes = max_response_bytes

    async def query(self, request: KnowledgeQueryV1) -> KnowledgeSnapshotResponseV1:
        """POST a version-1 query and validate the complete response."""
        headers = {"content-type": "application/json", "accept": "application/json"}
        if self._bearer_token is not None:
            headers["authorization"] = f"Bearer {self._bearer_token.get_secret_value()}"
        response = await self._transport.post(
            "/v1/knowledge/query",
            request.canonical_json_bytes(),
            headers,
            timeout_seconds=self._timeout_seconds,
            max_response_bytes=self._max_response_bytes,
        )
        if response.status_code == 429 or response.status_code >= 500:
            raise KnowledgeUnavailableError(
                f"GPU Wiki returned temporary status {response.status_code}"
            )
        if response.status_code != 200:
            raise RuntimeError(f"GPU Wiki query was rejected with status {response.status_code}")
        if len(response.body) > self._max_response_bytes:
            raise ValueError("GPU Wiki response exceeds byte limit")
        return KnowledgeSnapshotResponseV1.model_validate_json(response.body)
