"""Small shared ASGI protocol helpers used by Runtime HTTP surfaces."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable, Mapping

type AsgiMessage = dict[str, object]
type AsgiReceive = Callable[[], Awaitable[AsgiMessage]]
type AsgiSend = Callable[[AsgiMessage], Awaitable[None]]


async def read_request_body(
    receive: AsgiReceive,
    limit: int,
    *,
    oversized_message: str,
) -> bytes:
    """Read one bounded ASGI HTTP request body."""
    chunks: list[bytes] = []
    size = 0
    while True:
        message = await receive()
        if message.get("type") != "http.request":
            raise ValueError("unexpected ASGI request message")
        body = message.get("body", b"")
        if not isinstance(body, bytes):
            raise ValueError("ASGI request body must be bytes")
        size += len(body)
        if size > limit:
            raise ValueError(oversized_message)
        chunks.append(body)
        if not message.get("more_body", False):
            return b"".join(chunks)


def bearer_token(headers: object) -> str | None:
    """Extract one case-sensitive ASGI Bearer credential from raw headers."""
    if not isinstance(headers, list):
        return None
    for item in headers:
        if not isinstance(item, tuple) or len(item) != 2:
            continue
        name, value = item
        if name == b"authorization" and isinstance(value, bytes):
            decoded = value.decode("ascii", errors="ignore")
            if decoded.startswith("Bearer ") and len(decoded) > 7:
                return decoded[7:]
    return None


async def json_response(
    send: AsgiSend,
    status: int,
    value: Mapping[str, object],
) -> None:
    """Send one compact UTF-8 JSON response."""
    payload = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": [(b"content-type", b"application/json")],
        }
    )
    await send({"type": "http.response.body", "body": payload})


__all__ = [
    "AsgiMessage",
    "AsgiReceive",
    "AsgiSend",
    "bearer_token",
    "json_response",
    "read_request_body",
]
