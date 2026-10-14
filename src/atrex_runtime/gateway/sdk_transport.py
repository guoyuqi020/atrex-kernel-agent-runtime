"""Runtime-owned single-request policy for the installed Agate SDK.

The SDK's public transport_retries switch does not disable its separate 429
submission backoff. Override that delay hook locally so Runtime can await the
backoff instead. No SDK files or remote Agate services are modified.
"""

from __future__ import annotations

from atrex_gateway_client import Client  # type: ignore[import-untyped]


class RuntimeAgateClient(Client):  # type: ignore[misc]
    @staticmethod
    def _overload_delay(error: object, attempt: int = 0) -> None:
        """Surface overload immediately to the Runtime async retry boundary."""
        return None
