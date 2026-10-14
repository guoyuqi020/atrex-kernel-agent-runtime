"""Classify infrastructure rejections independently of Agate's HTTP status."""

from __future__ import annotations

import json
from collections.abc import Mapping
from http.client import InvalidURL

from ..domain.errors import InfrastructureError, UpstreamGatewayError


def _field(error: BaseException, name: str) -> object:
    payload = getattr(error, "payload", None)
    value = getattr(error, name, None)
    return value if value is not None else (
        payload.get(name) if isinstance(payload, Mapping) else None
    )


def is_invalid_url_error(error: BaseException) -> bool:
    """Recognize local URL rejection, including the SDK's transport envelope.

    The SDK can label InvalidURL as retryable infra even though no request was
    sent. Repeating the same URL cannot recover; do not classify by message text.
    """
    details = _field(error, "details")
    return isinstance(error, InvalidURL) or (
        isinstance(details, Mapping) and details.get("cause_class") == "InvalidURL"
    )


def is_infrastructure_request_error(error: BaseException) -> bool:
    """OSS placement can be HTTP 400/validation without executing the candidate.

    Use structured reasons, never substrings of messages or candidate source.
    Genuine source-validation, authentication and unknown 4xx errors keep their
    existing classification.
    """
    return (
        _field(error, "error_class") == "infra"
        or _field(error, "reason") == "oss_unavailable"
    )


def infrastructure_request_error(error: Exception) -> InfrastructureError:
    """Preserve correlation fields without publishing requests or upload URLs."""
    details = {
        name: value[:400]
        for name in ("reason", "error_class", "trace_id")
        if isinstance(value := _field(error, name), str)
    }
    message = f"Agate infrastructure request failed: {type(error).__name__}: {error}"
    status = getattr(error, "status", None)
    failure = (
        UpstreamGatewayError(status, message)
        if isinstance(status, int) and not isinstance(status, bool)
        else InfrastructureError(message)
    )
    failure.public_detail = (
        "Agate infrastructure request failed; no Kernel verdict was recorded: "
        + json.dumps(details, ensure_ascii=False)
    )
    return failure
