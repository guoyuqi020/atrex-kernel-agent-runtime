"""Small public diagnostics; full exception chains stay in trusted Runtime logs."""

from __future__ import annotations

import os
import re

from ..domain.errors import InfrastructureError

_MAX_DETAIL_BYTES = 8192


def infrastructure_detail(error: InfrastructureError) -> str:
    """Honor a private-payload override even through an SDK exception wrapper."""
    detail = str(error) or type(error).__name__
    current: BaseException | None = error
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, InfrastructureError) and current.public_detail is not None:
            detail = current.public_detail
            break
        current = current.__cause__ or (
            None if current.__suppress_context__ else current.__context__
        )
    # Git/HTTP diagnostics may include URLs or credentials. Do not echo them to workers.
    for name, value in os.environ.items():
        if value and (name in {"AGATE_AK", "AGATE_SK"} or
                      re.search(r"(?:TOKEN|SECRET|PASSWORD|API_KEY|SIGNING_KEY)$", name)):
            detail = detail.replace(value, "[REDACTED]")
    detail = re.sub(r"(https?://)[^\s/@]+:[^\s/@]+@", r"\1[REDACTED]@", detail)
    detail = re.sub(r"(?i)(bearer\s+)[^\s,;]+", r"\1[REDACTED]", detail)
    return detail.encode("utf-8")[:_MAX_DETAIL_BYTES].decode("utf-8", errors="ignore")
