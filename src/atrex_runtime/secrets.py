"""Deployment secret loading shared by trusted Runtime process roles."""

from __future__ import annotations

import base64
import binascii
from collections.abc import Mapping

from pydantic import SecretStr


def required_secret(environment: Mapping[str, str], name: str | None) -> SecretStr | None:
    """Read one explicitly named non-empty environment secret."""
    if name is None:
        return None
    value = environment.get(name)
    if not value:
        raise ValueError(f"required secret environment variable is missing: {name}")
    return SecretStr(value)


def read_capability_signing_key(environment: Mapping[str, str], name: str) -> bytes:
    """Decode and validate the persistent Gateway capability signing key."""
    secret = required_secret(environment, name)
    if secret is None:
        raise AssertionError("signing key environment name is required")
    encoded = secret.get_secret_value()
    try:
        key = base64.b64decode(encoded, altchars=b"-_", validate=True)
    except (binascii.Error, ValueError) as error:
        raise ValueError(f"capability signing key in {name} is not valid Base64") from error
    if len(key) < 32:
        raise ValueError(f"capability signing key in {name} must decode to at least 32 bytes")
    return key
