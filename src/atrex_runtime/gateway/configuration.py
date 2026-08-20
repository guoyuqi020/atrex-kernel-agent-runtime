"""Resolve deployment-owned Agate settings into an SDK connection value."""

from __future__ import annotations

from collections.abc import Mapping

from ..config import AgateSettings
from ..secrets import required_secret
from .agate import AgateConnectionConfig


def build_agate_connection(
    settings: AgateSettings,
    environment: Mapping[str, str],
) -> AgateConnectionConfig:
    """Resolve credential environment variables without exposing their values."""
    return AgateConnectionConfig(
        base_url=settings.base_url,
        auth_mode=settings.auth_mode,
        token=required_secret(environment, settings.token_env),
        access_key=required_secret(environment, settings.access_key_env),
        secret_key=required_secret(environment, settings.secret_key_env),
        http_timeout_s=settings.http_timeout_s,
        wait_timeout_s=settings.wait_timeout_s,
    )
