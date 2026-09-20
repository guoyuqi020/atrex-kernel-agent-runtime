"""Resolve deployment-owned Agate settings into an SDK connection value."""

from __future__ import annotations

from collections.abc import Mapping
from urllib.parse import urlsplit

from ..config import DEFAULT_AGATE_URL, AgateSettings
from ..secrets import required_secret
from .agate import AgateConnectionConfig


def agate_settings_from_environment(
    environment: Mapping[str, str], *, base_url: str | None = None
) -> AgateSettings:
    """Default to the official remote Gateway; keep explicit endpoint overrides."""
    url = base_url or environment.get("AGATE_URL") or DEFAULT_AGATE_URL
    local = urlsplit(url).hostname in {"127.0.0.1", "localhost", "::1"}
    credentials = bool(environment.get("AGATE_AK") or environment.get("AGATE_SK"))
    auth: dict[str, object] = (
        {"auth_mode": "none"}
        if local and not credentials
        else {"auth_mode": "ak_sk", "access_key_env": "AGATE_AK", "secret_key_env": "AGATE_SK"}
    )
    return AgateSettings.model_validate(
        {
            "base_url": url,
            **auth,
            "http_timeout_s": float(environment.get("AGATE_HTTP_TIMEOUT", "1800")),
            "wait_timeout_s": float(environment.get("AGATE_WAIT_TIMEOUT", "3900")),
            "health_check_interval_s": float(environment.get("AGATE_HEALTH_CHECK_INTERVAL", "30")),
        }
    )


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
