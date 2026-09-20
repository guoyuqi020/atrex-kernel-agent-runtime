"""Local Agate routing uses the normal HTTP SDK and optional server-owned authentication."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from atrex_runtime.gateway.agate import AgateConnectionConfig
from atrex_runtime.gateway.configuration import (
    agate_settings_from_environment,
    build_agate_connection,
)

ROOT = Path(__file__).resolve().parents[1]


def test_remote_defaults_require_remote_credentials() -> None:
    settings = agate_settings_from_environment({})
    assert settings.base_url == "https://atrex-gateway.alibaba-inc.com"
    assert settings.auth_mode == "ak_sk"
    assert settings.access_key_env == "AGATE_AK"
    with pytest.raises(ValueError, match="AGATE_AK"):
        build_agate_connection(settings, {})
    environment = {"AGATE_AK": "test-ak", "AGATE_SK": "test-sk"}
    connection = build_agate_connection(settings, environment)
    assert connection.base_url == settings.base_url
    assert connection.auth_mode == "ak_sk"
    assert (
        AgateConnectionConfig(auth_mode="none", http_timeout_s=10, wait_timeout_s=20).base_url
        == settings.base_url
    )


@pytest.mark.parametrize(
    "url", ["http://127.0.0.1:8000", "http://localhost:8080", "http://[::1]:8000"]
)
def test_local_gateway_can_still_require_ak_sk(url: str) -> None:
    environment = {"AGATE_URL": url, "AGATE_AK": "test-ak", "AGATE_SK": "test-sk"}
    settings = agate_settings_from_environment(environment)
    assert settings.auth_mode == "ak_sk"
    connection = build_agate_connection(settings, environment)
    assert connection.access_key is not None
    assert connection.access_key.get_secret_value() == "test-ak"
    assert connection.secret_key is not None
    assert connection.secret_key.get_secret_value() == "test-sk"
    assert "test-ak" not in settings.model_dump_json()


def test_explicit_remote_gateway_keeps_existing_auth_policy() -> None:
    settings = agate_settings_from_environment({"AGATE_URL": "https://agate.example.test"})
    assert settings.base_url == "https://agate.example.test"
    assert settings.auth_mode == "ak_sk"
    with pytest.raises(ValueError, match="AGATE_AK"):
        build_agate_connection(settings, {})


@pytest.mark.parametrize(
    "environment, expected",
    [
        ({}, 64),
        ({"AGATE_URL": "http://localhost:8080"}, 0),
        ({"AGATE_URL": "http://localhost"}, 0),
        ({"AGATE_URL": "http://localhost.attacker.test:8080"}, 64),
        ({"AGATE_AK": "ak"}, 64),
        ({"AGATE_SK": "sk"}, 64),
        ({"AGATE_URL": "https://remote.test"}, 64),
        ({"AGATE_URL": "https://remote.test", "AGATE_AK": "ak", "AGATE_SK": "sk"}, 0),
    ],
)
def test_shell_environment_defaults_and_credential_validation(
    environment: dict[str, str], expected: int
) -> None:
    clean = {key: value for key, value in os.environ.items() if not key.startswith("AGATE_")}
    result = subprocess.run(
        ("bash", "-c", "source scripts/shared/agate-service.sh; atrex_require_agate_environment"),
        cwd=ROOT,
        env={**clean, **environment},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == expected
    if expected:
        assert "both AGATE_AK and AGATE_SK" in result.stderr
