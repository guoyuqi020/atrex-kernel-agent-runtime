"""Runtime configuration, resource assembly, and ASGI lifecycle tests."""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import pytest

from atrex_runtime.api.app import RuntimeApplication, build_runtime_application
from atrex_runtime.config import RuntimeSettings
from atrex_runtime.gateway.agate import AgateConnectionConfig
from atrex_runtime.gateway.proxy import GatewayProxyAsgiApp


def _config_value() -> dict[str, object]:
    return {
        "schema_version": 1,
        "server": {"host": "127.0.0.1", "port": 8765},
        "storage": {
            "registry_database": "state/registry.sqlite",
            "gateway_database": "state/gateway.sqlite",
            "agate_jobs_database": "state/agate-jobs.sqlite",
            "artifacts_root": "state/artifacts",
        },
        "gateway_proxy": {
            "max_request_bytes": 65536,
            "max_candidate_files": 16,
            "max_candidate_bytes": 32768,
            "capability_signing_key_env": "TEST_SIGNING_KEY",
            "candidate_diff_allowed_paths": {
                "cuda": ["*.cu"],
                "triton": ["*.py"],
                "cutedsl": ["*.py"],
            },
            "candidate_diff_require_change": True,
        },
        "agate": {
            "base_url": "https://gateway.example.test",
            "auth_mode": "ak_sk",
            "access_key_env": "TEST_AGATE_AK",
            "secret_key_env": "TEST_AGATE_SK",
            "http_timeout_s": 60,
            "wait_timeout_s": 900,
        },
        "kernel_agent": {
            "max_bundle_files": 1024,
            "max_bundle_bytes": 8388608,
            "max_entrypoint_bytes": 524288,
            "max_agent_problem_bytes": 262144,
        },
    }


@dataclass
class UnusedClient:
    health_calls: int = 0

    def health(self) -> bool:
        self.health_calls += 1
        return True

    def submit_job(self, kind: str, request: dict[str, object]) -> dict[str, object]:
        del kind, request
        raise AssertionError("health check must not call Agate")

    def get_job(
        self,
        job_id: str,
        wait: bool = False,
        timeout: float = 30.0,
        include_spec: bool = False,
    ) -> dict[str, object]:
        del job_id, wait, timeout, include_spec
        raise AssertionError("health check must not call Agate")

    def cancel_job(self, job_id: str) -> dict[str, object]:
        del job_id
        raise AssertionError("health check must not call Agate")


@dataclass
class UnusedBuilder:
    def __call__(
        self,
        candidate: str,
        reference: object,
        gpu: str,
        **options: object,
    ) -> dict[str, object]:
        del candidate, reference, gpu, options
        raise AssertionError("health check must not build an Agate request")


@dataclass
class CapturingSdkLoader:
    configs: list[AgateConnectionConfig] = field(default_factory=list)
    clients: list[UnusedClient] = field(default_factory=list)

    def __call__(self, config: AgateConnectionConfig) -> tuple[UnusedClient, UnusedBuilder]:
        self.configs.append(config)
        client = UnusedClient()
        self.clients.append(client)
        return client, UnusedBuilder()


def _settings(tmp_path: Path) -> RuntimeSettings:
    path = tmp_path / "runtime.json"
    path.write_text(json.dumps(_config_value()))
    return RuntimeSettings.from_file(path)


def _environment() -> dict[str, str]:
    return {
        "TEST_SIGNING_KEY": base64.urlsafe_b64encode(b"s" * 32).decode(),
        "TEST_AGATE_AK": "access",
        "TEST_AGATE_SK": "secret",
    }


def _administration_value() -> dict[str, object]:
    return {
        "bearer_token_env": "TEST_ADMIN_TOKEN",
        "max_request_bytes": 4096,
        "event_page_limit": 100,
        "event_export_limit": 1000,
        "event_prune_limit": 100,
        "task_lease_seconds": 120,
        "task_heartbeat_seconds": 30,
        "task_poll_seconds": 1,
        "max_error_bytes": 4096,
    }


def test_config_resolves_storage_relative_to_its_file(tmp_path: Path) -> None:
    settings = _settings(tmp_path)

    assert settings.storage.registry_database == tmp_path / "state/registry.sqlite"
    assert settings.storage.artifacts_root == tmp_path / "state/artifacts"
    assert settings.agate.health_check_interval_s == 30


def test_config_rejects_shared_storage_locations() -> None:
    value = _config_value()
    storage = value["storage"]
    assert isinstance(storage, dict)
    storage["agate_jobs_database"] = storage["gateway_database"]

    with pytest.raises(ValueError, match="must be distinct"):
        RuntimeSettings.model_validate(value)


def test_application_requires_all_named_secrets(tmp_path: Path) -> None:
    environment = _environment()
    del environment["TEST_AGATE_SK"]

    with pytest.raises(ValueError, match="TEST_AGATE_SK"):
        build_runtime_application(_settings(tmp_path), environment, sdk_loader=CapturingSdkLoader())


@pytest.mark.anyio
async def test_application_owns_wiki_secret_and_routes_worker_proxy(tmp_path: Path) -> None:
    value = _config_value()
    value["gpu_wiki"] = {
        "base_url": "https://wiki.example.test",
        "bearer_token_env": "TEST_WIKI_TOKEN",
        "timeout_seconds": 10,
        "max_proxy_request_bytes": 4096,
        "max_query_bytes": 2048,
        "max_response_bytes": 8192,
    }
    path = tmp_path / "runtime.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    settings = RuntimeSettings.from_file(path)
    with pytest.raises(ValueError, match="TEST_WIKI_TOKEN"):
        build_runtime_application(settings, _environment(), sdk_loader=CapturingSdkLoader())

    app = build_runtime_application(
        settings,
        _environment() | {"TEST_WIKI_TOKEN": "wiki-secret"},
        sdk_loader=CapturingSdkLoader(),
    )
    sent: list[dict[str, object]] = []

    async def unused_receive() -> dict[str, object]:
        raise AssertionError("missing authentication must be rejected before reading the body")

    async def send(message: dict[str, object]) -> None:
        sent.append(message)

    await app(
        {"type": "http", "method": "POST", "path": "/v1/wiki/query", "headers": []},
        unused_receive,
        send,
    )
    assert sent[0]["status"] == 401
    app.close()


def test_config_rejects_unsafe_task_lease() -> None:
    value = _config_value()
    administration = _administration_value()
    administration["task_lease_seconds"] = 60
    value["administration"] = administration

    with pytest.raises(ValueError, match="exceed two heartbeat"):
        RuntimeSettings.model_validate(value)


@pytest.mark.anyio
async def test_application_serves_health_and_closes_on_lifespan_shutdown(tmp_path: Path) -> None:
    loader = CapturingSdkLoader()
    app = build_runtime_application(_settings(tmp_path), _environment(), sdk_loader=loader)
    sent: list[dict[str, object]] = []

    async def unused_receive() -> dict[str, object]:
        raise AssertionError("health endpoint must not read a request body")

    async def send(message: dict[str, object]) -> None:
        sent.append(message)

    await app(
        {"type": "http", "method": "GET", "path": "/healthz", "headers": []},
        unused_receive,
        send,
    )

    assert sent[0]["status"] == 200
    assert loader.configs[0].secret_key is not None
    assert str(loader.configs[0].secret_key) == "**********"

    sent.clear()
    await app(
        {"type": "http", "method": "GET", "path": "/readyz", "headers": []},
        unused_receive,
        send,
    )
    assert sent[0]["status"] == 200
    readiness_body = sent[-1]["body"]
    assert isinstance(readiness_body, bytes)
    assert json.loads(readiness_body) == {"status": "ready"}

    messages = iter(
        [
            {"type": "lifespan.startup"},
            {"type": "lifespan.shutdown"},
        ]
    )
    lifespan_sent: list[dict[str, object]] = []

    async def receive_lifespan() -> dict[str, object]:
        return next(messages)

    async def send_lifespan(message: dict[str, object]) -> None:
        lifespan_sent.append(message)

    await app({"type": "lifespan"}, receive_lifespan, send_lifespan)

    assert lifespan_sent == [
        {"type": "lifespan.startup.complete"},
        {"type": "lifespan.shutdown.complete"},
    ]
    app.close()


@pytest.mark.anyio
async def test_readiness_reports_named_dependency_failures() -> None:
    async def unused_proxy(scope: object, receive: object, send: object) -> None:
        del scope, receive, send
        raise AssertionError("readiness must not reach the proxy")

    def failed_probe() -> None:
        raise OSError("secret storage path")

    app = RuntimeApplication(
        cast(GatewayProxyAsgiApp, unused_proxy),
        None,
        None,
        {"registry": lambda: None, "artifact_store": failed_probe},
        (),
    )
    sent: list[dict[str, object]] = []

    async def unused_receive() -> dict[str, object]:
        raise AssertionError("readiness endpoint must not read a request body")

    async def send(message: dict[str, object]) -> None:
        sent.append(message)

    await app(
        {"type": "http", "method": "GET", "path": "/readyz", "headers": []},
        unused_receive,
        send,
    )
    assert sent[0]["status"] == 503
    readiness_body = sent[-1]["body"]
    assert isinstance(readiness_body, bytes)
    assert json.loads(readiness_body) == {
        "status": "unavailable",
        "failed_dependencies": ["artifact_store"],
    }


@pytest.mark.anyio
async def test_application_routes_authenticated_administration_events(tmp_path: Path) -> None:
    value = _config_value()
    value["administration"] = _administration_value()
    path = tmp_path / "runtime.json"
    path.write_text(json.dumps(value))
    environment = _environment()
    environment["TEST_ADMIN_TOKEN"] = "a" * 32
    app = build_runtime_application(
        RuntimeSettings.from_file(path),
        environment,
        sdk_loader=CapturingSdkLoader(),
    )

    status, payload = await _app_request(
        app,
        "/v1/admin/events",
        token="a" * 32,
    )
    assert status == 200
    assert payload == {"events": [], "next_after": 0}
    unauthorized, _ = await _app_request(app, "/v1/admin/events", token="wrong" * 8)
    assert unauthorized == 401
    app.close()


async def _app_request(
    app: RuntimeApplication,
    path: str,
    *,
    token: str,
) -> tuple[int, dict[str, object]]:
    sent: list[dict[str, object]] = []

    async def receive() -> dict[str, object]:
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message: dict[str, object]) -> None:
        sent.append(message)

    await app(
        {
            "type": "http",
            "method": "GET",
            "path": path,
            "query_string": b"",
            "headers": [(b"authorization", f"Bearer {token}".encode())],
        },
        receive,
        send,
    )
    return int(sent[0]["status"]), json.loads(sent[-1]["body"])
