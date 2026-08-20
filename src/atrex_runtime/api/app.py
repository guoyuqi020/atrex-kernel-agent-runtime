"""Runtime resource assembly and ASGI lifecycle."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from typing import Protocol

from ..artifacts.local import LocalArtifactStore
from ..asgi import AsgiReceive, AsgiSend, json_response
from ..composition.bootstrap import (
    build_campaign_bootstrapper,
    build_lineage_seeder,
)
from ..composition.gateway import compose_authoritative_candidate_evaluator
from ..config import RuntimeSettings
from ..dependency_health import PeriodicHealthMonitor
from ..gateway.agate import (
    AgateClient,
    AgateConnectionConfig,
    AgateGatewayAdapter,
    AgateRequestBuilder,
    SqliteAgateJobStore,
    load_agate_sdk,
)
from ..gateway.configuration import build_agate_connection
from ..gateway.contract import RegistryAgateEvaluationContextResolver
from ..gateway.control import SqliteGatewayControl
from ..gateway.diff_policy import CandidateDiffPolicy, RegistryCandidateDiffValidator
from ..gateway.production_policy import ProductionKernelPolicy, RegistryProductionKernelValidator
from ..gateway.proxy import (
    GatewayProxyAsgiApp,
    GatewayProxyLimits,
    GatewayProxyService,
)
from ..knowledge.client import HttpGpuWikiClient, HttpxGpuWikiTransport
from ..knowledge.proxy import WikiProxyAsgiApp, WikiProxyLimits, WikiProxyService
from ..registry.sqlite import SqliteRegistry
from ..secrets import read_capability_signing_key, required_secret
from .administration import AdministrationAsgiApp


class AgateSdkLoader(Protocol):
    """Construct the published Agate SDK client and request builder."""

    def __call__(self, config: AgateConnectionConfig) -> tuple[AgateClient, AgateRequestBuilder]:
        """Load one configured SDK client."""
        ...


class RuntimeApplication:
    """Own Runtime resources and serve HTTP plus ASGI lifespan scopes."""

    def __init__(
        self,
        proxy: GatewayProxyAsgiApp,
        wiki_proxy: WikiProxyAsgiApp | None,
        administration: AdministrationAsgiApp | None,
        readiness_probes: Mapping[str, Callable[[], None]],
        closers: tuple[Callable[[], None], ...],
        health_monitors: tuple[PeriodicHealthMonitor, ...] = (),
    ) -> None:
        self._proxy = proxy
        self._wiki_proxy = wiki_proxy
        self._administration = administration
        self._readiness_probes = dict(readiness_probes)
        self._closers = closers
        self._health_monitors = health_monitors
        self._closed = False

    async def __call__(
        self,
        scope: Mapping[str, object],
        receive: AsgiReceive,
        send: AsgiSend,
    ) -> None:
        scope_type = scope.get("type")
        if scope_type == "lifespan":
            await self._lifespan(receive, send)
            return
        if scope_type == "http" and scope.get("path") == "/healthz":
            await self._health(scope, send)
            return
        if scope_type == "http" and scope.get("path") == "/readyz":
            await self._readiness(scope, send)
            return
        if (
            scope_type == "http"
            and isinstance(scope.get("path"), str)
            and str(scope["path"]).startswith("/v1/admin/")
        ):
            if self._administration is None:
                await json_response(send, 404, {"error": "not_found"})
                return
            await self._administration(scope, receive, send)
            return
        if scope_type == "http" and scope.get("path") == "/v1/wiki/query":
            if self._wiki_proxy is None:
                await json_response(send, 404, {"error": "not_found"})
                return
            await self._wiki_proxy(scope, receive, send)
            return
        await self._proxy(scope, receive, send)

    def close(self) -> None:
        """Close every owned resource once, in reverse construction order."""
        if self._closed:
            return
        self._closed = True
        for monitor in self._health_monitors:
            monitor.cancel()
        for close in self._closers:
            close()

    async def _lifespan(self, receive: AsgiReceive, send: AsgiSend) -> None:
        while True:
            message = await receive()
            message_type = message.get("type")
            if message_type == "lifespan.startup":
                for monitor in self._health_monitors:
                    monitor.start()
                await send({"type": "lifespan.startup.complete"})
            elif message_type == "lifespan.shutdown":
                for monitor in reversed(self._health_monitors):
                    await monitor.stop()
                self.close()
                await send({"type": "lifespan.shutdown.complete"})
                return
            else:
                raise ValueError(f"unexpected ASGI lifespan message: {message_type!r}")

    @staticmethod
    async def _health(scope: Mapping[str, object], send: AsgiSend) -> None:
        if scope.get("method") != "GET":
            await json_response(send, 405, {"error": "method_not_allowed"})
            return
        await json_response(send, 200, {"status": "ok"})

    async def _readiness(self, scope: Mapping[str, object], send: AsgiSend) -> None:
        if scope.get("method") != "GET":
            await json_response(send, 405, {"error": "method_not_allowed"})
            return
        failed: list[str] = []
        for name, probe in self._readiness_probes.items():
            try:
                probe()
            except Exception:
                failed.append(name)
        if failed:
            await json_response(
                send,
                503,
                {"status": "unavailable", "failed_dependencies": failed},
            )
            return
        await json_response(send, 200, {"status": "ready"})


def build_runtime_application(
    settings: RuntimeSettings,
    environment: Mapping[str, str],
    *,
    sdk_loader: AgateSdkLoader = load_agate_sdk,
) -> RuntimeApplication:
    """Build the complete trusted Gateway service or close partial resources on failure."""
    signing_key = read_capability_signing_key(
        environment,
        settings.gateway_proxy.capability_signing_key_env,
    )
    connection = build_agate_connection(settings.agate, environment)
    registry = SqliteRegistry(settings.storage.registry_database)
    control: SqliteGatewayControl | None = None
    jobs: SqliteAgateJobStore | None = None
    try:
        artifacts = LocalArtifactStore(settings.storage.artifacts_root)
        control = SqliteGatewayControl(
            settings.storage.gateway_database,
            registry,
            signing_key=signing_key,
        )
        campaign = settings.campaign
        gate_policy = settings.gate_policy or (
            None if campaign is None else campaign.gate_policy
        )
        jobs = SqliteAgateJobStore(settings.storage.agate_jobs_database)
        client, request_builder = sdk_loader(connection)
        contexts = RegistryAgateEvaluationContextResolver(registry, artifacts, control)
        production_policy = ProductionKernelPolicy()
        adapter = AgateGatewayAdapter(
            client,
            request_builder,
            contexts,
            jobs,
            wait_timeout_s=connection.wait_timeout_s,
            optimizer_evaluate_repeats=(
                1 if gate_policy is None else gate_policy.optimizer.evaluate_repeats
            ),
            optimizer_correctness_cases=(
                1 if gate_policy is None else gate_policy.optimizer.correctness_cases
            ),
            optimizer_bench_iters=(
                100 if gate_policy is None else gate_policy.optimizer.bench_iters
            ),
            profile_without_roofline=True,
            connection_summary={
                "url": connection.base_url,
                "auth": connection.auth_mode,
            },
        )
        limits = GatewayProxyLimits(
            settings.gateway_proxy.max_request_bytes,
            settings.gateway_proxy.max_candidate_files,
            settings.gateway_proxy.max_candidate_bytes,
        )
        proxy = GatewayProxyAsgiApp(
            GatewayProxyService(
                control,
                artifacts,
                adapter,
                limits,
                registry,
                RegistryCandidateDiffValidator(
                    registry,
                    artifacts,
                    CandidateDiffPolicy(
                        settings.gateway_proxy.candidate_diff_allowed_paths,
                        settings.gateway_proxy.candidate_diff_require_change,
                    ),
                    control,
                ),
                RegistryProductionKernelValidator(
                    contexts,
                    artifacts,
                    production_policy,
                ),
            ),
            limits,
        )
        wiki_proxy = None
        if settings.gpu_wiki is not None:
            wiki = settings.gpu_wiki
            wiki_limits = WikiProxyLimits(
                wiki.max_proxy_request_bytes,
                wiki.max_query_bytes,
            )
            wiki_proxy = WikiProxyAsgiApp(
                WikiProxyService(
                    control,
                    control,
                    registry,
                    artifacts,
                    HttpGpuWikiClient(
                        HttpxGpuWikiTransport(wiki.base_url),
                        bearer_token=required_secret(environment, wiki.bearer_token_env),
                        timeout_seconds=wiki.timeout_seconds,
                        max_response_bytes=wiki.max_response_bytes,
                    ),
                    wiki_limits,
                    registry,
                ),
                wiki_limits,
            )
        administration = None
        if settings.administration is not None:
            secret = required_secret(
                environment,
                settings.administration.bearer_token_env,
            )
            if secret is None:
                raise AssertionError("administration bearer token name is required")
            administration = AdministrationAsgiApp(
                registry,
                artifacts,
                (
                    None
                    if campaign is None
                    else build_campaign_bootstrapper(
                        settings,
                        artifacts,
                        registry,
                        environment,
                        control=control,
                        finalizer=compose_authoritative_candidate_evaluator(
                            settings,
                            artifacts,
                            registry,
                            control,
                            client,
                            request_builder,
                            production_policy=production_policy,
                        ),
                    )
                ),
                lineage_seeder=(
                    None
                    if campaign is None
                    else build_lineage_seeder(
                        settings,
                        artifacts,
                        registry,
                        client,
                        request_builder,
                        production_policy=production_policy,
                    )
                ),
                bearer_token=secret.get_secret_value(),
                max_request_bytes=settings.administration.max_request_bytes,
                event_page_limit=settings.administration.event_page_limit,
                event_export_limit=settings.administration.event_export_limit,
                event_prune_limit=settings.administration.event_prune_limit,
                gateway_control=control,
                max_kernel_source_files=settings.gateway_proxy.max_candidate_files,
                max_kernel_source_bytes=settings.gateway_proxy.max_candidate_bytes,
            )
        readiness_probes: dict[str, Callable[[], None]] = {
            "registry": registry.check_health,
            "gateway_control": control.check_health,
            "agate_jobs": jobs.check_health,
            "artifact_store": artifacts.check_health,
        }
        return RuntimeApplication(
            proxy,
            wiki_proxy,
            administration,
            readiness_probes,
            (jobs.close, control.close, registry.close),
            (
                PeriodicHealthMonitor(
                    "Agate",
                    client.health,
                    interval_seconds=settings.agate.health_check_interval_s,
                    logger=logging.getLogger("uvicorn.error"),
                ),
            ),
        )
    except BaseException:
        if jobs is not None:
            jobs.close()
        if control is not None:
            control.close()
        registry.close()
        raise
