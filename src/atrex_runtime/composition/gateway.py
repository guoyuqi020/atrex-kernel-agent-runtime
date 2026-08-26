"""Shared construction of trusted Gateway evaluation components."""

from __future__ import annotations

from collections.abc import Mapping

from ..artifacts.local import LocalArtifactStore
from ..config import RuntimeSettings
from ..gateway.agate import AgateClient, AgateRequestBuilder, load_agate_sdk
from ..gateway.configuration import build_agate_connection
from ..gateway.contract import RegistryAgateEvaluationContextResolver
from ..gateway.control import SqliteGatewayControl
from ..gateway.finalization import (
    AgateAuthoritativeCandidateEvaluator,
    BootstrapEvaluationStage,
)
from ..gateway.production_policy import ProductionKernelPolicy
from ..registry.base import Registry


def compose_authoritative_candidate_evaluator(
    settings: RuntimeSettings,
    artifacts: LocalArtifactStore,
    registry: Registry,
    control: SqliteGatewayControl,
    client: AgateClient,
    request_builder: AgateRequestBuilder,
    *,
    production_policy: ProductionKernelPolicy | None = None,
) -> AgateAuthoritativeCandidateEvaluator:
    """Compose the evaluator from already-open Runtime and Agate resources."""
    campaign = settings.campaign
    gate_policy = settings.gate_policy or (None if campaign is None else campaign.gate_policy)
    return AgateAuthoritativeCandidateEvaluator(
        client,
        request_builder,
        RegistryAgateEvaluationContextResolver(registry, artifacts, control),
        artifacts,
        control,
        registry,
        wait_timeout_s=settings.agate.wait_timeout_s,
        bootstrap_stages=(
            (BootstrapEvaluationStage(1), BootstrapEvaluationStage(5))
            if gate_policy is None
            else tuple(
                BootstrapEvaluationStage(stage.correctness_cases, stage.evaluate_repeats)
                for stage in gate_policy.bootstrap.stages
            )
        ),
        bootstrap_bench_iters=(5 if gate_policy is None else gate_policy.bootstrap.bench_iters),
        profile_without_roofline=True,
        production_policy=production_policy or ProductionKernelPolicy(),
    )


def build_authoritative_candidate_evaluator(
    settings: RuntimeSettings,
    artifacts: LocalArtifactStore,
    registry: Registry,
    control: SqliteGatewayControl,
    environment: Mapping[str, str],
) -> AgateAuthoritativeCandidateEvaluator:
    """Load the Agate SDK and compose the trusted final evaluation path."""
    client, request_builder = load_agate_sdk(build_agate_connection(settings.agate, environment))
    return compose_authoritative_candidate_evaluator(
        settings,
        artifacts,
        registry,
        control,
        client,
        request_builder,
    )


__all__ = [
    "build_authoritative_candidate_evaluator",
    "compose_authoritative_candidate_evaluator",
]
