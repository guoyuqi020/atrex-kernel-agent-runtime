"""CLI composition for Optimizer and Evolver development shells."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

from ..artifacts.local import ArtifactKind, LocalArtifactStore
from ..bootstrap import load_campaign_spec
from ..composition.bootstrap import build_gate_contract_policy, build_optimizer_base_loader
from ..composition.campaign import (
    build_core_process_config,
    build_evolution_process_config,
    build_worker_launcher,
)
from ..config import RuntimeSettings
from ..controller.evidence import LocalEvidenceAssembler
from ..controller.leases import RegistryLineageLeaseManager
from ..controller.projection import EvidenceProjectionLimits
from ..dev_shell import (
    EvolverDevShell,
    OptimizerDevShell,
    TemporaryEvolverDevShell,
    TemporaryOptimizerDevShell,
    TemporaryOptimizerDevShellRequest,
    build_dev_shell_evidence,
    create_empty_attempt_evidence,
)
from ..domain.ids import (
    new_attempt_id,
    new_campaign_id,
    new_epoch_id,
    new_kernel_agent_revision_id,
    new_kernel_revision_id,
    new_lineage_id,
    parse_attempt_id,
    parse_lineage_id,
)
from ..domain.models import KernelAgentCatalogEntry, KernelAgentRevision
from ..gateway.contract import AgateEvaluationContractV1
from ..gateway.control import (
    AttemptTimedWorkerGatewayAuthorityProvider,
    SqliteGatewayControl,
)
from ..gateway.control_models import GatewayOperation
from ..ports import BuildChallengerRequest
from ..registry.sqlite import SqliteRegistry
from ..secrets import read_capability_signing_key
from ..workers.core import CoreOptimizerSessionDriver
from ..workers.evolution import EvolutionWorkspaceAssembler, SubprocessEvolutionSessionDriver
from ..workers.optimizer import OptimizerSessionConfig
from ..workers.problem_generalization import AgentProblemV1
from ..workers.workspace import LocalAttemptWorkspaceAssembler


def open_optimizer_dev_shell(
    config_path: str,
    lineage_value: str | None,
    attempt_value: str | None,
    shell_name: str,
) -> None:
    """Open an exact Optimizer environment while deliberately skipping Core entrypoint."""
    settings = RuntimeSettings.from_file(config_path)
    campaign = settings.campaign
    if campaign is None:
        raise ValueError("dev-shell requires Campaign runtime configuration")
    lineage_id = None if lineage_value is None else parse_lineage_id(lineage_value)
    attempt_id = None if attempt_value is None else parse_attempt_id(attempt_value)
    signing_key = read_capability_signing_key(
        os.environ,
        settings.gateway_proxy.capability_signing_key_env,
    )
    registry = SqliteRegistry(settings.storage.registry_database, require_fencing=True)
    control: SqliteGatewayControl | None = None
    try:
        artifacts = LocalArtifactStore(settings.storage.artifacts_root)
        control = SqliteGatewayControl(
            settings.storage.gateway_database,
            registry,
            signing_key=signing_key,
        )
        evidence_settings = campaign.evidence
        evidence = build_dev_shell_evidence(
            registry,
            artifacts,
            EvidenceProjectionLimits(
                max_trace_files=evidence_settings.max_trace_files,
                max_trace_bytes=evidence_settings.max_trace_bytes,
                max_trace_events=evidence_settings.max_trace_events,
                max_projection_text_bytes=evidence_settings.max_projection_text_bytes,
                max_diff_files=evidence_settings.max_diff_files,
                max_diff_bytes=evidence_settings.max_diff_bytes,
            ),
            redaction_patterns=evidence_settings.redaction_patterns,
        )
        operations = frozenset(
            {
                *(GatewayOperation(value) for value in campaign.gateway_operations),
                *((GatewayOperation.WIKI_QUERY,) if settings.gpu_wiki is not None else ()),
            }
        )
        service = OptimizerDevShell(
            registry,
            LocalAttemptWorkspaceAssembler(
                campaign.attempt_workspaces_root,
                registry,
                artifacts,
            ),
            evidence,
            CoreOptimizerSessionDriver(
                build_worker_launcher(settings, os.environ),
                build_core_process_config(campaign),
                artifacts,
            ),
            AttemptTimedWorkerGatewayAuthorityProvider(
                control,
                registry,
                campaign.gateway_proxy_url,
                operations=operations,
                max_calls=campaign.gateway_max_calls,
                lifetime=timedelta(seconds=campaign.gateway_capability_lifetime_seconds),
            ),
            RegistryLineageLeaseManager(
                registry,
                lease_seconds=campaign.fencing_lease_seconds,
                heartbeat_seconds=campaign.fencing_heartbeat_seconds,
            ),
            OptimizerSessionConfig(environment=campaign.optimizer.environment.resolve(os.environ)),
            wiki_enabled=settings.gpu_wiki is not None,
        )
        result = service.open(
            shell_name=shell_name,
            lineage_id=lineage_id,
            attempt_id=attempt_id,
        )
    finally:
        if control is not None:
            control.close()
        registry.close()
    print(
        json.dumps(
            {
                "attempt_id": result.attempt_id,
                "lineage_id": result.lineage_id,
                "workspace": str(result.workspace),
                "shell": str(result.shell),
                "returncode": result.returncode,
                "created_attempt": result.created_attempt,
                "attempt_status": "running",
            },
            sort_keys=True,
        )
    )


def open_temporary_optimizer_dev_shell(
    config_path: str,
    campaign_path: str,
    shell_name: str,
) -> None:
    """Open a disposable Optimizer workspace without bootstrapping durable state."""
    settings = RuntimeSettings.from_file(config_path)
    campaign = settings.campaign
    if campaign is None:
        raise ValueError("temporary-dev-shell requires Campaign runtime configuration")
    spec = load_campaign_spec(campaign_path)
    selected = spec.selected_dsls()
    if len(selected) != 1:
        raise ValueError("temporary-dev-shell requires exactly one configured DSL")
    if spec.agent_problem is None:
        raise ValueError("temporary-dev-shell requires a supplied public Agent Problem")
    dsl = selected[0]
    lineage_spec = spec.lineages[dsl]

    registry = SqliteRegistry(settings.storage.registry_database, require_fencing=True)
    control: SqliteGatewayControl | None = None
    try:
        artifacts = LocalArtifactStore(settings.storage.artifacts_root)
        loader = build_optimizer_base_loader(settings, artifacts)
        if loader is None:
            raise ValueError("temporary-dev-shell requires a Git Optimizer Base source")
        optimizer = loader.build_candidate(dsl, spec.base_revision.commit).candidate

        contract_value = json.loads(spec.evaluation_contract.read_bytes())
        contract = AgateEvaluationContractV1.model_validate(contract_value)
        gate_policy = build_gate_contract_policy(settings)
        if gate_policy is not None:
            contract = gate_policy.apply(contract)
        contract_digest = artifacts.put_json(
            contract.model_dump(mode="json"),
            ArtifactKind.EVALUATION_CONTRACT,
        )
        problem_value = json.loads(Path(spec.agent_problem).read_bytes())
        problem = AgentProblemV1.from_value(problem_value, private_shapes=contract.shapes)
        problem_digest = artifacts.put_json(
            problem.model_dump(mode="json"),
            ArtifactKind.AGENT_PROBLEM,
        )
        input_kernel_digest = artifacts.put_directory(
            lineage_spec.baseline_kernel,
            ArtifactKind.KERNEL,
        )

        attempt_id = new_attempt_id()
        campaign_id = new_campaign_id()
        lineage_id = new_lineage_id()
        epoch_id = new_epoch_id()
        agent_revision_id = new_kernel_agent_revision_id()
        checkpoint = LocalEvidenceAssembler(registry, artifacts).create_initial(
            lineage_id,
            lineage_spec.initial_evidence,
            source_label="temporary-optimizer-dev-shell",
        )
        attempt_evidence = create_empty_attempt_evidence(
            artifacts,
            epoch_id=epoch_id,
            attempt_id=attempt_id,
            epoch_evidence_checkpoint=checkpoint,
        )
        control = SqliteGatewayControl(
            settings.storage.gateway_database,
            registry,
            signing_key=read_capability_signing_key(
                os.environ,
                settings.gateway_proxy.capability_signing_key_env,
            ),
        )
        operations = frozenset(
            {
                *(GatewayOperation(value) for value in campaign.gateway_operations),
                *((GatewayOperation.WIKI_QUERY,) if settings.gpu_wiki is not None else ()),
            }
        )
        service = TemporaryOptimizerDevShell(
            artifacts,
            control,
            CoreOptimizerSessionDriver(
                build_worker_launcher(settings, os.environ),
                build_core_process_config(campaign),
                artifacts,
            ),
            OptimizerSessionConfig(environment=campaign.optimizer.environment.resolve(os.environ)),
            workspace_root=campaign.attempt_workspaces_root,
            gateway_endpoint=campaign.gateway_proxy_url,
            operations=operations,
            max_calls=campaign.gateway_max_calls,
            capability_lifetime=timedelta(seconds=campaign.gateway_capability_lifetime_seconds),
            wiki_enabled=settings.gpu_wiki is not None,
        )
        result = service.open(
            shell_name=shell_name,
            request=TemporaryOptimizerDevShellRequest(
                attempt_id=attempt_id,
                campaign_id=campaign_id,
                lineage_id=lineage_id,
                epoch_id=epoch_id,
                kernel_agent_revision_id=agent_revision_id,
                input_kernel_revision_id=new_kernel_revision_id(),
                optimizer_digest=optimizer.optimizer_digest,
                input_kernel_digest=input_kernel_digest,
                evaluation_contract_digest=contract_digest,
                agent_problem_digest=problem_digest,
                epoch_evidence_checkpoint=checkpoint,
                attempt_evidence_digest=attempt_evidence,
                dsl=dsl,
                operator=spec.operator,
                hardware_target=spec.hardware_target,
                model=lineage_spec.models.optimizer,
            ),
        )
    finally:
        if control is not None:
            control.close()
        registry.close()
    print(
        json.dumps(
            {
                "attempt_id": result.attempt_id,
                "workspace": str(result.workspace),
                "workspace_destroyed": not result.workspace.exists(),
                "shell": str(result.shell),
                "returncode": result.returncode,
                "durable_campaign_created": False,
                "durable_lineage_created": False,
            },
            sort_keys=True,
        )
    )


def open_evolver_dev_shell(
    config_path: str,
    lineage_value: str,
    epoch_number: int,
    shell_name: str,
) -> None:
    """Open an Epoch-anchored Evolution environment without starting the Evolver."""
    settings = RuntimeSettings.from_file(config_path)
    campaign = settings.campaign
    if campaign is None:
        raise ValueError("evolver-dev-shell requires Campaign runtime configuration")
    lineage_id = parse_lineage_id(lineage_value)
    registry = SqliteRegistry(settings.storage.registry_database, require_fencing=True)
    try:
        artifacts = LocalArtifactStore(settings.storage.artifacts_root)
        lineage = registry.get_lineage(lineage_id)
        frozen_evolver_commit = registry.get_campaign(lineage.campaign_id).evolver_commit
        if frozen_evolver_commit is None:
            raise ValueError(
                "Campaign predates Evolver commit freezing; resume Bootstrap or run-campaign "
                "once before opening an Evolver dev shell"
            )
        if frozen_evolver_commit != campaign.evolver.commit:
            raise ValueError(
                f"Campaign freezes Evolver commit {frozen_evolver_commit}; configured commit "
                f"is {campaign.evolver.commit}"
            )
        evolution_config = build_evolution_process_config(campaign, artifacts, os.environ)
        sessions = SubprocessEvolutionSessionDriver(
            build_worker_launcher(settings, os.environ),
            evolution_config,
        )
        service = EvolverDevShell(
            registry,
            EvolutionWorkspaceAssembler(
                campaign.evolution_workspaces_root,
                artifacts,
                evolver_bundle_digest=evolution_config.bundle_artifact_digest,
            ),
            sessions,
            RegistryLineageLeaseManager(
                registry,
                lease_seconds=campaign.fencing_lease_seconds,
                heartbeat_seconds=campaign.fencing_heartbeat_seconds,
            ),
        )
        result = service.open(
            shell_name=shell_name,
            lineage_id=lineage_id,
            epoch_number=epoch_number,
        )
    finally:
        registry.close()
    print(
        json.dumps(
            {
                "lineage_id": result.lineage_id,
                "epoch_id": result.epoch_id,
                "epoch_number": result.epoch_number,
                "epoch_status": result.epoch_status.value,
                "parent_revision_id": result.parent_revision_id,
                "workspace": str(result.workspace),
                "shell": str(result.shell),
                "returncode": result.returncode,
            },
            sort_keys=True,
        )
    )


def open_temporary_evolver_dev_shell(
    config_path: str,
    campaign_path: str,
    shell_name: str,
) -> None:
    """Open a disposable Evolver workspace without Bootstrap or Registry state."""
    settings = RuntimeSettings.from_file(config_path)
    campaign = settings.campaign
    if campaign is None:
        raise ValueError(
            "temporary-evolver-dev-shell requires Campaign runtime configuration"
        )
    spec = load_campaign_spec(campaign_path)
    selected = spec.selected_dsls()
    if len(selected) != 1:
        raise ValueError("temporary-evolver-dev-shell requires exactly one configured DSL")
    dsl = selected[0]
    lineage_spec = spec.lineages[dsl]

    registry = SqliteRegistry(settings.storage.registry_database, require_fencing=True)
    try:
        artifacts = LocalArtifactStore(settings.storage.artifacts_root)
        loader = build_optimizer_base_loader(settings, artifacts)
        if loader is None:
            raise ValueError(
                "temporary-evolver-dev-shell requires a Git Optimizer Base source"
            )
        base = loader.build_candidate(dsl, spec.base_revision.commit)
        campaign_id = new_campaign_id()
        lineage_id = new_lineage_id()
        epoch_id = new_epoch_id()
        parent_revision_id = new_kernel_agent_revision_id()
        now = datetime.now(UTC).isoformat()
        parent = KernelAgentRevision(
            id=parent_revision_id,
            parent_id=None,
            creation_key=f"temporary-evolver:{epoch_id}:agent-v0",
            dsl=dsl,
            optimizer_digest=base.candidate.optimizer_digest,
            created_by="lineage_seed",
            created_at=now,
            source_provenance_digest=base.source_provenance_digest,
        )
        checkpoint = LocalEvidenceAssembler(registry, artifacts).create_initial(
            lineage_id,
            lineage_spec.initial_evidence,
            source_label="temporary-evolver-dev-shell",
        )
        request = BuildChallengerRequest(
            parent_revision=parent,
            epoch_id=epoch_id,
            evidence_checkpoint=checkpoint,
            idempotency_key=f"temporary-evolver-dev-shell:{epoch_id}",
            agent_catalog=(
                KernelAgentCatalogEntry(
                    revision=parent,
                    revision_number=0,
                    parent_revision_number=None,
                    campaign_id=campaign_id,
                    lineage_id=lineage_id,
                    introduced_epoch_id=None,
                    introduced_epoch_number=None,
                    disposition="baseline",
                    active=True,
                ),
            ),
            kernel_catalog=(),
            model=lineage_spec.models.evolver,
        )
        evolution_config = build_evolution_process_config(campaign, artifacts, os.environ)
        service = TemporaryEvolverDevShell(
            EvolutionWorkspaceAssembler(
                campaign.evolution_workspaces_root,
                artifacts,
                evolver_bundle_digest=evolution_config.bundle_artifact_digest,
            ),
            SubprocessEvolutionSessionDriver(
                build_worker_launcher(settings, os.environ),
                evolution_config,
            ),
        )
        result = service.open(shell_name=shell_name, request=request)
    finally:
        registry.close()
    print(
        json.dumps(
            {
                "epoch_id": result.epoch_id,
                "parent_revision_id": result.parent_revision_id,
                "workspace": str(result.workspace),
                "workspace_destroyed": not result.workspace.exists(),
                "shell": str(result.shell),
                "returncode": result.returncode,
                "durable_campaign_created": False,
                "durable_lineage_created": False,
                "durable_epoch_created": False,
            },
            sort_keys=True,
        )
    )
