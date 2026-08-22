"""Authenticated Campaign task API and independent durable task worker."""

from __future__ import annotations

import base64
import hmac
import json
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Protocol
from urllib.parse import parse_qs

import anyio
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ..artifacts.local import ArtifactKind, LocalArtifactStore
from ..asgi import (
    AsgiReceive,
    AsgiSend,
    read_request_body,
)
from ..asgi import (
    json_response as _json_response,
)
from ..bootstrap import (
    CampaignBootstrapResult,
    CampaignSpecV3,
    parse_campaign_spec_json,
)
from ..domain.errors import InvalidTransitionError
from ..domain.ids import (
    ArtifactDigest,
    CampaignId,
    new_campaign_task_id,
    parse_attempt_id,
    parse_campaign_id,
    parse_campaign_task_id,
    parse_epoch_id,
    parse_kernel_agent_revision_id,
    parse_kernel_revision_id,
    parse_lineage_id,
    parse_worker_session_id,
)
from ..domain.models import (
    CampaignTask,
    CampaignTaskStatus,
    KernelCatalogEntry,
    RuntimeEvent,
)
from ..gateway.control import (
    SqliteGatewayControl,
)
from ..lineage_seed import (
    LineageSeedResult,
    LineageSeedSpecV1,
    parse_lineage_seed_spec_json,
)
from ..presentation import (
    agent_catalog_entry,
    agent_revision_value,
    attempt_value,
    bootstrap_result_value,
    bootstrap_run_value,
    evaluation_value,
    kernel_catalog_entry,
    kernel_value,
    lineage_attempt_values,
    lineage_epoch_values,
    lineage_seed_result_value,
    measurement_values,
    worker_session_value,
)
from ..registry.base import Registry


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _bootstrap_source_paths(spec: CampaignSpecV3) -> tuple[Path | None, ...]:
    """Return every host path named by the Campaign definition."""
    common: list[Path | None] = [
        spec.evaluation_contract,
        spec.agent_problem,
    ]
    for lineage in spec.lineages.values():
        common.extend(
            (
                lineage.baseline_kernel,
                lineage.initial_evidence,
            )
        )
    return tuple(common)


class CampaignBootstrapService(Protocol):
    """Trusted idempotent Campaign bootstrap operation."""

    def bootstrap_campaign(self, spec: CampaignSpecV3) -> CampaignBootstrapResult:
        """Initialize the configured DSL Lineages under one Campaign identity."""
        ...


class LineageSeedService(Protocol):
    """Trusted idempotent Lineage-root publication operation."""

    async def seed_lineage(
        self,
        campaign_id: CampaignId,
        spec: LineageSeedSpecV1,
    ) -> LineageSeedResult:
        """Publish one independent Lineage under an active Campaign."""
        ...


class CampaignTaskRequestV1(BaseModel):
    """Strict HTTP request for one idempotent Campaign scheduling task."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1]
    creation_key: str = Field(min_length=1, max_length=256)
    campaign_id: str
    target_epoch_number: int = Field(gt=0)
    finalize: bool = False


class EpochRecoveryRequestV1(BaseModel):
    """Operator authorization for one Failed Epoch recovery."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1]
    recovery_key: str = Field(min_length=1, max_length=256)
    reason: str = Field(min_length=1, max_length=4096)


class EventPruneRequestV1(BaseModel):
    """Bounded deletion request for an acknowledged Runtime Event prefix."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1]
    before_sequence: int = Field(gt=0)
    limit: int = Field(gt=0)


class AdministrationAsgiApp:
    """Bearer-authenticated task mutation, status, and event pagination API."""

    def __init__(
        self,
        registry: Registry,
        artifacts: LocalArtifactStore,
        bootstrapper: CampaignBootstrapService | None,
        *,
        lineage_seeder: LineageSeedService | None = None,
        bearer_token: str,
        max_request_bytes: int,
        event_page_limit: int,
        event_export_limit: int,
        event_prune_limit: int,
        gateway_control: SqliteGatewayControl | None = None,
        max_kernel_source_files: int = 64,
        max_kernel_source_bytes: int = 524288,
        clock: Callable[[], datetime] = _utc_now,
    ) -> None:
        if len(bearer_token.encode()) < 32:
            raise ValueError("administration bearer token must contain at least 32 bytes")
        if (
            min(
                max_request_bytes,
                event_page_limit,
                event_export_limit,
                event_prune_limit,
                max_kernel_source_files,
                max_kernel_source_bytes,
            )
            <= 0
        ):
            raise ValueError("administration limits must be positive")
        self._registry = registry
        self._artifacts = artifacts
        self._bootstrapper = bootstrapper
        self._lineage_seeder = lineage_seeder
        self._gateway_control = gateway_control
        self._bearer_token = bearer_token.encode()
        self._max_request_bytes = max_request_bytes
        self._event_page_limit = event_page_limit
        self._event_export_limit = event_export_limit
        self._event_prune_limit = event_prune_limit
        self._max_kernel_source_files = max_kernel_source_files
        self._max_kernel_source_bytes = max_kernel_source_bytes
        self._clock = clock

    async def __call__(
        self,
        scope: Mapping[str, object],
        receive: AsgiReceive,
        send: AsgiSend,
    ) -> None:
        if not self._authorized(scope):
            await _json_response(send, 401, {"error": "unauthorized"})
            return
        path = scope.get("path")
        method = scope.get("method")
        try:
            if path == "/v1/admin/tasks" and method == "POST":
                await self._create_task(receive, send)
                return
            if path == "/v1/admin/events" and method == "GET":
                await self._events(scope, send, export=False)
                return
            if path == "/v1/admin/events/export" and method == "GET":
                await self._events(scope, send, export=True)
                return
            if path == "/v1/admin/events/prune" and method == "POST":
                await self._prune_events(receive, send)
                return
            if path == "/v1/admin/metrics" and method == "GET":
                metrics = self._registry.summarize_runtime_metrics()
                await _json_response(
                    send,
                    200,
                    {
                        "latest_event_sequence": metrics.latest_event_sequence,
                        "event_counts": dict(metrics.event_counts),
                        "campaign_task_counts": dict(metrics.campaign_task_counts),
                    },
                )
                return
            if path == "/v1/admin/campaigns/bootstrap" and method == "POST":
                await self._bootstrap(receive, send)
                return
            if isinstance(path, str) and path.startswith("/v1/admin/bootstrap-attempts/"):
                await self._bootstrap_attempt_route(path, method, send)
                return
            if isinstance(path, str) and path.startswith("/v1/admin/worker-sessions/"):
                await self._worker_session_route(path, method, send)
                return
            if isinstance(path, str) and path.startswith("/v1/admin/attempts/"):
                await self._attempt_route(path, method, send)
                return
            if isinstance(path, str) and path.startswith("/v1/admin/tasks/"):
                await self._task_route(path, method, send)
                return
            if isinstance(path, str) and path.startswith("/v1/admin/campaigns/"):
                await self._campaign_route(path, method, receive, send)
                return
            if isinstance(path, str) and path.startswith("/v1/admin/lineages/"):
                await self._lineage_route(path, method, send)
                return
            if isinstance(path, str) and path.startswith("/v1/admin/kernels/"):
                await self._kernel_route(path, method, send)
                return
            if isinstance(path, str) and path.startswith("/v1/admin/agent-revisions/"):
                await self._agent_revision_route(path, method, send)
                return
            if isinstance(path, str) and path.startswith("/v1/admin/epochs/"):
                await self._epoch_route(path, method, receive, send)
                return
            await _json_response(send, 404, {"error": "not_found"})
        except KeyError as error:
            await _json_response(send, 404, {"error": str(error)})
        except InvalidTransitionError as error:
            await _json_response(send, 409, {"error": str(error)})
        except (ValueError, ValidationError, UnicodeError) as error:
            await _json_response(send, 400, {"error": str(error)})

    def _authorized(self, scope: Mapping[str, object]) -> bool:
        headers = scope.get("headers")
        if not isinstance(headers, list):
            return False
        expected = b"Bearer " + self._bearer_token
        for item in headers:
            if (
                isinstance(item, tuple)
                and len(item) == 2
                and item[0].lower() == b"authorization"
                and isinstance(item[1], bytes)
            ):
                return hmac.compare_digest(item[1], expected)
        return False

    async def _create_task(self, receive: AsgiReceive, send: AsgiSend) -> None:
        payload = await _read_body(receive, self._max_request_bytes)
        request = CampaignTaskRequestV1.model_validate_json(payload)
        now = self._clock().isoformat()
        task = self._registry.enqueue_campaign_task(
            CampaignTask(
                id=new_campaign_task_id(),
                creation_key=request.creation_key,
                campaign_id=parse_campaign_id(request.campaign_id),
                target_epoch_number=request.target_epoch_number,
                finalize=request.finalize,
                status=CampaignTaskStatus.QUEUED,
                attempt_count=0,
                lease_owner=None,
                lease_expires_at=None,
                last_error=None,
                created_at=now,
                started_at=None,
                completed_at=None,
            )
        )
        await _json_response(send, 202, _task_value(task))

    async def _bootstrap(self, receive: AsgiReceive, send: AsgiSend) -> None:
        if self._bootstrapper is None:
            await _json_response(send, 404, {"error": "not_found"})
            return
        payload = await _read_body(receive, self._max_request_bytes)
        spec = parse_campaign_spec_json(payload)
        sources = _bootstrap_source_paths(spec)
        if any(source is not None and not source.is_absolute() for source in sources):
            raise ValueError("HTTP bootstrap source paths must be absolute")
        # Core bootstrap phases call back into this Runtime's Gateway/Wiki HTTP routes.
        # Keep the ASGI loop available while the blocking Git/Core process runs.
        result = await anyio.to_thread.run_sync(
            self._bootstrapper.bootstrap_campaign,
            spec,
        )
        await _json_response(
            send,
            200,
            bootstrap_result_value(result),
        )

    async def _campaign_route(
        self,
        path: str,
        method: object,
        receive: AsgiReceive,
        send: AsgiSend,
    ) -> None:
        parts = path.removeprefix("/v1/admin/campaigns/").split("/")
        if len(parts) not in {1, 2}:
            await _json_response(send, 404, {"error": "not_found"})
            return
        campaign_id = parse_campaign_id(parts[0])
        action = None if len(parts) == 1 else parts[1]
        if action == "lineages" and method == "POST":
            if self._lineage_seeder is None:
                await _json_response(send, 404, {"error": "not_found"})
                return
            payload = await _read_body(receive, self._max_request_bytes)
            spec = parse_lineage_seed_spec_json(payload)
            if spec.initial_evidence is not None and not spec.initial_evidence.is_absolute():
                raise ValueError("HTTP Lineage seed Evidence path must be absolute")
            result = await self._lineage_seeder.seed_lineage(campaign_id, spec)
            await _json_response(send, 200, lineage_seed_result_value(result))
            return
        if action == "cancel" and method == "POST":
            await _json_response(
                send,
                200,
                _campaign_value(self._registry, campaign_id, cancel=True),
            )
            return
        if action == "kernels" and method == "GET":
            await _json_response(
                send,
                200,
                {
                    "campaign_id": campaign_id,
                    "kernels": [
                        kernel_value(entry, artifacts=self._artifacts)
                        for entry in self._registry.list_campaign_kernels(campaign_id)
                    ],
                },
            )
            return
        if action == "agent-revisions" and method == "GET":
            await _json_response(
                send,
                200,
                {
                    "campaign_id": campaign_id,
                    "agent_revisions": [
                        agent_revision_value(entry)
                        for entry in self._registry.list_campaign_agent_revisions(campaign_id)
                    ],
                },
            )
            return
        if action == "attempts" and method == "GET":
            await _json_response(
                send,
                200,
                {
                    "campaign_id": campaign_id,
                    "attempts": [
                        value
                        for lineage in self._registry.list_campaign_lineages(campaign_id)
                        for value in lineage_attempt_values(self._registry, lineage.id)
                    ],
                },
            )
            return
        if action == "worker-sessions" and method == "GET":
            await _json_response(
                send,
                200,
                {
                    "campaign_id": campaign_id,
                    "worker_sessions": [
                        worker_session_value(session)
                        for session in self._registry.list_worker_sessions(campaign_id=campaign_id)
                    ],
                },
            )
            return
        if action == "epochs" and method == "GET":
            await _json_response(
                send,
                200,
                {
                    "campaign_id": campaign_id,
                    "epochs": [
                        value
                        for lineage in self._registry.list_campaign_lineages(campaign_id)
                        for value in lineage_epoch_values(self._registry, lineage.id)
                    ],
                },
            )
            return
        if action is None and method == "GET":
            await _json_response(send, 200, _campaign_value(self._registry, campaign_id))
            return
        await _json_response(send, 405, {"error": "method_not_allowed"})

    async def _bootstrap_attempt_route(
        self,
        path: str,
        method: object,
        send: AsgiSend,
    ) -> None:
        if self._gateway_control is None:
            await _json_response(send, 404, {"error": "not_found"})
            return
        if method != "GET":
            await _json_response(send, 405, {"error": "method_not_allowed"})
            return
        parts = path.removeprefix("/v1/admin/bootstrap-attempts/").split("/")
        if len(parts) not in {2, 3} or parts[1] != "runs":
            await _json_response(send, 404, {"error": "not_found"})
            return
        attempt_id = parse_attempt_id(parts[0])
        if len(parts) == 2:
            runs = self._gateway_control.list_bootstrap_runs(attempt_id)
            await _json_response(
                send,
                200,
                {
                    "bootstrap_attempt_id": attempt_id,
                    "runs": [bootstrap_run_value(run) for run in runs],
                },
            )
            return
        try:
            generation = int(parts[2])
        except ValueError as error:
            raise ValueError("Bootstrap recovery generation must be an integer") from error
        await _json_response(
            send,
            200,
            bootstrap_run_value(self._gateway_control.get_bootstrap_run(attempt_id, generation)),
        )

    async def _lineage_route(
        self,
        path: str,
        method: object,
        send: AsgiSend,
    ) -> None:
        parts = path.removeprefix("/v1/admin/lineages/").split("/")
        if len(parts) != 2 or parts[1] not in {
            "kernels",
            "agent-revisions",
            "attempts",
            "epochs",
            "worker-sessions",
        }:
            await _json_response(send, 404, {"error": "not_found"})
            return
        lineage_id = parse_lineage_id(parts[0])
        if method != "GET":
            await _json_response(send, 405, {"error": "method_not_allowed"})
            return
        if parts[1] == "kernels":
            value: dict[str, object] = {
                "lineage_id": lineage_id,
                "kernels": [
                    kernel_value(entry, artifacts=self._artifacts)
                    for entry in self._registry.list_lineage_kernels(lineage_id)
                ],
            }
        elif parts[1] == "agent-revisions":
            value = {
                "lineage_id": lineage_id,
                "agent_revisions": [
                    agent_revision_value(entry)
                    for entry in self._registry.list_lineage_agent_revisions(lineage_id)
                ],
            }
        elif parts[1] == "attempts":
            value = {
                "lineage_id": lineage_id,
                "attempts": lineage_attempt_values(self._registry, lineage_id),
            }
        elif parts[1] == "epochs":
            value = {
                "lineage_id": lineage_id,
                "epochs": lineage_epoch_values(self._registry, lineage_id),
            }
        else:
            value = {
                "lineage_id": lineage_id,
                "worker_sessions": [
                    worker_session_value(session)
                    for session in self._registry.list_worker_sessions(lineage_id=lineage_id)
                ],
            }
        await _json_response(send, 200, value)

    async def _agent_revision_route(
        self,
        path: str,
        method: object,
        send: AsgiSend,
    ) -> None:
        if method != "GET":
            await _json_response(send, 405, {"error": "method_not_allowed"})
            return
        parts = path.removeprefix("/v1/admin/agent-revisions/").split("/")
        if len(parts) != 1:
            await _json_response(send, 404, {"error": "not_found"})
            return
        revision_id = parse_kernel_agent_revision_id(parts[0])
        await _json_response(
            send,
            200,
            agent_revision_value(agent_catalog_entry(self._registry, revision_id)),
        )

    async def _attempt_route(
        self,
        path: str,
        method: object,
        send: AsgiSend,
    ) -> None:
        if method != "GET":
            await _json_response(send, 405, {"error": "method_not_allowed"})
            return
        parts = path.removeprefix("/v1/admin/attempts/").split("/")
        if len(parts) == 1:
            attempt = self._registry.get_attempt(parse_attempt_id(parts[0]))
            await _json_response(send, 200, attempt_value(self._registry, attempt))
            return
        if len(parts) == 2 and parts[1] == "worker-sessions":
            attempt_id = parse_attempt_id(parts[0])
            await _json_response(
                send,
                200,
                {
                    "attempt_id": attempt_id,
                    "worker_sessions": [
                        worker_session_value(session)
                        for session in self._registry.list_worker_sessions(attempt_id=attempt_id)
                    ],
                },
            )
            return
        if len(parts) < 2 or parts[1] != "evaluations" or len(parts) > 4:
            await _json_response(send, 404, {"error": "not_found"})
            return

        if self._gateway_control is None:
            await _json_response(send, 404, {"error": "not_found"})
            return
        attempt_id = parse_attempt_id(parts[0])
        if len(parts) == 2:
            await _json_response(
                send,
                200,
                {
                    "attempt_id": attempt_id,
                    "evaluations": [
                        evaluation_value(evaluation)
                        for evaluation in self._gateway_control.list_evaluations(attempt_id)
                    ],
                },
            )
            return
        evaluation = self._gateway_control.get_evaluation(parts[2])
        if evaluation.attempt_id != attempt_id:
            raise KeyError(parts[2])
        if len(parts) == 3:
            await _json_response(send, 200, evaluation_value(evaluation))
            return
        action = parts[3]
        if action == "source":
            await _json_response(
                send,
                200,
                {
                    "evaluation_id": evaluation.id,
                    "evaluation_label": (
                        f"g{evaluation.recovery_generation}-e{evaluation.ordinal}"
                    ),
                    "attempt_id": attempt_id,
                    "artifact_digest": evaluation.candidate_artifact_digest,
                    "referenced_at": evaluation.created_at,
                    "files": self._artifact_files(
                        evaluation.candidate_artifact_digest,
                        expected_kind=ArtifactKind.KERNEL,
                    ),
                },
            )
            return
        if action == "result":
            stored = self._artifacts.verify(evaluation.gateway_result_digest)
            if stored.kind is not ArtifactKind.GATEWAY_RESULT:
                raise ValueError("evaluation result Artifact has the wrong kind")
            value = stored.payload_path / "value.json"
            if not value.is_file() or value.stat().st_size > self._max_kernel_source_bytes:
                raise ValueError("evaluation result exceeds the administration byte limit")
            await _json_response(
                send,
                200,
                {
                    "evaluation_id": evaluation.id,
                    "evaluation_label": (
                        f"g{evaluation.recovery_generation}-e{evaluation.ordinal}"
                    ),
                    "attempt_id": attempt_id,
                    "artifact_digest": evaluation.gateway_result_digest,
                    "referenced_at": evaluation.created_at,
                    "result": json.loads(value.read_bytes()),
                },
            )
            return
        await _json_response(send, 404, {"error": "not_found"})

    async def _worker_session_route(
        self,
        path: str,
        method: object,
        send: AsgiSend,
    ) -> None:
        if method != "GET":
            await _json_response(send, 405, {"error": "method_not_allowed"})
            return
        parts = path.removeprefix("/v1/admin/worker-sessions/").split("/")
        if len(parts) != 1:
            await _json_response(send, 404, {"error": "not_found"})
            return
        session = self._registry.get_worker_session(parse_worker_session_id(parts[0]))
        await _json_response(send, 200, worker_session_value(session))

    async def _kernel_route(
        self,
        path: str,
        method: object,
        send: AsgiSend,
    ) -> None:
        parts = path.removeprefix("/v1/admin/kernels/").split("/")
        if len(parts) not in {1, 2}:
            await _json_response(send, 404, {"error": "not_found"})
            return
        revision_id = parse_kernel_revision_id(parts[0])
        action = None if len(parts) == 1 else parts[1]
        if method != "GET":
            await _json_response(send, 405, {"error": "method_not_allowed"})
            return
        entry = kernel_catalog_entry(self._registry, revision_id)
        if action is None:
            await _json_response(
                send,
                200,
                kernel_value(
                    entry,
                    artifacts=self._artifacts,
                    include_measurements=True,
                    registry=self._registry,
                ),
            )
            return
        if action == "measurements":
            await _json_response(
                send,
                200,
                {
                    "kernel_revision_id": revision_id,
                    "measurements": measurement_values(self._registry, entry),
                },
            )
            return
        if action == "source":
            await _json_response(send, 200, self._kernel_source(entry))
            return
        await _json_response(send, 404, {"error": "not_found"})

    def _kernel_source(self, entry: KernelCatalogEntry) -> dict[str, object]:
        files = self._artifact_files(
            entry.revision.artifact_digest,
            expected_kind=ArtifactKind.KERNEL,
        )
        return {
            "kernel_revision_id": entry.revision.id,
            "version": f"v{entry.revision_number}",
            "artifact_digest": entry.revision.artifact_digest,
            "referenced_at": entry.revision.created_at,
            "files": files,
        }

    def _artifact_files(
        self,
        digest: ArtifactDigest,
        *,
        expected_kind: ArtifactKind,
    ) -> list[dict[str, object]]:
        stored = self._artifacts.verify(digest)
        if stored.kind is not expected_kind:
            raise ValueError("Artifact has the wrong kind")
        files: list[dict[str, object]] = []
        paths = [path for path in sorted(stored.payload_path.rglob("*")) if path.is_file()]
        if len(paths) > self._max_kernel_source_files:
            raise ValueError("Kernel source exceeds the administration file limit")
        if sum(path.stat().st_size for path in paths) > self._max_kernel_source_bytes:
            raise ValueError("Kernel source exceeds the administration byte limit")
        for path in paths:
            payload = path.read_bytes()
            try:
                content = payload.decode("utf-8")
                encoding = "utf-8"
            except UnicodeDecodeError:
                content = base64.b64encode(payload).decode("ascii")
                encoding = "base64"
            files.append(
                {
                    "path": path.relative_to(stored.payload_path).as_posix(),
                    "size": len(payload),
                    "encoding": encoding,
                    "content": content,
                }
            )
        return files

    async def _epoch_route(
        self,
        path: str,
        method: object,
        receive: AsgiReceive,
        send: AsgiSend,
    ) -> None:
        suffix = path.removeprefix("/v1/admin/epochs/")
        if suffix.endswith("/worker-sessions") and method == "GET":
            epoch_id = parse_epoch_id(suffix.removesuffix("/worker-sessions"))
            await _json_response(
                send,
                200,
                {
                    "epoch_id": epoch_id,
                    "worker_sessions": [
                        worker_session_value(session)
                        for session in self._registry.list_worker_sessions(epoch_id=epoch_id)
                    ],
                },
            )
            return
        if not suffix.endswith("/recover") or method != "POST":
            await _json_response(send, 405, {"error": "method_not_allowed"})
            return
        epoch_id = parse_epoch_id(suffix.removesuffix("/recover"))
        payload = await _read_body(receive, self._max_request_bytes)
        request = EpochRecoveryRequestV1.model_validate_json(payload)
        recovery = self._registry.recover_failed_epoch(
            epoch_id,
            recovery_key=request.recovery_key,
            reason=request.reason,
        )
        await _json_response(
            send,
            200,
            {
                "epoch_id": recovery.epoch_id,
                "lineage_id": recovery.lineage_id,
                "campaign_id": recovery.campaign_id,
                "recovery_key": recovery.recovery_key,
                "generation": recovery.generation,
                "attempt_ids": recovery.attempt_ids,
                "created_at": recovery.created_at,
            },
        )

    async def _task_route(
        self,
        path: str,
        method: object,
        send: AsgiSend,
    ) -> None:
        parts = path.removeprefix("/v1/admin/tasks/").split("/")
        if len(parts) not in {1, 2}:
            await _json_response(send, 404, {"error": "not_found"})
            return
        task_id = parse_campaign_task_id(parts[0])
        action = None if len(parts) == 1 else parts[1]
        if action == "cancel" and method == "POST":
            task = self._registry.cancel_campaign_task(task_id)
            await _json_response(send, 200, _task_value(task))
            return
        if action == "requeue" and method == "POST":
            task = self._registry.requeue_campaign_task(task_id)
            await _json_response(send, 200, _task_value(task))
            return
        if action is None and method == "GET":
            task = self._registry.get_campaign_task(task_id)
            await _json_response(send, 200, _task_value(task))
            return
        await _json_response(send, 405, {"error": "method_not_allowed"})

    async def _events(
        self,
        scope: Mapping[str, object],
        send: AsgiSend,
        *,
        export: bool,
    ) -> None:
        raw_query = scope.get("query_string", b"")
        if not isinstance(raw_query, bytes):
            raise ValueError("query string must be bytes")
        if len(raw_query) > self._max_request_bytes:
            raise ValueError("administration query exceeds byte limit")
        query = parse_qs(raw_query.decode("ascii"), strict_parsing=True)
        correlation_parsers = {
            "campaign_id": parse_campaign_id,
            "lineage_id": parse_lineage_id,
            "epoch_id": parse_epoch_id,
            "attempt_id": parse_attempt_id,
            "campaign_task_id": parse_campaign_task_id,
            "kernel_revision_id": parse_kernel_revision_id,
        }
        unknown = set(query).difference({"after", "limit", "kind", *correlation_parsers})
        if unknown:
            raise ValueError(f"unknown event query fields: {sorted(unknown)}")
        after = _one_int(query, "after", default=0, minimum=0)
        configured_limit = self._event_export_limit if export else self._event_page_limit
        limit = _one_int(query, "limit", default=configured_limit, minimum=1)
        if limit > configured_limit:
            raise ValueError("event page limit exceeds configured maximum")
        kinds = tuple(dict.fromkeys(query.get("kind", [])))
        if any(not kind or len(kind) > 200 for kind in kinds):
            raise ValueError("event kind filter is invalid")
        correlation: dict[str, str] = {}
        for name, parser in correlation_parsers.items():
            values = query.get(name)
            if values is None:
                continue
            if len(values) != 1:
                raise ValueError(f"event query field {name} must appear once")
            correlation[name] = parser(values[0])
        events = self._registry.list_runtime_events(
            after_sequence=after,
            limit=limit,
            kinds=kinds,
            correlation=correlation,
        )
        if export:
            await _ndjson_response(send, events)
            return
        await _json_response(
            send,
            200,
            {
                "events": [_event_value(event) for event in events],
                "next_after": after if not events else events[-1].sequence,
            },
        )

    async def _prune_events(self, receive: AsgiReceive, send: AsgiSend) -> None:
        payload = await _read_body(receive, self._max_request_bytes)
        request = EventPruneRequestV1.model_validate_json(payload)
        if request.limit > self._event_prune_limit:
            raise ValueError("event prune limit exceeds configured maximum")
        deleted = self._registry.prune_runtime_events(
            before_sequence=request.before_sequence,
            limit=request.limit,
        )
        await _json_response(send, 200, {"deleted": deleted})


def _one_int(
    query: dict[str, list[str]],
    name: str,
    *,
    default: int,
    minimum: int,
) -> int:
    values = query.get(name)
    if values is None:
        return default
    if len(values) != 1:
        raise ValueError(f"event query field {name} must appear once")
    value = int(values[0])
    if value < minimum:
        raise ValueError(f"event query field {name} is below its minimum")
    return value


async def _read_body(receive: AsgiReceive, limit: int) -> bytes:
    return await read_request_body(
        receive,
        limit,
        oversized_message="administration request exceeds byte limit",
    )


def _task_value(task: CampaignTask) -> dict[str, object]:
    return {
        "task_id": task.id,
        "creation_key": task.creation_key,
        "campaign_id": task.campaign_id,
        "target_epoch_number": task.target_epoch_number,
        "finalize": task.finalize,
        "status": task.status,
        "attempt_count": task.attempt_count,
        "lease_expires_at": task.lease_expires_at,
        "last_error": task.last_error,
        "created_at": task.created_at,
        "started_at": task.started_at,
        "completed_at": task.completed_at,
    }


def _event_value(event: RuntimeEvent) -> dict[str, object]:
    return {
        "sequence": event.sequence,
        "kind": event.kind,
        "aggregate_id": event.aggregate_id,
        "payload": event.payload,
        "created_at": event.created_at,
    }


def _campaign_value(
    registry: Registry,
    campaign_id: CampaignId,
    *,
    cancel: bool = False,
) -> dict[str, object]:
    campaign = (
        registry.cancel_campaign(campaign_id) if cancel else registry.get_campaign(campaign_id)
    )
    lineages: list[dict[str, object]] = []
    for lineage in registry.list_campaign_lineages(campaign_id):
        open_epoch = registry.find_open_epoch(lineage.id)
        agent_versions = {
            entry.revision.id: f"agent-v{entry.revision_number}"
            for entry in registry.list_lineage_agent_revisions(lineage.id)
        }
        lineages.append(
            {
                "lineage_id": lineage.id,
                "dsl": lineage.dsl,
                "status": lineage.status,
                "next_epoch_number": lineage.next_epoch_number,
                "models": {
                    "optimizer": lineage.optimizer_model,
                    "evolver": lineage.evolver_model,
                },
                "active_kernel_agent_revision_id": (lineage.active_kernel_agent_revision_id),
                "active_kernel_agent_version": agent_versions[
                    lineage.active_kernel_agent_revision_id
                ],
                "best_kernel_revision_id": lineage.best_kernel_revision_id,
                "evidence_checkpoint": lineage.evidence_checkpoint,
                "open_epoch": (
                    None
                    if open_epoch is None
                    else {
                        "epoch_id": open_epoch.id,
                        "number": open_epoch.number,
                        "status": open_epoch.status,
                    }
                ),
            }
        )
    return {
        "campaign_id": campaign.id,
        "operator": campaign.operator,
        "hardware_target": campaign.hardware_target,
        "evaluation_contract_digest": campaign.evaluation_contract_digest,
        "agent_problem_digest": campaign.agent_problem_digest,
        "problem_generalization_model": campaign.problem_generalization_model,
        "evolver_commit": campaign.evolver_commit,
        "status": campaign.status,
        "created_at": campaign.created_at,
        "lineages": lineages,
    }


async def _ndjson_response(send: AsgiSend, events: list[RuntimeEvent]) -> None:
    payload = b"".join(
        json.dumps(
            _event_value(event),
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode()
        + b"\n"
        for event in events
    )
    await send(
        {
            "type": "http.response.start",
            "status": 200,
            "headers": [(b"content-type", b"application/x-ndjson")],
        }
    )
    await send({"type": "http.response.body", "body": payload})
