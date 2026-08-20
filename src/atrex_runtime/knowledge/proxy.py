"""Attempt-scoped GPU Wiki proxy with freeze-before-return semantics."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol, cast

from pydantic import ValidationError

from ..artifacts.local import ArtifactKind, JsonValue, LocalArtifactStore
from ..asgi import AsgiReceive, AsgiSend, bearer_token, json_response, read_request_body
from ..domain.errors import InvalidTransitionError
from ..domain.ids import ArtifactDigest, AttemptId
from ..domain.models import BranchRole
from ..gateway.control import SqliteGatewayControl
from ..gateway.control_models import GatewayCapability, GatewayOperation
from ..ports import RuntimeEventRecorder
from ..registry.base import Registry
from ..serialization import canonical_json_digest
from .client import GpuWikiClient, KnowledgeUnavailableError
from .models import KnowledgeInteractionV1, KnowledgeQueryV1
from .protocol import (
    WIKI_PROXY_PROTOCOL_VERSION,
    WikiProxyRequestV1,
    WikiProxyResponseV1,
)


class WikiInteractionIndex(Protocol):
    """Persist and retrieve frozen Wiki results for idempotent Worker retries."""

    def get_operation_artifact(
        self,
        attempt_id: AttemptId,
        idempotency_key: str,
        operation: GatewayOperation,
    ) -> ArtifactDigest | None:
        """Return a committed result Artifact for one reserved operation."""
        ...

    def commit_operation_artifact(
        self,
        attempt_id: AttemptId,
        idempotency_key: str,
        operation: GatewayOperation,
        artifact_digest: ArtifactDigest,
    ) -> ArtifactDigest:
        """Commit or recover the immutable result Artifact for one operation."""
        ...


@dataclass(frozen=True, slots=True)
class WikiProxyLimits:
    """Worker request and model-authored query acquisition bounds."""

    max_request_bytes: int
    max_query_bytes: int

    def __post_init__(self) -> None:
        if self.max_request_bytes <= 0 or self.max_query_bytes <= 0:
            raise ValueError("Wiki Proxy limits must be positive")


class WikiProxyService:
    """Authorize, query, freeze, and return one external Wiki interaction."""

    def __init__(
        self,
        control: SqliteGatewayControl,
        interactions: WikiInteractionIndex,
        registry: Registry,
        artifacts: LocalArtifactStore,
        client: GpuWikiClient,
        limits: WikiProxyLimits,
        events: RuntimeEventRecorder,
    ) -> None:
        self._control = control
        self._interactions = interactions
        self._registry = registry
        self._artifacts = artifacts
        self._client = client
        self._limits = limits
        self._events = events

    async def query(self, token: str, payload: bytes) -> WikiProxyResponseV1:
        """Return a previously frozen response or execute and freeze a live query."""
        if len(payload) > self._limits.max_request_bytes:
            raise ValueError("Wiki Proxy request exceeds byte limit")
        request = WikiProxyRequestV1.model_validate_json(payload)
        if len(request.query.encode()) > self._limits.max_query_bytes:
            raise ValueError("Wiki query exceeds byte limit")
        request_digest = canonical_json_digest(request.model_dump(mode="json"))
        self._control.authorize(
            GatewayCapability(token, request.attempt_id),
            GatewayOperation.WIKI_QUERY,
            idempotency_key=request.idempotency_key,
            request_digest=str(request_digest),
        )
        existing = self._interactions.get_operation_artifact(
            request.attempt_id,
            request.idempotency_key,
            GatewayOperation.WIKI_QUERY,
        )
        if existing is not None:
            return self._load_response(existing)

        query = self._trusted_query(request)
        event_base: dict[str, object] = {
            "idempotency_key": request.idempotency_key,
            "request_digest": request_digest,
        }
        self._events.record_runtime_event(
            "wiki.query_submitted",
            request.attempt_id,
            event_base,
        )
        try:
            response = await self._client.query(query)
            interaction = KnowledgeInteractionV1(
                idempotency_key=request.idempotency_key,
                query=query,
                response=response,
            )
            artifact_digest = self._artifacts.put_json(
                cast(JsonValue, interaction.model_dump(mode="json")),
                ArtifactKind.WIKI_INTERACTION,
            )
            committed = self._interactions.commit_operation_artifact(
                request.attempt_id,
                request.idempotency_key,
                GatewayOperation.WIKI_QUERY,
                artifact_digest,
            )
            result = self._load_response(committed)
        except Exception as error:
            self._events.record_runtime_event(
                "wiki.query_failed",
                request.attempt_id,
                {**event_base, "error_type": type(error).__name__},
            )
            raise
        self._events.record_runtime_event(
            "wiki.query_completed",
            request.attempt_id,
            {
                **event_base,
                "interaction_artifact_digest": result.interaction_artifact_digest,
                "snapshot_id": result.snapshot_id,
                "content_digest": result.content_digest,
            },
        )
        return result

    def _trusted_query(self, request: WikiProxyRequestV1) -> KnowledgeQueryV1:
        try:
            attempt = self._registry.get_attempt(request.attempt_id)
        except KeyError:
            subject = self._control.get_bootstrap_subject(request.attempt_id)
            return KnowledgeQueryV1(
                campaign_id=subject.campaign_id,
                lineage_id=subject.lineage_id,
                epoch_id=subject.epoch_id,
                epoch_number=1,
                attempt_id=subject.attempt_id,
                branch=BranchRole.ACTIVE,
                attempt_ordinal=1,
                kernel_agent_revision_id=subject.kernel_agent_revision_id,
                operator=subject.operator,
                dsl=subject.dsl,
                hardware_target=subject.hardware_target,
                evaluation_contract_digest=subject.evaluation_contract_digest,
                epoch_evidence_checkpoint_digest=subject.evidence_digest,
                attempt_evidence_digest=subject.evidence_digest,
                query=request.query,
            )
        epoch = self._registry.get_epoch(attempt.epoch_id)
        lineage = self._registry.get_lineage(epoch.lineage_id)
        campaign = self._registry.get_campaign(lineage.campaign_id)
        return KnowledgeQueryV1(
            campaign_id=campaign.id,
            lineage_id=lineage.id,
            epoch_id=epoch.id,
            epoch_number=epoch.number,
            attempt_id=attempt.id,
            branch=attempt.branch,
            attempt_ordinal=attempt.ordinal,
            kernel_agent_revision_id=attempt.kernel_agent_revision_id,
            operator=campaign.operator,
            dsl=lineage.dsl,
            hardware_target=lineage.hardware_target,
            evaluation_contract_digest=campaign.evaluation_contract_digest,
            epoch_evidence_checkpoint_digest=epoch.evidence_checkpoint,
            attempt_evidence_digest=attempt.attempt_evidence_digest,
            query=request.query,
        )

    def _load_response(self, digest: ArtifactDigest) -> WikiProxyResponseV1:
        artifact = self._artifacts.verify(digest)
        if artifact.kind is not ArtifactKind.WIKI_INTERACTION:
            raise ValueError("Wiki operation result has the wrong Artifact kind")
        payload = (artifact.payload_path / "value.json").read_bytes()
        response = KnowledgeInteractionV1.model_validate_json(payload).response
        return WikiProxyResponseV1(
            schema_version=WIKI_PROXY_PROTOCOL_VERSION,
            interaction_artifact_digest=digest,
            snapshot_id=response.snapshot_id,
            content_digest=response.content_digest,
            content=response.content,
        )


class WikiProxyAsgiApp:
    """ASGI adapter for the Attempt-scoped Wiki query endpoint."""

    def __init__(self, service: WikiProxyService, limits: WikiProxyLimits) -> None:
        self._service = service
        self._limits = limits

    async def __call__(
        self,
        scope: Mapping[str, object],
        receive: AsgiReceive,
        send: AsgiSend,
    ) -> None:
        if scope.get("type") != "http":
            raise ValueError("Wiki Proxy supports only ASGI HTTP scopes")
        if scope.get("method") != "POST" or scope.get("path") != "/v1/wiki/query":
            await json_response(send, 404, {"error": "not_found"})
            return
        token = bearer_token(scope.get("headers"))
        if token is None:
            await json_response(send, 401, {"error": "missing_bearer_token"})
            return
        try:
            payload = await read_request_body(
                receive,
                self._limits.max_request_bytes,
                oversized_message="Wiki Proxy request exceeds byte limit",
            )
            result = await self._service.query(token, payload)
        except ValidationError as error:
            await json_response(send, 400, {"error": "invalid_request", "detail": str(error)})
        except ValueError as error:
            await json_response(send, 400, {"error": "invalid_request", "detail": str(error)})
        except PermissionError as error:
            await json_response(send, 403, {"error": "forbidden", "detail": str(error)})
        except InvalidTransitionError as error:
            await json_response(send, 409, {"error": "conflict", "detail": str(error)})
        except KnowledgeUnavailableError:
            await json_response(send, 503, {"error": "wiki_unavailable"})
        except RuntimeError:
            await json_response(send, 502, {"error": "wiki_rejected_response"})
        else:
            await json_response(send, 200, result.model_dump(mode="json"))
