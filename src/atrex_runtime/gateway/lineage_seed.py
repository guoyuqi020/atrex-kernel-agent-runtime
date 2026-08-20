"""Independent Agate evaluation for a Kernel selected as a new Lineage root."""

from __future__ import annotations

from collections.abc import Callable

from ..artifacts.local import JsonValue, LocalArtifactStore
from ..domain.errors import InfrastructureError
from ..domain.ids import ArtifactDigest, CampaignId, LineageId
from ..domain.models import Dsl
from ..ports import AttemptCandidateResult, RuntimeEventRecorder
from ..registry.base import Registry
from .agate import (
    AgateCandidateRejection,
    AgateClient,
    AgateRequestBuilder,
    parse_agate_evaluation,
)
from .batched_evaluate import ShapeBatch, ShapeBatchedEvaluateExecutor, ShapeBatchOutcome
from .candidate import resolve_kernel_candidate
from .contract import load_evaluation_contract
from .execution import (
    build_evaluation_request,
    call_agate_json,
    store_gateway_result,
    submit_agate_job,
)
from .production_policy import ProductionKernelPolicy
from .protocol import EvaluationV2

_TERMINAL = frozenset({"succeeded", "failed", "cancelled"})


class AgateLineageSeedEvaluator:
    """Evaluate exact seed content without creating an Agent-visible Attempt."""

    def __init__(
        self,
        client: AgateClient,
        request_builder: AgateRequestBuilder,
        registry: Registry,
        artifacts: LocalArtifactStore,
        events: RuntimeEventRecorder,
        *,
        wait_timeout_s: float,
        profile_without_roofline: bool = True,
        production_policy: ProductionKernelPolicy | None = None,
    ) -> None:
        if wait_timeout_s <= 0:
            raise ValueError("Lineage seed evaluation wait timeout must be positive")
        self._client = client
        self._request_builder = request_builder
        self._registry = registry
        self._artifacts = artifacts
        self._events = events
        self._wait_timeout_s = wait_timeout_s
        self._profile_without_roofline = profile_without_roofline
        self._production_policy = production_policy
        self._shape_batches = ShapeBatchedEvaluateExecutor()

    async def evaluate(
        self,
        *,
        campaign_id: CampaignId,
        lineage_id: LineageId,
        dsl: Dsl,
        kernel_artifact_digest: ArtifactDigest,
    ) -> AttemptCandidateResult:
        """Run one authoritative ordinary Evaluate and an optional SOL Profile."""
        campaign = self._registry.get_campaign(campaign_id)
        contract = load_evaluation_contract(
            self._artifacts,
            campaign.evaluation_contract_digest,
        )
        resolved = resolve_kernel_candidate(
            self._artifacts,
            kernel_artifact_digest,
            contract.candidate_path,
            error_type=ValueError,
            kind_error="Lineage seed candidate Artifact is not a Kernel",
            missing_error="Lineage seed Kernel does not contain the contract candidate path",
        )
        candidate = resolved.source
        if contract.production_gate and self._production_policy is not None:
            self._production_policy.validate(
                resolved.root,
                contract.candidate_path,
                dsl,
            )
        try:
            candidate_source = candidate.read_text(encoding="utf-8")
        except UnicodeDecodeError as error:
            raise ValueError("Lineage seed Kernel source must be UTF-8") from error

        idempotency_key = f"lineage-seed:{lineage_id}:{kernel_artifact_digest}"
        profile_payload = build_evaluation_request(
            self._request_builder,
            candidate_source=candidate_source,
            operator=campaign.operator,
            contract=contract,
            hardware_target=campaign.hardware_target,
            dsl=dsl,
            name=f"{campaign.operator}_{lineage_id}_seed",
            idempotency_key=idempotency_key,
        )

        async def evaluate_batch(batch: ShapeBatch) -> ShapeBatchOutcome:
            payload = build_evaluation_request(
                self._request_builder,
                candidate_source=candidate_source,
                operator=campaign.operator,
                contract=batch.contract,
                hardware_target=campaign.hardware_target,
                dsl=dsl,
                name=f"{campaign.operator}_{lineage_id}_seed_batch_{batch.index}",
                idempotency_key=batch.idempotency_key,
            )
            try:
                accepted = await self._submit("eval", payload)
            except AgateCandidateRejection as rejection:
                return ShapeBatchOutcome(
                    {"status": "rejected", "error": rejection.payload},
                    EvaluationV2(correct=False, latency_us=None),
                )
            job_id = accepted.get("job_id")
            if not isinstance(job_id, str) or not job_id:
                raise InfrastructureError("Lineage seed Agate acceptance has no job_id")
            self._events.record_runtime_event(
                "lineage_seed.evaluation_submitted",
                lineage_id,
                {
                    "campaign_id": campaign_id,
                    "kernel_artifact_digest": kernel_artifact_digest,
                    "agate_job_id": job_id,
                    "shape_batch": batch.index,
                    "shape_ids": list(batch.shape_ids),
                },
            )
            job = await self._call(
                lambda: self._client.get_job(job_id, wait=True, timeout=self._wait_timeout_s)
            )
            if job.get("status") not in _TERMINAL:
                raise InfrastructureError("Lineage seed Agate job did not reach a terminal state")
            evaluation = (
                parse_agate_evaluation(job, batch.shape_ids)
                if job.get("status") == "succeeded"
                else EvaluationV2(correct=False, latency_us=None)
            )
            return ShapeBatchOutcome(job, evaluation, job_id)

        batched = await self._shape_batches.run(contract, idempotency_key, evaluate_batch)
        evaluation = batched.evaluation
        job = batched.job
        job_id = batched.job_id
        profile: JsonValue | None = None
        if evaluation.correct and contract.roofline is None and self._profile_without_roofline:
            profile = await self._profile(lineage_id, kernel_artifact_digest, profile_payload)
        result = self._store_result(job, profile)
        self._completed(
            lineage_id,
            kernel_artifact_digest,
            result,
            evaluation.correct,
            evaluation.latency_us,
            job_id,
        )
        return AttemptCandidateResult(
            kernel_artifact_digest,
            result,
            evaluation.correct,
            evaluation.latency_us,
        )

    async def _profile(
        self,
        lineage_id: LineageId,
        kernel_digest: ArtifactDigest,
        evaluation_payload: dict[str, object],
    ) -> JsonValue:
        payload = dict(evaluation_payload)
        payload.update(
            {
                "idempotency_key": f"lineage-seed-profile:{lineage_id}:{kernel_digest}",
                "level": "sol",
                "top_kernels": 10,
            }
        )
        try:
            accepted = await self._submit("profile", payload)
            job_id = accepted.get("job_id")
            if not isinstance(job_id, str) or not job_id:
                raise InfrastructureError("Lineage seed Agate Profile acceptance has no job_id")
            self._events.record_runtime_event(
                "lineage_seed.profile_submitted",
                lineage_id,
                {"kernel_artifact_digest": kernel_digest, "agate_job_id": job_id},
            )
            job = await self._call(
                lambda: self._client.get_job(job_id, wait=True, timeout=self._wait_timeout_s)
            )
            if job.get("status") not in _TERMINAL:
                raise InfrastructureError("Lineage seed Agate Profile did not terminate")
            self._events.record_runtime_event(
                "lineage_seed.profile_completed",
                lineage_id,
                {
                    "kernel_artifact_digest": kernel_digest,
                    "agate_job_id": job_id,
                    "status": job.get("status"),
                },
            )
            return job
        except Exception as error:
            message = f"{type(error).__name__}: {error}"[:1000]
            self._events.record_runtime_event(
                "lineage_seed.profile_failed",
                lineage_id,
                {"kernel_artifact_digest": kernel_digest, "error": message},
            )
            return {"status": "failed", "error": {"message": message}}

    def _completed(
        self,
        lineage_id: LineageId,
        kernel_digest: ArtifactDigest,
        result_digest: ArtifactDigest,
        correct: bool,
        latency_us: float | None,
        job_id: str | None,
    ) -> None:
        self._events.record_runtime_event(
            "lineage_seed.evaluation_completed",
            lineage_id,
            {
                "kernel_artifact_digest": kernel_digest,
                "gateway_result_digest": result_digest,
                "agate_job_id": job_id,
                "correct": correct,
                "latency_us": latency_us,
            },
        )

    def _store_result(self, job: JsonValue, profile: JsonValue | None) -> ArtifactDigest:
        return store_gateway_result(
            self._artifacts,
            job,
            profile,
            temporary_prefix="lineage-seed-result-",
        )

    async def _submit(self, kind: str, payload: dict[str, object]) -> dict[str, JsonValue]:
        return await submit_agate_job(
            self._client,
            kind,
            payload,
            request_error="Lineage seed Agate request failed",
            invalid_response="Lineage seed Agate response is invalid JSON",
            non_object_response="Lineage seed Agate response is not an object",
        )

    async def _call(self, operation: Callable[[], object]) -> dict[str, JsonValue]:
        return await call_agate_json(
            operation,
            request_error="Lineage seed Agate request failed",
            invalid_response="Lineage seed Agate response is invalid JSON",
            non_object_response="Lineage seed Agate response is not an object",
        )
