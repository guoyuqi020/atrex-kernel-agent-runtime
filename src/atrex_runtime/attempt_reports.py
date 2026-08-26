"""Runtime-owned final Attempt Report projections."""

from __future__ import annotations

import json
import math
import statistics
from collections.abc import Mapping
from typing import Final, cast

from .artifacts.local import ArtifactKind, JsonValue, LocalArtifactStore
from .domain.ids import ArtifactDigest, KernelRevisionId, parse_artifact_digest
from .domain.models import Attempt, KernelRevision
from .gateway.contract import load_evaluation_contract
from .gateway.correctness import correctness_summary, merge_correctness_summaries
from .registry.base import Registry

RUNTIME_ATTEMPT_REPORT_VERSION: Final = 1


class RuntimeAttemptReportProjector:
    """Fuse one Agent handoff with authoritative Registry and Gateway facts."""

    def __init__(self, registry: Registry, artifacts: LocalArtifactStore) -> None:
        self._registry = registry
        self._artifacts = artifacts

    def project(self, attempt: Attempt) -> dict[str, JsonValue]:
        """Return the final report view for one Attempt with a terminal Agent handoff."""
        if attempt.attempt_report_digest is None or attempt.attempt_report_status is None:
            raise ValueError("Attempt has no terminal report")
        agent = self._agent_report(attempt)
        parent = self._registry.get_kernel_revision(attempt.input_kernel_revision_id)
        candidate = (
            None
            if attempt.output_kernel_revision_id is None
            else self._registry.get_kernel_revision(attempt.output_kernel_revision_id)
        )
        versions = self._kernel_versions(attempt)
        parent_value = self._kernel_value(parent, versions[parent.id])
        candidate_value = (
            None
            if candidate is None
            else self._candidate_value(
                attempt,
                candidate,
                versions[candidate.id],
                parent_value,
            )
        )
        agent_fields = {
            key: value
            for key, value in agent.items()
            if key not in {"schema_version", "attempt_id", "status", "experiments"}
        }
        experiments = agent.get("experiments")
        if not isinstance(experiments, list):
            raise ValueError("Agent Attempt Report has invalid experiments")
        return cast(
            dict[str, JsonValue],
            {
                "schema_version": RUNTIME_ATTEMPT_REPORT_VERSION,
                "attempt_id": str(attempt.id),
                "status": attempt.attempt_report_status.value,
                "created_at": attempt.created_at,
                "completed_at": attempt.completed_at,
                "parent_kernel": parent_value,
                "candidate_kernel": candidate_value,
                "production_gate": self._production_gate(attempt),
                **agent_fields,
                "experiments": experiments,
            },
        )

    def _production_gate(self, attempt: Attempt) -> dict[str, JsonValue]:
        epoch = self._registry.get_epoch(attempt.epoch_id)
        lineage = self._registry.get_lineage(epoch.lineage_id)
        campaign = self._registry.get_campaign(lineage.campaign_id)
        contract = load_evaluation_contract(
            self._artifacts,
            campaign.evaluation_contract_digest,
        )
        if not contract.production_gate:
            return {
                "enabled": False,
                "result": "NOT_ENABLED",
                "failure_reason": None,
            }
        if attempt.output_kernel_revision_id is not None:
            return {
                "enabled": True,
                "result": "PASS",
                "failure_reason": None,
            }
        failure = attempt.failure_reason
        if failure is not None and "production gate" in failure.lower():
            return {
                "enabled": True,
                "result": "FAIL",
                "failure_reason": failure,
            }
        return {
            "enabled": True,
            "result": "NOT_RUN",
            "failure_reason": None,
        }

    def _agent_report(self, attempt: Attempt) -> dict[str, JsonValue]:
        if attempt.attempt_report_digest is None:
            raise AssertionError("Agent report loader requires a Digest")
        artifact = self._artifacts.verify(attempt.attempt_report_digest)
        if artifact.kind is not ArtifactKind.ATTEMPT_REPORT:
            raise ValueError("Attempt terminal report has the wrong Artifact kind")
        try:
            value = json.loads((artifact.payload_path / "value.json").read_bytes())
        except (FileNotFoundError, json.JSONDecodeError) as error:
            raise ValueError("Attempt terminal report is not valid JSON") from error
        if (
            not isinstance(value, dict)
            or value.get("attempt_id") != attempt.id
            or value.get("status") != attempt.attempt_report_status
        ):
            raise ValueError("Attempt terminal report disagrees with Registry state")
        return cast(dict[str, JsonValue], value)

    def _kernel_versions(self, attempt: Attempt) -> dict[KernelRevisionId, int]:
        epoch = self._registry.get_epoch(attempt.epoch_id)
        return {
            entry.revision.id: entry.revision_number
            for entry in self._registry.list_lineage_kernels(epoch.lineage_id)
        }

    def _kernel_value(self, kernel: KernelRevision, version: int) -> dict[str, JsonValue]:
        return {
            "version": f"v{version}",
            "kernel_artifact_digest": str(kernel.artifact_digest),
            "gateway_result": self._gateway_result(kernel),
        }

    def _candidate_value(
        self,
        attempt: Attempt,
        candidate: KernelRevision,
        version: int,
        parent: dict[str, JsonValue],
    ) -> dict[str, JsonValue]:
        value = self._kernel_value(candidate, version)
        value["status"] = (
            "incorrect"
            if not candidate.evaluation.correct
            else "retained"
            if attempt.accepted_as_branch_best
            else "rejected"
        )
        value["comparison_with_parent"] = self._comparison(
            cast(dict[str, JsonValue] | None, parent.get("gateway_result")),
            cast(dict[str, JsonValue] | None, value.get("gateway_result")),
        )
        return value

    def _gateway_result(self, kernel: KernelRevision) -> dict[str, JsonValue]:
        latency = kernel.evaluation.latency_us
        operation, status, by_shape = self._gateway_projection(
            kernel.evaluation.gateway_result_digest
        )
        arithmetic_mean = statistics.fmean(by_shape.values()) if by_shape else None
        return {
            "operation": operation,
            "status": status,
            "correct": kernel.evaluation.correct,
            "correctness": self._correctness_projection(
                kernel.evaluation.gateway_result_digest,
                passed=kernel.evaluation.correct,
            ),
            "latency_us_geomean": latency,
            "latency_us_arith_mean": arithmetic_mean,
            "latency_us_by_shape": cast(dict[str, JsonValue], by_shape),
        }

    def _correctness_projection(
        self,
        digest: ArtifactDigest,
        *,
        passed: bool,
        seen: set[ArtifactDigest] | None = None,
    ) -> dict[str, JsonValue]:
        visited = set() if seen is None else set(seen)
        if digest in visited:
            raise ValueError("Gateway Result measurement graph contains a cycle")
        visited.add(digest)
        artifact = self._artifacts.verify(digest)
        if artifact.kind is not ArtifactKind.GATEWAY_RESULT:
            raise ValueError("Kernel evaluation does not reference a Gateway Result")
        try:
            value = json.loads((artifact.payload_path / "value.json").read_bytes())
        except (FileNotFoundError, json.JSONDecodeError) as error:
            raise ValueError("Gateway Result is not valid JSON") from error
        if not isinstance(value, dict):
            raise ValueError("Gateway Result must be an object")
        operation = value.get("operation")
        if operation == "same_allocation_abba":
            candidate = value.get("candidate")
            return correctness_summary(candidate, passed=passed)
        if operation == "evaluate_comparison":
            measurements = value.get("measurements")
            summaries: list[object] = []
            if isinstance(measurements, list):
                for measurement in measurements:
                    if not isinstance(measurement, Mapping):
                        continue
                    raw_digest = measurement.get("gateway_result_digest")
                    if not isinstance(raw_digest, str):
                        continue
                    summaries.append(
                        self._correctness_projection(
                            parse_artifact_digest(raw_digest),
                            passed=passed,
                            seen=visited,
                        )
                    )
            return merge_correctness_summaries(summaries, passed=passed)
        return correctness_summary(value, passed=passed)

    @staticmethod
    def _comparison(
        parent: dict[str, JsonValue] | None,
        candidate: dict[str, JsonValue] | None,
    ) -> dict[str, JsonValue] | None:
        if parent is None or candidate is None:
            return None
        parent_latency = _positive_number(parent.get("latency_us_geomean"))
        candidate_latency = _positive_number(candidate.get("latency_us_geomean"))
        if parent_latency is None or candidate_latency is None:
            return None
        parent_shapes = _shape_mapping(parent.get("latency_us_by_shape"))
        candidate_shapes = _shape_mapping(candidate.get("latency_us_by_shape"))
        shared = sorted(set(parent_shapes).intersection(candidate_shapes), key=_shape_sort_key)
        return {
            "latency_us_geomean_delta": candidate_latency - parent_latency,
            "improvement_percent": (parent_latency - candidate_latency) / parent_latency * 100.0,
            "latency_us_delta_by_shape": {
                shape_id: candidate_shapes[shape_id] - parent_shapes[shape_id]
                for shape_id in shared
            },
            "improvement_percent_by_shape": {
                shape_id: (parent_shapes[shape_id] - candidate_shapes[shape_id])
                / parent_shapes[shape_id]
                * 100.0
                for shape_id in shared
            },
        }

    def _gateway_projection(
        self,
        digest: ArtifactDigest,
    ) -> tuple[str, str, dict[str, float]]:
        artifact = self._artifacts.verify(digest)
        if artifact.kind is not ArtifactKind.GATEWAY_RESULT:
            raise ValueError("Kernel evaluation does not reference a Gateway Result")
        try:
            value = json.loads((artifact.payload_path / "value.json").read_bytes())
        except (FileNotFoundError, json.JSONDecodeError) as error:
            raise ValueError("Gateway Result is not valid JSON") from error
        if not isinstance(value, dict):
            raise ValueError("Gateway Result must be an object")
        operation = value.get("operation")
        status = value.get("status")
        return (
            operation if isinstance(operation, str) and operation else "evaluate",
            status if isinstance(status, str) and status else "completed",
            self._latency_by_shape_value(value, seen={digest}),
        )

    def _latency_by_shape_digest(
        self,
        digest: ArtifactDigest,
        *,
        seen: set[ArtifactDigest],
    ) -> dict[str, float]:
        if digest in seen:
            raise ValueError("Gateway Result measurement graph contains a cycle")
        seen.add(digest)
        artifact = self._artifacts.verify(digest)
        if artifact.kind is not ArtifactKind.GATEWAY_RESULT:
            raise ValueError("Kernel evaluation does not reference a Gateway Result")
        try:
            value = json.loads((artifact.payload_path / "value.json").read_bytes())
        except (FileNotFoundError, json.JSONDecodeError) as error:
            raise ValueError("Gateway Result is not valid JSON") from error
        if not isinstance(value, dict):
            raise ValueError("Gateway Result must be an object")
        return self._latency_by_shape_value(value, seen=seen)

    def _latency_by_shape_value(
        self,
        value: Mapping[str, object],
        *,
        seen: set[ArtifactDigest],
    ) -> dict[str, float]:
        operation = value.get("operation")
        if operation == "same_allocation_abba":
            candidate = value.get("candidate")
            if isinstance(candidate, Mapping):
                return _shape_mapping(candidate.get("latency_us_by_shape"))
        if operation == "evaluate_comparison":
            measurements = value.get("measurements")
            if isinstance(measurements, list):
                children: list[dict[str, float]] = []
                for measurement in measurements:
                    if not isinstance(measurement, Mapping):
                        continue
                    raw_digest = measurement.get("gateway_result_digest")
                    if not isinstance(raw_digest, str):
                        continue
                    children.append(
                        self._latency_by_shape_digest(
                            parse_artifact_digest(raw_digest),
                            seen=set(seen),
                        )
                    )
                return _mean_shape_mappings(children)
        direct = _shape_mapping(value.get("latency_us_by_shape"))
        if direct:
            return direct
        for key in ("result", "worker_result", "job"):
            nested = value.get(key)
            if isinstance(nested, Mapping):
                projected = self._latency_by_shape_value(nested, seen=set(seen))
                if projected:
                    return projected
        stages = value.get("completed_stages")
        if isinstance(stages, list):
            for stage in reversed(stages):
                if isinstance(stage, Mapping):
                    projected = self._latency_by_shape_value(stage, seen=set(seen))
                    if projected:
                        return projected
        jobs = value.get("jobs")
        if isinstance(jobs, list):
            return _mean_shape_mappings(
                [
                    projected
                    for job in jobs
                    if isinstance(job, Mapping)
                    and (projected := self._latency_by_shape_value(job, seen=set(seen)))
                ]
            )
        return {}


def _positive_number(value: object) -> float | None:
    if (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) > 0
    ):
        return float(value)
    return None


def _shape_mapping(value: object) -> dict[str, float]:
    if not isinstance(value, Mapping):
        return {}
    return {
        str(shape_id): number
        for shape_id, raw in value.items()
        if (number := _positive_number(raw)) is not None
    }


def _mean_shape_mappings(values: list[dict[str, float]]) -> dict[str, float]:
    if not values:
        return {}
    shared = set(values[0])
    for value in values[1:]:
        shared.intersection_update(value)
    return {
        shape_id: statistics.fmean(value[shape_id] for value in values)
        for shape_id in sorted(shared, key=_shape_sort_key)
    }


def _shape_sort_key(value: str) -> tuple[int, int | str]:
    return (0, int(value)) if value.isdigit() else (1, value)
