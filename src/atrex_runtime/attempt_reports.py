"""Runtime-owned final Attempt Report projections."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from typing import Final, cast

from .artifacts.local import ArtifactKind, JsonValue, LocalArtifactStore
from .domain.ids import KernelRevisionId
from .domain.models import Attempt, KernelRevision
from .gateway.contract import load_evaluation_contract
from .gateway.result_metrics import gateway_result_projection
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
        return gateway_result_projection(
            self._artifacts,
            kernel.evaluation.gateway_result_digest,
            correct=kernel.evaluation.correct,
            latency_us=kernel.evaluation.latency_us,
        )

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


def _shape_sort_key(value: str) -> tuple[int, int | str]:
    return (0, int(value)) if value.isdigit() else (1, value)
