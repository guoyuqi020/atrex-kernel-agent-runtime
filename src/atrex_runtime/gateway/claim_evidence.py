"""Evidence eligibility for scoped Agent claims, never a causal truth oracle."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, cast

from ..artifacts.local import ArtifactKind, LocalArtifactStore
from ..domain.errors import InfrastructureError
from ..domain.ids import AttemptId
from ..workers.attempt_report import AttemptExperimentSubjectV1, AttemptReportV12
from .artifact_subjects import resolve_artifact_subject
from .control import SqliteGatewayControl
from .control_models import GatewayKernelTrialObservation, GatewayKernelTrialRecord

type Assessment = Literal["unresolved", "supported", "refuted"]


@dataclass(frozen=True, slots=True)
class ClaimAssessment:
    assessment: Assessment
    notes: tuple[str, ...] = ()


def _result_payload(
    artifacts: LocalArtifactStore,
    observation: GatewayKernelTrialObservation,
    trial: GatewayKernelTrialRecord,
) -> dict[str, object]:
    digest = observation.result_artifact_digest
    assert digest is not None
    artifact = artifacts.verify(digest)
    if artifact.kind not in {ArtifactKind.RESULT_ARTIFACT, ArtifactKind.GATEWAY_RESULT}:
        raise InfrastructureError("Claim evidence has an invalid Result Artifact kind")
    try:
        value = json.loads((artifact.payload_path / "value.json").read_bytes())
        metadata = (
            json.loads((artifact.payload_path / "metadata.json").read_bytes())
            if artifact.kind is ArtifactKind.RESULT_ARTIFACT
            else value
        )
    except (OSError, ValueError) as error:
        raise InfrastructureError("Claim evidence has invalid Result JSON") from error
    if (
        not isinstance(value, dict)
        or value.get("operation") != observation.operation.value
        or not isinstance(value.get("status"), str)
        or not isinstance(metadata, dict)
        or metadata.get("kernel_artifact_digest") != trial.kernel_artifact_digest
    ):
        raise InfrastructureError("Claim Result Artifact disagrees with its recorded observation")
    return value


def _has_latency(payload: Mapping[str, object]) -> bool:
    value = payload.get("latency_us_geomean")
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value > 0
    )


def _eligible_result(value: Mapping[str, object], claim_kind: str) -> bool:
    if value.get("status") != "completed":
        return False
    # A completed Check can establish a compile/launch observation, not a bottleneck.
    if claim_kind == "observation":
        return True
    operation = value.get("operation")
    payload = value.get("result")
    if not isinstance(payload, dict):
        return False
    if payload.get("status") in {"failed", "cancelled", "rejected"}:
        return False
    if operation == "dev":
        return bool(payload.get("exit_code", 0) == 0)
    if operation == "profile":
        return claim_kind == "causal_hypothesis"
    if operation != "evaluate" or payload.get("mode") == "correctness_only":
        return False
    if payload.get("correct") is not True:
        return False
    if _has_latency(payload):
        return True
    # Exploratory ABBA has two measured sides rather than a top-level latency.
    return all(
        isinstance(side := payload.get(name), dict) and _has_latency(side)
        for name in ("baseline", "candidate")
    )


def assess_claim(
    *,
    attempt_id: AttemptId,
    assessment: str,
    claim_kind: str,
    claim: str | None,
    scope: str | None,
    supporting_results: Sequence[Mapping[str, object]],
    trials: Iterable[GatewayKernelTrialRecord],
    artifacts: LocalArtifactStore,
) -> ClaimAssessment:
    """Reject invalid references; downgrade insufficient evidence without blocking work.

    A retained supported/refuted label is still the Agent's scoped interpretation.
    Operation eligibility cannot prove that an experiment isolated the claimed cause.
    """
    if assessment not in {"unresolved", "supported", "refuted"}:
        raise ValueError("Claim assessment must be unresolved, supported, or refuted")
    if claim_kind not in {"observation", "implementation_outcome", "causal_hypothesis"}:
        raise ValueError(
            "Claim kind must be observation, implementation_outcome, or causal_hypothesis"
        )
    visible = tuple(trials)
    eligible = False
    for raw in supporting_results:
        subject = AttemptExperimentSubjectV1.model_validate(raw)
        trial = resolve_artifact_subject(
            visible, subject.model_dump(mode="json"), attempt_id=attempt_id
        )
        selected = set(subject.result_artifact_digests)
        for observation in trial.observations:
            if observation.result_artifact_digest in selected:
                value = _result_payload(artifacts, observation, trial)
                eligible = _eligible_result(value, claim_kind) or eligible
    if assessment == "unresolved":
        return ClaimAssessment("unresolved")
    missing: list[str] = []
    if claim is None or not claim.strip():
        missing.append("an explicit claim")
    if scope is None or not scope.strip():
        missing.append("the tested scope (kernel, hardware and workloads)")
    if not eligible:
        missing.append(
            "a completed Kernel-bound Result appropriate to the claim kind; "
            "Check/correctness-only evidence cannot establish performance or a bottleneck"
        )
    if missing:
        return ClaimAssessment(
            "unresolved",
            (f"{assessment} downgraded to unresolved: missing " + "; ".join(missing),),
        )
    return ClaimAssessment(cast(Assessment, assessment))


def normalize_report_findings(
    report: AttemptReportV12,
    *,
    control: SqliteGatewayControl,
    artifacts: LocalArtifactStore,
) -> tuple[AttemptReportV12, tuple[str, ...]]:
    """Apply the same evidence boundary with either, both, or neither Journal module."""
    if not report.findings:
        return report, ()
    _, visible_ids = control.visible_kernel_trial_attempt_ids(report.attempt_id)
    trials = control.list_kernel_trials(visible_ids, limit=5_000)
    experiments = {item.experiment_id: item for item in report.experiments}
    findings = []
    notes: list[str] = []
    for index, finding in enumerate(report.findings):
        subjects = [subject.model_dump(mode="json") for subject in finding.supporting_results]
        # Old free-text findings stay unresolved; their historical prose is not promoted.
        if finding.assessment != "unresolved":
            for experiment_id in finding.supporting_experiment_ids:
                experiment = experiments[experiment_id]
                # Measuring the baseline cannot validate an untested/failed intervention.
                # A claim about that baseline must explicitly select its Result instead.
                if experiment.after is not None:
                    subjects.append(experiment.after.model_dump(mode="json"))
        result = assess_claim(
            attempt_id=report.attempt_id,
            assessment=finding.assessment,
            claim_kind=finding.claim_kind,
            claim=finding.claim,
            scope=finding.scope,
            supporting_results=subjects,
            trials=trials,
            artifacts=artifacts,
        )
        findings.append(finding.model_copy(update={"assessment": result.assessment}))
        notes.extend(f"findings[{index}]: {note}" for note in result.notes)
    return report.model_copy(update={"findings": tuple(findings)}), tuple(notes)
