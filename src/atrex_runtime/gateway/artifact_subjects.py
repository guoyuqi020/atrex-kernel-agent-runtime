"""Resolve Agent-facing Artifact identities against Runtime-owned observations."""

from collections.abc import Iterable, Mapping

from ..domain.ids import AttemptId, parse_artifact_digest
from .control_models import GatewayKernelTrialRecord


def artifact_experiment_view(value: Mapping[str, object]) -> dict[str, object]:
    """Project an old immutable Journal without requiring its redundant Trial IDs."""
    result = dict(value)
    for side in ("before", "after"):
        subject = result.get(side)
        if isinstance(subject, Mapping):
            result[side] = {k: v for k, v in subject.items() if k != "kernel_trial_id"}
    return result


def resolve_artifact_subject(
    trials: Iterable[GatewayKernelTrialRecord],
    subject: Mapping[str, object],
    *,
    attempt_id: AttemptId | None = None,
    require_current: bool = False,
) -> GatewayKernelTrialRecord:
    """Choose one observation group, never combine evidence across generations.

    A Journal snapshot includes Result Artifact digests, which freeze its exact
    evidence. A result-only request prefers this Attempt, then the latest
    visible generation. A failed recent evaluation cannot fall back to older success.
    """
    kernel = subject.get("kernel_artifact_digest")
    digest = None if kernel is None else parse_artifact_digest(str(kernel))
    results = subject.get("result_artifact_digests", ())
    if "result_artifact_digest" in subject:
        results = (subject["result_artifact_digest"],)
    if not isinstance(results, (list, tuple)) or any(not isinstance(x, str) for x in results):
        raise ValueError("Kernel reference Result Artifacts must be an array of digests")
    if not results:
        raise ValueError("Kernel reference requires a Result Artifact digest")
    for result in results:
        parse_artifact_digest(str(result))
    # Only frozen historical records have this field; new Agent requests do not.
    historical_trial = subject.get("kernel_trial_id")
    candidates = [
        trial
        for trial in trials
        if (digest is None or trial.kernel_artifact_digest == digest)
        and (historical_trial is None or trial.id == historical_trial)
        and (not require_current or trial.attempt_id == attempt_id)
        and set(results).issubset({o.result_artifact_digest for o in trial.observations})
    ]
    if not candidates:
        raise ValueError(
            "Kernel/Result Artifacts are outside the permitted visible history"
            + (
                " for this logical Attempt; use action=adopt for historical evidence"
                if require_current
                else ""
            )
        )
    if len({t.kernel_artifact_digest for t in candidates}) != 1:
        raise ValueError("Result Artifact has ambiguous Kernel ownership")
    return max(
        candidates,
        key=lambda t: (
            t.attempt_id == attempt_id,
            t.created_at,
            t.recovery_generation,
            t.ordinal,
            t.id,
        ),
    )
