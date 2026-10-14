"""Deterministic Journal facts from the same immutable Results an Agent can read.

Never inspect private worker results, infer causes, or treat a partial measurement as
a full comparison. Predictions remain input-only audit data, not Agent feedback.
"""

from __future__ import annotations

import json
import math
import statistics
from collections.abc import Mapping
from typing import Any

from ..artifacts.local import ArtifactKind, LocalArtifactStore
from .control import SqliteGatewayControl
from .control_models import GatewayKernelTrialRecord
from .result_metrics import input_result_payload


def _positive(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) and value > 0 else None


def _latencies(value: Mapping[str, Any]) -> dict[str, float]:
    raw = value.get("latency_us_by_shape")
    if not isinstance(raw, dict):
        return {}
    valid = {str(k): number for k, v in raw.items() if (number := _positive(v)) is not None}
    # Do not silently exclude malformed Shapes and call the remainder a full result.
    return valid if len(valid) == len(raw) else {}


def _classification(before: float, after: float) -> str:
    ratio = after / before
    if math.isclose(ratio, 0.99, rel_tol=1e-12) or math.isclose(ratio, 1.01, rel_tol=1e-12):
        return "retained"
    return "improved" if ratio < 0.99 else "degraded" if ratio > 1.01 else "retained"


def _metrics(payload: Mapping[str, Any]) -> dict[str, Any]:
    by_shape = _latencies(payload)
    return {
        "correct": payload.get("correct"),
        "correctness": payload.get("correctness"),
        "measured_shape_count": len(by_shape),
        "shape_ids": sorted(by_shape),
        "latency_us_geomean": (
            math.exp(statistics.fmean(math.log(v) for v in by_shape.values()))
            if by_shape
            else _positive(payload.get("latency_us_geomean"))
        ),
    }


def _compare(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
    if before.get("correct") is not True or after.get("correct") is not True:
        return {"available": False, "reason": "both_sides_must_pass_correctness"}
    a, b = _latencies(before), _latencies(after)
    if not a or not b:
        return {"available": False, "reason": "per_shape_measurements_unavailable"}
    if a.keys() != b.keys():
        return {"available": False, "reason": "shape_coverage_mismatch"}
    ga = math.exp(statistics.fmean(math.log(v) for v in a.values()))
    gb = math.exp(statistics.fmean(math.log(v) for v in b.values()))
    counts = dict.fromkeys(("improved", "retained", "degraded"), 0)
    for key in a:
        counts[_classification(a[key], b[key])] += 1
    return {
        "available": True,
        "aggregation": "geometric_mean_latency_over_identical_shapes",
        "shape_count": len(a),
        "baseline_latency_us_geomean": ga,
        "candidate_latency_us_geomean": gb,
        "latency_reduction_pct": 100 * (1 - gb / ga),
        "speedup": ga / gb,
        "classification": _classification(ga, gb),
        "per_shape_classification_counts": counts,
        "classification_threshold_pct": 1.0,
    }


def experiment_measurement_summary(
    experiment: Mapping[str, Any],
    *,
    trials: Mapping[str, GatewayKernelTrialRecord],
    control: SqliteGatewayControl,
    artifacts: LocalArtifactStore,
) -> dict[str, Any]:
    """Summarize exact selected observations; ambiguity never chooses a winner."""
    observations = {
        (str(trial.kernel_artifact_digest), str(observation.result_artifact_digest)): (
            trial,
            observation,
        )
        for trial in trials.values()
        for observation in trial.observations
        if observation.result_artifact_digest is not None
    }
    sides: dict[str, list[dict[str, Any]]] = {"before": [], "after": []}
    # Private comparability keys are used internally, never emitted to the Agent.
    scopes: dict[str, object] = {}
    payloads: dict[str, dict[str, Any]] = {}
    for side, entries in sides.items():
        subject = experiment.get(side)
        if not isinstance(subject, Mapping):
            continue
        kernel = str(subject["kernel_artifact_digest"])
        for digest in subject["result_artifact_digests"]:
            trial, observation = observations[(kernel, str(digest))]
            assert observation.result_artifact_digest is not None
            artifact = artifacts.verify(observation.result_artifact_digest)
            # Only sealed input ancestry authorizes projecting an authoritative
            # Gateway result. Other private/legacy results remain unavailable.
            value: dict[str, Any]
            if trial.input_revision is not None:
                value = input_result_payload(artifacts, trial.input_revision)
            elif artifact.kind is not ArtifactKind.RESULT_ARTIFACT:
                entries.append(
                    {
                        "kernel_artifact_digest": kernel,
                        "result_artifact_digest": str(digest),
                        "status": "unavailable",
                        "reason": "agent_visible_result_required",
                    }
                )
                continue
            else:
                value = json.loads((artifact.payload_path / "value.json").read_bytes())
                metadata = json.loads((artifact.payload_path / "metadata.json").read_bytes())
                if metadata.get("kernel_artifact_digest") != kernel:
                    raise ValueError("Measurement summary Result does not match its Kernel")
            operator, hardware, dsl, contract = control.evaluation_identity(trial.attempt_id)
            payload = value.get("result")
            payload = payload if isinstance(payload, dict) else {}
            entry = {
                "kernel_artifact_digest": kernel,
                "result_artifact_digest": str(digest),
                "operation": value.get("operation"),
                "status": value.get("status"),
                "hardware": hardware,
                "operator": operator,
                "dsl": dsl.value,
            }
            if value.get("operation") == "evaluate":
                comparison_spec = payload.get("comparison")
                candidate_payload = payload.get("candidate")
                measured = (
                    candidate_payload
                    if isinstance(comparison_spec, dict)
                    and comparison_spec.get("method") == "abba"
                    and isinstance(candidate_payload, dict)
                    else payload
                )
                entry.update(
                    mode=payload.get("mode", "full"),
                    input_scope=payload.get("input_scope", "contract"),
                    **_metrics(measured),
                )
                entry["correct"] = payload.get("correct")
            for field in ("error", "failures"):
                if field in payload:
                    entry[field] = payload[field]
            entries.append(entry)
            payloads[str(digest)] = payload
            if observation.gateway_result_digest is not None:
                private = artifacts.verify(observation.gateway_result_digest)
                parameters_file = private.payload_path / "evaluation-request.json"
                if parameters_file.is_file():
                    # Only the request contract is used to check comparability. No private
                    # result, Test timing, input source or prediction enters the summary.
                    parameters = json.loads(parameters_file.read_bytes())
                    scopes[str(digest)] = (
                        operator,
                        hardware,
                        dsl,
                        contract,
                        parameters.get("input_py"),
                        parameters.get("shapes"),
                        payload.get("measurement_aggregation"),
                    )
                elif (
                    entry.get("operation") == "evaluate" and entry.get("input_scope") == "contract"
                ):
                    # Ordinary contract Evaluate deliberately stores no exploratory request.
                    # Its immutable campaign contract defines the input domain.
                    scopes[str(digest)] = (
                        operator,
                        hardware,
                        dsl,
                        contract,
                        None,
                        None,
                        payload.get("measurement_aggregation"),
                    )

    comparison: dict[str, Any] = {"available": False, "reason": "one_result_per_side_required"}
    after = sides["after"]
    if len(after) == 1:
        entry = after[0]
        payload = payloads.get(entry["result_artifact_digest"], {})
        abba = payload.get("comparison")
        if isinstance(abba, dict) and abba.get("method") == "abba":
            a, b = payload.get("baseline"), payload.get("candidate")
            comparison = {"available": False, "reason": "abba_not_completed"}
            if (
                entry["status"] == "completed"
                and payload.get("correct") is True
                and entry.get("mode") == "full"
                and isinstance(a, dict)
                and isinstance(b, dict)
            ):
                comparison = _compare(a, b)
            comparison.update(
                method="abba",
                repeats=abba.get("repeats"),
                baseline_kernel_artifact_digest=payload.get("baseline_kernel_artifact_digest"),
                candidate_kernel_artifact_digest=entry["kernel_artifact_digest"],
                result_artifact_digest=entry["result_artifact_digest"],
            )
        elif len(sides["before"]) == 1:
            before = sides["before"][0]
            keys = [before["result_artifact_digest"], entry["result_artifact_digest"]]
            a, b = (payloads.get(key, {}) for key in keys)
            comparison = {"available": False, "reason": "requires_completed_full_evaluations"}
            if all(
                item.get("operation") == "evaluate"
                and item.get("status") == "completed"
                and item.get("mode") == "full"
                for item in (before, entry)
            ) and not a.get("comparison"):
                if keys[0] not in scopes or scopes.get(keys[0]) != scopes.get(keys[1]):
                    comparison = {"available": False, "reason": "measurement_scope_mismatch"}
                else:
                    comparison = _compare(a, b)
            comparison.update(
                method="same_result_reuse" if keys[0] == keys[1] else "independent_evaluations"
            )
    return {
        "schema_version": 1,
        "source": "runtime",
        **sides,
        "comparison": comparison,
        "interpretation_notice": (
            "Measured observations only; no causal conclusion or noise exclusion. "
            "Shape counts describe returned timings, not an inferred correctness pass count."
        ),
    }


def compact_measurement_summary(summary: object) -> object:
    """Keep indexes bounded; exact per-Result details remain in load-experiment."""
    if not isinstance(summary, dict):
        return None
    return {"source": "runtime", "comparison": summary.get("comparison")}
