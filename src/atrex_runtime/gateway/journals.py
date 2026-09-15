"""Authoritative Runtime-owned Direction and Experiment Journals."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import cast
from uuid import uuid4

from ..artifacts.local import ArtifactKind, JsonValue, LocalArtifactStore
from ..direction_genealogy import RELATIONSHIP_FIELDS, relationship_fields, validate_relationship
from ..domain.errors import (
    DirectionConcurrencyError,
    DirectionLookupError,
    InfrastructureError,
    OptimizerSuggestionForbiddenError,
    SuggestedDirectionTransitionError,
)
from ..domain.ids import AttemptId, parse_artifact_digest
from ..workers.attempt_report import (
    AttemptDirectionEventV1,
    AttemptExperimentV8,
    AttemptReportV12,
)
from .artifact_subjects import resolve_artifact_subject
from .control import SqliteGatewayControl
from .control_models import GatewayAuthorization, GatewayKernelTrialRecord, GatewayOperation
from .protocol import (
    DirectionLoadRequestV2,
    DirectionUpdateRequestV2,
    ExperimentLoadRequestV2,
    ExperimentRecordRequestV2,
    GatewayProxyRequestV2,
)

_DIRECTION_PROPOSAL_FIELDS = {
    "action",
    "name",
    "hypothesis",
    "rationale",
    "plan",
    "success_criteria",
    "stop_conditions",
}
_DIRECTION_UPDATE_FIELDS = {"action", "direction_id", "analysis"}
_DIRECTION_CLOSURE_FIELDS = _DIRECTION_UPDATE_FIELDS | {
    "hypothesis_status",
    "supporting_experiment_ids",
}
_DIRECTION_CLOSURES = {"complete", "abandon", "block", "defer"}
_EXPERIMENT_FIELDS = {
    "direction_id",
    "name",
    "hypothesis",
    "change",
    "before",
    "after",
    "evidence",
    "analysis",
    "action",
}
_DIRECTION_STATUSES = {
    "propose": "proposed",
    "suggest": "suggested",
    "start": "in_progress",
    "complete": "completed",
    "abandon": "abandoned",
    "block": "blocked",
    "defer": "deferred",
}


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be non-empty text")
    return value


def _one_edit_apart(first: str, second: str) -> bool:
    """Match exactly one substitution, insertion, or deletion, not a fuzzy identity."""
    if len(first) == len(second):
        return sum(left != right for left, right in zip(first, second, strict=True)) == 1
    if abs(len(first) - len(second)) != 1:
        return False
    shorter, longer = (first, second) if len(first) < len(second) else (second, first)
    for index, character in enumerate(shorter):
        if character != longer[index]:
            return shorter[index:] == longer[index + 1 :]
    return True


def _text_array(value: object, label: str, *, required: bool = False) -> list[str]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise ValueError(f"{label} must be an array of non-empty text")
    if required and not value:
        raise ValueError(f"{label} must not be empty")
    return value


@dataclass(frozen=True, slots=True)
class RuntimeJournalService:
    """Validate, persist, and project live optimization Journals."""

    control: SqliteGatewayControl
    artifacts: LocalArtifactStore

    def validate_report_journal(self, report: AttemptReportV12) -> None:
        """An empty or edited report must not erase a live authoritative Journal."""
        if not self._is_bootstrap(report.attempt_id) and any(
            event.action == "suggest" for event in report.direction_events
        ):
            raise OptimizerSuggestionForbiddenError("report.direction_events.action")
        in_progress = sorted(
            direction_id
            for direction_id, direction in self._direction_views(report.attempt_id).items()
            if direction["status"] == "in_progress"
        )
        if in_progress:
            raise ValueError(
                "Attempt report cannot leave a Runtime-owned Direction in progress: "
                f"{in_progress}; record its Experiment, then block or defer it before submitting"
            )
        events = self._current_direction_events(report.attempt_id)
        experiments = self._current_experiments(report.attempt_id)
        if any(event.action == "suggest" for event in report.direction_events) and not events:
            raise ValueError(
                "Bootstrap suggestions must be recorded with update-direction before "
                "attempt-report; a report alone cannot create suggested Directions"
            )
        # Direct report-only clients must not bypass conclusion/evidence validation.
        declared = [
            event for event in report.direction_events if event.hypothesis_status is not None
        ]
        if declared:
            available = {
                str(item["experiment_id"]): item
                for item in self._visible_experiments(report.attempt_id)
            }
            for experiment in report.experiments:
                available.setdefault(experiment.experiment_id, experiment.model_dump(mode="json"))
            for event in declared:
                self._closure_support(
                    report.attempt_id, event.direction_id, event.model_dump(mode="json"), available
                )
        if not events and not experiments:
            if any(event.relationship for event in report.direction_events):
                raise ValueError(
                    "Record derived Directions with update-direction before submitting the report; "
                    "genealogy references must be validated by Runtime"
                )
            # Older report-only clients have no live Journal, but cannot invent the
            # new adoption action to claim Runtime has registered a reuse decision.
            if any(experiment.action == "adopt" for experiment in report.experiments):
                raise ValueError(
                    "First record an adopt Experiment in the Runtime Journal, "
                    "then submit the report"
                )
            return
        expected_events = tuple(
            AttemptDirectionEventV1.model_validate(
                item, context={"trusted_direction_history": True}
            )
            for item in events
        )
        expected_experiments = tuple(
            AttemptExperimentV8.model_validate(item) for item in experiments
        )
        if report.direction_events != expected_events or report.experiments != expected_experiments:
            raise ValueError(
                "Attempt report must match the Runtime-owned Direction and Experiment journals; "
                "refresh the journal snapshot using the attempt-report tool. Record an associated "
                "Experiment before closing any in-progress Direction and submitting"
            )

    def execute(
        self,
        request: GatewayProxyRequestV2,
        authorization: GatewayAuthorization,
    ) -> dict[str, JsonValue]:
        """Execute one Journal operation after capability authorization."""
        if isinstance(request, DirectionUpdateRequestV2):
            return self._update_direction(request, authorization)
        if request.operation == "directions_list":
            directions = self._direction_views(request.attempt_id)
            return cast(
                dict[str, JsonValue],
                {
                    "directions": [
                        {
                            "direction_id": direction["direction_id"],
                            "name": direction["name"],
                            "status": direction["status"],
                            "hypothesis_status": direction["hypothesis_status"],
                            **relationship_fields(direction),
                        }
                        for direction in directions.values()
                    ]
                },
            )
        if isinstance(request, DirectionLoadRequestV2):
            return cast(
                dict[str, JsonValue],
                self._require_direction(
                    request.attempt_id, request.direction_id, field_path="direction_id"
                ),
            )
        if isinstance(request, ExperimentRecordRequestV2):
            return self._record_experiment(request, authorization)
        if request.operation == "experiments_list":
            return cast(
                dict[str, JsonValue],
                {
                    "experiments": [
                        {
                            "experiment_id": experiment["experiment_id"],
                            "name": experiment["name"],
                            "hypothesis": experiment["hypothesis"],
                            "change": experiment["change"],
                            "evidence": experiment["evidence"],
                            "analysis": experiment["analysis"],
                            "action": experiment["action"],
                        }
                        for experiment in self._visible_experiments(request.attempt_id)
                    ]
                },
            )
        if isinstance(request, ExperimentLoadRequestV2):
            for experiment in self._visible_experiments(request.attempt_id):
                if experiment["experiment_id"] == request.experiment_id:
                    visible = dict(experiment)
                    visible.pop("sequence", None)
                    return cast(dict[str, JsonValue], visible)
            raise ValueError("Experiment ID is outside the current Attempt's visible history")
        if request.operation == "journal_snapshot":
            return cast(
                dict[str, JsonValue],
                {
                    "direction_events": list(self._current_direction_events(request.attempt_id)),
                    "experiments": list(self._current_experiments(request.attempt_id)),
                    "directions": list(self._direction_views(request.attempt_id).values()),
                    "citable_profile_results": self._citable_profile_results(request.attempt_id),
                },
            )
        raise ValueError(f"unsupported Runtime Journal operation: {request.operation}")

    def _citable_profile_results(self, attempt_id: AttemptId) -> list[JsonValue]:
        """Project durable visible Profile observations, independently of Experiments."""
        bindings: dict[tuple[str, str], None] = {}
        for trial in self.control.list_kernel_trials(
            self._visible_attempt_ids(attempt_id),
            limit=5_000,
        ):
            for observation in trial.observations:
                if (
                    observation.operation is not GatewayOperation.PROFILE
                    or observation.result_artifact_digest is None
                ):
                    continue
                bindings[
                    (
                        str(trial.kernel_artifact_digest),
                        str(observation.result_artifact_digest),
                    )
                ] = None
        return [
            cast(
                JsonValue,
                {
                    "kernel_artifact_digest": kernel,
                    "result_artifact_digest": result,
                },
            )
            for kernel, result in bindings
        ]

    def _is_bootstrap(self, attempt_id: AttemptId) -> bool:
        try:
            self.control.get_bootstrap_subject(attempt_id)
        except KeyError:
            return False
        return True

    def _report_values(
        self,
        attempt_id: AttemptId,
        field: str,
    ) -> dict[AttemptId, list[dict[str, object]]]:
        values: dict[AttemptId, list[dict[str, object]]] = {}
        for report_attempt_id, digest in self.control.visible_attempt_report_artifacts(attempt_id):
            artifact = self.artifacts.verify(digest)
            if artifact.kind is not ArtifactKind.ATTEMPT_REPORT:
                raise InfrastructureError("Attempt Report history has an invalid Artifact kind")
            try:
                report = json.loads((artifact.payload_path / "value.json").read_bytes())
            except (FileNotFoundError, json.JSONDecodeError) as error:
                raise InfrastructureError("Attempt Report history is invalid JSON") from error
            journal = report.get(field) if isinstance(report, dict) else None
            if not isinstance(journal, list):
                raise InfrastructureError(f"Attempt Report has invalid {field}")
            values[report_attempt_id] = [
                cast(dict[str, object], item) for item in journal if isinstance(item, dict)
            ]
        return values

    def _visible_attempt_ids(self, attempt_id: AttemptId) -> tuple[AttemptId, ...]:
        _lineage_id, attempt_ids = self.control.visible_journal_attempt_ids(attempt_id)
        return attempt_ids

    def _current_direction_events(self, attempt_id: AttemptId) -> tuple[dict[str, object], ...]:
        return tuple(
            AttemptDirectionEventV1.model_validate(
                event, context={"trusted_direction_history": True}
            ).model_dump(mode="json")
            for event in self.control.list_direction_events(attempt_id)
        )

    def _current_experiments(self, attempt_id: AttemptId) -> tuple[dict[str, object], ...]:
        values = self.control.list_experiments(attempt_id)
        trials = self._visible_trials(attempt_id) if self._needs_legacy_mapping(values) else {}
        return tuple(self._normalized_experiment(item, trials) for item in values)

    def _visible_trials(self, attempt_id: AttemptId) -> dict[str, GatewayKernelTrialRecord]:
        return {
            trial.id: trial
            for trial in self.control.list_kernel_trials(
                self._visible_attempt_ids(attempt_id),
                limit=5_000,
            )
        }

    @staticmethod
    def _needs_legacy_mapping(values: object) -> bool:
        if not isinstance(values, (list, tuple)):
            return False
        return any(
            isinstance(side, Mapping) and "gateway_result_digests" in side
            for value in values
            if isinstance(value, Mapping)
            for side in (value.get("before"), value.get("after"))
        )

    @staticmethod
    def _normalized_experiment(
        value: Mapping[str, object],
        trials: Mapping[str, GatewayKernelTrialRecord],
    ) -> dict[str, object]:
        """Upgrade historical raw Gateway references without exposing them to an Agent."""
        normalized = dict(value)
        for side_name in ("before", "after"):
            raw_side = normalized.get(side_name)
            if not isinstance(raw_side, Mapping):
                continue
            side = dict(raw_side)
            if "result_artifact_digests" in side:
                normalized[side_name] = side
                continue
            raw_results = side.pop("gateway_result_digests", None)
            if not isinstance(raw_results, (list, tuple)):
                normalized[side_name] = side
                continue
            trial = trials.get(str(side.get("kernel_trial_id")))
            if trial is None:
                raise InfrastructureError("Historical Experiment references a missing Kernel Trial")
            result_artifacts: list[str] = []
            for raw_result in raw_results:
                mapping = next(
                    (
                        observation.result_artifact_digest
                        for observation in trial.observations
                        if observation.gateway_result_digest is not None
                        and str(observation.gateway_result_digest) == str(raw_result)
                        and observation.result_artifact_digest is not None
                    ),
                    None,
                )
                if mapping is None:
                    raise InfrastructureError(
                        "Historical Experiment has no Agent-visible Result Artifact mapping"
                    )
                result_artifacts.append(str(mapping))
            side["result_artifact_digests"] = result_artifacts
            normalized[side_name] = side
        return cast(
            dict[str, object],
            AttemptExperimentV8.model_validate(
                normalized, context={"trusted_experiment_history": True}
            ).model_dump(mode="json"),
        )

    def _visible_direction_events(self, attempt_id: AttemptId) -> list[dict[str, object]]:
        reports = self._report_values(attempt_id, "direction_events")
        values: list[dict[str, object]] = []
        for visible_attempt_id in self._visible_attempt_ids(attempt_id):
            live = self.control.list_direction_events(visible_attempt_id)
            source = list(live) if live else reports.get(visible_attempt_id, [])
            values.extend(
                AttemptDirectionEventV1.model_validate(
                    item, context={"trusted_direction_history": True}
                ).model_dump(mode="json")
                for item in source
            )
        if len(values) > 4_096:
            raise ValueError("Visible Direction history exceeds its entry limit")
        event_ids = [str(event["direction_event_id"]) for event in values]
        if len(set(event_ids)) != len(event_ids):
            raise ValueError("Visible Direction history contains duplicate Event IDs")
        # A Direction is proposed in one Attempt and advanced in another, so replay
        # must follow when each event was recorded rather than Attempt grouping.
        values.sort(key=lambda event: (str(event["recorded_at"]), str(event["direction_event_id"])))
        return values

    def _visible_experiments(self, attempt_id: AttemptId) -> list[dict[str, object]]:
        reports = self._report_values(attempt_id, "experiments")
        raw_values: list[dict[str, object]] = []
        for visible_attempt_id in self._visible_attempt_ids(attempt_id):
            live = self.control.list_experiments(visible_attempt_id)
            source = list(live) if live else reports.get(visible_attempt_id, [])
            raw_values.extend(source)
        trials = self._visible_trials(attempt_id) if self._needs_legacy_mapping(raw_values) else {}
        values = [self._normalized_experiment(item, trials) for item in raw_values]
        if len(values) > 4_096:
            raise ValueError("Visible Experiment history exceeds its entry limit")
        experiment_ids = [str(experiment["experiment_id"]) for experiment in values]
        if len(set(experiment_ids)) != len(experiment_ids):
            raise ValueError("Visible Experiment history contains duplicate Experiment IDs")
        return values

    def _require_direction(
        self, attempt_id: AttemptId, direction_id: str, *, field_path: str
    ) -> dict[str, object]:
        directions = self._direction_views(attempt_id)
        direction = directions.get(direction_id)
        if direction is None:
            # Never search the global Journal: even typo hints must obey visibility.
            suggestions = tuple(
                sorted(
                    candidate
                    for candidate in directions
                    if _one_edit_apart(direction_id, candidate)
                )[:3]
            )
            raise DirectionLookupError(direction_id, suggestions, field_path=field_path)
        return direction

    def _direction_views(self, attempt_id: AttemptId) -> dict[str, dict[str, object]]:
        directions: dict[str, dict[str, object]] = {}
        for suggested in self.control.visible_suggested_directions(attempt_id):
            direction_id = str(suggested["direction_id"])
            if direction_id in directions:
                raise ValueError("Suggested Direction identity is duplicated")
            directions[direction_id] = {
                "direction_id": direction_id,
                "name": suggested["name"],
                "hypothesis": suggested["hypothesis"],
                "rationale": suggested["rationale"],
                "plan": suggested["plan"],
                "success_criteria": suggested["success_criteria"],
                "stop_conditions": suggested["stop_conditions"],
                "status": suggested["status"],
                "analysis": None,
                "supporting_experiment_ids": [],
                "associated_experiment_ids": [],
                "hypothesis_status": "unresolved",
                **relationship_fields(suggested),
            }
        for event in self._visible_direction_events(attempt_id):
            direction_id = str(event["direction_id"])
            action = str(event["action"])
            existing = directions.get(direction_id)
            if action in {"propose", "suggest"}:
                if existing is not None:
                    raise ValueError("Direction history contains duplicate proposals")
                directions[direction_id] = {
                    "direction_id": direction_id,
                    "name": event["name"],
                    "hypothesis": event["hypothesis"],
                    "rationale": event["rationale"],
                    "plan": event["plan"],
                    "success_criteria": event["success_criteria"],
                    "stop_conditions": event["stop_conditions"],
                    "status": (
                        self.control.suggestion_status(attempt_id, created_epoch_number=0)
                        if action == "suggest"
                        else _DIRECTION_STATUSES[action]
                    ),
                    "analysis": None,
                    "supporting_experiment_ids": [],
                    "associated_experiment_ids": [],
                    "hypothesis_status": "unresolved",
                    **relationship_fields(event),
                }
                continue
            if existing is None:
                raise ValueError("Direction update precedes its proposal")
            existing["status"] = _DIRECTION_STATUSES[action]
            existing["analysis"] = event["analysis"]
            # Lifecycle changes are not proof. A new investigation resets the current
            # assessment; historical closure assertions remain in append-only events.
            assessment = event.get("hypothesis_status")
            existing["hypothesis_status"] = assessment or "unresolved"
            existing["supporting_experiment_ids"] = (
                list(cast(list[str], event["supporting_experiment_ids"]))
                if assessment is not None and action in _DIRECTION_CLOSURES
                else []
            )
            associated = cast(list[str], existing["associated_experiment_ids"])
            for experiment_id in cast(list[str], event["supporting_experiment_ids"]):
                if experiment_id not in associated:
                    associated.append(experiment_id)
        adopted = {
            str(parent)
            for direction in directions.values()
            if direction.get("relationship") == "adoption"
            for parent in cast(list[str], direction.get("derived_from_direction_ids") or [])
        }
        for direction_id in adopted:
            direction = directions.get(direction_id)
            if direction is not None and direction["status"] == "expired":
                direction["status"] = "adopted"
        for experiment in self._visible_experiments(attempt_id):
            direction = directions.get(str(experiment["direction_id"]))
            if direction is None:
                continue
            associated = cast(list[str], direction["associated_experiment_ids"])
            experiment_id = str(experiment["experiment_id"])
            if experiment_id not in associated:
                associated.append(experiment_id)
        return directions

    def _closure_support(
        self,
        attempt_id: AttemptId,
        direction_id: str,
        value: Mapping[str, object],
        experiments: Mapping[str, Mapping[str, object]],
    ) -> list[str]:
        """Validate selected evidence, never certify the Agent's causal interpretation."""
        status = value.get("hypothesis_status")
        if not isinstance(status, str) or status not in {"unresolved", "supported", "refuted"}:
            raise ValueError("hypothesis_status must be unresolved, supported, or refuted")
        selected = value.get("supporting_experiment_ids")
        if (
            not isinstance(selected, list)
            or not selected
            or len(selected) > 32
            or any(not isinstance(item, str) for item in selected)
        ):
            raise ValueError(
                "Direction closure requires at least one associated Experiment explicitly selected "
                "in supporting_experiment_ids (maximum 32); record-experiment first if needed. "
                "Every Experiment must cite at least one real Kernel-bound Gateway Result. "
                "A diagnostic Result may support an unresolved closure, but not a performance claim"
            )
        supporting = cast(list[str], selected)
        if len(set(supporting)) != len(supporting):
            raise ValueError("Direction supporting_experiment_ids must be unique")
        for experiment_id in supporting:
            experiment = experiments.get(experiment_id)
            if experiment is None:
                raise ValueError(
                    f"Supporting Experiment {experiment_id} is outside visible history"
                )
            if experiment.get("direction_id") != direction_id:
                raise ValueError("Supporting Experiment must belong to the Direction being closed")
            if experiment.get("before") is None and experiment.get("after") is None:
                raise ValueError(
                    "Supporting Experiment requires at least one Gateway Result; "
                    "historical unmeasured notes cannot justify a new Direction closure"
                )
        if status != "unresolved":
            trials = tuple(self._visible_trials(attempt_id).values())
            for experiment_id in supporting:
                after = experiments[experiment_id].get("after")
                if not isinstance(after, Mapping):
                    raise ValueError(
                        "supported/refuted requires Gateway Result evidence for every selected "
                        "Experiment; use hypothesis_status=unresolved for unmeasured analysis"
                    )
                trial = resolve_artifact_subject(trials, after, attempt_id=attempt_id)
                results = cast(list[str], after["result_artifact_digests"])
                if not any(
                    item.result_artifact_digest is not None
                    and item.result_artifact_digest in results
                    and self._completed_result(str(item.result_artifact_digest))
                    for item in trial.observations
                ):
                    raise ValueError(
                        "supported/refuted requires a completed Gateway observation; "
                        "infrastructure failures must remain hypothesis_status=unresolved"
                    )
        return list(supporting)

    def _completed_result(self, digest: str) -> bool:
        artifact = self.artifacts.verify(parse_artifact_digest(digest))
        if artifact.kind not in {ArtifactKind.RESULT_ARTIFACT, ArtifactKind.GATEWAY_RESULT}:
            raise InfrastructureError("Direction evidence has an invalid Result Artifact kind")
        try:
            value = json.loads((artifact.payload_path / "value.json").read_bytes())
        except (OSError, json.JSONDecodeError) as error:
            raise InfrastructureError("Direction evidence has invalid Result JSON") from error
        return isinstance(value, dict) and value.get("status") == "completed"

    def _update_direction(
        self,
        request: DirectionUpdateRequestV2,
        authorization: GatewayAuthorization,
    ) -> dict[str, JsonValue]:
        value = dict(request.request)
        action = value.get("action")
        if not isinstance(action, str):
            raise ValueError("Direction action must be text")
        if action in {"propose", "suggest"}:
            if action == "suggest" and not self._is_bootstrap(request.attempt_id):
                raise OptimizerSuggestionForbiddenError("request.action")
            if set(value) - RELATIONSHIP_FIELDS != _DIRECTION_PROPOSAL_FIELDS:
                raise ValueError(
                    f"Direction proposal requires {sorted(_DIRECTION_PROPOSAL_FIELDS)}; "
                    f"optional genealogy fields are {sorted(RELATIONSHIP_FIELDS)}"
                )
            for field in (
                "name",
                "hypothesis",
                "rationale",
                "success_criteria",
                "stop_conditions",
            ):
                _text(value.get(field), f"Direction {field}")
            _text_array(value.get("plan"), "Direction plan", required=True)
            direction_id = f"direction_{uuid4().hex}"
            ancestry: dict[str, object] = {}
            if set(value) & RELATIONSHIP_FIELDS:
                ancestry = validate_relationship(
                    direction_id,
                    value,
                    self._direction_views(request.attempt_id),
                    {
                        str(item["experiment_id"]): item
                        for item in self._visible_experiments(request.attempt_id)
                    },
                )
            event: dict[str, object] = {
                "direction_event_id": f"directionevent_{uuid4().hex}",
                "direction_id": direction_id,
                "recorded_at": datetime.now(UTC).isoformat(),
                **value,
                **ancestry,
                "analysis": None,
                "supporting_experiment_ids": [],
            }
        else:
            if set(value) & RELATIONSHIP_FIELDS:
                raise ValueError(
                    "Direction genealogy is immutable; propose a new derived Direction "
                    "instead of changing ancestry in a lifecycle update"
                )
            expected_fields = (
                _DIRECTION_CLOSURE_FIELDS
                if action in _DIRECTION_CLOSURES
                else _DIRECTION_UPDATE_FIELDS
            )
            if set(value) != expected_fields:
                raise ValueError(
                    f"Direction {action} fields must be exactly {sorted(expected_fields)}; "
                    "closures must select supporting_experiment_ids and declare hypothesis_status"
                )
            if action not in {"start", "complete", "abandon", "block", "defer"}:
                raise ValueError("Direction update action is invalid")
            direction_id = _text(value.get("direction_id"), "Direction ID")
            direction = self._require_direction(
                request.attempt_id, direction_id, field_path="request.direction_id"
            )
            if direction["status"] in {"suggested", "expired", "adopted"}:
                raise SuggestedDirectionTransitionError(
                    direction_id, action, status=str(direction["status"])
                )
            _text(value.get("analysis"), "Direction analysis")
            if action == "start":
                started = {
                    str(event["direction_id"])
                    for event in self._current_direction_events(request.attempt_id)
                    if event["action"] == "start"
                }
                if direction_id not in started and len(started) >= 3:
                    raise ValueError(
                        "Attempt Direction advancement limit exceeded: maximum=3; "
                        f"requested_direction_id={direction_id}; "
                        f"already_advanced_direction_ids={sorted(started)}. "
                        "The requested Direction was not started; keep it proposed or deferred "
                        "for a future Attempt"
                    )
                in_progress = tuple(
                    sorted(
                        visible_direction_id
                        for visible_direction_id, visible_direction in self._direction_views(
                            request.attempt_id
                        ).items()
                        if visible_direction["status"] == "in_progress"
                        and visible_direction_id != direction_id
                    )
                )
                if in_progress:
                    raise DirectionConcurrencyError(direction_id, in_progress)
            supporting = []
            if action in _DIRECTION_CLOSURES:
                supporting = self._closure_support(
                    request.attempt_id,
                    direction_id,
                    value,
                    {
                        str(item["experiment_id"]): item
                        for item in self._visible_experiments(request.attempt_id)
                    },
                )
            event = {
                "direction_event_id": f"directionevent_{uuid4().hex}",
                "direction_id": direction_id,
                "recorded_at": datetime.now(UTC).isoformat(),
                "action": action,
                "name": None,
                "hypothesis": None,
                "rationale": None,
                "plan": [],
                "success_criteria": None,
                "stop_conditions": None,
                "analysis": value["analysis"],
                "supporting_experiment_ids": supporting,
                "hypothesis_status": value.get("hypothesis_status"),
            }
        validated = AttemptDirectionEventV1.model_validate(event).model_dump(mode="json")
        recorded = self.control.append_direction_event(
            request.attempt_id,
            request.idempotency_key,
            validated,
            recovery_generation=authorization.recovery_generation,
        )
        return {"status": "recorded", "direction_id": str(recorded["direction_id"])}

    def _record_experiment(
        self,
        request: ExperimentRecordRequestV2,
        authorization: GatewayAuthorization,
    ) -> dict[str, JsonValue]:
        value = dict(request.request)
        if set(value) != _EXPERIMENT_FIELDS:
            raise ValueError(f"Experiment fields must be exactly {sorted(_EXPERIMENT_FIELDS)}")
        for field in _EXPERIMENT_FIELDS - {"action", "before", "after"}:
            _text(value.get(field), f"Experiment {field}")
        direction_id = str(value["direction_id"])
        direction = self._require_direction(
            request.attempt_id, direction_id, field_path="request.direction_id"
        )
        if direction["status"] not in {
            "in_progress",
            "completed",
            "abandoned",
            "blocked",
            "deferred",
        }:
            raise ValueError(
                "Experiment Direction must be in progress or closed; "
                f"current status is {direction['status']}"
            )
        # Late evidence appends to the Journal without reopening research or
        # rewriting the Direction's lifecycle events.
        allow_baseline = self._is_bootstrap(request.attempt_id)
        actions = {"keep_after", "restore_before", "abandon_direction", "adopt"}
        if allow_baseline:
            actions.add("baseline")
        if value.get("action") not in actions:
            raise ValueError("Experiment action is invalid")
        current = list(self._current_experiments(request.attempt_id))
        if value.get("action") == "baseline" and any(
            experiment.get("action") == "baseline" for experiment in current
        ):
            raise ValueError("Bootstrap Experiment journal may contain only one baseline action")
        _lineage_id, visible_attempt_ids = self.control.visible_kernel_trial_attempt_ids(
            request.attempt_id
        )
        trials = {
            trial.id: trial
            for trial in self.control.list_kernel_trials(visible_attempt_ids, limit=5_000)
        }
        for side_name in ("before", "after"):
            value[side_name] = self._materialize_experiment_subject(
                request.attempt_id,
                side_name,
                value.get(side_name),
                trials,
                allow_historical_after=value.get("action") == "adopt",
            )
        experiment = AttemptExperimentV8.model_validate(
            {
                "experiment_id": f"experiment_{uuid4().hex}",
                "sequence": len(current) + 1,
                "recorded_at": datetime.now(UTC).isoformat(),
                **value,
            }
        ).model_dump(mode="json")
        self._validate_experiment_trials(request.attempt_id, experiment)
        recorded = self.control.append_experiment(
            request.attempt_id,
            request.idempotency_key,
            experiment,
            recovery_generation=authorization.recovery_generation,
        )
        all_current = list(self._current_experiments(request.attempt_id))
        self.control.record_kernel_trial_annotations(
            request.attempt_id,
            all_current,
            recovery_generation=authorization.recovery_generation,
            allow_baseline=allow_baseline,
        )
        return {"status": "recorded", "experiment_id": str(recorded["experiment_id"])}

    @staticmethod
    def _materialize_experiment_subject(
        attempt_id: AttemptId,
        side_name: str,
        value: object,
        trials: Mapping[str, GatewayKernelTrialRecord],
        *,
        allow_historical_after: bool = False,
    ) -> dict[str, JsonValue] | None:
        """Bind one exact Result Artifact to its immutable Kernel and observation."""
        if value is None:
            return None
        if not isinstance(value, Mapping) or set(value) != {"result_artifact_digest"}:
            raise ValueError(
                f"Experiment {side_name} fields must be exactly ['result_artifact_digest']; "
                "Runtime resolves the corresponding Kernel Artifact"
            )
        trial = resolve_artifact_subject(
            trials.values(),
            value,
            attempt_id=attempt_id,
            require_current=side_name == "after" and not allow_historical_after,
        )
        if side_name == "after" and trial.attempt_id != attempt_id and not allow_historical_after:
            raise ValueError(
                "Experiment after Kernel Trial must belong to this logical Attempt; "
                "use action=adopt to adopt eligible historical Trial evidence"
            )
        result_artifacts = [str(value["result_artifact_digest"])]
        if not result_artifacts:
            raise ValueError(
                f"Experiment {side_name} Kernel Trial has no recorded Result Artifacts"
            )
        return cast(
            dict[str, JsonValue],
            {
                "kernel_artifact_digest": str(trial.kernel_artifact_digest),
                "result_artifact_digests": result_artifacts,
            },
        )

    def _validate_experiment_trials(
        self,
        attempt_id: AttemptId,
        experiment: Mapping[str, object],
    ) -> None:
        _lineage_id, visible_attempt_ids = self.control.visible_kernel_trial_attempt_ids(attempt_id)
        trials = {
            trial.id: trial
            for trial in self.control.list_kernel_trials(visible_attempt_ids, limit=5_000)
        }
        for side_name in ("before", "after"):
            side = experiment.get(side_name)
            if side is None:
                continue
            if not isinstance(side, Mapping):
                raise ValueError(f"Experiment {side_name} evidence is invalid")
            trial = resolve_artifact_subject(
                trials.values(),
                side,
                attempt_id=attempt_id,
                require_current=side_name == "after" and experiment.get("action") != "adopt",
            )
            if (
                side_name == "after"
                and trial.attempt_id != attempt_id
                and experiment.get("action") != "adopt"
            ):
                raise ValueError(
                    "Experiment after Kernel Trial must belong to this logical Attempt; "
                    "use action=adopt to adopt eligible historical Trial evidence"
                )
            kernel_digest = parse_artifact_digest(str(side.get("kernel_artifact_digest")))
            if trial.kernel_artifact_digest != kernel_digest:
                raise ValueError(
                    f"Experiment {side_name} Kernel Trial does not match its Kernel Artifact"
                )
            observed = {
                observation.result_artifact_digest
                for observation in trial.observations
                if observation.result_artifact_digest is not None
            }
            result_values = side.get("result_artifact_digests")
            if not isinstance(result_values, (list, tuple)):
                raise ValueError(f"Experiment {side_name} Result Artifacts must be an array")
            for result_value in result_values:
                if parse_artifact_digest(str(result_value)) not in observed:
                    raise ValueError(
                        f"Experiment {side_name} references a Kernel/Result Artifact pair "
                        "not observed "
                        "in visible history"
                    )
            if side_name == "after" and experiment.get("action") == "adopt":
                self.control.validate_adoption_trial(
                    attempt_id,
                    trial.id,
                    result_artifact_digests=tuple(str(x) for x in result_values),
                )


__all__ = ["RuntimeJournalService"]
