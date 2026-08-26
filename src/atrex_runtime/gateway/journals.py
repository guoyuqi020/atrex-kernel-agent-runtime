"""Authoritative Runtime-owned Direction and Experiment Journals."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import cast
from uuid import uuid4

from ..artifacts.local import ArtifactKind, JsonValue, LocalArtifactStore
from ..domain.errors import DirectionConcurrencyError, InfrastructureError
from ..domain.ids import AttemptId, parse_artifact_digest
from ..workers.attempt_report import AttemptDirectionEventV1, AttemptExperimentV8
from .control import SqliteGatewayControl
from .control_models import GatewayAuthorization
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
                        }
                        for direction in directions.values()
                    ]
                },
            )
        if isinstance(request, DirectionLoadRequestV2):
            try:
                return cast(
                    dict[str, JsonValue],
                    self._direction_views(request.attempt_id)[request.direction_id],
                )
            except KeyError as error:
                raise ValueError(
                    "Direction ID is outside the current Attempt's visible history"
                ) from error
        if isinstance(request, ExperimentRecordRequestV2):
            return self._record_experiment(request, authorization)
        if request.operation == "experiments_list":
            return cast(
                dict[str, JsonValue],
                {
                    "experiments": [
                        {
                            "experiment_id": experiment["experiment_id"],
                            "sequence": experiment["sequence"],
                            "name": experiment["name"],
                            "action": experiment["action"],
                        }
                        for experiment in self._visible_experiments(request.attempt_id)
                    ]
                },
            )
        if isinstance(request, ExperimentLoadRequestV2):
            for experiment in self._visible_experiments(request.attempt_id):
                if experiment["experiment_id"] == request.experiment_id:
                    return cast(dict[str, JsonValue], experiment)
            raise ValueError("Experiment ID is outside the current Attempt's visible history")
        if request.operation == "journal_snapshot":
            return cast(
                dict[str, JsonValue],
                {
                    "direction_events": list(self._current_direction_events(request.attempt_id)),
                    "experiments": list(self._current_experiments(request.attempt_id)),
                    "directions": list(self._direction_views(request.attempt_id).values()),
                },
            )
        raise ValueError(f"unsupported Runtime Journal operation: {request.operation}")

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
        _lineage_id, attempt_ids = self.control.visible_kernel_trial_attempt_ids(attempt_id)
        return attempt_ids

    def _current_direction_events(self, attempt_id: AttemptId) -> tuple[dict[str, object], ...]:
        return self.control.list_direction_events(attempt_id)

    def _current_experiments(self, attempt_id: AttemptId) -> tuple[dict[str, object], ...]:
        return self.control.list_experiments(attempt_id)

    def _visible_direction_events(self, attempt_id: AttemptId) -> list[dict[str, object]]:
        reports = self._report_values(attempt_id, "direction_events")
        values: list[dict[str, object]] = []
        for visible_attempt_id in self._visible_attempt_ids(attempt_id):
            live = self.control.list_direction_events(visible_attempt_id)
            source = list(live) if live else reports.get(visible_attempt_id, [])
            values.extend(
                AttemptDirectionEventV1.model_validate(item).model_dump(mode="json")
                for item in source
            )
        if len(values) > 4_096:
            raise ValueError("Visible Direction history exceeds its entry limit")
        event_ids = [str(event["direction_event_id"]) for event in values]
        if len(set(event_ids)) != len(event_ids):
            raise ValueError("Visible Direction history contains duplicate Event IDs")
        return values

    def _visible_experiments(self, attempt_id: AttemptId) -> list[dict[str, object]]:
        reports = self._report_values(attempt_id, "experiments")
        values: list[dict[str, object]] = []
        for visible_attempt_id in self._visible_attempt_ids(attempt_id):
            live = self.control.list_experiments(visible_attempt_id)
            source = list(live) if live else reports.get(visible_attempt_id, [])
            values.extend(
                AttemptExperimentV8.model_validate(item).model_dump(mode="json") for item in source
            )
        if len(values) > 4_096:
            raise ValueError("Visible Experiment history exceeds its entry limit")
        experiment_ids = [str(experiment["experiment_id"]) for experiment in values]
        if len(set(experiment_ids)) != len(experiment_ids):
            raise ValueError("Visible Experiment history contains duplicate Experiment IDs")
        return values

    def _direction_views(self, attempt_id: AttemptId) -> dict[str, dict[str, object]]:
        directions: dict[str, dict[str, object]] = {}
        for event in self._visible_direction_events(attempt_id):
            direction_id = str(event["direction_id"])
            action = str(event["action"])
            existing = directions.get(direction_id)
            if action == "propose":
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
                    "status": _DIRECTION_STATUSES[action],
                    "analysis": None,
                    "supporting_experiment_ids": [],
                }
                continue
            if existing is None:
                raise ValueError("Direction update precedes its proposal")
            existing["status"] = _DIRECTION_STATUSES[action]
            existing["analysis"] = event["analysis"]
            supporting = cast(list[str], existing["supporting_experiment_ids"])
            for experiment_id in cast(list[str], event["supporting_experiment_ids"]):
                if experiment_id not in supporting:
                    supporting.append(experiment_id)
        for experiment in self._visible_experiments(attempt_id):
            direction = directions.get(str(experiment["direction_id"]))
            if direction is None:
                continue
            supporting = cast(list[str], direction["supporting_experiment_ids"])
            experiment_id = str(experiment["experiment_id"])
            if experiment_id not in supporting:
                supporting.append(experiment_id)
        return directions

    def _update_direction(
        self,
        request: DirectionUpdateRequestV2,
        authorization: GatewayAuthorization,
    ) -> dict[str, JsonValue]:
        value = dict(request.request)
        action = value.get("action")
        if action == "propose":
            if set(value) != _DIRECTION_PROPOSAL_FIELDS:
                raise ValueError(
                    "Direction proposal fields must be exactly "
                    f"{sorted(_DIRECTION_PROPOSAL_FIELDS)}"
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
            event: dict[str, object] = {
                "direction_event_id": f"directionevent_{uuid4().hex}",
                "direction_id": direction_id,
                "recorded_at": datetime.now(UTC).isoformat(),
                **value,
                "analysis": None,
                "supporting_experiment_ids": [],
            }
        else:
            if set(value) != _DIRECTION_UPDATE_FIELDS:
                raise ValueError(
                    f"Direction update fields must be exactly {sorted(_DIRECTION_UPDATE_FIELDS)}"
                )
            if action not in {"start", "complete", "abandon", "block", "defer"}:
                raise ValueError("Direction update action is invalid")
            direction_id = _text(value.get("direction_id"), "Direction ID")
            direction = self._direction_views(request.attempt_id).get(direction_id)
            if direction is None:
                raise ValueError("Direction ID is outside the current Attempt's visible history")
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
            supporting = list(cast(list[str], direction["supporting_experiment_ids"]))
            if action in {"complete", "abandon"} and not supporting:
                raise ValueError(f"Direction {action} requires at least one associated Experiment")
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
        direction = self._direction_views(request.attempt_id).get(direction_id)
        if direction is None:
            raise ValueError("Experiment Direction is outside visible history")
        if direction["status"] != "in_progress":
            raise ValueError(
                f"Experiment Direction must be in progress; current status is {direction['status']}"
            )
        allow_baseline = self._is_bootstrap(request.attempt_id)
        actions = {"keep_after", "restore_before", "abandon_direction"}
        if allow_baseline:
            actions.add("baseline")
        if value.get("action") not in actions:
            raise ValueError("Experiment action is invalid")
        current = list(self._current_experiments(request.attempt_id))
        if value.get("action") == "baseline" and any(
            experiment.get("action") == "baseline" for experiment in current
        ):
            raise ValueError("Bootstrap Experiment journal may contain only one baseline action")
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
            trial_id = side.get("kernel_trial_id")
            trial = trials.get(str(trial_id))
            if trial is None:
                raise ValueError(f"Experiment {side_name} Kernel Trial is outside visible history")
            if side_name == "after" and trial.attempt_id != attempt_id:
                raise ValueError(
                    "Experiment after Kernel Trial must belong to this logical Attempt; "
                    "historical Trials may only be used as before evidence"
                )
            kernel_digest = parse_artifact_digest(str(side.get("kernel_artifact_digest")))
            if trial.kernel_artifact_digest != kernel_digest:
                raise ValueError(
                    f"Experiment {side_name} Kernel Trial does not match its Kernel Artifact"
                )
            observed = {
                observation.gateway_result_digest
                for observation in trial.observations
                if observation.gateway_result_digest is not None
            }
            result_values = side.get("gateway_result_digests")
            if not isinstance(result_values, (list, tuple)):
                raise ValueError(f"Experiment {side_name} Gateway results must be an array")
            for result_value in result_values:
                if parse_artifact_digest(str(result_value)) not in observed:
                    raise ValueError(
                        f"Experiment {side_name} references a Kernel/Result pair not observed "
                        "in visible history"
                    )


__all__ = ["RuntimeJournalService"]
