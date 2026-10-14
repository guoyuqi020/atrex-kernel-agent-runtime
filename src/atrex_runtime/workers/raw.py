"""Direct Agate launch data and private controller handoff for the Raw ablation."""

from collections.abc import Mapping

from ..config import AgateSettings
from ..domain.ids import AttemptId
from .attempt_report import AttemptReportV12


def raw_agate_environment(
    settings: AgateSettings, environment: Mapping[str, str]
) -> tuple[tuple[str, str], ...]:
    """Resolve the configured destination/auth, injected only into Raw Sessions."""
    from ..gateway.configuration import build_agate_connection

    connection = build_agate_connection(settings, environment)
    values = {"AGATE_URL": connection.base_url, "AGATE_NO_UPDATE_CHECK": "1"}
    if connection.auth_mode == "token":
        values["AGATE_TOKEN"] = connection.token.get_secret_value() if connection.token else ""
    elif connection.auth_mode == "ak_sk":
        values.update(
            AGATE_AK=connection.access_key.get_secret_value() if connection.access_key else "",
            AGATE_SK=connection.secret_key.get_secret_value() if connection.secret_key else "",
        )
    return tuple(values.items())


def raw_handoff(attempt_id: AttemptId) -> AttemptReportV12:
    """Internal nomination only; neither an Agent claim nor reusable experience."""
    return AttemptReportV12.model_validate(
        {
            "schema_version": 13,
            "attempt_id": attempt_id,
            "status": "candidate_ready",
            "final_candidate": {"change_summary": "Raw Session final working source"},
            "contributing_result_artifact_digests": [],
            "blocker": None,
            "tool_modules": [],
            "experiments": [],
            "direction_events": [],
        }
    )
