"""Runtime-owned Agent Backend binding configuration."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
from conftest import with_local_interpreter
from pydantic import ValidationError

from atrex_runtime.composition.campaign import build_core_process_config
from atrex_runtime.config import RuntimeSettings

REPOSITORY = Path(__file__).resolve().parents[1]
CONFIG = REPOSITORY / "runtime.example.json"
BACKENDS = ("claude", "codex", "qodercli", "pi")


def test_runtime_defaults_both_workers_to_qodercli() -> None:
    value = json.loads(CONFIG.read_text(encoding="utf-8"))
    del value["campaign"]["optimizer"]["agent_backend"]
    del value["campaign"]["evolver"]["agent_backend"]

    settings = RuntimeSettings.model_validate(value, context={"base": CONFIG.parent})

    assert settings.campaign is not None
    assert settings.campaign.optimizer.agent_backend == "qodercli"
    assert settings.campaign.evolver.agent_backend == "qodercli"


@pytest.mark.parametrize("role", ("optimizer", "evolver"))
@pytest.mark.parametrize("backend", BACKENDS)
def test_runtime_accepts_every_backend_for_each_worker(role: str, backend: str) -> None:
    value = json.loads(CONFIG.read_text(encoding="utf-8"))
    value["campaign"][role]["agent_backend"] = backend

    settings = RuntimeSettings.model_validate(value, context={"base": CONFIG.parent})

    assert settings.campaign is not None
    assert getattr(settings.campaign, role).agent_backend == backend


@pytest.mark.parametrize("role", ("optimizer", "evolver"))
def test_runtime_rejects_unknown_backend(role: str) -> None:
    value = json.loads(CONFIG.read_text(encoding="utf-8"))
    value["campaign"][role]["agent_backend"] = "unknown"

    with pytest.raises(ValidationError):
        RuntimeSettings.model_validate(value)


def test_core_process_contract_contains_runtime_binding() -> None:
    settings = RuntimeSettings.from_file(CONFIG)
    assert settings.campaign is not None

    campaign = with_local_interpreter(settings.campaign)
    process = build_core_process_config(campaign)

    assert process.agent_backend == "qodercli"
    assert process.reasoning_effort == "max"
    assert process.session_settings == ""
    assert process.report_completion_retries == 2
    assert process.output_limit_recovery_retries == 2
    assert process.timeout_seconds == 28_800

    bootstrap = build_core_process_config(
        campaign,
        timeout_seconds=campaign.optimizer.bootstrap_timeout_seconds,
    )
    assert bootstrap.timeout_seconds == 14_400


@pytest.mark.parametrize("retries", (0, 3, 10))
def test_report_completion_configuration_reaches_process_policy(retries: int) -> None:
    value = json.loads(CONFIG.read_text(encoding="utf-8"))
    value["campaign"]["optimizer"]["report_completion_retries"] = retries
    settings = RuntimeSettings.model_validate(value, context={"base": CONFIG.parent})
    assert settings.campaign is not None
    process = build_core_process_config(with_local_interpreter(settings.campaign))
    assert process.report_completion_retries == retries


@pytest.mark.parametrize("retries", (-1, 11, True, "2", 2.0))
def test_runtime_rejects_invalid_report_completion_configuration(retries: object) -> None:
    value = json.loads(CONFIG.read_text(encoding="utf-8"))
    value["campaign"]["optimizer"]["report_completion_retries"] = retries
    with pytest.raises(ValidationError):
        RuntimeSettings.model_validate(value, context={"base": CONFIG.parent})


@pytest.mark.parametrize("retries", (0, 2, 10))
def test_output_limit_recovery_configuration_reaches_process_policy(retries: int) -> None:
    value = json.loads(CONFIG.read_text(encoding="utf-8"))
    value["campaign"]["optimizer"]["output_limit_recovery_retries"] = retries
    value["campaign"]["optimizer"]["report_completion_retries"] = 1
    settings = RuntimeSettings.model_validate(value, context={"base": CONFIG.parent})
    assert settings.campaign is not None
    process = build_core_process_config(with_local_interpreter(settings.campaign))
    assert process.output_limit_recovery_retries == retries
    assert process.report_completion_retries == 1


@pytest.mark.parametrize("retries", (-1, 11, True, "2", 2.0, None))
def test_runtime_rejects_invalid_output_limit_recovery_configuration(retries: object) -> None:
    value = json.loads(CONFIG.read_text(encoding="utf-8"))
    value["campaign"]["optimizer"]["output_limit_recovery_retries"] = retries
    with pytest.raises(ValidationError):
        RuntimeSettings.model_validate(value, context={"base": CONFIG.parent})


@pytest.mark.parametrize("retries", (-1, 11, True, "2", 2.0, None))
def test_process_policy_rejects_invalid_output_limit_recovery_configuration(
    retries: object,
) -> None:
    settings = RuntimeSettings.from_file(CONFIG)
    assert settings.campaign is not None
    process = build_core_process_config(with_local_interpreter(settings.campaign))
    with pytest.raises(ValueError, match="output limit recovery retries"):
        replace(process, output_limit_recovery_retries=retries)


def test_missing_output_limit_recovery_configuration_defaults_to_two() -> None:
    value = json.loads(CONFIG.read_text(encoding="utf-8"))
    value["campaign"]["optimizer"].pop("output_limit_recovery_retries", None)
    settings = RuntimeSettings.model_validate(value, context={"base": CONFIG.parent})
    assert settings.campaign is not None
    assert settings.campaign.optimizer.output_limit_recovery_retries == 2
