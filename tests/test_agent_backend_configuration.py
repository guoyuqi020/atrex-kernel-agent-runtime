"""Runtime-owned Agent Backend binding configuration."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
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

    process = build_core_process_config(settings.campaign)

    assert process.agent_backend == "qodercli"
    assert process.reasoning_effort == "max"
    assert process.session_settings == ""
    assert process.timeout_seconds == 3600

    bootstrap = build_core_process_config(
        settings.campaign,
        timeout_seconds=settings.campaign.optimizer.bootstrap_timeout_seconds,
    )
    assert bootstrap.timeout_seconds == 10_800
