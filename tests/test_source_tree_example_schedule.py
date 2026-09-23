"""Source-tree runs freeze only Runtime resource bounds, not Workflow topology."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from atrex_runtime.bootstrap import CampaignSpecV3
from atrex_runtime.composition.campaign import build_core_process_config
from atrex_runtime.config import RuntimeSettings

REPOSITORY = Path(__file__).resolve().parents[1]


def _module(relative: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "source_tree_schedule_example", REPOSITORY / relative,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str((REPOSITORY / relative).parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


@pytest.mark.parametrize("relative", [
    "data/GDN/campaign.json", "examples/source-tree/campaign.example.json",
])
def test_source_tree_configs_share_the_production_resource_envelope(relative: str) -> None:
    campaign = CampaignSpecV3.model_validate_json((REPOSITORY / relative).read_text())
    policy = json.loads((REPOSITORY / "scripts/production/policy.json").read_text())
    assert campaign.optimizer_attempt_budget == (
        policy["schedule"]["optimizer_attempt_budget"]
    ) == 6
    assert campaign.max_challengers == policy["schedule"]["max_challengers"] == 1


def test_gdn_source_tree_sessions_allow_one_hundred_million_tokens() -> None:
    settings = RuntimeSettings.model_validate_json(
        (REPOSITORY / "data/GDN/runtime.template.json").read_text(),
    )
    campaign = settings.campaign
    assert campaign is not None
    assert campaign.optimizer.max_session_tokens == 100_000_000
    optimizer = build_core_process_config(campaign)
    bootstrap = build_core_process_config(
        campaign, timeout_seconds=campaign.optimizer.bootstrap_timeout_seconds,
    )
    assert optimizer.max_session_tokens == bootstrap.max_session_tokens == 100_000_000
    assert "max_session_tokens" not in campaign.evolver.model_dump()


def test_source_tree_preparation_defaults_to_three_attempts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module("examples/source-tree/prepare.py")

    def stop_after_parser(parser: argparse.ArgumentParser) -> None:
        assert parser.get_default("attempts") == 3
        raise SystemExit(0)

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", stop_after_parser)
    with pytest.raises(SystemExit) as stopped:
        module.main()
    assert stopped.value.code == 0


@pytest.mark.parametrize(("arguments", "target"), [([], "100"), (["--target-epoch", "2"], "2")])
def test_gdn_runner_forwards_absolute_epoch_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, arguments: list[str], target: str,
) -> None:
    module = _module("scripts/gdn/run.py")
    monkeypatch.setattr(module.sys, "platform", "linux")
    monkeypatch.setattr(module.sys, "argv", [
        "run.py", "campaign", "--workspace", str(tmp_path), *arguments,
    ])
    # Isolate configuration/secrets and fake all process execution.
    (tmp_path / "runtime.json").write_text("{}")
    (tmp_path / "runtime-secrets.json").write_text(json.dumps({
        "ATREX_CAPABILITY_SIGNING_KEY": "test-signing", "ATREX_ADMIN_BEARER_TOKEN": "test-admin",
    }))
    calls: list[list[str]] = []

    def run(command: list[str], **kwargs: object) -> SimpleNamespace:
        calls.append(command)
        if command[1] == "bootstrap":
            return SimpleNamespace(stdout=json.dumps({"campaign_id": "campaign-test"}))
        assert command[1] == "run-campaign"
        assert command[-2:] == ["--target-epoch", target]
        return SimpleNamespace(stdout=json.dumps({"target_epoch_number": int(target)}))

    monkeypatch.setattr(module.subprocess, "run", run)
    module.main()
    assert [command[1] for command in calls] == ["bootstrap", "run-campaign"]
    assert json.loads((tmp_path / "epoch-result.json").read_text()) == {
        "target_epoch_number": int(target),
    }
