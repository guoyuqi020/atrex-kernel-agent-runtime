from __future__ import annotations

import json
import stat
from pathlib import Path

import pytest

from atrex_runtime.composition.campaign import build_optimizer_session_contract_policy
from atrex_runtime.config import RuntimeSettings
from atrex_runtime.workers.session_contract import (
    RUNTIME_CONTRACT_ENVIRONMENT_KEY,
    materialize_session_contract,
    runtime_contract_environment,
)


def test_session_contract_materializes_live_schema_environment_and_limits(
    tmp_path: Path,
) -> None:
    path = materialize_session_contract(
        tmp_path,
        phase="optimization_attempt",
        dsl="triton",
        hardware_target="sm_120",
        agent_backend="claude",
        model="test-model",
        session_timeout_seconds=3600,
        usage_unit="provider_tokens",
        usage_budget=20_000_000,
        max_attempt_report_bytes=1024 * 1024,
        wiki_available=False,
    )

    assert path == tmp_path / "input/runtime-contract"
    assert runtime_contract_environment(path) == {RUNTIME_CONTRACT_ENVIRONMENT_KEY: str(path)}
    tools = json.loads((path / "tools.json").read_text())
    environment = json.loads((path / "environment.json").read_text())
    limits = json.loads((path / "limits.json").read_text())
    assert set(tools["gateway"]["operations"]) == {
        "check",
        "dev",
        "disassemble",
        "env",
        "evaluate",
        "profile",
    }
    evaluate = tools["gateway"]["operations"]["evaluate"]
    assert evaluate["additionalProperties"] is False
    assert "candidate" not in evaluate["properties"]
    assert environment["phase"] == "optimization_attempt"
    assert environment["paths"]["writable"] == ["scratch/", "tools/", "work/"]
    assert limits["provider_usage"] == {
        "budget": 20_000_000,
        "unit": "provider_tokens",
    }
    assert stat.S_IMODE(path.stat().st_mode) & stat.S_IWUSR == 0
    assert stat.S_IMODE((path / "tools.json").stat().st_mode) & stat.S_IWUSR == 0


def test_session_contract_can_describe_the_next_optimizer_without_becoming_live(
    tmp_path: Path,
) -> None:
    path = materialize_session_contract(
        tmp_path,
        phase="optimization_attempt",
        dsl="cuda",
        hardware_target="sm_120",
        agent_backend="codex",
        model="next-model",
        session_timeout_seconds=7200,
        usage_unit="provider_tokens",
        usage_budget=1_000_000,
        max_attempt_report_bytes=64_000,
        wiki_available=False,
        relative_path=Path("input/next-session-contract"),
    )

    assert path == tmp_path / "input/next-session-contract"
    environment = json.loads((path / "environment.json").read_text())
    assert environment["hardware_target"] == "sm_120"
    assert environment["agent_backend"] == "codex"
    assert not (tmp_path / "input/runtime-contract").exists()


@pytest.mark.parametrize(
    "modules", [(), ("directions",), ("experiments",), ("directions", "experiments")]
)
def test_session_contract_exposes_only_enabled_tool_modules(
    tmp_path: Path, modules: tuple[str, ...]
) -> None:
    path = materialize_session_contract(
        tmp_path,
        phase="optimization_attempt",
        dsl="triton",
        hardware_target="sm_120",
        agent_backend="claude",
        model=None,
        session_timeout_seconds=3600,
        usage_unit="provider_tokens",
        usage_budget=1000,
        max_attempt_report_bytes=100_000,
        wiki_available=False,
        tool_modules=modules,
    )
    tools = json.loads((path / "tools.json").read_text())["bindings"]
    environment = json.loads((path / "environment.json").read_text())
    assert environment["tool_modules"] == list(modules)
    assert ("update-direction" in tools) == ("directions" in modules)
    assert ("list-directions" in tools) == ("directions" in modules)
    assert ("find-kernel-directions" in tools) == ("directions" in modules)
    assert ("record-experiment" in tools) == ("experiments" in modules)
    assert ("list-experiments" in tools) == ("experiments" in modules)
    assert ("find-kernel-experiments" in tools) == ("experiments" in modules)
    assert tools["kernel-pareto-frontier"]["operation"] == "kernel_pareto_frontier"
    assert "attempt-report" in tools


@pytest.mark.parametrize("wiki_available", (False, True))
@pytest.mark.parametrize("phase", ("framework_baseline", "optimization_attempt"))
def test_session_contract_exposes_wiki_independently_of_journal_modules(
    tmp_path: Path,
    wiki_available: bool,
    phase: str,
) -> None:
    path = materialize_session_contract(
        tmp_path,
        phase=phase,
        dsl="cuda",
        hardware_target="ZW-M890P",
        agent_backend="claude",
        model=None,
        session_timeout_seconds=3600,
        usage_unit="provider_tokens",
        usage_budget=1000,
        max_attempt_report_bytes=100_000,
        wiki_available=wiki_available,
        tool_modules=(),
    )
    tools = json.loads((path / "tools.json").read_text())
    environment = json.loads((path / "environment.json").read_text())
    assert environment["services"]["wiki"] is wiki_available
    assert ("wiki-query" in tools["bindings"]) is wiki_available
    if wiki_available:
        assert tools["bindings"]["wiki-query"] == {
            "kind": "runtime-query",
            "operation": "wiki_query",
        }
    # Wiki is a Runtime service, not an Agate GPU operation or Journal module.
    assert "wiki_query" not in tools["gateway"]["operations"]
    assert "update-direction" not in tools["bindings"]
    assert "record-experiment" not in tools["bindings"]


@pytest.mark.parametrize("wiki_available", (False, True))
def test_next_optimizer_contract_policy_preserves_wiki_availability(
    wiki_available: bool,
) -> None:
    settings = RuntimeSettings.from_file(
        Path(__file__).resolve().parents[1] / "runtime.example.json"
    )
    assert settings.campaign is not None
    policy = build_optimizer_session_contract_policy(
        settings.campaign,
        wiki_available=wiki_available,
    )
    assert policy.wiki_available is wiki_available
    assert policy.tool_modules == settings.campaign.optimizer.tool_modules
