from __future__ import annotations

import json
import stat
from pathlib import Path

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
    assert runtime_contract_environment(path) == {
        RUNTIME_CONTRACT_ENVIRONMENT_KEY: str(path)
    }
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
