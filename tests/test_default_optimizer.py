"""Default deployment selects KDA without changing explicitly pinned workspaces."""

from __future__ import annotations

import json
import runpy
from pathlib import Path
from typing import Any, cast

import pytest

from atrex_runtime.config import RuntimeSettings

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = [ROOT / "runtime.example.json", *sorted((ROOT / "examples").glob("*/runtime.json"))]


def assert_kda_settings(settings: RuntimeSettings) -> None:
    expected = json.loads(
        (ROOT / "examples/kernel-design-agents/kernel-agent.example.json").read_text()
    )
    actual = settings.kernel_agent.model_dump(mode="json")
    assert actual["base_source"]["repository"] == str(ROOT / "src/kernel-design-agents")
    actual["base_source"]["repository"] = expected["base_source"]["repository"]
    # System git may resolve through Homebrew in generated production configs.
    actual["base_source"]["git_executable"] = expected["base_source"]["git_executable"]
    assert actual == expected


@pytest.mark.parametrize("config", CONFIGS, ids=lambda path: str(path.relative_to(ROOT)))
def test_shipped_configs_select_complete_kda_bundle(config: Path) -> None:
    assert_kda_settings(RuntimeSettings.from_file(config))


def test_production_config_selects_kda(tmp_path: Path) -> None:
    prepare = runpy.run_path(str(ROOT / "scripts/production/prepare.py"))
    build = cast(Any, prepare["_runtime_config"])
    config = build(
        root=ROOT,
        workspace=tmp_path,
        backend="codex",
        hardware_target="test-gpu",
        policy=json.loads((ROOT / "scripts/production/policy.json").read_text()),
        host="127.0.0.1",
        port=8765,
        wiki_url="http://127.0.0.1:8091",
        worker_user="test-worker",
        host_home=str(tmp_path / "home"),
        launcher_mode="container",
        evolver_commit="a" * 40,
        bench_commit="b" * 40,
    )
    assert_kda_settings(RuntimeSettings.model_validate(config))
    assert config["campaign"]["optimizer"]["agent_backend"] == "codex"


def test_connectivity_probe_defaults_to_kda() -> None:
    from atrex_runtime.acceptance.backend_connectivity import _parser

    assert _parser().parse_args([]).core_root == ROOT / "src/kernel-design-agents"
