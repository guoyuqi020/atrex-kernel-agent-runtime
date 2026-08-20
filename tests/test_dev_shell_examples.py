"""Executable examples for interactive Optimizer and Evolver workspaces."""

from __future__ import annotations

import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from atrex_runtime.bootstrap import CampaignSpecV3
from atrex_runtime.config import RuntimeSettings

REPOSITORY = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("name", "expected_scripts"),
    (
        (
            "optimizer-dev-shell",
            {"common.sh", "open-shell.sh", "run.sh"},
        ),
        (
            "evolver-dev-shell",
            {"common.sh", "open-shell.sh", "run.sh"},
        ),
    ),
)
def test_dev_shell_wrappers_parse_and_entrypoints_are_executable(
    name: str,
    expected_scripts: set[str],
) -> None:
    scripts = sorted((REPOSITORY / "examples" / name).glob("*.sh"))
    assert {path.name for path in scripts} == expected_scripts
    subprocess.run(("bash", "-n", *(str(path) for path in scripts)), check=True)
    for path in scripts:
        if path.name != "common.sh":
            assert stat.S_IMODE(path.stat().st_mode) & 0o111


@pytest.mark.parametrize(
    ("name", "state_environment"),
    (
        ("optimizer-dev-shell", "ATREX_OPTIMIZER_DEV_SHELL_STATE_DIR"),
        ("evolver-dev-shell", "ATREX_EVOLVER_DEV_SHELL_STATE_DIR"),
    ),
)
def test_dev_shell_prepare_materializes_an_independent_active_only_campaign(
    name: str,
    state_environment: str,
    tmp_path: Path,
) -> None:
    environment = {
        **os.environ,
        "AGATE_URL": "https://agate.example.test",
        "AGATE_AK": "not-persisted-ak",
        "AGATE_SK": "not-persisted-sk",
        "AGATE_GPU": "TEST_GPU",
        "ATREX_PYTHON": sys.executable,
        state_environment: str(tmp_path),
    }
    for key in (
        "ATREX_CAMPAIGN",
        "ATREX_CHALLENGER_COUNT",
        "ATREX_CHALLENGER_START_EPOCH",
        "ATREX_TRAJECTORIES_PER_BRANCH",
        "ATREX_ATTEMPTS_PER_TRAJECTORY",
        "ATREX_OPTIMIZER_AGENT_BACKEND",
        "ATREX_EVOLVER_AGENT_BACKEND",
    ):
        environment.pop(key, None)
    subprocess.run(
        (
            "bash",
            "-c",
            f"source examples/{name}/common.sh; atrex_example_prepare_inputs",
        ),
        cwd=REPOSITORY,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    settings = RuntimeSettings.from_file(tmp_path / "runtime.json")
    spec = CampaignSpecV3.from_file(tmp_path / "campaign.json")
    serialized = (tmp_path / "runtime.json").read_text(encoding="utf-8")
    assert spec.challenger_count == 0
    assert spec.trajectories_per_branch == 1
    assert spec.attempts_per_trajectory == 1
    assert settings.campaign is not None
    assert settings.campaign.launcher.mode == "sandbox"
    assert settings.server.host == "127.0.0.1"
    assert settings.campaign.launcher.sandbox is not None
    assert settings.campaign.launcher.sandbox.resolv_conf.is_absolute()
    assert "not-persisted-ak" not in serialized
    assert "not-persisted-sk" not in serialized


def test_campaign_preparation_accepts_independent_optimizer_and_evolver_backends(
    tmp_path: Path,
) -> None:
    environment = {
        **os.environ,
        "AGATE_URL": "https://agate.example.test",
        "AGATE_AK": "not-persisted-ak",
        "AGATE_SK": "not-persisted-sk",
        "AGATE_GPU": "TEST_GPU",
        "ATREX_PYTHON": sys.executable,
        "ATREX_EVOLVER_DEV_SHELL_STATE_DIR": str(tmp_path),
        "ATREX_OPTIMIZER_AGENT_BACKEND": "claude",
        "ATREX_EVOLVER_AGENT_BACKEND": "codex",
    }
    subprocess.run(
        (
            "bash",
            "-c",
            "source examples/evolver-dev-shell/common.sh; atrex_example_prepare_inputs",
        ),
        cwd=REPOSITORY,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    settings = RuntimeSettings.from_file(tmp_path / "runtime.json")
    assert settings.campaign is not None
    assert settings.campaign.optimizer.agent_backend == "claude"
    assert settings.campaign.evolver.agent_backend == "codex"


def test_optimizer_example_opens_a_disposable_shell_without_bootstrap_or_agent() -> None:
    source = (REPOSITORY / "examples/optimizer-dev-shell/open-shell.sh").read_text(encoding="utf-8")
    runner = (REPOSITORY / "examples/optimizer-dev-shell/run.sh").read_text(encoding="utf-8")
    assert "atrex_example_bootstrap_campaign" not in source
    assert "atrex_example_require_agent_backend" not in source
    assert '"${atrex_runtime_cli}" temporary-dev-shell' in source
    assert "mktemp -d" in runner
    assert "rm -rf" in runner
    assert "sudo --preserve-env=" in runner
    assert "evolver-dev-shell" not in source


def test_evolver_example_opens_a_disposable_shell_without_bootstrap_or_agent() -> None:
    open_shell = (REPOSITORY / "examples/evolver-dev-shell/open-shell.sh").read_text(
        encoding="utf-8"
    )
    managed = (REPOSITORY / "examples/evolver-dev-shell/run.sh").read_text(encoding="utf-8")
    assert "atrex_example_bootstrap_campaign" not in open_shell
    assert "atrex_example_with_runtime" not in managed
    assert '"${atrex_runtime_cli}" temporary-evolver-dev-shell' in open_shell
    assert "mktemp -d" in managed
    assert "rm -rf" in managed
    assert "sudo --preserve-env=" in managed
    assert '"${script_dir}/open-shell.sh" --inputs-already-prepared' in managed
