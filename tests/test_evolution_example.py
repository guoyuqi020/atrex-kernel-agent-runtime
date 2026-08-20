"""Tests for the executable three-Epoch evolution example."""

from __future__ import annotations

import os
import stat
import subprocess
import sys
from pathlib import Path

from atrex_runtime.bootstrap import CampaignSpecV3

REPOSITORY = Path(__file__).resolve().parents[1]
EXAMPLE = REPOSITORY / "examples/evolution"


def test_all_evolution_shell_wrappers_parse_and_entrypoints_are_executable() -> None:
    scripts = sorted(EXAMPLE.glob("*.sh"))
    assert {path.name for path in scripts} == {
        "common.sh",
        "inspect.sh",
        "prepare.sh",
        "run-campaign.sh",
        "run.sh",
        "start-runtime.sh",
    }
    subprocess.run(("bash", "-n", *(str(path) for path in scripts)), check=True)
    for path in scripts:
        if path.name != "common.sh":
            assert stat.S_IMODE(path.stat().st_mode) & 0o111


def test_evolution_prepare_delays_challengers_and_uses_one_attempt_per_branch(
    tmp_path: Path,
) -> None:
    environment = {
        **os.environ,
        "AGATE_URL": "https://agate.example.test",
        "AGATE_AK": "not-persisted-ak",
        "AGATE_SK": "not-persisted-sk",
        "AGATE_GPU": "TEST_GPU",
        "ATREX_PYTHON": sys.executable,
        "ATREX_EVOLUTION_STATE_DIR": str(tmp_path),
    }
    for name in (
        "ATREX_CHALLENGER_COUNT",
        "ATREX_CHALLENGER_START_EPOCH",
        "ATREX_TRAJECTORIES_PER_BRANCH",
        "ATREX_ATTEMPTS_PER_TRAJECTORY",
    ):
        environment.pop(name, None)
    subprocess.run(
        (
            "bash",
            "-c",
            "source examples/evolution/common.sh; atrex_example_prepare_inputs",
        ),
        cwd=REPOSITORY,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    spec = CampaignSpecV3.from_file(tmp_path / "campaign.json")
    assert spec.challenger_count == 1
    assert spec.challenger_start_epoch == 2
    assert spec.trajectories_per_branch == 1
    assert spec.attempts_per_trajectory == 1


def test_evolution_wrapper_targets_epoch_three_and_inspects_history() -> None:
    run_campaign = (EXAMPLE / "run-campaign.sh").read_text(encoding="utf-8")
    managed = (EXAMPLE / "run.sh").read_text(encoding="utf-8")
    inspect = (EXAMPLE / "inspect.sh").read_text(encoding="utf-8")
    assert "--target-epoch 3" in run_campaign
    assert '"${script_dir}/run-campaign.sh" --inputs-already-prepared' in managed
    assert '"${script_dir}/inspect.sh"' in managed
    assert "atrex_example_prepare_gpu_wiki" in managed
    assert "list-epochs" in inspect
    assert "list-attempts" in inspect
    assert "list-kernels" in inspect
    assert "list-agent-revisions" in inspect
