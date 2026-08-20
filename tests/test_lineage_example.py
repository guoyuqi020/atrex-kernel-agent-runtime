"""Tests for the executable single-Epoch Lineage example."""

from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
from pathlib import Path

from atrex_runtime.bootstrap import CampaignSpecV3
from atrex_runtime.config import RuntimeSettings

REPOSITORY = Path(__file__).resolve().parents[1]
EXAMPLE = REPOSITORY / "examples/lineage"


def test_all_lineage_shell_wrappers_parse_and_entrypoints_are_executable() -> None:
    scripts = sorted(EXAMPLE.glob("*.sh"))
    assert {path.name for path in scripts} == {
        "common.sh",
        "inspect.sh",
        "prepare.sh",
        "run-epoch.sh",
        "run.sh",
        "start-runtime.sh",
    }
    subprocess.run(("bash", "-n", *(str(path) for path in scripts)), check=True)
    for path in scripts:
        if path.name != "common.sh":
            assert stat.S_IMODE(path.stat().st_mode) & 0o111


def test_lineage_prepare_uses_one_epoch_with_three_serial_attempts(tmp_path: Path) -> None:
    environment = {
        **os.environ,
        "AGATE_URL": "https://agate.example.test",
        "AGATE_AK": "not-persisted-ak",
        "AGATE_SK": "not-persisted-sk",
        "AGATE_GPU": "TEST_GPU",
        "ATREX_PYTHON": sys.executable,
        "ATREX_LINEAGE_STATE_DIR": str(tmp_path),
    }
    environment.pop("ATREX_CHALLENGER_COUNT", None)
    environment.pop("ATREX_CHALLENGER_START_EPOCH", None)
    environment.pop("ATREX_TRAJECTORIES_PER_BRANCH", None)
    environment.pop("ATREX_ATTEMPTS_PER_TRAJECTORY", None)
    subprocess.run(
        (
            "bash",
            "-c",
            "source examples/lineage/common.sh; atrex_example_prepare_inputs",
        ),
        cwd=REPOSITORY,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    config_path = tmp_path / "runtime.json"
    spec_path = tmp_path / "campaign.json"
    settings = RuntimeSettings.from_file(config_path)
    spec = CampaignSpecV3.from_file(spec_path)
    serialized = config_path.read_text(encoding="utf-8")
    assert spec.challenger_count == 0
    assert spec.challenger_start_epoch == 1
    assert spec.trajectories_per_branch == 1
    assert spec.attempts_per_trajectory == 3
    assert settings.campaign is not None
    assert settings.campaign.evolver.commit == ("853fbdc969c8102938bb4c3a0ebe492ba26a1a77")
    assert "not-persisted-ak" not in serialized
    assert "not-persisted-sk" not in serialized


def test_lineage_common_resolves_saved_single_lineage(tmp_path: Path) -> None:
    expected = "lineage_0123456789abcdef0123456789abcdef"
    (tmp_path / "bootstrap-result.json").write_text(
        json.dumps({"lineages": [{"lineage_id": expected}]}),
        encoding="utf-8",
    )
    result = subprocess.run(
        (
            "bash",
            "-c",
            "source examples/lineage/common.sh; atrex_example_lineage_id",
        ),
        cwd=REPOSITORY,
        env={**os.environ, "ATREX_LINEAGE_STATE_DIR": str(tmp_path)},
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip() == expected


def test_managed_lineage_wrapper_delegates_run_and_offline_inspection() -> None:
    source = (EXAMPLE / "run.sh").read_text(encoding="utf-8")
    assert '"${script_dir}/run-epoch.sh" --inputs-already-prepared' in source
    assert '"${script_dir}/inspect.sh"' in source
    assert "atrex_example_prepare_gpu_wiki" in source
    assert "list-epochs" in (EXAMPLE / "inspect.sh").read_text(encoding="utf-8")
