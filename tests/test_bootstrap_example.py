"""Tests for the executable real-Agate Bootstrap example."""

from __future__ import annotations

import json
import os
import stat
import subprocess
from pathlib import Path

from atrex_runtime.bootstrap import CampaignSpecV3
from atrex_runtime.config import RuntimeSettings
from atrex_runtime.domain.models import Dsl

REPOSITORY = Path(__file__).resolve().parents[1]
EXAMPLE = REPOSITORY / "examples/bootstrap"
SHARED = REPOSITORY / "examples/shared"


def _prepare_command(
    state: Path,
    config: Path,
    campaign: Path,
) -> tuple[str, ...]:
    return (
        str(REPOSITORY / ".venv/bin/python"),
        str(SHARED / "prepare_campaign.py"),
        "--state-dir",
        str(state),
        "--runtime-template",
        str(EXAMPLE / "runtime.json"),
        "--campaign-template",
        str(EXAMPLE / "campaign.json"),
        "--config",
        str(config),
        "--campaign",
        str(campaign),
    )


def test_all_bootstrap_shell_wrappers_parse() -> None:
    scripts = sorted(EXAMPLE.glob("*.sh"))
    assert scripts
    subprocess.run(("bash", "-n", *(str(path) for path in scripts)), check=True)


def test_bootstrap_prepare_builds_valid_remote_agate_inputs(tmp_path: Path) -> None:
    config_path = tmp_path / "runtime.json"
    spec_path = tmp_path / "campaign.json"
    environment = {
        **os.environ,
        "AGATE_URL": "https://agate.example.test",
        "AGATE_AK": "not-persisted-ak",
        "AGATE_SK": "not-persisted-sk",
        "AGATE_GPU": "TEST_GPU",
        "ATREX_BOOTSTRAP_STATE_DIR": str(tmp_path),
    }
    environment.pop("ATREX_WIKI_URL", None)
    subprocess.run(
        _prepare_command(tmp_path, config_path, spec_path),
        cwd=REPOSITORY,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    settings = RuntimeSettings.from_file(config_path)
    spec = CampaignSpecV3.from_file(spec_path)
    contract = json.loads(spec.evaluation_contract.read_text(encoding="utf-8"))
    serialized = config_path.read_text(encoding="utf-8")
    assert settings.agate.base_url == "https://agate.example.test"
    assert settings.agate.auth_mode == "ak_sk"
    assert settings.agate.access_key_env == "AGATE_AK"
    assert settings.agate.secret_key_env == "AGATE_SK"
    assert settings.gpu_wiki is None
    assert settings.campaign is not None
    assert settings.campaign.optimizer.max_session_tokens == 20_000_000
    assert settings.campaign.optimizer.bootstrap_timeout_seconds == 10_800
    assert spec.hardware_target == "TEST_GPU"
    assert tuple(spec.lineages) == (Dsl.TRITON,)
    assert spec.challenger_start_epoch == 1
    assert spec.creation_key.endswith(spec.base_revision.commit[:12])
    assert settings.campaign.gate_policy.atol == 0.01
    assert settings.campaign.gate_policy.rtol == 0.05
    assert contract["options"]["num_correctness_cases"] == 1
    assert "not-persisted-ak" not in serialized
    assert "not-persisted-sk" not in serialized


def test_bootstrap_prepare_accepts_optimizer_token_quota_override(tmp_path: Path) -> None:
    config_path = tmp_path / "runtime.json"
    spec_path = tmp_path / "campaign.json"
    environment = {
        **os.environ,
        "AGATE_URL": "https://agate.example.test",
        "AGATE_AK": "unused-ak",
        "AGATE_SK": "unused-sk",
        "AGATE_GPU": "TEST_GPU",
        "ATREX_OPTIMIZER_MAX_SESSION_TOKENS": "1234567",
    }
    subprocess.run(
        _prepare_command(tmp_path, config_path, spec_path),
        cwd=REPOSITORY,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    settings = RuntimeSettings.from_file(config_path)
    assert settings.campaign is not None
    assert settings.campaign.optimizer.max_session_tokens == 1_234_567


def test_bootstrap_prepare_preserves_existing_workspace_identity(tmp_path: Path) -> None:
    config_path = tmp_path / "runtime.json"
    spec_path = tmp_path / "campaign.json"
    command = _prepare_command(tmp_path, config_path, spec_path)
    environment = {
        **os.environ,
        "AGATE_URL": "https://agate.example.test",
        "AGATE_AK": "unused-ak",
        "AGATE_SK": "unused-sk",
        "AGATE_GPU": "FIRST_GPU",
        "ATREX_BOOTSTRAP_CREATION_KEY": "stable-campaign",
    }
    subprocess.run(command, cwd=REPOSITORY, env=environment, check=True)
    original_spec = spec_path.read_bytes()
    contract_path = tmp_path / "evaluation-contract.json"
    original_contract = contract_path.read_bytes()
    pinned_evolver_commit = "a" * 40
    existing_config = json.loads(config_path.read_text(encoding="utf-8"))
    existing_config["campaign"]["evolver"]["commit"] = pinned_evolver_commit
    config_path.write_text(json.dumps(existing_config), encoding="utf-8")

    changed = {
        **environment,
        "AGATE_GPU": "DIFFERENT_GPU",
        "ATREX_BOOTSTRAP_CREATION_KEY": "would-have-created-a-new-campaign",
        "AGATE_JOB_TIMEOUT": "999",
    }
    result = subprocess.run(
        command,
        cwd=REPOSITORY,
        env=changed,
        check=True,
        capture_output=True,
        text=True,
    )

    assert spec_path.read_bytes() == original_spec
    assert contract_path.read_bytes() == original_contract
    assert "stable-campaign (pinned existing workspace definition)" in result.stdout
    regenerated = RuntimeSettings.from_file(config_path)
    assert regenerated.campaign is not None
    assert regenerated.campaign.evolver.commit == pinned_evolver_commit
    assert f"Evolver commit: {pinned_evolver_commit}" in result.stdout


def test_bootstrap_local_secrets_are_private_and_stable(tmp_path: Path) -> None:
    env_file = tmp_path / "runtime.env"
    environment = {
        **os.environ,
        "ATREX_BOOTSTRAP_STATE_DIR": str(tmp_path),
        "ATREX_BOOTSTRAP_ENV_FILE": str(env_file),
    }
    command = "source examples/bootstrap/common.sh; atrex_example_ensure_local_secrets"
    subprocess.run(("bash", "-c", command), cwd=REPOSITORY, env=environment, check=True)
    initial = env_file.read_bytes()
    subprocess.run(("bash", "-c", command), cwd=REPOSITORY, env=environment, check=True)

    assert env_file.read_bytes() == initial
    assert stat.S_IMODE(env_file.stat().st_mode) == 0o600
    assert b"AGATE_AK" not in initial
    assert b"AGATE_SK" not in initial


def test_bootstrap_local_secrets_override_inherited_shell_values(tmp_path: Path) -> None:
    env_file = tmp_path / "runtime.env"
    environment = {
        **os.environ,
        "ATREX_BOOTSTRAP_STATE_DIR": str(tmp_path),
        "ATREX_BOOTSTRAP_ENV_FILE": str(env_file),
        "ATREX_CAPABILITY_SIGNING_KEY": "inherited-capability-key",
        "ATREX_ADMIN_BEARER_TOKEN": "inherited-admin-token",
    }
    command = (
        "source examples/bootstrap/common.sh; "
        "atrex_example_load_local_secrets; "
        "printf '%s\\n%s\\n' \"$ATREX_CAPABILITY_SIGNING_KEY\" "
        "\"$ATREX_ADMIN_BEARER_TOKEN\""
    )
    result = subprocess.run(
        ("bash", "-c", command),
        cwd=REPOSITORY,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    persisted = {}
    for line in env_file.read_text(encoding="utf-8").splitlines():
        name, value = line.removeprefix("export ").split("=", 1)
        persisted[name] = value.strip("'")
    assert result.stdout.splitlines() == [
        persisted["ATREX_CAPABILITY_SIGNING_KEY"],
        persisted["ATREX_ADMIN_BEARER_TOKEN"],
    ]
    assert "inherited-capability-key" not in result.stdout
    assert "inherited-admin-token" not in result.stdout
