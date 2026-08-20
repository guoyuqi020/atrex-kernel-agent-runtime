"""Strict production Sandbox composition validation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from atrex_runtime.composition.campaign import build_worker_launcher
from atrex_runtime.config import RuntimeSettings
from atrex_runtime.workers.launcher import BwrapSandboxLauncher

REPOSITORY = Path(__file__).resolve().parents[1]


def _config(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    value = json.loads((REPOSITORY / "runtime.example.json").read_text(encoding="utf-8"))
    path = tmp_path / "runtime.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    return path, value


def test_root_template_selects_strict_host_network_sandbox() -> None:
    settings = RuntimeSettings.from_file(REPOSITORY / "runtime.example.json")

    assert settings.campaign is not None
    assert settings.campaign.launcher.mode == "sandbox"
    assert settings.campaign.launcher.sandbox is not None
    assert settings.campaign.launcher.sandbox.workspace_mount.as_posix() == (
        "/home/agent/workspace"
    )
    assert settings.campaign.launcher.sandbox.systemd_user is False
    assert settings.campaign.launcher.sandbox.worker_user == "atrex-worker"
    assert settings.server.host == "127.0.0.1"
    assert settings.campaign.launcher.sandbox.resolv_conf == Path(
        "/run/systemd/resolve/resolv.conf"
    )


def test_development_launcher_does_not_require_sandbox_settings(tmp_path: Path) -> None:
    path, value = _config(tmp_path)
    campaign = value["campaign"]
    assert isinstance(campaign, dict)
    campaign["launcher"] = {"mode": "development", "env_executable": "/usr/bin/env"}
    path.write_text(json.dumps(value), encoding="utf-8")

    settings = RuntimeSettings.from_file(path)
    assert settings.campaign is not None
    assert settings.campaign.launcher.mode == "development"


def test_sandbox_launcher_automatically_masks_runtime_storage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = RuntimeSettings.from_file(REPOSITORY / "runtime.example.json")
    monkeypatch.setattr(BwrapSandboxLauncher, "check_host", lambda _self: None)

    launcher = build_worker_launcher(settings, {})

    assert isinstance(launcher, BwrapSandboxLauncher)
    hidden = set(launcher.settings.hidden_host_paths)
    assert settings.storage.artifacts_root in hidden
    assert settings.storage.registry_database.parent in hidden
    assert settings.storage.gateway_database.parent in hidden
    assert settings.storage.agate_jobs_database.parent in hidden


def test_sandbox_gateway_must_be_inside_runtime_only_cidr(tmp_path: Path) -> None:
    path, value = _config(tmp_path)
    campaign = value["campaign"]
    assert isinstance(campaign, dict)
    campaign["gateway_proxy_url"] = "http://10.77.0.2:8765"
    path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError, match="this Runtime API socket"):
        RuntimeSettings.from_file(path)


def test_sandbox_read_only_bind_cannot_reexpose_runtime_storage(tmp_path: Path) -> None:
    path, value = _config(tmp_path)
    campaign = value["campaign"]
    storage = value["storage"]
    assert isinstance(campaign, dict)
    assert isinstance(storage, dict)
    launcher = campaign["launcher"]
    assert isinstance(launcher, dict)
    sandbox = launcher["sandbox"]
    assert isinstance(sandbox, dict)
    sandbox["read_only_bind_paths"] = [str(tmp_path / str(storage["artifacts_root"]))]
    path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError, match="cannot expose Runtime storage"):
        RuntimeSettings.from_file(path)


def test_sandbox_rejects_user_systemd_manager(tmp_path: Path) -> None:
    path, value = _config(tmp_path)
    campaign = value["campaign"]
    assert isinstance(campaign, dict)
    launcher = campaign["launcher"]
    assert isinstance(launcher, dict)
    sandbox = launcher["sandbox"]
    assert isinstance(sandbox, dict)
    sandbox["systemd_user"] = True
    path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError):
        RuntimeSettings.from_file(path)


def test_sandbox_requires_nonempty_worker_user(tmp_path: Path) -> None:
    path, value = _config(tmp_path)
    campaign = value["campaign"]
    assert isinstance(campaign, dict)
    launcher = campaign["launcher"]
    assert isinstance(launcher, dict)
    sandbox = launcher["sandbox"]
    assert isinstance(sandbox, dict)
    sandbox["worker_user"] = ""
    path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError):
        RuntimeSettings.from_file(path)
