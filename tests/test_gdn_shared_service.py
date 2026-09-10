"""Two GDN input variants can share one control plane without sharing task inputs."""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_source_tree_example_schedule import REPOSITORY, _module

from atrex_runtime.bootstrap import CampaignSpecV3
from atrex_runtime.config import RuntimeSettings


def prepare_fixture(tmp_path, monkeypatch):
    module = _module("scripts/gdn/prepare.py")
    monkeypatch.setattr(module.sys, "platform", "linux")
    monkeypatch.setattr(module.pwd, "getpwnam", lambda name: SimpleNamespace(
        pw_name=name, pw_uid=1000, pw_dir="/home/worker",
    ))
    monkeypatch.setattr(module.pwd, "getpwuid", lambda _uid: SimpleNamespace(
        pw_name="worker", pw_uid=1000, pw_dir="/home/worker",
    ))
    # Test actual input import/policy validation, but no remote Git/network/GPU work.
    monkeypatch.setattr(module, "validate_runtime_sources", lambda *_args: {"verified": True})

    def prepare(workspace, *arguments, kit="GDN"):
        monkeypatch.setattr(module.sys, "argv", [
            "prepare.py", "--inputs", str(REPOSITORY / "data" / kit),
            "--workspace", str(workspace), "--worker-user", "worker", *arguments,
        ])
        module.main()

    service = tmp_path / "control"
    prepare(service, "--services-only", "--port", "8877", "--backend", "claude")
    return module, prepare, service


def test_shared_prepare_keeps_service_state_and_task_snapshots_separate(tmp_path, monkeypatch):
    module, prepare, service = prepare_fixture(tmp_path, monkeypatch)
    settings = RuntimeSettings.from_file(service / "runtime.json")
    assert settings.server.port == 8877
    assert settings.gpu_wiki is None
    assert settings.campaign.gateway_proxy_url == "http://127.0.0.1:8877"
    assert settings.storage.registry_database == service / "state/registry.sqlite"
    assert settings.campaign.attempt_workspaces_root == service / "state/attempt-workspaces"
    assert settings.campaign.launcher.mode == "container"
    assert settings.campaign.launcher.sandbox is None
    assert settings.campaign.launcher.backend_credentials.host_home == Path("/home/worker")
    assert not (service / "source").exists()
    assert not (service / "campaign.json").exists()
    assert not (service / "state").exists()
    before = (service / "runtime.json").read_bytes()
    helper = _module("scripts/gdn/gdn_workspace.py")
    specs = []
    tasks = []
    for kit in ("GDN", "GDN-full"):
        task = tmp_path / kit
        prepare(task, "--service-workspace", str(service), kit=kit)
        tasks.append(task)
        assert helper.resolve_service(task) == (service, service / "runtime.json")
        spec = CampaignSpecV3.from_file(task / "campaign.json")
        specs.append(spec)
        assert spec.shape_train == task / "task/shape_train.json"
        assert spec.lineages[next(iter(spec.lineages))].source_repository == task / "source"
        assert ("M64-oriented" in json.loads(spec.shape_train.read_text())["objective"]) == (
            kit == "GDN-full"
        )
        for name in ("runtime.json", "runtime-secrets.json", "state"):
            assert not (task / name).exists()
        binding = json.loads((task / "service-binding.json").read_text())
        assert not Path(binding["service_workspace"]).is_absolute()
        prepare(task, "--service-workspace", str(service), kit=kit)  # idempotent
    assert specs[0].creation_key != specs[1].creation_key
    assert (service / "runtime.json").read_bytes() == before
    with pytest.raises(SystemExit, match="Existing runtime state"):
        module.write_inputs(tasks[0], {"task/shape_train.json": b"changed"})
    # The marker protects a service even before persistent Runtime state exists.
    with pytest.raises(SystemExit, match=r"refusing to replace runtime\.json"):
        prepare(service, "--services-only", "--port", "8878")
    assert (service / "runtime.json").read_bytes() == before


@pytest.mark.parametrize("role", ["campaign", "ablation"])
def test_both_tasks_load_same_config_and_secrets_but_distinct_definitions(
    tmp_path, monkeypatch, role,
):
    _, prepare, service = prepare_fixture(tmp_path, monkeypatch)
    tasks = [tmp_path / kit for kit in ("GDN", "GDN-full")]
    for task in tasks:
        prepare(task, "--service-workspace", str(service), kit=task.name)
    runner = _module("scripts/gdn/run.py")
    calls, keys = [], []

    def record(command):
        calls.append(command)
        assert command[command.index("--config") + 1] == str(service / "runtime.json")
        keys.append((os.environ["ATREX_CAPABILITY_SIGNING_KEY"],
                     os.environ["ATREX_ADMIN_BEARER_TOKEN"]))

    def run(command, **_kwargs):
        record(command)
        return SimpleNamespace(stdout=json.dumps({"campaign_id": "test-campaign"}))

    def execute(_executable, command):
        record(command)
        raise SystemExit(0)

    monkeypatch.setattr(runner.subprocess, "run", run)
    monkeypatch.setattr(runner.os, "execv", execute)
    # Restore environment after the runner's normal os.environ.update side effect.
    monkeypatch.setenv("ATREX_CAPABILITY_SIGNING_KEY", "before")
    monkeypatch.setenv("ATREX_ADMIN_BEARER_TOKEN", "before")
    for task in tasks:
        monkeypatch.setattr(runner.sys, "argv", ["run.py", role, "--workspace", str(task)])
        if role == "ablation":
            with pytest.raises(SystemExit) as stopped:
                runner.main()
            assert stopped.value.code == 0
        else:
            runner.main()
        assert not (task / "runtime-secrets.json").exists()
    assert len(set(keys)) == 1
    for task in tasks:
        assert any(str(task / ("campaign.json" if role == "campaign" else "ablation-campaign.json"))
                   in command for command in calls)
    assert (service / "runtime-secrets.json").stat().st_mode & 0o777 == 0o600


def test_invalid_binding_and_roles_fail_before_any_process(tmp_path, monkeypatch):
    _, prepare, service = prepare_fixture(tmp_path, monkeypatch)
    task = tmp_path / "GDN"
    prepare(task, "--service-workspace", str(service))
    runner = _module("scripts/gdn/run.py")
    monkeypatch.setattr(runner.subprocess, "run", lambda *_a, **_k: pytest.fail("process started"))
    monkeypatch.setattr(runner.os, "execv", lambda *_a: pytest.fail("process started"))
    for arguments, message in (
        (["serve", "--workspace", str(task)], "Start the shared Runtime once"),
        (["campaign", "--workspace", str(service)], "Select a task workspace"),
        (["campaign", "--workspace", str(task), "--service-workspace", str(tmp_path / "other")],
         "differs from the task"),
    ):
        monkeypatch.setattr(runner.sys, "argv", ["run.py", *arguments])
        with pytest.raises(SystemExit, match=message):
            runner.main()
    with pytest.raises(SystemExit, match="--backend differs"):
        prepare(tmp_path / "other", "--service-workspace", str(service), "--backend", "codex")
    with pytest.raises(SystemExit, match="non-nested"):
        prepare(service / "task", "--service-workspace", str(service))
    with pytest.raises(SystemExit, match="Cannot attach a standalone"):
        prepare(service, "--service-workspace", str(service))
    config = json.loads((service / "runtime.json").read_text())
    config["server"]["port"] = 9999
    (service / "runtime.json").write_text(json.dumps(config))
    monkeypatch.setattr(runner.sys, "argv", ["run.py", "campaign", "--workspace", str(task)])
    with pytest.raises(SystemExit, match="Shared Runtime config changed"):
        runner.main()
    assert not (service / "runtime-secrets.json").exists()


def test_concurrent_runners_share_one_complete_key_file(tmp_path):
    helper = _module("scripts/gdn/gdn_workspace.py")
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: helper.load_service_secrets(tmp_path), range(16)))
    assert all(value == results[0] for value in results)
    path = tmp_path / "runtime-secrets.json"
    assert json.loads(path.read_text()) == results[0]
    assert path.stat().st_mode & 0o777 == 0o600
    path.write_text("broken")
    with pytest.raises(SystemExit, match="refusing to regenerate"):
        helper.load_service_secrets(tmp_path)
    assert path.read_text() == "broken"


@pytest.mark.parametrize("uid,name,home", [(1000, "worker", "/home/worker"), (0, "root", "/root")])
def test_container_preparation_never_switches_user_or_requires_systemd(
    tmp_path, monkeypatch, uid, name, home,
):
    import argparse

    module = _module("scripts/gdn/prepare.py")
    monkeypatch.setattr(module.pwd, "getpwuid", lambda _uid: SimpleNamespace(
        pw_name=name, pw_uid=uid, pw_dir=home,
    ))
    monkeypatch.setattr(module.pwd, "getpwnam", lambda _name: pytest.fail("user switch lookup"))
    monkeypatch.setenv("SUDO_USER", "unrelated-host-user")
    args = argparse.Namespace(worker_user=None, backend=None, port=None)
    for kit in ("GDN", "GDN-full"):
        value, worker, _, _ = module.configure_runtime(REPOSITORY / "data" / kit, tmp_path, args)
        settings = RuntimeSettings.model_validate(value)
        assert settings.gpu_wiki is None
        assert worker.pw_name == name
        launcher = settings.campaign.launcher
        assert launcher.mode == "container"
        assert launcher.sandbox is None
        assert launcher.backend_credentials.host_home == Path(home)
        assert launcher.container.resolv_conf == Path("/etc/resolv.conf")
        assert not {"resources", "worker_user", "systemd_run_executable"} & set(
            value["campaign"]["launcher"]["container"]
        )
    with pytest.raises(SystemExit, match="cannot switch users"):
        module.worker_for_mode("container", "someone-else")


@pytest.mark.parametrize("kit", ["GDN", "GDN-full"])
def test_gdn_defaults_select_container_launcher_for_both_agents(monkeypatch, kit):
    from atrex_runtime.composition.campaign import build_worker_launcher
    from atrex_runtime.workers.launcher import BwrapContainerLauncher

    settings = RuntimeSettings.from_file(REPOSITORY / "data" / kit / "runtime.template.json")
    checks = []
    monkeypatch.setattr(BwrapContainerLauncher, "check_host", lambda self: checks.append(self))
    launcher = build_worker_launcher(settings, {})
    assert isinstance(launcher, BwrapContainerLauncher)
    assert launcher.use_systemd_cgroup is False
    assert checks == [launcher]
    assert settings.storage.registry_database.parent in launcher.settings.hidden_host_paths


def test_legacy_sandbox_worker_selection_remains_supported(monkeypatch):
    module = _module("scripts/gdn/prepare.py")
    monkeypatch.setattr(module.pwd, "getpwnam", lambda name: SimpleNamespace(
        pw_name=name, pw_uid=1000 if name == "worker" else 0, pw_dir="/home/worker",
    ))
    assert module.worker_for_mode("sandbox", None, "worker").pw_name == "worker"
    with pytest.raises(SystemExit, match="non-root"):
        module.worker_for_mode("sandbox", "root")
