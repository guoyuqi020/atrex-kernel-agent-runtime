"""Tests for the explicit-environment worker launcher."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from atrex_runtime.config import (
    BackendCredentialSettings,
    BwrapContainerSettings,
    BwrapSandboxSettings,
    CgroupResourceSettings,
)
from atrex_runtime.workers.launcher import (
    BackendCredentialMounts,
    BwrapContainerLauncher,
    BwrapSandboxLauncher,
    CleanEnvironmentLauncher,
)


def _resolver(tmp_path: Path) -> Path:
    path = tmp_path / "resolver/resolv.conf"
    path.parent.mkdir(exist_ok=True)
    path.write_text("nameserver 192.0.2.53\n", encoding="utf-8")
    return path


def test_clean_environment_launcher_does_not_use_a_shell(tmp_path: Path) -> None:
    launcher = CleanEnvironmentLauncher(Path("/usr/bin/env"))

    argv = launcher.wrap(
        ("/opt/agent/runtime", "--flag"),
        workspace=tmp_path,
        environment={"ZED": "last", "ALPHA": "first"},
    )

    assert argv == (
        "/usr/bin/env",
        "-i",
        "ALPHA=first",
        "ZED=last",
        "/opt/agent/runtime",
        "--flag",
    )


def test_container_launcher_uses_bwrap_without_systemd_and_projects_credentials(
    tmp_path: Path,
) -> None:
    host_home = tmp_path / "host-home"
    qoder = host_home / ".qoder"
    qodersec = host_home / ".qodersec"
    (qoder / ".auth").mkdir(parents=True)
    (qoder / ".auth/token").write_text("secret", encoding="utf-8")
    (qoder / "tasks").mkdir()
    (qoder / "tasks/old-task").write_text("volatile", encoding="utf-8")
    (qoder / "settings.json").write_text("{}", encoding="utf-8")
    qodersec.mkdir()
    (qodersec / "identity").write_text("identity", encoding="utf-8")
    credentials = BackendCredentialMounts.from_environment(
        BackendCredentialSettings(host_home=host_home),
        {"HOME": str(host_home), "PATH": "/usr/bin"},
    )
    assert credentials is not None
    root = tmp_path / "attempt"
    workspace = root / "run"
    session_home = workspace / "sessions/core/agent-home"
    session_home.mkdir(parents=True)
    launcher = BwrapContainerLauncher(
        Path("/usr/bin/env"),
        BwrapContainerSettings(
            bwrap_executable=Path("/usr/bin/bwrap"),
            resolv_conf=_resolver(tmp_path),
        ),
        (root,),
        credentials,
    )

    argv = launcher.wrap(
        ("/opt/core/run.py",),
        workspace=workspace,
        environment={
            "ATREX_AGENT_BACKEND": "qodercli",
            "HOME": str(session_home),
            "PATH": "/usr/bin",
        },
    )

    assert argv[0] == "/usr/bin/bwrap"
    assert "/usr/bin/systemd-run" not in argv
    assert not any(value.startswith("--property=") for value in argv)
    assert "--proc" not in argv
    assert any(
        argv[index : index + 3] == ("--ro-bind", "/proc", "/proc")
        for index in range(len(argv) - 2)
    )
    assert "ATREX_SANDBOX=bwrap-container" in argv
    assert "ATREX_WORKSPACE=/home/agent/workspace" in argv
    assert str(qoder) in argv
    assert str(qodersec) in argv
    assert "/home/agent/workspace/sessions/core/agent-home/.qoder" in argv
    assert (session_home / ".qoder/tasks").is_dir()


def test_container_launcher_requires_session_home_inside_workspace(tmp_path: Path) -> None:
    host_home = tmp_path / "host-home"
    (host_home / ".codex").mkdir(parents=True)
    (host_home / ".codex/auth.json").write_text("{}", encoding="utf-8")
    credentials = BackendCredentialMounts.from_environment(
        BackendCredentialSettings(host_home=host_home),
        {"HOME": str(host_home), "PATH": "/usr/bin"},
    )
    assert credentials is not None
    workspace = tmp_path / "attempt/run"
    workspace.mkdir(parents=True)
    launcher = BwrapContainerLauncher(
        Path("/usr/bin/env"),
        BwrapContainerSettings(
            bwrap_executable=Path("/usr/bin/bwrap"),
            resolv_conf=_resolver(tmp_path),
        ),
        (workspace.parent,),
        credentials,
    )

    with pytest.raises(ValueError, match="HOME must be inside"):
        launcher.wrap(
            ("/bin/true",),
            workspace=workspace,
            environment={
                "ATREX_AGENT_BACKEND": "codex",
                "HOME": str(tmp_path / "outside"),
            },
        )

def test_launcher_rejects_invalid_environment_names(tmp_path: Path) -> None:
    launcher = CleanEnvironmentLauncher(Path("/usr/bin/env"))

    with pytest.raises(ValueError, match="invalid environment key"):
        launcher.wrap(("/bin/true",), workspace=tmp_path, environment={"BAD-NAME": "value"})


def test_development_launcher_rejects_private_evaluator_path_environment(
    tmp_path: Path,
) -> None:
    launcher = CleanEnvironmentLauncher(Path("/usr/bin/env"))

    with pytest.raises(ValueError, match="private evaluator paths"):
        launcher.wrap(
            ("/bin/true",),
            workspace=tmp_path,
            environment={"ATREX_PRIVATE_REFERENCE_DIR": "/trusted/private"},
        )


def test_development_launcher_mounts_qoder_login_read_only_and_state_writable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    host_home = tmp_path / "host-home"
    qoder = host_home / ".qoder"
    qodersec = host_home / ".qodersec"
    qoder.mkdir(parents=True)
    (qoder / ".auth").mkdir()
    (qoder / "tasks").mkdir()
    (qoder / "logs").mkdir()
    (qoder / "session-env").mkdir()
    (qoder / "settings.json").write_text('{"model":"test"}', encoding="utf-8")
    (qoder / "state.json").write_text("{}", encoding="utf-8")
    qodersec.mkdir()
    bwrap = tmp_path / "bin/bwrap"
    bwrap.parent.mkdir()
    bwrap.write_text("", encoding="utf-8")
    bwrap.chmod(0o700)
    credentials = BackendCredentialMounts.from_environment(
        BackendCredentialSettings(
            host_home=host_home,
            development_bwrap_executable=bwrap,
        ),
        {"HOME": str(host_home), "PATH": "/usr/bin"},
    )
    assert credentials is not None
    workspace = tmp_path / "attempt/run"
    session_home = workspace / "sessions/core/agent-home"
    session_home.mkdir(parents=True)
    monkeypatch.setattr("atrex_runtime.workers.launcher.platform.system", lambda: "Linux")
    launcher = CleanEnvironmentLauncher(Path("/usr/bin/env"), credentials)

    argv = launcher.wrap(
        ("/opt/core/run.py",),
        workspace=workspace,
        environment={
            "ATREX_AGENT_BACKEND": "qodercli",
            "HOME": str(session_home),
            "PATH": "/usr/bin",
        },
    )

    assert argv[0] == str(bwrap)
    joined = "\n".join(argv)
    assert f"{qoder}\n{session_home / '.qoder'}" in joined
    assert f"{qodersec}\n{session_home / '.qodersec'}" in joined
    assert f"--tmpfs\n{session_home / '.qoder/tasks'}" in joined
    assert f"--tmpfs\n{session_home / '.qoder/logs'}" in joined
    assert f"--tmpfs\n{session_home / '.qoder/session-env'}" in joined
    staged = session_home / ".atrex-provider-state/qoder"
    assert (staged / "settings.json").read_text(encoding="utf-8") == '{"model":"test"}'
    assert f"{staged / 'settings.json'}\n{session_home / '.qoder/settings.json'}" in joined
    assert f"{staged / 'state.json'}\n{session_home / '.qoder/state.json'}" in joined
    assert f"HOME={session_home}" in argv


def test_backend_credentials_cover_claude_codex_qoder_and_pi(tmp_path: Path) -> None:
    host_home = tmp_path / "host-home"
    for relative in (
        ".claude",
        ".claude/plugins",
        ".codex/skills",
        ".qoder/.auth",
        ".qoder/entry",
        ".qodersec",
        ".pi/agent",
    ):
        (host_home / relative).mkdir(parents=True, exist_ok=True)
    (host_home / ".claude.json").write_text("{}", encoding="utf-8")
    (host_home / ".claude/.credentials.json").write_text("{}", encoding="utf-8")
    (host_home / ".codex/auth.json").write_text("{}", encoding="utf-8")
    (host_home / ".codex/config.toml").write_text("", encoding="utf-8")
    credentials = BackendCredentialMounts.from_environment(
        BackendCredentialSettings(host_home=host_home),
        {"HOME": str(host_home), "PATH": "/usr/bin"},
    )
    assert credentials is not None

    assert {mount.home_relative.as_posix() for mount in credentials.mounts_for("claude")} == {
        ".claude/.credentials.json",
        ".claude/plugins",
    }
    assert {mount.home_relative.as_posix() for mount in credentials.mounts_for("codex")} == {
        ".codex/auth.json",
        ".codex/skills",
    }
    assert {mount.home_relative.as_posix() for mount in credentials.mounts_for("qodercli")} == {
        ".qoder",
        ".qodersec",
    }
    assert {mount.home_relative.as_posix() for mount in credentials.mounts_for("pi")} == {
        ".pi/agent"
    }


def test_sandbox_launcher_mounts_codex_home_into_session_workspace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    host_home = tmp_path / "host-home"
    codex_home = host_home / ".codex"
    codex_home.mkdir(parents=True)
    (codex_home / "auth.json").write_text("{}", encoding="utf-8")
    (codex_home / "config.toml").write_text("", encoding="utf-8")
    (codex_home / "skills").mkdir()
    ca_bundle = tmp_path / "ca-certificates.crt"
    ca_bundle.write_text("test CA bundle\n", encoding="utf-8")
    monkeypatch.setattr(
        "atrex_runtime.workers.launcher._CODEX_CA_BUNDLE_CANDIDATES",
        (ca_bundle,),
    )
    credentials = BackendCredentialMounts.from_environment(
        BackendCredentialSettings(host_home=host_home),
        {"HOME": str(host_home), "PATH": "/usr/bin"},
    )
    assert credentials is not None
    root = tmp_path / "attempt-workspaces"
    workspace = root / "attempt-1/run-1"
    session_home = workspace / "sessions/core/agent-home"
    session_home.mkdir(parents=True)
    launcher = BwrapSandboxLauncher(
        Path("/usr/bin/env"),
        BwrapSandboxSettings(
            bwrap_executable=Path("/usr/bin/bwrap"),
            systemd_run_executable=Path("/usr/bin/systemd-run"),
            worker_user="atrex-worker",
            resolv_conf=_resolver(tmp_path),
            resources=CgroupResourceSettings(
                memory_max_bytes=1,
                memory_swap_max_bytes=0,
                cpu_quota_percent=1,
                tasks_max=1,
            ),
        ),
        (root,),
        credentials,
    )

    argv = launcher.wrap(
        ("/opt/core/run.py",),
        workspace=workspace,
        environment={
            "ATREX_AGENT_BACKEND": "codex",
            "HOME": str(session_home),
            "PATH": "/usr/bin",
        },
    )

    joined = "\n".join(argv)
    mapped_home = Path("/home/agent/workspace/sessions/core/agent-home")
    assert f"{codex_home / 'auth.json'}\n{codex_home / 'auth.json'}" in joined
    assert f"{codex_home / 'auth.json'}\n{mapped_home / '.codex/auth.json'}" in joined
    assert f"{codex_home}\n{mapped_home / '.codex'}" not in joined
    assert (session_home / ".codex").is_dir()
    assert (session_home / ".codex/config.toml").is_file()
    assert (session_home / ".codex/config.toml").stat().st_mode & 0o200
    assert str(codex_home / "config.toml") not in argv
    assert f"HOME={mapped_home}" in argv
    assert f"CODEX_HOME={mapped_home / '.codex'}" in argv
    assert f"SSL_CERT_FILE={ca_bundle}" in argv


def test_interactive_sandbox_projects_all_requested_backend_credentials(
    tmp_path: Path,
) -> None:
    host_home = tmp_path / "host-home"
    for relative in (".claude", ".codex", ".qoder/.auth", ".qodersec"):
        (host_home / relative).mkdir(parents=True, exist_ok=True)
    (host_home / ".claude/.credentials.json").write_text("{}", encoding="utf-8")
    (host_home / ".codex/auth.json").write_text("{}", encoding="utf-8")
    (host_home / ".codex/config.toml").write_text("", encoding="utf-8")
    credentials = BackendCredentialMounts.from_environment(
        BackendCredentialSettings(host_home=host_home),
        {"HOME": str(host_home), "PATH": "/usr/bin"},
    )
    assert credentials is not None
    root = tmp_path / "attempt-workspaces"
    workspace = root / "attempt-1/run-1"
    session_home = workspace / "sessions/core/agent-home"
    session_home.mkdir(parents=True)
    launcher = BwrapSandboxLauncher(
        Path("/usr/bin/env"),
        BwrapSandboxSettings(
            bwrap_executable=Path("/usr/bin/bwrap"),
            systemd_run_executable=Path("/usr/bin/systemd-run"),
            worker_user="atrex-worker",
            resolv_conf=_resolver(tmp_path),
            resources=CgroupResourceSettings(
                memory_max_bytes=1,
                memory_swap_max_bytes=0,
                cpu_quota_percent=1,
                tasks_max=1,
            ),
        ),
        (root,),
        credentials,
    )

    argv = launcher.wrap(
        ("/bin/bash", "-i"),
        workspace=workspace,
        environment={
            "ATREX_AGENT_BACKEND": "qodercli",
            "ATREX_DEV_SHELL_BACKENDS": "claude,codex,qodercli",
            "HOME": str(session_home),
            "PATH": "/usr/bin",
        },
        interactive=True,
    )

    joined = "\n".join(argv)
    mapped_home = Path("/home/agent/workspace/sessions/core/agent-home")
    assert str(host_home / ".claude/.credentials.json") in argv
    assert str(host_home / ".codex/auth.json") in argv
    assert str(host_home / ".qoder") in argv
    assert str(host_home / ".qodersec") in argv
    assert f"CLAUDE_CONFIG_DIR={mapped_home / '.claude'}" in argv
    assert f"CODEX_HOME={mapped_home / '.codex'}" in argv
    assert f"--tmpfs\n{mapped_home / '.claude/cache'}" in joined
    assert (session_home / ".claude").is_dir()
    assert (session_home / ".codex").is_dir()
    assert (session_home / ".qoder").is_dir()


def test_noninteractive_launcher_rejects_dev_shell_credential_expansion(
    tmp_path: Path,
) -> None:
    launcher = CleanEnvironmentLauncher(Path("/usr/bin/env"))

    with pytest.raises(ValueError, match="restricted to interactive shells"):
        launcher.wrap(
            ("/bin/true",),
            workspace=tmp_path,
            environment={"ATREX_DEV_SHELL_BACKENDS": "claude,codex"},
        )


def test_bwrap_launcher_builds_private_workspace_cgroup_with_host_network(
    tmp_path: Path,
) -> None:
    root = tmp_path / "attempt-workspaces"
    workspace = root / "attempt-1/run-1"
    workspace.mkdir(parents=True)
    credentials = tmp_path / "credentials"
    credentials.mkdir()
    settings = BwrapSandboxSettings(
        bwrap_executable=Path("/usr/bin/bwrap"),
        systemd_run_executable=Path("/usr/bin/systemd-run"),
        systemd_user=False,
        worker_user="atrex-worker",
        sandbox_home=Path("/home/agent"),
        workspace_mount=Path("/home/agent/workspace"),
        resolv_conf=_resolver(tmp_path),
        read_only_bind_paths=(credentials,),
        hidden_host_paths=(),
        resources=CgroupResourceSettings(
            memory_max_bytes=1024 * 1024,
            memory_swap_max_bytes=0,
            cpu_quota_percent=200,
            tasks_max=64,
        ),
    )
    launcher = BwrapSandboxLauncher(
        Path("/usr/bin/env"),
        settings,
        (root,),
    )

    argv = launcher.wrap(
        (str(workspace / "agent/optimizer/run.py"),),
        workspace=workspace,
        environment={
            "PATH": "/usr/bin:/bin",
            "ATREX_ATTEMPT_MANIFEST": str(workspace / "attempt.json"),
            "ATREX_GATEWAY_PROXY_URL": "http://127.0.0.1:8765",
        },
    )

    systemd_index = argv.index("/usr/bin/systemd-run")
    assert argv[systemd_index : systemd_index + 4] == (
        "/usr/bin/systemd-run",
        "--quiet",
        "--wait",
        "--pipe",
    )
    assert "--service-type=exec" in argv
    assert "--uid=atrex-worker" in argv
    assert "--property=MemoryMax=1048576" in argv
    assert "--property=CPUQuota=200%" in argv
    assert "--property=TasksMax=64" in argv
    assert not any(value.startswith("--property=NetworkNamespacePath=") for value in argv)
    bwrap_index = argv.index("/usr/bin/bwrap")
    bwrap = argv[bwrap_index:]
    ro_bind_index = bwrap.index("--ro-bind")
    assert bwrap[ro_bind_index : ro_bind_index + 3] == ("--ro-bind", "/", "/")
    proc_index = bwrap.index("--proc")
    assert bwrap[proc_index : proc_index + 2] == ("--proc", "/proc")
    assert str(tmp_path / "resolver/resolv.conf") in bwrap
    assert "/run/systemd/resolve/stub-resolv.conf" in bwrap
    assert "--cap-drop" in bwrap and "ALL" in bwrap
    final_workspace_index = bwrap.index("/home/agent/workspace")
    assert bwrap[final_workspace_index - 2 : final_workspace_index + 1] == (
        "--bind",
        str(workspace),
        "/home/agent/workspace",
    )
    assert "/run/atrex-sandbox-staging" not in bwrap
    assert "/home/agent/workspace/attempt.json" in "\n".join(bwrap)
    assert not any(value.startswith("HTTPS_PROXY=") for value in bwrap)
    assert bwrap[-1] == "/home/agent/workspace/agent/optimizer/run.py"


def test_bwrap_launcher_uses_a_pty_only_for_interactive_commands(tmp_path: Path) -> None:
    root = tmp_path / "attempt-workspaces"
    workspace = root / "attempt-1/run-1"
    workspace.mkdir(parents=True)
    launcher = BwrapSandboxLauncher(
        Path("/usr/bin/env"),
        BwrapSandboxSettings(
            bwrap_executable=Path("/usr/bin/bwrap"),
            systemd_run_executable=Path("/usr/bin/systemd-run"),
            worker_user="atrex-worker",
            resolv_conf=_resolver(tmp_path),
            resources=CgroupResourceSettings(
                memory_max_bytes=1,
                memory_swap_max_bytes=0,
                cpu_quota_percent=1,
                tasks_max=1,
            ),
        ),
        (root,),
    )

    argv = launcher.wrap(
        ("/bin/bash", "-i"),
        workspace=workspace,
        environment={},
        interactive=True,
    )

    assert "--pty" in argv
    assert "--pipe" not in argv
    bwrap = argv[argv.index("/usr/bin/bwrap") :]
    assert bwrap.count("/dev/pts") == 2


def test_bwrap_launcher_rejects_workspace_outside_configured_roots(tmp_path: Path) -> None:
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    launcher = BwrapSandboxLauncher(
        Path("/usr/bin/env"),
        BwrapSandboxSettings(
            bwrap_executable=Path("/usr/bin/bwrap"),
            systemd_run_executable=Path("/usr/bin/systemd-run"),
            worker_user="atrex-worker",
            resolv_conf=_resolver(tmp_path),
            resources=CgroupResourceSettings(
                memory_max_bytes=1,
                memory_swap_max_bytes=0,
                cpu_quota_percent=1,
                tasks_max=1,
            ),
        ),
        (allowed,),
    )

    with pytest.raises(ValueError, match="outside"):
        launcher.wrap(("/bin/true",), workspace=outside, environment={})


def test_bwrap_launcher_never_silently_degrades_off_linux(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launcher = BwrapSandboxLauncher(
        Path("/usr/bin/env"),
        BwrapSandboxSettings(
            bwrap_executable=Path("/usr/bin/bwrap"),
            systemd_run_executable=Path("/usr/bin/systemd-run"),
            worker_user="atrex-worker",
            resolv_conf=_resolver(tmp_path),
            resources=CgroupResourceSettings(
                memory_max_bytes=1,
                memory_swap_max_bytes=0,
                cpu_quota_percent=1,
                tasks_max=1,
            ),
        ),
        (tmp_path,),
    )
    monkeypatch.setattr("atrex_runtime.workers.launcher.platform.system", lambda: "Darwin")

    with pytest.raises(RuntimeError, match="requires Linux"):
        launcher.check_host()


def test_bwrap_host_check_serializes_shared_workspace_preparation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable = Path("/usr/bin/env")
    workspace_root = tmp_path / "attempt-workspaces"
    launcher = BwrapSandboxLauncher(
        executable,
        BwrapSandboxSettings(
            bwrap_executable=executable,
            systemd_run_executable=executable,
            worker_user="atrex-worker",
            resolv_conf=_resolver(tmp_path),
            resources=CgroupResourceSettings(
                memory_max_bytes=1,
                memory_swap_max_bytes=0,
                cpu_quota_percent=1,
                tasks_max=1,
            ),
        ),
        (workspace_root,),
    )
    original_is_file = Path.is_file
    controllers = Path("/sys/fs/cgroup/cgroup.controllers")
    monkeypatch.setattr(
        Path,
        "is_file",
        lambda path: path == controllers or original_is_file(path),
    )
    monkeypatch.setattr("atrex_runtime.workers.launcher.platform.system", lambda: "Linux")
    worker = SimpleNamespace(pw_uid=1234, pw_gid=1234)
    monkeypatch.setattr("atrex_runtime.workers.launcher.pwd.getpwnam", lambda _name: worker)
    calls: list[object] = []
    monkeypatch.setattr(
        "atrex_runtime.workers.launcher.fcntl.flock",
        lambda _stream, _operation: calls.append("locked"),
    )
    monkeypatch.setattr(
        BwrapSandboxLauncher,
        "_check_host_workspace_access",
        lambda _self, selected_worker: calls.append(selected_worker),
    )

    launcher.check_host()

    assert calls == ["locked", worker]
    assert (tmp_path / ".atrex-sandbox-host.lock").is_file()


def test_container_bwrap_host_check_skips_systemd_user_and_cgroup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable = Path("/usr/bin/env")
    workspace_root = tmp_path / "attempt-workspaces"
    launcher = BwrapContainerLauncher(
        executable,
        BwrapContainerSettings(
            bwrap_executable=executable,
            resolv_conf=_resolver(tmp_path),
        ),
        (workspace_root,),
    )
    monkeypatch.setattr("atrex_runtime.workers.launcher.platform.system", lambda: "Linux")
    monkeypatch.setattr(
        "atrex_runtime.workers.launcher.pwd.getpwnam",
        lambda _name: pytest.fail("container mode must not resolve a systemd Worker user"),
    )
    calls: list[str] = []
    monkeypatch.setattr(
        BwrapSandboxLauncher,
        "_check_container_workspace_access",
        lambda _self: calls.append("probed"),
    )

    launcher.check_host()

    assert calls == ["probed"]
    assert (tmp_path / ".atrex-sandbox-host.lock").is_file()


def test_bwrap_recreates_empty_foreign_owned_root_as_worker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace_root = tmp_path / "attempt-workspaces"
    workspace_root.mkdir()
    launcher = BwrapSandboxLauncher(
        Path("/usr/bin/env"),
        BwrapSandboxSettings(
            bwrap_executable=Path("/usr/bin/bwrap"),
            systemd_run_executable=Path("/usr/bin/systemd-run"),
            worker_user="atrex-worker",
            resolv_conf=_resolver(tmp_path),
            resources=CgroupResourceSettings(
                memory_max_bytes=1,
                memory_swap_max_bytes=0,
                cpu_quota_percent=1,
                tasks_max=1,
            ),
        ),
        (workspace_root,),
    )
    worker = SimpleNamespace(pw_uid=1234, pw_gid=1234)
    monkeypatch.setattr("atrex_runtime.workers.launcher.os.geteuid", lambda: 0)
    commands: list[tuple[str, ...]] = []

    def run_as_worker(argv: tuple[str, ...], **_kwargs: object) -> SimpleNamespace:
        commands.append(argv)
        workspace_root.mkdir()
        return SimpleNamespace(returncode=0, stderr=b"")

    monkeypatch.setattr("atrex_runtime.workers.launcher.subprocess.run", run_as_worker)
    ownership_checks: list[object] = []
    monkeypatch.setattr(
        BwrapSandboxLauncher,
        "_require_worker_owned_path",
        staticmethod(
            lambda _path, selected_worker: ownership_checks.append(selected_worker)
        ),
    )

    launcher._ensure_worker_directory(workspace_root, worker)

    assert workspace_root.is_dir()
    assert len(commands) == 1
    assert "--uid=atrex-worker" in commands[0]
    assert commands[0][-2:] == ("--", str(workspace_root))
    assert ownership_checks == [worker]


def test_bwrap_accepts_worker_owned_virtiofs_root_from_worker_view(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace_root = tmp_path / "lineage-bootstrap-workspaces"
    workspace_root.mkdir()
    (workspace_root / "retained-bootstrap").mkdir()
    launcher = BwrapSandboxLauncher(
        Path("/usr/bin/env"),
        BwrapSandboxSettings(
            bwrap_executable=Path("/usr/bin/bwrap"),
            systemd_run_executable=Path("/usr/bin/systemd-run"),
            worker_user="atrex-worker",
            resolv_conf=_resolver(tmp_path),
            resources=CgroupResourceSettings(
                memory_max_bytes=1,
                memory_swap_max_bytes=0,
                cpu_quota_percent=1,
                tasks_max=1,
            ),
        ),
        (workspace_root,),
    )
    worker = SimpleNamespace(pw_uid=1234, pw_gid=5678)
    monkeypatch.setattr("atrex_runtime.workers.launcher.os.geteuid", lambda: 0)
    commands: list[tuple[str, ...]] = []

    def worker_stat(argv: tuple[str, ...], **_kwargs: object) -> SimpleNamespace:
        commands.append(argv)
        return SimpleNamespace(returncode=0, stdout=b"1234:5678\n", stderr=b"")

    monkeypatch.setattr("atrex_runtime.workers.launcher.subprocess.run", worker_stat)

    launcher._ensure_worker_directory(workspace_root, worker)

    assert workspace_root.joinpath("retained-bootstrap").is_dir()
    assert len(commands) == 1
    assert "--uid=atrex-worker" in commands[0]
    assert Path(commands[0][-5]).name == "stat"
    assert commands[0][-4:] == ("-c", "%u:%g", "--", str(workspace_root))


def test_bwrap_waits_for_transient_virtiofs_worker_ownership(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace_root = tmp_path / "lineage-bootstrap-workspaces"
    workspace_root.mkdir()
    launcher = BwrapSandboxLauncher(
        Path("/usr/bin/env"),
        BwrapSandboxSettings(
            bwrap_executable=Path("/usr/bin/bwrap"),
            systemd_run_executable=Path("/usr/bin/systemd-run"),
            worker_user="atrex-worker",
            resolv_conf=_resolver(tmp_path),
            resources=CgroupResourceSettings(
                memory_max_bytes=1,
                memory_swap_max_bytes=0,
                cpu_quota_percent=1,
                tasks_max=1,
            ),
        ),
        (workspace_root,),
    )
    worker = SimpleNamespace(pw_uid=1234, pw_gid=5678)
    monkeypatch.setattr(
        Path,
        "stat",
        lambda _path: SimpleNamespace(st_uid=0, st_gid=0),
    )
    views = iter(((0, 0), (0, 0), (1234, 5678)))
    monkeypatch.setattr(
        BwrapSandboxLauncher,
        "_worker_view_ownership",
        lambda _self, _path, _worker: next(views),
    )
    sleeps: list[float] = []
    monkeypatch.setattr(
        "atrex_runtime.workers.launcher.time.sleep",
        sleeps.append,
    )

    launcher._require_worker_owned_path(workspace_root, worker)

    assert sleeps == [1.0, 1.0]


def test_bwrap_rejects_foreign_root_in_both_root_and_worker_views(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace_root = tmp_path / "lineage-bootstrap-workspaces"
    workspace_root.mkdir()
    (workspace_root / "retained-bootstrap").mkdir()
    launcher = BwrapSandboxLauncher(
        Path("/usr/bin/env"),
        BwrapSandboxSettings(
            bwrap_executable=Path("/usr/bin/bwrap"),
            systemd_run_executable=Path("/usr/bin/systemd-run"),
            worker_user="atrex-worker",
            resolv_conf=_resolver(tmp_path),
            resources=CgroupResourceSettings(
                memory_max_bytes=1,
                memory_swap_max_bytes=0,
                cpu_quota_percent=1,
                tasks_max=1,
            ),
        ),
        (workspace_root,),
    )
    worker = SimpleNamespace(pw_uid=1234, pw_gid=5678)
    monkeypatch.setattr("atrex_runtime.workers.launcher.os.geteuid", lambda: 0)
    monkeypatch.setattr(
        "atrex_runtime.workers.launcher.subprocess.run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0,
            stdout=b"999:999\n",
            stderr=b"",
        ),
    )

    with pytest.raises(RuntimeError, match=r"Worker view is 999:999"):
        launcher._ensure_worker_directory(workspace_root, worker)
