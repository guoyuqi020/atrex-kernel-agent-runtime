"""Skill/Hook installation is per Session, never a host-global registration."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from atrex_runtime.config import BackendCredentialSettings, BwrapContainerSettings
from atrex_runtime.workers.extensions import install_optimizer_extensions
from atrex_runtime.workers.launcher import (
    BackendCredentialMounts,
    BwrapContainerLauncher,
    CleanEnvironmentLauncher,
)


def _workspace(root: Path, phase: str = "optimization_attempt") -> dict[str, str]:
    home = root / "sessions/core/agent-home"
    home.mkdir(parents=True)
    skill = root / "skills/profile-analysis"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        "---\nname: profile-analysis\ndescription: Analyze profiles\n---\n"
    )
    (skill / "analyze.py").write_text("print('analysis')\n")
    (root / "skills/README.md").write_text("index")
    (root / "hooks").mkdir()
    hook = {
        "hooks": {
            "SessionStart": [
                {
                    "hooks": [
                        {
                            "type": "command",
                            "command": 'python3 "$WORKSPACE_ROOT/hooks/start.py"',
                        }
                    ]
                }
            ]
        }
    }
    for backend in ("claude", "codex"):
        (root / f"hooks/{backend}.json").write_text(json.dumps(hook))
    (root / "hooks/start.py").write_text("raise RuntimeError('installer must not execute hooks')\n")
    return {"HOME": str(home), "ATREX_CORE_PHASE": phase, "ATREX_AGENT_BACKEND": "claude"}


@pytest.mark.parametrize("phase", ["optimization_attempt", "framework_baseline"])
def test_installation_is_private_snapshot_and_keeps_state_separate(
    tmp_path: Path, phase: str
) -> None:
    first, second = tmp_path / "attempt-a", tmp_path / "attempt-b"
    env_a, env_b = _workspace(first, phase), _workspace(second, phase)
    host = tmp_path / "host/.claude"
    host.mkdir(parents=True)
    (host / "settings.json").write_text('{"untouched": true}')
    for root, env in ((first, env_a), (second, env_b)):
        result = install_optimizer_extensions(root, env, ("claude", "codex"))
        home = Path(env["HOME"])
        assert result["CLAUDE_CONFIG_DIR"] == str(home / ".claude")
        assert result["CODEX_HOME"] == str(home / ".codex")
        assert result["ATREX_OPTIMIZER_CODEX_HOOKS"] == "1"
        assert result["ATREX_WORKSPACE"] == str(root)
        for target in (".claude/skills", ".agents/skills"):
            installed = home / target / "profile-analysis/SKILL.md"
            assert (
                installed.read_bytes() == (root / "skills/profile-analysis/SKILL.md").read_bytes()
            )
            assert not installed.is_symlink()
            assert (
                installed.stat().st_ino != (root / "skills/profile-analysis/SKILL.md").stat().st_ino
            )
        assert json.loads((home / ".codex/hooks.json").read_text())["hooks"]["SessionStart"]
    installed = Path(env_a["HOME"]) / ".claude/skills/profile-analysis/SKILL.md"
    installed.write_text("local install edit")
    assert (second / "skills/profile-analysis/SKILL.md").read_text().startswith("---")
    assert (first / "skills/profile-analysis/SKILL.md").read_text().startswith("---")
    assert (host / "settings.json").read_text() == '{"untouched": true}'
    assert not (first / ".claude").exists()


def test_repeat_install_refreshes_private_skills(tmp_path: Path) -> None:
    env = _workspace(tmp_path)
    install_optimizer_extensions(tmp_path, env, ("claude",))
    (tmp_path / "skills/profile-analysis/SKILL.md").unlink()
    install_optimizer_extensions(tmp_path, env, ("claude",))
    assert not (Path(env["HOME"]) / ".claude/skills/profile-analysis").exists()


def test_removing_hooks_does_not_leave_stale_registrations(tmp_path: Path) -> None:
    env = _workspace(tmp_path)
    install_optimizer_extensions(tmp_path, env, ("claude", "codex"))
    for backend in ("claude", "codex"):
        (tmp_path / f"hooks/{backend}.json").unlink()
    result = install_optimizer_extensions(tmp_path, env, ("claude", "codex"))
    assert result["ATREX_OPTIMIZER_CODEX_HOOKS"] == "0"
    home = Path(env["HOME"])
    assert json.loads((home / ".claude/settings.json").read_text())["hooks"] == {}
    assert json.loads((home / ".codex/hooks.json").read_text())["hooks"] == {}


def test_hook_install_replaces_hardlink_without_editing_external_file(tmp_path: Path) -> None:
    root = tmp_path / "attempt"
    env = _workspace(root)
    host = tmp_path / "host-settings.json"
    host.write_text('{"env":{"KEEP":"value"}}')
    home = Path(env["HOME"])
    (home / ".claude").mkdir()
    os.link(host, home / ".claude/settings.json")
    install_optimizer_extensions(root, env, ("claude",))
    assert host.read_text() == '{"env":{"KEEP":"value"}}'
    assert (home / ".claude/settings.json").stat().st_ino != host.stat().st_ino


@pytest.mark.parametrize("alias", ["symlink", "hardlink"])
def test_credential_projection_rejects_shared_config_before_copying(
    tmp_path: Path,
    alias: str,
) -> None:
    root = tmp_path / "attempt"
    env = _workspace(root)
    host = tmp_path / "host"
    (host / ".claude").mkdir(parents=True)
    (host / ".claude/settings.json").write_text('{"new":"settings"}')
    outside = tmp_path / "outside.json"
    outside.write_text('{"untouched":true}')
    home = Path(env["HOME"])
    (home / ".claude").mkdir()
    target = home / ".claude/settings.json"
    if alias == "symlink":
        target.symlink_to(outside)
    else:
        os.link(outside, target)
    credentials = BackendCredentialMounts.from_environment(
        BackendCredentialSettings(host_home=host),
        {"HOME": str(host)},
    )
    assert credentials is not None
    with pytest.raises(ValueError, match="alias"):
        credentials.prepare_writable_backend_home("claude", home)
    assert outside.read_text() == '{"untouched":true}'


@pytest.mark.parametrize("phase", ["evolution", "problem_generalization", ""])
def test_other_phases_do_not_install_candidate_hooks(tmp_path: Path, phase: str) -> None:
    env = _workspace(tmp_path, phase)
    assert install_optimizer_extensions(tmp_path, env, ("claude", "codex")) == {}
    assert not (Path(env["HOME"]) / ".claude").exists()


def test_only_selected_backend_config_is_parsed(tmp_path: Path) -> None:
    env = _workspace(tmp_path)
    (tmp_path / "hooks/codex.json").write_text("invalid")
    result = install_optimizer_extensions(tmp_path, env, ("claude",))
    assert "CODEX_HOME" not in result
    with pytest.raises(ValueError, match=r"Invalid Optimizer extension config.*codex\.json"):
        install_optimizer_extensions(tmp_path, env, ("codex",))


@pytest.mark.parametrize(
    "value",
    [
        [],
        {"hooks": []},
        {"hooks": {"Stop": {}}},
        {"hooks": {"Stop": [{"hooks": [{"type": "command"}]}]}},
    ],
)
def test_bad_hook_config_has_actionable_error(tmp_path: Path, value: object) -> None:
    env = _workspace(tmp_path)
    (tmp_path / "hooks/claude.json").write_text(json.dumps(value))
    with pytest.raises(ValueError, match=r"claude\.json"):
        install_optimizer_extensions(tmp_path, env, ("claude",))


@pytest.mark.parametrize(
    "relative", ["skills/profile-analysis", "hooks/claude.json", "sessions/core/agent-home/.claude"]
)
def test_symlinks_cannot_redirect_installation(tmp_path: Path, relative: str) -> None:
    root = tmp_path / "attempt"
    env = _workspace(root)
    original = root / relative
    outside = tmp_path / "outside"
    if original.exists():
        original.rename(outside)
    else:
        outside.mkdir()
    original.symlink_to(outside, target_is_directory=outside.is_dir())
    with pytest.raises(ValueError, match="symlink"):
        install_optimizer_extensions(root, env, ("claude",))


def test_global_home_is_rejected_and_no_global_environment_is_mutated(tmp_path: Path) -> None:
    env = _workspace(tmp_path / "attempt")
    env["HOME"] = str(tmp_path / "host")
    before = dict(os.environ)
    with pytest.raises(ValueError, match="HOME inside"):
        install_optimizer_extensions(tmp_path / "attempt", env, ("claude",))
    assert dict(os.environ) == before


def test_home_cannot_escape_through_parent_components(tmp_path: Path) -> None:
    root = tmp_path / "attempt"
    env = _workspace(root)
    env["HOME"] = str(root / "sessions/../../host")
    with pytest.raises(ValueError, match="traverse"):
        install_optimizer_extensions(root, env, ("claude",))
    assert not (tmp_path / "host").exists()


def test_development_launcher_without_credentials_still_isolates_cli_homes(tmp_path: Path) -> None:
    env = _workspace(tmp_path)
    env["CLAUDE_CONFIG_DIR"] = "/host/.claude"
    argv = CleanEnvironmentLauncher(Path("/usr/bin/env")).wrap(
        ("/bin/true",),
        workspace=tmp_path,
        environment=env,
    )
    assert f"CLAUDE_CONFIG_DIR={env['HOME']}/.claude" in argv
    assert "CLAUDE_CONFIG_DIR=/host/.claude" not in argv


def test_container_installs_after_credential_copy_and_maps_script_environment(
    tmp_path: Path,
) -> None:
    root = tmp_path / "attempt"
    env = _workspace(root)
    host = tmp_path / "host"
    (host / ".claude").mkdir(parents=True)
    settings = '{"env":{"API_BASE":"https://example.invalid"},"hooks":{"Stop":[]}}'
    (host / ".claude/settings.json").write_text(settings)
    credentials = BackendCredentialMounts.from_environment(
        BackendCredentialSettings(host_home=host),
        {"HOME": str(host)},
    )
    resolver = tmp_path / "resolv.conf"
    resolver.write_text("nameserver 192.0.2.53\n")
    launcher = BwrapContainerLauncher(
        Path("/usr/bin/env"),
        BwrapContainerSettings(
            bwrap_executable=Path("/usr/bin/bwrap"),
            resolv_conf=resolver,
        ),
        (root,),
        credentials,
    )
    argv = launcher.wrap(("/bin/true",), workspace=root, environment=env)
    copied = json.loads((Path(env["HOME"]) / ".claude/settings.json").read_text())
    assert copied["env"] == {"API_BASE": "https://example.invalid"}
    assert "SessionStart" in copied["hooks"] and "Stop" not in copied["hooks"]
    assert (host / ".claude/settings.json").read_text() == settings
    assert "CLAUDE_CONFIG_DIR=/home/agent/workspace/sessions/core/agent-home/.claude" in argv
    assert "ATREX_WORKSPACE=/home/agent/workspace" in argv
    assert (
        copied["hooks"]["SessionStart"][0]["hooks"][0]["command"]
        == 'python3 "$WORKSPACE_ROOT/hooks/start.py"'
    )
