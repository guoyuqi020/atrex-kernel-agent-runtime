"""Project adaptive Skills into one Optimizer Session's private Home."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

_PHASES = {"optimization_attempt", "framework_baseline"}


def _local_path(root: Path, path: Path) -> None:
    """Never traverse links or special files, even in a reused dev-shell Home."""
    relative = path.relative_to(root)
    if ".." in relative.parts:
        raise ValueError("Optimizer extension path cannot traverse outside its workspace")
    current = root
    for part in relative.parts:
        current /= part
        if current.is_symlink():
            raise ValueError(f"Optimizer extension path cannot be a symlink: {current}")
        if current.exists() and not (current.is_dir() or current.is_file()):
            raise ValueError(
                f"Optimizer extension path must be a regular file/directory: {current}"
            )


def _json_object(root: Path, path: Path) -> dict[str, Any]:
    _local_path(root, path)
    try:
        if path.stat().st_size > 1024 * 1024:
            raise ValueError("file exceeds 1 MiB")
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("expected a JSON object")
        return value
    except (OSError, ValueError) as error:
        raise ValueError(f"Invalid private Provider config {path}: {error}") from error


def _write_json(root: Path, path: Path, value: Mapping[str, Any]) -> None:
    _local_path(root, path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _disable_provider_hooks(workspace: Path, config_dir: Path, backend: str) -> None:
    """Remove host/provider Hooks from the Session-local copy.

    Terminal handoff is guarded by the Runtime protocol, not backend-native Hooks.
    The copied credential/configuration Home is private, so this never mutates host
    configuration.
    """
    if backend == "claude":
        path = config_dir / "settings.json"
        if not path.exists():
            return
        settings = _json_object(workspace, path)
        if settings.pop("hooks", None) is not None:
            _write_json(workspace, path, settings)
        return
    path = config_dir / "hooks.json"
    _local_path(workspace, path)
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Private Codex hooks path is invalid: {path}")
        path.unlink()


def _install_skills(workspace: Path, destination: Path) -> None:
    source = workspace / "skills"
    _local_path(workspace, source)
    _local_path(workspace, destination)
    if not source.is_dir():
        return
    skills: list[Path] = []
    for skill in sorted(source.iterdir()):
        _local_path(workspace, skill)
        if not skill.is_dir():
            continue  # README and loose notes are not CLI Skills.
        _local_path(workspace, skill / "SKILL.md")
        if not (skill / "SKILL.md").is_file():
            continue
        for item in skill.rglob("*"):
            _local_path(workspace, item)
        skills.append(skill)
    # This is a generated, Session-local discovery directory, not inherited State.
    # Re-wrapping a dev shell refreshes it, including Skills removed from the source.
    if destination.exists():
        for item in destination.rglob("*"):
            _local_path(workspace, item)
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True, mode=0o700)
    for skill in skills:
        shutil.copytree(skill, destination / skill.name)


def install_optimizer_extensions(
    workspace: Path,
    environment: Mapping[str, str],
    backends: Sequence[str],
    *,
    visible_workspace: Path | None = None,
) -> dict[str, str]:
    """Called after credential projection, before either launcher starts any process.

    Only the selected Optimizer backends are installed. Installation products live
    in this Session's private Home and are never persisted as adaptive State.
    """
    if environment.get("ATREX_CORE_PHASE") not in _PHASES:
        return {}
    workspace = workspace.resolve()
    home = Path(environment.get("HOME", ""))
    if not home.is_absolute() or home == workspace or not home.is_relative_to(workspace):
        raise ValueError("Optimizer extension installation requires HOME inside its workspace")
    _local_path(workspace, home)
    # Keep installation products outside the three checkpointed adaptive directories.
    if home.relative_to(workspace).parts[0] != "sessions":
        raise ValueError("Optimizer extension HOME must be under workspace/sessions")
    visible_workspace = visible_workspace or workspace
    visible_home = visible_workspace / home.relative_to(workspace)
    result = {
        "ATREX_WORKSPACE": str(visible_workspace),
        "WORKSPACE_ROOT": str(visible_workspace),
    }
    for backend in backends:
        if backend not in {"claude", "codex"}:
            continue
        config_dir = home / f".{backend}"
        _local_path(workspace, config_dir)
        config_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        result["CLAUDE_CONFIG_DIR" if backend == "claude" else "CODEX_HOME"] = str(
            visible_home / f".{backend}"
        )
        _disable_provider_hooks(workspace, config_dir, backend)
        _install_skills(
            workspace, home / (".claude/skills" if backend == "claude" else ".agents/skills")
        )
    return result
