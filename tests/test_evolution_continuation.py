"""Native Evolver history continues across fresh, isolated workspaces."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from atrex_runtime.domain.errors import InfrastructureError
from atrex_runtime.workers.evolution_continuation import (
    CONTINUATION_METADATA,
    EvolutionConversation,
)


def _previous(root: Path, backend: str = "claude", session_id: str = "session-1") -> Path:
    workspace = root / "agent-v0/run-1"
    home = workspace / "scratch/agent-home"
    home.mkdir(parents=True)
    (home / CONTINUATION_METADATA).write_text(
        json.dumps(
            {
                "backend": backend,
                "session_id": session_id,
            }
        )
    )
    return workspace


@pytest.mark.parametrize(
    "backend,relative",
    (
        ("claude", ".claude/projects/project/session-1.jsonl"),
        ("qodercli", ".qoder/projects/project/session-1.jsonl"),
        ("codex", ".codex/sessions/rollout-session-1.jsonl"),
        ("pi", ".atrex-pi/evolver.jsonl"),
    ),
)
def test_continuation_copies_only_native_history(
    tmp_path: Path,
    backend: str,
    relative: str,
) -> None:
    previous = _previous(tmp_path, backend)
    old_home = previous / "scratch/agent-home"
    native = old_home / relative
    native.parent.mkdir(parents=True)
    native.write_text('{"type":"user","text":"prior turn"}\n')
    (old_home / "auth.json").write_text("secret")
    (previous / "scratch/request.json").write_text("old request")
    (previous / "candidate").mkdir()
    (previous / "candidate/tool.py").write_text("old candidate")
    conversation = EvolutionConversation(tmp_path, "lineage-1", backend)
    conversation.remember(previous)
    current = tmp_path / "agent-v2/run-2"
    home = current / "scratch/agent-home"
    home.mkdir(parents=True)
    assert conversation.restore(current, home) == "session-1"
    assert (home / relative).read_bytes() == native.read_bytes()
    assert not (home / "auth.json").exists()
    assert not (current / "scratch/request.json").exists()
    assert not (current / "candidate").exists()
    assert EvolutionConversation(tmp_path, "lineage-1", backend).restore(current, home)


def test_lineages_and_backends_never_share_a_conversation(tmp_path: Path) -> None:
    previous = _previous(tmp_path)
    native = previous / "scratch/agent-home/.claude/projects/project/session-1.jsonl"
    native.parent.mkdir(parents=True)
    native.write_text("{}\n")
    EvolutionConversation(tmp_path, "lineage-1", "claude").remember(previous)
    home = tmp_path / "current/scratch/agent-home"
    home.mkdir(parents=True)
    assert (
        EvolutionConversation(tmp_path, "lineage-2", "claude").restore(home.parents[1], home)
        is None
    )
    assert (
        EvolutionConversation(tmp_path, "lineage-1", "codex").restore(home.parents[1], home) is None
    )


def test_failure_before_provider_start_does_not_resume_a_nonexistent_session(
    tmp_path: Path,
) -> None:
    previous = _previous(tmp_path)
    EvolutionConversation(tmp_path, "lineage", "claude").remember(previous)
    assert (
        EvolutionConversation(tmp_path, "lineage", "claude").restore(
            tmp_path / "current",
            tmp_path / "current/scratch/agent-home",
        )
        is None
    )


def test_hard_killed_codex_recovers_id_from_native_session_meta(tmp_path: Path) -> None:
    previous = _previous(tmp_path, "codex", "")
    native = previous / "scratch/agent-home/.codex/sessions/rollout-thread-1.jsonl"
    native.parent.mkdir(parents=True)
    native.write_text(json.dumps({"type": "session_meta", "payload": {"id": "thread-1"}}) + "\n")
    conversation = EvolutionConversation(tmp_path, "lineage", "codex")
    conversation.remember(previous)
    home = tmp_path / "current/scratch/agent-home"
    assert conversation.restore(tmp_path / "current", home) == "thread-1"


def test_unsafe_native_links_are_rejected(tmp_path: Path) -> None:
    previous = _previous(tmp_path)
    root = previous / "scratch/agent-home/.claude/projects/project"
    root.mkdir(parents=True)
    (tmp_path / "outside.jsonl").write_text("private\n")
    (root / "session-1.jsonl").symlink_to(tmp_path / "outside.jsonl")
    conversation = EvolutionConversation(tmp_path, "lineage", "claude")
    conversation.remember(previous)
    with pytest.raises(InfrastructureError, match="unsafe native state"):
        conversation.restore(tmp_path / "current", tmp_path / "current/scratch/agent-home")


def test_hard_killed_pi_recovers_its_native_identity(tmp_path: Path) -> None:
    previous = _previous(tmp_path, "pi", "")
    native = previous / "scratch/agent-home/.atrex-pi/evolver.jsonl"
    native.parent.mkdir(parents=True)
    native.write_text(json.dumps({"type": "session", "id": "pi-session"}) + "\n")
    conversation = EvolutionConversation(tmp_path, "lineage", "pi")
    conversation.remember(previous)
    assert (
        conversation.restore(
            tmp_path / "current",
            tmp_path / "current/scratch/agent-home",
        )
        == "pi-session"
    )
