"""Pure checks for the opt-in live Backend acceptance runner."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from atrex_runtime.acceptance import backend_connectivity as connectivity


@pytest.mark.parametrize("backend", ("claude", "codex", "qodercli"))
def test_core_registry_builds_the_backend_command(backend: connectivity.BackendName) -> None:
    core = Path(__file__).resolve().parents[1] / "src/atrex-kernel-agent-core"
    adapter = connectivity._load_registry(core).create(backend)

    command = adapter.build_command("ping", "00000000-0000-4000-8000-000000000001", "low", "")

    assert command[0] == backend
    assert "ping" in command


def test_user_local_executable_mounts_only_its_installation_root(tmp_path: Path) -> None:
    home = tmp_path / "home"
    executable = home / ".local/bin/codex"
    executable.parent.mkdir(parents=True)
    executable.write_text("#!/bin/sh\n")

    assert connectivity._executable_bind_paths(executable, home) == (home / ".local",)
    assert connectivity._executable_bind_paths(Path("/usr/bin/codex"), home) == ()


def test_codex_uses_writable_session_home_with_read_only_login_links(tmp_path: Path) -> None:
    source = tmp_path / "source"
    workspace = tmp_path / "workspace"
    source.mkdir()
    (source / "auth.json").write_text("{}")
    (workspace / "scratch").mkdir(parents=True)
    environment = {"CODEX_HOME": str(source)}

    connectivity._prepare_backend_home("codex", workspace, environment)

    session_home = workspace / "scratch/codex-home"
    assert environment["CODEX_HOME"] == str(session_home)
    assert (session_home / "auth.json").is_symlink()
    assert (session_home / "auth.json").resolve() == source / "auth.json"


def test_explicit_certificate_file_is_mounted_without_secret_discovery(tmp_path: Path) -> None:
    certificate = tmp_path / "corporate.pem"
    certificate.write_text("public certificate")

    assert connectivity._certificate_bind_paths(
        {"SSL_CERT_FILE": str(certificate), "ANTHROPIC_API_KEY": "secret"}
    ) == (certificate,)


def test_codex_write_probe_covers_session_home() -> None:
    command = connectivity._write_probe_command("codex")

    assert command[:2] == ("/bin/sh", "-c")
    assert '"$HOME"' in command[2]
    assert '"$CODEX_HOME"' in command[2]


def test_filesystem_failure_summary_keeps_only_failed_paths(tmp_path: Path) -> None:
    trace = tmp_path / "trace"
    trace.write_text(
        '1 openat("ok", O_RDONLY) = 3\n'
        '1 openat("readonly", O_WRONLY) = -1 EROFS (Read-only file system)\n'
    )

    summary = connectivity._filesystem_failure_summary(trace)

    assert "readonly" in summary
    assert '"ok"' not in summary


@pytest.mark.parametrize(
    ("backend", "event"),
    (
        ("claude", {"type": "result", "subtype": "success"}),
        ("qodercli", {"type": "result", "is_error": False}),
        ("codex", {"type": "turn.completed", "usage": {"output_tokens": 1}}),
    ),
)
def test_terminal_success_requires_backend_terminal_event(
    backend: connectivity.BackendName,
    event: dict[str, object],
) -> None:
    assert connectivity._terminal_success(backend, json.dumps(event))
    assert not connectivity._terminal_success(backend, json.dumps({"type": "progress"}))


def test_main_treats_skips_as_optional_unless_required(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        connectivity,
        "run_acceptance",
        lambda options: (
            connectivity.ConnectivityResult(
                backend="qodercli",
                status="skipped",
                detail="not installed",
            ),
        ),
    )

    assert connectivity.main(["--backend", "qodercli", "--json"]) == 0
    assert connectivity.main(["--backend", "qodercli", "--require-all", "--json"]) == 1


def test_duplicate_scalar_override_is_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate model"):
        connectivity._assignment_map(
            ("codex=gpt-a", "codex=gpt-b"),
            label="model",
        )
