"""Shared exact-commit Git import policy tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from atrex_runtime.git_import import SafeGitImporter


def test_fetch_commit_never_requests_partial_clone_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    importer = SafeGitImporter(
        Path("/usr/bin/git"),
        timeout_seconds=10,
        max_archive_bytes=1024,
        label="test",
    )
    calls: list[tuple[str, ...]] = []
    monkeypatch.setattr(importer, "run", lambda arguments: calls.append(arguments) or b"")

    importer.fetch_commit(tmp_path / "repository", "origin", "a" * 40)

    assert calls == [
        (
            "-C",
            str(tmp_path / "repository"),
            "fetch",
            "--depth=1",
            "--no-tags",
            "origin",
            "a" * 40,
        )
    ]
    assert all("--filter" not in argument for argument in calls[0])
