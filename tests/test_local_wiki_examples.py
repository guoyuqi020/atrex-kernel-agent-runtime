"""Executable Local Wiki example wrapper tests."""

from __future__ import annotations

import json
import os
import stat
import subprocess
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
EXAMPLE = REPOSITORY / "examples/local-wiki"
LOCAL_WIKI = REPOSITORY / "workspaces/local-wiki"


def _bash(command: str, *, environment: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ("bash", "-c", command),
        cwd=REPOSITORY,
        env={**os.environ, **environment},
        check=True,
        capture_output=True,
        text=True,
    )


def test_demo_environment_is_private_stable_and_not_printed(tmp_path: Path) -> None:
    demo_env = tmp_path / "state/demo.env"
    environment = {"ATREX_DEMO_ENV_FILE": str(demo_env)}

    first = _bash("bash examples/local-wiki/prepare-demo-env.sh", environment=environment)
    initial = demo_env.read_bytes()
    second = _bash("bash examples/local-wiki/prepare-demo-env.sh", environment=environment)

    assert initial == demo_env.read_bytes()
    assert stat.S_IMODE(demo_env.stat().st_mode) == 0o600
    assert b"ATREX_CAPABILITY_SIGNING_KEY" in initial
    assert b"ATREX_ADMIN_BEARER_TOKEN" in initial
    assert b"ANTHROPIC" not in initial
    for line in initial.decode().splitlines():
        _name, secret = line.split("=", 1)
        assert secret.strip("'") not in first.stdout
        assert secret.strip("'") not in second.stdout


def test_demo_accepts_the_default_qoder_credential() -> None:
    result = _bash(
        "source examples/local-wiki/demo-common.sh; "
        "atrex_demo_require_optimizer_backend >/dev/null; "
        "printf 'qoder-ready'",
        environment={"QODER_PERSONAL_ACCESS_TOKEN": "test-qoder-token"},
    )

    assert result.stdout == "qoder-ready"


def test_demo_resolves_the_first_saved_bootstrap_lineage(tmp_path: Path) -> None:
    result_file = tmp_path / "last-bootstrap.json"
    lineage_id = "lineage_0123456789abcdef0123456789abcdef"
    result_file.write_text(
        json.dumps({"lineages": [{"lineage_id": lineage_id}]}),
        encoding="utf-8",
    )

    result = _bash(
        "source examples/local-wiki/demo-common.sh; atrex_demo_last_lineage_id",
        environment={"ATREX_DEMO_BOOTSTRAP_RESULT_FILE": str(result_file)},
    )

    assert result.stdout.strip() == lineage_id


def test_temporary_wiki_shell_entrypoint_loads_without_starting_services() -> None:
    result = subprocess.run(
        (
            str(REPOSITORY / ".venv/bin/python"),
            str(EXAMPLE / "temporary_wiki_shell.py"),
            "--help",
        ),
        cwd=REPOSITORY,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "disposable Core-compatible shell" in result.stdout


def test_all_local_wiki_shell_wrappers_parse() -> None:
    scripts = sorted(EXAMPLE.glob("*.sh"))
    assert scripts
    subprocess.run(("bash", "-n", *(str(path) for path in scripts)), check=True)


def test_local_wiki_uses_the_vendored_pinned_upstream_corpus() -> None:
    config = json.loads((LOCAL_WIKI / "configs/local.example.json").read_text())
    lock = json.loads((LOCAL_WIKI / "reference.lock.json").read_text())
    corpus = LOCAL_WIKI / lock["vendored_path"]

    assert config["reference_root"] == "../corpus/gpu-wiki"
    assert lock == {
        "schema_version": 1,
        "repository": "git@github.com:alibaba/atrex-kernel-agent.git",
        "commit": "71b16928579474c93039053d2facfeaf7134e268",
        "git_tree": "f83b706953a6e62d08863db0f349995f3c7f0081",
        "vendored_path": "corpus/gpu-wiki",
        "sparse_paths": ["gpu-wiki"],
    }
    assert (corpus / "tools/query_nl.py").is_file()
    assert (corpus / "tools/ingest_feedback.py").is_file()
    assert (corpus / "tools/rebuild_importance.py").is_file()
    assert (corpus / "kernel_wiki").is_dir()
    assert (corpus / "hardware_wiki").is_dir()
    assert (LOCAL_WIKI / "corpus/atrex-kernel-agent.LICENSE").is_file()
    assert (LOCAL_WIKI / "corpus/atrex-kernel-agent.NOTICE").is_file()
