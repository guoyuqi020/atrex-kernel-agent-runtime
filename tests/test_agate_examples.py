"""Static tests for examples that target a real remote Agate service."""

from __future__ import annotations

import ast
import json
import os
import subprocess
from pathlib import Path

from atrex_gateway_client import build_eval_request

REPOSITORY = Path(__file__).resolve().parents[1]
EXAMPLE = REPOSITORY / "examples/agate"
VECADD = REPOSITORY / "examples/shared/vecadd"


def test_all_agate_shell_wrappers_parse() -> None:
    scripts = sorted(EXAMPLE.glob("*.sh"))
    assert scripts
    subprocess.run(("bash", "-n", *(str(path) for path in scripts)), check=True)


def test_agate_example_requires_an_explicit_real_service_url() -> None:
    environment = dict(os.environ)
    environment.pop("AGATE_URL", None)
    result = subprocess.run(
        (
            "bash",
            "-c",
            "source examples/agate/common.sh; agate_require_service",
        ),
        cwd=REPOSITORY,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 64
    assert "real Agate service" in result.stderr


def test_agate_example_resolves_the_cli_from_path(tmp_path: Path) -> None:
    executable = tmp_path / "agate"
    executable.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    executable.chmod(0o700)
    environment = {
        **os.environ,
        "PATH": f"{tmp_path}{os.pathsep}{os.environ['PATH']}",
        "AGATE_URL": "https://agate.test",
    }

    result = subprocess.run(
        (
            "/bin/bash",
            "-c",
            "source examples/agate/common.sh; agate_require_service; printf '%s' \"$agate_bin\"",
        ),
        cwd=REPOSITORY,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout == "agate"
    assert ".venv/bin/agate" not in (EXAMPLE / "common.sh").read_text(encoding="utf-8")


def test_agate_example_sources_and_shapes_are_well_formed() -> None:
    for path in (
        VECADD / "triton/agate-candidate/kernel.py",
        VECADD / "reference/reference.py",
        VECADD / "reference/input.py",
    ):
        compile(path.read_text(encoding="utf-8"), str(path), "exec")

    input_source = (VECADD / "reference/input.py").read_text(encoding="utf-8")
    input_tree = ast.parse(input_source)
    functions = {node.name for node in input_tree.body if isinstance(node, ast.FunctionDef)}
    assert "_make_inputs" in functions

    shapes = json.loads((VECADD / "reference/shapes.json").read_text(encoding="utf-8"))
    assert shapes == {
        "0": {
            "init_kwargs": None,
            "input_kwargs": {"num_elements": 1_048_576},
        }
    }


def test_official_agate_client_builds_the_vecadd_request_without_network() -> None:
    request = build_eval_request(
        str(VECADD / "triton/agate-candidate/kernel.py"),
        str(VECADD / "reference"),
        "not-submitted",
        operator="vector_add",
    )

    reference = request["reference"]
    assert reference["operator"] == "vector_add"
    assert "def _make_inputs(" in reference["input_py"]
    assert reference["shapes"]["0"]["input_kwargs"] == {"num_elements": 1_048_576}
    assert "getattr" not in reference["input_py"]


def test_agate_example_never_starts_a_local_gateway() -> None:
    contents = "\n".join(path.read_text(encoding="utf-8") for path in EXAMPLE.glob("*.sh"))
    assert "uvicorn" not in contents
    assert "127.0.0.1" not in contents
    assert "localhost" not in contents
    assert "AGATE_URL must name the real Agate service" in contents


def test_evaluate_and_profile_use_extended_overridable_timeouts() -> None:
    for script_name in ("evaluate.sh", "profile.sh"):
        source = (EXAMPLE / script_name).read_text(encoding="utf-8")
        assert "AGATE_HTTP_TIMEOUT:-1800" in source
        assert "AGATE_JOB_TIMEOUT:-3600" in source
        assert "AGATE_WAIT_TIMEOUT:-3900" in source
        assert "AGATE_POLL_SECONDS:-5" in source

    asynchronous = (EXAMPLE / "evaluate-async.sh").read_text(encoding="utf-8")
    assert "AGATE_HTTP_TIMEOUT:-1800" in asynchronous
    assert "AGATE_JOB_TIMEOUT:-3600" in asynchronous
    assert "AGATE_WAIT_TIMEOUT" not in asynchronous
