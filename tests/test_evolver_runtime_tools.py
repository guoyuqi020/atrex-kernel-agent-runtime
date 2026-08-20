"""End-to-end tests for Runtime-owned Evolver inspection and Candidate tools."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from atrex_runtime.workers import evolver_tools

ACTIVE = "agentrev_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
CHALLENGER = "agentrev_bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
START = "kernelrev_11111111111111111111111111111111"
BEST = "kernelrev_22222222222222222222222222222222"


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _workspace(root: Path) -> Path:
    workspace = root / "run"
    tools = workspace / "runtime-tools"
    tools.mkdir(parents=True)
    shutil.copyfile(Path(evolver_tools.__file__), tools / "evolver_tools.py")
    for revision, prompt in ((ACTIVE, "broad\n"), (CHALLENGER, "targeted\n")):
        repository = workspace / "input/agents" / revision
        (repository / "prompts").mkdir(parents=True)
        (repository / "prompts/evolve.md").write_text(prompt)
    candidate = workspace / "candidate"
    (candidate / "prompts").mkdir(parents=True)
    (candidate / "prompts/evolve.md").write_text("targeted\n")
    (candidate / "stale.txt").write_text("remove me\n")
    (workspace / "scratch").mkdir()
    for revision, source in ((START, "# start\n"), (BEST, "# best\n")):
        kernel = tools / "kernels" / revision
        kernel.mkdir(parents=True)
        (kernel / "kernel.py").write_text(source)
    catalog = {
        "schema_version": 1,
        "evidence_checkpoint": "sha256:" + "c" * 64,
        "agents": [
            {
                "revision_id": ACTIVE,
                "version": "agent-v0",
                "active": False,
                "repository_path": f"input/agents/{ACTIVE}",
            },
            {
                "revision_id": CHALLENGER,
                "version": "agent-v1",
                "active": True,
                "repository_path": f"input/agents/{CHALLENGER}",
            },
        ],
        "kernels": [
            {
                "revision_id": START,
                "version": "v0",
                "epoch_number": None,
                "latency_us": 12.0,
                "sol_percent": 40.0,
                "artifact_path": f"runtime-tools/kernels/{START}",
            },
            {
                "revision_id": BEST,
                "version": "v1",
                "epoch_number": 1,
                "latency_us": 9.0,
                "sol_percent": 55.0,
                "artifact_path": f"runtime-tools/kernels/{BEST}",
            },
        ],
    }
    _write(tools / "catalog.json", catalog)
    _write(
        workspace / "evolution-input.json",
        {
            "schema_version": 4,
            "parent_revision_id": CHALLENGER,
            "visible_agents": [
                {
                    "revision_id": ACTIVE,
                    "path": f"input/agents/{ACTIVE}",
                    "relationship": "lineage_history",
                },
                {
                    "revision_id": CHALLENGER,
                    "path": f"input/agents/{CHALLENGER}",
                    "relationship": "active",
                },
            ],
        },
    )
    evidence = workspace / "input/evidence"
    _write(
        evidence / "manifest.json",
        {
            "schema_version": 1,
            "role": "evolver",
            "lineage_checkpoint": catalog["evidence_checkpoint"],
            "prompt_fragment_sha256": "d" * 64,
            "through_completed_epoch": 1,
            "current_epoch": None,
            "visibility": {
                "completed_epochs": "all_completed_branches",
                "current_attempts_before": None,
            },
        },
    )
    epoch = evidence / "epochs/00000001"
    _write(
        epoch / "summary.json",
        {
            "schema_version": 1,
            "epoch_id": "epoch_" + "e" * 32,
            "number": 1,
            "active_kernel_agent_revision_id": ACTIVE,
            "challenger_kernel_agent_revision_ids": [CHALLENGER],
            "winner_kernel_agent_revision_id": CHALLENGER,
            "starting_kernel_revision_id": START,
            "best_kernel_revision_id": BEST,
            "branches": [
                {
                    "branch": "active",
                    "challenger_ordinal": 0,
                    "kernel_agent_revision_id": ACTIVE,
                    "selected": False,
                },
                {
                    "branch": "challenger",
                    "challenger_ordinal": 1,
                    "kernel_agent_revision_id": CHALLENGER,
                    "selected": True,
                },
            ],
        },
    )
    _write(
        epoch / "branches/challenger-0001/trajectories/00000001/attempts/00000001/summary.json",
        {
            "attempt_id": "attempt_" + "f" * 32,
            "branch": "challenger",
            "challenger_ordinal": 1,
            "trajectory_ordinal": 1,
            "ordinal": 1,
            "kernel_agent_revision_id": CHALLENGER,
            "input_kernel_revision_id": START,
            "accepted_as_branch_best": True,
            "output": {
                "kernel_revision_id": BEST,
                "correct": True,
                "latency_us": 9.0,
            },
        },
    )
    active_attempts = epoch / "branches/active/trajectories/00000001/attempts"
    active_attempts.mkdir(parents=True)
    trace = (
        epoch / "branches/challenger-0001/trajectories/00000001/attempts/00000001/traces/run-0001"
    )
    trace.mkdir(parents=True)
    (trace / "session.json").write_text("{}")
    return workspace


def _run(workspace: Path, *arguments: str) -> dict[str, object]:
    result = subprocess.run(
        [sys.executable, str(workspace / "runtime-tools/evolver_tools.py"), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    value = json.loads(result.stdout)
    assert isinstance(value, dict)
    return value


def test_evolver_runtime_tools_query_only_the_frozen_workspace(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)

    history = _run(workspace, "history")
    epoch = history["epochs"][0]  # type: ignore[index]
    assert epoch["winner"] == {"revision_id": CHALLENGER, "version": "agent-v1"}
    assert epoch["best_kernel"] == {"revision_id": BEST, "version": "v1"}

    branches = _run(workspace, "branches", "--epoch", "1")
    challenger = branches["branches"][1]  # type: ignore[index]
    assert challenger["selected"] is True
    assert challenger["best_kernel"]["latency_us"] == 9.0
    assert challenger["best_kernel"]["sol_percent"] == 55.0

    attempts = _run(
        workspace,
        "attempts",
        "--epoch",
        "1",
        "--branch",
        "challenger-0001",
    )
    assert attempts["attempts"][0]["output"]["version"] == "v1"  # type: ignore[index]

    kernels = _run(workspace, "kernels")
    assert [item["version"] for item in kernels["kernels"]] == ["v0", "v1"]  # type: ignore[union-attr]
    assert [item["sol_percent"] for item in kernels["kernels"]] == [40.0, 55.0]  # type: ignore[union-attr]
    source = _run(
        workspace,
        "kernel-read",
        "--revision",
        BEST,
        "--file",
        "kernel.py",
    )
    assert source["content"] == "# best\n"

    agents = _run(workspace, "agents")
    assert [item["version"] for item in agents["agents"]] == [  # type: ignore[union-attr]
        "agent-v0",
        "agent-v1",
    ]
    difference = _run(
        workspace,
        "agent-diff",
        "--base",
        ACTIVE,
        "--candidate",
        CHALLENGER,
    )
    assert difference["changes"][0]["path"] == "prompts/evolve.md"  # type: ignore[index]
    assert "targeted" in difference["changes"][0]["diff"]  # type: ignore[index]

    traces = _run(workspace, "trace-paths", "--epoch", "1")
    assert traces["trace_paths"] == [
        "input/evidence/epochs/00000001/branches/challenger-0001/trajectories/"
        "00000001/attempts/00000001/traces/run-0001"
    ]


def test_candidate_reset_atomically_loads_only_completed_lineage_history(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)

    reset = _run(workspace, "candidate-reset", "--base", ACTIVE)

    assert reset == {
        "schema_version": 1,
        "candidate_base_revision_id": ACTIVE,
        "candidate_repository": "candidate",
    }
    assert (workspace / "candidate/prompts/evolve.md").read_text() == "broad\n"
    assert not (workspace / "candidate/stale.txt").exists()
    assert os.stat(workspace / "candidate/prompts/evolve.md").st_mode & 0o200
    assert json.loads((workspace / "scratch/candidate-base.json").read_text()) == {
        "schema_version": 1,
        "base_revision_id": ACTIVE,
        "selection": "candidate_reset",
    }


def test_candidate_reset_rejects_the_current_active_agent(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)

    result = subprocess.run(
        [
            sys.executable,
            str(workspace / "runtime-tools/evolver_tools.py"),
            "candidate-reset",
            "--base",
            CHALLENGER,
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "completed Lineage history" in result.stderr
    assert (workspace / "candidate/prompts/evolve.md").read_text() == "targeted\n"
