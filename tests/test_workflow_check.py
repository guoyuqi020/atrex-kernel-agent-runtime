from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from atrex_runtime.kernel_agents.workflow_check import WorkflowCheckError, check_agent_workflow

ROOT = Path(__file__).resolve().parents[1]


def _candidate(tmp_path: Path, main: str) -> Path:
    candidate = tmp_path / "candidate"
    workflow = candidate / "workflow"
    workflow.mkdir(parents=True)
    shutil.copy2(ROOT / "src/kernel-design-agents/workflow/runtime.py", workflow / "runtime.py")
    (workflow / "main.py").write_text(main, encoding="utf-8")
    (candidate / "src").mkdir()
    (candidate / "src/main.py").write_text("raise SystemExit(0)\n", encoding="utf-8")
    (candidate / "atrex-bundle.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "bundle_format": "atrex-kernel-agent-bundle-v1",
                "entrypoint": {"command": "src/main.py"},
                "workflow": {"command": "workflow/main.py"},
            }
        ),
        encoding="utf-8",
    )
    return candidate


def test_check_agent_workflow_exercises_first_later_and_no_change_paths(
    tmp_path: Path,
) -> None:
    main = (ROOT / "src/kernel-design-agents/workflow/main.py").read_text(encoding="utf-8")
    result = check_agent_workflow(
        _candidate(tmp_path, main),
        dsl="triton",
        epoch_number=4,
        max_challengers=1,
        optimizer_attempt_budget=6,
    )

    assert result["status"] == "valid"
    scenarios = result["scenarios"]
    assert isinstance(scenarios, list)
    assert [item["name"] for item in scenarios] == [
        "first-epoch",
        "later-epoch-evolved",
        "later-epoch-no-change",
    ]
    assert all(item["optimizer_attempts"] == 6 for item in scenarios)


@pytest.mark.parametrize(
    ("template", "max_challengers", "optimizer_attempt_budget"),
    (
        ("isolated.py", 0, 3),
        ("retained.py", 0, 3),
        ("pool_3.py", 0, 6),
        ("pool_retained_3.py", 0, 6),
        ("evolve_3.py", 1, 6),
        ("evolve_isolated_3.py", 1, 3),
        ("evolve_retained_3.py", 1, 3),
        ("evolve_isolated_pool_3.py", 1, 12),
    ),
)
def test_packaged_workflow_templates_pass_the_candidate_dry_run(
    tmp_path: Path,
    template: str,
    max_challengers: int,
    optimizer_attempt_budget: int,
) -> None:
    main = (ROOT / "src/atrex_runtime/workflow_templates" / template).read_text(
        encoding="utf-8"
    )

    result = check_agent_workflow(
        _candidate(tmp_path, main),
        dsl="triton",
        epoch_number=4,
        max_challengers=max_challengers,
        optimizer_attempt_budget=optimizer_attempt_budget,
    )

    assert result["status"] == "valid"


def test_check_agent_workflow_reports_the_failing_runtime_path(tmp_path: Path) -> None:
    candidate = _candidate(
        tmp_path,
        """#!/usr/bin/env python3
from runtime import EpochRuntime, serve

def run_epoch(epoch: EpochRuntime) -> None:
    challenger = (
        epoch.replicate_active(1)
        if int(epoch.context["epoch_number"]) == 1
        else epoch.evolve_agent(1)
    )
    pool = epoch.create_pool(
        branch="challenger-1",
        trajectories=1,
        rounds=int(epoch.limits["optimizer_attempts"]),
    )
    epoch.run_pools([pool])
    epoch.complete()

if __name__ == "__main__":
    raise SystemExit(serve(run_epoch))
""",
    )

    with pytest.raises(WorkflowCheckError) as caught:
        check_agent_workflow(
            candidate,
            dsl="triton",
            epoch_number=2,
            max_challengers=1,
            optimizer_attempt_budget=3,
        )

    assert caught.value.scenario == "later-epoch-no-change"
    assert caught.value.phase in {"process", "runtime_service"}
    assert "Challenger" in str(caught.value) or "challenger" in str(caught.value)
