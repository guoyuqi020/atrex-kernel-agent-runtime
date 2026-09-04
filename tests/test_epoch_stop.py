"""Quiescent Epoch stops, phase recovery and workspace scoping."""

from __future__ import annotations

import json
import runpy
from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest
from conftest import NOW, seed_lineage

from atrex_runtime.domain.ids import new_epoch_id
from atrex_runtime.domain.models import Dsl, Epoch, EpochStatus
from atrex_runtime.registry.sqlite import SqliteRegistry

ROOT = Path(__file__).resolve().parents[1]
stop_workspace_epochs = cast(
    Callable[[Path], list[str]],
    runpy.run_path(str(ROOT / "scripts/production/stop_epochs.py"))["stop_workspace_epochs"],
)


def create_epoch(
    registry: SqliteRegistry, status: EpochStatus, dsl: Dsl = Dsl.TRITON,
) -> Epoch:
    seed = seed_lineage(
        registry, dsl=dsl, challenger_count=int(status is EpochStatus.BUILDING_CHALLENGER),
    )
    lineage = registry.get_lineage(seed.lineage_id)
    epoch = Epoch(
        id=new_epoch_id(), lineage_id=lineage.id, number=1,
        active_kernel_agent_revision_id=lineage.active_kernel_agent_revision_id,
        challenger_kernel_agent_revision_ids=(), starting_kernel_revision_id=seed.baseline.id,
        evidence_checkpoint=lineage.evidence_checkpoint, challenger_count=lineage.challenger_count,
        trajectories_per_branch=lineage.trajectories_per_branch,
        attempts_per_trajectory=lineage.attempts_per_trajectory, status=status,
        winner_kernel_agent_revision_id=None, best_kernel_revision_id=None,
        created_at=NOW, completed_at=None,
    )
    registry.insert_epoch(epoch)
    return epoch


@pytest.mark.parametrize("phase", [
    EpochStatus.BUILDING_CHALLENGER, EpochStatus.READY, EpochStatus.RUNNING, EpochStatus.SELECTING,
])
def test_stop_restores_exact_phase_across_database_reopen(
    tmp_path: Path, phase: EpochStatus,
) -> None:
    path = tmp_path / "registry.sqlite"
    with SqliteRegistry(path) as registry:
        epoch = create_epoch(registry, phase)
        assert registry.stop_epoch(epoch.id, "operator stopped").status is EpochStatus.STOPPED
    with SqliteRegistry(path) as registry:
        assert registry.get_epoch(epoch.id).status is EpochStatus.STOPPED
        assert registry.resume_stopped_epoch(epoch.id) == epoch
        assert registry.resume_stopped_epoch(epoch.id) == epoch
        registry.stop_epoch(epoch.id, "stopped again")
        assert registry.resume_stopped_epoch(epoch.id) == epoch


def test_stop_workspace_targets_only_its_bootstrap_and_arm_results(tmp_path: Path) -> None:
    database = tmp_path / "shared-registry.sqlite"
    workspace = tmp_path / "task"
    workspace.mkdir()
    config = json.loads((ROOT / "runtime.example.json").read_text())
    config["storage"]["registry_database"] = str(database)
    (workspace / "runtime.json").write_text(json.dumps(config))
    with SqliteRegistry(database) as registry:
        main = create_epoch(registry, EpochStatus.RUNNING)
        arm = create_epoch(registry, EpochStatus.SELECTING, Dsl.CUDA)
        other = create_epoch(registry, EpochStatus.RUNNING, Dsl.CUTEDSL)
        dsl = workspace / "dsls/triton"
        dsl.mkdir(parents=True)
        (dsl / "bootstrap-result.json").write_text(json.dumps({
            "campaign_id": registry.get_lineage(main.lineage_id).campaign_id,
            "lineages": [{"lineage_id": main.lineage_id}],
        }))
        (dsl / "ablation-retained").mkdir()
        (dsl / "ablation-retained/seed-result.json").write_text(json.dumps({
            "campaign_id": registry.get_lineage(arm.lineage_id).campaign_id,
            "lineage": {"lineage_id": arm.lineage_id},
        }))
    assert set(stop_workspace_epochs(workspace)) == {main.id, arm.id}
    assert set(stop_workspace_epochs(workspace)) == {main.id, arm.id}
    with SqliteRegistry(database) as registry:
        assert registry.get_epoch(other.id) == other
        assert registry.get_epoch(main.id).status is EpochStatus.STOPPED
        assert registry.get_epoch(arm.id).status is EpochStatus.STOPPED


def test_stop_script_reconciles_only_after_process_checks() -> None:
    script = (ROOT / "scripts/production/campaign.sh").read_text()
    body = script.split("stop_campaign() {", 1)[1].split("start_prepared()", 1)[0]
    assert body.count('"${script_dir}/stop_epochs.py"') == 2
    assert body.index('"${script_dir}/stop_epochs.py"') > body.index('${#pids[@]} == 0')
    assert body.rindex('"${script_dir}/stop_epochs.py"') > body.index('${survivors}')
    assert body.rindex('"${script_dir}/stop_epochs.py"') < body.rindex('write_state "stopped"')
