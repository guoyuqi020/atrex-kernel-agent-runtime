"""Source-tree and single-file arms share topology; launch tests never call models/GPUs."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_source_tree_example_schedule import REPOSITORY, _module

from atrex_runtime.ablation_plan import build_ablation_plan
from atrex_runtime.bootstrap import CampaignSpecV3


@pytest.mark.parametrize(
    "relative",
    [
        "data/GDN/ablation.json",
        "data/GDN-full/ablation.json",
        "examples/source-tree/ablation.example.json",
    ],
)
def test_source_tree_arms_keep_topology_with_one_hundred_epochs(relative: str) -> None:
    policy = json.loads((REPOSITORY / "scripts/production/policy.json").read_text())
    plan = json.loads((REPOSITORY / relative).read_text())
    assert plan == build_ablation_plan(policy, optimizer_attempt_budget_per_trajectory=300)
    single_file = build_ablation_plan(policy)
    assert single_file["optimizer_attempt_budget_per_trajectory"] == 15
    assert all(arm["target_epoch_number"] == 5 for arm in single_file["arms"])
    assert plan["main_evolve_enabled"] is False
    assert len(plan["arms"]) == 7
    assert all(arm["target_epoch_number"] == 100 for arm in plan["arms"])
    assert sum(arm["optimizer_attempt_budget_total"] for arm in plan["arms"]) == 2700
    campaign = CampaignSpecV3.from_file(REPOSITORY / "data/GDN/ablation-campaign.json")
    for key in ("max_challengers", "optimizer_attempt_budget"):
        assert getattr(campaign, key) == policy["schedule"][key]
    old = CampaignSpecV3.from_file(REPOSITORY / "data/GDN/campaign.json")
    assert campaign.creation_key != old.creation_key
    assert campaign.lineages == old.lineages
    assert campaign.base_revision == old.base_revision


def launch_fixture(tmp_path, monkeypatch, *, fail=None, target=2, control_epochs=100):
    module = _module("scripts/source-tree/run.py")
    config = tmp_path / "runtime.json"
    config.write_text("{}")
    campaign_path = tmp_path / "campaign.json"
    campaign = json.loads((REPOSITORY / "data/GDN/ablation-campaign.json").read_text())
    campaign_path.write_text(json.dumps(campaign))
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(
        json.dumps(
            build_ablation_plan(
                {"schedule": {**campaign, "event_only": True}},
                optimizer_attempt_budget_per_trajectory=control_epochs * 3,
            )
        )
    )
    workspace = tmp_path / "run"
    monkeypatch.setattr(
        module.sys,
        "argv",
        [
            "run.py",
            "--workspace",
            str(workspace),
            "--config",
            str(config),
            "--campaign",
            str(campaign_path),
            "--plan",
            str(plan_path),
            *([] if target is None else ["--target-epoch", str(target)]),
        ],
    )
    calls = []
    processes = []
    monkeypatch.setattr(module.shutil, "which", lambda _name: "/fake/runtime")

    def run(command, *, stdout, stderr, check):
        del stderr, check
        calls.append(command)
        if command[1] == "bootstrap":
            result = {
                "campaign_id": "campaign_" + "0" * 32,
                "lineages": [{"lineage_id": "lineage_" + "0" * 32}],
            }
        else:
            assert command[1] == "seed-ablation-arm"
            seed = json.loads(Path(command[-1]).read_text())
            assert seed["source_lineage_id"] == "lineage_" + "0" * 32
            assert seed["workflow_command"].startswith("workflow/")
            label = Path(command[-1]).parent.name
            planned_arms = json.loads(plan_path.read_text())["arms"]
            planned = next(arm for arm in planned_arms if arm["label"] == label)
            for key in ("max_challengers", "optimizer_attempt_budget", "workflow_command"):
                assert seed[key] == planned[key]
            if planned.get("observer_label") is None:
                assert seed["evolver_observer_lineage_id"] is None
            else:
                observer = json.loads(
                    (
                        Path(command[-1]).parent.parent
                        / planned["observer_label"]
                        / "seed-result.json"
                    ).read_text()
                )
                assert (
                    seed["evolver_observer_lineage_id"]
                    == observer["lineage"]["lineage_id"]
                )
            if fail == "seed":
                raise subprocess.CalledProcessError(1, command)
            # Deterministic arm identity across resumptions.
            labels = [arm["label"] for arm in planned_arms]
            campaign_suffix = f"{labels.index(label) + 1:032x}"
            lineage_suffix = f"{labels.index(label) + 1:032x}"
            result = {
                "campaign_id": "campaign_" + campaign_suffix,
                "lineage": {"lineage_id": "lineage_" + lineage_suffix},
            }
        stdout.write(json.dumps(result))
        stdout.flush()

    class Process:
        def __init__(self, command, *, stdout, stderr, start_new_session):
            assert start_new_session
            self.pid = len(processes) + 1
            self.campaign = command[command.index("--campaign") + 1]
            requested_target = int(command[-1])
            assert command[1] == "run-campaign"
            assert requested_target == (
                (100 if target is None else target)
                if self.campaign.endswith("0" * 32)
                else control_epochs
            )
            if fail != "result":
                stdout.write(
                    json.dumps(
                        {"campaign_id": self.campaign, "target_epoch_number": requested_target}
                    )
                )
            stdout.flush()
            stderr.write("attempt finished\n")
            stderr.flush()
            processes.append(self)

        def poll(self):
            # Every arm must start before the first wait: no accidental serialization.
            arm_count = len(json.loads(plan_path.read_text())["arms"])
            assert len(processes) % arm_count == 0
            return 1 if fail == "arm" and self.campaign.endswith("1".zfill(32)) else 0

    def run_json(cli, arguments, output, log):
        with output.open("w") as stdout, log.open("a") as stderr:
            run([cli, *arguments], stdout=stdout, stderr=stderr, check=True)
        return json.loads(output.read_text())

    monkeypatch.setattr(module, "run_json", run_json)
    monkeypatch.setattr(module.subprocess, "Popen", Process)
    return SimpleNamespace(
        module=module,
        calls=calls,
        processes=processes,
        workspace=workspace,
        plan=plan_path,
        campaign=campaign_path,
    )


def test_bootstrap_once_shared_seed_parallel_launch_and_resume(tmp_path, monkeypatch):
    test = launch_fixture(tmp_path, monkeypatch)
    test.module.main()
    summary = json.loads((test.workspace / "campaign-results.json").read_text())
    assert len(summary["arms"]) == 7
    assert all(arm["status"] == "completed" for arm in summary["arms"])
    assert len(test.calls) == 8  # one Bootstrap, seven measurement-free seed operations
    assert len(test.processes) == 7
    assert len({arm["campaign_id"] for arm in summary["arms"]}) == 7
    for arm in summary["arms"]:
        assert Path(arm["result_path"]).is_file()
        assert "attempt finished" in Path(arm["log"]).read_text()
    test.module.main()
    resumed = json.loads((test.workspace / "campaign-results.json").read_text())
    assert resumed == summary
    assert len(test.processes) == 14


def test_source_tree_runner_defaults_all_arms_to_one_hundred_epochs(tmp_path, monkeypatch):
    test = launch_fixture(tmp_path, monkeypatch, target=None)
    test.module.main()
    arms = json.loads((test.workspace / "campaign-results.json").read_text())["arms"]
    assert all(arm["target_epoch_number"] == 100 for arm in arms)
    assert sum(arm["optimizer_attempt_budget_total"] for arm in arms) == 2700


def test_existing_five_epoch_plan_is_not_rewritten(tmp_path, monkeypatch):
    test = launch_fixture(tmp_path, monkeypatch, target=5, control_epochs=5)
    test.module.main()
    original = (test.workspace / "launch-inputs.json").read_bytes()
    test.module.main()
    assert (test.workspace / "launch-inputs.json").read_bytes() == original
    arms = json.loads((test.workspace / "campaign-results.json").read_text())["arms"]
    assert all(arm["target_epoch_number"] == 5 for arm in arms)


@pytest.mark.parametrize("budget", [None, True, 0, -3, 1.5, "300"])
def test_ablation_budget_must_be_a_positive_integer(budget):
    policy = json.loads((REPOSITORY / "scripts/production/policy.json").read_text())
    with pytest.raises(ValueError, match="positive integer"):
        build_ablation_plan(policy, optimizer_attempt_budget_per_trajectory=budget)


@pytest.mark.parametrize("fail", ["seed", "arm", "result"])
def test_failure_is_reported_without_silently_losing_other_arms(tmp_path, monkeypatch, fail):
    test = launch_fixture(tmp_path, monkeypatch, fail=fail)
    with pytest.raises((SystemExit, subprocess.CalledProcessError)):
        test.module.main()
    if fail == "seed":
        assert not test.processes
    else:
        summary = json.loads((test.workspace / "campaign-results.json").read_text())
        statuses = [arm["status"] for arm in summary["arms"]]
        arm_count = len(json.loads(test.plan.read_text())["arms"])
        assert statuses.count("failed") == (1 if fail == "arm" else arm_count)
        assert statuses.count("completed") == (arm_count - 1 if fail == "arm" else 0)


def test_changed_plan_or_frozen_campaign_rejected_before_execution(tmp_path, monkeypatch):
    test = launch_fixture(tmp_path, monkeypatch)
    test.module.main()
    count = len(test.calls)
    value = json.loads(test.campaign.read_text())
    value["base_revision"]["commit"] = "b" * 40
    test.campaign.write_text(json.dumps(value))
    with pytest.raises(SystemExit):
        test.module.main()
    assert len(test.calls) == count
    test.plan.write_text('{"schema_version": 3}')
    with pytest.raises(SystemExit):
        test.module.main()
    assert len(test.calls) == count


def test_gdn_ablation_role_uses_separate_definition_and_output(tmp_path, monkeypatch):
    module = _module("scripts/gdn/run.py")
    root = tmp_path / "workspaces/GDN"
    root.mkdir(parents=True)
    (root / "runtime.json").write_text("{}")
    (root / "runtime-secrets.json").write_text(
        json.dumps(
            {
                "ATREX_CAPABILITY_SIGNING_KEY": "test-signing",
                "ATREX_ADMIN_BEARER_TOKEN": "test-admin",
            }
        )
    )
    monkeypatch.setattr(module, "__file__", str(tmp_path / "scripts/gdn/run.py"))
    monkeypatch.setattr(module.sys, "platform", "linux")
    monkeypatch.setattr(module.sys, "argv", ["run.py", "ablation"])

    def execute(_executable, arguments):
        assert arguments[1] == str(tmp_path / "scripts/source-tree/run.py")
        assert arguments[arguments.index("--campaign") + 1] == str(root / "ablation-campaign.json")
        assert arguments[arguments.index("--workspace") + 1] == str(root / "ablation")
        assert arguments[-2:] == ["--target-epoch", "100"]
        raise SystemExit(0)

    monkeypatch.setattr(module.os, "execv", execute)
    with pytest.raises(SystemExit) as stopped:
        module.main()
    assert stopped.value.code == 0


def test_interrupted_bootstrap_cleans_its_own_process_group(tmp_path, monkeypatch):
    module = _module("scripts/source-tree/run.py")
    signals = []

    class Process:
        pid = 43210
        stopped = False

        def wait(self, timeout=None):
            if timeout is None:
                raise KeyboardInterrupt
            self.stopped = True
            return -15

        def poll(self):
            return -15 if self.stopped else None

    process = Process()
    monkeypatch.setattr(module.subprocess, "Popen", lambda *_args, **_kwargs: process)
    monkeypatch.setattr(module.os, "killpg", lambda pid, sig: signals.append((pid, sig)))
    with pytest.raises(KeyboardInterrupt):
        module.run_json("runtime", ["bootstrap"], tmp_path / "result.json", tmp_path / "log")
    assert signals == [(43210, module.signal.SIGTERM)]
    assert not (tmp_path / "result.json").exists()
