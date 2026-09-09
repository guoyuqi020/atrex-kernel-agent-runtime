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


@pytest.mark.parametrize("relative", [
    "data/GDN/ablation.json", "examples/source-tree/ablation.example.json",
])
def test_source_tree_arms_match_single_file_exactly(relative: str) -> None:
    policy = json.loads((REPOSITORY / "scripts/production/policy.json").read_text())
    plan = json.loads((REPOSITORY / relative).read_text())
    assert plan == build_ablation_plan(policy)
    assert len(plan["arms"]) == 6
    assert sum(arm["optimizer_attempt_budget_total"] for arm in plan["arms"]) == 120
    campaign = CampaignSpecV3.from_file(REPOSITORY / "data/GDN/ablation-campaign.json")
    for key in ("attempts_per_trajectory", "trajectories_per_branch", "challenger_count",
                "challenger_start_epoch", "first_epoch_same_agent"):
        assert getattr(campaign, key) == policy["schedule"][key]
    old = CampaignSpecV3.from_file(REPOSITORY / "data/GDN/campaign.json")
    assert campaign.creation_key != old.creation_key
    assert campaign.lineages == old.lineages
    assert campaign.base_revision == old.base_revision


def launch_fixture(tmp_path, monkeypatch, *, fail=None):
    module = _module("scripts/source-tree/run.py")
    config = tmp_path / "runtime.json"
    config.write_text("{}")
    campaign_path = tmp_path / "campaign.json"
    campaign = json.loads((REPOSITORY / "data/GDN/ablation-campaign.json").read_text())
    campaign_path.write_text(json.dumps(campaign))
    plan_path = tmp_path / "plan.json"
    plan_path.write_text((REPOSITORY / "data/GDN/ablation.json").read_text())
    workspace = tmp_path / "run"
    monkeypatch.setattr(module.sys, "argv", [
        "run.py", "--workspace", str(workspace), "--config", str(config),
        "--campaign", str(campaign_path), "--plan", str(plan_path), "--target-epoch", "2",
    ])
    calls = []
    processes = []
    monkeypatch.setattr(module.shutil, "which", lambda _name: "/fake/runtime")

    def run(command, *, stdout, stderr, check):
        del stderr, check
        calls.append(command)
        if command[1] == "bootstrap":
            result = {"campaign_id": "campaign_" + "0" * 32,
                      "lineages": [{"lineage_id": "lineage_" + "0" * 32}]}
        else:
            assert command[1] == "seed-ablation-arm"
            seed = json.loads(Path(command[-1]).read_text())
            assert seed["source_lineage_id"] == "lineage_" + "0" * 32
            assert seed["challenger_count"] == 0
            if fail == "seed":
                raise subprocess.CalledProcessError(1, command)
            # Deterministic arm identity across resumptions.
            label = Path(command[-1]).parent.name
            labels = [arm["label"] for arm in json.loads(plan_path.read_text())["arms"]]
            suffix = f"{labels.index(label)+1:032x}"
            result = {"campaign_id": "campaign_" + suffix,
                      "lineage": {"lineage_id": "lineage_" + suffix}}
        stdout.write(json.dumps(result))
        stdout.flush()

    class Process:
        def __init__(self, command, *, stdout, stderr, start_new_session):
            assert start_new_session
            self.pid = len(processes) + 1
            self.campaign = command[command.index("--campaign") + 1]
            target = int(command[-1])
            assert command[1] == "run-campaign"
            assert target == (2 if self.campaign.endswith("0" * 32) else 5)
            if fail != "result":
                stdout.write(json.dumps({"campaign_id": self.campaign,
                                         "target_epoch_number": target}))
            stdout.flush()
            stderr.write("attempt finished\n")
            stderr.flush()
            processes.append(self)

        def poll(self):
            # Every arm must start before the first wait: no accidental serialization.
            assert len(processes) % 7 == 0
            return 1 if fail == "arm" and self.campaign.endswith("1".zfill(32)) else 0

    def run_json(cli, arguments, output, log):
        with output.open("w") as stdout, log.open("a") as stderr:
            run([cli, *arguments], stdout=stdout, stderr=stderr, check=True)
        return json.loads(output.read_text())

    monkeypatch.setattr(module, "run_json", run_json)
    monkeypatch.setattr(module.subprocess, "Popen", Process)
    return SimpleNamespace(module=module, calls=calls, processes=processes,
                           workspace=workspace, plan=plan_path, campaign=campaign_path)


def test_bootstrap_once_shared_seed_parallel_launch_and_resume(tmp_path, monkeypatch):
    test = launch_fixture(tmp_path, monkeypatch)
    test.module.main()
    summary = json.loads((test.workspace / "campaign-results.json").read_text())
    assert len(summary["arms"]) == 7
    assert all(arm["status"] == "completed" for arm in summary["arms"])
    assert len(test.calls) == 7  # one Bootstrap, six measurement-free seed operations
    assert len(test.processes) == 7
    assert len({arm["campaign_id"] for arm in summary["arms"]}) == 7
    for arm in summary["arms"]:
        assert Path(arm["result_path"]).is_file()
        assert "attempt finished" in Path(arm["log"]).read_text()
    test.module.main()
    resumed = json.loads((test.workspace / "campaign-results.json").read_text())
    assert resumed == summary
    assert len(test.processes) == 14


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
        assert statuses.count("failed") == (1 if fail == "arm" else 7)
        assert statuses.count("completed") == (6 if fail == "arm" else 0)


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
    module = _module("data/GDN/run.py")
    root = tmp_path / "data/GDN"
    root.mkdir(parents=True)
    (root / "runtime.json").write_text("{}")
    (root / "runtime-secrets.json").write_text("{}")
    monkeypatch.setattr(module, "__file__", str(root / "run.py"))
    monkeypatch.setattr(module.sys, "platform", "linux")
    monkeypatch.setattr(module.sys, "argv", ["run.py", "ablation"])

    def execute(_executable, arguments):
        assert arguments[1] == str(tmp_path / "scripts/source-tree/run.py")
        assert arguments[arguments.index("--campaign") + 1] == str(root / "ablation-campaign.json")
        assert arguments[arguments.index("--workspace") + 1] == str(root / "ablation")
        assert arguments[-2:] == ["--target-epoch", "5"]
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
