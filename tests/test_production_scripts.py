from __future__ import annotations

import hashlib
import json
import os
import runpy
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

import pytest

from atrex_runtime.bootstrap import CampaignSpecV3
from atrex_runtime.config import RuntimeSettings

ROOT = Path(__file__).resolve().parents[1]
PRODUCTION = ROOT / "scripts/production"
_prepare = runpy.run_path(str(PRODUCTION / "prepare.py"))
_git_commit = cast(Any, _prepare["_git_commit"])
_ablation_plan = cast(Any, _prepare["_ablation_plan"])


def _operator(root: Path) -> Path:
    root.mkdir()
    root.joinpath("reference.py").write_text(
        "import torch\nclass Model(torch.nn.Module):\n    def forward(self, x): return x + 1\n",
        encoding="utf-8",
    )
    root.joinpath("input.py").write_text(
        "import torch\ndef _make_inputs(n):\n    return {'x': torch.randn((n,), device='cuda')}\n",
        encoding="utf-8",
    )
    root.joinpath("shapes.json").write_text(
        json.dumps({"0": {"init_kwargs": None, "input_kwargs": {"n": 1024}}}),
        encoding="utf-8",
    )
    root.joinpath("metadata.json").write_text(json.dumps({"id": "test_add"}), encoding="utf-8")
    return root


def _shape_train_operator(root: Path) -> Path:
    operator = _operator(root)
    operator.joinpath("shapes.json").replace(operator / "shape_valid.json")
    operator.joinpath("shape_train.json").write_text(
        json.dumps(
            {
                "schema_version": "atrex.shape_train.v1",
                "generator": {"name": "test", "version": 1},
                "objective": "Optimize vector add across hidden exact cases.",
                "operator_contract": {"operation": "vector add"},
                "workload_profile": {"phase": "decode"},
                "shape_domain": {"n": {"type": "integer", "min": 1, "max": 4096}},
                "invariants": ["n >= 1"],
                "coverage_regimes": [],
                "development_cases": [],
            }
        ),
        encoding="utf-8",
    )
    return operator


def test_prepare_materializes_pinned_single_dsl_campaign_workspaces(tmp_path: Path) -> None:
    operator = _operator(tmp_path / "operator")
    workspace = tmp_path / "workspace"
    environment = dict(os.environ)
    environment.update({"AGATE_URL": "https://agate.invalid", "AGATE_GPU": "test-gpu"})
    command = (
        sys.executable,
        str(PRODUCTION / "prepare.py"),
        "--kernel",
        str(operator),
        "--backend",
        "codex",
        "--workspace",
        str(workspace),
    )
    subprocess.run(command, check=True, env=environment, capture_output=True, text=True)

    settings = RuntimeSettings.from_file(workspace / "runtime.json")
    creation_keys: set[str] = set()
    for dsl in ("cuda", "triton", "cutedsl"):
        dsl_workspace = workspace / "dsls" / dsl
        campaign = CampaignSpecV3.from_file(dsl_workspace / "campaign.json")
        assert tuple(value.value for value in campaign.selected_dsls()) == (dsl,)
        assert campaign.attempts_per_trajectory == 2
        assert campaign.challenger_count == 1
        assert campaign.challenger_start_epoch == 2
        assert campaign.lineages[campaign.selected_dsls()[0]].baseline_kernel == (
            dsl_workspace / "inputs/baseline-kernel"
        )
        assert (dsl_workspace / "inputs/baseline-kernel/kernel.py").is_file()
        assert (dsl_workspace / "evaluation-contract.json").is_file()
        dsl_manifest = json.loads(
            (dsl_workspace / "production-manifest.json").read_text(encoding="utf-8")
        )
        assert dsl_manifest["dsl"] == dsl
        creation_keys.add(campaign.creation_key)
    assert len(creation_keys) == 3
    assert not (workspace / "campaign.json").exists()
    assert settings.campaign is not None
    assert settings.campaign.bootstrap_max_parallel_lineages == 3
    assert settings.campaign.max_parallel_branches == 2
    assert settings.campaign.gate_policy.production_gate is True
    assert settings.campaign.optimizer.agent_backend == "codex"
    assert settings.campaign.optimizer.timeout_seconds == 28_800
    assert settings.campaign.optimizer.bootstrap_timeout_seconds == 14_400
    assert settings.campaign.evolver.agent_backend == "codex"
    assert settings.campaign.evolver.timeout_seconds == 14_400
    secrets = workspace / "runtime.env"
    assert secrets.stat().st_mode & 0o777 == 0o600

    plan = json.loads((workspace / "ablation.json").read_text(encoding="utf-8"))
    assert plan["schema_version"] == 2
    assert plan["enabled"] is True
    assert plan["attempts_per_trajectory"] == 2
    # 1 Trajectory x (1 Active + 1 Challenger) = 2 isolated arms, plus one pooled and one
    # retained arm, each sized to that same combined Trajectory count.
    assert [
        (arm["kind"], arm["trajectories_per_branch"], arm["ephemeral_agent_state"])
        for arm in plan["arms"]
    ] == [
        ("isolated", 1, True),
        ("isolated", 1, True),
        ("pooled", 2, True),
        ("retained", 2, False),
    ]

    manifest_path = workspace / "production-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 2
    assert manifest["layout"] == "shared-runtime-per-dsl-campaign-workspaces"
    assert len(manifest["production_policy_digest"]) == 64
    manifest["core_commit"] = "0" * 40
    manifest["evolver_commit"] = "1" * 40
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    subprocess.run(command, check=True, env=environment, capture_output=True, text=True)
    assert json.loads(manifest_path.read_text(encoding="utf-8")) == manifest


def test_ablation_plan_derives_all_three_compute_matched_arms() -> None:
    """Each arm must mirror Active plus every Challenger to be compute-matched on its own."""
    schedule = {
        "challenger_count": 2,
        "trajectories_per_branch": 3,
        "attempts_per_trajectory": 4,
        # Gating Challengers out of early Epochs must not shrink the arm set, or arm
        # identity would move between Epochs.
        "challenger_start_epoch": 5,
    }

    enabled = _ablation_plan({"schedule": {**schedule, "event_only": True}})
    disabled = _ablation_plan({"schedule": {**schedule, "event_only": False}})
    omitted = _ablation_plan({"schedule": schedule})

    assert enabled["schema_version"] == 2
    assert enabled["attempts_per_trajectory"] == 4
    by_kind: dict[str, list[dict[str, Any]]] = {}
    for arm in enabled["arms"]:
        by_kind.setdefault(str(arm["kind"]), []).append(arm)
    # 3 Trajectories x (1 Active + 2 Challengers) = 9.
    assert len(by_kind["isolated"]) == 9
    assert {arm["trajectories_per_branch"] for arm in by_kind["isolated"]} == {1}
    assert {arm["ephemeral_agent_state"] for arm in by_kind["isolated"]} == {True}
    assert [arm["label"] for arm in by_kind["isolated"][:2]] == [
        "ablation-isolated-01",
        "ablation-isolated-02",
    ]
    assert by_kind["pooled"] == [
        {
            "kind": "pooled",
            "label": "ablation-pooled",
            "trajectories_per_branch": 9,
            "ephemeral_agent_state": True,
        }
    ]
    # The retained arm keeps Skills and Tools, so only the Evolver is removed.
    assert by_kind["retained"] == [
        {
            "kind": "retained",
            "label": "ablation-retained",
            "trajectories_per_branch": 9,
            "ephemeral_agent_state": False,
        }
    ]
    assert disabled["arms"] == []
    assert disabled["enabled"] is False
    assert omitted == disabled


def test_prepare_seeds_each_dsl_campaign_from_its_own_kernel(tmp_path: Path) -> None:
    operator = _operator(tmp_path / "operator")
    workspace = tmp_path / "workspace"
    triton_seed = tmp_path / "triton-v1.py"
    cuda_seed = tmp_path / "cuda-v1.py"
    triton_seed.write_text("# triton framework baseline\n", encoding="utf-8")
    cuda_seed.write_text("# cuda framework baseline\n", encoding="utf-8")
    environment = dict(os.environ)
    environment.update({"AGATE_URL": "https://agate.invalid", "AGATE_GPU": "test-gpu"})
    command = (
        sys.executable,
        str(PRODUCTION / "prepare.py"),
        "--kernel",
        str(operator),
        "--backend",
        "codex",
        "--workspace",
        str(workspace),
        "--dsl-seed-source",
        f"triton={triton_seed}",
        "--dsl-seed-source",
        f"cuda={cuda_seed}",
    )
    subprocess.run(command, check=True, env=environment, capture_output=True, text=True)

    reference = (operator / "reference.py").read_text(encoding="utf-8")
    expected = {
        "triton": triton_seed.read_text(encoding="utf-8"),
        "cuda": cuda_seed.read_text(encoding="utf-8"),
        "cutedsl": reference,
    }
    for dsl, text in expected.items():
        baseline = workspace / "dsls" / dsl / "inputs/baseline-kernel/kernel.py"
        assert baseline.read_text(encoding="utf-8") == text
        manifest = json.loads(
            (workspace / "dsls" / dsl / "production-manifest.json").read_text(encoding="utf-8")
        )
        pinned = manifest["dsl_seed_sources"][dsl]
        assert pinned["sha256"] == hashlib.sha256(text.encode()).hexdigest()
    manifest = json.loads(
        (workspace / "dsls/triton/production-manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["dsl_seed_sources"]["triton"]["path"] == str(triton_seed)
    assert manifest["dsl_seed_sources"]["cutedsl"]["path"] == str(operator / "reference.py")

    rejected = subprocess.run(
        (*command[:-2], "--dsl-seed-source", f"gluon={triton_seed}"),
        env=environment,
        capture_output=True,
        text=True,
    )
    assert rejected.returncode != 0
    assert "--dsl-seed-source must be DSL=PATH" in rejected.stderr


def test_prepare_prefers_shape_train_and_keeps_shape_valid_private(tmp_path: Path) -> None:
    operator = _shape_train_operator(tmp_path / "operator")
    workspace = tmp_path / "workspace"
    environment = dict(os.environ)
    environment.update({"AGATE_URL": "https://agate.invalid", "AGATE_GPU": "test-gpu"})
    subprocess.run(
        (
            sys.executable,
            str(PRODUCTION / "prepare.py"),
            "--kernel",
            str(operator),
            "--backend",
            "codex",
            "--workspace",
            str(workspace),
        ),
        check=True,
        env=environment,
        capture_output=True,
        text=True,
    )

    for dsl in ("cuda", "triton", "cutedsl"):
        dsl_workspace = workspace / "dsls" / dsl
        campaign = CampaignSpecV3.from_file(dsl_workspace / "campaign.json")
        assert campaign.shape_train == operator / "shape_train.json"
        assert campaign.agent_problem is None
        contract = json.loads(
            dsl_workspace.joinpath("evaluation-contract.json").read_text(encoding="utf-8")
        )
        assert contract["shapes"] == {"0": {"init_kwargs": None, "input_kwargs": {"n": 1024}}}
        assert "shape_valid" not in json.dumps(campaign.model_dump(mode="json"))


def test_prepare_can_attach_task_to_existing_service_workspace(tmp_path: Path) -> None:
    operator = _operator(tmp_path / "operator")
    service_workspace = tmp_path / "service"
    task_workspace = tmp_path / "task"
    environment = dict(os.environ)
    environment.update({"AGATE_URL": "https://agate.invalid", "AGATE_GPU": "test-gpu"})
    base_command = (
        sys.executable,
        str(PRODUCTION / "prepare.py"),
        "--kernel",
        str(operator),
    )
    subprocess.run(
        (
            sys.executable,
            str(PRODUCTION / "prepare.py"),
            "--services-only",
            "--workspace",
            str(service_workspace),
        ),
        check=True,
        env=environment,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        (
            *base_command,
            "--backend",
            "claude",
            "--workspace",
            str(task_workspace),
            "--service-workspace",
            str(service_workspace),
        ),
        check=True,
        env=environment,
        capture_output=True,
        text=True,
    )

    service = RuntimeSettings.from_file(service_workspace / "runtime.json")
    task = RuntimeSettings.from_file(task_workspace / "runtime.json")
    assert service.campaign is None
    assert service.gate_policy is not None
    service_manifest = json.loads(
        service_workspace.joinpath("production-manifest.json").read_text(encoding="utf-8")
    )
    assert service_manifest["layout"] == "production-control-plane"
    assert "backend" not in service_manifest
    service_runtime_text = service_workspace.joinpath("runtime.json").read_text(encoding="utf-8")
    assert "agent_backend" not in service_runtime_text
    assert task.server == service.server
    assert task.storage == service.storage
    assert task.gateway_proxy == service.gateway_proxy
    assert task.agate == service.agate
    assert task.gpu_wiki == service.gpu_wiki
    assert task.campaign is not None
    assert task.campaign.optimizer.agent_backend == "claude"
    assert task.campaign.evolver.agent_backend == "claude"
    assert task.campaign.attempt_workspaces_root.is_relative_to(task_workspace)
    assert task_workspace.joinpath("runtime.env").read_text(encoding="utf-8") == (
        service_workspace / "runtime.env"
    ).read_text(encoding="utf-8")
    manifest = json.loads(
        task_workspace.joinpath("production-manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["service_workspace"] == str(service_workspace.resolve())


def test_prepare_container_mode_needs_no_systemd_or_cgroup_configuration(
    tmp_path: Path,
) -> None:
    operator = _operator(tmp_path / "operator")
    service_workspace = tmp_path / "service"
    task_workspace = tmp_path / "task"
    environment = dict(os.environ)
    environment.update({"AGATE_URL": "https://agate.invalid", "AGATE_GPU": "test-gpu"})
    subprocess.run(
        (
            sys.executable,
            str(PRODUCTION / "prepare.py"),
            "--services-only",
            "--workspace",
            str(service_workspace),
            "--launcher-mode",
            "container",
        ),
        check=True,
        env=environment,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        (
            sys.executable,
            str(PRODUCTION / "prepare.py"),
            "--kernel",
            str(operator),
            "--backend",
            "codex",
            "--workspace",
            str(task_workspace),
            "--service-workspace",
            str(service_workspace),
        ),
        check=True,
        env=environment,
        capture_output=True,
        text=True,
    )

    service_manifest = json.loads(
        service_workspace.joinpath("production-manifest.json").read_text(encoding="utf-8")
    )
    task_manifest = json.loads(
        task_workspace.joinpath("production-manifest.json").read_text(encoding="utf-8")
    )
    settings = RuntimeSettings.from_file(task_workspace / "runtime.json")
    runtime_document = json.loads(
        task_workspace.joinpath("runtime.json").read_text(encoding="utf-8")
    )
    launcher_document = runtime_document["campaign"]["launcher"]

    assert service_manifest["launcher_mode"] == "container"
    assert task_manifest["launcher_mode"] == "container"
    assert settings.campaign is not None
    assert settings.campaign.launcher.mode == "container"
    assert settings.campaign.launcher.container is not None
    assert settings.campaign.launcher.sandbox is None
    assert "systemd_run_executable" not in launcher_document["container"]
    assert "resources" not in launcher_document["container"]

    mismatch = subprocess.run(
        (
            sys.executable,
            str(PRODUCTION / "prepare.py"),
            "--kernel",
            str(operator),
            "--backend",
            "codex",
            "--workspace",
            str(tmp_path / "mismatched-task"),
            "--service-workspace",
            str(service_workspace),
            "--launcher-mode",
            "sandbox",
        ),
        env=environment,
        capture_output=True,
        text=True,
    )
    assert mismatch.returncode != 0
    assert "must match the pinned service workspace" in mismatch.stderr


def test_summarize_combines_independent_dsl_results(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    for index, dsl in enumerate(("cuda", "triton", "cutedsl"), start=1):
        dsl_workspace = workspace / "dsls" / dsl
        dsl_workspace.mkdir(parents=True)
        dsl_workspace.joinpath("campaign.json").write_text(
            json.dumps({"lineages": {dsl: {}}}), encoding="utf-8"
        )
        dsl_workspace.joinpath("bootstrap-result.json").write_text(
            json.dumps(
                {
                    "campaign_id": f"campaign_{index:032x}",
                    "lineages": [{"lineage_id": f"lineage_{index:032x}"}],
                }
            ),
            encoding="utf-8",
        )
        dsl_workspace.joinpath("campaign-result.json").write_text(
            json.dumps(
                {
                    "campaign_id": f"campaign_{index:032x}",
                    "target_epoch_number": 10,
                    "lineages": [{"dsl": dsl}],
                }
            ),
            encoding="utf-8",
        )

    for phase, extra in (("bootstrap", ()), ("campaign", ("--target-epoch", "10"))):
        subprocess.run(
            (
                sys.executable,
                str(PRODUCTION / "summarize.py"),
                "--workspace",
                str(workspace),
                "--phase",
                phase,
                *extra,
            ),
            check=True,
            capture_output=True,
            text=True,
        )
        summary = json.loads((workspace / f"{phase}-results.json").read_text(encoding="utf-8"))
        assert tuple(summary["dsls"]) == ("cuda", "cutedsl", "triton")
        result_dsls = {
            entry["result"]["lineages"][0].get("dsl") for entry in summary["dsls"].values()
        }
        assert result_dsls == ({None} if phase == "bootstrap" else {"cuda", "triton", "cutedsl"})


def test_summarize_pairs_each_dsl_with_its_ablation_arms(tmp_path: Path) -> None:
    """The summary must carry the evolution-versus-control pairing for later comparison."""
    workspace = tmp_path / "workspace"
    for index, dsl in enumerate(("cuda", "triton", "cutedsl"), start=1):
        dsl_workspace = workspace / "dsls" / dsl
        dsl_workspace.mkdir(parents=True)
        dsl_workspace.joinpath("campaign.json").write_text(
            json.dumps({"lineages": {dsl: {}}}), encoding="utf-8"
        )
        dsl_workspace.joinpath("campaign-result.json").write_text(
            json.dumps(
                {
                    "campaign_id": f"campaign_{index:032x}",
                    "target_epoch_number": 10,
                    "lineages": [{"dsl": dsl}],
                }
            ),
            encoding="utf-8",
        )
        dsl_workspace.joinpath("bootstrap-result.json").write_text(
            json.dumps(
                {
                    "campaign_id": f"campaign_{index:032x}",
                    "lineages": [{"lineage_id": f"lineage_{index:032x}"}],
                }
            ),
            encoding="utf-8",
        )
        first = dsl_workspace / "ablation-isolated-01"
        first.mkdir()
        first.joinpath("campaign-result.json").write_text(
            json.dumps(
                {
                    "campaign_id": f"campaign_{index + 100:032x}",
                    "target_epoch_number": 10,
                    "lineages": [{"dsl": dsl}],
                }
            ),
            encoding="utf-8",
        )
        first.joinpath("seed-result.json").write_text(
            json.dumps({"campaign_id": f"campaign_{index + 100:032x}", "event_only": True}),
            encoding="utf-8",
        )
        # A second arm that never produced a result must not sink the whole summary.
        (dsl_workspace / "ablation-pooled").mkdir()

    subprocess.run(
        (
            sys.executable,
            str(PRODUCTION / "summarize.py"),
            "--workspace",
            str(workspace),
            "--phase",
            "campaign",
            "--target-epoch",
            "10",
            "--allow-partial",
        ),
        check=True,
        capture_output=True,
        text=True,
    )
    summary = json.loads((workspace / "campaign-results.json").read_text(encoding="utf-8"))

    for dsl, entry in summary["dsls"].items():
        arms = entry["ablation"]
        assert [arm["arm"] for arm in arms] == ["ablation-isolated-01", "ablation-pooled"]
        assert arms[0]["campaign_id"] != entry["campaign_id"]
        assert arms[0]["result"]["lineages"][0]["dsl"] == dsl
        assert arms[0]["seed_result"]["event_only"] is True
        assert arms[1]["status"] == "missing"

    subprocess.run(
        (
            sys.executable,
            str(PRODUCTION / "summarize.py"),
            "--workspace",
            str(workspace),
            "--phase",
            "bootstrap",
        ),
        check=True,
        capture_output=True,
        text=True,
    )
    bootstrap = json.loads((workspace / "bootstrap-results.json").read_text(encoding="utf-8"))
    assert all(entry["ablation"] == [] for entry in bootstrap["dsls"].values())


def test_summarize_preserves_successful_dsl_results_when_one_is_missing(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    for index, dsl in enumerate(("cuda", "triton", "cutedsl"), start=1):
        dsl_workspace = workspace / "dsls" / dsl
        dsl_workspace.mkdir(parents=True)
        dsl_workspace.joinpath("campaign.json").write_text(
            json.dumps({"lineages": {dsl: {}}}), encoding="utf-8"
        )
        if dsl == "cuda":
            continue
        dsl_workspace.joinpath("campaign-result.json").write_text(
            json.dumps(
                {
                    "campaign_id": f"campaign_{index:032x}",
                    "target_epoch_number": 10,
                    "lineages": [{"dsl": dsl}],
                }
            ),
            encoding="utf-8",
        )

    subprocess.run(
        (
            sys.executable,
            str(PRODUCTION / "summarize.py"),
            "--workspace",
            str(workspace),
            "--phase",
            "campaign",
            "--target-epoch",
            "10",
            "--allow-partial",
        ),
        check=True,
        capture_output=True,
        text=True,
    )

    summary = json.loads((workspace / "campaign-results.json").read_text(encoding="utf-8"))
    assert summary["dsls"]["cuda"]["status"] == "missing"
    assert summary["dsls"]["cuda"]["result"] is None
    assert summary["dsls"]["triton"]["result"]["lineages"][0]["dsl"] == "triton"
    assert summary["dsls"]["cutedsl"]["result"]["lineages"][0]["dsl"] == "cutedsl"


def test_production_shell_scripts_parse() -> None:
    for script in sorted(PRODUCTION.glob("*.sh")):
        subprocess.run(("bash", "-n", str(script)), check=True)


def test_campaign_entrypoint_uses_external_services_mode() -> None:
    campaign = (PRODUCTION / "campaign.sh").read_text(encoding="utf-8")

    assert "--external-services" in campaign
    assert "services.sh" not in campaign
    assert "start|stop|status|restart|__run" in campaign
    assert 'campaign_control="${atrex_prod_workspace}/campaign-run"' in campaign
    assert "workspace_process_pids" in campaign
    assert 'write_state "succeeded"' in campaign


def test_production_runner_has_independent_per_dsl_pipelines() -> None:
    runner = (PRODUCTION / "run.sh").read_text(encoding="utf-8")

    assert 'run_dsl_pipeline "${dsl}" &' in runner
    assert 'if ! bootstrap_one "${dsl}"; then' in runner
    assert 'run_one "${dsl}" "${campaign_id}"' in runner
    assert "At least one DSL Bootstrap failed" not in runner


def test_services_start_initializes_control_plane() -> None:
    service = (PRODUCTION / "services.sh").read_text(encoding="utf-8")

    assert "--services-only" in service
    assert '"${action}" == "start"' in service


def test_managed_local_wiki_drops_root_privileges() -> None:
    service = (PRODUCTION / "services.sh").read_text(encoding="utf-8")

    assert "campaign.launcher.sandbox.worker_user" in service
    assert 'setpriv --reuid="${wiki_uid}" --regid="${wiki_gid}" --init-groups' in service


def test_prepare_rejects_disabled_production_gate(tmp_path: Path) -> None:
    operator = _operator(tmp_path / "operator")
    policy = json.loads((PRODUCTION / "policy.json").read_text(encoding="utf-8"))
    policy["gate_policy"]["production_gate"] = False
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(json.dumps(policy), encoding="utf-8")
    environment = dict(os.environ)
    environment.update({"AGATE_URL": "https://agate.invalid", "AGATE_GPU": "test-gpu"})

    result = subprocess.run(
        (
            sys.executable,
            str(PRODUCTION / "prepare.py"),
            "--kernel",
            str(operator),
            "--backend",
            "codex",
            "--workspace",
            str(tmp_path / "workspace"),
            "--policy",
            str(policy_path),
        ),
        env=environment,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "must enable gate_policy.production_gate" in result.stderr


def test_production_agent_commit_rejects_dirty_checkout(tmp_path: Path) -> None:
    repository = tmp_path / "agent"
    repository.mkdir()
    subprocess.run(("git", "init", "-q", str(repository)), check=True)
    subprocess.run(
        ("git", "-C", str(repository), "config", "user.email", "atrex-test@example.invalid"),
        check=True,
    )
    subprocess.run(
        ("git", "-C", str(repository), "config", "user.name", "Atrex Test"),
        check=True,
    )
    tracked = repository / "agent.py"
    tracked.write_text("print('clean')\n", encoding="utf-8")
    subprocess.run(("git", "-C", str(repository), "add", "agent.py"), check=True)
    subprocess.run(("git", "-C", str(repository), "commit", "-qm", "initial"), check=True)

    commit = _git_commit(repository, require_clean=True, label="Core")
    assert len(commit) == 40

    tracked.write_text("print('dirty')\n", encoding="utf-8")
    (repository / "untracked.txt").write_text("not committed\n", encoding="utf-8")
    with pytest.raises(SystemExit, match=r"Core working tree is dirty.*exact source bytes"):
        _git_commit(repository, require_clean=True, label="Core")
