from __future__ import annotations

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
_git_commit = cast(Any, runpy.run_path(str(PRODUCTION / "prepare.py"))["_git_commit"])


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
    assert settings.campaign.optimizer.timeout_seconds == 5400
    assert settings.campaign.optimizer.bootstrap_timeout_seconds == 10_800
    assert settings.campaign.evolver.agent_backend == "codex"
    assert settings.campaign.evolver.timeout_seconds == 10_800
    secrets = workspace / "runtime.env"
    assert secrets.stat().st_mode & 0o777 == 0o600

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
        assert contract["shapes"] == {
            "0": {"init_kwargs": None, "input_kwargs": {"n": 1024}}
        }
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
