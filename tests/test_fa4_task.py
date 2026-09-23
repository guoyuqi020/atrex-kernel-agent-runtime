"""The FA4 kit preserves its R0 source and strict, private target evaluation contract."""

from __future__ import annotations

import ast
import json
import math
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
from conftest import digest
from test_source_tree_example_schedule import REPOSITORY, _module

from atrex_runtime.ablation_plan import build_ablation_plan
from atrex_runtime.artifacts.local import LocalArtifactStore
from atrex_runtime.bootstrap import CampaignSpecV3
from atrex_runtime.config import RuntimeSettings
from atrex_runtime.domain.models import Dsl
from atrex_runtime.gateway.abba import CommitPinnedAtrexBenchEvaluator, build_abba_source_request
from atrex_runtime.gateway.contract import AgateEvaluationContractV1
from atrex_runtime.gateway.oss_client import AGATE_MAX_INLINE_DEV_BYTES, OssAgateClient
from atrex_runtime.gateway.source_tree import SourceTreeAgateClient, attach_source_tree
from atrex_runtime.kernel_sources import KernelSourceBundle, SourceManifest, import_source_tree
from atrex_runtime.workers.problem_generalization import validate_public_operator_contract

INPUTS = REPOSITORY / "data/FA4"
SM120_INPUTS = REPOSITORY / "data/FA4-SM120"
EVOLVER_REVIEW_CONTRACT_COMMIT = "2ac444b793bedcbc1c317ecb4d5a1cecf4ccf801"


@pytest.mark.parametrize("inputs", [INPUTS, SM120_INPUTS])
def test_fa4_tasks_pin_review_compatible_evolver(inputs: Path) -> None:
    settings = RuntimeSettings.from_file(inputs / "runtime.template.json")
    assert settings.campaign.evolver.commit == EVOLVER_REVIEW_CONTRACT_COMMIT


def test_fa4_sm120_exposes_only_sm103_reference_and_empty_target_implementation(
    tmp_path: Path,
) -> None:
    spec = CampaignSpecV3.from_file(SM120_INPUTS / "campaign.json")
    assert spec.hardware_target == "L20N"
    assert set(spec.lineages) == {Dsl.CUTEDSL}
    adapter = (SM120_INPUTS / "task/adapter.py").read_text()
    assert "from implementation.sm120 import flash_attention_sm120" in adapter
    assert "vendor" not in adapter.lower()
    assert (SM120_INPUTS / "task/reference.py").read_bytes() == (
        INPUTS / "task/reference.py"
    ).read_bytes()
    assert (SM120_INPUTS / "task/input.py").read_bytes() == (
        INPUTS / "task/input.py"
    ).read_bytes()
    assert (SM120_INPUTS / "task/shape_valid.json").read_bytes() == (
        INPUTS / "task/shape_valid.json"
    ).read_bytes()

    metadata = json.loads((SM120_INPUTS / "task/metadata.json").read_text())
    assert metadata["id"] == "qwen38_max_l20n_prefill_flashinfer_trtllm_attention_fp8"
    assert metadata["target_hardware"] == "L20N"
    assert metadata["target_arch"] == "sm_120"
    roofline = json.loads((SM120_INPUTS / "task/roofline.json").read_text())
    assert len(roofline["shapes"]) == 30
    assert all(
        set(shape["SOL_time_ms"])
        == {"NVIDIA RTX PRO 5000 72GB Blackwell (SM120)"}
        for shape in roofline["shapes"].values()
    )

    manifest = SourceManifest.model_validate_json(
        (SM120_INPUTS / "task/source_manifest.json").read_bytes()
    )
    assert manifest.editable_roots == ("implementation",)
    source = tmp_path / "sm120-source"
    subprocess.run(
        ["git", "clone", "--quiet", str(SM120_INPUTS / "source.bundle"), str(source)],
        check=True,
    )
    assert subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip() == manifest.source.revision
    assert not (source / "vendor").exists()
    assert not (source / "vendor_support").exists()
    implementation = (source / "implementation/sm120.py").read_text()
    assert "raise NotImplementedError" in implementation
    assert "import cutlass.cute as cute" in implementation
    reference = source / "reference_sm103/flash_attention/flash_attn/cute"
    assert (reference / "sm100_hd256_2cta_fmha_forward.py").is_file()
    assert not (reference / "interface.py").exists()
    assert not list(reference.glob("*sm120*"))

    original = tmp_path / "sm103-source"
    subprocess.run(
        ["git", "clone", "--quiet", str(INPUTS / "source.bundle"), str(original)],
        check=True,
    )
    assert (
        reference / "sm100_hd256_2cta_fmha_forward.py"
    ).read_bytes() == (
        original
        / "vendor/flash_attention/flash_attn/cute/sm100_hd256_2cta_fmha_forward.py"
    ).read_bytes()
    hint = (SM120_INPUTS / "initial-evidence/README.md").read_text()
    assert "SM120 source" in hint and "deliberately absent" in hint
    assert "not a runtime dependency" in hint
    assert "native FP8-compute FA4 task" in hint
    assert "principal matrix-multiply data path in FP8" in hint


def test_fa4_sm120_preparation_is_self_contained(tmp_path: Path, runner) -> None:
    workspace = tmp_path / "FA4-SM120"
    runner.prepare(SM120_INPUTS, workspace, None, None)
    settings = RuntimeSettings.from_file(workspace / "runtime.json")
    assert settings.server.port == 8771
    spec = CampaignSpecV3.from_file(workspace / "campaign.json")
    assert spec.hardware_target == "L20N"
    assert spec.lineages[Dsl.CUTEDSL].source_repository == workspace / "source"
    assert (workspace / "source/PROVENANCE.json").is_file()
    prepared = json.loads((workspace / "prepared.json").read_text())
    assert prepared["hardware_target"] == "L20N"
    assert prepared["source_commit"] == "acd55c80e66d4885a066c6b281b7fd98c9100e02"
    assert prepared["source_file_count"] == 71
    assert prepared["production_gate"] is True
    assert prepared["seed_static_policy_violations"] == []
    contract = AgateEvaluationContractV1.model_validate_json(
        (workspace / "evaluation-contract.json").read_bytes()
    )
    assert contract.agent_correctness_policy().model_dump(mode="json") == {
        "comparison": "elementwise",
        "formula": "abs(candidate - reference) <= atol + rtol * abs(reference)",
        "default_tolerance": {"atol": 0.06, "rtol": 0.04},
        "output_tolerances": {
            "output": {"atol": 0.06, "rtol": 0.04},
            "mutated_inputs.out": {"atol": 0.06, "rtol": 0.04},
        },
    }
    assert not (SM120_INPUTS / "smoke").exists()
    assert not (workspace / "smoke").exists()

    malformed = contract.model_copy(
        update={
            "metadata": {
                "benchmark_contract": {
                    "correctness_tolerances": {
                        "output": {"atol": -1.0, "rtol": 0.04}
                    }
                }
            }
        }
    )
    with pytest.raises(ValueError, match="greater than or equal to 0"):
        malformed.agent_correctness_policy()


def test_real_fa4_abba_uses_oss_and_restores_both_complete_snapshots(
    tmp_path: Path, runner, monkeypatch: pytest.MonkeyPatch
) -> None:
    from atrex_runtime.gateway import oss_remote

    workspace = tmp_path / "FA4"
    runner.prepare(INPUTS, workspace, None, None)
    settings = RuntimeSettings.from_file(workspace / "runtime.json")
    evaluator = CommitPinnedAtrexBenchEvaluator(
        **settings.campaign.gate_policy.evaluator.model_dump(exclude={"agate_package_version"})
    )
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    imported = import_source_tree(
        workspace / "task/source_manifest.json", workspace / "source", artifacts
    )
    source = KernelSourceBundle(
        imported.validate_tree(artifacts.verify(imported.seed_digest).payload_path),
        imported,
        "kernel.py",
    )
    hot = "vendor/flash_attention/flash_attn/cute/sm100_hd256_2cta_fmha_forward.py"
    candidate = KernelSourceBundle(
        {**source.files, hot: source.files[hot] + "\n# candidate-only change\n"},
        imported,
        "kernel.py",
    )
    contract = AgateEvaluationContractV1.model_validate_json(
        (workspace / "evaluation-contract.json").read_bytes()
    )
    payload = build_abba_source_request(
        hardware_target="L20D",
        contract=contract,
        shape_ids=["0"],
        schedule=[
            {"revision": side, "repeat": i // 2}
            for i, side in enumerate(["incumbent", "candidate", "candidate", "incumbent"])
        ],
        incumbent_source=source,
        candidate_source=candidate,
        evaluator_files=evaluator.files(),
        per_run_timeout_seconds=120,
        allocation_timeout_seconds=600,
    )
    assert (
        sum(len(text.encode()) for text in payload["files"].values()) > AGATE_MAX_INLINE_DEV_BYTES
    )
    remote = tmp_path / "remote"
    remote.mkdir()
    uploaded = []

    def prepare(gpu, files, *, kind):
        assert gpu == "L20D" and kind == "dev" and len(files) == 1
        return {
            "job_id": "dv_reserved",
            "uploads": [
                {
                    "path": files[0]["path"],
                    "put_url": "https://oss.example.test/upload",
                    "upload_ref": "opaque-reference",
                }
            ],
        }

    def upload(url, path):
        uploaded.append(Path(path).read_bytes())

    wire = OssAgateClient(
        SimpleNamespace(
            prepare_uploads=prepare,
            upload_file=upload,
            submit_job=lambda kind, request: request,
        )
    ).submit_job("dev", payload)
    assert len(uploaded) == 1
    assert sum(len(text.encode()) for text in wire["files"].values()) < AGATE_MAX_INLINE_DEV_BYTES
    assert wire["command"].endswith(f"&& (\n{payload['command']}\n)")
    attachment = wire["oss_files"][0]
    archive = remote / attachment["path"]
    archive.write_bytes(uploaded[0])
    monkeypatch.chdir(remote)
    oss_remote.unpack(archive, wire["command"].split()[3])
    for path, text in payload["files"].items():
        assert (remote / path).read_bytes() == text.encode("utf-8")
    assert (remote / "snapshots/incumbent" / hot).read_bytes() != (
        remote / "snapshots/candidate" / hot
    ).read_bytes()


@pytest.fixture
def runner(monkeypatch: pytest.MonkeyPatch):
    module = _module("scripts/source-tree/task.py")
    # Source/evaluator imports are real and offline. Agent submodules are not needed in these tests.
    monkeypatch.setattr(
        module,
        "build_optimizer_base_loader",
        lambda *_: SimpleNamespace(
            build_candidate=lambda *_: SimpleNamespace(
                candidate=SimpleNamespace(optimizer_digest=digest("test-fa4-optimizer"))
            )
        ),
    )
    monkeypatch.setattr(
        module,
        "GitEvolverBundleResolver",
        lambda *_, **kwargs: SimpleNamespace(
            resolve=lambda: SimpleNamespace(commit=kwargs["commit"])
        ),
    )
    return module


def test_fa4_target_contract_and_schedule_are_self_contained() -> None:
    spec = CampaignSpecV3.from_file(INPUTS / "campaign.json")
    assert spec.hardware_target == "L20D"
    assert set(spec.lineages) == {Dsl.CUTEDSL}
    assert spec.max_challengers == 1
    assert spec.optimizer_attempt_budget == 6
    plan = build_ablation_plan({"schedule": {**spec.model_dump(mode="json"), "event_only": True}})
    assert len(plan["arms"]) == 15
    assert plan["optimizer_attempt_budget_per_trajectory"] == 15
    assert all(arm["target_epoch_number"] == 5 for arm in plan["arms"])
    shapes = json.loads((INPUTS / "task/shape_valid.json").read_text())
    assert len(shapes) == 30
    public = json.loads((INPUTS / "task/shape_train.json").read_text())
    validate_public_operator_contract(public, private_shapes=shapes)
    for entry in shapes.values():
        for key, value in {
            "num_q_heads": 16,
            "num_kv_heads": 1,
            "head_dim": 256,
            "page_size": 64,
            "bmm2_scale": 1.0,
        }.items():
            assert entry["input_kwargs"][key] == value
    metadata = json.loads((INPUTS / "task/metadata.json").read_text())
    policy = metadata["benchmark_contract"]
    assert policy["correctness_tolerances"] == {
        "output": {"atol": 0.06, "rtol": 0.04},
        "mutated_inputs.out": {"atol": 0.06, "rtol": 0.04},
    }
    assert policy["mutates_inputs"] == ["out"]
    assert policy["scratch_inputs"] == ["workspace_buffer"]
    assert metadata["perf_gpu"] == "NVIDIA L20D"
    assert not (INPUTS / "task/solution.py").exists()


def test_preparation_is_offline_preserves_inputs_and_exposes_real_r0(
    tmp_path: Path,
    runner,
) -> None:
    before = {path: path.read_bytes() for path in INPUTS.rglob("*") if path.is_file()}
    workspace = tmp_path / "nested/FA4"
    runner.prepare(INPUTS, workspace, "codex", 18870)
    assert before == {path: path.read_bytes() for path in before}
    settings = RuntimeSettings.from_file(workspace / "runtime.json")
    assert settings.gpu_wiki is None
    assert settings.server.port == 18870
    assert settings.campaign is not None
    assert settings.campaign.optimizer.agent_backend == "codex"
    assert settings.campaign.evolver.agent_backend == "codex"
    assert settings.campaign.gateway_proxy_url == "http://127.0.0.1:18870"
    assert settings.campaign.roofline_builder is None
    assert settings.campaign.gate_policy.lock_clocks
    assert not settings.campaign.gate_policy.production_gate
    assert Path(settings.campaign.gate_policy.evaluator.repository) == workspace / "evaluator"
    spec = CampaignSpecV3.from_file(workspace / "campaign.json")
    assert spec.lineages[Dsl.CUTEDSL].source_repository == workspace / "source"
    frozen_plan = json.loads((workspace / "ablation.json").read_text())
    assert frozen_plan == build_ablation_plan(
        {"schedule": {**spec.model_dump(mode="json"), "event_only": True}}
    )
    contract = AgateEvaluationContractV1.model_validate_json(
        (workspace / "evaluation-contract.json").read_bytes()
    )
    assert len(contract.shapes) == 30 and contract.roofline is not None
    with pytest.raises(ValueError, match="already exists"):
        runner.prepare(INPUTS, workspace, None, None)
    assert not (workspace / "state").exists()
    assert not (workspace / "runtime-secrets.json").exists()
    prepared = json.loads((workspace / "prepared.json").read_text())
    assert prepared["gpu_jobs_submitted"] == 0
    assert prepared["seed_static_policy_violations"]  # Why this task disables that static scan.
    assert prepared["ablation_arm_count"] == 15
    assert prepared["ablation_optimizer_attempt_budget_per_trajectory"] == 15
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    source = import_source_tree(
        workspace / "task/source_manifest.json", workspace / "source", artifacts
    )
    root = artifacts.verify(source.seed_digest).payload_path
    assert (root / "kernel.py").read_bytes() == (INPUTS / "task/adapter.py").read_bytes()
    hot = root / "vendor/flash_attention/flash_attn/cute/sm100_hd256_2cta_fmha_forward.py"
    text = hot.read_text()
    assert "does not support seqused_q/seqused_k" in text
    assert "assert not pack_gqa" in text
    assert source.editable(hot.relative_to(root).as_posix())
    assert not source.editable("kernel.py")
    assert not source.editable("vendor_support/quack/utils.py")
    clone = tmp_path / "tampered"
    shutil.copytree(root, clone)
    (clone / "kernel.py").chmod(0o600)
    (clone / "kernel.py").write_text("pass\n")
    with pytest.raises(ValueError, match="fixed source file was modified"):
        source.validate_tree(clone)


def test_failed_preflight_never_publishes_workspace(tmp_path: Path, runner, monkeypatch) -> None:
    workspace = tmp_path / "FA4"
    monkeypatch.setattr(runner, "build_optimizer_base_loader", lambda *_: None)
    with pytest.raises(ValueError, match="Optimizer base source"):
        runner.prepare(INPUTS, workspace, None, None)
    assert not workspace.exists()
    assert not list(workspace.parent.glob(".source-tree-prepare-*"))


def test_evaluator_uses_metadata_tolerances_for_eval_and_abba(tmp_path: Path, runner) -> None:
    workspace = tmp_path / "FA4"
    runner.prepare(INPUTS, workspace, None, None)
    config = json.loads((workspace / "runtime.json").read_text())
    evaluator = CommitPinnedAtrexBenchEvaluator(
        **{
            **config["campaign"]["gate_policy"]["evaluator"],
            "repository": str(workspace / "evaluator"),
        }
    )
    files = evaluator.files()
    code = files["atrex-bench/src/atrex_bench/eval/correctness.py"]
    tree = ast.parse(code)
    selected = ast.Module(
        body=[
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name in {"_load_output_tolerance_contract", "load_minimum_correctness_cases"}
        ],
        type_ignores=[],
    )

    @dataclass
    class OutputTolerance:
        atol: float
        rtol: float

    namespace = {"Path": Path, "json": json, "math": math, "OutputTolerance": OutputTolerance}
    exec(compile(selected, "correctness-contract", "exec"), namespace)
    tolerances = namespace["_load_output_tolerance_contract"](workspace / "task/reference.py")
    assert tolerances == {
        "output": OutputTolerance(0.06, 0.04),
        "mutated_inputs.out": OutputTolerance(0.06, 0.04),
    }
    assert namespace["load_minimum_correctness_cases"](workspace / "task/reference.py") == 1
    assert "args.correctness_max_rel_l2 = None" in files["atrex-bench/scripts/run_eval.py"]
    assert "args.correctness_max_mismatch_rate = None" in files["atrex-bench/scripts/run_eval.py"]
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    source_contract = import_source_tree(
        workspace / "task/source_manifest.json", workspace / "source", artifacts
    )
    root = artifacts.verify(source_contract.seed_digest).payload_path
    source = KernelSourceBundle(source_contract.validate_tree(root), source_contract, "kernel.py")
    contract = AgateEvaluationContractV1.model_validate_json(
        (workspace / "evaluation-contract.json").read_bytes()
    )
    payload = build_abba_source_request(
        hardware_target="L20D",
        contract=contract,
        shape_ids=["0"],
        schedule=[{"revision": "incumbent", "repeat": 0}, {"revision": "candidate", "repeat": 0}],
        incumbent_source=source,
        candidate_source=source,
        evaluator_files=files,
        per_run_timeout_seconds=120,
        allocation_timeout_seconds=600,
    )
    metadata = json.loads(payload["files"]["reference/metadata.json"])
    assert metadata["benchmark_contract"]["correctness_tolerances"] == {
        "output": {"atol": 0.06, "rtol": 0.04},
        "mutated_inputs.out": {"atol": 0.06, "rtol": 0.04},
    }
    assert "atrex-bench/src/atrex_bench/eval/correctness.py" in payload["files"]


def test_packaged_commits_are_explicit_and_stable(tmp_path: Path, runner) -> None:
    manifest = SourceManifest.model_validate_json(
        (INPUTS / "task/source_manifest.json").read_bytes()
    )
    with pytest.raises(ValueError, match="expected"):
        runner.clone_bundle(INPUTS / "source.bundle", tmp_path / "source", "0" * 40)
    actual = subprocess.check_output(
        ["git", "-C", str(tmp_path / "source"), "rev-parse", "HEAD"], text=True
    ).strip()
    assert actual == manifest.source.revision


def test_runtime_keys_are_private_reused_and_not_silently_replaced(tmp_path: Path, runner) -> None:
    first = runner.service_secrets(tmp_path)
    assert first == runner.service_secrets(tmp_path)
    path = tmp_path / "runtime-secrets.json"
    assert path.stat().st_mode & 0o777 == 0o600
    path.write_text("{}")
    with pytest.raises(ValueError, match="refusing to regenerate"):
        runner.service_secrets(tmp_path)
    assert path.read_text() == "{}"


def test_runner_uses_current_python_and_absolute_epoch_target(
    tmp_path, runner, monkeypatch
) -> None:
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(stdout=json.dumps({"campaign_id": "campaign-test"}))

    monkeypatch.setattr(runner.subprocess, "run", run)
    result = runner.run_cli(
        ["run-campaign", "--target-epoch", "3"], output=tmp_path / "result.json"
    )
    assert result == {"campaign_id": "campaign-test"}
    assert calls[0][0] == runner.sys.executable
    assert calls[0][1:3] == ["-c", "from atrex_runtime.cli import main; main()"]
    assert calls[0][-2:] == ["--target-epoch", "3"]


def test_original_asset_hashes_cover_full_source_and_evaluator(tmp_path: Path, runner) -> None:
    integrity = json.loads((INPUTS / "asset-integrity.json").read_text())
    runner.verify_assets(INPUTS, integrity["files_sha256"])
    for name, count in (("source", 70), ("evaluator", 32)):
        component = integrity["components"][name]
        assert len(component["files_sha256"]) == count
        root = tmp_path / name
        runner.clone_bundle(INPUTS / f"{name}.bundle", root, component["commit"])
        runner.verify_component(root, component["files_sha256"])
        with pytest.raises(ValueError, match="file set"):
            runner.verify_component(root, {})


def test_modified_task_inputs_fail_before_any_workspace_is_created(tmp_path: Path, runner) -> None:
    inputs = tmp_path / "inputs"
    shutil.copytree(INPUTS, inputs)
    (inputs / "task/input.py").write_text("# changed generator\n")
    workspace = tmp_path / "workspace"
    with pytest.raises(ValueError, match=r"verified original: task/input\.py"):
        runner.prepare(inputs, workspace, None, None)
    assert not workspace.exists()
    assert not list(tmp_path.glob(".source-tree-prepare-*"))


def test_campaign_and_ablation_roles_use_distinct_default_epoch_targets(
    tmp_path: Path, runner, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "FA4"
    runner.prepare(INPUTS, workspace, None, None)
    cli_calls: list[list[str]] = []

    def run_cli(arguments: list[str], *, output: Path):
        del output
        cli_calls.append(arguments)
        return {"campaign_id": "campaign_" + "0" * 32}

    monkeypatch.setattr(runner, "run_cli", run_cli)
    monkeypatch.setattr(runner.sys, "argv", ["task.py", "campaign", "--workspace", str(workspace)])
    runner.main()
    assert [call[0] for call in cli_calls] == ["bootstrap", "run-campaign"]
    assert cli_calls[1][cli_calls[1].index("--target-epoch") + 1] == "1"

    calls: list[tuple[list[str], bool]] = []

    def run(command: list[str], *, check: bool) -> None:
        calls.append((command, check))

    monkeypatch.setattr(runner.subprocess, "run", run)
    monkeypatch.setattr(runner.sys, "argv", ["task.py", "ablation", "--workspace", str(workspace)])
    runner.main()
    assert len(calls) == 1 and calls[0][1]
    command = calls[0][0]
    assert command[:2] == [runner.sys.executable, str(REPOSITORY / "scripts/source-tree/run.py")]
    assert command[command.index("--workspace") + 1] == str(workspace / "ablation-run")
    assert command[command.index("--campaign") + 1] == str(workspace / "campaign.json")
    assert command[command.index("--plan") + 1] == str(workspace / "ablation.json")
    assert command[command.index("--target-epoch") + 1] == "5"


def test_smoke_stages_original_scripts_source_and_inputs_and_cleans_up(
    tmp_path: Path, runner, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "FA4"
    runner.prepare(INPUTS, workspace, None, None)
    calls = []
    staged = []

    def run(command, **kwargs):
        assert command[0] == "bash"
        assert command[2:] == ["target", "0"]
        root = Path(command[1]).parent
        staged.append(root)
        integrity = json.loads((INPUTS / "asset-integrity.json").read_text())
        runner.verify_assets(root, integrity["components"]["source"]["files_sha256"])
        for name in ("smoke.py", "run_agate_dev.sh", "reference/shapes.json"):
            assert (root / name).read_bytes() == (INPUTS / "smoke" / name).read_bytes()
        assert (root / "kernel.py").read_bytes() == (INPUTS / "task/adapter.py").read_bytes()
        for name in ("reference.py", "input.py"):
            assert (root / "reference" / name).read_bytes() == (INPUTS / "task" / name).read_bytes()
        assert not (root / ".git").exists()
        calls.append(command)

    monkeypatch.setattr(runner.subprocess, "run", run)
    runner.run_smoke(workspace, "target", "0")
    assert len(calls) == 1 and not staged[0].exists()
    assert not (workspace / "state").exists()
    with pytest.raises(ValueError, match="Unknown smoke Shape ID"):
        runner.run_smoke(workspace, "target", "999")
    assert len(calls) == 1


def test_actual_eval_transport_preserves_supplied_contract_and_sources(
    tmp_path: Path, runner
) -> None:
    workspace = tmp_path / "FA4"
    runner.prepare(INPUTS, workspace, None, None)
    settings = RuntimeSettings.from_file(workspace / "runtime.json")
    evaluator = CommitPinnedAtrexBenchEvaluator(
        **settings.campaign.gate_policy.evaluator.model_dump(exclude={"agate_package_version"})
    )
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    imported = import_source_tree(
        workspace / "task/source_manifest.json", workspace / "source", artifacts
    )
    root = artifacts.verify(imported.seed_digest).payload_path
    source = KernelSourceBundle(imported.validate_tree(root), imported, "kernel.py")
    contract = AgateEvaluationContractV1.model_validate_json(
        (workspace / "evaluation-contract.json").read_bytes()
    ).model_copy(
        update={
            "options": AgateEvaluationContractV1.model_validate_json(
                (workspace / "evaluation-contract.json").read_bytes()
            ).options.model_copy(update={"num_correctness_cases": 5}),
            "runner_overrides": {
                "candidate_timeout_s": 120,
                "perf_timeout_s": 120,
                "warmup_iters": 10,
                "benchmark_mode": "eager",
            },
        }
    )
    submissions = []

    class Client:
        def submit_job(self, kind, payload):
            submissions.append((kind, payload))
            return {"job_id": "test", "status": "submitted"}

    client = SourceTreeAgateClient(Client(), evaluator)
    client.submit_job("eval", attach_source_tree({}, source, contract, "L20D"))
    kind, payload = submissions[0]
    assert kind == "dev" and payload["spec"]["target_hardware"] == ["L20D"]
    files = payload["files"]
    request = json.loads(files["request.json"])
    assert request["raw_result"] is True and request["lock_clocks"] is True
    assert request["schedule"] == [{"revision": "candidate", "repeat": 0}]
    assert request["evaluator"]["num_correctness_cases"] == 5
    assert request["evaluator"]["warmup_iters"] == 10
    assert request["evaluator"]["bench_iters"] == 100
    assert request["evaluator"]["candidate_timeout_s"] == 120
    # Ordinary tree Evaluate currently gives its evaluator the outer performance budget.
    assert request["evaluator"]["perf_timeout_s"] == 600
    assert request["evaluator"]["clock_lock_mode"] == "external"
    assert json.loads(files["reference/shapes.json"]) == contract.shapes
    assert json.loads(files["reference/metadata.json"]) == contract.metadata
    assert files["reference/reference.py"] == contract.reference_py
    assert files["reference/input.py"] == contract.input_py
    for name, text in source.files.items():
        assert files[f"snapshots/candidate/{name}"] == text
    for name, text in evaluator.files().items():
        assert files[name] == text
    original_roofline = json.loads((INPUTS / "task/roofline.json").read_text())
    transmitted_roofline = json.loads(files["reference/roofline.json"])
    for shape_id, entry in original_roofline["shapes"].items():
        transmitted = transmitted_roofline["shapes"][shape_id]
        assert transmitted["SOL_time_ms"] == {
            "NVIDIA B300": entry["SOL_time_ms"]["NVIDIA B300 (SM100)"]
        }
        for name, value in entry.items():
            if name != "SOL_time_ms":
                assert transmitted[name] == value
