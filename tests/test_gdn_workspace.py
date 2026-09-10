"""Task inputs stay read-only; all GDN run products belong to a separate workspace."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_source_tree_example_schedule import REPOSITORY, _module

from atrex_runtime.bootstrap import CampaignSpecV3
from atrex_runtime.config import RuntimeSettings


def _files(root: Path) -> dict[str, bytes]:
    return {path.relative_to(root).as_posix(): path.read_bytes()
            for path in root.rglob("*") if path.is_file()}


@pytest.mark.parametrize("kit", ["GDN", "GDN-full"])
def test_gdn_pinned_optimizer_builds_without_skill_submodules(tmp_path, kit):
    from atrex_runtime.artifacts.local import LocalArtifactStore
    from atrex_runtime.composition.bootstrap import build_optimizer_base_loader
    from atrex_runtime.domain.models import Dsl

    inputs = REPOSITORY / "data" / kit
    settings = RuntimeSettings.from_file(inputs / "runtime.template.json")
    source = settings.kernel_agent.base_source
    assert source is not None
    assert source.allowed_submodules == {}
    if not (Path(source.repository) / ".git").exists():
        pytest.skip("KDA Optimizer submodule is not initialized")
    specs = [CampaignSpecV3.from_file(inputs / name) for name in (
        "campaign.json", "ablation-campaign.json",
    )]
    commit = specs[0].base_revision.commit
    assert all(spec.base_revision.commit == commit for spec in specs)
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    loader = build_optimizer_base_loader(settings, artifacts)
    assert loader is not None
    result = loader.build_candidate(Dsl.CUTEDSL, commit)
    provenance = artifacts.verify(result.source_provenance_digest).payload_path / "value.json"
    value = json.loads(provenance.read_text())
    assert value["commit"] == commit
    assert value["submodules"] == []
    sealed = artifacts.verify(result.candidate.optimizer_digest).payload_path
    assert (sealed / "src/main.py").is_file()
    assert (sealed / "skills/README.md").is_file()
    assert not (sealed / ".gitmodules").exists()
    for name in ("KernelWiki", "ncu-report-skill"):
        assert not (sealed / "skills" / name).exists()
    checked = _module("scripts/gdn/prepare.py").validate_runtime_sources(settings, tuple(specs))
    assert checked["optimizers"][0]["artifact_digest"] == result.candidate.optimizer_digest
    assert checked["evolver"]["commit"] == settings.campaign.evolver.commit
    assert checked["evaluator"]["commit"] == settings.campaign.gate_policy.evaluator.commit
    assert checked["evaluator"]["file_count"] > 0
    assert checked["roofline"]["commit"] == settings.campaign.roofline_builder.commit


@pytest.mark.parametrize("component", ["evolver", "evaluator", "roofline"])
def test_preflight_rejects_a_commit_that_exists_but_cannot_be_consumed(
    tmp_path, monkeypatch, component,
):
    from atrex_runtime.git_import import SafeGitImporter

    module = _module("scripts/gdn/prepare.py")
    inputs = REPOSITORY / "data/GDN"
    settings = RuntimeSettings.from_file(inputs / "runtime.template.json")
    spec = CampaignSpecV3.from_file(inputs / "campaign.json")
    if not (Path(settings.kernel_agent.base_source.repository) / ".git").exists():
        pytest.skip("KDA submodule is not initialized")
    section = {
        "evolver": settings.campaign.evolver,
        "evaluator": settings.campaign.gate_policy.evaluator,
        "roofline": settings.campaign.roofline_builder,
    }[component]
    # The old preflight passes; only the actual fetch-by-SHA consumer reveals the problem.
    module.git("-C", section.repository, "cat-file", "-e", f"{section.commit}^{{commit}}")
    original = SafeGitImporter.fetch_commit
    calls = []

    def fail_fetch(self, repository, remote, commit):
        calls.append(self._label)
        expected = {"evolver": "Evolver", "evaluator": "Atrex Bench comparison evaluator",
                    "roofline": "Atrex Bench Roofline"}[component]
        if self._label == expected:
            raise RuntimeError(f"upload-pack: not our ref {commit}")
        return original(self, repository, remote, commit)

    monkeypatch.setattr(SafeGitImporter, "fetch_commit", fail_fetch)
    with pytest.raises(ValueError, match=rf"{component.capitalize()} .*not our ref"):
        module.validate_runtime_sources(settings, (spec,))
    assert calls


@pytest.mark.parametrize(("kit", "custom_workspace"), [
    ("GDN", True), ("GDN-full", True), ("GDN-full", False),
])
@pytest.mark.parametrize("preflight_fails", [False, True])
def test_prepare_snapshots_inputs_without_writing_to_data(
    tmp_path, monkeypatch, kit, custom_workspace, preflight_fails,
):
    module = _module("scripts/gdn/prepare.py")
    inputs = tmp_path / "data" / kit
    inputs.mkdir(parents=True)
    for name in ("campaign.json", "ablation-campaign.json", "ablation.json",
                 "runtime.template.json", "source.bundle"):
        shutil.copyfile(REPOSITORY / "data" / kit / name, inputs / name)
    for name in ("task", "initial-evidence"):
        shutil.copytree(REPOSITORY / "data" / kit / name, inputs / name,
                        ignore=shutil.ignore_patterns("__pycache__", ".DS_Store"))
    before = _files(inputs)
    workspace = tmp_path / ("workspaces/nested" if custom_workspace else "workspaces") / kit
    monkeypatch.setattr(module, "__file__", str(tmp_path / "scripts/gdn/prepare.py"))
    monkeypatch.setattr(module.sys, "platform", "linux")
    monkeypatch.setattr(module.sys, "argv", [
        "prepare.py", "--worker-user", "worker",
        *(["--inputs", str(inputs)] if kit != "GDN" else []),
        *(["--workspace", str(workspace)] if custom_workspace else []),
    ])
    monkeypatch.setattr(module.pwd, "getpwnam", lambda _name: SimpleNamespace(
        pw_name="worker", pw_uid=1000, pw_dir="/home/worker",
    ))
    def preflight(settings, specs):
        assert {spec.base_revision.commit for spec in specs} == {
            json.loads((inputs / "campaign.json").read_text())["base_revision"]["commit"]
        }
        assert len(specs) == 2
        assert settings.kernel_agent.base_source.repository == str(
            tmp_path / "src/kernel-design-agents"
        )
        if preflight_fails:
            raise ValueError("Runtime source preflight failed: fetch-by-SHA unavailable")
        return {"verified": True}

    monkeypatch.setattr(module, "validate_runtime_sources", preflight)
    if preflight_fails:
        with pytest.raises(ValueError, match="fetch-by-SHA unavailable"):
            module.main()
        assert not (workspace / "prepared.json").exists()
        assert not (workspace / "runtime.json").exists()
        assert not (workspace / "state").exists()
        assert _files(inputs) == before
        return
    module.main()
    assert _files(inputs) == before
    settings = RuntimeSettings.from_file(workspace / "runtime.json")
    assert settings.storage.registry_database == workspace / "state/registry.sqlite"
    assert settings.campaign.attempt_workspaces_root == workspace / "state/attempt-workspaces"
    assert settings.kernel_agent.base_source.repository == str(
        tmp_path / "src/kernel-design-agents"
    )
    assert settings.campaign.evolver.repository == str(tmp_path / "src/atrex-kernel-agent-evolver")
    assert settings.campaign.gate_policy.evaluator.repository == str(
        tmp_path / "third_party/atrex-bench"
    )
    assert settings.campaign.launcher.sandbox.reference_projects_root == (
        tmp_path / "third_party/reference-projects"
    )
    for name in ("campaign.json", "ablation-campaign.json"):
        spec = CampaignSpecV3.from_file(workspace / name)
        assert spec.evaluation_contract == workspace / "evaluation-contract.json"
        assert spec.shape_train == workspace / "task/shape_train.json"
        for lineage in spec.lineages.values():
            assert lineage.source_repository == workspace / "source"
            assert lineage.source_manifest == workspace / "task/source_manifest.json"
    assert (workspace / "prepared.json").is_file()
    provenance = json.loads((workspace / "prepared.json").read_text())
    assert provenance["source_preflight"] == {"verified": True}
    assert (workspace / provenance["task_inputs"]).resolve() == inputs
    manifest = json.loads((inputs / "task/source_manifest.json").read_text())
    assert provenance["source_commit"] == manifest["source"]["revision"]
    public = json.loads((workspace / "task/shape_train.json").read_text())
    assert ("M64-oriented" in public["objective"]) == (kit == "GDN-full")
    assert not (workspace / "state").exists()
    module.main()  # idempotent preparation
    (workspace / "state").mkdir()
    module.main()  # identical inputs still accepted after state exists
    snapshots = _files(workspace / "task")
    public = json.loads((inputs / "task/shape_train.json").read_text())
    public["objective"] = "A changed objective"
    (inputs / "task/shape_train.json").write_text(json.dumps(public))
    with pytest.raises(SystemExit, match="Existing runtime state"):
        module.main()
    assert _files(workspace / "task") == snapshots


@pytest.mark.parametrize("script,arguments", [
    ("prepare", []), ("run", ["campaign"]),
])
def test_gdn_cannot_use_data_as_workspace(tmp_path, monkeypatch, script, arguments):
    module = _module(f"scripts/gdn/{script}.py")
    monkeypatch.setattr(module, "__file__", str(tmp_path / f"scripts/gdn/{script}.py"))
    monkeypatch.setattr(module.sys, "platform", "linux")
    monkeypatch.setattr(module.sys, "argv", [
        f"{script}.py", *arguments, "--workspace", str(tmp_path / "data/GDN/run"),
    ])
    with pytest.raises(SystemExit, match="separate from data"):
        module.main()
    assert not (tmp_path / "data").exists()


def test_gdn_preparation_refuses_all_changes_before_writing(tmp_path):
    module = _module("scripts/gdn/prepare.py")
    (tmp_path / "state").mkdir()
    (tmp_path / "runtime.json").write_bytes(b"old")
    with pytest.raises(SystemExit, match=r"refusing to replace runtime\.json"):
        module.write_inputs(tmp_path, {"runtime.json": b"new", "task/new.json": b"{}"})
    assert (tmp_path / "runtime.json").read_bytes() == b"old"
    assert not (tmp_path / "task").exists()


def test_gdn_full_restores_original_hints_without_changing_the_task(tmp_path):
    from atrex_runtime.artifacts.local import LocalArtifactStore
    from atrex_runtime.kernel_sources import import_source_tree

    clean = REPOSITORY / "data/GDN"
    full = REPOSITORY / "data/GDN-full"
    original = (full / "task/shape_train.json").read_bytes()
    assert hashlib.sha256(original).hexdigest() == (
        "7695093a901dc50f595ba0b0cd95df273882a3f14c5c457f6b050789d9437df9"
    )

    def without_hints(value):
        if isinstance(value, dict):
            return {key: without_hints(item) for key, item in value.items()
                    if key not in {"objective", "range_evidence", "value_evidence",
                                   "coverage_regimes"}}
        if isinstance(value, list):
            return [without_hints(item) for item in value]
        return value

    assert without_hints(json.loads(original)) == without_hints(
        json.loads((clean / "task/shape_train.json").read_text())
    )
    for path in (clean / "task").iterdir():
        if path.is_file() and path.name not in {"shape_train.json", "source_manifest.json"}:
            assert path.read_bytes() == (full / "task" / path.name).read_bytes()
    for name in ("runtime.template.json", "ablation.json"):
        assert (clean / name).read_bytes() == (full / name).read_bytes()
    keys = set()
    for name in ("campaign.json", "ablation-campaign.json"):
        a, b = [json.loads((kit / name).read_text()) for kit in (clean, full)]
        keys.update((a.pop("creation_key"), b.pop("creation_key")))
        assert a == b
    assert len(keys) == 4

    payloads = []
    for kit in (clean, full):
        source = tmp_path / kit.name
        subprocess.run(["git", "clone", "--quiet", str(kit / "source.bundle"), str(source)],
                       check=True, capture_output=True)
        artifacts = LocalArtifactStore(tmp_path / (kit.name + "-artifacts"))
        contract = import_source_tree(kit / "task/source_manifest.json", source, artifacts)
        payloads.append(_files(artifacts.verify(contract.seed_digest).payload_path))
    clean_tree, full_tree = payloads
    assert "all M64 implementations" not in clean_tree.pop("UPSTREAM_PROVENANCE.json").decode()
    assert "all M64 implementations" in full_tree.pop("UPSTREAM_PROVENANCE.json").decode()
    assert clean_tree == full_tree
