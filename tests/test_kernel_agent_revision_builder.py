"""Full-repository Kernel Agent revision validation and sealing tests."""

from __future__ import annotations

import json
import os
import sys
from dataclasses import replace
from pathlib import Path

import pytest
from conftest import NOW, kernel_agent_limits

from atrex_runtime.artifacts.local import LocalArtifactStore
from atrex_runtime.domain.ids import new_epoch_id, new_kernel_agent_revision_id
from atrex_runtime.domain.models import Dsl, KernelAgentRevision
from atrex_runtime.kernel_agents import KernelAgentRevisionBuilder
from atrex_runtime.kernel_agents.revision import KernelAgentBundleManifestV1
from atrex_runtime.kernel_agents.workflow import SandboxedAgentWorkflowRunner
from atrex_runtime.ports import RunAgentWorkflowRequest
from atrex_runtime.workers.launcher import CleanEnvironmentLauncher


def _source(tmp_path: Path) -> Path:
    root = tmp_path / "agent"
    prompt = root / "prompts/episode.md"
    prompt.parent.mkdir(parents=True)
    docs = root / "docs/design.md"
    docs.parent.mkdir(parents=True)
    (root / "src").mkdir()
    prompt.write_text("Optimize through Runtime tools.\n", encoding="utf-8")
    docs.write_text("# Agent design\n", encoding="utf-8")
    (root / "atrex-bundle.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "bundle_format": "atrex-kernel-agent-bundle-v1",
                "entrypoint": {
                    "command": "src/main.py",
                },
            }
        ),
        encoding="utf-8",
    )
    (root / "src/main.py").write_text("def optimize(): ...\n", encoding="utf-8")
    return root


def _add_executable_workflow(source: Path) -> None:
    manifest_path = source / "atrex-bundle.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["workflow"] = {"command": "workflow/main.py"}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    workflow = source / "workflow"
    workflow.mkdir()
    (workflow / "main.py").write_text(
        """import json
import sys

context = json.loads(sys.stdin.readline())
assert context["operation"] == "run_epoch"
print(json.dumps({
    "request_id": "call-1",
    "operation": "probe",
    "arguments": {"epoch_number": context["context"]["epoch_number"]},
}), flush=True)
response = json.loads(sys.stdin.readline())
assert response == {"request_id": "call-1", "ok": True, "result": {"accepted": True}}
""",
        encoding="utf-8",
    )


class _ProbeWorkflowOperations:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []

    async def execute_workflow_operation(
        self,
        operation: str,
        arguments: dict[str, object],
    ) -> dict[str, object]:
        self.calls.append((operation, arguments))
        assert operation == "probe"
        assert arguments["epoch_number"] == 2
        assert isinstance(arguments["_runtime_workflow_program_sha256"], str)
        return {"accepted": True}


def test_builder_seals_complete_repository(tmp_path: Path) -> None:
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    candidate = KernelAgentRevisionBuilder(
        artifacts,
        limits=kernel_agent_limits(),
    ).build_candidate(_source(tmp_path), Dsl.TRITON)

    assert candidate.dsl is Dsl.TRITON
    stored = artifacts.verify(candidate.optimizer_digest).payload_path
    assert (stored / "src/main.py").is_file()
    assert (stored / "prompts/episode.md").is_file()
    assert (stored / "docs/design.md").is_file()


def test_builder_derives_revision_with_selected_workflow(tmp_path: Path) -> None:
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    source = _source(tmp_path)
    _add_executable_workflow(source)
    builder = KernelAgentRevisionBuilder(artifacts, limits=kernel_agent_limits())
    base = builder.build_candidate(source, Dsl.TRITON)

    selected = builder.select_workflow(
        base.optimizer_digest,
        Dsl.TRITON,
        "workflow/pool_3.py",
    )

    assert selected.optimizer_digest != base.optimizer_digest
    base_manifest = KernelAgentBundleManifestV1.from_file(
        artifacts.verify(base.optimizer_digest).payload_path / "atrex-bundle.json"
    )
    selected_manifest = KernelAgentBundleManifestV1.from_file(
        artifacts.verify(selected.optimizer_digest).payload_path / "atrex-bundle.json"
    )
    assert base_manifest.workflow is not None
    assert base_manifest.workflow.command == "workflow/main.py"
    assert selected_manifest.workflow is not None
    assert selected_manifest.workflow.command == "workflow/main.py"
    assert (
        artifacts.verify(selected.optimizer_digest).payload_path / "workflow/main.py"
    ).read_bytes() == (
        Path(__file__).resolve().parents[1] / "src/atrex_runtime/workflow_templates/pool_3.py"
    ).read_bytes()
    assert not (
        artifacts.verify(selected.optimizer_digest).payload_path / "workflow/pool_3.py"
    ).exists()

    with pytest.raises(ValueError, match="Selected Agent Workflow command file does not exist"):
        builder.select_workflow(
            base.optimizer_digest,
            Dsl.TRITON,
            "workflow/missing.py",
        )


def test_builder_rejects_alternative_workflow_entries(tmp_path: Path) -> None:
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    source = _source(tmp_path)
    _add_executable_workflow(source)
    (source / "workflow/pool_3.py").write_text("print('pool')\n", encoding="utf-8")

    with pytest.raises(ValueError, match="alternative Workflow entries are not allowed"):
        KernelAgentRevisionBuilder(
            artifacts,
            limits=kernel_agent_limits(),
        ).build_candidate(source, Dsl.TRITON)


def test_builder_moves_custom_selected_entry_to_main(tmp_path: Path) -> None:
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    source = _source(tmp_path)
    _add_executable_workflow(source)
    (source / "workflow/custom.py").write_text("print('custom')\n", encoding="utf-8")
    builder = KernelAgentRevisionBuilder(artifacts, limits=kernel_agent_limits())
    base = builder.build_candidate(source, Dsl.TRITON)

    selected = builder.select_workflow(
        base.optimizer_digest,
        Dsl.TRITON,
        "workflow/custom.py",
    )

    root = artifacts.verify(selected.optimizer_digest).payload_path
    assert (root / "workflow/main.py").read_text() == "print('custom')\n"
    assert not (root / "workflow/custom.py").exists()


@pytest.mark.anyio
async def test_runtime_executes_agent_workflow_against_bounded_services(
    tmp_path: Path,
) -> None:
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    source = _source(tmp_path)
    _add_executable_workflow(source)
    candidate = KernelAgentRevisionBuilder(
        artifacts,
        limits=kernel_agent_limits(),
    ).build_candidate(source, Dsl.TRITON)
    revision = KernelAgentRevision(
        id=new_kernel_agent_revision_id(),
        parent_id=None,
        creation_key="bootstrap:executable-workflow",
        dsl=Dsl.TRITON,
        optimizer_digest=candidate.optimizer_digest,
        created_by="bootstrap",
        created_at=NOW,
        source_provenance_digest=candidate.optimizer_digest,
    )
    operations = _ProbeWorkflowOperations()

    await SandboxedAgentWorkflowRunner(
        artifacts,
        CleanEnvironmentLauncher(Path("/usr/bin/env")),
        tmp_path / "workflow-runs",
        command_prefix=(sys.executable,),
    ).run(
        RunAgentWorkflowRequest(
            revision=revision,
            epoch_id=new_epoch_id(),
            epoch_number=2,
            max_challengers=1,
            optimizer_attempt_budget=6,
        ),
        operations,
    )

    assert len(operations.calls) == 1
    assert list((tmp_path / "workflow-runs").rglob("protocol.jsonl"))


@pytest.mark.parametrize("name", ("prompts", "skills", "tools"))
def test_builder_seals_top_level_adaptive_state_seeds(
    tmp_path: Path,
    name: str,
) -> None:
    source = _source(tmp_path)
    seed = source / name
    seed.mkdir(exist_ok=True)
    (seed / "state.md").write_text("initial state\n", encoding="utf-8")
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    candidate = KernelAgentRevisionBuilder(artifacts, limits=kernel_agent_limits()).build_candidate(
        source, Dsl.TRITON
    )
    stored = artifacts.verify(candidate.optimizer_digest).payload_path
    assert (stored / name / "state.md").read_text() == "initial state\n"


def test_builder_allows_non_entry_repository_content(tmp_path: Path) -> None:
    source = _source(tmp_path)
    (source / "docs").mkdir(exist_ok=True)
    (source / "docs/design.md").write_text("supporting documentation\n")
    candidate = KernelAgentRevisionBuilder(
        LocalArtifactStore(tmp_path / "artifacts"),
        limits=kernel_agent_limits(),
    ).build_candidate(source, Dsl.CUDA)

    assert candidate.dsl is Dsl.CUDA


def test_builder_excludes_generated_cache_content_from_revision(tmp_path: Path) -> None:
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    source = _source(tmp_path)
    baseline = KernelAgentRevisionBuilder(
        artifacts,
        limits=kernel_agent_limits(),
    ).build_candidate(source, Dsl.TRITON)
    (source / "src/__pycache__").mkdir()
    (source / "src/__pycache__/main.cpython-314.pyc").write_bytes(b"generated")
    (source / ".pytest_cache").mkdir()
    (source / ".pytest_cache/state").write_text("generated", encoding="utf-8")
    (source / ".coverage").write_text("generated", encoding="utf-8")

    with_cache = KernelAgentRevisionBuilder(
        artifacts,
        limits=kernel_agent_limits(),
    ).build_candidate(source, Dsl.TRITON)

    assert with_cache.optimizer_digest == baseline.optimizer_digest
    stored = artifacts.verify(with_cache.optimizer_digest).payload_path
    assert not (stored / "src/__pycache__").exists()
    assert not (stored / ".pytest_cache").exists()
    assert not (stored / ".coverage").exists()


def test_builder_rejects_runtime_selected_agent_adapter(tmp_path: Path) -> None:
    source = _source(tmp_path)
    manifest_path = source / "atrex-bundle.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["entrypoint"]["dsh"] = {
        "prompt": "prompts/episode.md",
        "skills": ["skills/example/SKILL.md"],
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        KernelAgentRevisionBuilder(
            LocalArtifactStore(tmp_path / "artifacts"),
            limits=kernel_agent_limits(),
        ).build_candidate(source, Dsl.TRITON)


def test_builder_requires_current_bundle_manifest_name_and_format(tmp_path: Path) -> None:
    source = _source(tmp_path)
    manifest_path = source / "atrex-bundle.json"
    legacy_path = source / "atrex-optimizer.json"
    manifest_path.rename(legacy_path)
    builder = KernelAgentRevisionBuilder(
        LocalArtifactStore(tmp_path / "artifacts"), limits=kernel_agent_limits()
    )

    with pytest.raises(ValueError, match="Bundle manifest is unavailable"):
        builder.build_candidate(source, Dsl.TRITON)

    legacy_path.rename(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["bundle_format"] = "atrex-optimizer-repository-v1"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="Bundle manifest is invalid"):
        builder.build_candidate(source, Dsl.TRITON)


def test_builder_rejects_missing_or_oversized_entry_file(tmp_path: Path) -> None:
    source = _source(tmp_path)
    command = source / "src/main.py"
    command.write_text("x" * 20)
    builder = KernelAgentRevisionBuilder(
        LocalArtifactStore(tmp_path / "artifacts"),
        limits=replace(kernel_agent_limits(), max_entrypoint_bytes=16),
    )

    with pytest.raises(ValueError, match="Optimizer command file exceeds byte limit"):
        builder.build_candidate(source, Dsl.TRITON)

    command.unlink()
    with pytest.raises(ValueError, match="Optimizer command file does not exist"):
        builder.build_candidate(source, Dsl.TRITON)


def test_builder_rejects_git_metadata_and_links(tmp_path: Path) -> None:
    source = _source(tmp_path)
    (source / ".git").write_text("gitdir: elsewhere\n")
    builder = KernelAgentRevisionBuilder(
        LocalArtifactStore(tmp_path / "artifacts"), limits=kernel_agent_limits()
    )

    with pytest.raises(ValueError, match="Git metadata"):
        builder.build_candidate(source, Dsl.TRITON)

    (source / ".git").unlink()
    try:
        os.symlink(source / "src/main.py", source / "linked.py")
    except (OSError, NotImplementedError):
        pytest.skip("symbolic links are unavailable")
    with pytest.raises(ValueError, match="symbolic links"):
        builder.build_candidate(source, Dsl.TRITON)


def test_challenger_requires_same_dsl_and_repository_change(tmp_path: Path) -> None:
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    builder = KernelAgentRevisionBuilder(artifacts, limits=kernel_agent_limits())
    first = builder.build_candidate(_source(tmp_path), Dsl.TRITON)
    parent = KernelAgentRevision(
        id=new_kernel_agent_revision_id(),
        parent_id=None,
        creation_key="bootstrap:test",
        dsl=first.dsl,
        optimizer_digest=first.optimizer_digest,
        created_by="bootstrap",
        created_at=NOW,
        source_provenance_digest=first.optimizer_digest,
    )

    with pytest.raises(ValueError, match="no Optimizer repository changes"):
        builder.validate_challenger(parent, first)

    changed_root = _source(tmp_path / "changed")
    (changed_root / "src/main.py").write_text("def optimize(): return 1\n")
    changed = builder.build_candidate(changed_root, Dsl.TRITON)
    builder.validate_challenger(parent, changed)

    wrong_dsl = builder.build_candidate(changed_root, Dsl.CUDA)
    with pytest.raises(ValueError, match="lineage DSL"):
        builder.validate_challenger(parent, wrong_dsl)
