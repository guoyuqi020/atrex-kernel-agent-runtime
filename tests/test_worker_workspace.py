"""Tests for isolated Attempt workspace assembly."""

from __future__ import annotations

import json
import os
from pathlib import Path

from conftest import NOW, digest

from atrex_runtime.artifacts.local import ArtifactKind, LocalArtifactStore
from atrex_runtime.domain.ids import (
    ArtifactDigest,
    new_attempt_id,
    new_campaign_id,
    new_epoch_id,
    new_kernel_agent_revision_id,
    new_kernel_revision_id,
    new_lineage_id,
)
from atrex_runtime.domain.models import (
    Attempt,
    AttemptStatus,
    BranchRole,
    Campaign,
    Dsl,
    Epoch,
    EpochStatus,
    KernelAgentRevision,
    KernelEvaluation,
    KernelRevision,
    Lineage,
    LineageStatus,
)
from atrex_runtime.ports import RunAttemptRequest
from atrex_runtime.registry.sqlite import SqliteRegistry
from atrex_runtime.workers.manifest import AttemptInputManifestV6
from atrex_runtime.workers.workspace import LocalAttemptWorkspaceAssembler


def _put_text_artifact(
    store: LocalArtifactStore,
    tmp_path: Path,
    name: str,
    kind: ArtifactKind,
) -> ArtifactDigest:
    source = tmp_path / f"source-{name}"
    source.mkdir()
    (source / f"{name}.txt").write_text(name)
    return store.put_directory(source, kind)


def test_workspace_materializes_complete_optimizer_repository(tmp_path: Path) -> None:
    registry = SqliteRegistry(tmp_path / "registry.sqlite")
    store = LocalArtifactStore(tmp_path / "artifacts")
    optimizer = _put_text_artifact(store, tmp_path, "optimizer", ArtifactKind.KERNEL_AGENT)
    kernel_digest = _put_text_artifact(store, tmp_path, "kernel", ArtifactKind.KERNEL)
    contract = store.put_json(
        {"schema_version": 1, "candidate_path": "kernel.txt"},
        ArtifactKind.EVALUATION_CONTRACT,
    )
    problem = store.put_json(
        {"schema_version": "atrex.agent_problem.v1", "objective": "vector add"},
        ArtifactKind.AGENT_PROBLEM,
    )

    campaign_id = new_campaign_id()
    agent_id = new_kernel_agent_revision_id()
    kernel_id = new_kernel_revision_id()
    lineage_id = new_lineage_id()
    epoch_id = new_epoch_id()
    attempt_id = new_attempt_id()
    evidence_source = tmp_path / "source-evidence"
    (evidence_source / "bootstrap").mkdir(parents=True)
    (evidence_source / "bootstrap/seed.txt").write_text("seed")
    (evidence_source / "bootstrap-metadata.json").write_text(
        json.dumps({"schema_version": 1, "source": "test"})
    )
    (evidence_source / "checkpoint.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "lineage_id": str(lineage_id),
                "through_epoch": 0,
                "previous_checkpoint_digest": None,
            }
        )
    )
    evidence = store.put_directory(evidence_source, ArtifactKind.EVIDENCE)
    attempt_source = tmp_path / "source-attempt-evidence"
    for name in ("attempts", "traces", "diffs", "reports"):
        (attempt_source / name).mkdir(parents=True, exist_ok=True)
    (attempt_source / "context.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "epoch_id": str(epoch_id),
                "attempt_id": str(attempt_id),
                "branch": "active",
                "challenger_ordinal": 0,
                "trajectory_ordinal": 1,
                "ordinal": 1,
                "epoch_evidence_checkpoint": str(evidence),
                "previous_attempt_ids": [],
            }
        )
    )
    (attempt_source / "lessons.json").write_text(
        json.dumps({"schema_version": 1, "annotations": []})
    )
    attempt_evidence = store.put_directory(
        attempt_source,
        ArtifactKind.ATTEMPT_EVIDENCE,
    )
    registry.insert_campaign(Campaign(campaign_id, "vector_add", "h100", contract, problem, NOW))
    registry.register_kernel_agent_revision(
        KernelAgentRevision(
            agent_id,
            None,
            "bootstrap:triton",
            Dsl.TRITON,
            optimizer,
            "bootstrap",
            NOW,
            source_provenance_digest=digest("source"),
        )
    )
    registry.register_kernel_revision(
        KernelRevision(
            kernel_id,
            None,
            kernel_digest,
            None,
            KernelEvaluation(True, 100.0, digest("gateway")),
            NOW,
        )
    )
    registry.insert_lineage(
        Lineage(
            id=lineage_id,
            campaign_id=campaign_id,
            dsl=Dsl.TRITON,
            hardware_target="h100",
            active_kernel_agent_revision_id=agent_id,
            best_kernel_revision_id=kernel_id,
            evidence_checkpoint=evidence,
            challenger_count=0,
            trajectories_per_branch=1,
            attempts_per_trajectory=2,
            next_epoch_number=1,
            status=LineageStatus.READY,
        )
    )
    registry.insert_epoch(
        Epoch(
            id=epoch_id,
            lineage_id=lineage_id,
            number=1,
            active_kernel_agent_revision_id=agent_id,
            challenger_kernel_agent_revision_ids=(),
            starting_kernel_revision_id=kernel_id,
            evidence_checkpoint=evidence,
            challenger_count=0,
            trajectories_per_branch=1,
            attempts_per_trajectory=2,
            status=EpochStatus.RUNNING,
            winner_kernel_agent_revision_id=None,
            best_kernel_revision_id=None,
            created_at=NOW,
            completed_at=None,
        )
    )
    registry.insert_attempt(
        Attempt(
            id=attempt_id,
            epoch_id=epoch_id,
            branch=BranchRole.ACTIVE,
            challenger_ordinal=0,
            trajectory_ordinal=1,
            ordinal=1,
            kernel_agent_revision_id=agent_id,
            input_kernel_revision_id=kernel_id,
            attempt_evidence_digest=attempt_evidence,
            output_kernel_revision_id=None,
            accepted_as_branch_best=False,
            status=AttemptStatus.RUNNING,
            infrastructure_failures=0,
            recovery_generation=0,
            authority_started_at=NOW,
            failure_reason=None,
            created_at=NOW,
            completed_at=None,
        )
    )
    request = RunAttemptRequest(
        attempt_id,
        agent_id,
        kernel_id,
        evidence,
        attempt_evidence,
        Dsl.TRITON,
    )
    assembler = LocalAttemptWorkspaceAssembler(tmp_path / "workspaces", registry, store)

    first = assembler.prepare(request)
    second = assembler.prepare(request)
    manifest = AttemptInputManifestV6.from_json_bytes(first.manifest_path.read_bytes())

    assert first.root != second.root
    assert first.session_id != second.session_id
    assert first.session_root != second.session_root
    assert manifest.attempt_id == request.attempt_id
    assert manifest.context.operator == "vector_add"
    assert manifest.context.hardware_target == "h100"
    assert (first.root / "input/agent-problem/value.json").is_file()
    assert (first.root / "agent/optimizer/optimizer.txt").read_text() == "optimizer"
    evidence_view = first.root / "input/evidence"
    assert json.loads((evidence_view / "manifest.json").read_text())["role"] == "optimizer"
    assert (evidence_view / "bootstrap/seed.txt").read_text() == "seed"
    assert (evidence_view / "epochs/00000001/attempts").is_dir()
    assert not (first.root / "input/attempt-evidence").exists()
    assert list((first.root / "agent").iterdir()) == [first.root / "agent/optimizer"]
    working_file = first.root / "work/kernel/kernel.txt"
    assert os.stat(working_file).st_mode & 0o200
    assert not (os.stat(first.root / "input/kernel/kernel.txt").st_mode & 0o200)
    registry.close()
