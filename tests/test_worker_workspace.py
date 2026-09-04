"""Tests for isolated Attempt workspace assembly."""

from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import pytest
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
from atrex_runtime.workers.manifest import AttemptInputManifestV9
from atrex_runtime.workers.workspace import (
    REUSABLE_AGENT_DIRECTORIES,
    LocalAttemptWorkspaceAssembler,
    copy_reusable_agent_state,
    ensure_reusable_directories,
    initialize_reusable_agent_state,
    materialize_reusable_agent_state_snapshot,
    validate_reusable_agent_state_seed,
)


def _put_text_artifact(
    store: LocalArtifactStore,
    tmp_path: Path,
    name: str,
    kind: ArtifactKind,
) -> ArtifactDigest:
    source = tmp_path / f"source-{name}"
    source.mkdir(exist_ok=True)
    (source / f"{name}.txt").write_text(name)
    return store.put_directory(source, kind)


def test_next_active_and_evolver_share_winning_trajectory_terminal_state(
    tmp_path: Path,
) -> None:
    """Active selection uses the same terminal checkpoint exercised by Evolver tests."""
    winner_id = new_kernel_agent_revision_id()
    lineage_id = new_lineage_id()
    previous_epoch_id = new_epoch_id()
    next_epoch_id = new_epoch_id()
    starting_kernel_id = new_kernel_revision_id()
    best_kernel_id = new_kernel_revision_id()
    later_kernel_id = new_kernel_revision_id()
    best_attempt_id = new_attempt_id()
    final_attempt_id = new_attempt_id()
    start_state = digest("winning-trajectory-start")
    best_state = digest("winning-trajectory-best")
    terminal_state = digest("winning-trajectory-terminal")
    evidence = digest("evidence")

    previous = Epoch(
        id=previous_epoch_id,
        lineage_id=lineage_id,
        number=1,
        active_kernel_agent_revision_id=winner_id,
        challenger_kernel_agent_revision_ids=(),
        starting_kernel_revision_id=starting_kernel_id,
        evidence_checkpoint=evidence,
        challenger_count=0,
        trajectories_per_branch=1,
        attempts_per_trajectory=2,
        status=EpochStatus.COMPLETED,
        winner_kernel_agent_revision_id=winner_id,
        best_kernel_revision_id=best_kernel_id,
        created_at=NOW,
        completed_at=NOW,
    )
    next_epoch = Epoch(
        id=next_epoch_id,
        lineage_id=lineage_id,
        number=2,
        active_kernel_agent_revision_id=winner_id,
        challenger_kernel_agent_revision_ids=(),
        starting_kernel_revision_id=best_kernel_id,
        evidence_checkpoint=evidence,
        challenger_count=0,
        trajectories_per_branch=1,
        attempts_per_trajectory=1,
        status=EpochStatus.RUNNING,
        winner_kernel_agent_revision_id=None,
        best_kernel_revision_id=None,
        created_at=NOW,
        completed_at=None,
    )

    def attempt(
        attempt_id: object,
        ordinal: int,
        output_kernel_id: object,
        input_state: ArtifactDigest,
        terminal: ArtifactDigest,
        *,
        retained: bool,
    ) -> Attempt:
        return Attempt(
            id=attempt_id,
            epoch_id=previous_epoch_id,
            branch=BranchRole.ACTIVE,
            challenger_ordinal=0,
            trajectory_ordinal=1,
            ordinal=ordinal,
            kernel_agent_revision_id=winner_id,
            input_kernel_revision_id=starting_kernel_id,
            attempt_evidence_digest=digest(f"attempt-evidence-{ordinal}"),
            output_kernel_revision_id=output_kernel_id,
            accepted_as_branch_best=retained,
            status=AttemptStatus.COMPLETED,
            infrastructure_failures=0,
            recovery_generation=0,
            authority_started_at=NOW,
            failure_reason=None,
            created_at=NOW,
            completed_at=NOW,
            runtime_state_digest=terminal,
            input_runtime_state_digest=input_state,
        )

    attempts = (
        attempt(
            best_attempt_id,
            1,
            best_kernel_id,
            start_state,
            best_state,
            retained=True,
        ),
        attempt(
            final_attempt_id,
            2,
            later_kernel_id,
            best_state,
            terminal_state,
            retained=False,
        ),
    )
    kernels = {
        best_kernel_id: KernelRevision(
            best_kernel_id,
            starting_kernel_id,
            digest("best-kernel"),
            best_attempt_id,
            KernelEvaluation(True, 10.0, digest("best-result")),
            NOW,
        ),
        later_kernel_id: KernelRevision(
            later_kernel_id,
            best_kernel_id,
            digest("later-kernel"),
            final_attempt_id,
            KernelEvaluation(True, 12.0, digest("later-result")),
            NOW,
        ),
    }
    registry = Mock()
    registry.find_epoch.return_value = previous
    registry.list_attempts.return_value = list(attempts)
    registry.get_kernel_revision.side_effect = kernels.__getitem__
    assembler = LocalAttemptWorkspaceAssembler(
        tmp_path / "workspaces",
        registry,
        LocalArtifactStore(tmp_path / "artifacts"),
    )

    assert assembler._active_branch_seed(next_epoch) == terminal_state


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
    (evidence_source / "bootstrap/report.json").write_text(json.dumps({"status": "baseline_ready"}))
    (evidence_source / "bootstrap/conversation.jsonl").write_text("{}\n")
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
    bootstrap_state = (
        tmp_path / "workspaces/.reusable" / str(lineage_id) / str(agent_id) / "bootstrap"
    )
    (bootstrap_state / "skills").mkdir(parents=True)
    (bootstrap_state / "tools").mkdir()
    (bootstrap_state / "skills/bootstrap.md").write_text("baseline lesson\n")
    (bootstrap_state / "tools/README.md").write_text("# Bootstrap tools\n")

    first = assembler.prepare(request)
    recorded_input_state = registry.get_attempt(attempt_id).input_runtime_state_digest
    assert recorded_input_state is not None
    assert store.verify(recorded_input_state).kind is ArtifactKind.KERNEL_AGENT_RUNTIME_STATE
    assert (first.root / "skills/bootstrap.md").read_text() == "baseline lesson\n"
    assert (first.root / "tools/README.md").read_text() == "# Bootstrap tools\n"
    (first.root / "skills/vector-load.md").write_text("reuse aligned loads\n")
    (first.root / "tools/inspect_kernel.py").write_text("print('inspect')\n")
    (first.root / "tools/README.md").write_text(
        "# Reusable tools\n\n## inspect_kernel.py\n\nRun with Python.\n"
    )
    first.persist_reusable_directories()
    second = assembler.prepare(request)
    manifest = AttemptInputManifestV9.from_json_bytes(first.manifest_path.read_bytes())

    assert first.root != second.root
    assert first.session_id != second.session_id
    assert first.session_root != second.session_root
    assert (second.root / "skills/vector-load.md").read_text() == "reuse aligned loads\n"
    assert (second.root / "tools/inspect_kernel.py").read_text() == "print('inspect')\n"
    assert "inspect_kernel.py" in (second.root / "tools/README.md").read_text()
    child_id = new_kernel_agent_revision_id()
    child_revision = KernelAgentRevision(
        child_id,
        agent_id,
        "epoch:test:challenger:1",
        Dsl.TRITON,
        optimizer,
        "evolver",
        NOW,
        evolution_trace_digest=digest("legacy-evolution"),
    )
    child_state, _lock = assembler._persistent_root(
        lineage_id=lineage_id,
        revision=child_revision,
        trajectory_ordinal=1,
    )
    assert (child_state / "skills/vector-load.md").read_text() == "reuse aligned loads\n"
    assert (child_state / "tools/inspect_kernel.py").is_file()
    assert manifest.attempt_id == request.attempt_id
    assert manifest.context.operator == "vector_add"
    assert manifest.context.hardware_target == "h100"
    assert first.manifest_path == first.root / ".runtime/attempt.json"
    assert first.manifest_path.is_file()
    assert not (first.root / "attempt.json").exists()
    assert (first.root / ".runtime/agent-problem.json").is_file()
    assert not (first.root / "input/agent-problem").exists()
    assert (first.root / "agent/optimizer/optimizer.txt").read_text() == "optimizer"
    evidence_view = first.root / "input/evidence"
    assert json.loads((first.root / ".runtime/evidence-manifest.json").read_text())["role"] == (
        "optimizer"
    )
    assert not (evidence_view / "manifest.json").exists()
    assert not (evidence_view / "instructions.md").exists()
    assert json.loads((evidence_view / "bootstrap/report.json").read_text()) == {
        "status": "baseline_ready"
    }
    assert (evidence_view / "bootstrap/conversation.jsonl").read_text() == "{}\n"
    assert (evidence_view / "epochs/00000001/trajectories/00000001/attempts").is_dir()
    assert not (evidence_view / "epochs/00000001/attempts").exists()
    assert not (first.root / "input/attempt-evidence").exists()
    assert list((first.root / "agent").iterdir()) == [first.root / "agent/optimizer"]
    working_file = first.root / "work/kernel/kernel.txt"
    assert os.stat(working_file).st_mode & 0o200
    assert not (os.stat(first.root / "input/kernel/kernel.txt").st_mode & 0o200)
    assert not (first.root / "reference").exists()
    registry.close()


def _single_trajectory_workspace(
    tmp_path: Path,
    registry: SqliteRegistry,
    store: LocalArtifactStore,
    *,
    ephemeral_agent_state: bool,
    bootstrap_source_lineage_id: object | None = None,
    first_epoch_same_agent: bool = False,
    core_seed: bool = False,
) -> tuple[LocalAttemptWorkspaceAssembler, RunAttemptRequest, str, str]:
    """Assemble the least Registry state one prepare() call accepts."""
    optimizer = _put_text_artifact(store, tmp_path, "optimizer", ArtifactKind.KERNEL_AGENT)
    if core_seed:
        source = tmp_path / "source-optimizer"
        for name in REUSABLE_AGENT_DIRECTORIES:
            (source / name).mkdir()
            (source / name / "README.md").write_text(f"Initial {name} index")
            (source / name / "seed.md").write_text(f"Initial {name}")
        optimizer = store.put_directory(source, ArtifactKind.KERNEL_AGENT)
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
    (evidence_source / "bootstrap/report.json").write_text(json.dumps({"status": "baseline_ready"}))
    (evidence_source / "bootstrap/conversation.jsonl").write_text("{}\n")
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
    attempt_evidence = store.put_directory(attempt_source, ArtifactKind.ATTEMPT_EVIDENCE)
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
    mirrored = Lineage(
        id=lineage_id,
        campaign_id=campaign_id,
        dsl=Dsl.TRITON,
        hardware_target="h100",
        active_kernel_agent_revision_id=agent_id,
        best_kernel_revision_id=kernel_id,
        evidence_checkpoint=evidence,
        challenger_count=1 if first_epoch_same_agent else 0,
        first_epoch_same_agent=first_epoch_same_agent,
        trajectories_per_branch=1,
        attempts_per_trajectory=2,
        next_epoch_number=1,
        status=LineageStatus.READY,
        ephemeral_agent_state=ephemeral_agent_state,
        bootstrap_source_lineage_id=bootstrap_source_lineage_id,
    )
    deposit_agent_id = str(agent_id)
    if bootstrap_source_lineage_id is not None:
        source_agent_id = new_kernel_agent_revision_id()
        source_kernel_id = new_kernel_revision_id()
        registry.register_kernel_agent_revision(
            KernelAgentRevision(
                source_agent_id,
                None,
                "bootstrap:triton:source",
                Dsl.TRITON,
                optimizer,
                "bootstrap",
                NOW,
                source_provenance_digest=digest("source"),
            )
        )
        registry.register_kernel_revision(
            KernelRevision(
                source_kernel_id,
                None,
                kernel_digest,
                None,
                KernelEvaluation(True, 100.0, digest("gateway")),
                NOW,
            )
        )
        registry.insert_lineage(
            replace(
                mirrored,
                id=bootstrap_source_lineage_id,
                active_kernel_agent_revision_id=source_agent_id,
                best_kernel_revision_id=source_kernel_id,
                ephemeral_agent_state=False,
                bootstrap_source_lineage_id=None,
            )
        )
        deposit_agent_id = str(source_agent_id)
    registry.insert_lineage(mirrored)
    registry.insert_epoch(
        Epoch(
            id=epoch_id,
            lineage_id=lineage_id,
            number=1,
            active_kernel_agent_revision_id=agent_id,
            challenger_kernel_agent_revision_ids=(agent_id,) if first_epoch_same_agent else (),
            starting_kernel_revision_id=kernel_id,
            evidence_checkpoint=evidence,
            challenger_count=1 if first_epoch_same_agent else 0,
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
    return assembler, request, str(lineage_id), deposit_agent_id


@pytest.mark.parametrize("directory", REUSABLE_AGENT_DIRECTORIES)
def test_event_only_attempt_never_inherits_agent_state(tmp_path: Path, directory: str) -> None:
    """The ablation arm must start identical every time, including after a physical retry."""
    registry = SqliteRegistry(tmp_path / "registry.sqlite")
    store = LocalArtifactStore(tmp_path / "artifacts")
    assembler, request, lineage_id, _attempt_id = _single_trajectory_workspace(
        tmp_path,
        registry,
        store,
        ephemeral_agent_state=True,
    )

    first = assembler.prepare(request)
    assert [p.name for p in (first.root / "skills").iterdir()] == ["README.md"]
    assert (first.root / "tools/README.md").is_file()
    assert first.persistent_state_root is None
    assert first.persistent_lock_path is None

    (first.root / directory / "learned.md").write_text("reuse aligned loads\n")
    (first.root / directory / "README.md").write_text("custom index\n")
    first.persist_reusable_directories()
    # The Session seals its own post-Session state; a physical retry must still start empty.
    registry.record_attempt_runtime_state(
        request.attempt_id,
        first.seal_runtime_state(store),
    )
    retried = assembler.prepare(request)

    assert [p.name for p in (retried.root / "skills").iterdir()] == ["README.md"]
    assert not (retried.root / directory / "learned.md").exists()
    assert (retried.root / directory / "README.md").read_text() != "custom index\n"
    assert not (tmp_path / "workspaces/.reusable" / lineage_id).exists()
    registry.close()


def test_initial_replica_has_independent_persistent_state_and_retry(tmp_path: Path) -> None:
    with SqliteRegistry(tmp_path / "registry.sqlite") as registry:
        store = LocalArtifactStore(tmp_path / "artifacts")
        assembler, request, _lineage, _agent = _single_trajectory_workspace(
            tmp_path,
            registry,
            store,
            ephemeral_agent_state=False,
            first_epoch_same_agent=True,
        )
        active = assembler.prepare(request)
        (active.root / "skills/active.md").write_text("active only")
        active.persist_reusable_directories()
        registry.record_attempt_runtime_state(request.attempt_id, active.seal_runtime_state(store))
        attempt = registry.get_attempt(request.attempt_id)
        replica_id = new_attempt_id()
        context = json.loads((tmp_path / "source-attempt-evidence/context.json").read_text())
        context.update(attempt_id=str(replica_id), branch="challenger", challenger_ordinal=1)
        replica_evidence = tmp_path / "replica-evidence"
        replica_evidence.mkdir()
        (replica_evidence / "context.json").write_text(json.dumps(context))
        (replica_evidence / "lessons.json").write_text('{"schema_version": 1, "annotations": []}')
        evidence_digest = store.put_directory(replica_evidence, ArtifactKind.ATTEMPT_EVIDENCE)
        registry.insert_attempt(
            replace(
                attempt,
                id=replica_id,
                branch=BranchRole.CHALLENGER,
                challenger_ordinal=1,
                attempt_evidence_digest=evidence_digest,
                runtime_state_digest=None,
                input_runtime_state_digest=None,
            )
        )
        replica_request = replace(
            request, attempt_id=replica_id, attempt_evidence_digest=evidence_digest
        )
        replica = assembler.prepare(replica_request)
        assert replica.persistent_state_root != active.persistent_state_root
        assert not (replica.root / "skills/active.md").exists()
        (replica.root / "skills/replica.md").write_text("replica only")
        replica.persist_reusable_directories()
        registry.record_attempt_runtime_state(replica_id, replica.seal_runtime_state(store))
        active_retry = assembler.prepare(request)
        replica_retry = assembler.prepare(replica_request)
        assert (active_retry.root / "skills/active.md").is_file()
        assert not (active_retry.root / "skills/replica.md").exists()
        assert (replica_retry.root / "skills/replica.md").is_file()
        assert not (replica_retry.root / "skills/active.md").exists()


@pytest.mark.parametrize("directory", REUSABLE_AGENT_DIRECTORIES)
def test_a_normal_retry_does_inherit_agent_state(tmp_path: Path, directory: str) -> None:
    """Pin the inheritance the flag suppresses, so the ablation assertions cannot go vacuous."""
    registry = SqliteRegistry(tmp_path / "registry.sqlite")
    store = LocalArtifactStore(tmp_path / "artifacts")
    assembler, request, lineage_id, _attempt_id = _single_trajectory_workspace(
        tmp_path,
        registry,
        store,
        ephemeral_agent_state=False,
    )

    first = assembler.prepare(request)
    (first.root / directory / "learned.md").write_text("reuse aligned loads\n")
    (first.root / directory / "README.md").write_text("learned.md: aligned loads\n")
    first.persist_reusable_directories()
    registry.record_attempt_runtime_state(
        request.attempt_id,
        first.seal_runtime_state(store),
    )
    retried = assembler.prepare(request)

    assert (retried.root / directory / "learned.md").read_text() == "reuse aligned loads\n"
    assert (retried.root / directory / "README.md").read_text() == "learned.md: aligned loads\n"
    assert (tmp_path / "workspaces/.reusable" / lineage_id).is_dir()
    registry.close()


def test_retaining_clone_inherits_the_source_lineage_bootstrap_state(tmp_path: Path) -> None:
    """A cloned arm has no Bootstrap Attempt, so it must read the source Lineage's deposit."""
    registry = SqliteRegistry(tmp_path / "registry.sqlite")
    store = LocalArtifactStore(tmp_path / "artifacts")
    source_lineage_id = new_lineage_id()
    assembler, request, lineage_id, agent_id = _single_trajectory_workspace(
        tmp_path,
        registry,
        store,
        ephemeral_agent_state=False,
        bootstrap_source_lineage_id=source_lineage_id,
    )

    deposit = tmp_path / "workspaces/.reusable" / str(source_lineage_id) / agent_id / "bootstrap"
    (deposit / "skills").mkdir(parents=True)
    (deposit / "tools").mkdir()
    (deposit / "skills/bootstrap.md").write_text("shared baseline lesson\n")
    (deposit / "tools/refcheck.py").write_text("# shared checker\n")
    (deposit / "tools/README.md").write_text("# Bootstrap tools\n")

    prepared = assembler.prepare(request)

    assert (prepared.root / "skills/bootstrap.md").read_text() == "shared baseline lesson\n"
    assert (prepared.root / "tools/refcheck.py").read_text() == "# shared checker\n"
    assert (tmp_path / "workspaces/.reusable" / lineage_id).is_dir()
    registry.close()


def test_ephemeral_clone_still_ignores_the_source_lineage_bootstrap_state(tmp_path: Path) -> None:
    """Inheriting for a retaining clone must not leak into the always-empty arms."""
    registry = SqliteRegistry(tmp_path / "registry.sqlite")
    store = LocalArtifactStore(tmp_path / "artifacts")
    source_lineage_id = new_lineage_id()
    assembler, request, _lineage_id, agent_id = _single_trajectory_workspace(
        tmp_path,
        registry,
        store,
        ephemeral_agent_state=True,
        bootstrap_source_lineage_id=source_lineage_id,
    )

    deposit = tmp_path / "workspaces/.reusable" / str(source_lineage_id) / agent_id / "bootstrap"
    (deposit / "skills").mkdir(parents=True)
    (deposit / "skills/bootstrap.md").write_text("shared baseline lesson\n")

    prepared = assembler.prepare(request)

    assert [p.name for p in (prepared.root / "skills").iterdir()] == ["README.md"]
    assert (prepared.root / "tools/README.md").is_file()
    assert not (prepared.root / "tools/refcheck.py").exists()
    registry.close()


def test_evolved_revision_seeds_trajectory_from_candidate_runtime_state(tmp_path: Path) -> None:
    registry = SqliteRegistry(tmp_path / "registry.sqlite")
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    state = tmp_path / "candidate-state"
    (state / "skills").mkdir(parents=True)
    (state / "tools").mkdir()
    (state / "skills/evolved.md").write_text("candidate-selected lesson\n")
    (state / "tools/README.md").write_text("# Candidate tools\n")
    state_digest = artifacts.put_directory(
        state,
        ArtifactKind.KERNEL_AGENT_RUNTIME_STATE,
    )
    trace_digest = artifacts.put_json(
        {
            "schema_version": 9,
            "candidate": {
                "optimizer_digest": _put_text_artifact(
                    artifacts, tmp_path, "optimizer", ArtifactKind.KERNEL_AGENT
                ),
                "runtime_state_digest": state_digest,
            },
        },
        ArtifactKind.EVOLUTION,
    )
    revision = KernelAgentRevision(
        new_kernel_agent_revision_id(),
        new_kernel_agent_revision_id(),
        "epoch:test:challenger:1",
        Dsl.TRITON,
        _put_text_artifact(artifacts, tmp_path, "optimizer", ArtifactKind.KERNEL_AGENT),
        "evolver",
        NOW,
        evolution_trace_digest=trace_digest,
        runtime_state_digest=state_digest,
    )
    assembler = LocalAttemptWorkspaceAssembler(tmp_path / "workspaces", registry, artifacts)
    lineage_id = new_lineage_id()

    state, _lock = assembler._persistent_root(
        lineage_id=lineage_id,
        revision=revision,
        trajectory_ordinal=1,
    )
    second_state, _second_lock = assembler._persistent_root(
        lineage_id=lineage_id,
        revision=revision,
        trajectory_ordinal=2,
    )

    assert (state / "skills/evolved.md").read_text() == "candidate-selected lesson\n"
    assert (state / "tools/README.md").read_text() == "# Candidate tools\n"
    assert (second_state / "skills/evolved.md").read_text() == "candidate-selected lesson\n"
    assert (second_state / "tools/README.md").read_text() == "# Candidate tools\n"
    registry.close()


def test_a_runtime_state_seed_without_skills_still_seeds_a_trajectory(tmp_path: Path) -> None:
    registry = SqliteRegistry(tmp_path / "registry.sqlite")
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    # Sealed before empty directories were recorded, so an Agent that saved no Skill
    # produced a payload holding tools/ alone.
    state = tmp_path / "legacy-state"
    (state / "tools").mkdir(parents=True)
    (state / "tools/README.md").write_text("# Candidate tools\n")
    state_digest = artifacts.put_directory(state, ArtifactKind.KERNEL_AGENT_RUNTIME_STATE)
    trace_digest = artifacts.put_json(
        {
            "schema_version": 9,
            "candidate": {
                "optimizer_digest": _put_text_artifact(
                    artifacts, tmp_path, "optimizer", ArtifactKind.KERNEL_AGENT
                ),
                "runtime_state_digest": state_digest,
            },
        },
        ArtifactKind.EVOLUTION,
    )
    revision = KernelAgentRevision(
        new_kernel_agent_revision_id(),
        new_kernel_agent_revision_id(),
        "epoch:test:challenger:1",
        Dsl.TRITON,
        _put_text_artifact(artifacts, tmp_path, "optimizer", ArtifactKind.KERNEL_AGENT),
        "evolver",
        NOW,
        evolution_trace_digest=trace_digest,
        runtime_state_digest=state_digest,
    )
    assembler = LocalAttemptWorkspaceAssembler(tmp_path / "workspaces", registry, artifacts)

    state, _lock = assembler._persistent_root(
        lineage_id=new_lineage_id(),
        revision=revision,
        trajectory_ordinal=1,
    )

    assert (state / "skills").is_dir()
    assert [p.name for p in (state / "skills").iterdir()] == ["README.md"]
    assert (state / "tools/README.md").read_text() == "# Candidate tools\n"
    registry.close()


def test_source_only_evolution_preserves_parent_trajectory_state(tmp_path: Path) -> None:
    registry = SqliteRegistry(tmp_path / "registry.sqlite")
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    assembler = LocalAttemptWorkspaceAssembler(tmp_path / "workspaces", registry, artifacts)
    lineage_id = new_lineage_id()
    parent = KernelAgentRevision(
        new_kernel_agent_revision_id(),
        None,
        "bootstrap:test",
        Dsl.TRITON,
        _put_text_artifact(artifacts, tmp_path, "parent-optimizer", ArtifactKind.KERNEL_AGENT),
        "bootstrap",
        NOW,
        source_provenance_digest=digest("parent-provenance"),
    )
    parent_state, _parent_lock = assembler._persistent_root(
        lineage_id=lineage_id,
        revision=parent,
        trajectory_ordinal=1,
    )
    (parent_state / "skills/retained.md").write_text("parent trajectory lesson\n")
    trace_digest = artifacts.put_json(
        {
            "schema_version": 9,
            "candidate": {
                "optimizer_digest": _put_text_artifact(
                    artifacts, tmp_path, "child-optimizer", ArtifactKind.KERNEL_AGENT
                ),
                "runtime_state_digest": None,
            },
        },
        ArtifactKind.EVOLUTION,
    )
    child = KernelAgentRevision(
        new_kernel_agent_revision_id(),
        parent.id,
        "epoch:test:challenger:1",
        Dsl.TRITON,
        _put_text_artifact(artifacts, tmp_path, "child-optimizer", ArtifactKind.KERNEL_AGENT),
        "evolver",
        NOW,
        evolution_trace_digest=trace_digest,
    )

    child_state, _child_lock = assembler._persistent_root(
        lineage_id=lineage_id,
        revision=child,
        trajectory_ordinal=1,
    )

    assert (child_state / "skills/retained.md").read_text() == "parent trajectory lesson\n"
    registry.close()


def test_missing_trajectory_scope_restores_previous_attempt_runtime_state(
    tmp_path: Path,
) -> None:
    registry = SqliteRegistry(tmp_path / "registry.sqlite")
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    checkpoint = tmp_path / "previous-attempt-state"
    (checkpoint / "skills").mkdir(parents=True)
    (checkpoint / "tools").mkdir()
    (checkpoint / "skills/learned.md").write_text("terminal Attempt lesson\n")
    (checkpoint / "tools/README.md").write_text("# Terminal tools\n")
    state_digest = artifacts.put_directory(
        checkpoint,
        ArtifactKind.KERNEL_AGENT_RUNTIME_STATE,
    )
    revision = KernelAgentRevision(
        new_kernel_agent_revision_id(),
        None,
        "bootstrap:test",
        Dsl.TRITON,
        _put_text_artifact(artifacts, tmp_path, "optimizer", ArtifactKind.KERNEL_AGENT),
        "bootstrap",
        NOW,
        source_provenance_digest=digest("source"),
    )
    assembler = LocalAttemptWorkspaceAssembler(tmp_path / "workspaces", registry, artifacts)

    state, _lock = assembler._persistent_root(
        lineage_id=new_lineage_id(),
        revision=revision,
        trajectory_ordinal=1,
        previous_runtime_state_digest=state_digest,
    )

    assert (state / "skills/learned.md").read_text() == "terminal Attempt lesson\n"
    assert (state / "tools/README.md").read_text() == "# Terminal tools\n"
    registry.close()


def test_five_directory_state_survives_serial_attempts_and_isolates_trajectories(
    tmp_path: Path,
) -> None:
    with SqliteRegistry(tmp_path / "registry.sqlite") as registry:
        artifacts = LocalArtifactStore(tmp_path / "artifacts")
        assembler, request, _lineage, _agent = _single_trajectory_workspace(
            tmp_path,
            registry,
            artifacts,
            ephemeral_agent_state=False,
        )
        first = assembler.prepare(request)
        for name in REUSABLE_AGENT_DIRECTORIES:
            assert (first.root / name / "README.md").is_file()
            (first.root / name / "reusable.txt").write_text(f"{name} content")
            (first.root / name / "README.md").write_text(f"{name} current index")
        first.persist_reusable_directories()
        checkpoint = first.seal_runtime_state(artifacts)
        registry.record_attempt_runtime_state(request.attempt_id, checkpoint)
        old = registry.get_attempt(request.attempt_id)
        next_id = new_attempt_id()
        context = json.loads((tmp_path / "source-attempt-evidence/context.json").read_text())
        context.update(attempt_id=str(next_id), ordinal=2)
        evidence = tmp_path / "next-evidence"
        evidence.mkdir()
        (evidence / "context.json").write_text(json.dumps(context))
        (evidence / "lessons.json").write_text('{"schema_version": 1, "annotations": []}')
        (evidence / "attempts").mkdir()
        (evidence / "attempts/00000001.json").write_text(
            json.dumps(
                {
                    "ordinal": 1,
                    "branch": "active",
                    "challenger_ordinal": 0,
                    "trajectory_ordinal": 1,
                    "kernel_agent_revision_id": str(old.kernel_agent_revision_id),
                }
            )
        )
        evidence_digest = artifacts.put_directory(evidence, ArtifactKind.ATTEMPT_EVIDENCE)
        registry.insert_attempt(
            replace(
                old,
                id=next_id,
                ordinal=2,
                attempt_evidence_digest=evidence_digest,
                input_runtime_state_digest=None,
                runtime_state_digest=None,
            )
        )
        second = assembler.prepare(
            replace(request, attempt_id=next_id, attempt_evidence_digest=evidence_digest)
        )
        for name in REUSABLE_AGENT_DIRECTORIES:
            assert (second.root / name / "reusable.txt").read_text() == f"{name} content"
            assert (second.root / name / "README.md").read_text() == f"{name} current index"
        revision = registry.get_kernel_agent_revision(old.kernel_agent_revision_id)
        sibling, _lock = assembler._persistent_root(
            lineage_id=registry.get_epoch(old.epoch_id).lineage_id,
            revision=revision,
            trajectory_ordinal=2,
            previous_runtime_state_digest=checkpoint,
        )
        for name in REUSABLE_AGENT_DIRECTORIES:
            (second.root / name / "reusable.txt").unlink()
        second.persist_reusable_directories()
        for name in REUSABLE_AGENT_DIRECTORIES:
            assert (sibling / name / "reusable.txt").is_file()
            assert second.persistent_state_root is not None
            assert not (second.persistent_state_root / name / "reusable.txt").exists()


def test_legacy_state_upgrades_only_the_copy(tmp_path: Path) -> None:
    legacy = tmp_path / "legacy"
    (legacy / "tools").mkdir(parents=True)
    (legacy / "tools/README.md").write_text("existing index")
    store = LocalArtifactStore(tmp_path / "artifacts")
    digest = store.put_directory(legacy, ArtifactKind.KERNEL_AGENT_RUNTIME_STATE)
    source = store.verify(digest).payload_path
    copied = tmp_path / "copied"
    copy_reusable_agent_state(source, copied)
    validate_reusable_agent_state_seed(copied, require_complete=True)
    assert {path.name for path in source.iterdir()} == {"tools"}
    assert (copied / "tools/README.md").read_text() == "existing index"
    assert store.verify(digest).digest == digest


def test_legacy_state_loads_prompts_from_pinned_core_only(tmp_path: Path) -> None:
    core = tmp_path / "core"
    (core / "prompts").mkdir(parents=True)
    (core / "prompts/episode.md").write_text("Pinned methodology")
    legacy = tmp_path / "legacy"
    (legacy / "tools").mkdir(parents=True)
    (legacy / "tools/README.md").write_text("Historical tools")
    store = LocalArtifactStore(tmp_path / "artifacts")
    source_digest = store.put_directory(core, ArtifactKind.KERNEL_AGENT)
    state_digest = store.put_directory(legacy, ArtifactKind.KERNEL_AGENT_RUNTIME_STATE)
    source = store.verify(source_digest).payload_path
    state = store.verify(state_digest).payload_path
    copied = tmp_path / "copied"
    copy_reusable_agent_state(state, copied, optimizer_source=source)
    assert (copied / "prompts/episode.md").read_text() == "Pinned methodology"
    assert (copied / "tools/README.md").read_text() == "Historical tools"
    assert not (state / "prompts").exists()
    validate_reusable_agent_state_seed(copied, require_complete=True)
    assert store.verify(source_digest).digest == source_digest
    assert store.verify(state_digest).digest == state_digest


def test_state_without_hooks_gains_only_an_empty_indexed_copy(tmp_path: Path) -> None:
    legacy = tmp_path / "legacy"
    ensure_reusable_directories(legacy)
    (legacy / "hooks/README.md").unlink()
    (legacy / "hooks").rmdir()
    store = LocalArtifactStore(tmp_path / "artifacts")
    digest = store.put_directory(legacy, ArtifactKind.KERNEL_AGENT_RUNTIME_STATE)
    source = store.verify(digest).payload_path
    validate_reusable_agent_state_seed(source)
    with pytest.raises(ValueError, match=r"hooks/ must retain README\.md"):
        validate_reusable_agent_state_seed(source, require_complete=True)
    copied = tmp_path / "copied"
    copy_reusable_agent_state(source, copied)
    validate_reusable_agent_state_seed(copied, require_complete=True)
    assert [entry.name for entry in (copied / "hooks").iterdir()] == ["README.md"]
    assert not (source / "hooks").exists()
    assert store.verify(digest).digest == digest


@pytest.mark.parametrize("reset", (False, True))
def test_attempt_core_seed_and_retry_precedence(tmp_path: Path, reset: bool) -> None:
    with SqliteRegistry(tmp_path / "registry.sqlite") as registry:
        store = LocalArtifactStore(tmp_path / "artifacts")
        assembler, request, _, _ = _single_trajectory_workspace(
            tmp_path, registry, store, ephemeral_agent_state=reset, core_seed=True
        )
        first = assembler.prepare(request)
        for name in REUSABLE_AGENT_DIRECTORIES:
            assert (first.root / name / "seed.md").read_text() == f"Initial {name}"
            assert not (first.root / "agent/optimizer" / name).exists()
            (first.root / name / "seed.md").unlink()
            (first.root / name / "README.md").write_text("Pruned seed")
        first.persist_reusable_directories()
        registry.record_attempt_runtime_state(request.attempt_id, first.seal_runtime_state(store))
        retry = assembler.prepare(request)
        for name in REUSABLE_AGENT_DIRECTORIES:
            assert (retry.root / name / "seed.md").exists() is reset
            expected = f"Initial {name} index" if reset else "Pruned seed"
            assert (retry.root / name / "README.md").read_text() == expected


def test_core_initial_state_copies_resources_not_engineering_docs(tmp_path: Path) -> None:
    core = tmp_path / "core"
    for name in REUSABLE_AGENT_DIRECTORIES:
        (core / name).mkdir(parents=True)
        (core / name / "README.md").write_text(f"Core {name} index")
        (core / name / "nested").mkdir()
        (core / name / "nested/seed.md").write_text(f"Initial {name}")
    (core / "docs").mkdir()
    (core / "docs/design.md").write_text("Engineering only")
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    digest = artifacts.put_directory(core, ArtifactKind.KERNEL_AGENT)
    source = artifacts.verify(digest).payload_path
    root = tmp_path / "workspace"
    initialize_reusable_agent_state(root, source)
    for name in REUSABLE_AGENT_DIRECTORIES:
        assert (root / name / "README.md").read_text() == f"Core {name} index"
        assert (root / name / "nested/seed.md").read_text() == f"Initial {name}"
        (root / name / "nested/seed.md").write_text("Learned state")
        assert (source / name / "nested/seed.md").read_text() == f"Initial {name}"
    assert not (root / "docs").exists()
    assert not (root / "knowledge/design.md").exists()
    assert artifacts.verify(digest).digest == digest


def test_legacy_docs_migrates_without_rewriting_artifact(tmp_path: Path) -> None:
    legacy = tmp_path / "legacy"
    (legacy / "docs").mkdir(parents=True)
    (legacy / "docs/README.md").write_text("API knowledge: api.md")
    (legacy / "docs/api.md").write_text("Measured API constraints")
    (legacy / "tools").mkdir()
    (legacy / "tools/README.md").write_text("Tool index")
    store = LocalArtifactStore(tmp_path / "artifacts")
    digest = store.put_directory(legacy, ArtifactKind.KERNEL_AGENT_RUNTIME_STATE)
    source = store.verify(digest).payload_path
    validate_reusable_agent_state_seed(source)
    with pytest.raises(ValueError, match="may contain only"):
        validate_reusable_agent_state_seed(source, require_complete=True)
    copied = tmp_path / "copied"
    copy_reusable_agent_state(source, copied)
    validate_reusable_agent_state_seed(copied, require_complete=True)
    assert not (copied / "docs").exists()
    assert (copied / "knowledge/README.md").read_text() == "API knowledge: api.md"
    assert (copied / "knowledge/api.md").read_text() == "Measured API constraints"
    assert (copied / "knowledge/api.md").stat().st_mode & 0o200
    assert (source / "docs/api.md").is_file()
    assert not (source / "knowledge").exists()
    assert store.verify(digest).digest == digest


def test_existing_persistent_docs_is_renamed(tmp_path: Path) -> None:
    state = tmp_path / "state"
    (state / "docs").mkdir(parents=True)
    (state / "docs/note.md").write_text("Keep across retries")
    ensure_reusable_directories(state)
    ensure_reusable_directories(state)
    assert not (state / "docs").exists()
    assert (state / "knowledge/note.md").read_text() == "Keep across retries"
    validate_reusable_agent_state_seed(state, require_complete=True)


def test_knowledge_migration_rejects_conflicting_names(tmp_path: Path) -> None:
    ensure_reusable_directories(tmp_path)
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs/note.md").write_text("Do not discard")
    with pytest.raises(ValueError, match="both docs/ and knowledge/"):
        ensure_reusable_directories(tmp_path)
    assert (tmp_path / "docs/note.md").read_text() == "Do not discard"


@pytest.mark.parametrize("name", REUSABLE_AGENT_DIRECTORIES)
def test_reusable_readme_rejects_symlink_or_directory(tmp_path: Path, name: str) -> None:
    state = tmp_path / "state"
    ensure_reusable_directories(state)
    readme = state / name / "README.md"
    readme.unlink()
    outside = tmp_path / "outside"
    outside.write_text("unchanged")
    readme.symlink_to(outside)
    with pytest.raises(ValueError, match="regular file or directory"):
        ensure_reusable_directories(state)
    assert outside.read_text() == "unchanged"
    readme.unlink()
    readme.mkdir()
    with pytest.raises(ValueError, match="README must be a regular file"):
        ensure_reusable_directories(state)


@pytest.mark.parametrize("legacy_knowledge", (False, True))
def test_evolver_snapshot_preserves_all_state_indexes_read_only(
    tmp_path: Path, legacy_knowledge: bool
) -> None:
    workspaces = tmp_path / "attempts"
    source = workspaces / ".reusable/lineage/agent/trajectory-00000001"
    ensure_reusable_directories(source)
    for name in REUSABLE_AGENT_DIRECTORIES:
        (source / name / "entry.txt").write_text(name)
        (source / name / "README.md").write_text(f"{name}: entry.txt")
    if legacy_knowledge:
        (source / "knowledge").rename(source / "docs")
    target = tmp_path / "evolver-view"
    assert materialize_reusable_agent_state_snapshot(
        workspaces,
        target,
        agent_lineages={"agent": "lineage"},
    ) == {"agent": (1,)}
    for name in REUSABLE_AGENT_DIRECTORIES:
        copied = target / "agent/trajectories/trajectory-00000001" / name
        assert (copied / "entry.txt").read_text() == name
        assert (copied / "README.md").read_text() == f"{name}: entry.txt"
        assert not (copied.stat().st_mode & 0o222)
    assert (source / "docs").exists() is legacy_knowledge
    assert not (target / "agent/trajectories/trajectory-00000001/docs").exists()
