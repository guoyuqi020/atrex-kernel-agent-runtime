"""Branch-local Attempt Evidence assembly and isolation tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from conftest import NOW, digest

from atrex_runtime.artifacts.local import ArtifactKind, LocalArtifactStore
from atrex_runtime.controller import (
    AttemptEvidenceMetadataV2,
    EvidenceArtifactProjector,
    EvidenceProjectionLimits,
    LocalAttemptEvidenceAssembler,
)
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
    AttemptReportStatus,
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
    TokenUsage,
)
from atrex_runtime.ports import BuildAttemptEvidenceRequest
from atrex_runtime.registry.sqlite import SqliteRegistry


def _directory_artifact(
    artifacts: LocalArtifactStore,
    tmp_path: Path,
    label: str,
    kind: ArtifactKind,
    content: str,
) -> ArtifactDigest:
    source = tmp_path / f"source-{label}"
    source.mkdir()
    (source / "kernel.py").write_text(content, encoding="utf-8")
    return artifacts.put_directory(source, kind)


def _session_artifact(
    artifacts: LocalArtifactStore,
    tmp_path: Path,
    label: str,
    annotation: str,
) -> ArtifactDigest:
    source = tmp_path / f"session-{label}"
    source.mkdir()
    events = [
        {"type": "session", "version": 0, "id": f"session-{label}"},
        {
            "type": "assistant/message",
            "seq": 1,
            "time": 1,
            "data": {"message": {"content": [{"type": "text", "text": annotation}]}},
        },
        {
            "type": "turn/end",
            "seq": 2,
            "time": 2,
            "data": {"reason": {"kind": "completed"}},
        },
    ]
    (source / "session.jsonl").write_text(
        "".join(f"{json.dumps(event)}\n" for event in events),
        encoding="utf-8",
    )
    return artifacts.put_directory(source, ArtifactKind.SESSION_LOG)


def _projector(artifacts: LocalArtifactStore) -> EvidenceArtifactProjector:
    return EvidenceArtifactProjector(
        artifacts,
        EvidenceProjectionLimits(
            max_trace_files=8,
            max_trace_bytes=1_000_000,
            max_trace_events=100,
            max_projection_text_bytes=100_000,
            max_diff_files=16,
            max_diff_bytes=100_000,
        ),
    )


def _seed_epoch(
    registry: SqliteRegistry,
    artifacts: LocalArtifactStore,
    tmp_path: Path,
) -> tuple[Epoch, KernelRevision, ArtifactDigest]:
    evidence = _directory_artifact(
        artifacts,
        tmp_path,
        "epoch-evidence",
        ArtifactKind.EVIDENCE,
        "trusted epoch evidence\n",
    )
    baseline_digest = _directory_artifact(
        artifacts,
        tmp_path,
        "baseline",
        ArtifactKind.KERNEL,
        "VALUE = 0\n",
    )
    campaign_id = new_campaign_id()
    registry.insert_campaign(
        Campaign(
            campaign_id,
            "vector_add",
            "h100",
            digest("contract"),
            digest("problem"),
            NOW,
        )
    )
    agent_id = new_kernel_agent_revision_id()
    registry.register_kernel_agent_revision(
        KernelAgentRevision(
            id=agent_id,
            parent_id=None,
            creation_key="bootstrap:triton",
            dsl=Dsl.TRITON,
            optimizer_digest=digest("optimizer"),
            created_by="bootstrap",
            created_at=NOW,
            source_provenance_digest=digest("source"),
        )
    )
    baseline = registry.register_kernel_revision(
        KernelRevision(
            new_kernel_revision_id(),
            None,
            baseline_digest,
            None,
            KernelEvaluation(True, 100.0, digest("baseline-gateway")),
            NOW,
        )
    )
    lineage_id = new_lineage_id()
    registry.insert_lineage(
        Lineage(
            id=lineage_id,
            campaign_id=campaign_id,
            dsl=Dsl.TRITON,
            hardware_target="h100",
            active_kernel_agent_revision_id=agent_id,
            best_kernel_revision_id=baseline.id,
            evidence_checkpoint=evidence,
            challenger_count=1,
            trajectories_per_branch=1,
            attempts_per_trajectory=3,
            next_epoch_number=1,
            status=LineageStatus.READY,
        )
    )
    epoch = Epoch(
        id=new_epoch_id(),
        lineage_id=lineage_id,
        number=1,
        active_kernel_agent_revision_id=agent_id,
        challenger_kernel_agent_revision_ids=(agent_id,),
        starting_kernel_revision_id=baseline.id,
        evidence_checkpoint=evidence,
        challenger_count=1,
        trajectories_per_branch=1,
        attempts_per_trajectory=3,
        status=EpochStatus.RUNNING,
        winner_kernel_agent_revision_id=None,
        best_kernel_revision_id=None,
        created_at=NOW,
        completed_at=None,
    )
    registry.insert_epoch(epoch)
    return epoch, baseline, evidence


def _complete_attempt(
    registry: SqliteRegistry,
    artifacts: LocalArtifactStore,
    tmp_path: Path,
    epoch: Epoch,
    baseline: KernelRevision,
    branch: BranchRole,
    label: str,
    annotation: str,
) -> Attempt:
    attempt = Attempt(
        id=new_attempt_id(),
        epoch_id=epoch.id,
        branch=branch,
        challenger_ordinal=(0 if branch is BranchRole.ACTIVE else 1),
        trajectory_ordinal=1,
        ordinal=1,
        kernel_agent_revision_id=epoch.active_kernel_agent_revision_id,
        input_kernel_revision_id=baseline.id,
        attempt_evidence_digest=digest(f"seed-{label}-evidence"),
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
    registry.insert_attempt(attempt)
    output_digest = _directory_artifact(
        artifacts,
        tmp_path,
        f"output-{label}",
        ArtifactKind.KERNEL,
        f"VALUE = '{label}'\n",
    )
    output = registry.register_kernel_revision(
        KernelRevision(
            new_kernel_revision_id(),
            baseline.id,
            output_digest,
            attempt.id,
            KernelEvaluation(True, 90.0, digest(f"{label}-gateway")),
            NOW,
        )
    )
    trace = _session_artifact(artifacts, tmp_path, label, annotation)
    registry.record_attempt_session_trace(
        attempt.id,
        trace,
        "completed",
        1000,
        TokenUsage(10, 20, 30, 40),
    )
    report = artifacts.put_json(
        {
            "schema_version": 1,
            "attempt_id": str(attempt.id),
            "status": "candidate_ready",
            "hypothesis": f"hypothesis-{label}",
            "bottleneck": "memory bandwidth",
            "plan": ["coalesce loads"],
            "change_summary": f"change-{label}",
            "profile_evidence": "profile evidence",
            "evaluation_evidence": "Gateway evidence",
            "result_interpretation": f"interpretation-{label}",
            "decision": "keep",
            "research_sources": [],
            "lessons": [f"structured-lesson-{label}"],
            "next_directions": [],
        },
        ArtifactKind.ATTEMPT_REPORT,
    )
    registry.record_attempt_report(
        attempt.id,
        report,
        AttemptReportStatus.CANDIDATE_READY,
    )
    registry.complete_attempt(
        attempt.id,
        output.id,
        accepted_as_branch_best=True,
        failure_reason=None,
    )
    return registry.get_attempt(attempt.id)


def test_attempt_evidence_contains_only_earlier_same_branch_history(
    tmp_path: Path,
) -> None:
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    with SqliteRegistry(tmp_path / "registry.sqlite") as registry:
        epoch, baseline, epoch_evidence = _seed_epoch(registry, artifacts, tmp_path)
        active = _complete_attempt(
            registry,
            artifacts,
            tmp_path,
            epoch,
            baseline,
            BranchRole.ACTIVE,
            "active",
            "active-only lesson token=active-secret",
        )
        challenger = _complete_attempt(
            registry,
            artifacts,
            tmp_path,
            epoch,
            baseline,
            BranchRole.CHALLENGER,
            "challenger",
            "challenger lesson token=challenger-secret",
        )
        request = BuildAttemptEvidenceRequest(
            attempt_id=new_attempt_id(),
            epoch_id=epoch.id,
            branch=BranchRole.CHALLENGER,
            challenger_ordinal=1,
            trajectory_ordinal=1,
            ordinal=2,
            epoch_evidence_checkpoint=epoch_evidence,
        )
        assembler = LocalAttemptEvidenceAssembler(registry, artifacts, _projector(artifacts))

        first_digest = assembler.assemble(request)
        second_digest = assembler.assemble(request)

        assert first_digest == second_digest
        stored = artifacts.verify(first_digest)
        assert stored.kind is ArtifactKind.ATTEMPT_EVIDENCE
        metadata = AttemptEvidenceMetadataV2.from_file(stored.payload_path / "context.json")
        assert metadata.previous_attempt_ids == (challenger.id,)
        assert active.id not in metadata.previous_attempt_ids
        attempt_value = json.loads(
            (stored.payload_path / "attempts/00000001.json").read_text(encoding="utf-8")
        )
        assert attempt_value["attempt_id"] == challenger.id
        assert attempt_value["kernel_diff"] == "diffs/00000001.json"
        assert attempt_value["attempt_report"] == "reports/00000001.json"
        serialized = (stored.payload_path / "lessons.json").read_text(encoding="utf-8")
        assert "challenger lesson" in serialized
        assert "structured-lesson-challenger" in serialized
        assert "active-only" not in serialized
        assert "challenger-secret" not in serialized
        assert "[REDACTED]" in serialized

        mismatched = BuildAttemptEvidenceRequest(
            attempt_id=new_attempt_id(),
            epoch_id=epoch.id,
            branch=BranchRole.CHALLENGER,
            challenger_ordinal=1,
            trajectory_ordinal=1,
            ordinal=2,
            epoch_evidence_checkpoint=epoch_evidence,
        )
        with pytest.raises(ValueError, match="disagrees with its Attempt"):
            assembler.validate(first_digest, mismatched)


def test_attempt_evidence_rejects_missing_same_branch_ordinal(tmp_path: Path) -> None:
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    with SqliteRegistry(tmp_path / "registry.sqlite") as registry:
        epoch, baseline, epoch_evidence = _seed_epoch(registry, artifacts, tmp_path)
        _complete_attempt(
            registry,
            artifacts,
            tmp_path,
            epoch,
            baseline,
            BranchRole.ACTIVE,
            "active",
            "lesson",
        )
        request = BuildAttemptEvidenceRequest(
            attempt_id=new_attempt_id(),
            epoch_id=epoch.id,
            branch=BranchRole.ACTIVE,
            challenger_ordinal=0,
            trajectory_ordinal=1,
            ordinal=3,
            epoch_evidence_checkpoint=epoch_evidence,
        )

        with pytest.raises(ValueError, match="incomplete same-branch history"):
            LocalAttemptEvidenceAssembler(
                registry,
                artifacts,
                _projector(artifacts),
            ).assemble(request)
