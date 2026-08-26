"""Branch-local Attempt Evidence assembly and isolation tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from conftest import NOW, digest

from atrex_runtime.artifacts.local import ArtifactKind, LocalArtifactStore
from atrex_runtime.attempt_reports import RuntimeAttemptReportProjector
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
    evaluation_contract = artifacts.put_json(
        {
            "schema_version": 1,
            "candidate_path": "kernel.py",
            "reference_py": "class Model:\n    pass\n",
            "input_py": "def get_inputs():\n    return ()\n",
            "shapes": {"0": {}},
            "options": {
                "num_correctness_cases": 1,
                "bench_iters": 5,
                "atol": 0.001,
                "rtol": 0.01,
                "timeout_s": 60,
            },
            "production_gate": True,
        },
        ArtifactKind.EVALUATION_CONTRACT,
    )
    campaign_id = new_campaign_id()
    registry.insert_campaign(
        Campaign(
            campaign_id,
            "vector_add",
            "h100",
            evaluation_contract,
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
    baseline_samples = tuple(
        artifacts.put_json(
            {
                "operation": "evaluate",
                "status": "completed",
                "result": {
                    "correct": True,
                    "correctness": {
                        "status": "PASS",
                        "rel_err": 0.002,
                        "max_abs_err": 0.001,
                        "max_rel_err": 0.007,
                    },
                    "latency_us_by_shape": {"0": first, "1": second},
                },
            },
            ArtifactKind.GATEWAY_RESULT,
        )
        for first, second in ((79.0, 124.0), (81.0, 126.0))
    )
    baseline_gateway = artifacts.put_json(
        {
            "operation": "evaluate_comparison",
            "aggregation": "arithmetic_mean",
            "correct": True,
            "latency_us": 100.0,
            "measurements": [
                {"gateway_result_digest": str(value)} for value in baseline_samples
            ],
        },
        ArtifactKind.GATEWAY_RESULT,
    )
    baseline = registry.register_kernel_revision(
        KernelRevision(
            new_kernel_revision_id(),
            None,
            baseline_digest,
            None,
            KernelEvaluation(True, 100.0, baseline_gateway),
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
    *,
    reject_by_production_gate: bool = False,
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
    agent_evaluate_gateway = artifacts.put_json(
        {
            "operation": "evaluate",
            "status": "completed",
            "result": {
                "correct": True,
                "latency_us": 92.0,
                "latency_us_by_shape": {"0": 74.0, "1": 115.0},
            },
        },
        ArtifactKind.GATEWAY_RESULT,
    )
    agent_profile_gateway = artifacts.put_json(
        {
            "operation": "profile",
            "status": "completed",
            "result": {"kernel_count": 1, "dominant_bound": "memory"},
        },
        ArtifactKind.GATEWAY_RESULT,
    )
    runtime_abba_gateway = artifacts.put_json(
        {
            "operation": "same_allocation_abba",
            "status": "completed",
            "candidate": {
                "correct": True,
                "correctness": {
                    "status": "PASS",
                    "rel_err": 0.003,
                    "max_abs_err": 0.0009765625,
                    "max_rel_err": 0.0078125,
                },
                "latency_us": 90.0,
                "latency_us_by_shape": {"0": 72.0, "1": 112.5},
            },
        },
        ArtifactKind.GATEWAY_RESULT,
    )
    output = KernelRevision(
        new_kernel_revision_id(),
        baseline.id,
        output_digest,
        attempt.id,
        KernelEvaluation(True, 90.0, runtime_abba_gateway),
        NOW,
    )
    if not reject_by_production_gate:
        output = registry.register_kernel_revision(output)
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
            "schema_version": 12,
            "attempt_id": str(attempt.id),
            "status": "candidate_ready",
            "hypothesis": f"hypothesis-{label}",
            "diagnosis": {
                "bottleneck": "memory bandwidth",
                "evidence": "profile evidence",
            },
            "approach": {
                "summary": "Coalesce global loads",
                "steps": ["coalesce loads"],
                "expected_impact": "Reduce memory transactions",
                "risks": [],
            },
            "final_candidate": {"change_summary": f"change-{label}"},
            "evidence_summary": {
                "correctness": "Gateway correctness passed",
                "performance": "Gateway latency improved",
            },
            "profile_evidence": {
                "tool_used": "gateway-execute/profile",
                "profiler": "ncu",
                "profile_level": "sol",
                "bottleneck_type": "memory_bound",
                "evidence_summary": "profile evidence",
                "evidence_chain": "profile counters support the memory diagnosis",
                "supporting_results": [
                    {
                        "operation": "profile",
                        "kernel_artifact_digest": str(output.artifact_digest),
                        "kernel_trial_id": "gtrial_abcdef0123456789abcdef0123456789",
                        "gateway_result_digest": str(agent_profile_gateway),
                    }
                ],
            },
            "analysis": f"interpretation-{label}",
            "knowledge_used": [],
            "findings": [
                {
                    "category": "memory",
                    "observation": f"structured-lesson-{label}",
                    "root_cause": "Uncoalesced loads",
                    "resolution": "Kept the coalesced-load candidate",
                    "lesson": f"structured-lesson-{label}",
                    "supporting_experiment_ids": [
                        "experiment_0123456789abcdef0123456789abcdef"
                    ],
                }
            ],
            "blocker": None,
            "experiments": [
                {
                    "experiment_id": "experiment_0123456789abcdef0123456789abcdef",
                    "direction_id": "direction_0123456789abcdef0123456789abcdef",
                    "sequence": 1,
                    "recorded_at": NOW,
                    "name": "Coalesced loads",
                    "hypothesis": f"hypothesis-{label}",
                    "change": f"change-{label}",
                    "before": {
                        "kernel_artifact_digest": str(baseline.artifact_digest),
                        "kernel_trial_id": "gtrial_0123456789abcdef0123456789abcdef",
                        "gateway_result_digests": [
                            str(baseline.evaluation.gateway_result_digest)
                        ],
                    },
                    "after": {
                        "kernel_artifact_digest": str(output.artifact_digest),
                        "kernel_trial_id": "gtrial_abcdef0123456789abcdef0123456789",
                        "gateway_result_digests": [str(agent_evaluate_gateway)],
                    },
                    "evidence": "Correct and faster on both shapes",
                    "analysis": "The coalescing hypothesis is supported",
                    "action": "keep_after",
                }
            ],
            "direction_events": [
                {
                    "direction_event_id": "directionevent_0123456789abcdef0123456789abcdef",
                    "direction_id": "direction_0123456789abcdef0123456789abcdef",
                    "recorded_at": NOW,
                    "action": "propose",
                    "name": "Coalesced loads",
                    "hypothesis": f"hypothesis-{label}",
                    "rationale": "profile evidence",
                    "plan": ["coalesce loads"],
                    "success_criteria": "latency improves",
                    "stop_conditions": "correctness fails",
                    "analysis": None,
                    "supporting_experiment_ids": [],
                },
                {
                    "direction_event_id": "directionevent_fedcba9876543210fedcba9876543210",
                    "direction_id": "direction_0123456789abcdef0123456789abcdef",
                    "recorded_at": NOW,
                    "action": "complete",
                    "name": None,
                    "hypothesis": None,
                    "rationale": None,
                    "plan": [],
                    "success_criteria": None,
                    "stop_conditions": None,
                    "analysis": "the hypothesis was supported",
                    "supporting_experiment_ids": [
                        "experiment_0123456789abcdef0123456789abcdef"
                    ],
                },
            ],
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
        None if reject_by_production_gate else output.id,
        accepted_as_branch_best=not reject_by_production_gate,
        failure_reason=(
            "Candidate nomination was rejected: production gate rejected candidate: "
            "mixed/alternate framework marker is forbidden: cuda"
            if reject_by_production_gate
            else None
        ),
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
        report_value = json.loads(
            (stored.payload_path / "reports/00000001.json").read_text(encoding="utf-8")
        )
        assert report_value["schema_version"] == 1
        assert report_value["parent_kernel"] == {
            "version": "v0",
            "kernel_artifact_digest": str(baseline.artifact_digest),
            "gateway_result": {
                "operation": "evaluate_comparison",
                "status": "completed",
                "correct": True,
                "correctness": {
                    "status": "PASS",
                    "rel_err": 0.002,
                    "max_abs_err": 0.001,
                    "max_rel_err": 0.007,
                },
                "latency_us_geomean": 100.0,
                "latency_us_arith_mean": 102.5,
                "latency_us_by_shape": {"0": 80.0, "1": 125.0},
            },
        }
        assert challenger.output_kernel_revision_id is not None
        assert report_value["candidate_kernel"]["kernel_artifact_digest"] == str(
            registry.get_kernel_revision(challenger.output_kernel_revision_id).artifact_digest
        )
        assert report_value["candidate_kernel"]["status"] == "retained"
        assert report_value["candidate_kernel"]["gateway_result"] == {
            "operation": "same_allocation_abba",
            "status": "completed",
            "correct": True,
            "correctness": {
                "status": "PASS",
                "rel_err": 0.003,
                "max_abs_err": 0.0009765625,
                "max_rel_err": 0.0078125,
            },
            "latency_us_geomean": 90.0,
            "latency_us_arith_mean": 92.25,
            "latency_us_by_shape": {"0": 72.0, "1": 112.5},
        }
        assert report_value["candidate_kernel"]["comparison_with_parent"] == {
            "latency_us_geomean_delta": -10.0,
            "improvement_percent": 10.0,
            "latency_us_delta_by_shape": {"0": -8.0, "1": -12.5},
            "improvement_percent_by_shape": {"0": 10.0, "1": 10.0},
        }
        assert report_value["production_gate"] == {
            "enabled": True,
            "result": "PASS",
            "failure_reason": None,
        }
        assert "kernel_revision_id" not in json.dumps(report_value)
        assert "gateway_result_digest" not in json.dumps(report_value["parent_kernel"])
        assert challenger.attempt_report_digest is not None
        raw_agent_report = json.loads(
            (
                artifacts.verify(challenger.attempt_report_digest).payload_path
                / "value.json"
            ).read_text(encoding="utf-8")
        )
        assert raw_agent_report["experiments"][0]["after"]["gateway_result_digests"]
        assert "same_allocation_abba" not in json.dumps(raw_agent_report)
        assert "same_allocation_abba" in json.dumps(report_value)
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


def test_final_report_exposes_production_gate_rejection(tmp_path: Path) -> None:
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    with SqliteRegistry(tmp_path / "registry.sqlite") as registry:
        epoch, baseline, _evidence = _seed_epoch(registry, artifacts, tmp_path)
        attempt = _complete_attempt(
            registry,
            artifacts,
            tmp_path,
            epoch,
            baseline,
            BranchRole.ACTIVE,
            "production-rejected",
            "production policy rejected the candidate",
            reject_by_production_gate=True,
        )

        report = RuntimeAttemptReportProjector(registry, artifacts).project(attempt)

        assert report["candidate_kernel"] is None
        assert report["production_gate"] == {
            "enabled": True,
            "result": "FAIL",
            "failure_reason": (
                "Candidate nomination was rejected: production gate rejected candidate: "
                "mixed/alternate framework marker is forbidden: cuda"
            ),
        }
