"""Tests for bounded normalized summaries and exact raw Session projections."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest
from conftest import NOW, digest

from atrex_runtime.artifacts.local import ArtifactKind, LocalArtifactStore
from atrex_runtime.controller.evidence import LocalEvidenceAssembler
from atrex_runtime.controller.projection import (
    EvidenceArtifactProjector,
    EvidenceProjectionLimits,
)
from atrex_runtime.domain.ids import (
    AttemptId,
    new_epoch_id,
    new_kernel_agent_revision_id,
    new_kernel_revision_id,
    new_lineage_id,
)
from atrex_runtime.domain.models import (
    ChallengerProposalType,
    Dsl,
    Epoch,
    EpochChallenger,
    EpochStatus,
    KernelAgentRevision,
)
from atrex_runtime.gateway.control_models import (
    GatewayMeasurementPoint,
    GatewayMeasurementRecord,
    GatewayOperation,
)
from atrex_runtime.registry.base import Registry
from atrex_runtime.workers.evolution import (
    EvolutionAgentDescriptorV3,
    EvolutionCandidateTraceV3,
    EvolutionInputManifestV10,
    EvolutionOutput,
    EvolutionTraceV9,
    VisibleAgentRevisionV2,
)
from atrex_runtime.workers.token_usage import ProviderUsageReportV2, TokenUsageBucketsV2


def _projector(
    artifacts: LocalArtifactStore,
    *,
    trace_bytes: int = 100_000,
) -> EvidenceArtifactProjector:
    return EvidenceArtifactProjector(
        artifacts,
        EvidenceProjectionLimits(8, trace_bytes, 100, 10_000, 20, 100_000),
        redaction_patterns=(r"private-[0-9]+",),
    )


def test_session_projection_keeps_summary_bounded_and_marks_annotation(
    tmp_path: Path,
) -> None:
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    source = tmp_path / "session"
    source.mkdir()
    rows = [
        {
            "type": "session",
            "version": 0,
            "id": "session-1",
            "createdAt": 1,
            "delegationDepth": 0,
        },
        {
            "type": "user/message",
            "seq": 0,
            "time": 1,
            "data": {"content": [{"type": "text", "text": "private prompt"}]},
        },
        {
            "type": "assistant/message",
            "seq": 1,
            "time": 2,
            "data": {
                "turn": 1,
                "step": 1,
                "message": {
                    "content": [
                        {"type": "reasoning", "reasoning": "hidden chain"},
                        {
                            "type": "text",
                            "text": "Try vector loads; api_key=abc private-42",
                        },
                    ]
                },
            },
        },
        {
            "type": "tool/call",
            "seq": 2,
            "time": 3,
            "data": {
                "turn": 1,
                "step": 1,
                "callId": "call-1",
                "name": "gateway_execute",
                "arguments": '{"secret":"never-project"}',
            },
        },
        {
            "type": "tool/result",
            "seq": 3,
            "time": 4,
            "data": {
                "turn": 1,
                "step": 1,
                "message": {"content": [{"type": "text", "text": "raw result"}]},
            },
        },
        {
            "type": "turn/end",
            "seq": 4,
            "time": 5,
            "data": {"turn": 1, "reason": {"kind": "completed"}},
        },
    ]
    (source / "session.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    (source / "conversation.jsonl").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "sequence": 0,
                "type": "message",
                "source": "runtime_input",
                "role": "user",
                "content": [{"type": "text", "text": "unredacted transcript prompt"}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    digest = artifacts.put_directory(source, ArtifactKind.SESSION_LOG)

    projection = _projector(artifacts).session_projection(digest)

    encoded = json.dumps(projection)
    assert "private prompt" not in encoded
    assert "hidden chain" not in encoded
    assert "never-project" not in encoded
    assert "raw result" not in encoded
    assert "api_key=[REDACTED]" in encoded
    assert "private-42" not in encoded
    session = projection["sessions"][0]
    assert session["final_agent_annotation"] == "Try vector loads; api_key=[REDACTED] [REDACTED]"
    assert "raw_files" not in projection
    assert len(projection["sessions"]) == 1


def test_session_projection_rejects_compressed_or_unknown_required_events(
    tmp_path: Path,
) -> None:
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    compressed = tmp_path / "compressed"
    compressed.mkdir()
    (compressed / "session.jsonl.zstd").write_bytes(b"not decoded")
    compressed_digest = artifacts.put_directory(compressed, ArtifactKind.SESSION_LOG)
    with pytest.raises(ValueError, match="uncompressed JSONL"):
        _projector(artifacts).session_projection(compressed_digest)

    unknown = tmp_path / "unknown"
    unknown.mkdir()
    (unknown / "session.jsonl").write_text(
        json.dumps({"type": "session", "version": 0, "id": "s"})
        + "\n"
        + json.dumps({"type": "plugin/required", "seq": 0, "time": 1, "data": {}})
        + "\n",
        encoding="utf-8",
    )
    unknown_digest = artifacts.put_directory(unknown, ArtifactKind.SESSION_LOG)
    with pytest.raises(ValueError, match="does not recognize required"):
        _projector(artifacts).session_projection(unknown_digest)


def test_kernel_diff_is_deterministic_bounded_and_binary_safe(tmp_path: Path) -> None:
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    before = tmp_path / "before"
    after = tmp_path / "after"
    before.mkdir()
    after.mkdir()
    (before / "kernel.py").write_text("BLOCK = 64\n", encoding="utf-8")
    (after / "kernel.py").write_text("BLOCK = 128\n", encoding="utf-8")
    (after / "table.bin").write_bytes(b"\xff\x00")
    before_digest = artifacts.put_directory(before, ArtifactKind.KERNEL)
    after_digest = artifacts.put_directory(after, ArtifactKind.KERNEL)

    value = _projector(artifacts).kernel_diff(before_digest, after_digest)

    assert value == _projector(artifacts).kernel_diff(before_digest, after_digest)
    assert value["changes"][0]["unified_diff"].startswith("--- a/kernel.py")
    assert value["changes"][1]["binary"] is True
    with pytest.raises(ValueError, match="byte limit"):
        EvidenceArtifactProjector(
            artifacts,
            EvidenceProjectionLimits(8, 100_000, 100, 10_000, 20, 10),
        ).kernel_diff(before_digest, after_digest)


def test_derived_evidence_projects_evolver_session_as_untrusted_annotation(
    tmp_path: Path,
) -> None:
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    session_source = tmp_path / "evolver-session"
    session_source.mkdir()
    rows = [
        {"type": "session", "version": 0, "id": "evolver-session"},
        {
            "type": "assistant/message",
            "seq": 0,
            "time": 1,
            "data": {
                "message": {
                    "content": [
                        {"type": "reasoning", "reasoning": "do not retain"},
                        {"type": "text", "text": "Reduce register pressure"},
                    ]
                }
            },
        },
    ]
    (session_source / "events.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    (session_source / "input").mkdir()
    (session_source / "input/prompt.md").write_text(
        "raw private Evolver prompt",
        encoding="utf-8",
    )
    (session_source / "provider").mkdir()
    (session_source / "provider/stdout.stream-json").write_text(
        json.dumps(
            {
                "type": "system",
                "subtype": "thinking_tokens",
                "estimated_tokens": 12_345,
            }
        )
        + "\n"
        + json.dumps(
            {
                "type": "assistant",
                "message": {
                    "content": [
                        {"type": "thinking", "thinking": "raw private reasoning"},
                        {"type": "tool_use", "input": {"token": "raw secret"}},
                    ]
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (session_source / "provider/stderr.log").write_text(
        "raw provider credential",
        encoding="utf-8",
    )
    session_digest = artifacts.put_directory(session_source, ArtifactKind.SESSION_LOG)
    parent_id = new_kernel_agent_revision_id()
    challenger_id = new_kernel_agent_revision_id()
    candidate = EvolutionCandidateTraceV3(
        optimizer_digest=digest("optimizer"),
        runtime_state_digest=digest("runtime-state"),
    )
    trace = EvolutionTraceV9(
        input=EvolutionInputManifestV10(
            parent_revision_id=parent_id,
            evidence_checkpoint=digest("evidence"),
            idempotency_key="evolve:1",
            dsl=Dsl.TRITON,
            optimizer_digest=digest("parent-optimizer"),
            visible_agents=(
                VisibleAgentRevisionV2(
                    revision_id=parent_id,
                    optimizer_digest=digest("parent-optimizer"),
                    path="input/agents/active/source",
                    optimization_summary_path="input/evidence/active/optimization-summary.json",
                    sessions_path="input/evidence/active/sessions",
                    runtime_state_path="input/agents/active/runtime-state",
                    parent=True,
                    relationship="active",
                    challenger_ordinal=None,
                    parent_revision_id=None,
                    created_by="bootstrap",
                ),
            ),
        ),
        agent=EvolutionAgentDescriptorV3(
            bundle_commit="0" * 40,
            bundle_tree="1" * 40,
            bundle_artifact_digest=digest("evolver-bundle"),
            agent_backend="claude",
            model=None,
            reasoning_effort="max",
            session_settings_sha256="0" * 64,
            command_executable="/usr/bin/agent-cli",
            command_argv_sha256="0" * 64,
            environment_keys=(),
            isolated_home_environment_keys=("AGENT_TOOL_HOME",),
        ),
        process_returncode=0,
        stdout="",
        stderr="",
        session_trace_digest=session_digest,
        token_usage=ProviderUsageReportV2(
            usage_unit="provider_tokens",
            budget=100,
            consumed=15,
            token_usage=TokenUsageBucketsV2(
                uncached_input_tokens=10,
                output_tokens=5,
                cache_read_tokens=0,
                cache_write_tokens=0,
            ),
            credits=None,
            budget_exhausted=False,
            session_count=1,
            model_request_count=1,
            usage_complete=True,
        ),
        output=EvolutionOutput(
            proposal_type="evolved",
            kernel_agent_revision_id=parent_id,
            hypothesis="Register pressure limits occupancy",
            expected_effect="Increase occupancy",
            changed_paths=("prompts/episode.md",),
            unimplemented_capabilities=(
                {
                    "capability": "Live occupancy modeling",
                    "expected_benefit": "Choose launch configurations with fewer attempts",
                    "reason_unimplemented": "No live profiler is available to the Evolver",
                },
            ),
        ),
        candidate=candidate,
    )
    evolution_digest = artifacts.put_json(
        trace.model_dump(mode="json"),
        ArtifactKind.EVOLUTION,
    )
    challenger = KernelAgentRevision(
        id=challenger_id,
        parent_id=parent_id,
        creation_key="evolve:1",
        dsl=Dsl.TRITON,
        optimizer_digest=candidate.optimizer_digest,
        created_by="evolver",
        created_at=NOW,
        evolution_trace_digest=evolution_digest,
    )
    epoch_id = new_epoch_id()
    kernel_id = new_kernel_revision_id()
    epoch = Epoch(
        id=epoch_id,
        lineage_id=new_lineage_id(),
        number=1,
        active_kernel_agent_revision_id=parent_id,
        challenger_kernel_agent_revision_ids=(challenger_id,),
        starting_kernel_revision_id=kernel_id,
        evidence_checkpoint=digest("evidence"),
        challenger_count=1,
        trajectories_per_branch=1,
        attempts_per_trajectory=1,
        status=EpochStatus.COMPLETED,
        winner_kernel_agent_revision_id=challenger_id,
        best_kernel_revision_id=kernel_id,
        created_at=NOW,
        completed_at=NOW,
    )

    class EvolutionRegistry:
        def get_epoch(self, requested: object) -> Epoch:
            assert requested == epoch_id
            return epoch

        def get_kernel_agent_revision(self, requested: object) -> KernelAgentRevision:
            assert requested == challenger_id
            return challenger

        @staticmethod
        def list_epoch_challengers(_epoch_id: object) -> list[EpochChallenger]:
            return [
                EpochChallenger(
                    epoch_id=epoch_id,
                    challenger_ordinal=1,
                    kernel_agent_revision_id=challenger_id,
                    proposal_type=ChallengerProposalType.EVOLVED,
                    base_revision_id=parent_id,
                    evolution_trace_digest=evolution_digest,
                )
            ]

        @staticmethod
        def list_attempts(_epoch_id: object) -> tuple[()]:
            return ()

    class MeasurementSource:
        @staticmethod
        def list_evidence_measurements(
            attempt_ids: tuple[AttemptId, ...],
            *,
            limit: int,
        ) -> tuple[GatewayMeasurementRecord, ...]:
            assert attempt_ids == ()
            assert limit == 5_000
            return (
                GatewayMeasurementRecord(
                    id="gmeasure_0123456789abcdef0123456789abcdef",
                    attempt_id=cast(AttemptId, "attempt_history"),
                    recovery_generation=0,
                    ordinal=1,
                    source_operation=GatewayOperation.PROFILE,
                    idempotency_key="profile-1",
                    kernel_artifact_digest=digest("candidate"),
                    gateway_result_digest=digest("profile-result"),
                    point=GatewayMeasurementPoint(
                        kind=GatewayOperation.PROFILE,
                        profile_level="sol",
                        shape_id="opaque-shape-1",
                        kernel_name="kernel_0",
                        metrics={"compute_sol_pct": 72.5},
                    ),
                    created_at=NOW,
                ),
            )

        @staticmethod
        def list_kernel_trials(
            attempt_ids: tuple[AttemptId, ...],
            *,
            limit: int,
        ) -> tuple[()]:
            assert attempt_ids == ()
            assert limit == 5_000
            return ()

    staging = tmp_path / "staging"
    staging.mkdir()
    assembler = LocalEvidenceAssembler(
        cast(Registry, EvolutionRegistry()),
        artifacts,
        _projector(artifacts),
        MeasurementSource(),
    )

    derived = assembler._append_derived(staging, epoch_id, 1)

    assert derived["trace_projections"] == ["traces/00000001/evolver-0001.json"]
    lessons = json.loads((staging / "lessons/00000001.json").read_text(encoding="utf-8"))[
        "annotations"
    ]
    assert [lesson["kind"] for lesson in lessons] == [
        "evolver-authored-annotation",
        "evolver-session-annotation",
    ]
    assert lessons[1]["trusted"] is False
    assert lessons[1]["text"] == "Reduce register pressure"
    assert lessons[0]["unimplemented_capabilities"] == [
        {
            "capability": "Live occupancy modeling",
            "expected_benefit": "Choose launch configurations with fewer attempts",
            "reason_unimplemented": "No live profiler is available to the Evolver",
        }
    ]
    assert "do not retain" not in json.dumps(lessons)
    projection = json.loads((staging / "traces/00000001/evolver-0001.json").read_text())
    assert "raw_files" not in projection
    upload = _projector(artifacts).raw_session_projection(session_digest)
    raw = {entry["path"]: entry["content"] for entry in upload["files"]}
    assert raw["input/prompt.md"] == "raw private Evolver prompt"
    assert "raw private reasoning" in raw["provider/stdout.stream-json"]
    assert "thinking_tokens" not in raw["provider/stdout.stream-json"]
    assert raw["provider/stderr.log"] == "raw provider credential"
    measurements = json.loads((staging / "measurements/00000001.json").read_text())
    assert measurements["measurements"][0]["metrics"] == {"compute_sol_pct": 72.5}
    assert derived["measurements"] == "measurements/00000001.json"
