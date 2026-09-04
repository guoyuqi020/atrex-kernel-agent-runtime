"""Tests for role-scoped Evidence trees exposed to Optimizer and Evolver Agents."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from conftest import digest

from atrex_runtime.artifacts.local import ArtifactKind, LocalArtifactStore
from atrex_runtime.domain.ids import ArtifactDigest
from atrex_runtime.domain.models import BranchRole
from atrex_runtime.workers.evidence_view import (
    EVIDENCE_PROMPT_SHA256,
    EVIDENCE_PROMPT_TEXT,
    EVOLVER_EVIDENCE_PROMPT_TEXT,
    EvidenceViewManifestV1,
    _materialize_evolver_agent_reports,
    _materialize_evolver_agent_sessions,
    assemble_evolver_evidence_view,
    assemble_optimizer_evidence_view,
    evolver_agent_optimization_summary,
)


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _raw_trace(
    store: LocalArtifactStore,
    root: Path,
    label: str,
) -> ArtifactDigest:
    source = root / f"raw-{label}"
    (source / "input").mkdir(parents=True)
    (source / "provider").mkdir()
    (source / "input/prompt.md").write_text(
        f"private prompt {label}",
        encoding="utf-8",
    )
    provider_event = {
        "reasoning": f"hidden reasoning {label}",
        "tool_arguments": {"token": f"secret-{label}"},
        "tool_result": f"raw result {label}",
    }
    thinking_tokens = {
        "type": "system",
        "subtype": "thinking_tokens",
        "estimated_tokens": 12_345,
    }
    (source / "provider/stdout.stream-json").write_text(
        json.dumps(thinking_tokens) + "\n" + json.dumps(provider_event) + "\n",
        encoding="utf-8",
    )
    (source / "conversation.jsonl").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "sequence": 1,
                "type": "provider_event",
                "source": "provider",
                "path": "provider/stdout.stream-json",
                "event": thinking_tokens,
            }
        )
        + "\n"
        + json.dumps(
            {
                "schema_version": 1,
                "sequence": 2,
                "type": "provider_event",
                "source": "provider",
                "path": "provider/stdout.stream-json",
                "event": provider_event,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (source / "provider/stderr.log").write_text(
        f"Bearer credential-{label}",
        encoding="utf-8",
    )
    return store.put_directory(source, ArtifactKind.SESSION_LOG)


def _trace_digests(store: LocalArtifactStore, root: Path) -> dict[str, ArtifactDigest]:
    return {
        label: _raw_trace(store, root, label)
        for label in ("active", "challenger", "evolver", "current-1", "current-2")
    }


def _kernel_digest(
    store: LocalArtifactStore,
    root: Path,
    label: str,
) -> ArtifactDigest:
    source = root / f"kernel-{label}"
    source.mkdir(parents=True)
    (source / "kernel.py").write_text(f"# {label}\n", encoding="utf-8")
    return store.put_directory(source, ArtifactKind.KERNEL)


def _lineage(
    root: Path,
    trace_digests: dict[str, ArtifactDigest],
    store: LocalArtifactStore,
    *,
    winner: str = "active",
) -> Path:
    (root / "bootstrap").mkdir(parents=True)
    _write(root / "bootstrap/report.json", {"status": "baseline_ready"})
    (root / "bootstrap/conversation.jsonl").write_text(
        '{"type":"assistant/message","text":"bootstrap complete"}\n',
        encoding="utf-8",
    )
    _write(
        root / "checkpoint.json",
        {
            "schema_version": 1,
            "lineage_id": "lineage_0123456789abcdef0123456789abcdef",
            "through_epoch": 1,
            "previous_checkpoint_digest": str(digest("previous")),
        },
    )
    kernel_ids = {
        "starting": "kernelrev_00000000000000000000000000000000",
        "active": "kernelrev_11111111111111111111111111111111",
        "challenger": "kernelrev_22222222222222222222222222222222",
    }
    kernel_digests = {
        label: _kernel_digest(store, root / "kernel-sources", label) for label in kernel_ids
    }
    gateway_digests = {
        label: store.put_json(
            {
                "operation": "evaluate",
                "status": "completed",
                "result": {
                    "all_pass": True,
                    "latency_us_geomean": latency,
                    "latency_us_by_shape": {
                        "0": latency - 1.0,
                        "1": latency + 1.0,
                    },
                },
            },
            ArtifactKind.GATEWAY_RESULT,
        )
        for label, latency in (("starting", 12.0), ("active", 11.0), ("challenger", 9.0))
    }
    trial_digest = _kernel_digest(store, root / "kernel-sources", "reverted-trial")
    trial_result_digest = digest("raw-trial-result")
    trial_response_digest = store.put_json(
        {
            "schema_version": 2,
            "operation": "evaluate",
            "result": {"correct": True, "latency_us": 13.0},
        },
        ArtifactKind.GATEWAY_RESULT,
    )
    attempts = []
    for branch, ordinal, attempt_id in (
        ("active", 1, "attempt_active"),
        ("challenger", 1, "attempt_challenger"),
    ):
        attempts.append(
            {
                "attempt_id": attempt_id,
                "branch": branch,
                "challenger_ordinal": 0 if branch == "active" else 1,
                "trajectory_ordinal": 1,
                "ordinal": ordinal,
                "kernel_agent_revision_id": f"agent_{branch}",
                "input_kernel_revision_id": kernel_ids["starting"],
                "accepted_as_branch_best": branch == winner,
                "output": {
                    "kernel_revision_id": kernel_ids[branch],
                    "artifact_digest": str(kernel_digests[branch]),
                    "correct": True,
                    "latency_us": 9.0 if branch == winner else 11.0,
                    "gateway_result_digest": str(gateway_digests[branch]),
                },
            }
        )
        _write(
            root / f"reports/00000001/{attempt_id}.json",
            {
                "attempt_id": attempt_id,
                "branch": branch,
                "direction_events": [
                    {
                        "direction_event_id": "directionevent_"
                        + ("a" if branch == "active" else "b") * 32,
                        "direction_id": "direction_" + ("a" if branch == "active" else "b") * 32,
                        "recorded_at": "2026-08-24T00:00:00+00:00",
                        "action": "propose",
                        "name": f"{branch} direction",
                        "hypothesis": "test hypothesis",
                        "rationale": "test rationale",
                        "plan": ["test step"],
                        "success_criteria": "test succeeds",
                        "stop_conditions": "test fails",
                        "analysis": None,
                        "supporting_experiment_ids": [],
                    }
                ],
            },
        )
        _write(root / f"diffs/00000001/{attempt_id}.json", {"changes": []})
        _write(
            root / f"traces/00000001/{attempt_id}-run-0001.json",
            {
                "schema_version": 1,
                "source_session_log_digest": str(trace_digests[branch]),
                "sessions": [],
            },
        )
    _write(
        root / "traces/00000001/attempt_active-run-0002.json",
        {
            "schema_version": 1,
            "source_session_log_digest": str(trace_digests["current-1"]),
            "sessions": [],
        },
    )
    _write(
        root / "epochs/00000001.json",
        {
            "schema_version": 1,
            "number": 1,
            "active_kernel_agent_revision_id": "agent_active",
            "challenger_kernel_agent_revision_ids": ["agent_challenger"],
            "winner_kernel_agent_revision_id": f"agent_{winner}",
            "selection_reason": "authoritative_comparison",
            "starting_kernel_revision_id": kernel_ids["starting"],
            "starting_kernel": {
                "kernel_revision_id": kernel_ids["starting"],
                "artifact_digest": str(kernel_digests["starting"]),
                "correct": True,
                "latency_us": 12.0,
                "gateway_result_digest": str(gateway_digests["starting"]),
            },
            "best_kernel_revision_id": kernel_ids[winner],
            "best_kernel": {
                "kernel_revision_id": kernel_ids[winner],
                "artifact_digest": str(kernel_digests[winner]),
                "correct": True,
                "latency_us": 9.0,
                "gateway_result_digest": str(gateway_digests[winner]),
            },
            "attempts": attempts,
        },
    )
    _write(
        root / "lessons/00000001.json",
        {
            "schema_version": 1,
            "annotations": [
                {"branch": "active", "text": "promoted lesson"},
                {"branch": "challenger", "text": "losing negative lesson"},
            ],
        },
    )
    _write(
        root / "traces/00000001/evolver-0001.json",
        {
            "schema_version": 2,
            "source_session_log_digest": str(trace_digests["evolver"]),
            "sessions": [],
        },
    )
    _write(
        root / "measurements/00000001.json",
        {
            "schema_version": 1,
            "epoch_id": "epoch_0123456789abcdef0123456789abcdef",
            "measurements": [
                {
                    "measurement_id": f"measurement_{branch}",
                    "attempt_id": attempt_id,
                    "kernel_artifact_digest": str(kernel_digests[branch]),
                    "operation": "evaluate",
                    "source_operation": "evaluate",
                    "profile_level": None,
                    "shape_id": "opaque-shape-1",
                    "kernel_name": None,
                    "metrics": {"latency_us": 9.0 if branch == winner else 11.0},
                    "gateway_result_digest": str(gateway_digests[branch]),
                    "created_at": "2026-08-22T00:00:00+00:00",
                }
                for branch, attempt_id in (
                    ("active", "attempt_active"),
                    ("challenger", "attempt_challenger"),
                )
            ],
        },
    )
    _write(
        root / "kernel-trials/00000001.json",
        {
            "schema_version": 1,
            "epoch_id": "epoch_0123456789abcdef0123456789abcdef",
            "kernel_trials": [
                {
                    "kernel_trial_id": "gtrial_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                    "attempt_id": "attempt_active",
                    "recovery_generation": 0,
                    "ordinal": 1,
                    "kernel_artifact_digest": str(trial_digest),
                    "disposition": "revert",
                    "observations": [
                        {
                            "operation": "evaluate",
                            "gateway_result_digest": str(trial_result_digest),
                            "result_artifact_digest": str(trial_response_digest),
                        }
                    ],
                    "annotations": [{"disposition": "revert"}],
                    "created_at": "2026-08-22T00:00:00+00:00",
                }
            ],
        },
    )
    return root


def _current_branch(
    root: Path,
    checkpoint: str,
    trace_digests: dict[str, ArtifactDigest],
) -> Path:
    _write(
        root / "context.json",
        {
            "schema_version": 1,
            "epoch_id": "epoch_0123456789abcdef0123456789abcdef",
            "attempt_id": "attempt_current",
            "branch": "active",
            "challenger_ordinal": 0,
            "trajectory_ordinal": 1,
            "ordinal": 3,
            "epoch_evidence_checkpoint": checkpoint,
            "previous_attempt_ids": ["attempt_one", "attempt_two"],
        },
    )
    _write(root / "lessons.json", {"schema_version": 1, "annotations": []})
    for ordinal in (1, 2):
        _write(
            root / f"attempts/{ordinal:08d}.json",
            {
                "attempt_id": f"attempt_{ordinal}",
                "branch": "active",
                "challenger_ordinal": 0,
                "trajectory_ordinal": 1,
                "ordinal": ordinal,
                "kernel_agent_revision_id": "agent_active",
            },
        )
        _write(
            root / f"reports/{ordinal:08d}.json",
            {"ordinal": ordinal, "branch": "active"},
        )
        _write(root / f"diffs/{ordinal:08d}.json", {"ordinal": ordinal})
        _write(
            root / f"traces/{ordinal:08d}-run-0001.json",
            {
                "schema_version": 1,
                "source_session_log_digest": str(trace_digests[f"current-{ordinal}"]),
                "sessions": [],
            },
        )
    return root


def test_same_agent_branches_keep_both_sessions_reports_and_one_career_epoch(
    tmp_path: Path,
) -> None:
    evidence = tmp_path / "evidence"
    epoch = evidence / "epochs/00000001"
    _write(
        epoch / "summary.json",
        {
            "selection_reason": "incumbent_retained",
            "branches": [
                {
                    "branch": "active",
                    "challenger_ordinal": 0,
                    "kernel_agent_revision_id": "same",
                    "selected": True,
                },
                {
                    "branch": "challenger",
                    "challenger_ordinal": 1,
                    "kernel_agent_revision_id": "same",
                    "selected": True,
                },
            ],
        },
    )
    for branch in ("active", "challenger-0001"):
        root = epoch / "branches" / branch / "trajectories/00000001/attempts/00000001"
        _write(root / "summary.json", {"trajectory_ordinal": 1, "ordinal": 1})
        _write(root / "report.json", {"from": branch})
        trace = root / "traces/run-0001/conversation.jsonl"
        trace.parent.mkdir(parents=True)
        trace.write_text(branch)
    _materialize_evolver_agent_sessions(evidence, "same", tmp_path / "sessions")
    _materialize_evolver_agent_reports(evidence, "same", tmp_path / "reports")
    for ordinal, branch in enumerate(("active", "challenger-0001"), start=1):
        relative = f"trajectory-{ordinal:08d}/attempt-00000001"
        assert (tmp_path / "sessions" / f"{relative}.conversation.jsonl").read_text() == branch
        assert json.loads((tmp_path / "reports" / f"{relative}.report.json").read_text()) == {
            "from": branch
        }
    summary = evolver_agent_optimization_summary(
        evidence,
        "same",
        LocalArtifactStore(tmp_path / "artifacts"),
        version="agent-v0",
    )
    assert summary["latest_epoch"]["branch"] == "active_and_replica"
    assert summary["latest_epoch"]["attempt_count"] == 2
    assert summary["career"] == {"epoch_participation_count": 1, "win_count": 1, "loss_count": 0}


def test_evolver_sessions_expose_only_latest_epoch_and_latest_attempt_run(
    tmp_path: Path,
) -> None:
    evidence = tmp_path / "evidence"
    (evidence / "bootstrap").mkdir(parents=True)
    (evidence / "bootstrap/conversation.jsonl").write_text("bootstrap\n")
    for epoch in (1, 2):
        epoch_root = evidence / "epochs" / f"{epoch:08d}"
        _write(
            epoch_root / "summary.json",
            {
                "branches": [
                    {
                        "branch": "active",
                        "challenger_ordinal": 0,
                        "kernel_agent_revision_id": "agent_active",
                    },
                    {
                        "branch": "challenger",
                        "challenger_ordinal": 1,
                        "kernel_agent_revision_id": "agent_loser",
                    },
                ]
            },
        )
        for branch_label, trajectory in (("active", 1), ("challenger-0001", 2)):
            attempt = (
                epoch_root
                / f"branches/{branch_label}/trajectories/{trajectory:08d}/attempts/00000001"
            )
            _write(
                attempt / "summary.json",
                {"trajectory_ordinal": trajectory, "ordinal": 1},
            )
            for run in (1, 2):
                conversation = attempt / f"traces/run-{run:04d}/conversation.jsonl"
                conversation.parent.mkdir(parents=True)
                conversation.write_text(f"{branch_label} epoch {epoch} run {run}\n")

    destination = tmp_path / "sessions"
    _materialize_evolver_agent_sessions(evidence, "agent_active", destination)
    losing = tmp_path / "losing-sessions"
    _materialize_evolver_agent_sessions(evidence, "agent_loser", losing)

    conversation = destination / "trajectory-00000001/attempt-00000001.conversation.jsonl"
    assert conversation.read_text() == "active epoch 2 run 2\n"
    assert [path.relative_to(destination).as_posix() for path in destination.rglob("*")] == [
        "trajectory-00000001",
        "trajectory-00000001/attempt-00000001.conversation.jsonl",
    ]
    losing_conversation = losing / "trajectory-00000002/attempt-00000001.conversation.jsonl"
    assert losing_conversation.read_text() == "challenger-0001 epoch 2 run 2\n"
    assert [path.relative_to(losing).as_posix() for path in losing.rglob("*")] == [
        "trajectory-00000002",
        "trajectory-00000002/attempt-00000001.conversation.jsonl",
    ]


def test_evolver_view_summarizes_non_pool_versions_without_sessions_or_reports(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "view"
    store = LocalArtifactStore(tmp_path / "artifacts")
    traces = _trace_digests(store, tmp_path / "trace-sources")

    assemble_evolver_evidence_view(
        destination,
        control_root=tmp_path / ".runtime",
        lineage_payload=_lineage(tmp_path / "lineage", traces, store, winner="challenger"),
        lineage_checkpoint=digest("lineage"),
        artifacts=store,
        agent_versions={
            "agent-v0": "agent_active",
            "agent-v1": "agent_challenger",
            "agent-v2": "agent_fresh",
        },
        pool_versions=frozenset({"agent-v0", "agent-v1"}),
    )

    summary = json.loads((destination / "agent-v2/optimization-summary.json").read_text())
    assert summary == {
        "kernel_agent_revision_id": "agent_fresh",
        "version": "agent-v2",
        "path": "input/agents/agent-v2",
        "resources_path": "input/evidence/agent-v2/resources",
        "latest_epoch": None,
        "career": {"epoch_participation_count": 0, "win_count": 0, "loss_count": 0},
    }
    assert not (destination / "agent-v2/sessions").exists()
    assert not (destination / "agent-v2/reports").exists()
    for version in ("agent-v0", "agent-v1"):
        assert (destination / version / "sessions").is_dir()
        assert (destination / version / "reports").is_dir()


def test_evolver_view_rejects_a_pool_version_outside_the_visible_versions(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path / "artifacts")
    traces = _trace_digests(store, tmp_path / "trace-sources")

    with pytest.raises(ValueError, match="outside the visible Agent versions"):
        assemble_evolver_evidence_view(
            tmp_path / "view",
            control_root=tmp_path / ".runtime",
            lineage_payload=_lineage(tmp_path / "lineage", traces, store),
            lineage_checkpoint=digest("lineage"),
            artifacts=store,
            agent_versions={"agent-v0": "agent_active"},
            pool_versions=frozenset({"agent-v0", "agent-v9"}),
        )


def test_optimizer_view_projects_every_completed_branch_by_epoch(tmp_path: Path) -> None:
    checkpoint = digest("lineage")
    destination = tmp_path / "view"
    control_root = tmp_path / ".runtime"
    store = LocalArtifactStore(tmp_path / "artifacts")
    traces = _trace_digests(store, tmp_path / "trace-sources")

    manifest = assemble_optimizer_evidence_view(
        destination,
        control_root=control_root,
        lineage_payload=_lineage(tmp_path / "lineage", traces, store),
        lineage_checkpoint=checkpoint,
        attempt_payload=_current_branch(tmp_path / "attempt", str(checkpoint), traces),
        attempt_snapshot=digest("attempt"),
        current_epoch_number=2,
        branch=BranchRole.ACTIVE,
        challenger_ordinal=0,
        trajectory_ordinal=1,
        selected_revision="agent_active",
        attempt_ordinal=3,
        artifacts=store,
    )

    assert EvidenceViewManifestV1.from_file(control_root / "evidence-manifest.json") == manifest
    assert manifest.prompt_fragment_sha256 == EVIDENCE_PROMPT_SHA256
    assert manifest.visibility.completed_epochs == "all_completed_branches"
    assert manifest.visibility.current_trajectory_ordinal == 1
    assert (control_root / "evidence-instructions.md").read_text() == EVIDENCE_PROMPT_TEXT
    assert not (destination / "manifest.json").exists()
    assert not (destination / "instructions.md").exists()
    assert "layout v1" not in EVIDENCE_PROMPT_TEXT.lower()
    assert "call `kernel-trials`" not in EVIDENCE_PROMPT_TEXT
    assert "supplied Runtime-local query commands" in EVIDENCE_PROMPT_TEXT
    assert "durably appended by Runtime before its tool call returns" in EVIDENCE_PROMPT_TEXT
    assert "`kernel-trial-show`" not in EVIDENCE_PROMPT_TEXT
    assert "`kernel-artifact-read`" not in EVIDENCE_PROMPT_TEXT
    assert "`gateway-result-read`" not in EVIDENCE_PROMPT_TEXT
    assert "`list-directions`" not in EVIDENCE_PROMPT_TEXT
    assert "`load-direction`" not in EVIDENCE_PROMPT_TEXT
    assert "`measurements-query`" not in EVIDENCE_PROMPT_TEXT
    assert '"operation":"kernel_trials"' not in EVIDENCE_PROMPT_TEXT
    assert "Kernel, Trial, Result, Direction, and Experiment" in EVIDENCE_PROMPT_TEXT
    assert "Epochs then form one serial Lineage" in " ".join(EVIDENCE_PROMPT_TEXT.split())
    assert "controller independently selects" in EVIDENCE_PROMPT_TEXT
    assert "They may have different producers" in EVIDENCE_PROMPT_TEXT
    assert "promoted Agent/Kernel trajectory" not in EVIDENCE_PROMPT_TEXT
    assert "every branch that ran in it" in EVIDENCE_PROMPT_TEXT
    assert "never a concurrently running sibling" in EVIDENCE_PROMPT_TEXT
    completed = destination / "epochs/00000001"
    current = destination / "epochs/00000002"
    assert json.loads((completed / "summary.json").read_text()) == {
        "schema_version": 1,
        "number": 1,
        "branches": [
            {"branch": "active", "challenger_ordinal": 0, "selected": True},
            {"branch": "challenger", "challenger_ordinal": 1, "selected": False},
        ],
    }
    completed_attempt_root = completed / "branches/active/trajectories/00000001/attempts/00000001"
    losing_attempt_root = (
        completed / "branches/challenger-0001/trajectories/00000001/attempts/00000001"
    )
    for attempt_root, branch in (
        (completed_attempt_root, "active"),
        (losing_attempt_root, "challenger"),
    ):
        report = json.loads((attempt_root / "report.json").read_text())
        assert report["branch"] == branch
        assert {path.name for path in attempt_root.iterdir()} == {
            "report.json",
            "conversation.jsonl",
        }
    for aggregate in ("lessons.json", "measurements.json"):
        assert not (completed / aggregate).exists()
        assert not (current / aggregate).exists()
    assert not (current / "summary.json").exists()
    assert {path.name for path in completed.iterdir()} == {"summary.json", "branches"}
    assert {path.name for path in current.iterdir()} == {"trajectories"}
    assert not (completed / "trajectories").exists()
    assert not (completed / "evolution").exists()
    assert not (completed / "experiment-history.json").exists()
    assert not (completed / "direction-history.json").exists()
    assert not (control_root / "journal-history").exists()
    completed_trace = completed_attempt_root / "conversation.jsonl"
    assert "hidden reasoning current-1" in completed_trace.read_text()
    assert "secret-current-1" in completed_trace.read_text()
    assert "raw result current-1" in completed_trace.read_text()
    assert "thinking_tokens" not in completed_trace.read_text()
    losing_trace = losing_attempt_root / "conversation.jsonl"
    assert "hidden reasoning challenger" in losing_trace.read_text()
    assert "raw result challenger" in losing_trace.read_text()
    assert "thinking_tokens" not in losing_trace.read_text()
    source_trace = store.verify(traces["active"]).payload_path
    assert "thinking_tokens" in (source_trace / "provider/stdout.stream-json").read_text()
    assert "thinking_tokens" in (source_trace / "conversation.jsonl").read_text()
    current_attempt = current / "trajectories/00000001/attempts/00000002"
    assert (current_attempt / "report.json").is_file()
    assert '"branch"' not in (current_attempt / "report.json").read_text()
    assert "secret-current-2" in (current_attempt / "conversation.jsonl").read_text()
    assert {path.name for path in current_attempt.iterdir()} == {
        "report.json",
        "conversation.jsonl",
    }
    assert not (current / "attempts").exists()
    assert not (current / "branches").exists()
    assert not (destination / "lineage").exists()
    assert not (destination / "trigger-window").exists()
    assert os.stat(control_root / "evidence-manifest.json").st_mode & 0o200 == 0


def test_evolver_view_contains_only_completed_epoch_history(tmp_path: Path) -> None:
    destination = tmp_path / "view"
    control_root = tmp_path / ".runtime"
    store = LocalArtifactStore(tmp_path / "artifacts")
    traces = _trace_digests(store, tmp_path / "trace-sources")
    manifest = assemble_evolver_evidence_view(
        destination,
        control_root=control_root,
        lineage_payload=_lineage(
            tmp_path / "lineage",
            traces,
            store,
            winner="challenger",
        ),
        lineage_checkpoint=digest("lineage"),
        artifacts=store,
        agent_versions={
            "agent-v0": "agent_active",
            "agent-v1": "agent_challenger",
        },
        pool_versions=frozenset({"agent-v0", "agent-v1"}),
    )

    assert manifest.role == "evolver"
    assert not (control_root / "evidence-instructions.md").exists()
    assert EVOLVER_EVIDENCE_PROMPT_TEXT
    assert (
        "Each of the six reusable directories has a mandatory `README.md` index"
        in EVOLVER_EVIDENCE_PROMPT_TEXT
    )
    assert "`candidate/` starts as a writable" in EVOLVER_EVIDENCE_PROMPT_TEXT
    assert "{prompts,memory,knowledge,skills,tools,hooks}/" in EVOLVER_EVIDENCE_PROMPT_TEXT
    assert "candidate/runtime-state/" not in EVOLVER_EVIDENCE_PROMPT_TEXT
    assert "revision seed" in EVOLVER_EVIDENCE_PROMPT_TEXT
    assert (
        "Candidate resources seed its next optimization trajectories"
        in EVOLVER_EVIDENCE_PROMPT_TEXT
    )
    assert "`secondary_criteria`" in EVOLVER_EVIDENCE_PROMPT_TEXT
    assert "`incumbent_retained`" in EVOLVER_EVIDENCE_PROMPT_TEXT
    assert "`identical_kernel`" in EVOLVER_EVIDENCE_PROMPT_TEXT
    assert "does not imply the retained winner's raw" in EVOLVER_EVIDENCE_PROMPT_TEXT
    assert "not the complete tournament history" in EVOLVER_EVIDENCE_PROMPT_TEXT
    assert "tells you which `agent-vN` competed in it" in EVOLVER_EVIDENCE_PROMPT_TEXT
    assert "`current_epoch_challenger` — created earlier in the current Epoch" in (
        EVOLVER_EVIDENCE_PROMPT_TEXT
    )
    assert "`lineage_history` — a completed version outside that pool" in (
        EVOLVER_EVIDENCE_PROMPT_TEXT
    )
    assert "Read `latest_epoch.epoch_number` from either pool" in EVOLVER_EVIDENCE_PROMPT_TEXT
    assert manifest.prompt_fragment_sha256 != EVIDENCE_PROMPT_SHA256
    assert manifest.current_epoch is None
    assert manifest.visibility.completed_epochs == "all_completed_branches"
    assert manifest.visibility.current_trajectory_ordinal is None
    assert not (destination / "epochs").exists()
    assert not (destination / "bootstrap").exists()
    active_effect = json.loads((destination / "agent-v0/optimization-summary.json").read_text())
    assert active_effect["version"] == "agent-v0"
    assert active_effect["path"] == "input/agents/agent-v0"
    assert active_effect["resources_path"] == "input/evidence/agent-v0/resources"
    assert active_effect["latest_epoch"]["attempt_count"] == 1
    assert active_effect["latest_epoch"]["correct_attempt_count"] == 1
    assert active_effect["latest_epoch"]["branch"] == "active"
    assert active_effect["latest_epoch"]["challenger_ordinal"] is None
    assert active_effect["latest_epoch"]["outcome"] == "lost"
    assert active_effect["latest_epoch"]["selection_reason"] == "authoritative_comparison"
    assert active_effect["career"] == {
        "epoch_participation_count": 1,
        "loss_count": 1,
        "win_count": 0,
    }
    challenger_effect = json.loads((destination / "agent-v1/optimization-summary.json").read_text())
    assert challenger_effect["latest_epoch"] == {
        "attempt_count": 1,
        "branch": "challenger",
        "challenger_ordinal": 1,
        "outcome": "won",
        "selection_reason": "authoritative_comparison",
        "best_kernel": {
            "gateway_result": {
                "correct": True,
                "correctness": {
                    "max_abs_err": None,
                    "max_rel_err": None,
                    "rel_err": None,
                    "status": "PASS",
                },
                "latency_us_arith_mean": 9.0,
                "latency_us_by_shape": {"0": 8.0, "1": 10.0},
                "latency_us_geomean": 9.0,
                "status": "completed",
            },
        },
        "correct_attempt_count": 1,
        "epoch_number": 1,
        "incorrect_attempt_count": 0,
        "no_candidate_attempt_count": 0,
    }
    assert challenger_effect["career"] == {
        "epoch_participation_count": 1,
        "loss_count": 0,
        "win_count": 1,
    }
    challenger_sessions = destination / "agent-v1/sessions"
    assert [path.name for path in challenger_sessions.iterdir()] == ["trajectory-00000001"]
    challenger_conversation = (
        challenger_sessions / "trajectory-00000001/attempt-00000001.conversation.jsonl"
    )
    assert "hidden reasoning challenger" in challenger_conversation.read_text()
    active_conversation = (
        destination / "agent-v0/sessions/trajectory-00000001/attempt-00000001.conversation.jsonl"
    )
    assert "hidden reasoning current-1" in active_conversation.read_text()
    for version, branch in (("agent-v0", "active"), ("agent-v1", "challenger")):
        report = json.loads(
            (
                destination / version / "reports/trajectory-00000001/attempt-00000001.report.json"
            ).read_text()
        )
        assert report["branch"] == branch
        assert report["direction_events"]
    assert not any(destination.rglob("bootstrap.conversation.jsonl"))
    assert not (control_root / "evolver-evidence-data").exists()
    assert not (destination / "epochs/00000002").exists()
