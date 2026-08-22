"""Tests for role-scoped Evidence trees exposed to Optimizer and Evolver Agents."""

from __future__ import annotations

import json
import os
from pathlib import Path

from conftest import digest

from atrex_runtime.artifacts.local import ArtifactKind, LocalArtifactStore
from atrex_runtime.domain.ids import ArtifactDigest
from atrex_runtime.domain.models import BranchRole
from atrex_runtime.workers.evidence_view import (
    EVIDENCE_PROMPT_SHA256,
    EVIDENCE_PROMPT_TEXT,
    EvidenceViewManifestV1,
    assemble_evolver_evidence_view,
    assemble_optimizer_evidence_view,
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
    (root / "bootstrap/seed.md").write_text("seed", encoding="utf-8")
    _write(root / "bootstrap-metadata.json", {"schema_version": 1, "source": "test"})
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
    trial_digest = _kernel_digest(store, root / "kernel-sources", "reverted-trial")
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
                    "gateway_result_digest": str(digest(f"gateway-{branch}")),
                },
            }
        )
        _write(
            root / f"reports/00000001/{attempt_id}.json",
            {"attempt_id": attempt_id, "branch": branch},
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
        root / "epochs/00000001.json",
        {
            "schema_version": 1,
            "number": 1,
            "active_kernel_agent_revision_id": "agent_active",
            "challenger_kernel_agent_revision_ids": ["agent_challenger"],
            "winner_kernel_agent_revision_id": f"agent_{winner}",
            "starting_kernel_revision_id": kernel_ids["starting"],
            "starting_kernel": {
                "kernel_revision_id": kernel_ids["starting"],
                "artifact_digest": str(kernel_digests["starting"]),
                "correct": True,
                "latency_us": 12.0,
                "gateway_result_digest": str(digest("gateway-starting")),
            },
            "best_kernel_revision_id": kernel_ids[winner],
            "best_kernel": {
                "kernel_revision_id": kernel_ids[winner],
                "artifact_digest": str(kernel_digests[winner]),
                "correct": True,
                "latency_us": 9.0,
                "gateway_result_digest": str(digest(f"gateway-{winner}")),
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
                    "gateway_result_digest": str(digest(f"gateway-{branch}")),
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
                    "candidate_artifact_digest": str(trial_digest),
                    "disposition": "revert",
                    "observations": [{"operation": "evaluate"}],
                    "annotations": [{"decision": "revert"}],
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


def test_optimizer_view_projects_one_promoted_agent_lineage_by_epoch(tmp_path: Path) -> None:
    checkpoint = digest("lineage")
    destination = tmp_path / "view"
    store = LocalArtifactStore(tmp_path / "artifacts")
    traces = _trace_digests(store, tmp_path / "trace-sources")

    manifest = assemble_optimizer_evidence_view(
        destination,
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

    assert EvidenceViewManifestV1.from_file(destination / "manifest.json") == manifest
    assert manifest.prompt_fragment_sha256 == EVIDENCE_PROMPT_SHA256
    assert (destination / "instructions.md").read_text() == EVIDENCE_PROMPT_TEXT
    assert "layout v1" not in EVIDENCE_PROMPT_TEXT.lower()
    assert '"operation":"kernel_trials"' in EVIDENCE_PROMPT_TEXT
    assert '"operation":"kernel_trial_read"' in EVIDENCE_PROMPT_TEXT
    assert '"operation":"measurements","kernel_artifact_digest"' in EVIDENCE_PROMPT_TEXT
    assert "only `candidate_artifact_digest` identifies" in EVIDENCE_PROMPT_TEXT
    completed = destination / "epochs/00000001"
    current = destination / "epochs/00000002"
    completed_attempt_root = completed / "trajectories/00000001/attempts/00000001"
    assert (completed_attempt_root / "report.json").is_file()
    completed_attempt = json.loads((completed_attempt_root / "summary.json").read_text())
    assert completed_attempt["attempt_id"] == "attempt_active"
    assert '"branch"' not in (completed_attempt_root / "summary.json").read_text()
    assert '"branch"' not in (completed_attempt_root / "report.json").read_text()
    lessons = json.loads((completed / "lessons.json").read_text())
    assert [item["text"] for item in lessons["annotations"]] == [
        "promoted lesson",
        "losing negative lesson",
    ]
    assert '"branch"' not in json.dumps(lessons)
    assert not (completed / "branches").exists()
    assert not (completed / "evolution").exists()
    measurements = json.loads((completed / "measurements.json").read_text())["measurements"]
    assert [item["attempt_id"] for item in measurements] == ["attempt_active"]
    completed_trace = completed_attempt_root / "traces/run-0001/provider/stdout.stream-json"
    assert (
        "private prompt active"
        in (completed_attempt_root / "traces/run-0001/input/prompt.md").read_text()
    )
    assert "hidden reasoning active" in completed_trace.read_text()
    assert "secret-active" in completed_trace.read_text()
    assert "raw result active" in completed_trace.read_text()
    assert "thinking_tokens" not in completed_trace.read_text()
    assert (
        "thinking_tokens"
        not in (completed_attempt_root / "traces/run-0001/conversation.jsonl").read_text()
    )
    source_trace = store.verify(traces["active"]).payload_path
    assert "thinking_tokens" in (source_trace / "provider/stdout.stream-json").read_text()
    assert "thinking_tokens" in (source_trace / "conversation.jsonl").read_text()
    assert (
        "credential-active"
        in (completed_attempt_root / "traces/run-0001/provider/stderr.log").read_text()
    )
    assert not list((completed_attempt_root / "traces").glob("*.json"))
    assert (current / "attempts/00000002/report.json").is_file()
    assert '"branch"' not in (current / "attempts/00000002/report.json").read_text()
    assert (
        "secret-current-2"
        in (current / "attempts/00000002/traces/run-0001/provider/stdout.stream-json").read_text()
    )
    assert not (current / "branches").exists()
    assert not (destination / "lineage").exists()
    assert not (destination / "trigger-window").exists()
    assert os.stat(destination / "manifest.json").st_mode & 0o200 == 0


def test_evolver_view_contains_only_completed_epoch_history(tmp_path: Path) -> None:
    destination = tmp_path / "view"
    store = LocalArtifactStore(tmp_path / "artifacts")
    traces = _trace_digests(store, tmp_path / "trace-sources")

    manifest = assemble_evolver_evidence_view(
        destination,
        lineage_payload=_lineage(
            tmp_path / "lineage",
            traces,
            store,
            winner="challenger",
        ),
        lineage_checkpoint=digest("lineage"),
        artifacts=store,
    )

    assert manifest.role == "evolver"
    assert manifest.current_epoch is None
    assert manifest.visibility.completed_epochs == "all_completed_branches"
    assert (destination / "epochs/00000001/summary.json").is_file()
    epoch_root = destination / "epochs/00000001"
    summary = json.loads((epoch_root / "summary.json").read_text())
    assert summary["winner_kernel_agent_revision_id"] == "agent_challenger"
    assert [branch["selected"] for branch in summary["branches"]] == [False, True]
    active_root = epoch_root / "branches/active/trajectories/00000001/attempts/00000001"
    challenger_root = (
        epoch_root / "branches/challenger-0001/trajectories/00000001/attempts/00000001"
    )
    active_attempt = json.loads((active_root / "summary.json").read_text())
    challenger_attempt = json.loads((challenger_root / "summary.json").read_text())
    assert active_attempt["attempt_id"] == "attempt_active"
    assert active_attempt["branch"] == "active"
    assert challenger_attempt["attempt_id"] == "attempt_challenger"
    assert challenger_attempt["branch"] == "challenger"
    evolution_trace = epoch_root / "evolution/challenger-0001/trace/provider/stdout.stream-json"
    assert (
        "private prompt evolver"
        in (epoch_root / "evolution/challenger-0001/trace/input/prompt.md").read_text()
    )
    assert "secret-evolver" in evolution_trace.read_text()
    attempt_trace = challenger_root / "traces/run-0001/provider/stdout.stream-json"
    assert (
        "private prompt challenger"
        in (challenger_root / "traces/run-0001/input/prompt.md").read_text()
    )
    assert "hidden reasoning challenger" in attempt_trace.read_text()
    assert "secret-challenger" in attempt_trace.read_text()
    assert (
        "secret-active" in (active_root / "traces/run-0001/provider/stdout.stream-json").read_text()
    )
    lessons = json.loads((epoch_root / "lessons.json").read_text())
    assert {item["branch"] for item in lessons["annotations"]} == {
        "active",
        "challenger",
    }
    kernel_index = json.loads((epoch_root / "kernels/index.json").read_text())
    assert {item["kernel_revision_id"] for item in kernel_index["kernels"]} == {
        "kernelrev_00000000000000000000000000000000",
        "kernelrev_11111111111111111111111111111111",
        "kernelrev_22222222222222222222222222222222",
    }
    measurements = json.loads((epoch_root / "measurements.json").read_text())["measurements"]
    assert {item["attempt_id"] for item in measurements} == {
        "attempt_active",
        "attempt_challenger",
    }
    assert (
        epoch_root / "kernels/kernelrev_22222222222222222222222222222222/kernel.py"
    ).read_text() == "# challenger\n"
    trial_index = json.loads((epoch_root / "kernel-trials/index.json").read_text())
    assert trial_index["kernel_trials"][0]["disposition"] == "revert"
    assert (
        epoch_root / "kernel-trials/gtrial_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa/source/kernel.py"
    ).read_text() == "# reverted-trial\n"
    assert not (destination / "epochs/00000002").exists()
