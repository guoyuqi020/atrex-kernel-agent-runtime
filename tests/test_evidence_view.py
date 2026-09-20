"""Tests for role-scoped Evidence trees exposed to Optimizer and Evolver Agents."""

from __future__ import annotations

import ast
import json
import os
import re
from pathlib import Path

import pytest
from conftest import digest

from atrex_runtime.artifacts.local import ArtifactKind, LocalArtifactStore
from atrex_runtime.domain.ids import ArtifactDigest
from atrex_runtime.domain.models import BranchRole
from atrex_runtime.gateway.proxy import _supported_gateway_operations
from atrex_runtime.workers.evidence_view import (
    EVIDENCE_PROMPT_SHA256,
    EVIDENCE_PROMPT_TEXT,
    EVOLVER_EVIDENCE_PROMPT_TEXT,
    EvidenceViewManifestV1,
    _latest_evolver_epoch_facts,
    _materialize_evolver_agent_reports,
    _materialize_evolver_agent_sessions,
    _materialize_evolver_journal,
    assemble_evolver_evidence_view,
    assemble_optimizer_evidence_view,
    evolver_agent_optimization_summary,
)
from atrex_runtime.workers.evolver_review import materialize_evolver_review


def test_optimizer_prompt_enforces_evolver_owned_agent_content() -> None:
    assert "`prompts/`, `insights/`, and `skills/` belong" in EVIDENCE_PROMPT_TEXT
    assert "Only `tools/` is adaptive here" in EVIDENCE_PROMPT_TEXT
    assert "Direction and\nExperiment Journal" in EVIDENCE_PROMPT_TEXT
    assert "Evolver curates" in EVIDENCE_PROMPT_TEXT


def _evolver_service_catalog() -> str:
    _, catalog = EVOLVER_EVIDENCE_PROMPT_TEXT.split(
        "## Runtime services for the next Optimizer\n", 1
    )
    return catalog.split("Runtime injects this frozen view.", 1)[0]


@pytest.mark.parametrize("repository", ["kernel-design-agents", "atrex-kernel-agent-core"])
def test_evolver_service_catalog_matches_standard_public_cli(repository: str) -> None:
    tool_source = Path(__file__).resolve().parents[1] / "src" / repository / "src/runtime_tools.py"
    definitions = {
        node.targets[0].id: node.value
        for node in ast.parse(tool_source.read_text(encoding="utf-8")).body
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
    }
    queries = ast.literal_eval(definitions["_PUBLIC_RUNTIME_QUERY_COMMANDS"])
    journals = ast.literal_eval(definitions["_RUNTIME_JOURNAL_COMMANDS"])
    expected = {
        "gateway-execute",
        "attempt-report",
        *queries,
        *(name for name in journals if not name.startswith("_")),
    }
    columns = re.findall(r"^\| (`[^|]+) \|", _evolver_service_catalog(), re.MULTILINE)
    documented = {name for column in columns for name in re.findall(r"`([a-z-]+)`", column)}

    assert documented == expected


def test_evolver_service_catalog_matches_live_gateway_and_preserves_boundaries() -> None:
    catalog = _evolver_service_catalog()
    documented = set(re.findall(r"^- `([a-z]+)`:", catalog, re.MULTILINE))

    assert documented == set(_supported_gateway_operations("gateway"))
    assert 'comparison.method="abba"' in catalog
    assert "ABBA is an Evaluate option, not a separate operation" in catalog
    assert "It does not authorize this Evolver to call them" in " ".join(catalog.split())
    assert "Session-context `evolution_report.tool`" in catalog
    assert "not a new Runtime endpoint" in catalog
    assert "separating measured facts from new analysis" in catalog
    assert "read the relevant files through Runtime tools" not in EVOLVER_EVIDENCE_PROMPT_TEXT


def test_evolver_prior_report_guidance_limits_audit_to_observed_agent_changes() -> None:
    guidance = EVOLVER_EVIDENCE_PROMPT_TEXT.split("## Prior Evolutions\n", 1)[1].split(
        "## Agent Bundles and reusable resources\n", 1
    )[0]
    normalized = " ".join(guidance.split())

    assert "match `generated_agent.path`" in normalized
    assert "Session-context relationships" in normalized
    assert "may now be the Active" in normalized
    assert "`current_epoch_challenger` is unevaluated" in normalized
    assert "Missing observations cannot show whether a Tool was unused or ineffective" in normalized
    assert "terminal resource file alone does not establish the exact code executed" in normalized


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _review_session(path: Path, calls: list[tuple[str, str, bool, str]]) -> None:
    records: list[dict[str, object]] = []
    for index, (name, command, failed, result) in enumerate(calls, start=1):
        identifier = f"tool-{index}"
        records.extend(
            [
                {
                    "event": {
                        "message": {
                            "content": [
                                {
                                    "type": "tool_use",
                                    "id": identifier,
                                    "name": name,
                                    "input": {"command": command},
                                }
                            ]
                        }
                    }
                },
                {
                    "event": {
                        "message": {
                            "content": [
                                {
                                    "type": "tool_result",
                                    "tool_use_id": identifier,
                                    "is_error": failed,
                                    "content": result,
                                }
                            ]
                        }
                    }
                },
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(item) + "\n" for item in records), encoding="utf-8")


def test_evolver_review_indexes_observed_change_effects_trajectories_and_friction(
    tmp_path: Path,
) -> None:
    evidence = tmp_path / "evidence"
    shared_direction = "direction_" + "a" * 32
    _write(
        evidence / "latest-epoch-facts.json",
        {
            "epoch_number": 2,
            "selection_reason": "authoritative_comparison",
            "attempts": [
                {
                    "attempt_id": "attempt_active",
                    "kernel_agent_revision_id": "agent_active",
                    "branch": "active",
                    "challenger_ordinal": 0,
                    "trajectory_ordinal": 1,
                    "attempt_ordinal": 1,
                    "status": "completed",
                    "accepted_as_branch_best": True,
                    "failure_reason": None,
                    "candidate": {"correct": True, "latency_us": 12.0},
                    "direction_ids": [shared_direction],
                    "experiment_ids": ["experiment_" + "b" * 32],
                },
                {
                    "attempt_id": "attempt_challenger",
                    "kernel_agent_revision_id": "agent_challenger",
                    "branch": "challenger",
                    "challenger_ordinal": 1,
                    "trajectory_ordinal": 2,
                    "attempt_ordinal": 1,
                    "status": "failed",
                    "accepted_as_branch_best": False,
                    "failure_reason": "process-exit-1",
                    "candidate": None,
                    "direction_ids": [shared_direction],
                    "experiment_ids": [],
                },
            ],
        },
    )
    common_probe = "print('probe')"
    _review_session(
        evidence / "agent-v1/sessions/trajectory-00000002/attempt-00000001.conversation.jsonl",
        [
            ("Bash", "python3 tools/helper.py --mode screen", False, "ok"),
            (
                "Bash",
                "cat > scratch/probe-one.py <<'PY'\n" + common_probe + "\nPY",
                False,
                "",
            ),
            (
                "Bash",
                "python3 agent/optimizer/src/runtime_tools.py gateway-execute "
                "--request scratch/request-one.json",
                True,
                "validation failed",
            ),
            (
                "Bash",
                "python3 agent/optimizer/src/runtime_tools.py gateway-execute "
                "--request scratch/request-two.json",
                False,
                "completed",
            ),
        ],
    )
    _review_session(
        evidence / "agent-v1/sessions/trajectory-00000003/attempt-00000001.conversation.jsonl",
        [
            (
                "Bash",
                "cat > scratch/probe-two.py <<'PY'\n" + common_probe + "\nPY",
                False,
                "",
            )
        ],
    )
    _write(
        evidence / "agent-v1/reports/trajectory-00000002/attempt-00000001.report.json",
        {"analysis": "tools/helper.py supplied the screening result"},
    )
    reports = tmp_path / "evolution-reports"
    _write(
        reports / "evo-1.json",
        {
            "evolution_number": 1,
            "generated_agent": {"path": "input/agents/agent-v1"},
            "report": {
                "hypothesis": "A helper avoids repeated manual screening.",
                "expected_effect": "The Optimizer invokes the helper.",
                "changed_paths": [
                    "tools/helper.py",
                    "skills/screening/SKILL.md",
                    "prompts/episode.md",
                ],
            },
        },
    )

    materialize_evolver_review(
        evidence,
        evolution_reports_root=reports,
        agent_versions={"agent-v0": "agent_active", "agent-v1": "agent_challenger"},
        pool_versions=frozenset({"agent-v0", "agent-v1"}),
    )

    audit = json.loads((evidence / "review/evolution-change-audit.json").read_text())
    observations = {item["path"]: item for item in audit["evaluated_changes"][0]["observations"]}
    assert observations["tools/helper.py"]["status"] == "report_cited"
    assert observations["tools/helper.py"]["successful_in_sessions"]
    assert observations["skills/screening/SKILL.md"]["status"] == "not_observed"
    assert observations["prompts/episode.md"]["status"] == "not_observed"

    comparison = json.loads((evidence / "review/trajectory-comparison.json").read_text())
    assert len(comparison["trajectories"]) == 2
    assert comparison["exact_cross_trajectory_overlaps"]["direction_ids"] == [
        {
            "id": shared_direction,
            "trajectories": [
                "agent-v0:active:0:trajectory-00000001",
                "agent-v1:challenger:1:trajectory-00000002",
            ],
        }
    ]

    friction = json.loads((evidence / "review/workflow-friction.json").read_text())
    assert friction["runtime_operations"] == [
        {
            "operation": "gateway-execute",
            "call_count": 2,
            "failed_call_count": 1,
            "failed_then_later_succeeded_session_count": 1,
        }
    ]
    assert len(friction["tool_failures"]) == 1
    assert friction["repeated_construction_candidates"][0]["kind"] == ("probe_or_helper_script")
    assert friction["repeated_construction_candidates"][0]["occurrence_count"] == 2


def _raw_trace(
    store: LocalArtifactStore,
    root: Path,
    label: str,
) -> ArtifactDigest:
    source = root / f"raw-{label}"
    (source / "input").mkdir(parents=True)
    (source / "provider/claude-subagents").mkdir(parents=True)
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
    child_event = {
        "type": "assistant",
        "message": {
            "id": f"child-{label}",
            "content": [{"type": "text", "text": f"subagent result {label}"}],
        },
    }
    (source / "provider/claude-subagents/agent-child.jsonl").write_text(
        json.dumps(child_event) + "\n",
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
        + "\n"
        + json.dumps(
            {
                "schema_version": 1,
                "sequence": 3,
                "type": "provider_event",
                "source": "provider",
                "path": "provider/claude-subagents/agent-child.jsonl",
                "event": child_event,
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
    assert not (destination / "direction-proposals.json").exists()

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
    assert "`result-artifact-read`" not in EVIDENCE_PROMPT_TEXT
    assert "`list-directions`" in EVIDENCE_PROMPT_TEXT
    assert "`load-direction`" in EVIDENCE_PROMPT_TEXT
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
    assert "subagent result current-1" in completed_trace.read_text()
    assert "provider/claude-subagents/agent-child.jsonl" in completed_trace.read_text()
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
        "Each of the four reusable directories has a mandatory `README.md` index"
        in EVOLVER_EVIDENCE_PROMPT_TEXT
    )
    assert "`candidate/` starts as a writable" in EVOLVER_EVIDENCE_PROMPT_TEXT
    assert "{prompts,insights,skills,tools}/" in EVOLVER_EVIDENCE_PROMPT_TEXT
    assert "candidate/runtime-state/" not in EVOLVER_EVIDENCE_PROMPT_TEXT
    assert "revision seed" in EVOLVER_EVIDENCE_PROMPT_TEXT
    assert "Tool-to-Skill promotion as evidence-driven curation" in EVOLVER_EVIDENCE_PROMPT_TEXT
    assert "`skills/<skill-name>/SKILL.md`" in EVOLVER_EVIDENCE_PROMPT_TEXT
    assert "do not turn every one-off probe" in EVOLVER_EVIDENCE_PROMPT_TEXT
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
    assert {path.name for path in (destination / "review").iterdir()} == {
        "evolution-change-audit.json",
        "trajectory-comparison.json",
        "workflow-friction.json",
    }
    assert json.loads(
        (destination / "review/evolution-change-audit.json").read_text()
    )["status"] == "no_evolution_reports"
    facts = json.loads((destination / "latest-epoch-facts.json").read_text())
    assert facts["epoch_number"] == 1
    assert facts["selection_reason"] == "authoritative_comparison"
    assert [item["branch"] for item in facts["attempts"]] == ["active", "challenger"]
    assert facts["attempts"][0]["direction_ids"] == ["direction_" + "a" * 32]
    assert facts["attempts"][0]["failure_reason"] is None
    assert facts["attempts"][0]["candidate"]["correct"] is True
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


def test_latest_evolver_facts_preserve_runtime_failure_diagnosis(tmp_path: Path) -> None:
    lineage = tmp_path / "lineage"
    _write(
        lineage / "epochs/00000001.json",
        {
            "selection_reason": "incumbent_retained",
            "winner_kernel_agent_revision_id": "agent_active",
            "attempts": [
                {
                    "attempt_id": "attempt_failed",
                    "branch": "challenger",
                    "status": "failed",
                    "attempt_report_status": "candidate_ready",
                    "failure_reason": (
                        "Optimizer session did not complete successfully: process-exit-126"
                    ),
                    "output": None,
                }
            ],
        },
    )
    _write(
        lineage / "reports/00000001/attempt_failed.json",
        {
            "direction_events": [{"direction_id": "direction_one"}],
            "experiments": [{"experiment_id": "experiment_one"}],
        },
    )

    facts = _latest_evolver_epoch_facts(lineage, 1, LocalArtifactStore(tmp_path / "artifacts"))

    attempts = facts["attempts"]
    assert isinstance(attempts, list)
    attempt = attempts[0]
    assert isinstance(attempt, dict)
    assert attempt["status"] == "failed"
    assert attempt["attempt_report_status"] == "candidate_ready"
    failure_reason = attempt["failure_reason"]
    assert isinstance(failure_reason, str)
    assert failure_reason.endswith("process-exit-126")
    assert attempt["candidate"] is None
    assert attempt["direction_ids"] == ["direction_one"]
    assert attempt["experiment_ids"] == ["experiment_one"]


def test_evolver_reads_live_journal_from_attempt_without_terminal_report(tmp_path: Path) -> None:
    lineage = tmp_path / "lineage"
    attempt_id = "attempt_" + "a" * 32
    direction_id = "direction_" + "b" * 32
    experiment_id = "experiment_" + "c" * 32
    _write(
        lineage / "epochs/00000001.json",
        {
            "selection_reason": "latency",
            "winner_kernel_agent_revision_id": "agentrev_" + "d" * 32,
            "attempts": [{"attempt_id": attempt_id, "status": "infrastructure_failed"}],
        },
    )
    _write(
        lineage / f"journals/00000001/{attempt_id}.json",
        {
            "attempt_id": attempt_id,
            "direction_events": [
                {
                    "direction_id": direction_id,
                    "direction_event_id": "directionevent_" + "e" * 32,
                    "recorded_at": "2026-09-14T00:00:00+00:00",
                    "action": "propose",
                    "name": "Reorder loads",
                    "hypothesis": "Coalescing reduces transactions",
                }
            ],
            "experiments": [
                {
                    "experiment_id": experiment_id,
                    "direction_id": direction_id,
                    "name": "First probe",
                }
            ],
        },
    )

    facts = _latest_evolver_epoch_facts(lineage, 1, LocalArtifactStore(tmp_path / "artifacts"))
    assert facts["attempts"][0]["direction_ids"] == [direction_id]
    assert facts["attempts"][0]["experiment_ids"] == [experiment_id]
    destination = tmp_path / "evolver-journal"
    _materialize_evolver_journal(destination, lineage, 1)
    direction_index = json.loads((destination / "directions/index.json").read_text())
    direction = json.loads((destination / f"directions/{direction_id}.json").read_text())
    experiment = json.loads((destination / f"experiments/{experiment_id}.json").read_text())
    assert direction_index[0]["name"] == "Reorder loads"
    assert direction["events"][0]["hypothesis"] == "Coalescing reduces transactions"
    assert experiment["name"] == "First probe"


def test_evolver_journal_includes_bootstrap_direction_records(tmp_path: Path) -> None:
    lineage = tmp_path / "lineage"
    direction_id = "direction_" + "a" * 32
    experiment_id = "experiment_" + "b" * 32
    _write(
        lineage / "bootstrap/report.json",
        {
            "attempt_id": "attempt_" + "c" * 32,
            "direction_events": [
                {
                    "direction_id": direction_id,
                    "direction_event_id": "directionevent_" + "d" * 32,
                    "recorded_at": "2026-09-14T00:00:00+00:00",
                    "action": "propose",
                    "name": "Baseline construction",
                }
            ],
            "experiments": [{"experiment_id": experiment_id, "name": "Baseline test"}],
        },
    )

    destination = tmp_path / "evolver-journal"
    _materialize_evolver_journal(destination, lineage, 0)

    assert (
        json.loads((destination / "directions/index.json").read_text())[0]["direction_id"]
        == direction_id
    )
    assert (
        json.loads((destination / "experiments/index.json").read_text())[0]["experiment_id"]
        == experiment_id
    )


def test_evolver_journal_includes_prior_suggested_directions(tmp_path: Path) -> None:
    lineage = tmp_path / "lineage"
    direction_id = "direction_" + "a" * 32
    _write(
        lineage / "epochs/00000001.json",
        {
            "suggested_directions": [
                {
                    "direction_id": direction_id,
                    "status": "suggested",
                    "name": "Try a split reduction",
                    "hypothesis": "A split reduction could shorten the critical path",
                    "created_at": "2026-09-14T00:00:00+00:00",
                }
            ]
        },
    )
    destination = tmp_path / "evolver-journal"
    _materialize_evolver_journal(destination, lineage, 1)
    index = json.loads((destination / "directions/index.json").read_text())
    record = json.loads((destination / f"directions/{direction_id}.json").read_text())
    assert index[0]["direction_id"] == direction_id
    assert index[0]["latest_action"] == "suggest"
    assert record["events"][0]["status"] == "suggested"
