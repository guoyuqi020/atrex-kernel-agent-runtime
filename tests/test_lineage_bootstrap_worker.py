"""Core lineage-baseline workspace and process protocol tests."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from atrex_runtime.artifacts.local import ArtifactKind, LocalArtifactStore
from atrex_runtime.domain.models import Dsl, TokenUsage
from atrex_runtime.workers import (
    CleanEnvironmentLauncher,
    CoreLineageBootstrapSessionDriver,
    CoreOptimizerProcessConfig,
    LineageBootstrapManifestV2,
    LineageBootstrapSessionConfig,
    LineageBootstrapWorkspaceAssembler,
)
from atrex_runtime.workers.attempt_report import AttemptReportV12


def _write_agent(root: Path) -> None:
    (root / "src").mkdir(parents=True)
    (root / "atrex-bundle.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "bundle_format": "atrex-kernel-agent-bundle-v1",
                "entrypoint": {"command": "src/run.py"},
            }
        ),
        encoding="utf-8",
    )
    (root / "src/run.py").write_text(
        """import json
import os
from pathlib import Path

manifest = json.loads(Path(os.environ["ATREX_LINEAGE_BOOTSTRAP_MANIFEST"]).read_text())
Path("work/launch.json").write_text(json.dumps({
    "phase": os.environ["ATREX_CORE_PHASE"],
    "manifest": os.environ["ATREX_LINEAGE_BOOTSTRAP_MANIFEST"],
    "gateway": os.environ["ATREX_GATEWAY_PROXY_URL"],
    "repository": os.environ["ATREX_OPTIMIZER_REPOSITORY"],
}))
trace = Path("sessions/core")
trace.mkdir()
(trace / "events.jsonl").write_text('{"event":"turn-end"}\\n')
budget = float(os.environ["ATREX_USAGE_BUDGET"])
Path(os.environ["ATREX_TOKEN_USAGE_REPORT"]).write_text(json.dumps({
    "schema_version": 2,
    "usage_unit": os.environ["ATREX_USAGE_UNIT"],
    "budget": budget,
    "consumed": 18,
    "token_usage": {
        "uncached_input_tokens": 10,
        "output_tokens": 5,
        "cache_read_tokens": 2,
        "cache_write_tokens": 1
    },
    "credits": None,
    "budget_exhausted": False,
    "session_count": 1,
    "model_request_count": 1,
    "usage_complete": True
}))
experiment_id = "experiment_" + "4" * 32
direction_id = "direction_" + "5" * 32
subject = {
    "kernel_artifact_digest": "sha256:" + "a" * 64,
    "kernel_trial_id": "gtrial_" + "c" * 32,
    "result_artifact_digests": ["sha256:" + "b" * 64],
}
Path(os.environ["ATREX_ATTEMPT_REPORT_PATH"]).write_text(json.dumps({
    "schema_version": 12,
    "attempt_id": manifest["bootstrap_attempt_id"],
    "status": "candidate_ready",
    "hypothesis": "a direct implementation establishes the baseline",
    "diagnosis": {"bottleneck": "framework bring-up", "evidence": "seed inspection"},
    "approach": {
        "summary": "plain DSL baseline",
        "steps": ["implement the operator"],
        "expected_impact": "full correctness",
        "risks": [],
    },
    "final_candidate": {"change_summary": "implemented the operator"},
    "evidence_summary": {
        "correctness": "full evaluation passed",
        "performance": "positive measured latency",
    },
    "profile_evidence": None,
    "analysis": "the candidate is a valid first baseline",
    "knowledge_used": [],
    "findings": [{
        "category": "correctness",
        "observation": "the full evaluation passed",
        "root_cause": "the direct mapping preserves semantics",
        "resolution": "retain the implementation",
        "lesson": "simple baseline is correct",
        "supporting_experiment_ids": [experiment_id],
    }],
    "blocker": None
    ,"experiments": [{
        "experiment_id": experiment_id,
        "direction_id": direction_id,
        "sequence": 1,
        "recorded_at": "2026-08-25T00:00:00+00:00",
        "name": "establish baseline",
        "hypothesis": "the direct implementation is correct",
        "change": "implemented the operator",
        "before": None,
        "after": subject,
        "evidence": "full evaluation passed",
        "analysis": "the hypothesis held",
        "action": "baseline",
    }],
    "direction_events": [
        {
            "direction_event_id": "directionevent_" + "6" * 32,
            "direction_id": direction_id,
            "recorded_at": "2026-08-25T00:00:00+00:00",
            "action": "propose",
            "name": "establish a correct baseline",
            "hypothesis": "a direct implementation is sufficient",
            "rationale": "bootstrap prioritizes correctness",
            "plan": ["implement and evaluate"],
            "success_criteria": "full correctness passes",
            "stop_conditions": "the DSL cannot express the operator",
            "analysis": None,
            "supporting_experiment_ids": [],
        },
        {
            "direction_event_id": "directionevent_" + "7" * 32,
            "direction_id": direction_id,
            "recorded_at": "2026-08-25T00:01:00+00:00",
            "action": "complete",
            "name": None,
            "hypothesis": None,
            "rationale": None,
            "plan": [],
            "success_criteria": None,
            "stop_conditions": None,
            "analysis": "the full evaluation passed",
            "supporting_experiment_ids": [experiment_id],
        },
    ],
}))
print("framework baseline finished")
""",
        encoding="utf-8",
    )


def test_lineage_bootstrap_workspace_and_driver(tmp_path: Path) -> None:
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    kernel = tmp_path / "kernel"
    kernel.mkdir()
    (kernel / "kernel.py").write_text("def run(): return None\n", encoding="utf-8")
    agent = tmp_path / "agent"
    agent.mkdir()
    _write_agent(agent)
    for name in ("prompts", "memory", "knowledge", "skills", "tools", "hooks"):
        (agent / name).mkdir()
        (agent / name / "README.md").write_text(f"Initial {name} index")
        (agent / name / "seed.md").write_text(f"Initial {name}")
    kernel_digest = artifacts.put_directory(kernel, ArtifactKind.KERNEL)
    agent_digest = artifacts.put_directory(agent, ArtifactKind.KERNEL_AGENT)
    contract_digest = artifacts.put_json(
        {"schema_version": 1, "candidate_path": "kernel.py"},
        ArtifactKind.EVALUATION_CONTRACT,
    )
    problem_digest = artifacts.put_json(
        {"schema_version": "atrex.agent_problem.v1", "objective": "vector add"},
        ArtifactKind.AGENT_PROBLEM,
    )
    manifest = LineageBootstrapManifestV2(
        bootstrap_attempt_id="attempt_" + "1" * 32,
        lineage_id="lineage_" + "2" * 32,
        kernel_agent_revision_id="agentrev_" + "2" * 32,
        input_kernel_digest=kernel_digest,
        optimizer_digest=agent_digest,
        evaluation_contract_digest=contract_digest,
        agent_problem_digest=problem_digest,
        dsl=Dsl.TRITON,
        operator="vector_add",
        hardware_target="nvidia-h100",
    )
    attempt_workspaces = tmp_path / "attempt-workspaces"
    assembler = LineageBootstrapWorkspaceAssembler(
        tmp_path / "workspaces",
        artifacts,
        attempt_workspaces_root=attempt_workspaces,
    )
    prepared = assembler.prepare(manifest)
    assert (prepared.root / "input/kernel/kernel.py").is_file()
    assert (prepared.root / ".runtime/agent-problem.json").is_file()
    assert prepared.manifest_path == prepared.root / ".runtime/lineage-bootstrap.json"
    assert not (prepared.root / "input/agent-problem").exists()
    assert not (prepared.root / "input/evaluation-contract").exists()
    assert (prepared.root / "work/kernel/kernel.py").is_file()
    reference = prepared.root / "reference"
    assert reference.is_dir()
    assert list(reference.iterdir()) == []
    assert not (os.stat(reference).st_mode & 0o200)
    (prepared.root / "skills/baseline.md").write_text("reuse this lesson\n")
    (prepared.root / "tools/probe.py").write_text("print('probe')\n")
    for name in ("prompts", "memory", "knowledge", "skills", "tools", "hooks"):
        assert (prepared.root / name / "README.md").read_text() == f"Initial {name} index"
        assert not (prepared.root / "agent/optimizer" / name).exists()
        assert (prepared.root / name / "seed.md").read_text() == f"Initial {name}"
        (prepared.root / name / "entry.txt").write_text(f"bootstrap {name}")
        (prepared.root / name / "README.md").write_text(f"{name}: entry.txt")

    driver = CoreLineageBootstrapSessionDriver(
        CleanEnvironmentLauncher(Path("/usr/bin/env")),
        CoreOptimizerProcessConfig(
            agent_backend="claude",
            command_prefix=(sys.executable,),
            isolated_home_environment_keys=("HOME",),
            session_trace_relative_path="sessions/core",
            token_usage_report_relative_path="scratch/token-usage.json",
            max_attempt_report_bytes=65_536,
            timeout_seconds=10,
            terminate_grace_seconds=1,
            max_diagnostic_bytes=4096,
            max_session_tokens=1000,
        ),
        artifacts,
    )
    result = driver.run(
        prepared,
        LineageBootstrapSessionConfig(
            environment=(),
            gateway_endpoint="http://gateway-proxy",
            gateway_capability="bootstrap-capability",
        ),
    )

    launch = json.loads((prepared.root / "work/launch.json").read_text())
    assert launch["phase"] == "framework_baseline"
    assert launch["manifest"] == str(prepared.manifest_path)
    assert result.finish_reason == "completed"
    assert result.final_response == "framework baseline finished\n"
    assert result.token_usage == TokenUsage(10, 5, 2, 1)
    assert result.report is not None
    assert result.report.status == "candidate_ready"
    assert result.report_digest is not None
    assert result.session_trace_digest is not None
    assert artifacts.verify(result.report_digest).kind is ArtifactKind.ATTEMPT_REPORT
    assert artifacts.verify(result.session_trace_digest).kind is ArtifactKind.SESSION_LOG
    resumed = assembler.prepare(manifest)
    assert (resumed.root / "skills/baseline.md").read_text() == "reuse this lesson\n"
    assert (resumed.root / "tools/probe.py").read_text() == "print('probe')\n"
    for name in ("prompts", "memory", "knowledge", "skills", "tools", "hooks"):
        assert (resumed.root / name / "entry.txt").read_text() == f"bootstrap {name}"
        assert (resumed.root / name / "README.md").read_text() == f"{name}: entry.txt"


def test_lineage_bootstrap_rejects_the_legacy_terminal_report() -> None:
    value = {
        "schema_version": 3,
        "bootstrap_attempt_id": "attempt_" + "1" * 32,
        "status": "baseline_ready",
        "approach": "approach",
        "change_summary": "change",
        "correctness_evidence": "evidence",
        "latency_us": None,
        "kernel_artifact_digest": None,
        "kernel_trial_id": None,
        "gateway_result_digest": None,
        "research_sources": [],
        "lessons": "lesson",
        "next_directions": [],
        "blocker": None,
    }
    with pytest.raises(ValidationError):
        AttemptReportV12.model_validate(value)
