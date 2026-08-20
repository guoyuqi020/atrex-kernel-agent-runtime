"""Core lineage-baseline workspace and process protocol tests."""

from __future__ import annotations

import json
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
    LineageBootstrapManifestV1,
    LineageBootstrapReportV1,
    LineageBootstrapSessionConfig,
    LineageBootstrapWorkspaceAssembler,
)


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
Path(os.environ["ATREX_LINEAGE_BOOTSTRAP_REPORT_PATH"]).write_text(json.dumps({
    "schema_version": 1,
    "bootstrap_attempt_id": manifest["bootstrap_attempt_id"],
    "status": "baseline_ready",
    "approach": "plain DSL baseline",
    "change_summary": "implemented the operator",
    "correctness_evidence": "full evaluation passed",
    "latency_us": 12.5,
    "candidate_artifact_digest": "sha256:" + "a" * 64,
    "gateway_result_digest": "sha256:" + "b" * 64,
    "research_sources": ["wiki:snapshot"],
    "lessons": "simple baseline is correct",
    "next_directions": ["profile memory traffic"],
    "blocker": None
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
    manifest = LineageBootstrapManifestV1(
        bootstrap_attempt_id="attempt_" + "1" * 32,
        kernel_agent_revision_id="agentrev_" + "2" * 32,
        input_kernel_digest=kernel_digest,
        optimizer_digest=agent_digest,
        evaluation_contract_digest=contract_digest,
        agent_problem_digest=problem_digest,
        dsl=Dsl.TRITON,
        operator="vector_add",
        hardware_target="nvidia-h100",
    )
    prepared = LineageBootstrapWorkspaceAssembler(tmp_path / "workspaces", artifacts).prepare(
        manifest
    )
    assert (prepared.root / "input/kernel/kernel.py").is_file()
    assert (prepared.root / "input/agent-problem/value.json").is_file()
    assert not (prepared.root / "input/evaluation-contract").exists()
    assert (prepared.root / "work/kernel/kernel.py").is_file()

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
    assert result.report.status == "baseline_ready"
    assert result.report_digest is not None
    assert result.session_trace_digest is not None
    assert artifacts.verify(result.report_digest).kind is ArtifactKind.ATTEMPT_REPORT
    assert artifacts.verify(result.session_trace_digest).kind is ArtifactKind.SESSION_LOG


def test_lineage_bootstrap_report_rejects_incomplete_ready_result() -> None:
    value = {
        "schema_version": 1,
        "bootstrap_attempt_id": "attempt_" + "1" * 32,
        "status": "baseline_ready",
        "approach": "approach",
        "change_summary": "change",
        "correctness_evidence": "evidence",
        "latency_us": None,
        "candidate_artifact_digest": None,
        "gateway_result_digest": None,
        "research_sources": [],
        "lessons": "lesson",
        "next_directions": [],
        "blocker": None,
    }
    with pytest.raises(ValidationError, match="complete result"):
        LineageBootstrapReportV1.model_validate(value)
