"""Framework-neutral Core repository Optimizer process adapter tests."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import pytest
from conftest import digest

from atrex_runtime.artifacts.local import ArtifactKind, LocalArtifactStore
from atrex_runtime.domain.ids import (
    new_attempt_id,
    new_campaign_id,
    new_epoch_id,
    new_kernel_agent_revision_id,
    new_kernel_revision_id,
    new_lineage_id,
)
from atrex_runtime.domain.models import Dsl, TokenUsage
from atrex_runtime.workers.core import (
    CoreOptimizerProcessConfig,
    CoreOptimizerSessionDriver,
)
from atrex_runtime.workers.core_phase import CorePhaseRunner
from atrex_runtime.workers.launcher import CleanEnvironmentLauncher
from atrex_runtime.workers.manifest import AttemptInputManifestV6, AttemptTaskContextV5
from atrex_runtime.workers.optimizer import OptimizerSessionConfig
from atrex_runtime.workers.workspace import PreparedAttempt


def _attempt_manifest() -> AttemptInputManifestV6:
    return AttemptInputManifestV6(
        attempt_id=new_attempt_id(),
        kernel_agent_revision_id=new_kernel_agent_revision_id(),
        input_kernel_revision_id=new_kernel_revision_id(),
        input_kernel_digest=digest("kernel"),
        epoch_evidence_checkpoint=digest("evidence"),
        attempt_evidence_digest=digest("attempt-evidence"),
        optimizer_digest=digest("optimizer"),
        dsl=Dsl.TRITON,
        context=AttemptTaskContextV5(
            campaign_id=new_campaign_id(),
            lineage_id=new_lineage_id(),
            epoch_id=new_epoch_id(),
            epoch_number=1,
            attempt_ordinal=1,
            operator="vector_add",
            hardware_target="h100",
            evaluation_contract_digest=digest("contract"),
            agent_problem_digest=digest("problem"),
        ),
    )


def test_runtime_does_not_seal_an_unfinished_live_core_trace(tmp_path: Path) -> None:
    root = tmp_path / "phase"
    repository = root / "agent/optimizer"
    repository.mkdir(parents=True)
    (root / "sessions").mkdir()
    (repository / "atrex-bundle.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "bundle_format": "atrex-kernel-agent-bundle-v1",
                "entrypoint": {"command": "run.py"},
            }
        ),
        encoding="utf-8",
    )
    (repository / "run.py").write_text(
        """import json
import os
from pathlib import Path

trace = Path(os.environ["ATREX_SESSION_TRACE_PATH"])
trace.mkdir()
(trace / ".runtime-live-session").write_text("unsealed\\n")
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
""",
        encoding="utf-8",
    )
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    policy = CoreOptimizerProcessConfig(
        agent_backend="claude",
        command_prefix=(sys.executable,),
        isolated_home_environment_keys=(),
        session_trace_relative_path="sessions/core",
        token_usage_report_relative_path="scratch/token-usage.json",
        max_attempt_report_bytes=65_536,
        timeout_seconds=10,
        terminate_grace_seconds=1,
        max_diagnostic_bytes=4096,
        max_session_tokens=1000,
    )
    runner = CorePhaseRunner(CleanEnvironmentLauncher(Path("/usr/bin/env")), policy, artifacts)
    prepared = runner.prepare(root, root / "sessions")

    environment = runner.runtime_environment(
        prepared,
        phase="optimization_attempt",
        model="optimizer-model",
    )
    assert environment["ATREX_AGENT_MODEL"] == "optimizer-model"
    result = runner.run(
        prepared,
        environment,
        label="test Core",
    )

    assert result.finish_reason == "completed"
    assert result.session_trace_digest is None
    assert (root / "sessions/core/.runtime-live-session").is_file()


@pytest.mark.anyio
async def test_repository_driver_executes_core_declared_framework_neutral_entrypoint(
    tmp_path: Path,
) -> None:
    root = tmp_path / "attempt"
    repository = root / "agent/optimizer"
    repository.mkdir(parents=True)
    (root / "work").mkdir()
    (root / "scratch").mkdir()
    session_root = root / "sessions"
    session_root.mkdir()
    manifest = _attempt_manifest()
    manifest_path = root / "attempt.json"
    manifest_path.write_bytes(manifest.canonical_json_bytes())
    (repository / "atrex-bundle.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "bundle_format": "atrex-kernel-agent-bundle-v1",
                "entrypoint": {"command": "run.py"},
            }
        ),
        encoding="utf-8",
    )
    (repository / "run.py").write_text(
        """import json
import os
from pathlib import Path

attempt = json.loads(Path(os.environ["ATREX_ATTEMPT_MANIFEST"]).read_text())
Path("work/launch.json").write_text(json.dumps({
    "phase": os.environ["ATREX_CORE_PHASE"],
    "attempt": os.environ["ATREX_ATTEMPT_MANIFEST"],
    "gateway": os.environ["ATREX_GATEWAY_PROXY_URL"],
    "capability": os.environ["ATREX_GATEWAY_CAPABILITY"],
    "repository": os.environ["ATREX_OPTIMIZER_REPOSITORY"],
    "session_timeout": os.environ["ATREX_SESSION_TIMEOUT_SECONDS"],
    "session_trace": os.environ["ATREX_SESSION_TRACE_PATH"],
    "home": os.environ["HOME"],
    "codex_home": os.environ["CODEX_HOME"],
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
Path(os.environ["ATREX_ATTEMPT_REPORT_PATH"]).write_text(json.dumps({
    "schema_version": 3,
    "attempt_id": attempt["attempt_id"],
    "status": "blocked",
    "hypothesis": "test hypothesis",
    "bottleneck": "test bottleneck",
    "plan": ["test plan"],
    "change_summary": "no candidate change",
    "profile_evidence": "test profile evidence",
    "evaluation_evidence": "test evaluation evidence",
    "result_interpretation": "test result",
    "decision": "blocked",
    "research_sources": [],
    "lessons": ["test lesson"],
    "next_directions": ["test next direction"],
    "experiments": [{
        "sequence": 1,
        "recorded_at": "2026-08-17T00:00:00Z",
        "name": "test experiment",
        "hypothesis": "test hypothesis",
        "change": "none",
        "candidate_artifact_digest": None,
        "evidence": "test evidence",
        "result": "blocked",
        "decision": "pivot"
    }]
}))
print("core-owned optimizer finished")
""",
        encoding="utf-8",
    )
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    driver = CoreOptimizerSessionDriver(
        CleanEnvironmentLauncher(Path("/usr/bin/env")),
        CoreOptimizerProcessConfig(
            agent_backend="claude",
            command_prefix=(sys.executable,),
            isolated_home_environment_keys=("HOME", "CODEX_HOME"),
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

    result = await driver.run(
        PreparedAttempt(root, manifest_path, session_root, "session-id"),
        OptimizerSessionConfig(
            environment=(),
            gateway_endpoint="http://gateway-proxy",
            gateway_capability="attempt-capability",
        ),
    )

    launch = json.loads((root / "work/launch.json").read_text(encoding="utf-8"))
    assert result.finish_reason == "completed"
    assert result.final_response == "core-owned optimizer finished\n"
    assert result.token_usage == TokenUsage(10, 5, 2, 1)
    assert result.attempt_report is not None
    assert result.attempt_report.status == "blocked"
    assert result.attempt_report_digest is not None
    assert result.session_trace_digest is not None
    assert artifacts.verify(result.session_trace_digest).kind is ArtifactKind.SESSION_LOG
    assert launch == {
        "phase": "optimization_attempt",
        "attempt": str(manifest_path),
        "gateway": "http://gateway-proxy",
        "capability": "attempt-capability",
        "repository": str(repository),
        "session_timeout": "10",
        "session_trace": str(root / "sessions/core"),
        "home": str(session_root / "agent-home"),
        "codex_home": str(session_root / "agent-home"),
    }


@pytest.mark.anyio
async def test_runtime_executes_current_core_bundle_with_attempt_v6(tmp_path: Path) -> None:
    root = tmp_path / "attempt"
    repository = root / "agent/optimizer"
    source = Path(__file__).resolve().parents[1] / "src/atrex-kernel-agent-core"
    shutil.copytree(
        source,
        repository,
        ignore=shutil.ignore_patterns(
            ".git", ".mypy_cache", ".pytest_cache", ".ruff_cache", "__pycache__", "*.pyc"
        ),
    )
    for relative in ("input/kernel", "input/agent-problem", "work/kernel", "scratch", "sessions"):
        (root / relative).mkdir(parents=True, exist_ok=True)
    evidence = root / "input/evidence"
    (evidence / "epochs/00000001/attempts").mkdir(parents=True)
    prompt = "# Runtime evidence\n\nUse only the current Attempt evidence.\n"
    (evidence / "instructions.md").write_text(prompt, encoding="utf-8")
    manifest = _attempt_manifest()
    (evidence / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "role": "optimizer",
                "lineage_checkpoint": manifest.epoch_evidence_checkpoint,
                "prompt_fragment_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "through_completed_epoch": 0,
                "current_epoch": {
                    "number": 1,
                    "status": "in_progress",
                    "snapshot_digest": manifest.attempt_evidence_digest,
                    "trigger": None,
                },
                "visibility": {
                    "completed_epochs": "promoted_lineage",
                    "current_attempts_before": 1,
                },
            }
        ),
        encoding="utf-8",
    )
    manifest_path = root / "attempt.json"
    manifest_path.write_bytes(manifest.canonical_json_bytes())

    provider_bin = tmp_path / "provider-bin"
    provider_bin.mkdir()
    fake_codex = provider_bin / "codex"
    fake_codex.write_text(
        """#!/usr/bin/env python3
import json
import os
from pathlib import Path

attempt = json.loads(Path(os.environ["ATREX_ATTEMPT_MANIFEST"]).read_text())
thread_id = "01234567-89ab-cdef-0123-456789abcdef"
rollout = Path(os.environ["CODEX_HOME"]) / "sessions/2026" / f"rollout-test-{thread_id}.jsonl"
rollout.parent.mkdir(parents=True)
usage = {
    "input_tokens": 12,
    "output_tokens": 4,
    "cached_input_tokens": 2,
    "total_tokens": 16,
}
rollout.write_text(json.dumps({
    "type": "event_msg",
    "payload": {
        "type": "token_count",
        "info": {"last_token_usage": usage, "total_token_usage": usage},
    },
}) + "\\n")
Path(os.environ["ATREX_ATTEMPT_REPORT_PATH"]).write_text(json.dumps({
    "schema_version": 3,
    "attempt_id": attempt["attempt_id"],
    "status": "blocked",
    "hypothesis": "integration hypothesis",
    "bottleneck": "integration bottleneck",
    "plan": ["exercise the real Core entrypoint"],
    "change_summary": "no candidate produced by the smoke Provider",
    "profile_evidence": "not required for protocol integration",
    "evaluation_evidence": "no evaluation because this is a blocked report",
    "result_interpretation": "the Core protocol path completed",
    "decision": "blocked",
    "research_sources": [],
    "lessons": ["Runtime and Core protocol versions agree"],
    "next_directions": [],
    "experiments": [{
        "sequence": 1,
        "recorded_at": "2026-08-17T00:00:00Z",
        "name": "real-core-smoke",
        "hypothesis": "the current Core accepts Runtime manifest v6",
        "change": "none",
        "candidate_artifact_digest": None,
        "evidence": "Core reached the Provider and wrote a terminal report",
        "result": "completed",
        "decision": "pivot"
    }]
}))
print(json.dumps({"type": "thread.started", "thread_id": thread_id}), flush=True)
print(json.dumps({
    "type": "result",
    "usage": {
        "input_tokens": 12,
        "output_tokens": 4,
        "cached_input_tokens": 2,
        "total_tokens": 16
    }
}), flush=True)
""",
        encoding="utf-8",
    )
    fake_codex.chmod(0o700)
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    driver = CoreOptimizerSessionDriver(
        CleanEnvironmentLauncher(Path("/usr/bin/env")),
        CoreOptimizerProcessConfig(
            command_prefix=(sys.executable,),
            isolated_home_environment_keys=("HOME",),
            session_trace_relative_path="sessions/core",
            token_usage_report_relative_path="scratch/token-usage.json",
            max_attempt_report_bytes=65_536,
            timeout_seconds=10,
            terminate_grace_seconds=1,
            max_diagnostic_bytes=8192,
            max_session_tokens=1000,
            agent_backend="codex",
        ),
        artifacts,
    )

    result = await driver.run(
        PreparedAttempt(root, manifest_path, root / "sessions", "real-core"),
        OptimizerSessionConfig(
            environment=(("PATH", f"{provider_bin}{os.pathsep}{os.environ['PATH']}"),),
            gateway_endpoint="http://gateway-proxy",
            gateway_capability="attempt-capability",
        ),
    )

    assert result.finish_reason == "completed"
    assert result.attempt_report is not None
    assert result.attempt_report.status == "blocked"
    assert result.token_usage == TokenUsage(10, 4, 2, 0)
    assert result.session_trace_digest is not None
    trace = artifacts.verify(result.session_trace_digest).payload_path
    assert '"input_tokens": 12' in (trace / "provider/stdout.stream-json").read_text()
