"""Framework-neutral Core repository Optimizer process adapter tests."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from threading import Thread
from typing import Any

import pytest
from conftest import digest

from atrex_runtime.artifacts.local import ArtifactKind, LocalArtifactStore
from atrex_runtime.domain.errors import InfrastructureError
from atrex_runtime.domain.ids import (
    new_attempt_id,
    new_campaign_id,
    new_epoch_id,
    new_kernel_agent_revision_id,
    new_kernel_revision_id,
    new_lineage_id,
)
from atrex_runtime.domain.models import Dsl, TokenUsage
from atrex_runtime.workers.attempt_report import AttemptReportV12
from atrex_runtime.workers.core import (
    CoreOptimizerProcessConfig,
    CoreOptimizerSessionDriver,
)
from atrex_runtime.workers.core_phase import CorePhaseRunner
from atrex_runtime.workers.launcher import CleanEnvironmentLauncher
from atrex_runtime.workers.manifest import AttemptInputManifestV9, AttemptTaskContextV5
from atrex_runtime.workers.optimizer import OptimizerSessionConfig
from atrex_runtime.workers.workspace import PreparedAttempt

_CORRECTNESS_POLICY_JSON = json.dumps(
    {
        "comparison": "elementwise",
        "formula": "abs(candidate - reference) <= atol + rtol * abs(reference)",
        "default_tolerance": {"atol": 0.01, "rtol": 0.05},
        "output_tolerances": {},
    }
)


@pytest.mark.parametrize(
    "phase", ("problem_generalization", "framework_baseline", "optimization_attempt")
)
@pytest.mark.parametrize("backend", ("claude", "codex", "qodercli", "pi"))
@pytest.mark.parametrize("model", (None, "lineage-model"))
def test_visible_core_config_matches_runtime_binding(
    tmp_path: Path, phase: str, backend: str, model: str | None
) -> None:
    source = tmp_path / "source"
    (source / "src").mkdir(parents=True)
    (source / "src/main.py").write_text("pass\n")
    (source / "atrex-bundle.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "bundle_format": "atrex-kernel-agent-bundle-v1",
                "entrypoint": {"command": "src/main.py"},
            }
        )
    )
    original = {
        "schema_version": 2,
        "agent_backend": "codex",
        "reasoning_effort": "max",
        "session_settings": "stale settings",
        "model": "stale-model",
        "prompts": {"optimization_attempt": "prompts/episode.md"},
        "prompt_fragments": {"attempt_tools": "prompts/attempt-tools.md"},
    }
    (source / "atrex-agent.json").write_text(json.dumps(original))
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    source_digest = artifacts.put_directory(source, ArtifactKind.KERNEL_AGENT)
    root = tmp_path / "workspace"
    repository = artifacts.materialize(source_digest, root / "agent/optimizer")
    (root / "sessions").mkdir()
    repository.chmod(0o500)
    runner = CorePhaseRunner(
        CleanEnvironmentLauncher(Path("/usr/bin/env")),
        CoreOptimizerProcessConfig(
            agent_backend=backend,
            reasoning_effort="low",
            session_settings="effective settings",
            command_prefix=(sys.executable,),
            isolated_home_environment_keys=(),
            session_trace_relative_path=None,
            token_usage_report_relative_path="scratch/token-usage.json",
            max_attempt_report_bytes=65_536,
            timeout_seconds=30,
            terminate_grace_seconds=1,
            max_diagnostic_bytes=4096,
            max_session_tokens=1000,
            report_completion_retries=3,
        ),
        artifacts,
    )
    prepared = runner.prepare(root, root / "sessions")
    env = runner.runtime_environment(prepared, phase=phase, model=model)
    assert env["ATREX_REPORT_COMPLETION_RETRIES"] == "3"
    assert env["ATREX_ATTEMPT_REPORT_MAX_BYTES"] == "65536"
    effective = json.loads((repository / "atrex-agent.json").read_text())
    assert effective == {
        **original,
        "agent_backend": env["ATREX_AGENT_BACKEND"],
        "reasoning_effort": env["ATREX_AGENT_REASONING_EFFORT"],
        "session_settings": env["ATREX_AGENT_SESSION_SETTINGS"],
        "model": model,
        "prompt_root": "workspace",
    }
    assert not (repository.stat().st_mode & 0o222)
    assert not ((repository / "atrex-agent.json").stat().st_mode & 0o222)
    assert not list(repository.glob(".agent-config-*"))
    stored = artifacts.verify(source_digest).payload_path
    assert json.loads((stored / "atrex-agent.json").read_text()) == original


def _attempt_manifest() -> AttemptInputManifestV9:
    return AttemptInputManifestV9(
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
    manifest_path = root / ".runtime/attempt.json"
    manifest_path.parent.mkdir(exist_ok=True)
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
    "schema_version": 12,
    "attempt_id": attempt["attempt_id"],
    "status": "blocked",
    "hypothesis": "test hypothesis",
    "diagnosis": {"bottleneck": "test bottleneck", "evidence": "test profile evidence"},
    "approach": {
        "summary": "test approach",
        "steps": ["test plan"],
        "expected_impact": "test expected impact",
        "risks": []
    },
    "final_candidate": None,
    "evidence_summary": {
        "correctness": "test correctness evidence",
        "performance": "test evaluation evidence"
    },
    "profile_evidence": None,
    "analysis": "test result",
    "knowledge_used": [],
    "findings": [{
        "category": "infrastructure",
        "observation": "the task was blocked",
        "root_cause": "test blocker",
        "resolution": "no fix was possible within the Attempt",
        "lesson": "test lesson",
        "supporting_experiment_ids": ["experiment_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"]
    }],
    "blocker": "test blocker",
    "experiments": [{
        "experiment_id": "experiment_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "direction_id": "direction_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "sequence": 1,
        "recorded_at": "2026-08-17T00:00:00Z",
        "name": "test experiment",
        "hypothesis": "test hypothesis",
        "change": "none",
        "before": {"kernel_artifact_digest": "sha256:" + "a" * 64,
                   "result_artifact_digests": ["sha256:" + "b" * 64]},
        "after": None,
        "evidence": "test evidence",
        "analysis": "the hypothesis could not be measured",
        "action": "abandon_direction"
    }],
    "direction_events": [{
        "direction_event_id": "directionevent_11111111111111111111111111111111",
        "direction_id": "direction_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "recorded_at": "2026-08-17T00:00:00Z",
        "action": "propose",
        "name": "test direction",
        "hypothesis": "test hypothesis",
        "rationale": "test evidence",
        "plan": ["run the test"],
        "success_criteria": "the test succeeds",
        "stop_conditions": "the task is blocked",
        "analysis": None,
        "supporting_experiment_ids": []
    }, {
        "direction_event_id": "directionevent_22222222222222222222222222222222",
        "direction_id": "direction_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "recorded_at": "2026-08-17T00:01:00Z",
        "action": "abandon",
        "name": None,
        "hypothesis": None,
        "rationale": None,
        "plan": [],
        "success_criteria": None,
        "stop_conditions": None,
        "analysis": "the task was blocked",
        "supporting_experiment_ids": ["experiment_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"]
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
        PreparedAttempt(
            root,
            manifest_path,
            session_root,
            "session-id",
            correctness_policy_json=_CORRECTNESS_POLICY_JSON,
        ),
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
    assert result.runtime_state_digest is not None
    assert (
        artifacts.verify(result.runtime_state_digest).kind
        is ArtifactKind.KERNEL_AGENT_RUNTIME_STATE
    )
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
@pytest.mark.parametrize("accept_on_segment", [1, 2, None], ids=["accepted", "repair", "exhausted"])
@pytest.mark.parametrize("bundle", ["atrex-kernel-agent-core", "kernel-design-agents"])
@pytest.mark.parametrize("provider_case", ["codex", "claude-main", "claude-tree", "claude-stalled"])
async def test_runtime_executes_current_core_bundle_with_attempt_v9(
    tmp_path: Path,
    accept_on_segment: int | None,
    bundle: str,
    provider_case: str,
) -> None:
    root = tmp_path / "attempt"
    repository = root / "agent/optimizer"
    source = Path(__file__).resolve().parents[1] / "src" / bundle
    backend = provider_case.split("-")[0]
    shutil.copytree(
        source,
        repository,
        ignore=shutil.ignore_patterns(
            ".git", ".mypy_cache", ".pytest_cache", ".ruff_cache", "__pycache__", "*.pyc"
        ),
    )
    for relative in (
        "input/kernel",
        ".runtime",
        "work/kernel",
        "scratch",
        "sessions",
    ):
        (root / relative).mkdir(parents=True, exist_ok=True)
    (root / ".runtime/agent-problem.json").write_text(
        json.dumps(
            {
                "schema_version": "atrex.agent_problem.v1",
                "objective": "implement example while exact cases remain private",
            }
        ),
        encoding="utf-8",
    )
    evidence = root / "input/evidence"
    (evidence / "epochs/00000001/trajectories/00000001/attempts").mkdir(parents=True)
    prompt = "# Runtime evidence\n\nUse only the current Attempt evidence.\n"
    (root / ".runtime/evidence-instructions.md").write_text(prompt, encoding="utf-8")
    manifest = _attempt_manifest()
    (root / ".runtime/evidence-manifest.json").write_text(
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
                    "completed_epochs": "all_completed_branches",
                    "current_attempts_before": 1,
                    "current_trajectory_ordinal": 1,
                },
            }
        ),
        encoding="utf-8",
    )
    manifest_path = root / ".runtime/attempt.json"
    manifest_path.parent.mkdir(exist_ok=True)
    manifest_path.write_bytes(manifest.canonical_json_bytes())

    provider_bin = tmp_path / "provider-bin"
    provider_bin.mkdir()
    fake_provider = provider_bin / backend
    fake_provider.write_text(
        """#!/usr/bin/env python3
import json
import os
import sys
import urllib.request
from pathlib import Path

attempt = json.loads(Path(os.environ["ATREX_ATTEMPT_MANIFEST"]).read_text())
invocations_path = Path("scratch/fake-provider-invocations.json")
invocations = json.loads(invocations_path.read_text()) if invocations_path.exists() else []
ordinal = len(invocations) + 1
thread_id = f"00000000-0000-0000-0000-{ordinal:012d}"
provider_case = FAKE_PROVIDER_CASE
if provider_case != "codex":
    thread_id = sys.argv[sys.argv.index("--session-id") + 1]
invocations.append({
    "attempt_id": attempt["attempt_id"], "thread_id": thread_id,
    "prompt": sys.argv[-1],
})
invocations_path.write_text(json.dumps(invocations))
usage = {
    "input_tokens": 12,
    "output_tokens": 4,
    "cached_input_tokens": 2,
    "total_tokens": 16,
}
if provider_case == "codex":
    rollout = Path(os.environ["CODEX_HOME"]) / "sessions/2026" / f"rollout-test-{thread_id}.jsonl"
    rollout.parent.mkdir(parents=True)
    rollout.write_text(json.dumps({
        "type": "event_msg",
        "payload": {
            "type": "token_count",
            "info": {"last_token_usage": usage, "total_token_usage": usage},
        },
    }) + "\\n")
else:
    usage = {
        "input_tokens": 10, "output_tokens": 4,
        "cache_read_input_tokens": 2, "cache_creation_input_tokens": 0,
    }
    native = Path(os.environ["CLAUDE_CONFIG_DIR"]) / "projects/test" / f"{thread_id}.jsonl"
    native.parent.mkdir(parents=True, exist_ok=True)
    main_message = {"type": "assistant", "message": {"id": "main", "usage": usage}}
    native.write_text(json.dumps(main_message) + "\\n")
    print(json.dumps(main_message), flush=True)
    if provider_case in ("claude-main", "claude-tree"):
        child = native.with_suffix("") / "subagents/child.jsonl"
        child.parent.mkdir(parents=True)
        child.write_text(json.dumps({"type": "assistant", "message": {
            "id": "child", "usage": {
                "input_tokens": 1, "output_tokens": 6,
                "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0,
            },
        }}) + "\\n")
    if provider_case == "claude-tree":
        usage = {**usage, "input_tokens": 11, "output_tokens": 10}
    elif provider_case == "claude-stalled":
        usage = {**usage, "input_tokens": 1, "output_tokens": 1, "cache_read_input_tokens": 0}
report = {
    "schema_version": 12,
    "attempt_id": attempt["attempt_id"],
    "status": "blocked",
    "hypothesis": "integration hypothesis",
    "diagnosis": {
        "bottleneck": "integration bottleneck",
        "evidence": "not required for protocol integration"
    },
    "approach": {
        "summary": "exercise the real Core entrypoint",
        "steps": ["write a terminal report"],
        "expected_impact": "validate the protocol path",
        "risks": []
    },
    "final_candidate": None,
    "evidence_summary": {
        "correctness": "no evaluation because this is a blocked report",
        "performance": "no performance result"
    },
    "profile_evidence": None,
    "analysis": "the Core protocol path completed",
    "knowledge_used": [],
    "findings": [{
        "category": "protocol",
        "observation": "the Core wrote a terminal report",
        "root_cause": "the Runtime and Core protocols agree",
        "resolution": "sealed the compatible terminal handoff",
        "lesson": "Runtime and Core protocol versions agree",
        "supporting_experiment_ids": ["experiment_bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"]
    }],
    "blocker": "smoke Provider does not produce a candidate",
    "experiments": [{
        "experiment_id": "experiment_bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
        "direction_id": "direction_bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
        "sequence": 1,
        "recorded_at": "2026-08-17T00:00:00Z",
        "name": "real-core-smoke",
        "hypothesis": "the current Core accepts Runtime manifest v9",
        "change": "none",
        "before": {"kernel_artifact_digest": "sha256:" + "a" * 64,
                   "result_artifact_digests": ["sha256:" + "b" * 64]},
        "after": None,
        "evidence": "Core reached the Provider and wrote a terminal report",
        "analysis": "the protocol integration completed successfully",
        "action": "abandon_direction"
    }],
    "direction_events": [{
        "direction_event_id": "directionevent_33333333333333333333333333333333",
        "direction_id": "direction_bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
        "recorded_at": "2026-08-17T00:00:00Z",
        "action": "propose",
        "name": "protocol smoke",
        "hypothesis": "the current Core accepts Runtime manifest v9",
        "rationale": "the integration path needs coverage",
        "plan": ["write a terminal report"],
        "success_criteria": "Runtime accepts the report",
        "stop_conditions": "the protocol path is blocked",
        "analysis": None,
        "supporting_experiment_ids": []
    }, {
        "direction_event_id": "directionevent_44444444444444444444444444444444",
        "direction_id": "direction_bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
        "recorded_at": "2026-08-17T00:01:00Z",
        "action": "abandon",
        "name": None,
        "hypothesis": None,
        "rationale": None,
        "plan": [],
        "success_criteria": None,
        "stop_conditions": None,
        "analysis": "the smoke Provider cannot produce a candidate",
        "supporting_experiment_ids": ["experiment_bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"]
    }]
}
if ordinal == FAKE_ACCEPT_ON_SEGMENT:
    submission = urllib.request.Request(
        os.environ["ATREX_GATEWAY_PROXY_URL"] + "/v1/runtime/queries",
        data=json.dumps({
            "operation": "attempt_report", "attempt_id": attempt["attempt_id"],
            "report": report,
        }).encode(),
        headers={
            "authorization": "Bearer " + os.environ["ATREX_GATEWAY_CAPABILITY"],
            "content-type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(submission, timeout=5) as response:
        assert json.loads(response.read())["result"]["status"] == "registered"
    # Simulate a lost local write after Runtime accepted the report: Core must restore it.
else:
    Path(os.environ["ATREX_ATTEMPT_REPORT_PATH"]).write_text(json.dumps(report))
print(json.dumps({"type": "thread.started", "thread_id": thread_id}), flush=True)
print(json.dumps({
    "type": "result",
    "usage": usage,
}), flush=True)
""".replace("FAKE_ACCEPT_ON_SEGMENT", repr(accept_on_segment)).replace(
            "FAKE_PROVIDER_CASE", repr(provider_case)
        ),
        encoding="utf-8",
    )
    fake_provider.chmod(0o700)
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    driver = CoreOptimizerSessionDriver(
        CleanEnvironmentLauncher(Path("/usr/bin/env")),
        CoreOptimizerProcessConfig(
            command_prefix=(sys.executable,),
            isolated_home_environment_keys=("HOME",),
            session_trace_relative_path="sessions/core",
            token_usage_report_relative_path="scratch/token-usage.json",
            max_attempt_report_bytes=65_536,
            timeout_seconds=30,
            terminate_grace_seconds=1,
            max_diagnostic_bytes=8192,
            max_session_tokens=1000,
            agent_backend=backend,
            report_completion_retries=2,
        ),
        artifacts,
    )

    requests: list[dict[str, Any]] = []
    accepted_report: dict[str, Any] | None = None

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            nonlocal accepted_report
            assert self.path == "/v1/runtime/queries"
            assert self.headers["Authorization"] == "Bearer attempt-capability"
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(request)
            assert request["attempt_id"] == manifest.attempt_id
            if request["operation"] == "attempt_report":
                accepted_report = AttemptReportV12.model_validate(request["report"]).model_dump(
                    mode="json"
                )
                value: dict[str, Any] = {"status": "registered"}
            else:
                assert request["operation"] == "attempt_report_status"
                assert set(request) == {
                    "schema_version",
                    "operation",
                    "attempt_id",
                    "idempotency_key",
                }
                value = (
                    {"status": "missing"}
                    if accepted_report is None
                    else {
                        "status": "accepted",
                        "report": accepted_report,
                        "report_artifact_digest": str(digest("accepted-smoke-report")),
                    }
                )
            body = json.dumps(
                {
                    "operation": request["operation"],
                    "status": "completed",
                    "result": value,
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            pass

    with HTTPServer(("127.0.0.1", 0), Handler) as server:
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            result = await driver.run(
                PreparedAttempt(
                    root,
                    manifest_path,
                    root / "sessions",
                    "real-core",
                    correctness_policy_json=_CORRECTNESS_POLICY_JSON,
                ),
                OptimizerSessionConfig(
                    environment=(("PATH", f"{provider_bin}{os.pathsep}{os.environ['PATH']}"),),
                    gateway_endpoint=f"http://127.0.0.1:{server.server_port}",
                    gateway_capability="attempt-capability",
                ),
            )
        finally:
            server.shutdown()
            thread.join(timeout=5)

    count = accept_on_segment or 3
    assert result.finish_reason == (
        "completed" if accept_on_segment is not None else "report-completion-exhausted"
    )
    if accept_on_segment is None:
        assert result.attempt_report is None and result.attempt_report_digest is None
    else:
        assert result.attempt_report is not None
        assert result.attempt_report.status == "blocked"
        assert result.attempt_report.model_dump(mode="json") == accepted_report
    has_child = provider_case in ("claude-main", "claude-tree")
    assert result.token_usage == TokenUsage(
        (11 if has_child else 10) * count,
        (10 if has_child else 4) * count,
        2 * count,
        0,
    )
    assert result.usage_complete == (provider_case != "claude-stalled")
    if provider_case == "claude-stalled":
        assert result.usage_warnings == ("claude_response_usage_incomplete_or_unreconciled",)
    assert result.runtime_state_digest is not None
    invocations = json.loads((root / "scratch/fake-provider-invocations.json").read_text())
    assert len(invocations) == count
    assert {item["attempt_id"] for item in invocations} == {manifest.attempt_id}
    assert len({item["thread_id"] for item in invocations}) == count
    assert all("report-only" in item["prompt"] for item in invocations[1:])
    assert sum(item["operation"] == "attempt_report_status" for item in requests) == count
    assert sum(item["operation"] == "attempt_report" for item in requests) == (
        0 if accept_on_segment is None else 1
    )
    usage_report = json.loads((root / "scratch/token-usage.json").read_text())
    assert usage_report["session_count"] == count
    assert usage_report["model_request_count"] == (2 if has_child else 1) * count
    assert usage_report["consumed"] == (23 if has_child else 16) * count
    assert usage_report["usage_complete"] == (provider_case != "claude-stalled")
    assert result.session_trace_digest is not None
    trace = artifacts.verify(result.session_trace_digest).payload_path
    raw_input = 12 if backend == "codex" else 10
    assert f'"input_tokens": {raw_input}' in (trace / "provider/stdout.stream-json").read_text()
    metadata = json.loads((trace / "session.json").read_text())
    assert metadata["report_completion"]["retries_used"] == count - 1
    assert metadata["report_completion"]["state"] == (
        "complete" if accept_on_segment is not None else "exhausted"
    )
    assert metadata["exit_status"] == (0 if accept_on_segment is not None else 127)
    assert metadata["accounting_usage"] == usage_report
    assert len(metadata["segments"]) == count
    assert len({item["session_id"] for item in metadata["segments"]}) == count
    for ordinal in range(1, count):
        segment = trace / f"continuations/{ordinal:03d}"
        assert (
            f'"input_tokens": {raw_input}' in (segment / "provider/stdout.stream-json").read_text()
        )
        assert (segment / "events.jsonl").is_file()
        assert (segment / "conversation.jsonl").is_file()


def test_inner_agent_timeout_is_reported_as_a_timeout(tmp_path: Path) -> None:
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
    # A killed Provider leaves usage_complete false, which on its own reads as an
    # invalid report rather than the timeout that produced it.
    (repository / "run.py").write_text(
        """import json
import os
import sys
from pathlib import Path

Path(os.environ["ATREX_TOKEN_USAGE_REPORT"]).write_text(json.dumps({
    "schema_version": 2,
    "usage_unit": os.environ["ATREX_USAGE_UNIT"],
    "budget": float(os.environ["ATREX_USAGE_BUDGET"]),
    "consumed": 0,
    "token_usage": {
        "uncached_input_tokens": 0,
        "output_tokens": 0,
        "cache_read_tokens": 0,
        "cache_write_tokens": 0
    },
    "credits": None,
    "budget_exhausted": False,
    "session_count": 1,
    "model_request_count": 1,
    "usage_complete": False
}))
sys.exit(124)
""",
        encoding="utf-8",
    )
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    policy = CoreOptimizerProcessConfig(
        agent_backend="claude",
        command_prefix=(sys.executable,),
        isolated_home_environment_keys=(),
        session_trace_relative_path=None,
        token_usage_report_relative_path="scratch/token-usage.json",
        max_attempt_report_bytes=65_536,
        timeout_seconds=30,
        terminate_grace_seconds=1,
        max_diagnostic_bytes=4096,
        max_session_tokens=1000,
    )
    runner = CorePhaseRunner(CleanEnvironmentLauncher(Path("/usr/bin/env")), policy, artifacts)
    prepared = runner.prepare(root, root / "sessions")
    environment = runner.runtime_environment(prepared, phase="optimization_attempt")

    with pytest.raises(InfrastructureError, match="test Core timed out"):
        runner.run(prepared, environment, label="test Core")


def test_pre_session_worker_failure_preserves_process_diagnostic(tmp_path: Path) -> None:
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
        "import sys\nprint('bootstrap manifest paths disagree', file=sys.stderr)\nsys.exit(1)\n",
        encoding="utf-8",
    )
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    policy = CoreOptimizerProcessConfig(
        agent_backend="claude",
        command_prefix=(sys.executable,),
        isolated_home_environment_keys=(),
        session_trace_relative_path=None,
        token_usage_report_relative_path="scratch/token-usage.json",
        max_attempt_report_bytes=65_536,
        timeout_seconds=30,
        terminate_grace_seconds=1,
        max_diagnostic_bytes=4096,
        max_session_tokens=1000,
    )
    runner = CorePhaseRunner(CleanEnvironmentLauncher(Path("/usr/bin/env")), policy, artifacts)
    prepared = runner.prepare(root, root / "sessions")
    environment = runner.runtime_environment(prepared, phase="framework_baseline")

    with pytest.raises(InfrastructureError) as captured:
        runner.run(prepared, environment, label="Core lineage bootstrap")

    message = str(captured.value)
    assert "process exited with 1" in message
    assert "before producing a valid provider usage report" in message
    assert "bootstrap manifest paths disagree" in message
