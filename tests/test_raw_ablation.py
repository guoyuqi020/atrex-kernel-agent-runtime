"""Raw policy, isolated authored memory, direct access and private nomination."""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
from contextlib import closing
from dataclasses import replace
from pathlib import Path

import pytest
from conftest import digest, seed_lineage
from test_core_optimizer_driver import _CORRECTNESS_POLICY_JSON, _attempt_manifest
from test_gateway_proxy import _service
from test_worker_workspace import _single_trajectory_workspace

from atrex_runtime.ablation import AblationArmSpecV1
from atrex_runtime.artifacts.local import ArtifactKind, LocalArtifactStore
from atrex_runtime.config import AgateSettings
from atrex_runtime.domain.ids import new_attempt_id
from atrex_runtime.gateway import GatewayOperation
from atrex_runtime.registry.sqlite import SqliteRegistry
from atrex_runtime.workers.core import CoreOptimizerProcessConfig, CoreOptimizerSessionDriver
from atrex_runtime.workers.launcher import CleanEnvironmentLauncher
from atrex_runtime.workers.optimizer import OptimizerSessionConfig
from atrex_runtime.workers.raw import raw_agate_environment
from atrex_runtime.workers.session_contract import materialize_session_contract
from atrex_runtime.workers.workspace import PreparedAttempt

ROOT = Path(__file__).resolve().parents[1]


def enable_raw(registry, lineage_id):
    registry._connection.execute(
        "UPDATE lineages SET raw=1, dev_only=0, tool_modules_json='[]', "
        "result_artifact_read_enabled=0 WHERE id=?",
        (lineage_id,),
    )


def test_raw_migration_defaults_and_frozen_policy(tmp_path):
    path = tmp_path / "registry.sqlite"
    with SqliteRegistry(path) as registry:
        lineage_id = seed_lineage(registry).lineage_id
    with closing(sqlite3.connect(path)) as db:
        db.execute("ALTER TABLE lineages DROP COLUMN raw")
        db.execute("PRAGMA user_version=47")
        db.commit()
    with SqliteRegistry(path) as registry:
        assert not registry.get_lineage(lineage_id).raw
        enable_raw(registry, lineage_id)
    with SqliteRegistry(path) as registry:
        assert registry.get_lineage(lineage_id).raw


@pytest.mark.parametrize(
    "patch",
    [
        dict(raw=1),
        dict(raw="true"),
        dict(dev_only=True),
        dict(tool_modules=None),
        dict(tool_modules=["directions"]),
        dict(result_artifact_read_enabled=True),
    ],
)
def test_raw_rejects_conflicting_settings(patch):
    with pytest.raises(ValueError, match="raw"):
        AblationArmSpecV1.model_validate(
            dict(
                creation_key="raw",
                source_lineage_id="lineage_" + "0" * 32,
                optimizer_attempt_budget=3,
                **(
                    {"raw": True, "tool_modules": [], "result_artifact_read_enabled": False} | patch
                ),
            )
        )


def test_raw_disables_even_forged_broad_runtime_authority(tmp_path):
    registry, control, attempt, capability, _service_instance, adapter = _service(tmp_path)
    enable_raw(registry, registry.get_epoch(attempt.epoch_id).lineage_id)
    try:
        for operation in GatewayOperation:
            with pytest.raises(PermissionError, match="raw Lineage"):
                control.authorize(
                    capability,
                    operation,
                    idempotency_key=operation,
                    request_digest=digest("raw-request"),
                )
        assert not adapter.requests
    finally:
        control.close()
        registry.close()


@pytest.mark.parametrize("bundle", ["atrex-kernel-agent-core", "kernel-design-agents"])
def test_raw_contract_cli_and_session_skip_runtime_report_recovery(tmp_path, bundle):
    contract = materialize_session_contract(
        tmp_path,
        phase="optimization_attempt",
        dsl="cuda",
        hardware_target="L20D",
        agent_backend="claude",
        model=None,
        session_timeout_seconds=60,
        usage_unit="provider_tokens",
        usage_budget=1000,
        max_attempt_report_bytes=65536,
        wiki_available=True,
        tool_modules=(),
        result_artifact_read_enabled=False,
        raw=True,
    )
    tools = json.loads((contract / "tools.json").read_text())
    assert tools["bindings"] == tools["gateway"] == {}
    assert json.loads((contract / "environment.json").read_text())["services"] == {
        "gateway": False,
        "wiki": False,
    }
    script = r"""
import sys
from types import SimpleNamespace
sys.path.insert(0,sys.argv[1])
from pathlib import Path
import runtime_tools
from sessions import attempt, raw
for command in ('gateway-execute', 'attempt-report', 'runtime-contract', 'kernel-artifact-read'):
    assert runtime_tools.main([command]) == 2
root=Path(sys.argv[2])
ctx=SimpleNamespace(workspace=root,manifest={'dsl':'cuda','context':{'operator':'test','hardware_target':'L20D'}})
raw.RawContext.from_environment=lambda:ctx
raw.AgentConfig.load=lambda *a,**k:None
raw.json_object_file=lambda *a,**k:{'schema_version':'atrex.agent_problem.v1','objective':'test'}
raw.correctness_policy_value=lambda x:x
called=[]
def execute(context,config,prompt,**kwargs):
    assert 'completion_check' not in kwargs
    assert 'No Runtime tools' in kwargs['system_prompt']
    assert 'Agate directly' in kwargs['system_prompt']
    assert 'tools/memory/' in kwargs['system_prompt']
    called.append(True)
    return 0
raw.execute_agent_session=execute
assert attempt.run() == 0 and called == [True]
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(ROOT / "src" / bundle / "src"), str(tmp_path)],
        env={
            **os.environ,
            "ATREX_RUNTIME_CONTRACT_PATH": str(contract),
            "ATREX_CORRECTNESS_POLICY_JSON": "{}",
        },
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("source_tree", [False, True])
def test_raw_workspace_excludes_bootstrap_history_and_only_restores_own_notes(
    tmp_path, monkeypatch, source_tree
):
    with SqliteRegistry(tmp_path / "registry.sqlite") as registry:
        artifacts = LocalArtifactStore(tmp_path / "artifacts")
        assembler, request, lineage, _ = _single_trajectory_workspace(
            tmp_path, registry, artifacts, core_seed=True
        )
        enable_raw(registry, lineage)
        if source_tree:
            from types import SimpleNamespace

            monkeypatch.setattr(
                "atrex_runtime.kernel_sources.load_kernel_source_contract",
                lambda *args: SimpleNamespace(
                    editable_roots=("kernels",), package_root=".", runtime_requirements=()
                ),
            )
        request = replace(request, raw=True, tool_modules=(), result_artifact_read_enabled=False)
        first = assembler.prepare(request)
        assert not list((first.root / "input/evidence").iterdir())
        assert not (first.root / "tools/seed.md").exists()
        if source_tree:
            instructions = (first.root / ".runtime/source-instructions.md").read_text()
            assert "editable_roots" in instructions and "kernels" in instructions
            assert "Runtime-owned" not in instructions and "gateway-execute" not in instructions
        assert not list((first.root / "input").rglob("conversation.jsonl"))
        (first.root / "tools/my-notes.txt").write_text("authored observations")
        checkpoint = first.seal_runtime_state(artifacts)
        registry.record_attempt_runtime_state(request.attempt_id, checkpoint)
        retry = assembler.prepare(request)
        assert (retry.root / "tools/my-notes.txt").read_text() == "authored observations"
        old = registry.get_attempt(request.attempt_id)
        next_id = new_attempt_id()
        registry.insert_attempt(
            replace(
                old,
                id=next_id,
                ordinal=2,
                runtime_state_digest=None,
                input_runtime_state_digest=checkpoint,
            )
        )
        second = assembler.prepare(replace(request, attempt_id=next_id))
        assert (second.root / "tools/my-notes.txt").read_text() == "authored observations"
        # A peer cannot inherit these notes even if Workflow routes that checkpoint to it.
        registry._connection.execute(
            "UPDATE epoch_branch_workflows SET trajectories=2 WHERE epoch_id=?", (old.epoch_id,)
        )
        peer_id = new_attempt_id()
        registry.insert_attempt(
            replace(
                old,
                id=peer_id,
                ordinal=1,
                trajectory_ordinal=2,
                runtime_state_digest=None,
                input_runtime_state_digest=checkpoint,
            )
        )
        peer = assembler.prepare(replace(request, attempt_id=peer_id))
        assert not (peer.root / "tools/my-notes.txt").exists()
        assert not list((peer.root / "input/evidence").iterdir())


@pytest.mark.parametrize("mode", ["none", "token", "ak_sk"])
def test_direct_agate_uses_configured_auth_and_destination(mode):
    extra = (
        {}
        if mode == "none"
        else (
            {"token_env": "TOKEN"}
            if mode == "token"
            else {"access_key_env": "AK", "secret_key_env": "SK"}
        )
    )
    settings = AgateSettings(
        base_url="https://test.example",
        auth_mode=mode,
        http_timeout_s=30,
        wait_timeout_s=60,
        **extra,
    )
    result = dict(
        raw_agate_environment(settings, {"TOKEN": "test-token", "AK": "test-ak", "SK": "test-sk"})
    )
    assert result["AGATE_URL"] == "https://test.example"
    assert ("AGATE_TOKEN" in result) == (mode == "token")
    assert ("AGATE_AK" in result) == (mode == "ak_sk")
    assert ("AGATE_SK" in result) == (mode == "ak_sk")
    if mode == "ak_sk":
        assert result["AGATE_AK"] == "test-ak" and result["AGATE_SK"] == "test-sk"
    if mode == "token":
        assert result["AGATE_TOKEN"] == "test-token"


@pytest.mark.anyio
@pytest.mark.parametrize("raw", [False, True])
async def test_raw_process_gets_direct_auth_and_hands_off_without_report(tmp_path, raw):
    root = tmp_path / "workspace"
    repository = root / "agent/optimizer"
    repository.mkdir(parents=True)
    for path in (".runtime", "work/kernel", "sessions", "scratch"):
        (root / path).mkdir(parents=True)
    manifest = _attempt_manifest()
    manifest_path = root / ".runtime/attempt.json"
    manifest_path.write_bytes(manifest.canonical_json_bytes())
    (root / "work/kernel/kernel.py").write_text("def kernel(): pass\n")
    (repository / "atrex-bundle.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "bundle_format": "atrex-kernel-agent-bundle-v1",
                "entrypoint": {"command": "run.py"},
            }
        )
    )
    (repository / "run.py").write_text("""import json,os
from pathlib import Path
raw=json.loads((Path(os.environ['ATREX_RUNTIME_CONTRACT_PATH'])/'environment.json').read_text())['raw']
assert ('AGATE_TOKEN' in os.environ) == raw
assert ('ATREX_GATEWAY_CAPABILITY' in os.environ) != raw
assert ('ATREX_WIKI_CAPABILITY' in os.environ) != raw
if raw:
    assert os.environ['AGATE_URL']=='https://test.example'
    assert os.environ['AGATE_TOKEN']=='test-credential'
    assert os.environ['ATREX_AGATE_PYTHON']
Path(os.environ['ATREX_TOKEN_USAGE_REPORT']).write_text(json.dumps({
    'schema_version':2,'usage_unit':'provider_tokens','budget':1000,'consumed':18,
    'token_usage':{'uncached_input_tokens':10,'output_tokens':5,'cache_read_tokens':2,'cache_write_tokens':1},
    'credits':None,'budget_exhausted':False,'session_count':1,'model_request_count':1,'usage_complete':True}))
""")
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    driver = CoreOptimizerSessionDriver(
        CleanEnvironmentLauncher(Path("/usr/bin/env")),
        CoreOptimizerProcessConfig(
            command_prefix=(sys.executable,),
            isolated_home_environment_keys=("HOME",),
            session_trace_relative_path=None,
            token_usage_report_relative_path="scratch/usage.json",
            max_attempt_report_bytes=65536,
            timeout_seconds=30,
            terminate_grace_seconds=1,
            max_diagnostic_bytes=4096,
            max_session_tokens=1000,
            agent_backend="claude",
        ),
        artifacts,
    )
    config = OptimizerSessionConfig(
        environment=(),
        raw=raw,
        tool_modules=(),
        result_artifact_read_enabled=False,
        gateway_endpoint=None if raw else "http://runtime",
        gateway_capability=None if raw else "test-cap",
        wiki_endpoint="http://runtime",
        wiki_capability="test-wiki-cap",
        raw_agate_environment=(
            ("AGATE_URL", "https://test.example"),
            ("AGATE_TOKEN", "test-credential"),
        ),
    )
    assert "test-credential" not in repr(config)
    result = await driver.run(
        PreparedAttempt(
            root,
            manifest_path,
            root / "sessions",
            "raw-test",
            correctness_policy_json=_CORRECTNESS_POLICY_JSON,
        ),
        config,
    )
    assert result.finish_reason == "completed"
    assert bool(result.attempt_report) == raw
    assert bool(result.kernel_artifact_digest) == raw
    assert not (root / "scratch/attempt-report.json").exists()
    if raw:
        assert result.attempt_report.experiments == result.attempt_report.direction_events == ()
        assert result.attempt_report.findings == ()
        assert result.attempt_report.schema_version == 13
        assert "analysis" not in result.attempt_report.model_dump(mode="json")


@pytest.mark.anyio
async def test_raw_runner_does_not_issue_proxy_capability(tmp_path):
    from test_optimizer_runner import (
        FakeFinalizer,
        FakeGatewayAuthorities,
        FakeRuntimeEventRecorder,
        FakeSessionDriver,
        FakeSessionTraceRecorder,
        FakeWorkspaceAssembler,
        SequencedOutcomes,
        _config,
        _request,
        _session_result,
    )

    from atrex_runtime.ports import AttemptCandidateResult
    from atrex_runtime.workers.optimizer import SessionOptimizerRunner
    from atrex_runtime.workers.raw import raw_handoff

    request = replace(_request(), raw=True, tool_modules=(), result_artifact_read_enabled=False)
    candidate = AttemptCandidateResult(digest("candidate"), digest("result"), True, 1.0)
    configs = []
    authority = FakeGatewayAuthorities()
    finalizer = FakeFinalizer(candidate, [])
    session = replace(
        _session_result("completed", "done", with_report=True),
        attempt_report=raw_handoff(request.attempt_id),
    )
    runner = SessionOptimizerRunner(
        FakeWorkspaceAssembler(
            PreparedAttempt(
                tmp_path, tmp_path / ".runtime/attempt.json", tmp_path / "sessions", "raw"
            )
        ),
        FakeSessionDriver(session, configs=configs),
        SequencedOutcomes([None]),
        finalizer,
        authority,
        FakeSessionTraceRecorder([]),
        FakeRuntimeEventRecorder([]),
        _config(tmp_path),
        wiki_enabled=True,
    )
    assert (await runner.run_attempt(request)).candidate == candidate
    assert authority.calls == 0
    assert (
        configs[0].raw
        and configs[0].gateway_capability is None
        and configs[0].wiki_capability is None
    )
    assert len(finalizer.calls) == 1


@pytest.mark.anyio
@pytest.mark.parametrize("passing", [True, False])
async def test_raw_nomination_without_gateway_calls_always_runs_private_eval(tmp_path, passing):
    from test_gateway_finalization import FakeClient, FakeContexts, FakeEvents, _builder

    from atrex_runtime.gateway.finalization import (
        AgateAuthoritativeCandidateEvaluator,
        BootstrapEvaluationStage,
    )

    registry, control, attempt, _capability, service, adapter = _service(tmp_path)
    enable_raw(registry, registry.get_epoch(attempt.epoch_id).lineage_id)
    source = tmp_path / "candidate"
    source.mkdir()
    (source / "kernel.py").write_text("def kernel(): pass\n")
    candidate = service._artifacts.put_directory(source, ArtifactKind.KERNEL)

    class Client(FakeClient):
        def get_job(self, *args, **kwargs):
            job = super().get_job(*args, **kwargs)
            job["result"] = {"all_pass": passing, "latency_us_geomean": 7.5}
            return job

    client = Client()
    finalizer = AgateAuthoritativeCandidateEvaluator(
        client,
        _builder,
        FakeContexts(),
        service._artifacts,
        control,
        FakeEvents(),
        wait_timeout_s=100,
        bootstrap_stages=(BootstrapEvaluationStage(1),),
    )
    try:
        result = await finalizer.finalize(attempt.id, candidate, independent_evaluate=False)
        assert result.correct is passing
        assert result.latency_us == (7.5 if passing else None)
        assert len(client.submitted) == 1 and not adapter.requests
        again = await finalizer.finalize(attempt.id, candidate, independent_evaluate=False)
        assert again == result and len(client.submitted) == 1
    finally:
        control.close()
        registry.close()


@pytest.mark.anyio
@pytest.mark.parametrize("bundle", ["atrex-kernel-agent-core", "kernel-design-agents"])
async def test_raw_real_bundle_runs_one_agent_session_without_report_tool(tmp_path, bundle):
    import shutil

    root = tmp_path / "attempt"
    repository = root / "agent/optimizer"
    shutil.copytree(
        ROOT / "src" / bundle,
        repository,
        ignore=shutil.ignore_patterns(
            ".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"
        ),
    )
    for path in (
        ".runtime",
        "input/kernel",
        "input/evidence",
        "work/kernel",
        "scratch",
        "sessions",
    ):
        (root / path).mkdir(parents=True, exist_ok=True)
    manifest = _attempt_manifest()
    manifest_path = root / ".runtime/attempt.json"
    manifest_path.write_bytes(manifest.canonical_json_bytes())
    (root / ".runtime/agent-problem.json").write_text(
        json.dumps(
            {"schema_version": "atrex.agent_problem.v1", "objective": "optimize supplied kernel"}
        )
    )
    (root / "work/kernel/kernel.py").write_text("def kernel(): pass\n")
    bins = tmp_path / "bin"
    bins.mkdir()
    provider = bins / "claude"
    provider.write_text(r"""#!/usr/bin/env python3
import json,os,sys,subprocess
from pathlib import Path
assert not os.environ.get('ATREX_GATEWAY_CAPABILITY')
assert os.environ['AGATE_TOKEN']=='test-token'
assert not list(Path('input/evidence').iterdir())
probe=subprocess.run([os.environ['ATREX_AGATE_PYTHON'],'-m','atrex_gateway_client.cli','--help'],capture_output=True)
assert probe.returncode==0
counter=Path('scratch/calls.txt')
assert not counter.exists()
counter.write_text('1')
Path('tools/my-notes.md').write_text('Agent-owned memory')
session=sys.argv[sys.argv.index('--session-id')+1]
usage={'input_tokens':10,'output_tokens':4,'cache_read_input_tokens':2,'cache_creation_input_tokens':0}
native=Path(os.environ['CLAUDE_CONFIG_DIR'])/'projects/test'/f'{session}.jsonl'
native.parent.mkdir(parents=True,exist_ok=True)
message={'type':'assistant','message':{'id':'one','usage':usage}}
native.write_text(json.dumps(message)+'\n')
print(json.dumps(message),flush=True)
print(json.dumps({'type':'result','usage':usage}),flush=True)
""")
    provider.chmod(0o700)
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    driver = CoreOptimizerSessionDriver(
        CleanEnvironmentLauncher(Path("/usr/bin/env")),
        CoreOptimizerProcessConfig(
            command_prefix=(sys.executable,),
            isolated_home_environment_keys=("HOME",),
            session_trace_relative_path="sessions/core",
            token_usage_report_relative_path="scratch/usage.json",
            max_attempt_report_bytes=65536,
            timeout_seconds=30,
            terminate_grace_seconds=1,
            max_diagnostic_bytes=8192,
            max_session_tokens=1000,
            agent_backend="claude",
        ),
        artifacts,
    )
    result = await driver.run(
        PreparedAttempt(
            root,
            manifest_path,
            root / "sessions",
            "raw-real",
            correctness_policy_json=_CORRECTNESS_POLICY_JSON,
        ),
        OptimizerSessionConfig(
            environment=(("PATH", f"{bins}{os.pathsep}{os.environ['PATH']}"),),
            raw=True,
            tool_modules=(),
            result_artifact_read_enabled=False,
            raw_agate_environment=(
                ("AGATE_URL", "https://test.example"),
                ("AGATE_TOKEN", "test-token"),
            ),
        ),
    )
    assert result.finish_reason == "completed"
    assert result.token_usage.total_tokens == 16
    assert result.attempt_report.status == "candidate_ready"
    assert result.kernel_artifact_digest and result.session_trace_digest
    assert not (root / "scratch/attempt-report.json").exists()
    assert (root / "scratch/calls.txt").read_text() == "1"
    state = artifacts.verify(result.runtime_state_digest).payload_path
    assert (state / "tools/my-notes.md").read_text() == "Agent-owned memory"
