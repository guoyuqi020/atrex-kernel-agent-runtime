"""Fixed Evolver workspace, process, and Challenger collection tests."""

from __future__ import annotations

import json
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest
from conftest import NOW, digest, kernel_agent_limits

from atrex_runtime.artifacts.local import ArtifactKind, LocalArtifactStore
from atrex_runtime.domain.errors import InfrastructureError
from atrex_runtime.domain.ids import (
    ArtifactDigest,
    new_campaign_id,
    new_epoch_id,
    new_kernel_agent_revision_id,
    new_lineage_id,
)
from atrex_runtime.domain.models import (
    Dsl,
    KernelAgentCatalogEntry,
    KernelAgentRevision,
    WorkerSessionStatus,
)
from atrex_runtime.ports import (
    BuildChallengerRequest,
    KernelAgentCandidateProposal,
    KernelAgentReuseProposal,
)
from atrex_runtime.registry.sqlite import SqliteRegistry
from atrex_runtime.workers.evolution import (
    EvolutionInputManifestV4,
    EvolutionProcessConfig,
    EvolutionWorkspaceAssembler,
    EvolverBundleRunner,
    SubprocessEvolutionSessionDriver,
)
from atrex_runtime.workers.launcher import CleanEnvironmentLauncher


@dataclass
class FakeRuntimeEventRecorder:
    records: list[tuple[str, str, object]]

    def record_runtime_event(
        self,
        kind: str,
        aggregate_id: str,
        payload: object = None,
    ) -> None:
        self.records.append((kind, aggregate_id, payload))


def _optimizer_repository(
    artifacts: LocalArtifactStore,
    root: Path,
) -> ArtifactDigest:
    source = root / "source-optimizer"
    prompt = source / "prompts/episode.md"
    skill = source / "skills/loop/SKILL.md"
    prompt.parent.mkdir(parents=True)
    skill.parent.mkdir(parents=True)
    (source / "src").mkdir()
    prompt.write_text("parent optimizer\n", encoding="utf-8")
    skill.write_text("parent skill\n", encoding="utf-8")
    (source / "atrex-bundle.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "bundle_format": "atrex-kernel-agent-bundle-v1",
                "entrypoint": {
                    "command": "src/main.py",
                },
            }
        ),
        encoding="utf-8",
    )
    (source / "src/main.py").write_text("def main(): ...\n")
    return artifacts.put_directory(source, ArtifactKind.KERNEL_AGENT)


def _parent(artifacts: LocalArtifactStore, tmp_path: Path) -> KernelAgentRevision:
    return KernelAgentRevision(
        id=new_kernel_agent_revision_id(),
        parent_id=None,
        creation_key="bootstrap:test",
        dsl=Dsl.TRITON,
        optimizer_digest=_optimizer_repository(artifacts, tmp_path),
        created_by="bootstrap",
        created_at=NOW,
        source_provenance_digest=digest("source-provenance"),
    )


def _evidence(artifacts: LocalArtifactStore, tmp_path: Path) -> ArtifactDigest:
    source = tmp_path / "evidence-source"
    (source / "bootstrap").mkdir(parents=True)
    (source / "bootstrap/history.md").write_text("epoch evidence\n", encoding="utf-8")
    (source / "bootstrap-metadata.json").write_text(
        json.dumps({"schema_version": 1, "source": "test"}),
        encoding="utf-8",
    )
    (source / "checkpoint.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "lineage_id": "lineage_test",
                "through_epoch": 0,
                "previous_checkpoint_digest": None,
            }
        ),
        encoding="utf-8",
    )
    return artifacts.put_directory(source, ArtifactKind.EVIDENCE)


def _agent_script(
    tmp_path: Path,
    *,
    declared_path: str = "prompts/episode.md",
) -> Path:
    script = tmp_path / "evolve.py"
    script.write_text(
        """import json
import os
import sys
from pathlib import Path

assert sys.stdin.read() == "Run the versioned Evolver Bundle once."
manifest = json.loads(Path(os.environ["ATREX_EVOLUTION_INPUT"]).read_text())
candidate = Path(os.environ["ATREX_EVOLUTION_CANDIDATE"])
(Path(os.environ["ATREX_EVOLUTION_INPUT"]).parent / "scratch/agent-home/sessions").mkdir(
    parents=True
)
trace = Path(os.environ["ATREX_EVOLUTION_INPUT"]).parent
(trace / "scratch/agent-home/sessions/trace.jsonl").write_text(
    '{"event":"completed"}\\n'
)
Path(os.environ["ATREX_TOKEN_USAGE_REPORT"]).write_text(json.dumps({
    "schema_version": 2,
    "usage_unit": "provider_tokens",
    "budget": None,
    "consumed": 200,
    "token_usage": {
        "uncached_input_tokens": 120,
        "output_tokens": 30,
        "cache_read_tokens": 40,
        "cache_write_tokens": 10,
    },
    "credits": None,
    "budget_exhausted": False,
    "session_count": 1,
    "model_request_count": 1,
    "usage_complete": True,
}))
(candidate / "prompts/episode.md").write_text("challenger optimizer\\n")
Path(os.environ["ATREX_EVOLUTION_OUTPUT"]).write_text(json.dumps({
    "schema_version": 3,
    "proposal_type": "evolved",
    "base_revision_id": manifest["parent_revision_id"],
    "hypothesis": "Use a more targeted optimization strategy.",
    "expected_effect": "Reach a better Kernel within the fixed attempt budget.",
    "unimplemented_capabilities": [{
        "capability": "Automatic profiler-guided tool synthesis.",
        "expected_benefit": "Avoid spending attempts on irrelevant bottlenecks.",
        "reason_unimplemented": "The Evolver has no profiler capability."
    }],
    "changed_paths": ["""
        + json.dumps(declared_path)
        + """],
}))
""",
        encoding="utf-8",
    )
    return script


def _evolver_bundle(
    artifacts: LocalArtifactStore,
    tmp_path: Path,
    entrypoint: Path,
) -> ArtifactDigest:
    source = tmp_path / "source-evolver"
    (source / "src").mkdir(parents=True)
    shutil.copyfile(entrypoint, source / "src/main.py")
    (source / "atrex-evolver-bundle.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "bundle_format": "atrex-kernel-agent-evolver-bundle-v1",
                "entrypoint": {"command": "src/main.py"},
            }
        ),
        encoding="utf-8",
    )
    return artifacts.put_directory(source, ArtifactKind.EVOLVER_BUNDLE)


def _reuse_agent_script(tmp_path: Path, revision_id: str) -> Path:
    script = tmp_path / "reuse.py"
    script.write_text(
        """import json
import os
import sys
from pathlib import Path

assert sys.stdin.read() == "Run the versioned Evolver Bundle once."
Path(os.environ["ATREX_TOKEN_USAGE_REPORT"]).write_text(json.dumps({
    "schema_version": 2,
    "usage_unit": "provider_tokens",
    "budget": None,
    "consumed": 15,
    "token_usage": {
        "uncached_input_tokens": 10,
        "output_tokens": 5,
        "cache_read_tokens": 0,
        "cache_write_tokens": 0,
    },
    "credits": None,
    "budget_exhausted": False,
    "session_count": 1,
    "model_request_count": 1,
    "usage_complete": True,
}))
Path(os.environ["ATREX_EVOLUTION_OUTPUT"]).write_text(json.dumps({
    "schema_version": 3,
    "proposal_type": "reuse",
    "candidate_revision_id": """
        + json.dumps(revision_id)
        + """,
    "hypothesis": "Retry a historical design unchanged.",
    "expected_effect": "Reproduce its previously useful search behavior."
}))
""",
        encoding="utf-8",
    )
    return script


def _history_agent_script(tmp_path: Path, revision_id: str) -> Path:
    script = tmp_path / "evolve-from-history.py"
    script.write_text(
        """import json
import os
import subprocess
import sys
from pathlib import Path

assert sys.stdin.read() == "Run the versioned Evolver Bundle once."
workspace = Path(os.environ["ATREX_EVOLUTION_INPUT"]).parent
subprocess.run([
    sys.executable,
    "runtime-tools/evolver_tools.py",
    "candidate-reset",
    "--base",
    """
        + json.dumps(revision_id)
        + """,
], cwd=workspace, check=True)
candidate = Path(os.environ["ATREX_EVOLUTION_CANDIDATE"])
(candidate / "prompts/episode.md").write_text("history repaired optimizer\\n")
Path(os.environ["ATREX_TOKEN_USAGE_REPORT"]).write_text(json.dumps({
    "schema_version": 2,
    "usage_unit": "provider_tokens",
    "budget": None,
    "consumed": 30,
    "token_usage": {
        "uncached_input_tokens": 20,
        "output_tokens": 10,
        "cache_read_tokens": 0,
        "cache_write_tokens": 0,
    },
    "credits": None,
    "budget_exhausted": False,
    "session_count": 1,
    "model_request_count": 1,
    "usage_complete": True,
}))
Path(os.environ["ATREX_EVOLUTION_OUTPUT"]).write_text(json.dumps({
    "schema_version": 3,
    "proposal_type": "evolve_from_history",
    "base_revision_id": """
        + json.dumps(revision_id)
        + """,
    "hypothesis": "Repair one weak step in the historical design.",
    "expected_effect": "Retain its prior strengths with a narrower policy.",
    "changed_paths": ["prompts/episode.md"]
}))
""",
        encoding="utf-8",
    )
    return script


def _request(
    artifacts: LocalArtifactStore,
    tmp_path: Path,
) -> BuildChallengerRequest:
    return BuildChallengerRequest(
        parent_revision=_parent(artifacts, tmp_path),
        epoch_id=new_epoch_id(),
        evidence_checkpoint=_evidence(artifacts, tmp_path),
        idempotency_key="epoch:test:challenger",
        model="evolver-model",
    )


def _historical_catalog(revision: KernelAgentRevision) -> tuple[KernelAgentCatalogEntry, ...]:
    return (
        KernelAgentCatalogEntry(
            revision=revision,
            revision_number=1,
            parent_revision_number=0,
            campaign_id=new_campaign_id(),
            lineage_id=new_lineage_id(),
            introduced_epoch_id=None,
            introduced_epoch_number=None,
            disposition="rejected",
            active=False,
        ),
    )


def test_evolution_workspace_copies_full_parent_to_writable_candidate(tmp_path: Path) -> None:
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    request = _request(artifacts, tmp_path)

    prepared = EvolutionWorkspaceAssembler(tmp_path / "evolutions", artifacts).prepare(request)
    manifest = EvolutionInputManifestV4.model_validate_json(prepared.manifest_path.read_bytes())

    assert manifest.parent_revision_id == request.parent_revision.id
    assert len(manifest.visible_agents) == 1
    assert manifest.visible_agents[0].relationship == "active"
    assert manifest.visible_agents[0].challenger_ordinal is None
    assert manifest.visible_agents[0].created_by == "bootstrap"
    assert manifest.schema_version == 4
    assert prepared.model == "evolver-model"
    assert manifest.paths.runtime_tools == "runtime-tools"
    parent_prompt = prepared.root / "input/parent/prompts/episode.md"
    candidate_prompt = prepared.candidate_root / "prompts/episode.md"
    assert parent_prompt.is_file()
    assert candidate_prompt.is_file()
    assert os.stat(parent_prompt).st_mode & 0o200 == 0
    assert os.stat(candidate_prompt).st_mode & 0o200
    runtime_tools = prepared.root / "runtime-tools"
    assert (runtime_tools / "evolver_tools.py").is_file()
    catalog = json.loads((runtime_tools / "catalog.json").read_text())
    assert catalog["agents"][0]["revision_id"] == request.parent_revision.id
    assert not (os.stat(runtime_tools / "catalog.json").st_mode & 0o200)
    assert json.loads((prepared.root / "scratch/candidate-base.json").read_text()) == {
        "schema_version": 1,
        "base_revision_id": request.parent_revision.id,
        "selection": "active_seed",
    }


@pytest.mark.anyio
async def test_fixed_runner_collects_complete_repository_candidate(
    tmp_path: Path,
) -> None:
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    request = _request(artifacts, tmp_path)
    evolver_digest = _evolver_bundle(artifacts, tmp_path, _agent_script(tmp_path))
    sessions = SubprocessEvolutionSessionDriver(
        CleanEnvironmentLauncher(Path("/usr/bin/env")),
        EvolutionProcessConfig(
            bundle_commit="0" * 40,
            bundle_tree="1" * 40,
            bundle_artifact_digest=evolver_digest,
            command_argv=(
                str(Path(sys.executable).resolve()),
                "input/evolver/src/main.py",
            ),
            agent_backend="claude",
            isolated_home_environment_keys=(),
            session_trace_relative_path="scratch/agent-home/sessions",
            token_usage_report_relative_path="scratch/token-usage.json",
            environment=(),
            timeout_seconds=10,
            terminate_grace_seconds=1,
            max_diagnostic_bytes=4096,
        ),
    )
    events = FakeRuntimeEventRecorder([])
    registry = SqliteRegistry(tmp_path / "registry.sqlite", clock=lambda: NOW)
    runner = EvolverBundleRunner(
        EvolutionWorkspaceAssembler(
            tmp_path / "evolutions",
            artifacts,
            evolver_bundle_digest=evolver_digest,
        ),
        sessions,
        artifacts,
        events,
        kernel_agent_limits=kernel_agent_limits(),
        max_output_manifest_bytes=8192,
        worker_sessions=registry,
        backend="codex",
    )

    build = await runner.build_challenger(request)
    assert isinstance(build.proposal, KernelAgentCandidateProposal)
    candidate = build.proposal.candidate

    assert candidate.optimizer_digest != request.parent_revision.optimizer_digest
    optimizer = artifacts.verify(candidate.optimizer_digest).payload_path
    assert (optimizer / "prompts/episode.md").read_text() == ("challenger optimizer\n")
    assert (optimizer / "skills/loop/SKILL.md").is_file()
    trace_artifact = artifacts.verify(build.evolution_trace_digest)
    assert trace_artifact.kind is ArtifactKind.EVOLUTION
    trace = json.loads((trace_artifact.payload_path / "value.json").read_text())
    assert trace["agent"]["bundle_commit"] == "0" * 40
    assert trace["agent"]["bundle_tree"] == "1" * 40
    assert trace["agent"]["bundle_artifact_digest"] == evolver_digest
    assert trace["agent"]["model"] == "evolver-model"
    assert trace["schema_version"] == 7
    assert trace["token_usage"]["consumed"] == 200
    assert trace["output"]["unimplemented_capabilities"] == [
        {
            "capability": "Automatic profiler-guided tool synthesis.",
            "expected_benefit": "Avoid spending attempts on irrelevant bottlenecks.",
            "reason_unimplemented": "The Evolver has no profiler capability.",
        }
    ]
    assert [kind for kind, _aggregate, _payload in events.records] == [
        "worker.started",
        "worker.exited",
        "worker.cleaned",
        "evolution.proposal_sealed",
    ]
    assert all(aggregate == request.epoch_id for _kind, aggregate, _payload in events.records)
    assert events.records[-1][2]["unimplemented_capabilities"] == trace["output"][
        "unimplemented_capabilities"
    ]
    assert trace["candidate"]["optimizer_digest"] == candidate.optimizer_digest
    session_trace = artifacts.verify(trace["session_trace_digest"])
    assert session_trace.kind is ArtifactKind.SESSION_LOG
    assert (session_trace.payload_path / "trace.jsonl").is_file()
    worker_sessions = registry.list_worker_sessions(epoch_id=request.epoch_id)
    assert len(worker_sessions) == 1
    assert worker_sessions[0].status is WorkerSessionStatus.COMPLETED
    assert worker_sessions[0].trace_digest == trace["session_trace_digest"]
    assert worker_sessions[0].process_returncode == 0
    workspace_entrypoint = next(
        (tmp_path / "evolutions").glob("agentrev_*/run-*/input/evolver/src/main.py")
    )
    assert workspace_entrypoint.is_file()
    assert not (workspace_entrypoint.stat().st_mode & 0o200)
    registry.close()


@pytest.mark.anyio
async def test_fixed_runner_reuses_a_visible_historical_revision_without_new_content(
    tmp_path: Path,
) -> None:
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    base_request = _request(artifacts, tmp_path)
    historical = KernelAgentRevision(
        id=new_kernel_agent_revision_id(),
        parent_id=base_request.parent_revision.id,
        creation_key="epoch:old:challenger:1",
        dsl=base_request.parent_revision.dsl,
        optimizer_digest=base_request.parent_revision.optimizer_digest,
        created_by="evolver",
        created_at=NOW,
        evolution_trace_digest=digest("old-evolution"),
    )
    request = BuildChallengerRequest(
        parent_revision=base_request.parent_revision,
        epoch_id=base_request.epoch_id,
        evidence_checkpoint=base_request.evidence_checkpoint,
        idempotency_key=base_request.idempotency_key,
        agent_catalog=_historical_catalog(historical),
    )
    sessions = SubprocessEvolutionSessionDriver(
        CleanEnvironmentLauncher(Path("/usr/bin/env")),
        EvolutionProcessConfig(
            bundle_commit="0" * 40,
            bundle_tree="1" * 40,
            bundle_artifact_digest=digest("evolver-bundle"),
            command_argv=(
                str(Path(sys.executable).resolve()),
                str(_reuse_agent_script(tmp_path, historical.id)),
            ),
            agent_backend="claude",
            isolated_home_environment_keys=(),
            session_trace_relative_path=None,
            token_usage_report_relative_path="scratch/token-usage.json",
            environment=(),
            timeout_seconds=10,
            terminate_grace_seconds=1,
            max_diagnostic_bytes=4096,
        ),
    )
    runner = EvolverBundleRunner(
        EvolutionWorkspaceAssembler(tmp_path / "evolutions", artifacts),
        sessions,
        artifacts,
        FakeRuntimeEventRecorder([]),
        kernel_agent_limits=kernel_agent_limits(),
        max_output_manifest_bytes=8192,
    )

    build = await runner.build_challenger(request)

    assert isinstance(build.proposal, KernelAgentReuseProposal)
    assert build.proposal.candidate_revision_id == historical.id
    trace = json.loads(
        (artifacts.verify(build.evolution_trace_digest).payload_path / "value.json").read_text()
    )
    assert trace["output"]["proposal_type"] == "reuse"
    assert trace["candidate"] is None


@pytest.mark.anyio
async def test_fixed_runner_evolves_from_history_only_after_runtime_candidate_reset(
    tmp_path: Path,
) -> None:
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    base_request = _request(artifacts, tmp_path)
    historical_source = tmp_path / "historical-optimizer"
    shutil.copytree(
        artifacts.verify(base_request.parent_revision.optimizer_digest).payload_path,
        historical_source,
        copy_function=shutil.copyfile,
    )
    for directory in [historical_source, *historical_source.rglob("*")]:
        if directory.is_dir():
            directory.chmod(0o700)
        elif directory.is_file():
            directory.chmod(0o600)
    (historical_source / "prompts/episode.md").write_text("historical optimizer\n")
    historical_digest = artifacts.put_directory(
        historical_source,
        ArtifactKind.KERNEL_AGENT,
    )
    historical = KernelAgentRevision(
        id=new_kernel_agent_revision_id(),
        parent_id=base_request.parent_revision.id,
        creation_key="epoch:old:challenger:1",
        dsl=base_request.parent_revision.dsl,
        optimizer_digest=historical_digest,
        created_by="evolver",
        created_at=NOW,
        evolution_trace_digest=digest("old-evolution"),
    )
    request = BuildChallengerRequest(
        parent_revision=base_request.parent_revision,
        epoch_id=base_request.epoch_id,
        evidence_checkpoint=base_request.evidence_checkpoint,
        idempotency_key=base_request.idempotency_key,
        agent_catalog=_historical_catalog(historical),
    )
    sessions = SubprocessEvolutionSessionDriver(
        CleanEnvironmentLauncher(Path("/usr/bin/env")),
        EvolutionProcessConfig(
            bundle_commit="0" * 40,
            bundle_tree="1" * 40,
            bundle_artifact_digest=digest("evolver-bundle"),
            command_argv=(
                str(Path(sys.executable).resolve()),
                str(_history_agent_script(tmp_path, historical.id)),
            ),
            agent_backend="claude",
            isolated_home_environment_keys=(),
            session_trace_relative_path=None,
            token_usage_report_relative_path="scratch/token-usage.json",
            environment=(),
            timeout_seconds=10,
            terminate_grace_seconds=1,
            max_diagnostic_bytes=4096,
        ),
    )
    runner = EvolverBundleRunner(
        EvolutionWorkspaceAssembler(tmp_path / "evolutions", artifacts),
        sessions,
        artifacts,
        FakeRuntimeEventRecorder([]),
        kernel_agent_limits=kernel_agent_limits(),
        max_output_manifest_bytes=8192,
    )

    build = await runner.build_challenger(request)

    assert isinstance(build.proposal, KernelAgentCandidateProposal)
    assert build.proposal.proposal_type == "evolve_from_history"
    assert build.proposal.base_revision_id == historical.id
    candidate = artifacts.verify(build.proposal.candidate.optimizer_digest).payload_path
    assert (candidate / "prompts/episode.md").read_text() == "history repaired optimizer\n"
    assert (candidate / "skills/loop/SKILL.md").read_text() == "parent skill\n"


@pytest.mark.anyio
async def test_runtime_executes_current_evolver_bundle_entrypoint(tmp_path: Path) -> None:
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    request = _request(artifacts, tmp_path)
    provider_bin = tmp_path / "provider-bin"
    provider_bin.mkdir()
    fake_claude = provider_bin / "claude"
    fake_claude.write_text(
        """#!/usr/bin/env python3
import json
import os
from pathlib import Path

candidate = Path(os.environ["ATREX_EVOLUTION_CANDIDATE"])
(candidate / "prompts/integration.md").write_text("real Evolver Bundle ran\\n")
manifest = json.loads(Path(os.environ["ATREX_EVOLUTION_INPUT"]).read_text())
Path(os.environ["ATREX_EVOLUTION_OUTPUT"]).write_text(json.dumps({
    "schema_version": 3,
    "proposal_type": "evolved",
    "base_revision_id": manifest["parent_revision_id"],
    "hypothesis": "The current Evolver entrypoint accepts Runtime protocol v3.",
    "expected_effect": "Produce one valid complete Challenger Bundle.",
    "changed_paths": ["prompts/integration.md"]
}))
print(json.dumps({
    "type": "result",
    "usage": {
        "input_tokens": 16,
        "output_tokens": 5,
        "cache_read_input_tokens": 2,
        "cache_creation_input_tokens": 1
    }
}), flush=True)
""",
        encoding="utf-8",
    )
    fake_claude.chmod(0o700)
    bundle = Path(__file__).resolve().parents[1] / "src/atrex-kernel-agent-evolver"
    sessions = SubprocessEvolutionSessionDriver(
        CleanEnvironmentLauncher(Path("/usr/bin/env")),
        EvolutionProcessConfig(
            bundle_commit="0" * 40,
            bundle_tree="1" * 40,
            bundle_artifact_digest=digest("evolver-bundle"),
            command_argv=(str(Path(sys.executable).resolve()), str(bundle / "src/main.py")),
            isolated_home_environment_keys=("HOME",),
            session_trace_relative_path="scratch/evolver-session",
            token_usage_report_relative_path="scratch/token-usage.json",
            environment=(("PATH", f"{provider_bin}{os.pathsep}{os.environ['PATH']}"),),
            timeout_seconds=10,
            terminate_grace_seconds=1,
            max_diagnostic_bytes=8192,
            agent_backend="claude",
        ),
    )
    runner = EvolverBundleRunner(
        EvolutionWorkspaceAssembler(tmp_path / "evolutions", artifacts),
        sessions,
        artifacts,
        FakeRuntimeEventRecorder([]),
        kernel_agent_limits=kernel_agent_limits(),
        max_output_manifest_bytes=8192,
    )

    build = await runner.build_challenger(request)

    assert isinstance(build.proposal, KernelAgentCandidateProposal)
    candidate = artifacts.verify(build.proposal.candidate.optimizer_digest).payload_path
    assert (candidate / "prompts/integration.md").read_text() == "real Evolver Bundle ran\n"
    trace = json.loads(
        (artifacts.verify(build.evolution_trace_digest).payload_path / "value.json").read_text()
    )
    assert trace["process_returncode"] == 0
    assert trace["token_usage"]["consumed"] == 24
    session_trace = artifacts.verify(trace["session_trace_digest"]).payload_path
    assert '"input_tokens": 16' in (session_trace / "provider/stdout.stream-json").read_text()


@pytest.mark.anyio
async def test_fixed_runner_rejects_false_change_declaration(tmp_path: Path) -> None:
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    request = _request(artifacts, tmp_path)
    sessions = SubprocessEvolutionSessionDriver(
        CleanEnvironmentLauncher(Path("/usr/bin/env")),
        EvolutionProcessConfig(
            bundle_commit="0" * 40,
            bundle_tree="1" * 40,
            bundle_artifact_digest=digest("evolver-bundle"),
            command_argv=(
                str(Path(sys.executable).resolve()),
                str(_agent_script(tmp_path, declared_path="skills/loop/SKILL.md")),
            ),
            agent_backend="claude",
            isolated_home_environment_keys=(),
            session_trace_relative_path="scratch/agent-home/sessions",
            token_usage_report_relative_path="scratch/token-usage.json",
            environment=(),
            timeout_seconds=10,
            terminate_grace_seconds=1,
            max_diagnostic_bytes=4096,
        ),
    )
    events = FakeRuntimeEventRecorder([])
    runner = EvolverBundleRunner(
        EvolutionWorkspaceAssembler(tmp_path / "evolutions", artifacts),
        sessions,
        artifacts,
        events,
        kernel_agent_limits=kernel_agent_limits(),
        max_output_manifest_bytes=8192,
    )

    with pytest.raises(ValueError, match="disagrees with sealed content"):
        await runner.build_challenger(request)

    assert [kind for kind, _aggregate, _payload in events.records][-1] == (
        "evolution.candidate_rejected"
    )
    failure_payload = events.records[-1][2]
    assert isinstance(failure_payload, dict)
    failure_digest = failure_payload["failure_artifact_digest"]
    assert isinstance(failure_digest, str)
    failure_artifact = artifacts.verify(failure_digest)
    assert failure_artifact.kind is ArtifactKind.EVOLUTION
    failure = json.loads((failure_artifact.payload_path / "value.json").read_text())
    assert failure["status"] == "failed"
    assert failure["phase"] == "candidate_validation"
    assert failure["error_type"] == "ValueError"
    assert failure["process"]["returncode"] == 0
    assert failure["process"]["session_trace_digest"] is not None


@pytest.mark.anyio
async def test_process_driver_times_out_and_reaps_process(tmp_path: Path) -> None:
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    request = _request(artifacts, tmp_path)
    script = tmp_path / "sleep.py"
    script.write_text("import time\ntime.sleep(60)\n", encoding="utf-8")
    driver = SubprocessEvolutionSessionDriver(
        CleanEnvironmentLauncher(Path("/usr/bin/env")),
        EvolutionProcessConfig(
            bundle_commit="0" * 40,
            bundle_tree="1" * 40,
            bundle_artifact_digest=digest("evolver-bundle"),
            command_argv=(str(Path(sys.executable).resolve()), str(script)),
            isolated_home_environment_keys=(),
            session_trace_relative_path=None,
            token_usage_report_relative_path="scratch/token-usage.json",
            environment=(),
            timeout_seconds=0.1,
            terminate_grace_seconds=0.1,
            max_diagnostic_bytes=4096,
        ),
    )

    events = FakeRuntimeEventRecorder([])
    runner = EvolverBundleRunner(
        EvolutionWorkspaceAssembler(tmp_path / "evolutions", artifacts),
        driver,
        artifacts,
        events,
        kernel_agent_limits=kernel_agent_limits(),
        max_output_manifest_bytes=8192,
    )

    with pytest.raises(InfrastructureError, match="wall-time limit"):
        await runner.build_challenger(request)

    assert [kind for kind, _aggregate, _payload in events.records] == [
        "worker.started",
        "worker.timeout",
        "worker.cleaned",
    ]
    timeout_payload = events.records[1][2]
    assert isinstance(timeout_payload, dict)
    failure_digest = timeout_payload["failure_artifact_digest"]
    assert isinstance(failure_digest, str)
    failure_artifact = artifacts.verify(failure_digest)
    failure = json.loads((failure_artifact.payload_path / "value.json").read_text())
    assert failure["status"] == "failed"
    assert failure["phase"] == "session"
    assert failure["error_type"] == "InfrastructureError"
    assert failure["process"] is None


@pytest.mark.anyio
async def test_process_driver_uses_fixed_stdin_instruction_and_isolated_agent_home(
    tmp_path: Path,
) -> None:
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    prepared = EvolutionWorkspaceAssembler(tmp_path / "evolutions", artifacts).prepare(
        _request(artifacts, tmp_path)
    )
    capture = prepared.root / "scratch/capture.json"
    script = tmp_path / "argument-agent.py"
    script.write_text(
        """import json
import os
import sys
from pathlib import Path

Path(os.environ["ATREX_CAPTURE"]).write_text(json.dumps({
    "instruction": sys.stdin.read(),
    "home": os.environ["HOME"],
    "secondary_home": os.environ["SECONDARY_HOME"],
}))
Path(os.environ["ATREX_TOKEN_USAGE_REPORT"]).write_text(json.dumps({
    "schema_version": 2,
    "usage_unit": "provider_tokens",
    "budget": None,
    "consumed": 15,
    "token_usage": {
        "uncached_input_tokens": 10,
        "output_tokens": 5,
        "cache_read_tokens": 0,
        "cache_write_tokens": 0,
    },
    "credits": None,
    "budget_exhausted": False,
    "session_count": 1,
    "model_request_count": 1,
    "usage_complete": True,
}))
""",
        encoding="utf-8",
    )
    driver = SubprocessEvolutionSessionDriver(
        CleanEnvironmentLauncher(Path("/usr/bin/env")),
        EvolutionProcessConfig(
            bundle_commit="0" * 40,
            bundle_tree="1" * 40,
            bundle_artifact_digest=digest("evolver-bundle"),
            command_argv=(str(Path(sys.executable).resolve()), str(script)),
            agent_backend="claude",
            isolated_home_environment_keys=("HOME", "SECONDARY_HOME"),
            session_trace_relative_path=None,
            token_usage_report_relative_path="scratch/token-usage.json",
            environment=(("ATREX_CAPTURE", str(capture)),),
            timeout_seconds=10,
            terminate_grace_seconds=1,
            max_diagnostic_bytes=4096,
        ),
    )

    result = await driver.run(prepared)

    value = json.loads(capture.read_text(encoding="utf-8"))
    assert result.returncode == 0
    assert value["instruction"] == "Run the versioned Evolver Bundle once."
    assert value["home"] == str(prepared.root / "scratch/agent-home")
    assert value["secondary_home"] == value["home"]
