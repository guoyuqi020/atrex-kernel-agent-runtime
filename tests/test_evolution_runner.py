"""Fixed Evolver workspace, process, and Challenger collection tests."""

from __future__ import annotations

import json
import os
import shutil
import sys
from dataclasses import dataclass, replace
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
    EvolutionInputManifestV10,
    EvolutionProcessConfig,
    EvolutionSessionResult,
    EvolutionWorkspaceAssembler,
    EvolverBundleRunner,
    PreparedEvolution,
    SubprocessEvolutionSessionDriver,
    _active_next_epoch_runtime_state_seed,
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


class ProcessExitThenSuccessfulEvolutionSession:
    def __init__(self, delegate: SubprocessEvolutionSessionDriver) -> None:
        self._delegate = delegate
        self.calls = 0
        self.workspaces: list[Path] = []

    async def run(self, prepared: PreparedEvolution) -> EvolutionSessionResult:
        self.calls += 1
        self.workspaces.append(prepared.root)
        result = await self._delegate.run(prepared)
        if self.calls == 1:
            return replace(
                result,
                returncode=1,
                stderr="API Error: Connection lost mid-response.",
            )
        return result


def _optimizer_repository(
    artifacts: LocalArtifactStore,
    root: Path,
) -> ArtifactDigest:
    source = root / "source-optimizer"
    prompt = source / "prompts/episode.md"
    design = source / "docs/design.md"
    prompt.parent.mkdir(parents=True)
    design.parent.mkdir(parents=True)
    (source / "src").mkdir()
    prompt.write_text("parent optimizer\n", encoding="utf-8")
    design.write_text("parent design\n", encoding="utf-8")
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
    (source / "bootstrap/report.json").write_text(
        json.dumps({"status": "baseline_ready"}), encoding="utf-8"
    )
    (source / "bootstrap/conversation.jsonl").write_text("{}\n", encoding="utf-8")
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
manifest = json.loads(os.environ["ATREX_EVOLUTION_INPUT_JSON"])
candidate = Path(os.environ["ATREX_EVOLUTION_CANDIDATE"])
workspace = Path(os.environ["ATREX_EVOLUTION_WORKSPACE"])
(workspace / "scratch/agent-home/sessions").mkdir(
    parents=True
)
trace = workspace
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
(candidate / "source/prompts/episode.md").write_text("challenger optimizer\\n")
Path(os.environ["ATREX_EVOLUTION_OUTPUT"]).write_text(json.dumps({
    "proposal_type": "evolved",
    "kernel_agent_revision_id": manifest["parent_revision_id"],
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
    "proposal_type": "reuse",
    "kernel_agent_revision_id": """
        + json.dumps(revision_id)
        + """,
    "hypothesis": "Retry a historical design unchanged.",
    "expected_effect": "Reproduce its previously useful search behavior.",
    "changed_paths": [],
    "unimplemented_capabilities": []
}))
""",
        encoding="utf-8",
    )
    return script


def _runtime_state_agent_script(tmp_path: Path) -> Path:
    script = tmp_path / "evolve-runtime-state.py"
    script.write_text(
        """import json
import os
import sys
from pathlib import Path

assert sys.stdin.read() == "Run the versioned Evolver Bundle once."
manifest = json.loads(os.environ["ATREX_EVOLUTION_INPUT_JSON"])
candidate = Path(os.environ["ATREX_EVOLUTION_CANDIDATE"])
state = candidate / "runtime-state"
(state / "skills/search.md").write_text("prefer measured bottlenecks\\n")
(state / "tools/README.md").write_text("# Candidate tools\\n")
Path(os.environ["ATREX_TOKEN_USAGE_REPORT"]).write_text(json.dumps({
    "schema_version": 2,
    "usage_unit": "provider_tokens",
    "budget": None,
    "consumed": 20,
    "token_usage": {
        "uncached_input_tokens": 10,
        "output_tokens": 10,
        "cache_read_tokens": 0,
        "cache_write_tokens": 0
    },
    "credits": None,
    "budget_exhausted": False,
    "session_count": 1,
    "model_request_count": 1,
    "usage_complete": True
}))
Path(os.environ["ATREX_EVOLUTION_OUTPUT"]).write_text(json.dumps({
    "proposal_type": "evolved",
    "kernel_agent_revision_id": manifest["parent_revision_id"],
    "hypothesis": "Seed a reusable measured-bottleneck procedure.",
    "expected_effect": "Spend fewer Attempts on unsupported directions.",
    "changed_paths": [],
    "unimplemented_capabilities": []
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
import shutil
import sys
from pathlib import Path

assert sys.stdin.read() == "Run the versioned Evolver Bundle once."
workspace = Path(os.environ["ATREX_EVOLUTION_WORKSPACE"])
candidate = Path(os.environ["ATREX_EVOLUTION_CANDIDATE"])
shutil.rmtree(candidate / "source")
shutil.copytree(workspace / "input/historical/agent-v1/source", candidate / "source")
for path in [candidate, *candidate.rglob("*")]:
    path.chmod(0o700 if path.is_dir() else 0o600)
(candidate / "source/prompts/episode.md").write_text("history repaired optimizer\\n")
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
    "proposal_type": "evolve_from_history",
    "kernel_agent_revision_id": """
        + json.dumps(revision_id)
        + """,
    "hypothesis": "Repair one weak step in the historical design.",
    "expected_effect": "Retain its prior strengths with a narrower policy.",
    "changed_paths": ["prompts/episode.md"],
    "unimplemented_capabilities": []
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
    manifest = EvolutionInputManifestV10.model_validate_json(prepared.manifest_path.read_bytes())

    assert manifest.parent_revision_id == request.parent_revision.id
    assert len(manifest.visible_agents) == 1
    assert manifest.visible_agents[0].relationship == "active"
    assert manifest.visible_agents[0].challenger_ordinal is None
    assert manifest.visible_agents[0].created_by == "bootstrap"
    assert manifest.schema_version == 10
    assert prepared.manifest_path == prepared.control_root / ".runtime/evolution-input.json"
    assert not (prepared.root / ".runtime").exists()
    assert list((prepared.root / "input/evolution-reports").iterdir()) == []
    assert not (os.stat(prepared.root / "input/evolution-reports").st_mode & 0o200)
    assert prepared.model == "evolver-model"
    parent_prompt = prepared.root / "input/agents/active/source/prompts/episode.md"
    candidate_prompt = prepared.candidate_root / "source/prompts/episode.md"
    assert parent_prompt.is_file()
    assert candidate_prompt.is_file()
    assert os.stat(parent_prompt).st_mode & 0o200 == 0
    assert os.stat(candidate_prompt).st_mode & 0o200
    assert (prepared.candidate_root / "runtime-state/skills").is_dir()
    assert (prepared.candidate_root / "runtime-state/tools/README.md").is_file()
    assert not (prepared.candidate_root / "runtime-state/trajectories").exists()
    assert not (prepared.root / "input/parent").exists()
    assert not (prepared.root / "input/reusable-agents").exists()
    assert not (prepared.root / "runtime-tools").exists()
    assert not (prepared.root / "scratch/candidate-base.json").exists()


def test_evolution_workspace_copies_active_revision_runtime_state_seed(tmp_path: Path) -> None:
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    initial = _request(artifacts, tmp_path)
    state = tmp_path / "active-state"
    (state / "skills").mkdir(parents=True)
    (state / "tools").mkdir()
    (state / "skills/active.md").write_text("active reusable procedure\n")
    (state / "tools/README.md").write_text("# Active tools\n")
    state_digest = artifacts.put_directory(
        state,
        ArtifactKind.KERNEL_AGENT_RUNTIME_STATE,
    )
    trace_digest = artifacts.put_json(
        {
            "schema_version": 9,
            "candidate": {
                "optimizer_digest": initial.parent_revision.optimizer_digest,
                "runtime_state_digest": state_digest,
            },
        },
        ArtifactKind.EVOLUTION,
    )
    active = KernelAgentRevision(
        id=new_kernel_agent_revision_id(),
        parent_id=initial.parent_revision.id,
        creation_key="epoch:previous:challenger:1",
        dsl=initial.parent_revision.dsl,
        optimizer_digest=initial.parent_revision.optimizer_digest,
        created_by="evolver",
        created_at=NOW,
        evolution_trace_digest=trace_digest,
    )
    request = BuildChallengerRequest(
        parent_revision=active,
        epoch_id=initial.epoch_id,
        evidence_checkpoint=initial.evidence_checkpoint,
        idempotency_key=initial.idempotency_key,
        model=initial.model,
    )

    prepared = EvolutionWorkspaceAssembler(tmp_path / "evolutions", artifacts).prepare(request)

    assert (prepared.candidate_root / "runtime-state/skills/active.md").read_text() == (
        "active reusable procedure\n"
    )
    assert (prepared.candidate_root / "runtime-state/tools/README.md").read_text() == (
        "# Active tools\n"
    )
    assert (
        prepared.candidate_runtime_state_base_root / "skills/active.md"
    ).read_text() == "active reusable procedure\n"


def test_evolver_prefers_winning_trajectory_terminal_state(tmp_path: Path) -> None:
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    revision_id = new_kernel_agent_revision_id()

    def state_digest(label: str) -> ArtifactDigest:
        root = tmp_path / label
        (root / "skills").mkdir(parents=True)
        (root / "tools").mkdir()
        (root / "skills/state.md").write_text(f"{label}\n")
        (root / "tools/README.md").write_text("# Tools\n")
        return artifacts.put_directory(root, ArtifactKind.KERNEL_AGENT_RUNTIME_STATE)

    trajectory_one = state_digest("trajectory-one-final")
    trajectory_one_start = state_digest("trajectory-one-start")
    trajectory_two_start = state_digest("trajectory-two-start")
    trajectory_two_best = state_digest("trajectory-two-best")
    trajectory_two_final = state_digest("trajectory-two-final")
    evidence = tmp_path / "evidence"
    (evidence / "epochs").mkdir(parents=True)
    (evidence / "epochs/00000001.json").write_text(
        json.dumps(
            {
                "winner_kernel_agent_revision_id": revision_id,
                "active_kernel_agent_revision_id": revision_id,
                "challenger_kernel_agent_revision_ids": [],
                "best_kernel": {"produced_by_attempt_id": "attempt-best"},
                "attempts": [
                    {
                        "attempt_id": "attempt-one",
                        "branch": "active",
                        "challenger_ordinal": 0,
                        "trajectory_ordinal": 1,
                        "ordinal": 1,
                        "kernel_agent_revision_id": revision_id,
                        "input_runtime_state_digest": trajectory_one_start,
                        "runtime_state_digest": trajectory_one,
                        "accepted_as_branch_best": False,
                        "output": None,
                    },
                    {
                        "attempt_id": "attempt-best",
                        "branch": "active",
                        "challenger_ordinal": 0,
                        "trajectory_ordinal": 2,
                        "ordinal": 1,
                        "kernel_agent_revision_id": revision_id,
                        "input_runtime_state_digest": trajectory_two_start,
                        "runtime_state_digest": trajectory_two_best,
                        "accepted_as_branch_best": True,
                        "output": {"latency_us": 10.0},
                    },
                    {
                        "attempt_id": "attempt-two-final",
                        "branch": "active",
                        "challenger_ordinal": 0,
                        "trajectory_ordinal": 2,
                        "ordinal": 2,
                        "kernel_agent_revision_id": revision_id,
                        "input_runtime_state_digest": trajectory_two_best,
                        "runtime_state_digest": trajectory_two_final,
                        "accepted_as_branch_best": False,
                        "output": None,
                    },
                ],
            }
        )
    )

    selected = _active_next_epoch_runtime_state_seed(evidence, revision_id, artifacts)

    assert selected is not None
    assert (selected / "skills/state.md").read_text() == "trajectory-two-final\n"


def test_evolution_workspace_exposes_active_and_challenger_runtime_state(
    tmp_path: Path,
) -> None:
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    initial = _request(artifacts, tmp_path)
    active = initial.parent_revision
    challenger = KernelAgentRevision(
        id=new_kernel_agent_revision_id(),
        parent_id=active.id,
        creation_key=f"epoch:{initial.epoch_id}:challenger:1",
        dsl=active.dsl,
        optimizer_digest=active.optimizer_digest,
        created_by="evolver",
        created_at=NOW,
        evolution_trace_digest=digest("challenger-evolution"),
    )
    campaign_id = new_campaign_id()
    lineage_id = new_lineage_id()
    catalog = (
        KernelAgentCatalogEntry(
            revision=active,
            revision_number=0,
            parent_revision_number=None,
            campaign_id=campaign_id,
            lineage_id=lineage_id,
            introduced_epoch_id=None,
            introduced_epoch_number=None,
            disposition="baseline",
            active=True,
        ),
        KernelAgentCatalogEntry(
            revision=challenger,
            revision_number=1,
            parent_revision_number=0,
            campaign_id=campaign_id,
            lineage_id=lineage_id,
            introduced_epoch_id=initial.epoch_id,
            introduced_epoch_number=1,
            disposition="challenger",
            active=False,
        ),
    )
    attempt_workspaces = tmp_path / "attempt-workspaces"
    active_state = (
        attempt_workspaces / ".reusable" / str(lineage_id) / str(active.id) / "trajectory-00000001"
    )
    challenger_state = (
        attempt_workspaces
        / ".reusable"
        / str(lineage_id)
        / str(challenger.id)
        / "trajectory-00000002"
    )
    for state, label in ((active_state, "active"), (challenger_state, "challenger")):
        (state / "skills").mkdir(parents=True)
        (state / "tools").mkdir()
        (state / "skills/lesson.md").write_text(f"{label} skill\n")
        (state / "tools/helper.py").write_text(f"print('{label}')\n")
        (state / "tools/README.md").write_text(f"# {label} tools\n")
    request = BuildChallengerRequest(
        parent_revision=active,
        epoch_id=initial.epoch_id,
        evidence_checkpoint=initial.evidence_checkpoint,
        idempotency_key=initial.idempotency_key,
        agent_catalog=catalog,
        model=initial.model,
    )

    prepared = EvolutionWorkspaceAssembler(
        tmp_path / "evolutions",
        artifacts,
        attempt_workspaces_root=attempt_workspaces,
    ).prepare(request)

    assert (prepared.root / "input/agents/active/source/prompts/episode.md").is_file()
    assert (prepared.root / "input/agents/challenger-0001/source/prompts/episode.md").is_file()
    assert not any((prepared.root / "input/historical").iterdir())
    reusable = prepared.root / "input/agents"
    assert (
        reusable / "active/runtime-state" / "trajectories/trajectory-00000001/skills/lesson.md"
    ).read_text() == "active skill\n"
    assert (
        reusable
        / "challenger-0001/runtime-state"
        / "trajectories/trajectory-00000002/tools/helper.py"
    ).read_text() == "print('challenger')\n"
    assert not (os.stat(reusable / "active").st_mode & 0o200)
    agent_catalog = EvolutionInputManifestV10.model_validate_json(
        prepared.manifest_path.read_bytes()
    ).visible_agents
    reusable_by_id = {
        item.revision_id: (
            item.runtime_state_path,
            list((prepared.root / item.runtime_state_path / "trajectories").iterdir()),
        )
        for item in agent_catalog
    }
    assert reusable_by_id[active.id][0] == "input/agents/active/runtime-state"
    assert [path.name for path in reusable_by_id[active.id][1]] == ["trajectory-00000001"]
    assert reusable_by_id[challenger.id][0] == "input/agents/challenger-0001/runtime-state"
    assert [path.name for path in reusable_by_id[challenger.id][1]] == ["trajectory-00000002"]


def test_evolution_workspace_separates_historical_agent_source_and_effect(
    tmp_path: Path,
) -> None:
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    initial = _request(artifacts, tmp_path)
    historical = KernelAgentRevision(
        id=new_kernel_agent_revision_id(),
        parent_id=initial.parent_revision.id,
        creation_key="epoch:old:challenger:1",
        dsl=initial.parent_revision.dsl,
        optimizer_digest=initial.parent_revision.optimizer_digest,
        created_by="evolver",
        created_at=NOW,
        evolution_trace_digest=digest("historical-evolution"),
    )
    request = BuildChallengerRequest(
        parent_revision=initial.parent_revision,
        epoch_id=initial.epoch_id,
        evidence_checkpoint=initial.evidence_checkpoint,
        idempotency_key=initial.idempotency_key,
        agent_catalog=_historical_catalog(historical),
    )

    prepared = EvolutionWorkspaceAssembler(tmp_path / "evolutions", artifacts).prepare(request)

    historical_root = prepared.root / "input/historical/agent-v1"
    assert (historical_root / "source/prompts/episode.md").is_file()
    assert json.loads((historical_root / "optimization-summary.json").read_text()) == {
        "career": {
            "epoch_participation_count": 0,
            "loss_count": 0,
            "win_count": 0,
        },
        "kernel_agent_revision_id": historical.id,
        "latest_epoch": None,
    }
    assert (historical_root / "runtime-state/trajectories").is_dir()
    assert not (prepared.root / f"input/agents/{historical.id}").exists()
    assert not (prepared.root / "runtime-tools").exists()


def test_changed_paths_ignore_generated_candidate_cache_files(tmp_path: Path) -> None:
    parent = tmp_path / "parent"
    candidate = tmp_path / "candidate"
    (parent / "src").mkdir(parents=True)
    (parent / "src/main.py").write_text("before\n", encoding="utf-8")
    shutil.copytree(parent, candidate)
    (candidate / "src/main.py").write_text("after\n", encoding="utf-8")
    (candidate / "src/__pycache__").mkdir()
    (candidate / "src/__pycache__/main.cpython-314.pyc").write_bytes(b"generated")
    (candidate / ".ruff_cache").mkdir()
    (candidate / ".ruff_cache/state").write_text("generated", encoding="utf-8")

    assert EvolverBundleRunner._changed_paths(parent, candidate) == {"src/main.py"}


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
    assert (optimizer / "docs/design.md").read_text() == "parent design\n"
    trace_artifact = artifacts.verify(build.evolution_trace_digest)
    assert trace_artifact.kind is ArtifactKind.EVOLUTION
    trace = json.loads((trace_artifact.payload_path / "value.json").read_text())
    assert trace["agent"]["bundle_commit"] == "0" * 40
    assert trace["agent"]["bundle_tree"] == "1" * 40
    assert trace["agent"]["bundle_artifact_digest"] == evolver_digest
    assert trace["agent"]["model"] == "evolver-model"
    assert trace["schema_version"] == 9
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
    assert (
        events.records[-1][2]["unimplemented_capabilities"]
        == trace["output"]["unimplemented_capabilities"]
    )
    assert trace["candidate"]["optimizer_digest"] == candidate.optimizer_digest
    assert trace["candidate"]["runtime_state_digest"] is not None
    assert candidate.runtime_state_digest is not None
    report_test_state = tmp_path / "report-test-state"
    (report_test_state / "skills").mkdir(parents=True)
    (report_test_state / "tools").mkdir()
    (report_test_state / "skills/seed.md").write_text("# Seed\n")
    (report_test_state / "tools/README.md").write_text("# Tools\n")
    report_test_state_digest = artifacts.put_directory(
        report_test_state,
        ArtifactKind.KERNEL_AGENT_RUNTIME_STATE,
    )
    evolved_revision = KernelAgentRevision(
        id=new_kernel_agent_revision_id(),
        parent_id=request.parent_revision.id,
        creation_key="epoch:next:challenger:1",
        dsl=request.parent_revision.dsl,
        optimizer_digest=candidate.optimizer_digest,
        runtime_state_digest=report_test_state_digest,
        created_by="evolver",
        created_at=NOW,
        evolution_trace_digest=build.evolution_trace_digest,
    )
    next_workspace = EvolutionWorkspaceAssembler(
        tmp_path / "next-evolutions",
        artifacts,
    ).prepare(
        BuildChallengerRequest(
            parent_revision=evolved_revision,
            epoch_id=new_epoch_id(),
            evidence_checkpoint=request.evidence_checkpoint,
            idempotency_key="epoch:next:challenger",
            model="evolver-model",
            agent_catalog=(
                KernelAgentCatalogEntry(
                    revision=request.parent_revision,
                    revision_number=0,
                    parent_revision_number=None,
                    campaign_id=new_campaign_id(),
                    lineage_id=new_lineage_id(),
                    introduced_epoch_id=None,
                    introduced_epoch_number=None,
                    disposition="rejected",
                    active=False,
                ),
            ),
        )
    )
    previous_report = json.loads(
        (next_workspace.root / "input/evolution-reports/evo-1.json").read_text()
    )
    assert previous_report["evolution_number"] == 1
    expected_projected_report = dict(trace["output"])
    expected_projected_report.pop("kernel_agent_revision_id")
    expected_projected_report.pop("changed_paths")
    assert previous_report["report"] == expected_projected_report
    assert "kernel_agent_revision_id" not in json.dumps(previous_report)
    assert "changed_paths" not in json.dumps(previous_report)
    assert previous_report["parent"] == {
        "runtime_state_path": "input/historical/agent-v0/runtime-state",
        "source_path": "input/historical/agent-v0/source",
    }
    assert previous_report["generated_agent"] == {
        "runtime_state_path": "input/agents/active/runtime-state",
        "source_path": "input/agents/active/source",
    }
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
async def test_fixed_runner_accepts_runtime_state_only_candidate(tmp_path: Path) -> None:
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
                str(_runtime_state_agent_script(tmp_path)),
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
    assert build.proposal.candidate.optimizer_digest == request.parent_revision.optimizer_digest
    state_digest = build.proposal.candidate.runtime_state_digest
    assert state_digest is not None
    state = artifacts.verify(state_digest)
    assert state.kind is ArtifactKind.KERNEL_AGENT_RUNTIME_STATE
    assert (state.payload_path / "skills/search.md").read_text() == (
        "prefer measured bottlenecks\n"
    )


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
    assert (candidate / "docs/design.md").read_text() == "parent design\n"


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
import subprocess
import sys
from pathlib import Path

assert not any(key.startswith("ATREX_") for key in os.environ)
candidate = Path("candidate/source")
(candidate / "prompts/integration.md").write_text("real Evolver Bundle ran\\n")
Path("scratch/evolution-report-draft.json").write_text(json.dumps({
    "proposal_type": "evolved",
    "kernel_agent_revision_id": "REPLACE_PARENT_REVISION",
    "hypothesis": "The current Evolver entrypoint accepts the Runtime protocol.",
    "expected_effect": "Produce one valid complete Challenger Bundle.",
    "changed_paths": ["prompts/integration.md"],
    "unimplemented_capabilities": []
}))
published = subprocess.run(
    [
        sys.executable,
        "input/evolver/src/runtime_tools.py",
        "evolution-report",
        "--request",
        "scratch/evolution-report-draft.json",
    ],
    check=False,
    capture_output=True,
    text=True,
)
assert published.returncode == 0, published.stdout + published.stderr
assert json.loads(published.stdout)["status"] == "published"
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
    fake_claude.write_text(
        fake_claude.read_text().replace(
            "REPLACE_PARENT_REVISION",
            str(request.parent_revision.id),
        ),
        encoding="utf-8",
    )
    fake_claude.chmod(0o700)
    bundle = Path(__file__).resolve().parents[1] / "src/atrex-kernel-agent-evolver"
    evolver_bundle_digest = artifacts.put_directory(bundle, ArtifactKind.EVOLVER_BUNDLE)
    sessions = SubprocessEvolutionSessionDriver(
        CleanEnvironmentLauncher(Path("/usr/bin/env")),
        EvolutionProcessConfig(
            bundle_commit="0" * 40,
            bundle_tree="1" * 40,
            bundle_artifact_digest=evolver_bundle_digest,
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
        EvolutionWorkspaceAssembler(
            tmp_path / "evolutions",
            artifacts,
            evolver_bundle_digest=evolver_bundle_digest,
        ),
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
async def test_evolver_automatically_retries_process_exit_in_fresh_workspace(
    tmp_path: Path,
) -> None:
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    request = _request(artifacts, tmp_path)
    evolver_digest = _evolver_bundle(artifacts, tmp_path, _agent_script(tmp_path))
    delegate = SubprocessEvolutionSessionDriver(
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
    sessions = ProcessExitThenSuccessfulEvolutionSession(delegate)
    events = FakeRuntimeEventRecorder([])
    registry = SqliteRegistry(tmp_path / "registry.sqlite", clock=lambda: NOW)
    runner = EvolverBundleRunner(
        EvolutionWorkspaceAssembler(
            tmp_path / "evolutions",
            artifacts,
            evolver_bundle_digest=evolver_digest,
        ),
        sessions,  # type: ignore[arg-type]
        artifacts,
        events,
        kernel_agent_limits=kernel_agent_limits(),
        max_output_manifest_bytes=8192,
        worker_sessions=registry,
        backend="claude",
        max_infrastructure_retries=1,
    )

    build = await runner.build_challenger(request)

    assert isinstance(build.proposal, KernelAgentCandidateProposal)
    assert sessions.calls == 2
    assert len(set(sessions.workspaces)) == 2
    worker_sessions = registry.list_worker_sessions(epoch_id=request.epoch_id)
    assert [session.status for session in worker_sessions] == [
        WorkerSessionStatus.FAILED,
        WorkerSessionStatus.COMPLETED,
    ]
    assert [session.finish_reason for session in worker_sessions] == [
        "process-exit-1",
        "completed",
    ]
    assert [kind for kind, _aggregate, _payload in events.records] == [
        "worker.started",
        "worker.infrastructure_failed",
        "worker.cleaned",
        "evolution.retrying",
        "worker.started",
        "worker.exited",
        "worker.cleaned",
        "evolution.proposal_sealed",
    ]
    registry.close()


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
                str(_agent_script(tmp_path, declared_path="docs/design.md")),
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

    with pytest.raises(ValueError, match="disagrees with sealed Agent Source"):
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

    with pytest.raises(InfrastructureError, match=r"^Evolver timed out$"):
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
async def test_evolver_timeout_exit_precedes_partial_usage_validation(tmp_path: Path) -> None:
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    prepared = EvolutionWorkspaceAssembler(tmp_path / "evolutions", artifacts).prepare(
        _request(artifacts, tmp_path)
    )
    script = tmp_path / "provider-timeout.py"
    script.write_text(
        """import json
import os
from pathlib import Path

Path(os.environ["ATREX_TOKEN_USAGE_REPORT"]).write_text(json.dumps({
    "schema_version": 2,
    "usage_unit": "provider_tokens",
    "budget": None,
    "consumed": 12,
    "token_usage": {
        "uncached_input_tokens": 10,
        "output_tokens": 2,
        "cache_read_tokens": 0,
        "cache_write_tokens": 0,
    },
    "credits": None,
    "budget_exhausted": False,
    "session_count": 1,
    "model_request_count": 1,
    "usage_complete": False,
}))
raise SystemExit(124)
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
            isolated_home_environment_keys=(),
            session_trace_relative_path=None,
            token_usage_report_relative_path="scratch/token-usage.json",
            environment=(),
            timeout_seconds=10,
            terminate_grace_seconds=1,
            max_diagnostic_bytes=4096,
        ),
    )

    with pytest.raises(InfrastructureError, match=r"^Evolver timed out$"):
        await driver.run(prepared)

    usage = json.loads((prepared.root / "scratch/token-usage.json").read_text())
    assert usage["usage_complete"] is False
    assert usage["consumed"] == 12


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


def test_prepare_launch_binds_the_runtime_session_timeout(tmp_path: Path) -> None:
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    prepared = EvolutionWorkspaceAssembler(tmp_path / "evolutions", artifacts).prepare(
        _request(artifacts, tmp_path)
    )
    driver = SubprocessEvolutionSessionDriver(
        CleanEnvironmentLauncher(Path("/usr/bin/env")),
        EvolutionProcessConfig(
            bundle_commit="0" * 40,
            bundle_tree="1" * 40,
            bundle_artifact_digest=digest("evolver-bundle"),
            command_argv=(str(Path(sys.executable).resolve()),),
            agent_backend="claude",
            isolated_home_environment_keys=("HOME",),
            session_trace_relative_path=None,
            token_usage_report_relative_path="scratch/token-usage.json",
            environment=(),
            timeout_seconds=10_800,
            terminate_grace_seconds=1,
            max_diagnostic_bytes=4096,
        ),
    )

    launch = driver.prepare_launch(prepared)

    assert launch.environment["ATREX_SESSION_TIMEOUT_SECONDS"] == "10800"
    assert "ATREX_EVOLVER_QUERY_URL" not in launch.environment
    assert "ATREX_EVOLVER_QUERY_CAPABILITY" not in launch.environment
    assert launch.environment["ATREX_EVOLUTION_INPUT_JSON"].startswith("{")
    assert launch.environment["ATREX_EVIDENCE_PROMPT"].startswith("# Evidence input")
    assert not (prepared.root / ".runtime").exists()


def test_evolution_config_rejects_operator_override_of_the_session_timeout(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="Runtime-owned keys"):
        EvolutionProcessConfig(
            bundle_commit="0" * 40,
            bundle_tree="1" * 40,
            bundle_artifact_digest=digest("evolver-bundle"),
            command_argv=(str(Path(sys.executable).resolve()),),
            agent_backend="claude",
            isolated_home_environment_keys=("HOME",),
            session_trace_relative_path=None,
            token_usage_report_relative_path="scratch/token-usage.json",
            environment=(("ATREX_SESSION_TIMEOUT_SECONDS", "60"),),
            timeout_seconds=10_800,
            terminate_grace_seconds=1,
            max_diagnostic_bytes=4096,
        )
