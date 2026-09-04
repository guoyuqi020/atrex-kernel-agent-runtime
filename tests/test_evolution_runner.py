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
    BuildChallengerResult,
    KernelAgentCandidateProposal,
    KernelAgentReuseProposal,
)
from atrex_runtime.registry.sqlite import SqliteRegistry
from atrex_runtime.workers.evolution import (
    EvolutionInputManifestV11,
    EvolutionOutput,
    EvolutionProcessConfig,
    EvolutionSessionResult,
    EvolutionWorkspaceAssembler,
    EvolverBundleRunner,
    PreparedEvolution,
    SubprocessEvolutionSessionDriver,
    _active_next_epoch_runtime_state_seed,
    _upgrade_historical_output,
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


def test_legacy_contribution_ids_are_migrated_only_when_reading_history() -> None:
    from atrex_runtime.workers.evolution import EvolutionOutput

    revision = new_kernel_agent_revision_id()
    old = {
        "proposal_type": "evolved",
        "kernel_agent_revision_id": new_kernel_agent_revision_id(),
        "hypothesis": "historical intent",
        "expected_effect": "historical effect",
        "changed_paths": ["prompts/episode.md"],
        "contributing_revision_ids": [revision],
        "unimplemented_capabilities": [],
    }
    raw = {
        "output": old,
        "input": {
            "visible_agents": [{"revision_id": revision, "path": "input/agents/agent-v1/source"}]
        },
    }
    converted = EvolutionOutput.model_validate(_upgrade_historical_output(raw))
    assert converted.contributing_paths == ("input/agents/agent-v1",)
    assert old["contributing_revision_ids"] == [revision]
    with pytest.raises(ValueError, match="contributing_revision_ids"):
        EvolutionOutput.model_validate(old)


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


def _completed_epoch_evidence(
    artifacts: LocalArtifactStore,
    tmp_path: Path,
    *,
    active_id: str,
    challenger_id: str,
    winner_id: str,
) -> ArtifactDigest:
    source = tmp_path / f"evidence-epoch-{winner_id}"
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
                "through_epoch": 1,
                "previous_checkpoint_digest": None,
            }
        ),
        encoding="utf-8",
    )
    (source / "epochs").mkdir()
    (source / "epochs/00000001.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "number": 1,
                "active_kernel_agent_revision_id": active_id,
                "challenger_kernel_agent_revision_ids": [challenger_id],
                "winner_kernel_agent_revision_id": winner_id,
                "starting_kernel_revision_id": None,
                "starting_kernel": None,
                "best_kernel_revision_id": None,
                "best_kernel": None,
                "attempts": [
                    {
                        "attempt_id": f"attempt_{branch}",
                        "branch": branch,
                        "challenger_ordinal": 0 if branch == "active" else 1,
                        "trajectory_ordinal": 1,
                        "ordinal": 1,
                        "kernel_agent_revision_id": (
                            active_id if branch == "active" else challenger_id
                        ),
                        "input_kernel_revision_id": None,
                        "accepted_as_branch_best": False,
                        "output": None,
                    }
                    for branch in ("active", "challenger")
                ],
            }
        ),
        encoding="utf-8",
    )
    return artifacts.put_directory(source, ArtifactKind.EVIDENCE)


def _agent_script(
    tmp_path: Path,
    *,
    declared_path: str = "prompts/episode.md",
    contributing: tuple[str, ...] = (),
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
(candidate / "prompts/episode.md").write_text("challenger optimizer\\n")
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
    "contributing_paths": """
        + json.dumps(sorted(contributing))
        + """,
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
    "contributing_paths": [],
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
state = candidate
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
    "changed_paths": ["skills/search.md", "tools/README.md"],
    "contributing_paths": [],
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
shutil.rmtree(candidate)
shutil.copytree(workspace / "input/agents/agent-v1", candidate)
for path in [candidate, *candidate.rglob("*")]:
    path.chmod(0o700 if path.is_dir() else 0o600)
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
    "proposal_type": "evolve_from_history",
    "kernel_agent_revision_id": """
        + json.dumps(revision_id)
        + """,
    "hypothesis": "Repair one weak step in the historical design.",
    "expected_effect": "Retain its prior strengths with a narrower policy.",
    "changed_paths": ["prompts/episode.md"],
    "contributing_paths": [],
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
    parent = _parent(artifacts, tmp_path)
    return BuildChallengerRequest(
        parent_revision=parent,
        epoch_id=new_epoch_id(),
        evidence_checkpoint=_evidence(artifacts, tmp_path),
        idempotency_key="epoch:test:challenger",
        agent_catalog=(_baseline_catalog_entry(parent),),
        model="evolver-model",
    )


def _baseline_catalog_entry(revision: KernelAgentRevision) -> KernelAgentCatalogEntry:
    rooted = revision.parent_id is None
    return KernelAgentCatalogEntry(
        revision=revision,
        revision_number=0 if rooted else 1,
        parent_revision_number=None if rooted else 0,
        campaign_id=new_campaign_id(),
        lineage_id=new_lineage_id(),
        introduced_epoch_id=None,
        introduced_epoch_number=None,
        disposition="baseline",
        active=True,
    )


def _historical_catalog(
    parent: KernelAgentRevision,
    revision: KernelAgentRevision,
) -> tuple[KernelAgentCatalogEntry, ...]:
    campaign_id = new_campaign_id()
    lineage_id = new_lineage_id()
    return (
        KernelAgentCatalogEntry(
            revision=parent,
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
            revision=revision,
            revision_number=1,
            parent_revision_number=0,
            campaign_id=campaign_id,
            lineage_id=lineage_id,
            introduced_epoch_id=None,
            introduced_epoch_number=None,
            disposition="rejected",
            active=False,
        ),
    )


def test_unified_bundle_replaces_packaged_resources_without_resurrecting_files(
    tmp_path: Path,
) -> None:
    from atrex_runtime.workers.workspace import REUSABLE_AGENT_DIRECTORIES

    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    initial = _request(artifacts, tmp_path)
    source = tmp_path / "source-optimizer"
    state = tmp_path / "learned-state"
    for name in REUSABLE_AGENT_DIRECTORIES:
        (source / name).mkdir(exist_ok=True)
        (source / name / "README.md").write_text("packaged index")
        (source / name / "removed.md").write_text("do not resurrect")
        (state / name).mkdir(parents=True)
        (state / name / "README.md").write_text("learned index")
        (state / name / "learned.md").write_text("learned content")
    revision = replace(
        initial.parent_revision,
        optimizer_digest=artifacts.put_directory(source, ArtifactKind.KERNEL_AGENT),
        runtime_state_digest=artifacts.put_directory(
            state, ArtifactKind.KERNEL_AGENT_RUNTIME_STATE
        ),
    )
    request = replace(
        initial, parent_revision=revision, agent_catalog=(_baseline_catalog_entry(revision),)
    )
    prepared = EvolutionWorkspaceAssembler(tmp_path / "evolutions", artifacts).prepare(request)
    parent = prepared.root / "input/agents/agent-v0"
    for bundle in (parent, prepared.candidate_root):
        assert (bundle / "docs/design.md").read_text() == "parent design\n"
        for name in REUSABLE_AGENT_DIRECTORIES:
            assert (bundle / name / "README.md").read_text() == "learned index"
            assert (bundle / name / "learned.md").read_text() == "learned content"
            assert not (bundle / name / "removed.md").exists()
    assert not (parent.stat().st_mode & 0o200)
    assert prepared.candidate_root.stat().st_mode & 0o200
    assert (
        artifacts.verify(revision.optimizer_digest).payload_path / "memory/removed.md"
    ).is_file()


def test_evolution_workspace_copies_full_parent_to_writable_candidate(tmp_path: Path) -> None:
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    request = _request(artifacts, tmp_path)

    prepared = EvolutionWorkspaceAssembler(tmp_path / "evolutions", artifacts).prepare(request)
    assert not (prepared.candidate_root / "source").exists()
    assert not (prepared.candidate_root / "runtime-state").exists()
    assert (prepared.candidate_root / "docs/design.md").read_text() == "parent design\n"
    manifest = EvolutionInputManifestV11.model_validate_json(prepared.manifest_path.read_bytes())

    assert manifest.parent_revision_id == request.parent_revision.id
    assert len(manifest.visible_agents) == 1
    assert manifest.visible_agents[0].relationship == "active"
    assert manifest.visible_agents[0].challenger_ordinal is None
    assert manifest.visible_agents[0].created_by == "bootstrap"
    assert manifest.schema_version == 11
    assert prepared.manifest_path == prepared.control_root / ".runtime/evolution-input.json"
    assert not (prepared.root / ".runtime").exists()
    assert list((prepared.root / "input/evolution-reports").iterdir()) == []
    assert not (os.stat(prepared.root / "input/evolution-reports").st_mode & 0o200)
    assert prepared.model == "evolver-model"
    parent_prompt = prepared.root / "input/agents/agent-v0/prompts/episode.md"
    candidate_prompt = prepared.candidate_root / "prompts/episode.md"
    assert parent_prompt.is_file()
    assert candidate_prompt.is_file()
    assert os.stat(parent_prompt).st_mode & 0o200 == 0
    assert os.stat(candidate_prompt).st_mode & 0o200
    assert (prepared.candidate_root / "skills").is_dir()
    assert (prepared.candidate_root / "tools/README.md").is_file()
    assert not (prepared.candidate_root / "trajectories").exists()
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
        agent_catalog=(_baseline_catalog_entry(active),),
        model=initial.model,
    )

    prepared = EvolutionWorkspaceAssembler(tmp_path / "evolutions", artifacts).prepare(request)

    assert (prepared.candidate_root / "skills/active.md").read_text() == (
        "active reusable procedure\n"
    )
    assert (prepared.candidate_root / "tools/README.md").read_text() == ("# Active tools\n")
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


def _pool_request(
    artifacts: LocalArtifactStore,
    tmp_path: Path,
    *,
    winner: str,
) -> tuple[BuildChallengerRequest, KernelAgentRevision, KernelAgentRevision]:
    """Build a request whose last completed Epoch compared two distinct revisions."""
    incumbent = _parent(artifacts, tmp_path)
    rival_source = tmp_path / "rival-source"
    (rival_source / "prompts").mkdir(parents=True)
    (rival_source / "src").mkdir()
    (rival_source / "prompts/episode.md").write_text("rival optimizer\n", encoding="utf-8")
    (rival_source / "src/main.py").write_text("def main(): ...\n", encoding="utf-8")
    (rival_source / "atrex-bundle.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "bundle_format": "atrex-kernel-agent-bundle-v1",
                "entrypoint": {"command": "src/main.py"},
            }
        ),
        encoding="utf-8",
    )
    rival = KernelAgentRevision(
        id=new_kernel_agent_revision_id(),
        parent_id=incumbent.id,
        creation_key="epoch:previous:challenger:1",
        dsl=incumbent.dsl,
        optimizer_digest=artifacts.put_directory(rival_source, ArtifactKind.KERNEL_AGENT),
        created_by="evolver",
        created_at=NOW,
        evolution_trace_digest=digest("rival-evolution"),
    )
    parent, other = (rival, incumbent) if winner == "challenger" else (incumbent, rival)
    campaign_id = new_campaign_id()
    lineage_id = new_lineage_id()
    catalog = tuple(
        KernelAgentCatalogEntry(
            revision=revision,
            revision_number=number,
            parent_revision_number=None if number == 0 else 0,
            campaign_id=campaign_id,
            lineage_id=lineage_id,
            introduced_epoch_id=None,
            introduced_epoch_number=None,
            disposition="baseline" if number == 0 else "challenger",
            active=revision.id == parent.id,
        )
        for revision, number in ((incumbent, 0), (rival, 1))
    )
    request = BuildChallengerRequest(
        parent_revision=parent,
        epoch_id=new_epoch_id(),
        evidence_checkpoint=_completed_epoch_evidence(
            artifacts,
            tmp_path,
            active_id=str(incumbent.id),
            challenger_id=str(rival.id),
            winner_id=str(parent.id),
        ),
        idempotency_key="epoch:test:challenger",
        agent_catalog=catalog,
        model="evolver-model",
    )
    return request, parent, other


def test_evolution_workspace_pools_the_last_completed_epoch_challenger_winner(
    tmp_path: Path,
) -> None:
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    request, parent, loser = _pool_request(artifacts, tmp_path, winner="challenger")

    prepared = EvolutionWorkspaceAssembler(tmp_path / "evolutions", artifacts).prepare(request)
    manifest = EvolutionInputManifestV11.model_validate_json(prepared.manifest_path.read_bytes())

    by_id = {item.revision_id: item for item in manifest.visible_agents}
    assert by_id[loser.id].relationship == "active"
    assert by_id[loser.id].challenger_ordinal is None
    assert by_id[loser.id].parent is False
    assert by_id[loser.id].version == "agent-v0"
    assert by_id[loser.id].path == "input/agents/agent-v0"
    assert by_id[parent.id].relationship == "challenger"
    assert by_id[parent.id].challenger_ordinal == 1
    assert by_id[parent.id].parent is True
    assert by_id[parent.id].version == "agent-v1"
    assert by_id[parent.id].path == "input/agents/agent-v1"
    assert by_id[parent.id].sessions_path == "input/evidence/agent-v1/sessions"
    assert by_id[parent.id].reports_path == "input/evidence/agent-v1/reports"
    assert sorted(child.name for child in (prepared.root / "input/agents").iterdir()) == [
        "agent-v0",
        "agent-v1",
    ]
    assert sorted(child.name for child in (prepared.root / "input/evidence").iterdir()) == [
        "agent-v0",
        "agent-v1",
    ]
    assert not (prepared.root / "input/current-epoch-challengers").exists()
    assert not (prepared.root / "input/historical").exists()
    assert (
        prepared.root / "input/agents/agent-v0/prompts/episode.md"
    ).read_text() == "parent optimizer\n"
    assert (
        prepared.root / "input/agents/agent-v1/prompts/episode.md"
    ).read_text() == "rival optimizer\n"
    assert (prepared.candidate_root / "prompts/episode.md").read_text() == "rival optimizer\n"
    for version in ("agent-v0", "agent-v1"):
        assert (prepared.root / f"input/evidence/{version}/sessions").is_dir()
        assert (prepared.root / f"input/evidence/{version}/reports").is_dir()


def test_evolution_workspace_pools_the_last_completed_epoch_active_winner(
    tmp_path: Path,
) -> None:
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    request, parent, loser = _pool_request(artifacts, tmp_path, winner="active")

    prepared = EvolutionWorkspaceAssembler(tmp_path / "evolutions", artifacts).prepare(request)
    manifest = EvolutionInputManifestV11.model_validate_json(prepared.manifest_path.read_bytes())

    by_id = {item.revision_id: item for item in manifest.visible_agents}
    assert by_id[parent.id].relationship == "active"
    assert by_id[parent.id].parent is True
    assert by_id[loser.id].relationship == "challenger"
    assert by_id[loser.id].challenger_ordinal == 1
    assert by_id[loser.id].parent is False
    assert (prepared.candidate_root / "prompts/episode.md").read_text() == "parent optimizer\n"


def test_evolution_workspace_keys_same_ordinal_challengers_by_distinct_versions(
    tmp_path: Path,
) -> None:
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    request, parent, loser = _pool_request(artifacts, tmp_path, winner="active")
    fresh = KernelAgentRevision(
        id=new_kernel_agent_revision_id(),
        parent_id=parent.id,
        creation_key=f"epoch:{request.epoch_id}:challenger:1",
        dsl=parent.dsl,
        optimizer_digest=parent.optimizer_digest,
        created_by="evolver",
        created_at=NOW,
        evolution_trace_digest=digest("fresh-evolution"),
    )
    existing = request.agent_catalog[0]
    request = BuildChallengerRequest(
        parent_revision=request.parent_revision,
        epoch_id=request.epoch_id,
        evidence_checkpoint=request.evidence_checkpoint,
        idempotency_key=request.idempotency_key,
        agent_catalog=(
            *request.agent_catalog,
            replace(
                existing,
                revision=fresh,
                revision_number=2,
                parent_revision_number=0,
                disposition="challenger",
                active=False,
            ),
        ),
        model=request.model,
    )

    prepared = EvolutionWorkspaceAssembler(tmp_path / "evolutions", artifacts).prepare(request)
    manifest = EvolutionInputManifestV11.model_validate_json(prepared.manifest_path.read_bytes())

    by_id = {item.revision_id: item for item in manifest.visible_agents}
    assert by_id[loser.id].relationship == "challenger"
    assert by_id[loser.id].challenger_ordinal == 1
    assert by_id[loser.id].version == "agent-v1"
    assert by_id[loser.id].path == "input/agents/agent-v1"
    assert by_id[fresh.id].relationship == "current_epoch_challenger"
    assert by_id[fresh.id].challenger_ordinal == 1
    assert by_id[fresh.id].version == "agent-v2"
    assert by_id[fresh.id].path == "input/agents/agent-v2"
    assert by_id[fresh.id].sessions_path is None
    assert by_id[fresh.id].reports_path is None
    assert (prepared.root / "input/agents/agent-v1/prompts/episode.md").is_file()
    assert (prepared.root / "input/agents/agent-v2/prompts/episode.md").is_file()
    assert sorted(child.name for child in (prepared.root / "input/evidence").iterdir()) == [
        "agent-v0",
        "agent-v1",
        "agent-v2",
    ]
    assert not (prepared.root / "input/evidence/agent-v2/sessions").exists()
    assert not (prepared.root / "input/evidence/agent-v2/reports").exists()
    assert (prepared.root / "input/evidence/agent-v2/optimization-summary.json").is_file()


def test_evolution_workspace_separates_current_epoch_challenger_from_the_pool(
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

    assert (prepared.root / "input/agents/agent-v0/prompts/episode.md").is_file()
    assert sorted(child.name for child in (prepared.root / "input/agents").iterdir()) == [
        "agent-v0",
        "agent-v1",
    ]
    current_epoch = prepared.root / "input/agents/agent-v1"
    assert (current_epoch / "prompts/episode.md").is_file()
    assert (prepared.root / "input/evidence/agent-v1/optimization-summary.json").is_file()
    assert not (prepared.root / "input/evidence/agent-v1/sessions").exists()
    assert not (prepared.root / "input/evidence/agent-v1/reports").exists()
    assert not (prepared.root / "input/historical").exists()
    reusable = prepared.root / "input/evidence"
    assert (
        reusable / "agent-v0/resources" / "trajectories/trajectory-00000001/skills/lesson.md"
    ).read_text() == "active skill\n"
    assert (
        prepared.root
        / "input/evidence/agent-v1/resources"
        / "trajectories/trajectory-00000002/tools/helper.py"
    ).read_text() == "print('challenger')\n"
    assert not (os.stat(reusable / "agent-v0").st_mode & 0o200)
    assert not (os.stat(current_epoch).st_mode & 0o200)
    agent_catalog = EvolutionInputManifestV11.model_validate_json(
        prepared.manifest_path.read_bytes()
    ).visible_agents
    relationship_by_id = {item.revision_id: item.relationship for item in agent_catalog}
    assert relationship_by_id[active.id] == "active"
    assert relationship_by_id[challenger.id] == "current_epoch_challenger"
    sessions_by_id = {item.revision_id: item.sessions_path for item in agent_catalog}
    assert sessions_by_id[active.id] == "input/evidence/agent-v0/sessions"
    assert sessions_by_id[challenger.id] is None
    reusable_by_id = {
        item.revision_id: (
            item.resources_path,
            list((prepared.root / item.resources_path / "trajectories").iterdir()),
        )
        for item in agent_catalog
    }
    assert reusable_by_id[active.id][0] == "input/evidence/agent-v0/resources"
    assert [path.name for path in reusable_by_id[active.id][1]] == ["trajectory-00000001"]
    assert reusable_by_id[challenger.id][0] == "input/evidence/agent-v1/resources"
    assert [path.name for path in reusable_by_id[challenger.id][1]] == ["trajectory-00000002"]


def test_evolution_workspace_gives_a_non_pool_version_source_state_and_summary(
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
        agent_catalog=_historical_catalog(initial.parent_revision, historical),
    )

    prepared = EvolutionWorkspaceAssembler(tmp_path / "evolutions", artifacts).prepare(request)

    agent_root = prepared.root / "input/agents/agent-v1"
    evidence_root = prepared.root / "input/evidence/agent-v1"
    assert (agent_root / "prompts/episode.md").is_file()
    assert (evidence_root / "resources/trajectories").is_dir()
    assert json.loads((evidence_root / "optimization-summary.json").read_text()) == {
        "career": {
            "epoch_participation_count": 0,
            "loss_count": 0,
            "win_count": 0,
        },
        "kernel_agent_revision_id": historical.id,
        "version": "agent-v1",
        "path": "input/agents/agent-v1",
        "resources_path": "input/evidence/agent-v1/resources",
        "latest_epoch": None,
    }
    assert not (evidence_root / "sessions").exists()
    assert not (evidence_root / "reports").exists()
    assert not (prepared.root / "input/historical").exists()
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
    campaign_id = new_campaign_id()
    lineage_id = new_lineage_id()
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
                    campaign_id=campaign_id,
                    lineage_id=lineage_id,
                    introduced_epoch_id=None,
                    introduced_epoch_number=None,
                    disposition="rejected",
                    active=False,
                ),
                KernelAgentCatalogEntry(
                    revision=evolved_revision,
                    revision_number=1,
                    parent_revision_number=0,
                    campaign_id=campaign_id,
                    lineage_id=lineage_id,
                    introduced_epoch_id=None,
                    introduced_epoch_number=None,
                    disposition="baseline",
                    active=True,
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
    expected_projected_report.pop("contributing_paths")
    expected_projected_report["contributing_paths"] = []
    assert previous_report["report"] == expected_projected_report
    assert previous_report["report"]["changed_paths"] == trace["output"]["changed_paths"]
    assert "kernel_agent_revision_id" not in json.dumps(previous_report)
    assert "contributing_revision_ids" not in json.dumps(previous_report)
    assert previous_report["parent"] == {
        "path": "input/agents/agent-v0",
    }
    assert previous_report["generated_agent"] == {
        "path": "input/agents/agent-v1",
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
    assert build.proposal.candidate.optimizer_digest != request.parent_revision.optimizer_digest
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
        agent_catalog=_historical_catalog(base_request.parent_revision, historical),
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
        agent_catalog=_historical_catalog(base_request.parent_revision, historical),
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


def test_evolution_output_field_set_matches_the_frozen_evolver_contract() -> None:
    """A drift between the two definitions rejects valid drafts at one boundary only."""
    evolver_src = Path(__file__).resolve().parents[1] / "src/atrex-kernel-agent-evolver/src"
    sys.path.insert(0, str(evolver_src))
    try:
        from report import EVOLUTION_OUTPUT_FIELDS
    finally:
        sys.path.remove(str(evolver_src))

    assert set(EvolutionOutput.model_fields) == set(EVOLUTION_OUTPUT_FIELDS)


def _sibling_revision(
    artifacts: LocalArtifactStore,
    tmp_path: Path,
    parent: KernelAgentRevision,
    *,
    creation_key: str,
) -> KernelAgentRevision:
    """Register one further visible revision of the same Lineage and DSL."""
    source = tmp_path / f"sibling-{creation_key.replace(':', '-')}"
    shutil.copytree(
        artifacts.verify(parent.optimizer_digest).payload_path,
        source,
        copy_function=shutil.copyfile,
    )
    for path in [source, *source.rglob("*")]:
        path.chmod(0o700 if path.is_dir() else 0o600)
    (source / "prompts/episode.md").write_text(f"{creation_key} optimizer\n")
    return KernelAgentRevision(
        id=new_kernel_agent_revision_id(),
        parent_id=parent.id,
        creation_key=creation_key,
        dsl=parent.dsl,
        optimizer_digest=artifacts.put_directory(source, ArtifactKind.KERNEL_AGENT),
        created_by="evolver",
        created_at=NOW,
        evolution_trace_digest=digest(f"evolution-{creation_key}"),
    )


async def _build_with_contributor(
    artifacts: LocalArtifactStore,
    tmp_path: Path,
    *,
    current_epoch_challenger: bool,
    records: list[tuple[str, object, dict[str, object]]],
    contributing_paths: tuple[str, ...] = ("input/agents/agent-v1",),
) -> tuple[BuildChallengerResult, KernelAgentRevision]:
    base_request = _request(artifacts, tmp_path)
    creation_key = (
        f"epoch:{base_request.epoch_id}:challenger:1"
        if current_epoch_challenger
        else "epoch:old:challenger:1"
    )
    contributor = _sibling_revision(
        artifacts,
        tmp_path,
        base_request.parent_revision,
        creation_key=creation_key,
    )
    request = BuildChallengerRequest(
        parent_revision=base_request.parent_revision,
        epoch_id=base_request.epoch_id,
        evidence_checkpoint=base_request.evidence_checkpoint,
        idempotency_key=base_request.idempotency_key,
        agent_catalog=_historical_catalog(base_request.parent_revision, contributor),
    )
    attempt_workspaces = tmp_path / "attempt-workspaces"
    learned = (
        attempt_workspaces
        / ".reusable"
        / request.agent_catalog[0].lineage_id
        / request.parent_revision.id
        / "trajectory-00000002"
    )
    for directory in ("prompts", "memory", "knowledge", "skills", "tools", "hooks"):
        (learned / directory).mkdir(parents=True)
        (learned / directory / "README.md").write_text("index")
    (learned / "memory/lesson.md").write_text("measured memory before evolution")
    sessions = SubprocessEvolutionSessionDriver(
        CleanEnvironmentLauncher(Path("/usr/bin/env")),
        EvolutionProcessConfig(
            bundle_commit="0" * 40,
            bundle_tree="1" * 40,
            bundle_artifact_digest=digest("evolver-bundle"),
            command_argv=(
                str(Path(sys.executable).resolve()),
                str(_agent_script(tmp_path, contributing=contributing_paths)),
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
        EvolutionWorkspaceAssembler(
            tmp_path / "evolutions", artifacts, attempt_workspaces_root=attempt_workspaces
        ),
        sessions,
        artifacts,
        FakeRuntimeEventRecorder(records),
        kernel_agent_limits=kernel_agent_limits(),
        max_output_manifest_bytes=8192,
    )
    return await runner.build_challenger(request), contributor


@pytest.mark.anyio
async def test_contribution_snapshot_preserves_exact_parent_trajectory_resources(
    tmp_path: Path,
) -> None:
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    relative = "input/evidence/agent-v0/resources/trajectories/trajectory-00000002/memory"
    build, _ = await _build_with_contributor(
        artifacts,
        tmp_path,
        current_epoch_challenger=False,
        records=[],
        contributing_paths=(relative,),
    )
    trace = json.loads(
        (artifacts.verify(build.evolution_trace_digest).payload_path / "value.json").read_text()
    )
    snapshot = trace["contributions"][0]
    assert snapshot["path"] == relative
    assert snapshot["revision_id"] == trace["input"]["parent_revision_id"]
    for learned in (tmp_path / "attempt-workspaces/.reusable").rglob("lesson.md"):
        learned.write_text("later attempt replaced the lesson")
    frozen = artifacts.verify(snapshot["snapshot_digest"])
    assert (
        frozen.payload_path / "memory/lesson.md"
    ).read_text() == "measured memory before evolution"
    closure = artifacts.expand_reference_closure([build.evolution_trace_digest])
    assert snapshot["snapshot_digest"] in closure


@pytest.mark.anyio
@pytest.mark.parametrize(
    "relative",
    [
        "input/agents/agent-v99",
        "input/agents/agent-v1/missing",
        "input/evidence/agent-v0/sessions",
        "candidate/prompts",
        "input/agents/agent-v1/../agent-v0",
    ],
)
async def test_runtime_independently_rejects_invalid_contribution_paths(
    tmp_path: Path,
    relative: str,
) -> None:
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    with pytest.raises(ValueError, match="contributing_paths"):
        await _build_with_contributor(
            artifacts,
            tmp_path,
            current_epoch_challenger=False,
            records=[],
            contributing_paths=(relative,),
        )


@pytest.mark.anyio
async def test_fixed_runner_records_credited_completed_history(tmp_path: Path) -> None:
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    records: list[tuple[str, object, dict[str, object]]] = []

    build, contributor = await _build_with_contributor(
        artifacts,
        tmp_path,
        current_epoch_challenger=False,
        records=records,
    )

    assert isinstance(build.proposal, KernelAgentCandidateProposal)
    # Crediting fused content never moves the Source base off the Active revision.
    assert build.proposal.proposal_type == "evolved"
    trace = json.loads(
        (artifacts.verify(build.evolution_trace_digest).payload_path / "value.json").read_text()
    )
    assert trace["output"]["contributing_paths"] == ["input/agents/agent-v1"]
    assert trace["output"]["kernel_agent_revision_id"] != str(contributor.id)
    assert records[-1][2]["contributing_paths"] == ["input/agents/agent-v1"]
    reference = trace["contributions"][0]
    assert reference["path"] == "input/agents/agent-v1"
    assert reference["revision_id"] == contributor.id
    snapshot = artifacts.verify(reference["snapshot_digest"])
    assert (snapshot.payload_path / "agent-v1/prompts/episode.md").is_file()


@pytest.mark.anyio
async def test_fixed_runner_rejects_crediting_a_current_epoch_challenger(
    tmp_path: Path,
) -> None:
    artifacts = LocalArtifactStore(tmp_path / "artifacts")

    with pytest.raises(ValueError, match="only credit completed Lineage history"):
        await _build_with_contributor(
            artifacts,
            tmp_path,
            current_epoch_challenger=True,
            records=[],
        )


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
candidate = Path("candidate")
(candidate / "prompts/integration.md").write_text("real Evolver Bundle ran\\n")
Path("scratch/evolution-report-draft.json").write_text(json.dumps({
    "proposal_type": "evolved",
    "kernel_agent_revision_id": "REPLACE_PARENT_REVISION",
    "hypothesis": "The current Evolver entrypoint accepts the Runtime protocol.",
    "expected_effect": "Produce one valid complete Challenger Bundle.",
    "changed_paths": ["prompts/integration.md"],
    "contributing_paths": [],
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
session_id = sys.argv[sys.argv.index("--session-id") + 1]
native = Path(os.environ["HOME"]) / ".claude/projects/test" / (session_id + ".jsonl")
native.parent.mkdir(parents=True, exist_ok=True)
native.write_text(json.dumps({"type": "assistant", "sessionId": session_id,
    "message": {"id": "integration-response", "role": "assistant", "content": [], "usage": {
        "input_tokens": 16, "output_tokens": 5,
        "cache_read_input_tokens": 2, "cache_creation_input_tokens": 1
    }}}) + "\\n")
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

    with pytest.raises(ValueError, match="disagrees with sealed Agent Bundle"):
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
