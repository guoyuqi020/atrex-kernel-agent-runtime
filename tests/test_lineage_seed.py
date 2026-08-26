"""Independent Lineage roots selected from sealed Agent and Kernel content."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest
from conftest import NOW, kernel_agent_limits, seed_lineage

from atrex_runtime.artifacts.local import ArtifactKind, LocalArtifactStore
from atrex_runtime.domain.ids import ArtifactDigest, CampaignId, LineageId
from atrex_runtime.domain.models import Dsl
from atrex_runtime.kernel_agents import KernelAgentRevisionBuilder
from atrex_runtime.lineage_seed import LineageSeeder, LineageSeedSpecV1
from atrex_runtime.ports import AttemptCandidateResult
from atrex_runtime.registry.sqlite import SqliteRegistry


def _agent_artifact(artifacts: LocalArtifactStore, root: Path) -> ArtifactDigest:
    source = root / "agent"
    (source / "src").mkdir(parents=True)
    (source / "atrex-bundle.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "bundle_format": "atrex-kernel-agent-bundle-v1",
                "entrypoint": {"command": "src/main.py"},
            }
        ),
        encoding="utf-8",
    )
    (source / "src/main.py").write_text("def optimize(): ...\n", encoding="utf-8")
    return artifacts.put_directory(source, ArtifactKind.KERNEL_AGENT)


def _kernel_artifact(artifacts: LocalArtifactStore, root: Path) -> ArtifactDigest:
    source = root / "kernel"
    source.mkdir()
    (source / "kernel.py").write_text("class Model: pass\n", encoding="utf-8")
    return artifacts.put_directory(source, ArtifactKind.KERNEL)


@dataclass
class FakeEvaluator:
    artifacts: LocalArtifactStore
    calls: list[tuple[CampaignId, LineageId, Dsl, ArtifactDigest]]
    correct: bool = True

    async def evaluate(
        self,
        *,
        campaign_id: CampaignId,
        lineage_id: LineageId,
        dsl: Dsl,
        kernel_artifact_digest: ArtifactDigest,
    ) -> AttemptCandidateResult:
        self.calls.append((campaign_id, lineage_id, dsl, kernel_artifact_digest))
        result = self.artifacts.put_json(
            {"status": "succeeded", "result": {"all_pass": self.correct}},
            ArtifactKind.GATEWAY_RESULT,
        )
        return AttemptCandidateResult(
            kernel_artifact_digest,
            result,
            self.correct,
            12.5 if self.correct else None,
        )


@pytest.mark.anyio
async def test_artifact_seed_creates_independent_v0_roots_and_is_idempotent(
    tmp_path: Path,
) -> None:
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    agent_digest = _agent_artifact(artifacts, tmp_path)
    kernel_digest = _kernel_artifact(artifacts, tmp_path)
    with SqliteRegistry(tmp_path / "registry.sqlite", clock=lambda: NOW) as registry:
        existing = seed_lineage(registry)
        campaign_id = registry.get_lineage(existing.lineage_id).campaign_id
        evaluator = FakeEvaluator(artifacts, [])
        seeder = LineageSeeder(
            registry,
            artifacts,
            KernelAgentRevisionBuilder(artifacts, limits=kernel_agent_limits()),
            evaluator,
            evolver_commit="e" * 40,
            clock=lambda: NOW,
        )
        spec = LineageSeedSpecV1.model_validate(
            {
                "schema_version": 1,
                "creation_key": "fork-best-v7",
                "dsl": "triton",
                "seed": {
                    "source_type": "artifacts",
                    "agent_artifact_digest": agent_digest,
                    "kernel_artifact_digest": kernel_digest,
                },
                "models": {"optimizer": "gpt-5.6", "evolver": "claude-opus"},
                "challenger_count": 2,
                "challenger_start_epoch": 2,
                "trajectories_per_branch": 3,
                "attempts_per_trajectory": 4,
            }
        )

        result = await seeder.seed_lineage(campaign_id, spec)
        repeated = await seeder.seed_lineage(campaign_id, spec)

        assert repeated == result
        assert result.lineage_id != existing.lineage_id
        assert result.agent_artifact_digest == agent_digest
        assert result.kernel_artifact_digest == kernel_digest
        assert result.source_agent_revision_id is None
        assert result.source_kernel_revision_id is None
        assert len(evaluator.calls) == 1
        lineage = registry.get_lineage(result.lineage_id)
        assert lineage.optimizer_model == "gpt-5.6"
        assert lineage.evolver_model == "claude-opus"
        assert registry.get_campaign(campaign_id).evolver_commit == "e" * 40
        assert lineage.next_epoch_number == 1
        assert registry.list_lineage_agent_revisions(lineage.id)[0].revision_number == 0
        assert (
            registry.get_kernel_agent_revision(result.kernel_agent_revision_id).created_by
            == "lineage_seed"
        )
        assert registry.list_lineage_kernels(lineage.id)[0].revision_number == 0
        evidence = artifacts.verify(result.evidence_checkpoint).payload_path
        assert json.loads((evidence / "bootstrap/report.json").read_text()) == {
            "schema_version": 1,
            "status": "bootstrap_input",
            "source": "empty-lineage-seed",
            "source_files": [],
        }
        assert (evidence / "bootstrap/conversation.jsonl").read_text() == ""


@pytest.mark.anyio
async def test_revision_seed_reuses_content_but_creates_new_revision_identities(
    tmp_path: Path,
) -> None:
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    agent_digest = _agent_artifact(artifacts, tmp_path)
    kernel_digest = _kernel_artifact(artifacts, tmp_path)
    with SqliteRegistry(tmp_path / "registry.sqlite", clock=lambda: NOW) as registry:
        existing = seed_lineage(registry)
        campaign_id = registry.get_lineage(existing.lineage_id).campaign_id
        evaluator = FakeEvaluator(artifacts, [])
        seeder = LineageSeeder(
            registry,
            artifacts,
            KernelAgentRevisionBuilder(artifacts, limits=kernel_agent_limits()),
            evaluator,
            clock=lambda: NOW,
        )
        artifact_root = await seeder.seed_lineage(
            campaign_id,
            LineageSeedSpecV1.model_validate(
                {
                    "creation_key": "artifact-root",
                    "dsl": "triton",
                    "seed": {
                        "source_type": "artifacts",
                        "agent_artifact_digest": agent_digest,
                        "kernel_artifact_digest": kernel_digest,
                    },
                    "attempts_per_trajectory": 1,
                }
            ),
        )
        revision_root = await seeder.seed_lineage(
            campaign_id,
            LineageSeedSpecV1.model_validate(
                {
                    "creation_key": "revision-root",
                    "dsl": "triton",
                    "seed": {
                        "source_type": "revisions",
                        "agent_revision_id": artifact_root.kernel_agent_revision_id,
                        "kernel_revision_id": artifact_root.kernel_revision_id,
                    },
                    "attempts_per_trajectory": 1,
                }
            ),
        )

        assert revision_root.kernel_agent_revision_id != artifact_root.kernel_agent_revision_id
        assert revision_root.kernel_revision_id != artifact_root.kernel_revision_id
        assert revision_root.agent_artifact_digest == artifact_root.agent_artifact_digest
        assert revision_root.kernel_artifact_digest == artifact_root.kernel_artifact_digest
        assert revision_root.source_agent_revision_id == artifact_root.kernel_agent_revision_id
        assert revision_root.source_kernel_revision_id == artifact_root.kernel_revision_id
        assert len(evaluator.calls) == 2


@pytest.mark.anyio
async def test_incorrect_seed_kernel_does_not_publish_lineage(tmp_path: Path) -> None:
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    agent_digest = _agent_artifact(artifacts, tmp_path)
    kernel_digest = _kernel_artifact(artifacts, tmp_path)
    with SqliteRegistry(tmp_path / "registry.sqlite", clock=lambda: NOW) as registry:
        existing = seed_lineage(registry)
        campaign_id = registry.get_lineage(existing.lineage_id).campaign_id
        seeder = LineageSeeder(
            registry,
            artifacts,
            KernelAgentRevisionBuilder(artifacts, limits=kernel_agent_limits()),
            FakeEvaluator(artifacts, [], correct=False),
            clock=lambda: NOW,
        )
        before = registry.list_campaign_lineages(campaign_id)
        spec = LineageSeedSpecV1.model_validate(
            {
                "creation_key": "incorrect",
                "dsl": "triton",
                "seed": {
                    "source_type": "artifacts",
                    "agent_artifact_digest": agent_digest,
                    "kernel_artifact_digest": kernel_digest,
                },
                "attempts_per_trajectory": 1,
            }
        )

        with pytest.raises(ValueError, match="failed authoritative evaluation"):
            await seeder.seed_lineage(campaign_id, spec)
        assert registry.list_campaign_lineages(campaign_id) == before
