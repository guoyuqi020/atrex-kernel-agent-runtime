"""Ablation control arms isolated in their own Campaigns."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest
from conftest import NOW, kernel_agent_limits, seed_lineage

from atrex_runtime.ablation import AblationArmSeeder, AblationArmSpecV1
from atrex_runtime.artifacts.local import ArtifactKind, LocalArtifactStore
from atrex_runtime.domain.ids import ArtifactDigest, CampaignId, LineageId
from atrex_runtime.domain.models import Dsl
from atrex_runtime.kernel_agents import KernelAgentRevisionBuilder
from atrex_runtime.lineage_seed import LineageSeeder, LineageSeedSpecV1
from atrex_runtime.ports import AttemptCandidateResult
from atrex_runtime.presentation import ablation_arm_result_value
from atrex_runtime.registry.sqlite import SqliteRegistry


@dataclass
class FakeEvaluator:
    artifacts: LocalArtifactStore
    calls: list[ArtifactDigest]

    async def evaluate(
        self,
        *,
        campaign_id: CampaignId,
        lineage_id: LineageId,
        dsl: Dsl,
        kernel_artifact_digest: ArtifactDigest,
    ) -> AttemptCandidateResult:
        del campaign_id, lineage_id, dsl
        self.calls.append(kernel_artifact_digest)
        result = self.artifacts.put_json(
            {"status": "succeeded", "result": {"all_pass": True}},
            ArtifactKind.GATEWAY_RESULT,
        )
        return AttemptCandidateResult(kernel_artifact_digest, result, True, 12.5)


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


async def _evolution_arm(
    registry: SqliteRegistry,
    artifacts: LocalArtifactStore,
    tmp_path: Path,
    evaluator: FakeEvaluator,
) -> tuple[LineageSeeder, CampaignId, LineageId]:
    existing = seed_lineage(registry)
    campaign_id = registry.get_lineage(existing.lineage_id).campaign_id
    seeder = LineageSeeder(
        registry,
        artifacts,
        KernelAgentRevisionBuilder(artifacts, limits=kernel_agent_limits()),
        evaluator,
        evolver_commit="e" * 40,
        clock=lambda: NOW,
    )
    evolution = await seeder.seed_lineage(
        campaign_id,
        LineageSeedSpecV1.model_validate(
            {
                "creation_key": "evolution-arm",
                "dsl": "triton",
                "seed": {
                    "source_type": "artifacts",
                    "agent_artifact_digest": _agent_artifact(artifacts, tmp_path),
                    "kernel_artifact_digest": _kernel_artifact(artifacts, tmp_path),
                },
                "challenger_count": 1,
                "attempts_per_trajectory": 2,
                "models": {"optimizer": "optimizer-test", "evolver": "evolver-test"},
            }
        ),
    )
    return seeder, campaign_id, evolution.lineage_id


@pytest.mark.anyio
async def test_ablation_arm_owns_a_separate_campaign_sharing_the_exact_contract(
    tmp_path: Path,
) -> None:
    """The arm is only comparable if the contract is identical and the Campaign is separate."""
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    evaluator = FakeEvaluator(artifacts, [])
    with SqliteRegistry(tmp_path / "registry.sqlite", clock=lambda: NOW) as registry:
        seeder, evolution_campaign_id, evolution_lineage_id = await _evolution_arm(
            registry,
            artifacts,
            tmp_path,
            evaluator,
        )
        arms = AblationArmSeeder(registry, seeder, clock=lambda: NOW)
        spec = AblationArmSpecV1.model_validate(
            {
                "creation_key": "ablation-1",
                "source_lineage_id": str(evolution_lineage_id),
                "attempts_per_trajectory": 2,
            }
        )

        arm = await arms.seed_arm(spec)
        repeated = await arms.seed_arm(spec)

        assert repeated == arm
        # Reusing the baseline measurement means the evolution arm's eval is the only one.
        assert len(evaluator.calls) == 1
        assert arm.campaign_id != evolution_campaign_id
        assert arm.source_campaign_id == evolution_campaign_id
        assert arm.source_lineage_id == evolution_lineage_id

        evolution_campaign = registry.get_campaign(evolution_campaign_id)
        arm_campaign = registry.get_campaign(arm.campaign_id)
        assert arm_campaign.operator == evolution_campaign.operator
        assert arm_campaign.hardware_target == evolution_campaign.hardware_target
        assert (
            arm_campaign.evaluation_contract_digest == evolution_campaign.evaluation_contract_digest
        )
        assert arm_campaign.agent_problem_digest == evolution_campaign.agent_problem_digest

        arm_lineage = registry.get_lineage(arm.lineage.lineage_id)
        assert arm_lineage.ephemeral_agent_state is True
        assert arm_lineage.challenger_count == 0
        assert arm_lineage.optimizer_model == "optimizer-test"
        assert arm_lineage.evolver_model == "evolver-test"
        assert arm_lineage.trajectories_per_branch == 1
        assert arm_lineage.dsl is registry.get_lineage(evolution_lineage_id).dsl
        # Recording the shared Bootstrap is what lets the arm read that Bootstrap's
        # measurement history without seeing the evolution arm's own Attempts.
        assert arm_lineage.bootstrap_source_lineage_id == evolution_lineage_id
        # The arm's Campaign owns exactly one Lineage, so the single-Lineage production
        # assumptions keep holding.
        assert [entry.id for entry in registry.list_campaign_lineages(arm.campaign_id)] == [
            arm_lineage.id
        ]


@pytest.mark.anyio
@pytest.mark.parametrize("attempts,epochs,total", [(1, 15, 30), (5, 3, 30)])
async def test_evolving_arm_preserves_baseline_and_models_with_independent_schedule(
    tmp_path: Path,
    attempts: int,
    epochs: int,
    total: int,
) -> None:
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    evaluator = FakeEvaluator(artifacts, [])
    with SqliteRegistry(tmp_path / "registry.sqlite", clock=lambda: NOW) as registry:
        seeder, source_campaign_id, source_id = await _evolution_arm(
            registry,
            artifacts,
            tmp_path,
            evaluator,
        )
        arms = AblationArmSeeder(registry, seeder, clock=lambda: NOW)
        spec = AblationArmSpecV1(
            creation_key=f"ablation-evolve-{attempts}",
            source_lineage_id=source_id,
            attempts_per_trajectory=attempts,
            challenger_count=1,
            challenger_start_epoch=2,
            first_epoch_same_agent=True,
            ephemeral_agent_state=False,
        )
        result = await arms.seed_arm(spec)
        assert await arms.seed_arm(spec) == result
        response = ablation_arm_result_value(result)
        assert response["challenger_count"] == 1
        assert response["challenger_start_epoch"] == 2
        assert response["first_epoch_same_agent"] is True
        lineage = registry.get_lineage(result.lineage.lineage_id)
        assert lineage.challenger_count == 1
        assert lineage.challenger_start_epoch == 2
        assert lineage.attempts_per_trajectory == attempts
        assert lineage.trajectories_per_branch == 1
        assert lineage.ephemeral_agent_state is False
        assert lineage.optimizer_model == "optimizer-test"
        assert lineage.evolver_model == "evolver-test"
        assert lineage.bootstrap_source_lineage_id == source_id
        assert (
            sum(
                attempts * (1 + lineage.challengers_for_epoch(epoch))
                for epoch in range(1, epochs + 1)
            )
            == total
        )
        assert registry.get_campaign(result.campaign_id).evolver_commit == (
            registry.get_campaign(source_campaign_id).evolver_commit
        )
        assert registry.get_campaign(result.campaign_id).evolver_commit == "e" * 40
        assert len(evaluator.calls) == 1
        # A resumed arm cannot silently change its frozen evolution schedule.
        with pytest.raises(ValueError, match="different"):
            await arms.seed_arm(spec.model_copy(update={"first_epoch_same_agent": False}))


@pytest.mark.anyio
@pytest.mark.parametrize("kind,ephemeral", [("isolated", True), ("retained", False)])
async def test_two_ablation_arms_are_mutually_independent(
    tmp_path: Path, kind: str, ephemeral: bool
) -> None:
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    evaluator = FakeEvaluator(artifacts, [])
    with SqliteRegistry(tmp_path / "registry.sqlite", clock=lambda: NOW) as registry:
        seeder, _campaign_id, evolution_lineage_id = await _evolution_arm(
            registry,
            artifacts,
            tmp_path,
            evaluator,
        )
        arms = AblationArmSeeder(registry, seeder, clock=lambda: NOW)
        first = await arms.seed_arm(
            AblationArmSpecV1.model_validate(
                {
                    "creation_key": f"ablation-{kind}-01",
                    "source_lineage_id": str(evolution_lineage_id),
                    "attempts_per_trajectory": 3,
                    "ephemeral_agent_state": ephemeral,
                }
            )
        )
        second = await arms.seed_arm(
            AblationArmSpecV1.model_validate(
                {
                    "creation_key": f"ablation-{kind}-02",
                    "source_lineage_id": str(evolution_lineage_id),
                    "attempts_per_trajectory": 3,
                    "ephemeral_agent_state": ephemeral,
                }
            )
        )

        assert first.campaign_id != second.campaign_id
        assert first.lineage.lineage_id != second.lineage.lineage_id
        assert first.lineage.kernel_revision_id != second.lineage.kernel_revision_id
        # Identical starting point, independent identities.
        assert first.lineage.kernel_artifact_digest == second.lineage.kernel_artifact_digest
        assert first.lineage.agent_artifact_digest == second.lineage.agent_artifact_digest
        assert first.lineage.gateway_result_digest == second.lineage.gateway_result_digest
        assert first.lineage.latency_us == second.lineage.latency_us
        for result in (first, second):
            lineage = registry.get_lineage(result.lineage.lineage_id)
            assert lineage.ephemeral_agent_state is ephemeral
            assert lineage.trajectories_per_branch == 1
            assert lineage.attempts_per_trajectory == 3
            assert lineage.challenger_count == 0
            assert lineage.bootstrap_source_lineage_id == evolution_lineage_id
        assert len(evaluator.calls) == 1


@pytest.mark.anyio
async def test_control_arms_cross_pooling_and_agent_state_retention(
    tmp_path: Path,
) -> None:
    """Pool-Retained completes the pooling/state-retention controls; none evolves."""
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    evaluator = FakeEvaluator(artifacts, [])
    with SqliteRegistry(tmp_path / "registry.sqlite", clock=lambda: NOW) as registry:
        seeder, _campaign_id, evolution_lineage_id = await _evolution_arm(
            registry,
            artifacts,
            tmp_path,
            evaluator,
        )
        arms = AblationArmSeeder(registry, seeder, clock=lambda: NOW)
        shapes = {
            "isolated": {"trajectories_per_branch": 1, "ephemeral_agent_state": True},
            "pooled": {"trajectories_per_branch": 2, "ephemeral_agent_state": True},
            "retained": {"trajectories_per_branch": 1, "ephemeral_agent_state": False},
            "pool-retained": {"trajectories_per_branch": 2, "ephemeral_agent_state": False},
        }
        seeded = {
            kind: await arms.seed_arm(
                AblationArmSpecV1.model_validate(
                    {
                        "creation_key": f"ablation-{kind}",
                        "source_lineage_id": str(evolution_lineage_id),
                        "attempts_per_trajectory": 2,
                        **shape,
                    }
                )
            )
            for kind, shape in shapes.items()
        }

        for kind, shape in shapes.items():
            lineage = registry.get_lineage(seeded[kind].lineage.lineage_id)
            assert lineage.trajectories_per_branch == shape["trajectories_per_branch"]
            assert lineage.ephemeral_agent_state is shape["ephemeral_agent_state"]
            # No arm ever evolves, whatever its shape.
            assert lineage.challenger_count == 0
            assert lineage.bootstrap_source_lineage_id == evolution_lineage_id

        assert len({arm.campaign_id for arm in seeded.values()}) == 4
        # Every arm starts from the identical frozen baseline, measured exactly once.
        assert len({arm.lineage.kernel_artifact_digest for arm in seeded.values()}) == 1
        assert len({arm.lineage.latency_us for arm in seeded.values()}) == 1
        assert len({arm.lineage.kernel_revision_id for arm in seeded.values()}) == 4
        assert len(evaluator.calls) == 1


@pytest.mark.anyio
async def test_an_ablation_arm_cannot_be_cloned_from_another_ablation_arm(
    tmp_path: Path,
) -> None:
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    evaluator = FakeEvaluator(artifacts, [])
    with SqliteRegistry(tmp_path / "registry.sqlite", clock=lambda: NOW) as registry:
        seeder, _campaign_id, evolution_lineage_id = await _evolution_arm(
            registry,
            artifacts,
            tmp_path,
            evaluator,
        )
        arms = AblationArmSeeder(registry, seeder, clock=lambda: NOW)
        arm = await arms.seed_arm(
            AblationArmSpecV1.model_validate(
                {
                    "creation_key": "ablation-1",
                    "source_lineage_id": str(evolution_lineage_id),
                    "attempts_per_trajectory": 2,
                }
            )
        )

        with pytest.raises(ValueError, match="cannot be cloned from another ablation arm"):
            await arms.seed_arm(
                AblationArmSpecV1.model_validate(
                    {
                        "creation_key": "ablation-of-ablation",
                        "source_lineage_id": str(arm.lineage.lineage_id),
                        "attempts_per_trajectory": 2,
                    }
                )
            )
