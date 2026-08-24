"""Attempt workspace assembly from immutable Registry and Artifact Store state."""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
from uuid import uuid4

from ..artifacts.local import ArtifactKind, LocalArtifactStore
from ..filesystem import make_tree_owner_writable
from ..ports import RunAttemptRequest
from ..registry.base import Registry
from .evidence_view import assemble_optimizer_evidence_view
from .manifest import AttemptInputManifestV7, AttemptTaskContextV5


@dataclass(frozen=True, slots=True)
class PreparedAttempt:
    """Private filesystem allocation for one Core Optimizer process."""

    root: Path
    manifest_path: Path
    session_root: Path
    session_id: str


class AttemptWorkspaceAssembler(Protocol):
    """Materialize trusted Attempt inputs into a new private workspace."""

    def prepare(self, request: RunAttemptRequest) -> PreparedAttempt:
        """Create a new workspace; repeated calls must never reuse a session root."""
        ...


class LocalAttemptWorkspaceAssembler:
    """Local provider that exposes Optimizer inputs but never the Evolver artifact."""

    def __init__(
        self,
        root: str | Path,
        registry: Registry,
        artifacts: LocalArtifactStore,
    ) -> None:
        self._root = Path(root).resolve()
        self._registry = registry
        self._artifacts = artifacts
        self._root.mkdir(parents=True, exist_ok=True, mode=0o700)

    def prepare(self, request: RunAttemptRequest) -> PreparedAttempt:
        """Create one append-only run directory and materialize verified artifacts."""
        revision = self._registry.get_kernel_agent_revision(request.kernel_agent_revision_id)
        if revision.dsl is not request.dsl:
            raise ValueError("Attempt DSL disagrees with its Kernel Agent revision")
        kernel = self._registry.get_kernel_revision(request.input_kernel_revision_id)
        attempt = self._registry.get_attempt(request.attempt_id)
        if (
            attempt.kernel_agent_revision_id != request.kernel_agent_revision_id
            or attempt.input_kernel_revision_id != request.input_kernel_revision_id
            or attempt.attempt_evidence_digest != request.attempt_evidence_digest
        ):
            raise ValueError("Attempt request disagrees with Registry state")
        epoch = self._registry.get_epoch(attempt.epoch_id)
        if epoch.evidence_checkpoint != request.epoch_evidence_checkpoint:
            raise ValueError("Attempt request disagrees with its Epoch Evidence")
        lineage = self._registry.get_lineage(epoch.lineage_id)
        campaign = self._registry.get_campaign(lineage.campaign_id)

        attempt_root = self._root / str(request.attempt_id)
        attempt_root.mkdir(mode=0o700, exist_ok=True)
        root = attempt_root / f"run-{uuid4().hex}"
        root.mkdir(mode=0o700)

        manifest = AttemptInputManifestV7(
            attempt_id=request.attempt_id,
            kernel_agent_revision_id=request.kernel_agent_revision_id,
            input_kernel_revision_id=request.input_kernel_revision_id,
            input_kernel_digest=kernel.artifact_digest,
            epoch_evidence_checkpoint=request.epoch_evidence_checkpoint,
            attempt_evidence_digest=request.attempt_evidence_digest,
            optimizer_digest=revision.optimizer_digest,
            dsl=request.dsl,
            context=AttemptTaskContextV5(
                campaign_id=campaign.id,
                lineage_id=lineage.id,
                epoch_id=epoch.id,
                epoch_number=epoch.number,
                attempt_ordinal=attempt.ordinal,
                operator=campaign.operator,
                hardware_target=campaign.hardware_target,
                evaluation_contract_digest=campaign.evaluation_contract_digest,
                agent_problem_digest=campaign.agent_problem_digest,
            ),
        )
        paths = manifest.paths
        self._artifacts.materialize(kernel.artifact_digest, root / paths.input_kernel)
        epoch_evidence = self._artifacts.verify(request.epoch_evidence_checkpoint)
        if epoch_evidence.kind is not ArtifactKind.EVIDENCE:
            raise ValueError("Attempt epoch Evidence has the wrong artifact kind")
        attempt_evidence = self._artifacts.verify(request.attempt_evidence_digest)
        if attempt_evidence.kind is not ArtifactKind.ATTEMPT_EVIDENCE:
            raise ValueError("Attempt branch Evidence has the wrong artifact kind")
        assemble_optimizer_evidence_view(
            root / paths.evidence,
            lineage_payload=epoch_evidence.payload_path,
            lineage_checkpoint=request.epoch_evidence_checkpoint,
            attempt_payload=attempt_evidence.payload_path,
            attempt_snapshot=request.attempt_evidence_digest,
            current_epoch_number=epoch.number,
            branch=attempt.branch,
            challenger_ordinal=attempt.challenger_ordinal,
            trajectory_ordinal=attempt.trajectory_ordinal,
            selected_revision=request.kernel_agent_revision_id,
            attempt_ordinal=attempt.ordinal,
            artifacts=self._artifacts,
        )
        visible_digest = campaign.agent_problem_digest
        contract = self._artifacts.verify(visible_digest)
        if contract.kind is not ArtifactKind.AGENT_PROBLEM:
            raise ValueError("Campaign Agent Problem has the wrong artifact kind")
        self._artifacts.materialize(
            visible_digest,
            root / paths.agent_problem,
        )
        self._artifacts.materialize(revision.optimizer_digest, root / paths.optimizer)

        working_kernel = root / paths.working_kernel
        shutil.copytree(root / paths.input_kernel, working_kernel)
        make_tree_owner_writable(working_kernel)

        manifest_path = root / "attempt.json"
        manifest_path.write_bytes(manifest.canonical_json_bytes())
        os.chmod(manifest_path, 0o400)
        session_root = root / "sessions"
        session_root.mkdir(mode=0o700)
        (root / "scratch").mkdir(mode=0o700)
        # Stays empty on the host; the Sandbox binds the pinned reference tree over it.
        (root / paths.reference).mkdir(mode=0o500)
        return PreparedAttempt(
            root=root,
            manifest_path=manifest_path,
            session_root=session_root,
            session_id=f"attempt-session-{uuid4().hex}",
        )
