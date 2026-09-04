"""Attempt workspace assembly from immutable Registry and Artifact Store state."""

from __future__ import annotations

import fcntl
import json
import os
import shutil
import stat
import tempfile
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
from uuid import uuid4

from ..artifacts.local import ArtifactKind, LocalArtifactStore
from ..domain.ids import ArtifactDigest, parse_artifact_digest
from ..domain.models import BranchRole, Epoch, EpochStatus, KernelAgentRevision, Lineage
from ..filesystem import make_tree_owner_writable, make_tree_read_only
from ..ports import RunAttemptRequest
from ..registry.base import Registry
from .evidence_view import assemble_optimizer_evidence_view
from .manifest import (
    ATTEMPT_MANIFEST_RELATIVE_PATH,
    ATTEMPT_WORKSPACE_LAYOUT,
    AttemptInputManifestV9,
    AttemptTaskContextV5,
)
from .state_selection import RuntimeStateAttempt, select_winning_trajectory_terminal_state

REUSABLE_AGENT_DIRECTORIES = ("prompts", "memory", "knowledge", "skills", "tools", "hooks")
_LEGACY_STATE_DIRECTORIES = frozenset((*REUSABLE_AGENT_DIRECTORIES, "docs"))
REUSABLE_READMES = {
    name: (Path(__file__).parents[1] / "templates/runtime-state" / f"{name}.md").read_text(
        encoding="utf-8"
    )
    for name in REUSABLE_AGENT_DIRECTORIES
}


def _knowledge_directory(root: Path) -> Path:
    """Resolve the old State name without changing sealed history or merging conflicts."""
    current, legacy = root / "knowledge", root / "docs"
    if legacy.exists() or legacy.is_symlink():
        _validate_reusable_tree(legacy)
        if current.exists() or current.is_symlink():
            raise ValueError(
                "Runtime State contains both docs/ and knowledge/; "
                "merge their content into knowledge/ and remove the legacy docs/ directory"
            )
        return legacy
    return current


def ensure_reusable_directories(root: Path, *, optimizer_source: Path | None = None) -> None:
    """Fill missing State directories/indexes without overwriting Agent-authored content."""
    if root.is_symlink():
        raise ValueError(f"Reusable State root must not be a symlink: {root}")
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    prompts = root / "prompts"
    if optimizer_source is not None and not prompts.exists() and not prompts.is_symlink():
        seed = optimizer_source / "prompts"
        if seed.exists() or seed.is_symlink():
            _copy_reusable_tree(seed, prompts)
            make_tree_owner_writable(prompts)
    knowledge = _knowledge_directory(root)
    if knowledge.name == "docs":
        knowledge.rename(root / "knowledge")
        make_tree_owner_writable(root / "knowledge")
    for name in REUSABLE_AGENT_DIRECTORIES:
        directory = root / name
        if directory.is_symlink():
            raise ValueError(f"Reusable Workspace directory is invalid: {directory}")
        directory.mkdir(mode=0o700, exist_ok=True)
        _validate_reusable_tree(directory)
        readme = directory / "README.md"
        if readme.exists() and not readme.is_file():
            raise ValueError(f"Reusable {name} README must be a regular file")
        if not readme.exists():
            readme.write_text(REUSABLE_READMES[name], encoding="utf-8")


def copy_reusable_agent_state(
    source: Path, destination: Path, *, optimizer_source: Path | None = None
) -> None:
    """Copy just adaptive State, upgrading older snapshots only in the writable copy."""
    knowledge = _knowledge_directory(source)
    for name in REUSABLE_AGENT_DIRECTORIES:
        directory = knowledge if name == "knowledge" else source / name
        if directory.exists() or directory.is_symlink():
            _copy_reusable_tree(directory, destination / name)
            make_tree_owner_writable(destination / name)
    ensure_reusable_directories(destination, optimizer_source=optimizer_source)


def remove_optimizer_state_seeds(repository: Path) -> None:
    """Hide duplicate initial State only in a materialized Optimizer Source copy."""
    if repository.is_symlink() or not repository.is_dir():
        raise ValueError("Optimizer Source must be a real directory")
    mode = stat.S_IMODE(repository.stat().st_mode)
    try:
        repository.chmod(mode | stat.S_IWUSR)
        for name in REUSABLE_AGENT_DIRECTORIES:
            path = repository / name
            if path.exists() or path.is_symlink():
                _validate_reusable_tree(path)
                make_tree_owner_writable(path)
                shutil.rmtree(path)
    finally:
        repository.chmod(mode)


def materialize_reusable_agent_state_snapshot(
    attempt_workspaces_root: str | Path | None,
    destination: Path,
    *,
    agent_lineages: Mapping[str, object | None],
    optimizer_sources: Mapping[str, Path] | None = None,
    read_only: bool = True,
) -> dict[str, tuple[int, ...]]:
    """Freeze visible per-trajectory reusable Agent state for one Evolver session."""
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    destination.mkdir(parents=True, mode=0o700)
    trajectories_by_revision: dict[str, tuple[int, ...]] = {}
    persistent = (
        None
        if attempt_workspaces_root is None
        else Path(attempt_workspaces_root).resolve() / ".reusable"
    )

    def copy_snapshot() -> None:
        for revision_id, lineage_id in sorted(agent_lineages.items()):
            revision_root = destination / revision_id
            trajectories_root = revision_root / "trajectories"
            trajectories_root.mkdir(parents=True, mode=0o700)
            ordinals: list[int] = []
            source_root = (
                None
                if persistent is None or lineage_id is None
                else persistent / str(lineage_id) / revision_id
            )
            if source_root is not None and source_root.exists():
                if source_root.is_symlink() or not source_root.is_dir():
                    raise ValueError(f"Reusable Agent revision directory is invalid: {source_root}")
                for source in sorted(source_root.iterdir()):
                    if source.is_symlink() or not source.is_dir():
                        raise ValueError(f"Reusable trajectory entry is invalid: {source}")
                    # Bootstrap is the revision-wide seed copied into every new
                    # Optimizer trajectory. Evolver snapshots expose concrete
                    # trajectories only, so this internal seed is not a branch.
                    if source.name == "bootstrap":
                        continue
                    prefix = "trajectory-"
                    suffix = source.name.removeprefix(prefix)
                    if (
                        not source.name.startswith(prefix)
                        or len(suffix) != 8
                        or not suffix.isdigit()
                    ):
                        raise ValueError(f"Reusable trajectory directory is invalid: {source}")
                    ordinal = int(suffix)
                    if ordinal <= 0:
                        raise ValueError(f"Reusable trajectory ordinal is invalid: {source}")
                    children = {child.name for child in source.iterdir()}
                    if not children <= _LEGACY_STATE_DIRECTORIES:
                        raise ValueError(
                            f"Reusable trajectory has unexpected State directories: {source}"
                        )
                    target = trajectories_root / source.name
                    target.mkdir(mode=0o700)
                    copy_reusable_agent_state(
                        source,
                        target,
                        optimizer_source=(optimizer_sources or {}).get(revision_id),
                    )
                    ordinals.append(ordinal)
            trajectories_by_revision[revision_id] = tuple(ordinals)

    if persistent is None or not persistent.exists():
        copy_snapshot()
    else:
        with _exclusive_lock(persistent / ".lock"):
            copy_snapshot()
    if read_only:
        make_tree_read_only(destination)
    return trajectories_by_revision


def validate_reusable_agent_state_seed(
    root: str | Path,
    *,
    max_files: int | None = None,
    max_bytes: int | None = None,
    require_complete: bool = False,
) -> None:
    """Validate one revision-wide Candidate seed copied into every new Trajectory."""
    state_root = Path(root)
    _validate_reusable_tree(state_root)
    present = {child.name for child in state_root.iterdir()}
    allowed = set(REUSABLE_AGENT_DIRECTORIES) if require_complete else _LEGACY_STATE_DIRECTORIES
    if not present <= allowed:
        raise ValueError(
            "Candidate runtime-state may contain only "
            "prompts/, memory/, knowledge/, skills/, tools/, hooks/"
        )
    knowledge = _knowledge_directory(state_root)
    # Older immutable snapshots may lack newly introduced directories or empty ones.
    # Normalize only materialized copies; new Candidate output must have all six indexes.
    files = 0
    total_bytes = 0
    for name in REUSABLE_AGENT_DIRECTORIES:
        directory = knowledge if name == "knowledge" else state_root / name
        if require_complete and not (directory / "README.md").is_file():
            raise ValueError(f"Candidate runtime-state {name}/ must retain README.md")
        if not directory.exists():
            continue
        _validate_reusable_tree(directory)
        for entry in directory.rglob("*"):
            if entry.is_file():
                files += 1
                total_bytes += entry.stat().st_size
                if max_files is not None and files > max_files:
                    raise ValueError("Candidate runtime-state exceeds file limit")
                if max_bytes is not None and total_bytes > max_bytes:
                    raise ValueError("Candidate runtime-state exceeds byte limit")
    readme = state_root / "tools/README.md"
    if readme.is_symlink() or not readme.is_file():
        raise ValueError("Candidate runtime-state tools/ must retain README.md")


def resolve_revision_runtime_state_seed(
    artifacts: LocalArtifactStore,
    revision: KernelAgentRevision,
) -> Path | None:
    """Resolve the revision-owned common state seed recorded by Evolver."""
    if revision.runtime_state_digest is not None:
        state = artifacts.verify(revision.runtime_state_digest)
        if state.kind is not ArtifactKind.KERNEL_AGENT_RUNTIME_STATE:
            raise ValueError("Agent revision runtime-state has the wrong Artifact kind")
        validate_reusable_agent_state_seed(state.payload_path)
        return state.payload_path
    if revision.created_by != "evolver" or revision.evolution_trace_digest is None:
        return None
    try:
        trace = artifacts.verify(revision.evolution_trace_digest)
    except FileNotFoundError:
        return None
    if trace.kind is not ArtifactKind.EVOLUTION:
        raise ValueError("Agent revision Evolution trace has the wrong Artifact kind")
    value = json.loads((trace.payload_path / "value.json").read_bytes())
    if not isinstance(value, dict):
        raise ValueError("Agent revision Evolution trace must be a JSON object")
    if value.get("schema_version") != 9:
        return None
    candidate = value.get("candidate")
    if not isinstance(candidate, dict):
        raise ValueError("Evolved Agent trace has no Candidate state")
    raw_digest = candidate.get("runtime_state_digest")
    if raw_digest is None:
        return None
    if not isinstance(raw_digest, str):
        raise ValueError("Evolved Agent trace runtime-state digest is invalid")
    state = artifacts.verify(parse_artifact_digest(raw_digest))
    if state.kind is not ArtifactKind.KERNEL_AGENT_RUNTIME_STATE:
        raise ValueError("Evolved Agent runtime-state has the wrong Artifact kind")
    validate_reusable_agent_state_seed(state.payload_path)
    return state.payload_path


@dataclass(frozen=True, slots=True)
class PreparedAttempt:
    """Private filesystem allocation for one Core Optimizer process."""

    root: Path
    manifest_path: Path
    session_root: Path
    session_id: str
    persistent_state_root: Path | None = None
    persistent_lock_path: Path | None = None

    def persist_reusable_directories(self) -> None:
        """Atomically publish reusable Agent state after this physical Session exits."""
        persist_reusable_agent_state(
            self.root, self.persistent_state_root, self.persistent_lock_path
        )

    def seal_runtime_state(self, artifacts: LocalArtifactStore) -> ArtifactDigest:
        """Seal all six post-Session State directories as one immutable checkpoint."""
        ensure_reusable_directories(self.root)
        scratch = self.root / "scratch"
        scratch.mkdir(mode=0o700, exist_ok=True)
        state = Path(tempfile.mkdtemp(prefix="runtime-state-", dir=scratch))
        try:
            copy_reusable_agent_state(self.root, state)
            validate_reusable_agent_state_seed(state, require_complete=True)
            return artifacts.put_directory(state, ArtifactKind.KERNEL_AGENT_RUNTIME_STATE)
        finally:
            shutil.rmtree(state, ignore_errors=True)


def initialize_reusable_agent_state(root: Path, optimizer_source: Path) -> None:
    """Load the pinned Core's initial State, never its engineering docs/ directory."""
    for name in REUSABLE_AGENT_DIRECTORIES:
        directory = optimizer_source / name
        if directory.exists() or directory.is_symlink():
            _copy_reusable_tree(directory, root / name)
            make_tree_owner_writable(root / name)
    # Older Core commits do not contain State seeds. Missing directories/indexes
    # get the legacy empty defaults, without importing Source's engineering docs/.
    ensure_reusable_directories(root)


def persist_reusable_agent_state(root: Path, destination: Path | None, lock: Path | None) -> None:
    """Publish all adaptive directories with the same lock and persistence rules."""
    if destination is None or lock is None:
        return
    with _exclusive_lock(lock):
        ensure_reusable_directories(root)
        for name in REUSABLE_AGENT_DIRECTORIES:
            _replace_reusable_tree(root / name, destination / name)


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
        # An ephemeral-state Lineage is an ablation control arm, so every Attempt starts from the
        # same Core seed. That has to be decided before reading any prior digest:
        # attempt.runtime_state_digest is sealed after the Session, so a physical retry would
        # otherwise inherit the first run's Skills.
        previous_runtime_state_digest: ArtifactDigest | None = None
        reset_persistent_scope = False
        if not lineage.ephemeral_agent_state:
            # A physical retry of the same logical Attempt resumes from its latest
            # sealed Session state. A new serial Attempt resumes from its predecessor.
            previous_runtime_state_digest = (
                attempt.runtime_state_digest or attempt.input_runtime_state_digest
            )
            if previous_runtime_state_digest is None and attempt.ordinal > 1:
                previous = self._registry.find_attempt(
                    attempt.epoch_id,
                    attempt.branch,
                    attempt.challenger_ordinal,
                    attempt.trajectory_ordinal,
                    attempt.ordinal - 1,
                )
                if previous is None or previous.runtime_state_digest is None:
                    raise ValueError("Previous serial Attempt has no Runtime State checkpoint")
                previous_runtime_state_digest = previous.runtime_state_digest
            elif previous_runtime_state_digest is None:
                if attempt.branch is BranchRole.ACTIVE:
                    previous_runtime_state_digest = self._active_branch_seed(epoch)
                if previous_runtime_state_digest is None:
                    previous_runtime_state_digest = revision.runtime_state_digest
                # A first logical Attempt starts a fresh per-Trajectory copy of the
                # canonical Branch seed. It must not inherit an old same-ordinal cache.
                reset_persistent_scope = previous_runtime_state_digest is not None

        attempt_root = self._root / str(request.attempt_id)
        attempt_root.mkdir(mode=0o700, exist_ok=True)
        root = attempt_root / f"run-{uuid4().hex}"
        root.mkdir(mode=0o700)

        persistent_state: Path | None = None
        persistent_lock: Path | None = None
        if lineage.ephemeral_agent_state:
            initialize_reusable_agent_state(
                root, self._artifacts.verify(revision.optimizer_digest).payload_path
            )
        else:
            state_trajectory = attempt.trajectory_ordinal
            if (
                epoch.number == 1
                and lineage.first_epoch_same_agent
                and attempt.branch is BranchRole.CHALLENGER
            ):
                # One Agent ID owns two independent Branches in this Epoch. Reserve
                # the second range of its state slots for the replica, not Active's cache.
                state_trajectory += epoch.trajectories_per_branch
            persistent_state, persistent_lock = self._persistent_root(
                lineage_id=lineage.id,
                revision=revision,
                trajectory_ordinal=state_trajectory,
                previous_runtime_state_digest=previous_runtime_state_digest,
                reset_from_seed=reset_persistent_scope,
                bootstrap_seed=self._shared_bootstrap_seed(lineage),
            )
            copy_reusable_agent_state(persistent_state, root)

        manifest = AttemptInputManifestV9(
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
        paths = ATTEMPT_WORKSPACE_LAYOUT
        self._artifacts.materialize(kernel.artifact_digest, root / paths.input_kernel)
        epoch_evidence = self._artifacts.verify(request.epoch_evidence_checkpoint)
        if epoch_evidence.kind is not ArtifactKind.EVIDENCE:
            raise ValueError("Attempt epoch Evidence has the wrong artifact kind")
        attempt_evidence = self._artifacts.verify(request.attempt_evidence_digest)
        if attempt_evidence.kind is not ArtifactKind.ATTEMPT_EVIDENCE:
            raise ValueError("Attempt branch Evidence has the wrong artifact kind")
        assemble_optimizer_evidence_view(
            root / paths.evidence,
            control_root=root / ".runtime",
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
        self._artifacts.materialize_file(
            visible_digest,
            "value.json",
            root / paths.agent_problem,
        )
        self._artifacts.materialize(revision.optimizer_digest, root / paths.optimizer)
        remove_optimizer_state_seeds(root / paths.optimizer)

        working_kernel = root / paths.working_kernel
        shutil.copytree(root / paths.input_kernel, working_kernel)
        make_tree_owner_writable(working_kernel)

        manifest_path = root / ATTEMPT_MANIFEST_RELATIVE_PATH
        manifest_path.parent.mkdir(mode=0o700, exist_ok=True)
        manifest_path.write_bytes(manifest.canonical_json_bytes())
        os.chmod(manifest_path, 0o400)
        session_root = root / "sessions"
        session_root.mkdir(mode=0o700)
        (root / "scratch").mkdir(mode=0o700)
        prepared = PreparedAttempt(
            root=root,
            manifest_path=manifest_path,
            session_root=session_root,
            session_id=f"attempt-session-{uuid4().hex}",
            persistent_state_root=persistent_state,
            persistent_lock_path=persistent_lock,
        )
        if attempt.input_runtime_state_digest is None:
            self._registry.record_attempt_input_runtime_state(
                attempt.id,
                prepared.seal_runtime_state(self._artifacts),
            )
        return prepared

    def _active_branch_seed(self, epoch: Epoch) -> ArtifactDigest | None:
        """Resolve the prior winner's best-Trajectory terminal State for Active."""
        if epoch.number <= 1:
            return None
        previous = self._registry.find_epoch(epoch.lineage_id, epoch.number - 1)
        if previous is None or previous.status is not EpochStatus.COMPLETED:
            raise ValueError("Previous Epoch is unavailable for Active Runtime State seeding")
        winner = previous.winner_kernel_agent_revision_id
        if winner is None or winner != epoch.active_kernel_agent_revision_id:
            raise ValueError("Active Agent disagrees with the previous Epoch winner")

        best_kernel_producer_attempt_id: str | None = None
        if previous.best_kernel_revision_id is not None:
            best_kernel = self._registry.get_kernel_revision(previous.best_kernel_revision_id)
            if best_kernel.produced_by_attempt_id is not None:
                best_kernel_producer_attempt_id = str(best_kernel.produced_by_attempt_id)

        projections: list[RuntimeStateAttempt] = []
        for item in self._registry.list_attempts(previous.id):
            latency: float | None = None
            if item.output_kernel_revision_id is not None:
                output = self._registry.get_kernel_revision(item.output_kernel_revision_id)
                latency = output.evaluation.latency_us
            projections.append(
                RuntimeStateAttempt(
                    attempt_id=str(item.id),
                    branch=item.branch.value,
                    challenger_ordinal=item.challenger_ordinal,
                    trajectory_ordinal=item.trajectory_ordinal,
                    ordinal=item.ordinal,
                    kernel_agent_revision_id=str(item.kernel_agent_revision_id),
                    accepted_as_branch_best=item.accepted_as_branch_best,
                    output_latency_us=latency,
                    input_runtime_state_digest=item.input_runtime_state_digest,
                    runtime_state_digest=item.runtime_state_digest,
                )
            )
        return select_winning_trajectory_terminal_state(
            attempts=tuple(projections),
            winner_revision_id=winner,
            active_revision_id=previous.active_kernel_agent_revision_id,
            challenger_revision_ids=previous.challenger_kernel_agent_revision_ids,
            best_kernel_producer_attempt_id=best_kernel_producer_attempt_id,
        )

    def _shared_bootstrap_seed(self, lineage: Lineage) -> tuple[object, object] | None:
        """Locate the Bootstrap deposit of the Lineage this one was cloned from."""
        source_id = lineage.bootstrap_source_lineage_id
        if source_id is None:
            return None
        revisions = self._registry.list_lineage_agent_revisions(source_id)
        root = next((entry for entry in revisions if entry.revision_number == 0), None)
        if root is None:
            return None
        return source_id, root.revision.id

    def _persistent_root(
        self,
        *,
        lineage_id: object,
        revision: KernelAgentRevision,
        trajectory_ordinal: int,
        previous_runtime_state_digest: ArtifactDigest | None = None,
        reset_from_seed: bool = False,
        bootstrap_seed: tuple[object, object] | None = None,
    ) -> tuple[Path, Path]:
        revision_id = revision.id
        parent_revision_id = revision.parent_id
        optimizer_source = self._artifacts.verify(revision.optimizer_digest).payload_path
        persistent = self._root / ".reusable"
        lock_path = persistent / ".lock"
        scope = (
            persistent / str(lineage_id) / str(revision_id) / f"trajectory-{trajectory_ordinal:08d}"
        )
        with _exclusive_lock(lock_path):
            if reset_from_seed and scope.exists():
                if previous_runtime_state_digest is None:
                    raise ValueError("Runtime State reset requires an explicit State seed")
                if scope.is_symlink() or not scope.is_dir():
                    raise ValueError(f"Reusable Agent scope is invalid: {scope}")
                shutil.rmtree(scope)
            if not scope.exists():
                scope.mkdir(parents=True, mode=0o700)
                source_scope: Path | None = None
                state_seeded = previous_runtime_state_digest is not None
                if previous_runtime_state_digest is not None:
                    state = self._artifacts.verify(previous_runtime_state_digest)
                    if state.kind is not ArtifactKind.KERNEL_AGENT_RUNTIME_STATE:
                        raise ValueError(
                            "Previous Attempt Runtime State has the wrong Artifact kind"
                        )
                    validate_reusable_agent_state_seed(state.payload_path)
                    source_scope = state.payload_path
                else:
                    state_seeded, source_scope = self._evolved_runtime_state_seed(
                        revision,
                    )
                if not state_seeded and parent_revision_id is not None:
                    source_scope = (
                        persistent
                        / str(lineage_id)
                        / str(parent_revision_id)
                        / f"trajectory-{trajectory_ordinal:08d}"
                    )
                elif not state_seeded:
                    seed_lineage_id, seed_revision_id = bootstrap_seed or (lineage_id, revision_id)
                    bootstrap = (
                        persistent / str(seed_lineage_id) / str(seed_revision_id) / "bootstrap"
                    )
                    if bootstrap.is_dir() and not bootstrap.is_symlink():
                        source_scope = bootstrap
                if source_scope is not None and source_scope.exists():
                    _validate_reusable_tree(source_scope)
                    copy_reusable_agent_state(
                        source_scope,
                        scope,
                        optimizer_source=optimizer_source,
                    )
                else:
                    initialize_reusable_agent_state(scope, optimizer_source)
            ensure_reusable_directories(scope, optimizer_source=optimizer_source)
        return scope, lock_path

    def _evolved_runtime_state_seed(
        self,
        revision: KernelAgentRevision,
    ) -> tuple[bool, Path | None]:
        """Resolve an Evolver-sealed state seed; legacy revisions fall back to Parent state."""
        state = resolve_revision_runtime_state_seed(self._artifacts, revision)
        return (state is not None), state


@contextmanager
def _exclusive_lock(path: Path) -> Iterator[None]:
    """Hold the cross-process lock protecting reusable Workspace state."""
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with path.open("a+b") as file:
        fcntl.flock(file, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(file, fcntl.LOCK_UN)


def _validate_reusable_tree(root: Path) -> None:
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"Reusable Workspace directory is invalid: {root}")
    for path in root.rglob("*"):
        mode = path.lstat().st_mode
        if not (stat.S_ISDIR(mode) or stat.S_ISREG(mode)):
            raise ValueError(
                f"Reusable Workspace entry must be a regular file or directory: {path}"
            )


def _copy_reusable_tree(source: Path, destination: Path) -> None:
    _validate_reusable_tree(source)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    shutil.copytree(source, destination)


def _replace_reusable_tree(
    source: Path,
    destination: Path,
) -> None:
    _validate_reusable_tree(source)
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    staging = destination.parent / f".{destination.name}.next-{uuid4().hex}"
    backup = destination.parent / f".{destination.name}.previous-{uuid4().hex}"
    try:
        shutil.copytree(source, staging)
        destination.rename(backup)
        staging.rename(destination)
    except BaseException:
        if backup.exists() and not destination.exists():
            backup.rename(destination)
        raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)
        shutil.rmtree(backup, ignore_errors=True)
