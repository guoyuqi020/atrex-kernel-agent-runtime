"""Private-input Core session that derives one public Agent Problem Artifact."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Self
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..artifacts.local import ArtifactKind, JsonValue, LocalArtifactStore
from ..domain.ids import ArtifactDigest, new_worker_session_id, parse_artifact_digest
from ..domain.models import (
    Dsl,
    TokenUsage,
    WorkerSession,
    WorkerSessionRole,
    WorkerSessionStatus,
)
from ..gateway.contract import AgateEvaluationContractV1
from ..ports import WorkerSessionRecorder
from .core import CoreOptimizerProcessConfig
from .core_phase import CorePhaseRunner
from .launcher import WorkerLauncher

PROBLEM_GENERALIZATION_MANIFEST_VERSION: Literal[1] = 1
_PRIVATE_KEYS = frozenset(
    {"reference_py", "input_py", "shapes", "shape_ids", "metadata", "roofline"}
)
_RUNTIME_KEYS = {
    "ATREX_AGENT_BACKEND",
    "ATREX_AGENT_MODEL",
    "ATREX_AGENT_REASONING_EFFORT",
    "ATREX_AGENT_SESSION_SETTINGS",
    "ATREX_AGENT_PROBLEM_OUTPUT",
    "ATREX_CORE_PHASE",
    "ATREX_OPTIMIZER_REPOSITORY",
    "ATREX_PROBLEM_GENERALIZATION_MANIFEST",
    "ATREX_SESSION_TIMEOUT_SECONDS",
    "ATREX_SESSION_TRACE_PATH",
    "ATREX_USAGE_BUDGET",
    "ATREX_USAGE_UNIT",
    "ATREX_TOKEN_USAGE_REPORT",
}


class ProblemGeneralizationPathsV1(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    private_inputs: Literal["input/private"] = "input/private"
    output: Literal["work/output"] = "work/output"
    optimizer: Literal["agent/optimizer"] = "agent/optimizer"


class ProblemGeneralizationManifestV1(BaseModel):
    """Task identity passed to a Core problem-authoring session."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    schema_version: Literal[1] = PROBLEM_GENERALIZATION_MANIFEST_VERSION
    generalization_id: str = Field(min_length=1)
    optimizer_digest: ArtifactDigest
    evaluation_contract_digest: ArtifactDigest
    dsl: Dsl
    operator: str = Field(min_length=1)
    hardware_target: str = Field(min_length=1)
    paths: ProblemGeneralizationPathsV1 = ProblemGeneralizationPathsV1()

    @field_validator("optimizer_digest", "evaluation_contract_digest", mode="before")
    @classmethod
    def _digest(cls, value: object) -> ArtifactDigest:
        if not isinstance(value, str):
            raise ValueError("problem generalization digest must be a string")
        return parse_artifact_digest(value)

    def canonical_json_bytes(self) -> bytes:
        return json.dumps(
            self.model_dump(mode="json"),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()


class PublicEvaluationV1(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    exact_cases: Literal["private"]
    correctness_requirement: str = Field(min_length=1)
    performance_requirement: str = Field(min_length=1)
    development_cases_are_evaluation_cases: Literal[False]


class AgentProblemV1(BaseModel):
    """Structurally safe public problem contract emitted by Core."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    schema_version: Literal["atrex.agent_problem.v1"]
    objective: str = Field(min_length=1)
    evaluation: PublicEvaluationV1
    operator_contract: dict[str, JsonValue]
    workload_profile: dict[str, JsonValue] = Field(default_factory=dict)
    distribution_profile: dict[str, JsonValue] = Field(default_factory=dict)
    shape_domain: dict[str, JsonValue]
    invariants: tuple[str, ...]
    coverage_regimes: tuple[dict[str, JsonValue], ...]
    development_cases: tuple[dict[str, JsonValue], ...] = ()

    @field_validator("objective")
    @classmethod
    def _hidden_cases_are_explicit(cls, value: str) -> str:
        lowered = value.lower()
        if "hidden" not in lowered and "private" not in lowered:
            raise ValueError("Agent Problem objective must state that exact cases are hidden")
        return value

    @field_validator("invariants")
    @classmethod
    def _invariants_are_nonempty(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value or any(not item.strip() for item in value):
            raise ValueError("Agent Problem invariants must contain non-empty text")
        return value

    @field_validator("development_cases")
    @classmethod
    def _development_cases_are_synthetic_inputs(
        cls,
        value: tuple[dict[str, JsonValue], ...],
    ) -> tuple[dict[str, JsonValue], ...]:
        for case in value:
            if not isinstance(case.get("input_kwargs"), dict) or not isinstance(
                case.get("init_kwargs"), (dict, type(None))
            ):
                raise ValueError(
                    "Agent Problem development cases require input_kwargs and optional init_kwargs"
                )
        return value

    @classmethod
    def from_file(
        cls,
        path: Path,
        *,
        private_shapes: dict[str, JsonValue],
        max_bytes: int,
    ) -> Self:
        if path.is_symlink() or not path.is_file():
            raise ValueError("Agent Problem output must be a regular file")
        if path.stat().st_size > max_bytes:
            raise ValueError("Agent Problem output exceeds byte limit")
        problem = cls.model_validate_json(path.read_bytes())
        return cls._validate_visibility(problem, private_shapes=private_shapes)

    @classmethod
    def from_value(
        cls,
        value: object,
        *,
        private_shapes: dict[str, JsonValue],
    ) -> Self:
        """Validate a supplied public problem against the sealed evaluator cases."""
        return cls._validate_visibility(
            cls.model_validate(value),
            private_shapes=private_shapes,
        )

    @classmethod
    def _validate_visibility(
        cls,
        problem: Self,
        *,
        private_shapes: dict[str, JsonValue],
    ) -> Self:
        value = problem.model_dump(mode="json")
        cls._reject_private_keys(value)
        private_nodes = {
            cls._canonical(shape)
            for shape in private_shapes.values()
            if isinstance(shape, (dict, list)) and shape
        }
        for node in cls._nodes(value):
            if isinstance(node, (dict, list)) and node and cls._canonical(node) in private_nodes:
                raise ValueError("Agent Problem contains an exact private evaluator case")
        private_cases = {
            signature
            for shape in private_shapes.values()
            if isinstance(shape, dict) and (signature := cls._case_signature(shape)) is not None
        }
        for index, case in enumerate(problem.development_cases):
            signature = cls._case_signature(case)
            if signature is not None and signature in private_cases:
                raise ValueError(
                    f"Agent Problem development case {index} duplicates a private evaluator case"
                )
        return problem

    @classmethod
    def _reject_private_keys(cls, value: JsonValue) -> None:
        for node in cls._nodes(value):
            if isinstance(node, dict):
                leaked = _PRIVATE_KEYS.intersection(key.lower() for key in node)
                if leaked:
                    raise ValueError(f"Agent Problem contains private fields: {sorted(leaked)}")

    @classmethod
    def _nodes(cls, value: JsonValue) -> tuple[JsonValue, ...]:
        nodes: list[JsonValue] = [value]
        for node in nodes:
            if isinstance(node, dict):
                nodes.extend(node.values())
            elif isinstance(node, list):
                nodes.extend(node)
        return tuple(nodes)

    @staticmethod
    def _canonical(value: JsonValue) -> bytes:
        return json.dumps(
            value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
        ).encode()

    @classmethod
    def _case_signature(cls, value: dict[str, JsonValue]) -> bytes | None:
        input_kwargs = value.get("input_kwargs")
        init_kwargs = value.get("init_kwargs")
        if not isinstance(input_kwargs, dict) or not isinstance(init_kwargs, (dict, type(None))):
            return None
        return cls._canonical(
            {
                "init_kwargs": init_kwargs,
                "input_kwargs": input_kwargs,
            }
        )


@dataclass(frozen=True, slots=True)
class PreparedProblemGeneralization:
    root: Path
    manifest_path: Path
    output_path: Path
    session_root: Path
    private_shapes: dict[str, JsonValue]


class ProblemGeneralizationWorkspaceAssembler:
    """Expose private evaluator inputs only inside one dedicated Core workspace."""

    def __init__(self, root: str | Path, artifacts: LocalArtifactStore) -> None:
        self._root = Path(root).resolve()
        self._artifacts = artifacts
        self._root.mkdir(parents=True, exist_ok=True, mode=0o700)

    def prepare(
        self,
        manifest: ProblemGeneralizationManifestV1,
    ) -> PreparedProblemGeneralization:
        contract_artifact = self._artifacts.verify(manifest.evaluation_contract_digest)
        if contract_artifact.kind is not ArtifactKind.EVALUATION_CONTRACT:
            raise ValueError("problem generalization requires an Evaluation Contract")
        optimizer = self._artifacts.verify(manifest.optimizer_digest)
        if optimizer.kind is not ArtifactKind.KERNEL_AGENT:
            raise ValueError("problem generalization requires a Kernel Agent revision")
        contract = AgateEvaluationContractV1.model_validate_json(
            (contract_artifact.payload_path / "value.json").read_bytes()
        )
        root = self._root / manifest.generalization_id / f"run-{uuid4().hex}"
        private = root / manifest.paths.private_inputs
        output = root / manifest.paths.output
        private.mkdir(parents=True, mode=0o700)
        output.mkdir(parents=True, mode=0o700)
        self._write_private(private / "reference.py", contract.reference_py)
        self._write_private(private / "input.py", contract.input_py)
        self._write_json(private / "shapes.json", contract.shapes)
        if contract.metadata is not None:
            self._write_json(private / "metadata.json", contract.metadata)
        if contract.roofline is not None:
            self._write_json(private / "roofline.json", contract.roofline)
        self._artifacts.materialize(manifest.optimizer_digest, root / manifest.paths.optimizer)
        manifest_path = root / "problem-generalization.json"
        manifest_path.write_bytes(manifest.canonical_json_bytes())
        os.chmod(manifest_path, 0o400)
        session_root = root / "sessions"
        session_root.mkdir(mode=0o700)
        (root / "scratch").mkdir(mode=0o700)
        return PreparedProblemGeneralization(
            root,
            manifest_path,
            output / "agent_problem.json",
            session_root,
            contract.shapes,
        )

    @staticmethod
    def _write_private(path: Path, value: str) -> None:
        path.write_text(value, encoding="utf-8")
        os.chmod(path, 0o400)

    @staticmethod
    def _write_json(path: Path, value: JsonValue) -> None:
        path.write_text(
            json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True),
            encoding="utf-8",
        )
        os.chmod(path, 0o400)


@dataclass(frozen=True, slots=True)
class ProblemGeneralizationSessionResult:
    finish_reason: str
    final_response: str
    token_usage: TokenUsage
    token_budget: int
    problem: AgentProblemV1 | None
    problem_digest: ArtifactDigest | None
    problem_error: str | None
    session_trace_digest: ArtifactDigest | None


class CoreProblemGeneralizationSessionDriver:
    """Run Core without Gateway/Wiki authority and validate its only output."""

    def __init__(
        self,
        launcher: WorkerLauncher,
        config: CoreOptimizerProcessConfig,
        artifacts: LocalArtifactStore,
        *,
        max_problem_bytes: int,
    ) -> None:
        if max_problem_bytes <= 0:
            raise ValueError("Agent Problem byte limit must be positive")
        self._config = config
        self._artifacts = artifacts
        self._max_problem_bytes = max_problem_bytes
        self._phases = CorePhaseRunner(launcher, config, artifacts)

    def run(
        self,
        prepared: PreparedProblemGeneralization,
        environment_values: tuple[tuple[str, str], ...],
        model: str | None = None,
    ) -> ProblemGeneralizationSessionResult:
        environment = dict(environment_values)
        if len(environment) != len(environment_values):
            raise ValueError("problem generalization environment contains duplicate keys")
        overlap = _RUNTIME_KEYS.intersection(environment)
        if overlap:
            raise ValueError(
                f"problem generalization environment overrides Runtime keys: {overlap}"
            )
        phase = self._phases.prepare(prepared.root, prepared.session_root)
        environment.update(
            self._phases.runtime_environment(
                phase,
                phase="problem_generalization",
                model=model,
            )
        )
        environment.update(
            {
                "ATREX_AGENT_PROBLEM_OUTPUT": str(prepared.output_path),
                "ATREX_PROBLEM_GENERALIZATION_MANIFEST": str(prepared.manifest_path),
            }
        )
        result = self._phases.run(
            phase,
            environment,
            label="Core problem generalization",
        )
        problem = None
        problem_digest = None
        problem_error = None
        if prepared.output_path.exists() or prepared.output_path.is_symlink():
            try:
                output_entries = tuple(prepared.output_path.parent.iterdir())
                if output_entries != (prepared.output_path,):
                    raise ValueError("problem generalization produced unexpected output files")
                problem = AgentProblemV1.from_file(
                    prepared.output_path,
                    private_shapes=prepared.private_shapes,
                    max_bytes=self._max_problem_bytes,
                )
            except ValueError as error:
                problem_error = str(error)
            else:
                problem_digest = self._artifacts.put_json(
                    problem.model_dump(mode="json"), ArtifactKind.AGENT_PROBLEM
                )
        return ProblemGeneralizationSessionResult(
            result.finish_reason,
            result.process.stdout,
            result.token_usage.to_domain(),
            int(result.token_usage.require_budget()),
            problem,
            problem_digest,
            problem_error,
            result.session_trace_digest,
        )


class CoreAgentProblemGenerator:
    """Compose workspace and Core process into the Bootstrap generator contract."""

    def __init__(
        self,
        workspaces: ProblemGeneralizationWorkspaceAssembler,
        sessions: CoreProblemGeneralizationSessionDriver,
        environment: tuple[tuple[str, str], ...],
        *,
        worker_sessions: WorkerSessionRecorder | None = None,
        backend: str | None = None,
    ) -> None:
        self._workspaces = workspaces
        self._sessions = sessions
        self._environment = environment
        self._worker_sessions = worker_sessions
        self._backend = backend

    def generate(
        self,
        *,
        generalization_id: str,
        optimizer_digest: ArtifactDigest,
        evaluation_contract_digest: ArtifactDigest,
        dsl: Dsl,
        operator: str,
        hardware_target: str,
        model: str | None,
    ) -> ArtifactDigest:
        prepared = self._workspaces.prepare(
            ProblemGeneralizationManifestV1(
                generalization_id=generalization_id,
                optimizer_digest=optimizer_digest,
                evaluation_contract_digest=evaluation_contract_digest,
                dsl=dsl,
                operator=operator,
                hardware_target=hardware_target,
            )
        )
        worker_session_id = new_worker_session_id()
        if self._worker_sessions is not None:
            self._worker_sessions.start_worker_session(
                WorkerSession(
                    id=worker_session_id,
                    role=WorkerSessionRole.PROBLEM_GENERALIZATION,
                    subject_id=generalization_id,
                    external_run_id=prepared.root.name,
                    workspace_path=str(prepared.root),
                    status=WorkerSessionStatus.RUNNING,
                    started_at=datetime.now(UTC).isoformat(),
                    backend=self._backend,
                    model=model,
                )
            )
        try:
            result = self._sessions.run(prepared, self._environment, model)
        except BaseException as error:
            if self._worker_sessions is not None:
                timed_out = "wall-time limit" in str(error)
                self._worker_sessions.finish_worker_session(
                    worker_session_id,
                    status=(
                        WorkerSessionStatus.TIMED_OUT if timed_out else WorkerSessionStatus.FAILED
                    ),
                    finish_reason="timeout" if timed_out else "infrastructure-failed",
                    error_type=type(error).__name__,
                    error_message=str(error),
                )
            raise
        if self._worker_sessions is not None:
            self._worker_sessions.finish_worker_session(
                worker_session_id,
                status=(
                    WorkerSessionStatus.COMPLETED
                    if result.finish_reason == "completed"
                    else WorkerSessionStatus.FAILED
                ),
                finish_reason=result.finish_reason,
                trace_digest=result.session_trace_digest,
                token_budget=result.token_budget,
                token_usage=result.token_usage,
            )
        if result.finish_reason != "completed":
            raise RuntimeError(
                f"Core problem generalization did not complete: {result.finish_reason}"
            )
        if result.problem_digest is None:
            detail = result.problem_error or "Core produced no Agent Problem"
            raise ValueError(f"Core problem generalization was rejected: {detail}")
        return result.problem_digest
