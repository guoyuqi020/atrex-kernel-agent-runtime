"""Problem Generalization participation in the unified Worker session catalog."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from conftest import NOW, digest

from atrex_runtime.domain.models import Dsl, TokenUsage, WorkerSessionStatus
from atrex_runtime.registry.sqlite import SqliteRegistry
from atrex_runtime.workers.problem_generalization import (
    CoreAgentProblemGenerator,
    PreparedProblemGeneralization,
    ProblemGeneralizationManifestV1,
    ProblemGeneralizationSessionResult,
)


@dataclass
class FakeWorkspaces:
    prepared: PreparedProblemGeneralization

    def prepare(self, _manifest: ProblemGeneralizationManifestV1) -> PreparedProblemGeneralization:
        return self.prepared


@dataclass
class FakeSessions:
    result: ProblemGeneralizationSessionResult

    def run(
        self,
        _prepared: PreparedProblemGeneralization,
        _environment: tuple[tuple[str, str], ...],
        _model: str | None,
    ) -> ProblemGeneralizationSessionResult:
        return self.result


def test_problem_generalization_records_completed_session_before_campaign_exists(
    tmp_path: Path,
) -> None:
    root = tmp_path / "generalization-1" / "run-1"
    prepared = PreparedProblemGeneralization(
        root=root,
        manifest_path=root / "manifest.json",
        output_path=root / "output.json",
        session_root=root / "sessions",
        private_shapes={},
    )
    trace = digest("generalization-raw-trace")
    result = ProblemGeneralizationSessionResult(
        finish_reason="completed",
        final_response="done",
        token_usage=TokenUsage(40, 10, 0, 0),
        token_budget=20_000_000,
        problem=None,
        problem_digest=digest("agent-problem"),
        problem_error=None,
        session_trace_digest=trace,
    )
    with SqliteRegistry(tmp_path / "registry.sqlite", clock=lambda: NOW) as registry:
        generator = CoreAgentProblemGenerator(
            FakeWorkspaces(prepared),  # type: ignore[arg-type]
            FakeSessions(result),  # type: ignore[arg-type]
            (),
            worker_sessions=registry,
            backend="codex",
        )

        assert generator.generate(
            generalization_id="generalization-1",
            optimizer_digest=digest("optimizer"),
            evaluation_contract_digest=digest("contract"),
            dsl=Dsl.TRITON,
            operator="vector_add",
            hardware_target="nvidia-h100",
            model="gpt-5.6-codex",
        ) == digest("agent-problem")
        sessions = registry.list_worker_sessions(subject_id="generalization-1")

    assert len(sessions) == 1
    assert sessions[0].status is WorkerSessionStatus.COMPLETED
    assert sessions[0].campaign_id is None
    assert sessions[0].trace_digest == trace
    assert sessions[0].token_budget == 20_000_000
