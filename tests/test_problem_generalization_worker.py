"""Private Core problem-generalization protocol tests."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from atrex_runtime.artifacts.local import ArtifactKind, LocalArtifactStore
from atrex_runtime.domain.models import Dsl, TokenUsage
from atrex_runtime.workers import (
    AgentProblemV1,
    CleanEnvironmentLauncher,
    CoreOptimizerProcessConfig,
    CoreProblemGeneralizationSessionDriver,
    ProblemGeneralizationManifestV1,
    ProblemGeneralizationWorkspaceAssembler,
)


def _contract() -> dict[str, object]:
    return {
        "schema_version": 1,
        "candidate_path": "kernel.py",
        "reference_py": "class Model:\n    pass\n",
        "input_py": "def get_inputs():\n    return ()\n",
        "shapes": {"private-0": {"m": 17, "n": 31}, "private-1": {"m": 33, "n": 63}},
        "metadata": {"dtype": "bf16"},
        "options": {
            "num_correctness_cases": 2,
            "bench_iters": 10,
            "atol": 0.001,
            "rtol": 0.01,
            "timeout_s": 60,
        },
        "lock_clocks": True,
    }


def _write_agent(root: Path) -> None:
    (root / "src").mkdir(parents=True)
    (root / "atrex-bundle.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "bundle_format": "atrex-kernel-agent-bundle-v1",
                "entrypoint": {"command": "src/run.py"},
            }
        ),
        encoding="utf-8",
    )
    (root / "src/run.py").write_text(
        """import json
import os
from pathlib import Path

assert os.environ["ATREX_CORE_PHASE"] == "problem_generalization"
private = Path("input/private")
assert (private / "reference.py").is_file()
assert (private / "shapes.json").is_file()
problem = {
    "schema_version": "atrex.agent_problem.v1",
    "objective": "Implement vector add while exact evaluator cases remain hidden",
    "evaluation": {
        "exact_cases": "private",
        "correctness_requirement": "every hidden case must pass",
        "performance_requirement": "measure after correctness",
        "development_cases_are_evaluation_cases": False
    },
    "operator_contract": {"operation": "vector_add"},
    "workload_profile": {},
    "distribution_profile": {},
    "shape_domain": {"m": {"min": 1}, "n": {"min": 1}},
    "invariants": ["dispatch uses runtime dimensions"],
    "coverage_regimes": [],
    "development_cases": []
}
Path(os.environ["ATREX_AGENT_PROBLEM_OUTPUT"]).write_text(json.dumps(problem))
trace = Path("sessions/core")
trace.mkdir()
(trace / "events.jsonl").write_text('{"event":"turn-end"}\\n')
budget = float(os.environ["ATREX_USAGE_BUDGET"])
Path(os.environ["ATREX_TOKEN_USAGE_REPORT"]).write_text(json.dumps({
    "schema_version": 2,
    "usage_unit": os.environ["ATREX_USAGE_UNIT"],
    "budget": budget,
    "consumed": 14,
    "token_usage": {
        "uncached_input_tokens": 8,
        "output_tokens": 4,
        "cache_read_tokens": 2,
        "cache_write_tokens": 0
    },
    "credits": None,
    "budget_exhausted": False,
    "session_count": 1,
    "model_request_count": 1,
    "usage_complete": True
}))
print("problem generalized")
""",
        encoding="utf-8",
    )


def test_problem_generalization_runs_without_gateway_and_seals_public_problem(
    tmp_path: Path,
) -> None:
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    contract_digest = artifacts.put_json(_contract(), ArtifactKind.EVALUATION_CONTRACT)
    agent = tmp_path / "agent"
    agent.mkdir()
    _write_agent(agent)
    agent_digest = artifacts.put_directory(agent, ArtifactKind.KERNEL_AGENT)
    manifest = ProblemGeneralizationManifestV1(
        generalization_id="generalize-vector-add-triton",
        optimizer_digest=agent_digest,
        evaluation_contract_digest=contract_digest,
        dsl=Dsl.TRITON,
        operator="vector_add",
        hardware_target="nvidia-h100",
    )
    prepared = ProblemGeneralizationWorkspaceAssembler(tmp_path / "workspaces", artifacts).prepare(
        manifest
    )
    assert (prepared.root / "input/private/shapes.json").is_file()
    assert not prepared.output_path.exists()
    driver = CoreProblemGeneralizationSessionDriver(
        CleanEnvironmentLauncher(Path("/usr/bin/env")),
        CoreOptimizerProcessConfig(
            agent_backend="claude",
            command_prefix=(sys.executable,),
            isolated_home_environment_keys=("HOME",),
            session_trace_relative_path="sessions/core",
            token_usage_report_relative_path="scratch/token-usage.json",
            max_attempt_report_bytes=65_536,
            timeout_seconds=10,
            terminate_grace_seconds=1,
            max_diagnostic_bytes=4096,
            max_session_tokens=1000,
        ),
        artifacts,
        max_problem_bytes=65_536,
    )

    result = driver.run(prepared, ())

    assert result.finish_reason == "completed"
    assert result.final_response == "problem generalized\n"
    assert result.token_usage == TokenUsage(8, 4, 2, 0)
    assert result.problem is not None
    assert result.problem_digest is not None
    assert result.problem_error is None
    assert artifacts.verify(result.problem_digest).kind is ArtifactKind.AGENT_PROBLEM
    assert result.session_trace_digest is not None


@pytest.mark.parametrize(
    "mutation, message",
    [
        ({"shape_domain": {"shapes": {"private-0": {"m": 17}}}}, "private fields"),
        (
            {"development_cases": [{"init_kwargs": None, "input_kwargs": {"m": 17, "n": 31}}]},
            "exact private evaluator case",
        ),
    ],
)
def test_agent_problem_rejects_private_structure(
    tmp_path: Path,
    mutation: dict[str, object],
    message: str,
) -> None:
    value: dict[str, object] = {
        "schema_version": "atrex.agent_problem.v1",
        "objective": "Exact evaluator cases are hidden",
        "evaluation": {
            "exact_cases": "private",
            "correctness_requirement": "all hidden cases pass",
            "performance_requirement": "measure after correctness",
            "development_cases_are_evaluation_cases": False,
        },
        "operator_contract": {},
        "workload_profile": {},
        "distribution_profile": {},
        "shape_domain": {},
        "invariants": ["runtime dispatch only"],
        "coverage_regimes": [],
        "development_cases": [],
    }
    value.update(mutation)
    path = tmp_path / "agent_problem.json"
    path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        AgentProblemV1.from_file(
            path,
            private_shapes={"private-0": {"init_kwargs": None, "input_kwargs": {"m": 17, "n": 31}}},
            max_bytes=65_536,
        )
