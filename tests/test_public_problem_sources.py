"""Agent-visible public problem source projection tests."""

from __future__ import annotations

from pathlib import Path

from atrex_runtime.artifacts.local import ArtifactKind, LocalArtifactStore
from atrex_runtime.workers.public_problem import materialize_public_problem_sources


def test_evaluation_sources_stay_private_without_an_explicit_declaration(
    tmp_path: Path,
) -> None:
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    contract = artifacts.put_json(
        {
            "schema_version": 1,
            "candidate_path": "kernel.py",
            "reference_py": "SECRET_REFERENCE = True\n",
            "input_py": "SECRET_INPUT = True\n",
            "shapes": {"private-id": {"input_kwargs": {"m": 1}}},
            "metadata": {"description": "private by default"},
            "options": {
                "num_correctness_cases": 1,
                "bench_iters": 1,
                "atol": 0.01,
                "rtol": 0.05,
                "timeout_s": 60,
            },
        },
        ArtifactKind.EVALUATION_CONTRACT,
    )
    problem = artifacts.put_json(
        {"schema_version": "atrex.agent_problem.v1", "objective": "private task"},
        ArtifactKind.AGENT_PROBLEM,
    )
    destination = tmp_path / "workspace/input/problem"

    assert (
        materialize_public_problem_sources(
            destination,
            artifacts,
            evaluation_contract_digest=contract,
            agent_problem_digest=problem,
        )
        == ()
    )
    assert not destination.exists()
