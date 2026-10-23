"""Read-only public operator sources for managed Agent workspaces."""

from __future__ import annotations

import json
import os
from pathlib import Path

from ..artifacts.local import ArtifactKind, JsonValue, LocalArtifactStore
from ..domain.ids import ArtifactDigest
from ..gateway.contract import AgateEvaluationContractV1
from ..serialization import canonical_json_bytes

PUBLIC_PROBLEM_RELATIVE_PATH = Path("input/problem")

_PUBLIC_SOURCE_FILES = frozenset({"reference.py", "input.py", "metadata.json"})
_PUBLIC_METADATA_KEYS = frozenset(
    {
        "schema_version",
        "display_name",
        "description",
        "operator_family",
        "performance_objective",
        "real_inputs",
        "tags",
        "category",
        "dtype",
        "input_dtypes",
        "output_dtypes",
    }
)
_PUBLIC_BENCHMARK_CONTRACT_KEYS = frozenset(
    {
        "correctness_tolerances",
        "defined_outputs",
        "iteration_protocol",
        "mutates_inputs",
        "preserve_input_storage",
        "scratch_inputs",
    }
)


def materialize_public_problem_sources(
    destination: Path,
    artifacts: LocalArtifactStore,
    *,
    evaluation_contract_digest: ArtifactDigest,
    agent_problem_digest: ArtifactDigest,
) -> tuple[str, ...]:
    """Expose explicitly public sources without publishing evaluator Shape cases.

    The Agent Problem is the authority that decides whether reference/input sources are
    public. Evaluation Contracts remain private by default. When an Agent Problem names
    one of the supported files under ``authoritative_public_sources``, this function
    materializes that source plus an optional privacy-filtered metadata projection.
    """
    problem_artifact = artifacts.verify(agent_problem_digest)
    if problem_artifact.kind is not ArtifactKind.AGENT_PROBLEM:
        raise ValueError("Public problem sources require an Agent Problem")
    problem = json.loads((problem_artifact.payload_path / "value.json").read_bytes())
    declared = _declared_public_sources(problem)
    if not declared:
        return ()

    contract_artifact = artifacts.verify(evaluation_contract_digest)
    if contract_artifact.kind is not ArtifactKind.EVALUATION_CONTRACT:
        raise ValueError("Public problem sources require an Evaluation Contract")
    contract = AgateEvaluationContractV1.model_validate_json(
        (contract_artifact.payload_path / "value.json").read_bytes()
    )

    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    destination.mkdir(parents=True, mode=0o700)
    written: list[str] = []
    if "reference.py" in declared:
        _write_read_only(destination / "reference.py", contract.reference_py.encode("utf-8"))
        written.append("reference.py")
    if "input.py" in declared:
        _write_read_only(destination / "input.py", contract.input_py.encode("utf-8"))
        written.append("input.py")

    metadata = _public_metadata(contract.metadata)
    if metadata is not None and (declared & _PUBLIC_SOURCE_FILES):
        _write_read_only(destination / "metadata.json", canonical_json_bytes(metadata))
        written.append("metadata.json")

    readme = (
        "# Public problem sources\n\n"
        "Runtime copied these task-declared public sources from the sealed Evaluation "
        "Contract. Read them locally when reconstructing the operator contract. They do "
        "not contain exact evaluator Shape cases, case IDs, per-case weights, or private "
        "evaluation results. Do not use Dev jobs or evaluator temporary directories to "
        "recover alternate copies of these files.\n\n"
        f"Available files: {', '.join(written) if written else 'none'}.\n"
    )
    _write_read_only(destination / "README.md", readme.encode("utf-8"))
    written.append("README.md")
    os.chmod(destination, 0o500)
    return tuple(written)


def _declared_public_sources(problem: object) -> frozenset[str]:
    if not isinstance(problem, dict):
        return frozenset()
    operator_contract = problem.get("operator_contract")
    if not isinstance(operator_contract, dict):
        return frozenset()
    sources = operator_contract.get("authoritative_public_sources")
    if not isinstance(sources, dict):
        return frozenset()
    return frozenset(name for name in sources if name in _PUBLIC_SOURCE_FILES)


def _public_metadata(metadata: dict[str, JsonValue] | None) -> dict[str, JsonValue] | None:
    if metadata is None:
        return None
    public = {key: value for key, value in metadata.items() if key in _PUBLIC_METADATA_KEYS}
    benchmark = metadata.get("benchmark_contract")
    if isinstance(benchmark, dict):
        public_benchmark = {
            key: value for key, value in benchmark.items() if key in _PUBLIC_BENCHMARK_CONTRACT_KEYS
        }
        if public_benchmark:
            public["benchmark_contract"] = public_benchmark
    return public or None


def _write_read_only(path: Path, content: bytes) -> None:
    path.write_bytes(content)
    os.chmod(path, 0o400)
