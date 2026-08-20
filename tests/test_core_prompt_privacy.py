"""Ensure Agent-visible instructions do not reveal implementation identity."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

CORE_ROOT = Path(__file__).resolve().parents[1] / "src" / "atrex-kernel-agent-core"
sys.path.insert(0, str(CORE_ROOT / "src"))

from agent_config import AgentConfig  # noqa: E402
from sessions import attempt, lineage_bootstrap, problem_generalization  # noqa: E402

_IDENTITY_LEAK = re.compile(
    r"atrex|kernel agent|atrex runtime|runtime-owned|runtime-injected|runtime workflow|"
    r"core revision|core operation|preserves the upstream|replaces? the upstream|"
    r"upstream concept|bundle|control plane",
    re.IGNORECASE,
)
_INTERNAL_CONTEXT_FIELDS = {
    "attempt_id",
    "kernel_agent_revision_id",
    "input_kernel_revision_id",
    "campaign_id",
    "lineage_id",
    "epoch_id",
    "branch",
    "evaluation_contract_digest",
    "agent_problem_digest",
    "bootstrap_attempt_id",
    "optimizer_digest",
    "generalization_id",
}


def _rendered_prompts() -> dict[str, str]:
    config = AgentConfig.load(CORE_ROOT)
    return {
        "optimization_attempt": attempt.render_prompt(
            SimpleNamespace(
                evidence_prompt="# Evidence input\n\nInjected structure.\n",
                manifest={
                    "dsl": "cuda",
                    "context": {
                        "epoch_number": 1,
                        "attempt_ordinal": 1,
                        "operator": "example",
                        "hardware_target": "gpu",
                    },
                },
            ),
            config,
        ),
        "framework_baseline": lineage_bootstrap.render_prompt(
            SimpleNamespace(
                manifest={
                    "dsl": "cuda",
                    "operator": "example",
                    "hardware_target": "gpu",
                }
            ),
            config,
        ),
        "problem_generalization": problem_generalization.render_prompt(
            SimpleNamespace(
                manifest={
                    "dsl": "cuda",
                    "operator": "example",
                    "hardware_target": "gpu",
                }
            ),
            config,
        ),
    }


def test_rendered_prompts_do_not_reveal_implementation_identity() -> None:
    documents = _rendered_prompts()

    leaks = {
        name: match.group(0)
        for name, text in documents.items()
        if (match := _IDENTITY_LEAK.search(text)) is not None
    }

    assert leaks == {}


def test_rendered_context_omits_internal_control_identifiers() -> None:
    for prompt in _rendered_prompts().values():
        assert all(field not in prompt for field in _INTERNAL_CONTEXT_FIELDS)


def test_optimizer_prompt_contains_the_controller_injected_evidence_fragment() -> None:
    assert "# Evidence input" in _rendered_prompts()["optimization_attempt"]


def test_rendered_prompts_use_exact_cli_subcommands_and_valid_json_examples() -> None:
    prompts = _rendered_prompts()
    attempt_prompt = prompts["optimization_attempt"]
    baseline_prompt = prompts["framework_baseline"]

    for command in (
        "gateway-execute",
        "wiki-query",
        "record-experiment",
        "attempt-report",
    ):
        assert f"python agent/optimizer/src/runtime_tools.py {command} --request" in attempt_prompt
    assert "lineage-bootstrap-report" in baseline_prompt
    assert "python agent/optimizer/src/runtime_tools.py lineage-bootstrap-report --request" in (
        baseline_prompt
    )
    for alias in (
        "gateway_execute",
        "wiki_query",
        "attempt_record_experiment",
        "attempt_report",
        "lineage_bootstrap_report",
    ):
        assert all(alias not in prompt for prompt in prompts.values())

    for prompt in prompts.values():
        examples = re.findall(r"```json\n(.*?)\n```", prompt, re.DOTALL)
        for example in examples:
            json.loads(example)


def test_problem_schema_metadata_is_added_after_generation(tmp_path: Path) -> None:
    output = tmp_path / "agent_problem.json"
    contract = {
        "objective": "optimize while exact evaluator cases remain hidden",
        "evaluation": {
            "exact_cases": "private",
            "correctness_requirement": "every hidden case must pass",
            "performance_requirement": "measure only after correctness",
            "development_cases_are_evaluation_cases": False,
        },
        "operator_contract": {},
        "workload_profile": {},
        "distribution_profile": {},
        "shape_domain": {},
        "invariants": ["preserve semantics"],
        "coverage_regimes": [],
        "development_cases": [],
    }
    output.write_text(json.dumps(contract), encoding="utf-8")

    problem_generalization.finalize_output(SimpleNamespace(output_path=output))

    assert json.loads(output.read_text(encoding="utf-8")) == {
        "schema_version": "atrex.agent_problem.v1",
        **contract,
    }
    assert "atrex.agent_problem.v1" not in _rendered_prompts()["problem_generalization"]
