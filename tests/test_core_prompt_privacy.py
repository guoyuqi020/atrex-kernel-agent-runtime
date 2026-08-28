"""Ensure Agent-visible instructions do not reveal implementation identity."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

CORE_ROOT = Path(__file__).resolve().parents[1] / "src" / "atrex-kernel-agent-core"
RUNTIME_OPTIMIZER_EVIDENCE = (
    Path(__file__).resolve().parents[1]
    / "src/atrex_runtime/templates/evidence/optimizer.md"
).read_text(encoding="utf-8")
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


def _rendered_documents() -> dict[str, tuple[str, str]]:
    """Return the (user prompt, system prompt) pair delivered for each session phase."""
    config = AgentConfig.load(CORE_ROOT)
    attempt_context = SimpleNamespace(
        evidence_prompt=RUNTIME_OPTIMIZER_EVIDENCE,
        agent_problem={
            "schema_version": "atrex.agent_problem.v1",
            "generator": {"name": "private-generator-provenance", "version": 2},
            "objective": "implement example while exact cases remain private",
            "invariants": ["preserve semantics"],
        },
        manifest={
            "dsl": "cuda",
            "context": {
                "epoch_number": 1,
                "attempt_ordinal": 1,
                "operator": "example",
                "hardware_target": "gpu",
            },
        },
    )
    baseline_context = SimpleNamespace(
        agent_problem={
            "schema_version": "atrex.agent_problem.v1",
            "generator": {"name": "private-generator-provenance", "version": 2},
            "objective": "implement example while exact cases remain private",
            "invariants": ["preserve semantics"],
        },
        manifest={
            "dsl": "cuda",
            "operator": "example",
            "hardware_target": "gpu",
        },
    )
    generalization_context = SimpleNamespace(
        manifest={
            "dsl": "cuda",
            "operator": "example",
            "hardware_target": "gpu",
        }
    )
    return {
        "optimization_attempt": (
            attempt.render_prompt(attempt_context, config),
            attempt.render_system_prompt(attempt_context, config),
        ),
        "framework_baseline": (
            lineage_bootstrap.render_prompt(baseline_context, config),
            lineage_bootstrap.render_system_prompt(baseline_context, config),
        ),
        "problem_generalization": (
            problem_generalization.render_prompt(generalization_context, config),
            "",
        ),
    }


def _rendered_prompts() -> dict[str, str]:
    """Join both delivered channels so every instruction assertion covers the union."""
    documents = {}
    for phase, (prompt, system_prompt) in _rendered_documents().items():
        documents[phase] = (
            prompt.rstrip() + "\n\n" + system_prompt + "\n" if system_prompt else prompt
        )
    return documents


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
        assert all(f'"{field}"' not in prompt for field in _INTERNAL_CONTEXT_FIELDS)


def test_optimizer_prompt_contains_the_controller_injected_evidence_fragment() -> None:
    prompt = _rendered_prompts()["optimization_attempt"]
    assert "# Runtime workspace and Evidence contract" in prompt
    assert "tools/README.md" in prompt


def test_execution_prompts_embed_the_public_operator_contract() -> None:
    prompts = _rendered_prompts()
    for phase in ("optimization_attempt", "framework_baseline"):
        assert "## Public operator contract" in prompts[phase]
        assert '"objective": "implement example while exact cases remain private"' in prompts[phase]
        assert "private-generator-provenance" not in prompts[phase]
        assert "input/agent-problem" not in prompts[phase]


def test_rendered_prompts_use_exact_cli_subcommands_and_valid_json_examples() -> None:
    prompts = _rendered_prompts()
    attempt_prompt = prompts["optimization_attempt"]
    baseline_prompt = prompts["framework_baseline"]

    for command in (
        "gateway-execute",
        "kernel-trial-show",
        "kernel-artifact-read",
        "gateway-result-read",
        "wiki-query",
        "update-direction",
        "list-directions",
        "load-direction",
        "record-experiment",
        "list-experiments",
        "load-experiment",
        "attempt-report",
    ):
        assert f"python3 agent/optimizer/src/runtime_tools.py {command} --request" in attempt_prompt
    assert "measurements-query" not in attempt_prompt
    assert "lineage-bootstrap-report" not in baseline_prompt
    for command in (
        "gateway-execute",
        "kernel-trial-show",
        "kernel-artifact-read",
        "gateway-result-read",
        "wiki-query",
        "update-direction",
        "list-directions",
        "load-direction",
        "record-experiment",
        "list-experiments",
        "load-experiment",
        "attempt-report",
    ):
        expected = f"python3 agent/optimizer/src/runtime_tools.py {command} --request"
        assert expected in baseline_prompt
    assert "kernel-trials --request" not in baseline_prompt
    assert "kernel-trial-show --request" in baseline_prompt
    assert "kernel-artifact-read --request" in baseline_prompt
    assert "gateway-result-read --request" in baseline_prompt
    assert '"kernel_artifact_digest":"sha256:<digest>"' in baseline_prompt
    for prompt in (attempt_prompt, baseline_prompt):
        assert '"artifact_file"' in prompt
        assert "scratch/recovered/kernel.py" in prompt
        assert "source content is" in prompt.lower()
        assert any(value in prompt.lower() for value in ("not printed", "never printed"))
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


def test_session_tool_contract_survives_context_compaction_as_the_system_prompt() -> None:
    documents = _rendered_documents()

    for phase in ("optimization_attempt", "framework_baseline"):
        prompt, system_prompt = documents[phase]
        assert system_prompt.startswith("## Session tools")
        assert "python3 agent/optimizer/src/runtime_tools.py gateway-execute --request" in (
            system_prompt
        )
        assert "## Session tools" not in prompt
        assert "runtime_tools.py" not in prompt
    generalization_prompt, generalization_system = documents["problem_generalization"]
    assert generalization_system == ""
    assert "runtime_tools.py" not in generalization_prompt


def test_optimizer_prompt_teaches_the_end_to_end_evidence_handoff() -> None:
    prompt = _rendered_prompts()["optimization_attempt"]

    assert "## Optimization workflow" in prompt
    for step in (
        "Recover only relevant state",
        "Choose and plan one causal hypothesis",
        "Localize before broad changes",
        "Research progressively",
        "Implement and repair causally",
        "Validate the exact candidate",
        "Record as work proceeds",
    ):
        assert step in prompt
    assert '"before": {' in prompt
    assert '"after": {' in prompt
    assert '"action": "keep_after"' in prompt
    assert "supporting_experiment_ids" in prompt


def test_optimizer_prompt_reuses_measurements_but_revalidates_interpretations() -> None:
    prompt = _rendered_prompts()["optimization_attempt"]

    assert "Do not repeat a completed Evaluate or Profile" in prompt
    assert "Treat normalized Gateway operation status" in prompt
    assert "Agent-authored report, analysis" in prompt
    assert "Honor the injected measurement-reuse policy" in prompt
    assert prompt.count("Do not repeat a completed Evaluate or Profile") == 1
    assert prompt.count("Treat normalized Gateway operation status") == 1
    assert "Repeat surprising deltas" not in prompt


def test_optimizer_prompt_layers_are_concise_non_redundant_and_consistent() -> None:
    prompt = _rendered_prompts()["optimization_attempt"]

    for unique_contract in (
        "Do not repeat a completed Evaluate or Profile",
        "Treat normalized Gateway operation status",
        "scratch/attempt-report-draft.json",
        "tools/README.md",
    ):
        assert prompt.count(unique_contract) == 1
    assert prompt.index("# Kernel optimization attempt") < prompt.index(
        "# Runtime workspace and Evidence contract"
    )
    assert prompt.index("# Runtime workspace and Evidence contract") < prompt.index(
        "## Session tools"
    )
    assert "Use `reference/` for pinned production implementation patterns" in prompt
    assert "no local knowledge or reference checkout is available" not in prompt


def test_optimizer_prompt_limits_advancement_not_open_direction_count() -> None:
    prompt = _rendered_prompts()["optimization_attempt"]
    normalized = " ".join(prompt.split())

    assert "advance at most three inherited or new Directions" in normalized
    assert "proposals are unlimited and do not consume this limit" in normalized
    assert "Only one Direction may be `in_progress` at a time" in normalized
    assert "do not interleave their research, tools, edits, or measurements" in normalized
    assert "Before starting another, close the current one" in normalized
    assert "None may remain `in_progress` at handoff" in normalized
    assert "Without an Experiment use `defer` or `block`" in normalized
    assert "Leave at most three visible Directions" not in prompt


def test_optimizer_prompt_registers_direction_before_exploration() -> None:
    prompts = _rendered_prompts()

    for phase in ("optimization_attempt", "framework_baseline"):
        prompt = prompts[phase]
        normalized = " ".join(prompt.split())
        assert "durable unit of research and exploration" in normalized
        assert "immediately `propose` and `start`" in normalized
        assert "before its Wiki/reference research" in normalized
        assert "`TaskCreate`" in prompt
        assert "do not register it" in normalized
        assert "do not wait for measurement" in normalized


def test_bootstrap_defers_shared_direction_and_tool_protocols_to_one_contract() -> None:
    prompt = _rendered_prompts()["framework_baseline"]
    normalized = " ".join(prompt.split())

    assert prompt.count("durable unit of research and exploration") == 1
    assert normalized.count("`TaskCreate`, scratch plans, and prose do not register it") == 1
    assert "follow the shared Direction contract below" in normalized
    assert "Use the shared tool contract below" in normalized
    assert "Direction covers the whole research and exploration path" not in prompt
    assert "Route Runtime-local Trial, source, and result reads" not in prompt


def test_optimizer_prompt_builds_the_terminal_report_incrementally() -> None:
    prompt = _rendered_prompts()["optimization_attempt"]
    normalized = " ".join(prompt.split())

    assert "scratch/attempt-report-draft.json" in prompt
    assert "separate `scratch/attempt-report.json`" in prompt
    assert "After each receipt" in prompt
    assert "The draft may be overwritten" in prompt
    assert "first successful call" in prompt
    assert "An error response publishes nothing" in normalized
    assert "Never call it again after a successful response" in normalized
    assert prompt.count("scratch/attempt-report-draft.json") == 1


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
