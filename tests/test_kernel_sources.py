"""Multi-file source identity, edit scope, Dev execution, and fresh-process isolation."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from atrex_runtime.artifacts.local import ArtifactKind, LocalArtifactStore
from atrex_runtime.bootstrap import CampaignLineageSpecV2
from atrex_runtime.domain.ids import new_attempt_id
from atrex_runtime.domain.models import Dsl, TokenUsage
from atrex_runtime.gateway import abba_remote
from atrex_runtime.gateway.abba import build_abba_source_request
from atrex_runtime.gateway.contract import AgateEvaluationContext, AgateEvaluationContractV1
from atrex_runtime.gateway.control import (
    GatewayCapabilityPolicy,
    GatewayOperation,
    SqliteGatewayControl,
)
from atrex_runtime.gateway.execution import build_evaluation_request
from atrex_runtime.gateway.finalization import (
    AgateAuthoritativeCandidateEvaluator,
    BootstrapEvaluationStage,
)
from atrex_runtime.gateway.production_policy import ProductionKernelPolicy
from atrex_runtime.gateway.source_tree import SOURCE_REQUEST_KEY, SourceTreeAgateClient
from atrex_runtime.kernel_sources import (
    import_source_tree,
    inject_source_instructions,
    read_kernel_source,
)
from atrex_runtime.registry.sqlite import SqliteRegistry


@pytest.fixture
def source_seed(tmp_path):
    repository = tmp_path / "repository"
    package = repository / "example_kernel/impl"
    package.mkdir(parents=True)
    (package.parent / "__init__.py").write_text("")
    (package / "__init__.py").write_text("")
    (package / "value.py").write_text("VALUE = 2\n")
    (repository / "LICENSE").write_text("example\n")
    for args in (
        ("init",),
        ("add", "."),
        ("-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-m", "seed"),
    ):
        subprocess.run(["git", "-C", str(repository), *args], check=True, capture_output=True)
    revision = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
    ).strip()
    task = tmp_path / "task"
    task.mkdir()
    (task / "adapter.py").write_text("from example_kernel.impl.value import VALUE\n")
    manifest = task / "source_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "name": "example",
                "adapter": "adapter.py",
                "source": {
                    "name": "example",
                    "revision": revision,
                    "archive_paths": ["example_kernel", "LICENSE"],
                    "package_root": ".",
                },
                "editable_roots": ["example_kernel/impl"],
                "measurement": {"repeats": 999},
            }
        )
    )
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    contract = import_source_tree(manifest, repository, artifacts)
    working = tmp_path / "candidate"
    artifacts.materialize(contract.seed_digest, working)
    for path in (working, *working.rglob("*")):
        path.chmod(0o700 if path.is_dir() else 0o600)
    return SimpleNamespace(
        repository=repository,
        manifest=manifest,
        contract=contract,
        artifacts=artifacts,
        working=working,
    )


def evaluation_contract(source, *, mode="full"):
    return AgateEvaluationContractV1(
        candidate_path="kernel.py",
        reference_py="reference",
        input_py="inputs",
        shapes={"s0": {}},
        options={
            "num_correctness_cases": 5,
            "bench_iters": 100,
            "atol": 0.01,
            "rtol": 0.05,
            "timeout_s": 30,
        },
        lock_clocks=False,
        mode=mode,
        kernel_sources={Dsl.CUTEDSL: source},
    )


def builder(candidate, reference, gpu, **kwargs):
    return {"candidate": candidate, "reference": reference, "gpu": gpu, **kwargs}


def test_import_is_pinned_not_a_shared_checkout(source_seed):
    seed = source_seed
    (seed.repository / "example_kernel/impl/value.py").write_text("DIRTY = True\n")
    again = import_source_tree(seed.manifest, seed.repository, seed.artifacts)
    assert again == seed.contract
    assert (seed.working / "kernel.py").is_file()
    assert not (seed.working / "source").exists()
    assert seed.contract.seal(seed.working, seed.artifacts) == seed.contract.seed_digest


def test_complete_identity_and_historical_restore(source_seed, tmp_path):
    seed = source_seed
    (seed.working / "example_kernel/impl/value.py").write_text("VALUE = 3\n")
    (seed.working / "example_kernel/impl/new.py").write_text("NEW = True\n")
    digest = seed.contract.seal(seed.working, seed.artifacts)
    assert digest != seed.contract.seed_digest
    cache = seed.working / "example_kernel/impl/__pycache__"
    cache.mkdir()
    (cache / "ignored.pyc").write_bytes(b"\xff\x00")
    (seed.working / "example_kernel/impl/empty").mkdir()
    assert seed.contract.seal(seed.working, seed.artifacts) == digest
    restored = tmp_path / "restored"
    seed.artifacts.materialize(digest, restored)
    assert seed.contract.validate_tree(restored)["example_kernel/impl/value.py"] == "VALUE = 3\n"
    assert (restored / "LICENSE").read_text() == "example\n"


@pytest.mark.parametrize("change", ["adapter", "locked", "remove", "outside", "symlink", "binary"])
def test_edit_scope_is_runtime_enforced(source_seed, change):
    root, contract = source_seed.working, source_seed.contract
    if change == "adapter":
        (root / "kernel.py").write_text("FAKE = True\n")
    elif change == "locked":
        (root / "example_kernel/__init__.py").write_text("FAKE = True\n")
    elif change == "remove":
        (root / "LICENSE").unlink()
    elif change == "outside":
        (root / "fake.py").write_text("FAKE = True\n")
    elif change == "symlink":
        (root / "example_kernel/impl/alias.py").symlink_to(root / "kernel.py")
    else:
        (root / "example_kernel/impl/lib.so").write_text("even text is not a source library")
    with pytest.raises(ValueError):
        contract.seal(root, source_seed.artifacts)


def test_injected_scope_and_hash_match(source_seed, tmp_path):
    root = tmp_path / "workspace"
    (root / ".runtime").mkdir(parents=True)
    shutil.copytree(source_seed.working, root / "work/kernel")
    (root / ".runtime/evidence-instructions.md").write_text("base instructions\n")
    (root / ".runtime/evidence-manifest.json").write_text("{}")
    inject_source_instructions(root, source_seed.contract)
    prompt = (root / ".runtime/evidence-instructions.md").read_bytes()
    manifest = json.loads((root / ".runtime/evidence-manifest.json").read_bytes())
    assert manifest["prompt_fragment_sha256"] == hashlib.sha256(prompt).hexdigest()
    assert b"example_kernel/impl" in prompt
    assert b"work/kernel/" in prompt
    assert (root / "work/kernel/kernel.py").stat().st_mode & 0o200 == 0


def test_lineage_source_declaration_replaces_baseline(source_seed):
    values = {
        "source_manifest": source_seed.manifest,
        "source_repository": source_seed.repository,
        "initial_evidence": source_seed.working,
    }
    assert CampaignLineageSpecV2(**values).baseline_kernel is None
    with pytest.raises(ValueError):
        CampaignLineageSpecV2(**values, baseline_kernel=source_seed.working)
    with pytest.raises(ValueError):
        CampaignLineageSpecV2(
            source_manifest=source_seed.manifest, initial_evidence=source_seed.working
        )


# Small CPU evaluator double: imports the actual packaged module in a fresh process.
# It is deliberately not a GPU correctness claim.
EVALUATOR = """
import json, os
from pathlib import Path
def evaluate(config):
    from example_kernel.impl.value import VALUE
    cache = Path(os.environ["TRITON_CACHE_DIR"])
    assert not (cache / "used").exists()
    (cache / "used").write_text("yes")
    ids = json.loads((Path(config["reference_dir"]) / "shapes.json").read_text())
    return {
        "passed": {"compile": {"status": "passed"},
                   "correctness": {s: {"status": "passed"} for s in ids}},
        "correctness": {"shapes": {s: {"cases": [{"outputs": []}]} for s in ids}},
        "performance": {"shapes": {
            s: {"error": None, "samples": [{"end_to_end_time_ms": VALUE / 1000}]}
            for s in ids}},
        "mode": config.get("validation_mode"),
    }
"""


class EvaluatorFiles:
    def files(self):
        return {"atrex-bench/src/atrex_bench/__init__.py": EVALUATOR}


class LocalDevClient:
    def __init__(self, root):
        self.root, self.requests, self.jobs = root, [], {}

    def submit_job(self, kind, request):
        assert kind == "dev"
        job_id = f"dev_{len(self.requests)}"
        self.requests.append(request)
        root = self.root / job_id
        root.mkdir(parents=True)
        for name, content in request["files"].items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
        completed = subprocess.run(
            [sys.executable, "__atrex_abba.py", "request.json"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
        self.jobs[job_id] = {
            "job_id": job_id,
            "status": "succeeded",
            "command_ok": True,
            "result": {"stdout": completed.stdout, "stderr": completed.stderr},
        }
        return {"job_id": job_id}

    def get_job(self, job_id, **kwargs):
        return self.jobs[job_id]


@pytest.mark.parametrize("mode", ["full", "correctness_only"])
def test_source_evaluate_dev_and_recovered_poll(source_seed, tmp_path, mode):
    seed = source_seed
    contract = evaluation_contract(seed.contract, mode=mode)
    source = read_kernel_source(seed.working, "kernel.py", seed.contract)
    payload = build_evaluation_request(
        builder,
        candidate_source=source,
        contract=contract,
        operator="example",
        hardware_target="cpu-test",
        dsl=Dsl.CUTEDSL,
        name="example",
        idempotency_key="source-test",
    )
    assert SOURCE_REQUEST_KEY in json.loads(json.dumps(payload))
    sdk = LocalDevClient(tmp_path / "jobs")
    accepted = SourceTreeAgateClient(sdk, EvaluatorFiles()).submit_job("eval", payload)
    # New wrapper, no local ID map: this is the restarted Runtime's poll path.
    job = SourceTreeAgateClient(sdk, EvaluatorFiles()).get_job(accepted["job_id"])
    assert job["result"]["mode"] == mode
    assert job["result"]["performance"]["shapes"]["s0"]["samples"][0]["end_to_end_time_ms"] == 0.002
    assert "source_tree_execution" in job
    assert "example_kernel/impl/value.py" in source.files
    assert "measurement" not in json.loads(sdk.requests[0]["files"]["request.json"])["evaluator"]
    with pytest.raises(ValueError, match="unsupported source-tree operation"):
        SourceTreeAgateClient(sdk, EvaluatorFiles()).submit_job("unknown", payload)


def test_abba_uses_independent_whole_trees_and_jit_caches(source_seed, tmp_path):
    seed = source_seed
    a = read_kernel_source(seed.working, "kernel.py", seed.contract)
    (seed.working / "example_kernel/impl/value.py").write_text("VALUE = 1\n")
    b = read_kernel_source(seed.working, "kernel.py", seed.contract)
    schedule = [
        {"revision": revision, "repeat": index // 2}
        for index, revision in enumerate(("incumbent", "candidate", "candidate", "incumbent"))
    ]
    request = build_abba_source_request(
        hardware_target="cpu-test",
        contract=evaluation_contract(seed.contract),
        shape_ids=["s0"],
        schedule=schedule,
        incumbent_source=a,
        candidate_source=b,
        evaluator_files=EvaluatorFiles().files(),
        per_run_timeout_seconds=10,
        allocation_timeout_seconds=60,
    )
    sdk = LocalDevClient(tmp_path / "jobs")
    accepted = sdk.submit_job("dev", request)
    output = sdk.get_job(accepted["job_id"])["result"]["stdout"]
    result = json.loads(output.split(abba_remote.RESULT_PREFIX)[-1])
    assert result["error"] is None
    assert [run["result"]["latency_us_geomean"] for run in result["runs"]] == pytest.approx(
        [2, 1, 1, 2]
    )


def test_source_production_gate_allows_local_import_but_not_prebuilt_escape(source_seed):
    root = source_seed.working
    code = (
        "from . import helpers\nfrom .helpers import value\nfrom example_kernel import impl\n"
        "import cutlass.cute as cute\n@cute.kernel\ndef kernel(): pass\n"
    )
    (root / "example_kernel/impl/value.py").write_text(code)
    policy = ProductionKernelPolicy()
    policy.validate(root, "kernel.py", Dsl.CUTEDSL, source_seed.contract)
    (root / "example_kernel/impl/value.py").write_text(code + "import ctypes\n")
    with pytest.raises(ValueError, match="dynamic external-code"):
        policy.validate(root, "kernel.py", Dsl.CUTEDSL, source_seed.contract)


def test_source_bootstrap_runs_agent_journal_then_trusted_stages(
    source_seed, tmp_path, monkeypatch
):
    import anyio
    from test_bootstrap_runtime import _bootstrap_report, _record_agent_evaluate
    from test_gateway_finalization import NOW, FakeEvents, _subject
    from test_lineage_bootstrap_worker import _write_agent

    from atrex_runtime.composition.bootstrap import CoreLineageBaselineGenerator
    from atrex_runtime.gateway.journals import RuntimeJournalService
    from atrex_runtime.workers import (
        CleanEnvironmentLauncher,
        CoreLineageBootstrapSessionDriver,
        CoreOptimizerProcessConfig,
        LineageBootstrapWorkspaceAssembler,
    )
    from atrex_runtime.workers.attempt_report import AttemptReportV12

    seed = source_seed
    registry = SqliteRegistry(tmp_path / "registry.sqlite")
    control = SqliteGatewayControl(
        tmp_path / "gateway.sqlite", registry, signing_key=b"k" * 32, clock=lambda: NOW
    )
    attempt = new_attempt_id()
    subject = _subject(attempt)
    context = AgateEvaluationContext(
        "example", "cpu-test", Dsl.CUTEDSL, evaluation_contract(seed.contract)
    )
    contexts = SimpleNamespace(resolve=lambda _attempt: context)
    sdk = LocalDevClient(tmp_path / "jobs")
    evaluator = AgateAuthoritativeCandidateEvaluator(
        SourceTreeAgateClient(sdk, EvaluatorFiles()),
        builder,
        contexts,
        seed.artifacts,
        control,
        FakeEvents(),
        wait_timeout_s=100,
        bootstrap_stages=(BootstrapEvaluationStage(1), BootstrapEvaluationStage(5)),
        clock=lambda: NOW,
    )
    agent = tmp_path / "agent"
    _write_agent(agent)
    kda = Path(__file__).resolve().parents[1] / "src/kernel-design-agents"
    shutil.copyfile(kda / "atrex-agent.json", agent / "atrex-agent.json")
    shutil.copytree(kda / "prompts", agent / "prompts")
    agent_digest = seed.artifacts.put_directory(agent, ArtifactKind.KERNEL_AGENT)
    contract_digest = seed.artifacts.put_json(
        context.contract.model_dump(mode="json"), ArtifactKind.EVALUATION_CONTRACT
    )
    problem_digest = seed.artifacts.put_json({"objective": "example"}, ArtifactKind.AGENT_PROBLEM)
    driver = CoreLineageBootstrapSessionDriver(
        CleanEnvironmentLauncher(Path("/usr/bin/env")),
        CoreOptimizerProcessConfig(
            agent_backend="claude",
            command_prefix=(sys.executable,),
            isolated_home_environment_keys=("HOME",),
            session_trace_relative_path="sessions/core",
            token_usage_report_relative_path="scratch/token-usage.json",
            max_attempt_report_bytes=1_048_576,
            timeout_seconds=10,
            terminate_grace_seconds=1,
            max_diagnostic_bytes=4096,
            max_session_tokens=100,
        ),
        seed.artifacts,
    )
    session_calls = []

    def run_session(phase, environment, *, label):
        session_calls.append(phase.root)
        assert environment["ATREX_CORE_PHASE"] == "framework_baseline"
        assert environment["ATREX_AGENT_BACKEND"] == "claude"
        instructions = (phase.root / ".runtime/source-instructions.md").read_text()
        assert "Source-tree Bootstrap" in instructions
        assert "example_kernel/impl" in instructions
        assert "exactly one baseline Experiment" in instructions
        # Exercise both real Bundle renderers in fresh interpreters: the Runtime
        # projection must reach the model, not just exist as an unread file.
        for bundle_name in ("kernel-design-agents", "atrex-kernel-agent-core"):
            bundle = Path(__file__).resolve().parents[1] / "src" / bundle_name
            subprocess.run(
                [
                    sys.executable,
                    "-I",
                    "-B",
                    "-c",
                    "import sys; from pathlib import Path; from types import SimpleNamespace; "
                    "sys.path.insert(0, sys.argv[1]); "
                    "from agent_config import AgentConfig; "
                    "from sessions.lineage_bootstrap import render_prompt, render_system_prompt; "
                    "config=AgentConfig.load(Path(sys.argv[2]), environment={}); "
                    "context=SimpleNamespace(agent_problem={'objective':'test'}, "
                    "manifest={'dsl':'cutedsl','operator':'example',"
                    "'hardware_target':'cpu-test'}); "
                    "text=render_prompt(context, config)+render_system_prompt(context, config); "
                    "assert 'Source-tree Bootstrap' in text; "
                    "assert 'example_kernel/impl' in text; "
                    "assert 'record-experiment' in text",
                    str(bundle / "src"),
                    str(phase.repository),
                ],
                check=True,
                capture_output=True,
            )
        working = phase.root / "work/kernel"
        assert (working / "kernel.py").read_bytes() == (seed.working / "kernel.py").read_bytes()
        assert not ((working / "kernel.py").stat().st_mode & 0o200)
        assert (working / "example_kernel/impl/value.py").stat().st_mode & 0o200
        # Generated caches must not change the nomination's measured source identity.
        (working / "__pycache__").mkdir()
        (working / "__pycache__/kernel.pyc").write_bytes(b"cache")
        candidate = seed.contract.seal(working, seed.artifacts)
        gateway_result = seed.artifacts.put_json(
            {"correct": True, "latency_us": 2}, ArtifactKind.GATEWAY_RESULT
        )
        _record_agent_evaluate(
            control,
            capability_token=environment["ATREX_GATEWAY_CAPABILITY"],
            attempt_id=attempt,
            candidate=candidate,
            gateway_result=gateway_result,
            idempotency_key="bootstrap-agent",
            latency_us=2,
        )
        report = _bootstrap_report(
            attempt,
            candidate=candidate,
            gateway_result=gateway_result,
            generation=control.current_generation(attempt),
        )
        (phase.root / "scratch/attempt-report.json").write_text(report.model_dump_json())
        return SimpleNamespace(
            finish_reason="completed",
            process=SimpleNamespace(stdout="done"),
            token_usage=SimpleNamespace(
                to_domain=lambda: TokenUsage(10, 5, 0, 0), require_budget=lambda: 100
            ),
            session_trace_digest=seed.artifacts.put_json(
                {"session": "bootstrap-agent"}, ArtifactKind.SESSION_LOG
            ),
        )

    monkeypatch.setattr(driver._phases, "run", run_session)
    generator = CoreLineageBaselineGenerator(
        LineageBootstrapWorkspaceAssembler(tmp_path / "sessions", seed.artifacts),
        driver,
        control,
        registry,
        evaluator,
        seed.artifacts,
        gateway_endpoint="http://runtime.invalid",
        operations=frozenset({GatewayOperation.EVALUATE}),
        max_calls=100,
        capability_lifetime=timedelta(hours=1),
        environment=(),
        wiki_enabled=False,
        backend="claude",
    )
    arguments = dict(
        bootstrap_attempt_id=attempt,
        campaign_id=subject.campaign_id,
        lineage_id=subject.lineage_id,
        kernel_agent_revision_id=subject.kernel_agent_revision_id,
        optimizer_digest=agent_digest,
        input_kernel_digest=seed.contract.seed_digest,
        evaluation_contract_digest=contract_digest,
        agent_problem_digest=problem_digest,
        evidence_digest=subject.evidence_digest,
        dsl=Dsl.CUTEDSL,
        operator="example",
        hardware_target="cpu-test",
    )
    try:
        control.issue_bootstrap(
            replace(
                subject,
                epoch_id=generator._bootstrap_epoch_id(attempt),
                input_kernel_digest=seed.contract.seed_digest,
                evaluation_contract_digest=contract_digest,
                dsl=Dsl.CUTEDSL,
                operator="example",
                hardware_target="cpu-test",
            ),
            GatewayCapabilityPolicy(
                frozenset({GatewayOperation.EVALUATE}), 100, NOW + timedelta(hours=1)
            ),
        )
        with pytest.raises(ValueError, match="no matching Agent evaluation"):
            anyio.run(evaluator.finalize, attempt, seed.contract.seed_digest)
        baseline = generator.generate(**arguments)
        assert baseline.kernel_digest == seed.contract.seed_digest
        assert baseline.latency_us == pytest.approx(2)
        assert len(sdk.requests) == 2
        assert len(session_calls) == 1
        report = AttemptReportV12.model_validate_json(
            (seed.artifacts.verify(baseline.report_digest).payload_path / "value.json").read_bytes()
        )
        assert report.experiments[0].action == "baseline"
        assert report.direction_events
        journal = RuntimeJournalService(
            SimpleNamespace(
                visible_attempt_report_artifacts=lambda _: ((attempt, baseline.report_digest),)
            ),
            seed.artifacts,
        )
        assert journal._report_values(attempt, "experiments")
        assert journal._report_values(attempt, "direction_events")
        assert generator.generate(**arguments) == baseline
        assert len(session_calls) == 1 and len(sdk.requests) == 2
        assert not (seed.artifacts.verify(agent_digest).payload_path / "CLAUDE.md").exists()
        assert (
            "Source-tree Bootstrap"
            not in (session_calls[0] / "prompts/framework_baseline.md").read_text()
        )
    finally:
        control.close()
        registry.close()


@pytest.mark.anyio
@pytest.mark.parametrize("mode", ["full", "correctness_only"])
async def test_optimizer_source_evaluate_preserves_custom_inputs_and_gate(
    source_seed, tmp_path, mode
):
    from atrex_runtime.gateway.agate import AgateGatewayAdapter, SqliteAgateJobStore
    from atrex_runtime.gateway.proxy import GatewayAdapterRequest

    contract = evaluation_contract(source_seed.contract)
    context = AgateEvaluationContext("example", "cpu-test", Dsl.CUTEDSL, contract)
    jobs = SqliteAgateJobStore(tmp_path / "jobs.sqlite")
    sdk = LocalDevClient(tmp_path / "jobs")
    adapter = AgateGatewayAdapter(
        SourceTreeAgateClient(sdk, EvaluatorFiles()),
        builder,
        SimpleNamespace(resolve=lambda _attempt: context),
        jobs,
        wait_timeout_s=100,
        optimizer_correctness_cases=3,
        optimizer_bench_iters=17,
        profile_without_roofline=True,
    )
    try:
        request = GatewayAdapterRequest(
            new_attempt_id(),
            GatewayOperation.EVALUATE,
            "custom-source-eval",
            source_seed.contract.seed_digest,
            source_seed.working,
            None,
            None,
            None,
            parameters={
                "mode": mode,
                "input_py": "custom input",
                "shapes": {"99": {"init_kwargs": {}, "input_kwargs": {}}},
            },
        )
        result = await adapter.execute(request)
        assert result.status == "completed"
        assert len(sdk.requests) == 1  # no accidental single-file Profile fallback
        files = sdk.requests[0]["files"]
        assert files["reference/input.py"] == "custom input"
        assert json.loads(files["reference/shapes.json"]) == {
            "99": {"init_kwargs": {}, "input_kwargs": {}}
        }
        options = json.loads(files["request.json"])["evaluator"]
        assert options["num_correctness_cases"] == 3 and options["bench_iters"] == 17
        assert options["validation_mode"] == mode
        assert (result.evaluation is not None) == (mode == "full")
    finally:
        jobs.close()


def test_campaign_bootstrap_imports_and_reuses_source_seed(source_seed, tmp_path):
    from test_bootstrap import FakeBaselineGenerator, FakeGitLoader, _bootstrapper, _campaign_spec

    from atrex_runtime.bootstrap import CampaignSpecV3
    from atrex_runtime.gateway.contract import load_evaluation_contract

    spec_path = _campaign_spec(tmp_path, lineage_dsls=(Dsl.CUTEDSL,))
    value = json.loads(spec_path.read_bytes())
    value["lineages"]["cutedsl"].pop("baseline_kernel")
    value["lineages"]["cutedsl"].update(
        {
            "source_manifest": str(source_seed.manifest),
            "source_repository": str(source_seed.repository),
        }
    )
    spec_path.write_text(json.dumps(value))
    spec = CampaignSpecV3.from_file(spec_path)
    registry = SqliteRegistry(tmp_path / "registry.sqlite")

    class SeedGenerator(FakeBaselineGenerator):
        def generate(self, **values):
            assert "source_seed" not in values
            assert values["input_kernel_digest"] == source_seed.contract.seed_digest
            return super().generate(**values)

    baseline = SeedGenerator(source_seed.artifacts)
    bootstrapper = _bootstrapper(
        registry,
        source_seed.artifacts,
        FakeGitLoader(source_seed.artifacts),
        baseline,
    )
    try:
        first = bootstrapper.bootstrap_campaign(spec)
        assert bootstrapper.bootstrap_campaign(spec) == first
        assert baseline.calls == [Dsl.CUTEDSL]
        sealed = load_evaluation_contract(
            source_seed.artifacts, first.lineages[0].evaluation_contract_digest
        )
        assert sealed.kernel_sources[Dsl.CUTEDSL] == source_seed.contract
    finally:
        registry.close()


@pytest.mark.anyio
async def test_source_tree_ablation_preserves_full_v0_and_edit_boundary(source_seed, tmp_path):
    from conftest import kernel_agent_limits
    from test_ablation import _agent_artifact
    from test_bootstrap import FakeBaselineGenerator, FakeGitLoader, _bootstrapper, _campaign_spec

    from atrex_runtime.ablation import AblationArmSeeder, AblationArmSpecV1
    from atrex_runtime.ablation_plan import build_ablation_plan
    from atrex_runtime.bootstrap import CampaignSpecV3
    from atrex_runtime.gateway.contract import load_evaluation_contract
    from atrex_runtime.kernel_agents import KernelAgentRevisionBuilder
    from atrex_runtime.lineage_seed import LineageSeeder

    spec_path = _campaign_spec(tmp_path, lineage_dsls=(Dsl.CUTEDSL,))
    value = json.loads(spec_path.read_text())
    value["lineages"]["cutedsl"].pop("baseline_kernel")
    value["lineages"]["cutedsl"].update(
        source_manifest=str(source_seed.manifest), source_repository=str(source_seed.repository)
    )
    spec_path.write_text(json.dumps(value))
    loader = FakeGitLoader(source_seed.artifacts)
    loader.optimizer_digest = _agent_artifact(source_seed.artifacts, tmp_path)
    baseline = FakeBaselineGenerator(source_seed.artifacts)

    class NoEvaluation:
        async def evaluate(self, **_kwargs):
            pytest.fail("Control arms must reuse v0 without another GPU evaluation")

    with SqliteRegistry(tmp_path / "registry.sqlite") as registry:
        boot = _bootstrapper(registry, source_seed.artifacts, loader, baseline)
        result = boot.bootstrap_campaign(CampaignSpecV3.from_file(spec_path))
        seeder = LineageSeeder(
            registry,
            source_seed.artifacts,
            KernelAgentRevisionBuilder(source_seed.artifacts, limits=kernel_agent_limits()),
            NoEvaluation(),
            evolver_commit="e" * 40,
        )
        arm_seeder = AblationArmSeeder(registry, seeder)
        policy = json.loads(
            (Path(__file__).resolve().parents[1] / "scripts/production/policy.json").read_text()
        )
        ids = set()
        for arm in build_ablation_plan(policy)["arms"]:
            spec = AblationArmSpecV1(
                creation_key=arm["label"],
                source_lineage_id=result.lineages[0].lineage_id,
                **{
                    key: arm[key]
                    for key in (
                        "attempts_per_trajectory",
                        "trajectories_per_branch",
                        "ephemeral_agent_state",
                    )
                },
            )
            cloned = await arm_seeder.seed_arm(spec)
            assert await arm_seeder.seed_arm(spec) == cloned
            ids.add(cloned.campaign_id)
            original = registry.get_kernel_revision(result.lineages[0].baseline_kernel_revision_id)
            assert cloned.lineage.kernel_artifact_digest == original.artifact_digest
            assert cloned.lineage.gateway_result_digest == original.evaluation.gateway_result_digest
            campaign = registry.get_campaign(cloned.campaign_id)
            assert (
                campaign.evaluation_contract_digest == result.lineages[0].evaluation_contract_digest
            )
            contract = load_evaluation_contract(
                source_seed.artifacts, campaign.evaluation_contract_digest
            )
            source = contract.kernel_sources[Dsl.CUTEDSL]
            assert source == source_seed.contract
            restored = tmp_path / arm["label"]
            source_seed.artifacts.materialize(cloned.lineage.kernel_artifact_digest, restored)
            assert source.validate_tree(restored) == source.validate_tree(source_seed.working)
        assert len(ids) == 11
        assert baseline.calls == [Dsl.CUTEDSL]
