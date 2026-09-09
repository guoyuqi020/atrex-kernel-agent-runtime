"""Run the shipped Dev driver and source imports with CPU tool doubles, never a GPU."""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import subprocess
import sys
from dataclasses import replace
from types import SimpleNamespace

import pytest
from test_kernel_sources import builder, evaluation_contract
from test_kernel_sources import source_seed as source_seed

from atrex_runtime.domain.errors import InfrastructureError
from atrex_runtime.domain.ids import new_attempt_id
from atrex_runtime.domain.models import Dsl
from atrex_runtime.gateway.agate import AgateGatewayAdapter, SqliteAgateJobStore
from atrex_runtime.gateway.contract import AgateEvaluationContext
from atrex_runtime.gateway.control import GatewayOperation
from atrex_runtime.gateway.private_results import project_compile_job, project_private_job
from atrex_runtime.gateway.proxy import GatewayAdapterRequest
from atrex_runtime.gateway.source_diagnostics import (
    DIAGNOSTIC_PREFIX,
    TEXT_LIMIT,
    _export,
    _kernels,
)
from atrex_runtime.gateway.source_tree import SourceTreeAgateClient

FAKE_TORCH = """
from contextlib import nullcontext
from types import SimpleNamespace
version = SimpleNamespace(hip=None)
cuda = SimpleNamespace(is_available=lambda: True, get_device_capability=lambda: (10, 3),
    synchronize=lambda: None,
    nvtx=SimpleNamespace(range_push=lambda name: None, range_pop=lambda: None))
def manual_seed(seed): pass
no_grad = nullcontext
"""

MODEL = """
import os
from pathlib import Path
from example_kernel.impl.value import VALUE
class Model:
    def __init__(self, expected): assert expected == 7
    def eval(self): return self
    def __call__(self, x):
        assert x == 11 and VALUE == 2
        for key in ('CUTE_DSL_CACHE_DIR', 'TRITON_CACHE_DIR', 'TORCH_EXTENSIONS_DIR'):
            assert Path(os.environ[key]).is_dir()
        path = Path(os.environ['TRITON_CACHE_DIR']) / 'launches'
        path.write_text(path.read_text() + 'x' if path.exists() else 'x')
        if os.environ.get('FAKE_CANDIDATE_ERROR'): raise RuntimeError('PRIVATE_INPUT=11')
        return x
"""

LONG_METRICS = """"ID","Process ID","Kernel Name","Metric Name","Metric Unit","Metric Value"
"0","123","my_kernel","gpu__time_duration.sum","nsecond","2000"
"0","123","my_kernel","sm__throughput.avg.pct_of_peak_sustained_elapsed","%","60"
"0","123","my_kernel","dram__throughput.avg.pct_of_peak_sustained_elapsed","%","40"
"1","123","second","gpu__time_duration.sum","usecond","1"
"""

# NCU raw-page CSV: metric names in the header, then a units row, then launches.
METRICS = (
    '"ID","Process ID","Kernel Name","gpu__time_duration.sum",'
    '"sm__throughput.avg.pct_of_peak_sustained_elapsed",'
    '"gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed","launch__registers_per_thread"\n'
    '"","","","ns","%","%","register/thread"\n'
    '"0","123","my_kernel","2,000","60","40","168"\n'
    '"1","123","second","1,000","N/A","nan","32"\n'
)

FAKE_NCU = """
import json, os, subprocess, sys
from pathlib import Path
args = sys.argv[1:]
with open(os.environ['FAKE_TOOL_LOG'], 'a') as stream: stream.write(json.dumps(args) + '\\n')
if '--import' in args:
    if '--page' in args and args[args.index('--page')+1] == 'raw':
        if '--print-metric-name' in args:
            print("==ERROR== Option '--print-metric-name' is only supported for the details page.")
            sys.exit(1)
        if not os.environ.get('FAKE_EMPTY'): print(METRICS)
    elif args[args.index('--print-source')+1] == 'ptx':
        if os.environ.get('FAKE_NO_PTX'): sys.exit(2)
        print('.version 8.0\\n.target sm_103\\n.entry candidate() { ret; }')
    else: print('/*0010*/ MOV R1, R2;')
else:
    target = args.index('--target')
    child = subprocess.run(args[target-2:])
    if child.returncode: sys.exit(child.returncode)
    Path(args[args.index('--export')+1]).write_text('fake-report')
"""

FAKE_SANITIZER = """
import json, os, subprocess, sys
with open(os.environ['FAKE_TOOL_LOG'], 'a') as stream:
    stream.write(json.dumps(sys.argv[1:]) + '\\n')
assert '--error-exitcode' in sys.argv
child = subprocess.run(sys.argv[sys.argv.index('all')+1:])
if os.environ.get('FAKE_SANITIZER_ERROR'):
    print('========= ERROR SUMMARY: 1 error')
    sys.exit(86)
sys.exit(child.returncode)
"""


class DiagnosticDevClient:
    def __init__(self, root):
        self.root, self.requests, self.jobs = root, [], {}

    def submit_job(self, kind, payload):
        assert kind == "dev"
        index = len(self.requests)
        self.requests.append(payload)
        root = self.root / f"job-{index}"
        root.mkdir(parents=True)
        for name, content in payload["files"].items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
        (root / "torch.py").write_text(FAKE_TORCH)
        bin_dir = root / "bin"
        bin_dir.mkdir()
        for name, text in (
            ("ncu", f"METRICS={METRICS!r}\n" + FAKE_NCU),
            ("compute-sanitizer", FAKE_SANITIZER),
        ):
            path = bin_dir / name
            path.write_text(f"#!{sys.executable}\n" + text)
            path.chmod(0o700)
        env = {
            **os.environ,
            **payload["env_vars"],
            "PATH": str(bin_dir) + os.pathsep + os.environ["PATH"],
            "PYTHONPATH": str(root),
            "FAKE_TOOL_LOG": str(root / "tool.log"),
        }
        if env.get("FAKE_MISSING_NCU"):
            (bin_dir / "ncu").unlink()
        cmd = shlex.split(payload["command"])
        cmd[0] = sys.executable
        result = subprocess.run(
            cmd, cwd=root, env=env, text=True, capture_output=True, check=True, timeout=30
        )
        job_id = f"dv_{index}"
        self.jobs[job_id] = {
            "job_id": job_id,
            "status": "succeeded",
            "result": {"stdout": result.stdout, "stderr": result.stderr},
        }
        return {"job_id": job_id}

    def get_job(self, job_id, **kwargs):
        return self.jobs[job_id]


@pytest.fixture
def diagnostic(source_seed, tmp_path):
    seed = source_seed
    (seed.working / "kernel.py").write_text(MODEL)
    source = seed.contract.model_copy(
        update={
            "immutable_files": {
                **seed.contract.immutable_files,
                "kernel.py": hashlib.sha256(MODEL.encode()).hexdigest(),
            }
        }
    )
    contract = evaluation_contract(source).model_copy(
        update={
            "input_py": "def _make_inputs(value): return {'x': value}\n",
            "shapes": {
                "0": {"init_kwargs": {"expected": 7}, "input_kwargs": {"value": 11}},
                "1": {"input_kwargs": {"private": "must-not-upload"}},
            },
        }
    )
    context = AgateEvaluationContext("example", "test-cuda", Dsl.CUTEDSL, contract)
    sdk = DiagnosticDevClient(tmp_path / "jobs")
    client = SourceTreeAgateClient(sdk, None)  # Diagnostics do not need Atrex Bench.
    store = SqliteAgateJobStore(tmp_path / "jobs.sqlite")
    adapter = AgateGatewayAdapter(
        client, builder, SimpleNamespace(resolve=lambda _: context), store, wait_timeout_s=40
    )
    yield SimpleNamespace(
        seed=seed, adapter=adapter, client=client, sdk=sdk, context=context, store=store
    )
    store.close()


def submit(diagnostic, operation, parameters=None):
    operation = GatewayOperation(operation)
    parameters = parameters or {}
    request = GatewayAdapterRequest(
        attempt_id=new_attempt_id(),
        operation=operation,
        idempotency_key="diagnostic-test",
        candidate_path=diagnostic.seed.working,
        candidate_digest=diagnostic.context.kernel_source.seed_digest,
        kernel_regex=None,
        job_id=None,
        parameters=parameters,
        profile_level=parameters.get("level", "sol")
        if operation is GatewayOperation.PROFILE
        else None,
    )
    payload = diagnostic.adapter._build_request(request, diagnostic.context)
    kind = "compile" if operation is GatewayOperation.CHECK else operation.value
    accepted = diagnostic.client.submit_job(kind, payload)
    # Restart with no local request/job map. Durable ownership is in the adapter's job store.
    job = SourceTreeAgateClient(diagnostic.sdk, None).get_job(accepted["job_id"])
    diagnostic.adapter._require_source_diagnostic(diagnostic.context, operation, job)
    return job


@pytest.mark.parametrize("level", ["survey", "sol", "deep"])
def test_profile_full_driver_source_execution_and_projection(diagnostic, level):
    job = submit(
        diagnostic,
        "profile",
        {
            "level": level,
            "kernel_name": "my_kernel",
            "source": True,
            "top_kernels": 1,
            "counters": ["sm__cycles_elapsed.avg"],
            "launch_skip": 2,
            "launch_count": 3,
        },
    )
    value = job["result"]
    assert value["passed"] and value["clock_lock"]["requested"] is False
    assert value["kernels"][0]["duration_us"] == 2
    assert value["kernels"][0]["compute_sol_pct"] == 60
    assert value["kernels"][0]["memory_sol_pct"] == 40
    assert len(value["kernels"]) == 1
    assert "MOV" in value["exports"]["sass.txt"]["text"]
    assert "logs" not in project_private_job(job)["result"]
    root = diagnostic.sdk.root / "job-0"
    assert (root / ".caches/triton/launches").read_text() == "xx"
    calls = [json.loads(line) for line in (root / "tool.log").read_text().splitlines()]
    assert calls[0][calls[0].index("--launch-skip") + 1] == "2"
    assert calls[0][calls[0].index("--launch-count") + 1] == "3"
    assert calls[0][calls[0].index("--kernel-name") + 1] == "regex:^my_kernel$"
    assert {"LaunchStats", "SpeedOfLight", "full"}.intersection(calls[0])
    raw_export = next(call for call in calls if "--page" in call and "raw" in call)
    assert "--print-metric-name" not in raw_export
    assert raw_export[-4:] == ["raw", "--csv", "--print-units", "base"]
    staged = diagnostic.sdk.requests[0]["files"]
    assert "candidate/example_kernel/impl/value.py" in staged
    assert "must-not-upload" not in staged["request.json"]
    assert "reference.py" not in staged


@pytest.mark.parametrize("sanitize", [None, "memcheck", "racecheck", "initcheck", "synccheck"])
def test_check_compiles_and_launches_but_does_not_claim_correctness(diagnostic, sanitize):
    job = submit(diagnostic, "check", {"sanitize": sanitize, "arch": "sm_103"})
    value = project_compile_job(job)["result"]
    assert value["passed"] and value["compile_ok"] and value["launch_ok"]
    assert value["correctness_checked"] is False
    assert value["sanitizer_passed"] is (True if sanitize else None)
    root = diagnostic.sdk.root / "job-0"
    assert (root / ".caches/triton/launches").read_text() == "x"
    if sanitize:
        args = json.loads((root / "tool.log").read_text())
        assert args[args.index("--tool") + 1] == sanitize
        assert args[args.index("--error-exitcode") + 1] == "86"


@pytest.mark.parametrize("fmt", ["auto", "sass", "ptx"])
def test_disassemble_exports_real_tool_text(diagnostic, fmt):
    job = submit(diagnostic, "disassemble", {"fmt": fmt})
    value = project_compile_job(job)["result"]
    assert value["passed"]
    assert value["format"] == ("sass" if fmt == "auto" else fmt)
    text = value["exports"][f"{value['format']}.txt"]["text"]
    assert (".version" if fmt == "ptx" else "MOV") in text
    assert value["kernels"][0]["duration_us"] == 2


@pytest.mark.parametrize(
    "operation,parameters,stage",
    [
        ("check", {"sanitize": "memcheck", "env_vars": {"FAKE_SANITIZER_ERROR": "1"}}, "execution"),
        ("check", {"env_vars": {"FAKE_CANDIDATE_ERROR": "1"}}, "execution"),
        ("check", {"arch": "sm_90"}, "execution"),
        ("profile", {"env_vars": {"FAKE_EMPTY": "1"}}, "collection"),
        ("profile", {"env_vars": {"FAKE_MISSING_NCU": "1"}}, "environment"),
        ("disassemble", {"fmt": "ptx", "env_vars": {"FAKE_NO_PTX": "1"}}, "execution"),
    ],
)
def test_failures_are_durable_and_never_report_pass(diagnostic, operation, parameters, stage):
    job = submit(diagnostic, operation, parameters)
    assert job["status"] == "succeeded"  # Delivery succeeded; diagnostic did not.
    assert job["result"]["passed"] is False
    assert job["result"]["failure_stage"] == stage
    raw = json.dumps(job["source_tree_execution"])
    assert DIAGNOSTIC_PREFIX in raw
    safe = project_compile_job(job) if operation != "profile" else project_private_job(job)
    assert "PRIVATE_INPUT" not in json.dumps(safe)


def test_missing_structured_result_is_not_success(diagnostic):
    with pytest.raises(InfrastructureError, match="structured result"):
        diagnostic.adapter._require_source_diagnostic(
            diagnostic.context, GatewayOperation.PROFILE, {"status": "succeeded", "result": {}}
        )


@pytest.mark.parametrize(
    "operation,parameters",
    [
        ("profile", {"profiler": "rocprofv3"}),
        ("disassemble", {"fmt": "isa"}),
    ],
)
def test_unsupported_tools_rejected_before_submission(diagnostic, operation, parameters):
    with pytest.raises(ValueError):
        submit(diagnostic, operation, parameters)
    assert diagnostic.sdk.requests == []


def test_export_limits_and_missing_metrics_are_explicit():
    assert _kernels("NCU banner but no metrics") == []
    exported = _export("x" * (TEXT_LIMIT + 5))
    assert exported["truncated"] is True
    assert exported["size_bytes"] == TEXT_LIMIT + 5
    assert len(exported["text"]) == TEXT_LIMIT


@pytest.mark.parametrize("text", [METRICS, LONG_METRICS])
def test_ncu_csv_layouts_preserve_metric_names_units_and_values(text):
    kernels = _kernels("==NCU banner==\n" + text)
    assert len(kernels) == 2
    assert kernels[0]["name"] == "my_kernel"
    assert kernels[0]["duration_us"] == 2
    assert kernels[0]["compute_sol_pct"] == 60
    assert kernels[0]["memory_sol_pct"] == 40
    assert kernels[1]["duration_us"] == 1


def test_raw_csv_handles_units_headers_and_unavailable_metrics():
    kernels = _kernels(METRICS + METRICS.replace('"123"', '"456"'))
    assert len(kernels) == 4  # Same launch IDs in distinct processes are separate.
    assert kernels[0]["registers_per_thread"] == 168
    assert kernels[1]["metrics"]["gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed"] == {
        "value": None, "unit": "%",
    }
    assert "memory_sol_pct" not in kernels[1]
    assert "compute_sol_pct" not in kernels[1]
    assert _kernels("\n".join(METRICS.splitlines()[:2])) == []


@pytest.mark.anyio
async def test_adapter_records_logical_job_and_recovered_poll_without_evaluation(diagnostic):
    request = GatewayAdapterRequest(
        attempt_id=new_attempt_id(),
        operation=GatewayOperation.CHECK,
        idempotency_key="bound",
        candidate_path=diagnostic.seed.working,
        candidate_digest=diagnostic.context.kernel_source.seed_digest,
        kernel_regex=None,
        job_id=None,
        profile_level=None,
        parameters={"sanitize": "racecheck"},
    )
    result = await diagnostic.adapter.execute(request)
    assert result.status == "completed" and result.evaluation is None
    binding = diagnostic.store.require_owned(request.attempt_id, result.job_id)
    assert binding.operation is GatewayOperation.CHECK and binding.kind == "compile"
    restarted = AgateGatewayAdapter(
        SourceTreeAgateClient(diagnostic.sdk, None),
        builder,
        SimpleNamespace(resolve=lambda _: diagnostic.context),
        diagnostic.store,
        wait_timeout_s=40,
    )
    poll = replace(request, operation=GatewayOperation.POLL, job_id=result.job_id, parameters={})
    recovered = await restarted.execute(poll)
    assert recovered.worker_result == result.worker_result
    assert recovered.evaluation is None and len(diagnostic.sdk.requests) == 1


@pytest.mark.anyio
@pytest.mark.parametrize("operation", ["profile", "check", "disassemble"])
async def test_diagnostic_restart_scopes_jobs_to_recovery_generation(diagnostic, operation):
    operation = GatewayOperation(operation)
    request = GatewayAdapterRequest(
        attempt_id=new_attempt_id(),
        operation=operation,
        idempotency_key="same-agent-request-after-restart",
        candidate_path=diagnostic.seed.working,
        candidate_digest=diagnostic.context.kernel_source.seed_digest,
        kernel_regex=None,
        job_id=None,
        profile_level="sol" if operation is GatewayOperation.PROFILE else None,
    )
    previous_jobs = set()
    for generation in range(3):
        current = replace(request, recovery_generation=generation)
        result = await diagnostic.adapter.execute(current)
        assert result.job_id not in previous_jobs
        previous_jobs.add(result.job_id)
        assert result.status == "completed" and result.evaluation is None
        binding = diagnostic.store.require_owned(request.attempt_id, result.job_id)
        assert binding.operation is operation
        if generation == 0:
            assert binding.idempotency_key == request.idempotency_key
        else:
            assert binding.idempotency_key != request.idempotency_key

        # Simulate a Runtime crash after binding/polling but before GatewayProxy
        # persisted the response. Dev submissions always allocate a fresh job in
        # this double, so recovery must look up the durable binding before submit.
        restarted = AgateGatewayAdapter(
            SourceTreeAgateClient(diagnostic.sdk, None),
            builder,
            SimpleNamespace(resolve=lambda _: diagnostic.context),
            diagnostic.store,
            wait_timeout_s=40,
        )
        replay = await restarted.execute(current)
        assert replay == result
        assert len(diagnostic.sdk.requests) == generation + 1
        poll = replace(current, operation=GatewayOperation.POLL, job_id=result.job_id)
        assert (await restarted.execute(poll)).worker_result == result.worker_result
    assert len(diagnostic.store.list_owned(request.attempt_id)) == 3
    assert len({p["idempotency_key"] for p in diagnostic.sdk.requests}) == 3


def test_allocation_limit_and_installed_dependencies(diagnostic):
    diagnostic.context = replace(
        diagnostic.context,
        contract=diagnostic.context.contract.model_copy(
            update={
                "options": diagnostic.context.contract.options.model_copy(update={"timeout_s": 900})
            },
        ),
    )
    job = submit(diagnostic, "check", {"requirements": ["packaging>=1"]})
    assert job["result"]["passed"]
    request = diagnostic.sdk.requests[0]
    assert request["timeout_s"] == 600
    assert json.loads(request["files"]["request.json"])["timeout_s"] == 570
    assert "requirements" not in request  # Dev does not have typed dependency installation.
