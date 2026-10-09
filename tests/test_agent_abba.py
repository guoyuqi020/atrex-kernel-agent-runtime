"""Agent native Eval ABBA preserves trusted evidence without promotion authority."""

from __future__ import annotations

import json
import threading
import time
from copy import deepcopy
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import pytest
from atrex_gateway_client import build_eval_request_from_content
from pydantic import ValidationError

from atrex_runtime.artifacts.local import ArtifactKind, LocalArtifactStore
from atrex_runtime.domain.errors import InfrastructureError
from atrex_runtime.domain.ids import new_attempt_id
from atrex_runtime.domain.models import Dsl
from atrex_runtime.gateway.agent_abba import AgentAbbaGatewayAdapter
from atrex_runtime.gateway.contract import (
    AgateEvaluationContext,
    AgateEvaluationContractV1,
    AgateEvaluationOptionsV1,
)
from atrex_runtime.gateway.control_models import GatewayOperation
from atrex_runtime.gateway.proxy import GatewayAdapterRequest, GatewayAdapterResult
from atrex_runtime.kernel_sources import KernelSourceContract

_PRIVATE = "private-input-reference-or-log-must-not-escape"


@dataclass
class FakeDelegate:
    requests: list[GatewayAdapterRequest] = field(default_factory=list)

    async def execute(self, request: GatewayAdapterRequest) -> GatewayAdapterResult:
        self.requests.append(request)
        return GatewayAdapterResult("completed", {"delegated": True})


@dataclass
class FakeEvaluator:
    commit: str = "a" * 40
    calls: int = 0
    digest_calls: int = 0

    def files(self) -> dict[str, str]:
        self.calls += 1
        return {"atrex-bench/src/atrex_bench/__init__.py": ""}

    def bundle_digest(self) -> str:
        self.digest_calls += 1
        return "sha256:" + "b" * 64


@dataclass
class FakeContexts:
    context: AgateEvaluationContext

    def resolve(self, _attempt_id: object) -> AgateEvaluationContext:
        return self.context


def _native_request_builder(
    candidate: str | dict[str, object],
    reference: dict[str, object],
    gpu: str,
    **fields: object,
) -> dict[str, object]:
    return {"candidate": candidate, "reference": reference, "gpu": gpu, **fields}


class NativeFakeAgate:
    """Return native ABBA evidence while enforcing upstream idempotency."""

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.jobs: dict[str, dict[str, Any]] = {}
        self.keys: dict[str, str] = {}
        self.failure: str | None = None
        self.vary_repeats = False
        self.fetch_delay = 0.0
        self.active = 0
        self.peak_active = 0
        self._lock = threading.Lock()

    def submit_job(self, kind: str, request: dict[str, Any]) -> dict[str, Any]:
        assert kind == "eval"
        assert "command" not in request and "files" not in request
        with self._lock:
            key = request["idempotency_key"]
            if key in self.keys:
                return {"job_id": self.keys[key], "status": "queued"}
            job_id = f"ev_agent_abba_{len(self.requests)}"
            self.keys[key] = job_id
            self.requests.append(deepcopy(request))
            blocks = self._blocks(request)
            job: dict[str, Any] = {
                "job_id": job_id,
                "status": "succeeded",
                "command_ok": True,
                "result": {"abba": {"sdk_results": blocks, "valid": True}},
                "stdout_tail": _PRIVATE,
                "stderr_tail": _PRIVATE,
            }
            first = blocks[0]["abba"]["runs"][0]
            if self.failure == "schedule":
                first["revision"] = "candidate"
            elif self.failure == "missing_shape":
                first["result"]["performance"]["shapes"] = {"999": {"samples": []}}
            elif self.failure == "result_error":
                first["result"]["error"] = _PRIVATE
            elif self.failure == "incorrect":
                for block in blocks:
                    for run in block["abba"]["runs"]:
                        if run["revision"] == "candidate":
                            for value in run["result"]["passed"]["correctness"].values():
                                value["status"] = "failed"
            elif self.failure == "sdk_evidence":
                job["result"] = {"error": _PRIVATE, "abba": {"sdk_results": None}}
            if self.failure == "job":
                job.update(status="failed", error={"reason": "code_execution_failed",
                                                   "details": {"logs_tail": _PRIVATE}})
            elif self.failure == "queued":
                job.update(status="running", result=None)
            elif self.failure in {"logs_once", "infra_once"} and len(self.requests) == 1:
                job.update(status="failed", error={
                    "error_class": "infra",
                    "reason": "logs_unavailable" if self.failure == "logs_once" else "exec_failed",
                    "details": {"backend_state": "succeeded"},
                })
            self.jobs[job_id] = job
        return {"job_id": job_id, "status": "queued"}

    def _blocks(self, request: dict[str, Any]) -> list[dict[str, Any]]:
        shape_ids = list(request["reference"]["shapes"])
        expected: list[dict[str, Any]] = [
            {"index": 0, "revision": "baseline", "label": "A", "repeat": 0},
            {"index": 1, "revision": "candidate", "label": "B", "repeat": 0},
            {"index": 2, "revision": "candidate", "label": "B", "repeat": 1},
            {"index": 3, "revision": "baseline", "label": "A", "repeat": 1},
        ]
        blocks = []
        for index in range(request["abba"]["repeats"]):
            runs = []
            for step in expected:
                factor = 1 if step["revision"] == "baseline" else 0.5
                if self.vary_repeats and index * 2 + step["repeat"]:
                    factor *= 4 if step["revision"] == "baseline" else 9
                raw = {
                    "error": None,
                    "private_shape": _PRIVATE,
                    "passed": {
                        "compile": {"status": "passed"},
                        "correctness": {
                            shape_id: {"status": "passed"} for shape_id in shape_ids
                        },
                    },
                    "correctness": {
                        "shapes": {
                            shape_id: {
                                "max_elementwise_abs_diff": 0.001,
                                "max_elementwise_rel_diff": 0.002,
                            }
                            for shape_id in shape_ids
                        }
                    },
                    "performance": {
                        "shapes": {
                            shape_id: {
                                "error": None,
                                "samples": [{
                                    "end_to_end_time_ms": (
                                        (100 if shape_id == "0" else 400) * factor / 1000
                                    ),
                                }],
                            }
                            for shape_id in shape_ids
                        }
                    },
                }
                runs.append({**step, "result": raw})
            blocks.append({
                "eval_mode": "abba",
                "error": None,
                "passed": {"abba": {"status": "passed"}},
                "abba": {"schedule": expected, "runs": runs, "comparison": {}},
            })
        return blocks

    def get_job(
        self, job_id: str, wait: bool = False, timeout: float = 30.0,
        include_spec: bool = False,
    ) -> dict[str, Any]:
        assert wait and timeout == 90 and not include_spec
        with self._lock:
            self.active += 1
            self.peak_active = max(self.active, self.peak_active)
        try:
            if self.fetch_delay:
                time.sleep(self.fetch_delay)
            return deepcopy(self.jobs[job_id])
        finally:
            with self._lock:
                self.active -= 1


@dataclass
class Case:
    adapter: AgentAbbaGatewayAdapter
    request: GatewayAdapterRequest
    client: NativeFakeAgate
    contexts: FakeContexts
    evaluator: FakeEvaluator
    artifacts: LocalArtifactStore
    delegate: FakeDelegate


@pytest.fixture
def case(tmp_path: Path) -> Case:
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    for name in ("baseline", "candidate"):
        source = tmp_path / name
        source.mkdir()
        source.joinpath("kernel.py").write_text(f"SIDE = {name!r}\n", encoding="utf-8")
    baseline = artifacts.put_directory(tmp_path / "baseline", ArtifactKind.KERNEL)
    candidate = artifacts.put_directory(tmp_path / "candidate", ArtifactKind.KERNEL)
    contract = AgateEvaluationContractV1(
        candidate_path="kernel.py", reference_py=f"# {_PRIVATE}\nclass Model: pass\n",
        input_py=f"# {_PRIVATE}\ndef _make_inputs(n): return (n,)\n",
        shapes={"0": {"input_kwargs": {"n": 12345}}, "1": {"input_kwargs": {"n": 67890}}},
        metadata={"shapes": {"0": {"hidden": _PRIVATE}, "1": {"hidden": _PRIVATE}}},
        roofline={"shapes": {"0": {"SOL_time_ms": {"Test GPU": 1}}}},
        options=AgateEvaluationOptionsV1(
            num_correctness_cases=1, bench_iters=10, atol=0.01, rtol=0.02, timeout_s=60,
        ),
        env_vars={"TRUSTED_ENV": "yes"}, lock_clocks=True,
    )
    contexts = FakeContexts(AgateEvaluationContext("vector_add", "H20", Dsl.TRITON, contract))
    client = NativeFakeAgate()
    delegate = FakeDelegate()
    evaluator = FakeEvaluator()
    adapter = AgentAbbaGatewayAdapter(
        delegate, client, contexts, artifacts, evaluator,  # type: ignore[arg-type]
        build_eval_request_from_content, wait_timeout_s=90, correctness_cases=3, bench_iters=17,
    )
    request = GatewayAdapterRequest(
        attempt_id=new_attempt_id(), operation=GatewayOperation.EVALUATE,
        idempotency_key="agent-abba-request", candidate_digest=candidate,
        candidate_path=artifacts.verify(candidate).payload_path,
        profile_level=None, kernel_regex=None, job_id=None,
        baseline_candidate_digest=baseline,
        baseline_candidate_path=artifacts.verify(baseline).payload_path,
        parameters={"comparison": {"method": "abba"}},
    )
    return Case(adapter, request, client, contexts, evaluator, artifacts, delegate)


@pytest.mark.anyio
@pytest.mark.parametrize("lock_clocks", [False, True])
async def test_agent_abba_uses_exact_sources_schedule_and_gate_policy(
    case: Case, lock_clocks: bool,
) -> None:
    contract = case.contexts.context.contract.model_copy(update={"lock_clocks": lock_clocks})
    case.contexts.context = replace(case.contexts.context, contract=contract)
    original = contract.model_dump(mode="json")

    result = await case.adapter.execute(case.request)

    assert result.status == "completed" and result.evaluation is None
    assert result.profile_result is None and case.delegate.requests == []
    assert len(case.client.requests) == 2
    assert case.evaluator.calls == 0
    for submitted in case.client.requests:
        assert submitted["spec"]["target_hardware"] == ["H20"]
        assert submitted["spec"]["languages"] == ["triton"]
        assert submitted["env_vars"] == {"TRUSTED_ENV": "yes"}
        assert submitted["candidate"] == "SIDE = 'candidate'\n"
        assert submitted["abba"] == {"baseline": "SIDE = 'baseline'\n", "repeats": 1}
        assert submitted["reference"]["reference_py"] == contract.reference_py
        assert submitted["reference"]["input_py"] == contract.input_py
        assert submitted.get("lock_clocks", False) is lock_clocks
        assert submitted["mode"] == "full"
        config = submitted["options"]
        assert config["timeout_s"] == 600
        assert config["atol"] == 0.01 and config["rtol"] == 0.02
        assert config["num_correctness_cases"] == 3 and config["bench_iters"] == 17
        assert len(submitted["reference"]["shapes"]) == 1
        assert "command" not in submitted and "files" not in submitted
    public = result.worker_result
    assert public["schedule"] == [
        {"side": "A", "repeat": 0}, {"side": "B", "repeat": 0},
        {"side": "B", "repeat": 1}, {"side": "A", "repeat": 1},
    ]
    assert public["baseline_kernel_artifact_digest"] == case.request.baseline_candidate_digest
    assert public["kernel_artifact_digest"] == case.request.candidate_digest
    assert public["baseline"]["latency_us_geomean"] == pytest.approx(200)
    assert public["candidate"]["latency_us_geomean"] == pytest.approx(100)
    assert public["speedup"] == pytest.approx(2)
    assert public["improvement_pct"] == pytest.approx(50)
    assert public["correct"] is True and public["shape_batch_count"] == 2
    assert len(public["measurements"]) == 4
    assert public["mode"] == "full" and public["input_scope"] == "contract"
    assert public["comparison"] == {"method": "abba", "repeats": 2}
    assert "repeats" not in public
    encoded = json.dumps(public)
    assert _PRIVATE not in encoded and "12345" not in encoded and "67890" not in encoded
    assert "stdout" not in encoded and "reference_py" not in encoded
    assert "baseline_kernel_trial_id" not in public
    assert contract.model_dump(mode="json") == original


@pytest.mark.anyio
async def test_agent_single_file_abba_uses_native_agate_eval(case: Case) -> None:
    client = NativeFakeAgate()
    adapter = AgentAbbaGatewayAdapter(
        case.delegate,
        client,
        case.contexts,
        case.artifacts,
        case.evaluator,
        _native_request_builder,  # type: ignore[arg-type]
        wait_timeout_s=90,
        correctness_cases=3,
        bench_iters=17,
    )

    result = await adapter.execute(case.request)

    assert result.status == "completed"
    assert len(client.requests) == 2
    assert case.evaluator.calls == 0
    assert all(request["abba"]["repeats"] == 1 for request in client.requests)
    assert all(request["options"]["timeout_s"] == 600 for request in client.requests)
    assert result.worker_result["baseline"]["latency_us_geomean"] == pytest.approx(200)
    assert result.worker_result["candidate"]["latency_us_geomean"] == pytest.approx(100)
    assert result.result["execution_transport"] == "agate_native_eval_abba"


@pytest.mark.anyio
async def test_agent_multi_file_abba_uses_native_eval_without_custom_evaluator(
    case: Case, tmp_path: Path,
) -> None:
    digests = []
    for label in ("baseline", "candidate"):
        tree = tmp_path / f"tree-{label}"
        (tree / "impl").mkdir(parents=True)
        (tree / "kernel.py").write_text("from impl.kernel import run\n", encoding="utf-8")
        (tree / "impl/kernel.py").write_text(f"SIDE = {label!r}\n", encoding="utf-8")
        digests.append(case.artifacts.put_directory(tree, ArtifactKind.KERNEL))
    source_contract = KernelSourceContract(
        source_revision="a" * 40, seed_digest=digests[0], package_root=".",
        editable_roots=("kernel.py", "impl"), immutable_files={},
        runtime_requirements=({"distribution": "triton", "import": "triton", "version": ">=3"},),
    )
    contract = case.contexts.context.contract.model_copy(update={
        "kernel_sources": {Dsl.TRITON: source_contract},
        "requirements": ("numpy>=2",),
    })
    case.contexts.context = replace(case.contexts.context, contract=contract)
    request = replace(
        case.request,
        baseline_candidate_digest=digests[0],
        baseline_candidate_path=case.artifacts.verify(digests[0]).payload_path,
        candidate_digest=digests[1],
        candidate_path=case.artifacts.verify(digests[1]).payload_path,
    )
    client = NativeFakeAgate()
    adapter = AgentAbbaGatewayAdapter(
        case.delegate, client, case.contexts, case.artifacts, None,
        build_eval_request_from_content, wait_timeout_s=90, correctness_cases=3, bench_iters=17,
    )

    result = await adapter.execute(request)
    replayed = await adapter.execute(request)

    assert result.status == replayed.status == "completed"
    assert result.result["execution_transport"] == "agate_native_eval_abba"
    assert result.result["evaluator_bundle_digest"] is None
    assert len(client.requests) == 2  # One same-allocation ABBA per Shape, with stable retry keys.
    assert len({payload["idempotency_key"] for payload in client.requests}) == 2
    assert result.worker_result == replayed.worker_result
    assert result.worker_result["shape_batch_count"] == 2
    assert result.worker_result["speedup"] == pytest.approx(2)
    assert _PRIVATE not in json.dumps(result.worker_result)
    for payload in client.requests:
        assert len(payload["reference"]["shapes"]) == 1
        assert payload["candidate"] == {
            "archive": "archives/candidate.tar.gz", "entry_point": "kernel.py"
        }
        assert payload["abba"] == {
            "baseline": {"archive": "archives/baseline.tar.gz", "entry_point": "kernel.py"},
            "repeats": 1,
        }
        archives = payload["__atrex_eval_archives"]
        assert archives["archives/baseline.tar.gz"]["impl/kernel.py"] == "SIDE = 'baseline'\n"
        assert archives["archives/candidate.tar.gz"]["impl/kernel.py"] == "SIDE = 'candidate'\n"
        assert payload["requirements"] == ["numpy>=2", "triton>=3"]
        assert payload["options"]["timeout_s"] == 600
        assert payload["lock_clocks"] is True
        assert "files" not in payload and "command" not in payload


@pytest.mark.anyio
async def test_agent_abba_aggregates_both_shapes_and_repeats_geometrically(case: Case) -> None:
    case.client.vary_repeats = True
    result = await case.adapter.execute(case.request)
    public = result.worker_result
    assert public["baseline"]["latency_us_geomean"] == pytest.approx(400)
    assert public["candidate"]["latency_us_geomean"] == pytest.approx(300)
    assert public["speedup"] == pytest.approx(4 / 3)
    assert public["improvement_pct"] == pytest.approx(25)


@pytest.mark.anyio
async def test_agent_abba_uses_nested_comparison_repeats_for_each_side(case: Case) -> None:
    adapter = AgentAbbaGatewayAdapter(
        case.delegate, case.client, case.contexts, case.artifacts, case.evaluator,  # type: ignore[arg-type]
        build_eval_request_from_content, wait_timeout_s=90, per_run_timeout_seconds=60,
    )
    result = await adapter.execute(replace(case.request, parameters={
        "comparison": {"method": "abba", "repeats": 4},
    }))
    assert result.evaluation is None
    assert result.worker_result["comparison"] == {"method": "abba", "repeats": 4}
    assert result.worker_result["schedule"] == [
        {"side": "A", "repeat": 0}, {"side": "B", "repeat": 0},
        {"side": "B", "repeat": 1}, {"side": "A", "repeat": 1},
        {"side": "A", "repeat": 2}, {"side": "B", "repeat": 2},
        {"side": "B", "repeat": 3}, {"side": "A", "repeat": 3},
    ]
    assert len(result.worker_result["measurements"]) == 8
    assert "repeats" not in result.worker_result


@pytest.mark.anyio
@pytest.mark.parametrize("overrides", [
    {"input_py": "def _make_inputs(n): return (n + 1,)\n"},
    {"shapes": {"7": {"input_kwargs": {"n": 88}}}},
    {"input_py": "def _make_inputs(n): return (n,)\n", "shapes": {"7": {"n": 88}}},
])
async def test_agent_abba_custom_components_apply_to_both_sides_without_mutation(
    case: Case, overrides: dict[str, Any],
) -> None:
    original = case.contexts.context.contract.model_dump(mode="json")
    before = deepcopy(overrides)
    result = await case.adapter.execute(replace(
        case.request, parameters={**case.request.parameters, **overrides},
    ))
    for submitted in case.client.requests:
        reference = submitted["reference"]
        assert "metadata" not in reference and "roofline" not in reference
        assert reference["reference_py"] == original["reference_py"]
        assert reference["input_py"] == overrides.get("input_py", original["input_py"])
        expected = overrides.get("shapes", original["shapes"])
        assert set(reference["shapes"]).issubset(expected)
    assert result.worker_result["input_scope"] == "custom"
    assert case.contexts.context.contract.model_dump(mode="json") == original
    assert overrides == before


@pytest.mark.anyio
async def test_agent_abba_replay_is_idempotent_and_generation_scoped(case: Case) -> None:
    first = await case.adapter.execute(case.request)
    replay = await case.adapter.execute(case.request)
    assert first == replay and len(case.client.requests) == 2
    original_keys = set(case.client.keys)
    rotated = await case.adapter.execute(replace(case.request, recovery_generation=1))
    assert len(case.client.requests) == 4
    assert len(set(case.client.keys) - original_keys) == 2
    assert first.result["comparison_id"] != rotated.result["comparison_id"]


@pytest.mark.anyio
@pytest.mark.parametrize("failure", ["incorrect", "schedule", "missing_shape", "sdk_evidence",
                                      "result_error", "job"])
async def test_agent_abba_negative_or_malformed_outcomes_never_report_speedup(
    case: Case, failure: str,
) -> None:
    case.client.failure = failure
    result = await case.adapter.execute(case.request)
    measured_failure = failure in {"incorrect", "missing_shape", "result_error"}
    assert result.status == ("completed" if measured_failure else "failed")
    assert result.evaluation is None
    assert result.worker_result["correct"] is False
    assert result.worker_result["speedup"] is None
    assert result.worker_result["improvement_pct"] is None
    failed_side = "baseline" if failure in {"missing_shape", "result_error"} else "candidate"
    assert result.worker_result[failed_side]["correctness"]["status"] == "FAIL"
    assert _PRIVATE not in json.dumps(result.worker_result)
    assert len(result.result["jobs"]) == 2


@pytest.mark.anyio
async def test_agent_abba_rejects_oversized_schedule_before_evaluator_export(case: Case) -> None:
    with pytest.raises(ValueError, match=r"repeats=4.*990s.*600s"):
        await case.adapter.execute(replace(case.request, parameters={
            "comparison": {"method": "abba", "repeats": 4},
        }))
    assert case.client.requests == [] and case.evaluator.calls == 0


@pytest.mark.anyio
@pytest.mark.parametrize("repeats", (0, 1, 3, 17, 18, 20, True, False, 2.0, "2", None))
async def test_agent_abba_rejects_invalid_repeats_before_budget_or_dev(
    case: Case, repeats: object,
) -> None:
    with pytest.raises(ValidationError) as failure:
        await case.adapter.execute(replace(case.request, parameters={
            "comparison": {"method": "abba", "repeats": repeats},
        }))
    assert "2, 4, 6, 8, 10, 12, 14, 16" in str(failure.value)
    assert "measurements per side; 2 means A, B, B, A" in str(failure.value)
    assert f"got {repeats!r}" in str(failure.value)
    assert "not executed through Dev" in str(failure.value)
    assert case.client.requests == [] and case.evaluator.calls == 0


@pytest.mark.anyio
@pytest.mark.parametrize("parameters", [
    {"comparison": {"method": "abba", "repeats": 1}},
    {"comparison": {"method": "abba", "repeats": 21}},
    {"comparison": {"method": "unknown"}}, {"comparison": {}},
    {"repeats": 2}, {"mode": "correctness_only"},
    {"mode": "correctness"}, {"reference_py": "replace oracle"},
    {"baseline_kernel_trial_id": "gtrial_" + "c" * 32},
])
async def test_agent_abba_validates_direct_adapter_parameters(
    case: Case, parameters: dict[str, Any],
) -> None:
    with pytest.raises(ValidationError):
        await case.adapter.execute(replace(
            case.request, parameters={**case.request.parameters, **parameters},
        ))
    assert case.client.requests == [] and case.evaluator.calls == 0


@pytest.mark.anyio
@pytest.mark.parametrize("with_evaluator", [False, True])
@pytest.mark.parametrize("source_tree", [False, True])
async def test_agent_abba_requires_native_builder_without_export_or_gpu_submission(
    case: Case, tmp_path: Path, with_evaluator: bool, source_tree: bool,
) -> None:
    request = case.request
    if source_tree:
        digests = []
        for label in ("baseline", "candidate"):
            root = tmp_path / f"missing-builder-{label}"
            root.mkdir()
            (root / "kernel.py").write_text("from helper import SIDE\n", encoding="utf-8")
            (root / "helper.py").write_text(f"SIDE = {label!r}\n", encoding="utf-8")
            digests.append(case.artifacts.put_directory(root, ArtifactKind.KERNEL))
        source_contract = KernelSourceContract(
            source_revision="a" * 40, seed_digest=digests[0], package_root=".",
            editable_roots=("kernel.py", "helper.py"), immutable_files={},
            runtime_requirements=(),
        )
        contract = case.contexts.context.contract.model_copy(update={
            "kernel_sources": {Dsl.TRITON: source_contract},
        })
        case.contexts.context = replace(case.contexts.context, contract=contract)
        request = replace(
            request,
            baseline_candidate_digest=digests[0],
            baseline_candidate_path=case.artifacts.verify(digests[0]).payload_path,
            candidate_digest=digests[1],
            candidate_path=case.artifacts.verify(digests[1]).payload_path,
        )
    adapter = AgentAbbaGatewayAdapter(
        case.delegate, case.client, case.contexts, case.artifacts,
        case.evaluator if with_evaluator else None,  # type: ignore[arg-type]
        request_builder=None, wait_timeout_s=90,
    )

    with pytest.raises(ValueError) as failure:
        await adapter.execute(request)

    assert str(failure.value) == (
        "Native Agate Eval request builder is unavailable. ABBA requires native Eval; "
        "Dev fallback is disabled. Ask the Runtime operator to repair the Agate SDK configuration."
    )
    assert case.client.requests == [] and case.client.jobs == {}
    assert case.delegate.requests == []
    assert case.evaluator.calls == case.evaluator.digest_calls == 0


@pytest.mark.anyio
@pytest.mark.parametrize("operation,parameters", [
    (GatewayOperation.DEV, {"comparison": {"method": "abba"}}),
    (GatewayOperation.EVALUATE, {}),
    (GatewayOperation.EVALUATE, {"comparison": None}),
    (GatewayOperation.EVALUATE, {"mode": "correctness_only"}),
    (GatewayOperation.EVALUATE, {"input_py": "def _make_inputs(): return ()\n",
                                 "shapes": {"4": {}}}),
])
async def test_agent_abba_keeps_non_comparison_delegation_lazy_without_evaluator(
    case: Case, operation: GatewayOperation, parameters: dict[str, Any],
) -> None:
    adapter = AgentAbbaGatewayAdapter(
        case.delegate, case.client, case.contexts, case.artifacts, None,  # type: ignore[arg-type]
        wait_timeout_s=90,
    )
    delegated = replace(case.request, operation=operation, parameters=parameters)
    assert (await adapter.execute(delegated)).result == {"delegated": True}
    assert case.delegate.requests == [delegated]
    assert case.client.requests == [] and case.evaluator.calls == 0


@pytest.mark.anyio
async def test_agent_abba_caps_shape_batch_concurrency(case: Case) -> None:
    case.client.fetch_delay = 0.02
    case.contexts.context = replace(case.contexts.context, contract=(
        case.contexts.context.contract.model_copy(update={
            "shapes": {str(index): {} for index in range(20)},
        })
    ))
    result = await case.adapter.execute(case.request)
    assert result.worker_result["correct"] is True
    assert len(case.client.requests) == 20
    assert 1 < case.client.peak_active <= 16


@pytest.mark.anyio
@pytest.mark.parametrize("failure", ["logs_once", "infra_once"])
async def test_agent_abba_infra_recovery_retains_exact_schedule_and_source(
    case: Case, monkeypatch: pytest.MonkeyPatch, failure: str,
) -> None:
    async def no_delay(_seconds: float) -> None:
        return None

    monkeypatch.setattr("atrex_runtime.gateway.job_recovery.anyio.sleep", no_delay)
    case.client.failure = failure
    result = await case.adapter.execute(case.request)
    assert result.worker_result["correct"] is True
    assert len(case.client.requests) == 3
    original = case.client.requests[0]
    prefix = "logs-retry:" if failure == "logs_once" else "infra-retry:"
    retried = next(row for row in case.client.requests
                   if row["idempotency_key"].startswith(prefix))
    assert {key: value for key, value in retried.items() if key != "idempotency_key"} == {
        key: value for key, value in original.items() if key != "idempotency_key"
    }
    assert retried["abba"] == original["abba"]
    assert retried["candidate"] == original["candidate"]


@pytest.mark.anyio
async def test_agent_abba_does_not_cache_incomplete_jobs(case: Case) -> None:
    case.client.failure = "queued"
    with pytest.raises(InfrastructureError, match="did not reach a terminal state"):
        await case.adapter.execute(case.request)
