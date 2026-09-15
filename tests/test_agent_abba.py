"""Agent source-based ABBA preserves the trusted runner without promotion authority."""

from __future__ import annotations

import json
import threading
import time
from copy import deepcopy
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from atrex_runtime.artifacts.local import ArtifactKind, LocalArtifactStore
from atrex_runtime.domain.errors import InfrastructureError
from atrex_runtime.domain.ids import new_attempt_id
from atrex_runtime.domain.models import Dsl
from atrex_runtime.gateway.abba_remote import RESULT_PREFIX
from atrex_runtime.gateway.agent_abba import AgentAbbaGatewayAdapter
from atrex_runtime.gateway.contract import (
    AgateEvaluationContext,
    AgateEvaluationContractV1,
    AgateEvaluationOptionsV1,
)
from atrex_runtime.gateway.control_models import GatewayOperation
from atrex_runtime.gateway.proxy import GatewayAdapterRequest, GatewayAdapterResult

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

    def files(self) -> dict[str, str]:
        self.calls += 1
        return {"atrex-bench/src/atrex_bench/__init__.py": ""}

    def bundle_digest(self) -> str:
        return "sha256:" + "b" * 64


@dataclass
class FakeContexts:
    context: AgateEvaluationContext

    def resolve(self, _attempt_id: object) -> AgateEvaluationContext:
        return self.context


class FakeAgate:
    """Return deterministic trusted-driver summaries while enforcing upstream idempotency."""

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
        assert kind == "dev"
        key = request["idempotency_key"]
        with self._lock:
            if key in self.keys:
                return {"job_id": self.keys[key], "status": "queued"}
            job_id = f"dv_agent_abba_{len(self.requests)}"
            self.keys[key] = job_id
            self.requests.append(deepcopy(request))
            driver = json.loads(request["files"]["request.json"])
            shape_ids = driver["shape_ids"]
            runs = []
            for step in driver["schedule"]:
                factor = 1 if step["revision"] == "incumbent" else 0.5
                if self.vary_repeats and step["repeat"]:
                    factor *= 4 if step["revision"] == "incumbent" else 9
                by_shape = {
                    shape_id: (100 if shape_id == "0" else 400) * factor
                    for shape_id in shape_ids
                }
                runs.append({
                    **step,
                    "exit_code": 0,
                    "result": {
                        "all_pass": True,
                        "correctness": {"max_abs_err": 0.001, "max_rel_err": 0.002},
                        "latency_us_by_shape": by_shape,
                        "private_shape": _PRIVATE,
                    },
                    "stdout_tail": _PRIVATE,
                    "stderr_tail": _PRIVATE,
                })
            payload: dict[str, Any] = {"schema_version": 1, "runs": runs, "error": None}
            if self.failure == "schedule":
                runs[0]["revision"] = "candidate"
            elif self.failure == "missing_shape":
                runs[0]["result"]["latency_us_by_shape"] = {"999": 1.0}
            elif self.failure == "result_error":
                runs[0]["result"]["error"] = _PRIVATE
            elif self.failure == "incorrect":
                for run in runs:
                    if run["revision"] == "candidate":
                        run["result"]["all_pass"] = False
                        run["result"]["error"] = _PRIVATE
            elif self.failure == "driver":
                payload["error"] = _PRIVATE
            job: dict[str, Any] = {
                "job_id": job_id, "status": "succeeded", "command_ok": True,
                "result": {"stdout": RESULT_PREFIX + json.dumps(payload), "exit_code": 0},
            }
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
    client: FakeAgate
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
    client = FakeAgate()
    delegate = FakeDelegate()
    evaluator = FakeEvaluator()
    adapter = AgentAbbaGatewayAdapter(
        delegate, client, contexts, artifacts, evaluator,  # type: ignore[arg-type]
        wait_timeout_s=90, correctness_cases=3, bench_iters=17,
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
    for submitted in case.client.requests:
        assert submitted["command"] == "python3 __atrex_abba.py request.json"
        assert submitted["spec"] == {"target_hardware": ["H20"]}
        assert submitted["timeout_s"] == 600
        assert submitted["env_vars"] == {"TRUSTED_ENV": "yes"}
        files = submitted["files"]
        assert files["snapshots/incumbent.py"] == "SIDE = 'baseline'\n"
        assert files["snapshots/candidate.py"] == "SIDE = 'candidate'\n"
        assert files["reference/reference.py"] == contract.reference_py
        assert files["reference/input.py"] == contract.input_py
        driver = json.loads(files["request.json"])
        assert driver["schedule"] == [
            {"revision": "incumbent", "repeat": 0}, {"revision": "candidate", "repeat": 0},
            {"revision": "candidate", "repeat": 1}, {"revision": "incumbent", "repeat": 1},
        ]
        assert driver["lock_clocks"] is lock_clocks
        config = driver["evaluator"]
        assert config["clock_lock_mode"] == ("external" if lock_clocks else "off")
        assert config["atol"] == 0.01 and config["rtol"] == 0.02
        assert config["num_correctness_cases"] == 3 and config["bench_iters"] == 17
        assert config["validation_mode"] == "full"
        assert len(driver["shape_ids"]) == 1
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
        wait_timeout_s=90, per_run_timeout_seconds=80,
    )
    result = await adapter.execute(replace(case.request, parameters={
        "comparison": {"method": "abba", "repeats": 3},
    }))
    assert result.evaluation is None
    assert result.worker_result["comparison"] == {"method": "abba", "repeats": 3}
    assert result.worker_result["schedule"] == [
        {"side": "A", "repeat": 0}, {"side": "B", "repeat": 0},
        {"side": "B", "repeat": 1}, {"side": "A", "repeat": 1},
        {"side": "A", "repeat": 2}, {"side": "B", "repeat": 2},
    ]
    assert len(result.worker_result["measurements"]) == 6
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
        files = submitted["files"]
        assert "reference/metadata.json" not in files and "reference/roofline.json" not in files
        assert files["reference/reference.py"] == original["reference_py"]
        assert files["reference/input.py"] == overrides.get("input_py", original["input_py"])
        expected = overrides.get("shapes", original["shapes"])
        assert set(json.loads(files["reference/shapes.json"])).issubset(expected)
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
@pytest.mark.parametrize("failure", ["incorrect", "schedule", "missing_shape", "driver",
                                      "result_error", "job"])
async def test_agent_abba_negative_or_malformed_outcomes_never_report_speedup(
    case: Case, failure: str,
) -> None:
    case.client.failure = failure
    result = await case.adapter.execute(case.request)
    assert result.status == ("completed" if failure == "incorrect" else "failed")
    assert result.evaluation is None
    assert result.worker_result["correct"] is False
    assert result.worker_result["speedup"] is None
    assert result.worker_result["improvement_pct"] is None
    assert result.worker_result["candidate"]["correctness"]["status"] == "FAIL"
    assert _PRIVATE not in json.dumps(result.worker_result)
    assert len(result.result["jobs"]) == 2


@pytest.mark.anyio
async def test_agent_abba_rejects_oversized_schedule_before_evaluator_export(case: Case) -> None:
    with pytest.raises(ValueError, match=r"repeats=3.*750s.*600s"):
        await case.adapter.execute(replace(case.request, parameters={
            "comparison": {"method": "abba", "repeats": 3},
        }))
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
    with pytest.raises(ValueError, match="commit-pinned Atrex Bench evaluator"):
        await adapter.execute(case.request)
    assert case.client.requests == []


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
    assert retried["files"] == original["files"]


@pytest.mark.anyio
async def test_agent_abba_does_not_cache_incomplete_jobs(case: Case) -> None:
    case.client.failure = "queued"
    with pytest.raises(InfrastructureError, match="did not reach a terminal state"):
        await case.adapter.execute(case.request)
