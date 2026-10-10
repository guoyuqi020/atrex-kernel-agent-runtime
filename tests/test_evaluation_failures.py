"""Device recovery must not retry real candidate failures or arbitrary log text."""

import pytest
from evaluation_failure_fixture import DEVICE_ERROR, device_failure_result

from atrex_runtime.gateway.evaluation_failures import embedded_device_failure


@pytest.mark.parametrize("message", [
    DEVICE_ERROR,
    "CUDA error: all CUDA-capable devices are busy or unavailable",
    "CUDA error: system not yet initialized",
    "cudaErrorDevicesUnavailable",
    "hggcErrorDevicesUnavailable",
    "RuntimeError: No CUDA GPUs are available",
])
@pytest.mark.parametrize("native_abba", [False, True])
def test_device_failure_in_eval_and_native_abba(message: str, native_abba: bool) -> None:
    result = device_failure_result("10", message)
    if native_abba:
        result = {"abba": {"sdk_results": [{"abba": {"runs": [{"result": result}]}}]}}
    assert embedded_device_failure({"status": "succeeded", "result": result}) == message


@pytest.mark.parametrize("message", [
    "CUDA out of memory", "CUDA error: an illegal memory access was encountered",
    "candidate timed out", "SyntaxError: invalid syntax", "numerical mismatch",
    "unspecified launch failure", "device-side assert triggered",
])
def test_candidate_and_unknown_errors_are_not_inferred_to_be_infrastructure(message: str) -> None:
    assert embedded_device_failure({
        "status": "succeeded", "result": device_failure_result("10", message),
    }) is None


def test_ignore_device_words_in_source_stdout_and_metadata() -> None:
    assert embedded_device_failure({
        "status": "succeeded",
        "result": {"stdout": DEVICE_ERROR, "metadata": {"error": DEVICE_ERROR}},
        "candidate": DEVICE_ERROR,
    }) is None


@pytest.mark.parametrize("status", ["running", "queued", "cancelled"])
def test_nonterminal_and_cancelled_jobs_are_not_recovered(status: str) -> None:
    assert embedded_device_failure({
        "status": status, "result": device_failure_result("10"),
    }) is None


def test_real_numerical_failure_vetoes_device_recovery_in_same_batch() -> None:
    result = device_failure_result("10")
    result["correctness"] = {"shapes": {
        "10": {"cases": []},
        "11": {"cases": [{"outputs": [{"passed": False, "max_elementwise_abs_diff": 1}]}]},
    }}
    assert embedded_device_failure({"status": "succeeded", "result": result}) is None
