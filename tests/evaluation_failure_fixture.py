"""Small faithful reproduction of CSA's device failure before correctness cases."""

from atrex_runtime.artifacts.local import JsonValue

DEVICE_ERROR = (
    "torch.AcceleratorError: CUDA error: "
    "all HGGC-capable devices are busy or unavailable"
)


def device_failure_result(shape_id: str, message: str = DEVICE_ERROR) -> dict[str, JsonValue]:
    return {
        "error": None,
        "passed": {
            "compile": {shape_id: {"status": "failed", "reason": message}},
            "correctness": {shape_id: {"status": "failed", "reason": message}},
            "performance": {shape_id: {"status": "skipped"}},
        },
        "correctness": {"shapes": {shape_id: {"cases": []}}},
    }
