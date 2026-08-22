from __future__ import annotations

from pathlib import Path

from atrex_runtime.domain.ids import new_attempt_id
from atrex_runtime.gateway.control_models import GatewayOperation
from atrex_runtime.gateway.measurement_history import normalized_measurement_points
from atrex_runtime.gateway.proxy import GatewayAdapterRequest, GatewayAdapterResult


def test_profile_result_is_normalized_to_safe_comparable_metrics(tmp_path: Path) -> None:
    request = GatewayAdapterRequest(
        attempt_id=new_attempt_id(),
        operation=GatewayOperation.PROFILE,
        idempotency_key="profile-1",
        candidate_digest=None,
        candidate_path=tmp_path,
        profile_level="sol",
        kernel_regex=None,
        job_id=None,
        parameters={"shape_id": "opaque-shape-1"},
    )
    result = GatewayAdapterResult(
        "completed",
        {"status": "succeeded"},
        worker_result={
            "status": "succeeded",
            "result": {
                "kernels": [
                    {
                        "name": "fused_kernel",
                        "bound": "memory",
                        "compute_sol_pct": 22.5,
                        "mem_sol_pct": 71.25,
                        "occupancy_pct": 48.0,
                        "duration": 12_500,
                        "duration_unit": "ns",
                        "traffic": {
                            "achieved_dram_gbps": 802.5,
                            "dram_bytes": 4096,
                        },
                        "request": {"private": "must not be retained"},
                    }
                ]
            },
        },
    )

    points = normalized_measurement_points(request, result)

    assert len(points) == 1
    point = points[0]
    assert point.kind is GatewayOperation.PROFILE
    assert point.shape_id == "opaque-shape-1"
    assert point.kernel_name == "fused_kernel"
    assert point.metrics == {
        "achieved_dram_gbps": 802.5,
        "bound": "memory",
        "compute_sol_pct": 22.5,
        "dram_bytes": 4096.0,
        "duration_us": 12.5,
        "memory_sol_pct": 71.25,
        "occupancy_pct": 48.0,
    }
    assert "request" not in point.metrics
