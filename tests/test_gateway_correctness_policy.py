from __future__ import annotations

from atrex_runtime.gateway.contract import AgateEvaluationContractV1


def test_agent_policy_preserves_aggregate_tolerance_bounds() -> None:
    contract = AgateEvaluationContractV1(
        candidate_path="kernel.py",
        reference_py="def reference():\n    return None\n",
        input_py="def _make_inputs():\n    return {}\n",
        shapes={"0": {}},
        metadata={
            "benchmark_contract": {
                "correctness_tolerances": {
                    "output": {
                        "atol": 0.01,
                        "rtol": 0.05,
                        "max_mismatch_fraction": 0.001,
                        "max_rel_l2": 0.01,
                    }
                }
            }
        },
        options={
            "num_correctness_cases": 1,
            "bench_iters": 1,
            "atol": 0.01,
            "rtol": 0.05,
            "timeout_s": 60,
        },
    )

    policy = contract.agent_correctness_policy().model_dump(mode="json")

    assert policy["output_tolerances"]["output"] == {
        "atol": 0.01,
        "rtol": 0.05,
        "max_mismatch_fraction": 0.001,
        "max_rel_l2": 0.01,
    }
