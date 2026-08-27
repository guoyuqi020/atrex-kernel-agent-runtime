"""Trusted analytical Roofline Builder tests."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from atrex_runtime.gateway.contract import AgateEvaluationContractV1
from atrex_runtime.roofline import (
    AtrexBenchRooflineBuilder,
    strip_roofline_hardware_suffix,
    validate_roofline,
)


def _contract() -> AgateEvaluationContractV1:
    return AgateEvaluationContractV1.model_validate(
        {
            "schema_version": 1,
            "candidate_path": "kernel.py",
            "reference_py": "class Model: pass\n",
            "input_py": "def _make_inputs(): return {}\n",
            "shapes": {"0": {"input_kwargs": {"n": 1024}}},
            "metadata": {"shapes": {"0": {"dtype_compute": "fp32"}}},
            "options": {
                "num_correctness_cases": 1,
                "bench_iters": 10,
                "atol": 0.0,
                "rtol": 0.0,
                "timeout_s": 60,
            },
            "lock_clocks": True,
        }
    )


def _builder_repository(root: Path) -> tuple[Path, str]:
    repository = root / "atrex-bench"
    unrelated = repository / "data/unrelated-large-benchmark.bin"
    unrelated.parent.mkdir(parents=True)
    unrelated.write_bytes(b"x" * 1_100_000)
    script = repository / "skills/benchmark-converter/scripts/generate_roofline.py"
    script.parent.mkdir(parents=True)
    script.write_text(
        """import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', type=Path, required=True)
parser.add_argument('--op', required=True)
parser.add_argument('--sku')
args = parser.parse_args()
root = args.data_dir / args.op
shapes = json.loads((root / 'shapes.json').read_text())
roofline = {'shapes': {shape_id: {
    'semantic_W_flops': {'fp32': 1024},
    'semantic_Q_read_bytes': 8192,
    'semantic_Q_write_bytes': 4096,
    'SOL_time_ms': {args.sku or 'inferred': 0.01},
    'bottleneck': 'memory',
} for shape_id in shapes}}
(root / 'roofline.json').write_text(json.dumps(roofline))
""",
        encoding="utf-8",
    )
    subprocess.run(("/usr/bin/git", "init", str(repository)), check=True, capture_output=True)
    subprocess.run(
        ("/usr/bin/git", "-C", str(repository), "add", "."),
        check=True,
        capture_output=True,
    )
    subprocess.run(
        (
            "/usr/bin/git",
            "-C",
            str(repository),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-m",
            "builder",
        ),
        check=True,
        capture_output=True,
    )
    commit = subprocess.run(
        ("/usr/bin/git", "-C", str(repository), "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return repository, commit


def test_commit_pinned_atrex_bench_builder_generates_complete_roofline(
    tmp_path: Path,
) -> None:
    repository, commit = _builder_repository(tmp_path)
    builder = AtrexBenchRooflineBuilder(
        repository=str(repository),
        commit=commit,
        git_executable="/usr/bin/git",
        python_executable=sys.executable,
        fetch_timeout_seconds=10,
        execution_timeout_seconds=10,
        max_archive_bytes=1_000_000,
        max_output_bytes=100_000,
        sku_by_hardware_target={"L20N": "Test L20N"},
    )

    result = builder.build(
        operator="vector_add",
        hardware_target="L20N",
        contract=_contract(),
    )

    shape = result["shapes"]["0"]
    assert isinstance(shape, dict)
    assert shape["SOL_time_ms"] == {"Test L20N": 0.01}


def test_roofline_validation_requires_exact_shape_coverage() -> None:
    with pytest.raises(ValueError, match="exactly match"):
        validate_roofline({"shapes": {}}, expected_shape_ids={"0"})


def test_forwarded_roofline_keys_the_bare_device_name() -> None:
    source = {
        "shapes": {
            "0": {
                "semantic_Q_read_bytes": 8,
                "SOL_time_ms": {"NVIDIA RTX PRO 5000 72GB Blackwell (SM120)": 0.01},
            }
        }
    }

    result = strip_roofline_hardware_suffix(source)

    shape = result["shapes"]["0"]
    assert isinstance(shape, dict)
    assert shape["SOL_time_ms"] == {"NVIDIA RTX PRO 5000 72GB Blackwell": 0.01}
    assert shape["semantic_Q_read_bytes"] == 8
    assert source["shapes"]["0"]["SOL_time_ms"] == {
        "NVIDIA RTX PRO 5000 72GB Blackwell (SM120)": 0.01
    }


def test_forwarded_roofline_rejects_colliding_hardware_keys() -> None:
    with pytest.raises(ValueError, match="two hardware keys"):
        strip_roofline_hardware_suffix(
            {"shapes": {"0": {"SOL_time_ms": {"L20N (SM120)": 0.01, "L20N": 0.02}}}}
        )


def test_builder_requires_metadata(tmp_path: Path) -> None:
    repository, commit = _builder_repository(tmp_path)
    builder = AtrexBenchRooflineBuilder(
        repository=str(repository),
        commit=commit,
        git_executable="/usr/bin/git",
        python_executable=sys.executable,
        fetch_timeout_seconds=10,
        execution_timeout_seconds=10,
        max_archive_bytes=1_000_000,
        max_output_bytes=100_000,
    )
    contract = _contract().model_copy(update={"metadata": None})

    with pytest.raises(ValueError, match="requires evaluation metadata"):
        builder.build(operator="vector_add", hardware_target="L20N", contract=contract)
