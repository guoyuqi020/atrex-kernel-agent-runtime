"""Remote same-allocation driver contract tests."""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest

from atrex_runtime.gateway import abba_remote


def test_remote_driver_executes_exact_abba_schedule_in_one_process_boundary(
    tmp_path: Path,
    monkeypatch: object,
    capsys: object,
) -> None:
    package = tmp_path / "atrex-bench/src/atrex_bench"
    package.mkdir(parents=True)
    package.joinpath("__init__.py").write_text(
        """
import json
from pathlib import Path

def evaluate(config):
    candidate = Path(config["input"]).read_text()
    latency_ms = 0.01 if "CANDIDATE" in candidate else 0.02
    shapes = json.loads((Path(config["reference_dir"]) / "shapes.json").read_text())
    ids = list(shapes)
    return {
        "error": None,
        "passed": {
            "compile": {"status": "passed"},
            "correctness": {shape_id: {"status": "passed"} for shape_id in ids},
        },
        "correctness": {"shapes": {shape_id: {} for shape_id in ids}},
        "performance": {
            "shapes": {
                shape_id: {
                    "error": None,
                    "samples": [{"end_to_end_time_ms": latency_ms}],
                    "sol": {"pct": 50.0 if "CANDIDATE" in candidate else 25.0},
                }
                for shape_id in ids
            }
        },
    }
""".strip()
        + "\n",
        encoding="utf-8",
    )
    reference = tmp_path / "reference"
    reference.mkdir()
    reference.joinpath("reference.py").write_text("", encoding="utf-8")
    reference.joinpath("input.py").write_text("", encoding="utf-8")
    reference.joinpath("shapes.json").write_text('{"s0": [1], "s1": [2]}', encoding="utf-8")
    snapshots = tmp_path / "snapshots"
    snapshots.mkdir()
    snapshots.joinpath("incumbent.py").write_text("INCUMBENT = True\n", encoding="utf-8")
    snapshots.joinpath("candidate.py").write_text("CANDIDATE = True\n", encoding="utf-8")
    schedule = [
        {"revision": "incumbent", "repeat": 0},
        {"revision": "candidate", "repeat": 0},
        {"revision": "candidate", "repeat": 1},
        {"revision": "incumbent", "repeat": 1},
    ]
    request = {
        "schema_version": 1,
        "schedule": schedule,
        "shape_ids": ["s0", "s1"],
        "sources": {
            "incumbent": "snapshots/incumbent.py",
            "candidate": "snapshots/candidate.py",
        },
        "evaluator": {},
        "per_run_timeout_seconds": 10,
        "lock_clocks": True,
    }
    request_path = tmp_path / "request.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")
    monkeypatch.chdir(tmp_path)  # type: ignore[attr-defined]
    lock_calls: list[bool] = []

    @contextmanager
    def fake_clock_lock(enabled: bool) -> Iterator[dict[str, object]]:
        lock_calls.append(enabled)
        yield {"requested": enabled, "applied": enabled, "backend": "test"}

    monkeypatch.setattr(abba_remote, "_clock_lock", fake_clock_lock)  # type: ignore[attr-defined]

    assert abba_remote.main([str(request_path)]) == 0

    output = capsys.readouterr().out  # type: ignore[attr-defined]
    payload = json.loads(output.removeprefix(abba_remote.RESULT_PREFIX))
    assert lock_calls == [True]
    assert payload["clock_lock"] == {
        "requested": True,
        "applied": True,
        "backend": "test",
    }
    assert [
        {"revision": run["revision"], "repeat": run["repeat"]} for run in payload["runs"]
    ] == schedule
    assert [run["result"]["latency_us_geomean"] for run in payload["runs"]] == pytest.approx(
        [20.0, 10.0, 10.0, 20.0]
    )
    assert [run["result"]["sol_pct_by_shape"]["s0"] for run in payload["runs"]] == [
        25.0,
        50.0,
        50.0,
        25.0,
    ]
    assert all(run["result"]["all_pass"] for run in payload["runs"])


def test_nvidia_clock_lock_uses_max_supported_clock_and_restores(
    monkeypatch: object,
) -> None:
    commands: list[list[str]] = []

    def fake_command(argv: list[str]) -> str:
        commands.append(argv)
        if "--query-gpu=index,uuid" in argv:
            return "0, GPU-aabb\n"
        if "--query-supported-clocks=gr" in argv:
            return "1200\n1500\n"
        return ""

    monkeypatch.setattr(abba_remote.shutil, "which", lambda name: f"/bin/{name}")  # type: ignore[attr-defined]
    monkeypatch.setattr(abba_remote, "_run_clock_command", fake_command)  # type: ignore[attr-defined]
    for name in (
        "ATREX_BENCH_CLOCK_DEVICE",
        "CUDA_VISIBLE_DEVICES",
        "NVIDIA_VISIBLE_DEVICES",
        "ATREX_BENCH_CLOCKS_LOCKED",
        "SOL_EXECBENCH_CLOCKS_LOCKED",
        "ATREX_BENCH_CLOCK_LOCK_SOURCE",
    ):
        monkeypatch.delenv(name, raising=False)  # type: ignore[attr-defined]

    with abba_remote._clock_lock(True) as report:
        assert report["graphics_mhz"] == 1500
        assert abba_remote.os.environ["ATREX_BENCH_CLOCKS_LOCKED"] == "1"
        assert abba_remote.os.environ["SOL_EXECBENCH_CLOCKS_LOCKED"] == "1"
        assert abba_remote.os.environ["ATREX_BENCH_CLOCK_LOCK_SOURCE"] == ("atrex-runtime-abba")

    assert report["restored"] is True
    assert "ATREX_BENCH_CLOCKS_LOCKED" not in abba_remote.os.environ
    assert commands[-1] == ["/bin/nvidia-smi", "-i", "GPU-aabb", "--reset-gpu-clocks"]
