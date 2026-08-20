#!/usr/bin/env python3
"""Self-contained remote driver for one same-allocation ABBA Shape batch.

This module is uploaded as source and executed inside an Agate dev allocation. It
must therefore use only the Python standard library plus the bundled Atrex Bench.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import statistics
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

RESULT_PREFIX = "__ATREX_RUNTIME_ABBA_RESULT__="
RUN_TIMEOUT_GRACE_SECONDS = 60
_CLOCK_MARKERS = (
    "ATREX_BENCH_CLOCKS_LOCKED",
    "SOL_EXECBENCH_CLOCKS_LOCKED",
)
_CLOCK_SOURCE = "ATREX_BENCH_CLOCK_LOCK_SOURCE"


def _run_clock_command(argv: list[str]) -> str:
    completed = subprocess.run(
        argv,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    if completed.returncode != 0:
        detail = next(
            (line.strip() for line in completed.stderr.splitlines() if line.strip()),
            "no stderr",
        )
        raise RuntimeError(
            f"clock command exited {completed.returncode}: {' '.join(argv)}; {detail}"
        )
    return completed.stdout


def _nvidia_selector(executable: str) -> str:
    explicit = os.environ.get("ATREX_BENCH_CLOCK_DEVICE", "").strip()
    if explicit:
        return explicit
    for name in ("CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES"):
        value = os.environ.get(name, "").strip()
        tokens = tuple(token.strip() for token in value.split(",") if token.strip())
        if len(tokens) == 1 and tokens[0].lower() not in {"all", "none", "void"}:
            return tokens[0]
    rows = [
        line.strip()
        for line in _run_clock_command(
            [
                executable,
                "--query-gpu=index,uuid",
                "--format=csv,noheader,nounits",
            ]
        ).splitlines()
        if line.strip()
    ]
    if len(rows) != 1:
        raise RuntimeError(
            "clock locking requires exactly one visible NVIDIA GPU or an explicit selector"
        )
    parts = [part.strip() for part in rows[0].split(",")]
    if len(parts) != 2 or not all(parts):
        raise RuntimeError(f"could not parse NVIDIA GPU identity: {rows[0]!r}")
    return parts[1]


@contextmanager
def _clock_lock(enabled: bool) -> Iterator[dict[str, object]]:
    if not enabled:
        yield {"requested": False, "applied": False, "backend": None}
        return

    reset: list[str]
    report: dict[str, object]
    nvidia_smi = shutil.which("nvidia-smi")
    rocm_smi = shutil.which("rocm-smi")
    if nvidia_smi is not None:
        selector = _nvidia_selector(nvidia_smi)
        supported = _run_clock_command(
            [
                nvidia_smi,
                "-i",
                selector,
                "--query-supported-clocks=gr",
                "--format=csv,noheader,nounits",
            ]
        )
        clocks: list[float] = []
        for line in supported.splitlines():
            text = line.strip().split(",")[-1].strip()
            if not text:
                continue
            try:
                clocks.append(float(text))
            except ValueError:
                continue
        if not clocks:
            raise RuntimeError("nvidia-smi returned no supported graphics clocks")
        graphics_mhz = int(max(clocks))
        _run_clock_command(
            [
                nvidia_smi,
                "-i",
                selector,
                "--lock-gpu-clocks",
                f"{graphics_mhz},{graphics_mhz}",
            ]
        )
        reset = [nvidia_smi, "-i", selector, "--reset-gpu-clocks"]
        report = {
            "requested": True,
            "applied": True,
            "backend": "nvidia-smi",
            "device_selector": selector,
            "graphics_mhz": graphics_mhz,
        }
    elif rocm_smi is not None:
        _run_clock_command([rocm_smi, "--setperflevel", "high"])
        reset = [rocm_smi, "--setperflevel", "auto"]
        report = {
            "requested": True,
            "applied": True,
            "backend": "rocm-smi",
            "performance_level": "high",
        }
    else:
        raise RuntimeError("clock locking requires nvidia-smi or rocm-smi")

    saved = {name: os.environ.get(name) for name in (*_CLOCK_MARKERS, _CLOCK_SOURCE)}
    for name in _CLOCK_MARKERS:
        os.environ[name] = "1"
    os.environ[_CLOCK_SOURCE] = "atrex-runtime-abba"
    try:
        yield report
    finally:
        reset_error: BaseException | None = None
        try:
            _run_clock_command(reset)
        except BaseException as error:  # fail closed after attempting environment restoration
            reset_error = error
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        if reset_error is not None:
            raise RuntimeError(f"failed to reset GPU clocks: {reset_error}") from reset_error
        report["restored"] = True


def _positive_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if number > 0 and math.isfinite(number) else None


def _nonnegative_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if number >= 0 and math.isfinite(number) else None


def _status_passed(value: object) -> bool:
    return isinstance(value, dict) and value.get("status") == "passed"


def _compile_passed(value: object, shape_ids: list[str]) -> bool:
    if not isinstance(value, dict):
        return False
    aggregate = value.get("status")
    if aggregate is not None:
        return bool(aggregate == "passed")
    return all(_status_passed(value.get(shape_id)) for shape_id in shape_ids)


def _summarize(payload: object, shape_ids: list[str]) -> dict[str, object]:
    if not isinstance(payload, dict):
        return {"all_pass": False, "latency_us_by_shape": {}, "error": "invalid result"}
    passed = payload.get("passed")
    passed = passed if isinstance(passed, dict) else {}
    correctness = payload.get("correctness")
    correctness = correctness if isinstance(correctness, dict) else {}
    correctness_shapes = correctness.get("shapes")
    correctness_shapes = correctness_shapes if isinstance(correctness_shapes, dict) else {}
    performance = payload.get("performance")
    performance = performance if isinstance(performance, dict) else {}
    performance_shapes = performance.get("shapes")
    performance_shapes = performance_shapes if isinstance(performance_shapes, dict) else {}

    correct = (
        payload.get("error") is None
        and _compile_passed(passed.get("compile"), shape_ids)
        and isinstance(passed.get("correctness"), dict)
        and all(_status_passed(passed["correctness"].get(shape_id)) for shape_id in shape_ids)
        and all(shape_id in correctness_shapes for shape_id in shape_ids)
    )
    latency_by_shape: dict[str, float] = {}
    sol_pct_by_shape: dict[str, float] = {}
    for shape_id in shape_ids:
        shape = performance_shapes.get(shape_id)
        if not isinstance(shape, dict) or shape.get("error") is not None:
            correct = False
            continue
        samples = shape.get("samples")
        values = []
        for sample in samples if isinstance(samples, list) else []:
            if not isinstance(sample, dict):
                continue
            number = _positive_number(sample.get("end_to_end_time_ms"))
            if number is not None:
                values.append(number)
        if not values:
            correct = False
            continue
        latency_by_shape[shape_id] = statistics.median(values) * 1000.0
        sol = shape.get("sol")
        if isinstance(sol, dict):
            percentage = _nonnegative_number(sol.get("pct"))
            if percentage is not None:
                sol_pct_by_shape[shape_id] = percentage
    correct = correct and len(latency_by_shape) == len(shape_ids) and bool(shape_ids)
    latencies = [
        latency_by_shape[shape_id] for shape_id in shape_ids if shape_id in latency_by_shape
    ]
    return {
        "all_pass": correct,
        "latency_us_geomean": (
            math.exp(statistics.fmean(math.log(value) for value in latencies)) if correct else None
        ),
        "latency_us_by_shape": latency_by_shape,
        "sol_pct_by_shape": sol_pct_by_shape,
        "error": None if correct else "compile, correctness, or performance did not pass",
    }


def _single(config_path: Path, result_path: Path) -> int:
    try:
        root = Path.cwd()
        runtime_src = str(root / "atrex-bench" / "src")
        sys.path.insert(0, runtime_src)
        os.environ["PYTHONPATH"] = (
            runtime_src
            if not os.environ.get("PYTHONPATH")
            else runtime_src + os.pathsep + os.environ["PYTHONPATH"]
        )
        from atrex_bench import evaluate  # type: ignore[import-not-found]

        request = json.loads(config_path.read_text(encoding="utf-8"))
        shape_ids = request.pop("shape_ids")
        payload = evaluate(request)
        result = _summarize(payload, shape_ids)
    except Exception as error:
        result = {
            "all_pass": False,
            "latency_us_geomean": None,
            "latency_us_by_shape": {},
            "sol_pct_by_shape": {},
            "error": f"{type(error).__name__}: {error}"[:2000],
        }
    result_path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    return 0


def _driver(request_path: Path) -> int:
    runs: list[dict[str, object]] = []
    try:
        request = json.loads(request_path.read_text(encoding="utf-8"))
        if request.get("schema_version") != 1:
            raise ValueError("unsupported ABBA request schema")
        schedule = request["schedule"]
        shape_ids = request["shape_ids"]
        sources = request["sources"]
        evaluator = request["evaluator"]
        timeout = float(request["per_run_timeout_seconds"]) + RUN_TIMEOUT_GRACE_SECONDS
        lock_clocks = request.get("lock_clocks", True)
        if not isinstance(schedule, list) or not schedule:
            raise ValueError("ABBA schedule must be non-empty")
        if not isinstance(shape_ids, list) or not all(isinstance(item, str) for item in shape_ids):
            raise ValueError("ABBA shape_ids must be strings")
        if not isinstance(sources, dict) or set(sources) != {"incumbent", "candidate"}:
            raise ValueError("ABBA sources are incomplete")
        if not isinstance(evaluator, dict):
            raise ValueError("ABBA evaluator config is invalid")
        if not isinstance(lock_clocks, bool):
            raise ValueError("ABBA lock_clocks must be a boolean")

        root = Path.cwd()
        kernel = root / "kernel.py"
        with _clock_lock(lock_clocks) as clock_report:
            for index, step in enumerate(schedule):
                if not isinstance(step, dict):
                    raise ValueError("ABBA schedule entry must be an object")
                revision = step.get("revision")
                repeat = step.get("repeat")
                if revision not in {"incumbent", "candidate"} or not isinstance(repeat, int):
                    raise ValueError("invalid ABBA schedule entry")
                source_path = root / str(sources[revision])
                shutil.copyfile(source_path, kernel)
                config = dict(evaluator)
                config.update(
                    {
                        "input": str(kernel),
                        "reference_dir": str(root / "reference"),
                        "output": str(root / "outputs" / f"run-{index:04d}"),
                        "shape_ids": shape_ids,
                    }
                )
                config_path = root / f"single-{index:04d}.json"
                result_path = root / f"single-{index:04d}-result.json"
                config_path.write_text(json.dumps(config, ensure_ascii=False), encoding="utf-8")
                try:
                    process = subprocess.run(
                        [
                            sys.executable,
                            str(Path(__file__).resolve()),
                            "--single",
                            str(config_path),
                            str(result_path),
                        ],
                        capture_output=True,
                        text=True,
                        timeout=timeout,
                        env=os.environ.copy(),
                    )
                    result = (
                        json.loads(result_path.read_text(encoding="utf-8"))
                        if result_path.is_file()
                        else None
                    )
                    exit_code = process.returncode
                    stdout_tail = process.stdout[-3000:]
                    stderr_tail = process.stderr[-3000:]
                except subprocess.TimeoutExpired as error:
                    result = None
                    exit_code = -1
                    stdout_tail = str(error.stdout or "")[-3000:]
                    stderr_tail = "evaluation timed out"
                runs.append(
                    {
                        "revision": revision,
                        "repeat": repeat,
                        "exit_code": exit_code,
                        "result": result,
                        "stdout_tail": stdout_tail,
                        "stderr_tail": stderr_tail,
                    }
                )
        payload = {
            "schema_version": 1,
            "runs": runs,
            "clock_lock": clock_report,
            "error": None,
        }
    except Exception as error:
        payload = {
            "schema_version": 1,
            "runs": runs,
            "error": f"{type(error).__name__}: {error}"[:2000],
        }
    print(
        RESULT_PREFIX + json.dumps(payload, ensure_ascii=False, separators=(",", ":")), flush=True
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) == 3 and args[0] == "--single":
        return _single(Path(args[1]), Path(args[2]))
    if len(args) == 1:
        return _driver(Path(args[0]))
    print("usage: abba_remote.py REQUEST | --single CONFIG RESULT", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
