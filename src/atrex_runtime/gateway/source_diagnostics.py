#!/usr/bin/env python3
"""Self-contained NVIDIA source-tree diagnostics uploaded to an Agate Dev allocation.

No Runtime installation or ncu_report Python package is required on the worker.
Diagnostics are not correctness or performance Gate evidence.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import io
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import suppress
from pathlib import Path
from typing import Any

DIAGNOSTIC_PREFIX = "__ATREX_RUNTIME_SOURCE_DIAGNOSTIC__="
TEXT_LIMIT = 512 * 1024


def _load(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path.name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _target(request_path: Path) -> int:
    """One isolated JIT/launch probe; no hidden inputs are printed by this driver."""
    request = json.loads(request_path.read_text())
    stage = request_path.parent
    root = stage / "candidate"
    sys.path[:0] = [str(root), str(root / request["package_root"])]
    os.chdir(root)
    clocks = _load(stage / "__atrex_abba.py", "_atrex_clock_support")
    clocks._check_requirements(request["runtime_requirements"])
    if request.get("requirements"):
        from importlib.metadata import version

        from packaging.requirements import Requirement

        for raw in request["requirements"]:
            requirement = Requirement(raw)
            if requirement.marker and not requirement.marker.evaluate():
                continue
            if requirement.url or version(requirement.name) not in requirement.specifier:
                raise RuntimeError("diagnostic dependency is not provisioned in the GPU image")
    import torch  # type: ignore[import-not-found]

    if not torch.cuda.is_available() or getattr(torch.version, "hip", None):
        raise RuntimeError("source-tree diagnostics require a CUDA GPU")
    arch = request["parameters"].get("arch")
    if arch:
        actual = "".join(map(str, torch.cuda.get_device_capability()))
        requested = re.fullmatch(r"(?:sm_|compute_)?(\d+)([af]?)", arch.replace(".", ""))
        if requested is None or requested.group(1) != actual or requested.group(2):
            raise ValueError("arch must match the allocated GPU; cross-compilation is unsupported")
    torch.manual_seed(0)
    model_module = _load(root / request["entrypoint"], "kernel")
    input_module = _load(stage / "input.py", "_atrex_diagnostic_input")
    shape = next(iter(request["shapes"].values()))
    model = model_module.Model(**(shape.get("init_kwargs") or {})).eval()

    def inputs() -> dict[str, Any]:
        value = input_module._make_inputs(**(shape.get("input_kwargs") or {}))
        if not isinstance(value, dict):
            raise TypeError("_make_inputs must return a dictionary")
        return value

    with torch.no_grad():
        if request["operation"] != "check":
            model(**inputs())  # JIT and setup are outside the profiled NVTX range.
            torch.cuda.synchronize()
        values = inputs()  # Fresh buffers, including mutable output/state buffers.
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_push("ATREX_CANDIDATE")
        try:
            model(**values)
            torch.cuda.synchronize()
        finally:
            torch.cuda.nvtx.range_pop()
    return 0


class DiagnosticError(RuntimeError):
    def __init__(self, stage: str, message: str):
        super().__init__(message)
        self.stage = stage


def _execute(argv: list[str], root: Path, deadline: float, log: list[dict[str, Any]]) -> str:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise DiagnosticError("timeout", "diagnostic time budget exhausted")
    with subprocess.Popen(
        argv,
        cwd=root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        errors="replace",
        start_new_session=True,
    ) as process:
        try:
            stdout, stderr = process.communicate(timeout=remaining)
        except subprocess.TimeoutExpired as error:
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            stdout, stderr = process.communicate()
            log.append({"command": argv, "timeout": True, "stdout": stdout, "stderr": stderr})
            raise DiagnosticError("timeout", "diagnostic command timed out") from error
        code = process.returncode
    log.append({"command": argv, "exit_code": code, "stdout": stdout, "stderr": stderr})
    if code:
        raise DiagnosticError("execution", "compiler, GPU tool or candidate execution failed")
    return stdout


def _tool(name: str) -> str:
    result = shutil.which(name)
    if result is None:
        raise DiagnosticError("environment", f"required GPU tool is not installed: {name}")
    return result


def _export(text: str) -> dict[str, Any]:
    raw = text.encode("utf-8")
    return {
        "text": raw[:TEXT_LIMIT].decode("utf-8", errors="ignore"),
        "size_bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "truncated": len(raw) > TEXT_LIMIT,
    }


def _metric_rows(text: str) -> Iterator[dict[str, str]]:
    """Normalize raw-page columnar CSV and row-oriented metric CSV."""
    fields: list[str] = []
    metric_fields: list[str] = []
    units: dict[str, str] = {}
    for cells in csv.reader(io.StringIO(text)):
        if "ID" in cells and "Kernel Name" in cells:
            fields = cells
            metric_fields = [name for name in fields if "__" in name]
            units = {}
            continue
        if not fields or len(cells) != len(fields):
            continue
        row = dict(zip(fields, cells, strict=True))
        if "Metric Name" in fields and "Metric Value" in fields:
            yield row
            continue
        if not row.get("ID") and not row.get("Kernel Name"):
            units = row  # Raw-page units row, not a Kernel launch.
            continue
        if not row.get("Kernel Name"):
            continue
        for metric in metric_fields:
            yield {
                "ID": row["ID"],
                "Process ID": row.get("Process ID", ""),
                "Kernel Name": row["Kernel Name"],
                "Metric Name": metric,
                "Metric Unit": units.get(metric, ""),
                "Metric Value": row[metric],
            }


def _kernels(text: str) -> list[dict[str, Any]]:
    """Parse NCU CSV with exact metric names, not locale-dependent display labels."""
    kernels: dict[tuple[str, str], dict[str, Any]] = {}
    for row in _metric_rows(text):
        name = row.get("Kernel Name")
        metric = row.get("Metric Name")
        if not name or not metric or name == "Kernel Name" or metric == "Metric Name":
            continue
        key = (row.get("Process ID", ""), row.get("ID", name))
        kernel = kernels.setdefault(key, {"name": name, "metrics": {}})
        value: Any = row.get("Metric Value", "")
        try:
            number = float(value.replace(",", ""))
            value = number if math.isfinite(number) else None
        except (ValueError, AttributeError):
            pass
        unit = row.get("Metric Unit", "")
        kernel["metrics"][metric] = {"value": value, "unit": unit}
        if isinstance(value, (int, float)):
            if metric == "gpu__time_duration.sum":
                scale = {
                    "nsecond": 0.001,
                    "usecond": 1,
                    "msecond": 1000,
                    "second": 1_000_000,
                    "ns": 0.001,
                    "us": 1,
                    "ms": 1000,
                }.get(unit)
                if scale is not None:
                    kernel["duration_us"] = value * scale
            for source, target in (
                ("sm__throughput.avg.pct_of_peak_sustained_elapsed", "compute_sol_pct"),
                ("dram__throughput.avg.pct_of_peak_sustained_elapsed", "memory_sol_pct"),
                ("gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed", "memory_sol_pct"),
                ("launch__registers_per_thread", "registers_per_thread"),
            ):
                if metric == source:
                    kernel[target] = value
    return list(kernels.values())


def _ncu_command(
    ncu: str,
    request: dict[str, Any],
    report: Path,
    target: list[str],
) -> list[str]:
    parameters = request["parameters"]
    level = parameters.get("level", "sol")
    argv = [
        ncu,
        "--force-overwrite",
        "--export",
        str(report),
        "--clock-control",
        "none",
        "--nvtx",
        "--nvtx-include",
        "ATREX_CANDIDATE/",
        "--launch-skip",
        str(parameters.get("launch_skip") or 0),
        "--launch-count",
        str(parameters.get("launch_count") or 10),
        "--kernel-name-base",
        "demangled",
    ]
    if request["operation"] == "disassemble" or level == "survey":
        argv += ["--section", "LaunchStats"]
    elif level == "deep":
        argv += ["--set", "full"]
    else:
        argv += ["--section", "SpeedOfLight"]
    if parameters.get("source"):
        argv += ["--section", "SourceCounters", "--import-source", "yes"]
    metrics = ["gpu__time_duration.sum", *(parameters.get("counters") or [])]
    argv += ["--metrics", ",".join(dict.fromkeys(metrics))]
    name = parameters.get("kernel_name")
    regex = parameters.get("kernel_regex")
    if name:
        argv += ["--kernel-name", f"regex:^{re.escape(name)}$"]
    elif regex:
        argv += ["--kernel-name", f"regex:{regex}"]
    return argv + target


def _collect(
    request: dict[str, Any],
    root: Path,
    deadline: float,
    log: list[dict[str, Any]],
) -> dict[str, Any]:
    operation, parameters = request["operation"], request["parameters"]
    target = [
        sys.executable,
        str(root / "__atrex_diagnostic.py"),
        "--target",
        str(root / "request.json"),
    ]
    if operation == "check":
        sanitizer = parameters.get("sanitize")
        if sanitizer:
            target = [
                _tool("compute-sanitizer"),
                "--tool",
                sanitizer,
                "--error-exitcode",
                "86",
                "--target-processes",
                "all",
                *target,
            ]
        _execute(target, root, deadline, log)
        return {
            "compile_ok": True,
            "launch_ok": True,
            "sanitize": sanitizer,
            "sanitizer_passed": True if sanitizer else None,
            "scope": "one_shape_launch_probe",
            "correctness_checked": False,
        }

    ncu = _tool("ncu")
    report = root / "diagnostic.ncu-rep"
    _execute(_ncu_command(ncu, request, report, target), root, deadline, log)
    if not report.is_file():
        raise DiagnosticError(
            "collection", "NCU produced no report; check kernel filter/launch range"
        )
    # The raw CSV page already exports canonical metric names. NCU restricts
    # --print-metric-name to the details page and rejects it for raw exports.
    raw = _execute(
        [
            ncu,
            "--import",
            str(report),
            "--page",
            "raw",
            "--csv",
            "--print-units",
            "base",
        ],
        root,
        deadline,
        log,
    )
    kernels = _kernels(raw)
    if not kernels:
        raise DiagnosticError(
            "collection", "no kernels collected; check kernel filter/launch range"
        )
    result: dict[str, Any] = {
        "profiler": "ncu",
        "kernels": kernels,
        "exports": {"metrics.csv": _export(raw)},
    }
    if operation == "profile":
        result["level"] = parameters.get("level", "sol")
        top = parameters.get("top_kernels")
        if top:
            result["collected_kernel_count"] = len(kernels)
            result["kernels"] = sorted(
                kernels, key=lambda k: k.get("duration_us", 0), reverse=True
            )[:top]
    if operation == "disassemble" or parameters.get("source"):
        fmt = parameters.get("fmt", "auto")
        fmt = "sass" if fmt == "auto" else fmt
        view = "cuda,sass" if operation == "profile" else fmt
        assembly = _execute(
            [ncu, "--import", str(report), "--page", "source", "--print-source", view],
            root,
            deadline,
            log,
        )
        # Tool banners alone are not evidence that assembly was obtained.
        signature = (
            r"(?:\.version|\.target|\.entry)\b"
            if fmt == "ptx"
            else r"(?:/\*[0-9a-fA-F]+\*/|\b0x[0-9a-fA-F]+\b)"
        )
        if not re.search(signature, assembly):
            raise DiagnosticError(
                "assembly",
                "assembly unavailable in NCU report; try sass or check toolchain support",
            )
        result["exports"][f"{fmt}.txt"] = _export(assembly)
        result["format"] = fmt
    return result


def _driver(request_path: Path) -> int:
    root = request_path.resolve().parent
    request = json.loads(request_path.read_text())
    parameters = request["parameters"]
    deadline = time.monotonic() + float(request["timeout_s"])
    log: list[dict[str, Any]] = []
    result: dict[str, Any] = {
        "operation": request["operation"],
        "shape_id": next(iter(request["shapes"])),
        "passed": False,
        "status": "error",
    }
    try:
        # Always override caller caches; each job compiles the exact uploaded tree.
        for name, folder in (
            ("CUTE_DSL_CACHE_DIR", "cute"),
            ("TRITON_CACHE_DIR", "triton"),
            ("TORCH_EXTENSIONS_DIR", "torch"),
            ("TMPDIR", "tmp"),
            ("CUDA_CACHE_PATH", "cuda"),
            ("CUTE_DSL_DUMP_DIR", "dump"),
        ):
            path = root / ".caches" / folder
            path.mkdir(parents=True, exist_ok=True)
            os.environ[name] = str(path)
        if parameters.get("fmt") == "ptx":
            os.environ["CUTE_DSL_KEEP"] = "ptx,cubin"
            os.environ["CUTE_DSL_KEEP_PTX"] = "1"
        clocks = _load(root / "__atrex_abba.py", "_atrex_clock_support")
        with clocks._clock_lock(bool(request["lock_clocks"])) as report:
            result.update(_collect(request, root, deadline, log))
            result["clock_lock"] = report
        result.update(passed=True, status="passed")
    except DiagnosticError as error:
        result.update(error=str(error), failure_stage=error.stage)
    except Exception as error:
        log.append({"error": f"{type(error).__name__}: {error}"})
        result.update(
            error="diagnostic environment or clock setup failed", failure_stage="environment"
        )
    if request["operation"] == "check" and not result["passed"]:
        result.update(
            compile_ok=None,
            launch_ok=False,
            sanitize=parameters.get("sanitize"),
            sanitizer_passed=False if parameters.get("sanitize") else None,
        )
    # Kept in trusted raw evidence; normal private-result projection removes logs.
    result["logs"] = [_export(json.dumps(item)) for item in log]
    print(DIAGNOSTIC_PREFIX + json.dumps(result), flush=True)
    # A failed diagnostic is a recorded tool result, not a lost Dev transport job.
    return 0


def main() -> int:
    if len(sys.argv) == 3 and sys.argv[1] == "--target":
        return _target(Path(sys.argv[2]).resolve())
    return _driver(Path(sys.argv[1]).resolve())


if __name__ == "__main__":
    raise SystemExit(main())
