"""Explicit, process-local activation of the private Gateway dependencies."""

import os
import platform
import sys
from pathlib import Path


def activate() -> Path:
    if (
        sys.implementation.name != "cpython"
        or sys.version_info[:2] != (3, 12)
        or sys.platform != "linux"
        or platform.machine().lower() not in {"x86_64", "amd64"}
    ):
        raise RuntimeError("This bundle requires CPython 3.12 on Linux x86_64 (glibc >= 2.17)")
    vendor = Path(__file__).resolve().parent / "_vendor"
    path = str(vendor)
    if path not in sys.path:
        sys.path.insert(0, path)
    # Child Python interpreters must also see the original app and dependency modules.
    paths = [item for item in os.environ.get("PYTHONPATH", "").split(os.pathsep) if item]
    os.environ["PYTHONPATH"] = os.pathsep.join([path, *[item for item in paths if item != path]])
    return vendor
