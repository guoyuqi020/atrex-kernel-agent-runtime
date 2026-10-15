"""Dependency-free source validation shared with isolated Agent sessions.

This exact module is projected read-only; candidate checks and final sealing use
one implementation. It imports no Runtime state, credentials, or gateway tools.
"""

from __future__ import annotations

import hashlib
from pathlib import Path, PurePosixPath


def source_path(value: str, *, dot: bool = False) -> str:
    """Validate portable paths before Git, filesystem, or remote use."""
    if dot and value == ".":
        return value
    if not value or "\\" in value or "\x00" in value or value.startswith(("/", "-")):
        raise ValueError(f"unsafe source path: {value!r}")
    if any(part in {"", ".", ".."} for part in value.split("/")):
        raise ValueError(f"unsafe source path: {value!r}")
    return value


def ignored_source_path(path: PurePosixPath | Path, directory: bool = False) -> bool:
    """Only generated caches are omitted; source files never disappear silently."""
    return any(
        part in {".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
        for part in path.parts
    ) or path.suffix in {".pyc", ".pyo"}


def validate_source_tree(
    root: Path,
    *,
    package_root: str,
    editable_roots: tuple[str, ...] | list[str],
    immutable_files: dict[str, str],
) -> dict[str, str]:
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Kernel source root must be a real directory")
    files: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if path.is_symlink():
            raise ValueError(f"source tree cannot contain symlinks: {relative}")
        if ignored_source_path(relative) or path.is_dir():
            continue
        name = source_path(relative.as_posix())
        package_path = PurePosixPath(name)
        if package_root != "." and package_path.is_relative_to(package_root):
            package_path = package_path.relative_to(package_root)
        if package_path.parts[0].removesuffix(".py") in {
            "atrex_bench",
            "torch",
            "cutlass",
            "cuda",
            "triton",
            "site",
            "sitecustomize",
            "usercustomize",
            "importlib",
            "packaging",
            "json",
            "subprocess",
            "os",
            "sys",
        }:
            raise ValueError(f"source tree shadows a protected import: {name}")
        if not path.is_file():
            raise ValueError(f"source tree contains a special file: {name}")
        content = path.read_bytes()
        try:
            files[name] = content.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError(
                f"source tree must contain UTF-8 source, not build products: {name}"
            ) from error
        expected = immutable_files.get(name)
        if expected is not None:
            if hashlib.sha256(content).hexdigest() != expected:
                raise ValueError(f"fixed source file was modified: {name}")
        elif not any(name == item or name.startswith(item + "/") for item in editable_roots):
            raise ValueError(f"source file is outside editable_roots: {name}")
        elif path.suffix in {".so", ".a", ".o", ".cubin", ".whl", ".pth"}:
            raise ValueError(f"prebuilt/loading artifact is not permitted: {name}")
    missing = immutable_files.keys() - files.keys()
    if missing:
        raise ValueError(f"fixed source files were removed: {', '.join(sorted(missing))}")
    return files
