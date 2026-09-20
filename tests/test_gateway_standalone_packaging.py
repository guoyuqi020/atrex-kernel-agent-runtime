from __future__ import annotations

import csv
import hashlib
import io
import json
import runpy
import zipfile
from pathlib import Path

import pytest

BUILDER = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "scripts/packaging/build_gateway_standalone.py")
)
build = BUILDER["build"]
read_wheels = BUILDER["read_wheels"]
dependency_closure = BUILDER["dependency_closure"]
safe_path = BUILDER["safe_path"]
record_hash = BUILDER["record_hash"]


def _wheel(
    directory: Path,
    name: str,
    *,
    version: str = "1.0",
    requires: tuple[str, ...] = (),
    extras: tuple[str, ...] = (),
    tag: str = "py3-none-any",
    python: str = ">=3.9",
    files: dict[str, bytes] | None = None,
) -> Path:
    normalized = name.replace("-", "_")
    path = directory / f"{normalized}-{version}-{tag}.whl"
    metadata = (
        f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\nRequires-Python: {python}\n"
    )
    metadata += "".join(f"Requires-Dist: {value}\n" for value in requires)
    metadata += "".join(f"Provides-Extra: {value}\n" for value in extras)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(f"{normalized}-{version}.dist-info/METADATA", metadata)
        for member, data in (files or {}).items():
            archive.writestr(member, data)
    return path


def _roots(directory: Path, requires: tuple[str, ...] = ()) -> None:
    _wheel(directory, "atrex-gateway-server", requires=requires, files={"app/__init__.py": b""})
    _wheel(directory, "atrex-gateway-client", files={"atrex_gateway_client/__init__.py": b""})


def test_target_marker_extras_and_private_vendor_record(tmp_path: Path) -> None:
    _roots(tmp_path, ("helper[standard]>=2", "windows-only; sys_platform == 'win32'"))
    _wheel(
        tmp_path,
        "helper",
        version="2.0",
        requires=("native; extra == 'standard' and python_version == '3.12'",),
        extras=("standard",),
    )
    _wheel(tmp_path, "native", tag="cp312-cp312-manylinux_2_17_x86_64")
    output = build(tmp_path, tmp_path / "dist")
    with zipfile.ZipFile(output) as archive:
        manifest = json.loads(archive.read("atrex_gateway_standalone/manifest.json"))
        assert len(manifest["wheels"]) == 4
        assert (
            "Requires-Dist:"
            not in archive.read("atrex_gateway_standalone-1.0.dist-info/METADATA").decode()
        )
        assert "atrex_gateway_standalone/_vendor/app/__init__.py" in archive.namelist()
        assert "app/__init__.py" not in archive.namelist()
        assert b"atrex_gateway_standalone.cli:server_main" in archive.read(
            "atrex_gateway_standalone-1.0.dist-info/entry_points.txt"
        )
        rows = csv.reader(
            io.StringIO(archive.read("atrex_gateway_standalone-1.0.dist-info/RECORD").decode())
        )
        names = set()
        for member, digest, size in rows:
            names.add(member)
            if member.endswith(".dist-info/RECORD"):
                assert not digest and not size
            else:
                data = archive.read(member)
                assert record_hash(data) == digest
                assert len(data) == int(size)
        assert names == set(archive.namelist())
        for wheel in manifest["wheels"]:
            assert (
                wheel["sha256"]
                == hashlib.sha256((tmp_path / wheel["wheel"]).read_bytes()).hexdigest()
            )
    repeated = build(tmp_path, tmp_path / "second-dist")
    assert output.read_bytes() == repeated.read_bytes()
    with pytest.raises(FileExistsError):
        build(tmp_path, tmp_path / "dist")


@pytest.mark.parametrize("requirement", ["missing>=1", "helper>=3"])
def test_incomplete_or_incompatible_dependency_rejected(tmp_path: Path, requirement: str) -> None:
    _roots(tmp_path, (requirement,))
    _wheel(tmp_path, "helper", version="2.0")
    with pytest.raises(ValueError, match=r"Missing required wheel|Unsatisfied dependency"):
        dependency_closure(read_wheels(tmp_path))


@pytest.mark.parametrize(
    "tag", ["cp314-cp314-manylinux_2_17_x86_64", "cp312-cp312-manylinux_2_17_aarch64"]
)
def test_incompatible_native_wheel_rejected(tmp_path: Path, tag: str) -> None:
    _roots(tmp_path)
    _wheel(tmp_path, "native", tag=tag)
    with pytest.raises(ValueError, match="does not support"):
        read_wheels(tmp_path)


def test_requires_python_checked_against_target_not_host(tmp_path: Path) -> None:
    _roots(tmp_path)
    _wheel(tmp_path, "future", python=">=3.13")
    with pytest.raises(ValueError, match="requires Python"):
        read_wheels(tmp_path)


@pytest.mark.parametrize(
    "member", ["../escape", "/absolute", "bad\\path", "thing.data/scripts/a", "bad.pth"]
)
def test_unsafe_or_unsupported_wheel_members_rejected(member: str) -> None:
    with pytest.raises(ValueError):
        safe_path(zipfile.ZipInfo(member))


def test_conflicting_vendor_files_rejected(tmp_path: Path) -> None:
    _roots(tmp_path, ("other",))
    _wheel(tmp_path, "other", files={"app/__init__.py": b"conflict"})
    with pytest.raises(ValueError, match="Conflicting vendored files"):
        build(tmp_path, tmp_path / "dist")
