#!/usr/bin/env python3
"""Repackage a target-compatible Gateway wheelhouse into one offline wheel."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
import stat
import zipfile
from email.parser import BytesParser
from pathlib import Path, PurePosixPath

from packaging.markers import default_environment
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.tags import compatible_tags, cpython_tags
from packaging.utils import canonicalize_name, parse_wheel_filename

PACKAGE = "atrex_gateway_standalone"
TARGET_TAG = "cp312-cp312-manylinux_2_17_x86_64"
ROOTS = ("atrex-gateway-server", "atrex-gateway-client")
PLATFORMS = [f"manylinux_2_{minor}_x86_64" for minor in range(17, 4, -1)] + [
    "manylinux2014_x86_64",
    "manylinux2010_x86_64",
    "manylinux1_x86_64",
]
TARGET_TAGS = set(cpython_tags((3, 12), abis=["cp312"], platforms=PLATFORMS)) | set(
    compatible_tags((3, 12), interpreter="cp312", platforms=PLATFORMS)
)
TARGET_ENV = {
    **default_environment(),
    "implementation_name": "cpython",
    "implementation_version": "3.12.0",
    "os_name": "posix",
    "platform_machine": "x86_64",
    "platform_python_implementation": "CPython",
    "platform_release": "",
    "platform_system": "Linux",
    "platform_version": "",
    "python_full_version": "3.12.0",
    "python_version": "3.12",
    "sys_platform": "linux",
}


def safe_path(info: zipfile.ZipInfo) -> str:
    path = PurePosixPath(info.filename)
    if (
        path.is_absolute()
        or ".." in path.parts
        or "\\" in info.filename
        or stat.S_ISLNK(info.external_attr >> 16)
    ):
        raise ValueError(f"Unsafe wheel member: {info.filename}")
    if not path.parts:
        raise ValueError("Empty wheel member")
    if path.parts[0].endswith(".data"):
        if len(path.parts) < 3 or path.parts[1] not in {"purelib", "platlib"}:
            raise ValueError(f"Unsupported wheel installation scheme: {info.filename}")
        path = PurePosixPath(*path.parts[2:])
    if path.suffix == ".pth":
        raise ValueError(f"Vendored .pth files require explicit handling: {info.filename}")
    return str(path)


def read_wheels(wheelhouse: Path) -> dict[str, dict]:
    wheels: dict[str, dict] = {}
    for path in sorted(wheelhouse.glob("*.whl")):
        name, version, _, tags = parse_wheel_filename(path.name)
        name = canonicalize_name(name)
        if name in wheels:
            raise ValueError(f"Multiple wheels for {name}; use a clean wheelhouse")
        if not tags & TARGET_TAGS:
            raise ValueError(f"Wheel does not support {TARGET_TAG}: {path.name}")
        with zipfile.ZipFile(path) as archive:
            metadata_paths = [
                item.filename
                for item in archive.infolist()
                if item.filename.endswith(".dist-info/METADATA")
                and len(PurePosixPath(item.filename).parts) == 2
            ]
            if len(metadata_paths) != 1:
                raise ValueError(f"Expected one distribution METADATA in {path.name}")
            metadata = BytesParser().parsebytes(archive.read(metadata_paths[0]))
        if canonicalize_name(metadata["Name"]) != name or metadata["Version"] != str(version):
            raise ValueError(f"Filename/METADATA identity mismatch: {path.name}")
        requires_python = metadata.get("Requires-Python", "")
        if not SpecifierSet(requires_python).contains("3.12.0"):
            raise ValueError(f"{path.name} requires Python {requires_python}")
        wheels[name] = {
            "path": path,
            "version": version,
            "requires": [Requirement(value) for value in metadata.get_all("Requires-Dist", [])],
            "extras": set(metadata.get_all("Provides-Extra", [])),
        }
    return wheels


def dependency_closure(wheels: dict[str, dict]) -> set[str]:
    requested = {name: {""} for name in ROOTS}
    processed: dict[str, set[str]] = {}
    pending = list(ROOTS)
    while pending:
        name = pending.pop()
        if name not in wheels:
            raise ValueError(f"Missing required wheel: {name}")
        wheel = wheels[name]
        extras = requested[name]
        if extras - {""} - wheel["extras"]:
            raise ValueError(
                f"{name} has no requested extras: {sorted(extras - {''} - wheel['extras'])}"
            )
        if processed.get(name) == extras:
            continue
        processed[name] = set(extras)
        for requirement in wheel["requires"]:
            if requirement.marker and not any(
                requirement.marker.evaluate({**TARGET_ENV, "extra": extra}) for extra in extras
            ):
                continue
            dependency = canonicalize_name(requirement.name)
            if requirement.url:
                raise ValueError(f"Direct URL dependency needs explicit pinning: {requirement}")
            if dependency not in wheels:
                raise ValueError(f"Missing required wheel: {requirement} (from {name})")
            if not requirement.specifier.contains(wheels[dependency]["version"]):
                raise ValueError(f"Unsatisfied dependency: {name} needs {requirement}")
            previous = set(requested.get(dependency, set()))
            requested.setdefault(dependency, set()).update({"", *requirement.extras})
            if dependency not in processed or previous != requested[dependency]:
                pending.append(dependency)
    return set(requested)


def record_hash(data: bytes) -> str:
    return "sha256=" + base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()


def build(wheelhouse: Path, output_dir: Path) -> Path:
    wheels = read_wheels(wheelhouse)
    selected = dependency_closure(wheels)
    version = str(wheels[ROOTS[0]]["version"])
    if str(wheels[ROOTS[1]]["version"]) != version:
        raise ValueError("Server and Client versions must match")
    payloads: dict[str, bytes] = {}
    inputs = []
    for name in sorted(selected):
        path = wheels[name]["path"]
        inputs.append(
            {
                "name": name,
                "version": str(wheels[name]["version"]),
                "wheel": path.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
        with zipfile.ZipFile(path) as archive:
            for info in archive.infolist():
                member = safe_path(info)
                if info.is_dir():
                    continue
                target = f"{PACKAGE}/_vendor/{member}"
                data = archive.read(info)
                if target in payloads and payloads[target] != data:
                    raise ValueError(f"Conflicting vendored files: {member}")
                payloads[target] = data
    template_dir = Path(__file__).with_name("gateway_standalone")
    for path in sorted(template_dir.glob("*.py")):
        payloads[f"{PACKAGE}/{path.name}"] = path.read_bytes()
    for filename in ("README.md", "README.zh.md"):
        payloads[f"{PACKAGE}/{filename}"] = Path(__file__).with_name(filename).read_bytes()
    manifest = {
        "distribution": "atrex-gateway-standalone",
        "version": version,
        "target": {
            "implementation": "CPython",
            "python": "3.12",
            "platform": "Linux x86_64",
            "glibc_min": "2.17",
        },
        "scope": (
            "Server base dependencies and Client; "
            "excludes deployment extras and GPU/evaluator environment"
        ),
        "wheels": inputs,
    }
    payloads[f"{PACKAGE}/manifest.json"] = (json.dumps(manifest, indent=2) + "\n").encode()
    dist_info = f"{PACKAGE}-{version}.dist-info"
    payloads[f"{dist_info}/METADATA"] = (
        "Metadata-Version: 2.1\n"
        "Name: atrex-gateway-standalone\n"
        f"Version: {version}\n"
        "Summary: Offline private-vendor repack of Atrex Gateway Server and Client\n"
        "Requires-Python: >=3.12,<3.13\n\n"
        "Unofficial repack; upstream distributions and licenses are retained in _vendor.\n"
        "Requires a separately provisioned GPU and evaluator environment for GPU jobs.\n"
    ).encode()
    payloads[f"{dist_info}/WHEEL"] = (
        "Wheel-Version: 1.0\nGenerator: atrex-offline-repack\n"
        f"Root-Is-Purelib: false\nTag: {TARGET_TAG}\n"
    ).encode()
    payloads[f"{dist_info}/entry_points.txt"] = (
        b"[console_scripts]\n"
        b"atrex-gateway = atrex_gateway_standalone.cli:server_main\n"
        b"agate = atrex_gateway_standalone.cli:client_main\n"
        b"atrex-gateway-standalone-check = atrex_gateway_standalone.cli:check_main\n"
    )
    record_path = f"{dist_info}/RECORD"
    record = io.StringIO(newline="")
    writer = csv.writer(record)
    for member, data in sorted(payloads.items()):
        writer.writerow([member, record_hash(data), len(data)])
    writer.writerow([record_path, "", ""])
    payloads[record_path] = record.getvalue().encode()
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{PACKAGE}-{version}-{TARGET_TAG}.whl"
    with (
        output.open("xb") as destination,
        zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive,
    ):
        for member, data in sorted(payloads.items()):
            info = zipfile.ZipInfo(member, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (stat.S_IFREG | 0o644) << 16
            archive.writestr(info, data)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheelhouse", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = build(args.wheelhouse, args.output_dir)
    print(f"Built: {output}")
    print(f"Size: {output.stat().st_size:,} bytes")
    print(f"SHA256: {hashlib.sha256(output.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
