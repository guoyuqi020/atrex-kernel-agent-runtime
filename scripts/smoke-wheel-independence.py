"""Validate release archive boundaries and import the isolated Runtime wheel."""

from __future__ import annotations

import subprocess
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path

_FORBIDDEN_SOURCE_PARTS = frozenset(
    {
        ".git",
        "third_party",
        "workspaces",
        "local-wiki",
        "atrex-kernel-agent-core",
        "kernel-design-agents",
        "atrex-kernel-agent-evolver",
    }
)


def _assert_release_member(name: str) -> None:
    parts = frozenset(Path(name).parts)
    forbidden = sorted(parts & _FORBIDDEN_SOURCE_PARTS)
    if forbidden:
        raise RuntimeError(f"release archive contains forbidden source {forbidden}: {name}")


def main() -> None:
    """Build, inspect, install, and import the release archives."""
    repository = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix="atrex-wheel-smoke-") as temporary_value:
        temporary = Path(temporary_value)
        distribution = temporary / "dist"
        installation = temporary / "install"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "hatchling",
                "build",
                "-d",
                str(distribution),
            ],
            cwd=repository,
            check=True,
        )
        wheels = tuple(distribution.glob("*.whl"))
        if len(wheels) != 1:
            raise RuntimeError(f"expected exactly one wheel, found {len(wheels)}")
        source_archives = tuple(distribution.glob("*.tar.gz"))
        if len(source_archives) != 1:
            raise RuntimeError(f"expected exactly one sdist, found {len(source_archives)}")
        with zipfile.ZipFile(wheels[0]) as archive:
            for name in archive.namelist():
                _assert_release_member(name)
        with tarfile.open(source_archives[0], mode="r:gz") as archive:
            source_names = tuple(member.name for member in archive.getmembers())
        for name in source_names:
            _assert_release_member(name)
        if not any(name.endswith("/src/atrex_runtime/cli/__init__.py") for name in source_names):
            raise RuntimeError("sdist does not contain the Runtime package")
        if not any(name.endswith("/LICENSE") for name in source_names):
            raise RuntimeError("sdist does not contain the project license")
        rebuilt_distribution = temporary / "rebuilt-dist"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "wheel",
                "--disable-pip-version-check",
                "--no-index",
                "--no-deps",
                "--no-build-isolation",
                "--wheel-dir",
                str(rebuilt_distribution),
                str(source_archives[0]),
            ],
            cwd=temporary,
            check=True,
        )
        rebuilt_wheels = tuple(rebuilt_distribution.glob("*.whl"))
        if len(rebuilt_wheels) != 1:
            raise RuntimeError(
                f"expected one wheel rebuilt from sdist, found {len(rebuilt_wheels)}"
            )
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--no-index",
                "--no-cache-dir",
                "--no-deps",
                "--target",
                str(installation),
                str(wheels[0]),
            ],
            cwd=temporary,
            check=True,
        )
        smoke = """
import importlib.metadata
import pathlib
import sys

repository = pathlib.Path(sys.argv[1]).resolve()
installation = pathlib.Path(sys.argv[2]).resolve()
forbidden_paths = {repository, repository / 'src'}
sys.path[:] = [
    value for value in sys.path
    if pathlib.Path(value or '.').resolve() not in forbidden_paths
]
sys.path.insert(0, str(installation))

import atrex_runtime
from atrex_runtime.kernel_agents import GitOptimizerBaseLoader, KernelAgentRevisionBuilder

if GitOptimizerBaseLoader is None or KernelAgentRevisionBuilder is None:
    raise RuntimeError('full-repository Optimizer importer is unavailable')

for name, module in tuple(sys.modules.items()):
    if not name.startswith('atrex_runtime'):
        continue
    location = getattr(module, '__file__', None)
    if location is None or not pathlib.Path(location).resolve().is_relative_to(installation):
        raise RuntimeError(f'{name} resolved outside wheel installation: {location}')

distribution = importlib.metadata.distribution('atrex-kernel-agent-runtime')
if not pathlib.Path(distribution.locate_file('')).resolve().is_relative_to(installation):
    raise RuntimeError('package metadata resolved outside wheel installation')
requirements = distribution.requires or []
for forbidden in ('deepseek-harness', 'atrex-kernel-agent'):
    if any(forbidden in requirement.lower() for requirement in requirements):
        raise RuntimeError(f'wheel depends on reference checkout: {forbidden}')
print('isolated ATREX Runtime wheel and Git Optimizer importer loaded')
"""
        subprocess.run(
            [
                sys.executable,
                "-I",
                "-c",
                smoke,
                str(repository),
                str(installation),
            ],
            cwd=temporary,
            check=True,
        )


if __name__ == "__main__":
    main()
