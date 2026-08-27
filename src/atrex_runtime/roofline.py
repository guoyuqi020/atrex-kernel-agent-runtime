"""Trusted, commit-pinned Atrex Bench Roofline construction."""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import tempfile
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Protocol, cast

from .artifacts.local import JsonValue
from .gateway.contract import AgateEvaluationContractV1
from .git_import import SafeGitImporter

_FULL_COMMIT = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_SAFE_OPERATOR = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_GENERATOR_ROOT = PurePosixPath("skills/benchmark-converter")
_GENERATOR_PATH = PurePosixPath("skills/benchmark-converter/scripts/generate_roofline.py")


class RooflineBuilder(Protocol):
    """Build one evaluator-compatible Roofline from trusted Campaign inputs."""

    def build(
        self,
        *,
        operator: str,
        hardware_target: str,
        contract: AgateEvaluationContractV1,
    ) -> dict[str, JsonValue]: ...


class AtrexBenchRooflineBuilder:
    """Execute the canonical Atrex Bench converter from one exact Git commit."""

    def __init__(
        self,
        *,
        repository: str,
        commit: str,
        git_executable: str | Path,
        python_executable: str | Path,
        fetch_timeout_seconds: float,
        execution_timeout_seconds: float,
        max_archive_bytes: int,
        max_output_bytes: int,
        sku_by_hardware_target: Mapping[str, str] | None = None,
    ) -> None:
        if not repository.strip():
            raise ValueError("Atrex Bench Roofline repository cannot be empty")
        if _FULL_COMMIT.fullmatch(commit) is None:
            raise ValueError("Atrex Bench Roofline revision must be a full lowercase commit SHA")
        python = Path(python_executable)
        if not python.is_absolute():
            raise ValueError("Roofline Python executable must be absolute")
        if execution_timeout_seconds <= 0 or max_output_bytes <= 0:
            raise ValueError("Roofline execution limits must be positive")
        self._repository = repository
        self._commit = commit
        self._python_executable = python
        self._execution_timeout_seconds = execution_timeout_seconds
        self._max_output_bytes = max_output_bytes
        self._sku_by_hardware_target = dict(sku_by_hardware_target or {})
        self._importer = SafeGitImporter(
            git_executable,
            timeout_seconds=fetch_timeout_seconds,
            max_archive_bytes=max_archive_bytes,
            label="Atrex Bench Roofline",
        )

    def build(
        self,
        *,
        operator: str,
        hardware_target: str,
        contract: AgateEvaluationContractV1,
    ) -> dict[str, JsonValue]:
        """Generate and validate one complete per-Shape Roofline document."""
        if contract.roofline is not None:
            raise ValueError("Roofline Builder cannot replace an explicit roofline")
        if contract.metadata is None:
            raise ValueError("Atrex Bench Roofline generation requires evaluation metadata")
        if _SAFE_OPERATOR.fullmatch(operator) is None:
            raise ValueError("Roofline operator must be a safe single path component")

        with tempfile.TemporaryDirectory(prefix="atrex-roofline-") as temporary:
            root = Path(temporary)
            export = self._export(root)
            generator = export.joinpath(*_GENERATOR_PATH.parts)
            if generator.is_symlink() or not generator.is_file():
                raise ValueError(
                    f"Atrex Bench commit does not contain {_GENERATOR_PATH.as_posix()}"
                )
            data_root = root / "input"
            operator_root = data_root / operator
            operator_root.mkdir(parents=True, mode=0o700)
            self._write_json(operator_root / "shapes.json", contract.shapes)
            self._write_json(operator_root / "metadata.json", contract.metadata)
            command = [
                str(self._python_executable),
                str(generator),
                "--data-dir",
                str(data_root),
                "--op",
                operator,
            ]
            sku = self._sku_by_hardware_target.get(hardware_target)
            if sku is not None:
                command.extend(("--sku", sku))
            try:
                process = subprocess.run(
                    command,
                    check=False,
                    capture_output=True,
                    timeout=self._execution_timeout_seconds,
                    env={"PYTHONIOENCODING": "utf-8"},
                )
            except (OSError, subprocess.TimeoutExpired) as error:
                raise RuntimeError("Atrex Bench Roofline Builder failed to execute") from error
            diagnostic = (process.stdout + process.stderr)[: self._max_output_bytes].decode(
                "utf-8", errors="replace"
            )
            if process.returncode != 0:
                raise RuntimeError(
                    f"Atrex Bench Roofline Builder failed with {process.returncode}: "
                    f"{diagnostic.strip()}"
                )
            output = operator_root / "roofline.json"
            if output.is_symlink() or not output.is_file():
                raise ValueError("Atrex Bench Roofline Builder produced no roofline.json")
            if output.stat().st_size > self._max_output_bytes:
                raise ValueError("generated roofline.json exceeds byte limit")
            try:
                value: object = json.loads(output.read_bytes())
            except json.JSONDecodeError as error:
                raise ValueError("generated roofline.json is invalid JSON") from error
            return validate_roofline(value, expected_shape_ids=set(contract.shapes))

    def _export(self, root: Path) -> Path:
        repository = root / "repository"
        export = root / "export"
        archive = root / "source.tar"
        self._importer.run(("init", "--bare", str(repository)))
        self._importer.run(("-C", str(repository), "remote", "add", "origin", self._repository))
        self._importer.fetch_commit(repository, "origin", self._commit)
        resolved = self._importer.object_id(
            self._importer.run(("-C", str(repository), "rev-parse", "FETCH_HEAD^{commit}"))
        )
        if resolved != self._commit:
            raise ValueError("Git fetch resolved a different Atrex Bench commit")
        self._validate_tree(
            self._importer.run(("-C", str(repository), "ls-tree", "-rz", "-r", "FETCH_HEAD"))
        )
        self._importer.archive(
            repository,
            "FETCH_HEAD",
            archive,
            paths=(_GENERATOR_ROOT.as_posix(),),
        )
        export.mkdir(mode=0o700)
        self._importer.extract(archive, export)
        return export

    @staticmethod
    def _validate_tree(payload: bytes) -> None:
        for record in payload.split(b"\0"):
            if not record:
                continue
            try:
                metadata, raw_path = record.split(b"\t", 1)
                mode, kind, _object_id = metadata.split(b" ", 2)
                path = raw_path.decode("utf-8")
            except (ValueError, UnicodeDecodeError) as error:
                raise ValueError("Atrex Bench Git tree listing is malformed") from error
            relative = PurePosixPath(path)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Atrex Bench Git tree contains an unsafe path")
            if mode in {b"120000", b"160000"} or kind != b"blob":
                raise ValueError(f"Atrex Bench Git tree contains a link or submodule: {path}")

    @staticmethod
    def _write_json(path: Path, value: object) -> None:
        path.write_text(
            json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True),
            encoding="utf-8",
        )
        os.chmod(path, 0o600)


def validate_roofline(
    value: object,
    *,
    expected_shape_ids: set[str],
) -> dict[str, JsonValue]:
    """Validate the stable subset consumed by Atrex Bench evaluation."""
    if not isinstance(value, dict) or set(value) != {"shapes"}:
        raise ValueError("roofline.json must contain exactly one shapes object")
    raw_shapes = value.get("shapes")
    if not isinstance(raw_shapes, dict):
        raise ValueError("roofline.json shapes must be an object")
    if set(raw_shapes) != expected_shape_ids:
        raise ValueError("roofline.json Shape IDs must exactly match the Evaluation Contract")
    for shape_id, raw_shape in raw_shapes.items():
        if not isinstance(raw_shape, dict):
            raise ValueError(f"roofline Shape {shape_id} must be an object")
        flops = raw_shape.get("semantic_W_flops")
        if not isinstance(flops, dict) or not flops:
            raise ValueError(f"roofline Shape {shape_id} requires semantic_W_flops")
        for dtype, amount in flops.items():
            if not isinstance(dtype, str) or not _nonnegative_number(amount):
                raise ValueError(f"roofline Shape {shape_id} has invalid FLOPs")
        for field in ("semantic_Q_read_bytes", "semantic_Q_write_bytes"):
            if not _nonnegative_number(raw_shape.get(field)):
                raise ValueError(f"roofline Shape {shape_id} has invalid {field}")
        sol = raw_shape.get("SOL_time_ms")
        if not isinstance(sol, dict) or not sol:
            raise ValueError(f"roofline Shape {shape_id} requires SOL_time_ms")
        for hardware, duration in sol.items():
            if not isinstance(hardware, str) or not hardware or not _nonnegative_number(duration):
                raise ValueError(f"roofline Shape {shape_id} has invalid SOL_time_ms")
    return cast(dict[str, JsonValue], value)


_HARDWARE_SUFFIX = re.compile(r"\s*\([^()]*\)\s*$")


def strip_roofline_hardware_suffix(value: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    """Key SOL_time_ms on the bare device name the evaluation job reports."""
    raw_shapes = value.get("shapes")
    if not isinstance(raw_shapes, dict):
        return dict(value)
    shapes: dict[str, JsonValue] = {}
    for shape_id, raw_shape in raw_shapes.items():
        if not isinstance(raw_shape, dict):
            shapes[shape_id] = raw_shape
            continue
        sol = raw_shape.get("SOL_time_ms")
        if not isinstance(sol, dict):
            shapes[shape_id] = raw_shape
            continue
        rewritten: dict[str, JsonValue] = {}
        for hardware, duration in sol.items():
            bare = _HARDWARE_SUFFIX.sub("", hardware) if isinstance(hardware, str) else hardware
            if bare in rewritten:
                raise ValueError(f"roofline Shape {shape_id} maps two hardware keys onto {bare!r}")
            rewritten[bare] = duration
        shapes[shape_id] = {**raw_shape, "SOL_time_ms": rewritten}
    return {**value, "shapes": shapes}


def _nonnegative_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value >= 0
    )
