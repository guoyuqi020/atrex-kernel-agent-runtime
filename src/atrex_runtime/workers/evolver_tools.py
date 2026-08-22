#!/usr/bin/env python3
"""Snapshot-scoped inspection and Candidate-control tools for an Evolver workspace."""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import os
import shutil
import stat
import sys
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any

MAX_JSON_BYTES = 16 * 1024 * 1024
MAX_SOURCE_BYTES = 1024 * 1024
MAX_RESULTS = 1000
REVISION_PREFIXES = {"agent": "agentrev_", "kernel": "kernelrev_"}


def _workspace() -> Path:
    tool = Path(__file__)
    if tool.is_symlink():
        raise ValueError("Runtime Tool cannot be a symbolic link")
    return tool.resolve().parents[1]


def _regular(path: Path, label: str, *, max_bytes: int | None = None) -> Path:
    metadata = path.lstat()
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"{label} must be a regular file")
    if max_bytes is not None and metadata.st_size > max_bytes:
        raise ValueError(f"{label} exceeds its byte limit")
    return path


def _directory(path: Path, label: str) -> Path:
    metadata = path.lstat()
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise ValueError(f"{label} must be a real directory")
    return path


def _json(path: Path, label: str) -> dict[str, Any]:
    value = json.loads(_regular(path, label, max_bytes=MAX_JSON_BYTES).read_bytes())
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{label} must contain a JSON object")
    return value


def _catalog(workspace: Path) -> dict[str, Any]:
    value = _json(workspace / "runtime-tools/catalog.json", "Runtime Tools catalog")
    if (
        value.get("schema_version") != 1
        or not isinstance(value.get("agents"), list)
        or not isinstance(value.get("kernels"), list)
    ):
        raise ValueError("Runtime Tools catalog is invalid")
    return value


def _evidence(workspace: Path) -> Path:
    root = _directory(workspace / "input/evidence", "Evidence root")
    manifest = _json(root / "manifest.json", "Evidence manifest")
    catalog = _catalog(workspace)
    if (
        manifest.get("role") != "evolver"
        or manifest.get("lineage_checkpoint") != catalog.get("evidence_checkpoint")
        or manifest.get("current_epoch") is not None
        or not isinstance(manifest.get("through_completed_epoch"), int)
        or manifest.get("visibility")
        != {
            "completed_epochs": "all_completed_branches",
            "current_attempts_before": None,
        }
    ):
        raise ValueError("Runtime Tools disagree with the frozen Evolver Evidence view")
    return root


def _bounded_limit(value: int) -> int:
    if value <= 0 or value > MAX_RESULTS:
        raise ValueError(f"limit must be between 1 and {MAX_RESULTS}")
    return value


def _version_maps(catalog: dict[str, Any]) -> tuple[dict[str, str], dict[str, str]]:
    agents = {
        item["revision_id"]: item["version"]
        for item in catalog["agents"]
        if isinstance(item, dict)
        and isinstance(item.get("revision_id"), str)
        and isinstance(item.get("version"), str)
    }
    kernels = {
        item["revision_id"]: item["version"]
        for item in catalog["kernels"]
        if isinstance(item, dict)
        and isinstance(item.get("revision_id"), str)
        and isinstance(item.get("version"), str)
    }
    return agents, kernels


def _epoch_root(evidence: Path, number: int) -> Path:
    if number <= 0:
        raise ValueError("epoch must be positive")
    return _directory(evidence / "epochs" / f"{number:08d}", "Epoch Evidence")


def _with_version(value: object, versions: dict[str, str]) -> dict[str, object] | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("revision identity must be a string")
    return {"revision_id": value, "version": versions.get(value)}


def _history(workspace: Path, limit: int) -> dict[str, object]:
    catalog = _catalog(workspace)
    evidence = _evidence(workspace)
    agent_versions, kernel_versions = _version_maps(catalog)
    epochs: list[dict[str, object]] = []
    for epoch_dir in sorted((evidence / "epochs").iterdir()):
        if not epoch_dir.is_dir() or not epoch_dir.name.isdigit():
            continue
        summary = _json(epoch_dir / "summary.json", "Epoch summary")
        branches = []
        for branch in summary.get("branches", []):
            if not isinstance(branch, dict):
                continue
            revision = branch.get("kernel_agent_revision_id")
            branches.append(
                {
                    **branch,
                    "agent_version": (
                        agent_versions.get(revision) if isinstance(revision, str) else None
                    ),
                }
            )
        epochs.append(
            {
                "epoch": summary.get("number"),
                "epoch_id": summary.get("epoch_id"),
                "active": _with_version(
                    summary.get("active_kernel_agent_revision_id"), agent_versions
                ),
                "branches": branches,
                "winner": _with_version(
                    summary.get("winner_kernel_agent_revision_id"), agent_versions
                ),
                "starting_kernel": _with_version(
                    summary.get("starting_kernel_revision_id"), kernel_versions
                ),
                "best_kernel": _with_version(
                    summary.get("best_kernel_revision_id"), kernel_versions
                ),
            }
        )
    return {
        "schema_version": 1,
        "evidence_checkpoint": catalog.get("evidence_checkpoint"),
        "epochs": epochs[-limit:],
        "truncated": len(epochs) > limit,
    }


def _attempt_values(
    workspace: Path,
    epoch: int,
    branch: str,
    trajectory: int | None,
    limit: int,
) -> list[dict[str, object]]:
    if branch != "active" and not (
        branch.startswith("challenger-")
        and len(branch) == len("challenger-0000")
        and branch.removeprefix("challenger-").isdigit()
    ):
        raise ValueError("branch must be active or challenger-NNNN")
    if trajectory is not None and trajectory <= 0:
        raise ValueError("trajectory must be positive")
    catalog = _catalog(workspace)
    _, kernel_versions = _version_maps(catalog)
    root = _epoch_root(_evidence(workspace), epoch) / "branches" / branch / "trajectories"
    _directory(root, "Branch trajectories")
    values: list[dict[str, object]] = []
    trajectory_dirs = (
        [root / f"{trajectory:08d}"] if trajectory is not None else sorted(root.iterdir())
    )
    for trajectory_dir in trajectory_dirs:
        if not trajectory_dir.is_dir() or not trajectory_dir.name.isdigit():
            continue
        attempts_root = _directory(trajectory_dir / "attempts", "Trajectory Attempts")
        for attempt_dir in sorted(attempts_root.iterdir()):
            if not attempt_dir.is_dir() or not attempt_dir.name.isdigit():
                continue
            value = _json(attempt_dir / "summary.json", "Attempt summary")
            output = value.get("output")
            enriched_output: object = output
            if isinstance(output, dict):
                revision = output.get("kernel_revision_id")
                enriched_output = {
                    **output,
                    "version": (
                        kernel_versions.get(revision) if isinstance(revision, str) else None
                    ),
                }
            input_revision = value.get("input_kernel_revision_id")
            values.append(
                {
                    **value,
                    "input_kernel_version": (
                        kernel_versions.get(input_revision)
                        if isinstance(input_revision, str)
                        else None
                    ),
                    "output": enriched_output,
                    "evidence_path": str(attempt_dir.relative_to(workspace)),
                }
            )
            if len(values) >= limit:
                return values
    return values


def _branches(workspace: Path, epoch: int) -> dict[str, object]:
    catalog = _catalog(workspace)
    agent_versions, kernel_versions = _version_maps(catalog)
    summary = _json(_epoch_root(_evidence(workspace), epoch) / "summary.json", "Epoch summary")
    starting_id = summary.get("starting_kernel_revision_id")
    kernel_by_id = {
        item.get("revision_id"): item
        for item in catalog["kernels"]
        if isinstance(item, dict) and isinstance(item.get("revision_id"), str)
    }
    results = []
    for raw in summary.get("branches", []):
        if not isinstance(raw, dict):
            continue
        branch_name = raw.get("branch")
        challenger = raw.get("challenger_ordinal")
        label = (
            "active"
            if branch_name == "active"
            else f"challenger-{challenger:04d}"
            if isinstance(challenger, int)
            else "invalid"
        )
        attempts = _attempt_values(workspace, epoch, label, None, MAX_RESULTS)
        candidate_ids = [
            output.get("kernel_revision_id")
            for attempt in attempts
            if isinstance((output := attempt.get("output")), dict) and output.get("correct") is True
        ]
        best_ids = [starting_id, *candidate_ids]
        best = min(
            (
                kernel_by_id[item]
                for item in best_ids
                if isinstance(item, str)
                and item in kernel_by_id
                and isinstance(kernel_by_id[item].get("latency_us"), (int, float))
            ),
            key=lambda item: float(item["latency_us"]),
            default=None,
        )
        revision = raw.get("kernel_agent_revision_id")
        results.append(
            {
                **raw,
                "label": label,
                "agent_version": (
                    agent_versions.get(revision) if isinstance(revision, str) else None
                ),
                "attempt_count": len(attempts),
                "valid_candidates": sum(
                    output.get("correct") is True
                    for item in attempts
                    if isinstance((output := item.get("output")), dict)
                ),
                "failed_candidates": sum(
                    not isinstance((output := item.get("output")), dict)
                    or output.get("correct") is not True
                    for item in attempts
                ),
                "retained_candidates": sum(
                    item.get("accepted_as_branch_best") is True for item in attempts
                ),
                "best_kernel": (
                    None
                    if best is None
                    else {
                        "revision_id": best.get("revision_id"),
                        "version": kernel_versions.get(str(best.get("revision_id"))),
                        "latency_us": best.get("latency_us"),
                        "sol_percent": best.get("sol_percent"),
                    }
                ),
            }
        )
    return {"schema_version": 1, "epoch": epoch, "branches": results}


def _kernels(workspace: Path, epoch: int | None, limit: int) -> dict[str, object]:
    catalog = _catalog(workspace)
    values = [
        item
        for item in catalog["kernels"]
        if isinstance(item, dict) and (epoch is None or item.get("epoch_number") == epoch)
    ]
    return {
        "schema_version": 1,
        "kernels": values[:limit],
        "truncated": len(values) > limit,
    }


def _safe_revision(value: str, kind: str) -> str:
    prefix = REVISION_PREFIXES[kind]
    suffix = value.removeprefix(prefix)
    if (
        not value.startswith(prefix)
        or len(suffix) != 32
        or any(c not in "0123456789abcdef" for c in suffix)
    ):
        raise ValueError(f"invalid {kind} revision ID")
    return value


def _safe_relative(value: str) -> PurePosixPath:
    relative = PurePosixPath(value)
    if relative.is_absolute() or relative.as_posix() == "." or ".." in relative.parts:
        raise ValueError("file must be a safe artifact-relative path")
    return relative


def _kernel_read(workspace: Path, revision: str, file: str | None) -> dict[str, object]:
    revision = _safe_revision(revision, "kernel")
    catalog = _catalog(workspace)
    record = next(
        (
            item
            for item in catalog["kernels"]
            if isinstance(item, dict) and item.get("revision_id") == revision
        ),
        None,
    )
    if record is None:
        raise ValueError("Kernel revision is not present in the frozen catalog")
    root = _directory(workspace / "runtime-tools/kernels" / revision, "Kernel artifact")
    files = sorted(path for path in root.rglob("*") if path.is_file() and not path.is_symlink())
    if file is None:
        return {
            "schema_version": 1,
            "kernel": record,
            "artifact_root": str(root.relative_to(workspace)),
            "files": [
                {
                    "path": path.relative_to(root).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
                for path in files
            ],
        }
    relative = _safe_relative(file)
    target = root.joinpath(*relative.parts)
    payload = _regular(target, "Kernel source", max_bytes=MAX_SOURCE_BYTES).read_bytes()
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("Kernel source is not UTF-8") from error
    return {
        "schema_version": 1,
        "kernel": record,
        "file": relative.as_posix(),
        "content": text,
    }


def _trial_catalog(workspace: Path) -> list[dict[str, object]]:
    values: list[dict[str, object]] = []
    epochs = _directory(_evidence(workspace) / "epochs", "Evidence Epochs")
    for epoch_root in sorted(epochs.iterdir()):
        if not epoch_root.is_dir() or not epoch_root.name.isdigit():
            continue
        index = epoch_root / "kernel-trials/index.json"
        if not index.is_file():
            continue
        document = _json(index, "Kernel Trial index")
        trials = document.get("kernel_trials")
        if document.get("schema_version") != 1 or not isinstance(trials, list):
            raise ValueError("Kernel Trial index is invalid")
        for raw in trials:
            if not isinstance(raw, dict) or not isinstance(raw.get("kernel_trial_id"), str):
                raise ValueError("Kernel Trial record is invalid")
            values.append({**raw, "epoch": int(epoch_root.name)})
    return values


def _trials(
    workspace: Path,
    epoch: int | None,
    decision: str | None,
    limit: int,
) -> dict[str, object]:
    if epoch is not None and epoch <= 0:
        raise ValueError("epoch must be positive")
    if decision is not None and decision not in {"observed", "continue", "revert", "pivot"}:
        raise ValueError("decision must be observed, continue, revert, or pivot")
    values = [
        item
        for item in _trial_catalog(workspace)
        if (epoch is None or item.get("epoch") == epoch)
        and (decision is None or item.get("disposition") == decision)
    ]
    return {
        "schema_version": 1,
        "kernel_trials": values[:limit],
        "truncated": len(values) > limit,
    }


def _trial_read(workspace: Path, trial_id: str, file: str | None) -> dict[str, object]:
    if (
        not trial_id.startswith("gtrial_")
        or len(trial_id.removeprefix("gtrial_")) != 32
        or any(
            character not in "0123456789abcdef" for character in trial_id.removeprefix("gtrial_")
        )
    ):
        raise ValueError("invalid Kernel Trial ID")
    record = next(
        (item for item in _trial_catalog(workspace) if item.get("kernel_trial_id") == trial_id),
        None,
    )
    if record is None:
        raise ValueError("Kernel Trial is not present in frozen Evidence")
    epoch = record.get("epoch")
    if not isinstance(epoch, int):
        raise ValueError("Kernel Trial epoch is invalid")
    root = _directory(
        workspace
        / "input/evidence/epochs"
        / f"{epoch:08d}"
        / "kernel-trials"
        / trial_id
        / "source",
        "Kernel Trial source",
    )
    files = sorted(path for path in root.rglob("*") if path.is_file() and not path.is_symlink())
    if file is None:
        return {
            "schema_version": 1,
            "kernel_trial": record,
            "files": [
                {
                    "path": path.relative_to(root).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
                for path in files
            ],
        }
    relative = _safe_relative(file)
    payload = _regular(
        root.joinpath(*relative.parts),
        "Kernel Trial source",
        max_bytes=MAX_SOURCE_BYTES,
    ).read_bytes()
    try:
        content = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("Kernel Trial source is not UTF-8") from error
    return {
        "schema_version": 1,
        "kernel_trial": record,
        "file": relative.as_posix(),
        "content": content,
    }


def _agents(workspace: Path, limit: int) -> dict[str, object]:
    catalog = _catalog(workspace)
    values = [item for item in catalog["agents"] if isinstance(item, dict)]
    return {
        "schema_version": 1,
        "agents": values[:limit],
        "truncated": len(values) > limit,
    }


def _repository(workspace: Path, revision: str) -> Path:
    revision = _safe_revision(revision, "agent")
    catalog = _catalog(workspace)
    if not any(
        isinstance(item, dict) and item.get("revision_id") == revision for item in catalog["agents"]
    ):
        raise ValueError("Agent revision is not present in the frozen catalog")
    return _directory(workspace / "input/agents" / revision, "Agent repository")


def _historical_repository(workspace: Path, revision: str) -> Path:
    revision = _safe_revision(revision, "agent")
    manifest = _json(workspace / "evolution-input.json", "Evolution input manifest")
    visible = manifest.get("visible_agents")
    if manifest.get("schema_version") != 4 or not isinstance(visible, list):
        raise ValueError("Evolution input manifest is invalid")
    record = next(
        (
            item
            for item in visible
            if isinstance(item, dict) and item.get("revision_id") == revision
        ),
        None,
    )
    if record is None:
        raise ValueError("Agent revision is not present in the frozen Evolution input")
    if record.get("relationship") != "lineage_history":
        raise ValueError("Candidate base must be completed Lineage history")
    expected_path = f"input/agents/{revision}"
    if record.get("path") != expected_path:
        raise ValueError("Historical Agent repository path is invalid")
    return _directory(workspace / expected_path, "Historical Agent repository")


def _copy_writable_repository(source: Path, destination: Path) -> None:
    destination.mkdir(mode=0o700)
    for item in sorted(source.rglob("*")):
        metadata = item.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError("Historical Agent repository cannot contain symbolic links")
        relative = item.relative_to(source)
        target = destination / relative
        if stat.S_ISDIR(metadata.st_mode):
            target.mkdir(mode=0o700)
        elif stat.S_ISREG(metadata.st_mode):
            target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            shutil.copyfile(item, target)
            os.chmod(target, 0o600 | (metadata.st_mode & 0o111))
        else:
            raise ValueError("Historical Agent repository contains a special file")


def _write_candidate_base(workspace: Path, revision: str) -> None:
    scratch = _directory(workspace / "scratch", "Evolution scratch")
    destination = scratch / "candidate-base.json"
    temporary = scratch / ".candidate-base.json.tmp"
    payload = json.dumps(
        {
            "schema_version": 1,
            "base_revision_id": revision,
            "selection": "candidate_reset",
        },
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    with temporary.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.chmod(temporary, 0o600)
    os.replace(temporary, destination)


def _candidate_reset(workspace: Path, revision: str) -> dict[str, object]:
    revision = _safe_revision(revision, "agent")
    source = _historical_repository(workspace, revision)
    candidate = _directory(workspace / "candidate", "Candidate repository")
    scratch = _directory(workspace / "scratch", "Evolution scratch")
    transaction = Path(tempfile.mkdtemp(prefix="candidate-reset-", dir=scratch))
    staged = transaction / "repository"
    previous = transaction / "previous"
    try:
        _copy_writable_repository(source, staged)
        candidate.rename(previous)
        try:
            staged.rename(candidate)
        except BaseException:
            previous.rename(candidate)
            raise
        shutil.rmtree(previous)
        _write_candidate_base(workspace, revision)
    finally:
        shutil.rmtree(transaction, ignore_errors=True)
    return {
        "schema_version": 1,
        "candidate_base_revision_id": revision,
        "candidate_repository": "candidate",
    }


def _repo_files(root: Path) -> dict[str, Path]:
    values: dict[str, Path] = {}
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError("Agent repository cannot contain symbolic links")
        if path.is_file():
            values[path.relative_to(root).as_posix()] = path
    return values


def _agent_diff(
    workspace: Path,
    base: str,
    candidate: str,
    max_diff_chars: int,
) -> dict[str, object]:
    if max_diff_chars <= 0 or max_diff_chars > 100_000:
        raise ValueError("max-diff-chars must be between 1 and 100000")
    base_files = _repo_files(_repository(workspace, base))
    candidate_files = _repo_files(_repository(workspace, candidate))
    changes = []
    remaining = max_diff_chars
    for relative in sorted(base_files.keys() | candidate_files.keys()):
        before_path = base_files.get(relative)
        after_path = candidate_files.get(relative)
        before = None if before_path is None else before_path.read_bytes()
        after = None if after_path is None else after_path.read_bytes()
        if before == after:
            continue
        status_value = "added" if before is None else "removed" if after is None else "modified"
        rendered: str | None = None
        try:
            before_lines = (
                [] if before is None else before.decode("utf-8").splitlines(keepends=True)
            )
            after_lines = [] if after is None else after.decode("utf-8").splitlines(keepends=True)
            rendered = "".join(
                difflib.unified_diff(
                    before_lines,
                    after_lines,
                    fromfile=f"{base}/{relative}",
                    tofile=f"{candidate}/{relative}",
                )
            )
            rendered = rendered[:remaining]
            remaining -= len(rendered)
        except UnicodeDecodeError:
            rendered = None
        changes.append({"path": relative, "status": status_value, "diff": rendered})
    return {
        "schema_version": 1,
        "base_revision_id": base,
        "candidate_revision_id": candidate,
        "changes": changes,
        "diff_truncated": remaining == 0,
    }


def _trace_paths(workspace: Path, epoch: int | None, limit: int) -> dict[str, object]:
    evidence = _evidence(workspace)
    roots = (
        [_epoch_root(evidence, epoch)]
        if epoch is not None
        else [path for path in sorted((evidence / "epochs").iterdir()) if path.is_dir()]
    )
    paths: list[str] = []
    for root in roots:
        for path in sorted(root.rglob("*")):
            if not path.is_dir() or path.is_symlink():
                continue
            if path.name.startswith("run-") or (
                path.name == "trace" and path.parent.name.startswith("challenger-")
            ):
                paths.append(str(path.relative_to(workspace)))
                if len(paths) >= limit:
                    return {"schema_version": 1, "trace_paths": paths, "truncated": True}
    return {"schema_version": 1, "trace_paths": paths, "truncated": False}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    history = commands.add_parser("history", help="summarize completed Epoch winners")
    history.add_argument("--limit", type=int, default=100)
    branches = commands.add_parser("branches", help="summarize every Branch in one Epoch")
    branches.add_argument("--epoch", type=int, required=True)
    attempts = commands.add_parser("attempts", help="list Attempts in one completed Branch")
    attempts.add_argument("--epoch", type=int, required=True)
    attempts.add_argument("--branch", required=True)
    attempts.add_argument("--trajectory", type=int)
    attempts.add_argument("--limit", type=int, default=200)
    kernels = commands.add_parser("kernels", help="list versioned historical Kernels")
    kernels.add_argument("--epoch", type=int)
    kernels.add_argument("--limit", type=int, default=200)
    kernel_read = commands.add_parser("kernel-read", help="inspect one exact Kernel artifact")
    kernel_read.add_argument("--revision", required=True)
    kernel_read.add_argument("--file")
    trials = commands.add_parser(
        "kernel-trials",
        help="list exact experimental Kernels, including reverted candidates",
    )
    trials.add_argument("--epoch", type=int)
    trials.add_argument("--decision")
    trials.add_argument("--limit", type=int, default=200)
    trial_read = commands.add_parser(
        "kernel-trial-read",
        help="inspect one exact unversioned Kernel Trial artifact",
    )
    trial_read.add_argument("--trial", required=True)
    trial_read.add_argument("--file")
    agents = commands.add_parser("agents", help="list versioned Agent designs")
    agents.add_argument("--limit", type=int, default=200)
    agent_diff = commands.add_parser("agent-diff", help="diff two visible Agent repositories")
    agent_diff.add_argument("--base", required=True)
    agent_diff.add_argument("--candidate", required=True)
    agent_diff.add_argument("--max-diff-chars", type=int, default=20_000)
    candidate_reset = commands.add_parser(
        "candidate-reset",
        help="atomically reset Candidate to one completed historical Agent",
    )
    candidate_reset.add_argument("--base", required=True)
    traces = commands.add_parser("trace-paths", help="locate original Session Trace trees")
    traces.add_argument("--epoch", type=int)
    traces.add_argument("--limit", type=int, default=200)
    return parser


def main() -> int:
    try:
        arguments = _parser().parse_args()
        workspace = _workspace()
        if arguments.command == "history":
            result = _history(workspace, _bounded_limit(arguments.limit))
        elif arguments.command == "branches":
            result = _branches(workspace, arguments.epoch)
        elif arguments.command == "attempts":
            result = {
                "schema_version": 1,
                "epoch": arguments.epoch,
                "branch": arguments.branch,
                "attempts": _attempt_values(
                    workspace,
                    arguments.epoch,
                    arguments.branch,
                    arguments.trajectory,
                    _bounded_limit(arguments.limit),
                ),
            }
        elif arguments.command == "kernels":
            result = _kernels(workspace, arguments.epoch, _bounded_limit(arguments.limit))
        elif arguments.command == "kernel-read":
            result = _kernel_read(workspace, arguments.revision, arguments.file)
        elif arguments.command == "kernel-trials":
            result = _trials(
                workspace,
                arguments.epoch,
                arguments.decision,
                _bounded_limit(arguments.limit),
            )
        elif arguments.command == "kernel-trial-read":
            result = _trial_read(workspace, arguments.trial, arguments.file)
        elif arguments.command == "agents":
            result = _agents(workspace, _bounded_limit(arguments.limit))
        elif arguments.command == "agent-diff":
            result = _agent_diff(
                workspace,
                arguments.base,
                arguments.candidate,
                arguments.max_diff_chars,
            )
        elif arguments.command == "candidate-reset":
            result = _candidate_reset(workspace, arguments.base)
        elif arguments.command == "trace-paths":
            result = _trace_paths(
                workspace,
                arguments.epoch,
                _bounded_limit(arguments.limit),
            )
        else:
            raise AssertionError("unreachable command")
        print(json.dumps(result, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2))
        return 0
    except (FileNotFoundError, json.JSONDecodeError, OSError, ValueError) as error:
        print(
            json.dumps(
                {"error": type(error).__name__, "message": str(error)},
                ensure_ascii=False,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
