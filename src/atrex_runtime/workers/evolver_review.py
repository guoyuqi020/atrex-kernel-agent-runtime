"""Conservative, mechanically derived review inputs for an Evolver session.

The projections in this module deliberately stop before semantic attribution.  They expose
observable action/result chains, exact cross-Trajectory overlaps, and repeated command construction;
the Evolver still decides whether an observation is useful, harmful, intentional, or irrelevant.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from ..artifacts.local import JsonValue
from ..serialization import write_canonical_json

_RUNTIME_TOOL = re.compile(r"runtime_tools\.py\s+([a-z][a-z-]+)(?:\s|$)")
_IDENTITY = re.compile(
    r"\b(?:attempt|direction|experiment|gtrial|kernelrev|agentrev|epoch|lineage)_"
    r"[0-9a-f]{16,64}\b"
)
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
_SCRATCH_PATH = re.compile(r"\bscratch/[A-Za-z0-9_.-]+")
_SPACE = re.compile(r"\s+")
_WRITE_COMMAND = re.compile(r"(?:cat\s*>|tee\s+|apply_patch|sed\s+-i)")
_MAX_FAILURE_OBSERVATIONS = 256
_MAX_REPEATED_CONSTRUCTIONS = 128


@dataclass(frozen=True, slots=True)
class _ToolCall:
    identifier: str
    name: str
    input_text: str
    result_text: str
    failed: bool


@dataclass(frozen=True, slots=True)
class _Session:
    version: str
    relative_path: str
    calls: tuple[_ToolCall, ...]


def materialize_evolver_review(
    destination: Path,
    *,
    evolution_reports_root: Path | None,
    agent_versions: dict[str, str],
    pool_versions: frozenset[str],
) -> None:
    """Write three bounded review projections under one Evolver Evidence tree."""
    review = destination / "review"
    review.mkdir(mode=0o700)
    sessions = _visible_sessions(destination)
    reports = _visible_attempt_reports(destination)
    write_canonical_json(
        review / "evolution-change-audit.json",
        _evolution_change_audit(
            sessions,
            reports,
            evolution_reports_root=evolution_reports_root,
            evaluated_versions=pool_versions,
        ),
    )
    write_canonical_json(
        review / "trajectory-comparison.json",
        _trajectory_comparison(destination, agent_versions),
    )
    write_canonical_json(
        review / "workflow-friction.json",
        _workflow_friction(sessions),
    )


def _visible_sessions(root: Path) -> list[_Session]:
    sessions: list[_Session] = []
    for path in sorted(root.glob("agent-v*/sessions/**/*.conversation.jsonl")):
        if path.is_symlink() or not path.is_file():
            raise ValueError("Evolver review Session must be a regular file")
        version = path.relative_to(root).parts[0]
        sessions.append(
            _Session(
                version=version,
                relative_path=path.relative_to(root).as_posix(),
                calls=_tool_calls(path),
            )
        )
    return sessions


def _visible_attempt_reports(root: Path) -> dict[str, list[tuple[str, str]]]:
    reports: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for path in sorted(root.glob("agent-v*/reports/**/*.report.json")):
        if path.is_symlink() or not path.is_file():
            raise ValueError("Evolver review Attempt Report must be a regular file")
        version = path.relative_to(root).parts[0]
        reports[version].append(
            (path.relative_to(root).as_posix(), path.read_text(encoding="utf-8"))
        )
    return reports


def _tool_calls(path: Path) -> tuple[_ToolCall, ...]:
    uses: dict[str, tuple[str, str]] = {}
    results: dict[str, tuple[str, bool]] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        for node in _walk_json(value):
            if node.get("type") == "tool_use":
                identifier = node.get("id")
                name = node.get("name")
                if isinstance(identifier, str) and isinstance(name, str):
                    uses.setdefault(identifier, (name, _json_text(node.get("input"))))
            if node.get("type") == "tool_result":
                identifier = node.get("tool_use_id")
                if isinstance(identifier, str):
                    text = _json_text(node.get("content"))
                    failed = node.get("is_error") is True or _failure_text(text)
                    previous = results.get(identifier)
                    if previous is None or (not previous[1] and failed):
                        results[identifier] = (text, failed)
        event = value.get("event") if isinstance(value, dict) else None
        if isinstance(event, dict):
            raw = event.get("toolUseResult") or event.get("tool_use_result")
            if isinstance(raw, dict):
                identifier = _result_identifier(event)
                if identifier is not None:
                    text = _json_text(raw)
                    failed = _result_failed(raw, text)
                    previous = results.get(identifier)
                    if previous is None or (not previous[1] and failed):
                        results[identifier] = (text, failed)
    return tuple(
        _ToolCall(
            identifier=identifier,
            name=name,
            input_text=input_text,
            result_text=results.get(identifier, ("", False))[0],
            failed=results.get(identifier, ("", False))[1],
        )
        for identifier, (name, input_text) in uses.items()
    )


def _walk_json(value: object) -> Iterator[dict[str, Any]]:
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk_json(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_json(child)


def _result_identifier(event: dict[str, Any]) -> str | None:
    message = event.get("message")
    if isinstance(message, dict):
        for content in message.get("content", []):
            if isinstance(content, dict) and isinstance(content.get("tool_use_id"), str):
                return cast(str, content["tool_use_id"])
    value = event.get("sourceToolAssistantUUID")
    return value if isinstance(value, str) else None


def _json_text(value: object) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError):
        return repr(value)


def _failure_text(text: str) -> bool:
    lowered = text.lower()
    return any(
        marker in lowered
        for marker in (
            '"status": "error"',
            '"ok": false',
            "traceback (most recent call last)",
            "permission denied",
            "validation failed",
        )
    )


def _result_failed(value: dict[str, Any], text: str) -> bool:
    exit_code = value.get("exitCode", value.get("exit_code"))
    return (
        value.get("is_error") is True
        or value.get("interrupted") is True
        or (isinstance(exit_code, int) and exit_code != 0)
        or _failure_text(text)
    )


def _evolution_change_audit(
    sessions: list[_Session],
    reports: dict[str, list[tuple[str, str]]],
    *,
    evolution_reports_root: Path | None,
    evaluated_versions: frozenset[str],
) -> dict[str, JsonValue]:
    audited: list[JsonValue] = []
    if evolution_reports_root is None or not evolution_reports_root.is_dir():
        return {
            "status": "no_evolution_reports",
            "evaluated_changes": [],
            "interpretation": (
                "No prior Evolution report is visible; absence of observations is not evidence "
                "that a change was unused."
            ),
        }
    sessions_by_version: dict[str, list[_Session]] = defaultdict(list)
    for session in sessions:
        sessions_by_version[session.version].append(session)
    for path in sorted(evolution_reports_root.glob("evo-*.json"), key=_evolution_report_order):
        value = _json_object(path, "Evolution report")
        generated = value.get("generated_agent")
        report = value.get("report")
        if not isinstance(generated, dict) or not isinstance(report, dict):
            raise ValueError("Evolution report has no generated Agent or report")
        generated_path = generated.get("path")
        if not isinstance(generated_path, str):
            raise ValueError("Evolution report generated Agent path is invalid")
        version = Path(generated_path).name
        if version not in evaluated_versions:
            continue
        changed = report.get("changed_paths", [])
        if not isinstance(changed, list) or not all(isinstance(item, str) for item in changed):
            raise ValueError("Evolution report changed paths are invalid")
        version_sessions = sessions_by_version.get(version, [])
        observations = [
            _changed_path_observation(
                cast(str, changed_path),
                version_sessions,
                reports.get(version, []),
            )
            for changed_path in changed
        ]
        audited.append(
            cast(
                JsonValue,
                {
                    "evolution_number": value.get("evolution_number"),
                    "generated_agent": generated_path,
                    "hypothesis": report.get("hypothesis"),
                    "expected_effect": report.get("expected_effect"),
                    "sessions_available": len(version_sessions),
                    "observations": observations,
                },
            )
        )
    return {
        "status": "completed" if audited else "no_evaluated_evolution",
        "evaluated_changes": audited,
        "interpretation": (
            "Statuses describe observed discovery, invocation, failure, and report citation only. "
            "They do not prove causal benefit or harm."
        ),
    }


def _evolution_report_order(path: Path) -> tuple[int, str]:
    match = re.fullmatch(r"evo-([0-9]+)\.json", path.name)
    return (int(match.group(1)) if match else 0, path.name)


def _changed_path_observation(
    changed_path: str,
    sessions: list[_Session],
    reports: list[tuple[str, str]],
) -> dict[str, JsonValue]:
    kind = changed_path.split("/", 1)[0] if "/" in changed_path else "source"
    markers = _path_markers(changed_path)
    report_markers = _report_markers(changed_path)
    discovered: set[str] = set()
    invoked: set[str] = set()
    successful: set[str] = set()
    failed: set[str] = set()
    for session in sessions:
        for call in session.calls:
            if not any(marker in call.input_text for marker in markers):
                continue
            discovered.add(session.relative_path)
            if _is_invocation(changed_path, call):
                invoked.add(session.relative_path)
                (failed if call.failed else successful).add(session.relative_path)
    cited = sorted(
        relative for relative, text in reports if any(marker in text for marker in report_markers)
    )
    if cited:
        status = "report_cited"
    elif successful:
        status = "invoked_successfully"
    elif failed:
        status = "invoked_but_failed"
    elif discovered:
        status = "discovered_only"
    elif sessions:
        status = "not_observed"
    else:
        status = "not_evaluated"
    return {
        "path": changed_path,
        "kind": kind,
        "status": status,
        "discovered_in_sessions": cast(JsonValue, sorted(discovered)),
        "invoked_in_sessions": cast(JsonValue, sorted(invoked)),
        "successful_in_sessions": cast(JsonValue, sorted(successful)),
        "failed_in_sessions": cast(JsonValue, sorted(failed)),
        "cited_in_attempt_reports": cast(JsonValue, cited),
    }


def _path_markers(path: str) -> tuple[str, ...]:
    parts = Path(path).parts
    markers = {path, Path(path).name}
    if len(parts) >= 2 and parts[0] in {"skills", "tools"}:
        markers.add(parts[1])
        markers.add("/".join(parts[:2]))
    return tuple(sorted((item for item in markers if item), key=len, reverse=True))


def _report_markers(path: str) -> tuple[str, ...]:
    """Use only explicit resource identities when interpreting an Agent-authored report."""
    parts = Path(path).parts
    markers = {path}
    if len(parts) >= 2 and parts[0] == "tools":
        markers.add(Path(path).name)
    elif len(parts) >= 2 and parts[0] == "skills":
        markers.add("/".join(parts[:2]))
    return tuple(sorted(markers, key=len, reverse=True))


def _is_invocation(changed_path: str, call: _ToolCall) -> bool:
    parts = Path(changed_path).parts
    kind = parts[0] if parts else "source"
    if kind == "skills":
        skill = parts[1] if len(parts) > 1 else Path(changed_path).stem
        return call.name.lower() == "skill" and skill in call.input_text
    if kind != "tools":
        return False
    if "py_compile" in call.input_text:
        return False
    name = re.escape(Path(changed_path).name)
    return bool(
        re.search(rf"(?:python3?|bash|sh|zsh)\s+(?:\./)?(?:tools/)?{name}(?:\s|$)", call.input_text)
        or re.search(rf"(?:^|[;&|]\s*)(?:\./)?tools/{name}(?:\s|$)", call.input_text)
    )


def _trajectory_comparison(
    destination: Path,
    agent_versions: dict[str, str],
) -> dict[str, JsonValue]:
    facts = _json_object(destination / "latest-epoch-facts.json", "latest Epoch facts")
    raw_attempts = facts.get("attempts", [])
    if not isinstance(raw_attempts, list):
        raise ValueError("latest Epoch Attempt facts are invalid")
    version_by_revision = {revision: version for version, revision in agent_versions.items()}
    grouped: dict[tuple[str, str, int, int], list[dict[str, JsonValue]]] = defaultdict(list)
    for raw in raw_attempts:
        if not isinstance(raw, dict):
            raise ValueError("latest Epoch Attempt fact is invalid")
        revision = raw.get("kernel_agent_revision_id")
        branch = raw.get("branch")
        challenger = raw.get("challenger_ordinal")
        trajectory = raw.get("trajectory_ordinal")
        if not isinstance(revision, str) or not isinstance(branch, str):
            continue
        if not isinstance(challenger, int) or not isinstance(trajectory, int):
            continue
        identity = (
            version_by_revision.get(revision, revision),
            branch,
            challenger,
            trajectory,
        )
        grouped[identity].append(raw)
    trajectories: list[JsonValue] = []
    directions: dict[str, set[str]] = defaultdict(set)
    experiments: dict[str, set[str]] = defaultdict(set)
    for (version, branch, challenger, trajectory), attempts in sorted(grouped.items()):
        label = f"{version}:{branch}:{challenger}:trajectory-{trajectory:08d}"
        statuses = Counter(str(item.get("status")) for item in attempts)
        candidates = [item.get("candidate") for item in attempts]
        correct_candidates = [
            item for item in candidates if isinstance(item, dict) and item.get("correct") is True
        ]
        latencies = [
            latency
            for item in correct_candidates
            if (latency := _finite_number(item.get("latency_us"))) is not None
        ]
        direction_ids = sorted(
            {
                direction
                for item in attempts
                for direction in _string_values(item.get("direction_ids"))
            }
        )
        experiment_ids = sorted(
            {
                experiment
                for item in attempts
                for experiment in _string_values(item.get("experiment_ids"))
            }
        )
        for direction in direction_ids:
            directions[direction].add(label)
        for experiment in experiment_ids:
            experiments[experiment].add(label)
        trajectories.append(
            {
                "trajectory": label,
                "agent_version": version,
                "branch": branch,
                "challenger_ordinal": challenger,
                "trajectory_ordinal": trajectory,
                "attempt_count": len(attempts),
                "status_counts": cast(JsonValue, dict(sorted(statuses.items()))),
                "correct_candidate_count": len(correct_candidates),
                "accepted_as_branch_best_count": sum(
                    item.get("accepted_as_branch_best") is True for item in attempts
                ),
                "best_valid_latency_us": min(latencies) if latencies else None,
                "direction_ids": cast(JsonValue, direction_ids),
                "experiment_ids": cast(JsonValue, experiment_ids),
                "attempts": cast(
                    JsonValue,
                    [
                        {
                            "attempt_id": item.get("attempt_id"),
                            "attempt_ordinal": item.get("attempt_ordinal"),
                            "status": item.get("status"),
                            "failure_reason": item.get("failure_reason"),
                            "accepted_as_branch_best": item.get("accepted_as_branch_best"),
                            "candidate": item.get("candidate"),
                            "direction_ids": item.get("direction_ids", []),
                            "experiment_ids": item.get("experiment_ids", []),
                        }
                        for item in attempts
                    ],
                ),
            }
        )
    return {
        "epoch_number": facts.get("epoch_number"),
        "selection_reason": facts.get("selection_reason"),
        "trajectories": trajectories,
        "exact_cross_trajectory_overlaps": {
            "direction_ids": _overlaps(directions),
            "experiment_ids": _overlaps(experiments),
        },
        "interpretation": (
            "Only exact stable-ID overlap is reported. Similar hypotheses or implementations are "
            "not merged mechanically."
        ),
    }


def _finite_number(value: JsonValue) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _string_values(value: JsonValue) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(item for item in value if isinstance(item, str))


def _overlaps(values: dict[str, set[str]]) -> JsonValue:
    return cast(
        JsonValue,
        [
            {"id": identifier, "trajectories": sorted(trajectories)}
            for identifier, trajectories in sorted(values.items())
            if len(trajectories) > 1
        ],
    )


def _workflow_friction(sessions: list[_Session]) -> dict[str, JsonValue]:
    operation_counts: Counter[str] = Counter()
    operation_failures: Counter[str] = Counter()
    failures: list[JsonValue] = []
    patterns: dict[tuple[str, str], list[tuple[str, str]]] = defaultdict(list)
    recovery: Counter[str] = Counter()
    for session in sessions:
        failed_operations: set[str] = set()
        for call in session.calls:
            operation = _runtime_operation(call)
            if operation is not None:
                operation_counts[operation] += 1
                if call.failed:
                    operation_failures[operation] += 1
                    failed_operations.add(operation)
                elif operation in failed_operations:
                    recovery[operation] += 1
            if call.failed:
                failures.append(
                    {
                        "agent_version": session.version,
                        "session": session.relative_path,
                        "tool": call.name,
                        "runtime_operation": operation,
                        "input_summary": call.input_text[:400],
                        "error_summary": call.result_text[:600],
                    }
                )
            command = _command(call)
            pattern_kind = _construction_kind(command)
            if pattern_kind is not None:
                normalized = _normalize_construction(command)
                digest = hashlib.sha256(normalized.encode()).hexdigest()
                patterns[(pattern_kind, digest)].append((session.relative_path, command[:500]))
    repeated: list[dict[str, JsonValue]] = []
    for (kind, digest), occurrences in sorted(patterns.items()):
        distinct_sessions = sorted({item[0] for item in occurrences})
        if len(distinct_sessions) < 2:
            continue
        repeated.append(
            {
                "kind": kind,
                "fingerprint": f"sha256:{digest}",
                "occurrence_count": len(occurrences),
                "sessions": cast(JsonValue, distinct_sessions),
                "example": occurrences[0][1],
            }
        )
    failure_count = len(failures)
    repeated.sort(key=lambda item: cast(int, item["occurrence_count"]), reverse=True)
    repeated_count = len(repeated)
    return {
        "session_count": len(sessions),
        "runtime_operations": cast(
            JsonValue,
            [
                {
                    "operation": operation,
                    "call_count": count,
                    "failed_call_count": operation_failures[operation],
                    "failed_then_later_succeeded_session_count": recovery[operation],
                }
                for operation, count in sorted(operation_counts.items())
            ],
        ),
        "tool_failure_count": failure_count,
        "tool_failures_truncated": failure_count > _MAX_FAILURE_OBSERVATIONS,
        "tool_failures": failures[:_MAX_FAILURE_OBSERVATIONS],
        "repeated_construction_candidate_count": repeated_count,
        "repeated_construction_candidates_truncated": (
            repeated_count > _MAX_REPEATED_CONSTRUCTIONS
        ),
        "repeated_construction_candidates": cast(JsonValue, repeated[:_MAX_REPEATED_CONSTRUCTIONS]),
        "interpretation": (
            "Failures and repeated normalized constructions are review candidates. A retry, "
            "revalidation, or repeated request can be intentional and is not labeled waste."
        ),
    }


def _runtime_operation(call: _ToolCall) -> str | None:
    match = _RUNTIME_TOOL.search(call.input_text)
    return None if match is None else match.group(1)


def _command(call: _ToolCall) -> str:
    try:
        value = json.loads(call.input_text)
    except json.JSONDecodeError:
        return call.input_text
    if isinstance(value, dict):
        command = value.get("command")
        if isinstance(command, str):
            return command
    return call.input_text


def _construction_kind(command: str) -> str | None:
    if _WRITE_COMMAND.search(command) is None:
        return None
    if re.search(r"(?:scratch/)?[^\s'\"]+\.py\b", command):
        return "probe_or_helper_script"
    if re.search(r"(?:scratch/)?[^\s'\"]+\.json\b", command):
        return "runtime_request_json"
    return None


def _normalize_construction(command: str) -> str:
    value = _IDENTITY.sub("<id>", command)
    value = _DIGEST.sub("sha256:<digest>", value)
    value = _SCRATCH_PATH.sub("scratch/<file>", value)
    return _SPACE.sub(" ", value).strip()


def _json_object(path: Path, label: str) -> dict[str, JsonValue]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not valid JSON: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return cast(dict[str, JsonValue], value)


__all__ = ["materialize_evolver_review"]
