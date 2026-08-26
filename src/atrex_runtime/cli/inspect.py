"""Read-only CLI inspection commands and table rendering."""

from __future__ import annotations

import base64
import json
import os
from collections.abc import Sequence
from typing import cast

from ..artifacts.local import LocalArtifactStore
from ..config import RuntimeSettings
from ..domain.ids import (
    parse_attempt_id,
    parse_campaign_id,
    parse_epoch_id,
    parse_kernel_agent_revision_id,
    parse_kernel_revision_id,
    parse_lineage_id,
    parse_worker_session_id,
)
from ..gateway.control import SqliteGatewayControl
from ..presentation import (
    agent_catalog_entry,
    agent_revision_value,
    attempt_value,
    bootstrap_run_value,
    evaluation_value,
    kernel_catalog_entry,
    kernel_trial_value,
    kernel_value,
    lineage_attempt_values,
    lineage_epoch_values,
    worker_session_value,
)
from ..registry.sqlite import SqliteRegistry
from ..secrets import read_capability_signing_key


def list_kernels(
    config_path: str,
    campaign_value: str | None,
    lineage_value: str | None,
    output_format: str = "json",
) -> None:
    """Print the complete terminal-Kernel catalog for one scope."""
    settings = RuntimeSettings.from_file(config_path)
    artifacts = LocalArtifactStore(settings.storage.artifacts_root)
    with SqliteRegistry(settings.storage.registry_database) as registry:
        if campaign_value is not None:
            campaign_id = parse_campaign_id(campaign_value)
            entries = registry.list_campaign_kernels(campaign_id)
            scope: dict[str, object] = {"campaign_id": campaign_id}
        elif lineage_value is not None:
            lineage_id = parse_lineage_id(lineage_value)
            entries = registry.list_lineage_kernels(lineage_id)
            scope = {"lineage_id": lineage_id}
        else:
            raise AssertionError("validated Kernel catalog scope is absent")
    values = [kernel_value(entry, artifacts=artifacts) for entry in entries]
    if output_format == "table":
        _print_kernel_table(values)
        return
    if output_format != "json":
        raise ValueError(f"unsupported Kernel catalog format: {output_format}")
    print(json.dumps({**scope, "kernels": values}, sort_keys=True))


def list_epochs(
    config_path: str,
    campaign_value: str | None,
    lineage_value: str | None,
    output_format: str = "json",
) -> None:
    """Print every Epoch with pre-competition Active and terminal winner."""
    settings = RuntimeSettings.from_file(config_path)
    with SqliteRegistry(settings.storage.registry_database) as registry:
        if campaign_value is not None:
            campaign_id = parse_campaign_id(campaign_value)
            values = [
                value
                for lineage in registry.list_campaign_lineages(campaign_id)
                for value in lineage_epoch_values(registry, lineage.id)
            ]
            scope: dict[str, object] = {"campaign_id": campaign_id}
        elif lineage_value is not None:
            lineage_id = parse_lineage_id(lineage_value)
            values = lineage_epoch_values(registry, lineage_id)
            scope = {"lineage_id": lineage_id}
        else:
            raise AssertionError("validated Epoch scope is absent")
    if output_format == "table":
        _print_epoch_table(values)
        return
    if output_format != "json":
        raise ValueError(f"unsupported Epoch history format: {output_format}")
    print(json.dumps({**scope, "epochs": values}, sort_keys=True))


def _print_epoch_table(epochs: list[dict[str, object]]) -> None:
    """Print a compact Active/Challenger competition history."""

    def challenger_labels(epoch: dict[str, object]) -> str:
        proposals = cast(list[dict[str, object]], epoch["challenger_proposals"])
        return (
            ",".join(
                f"{item['agent_version']}:{item['proposal_type']}({item['base_agent_version']})"
                for item in proposals
            )
            or "-"
        )

    headers = (
        "DSL",
        "EPOCH",
        "STATUS",
        "ACTIVE_BEFORE",
        "CHALLENGERS",
        "WINNER",
        "DECISION",
        "START_KERNEL",
        "BEST_KERNEL",
        "COMPLETED",
    )
    rows = [
        (
            str(epoch["dsl"]),
            str(epoch["epoch_number"]),
            str(epoch["status"]),
            str(epoch["active_agent_version"]),
            challenger_labels(epoch),
            "-" if epoch["winner_agent_version"] is None else str(epoch["winner_agent_version"]),
            "-" if epoch["decision"] is None else str(epoch["decision"]),
            str(epoch["starting_kernel_version"]),
            "-" if epoch["best_kernel_version"] is None else str(epoch["best_kernel_version"]),
            "-" if epoch["completed_at"] is None else str(epoch["completed_at"]),
        )
        for epoch in epochs
    ]
    _print_table(headers, rows)


def list_attempts(
    config_path: str,
    campaign_value: str | None,
    lineage_value: str | None,
    output_format: str = "json",
) -> None:
    """Print every scheduled Attempt, whether or not it produced a Kernel."""
    settings = RuntimeSettings.from_file(config_path)
    with SqliteRegistry(settings.storage.registry_database) as registry:
        if campaign_value is not None:
            campaign_id = parse_campaign_id(campaign_value)
            values = [
                value
                for lineage in registry.list_campaign_lineages(campaign_id)
                for value in lineage_attempt_values(registry, lineage.id)
            ]
            scope: dict[str, object] = {"campaign_id": campaign_id}
        elif lineage_value is not None:
            lineage_id = parse_lineage_id(lineage_value)
            values = lineage_attempt_values(registry, lineage_id)
            scope = {"lineage_id": lineage_id}
        else:
            raise AssertionError("validated Attempt scope is absent")
    if output_format == "table":
        _print_attempt_table(values)
        return
    if output_format != "json":
        raise ValueError(f"unsupported Attempt history format: {output_format}")
    print(json.dumps({**scope, "attempts": values}, sort_keys=True))


def show_attempt(config_path: str, attempt_id_value: str) -> None:
    """Print one durable Attempt, including no-Candidate terminal state."""
    settings = RuntimeSettings.from_file(config_path)
    with SqliteRegistry(settings.storage.registry_database) as registry:
        value = attempt_value(registry, registry.get_attempt(parse_attempt_id(attempt_id_value)))
    print(json.dumps(value, sort_keys=True))


def list_worker_sessions(
    config_path: str,
    campaign_value: str | None,
    lineage_value: str | None,
    epoch_value: str | None,
    attempt_id_value: str | None,
    subject_value: str | None,
    output_format: str = "json",
) -> None:
    """Print the unified Worker-process catalog for one exact scope."""
    settings = RuntimeSettings.from_file(config_path)
    filters: dict[str, object] = {}
    scope: dict[str, object]
    if campaign_value is not None:
        campaign_id = parse_campaign_id(campaign_value)
        filters["campaign_id"] = campaign_id
        scope = {"campaign_id": campaign_id}
    elif lineage_value is not None:
        lineage_id = parse_lineage_id(lineage_value)
        filters["lineage_id"] = lineage_id
        scope = {"lineage_id": lineage_id}
    elif epoch_value is not None:
        epoch_id = parse_epoch_id(epoch_value)
        filters["epoch_id"] = epoch_id
        scope = {"epoch_id": epoch_id}
    elif attempt_id_value is not None:
        attempt_id = parse_attempt_id(attempt_id_value)
        filters["attempt_id"] = attempt_id
        scope = {"attempt_id": attempt_id}
    elif subject_value is not None:
        filters["subject_id"] = subject_value
        scope = {"subject_id": subject_value}
    else:
        raise AssertionError("validated Worker session scope is absent")
    with SqliteRegistry(settings.storage.registry_database) as registry:
        sessions = registry.list_worker_sessions(**filters)  # type: ignore[arg-type]
    values = [worker_session_value(session) for session in sessions]
    if output_format == "table":
        _print_worker_session_table(values)
        return
    print(json.dumps({**scope, "worker_sessions": values}, sort_keys=True))


def show_worker_session(config_path: str, session_value: str) -> None:
    """Print one Worker session including terminal diagnostics and trace digest."""
    settings = RuntimeSettings.from_file(config_path)
    with SqliteRegistry(settings.storage.registry_database) as registry:
        value = worker_session_value(
            registry.get_worker_session(parse_worker_session_id(session_value))
        )
    print(json.dumps(value, sort_keys=True))


def _print_worker_session_table(sessions: list[dict[str, object]]) -> None:
    headers = (
        "SESSION",
        "ROLE",
        "STATUS",
        "STARTED",
        "SUBJECT",
        "RUN",
        "BACKEND",
        "MODEL",
        "USAGE",
        "TRACE",
        "FINISH",
    )
    rows = [
        (
            str(session["worker_session_id"]),
            str(session["role"]),
            str(session["status"]),
            str(session["started_at"]),
            str(session["subject_id"]),
            str(session["external_run_id"]),
            "-" if session["backend"] is None else str(session["backend"]),
            "-" if session["model"] is None else str(session["model"]),
            (
                "-"
                if session["token_usage"] is None
                else (
                    f"{cast(dict[str, object], session['token_usage'])['consumed']} "
                    + (
                        "credits"
                        if cast(dict[str, object], session["token_usage"])["usage_unit"]
                        == "credits"
                        else "tokens"
                    )
                )
            ),
            (
                "-"
                if session["session_trace_digest"] is None
                else str(session["session_trace_digest"])
            ),
            "-" if session["finish_reason"] is None else str(session["finish_reason"]),
        )
        for session in sessions
    ]
    _print_table(headers, rows)


def _print_attempt_table(attempts: list[dict[str, object]]) -> None:
    """Print Attempt status independently from the sparse Kernel version history."""
    headers = (
        "DSL",
        "EPOCH",
        "BRANCH",
        "TRAJECTORY",
        "ATTEMPT",
        "INPUT",
        "OUTPUT",
        "RESULT",
        "REPORT",
        "COMPLETED",
    )
    rows = [
        (
            str(attempt["dsl"]),
            str(attempt["epoch_number"]),
            str(attempt["branch_label"]),
            str(attempt["trajectory_ordinal"]),
            str(attempt["attempt_ordinal"]),
            str(attempt["input_kernel_version"]),
            (
                "-"
                if attempt["output_kernel_version"] is None
                else str(attempt["output_kernel_version"])
            ),
            str(attempt["disposition"]),
            (
                "-"
                if attempt["attempt_report_status"] is None
                else str(attempt["attempt_report_status"])
            ),
            "-" if attempt["completed_at"] is None else str(attempt["completed_at"]),
        )
        for attempt in attempts
    ]
    _print_table(headers, rows)


def _print_kernel_table(kernels: list[dict[str, object]]) -> None:
    """Print a compact lineage-aware Kernel evolution history."""
    headers = (
        "DSL",
        "VERSION",
        "PARENT",
        "CREATED",
        "EPOCH",
        "BRANCH",
        "TRAJECTORY",
        "ATTEMPT",
        "RESULT",
        "LATENCY_US",
        "SOL_%",
        "DELTA_PARENT_%",
    )
    rows = [
        (
            str(kernel["dsl"]),
            str(kernel["version"]),
            "-" if kernel["parent_version"] is None else str(kernel["parent_version"]),
            str(kernel["created_at"]),
            "-" if kernel["epoch_number"] is None else str(kernel["epoch_number"]),
            "-" if kernel["branch_label"] is None else str(kernel["branch_label"]),
            ("-" if kernel["trajectory_ordinal"] is None else str(kernel["trajectory_ordinal"])),
            "-" if kernel["attempt_ordinal"] is None else str(kernel["attempt_ordinal"]),
            str(kernel["disposition"]),
            _format_optional_number(kernel["latency_us"], ".6g"),
            _format_optional_number(kernel["sol_percent"], ".3f"),
            _format_optional_number(kernel["improvement_over_parent_percent"], "+.3f"),
        )
        for kernel in kernels
    ]
    _print_table(headers, rows)


def _format_optional_number(value: object, format_spec: str) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("Kernel catalog numeric value has the wrong type")
    return format(float(value), format_spec)


def _print_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> None:
    widths = [
        max(len(header), *(len(row[index]) for row in rows)) for index, header in enumerate(headers)
    ]
    print("  ".join(header.ljust(widths[index]) for index, header in enumerate(headers)))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(value.ljust(widths[index]) for index, value in enumerate(row)))


def show_kernel(config_path: str, kernel_revision_value: str) -> None:
    """Print one Kernel with its Agent relation and durable measurements."""
    settings = RuntimeSettings.from_file(config_path)
    artifacts = LocalArtifactStore(settings.storage.artifacts_root)
    with SqliteRegistry(settings.storage.registry_database) as registry:
        entry = kernel_catalog_entry(registry, parse_kernel_revision_id(kernel_revision_value))
        value = kernel_value(
            entry,
            artifacts=artifacts,
            include_measurements=True,
            registry=registry,
        )
    print(json.dumps(value, sort_keys=True))


def list_agent_revisions(
    config_path: str,
    campaign_value: str | None,
    lineage_value: str | None,
    output_format: str = "json",
) -> None:
    """Print the complete versioned Agent history for one scope."""
    settings = RuntimeSettings.from_file(config_path)
    with SqliteRegistry(settings.storage.registry_database) as registry:
        if campaign_value is not None:
            campaign_id = parse_campaign_id(campaign_value)
            entries = registry.list_campaign_agent_revisions(campaign_id)
            scope: dict[str, object] = {"campaign_id": campaign_id}
        elif lineage_value is not None:
            lineage_id = parse_lineage_id(lineage_value)
            entries = registry.list_lineage_agent_revisions(lineage_id)
            scope = {"lineage_id": lineage_id}
        else:
            raise AssertionError("validated Agent catalog scope is absent")
    values = [agent_revision_value(entry) for entry in entries]
    if output_format == "table":
        _print_agent_revision_table(values)
        return
    if output_format != "json":
        raise ValueError(f"unsupported Agent catalog format: {output_format}")
    print(json.dumps({**scope, "agent_revisions": values}, sort_keys=True))


def show_agent_revision(config_path: str, revision_value: str) -> None:
    """Print one Agent revision with lineage-local version provenance."""
    settings = RuntimeSettings.from_file(config_path)
    with SqliteRegistry(settings.storage.registry_database) as registry:
        entry = agent_catalog_entry(
            registry,
            parse_kernel_agent_revision_id(revision_value),
        )
        value = agent_revision_value(entry)
    print(json.dumps(value, sort_keys=True))


def _print_agent_revision_table(revisions: list[dict[str, object]]) -> None:
    headers = (
        "DSL",
        "AGENT_VERSION",
        "PARENT",
        "CREATED",
        "EPOCH",
        "RESULT",
        "ACTIVE",
    )
    rows = [
        (
            str(revision["dsl"]),
            str(revision["agent_version"]),
            (
                "-"
                if revision["parent_agent_version"] is None
                else str(revision["parent_agent_version"])
            ),
            str(revision["created_at"]),
            (
                "-"
                if revision["introduced_epoch_number"] is None
                else str(revision["introduced_epoch_number"])
            ),
            str(revision["disposition"]),
            "yes" if revision["active"] is True else "no",
        )
        for revision in revisions
    ]
    _print_table(headers, rows)


def bootstrap_runs(
    config_path: str,
    attempt_value: str,
    generation: int | None,
) -> None:
    """Print append-only Bootstrap Session history from Gateway Control."""
    settings = RuntimeSettings.from_file(config_path)
    signing_key = read_capability_signing_key(
        os.environ,
        settings.gateway_proxy.capability_signing_key_env,
    )
    attempt_id = parse_attempt_id(attempt_value)
    with SqliteRegistry(settings.storage.registry_database) as registry:
        control = SqliteGatewayControl(
            settings.storage.gateway_database,
            registry,
            signing_key=signing_key,
        )
        try:
            if generation is None:
                value: dict[str, object] = {
                    "bootstrap_attempt_id": attempt_id,
                    "runs": [
                        bootstrap_run_value(run) for run in control.list_bootstrap_runs(attempt_id)
                    ],
                }
            else:
                if generation < 0:
                    raise ValueError("Bootstrap recovery generation cannot be negative")
                value = bootstrap_run_value(control.get_bootstrap_run(attempt_id, generation))
        finally:
            control.close()
    print(json.dumps(value, sort_keys=True))


def evaluations(
    config_path: str,
    *,
    attempt_value: str | None,
    evaluation_id: str | None,
    include_source: bool = False,
    include_result: bool = False,
) -> None:
    """Print immutable exploration/final evaluations and optionally their Artifacts."""
    settings = RuntimeSettings.from_file(config_path)
    signing_key = read_capability_signing_key(
        os.environ,
        settings.gateway_proxy.capability_signing_key_env,
    )
    artifacts = LocalArtifactStore(settings.storage.artifacts_root)
    with SqliteRegistry(settings.storage.registry_database) as registry:
        control = SqliteGatewayControl(
            settings.storage.gateway_database,
            registry,
            signing_key=signing_key,
        )
        try:
            if attempt_value is not None:
                attempt_id = parse_attempt_id(attempt_value)
                value: dict[str, object] = {
                    "attempt_id": attempt_id,
                    "evaluations": [
                        evaluation_value(item) for item in control.list_evaluations(attempt_id)
                    ],
                }
            elif evaluation_id is not None:
                evaluation = control.get_evaluation(evaluation_id)
                value = evaluation_value(evaluation)
                if include_source:
                    value["source_files"] = _artifact_files(
                        artifacts,
                        evaluation.kernel_artifact_digest,
                        max_files=settings.gateway_proxy.max_candidate_files,
                        max_bytes=settings.gateway_proxy.max_candidate_bytes,
                    )
                if include_result:
                    stored = artifacts.verify(evaluation.gateway_result_digest)
                    if stored.kind.value != "gateway_result":
                        raise ValueError("evaluation result Artifact has the wrong kind")
                    value_path = stored.payload_path / "value.json"
                    value["result"] = json.loads(value_path.read_bytes())
            else:
                raise AssertionError("evaluation CLI target is absent")
        finally:
            control.close()
    print(json.dumps(value, sort_keys=True))


def kernel_trials(
    config_path: str,
    *,
    attempt_value: str | None,
    trial_id: str | None,
    include_source: bool = False,
    include_results: bool = False,
) -> None:
    """Print immutable experimental Kernel snapshots and optional exact source files."""
    settings = RuntimeSettings.from_file(config_path)
    signing_key = read_capability_signing_key(
        os.environ,
        settings.gateway_proxy.capability_signing_key_env,
    )
    artifacts = LocalArtifactStore(settings.storage.artifacts_root)
    with SqliteRegistry(settings.storage.registry_database) as registry:
        control = SqliteGatewayControl(
            settings.storage.gateway_database,
            registry,
            signing_key=signing_key,
        )
        try:
            if attempt_value is not None:
                attempt_id = parse_attempt_id(attempt_value)
                value: dict[str, object] = {
                    "attempt_id": attempt_id,
                    "kernel_trials": [
                        kernel_trial_value(item)
                        for item in control.list_kernel_trials((attempt_id,))
                    ],
                }
            elif trial_id is not None:
                trial = control.get_kernel_trial(trial_id)
                value = kernel_trial_value(trial)
                if include_source:
                    value["source_files"] = _artifact_files(
                        artifacts,
                        trial.kernel_artifact_digest,
                        max_files=settings.gateway_proxy.max_candidate_files,
                        max_bytes=settings.gateway_proxy.max_candidate_bytes,
                    )
                if include_results:
                    results: list[dict[str, object]] = []
                    for observation in trial.observations:
                        digest = observation.result_artifact_digest
                        if digest is None:
                            results.append(
                                {
                                    "idempotency_key": observation.idempotency_key,
                                    "operation": observation.operation.value,
                                    "result": None,
                                }
                            )
                            continue
                        stored = artifacts.verify(digest)
                        value_path = stored.payload_path / "value.json"
                        if (
                            stored.kind.value != "gateway_result"
                            or not value_path.is_file()
                            or value_path.stat().st_size
                            > settings.gateway_proxy.max_candidate_bytes
                        ):
                            raise ValueError("Kernel Trial result Artifact is invalid")
                        results.append(
                            {
                                "idempotency_key": observation.idempotency_key,
                                "operation": observation.operation.value,
                                "result_artifact_digest": digest,
                                "result": json.loads(value_path.read_bytes()),
                            }
                        )
                    value["results"] = results
            else:
                raise AssertionError("Kernel Trial CLI target is absent")
        finally:
            control.close()
    print(json.dumps(value, sort_keys=True))


def _artifact_files(
    artifacts: LocalArtifactStore,
    digest: object,
    *,
    max_files: int,
    max_bytes: int,
) -> list[dict[str, object]]:
    from ..domain.ids import parse_artifact_digest

    stored = artifacts.verify(parse_artifact_digest(str(digest)))
    paths = [path for path in sorted(stored.payload_path.rglob("*")) if path.is_file()]
    if len(paths) > max_files or sum(path.stat().st_size for path in paths) > max_bytes:
        raise ValueError("evaluation candidate exceeds configured administration limits")
    files: list[dict[str, object]] = []
    for path in paths:
        payload = path.read_bytes()
        try:
            content = payload.decode("utf-8")
            encoding = "utf-8"
        except UnicodeDecodeError:
            content = base64.b64encode(payload).decode("ascii")
            encoding = "base64"
        files.append(
            {
                "path": path.relative_to(stored.payload_path).as_posix(),
                "size": len(payload),
                "encoding": encoding,
                "content": content,
            }
        )
    return files
