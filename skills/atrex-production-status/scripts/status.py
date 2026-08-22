#!/usr/bin/env python3
# ruff: noqa: E501
"""Read-only ATREX production status inspection through a Lima guest."""

from __future__ import annotations

import argparse
import base64
import os
import shlex
import subprocess
import sys
from textwrap import dedent

REMOTE_SOURCE = dedent(
    r'''
    from __future__ import annotations

    import json
    import sqlite3
    import sys
    import urllib.error
    import urllib.request
    from collections import defaultdict
    from datetime import datetime, timezone
    from pathlib import Path


    repo = Path.cwd()
    service_workspace = Path(sys.argv[1])
    if not service_workspace.is_absolute():
        service_workspace = repo / service_workspace
    runtime_url = sys.argv[2].rstrip("/")
    wiki_url = sys.argv[3].rstrip("/")
    max_error_lines = int(sys.argv[4])

    registry_path = service_workspace / "state" / "registry.sqlite"
    runtime_log = service_workspace / "services" / "runtime.log"
    core_source = repo / "src" / "atrex-kernel-agent-core" / "src"
    sys.path.insert(0, str(core_source))

    warnings = []


    def http_status(url: str, path: str) -> dict[str, object]:
        endpoint = f"{url}{path}"
        try:
            with urllib.request.urlopen(endpoint, timeout=3) as response:
                body = response.read(4096).decode("utf-8", errors="replace")
                try:
                    payload = json.loads(body)
                except json.JSONDecodeError:
                    payload = body
                return {"ok": 200 <= response.status < 300, "status": response.status, "body": payload}
        except (OSError, urllib.error.URLError) as error:
            return {"ok": False, "error": str(error)}


    def process_snapshot() -> tuple[list[Path], int]:
        workspaces: set[Path] = set()
        task_processes = 0
        for cmdline_path in Path("/proc").glob("[0-9]*/cmdline"):
            try:
                parts = [
                    value.decode("utf-8", errors="replace")
                    for value in cmdline_path.read_bytes().split(b"\0")
                    if value
                ]
            except (OSError, PermissionError):
                continue
            joined = " ".join(parts)
            is_campaign_runner = "scripts/production/campaign.sh __run" in joined
            is_runtime_task = "atrex-kernel-agent-runtime bootstrap" in joined or (
                "atrex-kernel-agent-runtime run-campaign" in joined
            )
            if is_campaign_runner or is_runtime_task:
                task_processes += 1
            if is_campaign_runner and "--workspace" in parts:
                index = parts.index("--workspace")
                if index + 1 < len(parts):
                    workspaces.add(Path(parts[index + 1]).resolve())
        return sorted(workspaces), task_processes


    def under_active_workspace(path: str, active: list[Path]) -> bool:
        candidate = Path(path).resolve()
        return any(candidate == root or root in candidate.parents for root in active)


    def manifest_context(workspace: Path) -> tuple[str | None, str | None]:
        manifest = workspace / "lineage-bootstrap.json"
        if not manifest.exists():
            return None, None
        try:
            value = json.loads(manifest.read_text(encoding="utf-8"))
            return value.get("operator"), value.get("dsl")
        except (OSError, json.JSONDecodeError) as error:
            warnings.append(f"Cannot read {manifest}: {error}")
            return None, None


    def activity_path(workspace: Path, role: str) -> Path | None:
        candidate = (
            workspace / "scratch" / "evolver-session" / "conversation.jsonl"
            if role == "evolver"
            else workspace / "sessions" / "core" / "conversation.jsonl"
        )
        return candidate if candidate.exists() else None


    def provider_stream(workspace: Path, role: str) -> Path:
        if role == "evolver":
            return workspace / "scratch" / "evolver-session" / "provider" / "stdout.stream-json"
        return workspace / "sessions" / "core" / "provider" / "stdout.stream-json"


    if not registry_path.exists():
        raise SystemExit(f"Registry not found: {registry_path}")

    active_workspaces, task_processes = process_snapshot()
    connection = sqlite3.connect(registry_path)
    connection.row_factory = sqlite3.Row

    all_sessions = connection.execute(
        """SELECT ws.*, l.dsl AS registered_dsl, c.operator AS registered_operator
           FROM worker_sessions ws
           LEFT JOIN lineages l ON l.id = ws.lineage_id
           LEFT JOIN campaigns c ON c.id = l.campaign_id
           ORDER BY ws.started_at"""
    ).fetchall()
    if active_workspaces:
        sessions = [
            row for row in all_sessions if under_active_workspace(row["workspace_path"], active_workspaces)
        ]
    else:
        sessions = [row for row in all_sessions if row["status"] == "running"]
        warnings.append("No campaign runner process was found; reporting running Sessions only")

    context_by_session: dict[str, tuple[str | None, str | None]] = {}
    for row in sessions:
        operator = row["registered_operator"]
        dsl = row["registered_dsl"]
        if operator is None or dsl is None:
            manifest_operator, manifest_dsl = manifest_context(Path(row["workspace_path"]))
            operator = operator or manifest_operator
            dsl = dsl or manifest_dsl
        context_by_session[row["id"]] = (operator, dsl)

    relevant_lineages = sorted(
        {
            row["lineage_id"]
            for row in sessions
            if row["lineage_id"] is not None
            and connection.execute("SELECT 1 FROM lineages WHERE id = ?", (row["lineage_id"],)).fetchone()
        }
    )

    routes: list[dict[str, object]] = []
    registered_ids: set[str] = set()
    for lineage_id in relevant_lineages:
        lineage = connection.execute(
            """SELECT l.*, c.operator, c.status AS campaign_status
               FROM lineages l JOIN campaigns c ON c.id = l.campaign_id
               WHERE l.id = ?""",
            (lineage_id,),
        ).fetchone()
        if lineage is None:
            continue
        registered_ids.add(lineage_id)
        lineage_sessions = [row for row in sessions if row["lineage_id"] == lineage_id]
        backend = lineage_sessions[-1]["backend"] if lineage_sessions else None
        epoch = connection.execute(
            "SELECT * FROM epochs WHERE lineage_id = ? ORDER BY number DESC LIMIT 1",
            (lineage_id,),
        ).fetchone()
        attempts = []
        if epoch is not None:
            attempts = connection.execute(
                """SELECT a.*, input.latency_us AS input_latency_us,
                          output.latency_us AS output_latency_us
                   FROM attempts a
                   JOIN kernel_revisions input ON input.id = a.input_kernel_revision_id
                   LEFT JOIN kernel_revisions output ON output.id = a.output_kernel_revision_id
                   WHERE a.epoch_id = ?
                   ORDER BY a.created_at""",
                (epoch["id"],),
            ).fetchall()
        baseline = connection.execute(
            """SELECT kr.latency_us
               FROM lineage_kernel_versions lkv
               JOIN kernel_revisions kr ON kr.id = lkv.kernel_revision_id
               WHERE lkv.lineage_id = ?
               ORDER BY lkv.revision_number LIMIT 1""",
            (lineage_id,),
        ).fetchone()
        baseline_us = None if baseline is None else baseline["latency_us"]
        registered_best = connection.execute(
            "SELECT latency_us FROM kernel_revisions WHERE id = ?",
            (lineage["best_kernel_revision_id"],),
        ).fetchone()
        registered_best_us = None if registered_best is None else registered_best["latency_us"]
        accepted_latencies = [
            attempt["output_latency_us"]
            for attempt in connection.execute(
                """SELECT a.accepted_as_branch_best, output.latency_us AS output_latency_us
                   FROM attempts a
                   JOIN epochs e ON e.id = a.epoch_id
                   LEFT JOIN kernel_revisions output ON output.id = a.output_kernel_revision_id
                   WHERE e.lineage_id = ?""",
                (lineage_id,),
            ).fetchall()
            if attempt["accepted_as_branch_best"] and attempt["output_latency_us"] is not None
        ]
        best_candidates = [value for value in [baseline_us, registered_best_us, *accepted_latencies] if value]
        observed_best_us = min(best_candidates) if best_candidates else None
        improvement = None
        if baseline_us and observed_best_us:
            improvement = (baseline_us - observed_best_us) / baseline_us * 100.0

        running_attempt = next((value for value in reversed(attempts) if value["status"] == "running"), None)
        if epoch is None:
            phase = "baseline_registered"
        elif epoch["status"] == "building_challenger":
            phase = "evolving"
        elif running_attempt is not None:
            phase = "optimizing"
        else:
            phase = f"epoch_{epoch['status']}"

        routes.append(
            {
                "operator": lineage["operator"],
                "backend": backend,
                "dsl": lineage["dsl"],
                "campaign_id": lineage["campaign_id"],
                "lineage_id": lineage_id,
                "lineage_status": lineage["status"],
                "phase": phase,
                "epoch_number": None if epoch is None else epoch["number"],
                "epoch_status": None if epoch is None else epoch["status"],
                "completed_attempts": sum(value["status"] == "completed" for value in attempts),
                "running_attempts": sum(value["status"] == "running" for value in attempts),
                "current_attempt": None
                if running_attempt is None
                else {
                    "branch": running_attempt["branch"],
                    "trajectory": running_attempt["trajectory_ordinal"],
                    "iteration": running_attempt["iteration_ordinal"],
                    "attempt_id": running_attempt["id"],
                },
                "baseline_latency_us": baseline_us,
                "registered_best_latency_us": registered_best_us,
                "observed_best_latency_us": observed_best_us,
                "improvement_percent": improvement,
                "latest_completed_attempt": next(
                    (
                        {
                            "accepted": bool(value["accepted_as_branch_best"]),
                            "input_latency_us": value["input_latency_us"],
                            "output_latency_us": value["output_latency_us"],
                            "failure_reason": value["failure_reason"],
                        }
                        for value in reversed(attempts)
                        if value["status"] == "completed"
                    ),
                    None,
                ),
            }
        )

    bootstrap_routes = []
    for row in sessions:
        if row["role"] != "framework_baseline" or row["lineage_id"] in registered_ids:
            continue
        operator, dsl = context_by_session[row["id"]]
        workspace = Path(row["workspace_path"])
        activity = activity_path(workspace, row["role"])
        bootstrap_routes.append(
            {
                "operator": operator,
                "backend": row["backend"],
                "dsl": dsl,
                "phase": "bootstrap" if row["status"] == "running" else "bootstrap_finalizing",
                "session_status": row["status"],
                "attempt_id": row["attempt_id"],
                "started_at": row["started_at"],
                "last_activity_at": None
                if activity is None
                else datetime.fromtimestamp(activity.stat().st_mtime, timezone.utc).isoformat(),
                "finish_reason": row["finish_reason"],
                "error_type": row["error_type"],
                "error_message": row["error_message"],
            }
        )

    session_status = []
    now = datetime.now(timezone.utc)
    for row in sessions:
        if row["status"] != "running":
            continue
        workspace = Path(row["workspace_path"])
        activity = activity_path(workspace, row["role"])
        activity_at = None
        age_seconds = None
        if activity is not None:
            activity_at = datetime.fromtimestamp(activity.stat().st_mtime, timezone.utc)
            age_seconds = max(0.0, (now - activity_at).total_seconds())
        operator, dsl = context_by_session[row["id"]]
        session_status.append(
            {
                "operator": operator,
                "backend": row["backend"],
                "dsl": dsl,
                "role": row["role"],
                "session_id": row["id"],
                "attempt_id": row["attempt_id"],
                "started_at": row["started_at"],
                "last_activity_at": None if activity_at is None else activity_at.isoformat(),
                "activity_age_seconds": age_seconds,
                "possibly_stale": age_seconds is None or age_seconds > 300,
            }
        )

    usage_groups: dict[tuple[str, str, str, str], dict[str, object]] = defaultdict(
        lambda: {
            "settled_sessions": 0,
            "running_sessions": 0,
            "settled": {
                "uncached_input_tokens": 0,
                "output_tokens": 0,
                "cache_read_tokens": 0,
                "cache_write_tokens": 0,
                "total_tokens": 0,
                "credits": 0.0,
            },
            "running_partial": {
                "uncached_input_tokens": 0,
                "output_tokens": 0,
                "cache_read_tokens": 0,
                "cache_write_tokens": 0,
                "total_tokens": 0,
                "credits": 0.0,
            },
        }
    )
    try:
        from backends.adapter import ClaudeAdapter, QoderAdapter
    except ImportError as error:
        ClaudeAdapter = QoderAdapter = None
        warnings.append(f"Live usage adapters unavailable: {error}")

    stage_names = {
        "framework_baseline": "bootstrap",
        "optimizer": "optimizer",
        "evolver": "evolver",
    }
    for row in sessions:
        operator, dsl = context_by_session[row["id"]]
        if operator is None or dsl is None:
            continue
        stage = stage_names.get(row["role"], row["role"])
        key = (operator, row["backend"], dsl, stage)
        group = usage_groups[key]
        unit = "credits" if row["backend"] == "qodercli" else "provider_tokens"
        if row["status"] == "completed":
            group["settled_sessions"] += 1
            settled = group["settled"]
            if unit == "credits":
                settled["credits"] += row["credits"] or 0.0
            else:
                for column in (
                    "uncached_input_tokens",
                    "output_tokens",
                    "cache_read_tokens",
                    "cache_write_tokens",
                ):
                    settled[column] += row[column] or 0
                settled["total_tokens"] = sum(
                    settled[column]
                    for column in (
                        "uncached_input_tokens",
                        "output_tokens",
                        "cache_read_tokens",
                        "cache_write_tokens",
                    )
                )
        elif row["status"] == "running":
            group["running_sessions"] += 1
            stream = provider_stream(Path(row["workspace_path"]), row["role"])
            if not stream.exists() or ClaudeAdapter is None:
                continue
            try:
                adapter = QoderAdapter() if row["backend"] == "qodercli" else ClaudeAdapter()
                _, observed = adapter.normalize_stream(stream.read_text(encoding="utf-8", errors="replace"))
                partial = group["running_partial"]
                if unit == "credits":
                    partial["credits"] += observed.credits or 0.0
                else:
                    component_map = {
                        "uncached_input_tokens": observed.input_tokens,
                        "output_tokens": observed.output_tokens,
                        "cache_read_tokens": observed.cache_read_tokens,
                        "cache_write_tokens": observed.cache_write_tokens,
                    }
                    for name, value in component_map.items():
                        partial[name] += value or 0
                    partial["total_tokens"] += observed.total_tokens or 0
            except (OSError, ValueError, json.JSONDecodeError) as error:
                warnings.append(f"Cannot parse live usage for Session {row['id']}: {error}")

    usage = []
    for key, value in sorted(usage_groups.items()):
        operator, backend, dsl, stage = key
        usage.append(
            {
                "operator": operator,
                "backend": backend,
                "dsl": dsl,
                "stage": stage,
                "unit": "credits" if backend == "qodercli" else "provider_tokens",
                **value,
            }
        )

    recent_errors = []
    if runtime_log.exists():
        lines = runtime_log.read_text(encoding="utf-8", errors="replace").splitlines()[-500:]
        recent_errors = [
            line[-2000:]
            for line in lines
            if "ERROR" in line or "Traceback" in line or "Exception" in line
        ][-max_error_lines:]

    output = {
        "generated_at": now.isoformat(),
        "services": {
            "runtime": http_status(runtime_url, "/readyz"),
            "wiki": http_status(wiki_url, "/readyz"),
        },
        "active_campaign_workspaces": [str(path) for path in active_workspaces],
        "task_processes": task_processes,
        "summary": {
            "registered_routes": len(routes),
            "bootstrap_pending_routes": len(bootstrap_routes),
            "running_bootstrap_sessions": sum(
                row["role"] == "framework_baseline" and row["status"] == "running"
                for row in sessions
            ),
            "running_optimizer_sessions": sum(
                row["role"] == "optimizer" and row["status"] == "running" for row in sessions
            ),
            "running_evolver_sessions": sum(
                row["role"] == "evolver" and row["status"] == "running" for row in sessions
            ),
            "possibly_stale_sessions": sum(item["possibly_stale"] for item in session_status),
            "recent_runtime_errors": len(recent_errors),
        },
        "routes": sorted(routes, key=lambda item: (item["operator"], item["backend"] or "", item["dsl"])),
        "bootstrap_routes": sorted(
            bootstrap_routes,
            key=lambda item: (item["operator"] or "", item["backend"] or "", item["dsl"] or ""),
        ),
        "running_sessions": sorted(
            session_status,
            key=lambda item: (item["operator"] or "", item["backend"] or "", item["dsl"] or "", item["role"]),
        ),
        "usage": usage,
        "recent_runtime_errors": recent_errors,
        "warnings": warnings,
    }
    print(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True))
    '''
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--instance",
        default=os.environ.get("ATREX_LIMA_INSTANCE", "ubuntu"),
        help="Lima instance name (default: ubuntu)",
    )
    parser.add_argument(
        "--repo",
        default=os.environ.get("ATREX_LIMA_REPO"),
        help="absolute repository path in the guest (default: $HOME/atrex-runtime)",
    )
    parser.add_argument(
        "--python",
        dest="guest_python",
        default=os.environ.get("ATREX_LIMA_PYTHON"),
        help="absolute Python path in the guest (default: $HOME/.venvs/atrex-runtime/bin/python)",
    )
    parser.add_argument(
        "--service-workspace",
        default=os.environ.get("ATREX_SERVICE_WORKSPACE", "workspaces/production/control-l20n"),
    )
    parser.add_argument("--runtime-url", default="http://127.0.0.1:8765")
    parser.add_argument("--wiki-url", default="http://127.0.0.1:8091")
    parser.add_argument("--max-error-lines", type=int, default=15)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_error_lines < 0:
        raise SystemExit("--max-error-lines must be non-negative")

    encoded = base64.b64encode(REMOTE_SOURCE.encode("utf-8")).decode("ascii")
    loader = (
        "import base64;"
        f"exec(compile(base64.b64decode({encoded!r}), '<atrex-production-status>', 'exec'))"
    )
    repo = shlex.quote(args.repo) if args.repo else '"$HOME/atrex-runtime"'
    guest_python = (
        shlex.quote(args.guest_python)
        if args.guest_python
        else '"$HOME/.venvs/atrex-runtime/bin/python"'
    )
    remote_command = " ".join(
        [
            f"cd {repo}",
            "&&",
            guest_python,
            "-c",
            shlex.quote(loader),
            shlex.quote(args.service_workspace),
            shlex.quote(args.runtime_url),
            shlex.quote(args.wiki_url),
            str(args.max_error_lines),
        ]
    )
    completed = subprocess.run(
        ["limactl", "shell", args.instance, "bash", "-lc", remote_command],
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.stdout:
        sys.stdout.write(completed.stdout)
    if completed.returncode != 0:
        if completed.stderr:
            sys.stderr.write(completed.stderr)
        return completed.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
