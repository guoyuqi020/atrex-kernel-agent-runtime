#!/usr/bin/env python3
"""Materialize three isolated DSL Campaign workspaces with one shared Runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pwd
import re
import secrets
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

SUPPORTED_BACKENDS = ("claude", "codex", "qodercli", "pi")
SUPPORTED_DSLS = ("cuda", "triton", "cutedsl")
SUPPORTED_LAUNCHER_MODES = ("sandbox", "container")
ABLATION_OPTIMIZER_ATTEMPT_BUDGET_PER_TRAJECTORY = 15


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build three single-DSL production Campaigns from one Atrex-Bench operator"
    )
    parser.add_argument(
        "--kernel",
        help=(
            "Atrex-Bench operator directory, data-relative suite/operator, or a unique "
            "operator directory name"
        ),
    )
    parser.add_argument("--backend", choices=SUPPORTED_BACKENDS)
    parser.add_argument(
        "--services-only",
        action="store_true",
        help="initialize a backend-neutral long-running control-plane workspace",
    )
    parser.add_argument("--workspace", type=Path)
    parser.add_argument(
        "--service-workspace",
        type=Path,
        help=(
            "prepared production workspace whose already-running Runtime/Wiki control "
            "plane is shared by this task"
        ),
    )
    parser.add_argument("--hardware-target", default=os.environ.get("AGATE_GPU"))
    parser.add_argument("--operator")
    parser.add_argument(
        "--seed-source",
        help=(
            "operator-relative seed copied to kernel.py; defaults to the --kernel file "
            "or reference.py when --kernel names a directory"
        ),
    )
    parser.add_argument(
        "--dsl-seed-source",
        action="append",
        default=[],
        metavar="DSL=PATH",
        help=(
            "seed one DSL Campaign from its own kernel, repeatable as cuda=..., "
            "triton=..., cutedsl=...; PATH may be operator-relative or absolute, and "
            "any DSL left unspecified falls back to --seed-source"
        ),
    )
    parser.add_argument("--optimizer-model")
    parser.add_argument("--evolver-model")
    parser.add_argument("--runtime-host")
    parser.add_argument("--runtime-port", type=int)
    parser.add_argument("--wiki-url", default=os.environ.get("ATREX_WIKI_URL"))
    parser.add_argument("--worker-user", default=os.environ.get("ATREX_SANDBOX_WORKER_USER"))
    parser.add_argument(
        "--launcher-mode",
        choices=SUPPORTED_LAUNCHER_MODES,
        default=os.environ.get("ATREX_LAUNCHER_MODE"),
        help=(
            "sandbox uses bwrap+systemd+cgroup; container uses bwrap directly and "
            "delegates aggregate resource limits to the outer OCI container"
        ),
    )
    parser.add_argument(
        "--policy",
        type=Path,
        default=Path(__file__).with_name("policy.json"),
    )
    parser.add_argument(
        "--workspace-output",
        type=Path,
        help="write the resolved workspace path for a calling script",
    )
    return parser.parse_args()


def _load_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise SystemExit(f"{label} not found: {path}") from error
    except json.JSONDecodeError as error:
        raise SystemExit(f"{label} is not valid JSON: {path}: {error}") from error
    if not isinstance(value, dict):
        raise SystemExit(f"{label} must contain one JSON object: {path}")
    return cast(dict[str, Any], value)


def _require_production_gate(policy: dict[str, Any]) -> None:
    """Reject production materialization without the content-level policy gate."""
    gate_policy = policy.get("gate_policy")
    if not isinstance(gate_policy, dict) or gate_policy.get("production_gate") is not True:
        raise SystemExit(
            "production policy must enable gate_policy.production_gate; "
            "production scripts do not permit an ungated Campaign"
        )


def _write_text(path: Path, value: str, *, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(value, encoding="utf-8")
    temporary.chmod(mode)
    temporary.replace(path)


def _write_json(path: Path, value: object, *, mode: int = 0o600) -> None:
    _write_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n", mode=mode)


def _git_commit(
    repository: Path,
    *,
    require_clean: bool = False,
    label: str | None = None,
) -> str:
    result = subprocess.run(
        ("git", "-C", str(repository), "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
    )
    commit = result.stdout.strip()
    if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise SystemExit(f"repository did not resolve to a full commit: {repository}")
    if require_clean:
        status = subprocess.run(
            (
                "git",
                "-C",
                str(repository),
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
            ),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
        if status:
            shown = status[:20]
            suffix = "" if len(status) <= len(shown) else f"\n  ... {len(status) - len(shown)} more"
            detail = "\n  ".join(shown)
            raise SystemExit(
                f"{label or repository.name} working tree is dirty; production Agent Bundle "
                "commits must identify the exact source bytes. Commit or clean these changes "
                f"before preparing a new Campaign:\n  {detail}{suffix}"
            )
    return commit


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-") or "kernel"


def _private_shapes_path(operator_root: Path) -> Path:
    """Resolve the evaluator-owned exact Shape contract using Atrex-Bench precedence."""
    preferred = operator_root / "shape_valid.json"
    if preferred.is_file() and not preferred.is_symlink():
        return preferred
    legacy = operator_root / "shapes.json"
    if legacy.is_file() and not legacy.is_symlink():
        return legacy
    raise SystemExit(
        "Atrex-Bench kernel requires a regular shape_valid.json "
        f"(or migration shapes.json): {operator_root}"
    )


def _resolve_kernel(data_root: Path, raw: str) -> tuple[Path, str | None]:
    direct = Path(raw).expanduser()
    candidates: list[tuple[Path, str | None]] = []
    if direct.is_dir():
        candidates.append((direct.resolve(), None))
    elif direct.is_file():
        candidates.append((direct.parent.resolve(), direct.name))
    relative = data_root / raw
    if relative.is_dir():
        candidates.append((relative.resolve(), None))
    elif relative.is_file():
        candidates.append((relative.parent.resolve(), relative.name))
    if "/" not in raw and os.sep not in raw:
        candidates.extend(
            (path.resolve(), None) for path in data_root.glob(f"*/{raw}") if path.is_dir()
        )
    unique = sorted(set(candidates))
    if not unique:
        raise SystemExit(
            f"Atrex-Bench kernel not found: {raw!r}; use an operator directory or suite/operator"
        )
    if len(unique) != 1:
        choices = "\n  ".join(str(path) for path, _seed in unique[:20])
        raise SystemExit(f"kernel name is ambiguous; use suite/operator:\n  {choices}")
    operator_root, selected_seed = unique[0]
    for name in ("reference.py", "input.py"):
        path = operator_root / name
        if path.is_symlink() or not path.is_file():
            raise SystemExit(f"Atrex-Bench kernel requires a regular {name}: {operator_root}")
    _private_shapes_path(operator_root)
    return operator_root, selected_seed


def _resolve_executable(name: str, fallback: str) -> str:
    return shutil.which(name) or fallback


def _resolver_source() -> str:
    preferred = Path("/run/systemd/resolve/resolv.conf")
    if preferred.is_file() and not preferred.is_symlink():
        return str(preferred)
    resolved = Path("/etc/resolv.conf").resolve()
    return str(resolved)


def _ensure_runtime_secrets(path: Path) -> None:
    if path.is_file():
        return
    lines = (
        "# Generated local control-plane secrets. Keep this file private.\n"
        f"export ATREX_CAPABILITY_SIGNING_KEY={shlex.quote(secrets.token_urlsafe(48))}\n"
        f"export ATREX_ADMIN_BEARER_TOKEN={shlex.quote(secrets.token_hex(32))}\n"
    )
    _write_text(path, lines)


def _shared_control_plane(
    service_workspace: Path,
) -> tuple[dict[str, Any], str, str]:
    """Load the immutable state and endpoint configuration of a service workspace."""
    service_workspace = service_workspace.expanduser().resolve()
    manifest = _load_object(
        service_workspace / "production-manifest.json", "service production manifest"
    )
    if manifest.get("layout") not in {
        "production-control-plane",
        "shared-runtime-per-dsl-campaign-workspaces",
    }:
        raise SystemExit("service workspace does not use the supported production layout")
    runtime = _load_object(service_workspace / "runtime.json", "service Runtime config")
    secrets_path = service_workspace / "runtime.env"
    try:
        secrets_text = secrets_path.read_text(encoding="utf-8")
    except FileNotFoundError as error:
        raise SystemExit(f"service Runtime secrets not found: {secrets_path}") from error
    required_sections = (
        "server",
        "administration",
        "storage",
        "gateway_proxy",
        "agate",
        "gpu_wiki",
        "gate_policy",
    )
    missing = [name for name in required_sections if not isinstance(runtime.get(name), dict)]
    if missing:
        raise SystemExit("service Runtime config is missing shared sections: " + ", ".join(missing))
    launcher_mode = str(manifest.get("launcher_mode", "sandbox"))
    if launcher_mode not in SUPPORTED_LAUNCHER_MODES:
        raise SystemExit(f"unsupported service launcher mode: {launcher_mode}")
    return runtime, secrets_text, launcher_mode


def _attach_shared_control_plane(runtime: dict[str, Any], service_runtime: dict[str, Any]) -> None:
    """Point a task-local scheduler config at one long-running trusted control plane."""
    for section in (
        "server",
        "administration",
        "storage",
        "gateway_proxy",
        "agate",
        "gpu_wiki",
        "gate_policy",
    ):
        runtime[section] = service_runtime[section]
    server = cast(dict[str, Any], service_runtime["server"])
    campaign = cast(dict[str, Any], runtime["campaign"])
    campaign["gateway_proxy_url"] = f"http://{server['host']}:{server['port']}"
    campaign["gate_policy"] = service_runtime["gate_policy"]


def _local_wiki_config(root: Path, workspace: Path) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "host": "127.0.0.1",
        "port": 8091,
        "reference_root": str(root / "local-wiki/corpus/gpu-wiki"),
        "store_root": str(workspace / "wiki-state/gpu-wiki"),
        "database": str(workspace / "wiki-state/local-wiki.sqlite"),
        "bearer_token_env": None,
        "max_request_bytes": 4194304,
        "max_concurrent_queries": 16,
        "max_response_bytes": 1048576,
    }


def _prepare_service_workspace(
    args: argparse.Namespace,
    *,
    root: Path,
    policy: dict[str, Any],
    hardware_target: str,
) -> None:
    """Create the persistent Runtime/Wiki control plane without a task backend."""
    if args.workspace is None:
        raise SystemExit("--workspace is required with --services-only")
    if args.kernel is not None or args.backend is not None or args.service_workspace is not None:
        raise SystemExit(
            "--services-only does not accept --kernel, --backend, or --service-workspace"
        )
    runtime_policy = cast(dict[str, Any], policy["runtime"])
    host = str(args.runtime_host or runtime_policy["host"])
    port = int(args.runtime_port or runtime_policy["port"])
    wiki_url = str(args.wiki_url or runtime_policy["wiki_url"])
    worker_user = str(
        args.worker_user or os.environ.get("SUDO_USER") or pwd.getpwuid(os.getuid()).pw_name
    )
    launcher_mode = str(args.launcher_mode or "sandbox")
    workspace = args.workspace.expanduser().resolve()
    manifest_path = workspace / "production-manifest.json"
    if manifest_path.is_file():
        manifest = _load_object(manifest_path, "production control-plane manifest")
        if manifest.get("layout") != "production-control-plane":
            raise SystemExit("workspace is not a production control plane")
        required = (
            workspace / "runtime.json",
            workspace / "runtime.env",
            workspace / "local-wiki.json",
        )
        if not all(path.is_file() for path in required):
            raise SystemExit("production control-plane workspace is incomplete")
        existing_mode = str(manifest.get("launcher_mode", "sandbox"))
        if args.launcher_mode is not None and existing_mode != launcher_mode:
            raise SystemExit(
                "production control-plane workspace is pinned to launcher mode "
                f"{existing_mode!r}, not {launcher_mode!r}"
            )
        print(f"Reusing production control plane: {workspace}")
        return

    core_commit = _git_commit(
        root / "src/atrex-kernel-agent-core",
        require_clean=True,
        label="Core",
    )
    evolver_commit = _git_commit(
        root / "src/atrex-kernel-agent-evolver",
        require_clean=True,
        label="Evolver",
    )
    bench_commit = _git_commit(root / "third_party/atrex-bench")
    host_home = os.environ.get("ATREX_SANDBOX_HOST_HOME", str(Path.home()))
    # Construct the common trusted policy, then remove all task Worker
    # configuration from the persistent control plane.
    runtime = _runtime_config(
        root=root,
        workspace=workspace,
        backend="pi",
        hardware_target=hardware_target,
        policy=policy,
        host=host,
        port=port,
        wiki_url=wiki_url,
        worker_user=worker_user,
        host_home=host_home,
        launcher_mode=launcher_mode,
        evolver_commit=evolver_commit,
        bench_commit=bench_commit,
    )
    runtime["campaign"] = None
    workspace.mkdir(parents=True, exist_ok=True)
    _write_json(workspace / "runtime.json", runtime)
    _write_json(workspace / "local-wiki.json", _local_wiki_config(root, workspace))
    _ensure_runtime_secrets(workspace / "runtime.env")
    _write_json(
        manifest_path,
        {
            "schema_version": 1,
            "layout": "production-control-plane",
            "hardware_target": hardware_target,
            "production_policy_digest": hashlib.sha256(
                json.dumps(policy, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
            "core_commit": core_commit,
            "evolver_commit": evolver_commit,
            "atrex_bench_commit": bench_commit,
            "worker_user": worker_user,
            "launcher_mode": launcher_mode,
        },
    )
    from atrex_runtime.config import RuntimeSettings

    RuntimeSettings.from_file(workspace / "runtime.json")
    print(f"Production control plane: {workspace}")
    print(f"Agate GPU environment: {hardware_target}")
    print(f"Runtime config: {workspace / 'runtime.json'}")


def _evaluation_contract(operator_root: Path) -> dict[str, Any]:
    shapes_path = _private_shapes_path(operator_root)
    shapes = _load_object(shapes_path, shapes_path.name)
    invalid_shapes = [
        shape_id
        for shape_id, shape in shapes.items()
        if not isinstance(shape, dict)
        or not isinstance(shape.get("init_kwargs"), (dict, type(None)))
        or not isinstance(shape.get("input_kwargs"), dict)
    ]
    if invalid_shapes:
        raise SystemExit(
            f"{shapes_path} entries require optional init_kwargs and object input_kwargs: "
            + ", ".join(invalid_shapes[:8])
        )
    metadata_path = operator_root / "metadata.json"
    roofline_path = operator_root / "roofline.json"
    metadata = _load_object(metadata_path, "metadata.json") if metadata_path.is_file() else None
    roofline = _load_object(roofline_path, "roofline.json") if roofline_path.is_file() else None
    return {
        "schema_version": 1,
        "candidate_path": "kernel.py",
        "reference_py": (operator_root / "reference.py").read_text(encoding="utf-8"),
        "input_py": (operator_root / "input.py").read_text(encoding="utf-8"),
        "shapes": shapes,
        "metadata": metadata,
        "roofline": roofline,
        "options": {
            "num_correctness_cases": 1,
            "bench_iters": 100,
            "atol": 0.01,
            "rtol": 0.05,
            "timeout_s": 600,
        },
        "env_vars": {},
        "requirements": [],
        "deps_mode": "freeze_installed",
        "mode": "full",
        "lock_clocks": True,
        "harness": "atrex_bench",
        "runner_overrides": {},
    }


def _runtime_config(
    *,
    root: Path,
    workspace: Path,
    backend: str,
    hardware_target: str,
    policy: dict[str, Any],
    host: str,
    port: int,
    wiki_url: str,
    worker_user: str,
    host_home: str,
    launcher_mode: str,
    evolver_commit: str,
    bench_commit: str,
) -> dict[str, Any]:
    runtime_policy = cast(dict[str, Any], policy["runtime"])
    workers = cast(dict[str, Any], policy["workers"])
    sandbox = cast(dict[str, Any], policy["sandbox"])
    state = workspace / "state"
    bench = root / "third_party/atrex-bench"
    python = str(Path(sys.executable).resolve())
    comparison = cast(dict[str, Any], policy["comparison"])
    roofline_sku = os.environ.get("ATREX_ROOFLINE_SKU", "").strip()
    gate_policy = dict(cast(dict[str, Any], policy["gate_policy"]))
    gate_policy["evaluator"] = {
        "repository": str(bench),
        "commit": bench_commit,
        "git_executable": _resolve_executable("git", "/usr/bin/git"),
        "fetch_timeout_seconds": 120,
        "max_archive_bytes": 8388608,
        "max_bundle_files": 128,
        "max_bundle_bytes": 4194304,
    }
    optional_provider_env = [
        "ANTHROPIC_AUTH_TOKEN",
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_BASE_URL",
        "ANTHROPIC_MODEL",
        "ANTHROPIC_DEFAULT_HAIKU_MODEL",
        "ANTHROPIC_DEFAULT_OPUS_MODEL",
        "ANTHROPIC_DEFAULT_SONNET_MODEL",
        "CODEX_HOME",
        "QODER_PERSONAL_ACCESS_TOKEN",
    ]
    worker_environment = {
        "values": {},
        "inherit": ["PATH"],
        "inherit_optional": optional_provider_env,
    }
    if launcher_mode == "container":
        launcher: dict[str, Any] = {
            "mode": "container",
            "env_executable": _resolve_executable("env", "/usr/bin/env"),
            "backend_credentials": {
                "enabled": True,
                "host_home": host_home,
            },
            "container": {
                "bwrap_executable": _resolve_executable("bwrap", "/usr/bin/bwrap"),
                "sandbox_home": str(sandbox["sandbox_home"]),
                "workspace_mount": str(sandbox["workspace_mount"]),
                "resolv_conf": _resolver_source(),
                "read_only_bind_paths": [],
                "hidden_host_paths": [],
                "reference_projects_root": str(root / "third_party/reference-projects"),
            },
        }
    else:
        launcher = {
            "mode": "sandbox",
            "env_executable": _resolve_executable("env", "/usr/bin/env"),
            "backend_credentials": {
                "enabled": True,
                "host_home": host_home,
                "development_bwrap_executable": _resolve_executable("bwrap", "/usr/bin/bwrap"),
            },
            "sandbox": {
                "bwrap_executable": _resolve_executable("bwrap", "/usr/bin/bwrap"),
                "systemd_run_executable": _resolve_executable(
                    "systemd-run", "/usr/bin/systemd-run"
                ),
                "systemd_user": False,
                "worker_user": worker_user,
                "sandbox_home": str(sandbox["sandbox_home"]),
                "workspace_mount": str(sandbox["workspace_mount"]),
                "resolv_conf": _resolver_source(),
                "read_only_bind_paths": [],
                "hidden_host_paths": [],
                "reference_projects_root": str(root / "third_party/reference-projects"),
                "resources": {
                    "memory_max_bytes": int(sandbox["memory_max_bytes"]),
                    "memory_swap_max_bytes": int(sandbox["memory_swap_max_bytes"]),
                    "cpu_quota_percent": int(sandbox["cpu_quota_percent"]),
                    "tasks_max": int(sandbox["tasks_max"]),
                },
            },
        }
    campaign = {
        "attempt_workspaces_root": str(state / "attempt-workspaces"),
        "evolution_workspaces_root": str(state / "evolution-workspaces"),
        "problem_generalization_workspaces_root": str(state / "problem-generalization-workspaces"),
        "lineage_bootstrap_workspaces_root": str(state / "lineage-bootstrap-workspaces"),
        "fencing_lease_seconds": 120,
        "fencing_heartbeat_seconds": 30,
        "gateway_proxy_url": f"http://{host}:{port}",
        "gateway_operations": [
            "evaluate",
            "profile",
            "dev",
            "check",
            "disassemble",
            "poll",
            "jobs",
            "cancel",
            "env",
            "health",
            "config",
        ],
        "gateway_max_calls": int(runtime_policy["gateway_max_calls"]),
        "gateway_capability_lifetime_seconds": int(
            runtime_policy["gateway_capability_lifetime_seconds"]
        ),
        "gate_policy": gate_policy,
        "max_infrastructure_retries": int(runtime_policy["max_infrastructure_retries"]),
        "bootstrap_max_parallel_lineages": int(runtime_policy["bootstrap_max_parallel_lineages"]),
        "max_parallel_branches": int(runtime_policy["max_parallel_branches"]),
        "roofline_builder": {
            "repository": str(bench),
            "commit": bench_commit,
            "git_executable": _resolve_executable("git", "/usr/bin/git"),
            "python_executable": python,
            "fetch_timeout_seconds": 120,
            "execution_timeout_seconds": 120,
            "max_archive_bytes": 268435456,
            "max_output_bytes": 8388608,
            "sku_by_hardware_target": ({hardware_target: roofline_sku} if roofline_sku else {}),
        },
        "kernel_retention_comparison": dict(comparison),
        "agent_promotion_comparison": dict(comparison),
        "evidence": {
            "max_trace_files": 8,
            "max_trace_bytes": 67108864,
            "max_trace_events": 100000,
            "max_projection_text_bytes": 1048576,
            "max_diff_files": 128,
            "max_diff_bytes": 2097152,
            "redaction_patterns": [],
        },
        "optimizer": {
            "agent_backend": backend,
            "reasoning_effort": str(workers["reasoning_effort"]),
            "session_settings": "",
            "command_prefix": [python],
            "environment": worker_environment,
            "isolated_home_environment_keys": ["HOME"],
            "session_trace_relative_path": "sessions/core",
            "token_usage_report_relative_path": "scratch/token-usage.json",
            "max_attempt_report_bytes": 65536,
            "timeout_seconds": int(workers["optimizer_timeout_seconds"]),
            "bootstrap_timeout_seconds": int(workers["bootstrap_timeout_seconds"]),
            "terminate_grace_seconds": 10,
            "max_diagnostic_bytes": 131072,
            "max_session_tokens": int(workers["optimizer_max_session_tokens"]),
            "max_session_credits": int(workers["optimizer_max_session_credits"]),
        },
        "evolver": {
            "agent_backend": backend,
            "reasoning_effort": str(workers["reasoning_effort"]),
            "session_settings": "",
            "repository": str(root / "src/atrex-kernel-agent-evolver"),
            "commit": evolver_commit,
            "git_executable": _resolve_executable("git", "/usr/bin/git"),
            "fetch_timeout_seconds": 120,
            "max_archive_bytes": 16777216,
            "command_prefix": [python],
            "max_bundle_files": 1024,
            "max_bundle_bytes": 8388608,
            "isolated_home_environment_keys": ["HOME"],
            "session_trace_relative_path": "scratch/evolver-session",
            "token_usage_report_relative_path": "scratch/token-usage.json",
            "environment": worker_environment,
            "timeout_seconds": int(workers["evolver_timeout_seconds"]),
            "terminate_grace_seconds": 10,
            "max_diagnostic_bytes": 131072,
            "max_output_manifest_bytes": 16384,
        },
        "launcher": launcher,
    }
    gpu_wiki: dict[str, Any] = {
        "base_url": wiki_url,
        "timeout_seconds": 30,
        "max_proxy_request_bytes": 131072,
        "max_query_bytes": 65536,
        "max_response_bytes": 1048576,
    }
    wiki_token_env = os.environ.get("ATREX_WIKI_TOKEN_ENV", "").strip()
    if wiki_token_env:
        gpu_wiki["bearer_token_env"] = wiki_token_env
    return {
        "schema_version": 1,
        "server": {"host": host, "port": port},
        "administration": {
            "bearer_token_env": "ATREX_ADMIN_BEARER_TOKEN",
            "max_request_bytes": 65536,
            "event_page_limit": 500,
            "event_export_limit": 10000,
            "event_prune_limit": 1000,
            "task_lease_seconds": 7200,
            "task_heartbeat_seconds": 60,
            "task_poll_seconds": 2,
            "max_error_bytes": 8192,
        },
        "storage": {
            "registry_database": str(state / "registry.sqlite"),
            "gateway_database": str(state / "gateway.sqlite"),
            "agate_jobs_database": str(state / "agate-jobs.sqlite"),
            "artifacts_root": str(state / "artifacts"),
        },
        "gateway_proxy": {
            "max_request_bytes": 1048576,
            "max_candidate_files": 64,
            "max_candidate_bytes": 524288,
            "capability_signing_key_env": "ATREX_CAPABILITY_SIGNING_KEY",
            "candidate_diff_allowed_paths": {
                "cuda": ["**/*.py", "*.py", "**/*.cu", "*.cu", "**/*.cuh", "*.cuh"],
                "triton": ["**/*.py", "*.py"],
                "cutedsl": ["**/*.py", "*.py"],
            },
            "candidate_diff_require_change": True,
        },
        "agate": {
            "base_url": os.environ.get("AGATE_URL", "http://127.0.0.1:9000"),
            "auth_mode": "ak_sk",
            "access_key_env": "AGATE_AK",
            "secret_key_env": "AGATE_SK",
            "http_timeout_s": int(os.environ.get("AGATE_HTTP_TIMEOUT", "1800")),
            "wait_timeout_s": int(os.environ.get("AGATE_WAIT_TIMEOUT", "3900")),
            "health_check_interval_s": int(os.environ.get("AGATE_HEALTH_CHECK_INTERVAL", "30")),
        },
        "kernel_agent": {
            "max_bundle_files": 1024,
            "max_bundle_bytes": 8388608,
            "max_entrypoint_bytes": 524288,
            "max_agent_problem_bytes": 262144,
            "base_source": {
                "repository": str(root / "src/atrex-kernel-agent-core"),
                "git_executable": _resolve_executable("git", "/usr/bin/git"),
                "fetch_timeout_seconds": 120,
                "max_archive_bytes": 67108864,
                "allowed_submodules": {},
            },
        },
        "gpu_wiki": gpu_wiki,
        "gate_policy": gate_policy,
        "campaign": campaign,
    }


def _ablation_plan(policy: dict[str, Any]) -> dict[str, Any]:
    """Derive control arms with 15 post-Bootstrap Active Attempts per Trajectory."""
    schedule = cast(dict[str, Any], policy["schedule"])
    enabled = bool(schedule.get("event_only", False))
    trajectories = int(schedule["trajectories_per_branch"])
    challengers = int(schedule["challenger_count"])
    default_attempts = int(schedule["attempts_per_trajectory"])
    attempt_budget = ABLATION_OPTIMIZER_ATTEMPT_BUDGET_PER_TRAJECTORY
    # Pair Isolated and Retained replicas for every configured Active/Challenger Trajectory. The
    # configured Challenger count is used rather than the challenger_start_epoch-gated one
    # so arm identity is stable across Epochs.
    total = trajectories * (1 + challengers)
    arms: list[dict[str, Any]] = []

    def arm(
        *,
        kind: str,
        label: str,
        attempts_per_trajectory: int,
        ephemeral_agent_state: bool,
        trajectories_per_branch: int = 1,
        challenger_count: int = 0,
    ) -> dict[str, Any]:
        if attempt_budget % attempts_per_trajectory:
            raise ValueError(
                f"Ablation Arm {label} cannot spend exactly {attempt_budget} Attempts "
                "per Trajectory: "
                f"{attempts_per_trajectory} Attempts per Epoch does not divide the budget"
            )
        target_epoch = attempt_budget // attempts_per_trajectory
        challenger_epochs = max(0, target_epoch - 1)
        return {
            "kind": kind,
            "label": label,
            "trajectories_per_branch": trajectories_per_branch,
            "attempts_per_trajectory": attempts_per_trajectory,
            "target_epoch_number": target_epoch,
            "ephemeral_agent_state": ephemeral_agent_state,
            "challenger_count": challenger_count,
            "challenger_start_epoch": 2,
            "first_epoch_same_agent": challenger_count > 0,
            "optimizer_attempt_budget_total": trajectories_per_branch
            * attempt_budget
            * (1 + challenger_count),
            "evolution_count": challenger_count * challenger_epochs,
        }

    if enabled:
        # Isolated arms remain independent replicas. Each one receives the full budget.
        arms.extend(
            arm(
                kind="isolated",
                label=f"ablation-isolated-{ordinal:02d}",
                attempts_per_trajectory=default_attempts,
                ephemeral_agent_state=True,
            )
            for ordinal in range(1, total + 1)
        )
        # Every Pool runs two parallel Trajectories, each with its own full budget.
        # Labels count serial Attempts per Trajectory per Epoch, not the arm total.
        arms.extend(
            arm(
                kind="pooled",
                label=f"ablation-pool-{attempts}",
                attempts_per_trajectory=attempts,
                ephemeral_agent_state=True,
                trajectories_per_branch=2,
            )
            for attempts in (3, 1, 5)
        )
        # Pair each Pool schedule with persistent Skills/Tools, without an Evolver.
        arms.extend(
            arm(
                kind="pool-retained",
                label=f"ablation-pool-retained-{attempts}",
                attempts_per_trajectory=attempts,
                ephemeral_agent_state=False,
                trajectories_per_branch=2,
            )
            for attempts in (1, 3, 5)
        )
        # Match each Isolated replica, retaining only its own Skills and Tools.
        arms.extend(
            arm(
                kind="retained",
                label=f"ablation-retained-{ordinal:02d}",
                attempts_per_trajectory=default_attempts,
                ephemeral_agent_state=False,
            )
            for ordinal in range(1, total + 1)
        )
        # Reuse the same frozen v0 while varying how often a new Challenger is created.
        # The main Campaign already supplies evolve-3 under the default policy.
        arms.extend(
            arm(
                kind="evolve",
                label=f"ablation-evolve-{attempts}",
                attempts_per_trajectory=attempts,
                ephemeral_agent_state=False,
                challenger_count=1,
            )
            for attempts in (1, 5)
        )
    return {
        "schema_version": 4,
        "enabled": enabled,
        "optimizer_attempt_budget_per_trajectory": attempt_budget,
        "arms": arms,
    }


def _campaign(
    *,
    workspace: Path,
    dsl: str,
    operator_root: Path,
    operator: str,
    hardware_target: str,
    backend: str,
    core_commit: str,
    evolver_commit: str,
    policy: dict[str, Any],
    optimizer_model: str | None,
    evolver_model: str | None,
) -> dict[str, Any]:
    schedule = cast(dict[str, Any], policy["schedule"])
    dsls = cast(list[str], schedule["dsls"])
    if tuple(dsls) != SUPPORTED_DSLS:
        raise SystemExit(
            "production policy must select cuda, triton, and cutedsl in canonical order"
        )
    relative_identity = f"{operator_root.parent.name}/{operator_root.name}"
    digest = hashlib.sha256(relative_identity.encode()).hexdigest()[:12]
    if dsl not in dsls:
        raise SystemExit(f"production DSL is not enabled by policy: {dsl}")
    creation_key = "-".join(
        (
            _slug(operator),
            _slug(hardware_target),
            _slug(backend),
            dsl,
            digest,
            core_commit[:12],
        )
    )[:200]
    lineages = {
        dsl: {
            "models": {"optimizer": optimizer_model, "evolver": evolver_model},
            "baseline_kernel": str(workspace / "inputs" / "baseline-kernel"),
            "initial_evidence": str(workspace / "inputs" / "initial-evidence"),
        }
    }
    shape_train = operator_root / "shape_train.json"
    agent_problem = operator_root / "agent_problem.json"
    return {
        "schema_version": 3,
        "creation_key": creation_key,
        "operator": operator,
        "hardware_target": hardware_target,
        "evaluation_contract": str(workspace / "evaluation-contract.json"),
        "shape_train": str(shape_train) if shape_train.is_file() else None,
        "agent_problem": (
            str(agent_problem) if not shape_train.is_file() and agent_problem.is_file() else None
        ),
        "problem_generalization_model": None,
        "base_revision": {"commit": core_commit},
        "challenger_count": int(schedule["challenger_count"]),
        "challenger_start_epoch": int(schedule["challenger_start_epoch"]),
        "first_epoch_same_agent": bool(schedule.get("first_epoch_same_agent", False)),
        "trajectories_per_branch": int(schedule["trajectories_per_branch"]),
        "attempts_per_trajectory": int(schedule["attempts_per_trajectory"]),
        "lineages": lineages,
    }


def main() -> None:
    args = _arguments()
    root = Path(__file__).resolve().parents[2]
    policy = _load_object(args.policy.resolve(), "production policy")
    if policy.get("schema_version") != 1:
        raise SystemExit("unsupported production policy schema_version")
    _require_production_gate(policy)
    hardware_target = str(args.hardware_target or "").strip()
    if not hardware_target:
        raise SystemExit("--hardware-target or AGATE_GPU is required")
    if not os.environ.get("AGATE_URL", "").strip():
        raise SystemExit("AGATE_URL is required when preparing a production Runtime")
    if args.services_only:
        _prepare_service_workspace(
            args,
            root=root,
            policy=policy,
            hardware_target=hardware_target,
        )
        return
    if args.kernel is None or args.backend is None:
        raise SystemExit("--kernel and --backend are required for a Campaign task")
    data_root = root / "third_party/atrex-bench/data"
    operator_root, selected_seed = _resolve_kernel(data_root, str(args.kernel))
    operator = str(args.operator or operator_root.name).strip()
    if not operator:
        raise SystemExit("operator name cannot be empty")
    runtime_policy = cast(dict[str, Any], policy["runtime"])
    service_workspace = (
        args.service_workspace.expanduser().resolve()
        if args.service_workspace is not None
        else None
    )
    service_runtime: dict[str, Any] | None = None
    service_secrets: str | None = None
    launcher_mode = str(args.launcher_mode or "sandbox")
    if service_workspace is not None:
        if (
            args.runtime_host is not None
            or args.runtime_port is not None
            or args.wiki_url is not None
        ):
            raise SystemExit(
                "--runtime-host, --runtime-port, and --wiki-url cannot override a shared "
                "service workspace"
            )
        service_runtime, service_secrets, service_launcher_mode = _shared_control_plane(
            service_workspace
        )
        if args.launcher_mode is not None and launcher_mode != service_launcher_mode:
            raise SystemExit(
                "Campaign launcher mode must match the pinned service workspace: "
                f"{service_launcher_mode}"
            )
        launcher_mode = service_launcher_mode
        service_server = cast(dict[str, Any], service_runtime["server"])
        service_wiki = cast(dict[str, Any], service_runtime["gpu_wiki"])
        host = str(service_server["host"])
        port = int(service_server["port"])
        wiki_url = str(service_wiki["base_url"])
    else:
        host = str(args.runtime_host or runtime_policy["host"])
        port = int(args.runtime_port or runtime_policy["port"])
        wiki_url = str(args.wiki_url or runtime_policy["wiki_url"])
    worker_user = str(
        args.worker_user or os.environ.get("SUDO_USER") or pwd.getpwuid(os.getuid()).pw_name
    )
    workspace_name = "--".join(
        (
            _slug(operator_root.parent.name),
            _slug(operator),
            _slug(hardware_target),
            args.backend,
        )
    )
    default_workspace = root / "workspaces/production" / workspace_name
    workspace = (args.workspace or default_workspace).expanduser().resolve()
    attached_service_workspace = service_workspace if service_workspace != workspace else None
    seed_source = args.seed_source or selected_seed or "reference.py"
    seed = (operator_root / seed_source).resolve()
    if not seed.is_relative_to(operator_root) or seed.is_symlink() or not seed.is_file():
        raise SystemExit(f"seed source must be a regular file inside the operator: {seed}")

    dsl_seeds: dict[str, Path] = {}
    for raw in args.dsl_seed_source:
        name, separator, value = raw.partition("=")
        dsl_name = name.strip().lower()
        if not separator or dsl_name not in SUPPORTED_DSLS or not value.strip():
            raise SystemExit(
                "--dsl-seed-source must be DSL=PATH with DSL in "
                f"{', '.join(SUPPORTED_DSLS)}: {raw!r}"
            )
        if dsl_name in dsl_seeds:
            raise SystemExit(f"--dsl-seed-source repeats {dsl_name}")
        candidate = Path(value.strip()).expanduser()
        resolved = (candidate if candidate.is_absolute() else operator_root / candidate).resolve()
        if resolved.is_symlink() or not resolved.is_file():
            raise SystemExit(f"{dsl_name} seed source must be a regular file: {resolved}")
        dsl_seeds[dsl_name] = resolved
    seed_by_dsl = {dsl: dsl_seeds.get(dsl, seed) for dsl in SUPPORTED_DSLS}
    seed_text_by_dsl: dict[str, str] = {}
    for dsl, path in seed_by_dsl.items():
        text = path.read_text(encoding="utf-8")
        if not text.strip():
            raise SystemExit(f"{dsl} seed source is empty: {path}")
        seed_text_by_dsl[dsl] = text

    core_commit = _git_commit(
        root / "src/atrex-kernel-agent-core",
        require_clean=True,
        label="Core",
    )
    evolver_commit = _git_commit(
        root / "src/atrex-kernel-agent-evolver",
        require_clean=True,
        label="Evolver",
    )
    bench_commit = _git_commit(root / "third_party/atrex-bench")
    manifest_path = workspace / "production-manifest.json"
    requested_identity = {
        "schema_version": 2,
        "layout": "shared-runtime-per-dsl-campaign-workspaces",
        "production_policy_digest": hashlib.sha256(
            json.dumps(policy, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "kernel_root": str(operator_root),
        "seed_source": str(seed),
        "dsl_seed_sources": {
            dsl: {
                "path": str(seed_by_dsl[dsl]),
                "sha256": hashlib.sha256(seed_text_by_dsl[dsl].encode()).hexdigest(),
            }
            for dsl in SUPPORTED_DSLS
        },
        "operator": operator,
        "hardware_target": hardware_target,
        "backend": args.backend,
        "optimizer_model": args.optimizer_model,
        "evolver_model": args.evolver_model,
        "core_commit": core_commit,
        "evolver_commit": evolver_commit,
        "atrex_bench_commit": bench_commit,
        "service_workspace": (
            str(attached_service_workspace) if attached_service_workspace is not None else None
        ),
        "launcher_mode": launcher_mode,
    }
    if manifest_path.is_file():
        existing = _load_object(manifest_path, "production manifest")
        if existing.get("layout") != requested_identity["layout"]:
            raise SystemExit(
                "production workspace uses the retired shared-Campaign layout; "
                "move it aside or choose a new --workspace"
            )
        existing_ablation_path = workspace / "ablation.json"
        if existing_ablation_path.is_file():
            existing_ablation = _load_object(existing_ablation_path, "Ablation Plan")
            if existing_ablation != _ablation_plan(policy):
                raise SystemExit(
                    "production workspace Ablation Plan does not match the requested Arm "
                    "topology and 15-Attempt-per-Trajectory budget; choose a new --workspace"
                )
        mutable_checkout_fields = {"core_commit", "evolver_commit", "atrex_bench_commit"}
        # Manifests created before container mode implicitly used the strict
        # systemd/cgroup sandbox.
        existing.setdefault("launcher_mode", "sandbox")
        requested_inputs = {
            key: value
            for key, value in requested_identity.items()
            if key not in mutable_checkout_fields
        }
        existing_inputs = {
            key: value for key, value in existing.items() if key not in mutable_checkout_fields
        }
        if existing_inputs != requested_inputs:
            raise SystemExit(
                "production workspace is pinned to different inputs; choose a new --workspace"
            )
        required = (
            workspace / "runtime.json",
            workspace / "local-wiki.json",
            existing_ablation_path,
            *(workspace / "dsls" / dsl / "campaign.json" for dsl in SUPPORTED_DSLS),
        )
        if not all(path.is_file() for path in required):
            raise SystemExit("pinned production workspace is incomplete")
        if service_secrets is None:
            _ensure_runtime_secrets(workspace / "runtime.env")
        else:
            _write_text(workspace / "runtime.env", service_secrets)
        if args.workspace_output is not None:
            _write_text(args.workspace_output.resolve(), str(workspace) + "\n")
        print(f"Reusing pinned production workspace: {workspace}")
        return

    workspace.mkdir(parents=True, exist_ok=True)
    contract = _evaluation_contract(operator_root)
    for dsl in SUPPORTED_DSLS:
        dsl_workspace = workspace / "dsls" / dsl
        baseline = dsl_workspace / "inputs" / "baseline-kernel" / "kernel.py"
        _write_text(baseline, seed_text_by_dsl[dsl])
        evidence = dsl_workspace / "inputs" / "initial-evidence" / "README.md"
        _write_text(
            evidence,
            "\n".join(
                (
                    f"# Initial {dsl} lineage evidence",
                    "",
                    f"Operator: `{operator}`",
                    f"Hardware target: `{hardware_target}`",
                    f"Seed source: `{seed_by_dsl[dsl].name}`.",
                    "",
                    "No prior optimization experiments exist; bootstrap must create the first "
                    f"correct self-contained {dsl} implementation.",
                    "",
                )
            ),
        )
        _write_json(dsl_workspace / "evaluation-contract.json", contract)
        campaign = _campaign(
            workspace=dsl_workspace,
            dsl=dsl,
            operator_root=operator_root,
            operator=operator,
            hardware_target=hardware_target,
            backend=args.backend,
            core_commit=core_commit,
            evolver_commit=evolver_commit,
            policy=policy,
            optimizer_model=args.optimizer_model,
            evolver_model=args.evolver_model,
        )
        _write_json(dsl_workspace / "campaign.json", campaign)
        _write_json(
            dsl_workspace / "production-manifest.json",
            {**requested_identity, "dsl": dsl},
        )
    _write_json(workspace / "ablation.json", _ablation_plan(policy))
    host_home = os.environ.get("ATREX_SANDBOX_HOST_HOME", str(Path.home()))
    runtime = _runtime_config(
        root=root,
        workspace=workspace,
        backend=args.backend,
        hardware_target=hardware_target,
        policy=policy,
        host=host,
        port=port,
        wiki_url=wiki_url,
        worker_user=worker_user,
        host_home=host_home,
        launcher_mode=launcher_mode,
        evolver_commit=evolver_commit,
        bench_commit=bench_commit,
    )
    if service_runtime is not None:
        _attach_shared_control_plane(runtime, service_runtime)
    _write_json(workspace / "runtime.json", runtime)
    _write_json(workspace / "local-wiki.json", _local_wiki_config(root, workspace))
    if service_secrets is None:
        _ensure_runtime_secrets(workspace / "runtime.env")
    else:
        _write_text(workspace / "runtime.env", service_secrets)
    # Validate through the same strict parsers used by production entrypoints.
    from atrex_runtime.bootstrap import CampaignSpecV3
    from atrex_runtime.config import RuntimeSettings

    RuntimeSettings.from_file(workspace / "runtime.json")
    for dsl in SUPPORTED_DSLS:
        parsed_campaign = CampaignSpecV3.from_file(workspace / "dsls" / dsl / "campaign.json")
        selected = tuple(value.value for value in parsed_campaign.selected_dsls())
        if selected != (dsl,):
            raise RuntimeError(f"{dsl} Campaign selected unexpected DSLs: {selected}")
    _write_json(manifest_path, requested_identity)
    if args.workspace_output is not None:
        _write_text(args.workspace_output.resolve(), str(workspace) + "\n")
    print(f"Production workspace: {workspace}")
    print(f"Atrex-Bench kernel: {operator_root}")
    print(f"Backend: {args.backend}")
    print(f"Worker launcher: {launcher_mode}")
    print(f"DSL Campaign workspaces: {', '.join(SUPPORTED_DSLS)}")
    print("Production content policy gate: enabled")
    schedule = policy["schedule"]
    print(
        f"Schedule: bootstrap, then {schedule['attempts_per_trajectory']} "
        f"Attempts/Trajectory/Epoch; {schedule['trajectories_per_branch']} "
        f"Trajectories/Branch; {schedule['challenger_count']} Challenger(s) "
        f"from Epoch {schedule['challenger_start_epoch']}"
    )
    print(f"Runtime config: {workspace / 'runtime.json'}")
    for dsl in SUPPORTED_DSLS:
        print(f"{dsl} Campaign config: {workspace / 'dsls' / dsl / 'campaign.json'}")


if __name__ == "__main__":
    main()
