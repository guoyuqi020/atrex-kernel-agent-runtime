"""Live Agent Backend connectivity checks through the production Sandbox boundary."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import pwd
import shutil
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Literal, Protocol, cast
from uuid import uuid4

from ..config import (
    BackendCredentialSettings,
    BwrapSandboxSettings,
    CgroupResourceSettings,
)
from ..workers.launcher import BackendCredentialMounts, BwrapSandboxLauncher

BackendName = Literal["claude", "codex", "qodercli"]
Status = Literal["passed", "skipped", "failed"]

_BACKENDS: tuple[BackendName, ...] = ("claude", "codex", "qodercli")
_COMMON_ENVIRONMENT = (
    "NODE_EXTRA_CA_CERTS",
    "SSL_CERT_DIR",
    "SSL_CERT_FILE",
)
_BACKEND_ENVIRONMENT: dict[BackendName, tuple[str, ...]] = {
    "claude": (
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "ANTHROPIC_BASE_URL",
        "ANTHROPIC_MODEL",
        "CLAUDE_CONFIG_DIR",
        "CLAUDE_CODE_USE_BEDROCK",
        "CLAUDE_CODE_USE_VERTEX",
    ),
    "codex": (
        "CODEX_HOME",
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
    ),
    "qodercli": ("QODER_PERSONAL_ACCESS_TOKEN",),
}
_CREDENTIAL_NAMES: dict[BackendName, tuple[str, ...]] = {
    "claude": (".claude", ".claude.json"),
    "codex": (".codex",),
    "qodercli": (".qoder", ".qodersec"),
}
_CODEX_SHARED_ENTRIES = (
    "auth.json",
    "config.toml",
    "skills",
    "plugins",
    "hooks.json",
    "models_cache.json",
    "vendor_imports",
    "mcp-oauth-locks",
)
_CODEX_WRITABLE_COPIES = ("installation_id",)


class BackendAdapter(Protocol):
    """Core-owned command construction required by this acceptance check."""

    def build_command(
        self,
        prompt: str,
        session_id: str,
        reasoning_effort: str,
        settings: str,
        model: str | None = None,
    ) -> list[str]: ...


class BackendRegistry(Protocol):
    def create(self, runtime_id: str) -> BackendAdapter: ...


@dataclass(frozen=True, slots=True)
class ConnectivityResult:
    backend: BackendName
    status: Status
    detail: str
    executable: str | None = None
    exit_code: int | None = None
    duration_seconds: float | None = None


@dataclass(frozen=True, slots=True)
class AcceptanceOptions:
    backends: tuple[BackendName, ...]
    core_root: Path
    temporary_parent: Path
    timeout_seconds: float
    executable_overrides: Mapping[BackendName, Path]
    credential_overrides: Mapping[BackendName, tuple[Path, ...]]
    model_overrides: Mapping[BackendName, str]
    settings_overrides: Mapping[BackendName, str]
    require_all: bool
    trace_filesystem: bool = False
    reasoning_effort: str = "low"
    selective_credentials: bool = False
    prompt: str = "Reply with exactly ATREX_BACKEND_CONNECTIVITY_OK. Do not use tools."


def _worker_identity() -> tuple[pwd.struct_passwd, Path]:
    user_name = os.environ.get("SUDO_USER") or pwd.getpwuid(os.getuid()).pw_name
    worker = pwd.getpwnam(user_name)
    if worker.pw_uid == 0:
        raise RuntimeError("Backend acceptance must target a non-root Worker user")
    return worker, Path(worker.pw_dir).resolve()


def _require_executable(name: str) -> Path:
    value = shutil.which(name)
    if value is None:
        raise RuntimeError(f"required acceptance executable is missing: {name}")
    return Path(value).absolute()


def _host_resolv_conf() -> Path:
    routed = Path("/run/systemd/resolve/resolv.conf")
    if routed.is_file():
        return routed
    resolved = Path("/etc/resolv.conf").resolve()
    if not resolved.is_file():
        raise RuntimeError("host resolver configuration is unavailable")
    return resolved


def _load_registry(core_root: Path) -> BackendRegistry:
    source = core_root.resolve() / "src"
    if not (source / "backends/adapter.py").is_file():
        raise RuntimeError(f"Core Backend Adapter is unavailable: {source}")
    source_value = str(source)
    if source_value not in sys.path:
        sys.path.insert(0, source_value)
    module = importlib.import_module("backends.adapter")
    return cast(BackendRegistry, module.DEFAULT_BACKEND_REGISTRY)


def _parse_assignment(raw: str, *, label: str) -> tuple[BackendName, str]:
    backend, separator, value = raw.partition("=")
    if not separator or backend not in _BACKENDS or not value:
        raise argparse.ArgumentTypeError(
            f"{label} must be BACKEND=VALUE for {', '.join(_BACKENDS)}"
        )
    return backend, value


def _assignment_map(values: Sequence[str], *, label: str) -> dict[BackendName, str]:
    parsed: dict[BackendName, str] = {}
    for raw in values:
        backend, value = _parse_assignment(raw, label=label)
        if backend in parsed:
            raise ValueError(f"duplicate {label} override for {backend}")
        parsed[backend] = value
    return parsed


def _credential_map(values: Sequence[str]) -> dict[BackendName, tuple[Path, ...]]:
    parsed: dict[BackendName, list[Path]] = {}
    for raw in values:
        backend, value = _parse_assignment(raw, label="credential")
        parsed.setdefault(backend, []).append(Path(value).expanduser().resolve())
    return {backend: tuple(paths) for backend, paths in parsed.items()}


def _environment_for(backend: BackendName, home: Path) -> dict[str, str]:
    names = (*_COMMON_ENVIRONMENT, *_BACKEND_ENVIRONMENT[backend])
    environment = {name: os.environ[name] for name in names if os.environ.get(name)}
    if backend == "codex":
        environment["CODEX_HOME"] = str(
            Path(environment.get("CODEX_HOME", home / ".codex")).expanduser().resolve()
        )
    return environment


def _credential_paths(
    backend: BackendName,
    home: Path,
    environment: Mapping[str, str],
    overrides: Mapping[BackendName, tuple[Path, ...]],
) -> tuple[Path, ...]:
    configured = overrides.get(backend)
    if configured is not None:
        paths = configured
    elif backend == "codex":
        paths = (Path(environment.get("CODEX_HOME", home / ".codex")),)
    elif backend == "claude" and environment.get("CLAUDE_CONFIG_DIR"):
        paths = (Path(environment["CLAUDE_CONFIG_DIR"]), home / ".claude.json")
    else:
        paths = tuple(home / name for name in _CREDENTIAL_NAMES[backend])
    return tuple(path.resolve() for path in paths if path.exists() and not path.is_symlink())


def _executable_bind_paths(executable: Path, home: Path) -> tuple[Path, ...]:
    """Expose one user-local CLI installation hidden by the private Home tmpfs."""
    absolute = executable.absolute()
    try:
        relative = absolute.relative_to(home)
    except ValueError:
        return ()
    if not relative.parts:
        raise ValueError("Backend executable cannot be the Worker home directory")
    installation_root = home / relative.parts[0]
    if installation_root.is_symlink() or not installation_root.is_dir():
        raise ValueError("User-local Backend installation root must be a real directory")
    return (installation_root,)


def _certificate_bind_paths(environment: Mapping[str, str]) -> tuple[Path, ...]:
    """Expose explicitly selected CA files/directories when private Home masks them."""
    paths: list[Path] = []
    for name in _COMMON_ENVIRONMENT:
        value = environment.get(name)
        if not value:
            continue
        path = Path(value).expanduser().absolute()
        if path.exists() and not path.is_symlink():
            paths.append(path)
    return tuple(dict.fromkeys(paths))


def _has_credentials(
    backend: BackendName,
    home: Path,
    environment: Mapping[str, str],
    credential_paths: tuple[Path, ...],
) -> bool:
    if backend == "claude":
        return bool(
            environment.get("ANTHROPIC_AUTH_TOKEN")
            or environment.get("ANTHROPIC_API_KEY")
            or credential_paths
            or (home / ".claude/.credentials.json").is_file()
            or (home / ".claude.json").is_file()
        )
    if backend == "codex":
        codex_home = Path(environment.get("CODEX_HOME", home / ".codex"))
        return bool(
            environment.get("OPENAI_API_KEY")
            or (codex_home / "auth.json").is_file()
            or credential_paths
        )
    return bool(
        environment.get("QODER_PERSONAL_ACCESS_TOKEN")
        or credential_paths
        or (home / ".qoder").is_dir()
        or (home / ".qodersec").exists()
    )


def _terminal_success(backend: BackendName, stdout: str) -> bool:
    for line in stdout.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        if backend == "codex" and event.get("type") == "turn.completed":
            return True
        if backend in {"claude", "qodercli"} and event.get("type") == "result":
            return event.get("is_error") is not True and event.get("subtype") != "error"
    return False


def _redacted_diagnostic(
    stdout: str,
    stderr: str,
    environment: Mapping[str, str],
    *,
    limit: int = 4000,
) -> str:
    parts = []
    if stderr.strip():
        parts.append(f"stderr: {stderr.strip()}")
    if stdout.strip():
        parts.append(f"stdout: {stdout.strip()}")
    value = " ".join(parts)[-limit:]
    for secret in environment.values():
        if len(secret) >= 8:
            value = value.replace(secret, "<redacted>")
    return value.replace("\n", " ")


def _filesystem_failure_summary(path: Path, *, limit: int = 20) -> str:
    if not path.is_file():
        return ""
    failures = [
        line.strip()
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines()
        if any(marker in line for marker in ("EROFS", "EACCES", "ENOENT"))
    ]
    return " | ".join(failures[-limit:])[-4000:]


def _prepare_backend_home(
    backend: BackendName,
    workspace: Path,
    environment: dict[str, str],
) -> None:
    """Mirror Core's writable per-Session Codex Home over read-only login state."""
    if backend != "codex":
        return
    source = Path(environment["CODEX_HOME"])
    isolated = workspace / "scratch/codex-home"
    isolated.mkdir(mode=0o700)
    for name in _CODEX_SHARED_ENTRIES:
        entry = source / name
        if entry.exists():
            (isolated / name).symlink_to(entry, target_is_directory=entry.is_dir())
    for name in _CODEX_WRITABLE_COPIES:
        entry = source / name
        if entry.is_file():
            shutil.copyfile(entry, isolated / name)
    environment["CODEX_HOME"] = str(isolated)


def _chown_tree(root: Path, uid: int, gid: int) -> None:
    os.chown(root, uid, gid, follow_symlinks=False)
    for path in root.rglob("*"):
        os.chown(path, uid, gid, follow_symlinks=False)


def _write_probe_command(backend: BackendName) -> tuple[str, ...]:
    targets = '"$HOME" "$PWD" "$PWD/scratch"'
    if backend == "codex":
        targets += ' "$CODEX_HOME"'
    script = (
        f"for path in {targets}; do "
        'test -w "$path" || { echo "not-writable:$path" >&2; exit 73; }; '
        'probe="$path/.atrex-connectivity-write-probe"; '
        'printf ok > "$probe" || exit 73; rm -f "$probe" || exit 73; done'
    )
    return ("/bin/sh", "-c", script)


def _run_backend(
    backend: BackendName,
    options: AcceptanceOptions,
    registry: BackendRegistry,
    worker: pwd.struct_passwd,
    home: Path,
) -> ConnectivityResult:
    environment = _environment_for(backend, home)
    override = options.executable_overrides.get(backend)
    executable_value = str(override) if override is not None else shutil.which(backend)
    if executable_value is None:
        return ConnectivityResult(backend, "skipped", "Backend executable is unavailable")
    executable = Path(executable_value).expanduser().absolute()
    if not executable.is_file() or not os.access(executable, os.X_OK):
        return ConnectivityResult(
            backend,
            "skipped",
            "Backend executable is not an executable file",
            str(executable),
        )
    credentials = _credential_paths(
        backend,
        home,
        environment,
        options.credential_overrides,
    )
    if not _has_credentials(backend, home, environment, credentials):
        return ConnectivityResult(
            backend,
            "skipped",
            "No login state or supported authentication environment is available",
            str(executable),
        )

    workspace_root = options.temporary_parent / f"atrex-backend-{backend}-{uuid4().hex}"
    workspace = workspace_root / "attempts/connectivity/run"
    workspace.mkdir(parents=True, mode=0o700)
    (workspace / "scratch").mkdir(mode=0o700)
    _prepare_backend_home(backend, workspace, environment)
    if os.geteuid() == 0:
        _chown_tree(workspace_root, worker.pw_uid, worker.pw_gid)

    adapter = registry.create(backend)
    command = adapter.build_command(
        options.prompt,
        str(uuid4()),
        options.reasoning_effort,
        options.settings_overrides.get(backend, ""),
        options.model_overrides.get(backend),
    )
    if backend == "codex":
        command.insert(command.index("--json") + 1, "--ephemeral")
    command[0] = str(executable)
    trace_path = workspace / "scratch/backend-files.strace"
    if options.trace_filesystem:
        command = [
            str(_require_executable("strace")),
            "-f",
            "-e",
            "trace=file",
            "-o",
            str(trace_path),
            *command,
        ]
    environment["PATH"] = os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin")
    environment["TERM"] = "dumb"
    credential_projection = None
    if options.selective_credentials:
        environment["ATREX_AGENT_BACKEND"] = backend
        environment["HOME"] = str(workspace / "sessions/agent-home")
        credential_projection = BackendCredentialMounts.from_environment(
            BackendCredentialSettings(
                host_home=home,
                development_bwrap_executable=_require_executable("bwrap"),
            ),
            {"HOME": str(home), "PATH": environment["PATH"]},
        )

    launcher = BwrapSandboxLauncher(
        _require_executable("env"),
        BwrapSandboxSettings(
            bwrap_executable=_require_executable("bwrap"),
            systemd_run_executable=_require_executable("systemd-run"),
            systemd_user=False,
            worker_user=worker.pw_name,
            sandbox_home=PurePosixPath(
                "/home/agent" if options.selective_credentials else home.as_posix()
            ),
            workspace_mount=PurePosixPath(
                "/home/agent/workspace"
                if options.selective_credentials
                else (home / "workspace").as_posix()
            ),
            resolv_conf=_host_resolv_conf(),
            read_only_bind_paths=tuple(
                dict.fromkeys(
                    (
                        *(() if options.selective_credentials else credentials),
                        *_executable_bind_paths(executable, home),
                        *_certificate_bind_paths(environment),
                    )
                )
            ),
            resources=CgroupResourceSettings(
                memory_max_bytes=2 * 1024 * 1024 * 1024,
                memory_swap_max_bytes=0,
                cpu_quota_percent=200,
                tasks_max=128,
            ),
        ),
        (workspace_root / "attempts",),
        credentials=credential_projection,
    )
    started = time.monotonic()
    try:
        launcher.check_host()
        probe = subprocess.run(
            launcher.wrap(
                _write_probe_command(backend),
                workspace=workspace,
                environment=environment,
            ),
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
            start_new_session=True,
        )
        if probe.returncode != 0:
            diagnostic = _redacted_diagnostic(probe.stdout, probe.stderr, environment)
            return ConnectivityResult(
                backend,
                "failed",
                f"Sandbox writable-path probe failed: {diagnostic}",
                str(executable),
                probe.returncode,
                time.monotonic() - started,
            )
        argv = launcher.wrap(tuple(command), workspace=workspace, environment=environment)
        completed = subprocess.run(
            argv,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=options.timeout_seconds,
            check=False,
            start_new_session=True,
        )
        duration = time.monotonic() - started
        if completed.returncode == 0 and _terminal_success(backend, completed.stdout):
            return ConnectivityResult(
                backend,
                "passed",
                "Sandboxed non-interactive model request completed",
                str(executable),
                completed.returncode,
                duration,
            )
        diagnostic = _redacted_diagnostic(
            completed.stdout,
            completed.stderr,
            environment,
        )
        detail = "Backend request failed or returned no terminal success event"
        if diagnostic:
            detail += f": {diagnostic}"
        filesystem = _filesystem_failure_summary(trace_path)
        if filesystem:
            detail += f"; filesystem: {filesystem}"
        return ConnectivityResult(
            backend,
            "failed",
            detail,
            str(executable),
            completed.returncode,
            duration,
        )
    except subprocess.TimeoutExpired:
        return ConnectivityResult(
            backend,
            "failed",
            f"Backend request exceeded {options.timeout_seconds:g} seconds",
            str(executable),
            duration_seconds=time.monotonic() - started,
        )
    except Exception as error:
        return ConnectivityResult(
            backend,
            "failed",
            f"{type(error).__name__}: {error}",
            str(executable),
            duration_seconds=time.monotonic() - started,
        )
    finally:
        shutil.rmtree(workspace_root, ignore_errors=True)


def run_acceptance(options: AcceptanceOptions) -> tuple[ConnectivityResult, ...]:
    """Run selected Backends independently through a fresh production Sandbox."""
    if platform.system() != "Linux":
        raise RuntimeError("Backend connectivity acceptance must run on Linux")
    options.temporary_parent.mkdir(parents=True, exist_ok=True)
    worker, home = _worker_identity()
    registry = _load_registry(options.core_root)
    return tuple(
        _run_backend(backend, options, registry, worker, home) for backend in options.backends
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend",
        action="append",
        choices=_BACKENDS,
        dest="backends",
        help="Backend to check; repeat as needed (default: all three)",
    )
    parser.add_argument(
        "--core-root",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "atrex-kernel-agent-core",
    )
    parser.add_argument("--temporary-parent", type=Path, default=Path("/var/tmp"))
    parser.add_argument("--timeout-seconds", type=float, default=180)
    parser.add_argument(
        "--reasoning-effort",
        default="low",
        help="Backend reasoning effort used by the live model request (default: low)",
    )
    parser.add_argument("--executable", action="append", default=[], metavar="BACKEND=PATH")
    parser.add_argument("--credential", action="append", default=[], metavar="BACKEND=PATH")
    parser.add_argument("--model", action="append", default=[], metavar="BACKEND=MODEL")
    parser.add_argument("--settings", action="append", default=[], metavar="BACKEND=JSON")
    parser.add_argument(
        "--require-all",
        action="store_true",
        help="Treat missing executable or credentials as failure",
    )
    parser.add_argument("--json", action="store_true", help="Emit one machine-readable summary")
    parser.add_argument(
        "--trace-filesystem",
        action="store_true",
        help="Trace file syscalls and summarize failed paths (diagnostic only)",
    )
    parser.add_argument(
        "--selective-credentials",
        action="store_true",
        help="Use the same isolated writable Home and selective login-state projection as Workers",
    )
    parser.add_argument(
        "--prompt-file",
        type=Path,
        help="Use a UTF-8 diagnostic Prompt from a file instead of the minimal probe",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.timeout_seconds <= 0:
        raise ValueError("--timeout-seconds must be positive")
    executable_values = _assignment_map(arguments.executable, label="executable")
    prompt = "Reply with exactly ATREX_BACKEND_CONNECTIVITY_OK. Do not use tools."
    if arguments.prompt_file is not None:
        prompt_path = arguments.prompt_file.expanduser().resolve()
        if prompt_path.stat().st_size > 1024 * 1024:
            raise ValueError("--prompt-file exceeds 1 MiB")
        prompt = prompt_path.read_text(encoding="utf-8")
        if not prompt.strip():
            raise ValueError("--prompt-file cannot be empty")
    options = AcceptanceOptions(
        backends=tuple(arguments.backends or _BACKENDS),
        core_root=arguments.core_root,
        temporary_parent=arguments.temporary_parent.resolve(),
        timeout_seconds=arguments.timeout_seconds,
        executable_overrides={
            backend: Path(value).expanduser().absolute()
            for backend, value in executable_values.items()
        },
        credential_overrides=_credential_map(arguments.credential),
        model_overrides=_assignment_map(arguments.model, label="model"),
        settings_overrides=_assignment_map(arguments.settings, label="settings"),
        require_all=arguments.require_all,
        trace_filesystem=arguments.trace_filesystem,
        reasoning_effort=arguments.reasoning_effort,
        selective_credentials=arguments.selective_credentials,
        prompt=prompt,
    )
    results = run_acceptance(options)
    if arguments.json:
        print(json.dumps({"results": [asdict(result) for result in results]}, sort_keys=True))
    else:
        for result in results:
            duration = (
                "" if result.duration_seconds is None else f" ({result.duration_seconds:.1f}s)"
            )
            print(f"{result.backend}: {result.status}{duration} - {result.detail}")
    failed = any(result.status == "failed" for result in results)
    skipped = any(result.status == "skipped" for result in results)
    return 1 if failed or (options.require_all and skipped) else 0


__all__ = [
    "AcceptanceOptions",
    "ConnectivityResult",
    "main",
    "run_acceptance",
]
