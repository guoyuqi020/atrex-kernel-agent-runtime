"""Explicit-environment worker process launch policy."""

from __future__ import annotations

import fcntl
import os
import platform
import pwd
import re
import shutil
import stat
import subprocess
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
from uuid import uuid4

from ..config import BackendCredentialSettings, BwrapSandboxSettings

_ENVIRONMENT_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_PRIVATE_PATH_ENVIRONMENT_KEYS = frozenset(
    {
        "ATREX_EVALUATION_CONTRACT_PATH",
        "ATREX_PRIVATE_INPUTS",
        "ATREX_PRIVATE_REFERENCE_DIR",
    }
)

_SUPPORTED_BACKENDS = frozenset({"claude", "codex", "qodercli", "pi"})
_VIRTIOFS_OWNERSHIP_SETTLE_ATTEMPTS = 31
_VIRTIOFS_OWNERSHIP_SETTLE_SECONDS = 1.0
_DEV_SHELL_BACKENDS_ENV = "ATREX_DEV_SHELL_BACKENDS"
_CODEX_CA_BUNDLE_CANDIDATES = (
    Path("/etc/ssl/certs/ca-certificates.crt"),
    Path("/etc/pki/tls/certs/ca-bundle.crt"),
    Path("/etc/ssl/ca-bundle.pem"),
)
_CLAUDE_READ_ONLY_HOME_SUBPATHS = (
    Path(".claude/.credentials.json"),
    Path(".claude/plugins"),
)
_CLAUDE_WRITABLE_HOME_SUBPATHS = tuple(
    Path(".claude") / name for name in ("backups", "cache", "projects", "sessions")
)
_CLAUDE_SESSION_STATE_FILES = (
    Path(".claude.json"),
    Path(".claude/settings.json"),
)
_CODEX_READ_ONLY_HOME_SUBPATHS = (
    Path(".codex/auth.json"),
    Path(".codex/skills"),
)
_CODEX_WRITABLE_HOME_SUBPATHS = tuple(
    Path(".codex") / name
    for name in ("cache", "sessions", "shell_snapshots", "thread-writer-locks", "tmp")
)
_CODEX_SESSION_STATE_FILES = tuple(
    Path(".codex") / name
    for name in (".sandbox_migration", "config.toml", "installation_id", "models_cache.json")
)
_QODER_WRITABLE_HOME_SUBPATHS = (
    *(
        Path(".qoder") / name
        for name in (
            "tasks",
            "projects",
            "logs",
            "tmp",
            "cache",
            ".cache",
            ".codebase-status",
            "session-env",
            "shell-snapshots",
            "external-commands/locks",
        )
    ),
    Path(".qodersec/logs"),
)
_QODER_SESSION_STATE_FILES = tuple(
    Path(".qoder") / name
    for name in ("settings.json", "state.json", "installation_id", ".last-cleanup")
)
_QODER_STATE_STAGING_ROOT = Path(".atrex-provider-state/qoder")


@dataclass(frozen=True, slots=True)
class BackendCredentialMount:
    """One host login-state entry projected into a Session-specific Home."""

    source: Path
    home_relative: Path


@dataclass(frozen=True, slots=True)
class BackendCredentialMounts:
    """Resolve coding-agent login state without copying it into Agent artifacts."""

    settings: BackendCredentialSettings
    environment: Mapping[str, str]

    @classmethod
    def from_environment(
        cls,
        settings: BackendCredentialSettings,
        environment: Mapping[str, str],
    ) -> BackendCredentialMounts | None:
        if not settings.enabled:
            return None
        raw_home = settings.host_home or (
            Path(environment["HOME"]) if environment.get("HOME") else None
        )
        if raw_home is None:
            return None
        expanded_home = raw_home.expanduser()
        if expanded_home.is_symlink():
            raise RuntimeError(f"Backend credential Home cannot be a symlink: {expanded_home}")
        home = expanded_home.resolve()
        if not home.is_dir():
            raise RuntimeError(f"Backend credential Home is unavailable: {home}")
        ambient = {
            key: value
            for key in (
                "PATH",
                "CLAUDE_CONFIG_DIR",
                "CODEX_HOME",
                "PI_CODING_AGENT_DIR",
            )
            if (value := environment.get(key))
        }
        ambient["HOME"] = str(home)
        return cls(settings, ambient)

    @property
    def host_home(self) -> Path:
        return Path(self.environment["HOME"])

    def mounts_for(self, backend: str) -> tuple[BackendCredentialMount, ...]:
        if backend not in _SUPPORTED_BACKENDS:
            return ()
        home = self.host_home
        candidates: tuple[tuple[Path, Path], ...]
        if backend == "claude":
            candidates = tuple(
                (home / relative, relative) for relative in _CLAUDE_READ_ONLY_HOME_SUBPATHS
            )
        elif backend == "codex":
            candidates = tuple(
                (home / relative, relative) for relative in _CODEX_READ_ONLY_HOME_SUBPATHS
            )
        elif backend == "qodercli":
            candidates = (
                (home / ".qoder", Path(".qoder")),
                (home / ".qodersec", Path(".qodersec")),
            )
        else:
            configured = self.environment.get("PI_CODING_AGENT_DIR")
            directory = Path(configured).expanduser() if configured else home / ".pi/agent"
            candidates = ((directory, Path(".pi/agent")),)
        mounts: list[BackendCredentialMount] = []
        for raw_source, relative in candidates:
            expanded_source = raw_source.expanduser()
            if not expanded_source.exists():
                continue
            if expanded_source.is_symlink():
                raise RuntimeError(
                    f"Backend credential source cannot be a symlink: {expanded_source}"
                )
            source = expanded_source.resolve()
            mounts.append(BackendCredentialMount(source, relative))
        return tuple(mounts)

    def installation_roots_for(self, backend: str) -> tuple[Path, ...]:
        """Restore a user-local CLI hidden by the strict Sandbox Home tmpfs."""
        if backend not in _SUPPORTED_BACKENDS:
            return ()
        executable = shutil.which(backend, path=self.environment.get("PATH"))
        if executable is None:
            return ()
        absolute = Path(executable).absolute()
        try:
            relative = absolute.relative_to(self.host_home)
        except ValueError:
            return ()
        if not relative.parts:
            return ()
        root = self.host_home / relative.parts[0]
        if root.is_dir() and not root.is_symlink():
            return (root,)
        return ()

    def writable_home_subpaths_for(self, backend: str) -> tuple[Path, ...]:
        """Return Provider runtime-state directories overlaid per Session."""
        if backend == "claude":
            return _CLAUDE_WRITABLE_HOME_SUBPATHS
        if backend == "codex":
            return _CODEX_WRITABLE_HOME_SUBPATHS
        if backend != "qodercli":
            return ()
        return tuple(
            relative
            for relative in _QODER_WRITABLE_HOME_SUBPATHS
            if (self.host_home / relative).is_dir()
        )

    def requires_writable_backend_home(self, backend: str) -> bool:
        """Return whether Provider state needs a private writable Session root."""
        roots = {
            "claude": self.host_home / ".claude",
            "codex": self.host_home / ".codex",
            "qodercli": self.host_home / ".qoder",
        }
        root = roots.get(backend)
        return root is not None and root.is_dir()

    def prepare_writable_backend_home(self, backend: str, session_home: Path) -> None:
        """Create login-free writable Provider state with read-only credentials."""
        if not self.requires_writable_backend_home(backend):
            return
        roots = {
            "claude": Path(".claude"),
            "codex": Path(".codex"),
            "qodercli": Path(".qoder"),
        }
        state_files = {
            "claude": _CLAUDE_SESSION_STATE_FILES,
            "codex": _CODEX_SESSION_STATE_FILES,
            "qodercli": _QODER_SESSION_STATE_FILES,
        }
        relative_root = roots[backend]
        destination_root = session_home / relative_root
        destination_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        for relative in state_files[backend]:
            source = self.host_home / relative
            if not source.exists():
                continue
            if source.is_symlink() or not source.is_file():
                raise RuntimeError(f"Provider Session state source is unsafe: {source}")
            if source.stat().st_size > 8 * 1024 * 1024:
                raise RuntimeError(f"Provider Session state source is too large: {source}")
            destination = (
                session_home / _QODER_STATE_STAGING_ROOT / relative.relative_to(".qoder")
                if backend == "qodercli"
                else session_home / relative
            )
            destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            shutil.copyfile(source, destination)
            destination.chmod(0o600)
        for relative in self.writable_home_subpaths_for(backend):
            (session_home / relative).mkdir(parents=True, exist_ok=True, mode=0o700)

    def qoder_state_overlays(
        self, session_home: Path
    ) -> tuple[tuple[Path, Path], ...]:
        """Return private writable Qoder root-state files overlaid on its read-only Home."""
        overlays: list[tuple[Path, Path]] = []
        for relative in _QODER_SESSION_STATE_FILES:
            staged = session_home / _QODER_STATE_STAGING_ROOT / relative.relative_to(".qoder")
            if staged.is_file():
                overlays.append((staged, session_home / relative))
        return tuple(overlays)

    @staticmethod
    def update_environment(
        environment: dict[str, str],
        backend: str,
        session_home: Path,
    ) -> None:
        if backend == "claude":
            environment["CLAUDE_CONFIG_DIR"] = str(session_home / ".claude")
        elif backend == "codex":
            environment["CODEX_HOME"] = str(session_home / ".codex")
        elif backend == "pi":
            environment["PI_CODING_AGENT_DIR"] = str(session_home / ".pi/agent")


def _projected_backends(
    environment: Mapping[str, str],
    *,
    interactive: bool,
) -> tuple[str, ...]:
    """Resolve the selected Backend plus explicitly requested dev-shell peers."""
    selected = environment.get("ATREX_AGENT_BACKEND", "")
    values = [selected] if selected else []
    raw_peers = environment.get(_DEV_SHELL_BACKENDS_ENV, "")
    if raw_peers:
        if not interactive:
            raise ValueError(f"{_DEV_SHELL_BACKENDS_ENV} is restricted to interactive shells")
        peers = raw_peers.split(",")
        if any(not peer for peer in peers):
            raise ValueError(f"{_DEV_SHELL_BACKENDS_ENV} contains an empty Backend")
        values.extend(peers)
    invalid = sorted(set(values).difference(_SUPPORTED_BACKENDS))
    if invalid:
        raise ValueError(f"unsupported Backend credential projection: {invalid}")
    return tuple(dict.fromkeys(values))


def _credential_mounts_for(
    credentials: BackendCredentialMounts,
    backends: tuple[str, ...],
) -> tuple[BackendCredentialMount, ...]:
    """Merge non-conflicting credential mounts for one interactive environment."""
    by_destination: dict[Path, BackendCredentialMount] = {}
    for backend in backends:
        for mount in credentials.mounts_for(backend):
            existing = by_destination.get(mount.home_relative)
            if existing is not None and existing.source != mount.source:
                raise RuntimeError(
                    f"Backend credential destination conflicts: {mount.home_relative}"
                )
            by_destination[mount.home_relative] = mount
    return tuple(by_destination.values())


def _writable_home_subpaths_for(
    credentials: BackendCredentialMounts,
    backends: tuple[str, ...],
) -> tuple[Path, ...]:
    return tuple(
        dict.fromkeys(
            relative
            for backend in backends
            for relative in credentials.writable_home_subpaths_for(backend)
        )
    )


def _inject_codex_ca_bundle(
    environment: dict[str, str],
    backends: tuple[str, ...],
) -> None:
    """Make the system trust store explicit for Codex HTTP/MCP clients."""
    if "codex" not in backends or environment.get("SSL_CERT_FILE"):
        return
    for candidate in _CODEX_CA_BUNDLE_CANDIDATES:
        if candidate.is_file():
            environment["SSL_CERT_FILE"] = str(candidate)
            return


def validate_worker_environment(environment: Mapping[str, str]) -> None:
    """Reject environment entries that cannot be passed as exact process values."""
    leaked = sorted(_PRIVATE_PATH_ENVIRONMENT_KEYS.intersection(environment))
    if leaked:
        raise ValueError(f"private evaluator paths cannot enter Worker environments: {leaked}")
    for key, value in environment.items():
        if _ENVIRONMENT_KEY.fullmatch(key) is None:
            raise ValueError(f"invalid environment key: {key!r}")
        if "\x00" in value:
            raise ValueError(f"environment value contains NUL: {key}")


class WorkerLauncher(Protocol):
    """Build argv for one worker without claiming OS-level isolation."""

    def wrap(
        self,
        runtime_argv: tuple[str, ...],
        *,
        workspace: Path,
        environment: Mapping[str, str],
        interactive: bool = False,
    ) -> tuple[str, ...]:
        """Return an argv with a clean, explicit environment."""
        ...


@dataclass(frozen=True, slots=True)
class CleanEnvironmentLauncher:
    """Clear ambient environment variables before starting a worker process."""

    env_executable: Path
    credentials: BackendCredentialMounts | None = None

    def wrap(
        self,
        runtime_argv: tuple[str, ...],
        *,
        workspace: Path,
        environment: Mapping[str, str],
        interactive: bool = False,
    ) -> tuple[str, ...]:
        """Prefix the worker command with ``env -i`` and exact assignments."""
        if not runtime_argv:
            raise ValueError("runtime argv cannot be empty")
        if not self.env_executable.is_absolute():
            raise ValueError("env executable must be absolute")
        validate_worker_environment(environment)
        exact_environment = dict(environment)
        mounts: tuple[BackendCredentialMount, ...] = ()
        backends = _projected_backends(exact_environment, interactive=interactive)
        _inject_codex_ca_bundle(exact_environment, backends)
        credentials = self.credentials
        if credentials is not None:
            mounts = _credential_mounts_for(credentials, backends)
        writable_backends = (
            ()
            if credentials is None
            else tuple(
                backend
                for backend in backends
                if credentials.requires_writable_backend_home(backend)
            )
        )
        writable_projection = (
            credentials is not None and bool(writable_backends)
        )
        if not mounts and not writable_projection:
            assignments = tuple(
                f"{key}={value}" for key, value in sorted(exact_environment.items())
            )
            return (str(self.env_executable), "-i", *assignments, *runtime_argv)

        if platform.system() != "Linux":
            raise RuntimeError("Read-only Backend credentials require Linux bubblewrap")
        assert credentials is not None
        bwrap_executable = credentials.settings.development_bwrap_executable
        if (
            not bwrap_executable.is_absolute()
            or not bwrap_executable.is_file()
            or not os.access(bwrap_executable, os.X_OK)
        ):
            raise RuntimeError(f"Backend credential bubblewrap is unavailable: {bwrap_executable}")
        workspace = workspace.resolve()
        home_value = exact_environment.get("HOME")
        if home_value is None:
            raise ValueError("Backend credential mounts require an isolated HOME")
        session_home = Path(home_value).resolve()
        if not session_home.is_relative_to(workspace):
            raise ValueError("Backend credential HOME must be inside the Worker workspace")
        for backend in writable_backends:
            credentials.prepare_writable_backend_home(backend, session_home)
        qoder_state_overlays = (
            credentials.qoder_state_overlays(session_home)
            if "qodercli" in writable_backends
            else ()
        )
        for backend in backends:
            credentials.update_environment(exact_environment, backend, session_home)
        assignments = tuple(f"{key}={value}" for key, value in sorted(exact_environment.items()))
        bwrap = [
            str(bwrap_executable),
            "--die-with-parent",
            "--new-session",
            "--ro-bind",
            "/",
            "/",
            "--dev",
            "/dev",
            "--proc",
            "/proc",
            "--tmpfs",
            "/tmp",
            "--bind",
            str(workspace),
            str(workspace),
        ]
        for mount in mounts:
            destination = session_home / mount.home_relative
            destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            bwrap.extend(("--ro-bind", str(mount.source), str(destination)))
        for relative in _writable_home_subpaths_for(credentials, backends):
            bwrap.extend(("--tmpfs", str(session_home / relative)))
        for source, destination in qoder_state_overlays:
            bwrap.extend(("--bind", str(source), str(destination)))
        bwrap.extend(
            (
                "--chdir",
                str(workspace),
                "--",
                str(self.env_executable),
                "-i",
                *assignments,
                *runtime_argv,
            )
        )
        return tuple(bwrap)


@dataclass(frozen=True, slots=True)
class BwrapSandboxLauncher:
    """Build one bwrap mount sandbox inside a resource-restricted cgroup."""

    env_executable: Path
    settings: BwrapSandboxSettings
    workspace_roots: tuple[Path, ...]
    credentials: BackendCredentialMounts | None = None

    def check_host(self) -> None:
        """Fail before scheduling when mandatory Linux isolation primitives are absent."""
        if platform.system() != "Linux":
            raise RuntimeError("Sandbox launcher requires Linux; use explicit development mode")
        for label, path in (
            ("bubblewrap", self.settings.bwrap_executable),
            ("systemd-run", self.settings.systemd_run_executable),
            ("env", self.env_executable),
        ):
            if not path.is_absolute() or not path.is_file() or not os.access(path, os.X_OK):
                raise RuntimeError(f"Sandbox {label} executable is unavailable: {path}")
        resolver = self.settings.resolv_conf
        if resolver.is_symlink():
            resolver = resolver.resolve()
        if not resolver.is_file():
            raise RuntimeError(f"Sandbox resolver file is unavailable: {resolver}")
        controllers = Path("/sys/fs/cgroup/cgroup.controllers")
        if not controllers.is_file():
            raise RuntimeError("Sandbox requires a mounted cgroup v2 hierarchy")
        for path in self.settings.read_only_bind_paths:
            if path.is_symlink() or not path.exists():
                raise RuntimeError(f"Sandbox read-only bind source is unavailable: {path}")
        try:
            worker = pwd.getpwnam(self.settings.worker_user)
        except KeyError as error:
            raise RuntimeError(
                f"Sandbox Worker user does not exist: {self.settings.worker_user}"
            ) from error
        if worker.pw_uid == 0:
            raise RuntimeError("Sandbox Worker user cannot be root")
        root_parents = {root.parent.resolve() for root in self.workspace_roots}
        lock_directory = (
            next(iter(root_parents)).parent
            if len(self.workspace_roots) > 1 and len(root_parents) == 1
            else self.workspace_roots[0].parent.resolve()
        )
        if not lock_directory.is_dir():
            raise RuntimeError(
                f"Sandbox workspace parent is unavailable: {lock_directory}"
            )
        lock_path = lock_directory / ".atrex-sandbox-host.lock"
        with lock_path.open("a+b") as lock:
            # A production Campaign starts one Bootstrap process per DSL. They
            # share these roots, so host preparation must be cross-process safe.
            fcntl.flock(lock, fcntl.LOCK_EX)
            self._check_host_workspace_access(worker)

    def _check_host_workspace_access(self, worker: pwd.struct_passwd) -> None:
        for root in self.workspace_roots:
            self._ensure_worker_directory(root, worker)
        probe = self.workspace_roots[0] / f".sandbox-probe-{uuid4().hex}"
        self._ensure_worker_directory(probe, worker)
        try:
            argv = self.wrap(("/bin/true",), workspace=probe, environment={})
            try:
                result = subprocess.run(
                    argv,
                    stdin=subprocess.DEVNULL,
                    capture_output=True,
                    timeout=30,
                    check=False,
                    start_new_session=True,
                )
            except (OSError, subprocess.TimeoutExpired) as error:
                raise RuntimeError(
                    f"Sandbox host probe could not run: {type(error).__name__}"
                ) from error
            if result.returncode != 0:
                diagnostic = result.stderr[:4096].decode(errors="replace").strip()
                permissions = self._path_permissions(probe)
                raise RuntimeError(
                    f"Sandbox host probe failed with exit {result.returncode}: {diagnostic}; "
                    f"workspace path permissions: {permissions}"
                )
        finally:
            shutil.rmtree(probe, ignore_errors=True)

    def _ensure_worker_directory(self, path: Path, worker: pwd.struct_passwd) -> None:
        if path.exists():
            if not path.is_dir() or path.is_symlink():
                raise RuntimeError(f"Sandbox workspace path is unsafe: {path}")
            metadata = path.stat()
            if metadata.st_uid == worker.pw_uid and metadata.st_gid == worker.pw_gid:
                if os.geteuid() == 0 or os.geteuid() == worker.pw_uid:
                    os.chmod(path, 0o700)
                return
            if os.geteuid() != 0 or any(path.iterdir()):
                self._require_worker_owned_path(path, worker)
                return
            # A failed virtiofs preparation can leave an empty root-owned
            # directory. chown is a no-op on that filesystem, so recreate only
            # the proven-empty directory as the configured Worker.
            path.rmdir()

        if os.geteuid() == worker.pw_uid:
            path.mkdir(parents=True, mode=0o700)
            self._require_worker_owned_path(path, worker)
            return

        mkdir = shutil.which("mkdir", path="/usr/sbin:/usr/bin:/sbin:/bin")
        if mkdir is None:
            raise RuntimeError("Sandbox Worker directory creation requires mkdir")
        result = subprocess.run(
            (
                str(self.settings.systemd_run_executable),
                "--quiet",
                "--wait",
                "--pipe",
                "--collect",
                "--service-type=exec",
                f"--uid={self.settings.worker_user}",
                f"--unit=atrex-worker-prepare-{uuid4().hex}.service",
                "--",
                mkdir,
                "-m",
                "700",
                "-p",
                "--",
                str(path),
            ),
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=30,
            check=False,
            start_new_session=True,
        )
        if result.returncode != 0:
            diagnostic = result.stderr[:4096].decode(errors="replace").strip()
            raise RuntimeError(
                f"Sandbox Worker could not create {path}: {diagnostic or result.returncode}"
            )
        self._require_worker_owned_path(path, worker)

    def _require_worker_owned_path(self, path: Path, worker: pwd.struct_passwd) -> None:
        metadata = path.stat()
        if metadata.st_uid == worker.pw_uid and metadata.st_gid == worker.pw_gid:
            return
        worker_view = self._worker_view_ownership(path, worker)
        if worker_view == (worker.pw_uid, worker.pw_gid):
            return
        # After a transient systemd/bwrap service exits, Lima virtiofs can
        # briefly retain the root mount view for a path that the login/Worker
        # view owns. Restart is allowed to wait for that view to converge, but
        # only for the characteristic root:root mismatch and for a bounded
        # interval. Every other mismatch still fails immediately.
        if (metadata.st_uid, metadata.st_gid) == (0, 0):
            for _ in range(_VIRTIOFS_OWNERSHIP_SETTLE_ATTEMPTS - 1):
                time.sleep(_VIRTIOFS_OWNERSHIP_SETTLE_SECONDS)
                worker_view = self._worker_view_ownership(path, worker)
                if worker_view == (worker.pw_uid, worker.pw_gid):
                    return
        suffix = (
            ""
            if worker_view is None
            else f"; Worker view is {worker_view[0]}:{worker_view[1]}"
        )
        raise RuntimeError(
            f"Sandbox path ownership handoff failed: {path} is "
            f"{metadata.st_uid}:{metadata.st_gid}, expected "
            f"{worker.pw_uid}:{worker.pw_gid}{suffix}"
        )

    def _worker_view_ownership(
        self,
        path: Path,
        worker: pwd.struct_passwd,
    ) -> tuple[int, int] | None:
        """Resolve virtiofs ownership through the identity that will execute the Worker."""
        if os.geteuid() != 0:
            return None
        stat_executable = shutil.which("stat", path="/usr/sbin:/usr/bin:/sbin:/bin")
        if stat_executable is None:
            return None
        try:
            result = subprocess.run(
                (
                    str(self.settings.systemd_run_executable),
                    "--quiet",
                    "--wait",
                    "--pipe",
                    "--collect",
                    "--service-type=exec",
                    f"--uid={self.settings.worker_user}",
                    f"--unit=atrex-worker-stat-{uuid4().hex}.service",
                    "--",
                    stat_executable,
                    "-c",
                    "%u:%g",
                    "--",
                    str(path),
                ),
                stdin=subprocess.DEVNULL,
                capture_output=True,
                timeout=30,
                check=False,
                start_new_session=True,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        if result.returncode != 0:
            return None
        match = re.fullmatch(rb"([0-9]+):([0-9]+)\s*", result.stdout)
        return None if match is None else (int(match.group(1)), int(match.group(2)))

    def wrap(
        self,
        runtime_argv: tuple[str, ...],
        *,
        workspace: Path,
        environment: Mapping[str, str],
        interactive: bool = False,
    ) -> tuple[str, ...]:
        """Return a shell-free systemd-run + bwrap argv for one private workspace."""
        if not runtime_argv:
            raise ValueError("runtime argv cannot be empty")
        validate_worker_environment(environment)
        workspace = workspace.resolve()
        if not workspace.is_dir() or workspace.is_symlink():
            raise ValueError("Sandbox workspace must be a real directory")
        if workspace.is_relative_to("/run"):
            raise ValueError("Sandbox workspace cannot be below private /run")
        if not any(workspace.is_relative_to(root.resolve()) for root in self.workspace_roots):
            raise ValueError("Sandbox workspace is outside the configured Worker roots")

        sandbox_workspace = Path(self.settings.workspace_mount.as_posix())
        mapped_environment = {
            key: self._translate_value(value, workspace, sandbox_workspace)
            for key, value in environment.items()
        }
        mapped_environment.setdefault("HOME", self.settings.sandbox_home.as_posix())
        mapped_environment.update(
            {
                "ATREX_SANDBOX": "bwrap-cgroup-v1",
                "ATREX_WORKSPACE": self.settings.workspace_mount.as_posix(),
            }
        )
        backends = _projected_backends(mapped_environment, interactive=interactive)
        _inject_codex_ca_bundle(mapped_environment, backends)
        credential_mounts = (
            ()
            if self.credentials is None
            else _credential_mounts_for(self.credentials, backends)
        )
        installation_roots = (
            ()
            if self.credentials is None
            else tuple(
                dict.fromkeys(
                    root
                    for backend in backends
                    for root in self.credentials.installation_roots_for(backend)
                )
            )
        )
        restored_host_paths = tuple(
            dict.fromkeys((*installation_roots, *(mount.source for mount in credential_mounts)))
        )
        forbidden_roots = tuple(
            path.resolve() for path in (*self.workspace_roots, *self.settings.hidden_host_paths)
        )
        if any(
            path.resolve().is_relative_to(root)
            for path in restored_host_paths
            for root in forbidden_roots
        ):
            raise ValueError("Backend credential mounts cannot expose hidden Runtime paths")
        session_home = Path(mapped_environment["HOME"])
        writable_backends = (
            ()
            if self.credentials is None
            else tuple(
                backend
                for backend in backends
                if self.credentials.requires_writable_backend_home(backend)
            )
        )
        writable_projection = self.credentials is not None and bool(writable_backends)
        if credential_mounts or writable_projection:
            host_home_value = environment.get("HOME")
            if host_home_value is None:
                raise ValueError("Sandbox Backend credential mounts require an isolated HOME")
            host_session_home = Path(host_home_value).resolve()
            if not host_session_home.is_relative_to(workspace):
                raise ValueError(
                    "Sandbox Backend credential HOME must be inside the Worker workspace"
                )
            for mount in credential_mounts:
                (host_session_home / mount.home_relative).parent.mkdir(
                    parents=True,
                    exist_ok=True,
                    mode=0o700,
                )
            if not session_home.is_relative_to(sandbox_workspace):
                raise ValueError(
                    "Sandbox Backend credential HOME must be inside the Worker workspace"
                )
            assert self.credentials is not None
            for backend in writable_backends:
                self.credentials.prepare_writable_backend_home(backend, host_session_home)
            qoder_state_overlays = (
                self.credentials.qoder_state_overlays(host_session_home)
                if "qodercli" in writable_backends
                else ()
            )
            for backend in backends:
                self.credentials.update_environment(mapped_environment, backend, session_home)
        else:
            qoder_state_overlays = ()
        translated_argv = tuple(
            self._translate_value(value, workspace, sandbox_workspace) for value in runtime_argv
        )
        assignments = tuple(f"{key}={value}" for key, value in sorted(mapped_environment.items()))
        hidden = self._minimal_hidden_paths(
            (*self.workspace_roots, *self.settings.hidden_host_paths)
        )
        bwrap = [
            str(self.settings.bwrap_executable),
            "--die-with-parent",
            "--new-session",
            "--unshare-user",
            "--unshare-pid",
            "--unshare-ipc",
            "--unshare-uts",
            "--unshare-cgroup-try",
            "--hostname",
            "atrex-agent",
            "--cap-drop",
            "ALL",
            "--ro-bind",
            "/",
            "/",
            "--dev",
            "/dev",
            "--proc",
            "/proc",
            "--tmpfs",
            "/run",
            "--dir",
            "/run/systemd",
            "--dir",
            "/run/systemd/resolve",
            "--ro-bind",
            str(self.settings.resolv_conf.resolve()),
            "/run/systemd/resolve/resolv.conf",
            "--ro-bind",
            str(self.settings.resolv_conf.resolve()),
            "/run/systemd/resolve/stub-resolv.conf",
        ]
        if interactive:
            # systemd-run owns the PTY. Keep /dev private, but make the
            # already-open devpts endpoint visible so terminal-aware CLIs can
            # recognize and control their inherited streams.
            bwrap.extend(("--ro-bind", "/dev/pts", "/dev/pts"))
        bwrap.extend(
            (
                "--tmpfs",
                "/tmp",
                "--tmpfs",
                "/home",
                "--tmpfs",
                "/root",
            )
        )
        for path in hidden:
            if path.as_posix() not in {"/home", "/root", "/run", "/tmp"}:
                bwrap.extend(("--tmpfs", str(path)))
        for path in self.settings.read_only_bind_paths:
            for parent in self._missing_masked_parents(path):
                bwrap.extend(("--dir", str(parent)))
            bwrap.extend(("--ro-bind", str(path), str(path)))
        for path in restored_host_paths:
            for parent in self._missing_masked_parents(path):
                bwrap.extend(("--dir", str(parent)))
            bwrap.extend(("--ro-bind", str(path), str(path)))
        bwrap.extend(
            (
                "--dir",
                self.settings.sandbox_home.as_posix(),
                "--bind",
                str(workspace),
                self.settings.workspace_mount.as_posix(),
            )
        )
        for mount in credential_mounts:
            destination = session_home / mount.home_relative
            bwrap.extend(("--ro-bind", str(mount.source), str(destination)))
        if self.credentials is not None:
            for relative in _writable_home_subpaths_for(self.credentials, backends):
                bwrap.extend(("--tmpfs", str(session_home / relative)))
        for source, destination in qoder_state_overlays:
            bwrap.extend(
                (
                    "--bind",
                    str(source),
                    self._translate_value(str(destination), workspace, sandbox_workspace),
                )
            )
        for name in ("input", "agent", "runtime-tools"):
            source = workspace / name
            if source.exists():
                if source.is_symlink():
                    raise ValueError("Sandbox immutable workspace input cannot be a symlink")
                bwrap.extend(
                    (
                        "--ro-bind",
                        str(source),
                        str(sandbox_workspace / name),
                    )
                )
        for source in sorted(workspace.glob("*.json")):
            if source.is_symlink() or not source.is_file():
                raise ValueError("Sandbox top-level manifest must be a regular file")
            bwrap.extend(
                (
                    "--ro-bind",
                    str(source),
                    str(sandbox_workspace / source.name),
                )
            )
        bwrap.extend(
            (
                "--chdir",
                self.settings.workspace_mount.as_posix(),
                "--",
                str(self.env_executable),
                "-i",
                *assignments,
                *translated_argv,
            )
        )
        unit = f"atrex-worker-{uuid4().hex}.service"
        systemd = [str(self.settings.systemd_run_executable)]
        resources = self.settings.resources
        systemd.extend(
            (
                "--quiet",
                "--wait",
                "--pty" if interactive else "--pipe",
                "--collect",
                "--service-type=exec",
                f"--uid={self.settings.worker_user}",
                f"--unit={unit}",
                f"--property=MemoryMax={resources.memory_max_bytes}",
                f"--property=MemorySwapMax={resources.memory_swap_max_bytes}",
                f"--property=CPUQuota={resources.cpu_quota_percent}%",
                f"--property=TasksMax={resources.tasks_max}",
            )
        )
        if os.geteuid() == 0:
            worker = pwd.getpwnam(self.settings.worker_user)
            self._chown_tree(workspace, worker.pw_uid, worker.pw_gid)
        return (*systemd, "--", *bwrap)

    @staticmethod
    def _chown_tree(root: Path, uid: int, gid: int) -> None:
        """Hand a trusted, freshly assembled Workspace to the configured Worker."""
        os.chown(root, uid, gid, follow_symlinks=False)
        for directory, names, files in os.walk(root, followlinks=False):
            parent = Path(directory)
            for name in (*names, *files):
                os.chown(parent / name, uid, gid, follow_symlinks=False)

    @staticmethod
    def _path_permissions(path: Path) -> str:
        """Render bounded ownership/mode diagnostics for Sandbox path failures."""
        values: list[str] = []
        current = Path("/")
        for part in path.resolve().parts[1:]:
            current /= part
            metadata = current.stat()
            values.append(
                f"{current}={stat.S_IMODE(metadata.st_mode):04o}:"
                f"{metadata.st_uid}:{metadata.st_gid}"
            )
        return ", ".join(values)

    @staticmethod
    def _translate_value(value: str, workspace: Path, sandbox_workspace: Path) -> str:
        """Translate exact absolute Worker paths without rewriting arbitrary prompt text."""
        prefix = str(workspace)
        if value == prefix:
            return str(sandbox_workspace)
        if value.startswith(prefix + os.sep):
            return str(sandbox_workspace / value[len(prefix + os.sep) :])
        candidate = Path(value)
        if candidate.is_absolute():
            resolved = candidate.resolve(strict=False)
            if resolved == workspace:
                return str(sandbox_workspace)
            if resolved.is_relative_to(workspace):
                return str(sandbox_workspace / resolved.relative_to(workspace))
        return value

    @staticmethod
    def _minimal_hidden_paths(paths: tuple[Path, ...]) -> tuple[Path, ...]:
        resolved = sorted({path.resolve() for path in paths}, key=lambda path: len(path.parts))
        selected: list[Path] = []
        for path in resolved:
            if path == Path("/"):
                raise ValueError("Sandbox cannot hide the host root")
            if not any(path.is_relative_to(parent) for parent in selected):
                selected.append(path)
        return tuple(selected)

    @staticmethod
    def _missing_masked_parents(path: Path) -> tuple[Path, ...]:
        if path.is_relative_to("/home"):
            boundary = Path("/home")
        elif path.is_relative_to("/root"):
            boundary = Path("/root")
        else:
            return ()
        parents: list[Path] = []
        current = path.parent
        while current != boundary:
            parents.append(current)
            current = current.parent
        return tuple(reversed(parents))
