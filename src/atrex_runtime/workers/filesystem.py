"""Allowlisted OS dependencies for an otherwise empty Agent mount namespace."""

import os
from pathlib import Path

# Executables, their shared libraries, and public OS support data. Never mount
# /, /etc, /opt, a home, a repository, or an experiment archive wholesale.
_SYSTEM_PATHS = (
    "/usr/bin",
    "/usr/sbin",
    "/usr/lib",
    "/usr/lib64",
    "/usr/libexec",
    "/bin",
    "/sbin",
    "/lib",
    "/lib64",
    "/usr/share/terminfo",
    "/usr/share/locale",
    "/usr/share/git-core",
    "/usr/share/zoneinfo",
    "/etc/hosts",
    "/etc/nsswitch.conf",
    "/etc/ld.so.cache",
    "/etc/localtime",
)

_CERTIFICATE_DIRECTORIES = ("/etc/ssl/certs", "/etc/pki/tls/certs")
_CERTIFICATE_BUNDLES = (
    "/etc/ssl/cert.pem",
    "/etc/ssl/ca-bundle.pem",
    "/etc/pki/tls/cert.pem",
)


def dependency_mounts(paths: tuple[Path, ...]) -> list[str]:
    """Bind exact dependencies, retaining symlink aliases without broad parent mounts."""
    args: list[str] = []
    mounted: set[Path] = set()

    def mount(path: Path) -> None:
        if path in mounted:
            return
        mounted.add(path)
        if path.is_symlink():
            target = path.resolve(strict=True)
            mount(target)
            args.extend(("--symlink", str(target), str(path)))
        else:
            args.extend(("--ro-bind", str(path), str(path)))

    for path in paths:
        mount(path)
    return args


def certificate_mounts() -> list[str]:
    """Complete public trust-store links without exposing their parent directories.

    RHEL's ca-bundle.crt and Debian's hashed certificates can point outside the
    mounted certificate directory. Follow file links (including intermediate
    aliases) while leaving existing aliases inside read-only mounts untouched.
    """
    directories = tuple(Path(p) for p in _CERTIFICATE_DIRECTORIES if Path(p).is_dir())
    args = dependency_mounts(directories)
    covered = {*directories, *(p.resolve() for p in directories)}
    visited: set[Path] = set()

    def expose(path: Path) -> None:
        path = Path(os.path.abspath(path))
        if path in visited or not path.is_file():
            return
        visited.add(path)
        already_visible = any(path.is_relative_to(root) for root in covered)
        if path.is_symlink():
            target = path.readlink()
            if not target.is_absolute():
                target = path.parent / target
            target = Path(os.path.abspath(target))
            expose(target)
            if not already_visible:
                args.extend(("--symlink", str(target), str(path)))
        elif not already_visible:
            args.extend(("--ro-bind", str(path), str(path)))

    for directory in directories:
        for entry in sorted(directory.iterdir()):
            if entry.is_symlink():
                expose(entry)
    for name in _CERTIFICATE_BUNDLES:
        expose(Path(name))
    return args


def system_mounts() -> list[str]:
    """Only mount installed public OS dependencies; no implicit local package roots."""
    return [
        *dependency_mounts(tuple(Path(p) for p in _SYSTEM_PATHS if Path(p).exists())),
        *certificate_mounts(),
    ]


def restricted_proc_mounts() -> list[str]:
    """OCI fallback without host PIDs, fd/root links, environments or kernel memory.

    Some container seccomp/mount policies prohibit a new procfs. In that mode
    expose only public machine statistics, never the outer procfs directory.
    """
    paths = ("cpuinfo", "meminfo", "stat", "uptime", "version", "loadavg")
    return [
        "--dir",
        "/proc",
        *dependency_mounts(
            tuple(Path("/proc") / name for name in paths if (Path("/proc") / name).is_file())
        ),
    ]
