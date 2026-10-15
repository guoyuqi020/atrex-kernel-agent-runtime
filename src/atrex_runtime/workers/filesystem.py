"""Allowlisted OS dependencies for an otherwise empty Agent mount namespace."""

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
    "/etc/ssl/certs",
    "/etc/pki/tls/certs",
    "/etc/hosts",
    "/etc/nsswitch.conf",
    "/etc/ld.so.cache",
    "/etc/localtime",
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


def system_mounts() -> list[str]:
    """Only mount installed public OS dependencies; no implicit local package roots."""
    return dependency_mounts(tuple(Path(p) for p in _SYSTEM_PATHS if Path(p).exists()))


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
