#!/usr/bin/env python3
"""Run destructive-negative acceptance checks against the production Linux launcher."""

from __future__ import annotations

import argparse
import contextlib
import os
import pwd
import shutil
import socket
import socketserver
import subprocess
import tempfile
import threading
import urllib.request
from collections.abc import Iterator
from pathlib import Path

from atrex_runtime.config import (
    BwrapSandboxSettings,
    CgroupResourceSettings,
)
from atrex_runtime.workers.launcher import BwrapSandboxLauncher

_MEMORY_MAX = 256 * 1024 * 1024
_CPU_QUOTA_PERCENT = 100
_TASKS_MAX = 64
_EXTERNAL_URL = "http://archive.ubuntu.com/ubuntu/dists/resolute/Release"


class _RuntimeHandler(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        self.request.recv(4096)
        self.request.sendall(
            b"HTTP/1.1 200 OK\r\nContent-Length: 10\r\nConnection: close\r\n\r\nruntime-ok"
        )


class _RuntimeServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


@contextlib.contextmanager
def _runtime_server(host: str, port: int) -> Iterator[int]:
    server = _RuntimeServer((host, port), _RuntimeHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_address[1]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _free_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _require_executable(name: str) -> Path:
    value = shutil.which(name)
    if value is None:
        raise RuntimeError(f"required acceptance executable is missing: {name}")
    return Path(value).absolute()


def _assert_external_baseline() -> str:
    address = socket.gethostbyname("archive.ubuntu.com")
    with urllib.request.urlopen(_EXTERNAL_URL, timeout=15) as response:
        if response.status != 200 or not response.read(64):
            raise RuntimeError("external HTTP baseline did not return a non-empty 200 response")
    return address


def _sandbox_script(
    *,
    sibling_secret: Path,
    host_sentinel: Path,
    host_home: Path,
    runtime_port: int,
    runtime_host: str,
    external_address: str,
) -> str:
    return f"""
set -eu
fail() {{ printf 'FAIL=%s\\n' "$1"; exit 91; }}
[ "$HOME" = /home/agent ] || fail home
[ "$PWD" = /home/agent/workspace ] || fail working-directory
[ -r /etc/os-release ] || fail system-read
[ ! -e {sibling_secret} ] || fail sibling-visible
[ ! -e {host_home} ] || fail host-home-visible
(printf bad > /etc/atrex-sandbox-write) >/dev/null 2>&1 && fail system-write
(printf bad > {host_sentinel}) >/dev/null 2>&1 && fail host-write
(printf bad > input/immutable.txt) >/dev/null 2>&1 && fail input-write
(printf bad > agent/immutable.txt) >/dev/null 2>&1 && fail agent-write
(printf bad > runtime-tools/immutable.txt) >/dev/null 2>&1 && fail tools-write
(printf bad > manifest.json) >/dev/null 2>&1 && fail manifest-write
printf writable > scratch/worker-output.txt
[ "$(cat scratch/worker-output.txt)" = writable ] || fail workspace-write
cap_eff=$(awk '/^CapEff:/ {{print $2}}' /proc/self/status)
[ "$cap_eff" = 0000000000000000 ] || fail capabilities
runtime_body=$(curl --silent --show-error --fail --max-time 5 --noproxy '*' \
  http://{runtime_host}:{runtime_port}/)
[ "$runtime_body" = runtime-ok ] || fail runtime-network
if ! curl --silent --show-error --fail --max-time 15 --noproxy '*' \
  {_EXTERNAL_URL} >/dev/null; then
  printf 'RESOLV_CONF_BEGIN\n'; cat /etc/resolv.conf; printf 'RESOLV_CONF_END\n'
  ip -4 route || true
  curl --silent --show-error --fail --max-time 15 --noproxy '*' \
    --resolve archive.ubuntu.com:80:{external_address} {_EXTERNAL_URL} >/dev/null \
    || fail direct-external-network
  fail direct-external-dns
fi
cgdir=$(find /sys/fs/cgroup/system.slice -maxdepth 1 -type d \
  -name 'atrex-worker-*.service' -print -quit)
[ -n "$cgdir" ] || fail cgroup-path
printf 'CG_MEMORY=%s\\n' "$(cat "$cgdir/memory.max")"
printf 'CG_SWAP=%s\\n' "$(cat "$cgdir/memory.swap.max")"
printf 'CG_CPU=%s\\n' "$(cat "$cgdir/cpu.max")"
printf 'CG_PIDS=%s\\n' "$(cat "$cgdir/pids.max")"
printf 'SANDBOX_ACCEPTANCE=passed\\n'
"""


def _verify_cgroup(output: str) -> None:
    values = dict(
        line.split("=", 1)
        for line in output.splitlines()
        if line.startswith(("CG_MEMORY=", "CG_SWAP=", "CG_CPU=", "CG_PIDS="))
    )
    expected = {
        "CG_MEMORY": str(_MEMORY_MAX),
        "CG_SWAP": "0",
        "CG_PIDS": str(_TASKS_MAX),
    }
    for key, value in expected.items():
        if values.get(key) != value:
            raise RuntimeError(
                f"unexpected {key}: {values.get(key)!r}; expected {value!r}\n{output}"
            )
    cpu = values.get("CG_CPU", "").split()
    if len(cpu) != 2 or cpu[0] == "max" or int(cpu[0]) * 100 != int(cpu[1]) * _CPU_QUOTA_PERCENT:
        raise RuntimeError(f"unexpected CG_CPU: {values.get('CG_CPU')!r}\n{output}")


def run_acceptance(temporary_parent: Path) -> None:
    """Create four roots and validate one complete production launcher invocation."""
    if os.uname().sysname != "Linux":
        raise RuntimeError("Linux Sandbox acceptance must run on Linux")
    external_address = _assert_external_baseline()
    worker_user = os.environ.get("SUDO_USER") or pwd.getpwuid(os.getuid()).pw_name
    worker = pwd.getpwnam(worker_user)
    if worker.pw_uid == 0:
        raise RuntimeError("acceptance must select a non-root SUDO_USER/Worker user")
    temporary_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="atrex-sandbox-", dir=temporary_parent) as base_value:
        base = Path(base_value)
        base.chmod(0o711)
        roots = tuple(base / name for name in ("attempts", "evolution", "problems", "bootstrap"))
        for root in roots:
            root.mkdir(mode=0o700)
        workspace = roots[0] / "attempt-test" / "run-test"
        for relative in ("input", "agent", "runtime-tools", "scratch"):
            (workspace / relative).mkdir(parents=True, mode=0o700)
        for relative in (
            "input/immutable.txt",
            "agent/immutable.txt",
            "runtime-tools/immutable.txt",
            "manifest.json",
        ):
            (workspace / relative).write_text("immutable\n", encoding="utf-8")
        sibling_secret = roots[1] / "sibling-secret.txt"
        sibling_secret.write_text("must remain invisible\n", encoding="utf-8")
        host_sentinel = base / "host-sentinel.txt"
        host_sentinel.write_text("must remain unchanged\n", encoding="utf-8")
        if os.geteuid() == 0:
            for path in (*roots, workspace, *workspace.iterdir()):
                os.chown(path, worker.pw_uid, worker.pw_gid)

        runtime_port = _free_port()
        runtime_host = "127.0.0.1"
        launcher = BwrapSandboxLauncher(
            _require_executable("env"),
            BwrapSandboxSettings(
                bwrap_executable=_require_executable("bwrap"),
                systemd_run_executable=_require_executable("systemd-run"),
                systemd_user=False,
                worker_user=worker_user,
                resolv_conf=Path("/run/systemd/resolve/resolv.conf"),
                resources=CgroupResourceSettings(
                    memory_max_bytes=_MEMORY_MAX,
                    memory_swap_max_bytes=0,
                    cpu_quota_percent=_CPU_QUOTA_PERCENT,
                    tasks_max=_TASKS_MAX,
                ),
            ),
            roots,
        )
        launcher.check_host()
        with _runtime_server(runtime_host, runtime_port):
            script = _sandbox_script(
                sibling_secret=sibling_secret,
                host_sentinel=host_sentinel,
                host_home=Path(worker.pw_dir),
                runtime_port=int(runtime_port),
                runtime_host=runtime_host,
                external_address=external_address,
            )
            argv = launcher.wrap(
                (_require_executable("bash").as_posix(), "-c", script),
                workspace=workspace,
                environment={},
            )
            result = subprocess.run(
                argv,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
                start_new_session=True,
            )
        if result.returncode != 0:
            raise RuntimeError(
                f"Sandbox acceptance exited {result.returncode}\n"
                f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
            )
        _verify_cgroup(result.stdout)
        if "SANDBOX_ACCEPTANCE=passed" not in result.stdout:
            raise RuntimeError(f"Sandbox did not emit its success marker:\n{result.stdout}")
        if host_sentinel.read_text(encoding="utf-8") != "must remain unchanged\n":
            raise RuntimeError("Sandbox modified the host sentinel")
        print(result.stdout, end="")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--temporary-parent",
        type=Path,
        default=Path("/var/tmp"),
        help="Linux host directory used for automatically cleaned test workspaces",
    )
    arguments = parser.parse_args()
    run_acceptance(arguments.temporary_parent.resolve())


if __name__ == "__main__":
    main()
