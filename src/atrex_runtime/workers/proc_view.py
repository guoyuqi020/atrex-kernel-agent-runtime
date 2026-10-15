"""Read-only process metadata for one bwrap PID namespace, without host procfs.

OCI may reject mounting a new procfs. The trusted parent mirrors only stat,
status, cmdline and task/children of its own descendants into a private directory.
There are no host fd, root, cwd, mem or environ links. The launcher may add a
fixed exe alias to the allowlisted Provider binary for standalone Bun. Core can still identify,
suspend and clean up detached Agent subprocesses using namespace-local PIDs.
"""

from __future__ import annotations

import array
import hashlib
import os
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path


def _write(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".next")
    temporary.write_bytes(value)
    temporary.replace(path)


def snapshot(root_pid: int, destination: Path, *, proc: Path = Path("/proc")) -> bool:
    """Publish a namespace-local census; never enumerate unrelated host processes."""
    pending = [root_pid]
    records: dict[int, tuple[list[str], bytes, str]] = {}
    seen: set[int] = set()
    namespace_depth = None
    while pending:
        pid = pending.pop()
        if pid in seen:
            continue
        seen.add(pid)
        path = proc / str(pid)
        try:
            children: set[int] = set()
            for task in (path / "task").iterdir():
                children.update(int(v) for v in (task / "children").read_text().split())
            pending.extend(children)
            status = (path / "status").read_text()
            ns = {
                parts[0].rstrip(":"): parts[1:]
                for line in status.splitlines()
                if (parts := line.split()) and parts[0] in {"NSpid:", "NSpgid:", "NSsid:"}
            }
            if pid == root_pid:
                namespace_depth = len(ns.get("NSpid", []))
            if namespace_depth is None or len(ns.get("NSpid", [])) <= namespace_depth:
                continue
            raw = (path / "stat").read_text()
            name, rest = raw.rsplit(")", 1)
            fields = rest.split()
            local = ns["NSpid"][namespace_depth]
            fields[2] = ns["NSpgid"][namespace_depth]
            fields[3] = ns["NSsid"][namespace_depth]
            records[pid] = (
                [local, name[name.index("(") :] + ")", *fields],
                (path / "cmdline").read_bytes(),
                status,
            )
        except (OSError, ValueError, IndexError, KeyError):
            continue  # Exit races are normal; never substitute unrelated PID data.
    if not records:
        return False
    ids = {outer: fields[0] for outer, (fields, _, _) in records.items()}
    if "1" not in ids.values():
        return False
    children_by_parent: dict[str, list[str]] = {}
    for fields, command, status in records.values():
        fields[3] = ids.get(int(fields[3]), "0")  # stat ppid
        children_by_parent.setdefault(fields[3], []).append(fields[0])
        directory = destination / fields[0]
        _write(directory / "stat", (" ".join(fields) + "\n").encode())
        # Core sees local PIDs; don't expose host identities in NSpid mappings.
        statuses = []
        for line in status.splitlines():
            key = line.split(":", 1)[0]
            if key in {"Pid", "Tgid", "PPid", "NSpid", "NStgid", "NSpgid", "NSsid"}:
                continue
            # Name, State and other public statistics describe only this Agent.
            statuses.append(line)
        statuses.extend(
            (
                f"Pid:\t{fields[0]}",
                f"PPid:\t{fields[3]}",
                f"NSpid:\t{fields[0]}",
                f"NSpgid:\t{fields[4]}",
                f"NSsid:\t{fields[5]}",
            )
        )
        _write(directory / "status", ("\n".join(statuses) + "\n").encode())
        _write(directory / "cmdline", command)
    for local in ids.values():
        _write(
            destination / local / "task" / local / "children",
            " ".join(children_by_parent.get(local, [])).encode(),
        )
    for old in destination.iterdir():
        if old.name.isdecimal() and old.name not in ids.values():
            shutil.rmtree(old)
    # A synthetic self supplies the namespace identity to Core, not host files.
    if not (destination / "self").is_symlink():
        (destination / "self").symlink_to("1")
    return True


def owns_process(root: int, pid: int, *, proc: Path = Path("/proc")) -> bool:
    """Verify host ancestry; a client cannot request another PID's mappings."""
    visited: set[int] = set()
    while pid > 1 and pid not in visited:
        if pid == root:
            return True
        visited.add(pid)
        try:
            pid = int((proc / str(pid) / "stat").read_text().rsplit(")", 1)[1].split()[1])
        except (OSError, ValueError, IndexError):
            return False
    return False


def _serve_maps(
    listener: socket.socket,
    root: int,
    stopped: threading.Event,
    workspace: Path,
    visible_workspace: str,
) -> None:
    while not stopped.is_set():
        try:
            client, _ = listener.accept()
        except TimeoutError:
            continue
        except OSError:
            return
        with client:
            client.settimeout(0.5)
            pid, _uid, _gid = struct.unpack(
                "3i",
                client.getsockopt(socket.SOL_SOCKET, int(getattr(socket, "SO_PEERCRED", 17)), 12),
            )
            if not owns_process(root, pid):
                continue
            try:
                request = b""
                while b"\n" not in request and len(request) < 32:
                    part = client.recv(32 - len(request))
                    if not part:
                        break
                    request += part
                if request == b"M\n":
                    fd = os.open(f"/proc/{pid}/maps", os.O_RDONLY | os.O_CLOEXEC)
                    try:
                        client.sendmsg(
                            [b"M"], [(socket.SOL_SOCKET, socket.SCM_RIGHTS, array.array("i", [fd]))]
                        )
                    finally:
                        os.close(fd)
                elif request.startswith(b"R") and request.endswith(b"\n"):
                    number = request[1:-1]
                    if not number.isdigit() or len(number) > 10:
                        continue
                    # Readlink metadata only. Never send a descriptor for a requested
                    # FD, follow its target, or accept a target PID/path from the client.
                    target = os.readlink(f"/proc/{pid}/fd/{int(number)}")
                    host = str(workspace)
                    if target == host or target.startswith(host + "/"):
                        target = visible_workspace + target[len(host) :]
                    client.sendall(target.encode())
            except OSError:
                continue


def _install_maps_shim(destination: Path) -> None:
    source = Path(__file__).with_name("proc_maps.c")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    cache = Path(tempfile.gettempdir()) / f"atrex-proc-maps-{os.getuid()}-{digest}"
    cache.mkdir(mode=0o700, exist_ok=True)
    if cache.is_symlink() or cache.stat().st_uid != os.getuid():
        raise RuntimeError("Unsafe process-view compiler cache")
    compiled = cache / "libproc-maps.so"
    if not compiled.exists():
        with tempfile.TemporaryDirectory(dir=cache) as temporary:
            output = Path(temporary) / compiled.name
            subprocess.run(
                ["cc", "-shared", "-fPIC", "-O2", str(source), "-ldl", "-o", str(output)],
                check=True,
                capture_output=True,
            )
            output.chmod(0o500)
            output.replace(compiled)
    target = destination.parent / compiled.name
    target.unlink(missing_ok=True)
    shutil.copy2(compiled, target)


def main() -> int:
    """Own bwrap and its process view for precisely the lifetime of one Worker."""
    destination = Path(sys.argv[1])
    argv = sys.argv[2:]
    if not argv or not destination.is_dir():
        raise ValueError("proc view requires a prepared directory and bwrap argv")
    _install_maps_shim(destination)
    endpoint = destination / ".maps.sock"
    endpoint.unlink(missing_ok=True)
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    # Attempt paths exceed AF_UNIX's short sockaddr pathname limit. Bind relative
    # to the private directory before starting any thread, then restore cwd.
    previous = os.open(".", os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.chdir(destination)
        listener.bind(endpoint.name)
    finally:
        os.fchdir(previous)
        os.close(previous)
    listener.listen(32)
    listener.settimeout(0.2)
    process = subprocess.Popen(argv)
    stopped = threading.Event()
    workspace = destination.parent.parent
    visible_workspace = next(
        argv[i + 2]
        for i in range(len(argv) - 2)
        if argv[i] == "--bind" and argv[i + 1] == str(workspace)
    )
    broker = threading.Thread(
        target=_serve_maps,
        args=(listener, process.pid, stopped, workspace, visible_workspace),
        daemon=True,
    )
    broker.start()
    try:
        while process.poll() is None:
            snapshot(process.pid, destination)
            time.sleep(0.025)
        return process.wait()
    finally:
        stopped.set()
        listener.close()
        broker.join(timeout=1)
        if process.poll() is None:
            process.kill()
            process.wait()
        shutil.rmtree(destination, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
