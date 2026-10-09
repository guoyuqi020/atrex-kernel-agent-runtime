"""Single-threaded native Wiki preload/fork server; private local IPC only."""

from __future__ import annotations

import contextlib
import json
import os
import signal
import socket
import sys
import time
from pathlib import Path
from types import FrameType
from typing import TYPE_CHECKING

# This process uses the configured native interpreter, which need not have the
# HTTP application's third-party dependencies. Keep its imports stdlib-only.
sys.path.insert(0, str(Path(__file__).resolve().parent))
if TYPE_CHECKING:
    from atrex_local_wiki.native_preload import NativePreload
    from atrex_local_wiki.preloaded_worker import reap_queries, stop_query
else:
    from native_preload import NativePreload
    from preloaded_worker import reap_queries, stop_query


def serve(root: Path, address: str, concurrency: int, max_bytes: int) -> None:
    started = time.monotonic()
    native = NativePreload(root)
    children: dict[int, tuple[float | None, bool]] = {}
    running = True

    def shutdown(_signum: int, _frame: FrameType | None) -> None:
        nonlocal running
        running = False

    def cancel_query(_signum: int, _frame: FrameType | None) -> None:
        # Unwind NativePreload.run so separately-sessioned model CLIs are reaped.
        raise SystemExit(124)

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
        listener.bind(address)
        os.chmod(address, 0o600)
        listener.listen(concurrency)
        listener.settimeout(0.1)
        print(
            json.dumps({"status": "ready", "preload_seconds": time.monotonic() - started}),
            flush=True,
        )
        try:
            while running:
                reap_queries(children)
                if len(children) >= concurrency:
                    time.sleep(0.05)
                    continue
                try:
                    connection, _ = listener.accept()
                except TimeoutError:
                    continue
                with connection:
                    connection.settimeout(5)
                    try:
                        with connection.makefile("rb") as stream:
                            line = stream.readline(8 * 1024 * 1024 + 1)
                        request = json.loads(line)
                    except (OSError, ValueError):
                        continue
                    if len(line) > 8 * 1024 * 1024 or not line.endswith(b"\n"):
                        continue
                    if not isinstance(request, dict):
                        continue
                    argv = request.get("argv")
                    timeout = request.get("timeout")
                    if not isinstance(argv, list) or any(not isinstance(v, str) for v in argv):
                        continue
                    if timeout is not None and (
                        not isinstance(timeout, (float, int)) or timeout <= 0
                    ):
                        continue
                    # The parent is single threaded and never runs request code.
                    # Model waits run concurrently in children with separate env/stdout.
                    pid = os.fork()
                    if pid == 0:
                        os.setsid()
                        signal.signal(signal.SIGTERM, cancel_query)
                        signal.signal(signal.SIGINT, cancel_query)
                        listener.close()
                        try:
                            code, out, err = native.run(argv)
                            if len(out.encode()) > max_bytes:
                                code, out, err = (
                                    1,
                                    "",
                                    "GPU Wiki response exceeded the configured byte limit",
                                )
                            response = {"returncode": code, "stdout": out, "stderr": err[-1000:]}
                            connection.sendall(
                                json.dumps(response, ensure_ascii=False).encode() + b"\n"
                            )
                        except BaseException:
                            # Query diagnostics belong to its envelope, never the daemon pipe.
                            with contextlib.suppress(OSError):
                                connection.sendall(
                                    b'{"returncode":1,"stdout":"",'
                                    b'"stderr":"GPU Wiki native query failed"}\n'
                                )
                        finally:
                            connection.close()
                            os._exit(0)
                    children[pid] = (
                        time.monotonic() + timeout if timeout is not None else None,
                        False,
                    )
        finally:
            for pid in children:
                stop_query(pid)
            deadline = time.monotonic() + 2
            while children and time.monotonic() < deadline:
                reap_queries(children)
                time.sleep(0.05)
            for pid in children:
                stop_query(pid, force=True)
                with contextlib.suppress(ChildProcessError):
                    os.waitpid(pid, 0)


if __name__ == "__main__":
    os.umask(0o077)
    try:
        serve(Path(sys.argv[1]), sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
    except Exception as error:
        print(json.dumps({"status": "error", "error": str(error)[:1000]}), flush=True)
        raise SystemExit(1) from None
