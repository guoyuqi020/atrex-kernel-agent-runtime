"""Own one preheated native index and isolated, copy-on-write query processes."""

from __future__ import annotations

import contextlib
import json
import os
import selectors
import signal
import socket
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any


class PreloadedWorkerError(ValueError):
    """A native index worker could not start or complete a query."""


class PreloadedWorker:
    """Manage a single-threaded fork server outside the HTTP service's threads."""

    def __init__(
        self,
        root: Path,
        python: Path,
        *,
        concurrency: int,
        max_response_bytes: int,
    ) -> None:
        if not hasattr(os, "fork"):
            raise PreloadedWorkerError("GPU Wiki preloading requires a POSIX fork platform")
        self._directory = tempfile.TemporaryDirectory(prefix="atrex-wiki-")
        self._socket = Path(self._directory.name) / "query.sock"
        self._limit = max_response_bytes * 6 + 65536
        self._closed = False
        self._process: subprocess.Popen[bytes] | None = None
        self._stderr = tempfile.TemporaryFile()  # noqa: SIM115 - owned until close()
        try:
            self._process = subprocess.Popen(
                [
                    str(python),
                    str(Path(__file__).with_name("preloaded_server.py")),
                    str(root),
                    str(self._socket),
                    str(concurrency),
                    str(max_response_bytes),
                ],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=self._stderr,
                start_new_session=True,
            )
            assert self._process.stdout is not None
            with selectors.DefaultSelector() as selector:
                selector.register(self._process.stdout, selectors.EVENT_READ)
                if not selector.select(timeout=120):
                    raise PreloadedWorkerError("GPU Wiki index preloading timed out")
                line = self._process.stdout.readline(65536)
                if not line:
                    self._stderr.seek(0)
                    detail = self._stderr.read(4096).decode(errors="replace").strip()
                    raise PreloadedWorkerError(
                        f"GPU Wiki index worker exited before readiness: {detail}"
                    )
                message = json.loads(line)
            if not isinstance(message, dict) or message.get("status") != "ready":
                detail = (
                    message.get("error", "invalid startup response")
                    if isinstance(message, dict)
                    else "invalid startup response"
                )
                raise PreloadedWorkerError(f"GPU Wiki index preloading failed: {detail}")
            self.startup: dict[str, Any] = message
        except (OSError, ValueError) as error:
            self.close()
            if isinstance(error, PreloadedWorkerError):
                raise
            raise PreloadedWorkerError(f"GPU Wiki index worker failed: {error}") from error

    def check_health(self) -> None:
        if self._closed or self._process is None or self._process.poll() is not None:
            raise PreloadedWorkerError("GPU Wiki preloaded index worker is unavailable")

    def query(self, argv: list[str], timeout: float | None) -> subprocess.CompletedProcess[bytes]:
        self.check_health()
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
                connection.settimeout(timeout)
                connection.connect(str(self._socket))
                connection.sendall(json.dumps({"argv": argv, "timeout": timeout}).encode() + b"\n")
                with connection.makefile("rb") as stream:
                    line = stream.readline(self._limit + 1)
                if len(line) > self._limit or not line.endswith(b"\n"):
                    raise PreloadedWorkerError("GPU Wiki index worker returned an invalid frame")
                value = json.loads(line)
            if (
                not isinstance(value, dict)
                or type(value.get("returncode")) is not int
                or not isinstance(value.get("stdout"), str)
                or not isinstance(value.get("stderr"), str)
            ):
                raise PreloadedWorkerError("GPU Wiki index worker returned an invalid result")
            return subprocess.CompletedProcess(
                argv, value["returncode"], value["stdout"].encode(), value["stderr"].encode()
            )
        except TimeoutError as error:
            raise PreloadedWorkerError("GPU Wiki natural-language query timed out") from error
        except (OSError, ValueError) as error:
            raise PreloadedWorkerError(f"GPU Wiki index query failed: {error}") from error

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        process = self._process
        if process is not None:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=5)
            if process.stdout is not None:
                process.stdout.close()
        self._directory.cleanup()
        self._stderr.close()


def stop_query(pid: int, *, force: bool = False) -> None:
    """Stop only this query's group; allow its CLI cleanup to run first."""
    with contextlib.suppress(ProcessLookupError):
        os.killpg(pid, signal.SIGKILL if force else signal.SIGTERM)


def reap_queries(children: dict[int, tuple[float | None, bool]]) -> None:
    now = time.monotonic()
    for pid, (deadline, terminating) in tuple(children.items()):
        if deadline is not None and now >= deadline:
            stop_query(pid, force=terminating)
            children[pid] = (now + 2, True)
        try:
            finished, _ = os.waitpid(pid, os.WNOHANG)
        except ChildProcessError:
            finished = pid
        if finished:
            stop_query(pid, force=True)
            del children[pid]
