"""Bounded ownership of one worker process tree."""

from __future__ import annotations

import os
import signal
import subprocess
import tempfile
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO


@dataclass(frozen=True, slots=True)
class BoundedProcessConfig:
    """Wall-time, shutdown, and output acquisition limits."""

    timeout_seconds: float
    terminate_grace_seconds: float
    max_output_bytes: int

    def __post_init__(self) -> None:
        if self.timeout_seconds <= 0 or self.terminate_grace_seconds <= 0:
            raise ValueError("Worker process timeouts must be positive")
        if self.max_output_bytes <= 0:
            raise ValueError("Worker process output byte limit must be positive")


@dataclass(frozen=True, slots=True)
class BoundedProcessResult:
    """Exit status and bounded UTF-8 diagnostics from a reaped process tree."""

    returncode: int
    stdout: str
    stderr: str


class BoundedProcessRunner:
    """Start a process group and ensure no owned descendants survive the call."""

    def __init__(self, config: BoundedProcessConfig) -> None:
        self._config = config

    def run(
        self,
        argv: tuple[str, ...],
        *,
        cwd: Path,
        stdin: bytes | None,
    ) -> BoundedProcessResult:
        """Execute one argv without a shell and acquire bounded output."""
        if not argv:
            raise ValueError("Worker process argv cannot be empty")
        with tempfile.TemporaryFile() as stdout_file, tempfile.TemporaryFile() as stderr_file:
            process = subprocess.Popen(
                argv,
                cwd=cwd,
                stdin=subprocess.PIPE if stdin is not None else subprocess.DEVNULL,
                stdout=stdout_file,
                stderr=stderr_file,
                start_new_session=True,
            )
            try:
                process.communicate(stdin, timeout=self._config.timeout_seconds)
            except subprocess.TimeoutExpired as error:
                self._terminate_process_group(process)
                raise TimeoutError("worker process exceeded its wall-time limit") from error
            self._terminate_process_group(process)
            if process.returncode is None:
                raise AssertionError("reaped worker process has no return code")
            return BoundedProcessResult(
                process.returncode,
                self._read_output(stdout_file),
                self._read_output(stderr_file),
            )

    def _terminate_process_group(self, process: subprocess.Popen[bytes]) -> None:
        if process.poll() is not None:
            self._signal_process_group(process.pid, signal.SIGKILL)
            return
        self._signal_process_group(process.pid, signal.SIGTERM)
        with suppress(subprocess.TimeoutExpired):
            process.wait(timeout=self._config.terminate_grace_seconds)
        self._signal_process_group(process.pid, signal.SIGKILL)
        if process.poll() is None:
            process.wait()

    @staticmethod
    def _signal_process_group(process_group_id: int, requested_signal: signal.Signals) -> None:
        with suppress(ProcessLookupError):
            os.killpg(process_group_id, requested_signal)

    def _read_output(self, stream: BinaryIO) -> str:
        stream.seek(0)
        value = stream.read(self._config.max_output_bytes + 1)
        suffix = "\n[truncated]" if len(value) > self._config.max_output_bytes else ""
        return value[: self._config.max_output_bytes].decode(errors="replace") + suffix
