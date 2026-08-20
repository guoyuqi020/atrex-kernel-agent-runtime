"""Shared non-executing Git command, archive, and extraction boundary."""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
import tarfile
from pathlib import Path, PurePosixPath


class SafeGitImporter:
    """Run one trusted Git executable and extract only plain bounded tar trees."""

    def __init__(
        self,
        executable: str | Path,
        *,
        timeout_seconds: float,
        max_archive_bytes: int,
        label: str,
    ) -> None:
        path = Path(executable)
        if not path.is_absolute():
            raise ValueError("Git executable must be absolute")
        if timeout_seconds <= 0 or max_archive_bytes <= 0 or not label.strip():
            raise ValueError("Git import policy must be positive and labeled")
        self._executable = str(path)
        self._timeout_seconds = timeout_seconds
        self._max_archive_bytes = max_archive_bytes
        self._label = label

    def run(self, arguments: tuple[str, ...]) -> bytes:
        """Run Git without prompting and return stdout or a bounded diagnostic."""
        try:
            process = subprocess.run(
                (self._executable, *arguments),
                check=False,
                capture_output=True,
                timeout=self._timeout_seconds,
                env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise RuntimeError(f"Git {self._label} import failed to execute") from error
        if process.returncode != 0:
            diagnostic = process.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(
                f"Git {self._label} import failed with {process.returncode}: {diagnostic[:1000]}"
            )
        return process.stdout

    @staticmethod
    def object_id(payload: bytes) -> str:
        """Decode one ASCII Git object identity."""
        try:
            return payload.decode("ascii").strip()
        except UnicodeDecodeError as error:
            raise ValueError("Git returned a non-ASCII object identity") from error

    def archive(
        self,
        repository: Path,
        revision: str,
        destination: Path,
        *,
        paths: tuple[str, ...] = (),
    ) -> int:
        """Write and size-check a plain Git tree archive, optionally restricted to paths."""
        try:
            with destination.open("wb") as output:
                process = subprocess.run(
                    (
                        self._executable,
                        "-C",
                        str(repository),
                        "archive",
                        "--format=tar",
                        revision,
                        *paths,
                    ),
                    check=False,
                    stdout=output,
                    stderr=subprocess.PIPE,
                    timeout=self._timeout_seconds,
                    env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
                )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise RuntimeError(f"Git {self._label} archive failed to execute") from error
        if process.returncode != 0:
            diagnostic = process.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"Git {self._label} archive failed: {diagnostic[:1000]}")
        size = destination.stat().st_size
        if size > self._max_archive_bytes:
            raise ValueError(f"Git {self._label} archive exceeds byte limit")
        return size

    @staticmethod
    def extract(archive_path: Path, destination: Path) -> None:
        """Extract only normalized directories and regular files without links."""
        with tarfile.open(archive_path, mode="r:") as archive:
            for member in archive.getmembers():
                relative = PurePosixPath(member.name)
                if relative.is_absolute() or ".." in relative.parts:
                    raise ValueError("Git archive contains an unsafe path")
                target = destination.joinpath(*relative.parts)
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True, mode=0o700)
                    continue
                if not member.isfile():
                    raise ValueError("Git archive contains a link or special file")
                target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                source = archive.extractfile(member)
                if source is None:
                    raise ValueError("Git archive regular file has no payload")
                with source, target.open("wb") as output:
                    shutil.copyfileobj(source, output)
                os.chmod(target, 0o700 if member.mode & stat.S_IXUSR else 0o600)
