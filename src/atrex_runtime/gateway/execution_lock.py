"""Process-safe execution leases released automatically when a Runtime dies."""

from __future__ import annotations

import fcntl
import hashlib
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def execution_lock(directory: Path, key: str) -> Iterator[bool]:
    """Try to hold one task's lease; never unlink a possibly shared lock inode."""
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    path = directory / hashlib.sha256(key.encode()).hexdigest()
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield False
        else:
            try:
                yield True
            finally:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)
