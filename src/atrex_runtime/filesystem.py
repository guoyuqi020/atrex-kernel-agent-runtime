"""Shared permission transitions for private Runtime directory trees."""

from __future__ import annotations

import os
from pathlib import Path


def make_tree_owner_writable(root: Path) -> None:
    """Give only the owner read/write access to files and full access to directories."""
    _set_tree_modes(root, directory_mode=0o700, file_mode=0o600)


def make_tree_read_only(root: Path) -> None:
    """Make files owner-readable and directories owner-readable/traversable."""
    _set_tree_modes(root, directory_mode=0o500, file_mode=0o400)


def _set_tree_modes(root: Path, *, directory_mode: int, file_mode: int) -> None:
    for child in root.rglob("*"):
        os.chmod(child, directory_mode if child.is_dir() else file_mode)
    os.chmod(root, directory_mode)


__all__ = ["make_tree_owner_writable", "make_tree_read_only"]
