"""Authoritative state storage interfaces and providers."""

from .base import Registry
from .sqlite import SqliteRegistry

__all__ = ["Registry", "SqliteRegistry"]
