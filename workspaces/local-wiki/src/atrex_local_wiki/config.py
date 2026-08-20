"""Strict file configuration for the local Wiki test service."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class LocalWikiSettings(BaseModel):
    """All deployment-varying local Wiki settings."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1] = 1
    host: str = Field(min_length=1)
    port: int = Field(ge=1, le=65_535)
    reference_root: Path
    store_root: Path
    database: Path
    python_executable: Path = Field(default_factory=lambda: Path(sys.executable).resolve())
    agent_cli: str | None = Field(default=None, min_length=1)
    query_timeout_seconds: int | None = Field(default=None, gt=0)
    bearer_token_env: str | None = None
    max_request_bytes: int = Field(gt=0)
    max_results: int | None = Field(default=None, gt=0)
    max_response_bytes: int = Field(gt=0)

    @model_validator(mode="after")
    def _validate_paths(self) -> Self:
        if not self.reference_root.is_dir():
            raise ValueError(f"reference_root is not a directory: {self.reference_root}")
        if self.store_root.exists() and not self.store_root.is_dir():
            raise ValueError(f"store_root is not a directory: {self.store_root}")
        if self.store_root == self.reference_root:
            raise ValueError("store_root must be separate from the pinned reference_root")
        if not self.python_executable.is_absolute() or not self.python_executable.is_file():
            raise ValueError("python_executable must be an existing absolute file")
        if self.database.exists() and not self.database.is_file():
            raise ValueError(f"database is not a file: {self.database}")
        return self

    @classmethod
    def from_file(cls, path: str | Path) -> Self:
        """Load JSON and resolve relative paths from the configuration directory."""
        config_path = Path(path).resolve()
        value = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("local Wiki config root must be an object")
        base = config_path.parent
        for name in ("reference_root", "store_root", "database", "python_executable"):
            raw = value.get(name)
            if isinstance(raw, str) and not Path(raw).is_absolute():
                value[name] = str((base / raw).resolve())
        return cls.model_validate(value)
