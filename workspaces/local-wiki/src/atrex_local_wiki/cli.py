"""Command-line entry point for the local GPU Wiki test service."""

from __future__ import annotations

import argparse
import importlib
import os
from collections.abc import Callable
from typing import Protocol, cast

from .app import LocalWikiApplication, build_application
from .config import LocalWikiSettings


class UvicornRun(Protocol):
    """Subset of the Uvicorn entry point used by this workspace."""

    def __call__(
        self,
        app: LocalWikiApplication,
        *,
        host: str,
        port: int,
        lifespan: str,
    ) -> None:
        """Serve the local Wiki until shutdown."""
        ...


def main(argv: list[str] | None = None) -> int:
    """Load one strict config and serve the local test double."""
    parser = argparse.ArgumentParser(prog="atrex-local-gpu-wiki")
    parser.add_argument("serve", choices=("serve",))
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    settings = LocalWikiSettings.from_file(args.config)
    app = build_application(settings, os.environ)
    try:
        uvicorn = importlib.import_module("uvicorn")
        run = cast(UvicornRun, cast(Callable[..., object], uvicorn.__dict__["run"]))
        run(app, host=settings.host, port=settings.port, lifespan="on")
    finally:
        app.close()
    return 0
