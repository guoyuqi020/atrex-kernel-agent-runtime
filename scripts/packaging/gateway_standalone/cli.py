"""CLI wrappers; never add private dependencies to unrelated Python processes."""

import importlib
import importlib.metadata
import json
from pathlib import Path

from . import activate


def server_main() -> None:
    activate()
    from app.cli import main

    main()


def client_main() -> None:
    activate()
    from atrex_gateway_client.cli import main

    main()


def check_main() -> None:
    vendor = activate()
    modules = (
        "app.cli",
        "atrex_gateway_client.cli",
        "fastapi",
        "jsonschema",
        "pydantic_settings",
        "uvicorn",
        "pydantic_core",
        "httptools",
        "yaml",
        "rpds",
        "uvloop",
        "watchfiles",
        "websockets",
    )
    for name in modules:
        module = importlib.import_module(name)
        if not module.__file__ or not Path(module.__file__).resolve().is_relative_to(vendor):
            raise RuntimeError(f"Dependency {name} did not load from the private bundle")
    manifest = json.loads(Path(__file__).with_name("manifest.json").read_text())
    for wheel in manifest["wheels"]:
        actual = importlib.metadata.version(wheel["name"])
        if actual != wheel["version"]:
            raise RuntimeError(f"Unexpected {wheel['name']} version: {actual}")
    print(
        json.dumps(
            {
                "status": "ok",
                "version": manifest["version"],
                "target": manifest["target"],
                "bundled_distributions": len(manifest["wheels"]),
            },
            indent=2,
        )
    )
