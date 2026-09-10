"""Shared GDN service bindings; task inputs and control-plane state stay separate."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import secrets
import tempfile
from pathlib import Path


def config_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def service_config(service: Path) -> Path:
    """Only attach to an explicitly prepared, unchanged GDN service workspace."""
    config = service / "runtime.json"
    marker = service / "service.json"
    if not config.is_file() or not marker.is_file():
        raise SystemExit(f"Prepare the shared service first: {service}")
    value = json.loads(marker.read_text())
    if (value.get("kind") != "gdn-shared-runtime" or
            value.get("runtime_config_sha256") != config_digest(config)):
        raise SystemExit("Shared Runtime config changed; do not silently rebind existing tasks")
    return config


def binding_for(workspace: Path, service: Path) -> dict:
    if workspace.is_relative_to(service) or service.is_relative_to(workspace):
        raise SystemExit("Task and service workspaces must be separate, non-nested directories")
    config = service_config(service)
    return {
        "schema_version": 1,
        "service_workspace": os.path.relpath(service, workspace),
        "runtime_config_sha256": config_digest(config),
    }


def resolve_service(workspace: Path, requested: Path | None = None) -> tuple[Path, Path]:
    """Resolve a frozen task binding, or retain the standalone workspace behavior."""
    binding = workspace / "service-binding.json"
    if binding.is_file():
        value = json.loads(binding.read_text())
        if value.get("schema_version") != 1:
            raise SystemExit("Unsupported GDN service binding")
        service = (workspace / value["service_workspace"]).resolve()
        if requested is not None and requested.resolve() != service:
            raise SystemExit("--service-workspace differs from the task's frozen service binding")
        if value != binding_for(workspace, service):
            raise SystemExit("Task service binding no longer matches the shared Runtime config")
        if (workspace / "runtime.json").exists() or (workspace / "runtime-secrets.json").exists():
            raise SystemExit("Attached task must not contain a second Runtime config or secrets")
        return service, service_config(service)
    if requested is not None:
        raise SystemExit("Task is not attached; prepare a new workspace with --service-workspace")
    config = workspace / "runtime.json"
    if not config.is_file():
        raise SystemExit(f"Run scripts/gdn/prepare.py --workspace {workspace} first")
    if (workspace / "service.json").is_file():
        service_config(workspace)
    return workspace, config


def load_service_secrets(service: Path) -> dict[str, str]:
    """Publish one complete key file even when several task runners start together."""
    path = service / "runtime-secrets.json"
    names = ("ATREX_CAPABILITY_SIGNING_KEY", "ATREX_ADMIN_BEARER_TOKEN")
    descriptor = os.open(service / ".runtime-secrets.lock", os.O_RDWR | os.O_CREAT, 0o600)
    with os.fdopen(descriptor, "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not path.exists():
            value = dict(zip(
                names, (secrets.token_urlsafe(48), secrets.token_hex(32)), strict=True,
            ))
            fd, temporary = tempfile.mkstemp(prefix=".runtime-secrets-", dir=service)
            try:
                with os.fdopen(fd, "w") as output:
                    json.dump(value, output)
                    output.flush()
                    os.fsync(output.fileno())
                os.replace(temporary, path)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
        try:
            value = json.loads(path.read_text())
        except (OSError, ValueError) as error:
            raise SystemExit(
                f"Cannot read Runtime secrets: {path}; refusing to regenerate"
            ) from error
        if not isinstance(value, dict) or set(value) != set(names) or any(
            not isinstance(value[name], str) or not value[name] for name in names
        ):
            raise SystemExit(f"Invalid Runtime secrets: {path}; refusing to regenerate")
        return value
