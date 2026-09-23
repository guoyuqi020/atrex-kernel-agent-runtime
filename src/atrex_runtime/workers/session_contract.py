"""Materialize the live Agent-facing tool, environment, and limit contract."""

from __future__ import annotations

import json
import shutil
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from ..filesystem import make_tree_read_only
from ..serialization import write_canonical_json

RUNTIME_CONTRACT_RELATIVE_PATH = Path("input/runtime-contract")
RUNTIME_CONTRACT_ENVIRONMENT_KEY = "ATREX_RUNTIME_CONTRACT_PATH"
RUNTIME_CONTRACT_VERSION: Literal[1] = 1

_GATEWAY_OPERATIONS = frozenset(
    {
        "evaluate",
        "profile",
        "dev",
        "check",
        "disassemble",
        "env",
    }
)
_COMMAND_BINDINGS: dict[str, dict[str, object]] = {
    "gateway-execute": {
        "kind": "gateway",
        "operations": sorted(_GATEWAY_OPERATIONS),
    },
    "kernel-artifact-read": {
        "kind": "runtime-query",
        "operation": "kernel_artifact_read",
    },
    "result-artifact-read": {
        "kind": "runtime-query",
        "operation": "result_artifact_read",
    },
    "update-direction": {
        "kind": "runtime-journal",
        "operation": "direction_update",
    },
    "list-directions": {
        "kind": "runtime-journal",
        "operation": "directions_list",
    },
    "load-direction": {
        "kind": "runtime-journal",
        "operation": "direction_load",
    },
    "record-experiment": {
        "kind": "runtime-journal",
        "operation": "experiment_record",
    },
    "list-experiments": {
        "kind": "runtime-journal",
        "operation": "experiments_list",
    },
    "load-experiment": {
        "kind": "runtime-journal",
        "operation": "experiment_load",
    },
    "attempt-report": {
        "kind": "runtime-terminal",
        "operation": "attempt_report",
    },
    "runtime-contract": {
        "kind": "local-discovery",
    },
}


@dataclass(frozen=True, slots=True)
class SessionContractPolicy:
    """Runtime-owned policy needed to describe one future Optimizer Session."""

    agent_backend: str
    session_timeout_seconds: float
    usage_unit: Literal["provider_tokens", "credits"]
    usage_budget: float
    max_attempt_report_bytes: int
    wiki_available: bool = False

    def __post_init__(self) -> None:
        if self.agent_backend not in {"claude", "codex", "qodercli", "pi"}:
            raise ValueError("Runtime contract Agent backend is unsupported")
        if (
            self.session_timeout_seconds <= 0
            or self.usage_budget <= 0
            or self.max_attempt_report_bytes <= 0
        ):
            raise ValueError("Runtime contract limits must be positive")


def materialize_session_contract(
    workspace: Path,
    *,
    phase: Literal["framework_baseline", "optimization_attempt"],
    dsl: str,
    hardware_target: str,
    agent_backend: str,
    model: str | None,
    session_timeout_seconds: float,
    usage_unit: str,
    usage_budget: float,
    max_attempt_report_bytes: int,
    wiki_available: bool,
    relative_path: Path = RUNTIME_CONTRACT_RELATIVE_PATH,
) -> Path:
    """Create one immutable, credential-free contract for the current Session.

    Runtime owns the live wire schemas and enforced limits. The Agent Bundle may
    project them through its own CLI or native-tool adapter, but must not copy a
    stale snapshot into versioned Prompts.
    """
    if (
        relative_path.is_absolute()
        or relative_path == Path(".")
        or ".." in relative_path.parts
        or not relative_path.is_relative_to("input")
    ):
        raise ValueError("Runtime contract location must be a safe path below input/")
    destination = workspace / relative_path
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Runtime contract already exists: {destination}")
    if session_timeout_seconds <= 0 or usage_budget <= 0 or max_attempt_report_bytes <= 0:
        raise ValueError("Runtime contract limits must be positive")
    if usage_unit not in {"provider_tokens", "credits"}:
        raise ValueError("Runtime contract usage unit is unsupported")

    destination.mkdir(parents=True, mode=0o700)
    try:
        # Import lazily: gateway.protocol validates Attempt reports from this
        # package, while workers.__init__ exports Core launchers.
        from ..gateway.protocol import gateway_agent_request_schema

        write_canonical_json(
            destination / "tools.json",
            {
                "schema_version": RUNTIME_CONTRACT_VERSION,
                "discovery": {
                    "command": (
                        "python3 agent/optimizer/src/runtime_tools.py runtime-contract "
                        "--output scratch/runtime-contract.json"
                    ),
                    "policy": "query_on_demand_do_not_copy_into_prompts",
                },
                "bindings": _COMMAND_BINDINGS,
                "gateway": gateway_agent_request_schema(
                    allowed_operations=_GATEWAY_OPERATIONS
                ),
            },
        )
        write_canonical_json(
            destination / "environment.json",
            {
                "schema_version": RUNTIME_CONTRACT_VERSION,
                "phase": phase,
                "dsl": dsl,
                "hardware_target": hardware_target,
                "agent_backend": agent_backend,
                "model": model,
                "services": {
                    "gateway": True,
                    "wiki": wiki_available,
                },
                "paths": {
                    "read_only": ["agent/", "input/", "prompts/", "skills/"],
                    "writable": ["scratch/", "tools/", "work/"],
                    "ephemeral": ["scratch/", "work/"],
                },
            },
        )
        write_canonical_json(
            destination / "limits.json",
            {
                "schema_version": RUNTIME_CONTRACT_VERSION,
                "session_timeout_seconds": session_timeout_seconds,
                "provider_usage": {
                    "unit": usage_unit,
                    "budget": usage_budget,
                },
                "attempt_report_max_bytes": max_attempt_report_bytes,
            },
        )
        make_tree_read_only(destination)
        return destination
    except BaseException:
        if destination.exists() and not destination.is_symlink():
            for path in destination.rglob("*"):
                with suppress(OSError):
                    path.chmod(0o700 if path.is_dir() else 0o600)
            destination.chmod(0o700)
            shutil.rmtree(destination, ignore_errors=True)
        raise


def runtime_contract_environment(path: Path) -> dict[str, str]:
    """Return the one stable environment binding consumed by Agent Source."""
    if path.is_symlink() or not path.is_dir():
        raise ValueError("Runtime contract path must be a real directory")
    # Parse every file at the trusted boundary before exposing the directory.
    for name in ("tools.json", "environment.json", "limits.json"):
        file = path / name
        if file.is_symlink() or not file.is_file():
            raise ValueError(f"Runtime contract is missing {name}")
        value = json.loads(file.read_text(encoding="utf-8"))
        if not isinstance(value, dict) or value.get("schema_version") != 1:
            raise ValueError(f"Runtime contract {name} is invalid")
    return {RUNTIME_CONTRACT_ENVIRONMENT_KEY: str(path)}
