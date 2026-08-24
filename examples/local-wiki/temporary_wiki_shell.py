#!/usr/bin/env python3
"""Open a disposable Core-compatible shell scoped only to the local GPU Wiki."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import queue
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import uvicorn

from atrex_runtime.artifacts import LocalArtifactStore
from atrex_runtime.config import RuntimeSettings
from atrex_runtime.domain.ids import (
    new_attempt_id,
    new_campaign_id,
    new_epoch_id,
    new_kernel_agent_revision_id,
    new_kernel_revision_id,
    new_lineage_id,
    parse_artifact_digest,
)
from atrex_runtime.domain.models import Dsl
from atrex_runtime.gateway.control import (
    BootstrapGatewaySubject,
    GatewayCapabilityPolicy,
    GatewayOperation,
    SqliteGatewayControl,
)
from atrex_runtime.knowledge.client import HttpGpuWikiClient, HttpxGpuWikiTransport
from atrex_runtime.knowledge.proxy import WikiProxyAsgiApp, WikiProxyLimits, WikiProxyService
from atrex_runtime.registry.sqlite import SqliteRegistry
from atrex_runtime.secrets import read_capability_signing_key, required_secret

_DIGEST = parse_artifact_digest("sha256:" + "0" * 64)


def _arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--shell", choices=("zsh", "bash"), default="zsh")
    parser.add_argument("--dsl", choices=tuple(item.value for item in Dsl), default="triton")
    parser.add_argument("--operator", default="wiki_debug")
    parser.add_argument("--hardware-target", default="nvidia-h100")
    return parser.parse_args(argv)


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True),
        encoding="utf-8",
    )


def _prepare_workspace(
    root: Path,
    core_repository: Path,
    *,
    subject: BootstrapGatewaySubject,
    dsl: Dsl,
) -> tuple[Path, dict[str, str]]:
    workspace = root / "workspace"
    optimizer = workspace / "agent/optimizer"
    shutil.copytree(
        core_repository,
        optimizer,
        ignore=shutil.ignore_patterns(
            ".git", ".mypy_cache", ".pytest_cache", ".ruff_cache", "__pycache__", "*.pyc"
        ),
    )
    for relative in (
        "input/kernel",
        "input/agent-problem",
        "input/evidence/epochs/00000001/attempts",
        "work/kernel",
        "scratch",
        "sessions",
    ):
        (workspace / relative).mkdir(parents=True, exist_ok=True)

    prompt = "# Temporary Wiki session\n\nUse wiki-query for complete safe GPU Wiki Records.\n"
    prompt_path = workspace / "input/evidence/instructions.md"
    prompt_path.write_text(prompt, encoding="utf-8")
    _write_json(
        workspace / "input/evidence/manifest.json",
        {
            "schema_version": 1,
            "role": "optimizer",
            "lineage_checkpoint": str(subject.evidence_digest),
            "prompt_fragment_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "through_completed_epoch": 0,
            "current_epoch": {
                "number": 1,
                "snapshot_digest": str(subject.evidence_digest),
                "status": "in_progress",
                "trigger": None,
            },
            "visibility": {
                "completed_epochs": "promoted_lineage",
                "current_attempts_before": 1,
            },
        },
    )
    manifest_path = workspace / "attempt.json"
    _write_json(
        manifest_path,
        {
            "schema_version": 6,
            "attempt_id": str(subject.attempt_id),
            "kernel_agent_revision_id": str(subject.kernel_agent_revision_id),
            "input_kernel_revision_id": str(new_kernel_revision_id()),
            "input_kernel_digest": str(subject.input_kernel_digest),
            "epoch_evidence_checkpoint": str(subject.evidence_digest),
            "attempt_evidence_digest": str(subject.evidence_digest),
            "optimizer_digest": str(_DIGEST),
            "dsl": dsl.value,
            "context": {
                "campaign_id": str(subject.campaign_id),
                "lineage_id": str(subject.lineage_id),
                "epoch_id": str(subject.epoch_id),
                "epoch_number": 1,
                "attempt_ordinal": 1,
                "operator": subject.operator,
                "hardware_target": subject.hardware_target,
                "evaluation_contract_digest": str(subject.evaluation_contract_digest),
                "agent_problem_digest": str(_DIGEST),
            },
            "paths": {
                "input_kernel": "input/kernel",
                "working_kernel": "work/kernel",
                "evidence": "input/evidence",
                "agent_problem": "input/agent-problem",
                "optimizer": "agent/optimizer",
            },
        },
    )
    example = Path(__file__).resolve().parent
    shutil.copy2(example / "wiki-query.json", workspace / "scratch/wiki-query.json")
    environment = {
        "ATREX_CORE_PHASE": "optimization_attempt",
        "ATREX_ATTEMPT_MANIFEST": str(manifest_path),
        "ATREX_ATTEMPT_REPORT_PATH": str(workspace / "scratch/attempt-report.json"),
        "ATREX_EVIDENCE_PROMPT_PATH": str(prompt_path),
        "ATREX_OPTIMIZER_REPOSITORY": str(optimizer),
        "ATREX_SESSION_TIMEOUT_SECONDS": "3600",
        "ATREX_TOKEN_BUDGET": "1",
        "ATREX_TOKEN_USAGE_REPORT": str(workspace / "scratch/token-usage.json"),
        "ATREX_SESSION_TRACE_PATH": str(workspace / "sessions/core"),
    }
    return workspace, environment


def _serve(
    root: Path,
    settings: RuntimeSettings,
    subject: BootstrapGatewaySubject,
    signing_key: bytes,
) -> tuple[uvicorn.Server, threading.Thread, socket.socket, int, str]:
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    port = int(listener.getsockname()[1])
    started: queue.Queue[tuple[uvicorn.Server, str] | BaseException] = queue.Queue(maxsize=1)

    def run_proxy() -> None:
        registry: SqliteRegistry | None = None
        control: SqliteGatewayControl | None = None
        try:
            wiki = settings.gpu_wiki
            if wiki is None:
                raise ValueError("temporary Wiki shell requires gpu_wiki configuration")
            registry = SqliteRegistry(root / "registry.sqlite")
            control = SqliteGatewayControl(
                root / "gateway.sqlite", registry, signing_key=signing_key
            )
            artifacts = LocalArtifactStore(root / "artifacts")
            capability = control.issue_bootstrap(
                subject,
                GatewayCapabilityPolicy(
                    frozenset({GatewayOperation.WIKI_QUERY}),
                    1,
                    subject.created_at + timedelta(hours=2),
                ),
            )
            limits = WikiProxyLimits(wiki.max_proxy_request_bytes, wiki.max_query_bytes)
            app = WikiProxyAsgiApp(
                WikiProxyService(
                    control,
                    control,
                    registry,
                    artifacts,
                    HttpGpuWikiClient(
                        HttpxGpuWikiTransport(wiki.base_url),
                        bearer_token=required_secret(os.environ, wiki.bearer_token_env),
                        timeout_seconds=wiki.timeout_seconds,
                        max_response_bytes=wiki.max_response_bytes,
                    ),
                    limits,
                    registry,
                ),
                limits,
            )
            server = uvicorn.Server(
                uvicorn.Config(app, log_level="warning", access_log=False, lifespan="off")
            )
            started.put((server, capability.token))
            server.run(sockets=[listener])
        except BaseException as error:
            if started.empty():
                started.put(error)
            raise
        finally:
            if control is not None:
                control.close()
            if registry is not None:
                registry.close()

    thread = threading.Thread(
        target=run_proxy,
        name="temporary-wiki-proxy",
        daemon=True,
    )
    thread.start()
    value = started.get(timeout=10)
    if isinstance(value, BaseException):
        raise RuntimeError("temporary Wiki Proxy failed to initialize") from value
    server, capability = value
    deadline = time.monotonic() + 10
    while not server.started:
        if not thread.is_alive():
            raise RuntimeError("temporary Wiki Proxy failed to start")
        if time.monotonic() >= deadline:
            raise TimeoutError("temporary Wiki Proxy did not become ready")
        time.sleep(0.02)
    return server, thread, listener, port, capability


def main(argv: list[str] | None = None) -> int:
    args = _arguments(argv)
    settings = RuntimeSettings.from_file(args.config)
    wiki = settings.gpu_wiki
    if wiki is None:
        raise ValueError("temporary Wiki shell requires gpu_wiki configuration")
    core_repository = Path(__file__).resolve().parents[2] / "src/atrex-kernel-agent-core"
    if not (core_repository / "atrex-bundle.json").is_file():
        raise FileNotFoundError(f"Core Agent Bundle is unavailable: {core_repository}")
    shell = shutil.which(args.shell, path="/bin:/usr/bin:/usr/local/bin:/opt/homebrew/bin")
    if shell is None:
        raise FileNotFoundError(f"requested shell is unavailable: {args.shell}")

    signing_key = read_capability_signing_key(
        os.environ,
        settings.gateway_proxy.capability_signing_key_env,
    )
    created_at = datetime.now(UTC)
    dsl = Dsl(args.dsl)
    subject = BootstrapGatewaySubject(
        attempt_id=new_attempt_id(),
        campaign_id=new_campaign_id(),
        lineage_id=new_lineage_id(),
        epoch_id=new_epoch_id(),
        kernel_agent_revision_id=new_kernel_agent_revision_id(),
        operator=args.operator,
        hardware_target=args.hardware_target,
        dsl=dsl,
        evaluation_contract_digest=_DIGEST,
        input_kernel_digest=_DIGEST,
        evidence_digest=_DIGEST,
        created_at=created_at,
    )

    state_parent = Path(__file__).resolve().parents[2] / "local-wiki/state"
    state_parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with tempfile.TemporaryDirectory(prefix="temporary-lineage-", dir=state_parent) as temporary:
        root = Path(temporary).resolve()
        server: uvicorn.Server | None = None
        thread: threading.Thread | None = None
        listener: socket.socket | None = None
        try:
            server, thread, listener, port, capability = _serve(
                root,
                settings,
                subject,
                signing_key,
            )
            workspace, injected = _prepare_workspace(
                root,
                core_repository,
                subject=subject,
                dsl=dsl,
            )
            endpoint = f"http://127.0.0.1:{port}"
            environment: dict[str, Any] = dict(os.environ)
            environment.update(injected)
            environment.update(
                {
                    "ATREX_GATEWAY_PROXY_URL": endpoint,
                    "ATREX_GATEWAY_CAPABILITY": capability,
                    "ATREX_WIKI_PROXY_URL": endpoint,
                    "ATREX_WIKI_CAPABILITY": capability,
                }
            )
            print(
                "\n".join(
                    (
                        "Temporary Local Wiki Agent shell",
                        f"Lineage:  {subject.lineage_id}",
                        f"Attempt:  {subject.attempt_id}",
                        f"Workspace: {workspace}",
                        "Bootstrap: skipped",
                        "Agate Gateway: not required",
                        "Try:",
                        "  python agent/optimizer/src/runtime_tools.py wiki-query "
                        "--request scratch/wiki-query.json",
                        "Exit the shell to destroy the temporary Lineage, Proxy, and Workspace.",
                        "",
                    )
                ),
                flush=True,
            )
            return subprocess.run(
                (shell, "-i"),
                cwd=workspace,
                env=environment,
                check=False,
            ).returncode
        finally:
            if server is not None:
                server.should_exit = True
            if thread is not None:
                thread.join(timeout=10)
            if listener is not None:
                listener.close()
            print("Temporary Local Wiki Lineage destroyed.", flush=True)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
