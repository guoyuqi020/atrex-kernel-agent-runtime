"""Trusted supervisor for untrusted, versioned Agent Workflow programs."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path, PurePosixPath
from typing import cast
from uuid import uuid4

import anyio

from ..artifacts.local import ArtifactKind, LocalArtifactStore
from ..ports import AgentWorkflowOperationHandler, RunAgentWorkflowRequest
from ..serialization import canonical_json_bytes
from ..workers.launcher import WorkerLauncher
from .revision import KernelAgentBundleManifestV1


class SandboxedAgentWorkflowRunner:
    """Run Agent-owned Epoch orchestration against capability-bounded Runtime services.

    The Workflow remains an untrusted subprocess. Runtime sends one immutable Epoch
    context line, then serves a bounded request/response JSONL protocol over stdio.
    Workflow code never imports controller objects or receives Registry access.
    """

    _MAX_PROTOCOL_LINE_BYTES = 256 * 1024
    _MAX_SERVICE_CALLS = 128

    def __init__(
        self,
        artifacts: LocalArtifactStore,
        launcher: WorkerLauncher,
        workspace_root: str | Path,
        *,
        command_prefix: tuple[str, ...],
    ) -> None:
        self._artifacts = artifacts
        self._launcher = launcher
        if not command_prefix:
            raise ValueError("Agent Workflow command prefix cannot be empty")
        self._command_prefix = command_prefix
        self._workspace_root = Path(workspace_root).resolve()
        self._workspace_root.mkdir(parents=True, exist_ok=True, mode=0o700)

    async def run(
        self,
        request: RunAgentWorkflowRequest,
        operations: AgentWorkflowOperationHandler,
    ) -> None:
        """Execute one Epoch Workflow until it commits or exits with an error."""
        if request.epoch_number <= 0:
            raise ValueError("Workflow Epoch number must be positive")
        if request.max_challengers < 0 or request.optimizer_attempt_budget <= 0:
            raise ValueError("Workflow limits are invalid")
        stored = self._artifacts.verify(request.revision.optimizer_digest)
        if stored.kind is not ArtifactKind.KERNEL_AGENT:
            raise ValueError("Agent revision does not reference a Kernel Agent Bundle")
        manifest = KernelAgentBundleManifestV1.from_file(stored.payload_path / "atrex-bundle.json")
        if manifest.workflow is None:
            raise ValueError("Agent revision does not declare an executable Workflow")

        command_relative = PurePosixPath(manifest.workflow.command)
        program_sha256 = self._program_sha256(
            stored.payload_path / "workflow",
            command_relative,
        )
        workspace = self._prepare_epoch_workspace(request)
        command = workspace / "agent" / Path(*command_relative.parts)
        environment = {
            "HOME": str(workspace / "scratch/home"),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONUNBUFFERED": "1",
        }
        argv = self._launcher.wrap(
            (*self._command_prefix, str(command)),
            workspace=workspace,
            environment=environment,
        )
        initial = self._run_request_value(request, program_sha256)
        (workspace / "input/workflow-context.json").write_bytes(canonical_json_bytes(initial))
        os.chmod(workspace / "input/workflow-context.json", 0o400)
        stderr_path = workspace / "scratch/stderr.log"
        calls_path = workspace / "scratch/protocol.jsonl"
        last_service_error: BaseException | None = None
        with (
            stderr_path.open("wb") as stderr_file,
            calls_path.open("w", encoding="utf-8") as calls_file,
        ):
            process = await anyio.open_process(
                argv,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=stderr_file,
                cwd=workspace,
            )
            assert process.stdin is not None
            assert process.stdout is not None
            await process.stdin.send(canonical_json_bytes(initial) + b"\n")
            buffer = bytearray()
            service_calls = 0
            while True:
                try:
                    chunk = await process.stdout.receive()
                except anyio.EndOfStream:
                    break
                buffer.extend(chunk)
                if len(buffer) > self._MAX_PROTOCOL_LINE_BYTES:
                    process.kill()
                    await process.wait()
                    raise ValueError("Agent Workflow protocol line exceeds byte limit")
                while b"\n" in buffer:
                    raw_line, _, remainder = buffer.partition(b"\n")
                    buffer = bytearray(remainder)
                    if not raw_line.strip():
                        continue
                    service_calls += 1
                    if service_calls > self._MAX_SERVICE_CALLS:
                        process.kill()
                        await process.wait()
                        raise ValueError("Agent Workflow exceeded Runtime service-call limit")
                    calls_file.write(raw_line.decode("utf-8", errors="replace") + "\n")
                    calls_file.flush()
                    response, service_error = await self._dispatch(
                        bytes(raw_line),
                        operations,
                        program_sha256=program_sha256,
                    )
                    if service_error is not None:
                        last_service_error = service_error
                    await process.stdin.send(canonical_json_bytes(response) + b"\n")
            if buffer.strip():
                process.kill()
                await process.wait()
                raise ValueError("Agent Workflow emitted an unterminated protocol record")
            await process.stdin.aclose()
            returncode = await process.wait()
        if returncode != 0:
            diagnostic = stderr_path.read_text(encoding="utf-8", errors="replace").strip()
            message = diagnostic or "no diagnostic"
            if last_service_error is not None:
                raise last_service_error
            raise ValueError(f"Agent Workflow program exited with {returncode}: {message}")

    async def _dispatch(
        self,
        raw_line: bytes,
        operations: AgentWorkflowOperationHandler,
        *,
        program_sha256: str,
    ) -> tuple[dict[str, object], BaseException | None]:
        request_id = "unknown"
        try:
            value = json.loads(raw_line)
            if not isinstance(value, dict):
                raise ValueError("Workflow service request must be an object")
            request_id_value = value.get("request_id")
            operation = value.get("operation")
            arguments = value.get("arguments", {})
            if not isinstance(request_id_value, str) or not request_id_value.strip():
                raise ValueError("Workflow service request_id must be non-empty")
            request_id = request_id_value
            if not isinstance(operation, str) or not operation.strip():
                raise ValueError("Workflow service operation must be non-empty")
            if not isinstance(arguments, dict):
                raise ValueError("Workflow service arguments must be an object")
            trusted_arguments: dict[str, object] = dict(cast(dict[str, object], arguments))
            trusted_arguments["_runtime_workflow_program_sha256"] = program_sha256
            result = await operations.execute_workflow_operation(
                operation,
                trusted_arguments,
            )
            return (
                {
                    "request_id": request_id,
                    "ok": True,
                    "result": dict(result),
                },
                None,
            )
        except BaseException as error:
            return (
                {
                    "request_id": request_id,
                    "ok": False,
                    "error": {
                        "type": type(error).__name__,
                        "message": str(error),
                    },
                },
                error,
            )

    def _prepare_epoch_workspace(self, request: RunAgentWorkflowRequest) -> Path:
        parent = self._workspace_root / str(request.epoch_id)
        parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        workspace = parent / f"run-{uuid4().hex}"
        workspace.mkdir(mode=0o700)
        self._artifacts.materialize(request.revision.optimizer_digest, workspace / "agent")
        (workspace / "input").mkdir(mode=0o700)
        (workspace / "scratch/home").mkdir(parents=True, mode=0o700)
        return workspace

    @staticmethod
    def _program_sha256(workflow_root: Path, command: PurePosixPath) -> str:
        """Hash the selected entry plus the complete Workflow support package."""
        digest = hashlib.sha256()
        selected = command.as_posix().encode("utf-8")
        digest.update(len(selected).to_bytes(8, "big"))
        digest.update(selected)
        for source in sorted(path for path in workflow_root.rglob("*") if path.is_file()):
            relative = source.relative_to(workflow_root).as_posix().encode("utf-8")
            payload = source.read_bytes()
            digest.update(len(relative).to_bytes(8, "big"))
            digest.update(relative)
            digest.update(len(payload).to_bytes(8, "big"))
            digest.update(payload)
        return digest.hexdigest()

    @staticmethod
    def _run_request_value(
        request: RunAgentWorkflowRequest,
        program_sha256: str,
    ) -> dict[str, object]:
        return {
            "schema_version": 1,
            "operation": "run_epoch",
            "context": {
                "kernel_agent_revision_id": request.revision.id,
                "dsl": request.revision.dsl.value,
                "epoch_id": request.epoch_id,
                "epoch_number": request.epoch_number,
                "workflow_program_sha256": program_sha256,
            },
            "limits": {
                "max_challengers": request.max_challengers,
                "optimizer_attempts": request.optimizer_attempt_budget,
            },
        }


__all__ = ["SandboxedAgentWorkflowRunner"]
