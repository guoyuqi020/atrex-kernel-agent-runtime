"""Sandbox-local dry-run validation for an Agent-owned Epoch Workflow."""

from __future__ import annotations

import argparse
import json
import os
import selectors
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal

from .revision import KernelAgentBundleManifestV1

_MAX_PROTOCOL_LINE_BYTES = 256 * 1024
_MAX_SERVICE_CALLS = 128
_SCENARIO_TIMEOUT_SECONDS = 10.0


class WorkflowCheckError(ValueError):
    """One actionable failure found while dry-running a Candidate Workflow."""

    def __init__(self, scenario: str, phase: str, detail: str) -> None:
        super().__init__(detail)
        self.scenario = scenario
        self.phase = phase


@dataclass(frozen=True, slots=True)
class WorkflowCheckScenario:
    """One deterministic Runtime response path exercised by the checker."""

    name: str
    epoch_number: int
    evolution: Literal["created", "no_change"]


class _DryRunServices:
    """Stateful, non-persistent projection of the trusted Epoch services."""

    def __init__(
        self,
        *,
        max_challengers: int,
        optimizer_attempt_budget: int,
        evolution: Literal["created", "no_change"],
    ) -> None:
        self.max_challengers = max_challengers
        self.optimizer_attempt_budget = optimizer_attempt_budget
        self.evolution = evolution
        self.challengers: dict[int, str] = {}
        self.trajectories: dict[tuple[str, int], tuple[int, int]] = {}
        self.next_attempt: dict[tuple[str, int], int] = {}
        self.attempt_ids: set[str] = set()
        self.kernel_ids: set[str] = set()
        self.attempt_count = 0
        self.selected_kernel: str | None = None
        self.selected_agent: str | None = None
        self.completed = False

    def execute(self, operation: str, arguments: Mapping[str, object]) -> dict[str, object]:
        if self.completed:
            raise ValueError("Workflow called a Runtime service after complete_epoch")
        if operation in {"replicate_active", "evolve_agent"}:
            ordinal = self._positive_int(arguments, "challenger_ordinal")
            if ordinal > self.max_challengers:
                raise ValueError(
                    f"Challenger {ordinal} exceeds max_challengers={self.max_challengers}"
                )
            if ordinal in self.challengers:
                raise ValueError(f"Challenger {ordinal} was materialized more than once")
            if operation == "evolve_agent" and self.evolution == "no_change":
                return {"kernel_agent_revision_id": None, "created": False}
            revision = f"agentrev_{ordinal:032x}"
            self.challengers[ordinal] = revision
            return {"kernel_agent_revision_id": revision, "created": True}
        if operation == "create_trajectory":
            branch = arguments.get("branch")
            ordinal = self._positive_int(arguments, "trajectory_ordinal")
            count = self._positive_int(arguments, "trajectory_count")
            capacity = self._positive_int(arguments, "attempt_capacity")
            if not isinstance(branch, str) or not branch:
                raise ValueError("create_trajectory requires a non-empty branch")
            self._validate_branch(branch)
            if ordinal > count:
                raise ValueError("trajectory_ordinal exceeds trajectory_count")
            key = (branch, ordinal)
            if key in self.trajectories:
                raise ValueError(f"Trajectory {branch}/{ordinal} was created more than once")
            for (existing_branch, _), existing in self.trajectories.items():
                if existing_branch == branch and existing != (count, capacity):
                    raise ValueError(
                        f"Branch {branch!r} uses inconsistent Trajectory topology"
                    )
            self.trajectories[key] = (count, capacity)
            self.next_attempt[key] = 1
            agent = (
                "agentrev_" + "0" * 32
                if branch == "active"
                else self.challengers[int(branch.removeprefix("challenger-"))]
            )
            return {
                "branch": branch,
                "trajectory_ordinal": ordinal,
                "attempt_capacity": capacity,
                "kernel_agent_revision_id": agent,
            }
        if operation == "run_attempts_parallel":
            launches = arguments.get("launches")
            if not isinstance(launches, list) or not launches:
                raise ValueError("run_attempts_parallel requires a non-empty launches array")
            outcomes: list[dict[str, object]] = []
            batch_keys: set[tuple[str, int]] = set()
            for raw in launches:
                if not isinstance(raw, dict):
                    raise ValueError("Attempt launch must be an object")
                branch = raw.get("branch")
                trajectory_ordinal = raw.get("trajectory_ordinal")
                attempt_ordinal = raw.get("attempt_ordinal")
                if (
                    not isinstance(branch, str)
                    or isinstance(trajectory_ordinal, bool)
                    or not isinstance(trajectory_ordinal, int)
                    or isinstance(attempt_ordinal, bool)
                    or not isinstance(attempt_ordinal, int)
                ):
                    raise ValueError("Attempt launch identity is invalid")
                key = (branch, trajectory_ordinal)
                if key not in self.trajectories:
                    raise ValueError(
                        f"Attempt names unknown Trajectory {branch}/{trajectory_ordinal}"
                    )
                if key in batch_keys:
                    raise ValueError(
                        f"Attempt batch repeats Trajectory {branch}/{trajectory_ordinal}"
                    )
                batch_keys.add(key)
                expected = self.next_attempt[key]
                if attempt_ordinal != expected:
                    raise ValueError(
                        f"Trajectory {branch}/{trajectory_ordinal} expected Attempt {expected}, "
                        f"got {attempt_ordinal}"
                    )
                capacity = self.trajectories[key][1]
                if attempt_ordinal > capacity:
                    raise ValueError(
                        f"Trajectory {branch}/{trajectory_ordinal} exceeded its capacity"
                    )
                input_kernel = raw.get("input_kernel_revision_id")
                if input_kernel is not None and input_kernel not in self.kernel_ids:
                    raise ValueError("Kernel route does not name a completed dry-run Attempt")
                input_state = raw.get("input_state_from_attempt_id")
                if input_state is not None and input_state not in self.attempt_ids:
                    raise ValueError("State route does not name a completed dry-run Attempt")
                self.attempt_count += 1
                if self.attempt_count > self.optimizer_attempt_budget:
                    raise ValueError("Workflow exceeded the Optimizer Attempt budget")
                attempt_id = f"attempt_{self.attempt_count:032x}"
                kernel_id = f"kernelrev_{self.attempt_count:032x}"
                self.attempt_ids.add(attempt_id)
                self.kernel_ids.add(kernel_id)
                self.next_attempt[key] = expected + 1
                outcomes.append(
                    {
                        "attempt_id": attempt_id,
                        "branch": branch,
                        "trajectory_ordinal": trajectory_ordinal,
                        "attempt_ordinal": attempt_ordinal,
                        "correct": True,
                        "accepted": True,
                        "latency_us": float(1000 - self.attempt_count),
                        "trajectory_kernel_revision_id": kernel_id,
                        "output_state_from_attempt_id": attempt_id,
                    }
                )
            return {"attempts": outcomes}
        if operation == "trajectory_status":
            raise ValueError("trajectory_status is not part of the current public Workflow SDK")
        if operation == "select_best_kernel":
            self._require_work_complete()
            self.selected_kernel = min(self.kernel_ids) if self.kernel_ids else None
            if self.selected_kernel is None:
                raise ValueError("Workflow produced no Kernel to select")
            return {"kernel_revision_id": self.selected_kernel, "latency_us": 1.0}
        if operation == "compare_agents":
            self._require_work_complete()
            self.selected_agent = "agentrev_" + "0" * 32
            return {"kernel_agent_revision_id": self.selected_agent, "reason": "active_retained"}
        if operation == "complete_epoch":
            if arguments.get("kernel_revision_id") != self.selected_kernel:
                raise ValueError("complete_epoch did not use select_best_kernel's result")
            if arguments.get("kernel_agent_revision_id") != self.selected_agent:
                raise ValueError("complete_epoch did not use compare_agents' result")
            self.completed = True
            return {"epoch_id": "epoch_" + "0" * 32, "status": "completed"}
        raise ValueError(f"unsupported Workflow Runtime operation: {operation}")

    def _validate_branch(self, branch: str) -> None:
        if branch == "active":
            return
        if not branch.startswith("challenger-") or not branch[11:].isdigit():
            raise ValueError(f"unsupported Workflow Branch: {branch!r}")
        ordinal = int(branch[11:])
        if ordinal not in self.challengers:
            raise ValueError(f"Branch {branch!r} has no materialized Challenger")

    def _require_work_complete(self) -> None:
        if self.attempt_count != self.optimizer_attempt_budget:
            raise ValueError(
                f"Workflow used {self.attempt_count} Optimizer Attempts; "
                f"the exact budget is {self.optimizer_attempt_budget}"
            )
        for key, (_count, capacity) in self.trajectories.items():
            if self.next_attempt[key] != capacity + 1:
                raise ValueError(f"Trajectory {key[0]}/{key[1]} did not finish its capacity")

    @staticmethod
    def _positive_int(arguments: Mapping[str, object], name: str) -> int:
        value = arguments.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
        return value


def _context(
    *,
    dsl: str,
    epoch_number: int,
    max_challengers: int,
    optimizer_attempt_budget: int,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "operation": "run_epoch",
        "context": {
            "kernel_agent_revision_id": "agentrev_" + "0" * 32,
            "dsl": dsl,
            "epoch_id": "epoch_" + "0" * 32,
            "epoch_number": epoch_number,
            "workflow_program_sha256": "0" * 64,
        },
        "limits": {
            "max_challengers": max_challengers,
            "optimizer_attempts": optimizer_attempt_budget,
        },
    }


def _run_scenario(
    workflow_root: Path,
    command: PurePosixPath,
    scenario: WorkflowCheckScenario,
    *,
    dsl: str,
    max_challengers: int,
    optimizer_attempt_budget: int,
) -> dict[str, object]:
    services = _DryRunServices(
        max_challengers=max_challengers,
        optimizer_attempt_budget=optimizer_attempt_budget,
        evolution=scenario.evolution,
    )
    with tempfile.TemporaryDirectory(prefix="atrex-workflow-check-") as temporary:
        copied = Path(temporary) / "workflow"
        shutil.copytree(workflow_root, copied)
        program = copied.joinpath(*command.parts[1:])
        stderr_path = Path(temporary) / "stderr.log"
        environment = {
            "HOME": str(Path(temporary) / "home"),
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": str(copied),
            "PYTHONUNBUFFERED": "1",
        }
        Path(environment["HOME"]).mkdir()
        initial = _context(
            dsl=dsl,
            epoch_number=scenario.epoch_number,
            max_challengers=max_challengers,
            optimizer_attempt_budget=optimizer_attempt_budget,
        )
        with stderr_path.open("wb") as stderr:
            process = subprocess.Popen(
                (sys.executable, str(program)),
                cwd=copied,
                env=environment,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=stderr,
            )
            assert process.stdin is not None and process.stdout is not None
            process.stdin.write(json.dumps(initial, separators=(",", ":")).encode() + b"\n")
            process.stdin.flush()
            selector = selectors.DefaultSelector()
            selector.register(process.stdout, selectors.EVENT_READ)
            buffer = bytearray()
            calls: list[str] = []
            deadline = time.monotonic() + _SCENARIO_TIMEOUT_SECONDS
            try:
                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise WorkflowCheckError(
                            scenario.name,
                            "timeout",
                            f"Workflow did not finish within {_SCENARIO_TIMEOUT_SECONDS:g}s",
                        )
                    events = selector.select(min(remaining, 0.1))
                    if not events:
                        if process.poll() is not None:
                            break
                        continue
                    chunk = os.read(process.stdout.fileno(), 65536)
                    if not chunk:
                        break
                    buffer.extend(chunk)
                    if len(buffer) > _MAX_PROTOCOL_LINE_BYTES:
                        raise WorkflowCheckError(
                            scenario.name,
                            "protocol",
                            "Workflow protocol line exceeds the Runtime byte limit",
                        )
                    while b"\n" in buffer:
                        raw, _, remainder = buffer.partition(b"\n")
                        buffer = bytearray(remainder)
                        if not raw.strip():
                            continue
                        if len(calls) >= _MAX_SERVICE_CALLS:
                            raise WorkflowCheckError(
                                scenario.name,
                                "protocol",
                                "Workflow exceeded the Runtime service-call limit",
                            )
                        try:
                            request: object = json.loads(raw)
                            if not isinstance(request, dict):
                                raise ValueError("request is not an object")
                            request_id = request.get("request_id")
                            operation = request.get("operation")
                            arguments = request.get("arguments", {})
                            if not isinstance(request_id, str) or not request_id:
                                raise ValueError("request_id is missing")
                            if not isinstance(operation, str) or not operation:
                                raise ValueError("operation is missing")
                            if not isinstance(arguments, dict):
                                raise ValueError("arguments is not an object")
                            result = services.execute(operation, arguments)
                        except (KeyError, TypeError, ValueError) as error:
                            raise WorkflowCheckError(
                                scenario.name,
                                "runtime_service",
                                f"Runtime rejected Workflow call {len(calls) + 1}: {error}",
                            ) from error
                        calls.append(operation)
                        response = {
                            "request_id": request_id,
                            "ok": True,
                            "result": result,
                        }
                        process.stdin.write(
                            json.dumps(response, separators=(",", ":")).encode() + b"\n"
                        )
                        process.stdin.flush()
                if buffer.strip():
                    raise WorkflowCheckError(
                        scenario.name,
                        "protocol",
                        "Workflow emitted an unterminated protocol record",
                    )
                returncode = process.wait(timeout=1)
            finally:
                selector.close()
                if process.poll() is None:
                    process.kill()
                    process.wait()
        diagnostic = stderr_path.read_text(encoding="utf-8", errors="replace").strip()
        if returncode != 0:
            raise WorkflowCheckError(
                scenario.name,
                "process",
                f"Workflow exited with {returncode}: {diagnostic[:2000] or 'no diagnostic'}",
            )
        if not services.completed:
            raise WorkflowCheckError(
                scenario.name,
                "terminal",
                "Workflow exited without completing the Epoch",
            )
        return {
            "name": scenario.name,
            "status": "passed",
            "epoch_number": scenario.epoch_number,
            "evolution": scenario.evolution,
            "service_calls": len(calls),
            "optimizer_attempts": services.attempt_count,
            "operations": calls,
        }


def check_agent_workflow(
    candidate_root: Path,
    *,
    dsl: str,
    epoch_number: int,
    max_challengers: int,
    optimizer_attempt_budget: int,
) -> dict[str, object]:
    """Run a Candidate Workflow against deterministic non-persistent Runtime services."""
    candidate = candidate_root.resolve()
    if epoch_number <= 0 or max_challengers < 0 or optimizer_attempt_budget <= 0:
        raise ValueError("Workflow check context has invalid Epoch limits")
    manifest = KernelAgentBundleManifestV1.from_file(candidate / "atrex-bundle.json")
    if manifest.workflow is None:
        raise ValueError("Candidate does not declare an executable Workflow")
    command = PurePosixPath(manifest.workflow.command)
    if not command.parts or command.parts[0] != "workflow":
        raise ValueError("Candidate Workflow command must be under workflow/")
    workflow_root = candidate / "workflow"
    program = candidate.joinpath(*command.parts)
    if program.is_symlink() or not program.is_file():
        raise ValueError("Candidate Workflow command is not a regular file")
    scenarios = [
        WorkflowCheckScenario("first-epoch", 1, "created"),
        WorkflowCheckScenario("later-epoch-evolved", max(2, epoch_number), "created"),
    ]
    if max_challengers:
        scenarios.append(
            WorkflowCheckScenario("later-epoch-no-change", max(2, epoch_number), "no_change")
        )
    results = [
        _run_scenario(
            workflow_root,
            command,
            scenario,
            dsl=dsl,
            max_challengers=max_challengers,
            optimizer_attempt_budget=optimizer_attempt_budget,
        )
        for scenario in scenarios
    ]
    return {
        "status": "valid",
        "workflow": manifest.workflow.command,
        "scenarios": results,
    }


__all__ = ["WorkflowCheckError", "check_agent_workflow"]


def main(argv: list[str] | None = None) -> int:
    """Run the checker inside the existing Evolver sandbox boundary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--dsl", required=True)
    parser.add_argument("--epoch-number", required=True, type=int)
    parser.add_argument("--max-challengers", required=True, type=int)
    parser.add_argument("--optimizer-attempt-budget", required=True, type=int)
    args = parser.parse_args(argv)
    try:
        result = check_agent_workflow(
            args.candidate,
            dsl=args.dsl,
            epoch_number=args.epoch_number,
            max_challengers=args.max_challengers,
            optimizer_attempt_budget=args.optimizer_attempt_budget,
        )
    except (OSError, RuntimeError, ValueError) as error:
        response: dict[str, object] = {
            "status": "error",
            "error": "invalid_workflow",
            "detail": str(error) or type(error).__name__,
        }
        if isinstance(error, WorkflowCheckError):
            response.update({"scenario": error.scenario, "phase": error.phase})
        print(json.dumps(response, ensure_ascii=False, sort_keys=True))
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
