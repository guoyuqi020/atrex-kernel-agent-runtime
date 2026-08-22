"""Command-line entry point for the Runtime ASGI service."""

from __future__ import annotations

import importlib
import os
from collections.abc import Callable
from typing import Protocol, cast

from ..api.app import RuntimeApplication, build_runtime_application
from ..config import RuntimeSettings
from .campaign import (
    bootstrap_campaign,
    cancel_campaign,
    recover_epoch,
    run_campaign,
    run_task_worker,
    seed_lineage,
)
from .dev_shell import (
    open_evolver_dev_shell,
    open_optimizer_dev_shell,
    open_temporary_evolver_dev_shell,
    open_temporary_optimizer_dev_shell,
)
from .inspect import (
    bootstrap_runs,
    evaluations,
    kernel_trials,
    list_agent_revisions,
    list_attempts,
    list_epochs,
    list_kernels,
    list_worker_sessions,
    show_agent_revision,
    show_attempt,
    show_kernel,
    show_worker_session,
)
from .maintenance import (
    digest_evolver_bundle,
    gc_artifacts,
    gc_workspaces,
)
from .parser import build_parser


class UvicornRun(Protocol):
    """Uvicorn entry point used by the Runtime CLI."""

    def __call__(
        self,
        app: RuntimeApplication,
        *,
        host: str,
        port: int,
        lifespan: str,
    ) -> None:
        """Serve one ASGI application until shutdown."""
        ...


def _serve(config_path: str) -> None:
    """Load strict configuration and serve the assembled Runtime."""
    settings = RuntimeSettings.from_file(config_path)
    app = build_runtime_application(settings, os.environ)
    try:
        uvicorn = importlib.import_module("uvicorn")
    except ModuleNotFoundError as error:
        app.close()
        raise RuntimeError("uvicorn is required to serve the Runtime") from error
    run = cast(UvicornRun, cast(Callable[..., object], uvicorn.__dict__["run"]))
    try:
        run(
            app,
            host=settings.server.host,
            port=settings.server.port,
            lifespan="on",
        )
    finally:
        app.close()


def main(argv: list[str] | None = None) -> None:
    """Dispatch one trusted Runtime administration command."""
    args = build_parser().parse_args(argv)
    if args.command == "serve":
        _serve(cast(str, args.config))
        return
    if args.command == "bootstrap":
        bootstrap_campaign(cast(str, args.config), cast(str, args.campaign))
        return
    if args.command == "seed-lineage":
        seed_lineage(
            cast(str, args.config),
            cast(str, args.campaign),
            cast(str, args.spec),
        )
        return
    if args.command == "run-campaign":
        run_campaign(
            cast(str, args.config),
            (
                None
                if cast(list[str] | None, args.lineage) is None
                else tuple(cast(list[str], args.lineage))
            ),
            cast(str | None, args.campaign),
            cast(int, args.target_epoch),
            finalize=cast(bool, args.finalize),
        )
        return
    if args.command == "dev-shell":
        open_optimizer_dev_shell(
            cast(str, args.config),
            cast(str | None, args.lineage),
            cast(str | None, args.attempt),
            cast(str, args.shell),
        )
        return
    if args.command == "temporary-dev-shell":
        open_temporary_optimizer_dev_shell(
            cast(str, args.config),
            cast(str, args.campaign),
            cast(str, args.shell),
        )
        return
    if args.command == "temporary-evolver-dev-shell":
        open_temporary_evolver_dev_shell(
            cast(str, args.config),
            cast(str, args.campaign),
            cast(str, args.shell),
        )
        return
    if args.command == "evolver-dev-shell":
        open_evolver_dev_shell(
            cast(str, args.config),
            cast(str, args.lineage),
            cast(int, args.epoch),
            cast(str, args.shell),
        )
        return
    if args.command == "cancel-campaign":
        cancel_campaign(cast(str, args.config), cast(str, args.campaign))
        return
    if args.command == "run-task-worker":
        run_task_worker(cast(str, args.config), watch=cast(bool, args.watch))
        return
    if args.command == "recover-epoch":
        recover_epoch(
            cast(str, args.config),
            cast(str, args.epoch),
            cast(str, args.recovery_key),
            cast(str, args.reason),
        )
        return
    if args.command == "list-epochs":
        list_epochs(
            cast(str, args.config),
            cast(str | None, args.campaign),
            cast(str | None, args.lineage),
            cast(str, args.output_format),
        )
        return
    if args.command == "list-attempts":
        list_attempts(
            cast(str, args.config),
            cast(str | None, args.campaign),
            cast(str | None, args.lineage),
            cast(str, args.output_format),
        )
        return
    if args.command == "show-attempt":
        show_attempt(cast(str, args.config), cast(str, args.attempt))
        return
    if args.command == "list-worker-sessions":
        list_worker_sessions(
            cast(str, args.config),
            cast(str | None, args.campaign),
            cast(str | None, args.lineage),
            cast(str | None, args.epoch),
            cast(str | None, args.attempt),
            cast(str | None, args.subject),
            cast(str, args.output_format),
        )
        return
    if args.command == "show-worker-session":
        show_worker_session(cast(str, args.config), cast(str, args.session))
        return
    if args.command == "list-kernels":
        list_kernels(
            cast(str, args.config),
            cast(str | None, args.campaign),
            cast(str | None, args.lineage),
            cast(str, args.output_format),
        )
        return
    if args.command == "show-kernel":
        show_kernel(cast(str, args.config), cast(str, args.kernel))
        return
    if args.command == "list-agent-revisions":
        list_agent_revisions(
            cast(str, args.config),
            cast(str | None, args.campaign),
            cast(str | None, args.lineage),
            cast(str, args.output_format),
        )
        return
    if args.command == "show-agent-revision":
        show_agent_revision(
            cast(str, args.config),
            cast(str, args.agent_revision),
        )
        return
    if args.command == "list-bootstrap-runs":
        bootstrap_runs(cast(str, args.config), cast(str, args.attempt), None)
        return
    if args.command == "show-bootstrap-run":
        bootstrap_runs(
            cast(str, args.config),
            cast(str, args.attempt),
            cast(int, args.generation),
        )
        return
    if args.command == "list-evaluations":
        evaluations(
            cast(str, args.config),
            attempt_value=cast(str, args.attempt),
            evaluation_id=None,
        )
        return
    if args.command == "show-evaluation":
        evaluations(
            cast(str, args.config),
            attempt_value=None,
            evaluation_id=cast(str, args.evaluation),
            include_source=cast(bool, args.source),
            include_result=cast(bool, args.result),
        )
        return
    if args.command == "list-kernel-trials":
        kernel_trials(
            cast(str, args.config),
            attempt_value=cast(str, args.attempt),
            trial_id=None,
        )
        return
    if args.command == "show-kernel-trial":
        kernel_trials(
            cast(str, args.config),
            attempt_value=None,
            trial_id=cast(str, args.trial),
            include_source=cast(bool, args.source),
            include_results=cast(bool, args.result),
        )
        return
    if args.command == "gc-artifacts":
        gc_artifacts(
            cast(str, args.config),
            minimum_age_seconds=cast(float, args.minimum_age_seconds),
            limit=cast(int, args.limit),
            apply=cast(bool, args.apply),
            confirm_runtime_stopped=cast(bool, args.confirm_runtime_stopped),
        )
        return
    if args.command == "gc-workspaces":
        gc_workspaces(
            cast(str, args.config),
            minimum_age_seconds=cast(float, args.minimum_age_seconds),
            limit=cast(int, args.limit),
            apply=cast(bool, args.apply),
            confirm_runtime_stopped=cast(bool, args.confirm_runtime_stopped),
        )
        return
    if args.command == "digest-evolver-bundle":
        digest_evolver_bundle(
            cast(str, args.path),
            max_files=cast(int, args.max_files),
            max_bytes=cast(int, args.max_bytes),
        )
        return
    raise AssertionError(f"unhandled command: {args.command}")
